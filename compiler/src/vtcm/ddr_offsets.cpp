// ddr_offsets.cpp —— 第7步阶段一: 静态 DDR 偏移表发射
// =====================================================================
// 复用 FancyAllocator::allocate_with_lifetime(RE 实证的生命期分配器:
// reverse-life-end 排序 + 同生命期分组 + 事件扫描复用), 把预算从 VTCM
// 换成 DDR 池, 对每个存活张量(按执行序 = plan_order_)分配池内偏移。
// 产物两种消费:
//   - compute_ddr_offsets → entries vector(wtop_emit 打包进 blob TEMPOFF 槽)
//   - emit_ddr_offset_table → TSV(kernels/host/check_offsets 裁判)
// 生命期约定: [生产 op 的执行序位置, 最后消费者位置+1); 图输出 producer
// 活到图尾。temp id 与 op_id 的映射(wtop_emit fresh_temp)在 D 步处理。
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace hnnx {

// 有效生命期分析: direct = 最后直接消费位置; eff = 沿 Reshape 引用消除别名链
// 取 max(与 wtop_ops::alias_temp 的 last_use 并集语义对齐)。compute_ddr_offsets
// 与 emit_ddr_offset_table 共享, 保证 TEMPOFF 与 TSV 裁判口径一致。
struct Liveness {
    std::unordered_map<op_id_t, uint32_t> direct;
    std::unordered_map<op_id_t, uint32_t> eff;
};

static Liveness compute_eff_last_use(const GraphPrepare& gp,
                                     const std::vector<op_id_t>& order) {
    Liveness lv;
    // 1. 直接消费: 逐 op 扫 inputs 边(out_idx==0 为 DDR 静态缓冲)。
    for (size_t p = 0; p < order.size(); p++) {
        const OpDef* c = gp.get_op_at(order[p]);
        if (!c) continue;
        for (const auto& e : c->inputs)
            if (e.out_idx == 0)
                lv.direct[e.src_id] = (uint32_t)p;
    }
    // 2. Reshape 引用消除(op_reshape → alias_temp): Reshape 不物化, 其输出
    // 别名到输入 temp。producer 缓冲真实生命期须沿 Reshape 链延伸到最后一个
    // "真读者"。
    auto is_alias_op = [&](op_id_t id) -> bool {
        const OpDef* o = gp.get_op_at(id);
        return o && o->name_tag && o->name_tag->name() &&
               std::strcmp(o->name_tag->name(), "Reshape") == 0;
    };
    std::unordered_map<op_id_t, std::vector<op_id_t>> alias_next;  // producer → 别名它的 Reshape
    for (op_id_t id : order) {
        const OpDef* o = gp.get_op_at(id);
        if (!o || !is_alias_op(id) || o->inputs.empty()) continue;
        if (o->inputs[0].out_idx == 0)
            alias_next[o->inputs[0].src_id].push_back(id);
    }
    // 3. 逆拓扑序传播: alias_next 边指向更晚的 op, 倒序处理保证子节点已算。
    for (size_t i = order.size(); i-- > 0;) {
        op_id_t pid = order[i];
        uint32_t best = UINT32_MAX;
        auto lu = lv.direct.find(pid);
        if (lu != lv.direct.end()) best = lu->second;
        auto nit = alias_next.find(pid);
        if (nit != alias_next.end()) {
            for (op_id_t r : nit->second) {
                uint32_t sub = lv.eff[r];
                if (sub != UINT32_MAX && (best == UINT32_MAX || sub > best)) best = sub;
            }
        }
        lv.eff[pid] = best;
    }
    return lv;
}

int GraphPrepare::compute_ddr_offsets(uint64_t budget,
                                      std::vector<DdrTempEntry>* entries,
                                      uint64_t* cap_out,
                                      std::vector<DdrSpillEntry>* spilled_out,
                                      uint64_t* spill_total_out,
                                      uint64_t vtcm_budget,
                                      uint64_t* vtcm_cap_out) {
    if (!entries || !cap_out) return -1;
    entries->clear();
    *cap_out = 0;
    if (spilled_out) spilled_out->clear();
    if (spill_total_out) *spill_total_out = 0;
    if (vtcm_cap_out) *vtcm_cap_out = vtcm_budget;

    std::vector<op_id_t> order = plan_order_;
    if (order.empty()) order = get_ordering();  // 未调度: 退回拓扑序
    if (order.empty()) return -1;

    uint32_t n = (uint32_t)order.size();

    // 活性分析(与 wtop_ops::alias_temp 语义对齐): 逐 op 扫 inputs 边(out_idx==0
    // 为 DDR 静态缓冲)记最后直接消费位置, 再沿 Reshape 引用消除别名链取 max
    // 得到有效生命期终点。40_qwen_layer3 temp6/temp25 重叠根因: Eltwise_Binary
    // 产 K 张量只被一个 Reshape 直接引用, 而该 Reshape 别名链一路拉到 softmax
    // 之后的 PV MatMul —— 直接读者法把生命期低估到 [生产, Reshape), 与后段
    // temp25 重叠。eff = 沿 Reshape 链递归取 max; UINT32_MAX = 全链无消费者。
    Liveness lv = compute_eff_last_use(*this, order);
    std::unordered_map<op_id_t, uint32_t>& eff_use = lv.eff;

    std::vector<fa::FancyAllocator::AllocRequest> reqs;
    for (size_t p = 0; p < order.size(); p++) {
        OpDef* od = get_op_at(order[p]);
        if (!od || !od->is_enabled() || od->is_dead() || od->is_const()) continue;
        if (!od->name_tag || !od->name_tag->name()) continue;
        std::string nm = od->name_tag->name();
        if (nm == "Input" || nm == "Output") continue;

        uint64_t elems = 1;
        for (uint32_t i = 0; i < od->output_def.rank && i < 5; ++i)
            elems *= od->output_def.dims[i];
        if (elems == 0) elems = 1;
        size_t es = od->output_def.element_size;
        if (es == 0 || es > 4) es = 4;
        uint64_t bytes = (elems * es + 127) & ~uint64_t(127);

        // 活性: 有效生命期 = 沿 Reshape 别名链取 max 最后读者 + 1;
        // 无消费者(图输出 producer)活到图尾。
        uint32_t life_end = (uint32_t)p + 1;
        uint32_t last = eff_use[od->op_id];
        if (last != UINT32_MAX)
            life_end = std::max(life_end, last + 1);
        else
            life_end = n;

        reqs.push_back({od->op_id, (size_t)bytes, (uint32_t)p, life_end});
    }

    uint64_t total = 0;
    for (const auto& r : reqs) total += r.size;
    uint64_t cap = budget ? budget : total;  // 默认总尺寸和 = 纯 bump 无复用

    // 第7步阶段二: 两级分配。先跑 VTCM 池(驻留); 装不下的进 DDR 池。
    // 两池用同一分配器实例独立跑(生命期复用各自池内生效)。
    if (vtcm_budget > 0) {
        fa::FancyAllocator fa_v;
        auto vac = fa_v.allocate_with_lifetime(reqs, (size_t)vtcm_budget, 128);
        std::vector<fa::FancyAllocator::AllocRequest> ddr_reqs;
        for (const auto& r : reqs) {
            auto it = vac.find(r.op_id);
            if (it != vac.end() && !it->second.spilled)
                entries->push_back({r.op_id, it->second.offset, r.size, true});
            else
                ddr_reqs.push_back(r);
        }
        fa::FancyAllocator fa_d;
        auto dac = fa_d.allocate_with_lifetime(ddr_reqs, (size_t)cap, 128);
        std::vector<const fa::FancyAllocator::AllocRequest*> spill_reqs;
        for (const auto& r : ddr_reqs) {
            auto it = dac.find(r.op_id);
            if (it != dac.end() && !it->second.spilled)
                entries->push_back({r.op_id, it->second.offset, r.size, false});
            else
                spill_reqs.push_back(&r);
        }
        if (budget == 0) {
            /* ddr 预算缺省(仅 vtcm 池开): cap 记生命期打包实际水位而非 Σ全量
             * 尺寸 —— 运行时按 blob cap memalign, Σ口径会虚耗数百 MB DDR */
            uint64_t hw = 0;
            for (const auto& e : *entries)
                if (!e.in_vtcm) hw = std::max<uint64_t>(hw, e.offset + e.size);
            *cap_out = hw;
        } else {
            *cap_out = cap;
        }
        finalize_spills(spill_reqs, spilled_out, spill_total_out, total);
        return 0;
    }

    // 阶段一: 全 DDR 分配
    fa::FancyAllocator fa;
    auto allocs = fa.allocate_with_lifetime(reqs, (size_t)cap, 128);
    std::vector<const fa::FancyAllocator::AllocRequest*> spill_reqs;
    for (const auto& r : reqs) {
        auto it = allocs.find(r.op_id);
        if (it != allocs.end() && !it->second.spilled) {
            entries->push_back({r.op_id, it->second.offset, r.size, false});
        } else {
            spill_reqs.push_back(&r);
        }
    }
    *cap_out = cap;
    finalize_spills(spill_reqs, spilled_out, spill_total_out, total);
    return 0;
}

void GraphPrepare::finalize_spills(
    const std::vector<const fa::FancyAllocator::AllocRequest*>& spill_reqs,
    std::vector<DdrSpillEntry>* spilled_out,
    uint64_t* spill_total_out,
    uint64_t total) {
    auto sorted = spill_reqs;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto* a, const auto* b) { return a->op_id < b->op_id; });
    uint64_t spill_total = 0;
    if (spilled_out) {
        for (const auto* r : sorted) {
            spilled_out->push_back({r->op_id, spill_total, r->size});
            spill_total += r->size;
        }
    }
    if (spill_total_out) *spill_total_out = spill_total;
    if (!sorted.empty()) {
        std::fprintf(stderr, "ddr_offsets: %zu tensors spilled (total=%llu spill=%llu)\n",
                     sorted.size(), (unsigned long long)total,
                     (unsigned long long)spill_total);
    }
}

int GraphPrepare::emit_ddr_offset_table(const char* path, uint64_t budget) {
    if (!path) return -1;
    std::vector<DdrTempEntry> entries;
    uint64_t cap = 0;
    if (compute_ddr_offsets(budget, &entries, &cap) != 0) return -1;

    std::ofstream f(path);
    if (!f) return -2;
    f << "# GEHTP 第7步阶段一: 静态 DDR 偏移表 (check_offsets 消费)\n";
    f << "# n_temps=" << entries.size() << " cap=" << cap << "\n";
    f << "pool_cap " << cap << "\n";
    // 生命期字段从 plan_order 重算(TSV 裁判用; 表本体只有 offset/size)
    std::vector<op_id_t> order = plan_order_;
    if (order.empty()) order = get_ordering();
    std::unordered_map<op_id_t, uint32_t> pos;
    for (size_t i = 0; i < order.size(); i++) pos[order[i]] = (uint32_t)i;
    uint32_t n = (uint32_t)order.size();
    // 活性用 inputs 真值反推(与 compute_ddr_offsets 同源; consumers 缓存陈旧;
    // 沿 Reshape 别名链取 max 的口径与 TEMPOFF 一致)。
    Liveness lv = compute_eff_last_use(*this, order);
    for (const auto& e : entries) {
        uint32_t lb = 0, le = 0;
        auto pit = pos.find(e.op_id);
        if (pit != pos.end()) {
            lb = pit->second;
            le = lb + 1;
            auto lu = lv.eff.find(e.op_id);
            if (lu != lv.eff.end()) le = std::max(le, lu->second + 1);
            else le = n;
        }
        f << e.op_id << " " << e.offset << " " << e.size << " " << lb << " " << le << "\n";
    }
    f.close();
    return 0;
}

} // namespace hnnx
