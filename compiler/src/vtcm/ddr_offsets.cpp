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
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace hnnx {

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

    std::unordered_map<op_id_t, uint32_t> pos;
    for (size_t i = 0; i < order.size(); i++) pos[order[i]] = (uint32_t)i;
    uint32_t n = (uint32_t)order.size();

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

        uint32_t life_end = (uint32_t)p + 1;
        bool has_consumer = false;
        for (op_id_t cid : od->consumers) {
            OpDef* c = get_op_at(cid);
            if (!c || c->is_dead()) continue;
            has_consumer = true;
            auto it = pos.find(cid);
            life_end = (it != pos.end())
                           ? std::max(life_end, it->second + 1)
                           : n;  // 消费者不在计划序(如 Output)→ 活到图尾
        }
        if (!has_consumer) life_end = n;  // 图输出 producer

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
        *cap_out = cap;
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
    for (const auto& e : entries) {
        uint32_t lb = 0, le = 0;
        auto pit = pos.find(e.op_id);
        if (pit != pos.end()) {
            lb = pit->second;
            le = lb + 1;
            const OpDef* od = get_op_at(e.op_id);
            if (od) {
                bool has_consumer = false;
                for (op_id_t cid : od->consumers) {
                    const OpDef* c = get_op_at(cid);
                    if (!c || c->is_dead()) continue;
                    has_consumer = true;
                    auto cit = pos.find(cid);
                    le = (cit != pos.end()) ? std::max(le, cit->second + 1) : n;
                }
                if (!has_consumer) le = n;
            }
        }
        f << e.op_id << " " << e.offset << " " << e.size << " " << lb << " " << le << "\n";
    }
    f.close();
    return 0;
}

} // namespace hnnx
