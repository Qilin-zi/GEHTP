// wtop_emit: GEHTP tagged .bin → WTOP blob (阶段 8, host 闭环枢纽)
//
// 用法:
//   wtop_emit --bin <tagged.bin> [--input-f16 <X.f16.raw>]
//             --out <blob.wtop> [--manifest <manifest.json>]
//
// 流程: deserialize .bin → 按 plan_order 遍历 → 逐 op 发射 WTOP:
//   Transpose        → OP_TRANSPOSE_F16
//   Conv2d           → 每 tile: OP_IM2COL(切片+halo) + OP_CONV2D_F16(纯 GEMM)
//   Eltwise_Binary   → OP_ADD_F16
//   溢出张量(SF 记录)→ OP_SPILL / OP_FILL(见 --spill-pool 说明)
// 受支持几何门: conv sh=sw=1、dh=dw=1、group=1、same-pad; 不满足报错退出
// (特性门非模型特判; 字段已参数化, 扩支持 = 解门)。
//
// 权重: 池内 [kh,kw,ci,co] f32 → f16(RNE)K×N 进 slot; 输入 slot 0 由
// --input-f16 提供(Level 1 外部槽: 阶段 9 引擎把该槽标为 external)。
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_extra.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/serialize/serializer.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// WTOP 契约(kernels/include/oplist_parse.h 的编译期副本语义; 头从 kernels 取)
#include "oplist_parse.h"

#include "wtop_ops/wtop_ops.hpp"

using namespace hnnx;
using namespace wtop;

namespace {


bool load_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz > 0 ? (size_t)sz : 0);
    if (sz > 0 && std::fread(out.data(), 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}


// 字符串按分隔符切分
static std::vector<std::string> split_str(const std::string& st, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= st.size()) {
        size_t pp = st.find(sep, start);
        out.push_back(st.substr(start, pp == std::string::npos ? std::string::npos : pp - start));
        if (pp == std::string::npos) break;
        start = pp + 1;
    }
    return out;
}

}


int emit(const std::string& bin_path, const std::string& in_f16_path,
         const std::string& out_path, const std::string& manifest_path,
         const std::string& gguf_path, const std::string& match_path,
         bool ext_weights = false, const std::string& ext_weights_path = "",
         bool force_v2 = false, bool emit_dma = false) {
    // 1. deserialize .bin
    std::vector<uint8_t> bin;
    if (!load_file(bin_path, bin)) { std::fprintf(stderr, "error: cannot open %s\n", bin_path.c_str()); return 2; }
    GraphPrepare gp;
    if (!gp.deserialize(bin.data(), bin.size())) { std::fprintf(stderr, "error: deserialize failed\n"); return 2; }

    // 1b. 第7步阶段一: 静态 DDR 偏移(编译期生命期分配, 打包进 TEMPOFF 槽)。
    //     compute 在 Emitter em 声明后执行(见下方 ddr_static 块)。

    // 2. 图级信息
    // 主输入 = Input 节点按 op_id 升序首项(与输入槽分配同约定;
    // gp.get_input_node_id() 可能指向 cos 等次输入)
    std::vector<uint64_t> in_ids;
    {
        gp.for_each_op([&](OpDef* od) {
            if (od && od->name_tag && od->name_tag->name() &&
                std::string(od->name_tag->name()) == "Input" && !od->is_dead())
                in_ids.push_back(od->op_id);
        });
        std::sort(in_ids.begin(), in_ids.end());
        if (in_ids.empty()) in_ids.push_back(gp.get_input_node_id());
    }
    const OpDef* input_op = gp.get_op_at(in_ids[0]);
    if (!input_op || input_op->output_def.rank == 0) { std::fprintf(stderr, "error: bad input node\n"); return 2; }
    // 输入形状/元素数: 任意 rank; dtype 决定字节/元素(f32/f16→2B 外部, int32→4B)
    size_t input_elems = 1;
    for (uint32_t i = 0; i < input_op->output_def.rank && i < 5; ++i)
        input_elems *= (size_t)input_op->output_def.dims[i];
    uint32_t input_dtype = input_op->output_def.dtype;
    size_t input_elem_bytes = (input_dtype == 1 || input_dtype == 0x21) ? 2 : 4;
    // NCHW 便捷(conv 路径沿用)
    uint32_t H = input_op->output_def.rank >= 4 ? (uint32_t)input_op->output_def.dims[1] : 1;
    uint32_t W = input_op->output_def.rank >= 4 ? (uint32_t)input_op->output_def.dims[2] : 1;
    uint32_t C = input_op->output_def.rank >= 4 ? (uint32_t)input_op->output_def.dims[3] : 1;

    Emitter em;
    em.ext_weights = ext_weights;  // 路线B: 权重外置(ensure_weight_slot → ext_weight_area)
    if (gp.has_mem_plan()) {
        // M2: 静态规划由编译器定稿(TAG_MEM_PLAN)——照抄, 不再自算。
        const auto& mp = gp.mem_plan();
        for (const auto& e : mp.entries) {
            em.ddr_op_map[e.op_id] = {(uint32_t)e.offset, (uint32_t)e.size};
            if (e.in_vtcm) em.ddr_op_vtcm[e.op_id] = true;
        }
        em.ddr_static_cap = mp.ddr_cap;
        em.ddr_vtcm_cap = mp.vtcm_cap;
        em.bump_reserve = mp.bump_reserve;
        em.ddr_static = true;
        for (const auto& s : mp.spills)
            em.ddr_spill_map[s.op_id] = {(uint32_t)s.offset, (uint32_t)s.size};
        std::fprintf(stderr, "[ddr] TAG_MEM_PLAN: cap=%llu vtcm_cap=%llu n_temps=%zu n_spills=%zu\n",
                     (unsigned long long)em.ddr_static_cap, (unsigned long long)em.ddr_vtcm_cap,
                     mp.entries.size(), mp.spills.size());
    }

    // 输入槽: 主输入 = slot0 EXT_IN; 其余图输入(--input-f16 逗号分隔
    // 后续文件, 或全零)固化为普通槽 — 全注意力层 cos/sin/mask 静态化。
    {
        std::vector<std::string> in_paths;
        if (!in_f16_path.empty()) {
            size_t pos = 0;
            while (pos <= in_f16_path.size()) {
                size_t comma = in_f16_path.find(',', pos);
                in_paths.push_back(in_f16_path.substr(
                    pos, comma == std::string::npos ? std::string::npos : comma - pos));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        /* Input 节点按 op_id 升序(与 host_run 同约定) — in_ids 已在图级信息段收集 */
        em.main_input_id = in_ids[0];
        for (size_t k = 0; k < in_ids.size(); k++) {
            const OpDef* iop = gp.get_op_at(in_ids[k]);
            if (!iop) continue;
            size_t elems = 1;
            for (uint32_t i = 0; i < iop->output_def.rank && i < 5; ++i)
                elems *= (size_t)iop->output_def.dims[i];
            /* 设备槽面统一 f16: f32 文件(4B/元素)转 f16 后塞槽;
             * 已是 f16(2B/元素)直塞。 */
            std::vector<uint8_t> data;
            /* 主输入 dtype 跟随图输入节点: int32(ids/gather 索引)保持 4B/元素,
             * 不参与槽面 f16 化; 其余 f16 直塞 / f32 转 f16。 */
            bool is_i32_in = (k == 0 && iop->output_def.dtype == (uint32_t)DType::Int32);
            if (is_i32_in) {
                if (k < in_paths.size()) {
                    if (!load_file(in_paths[k], data)) {
                        std::fprintf(stderr, "error: cannot open %s\n", in_paths[k].c_str());
                        return 2;
                    }
                    if (data.size() != elems * 4) {
                        std::fprintf(stderr, "error: input %zu size %zu != %zu*4 (int32, elems=%zu)\n",
                                     k, data.size(), elems, elems);
                        return 2;
                    }
                } else {
                    data.assign(elems * 4, 0);
                }
            } else if (k < in_paths.size()) {
                if (!load_file(in_paths[k], data)) {
                    std::fprintf(stderr, "error: cannot open %s\n", in_paths[k].c_str());
                    return 2;
                }
                if (data.size() == elems * 4) {
                    const float* f = reinterpret_cast<const float*>(data.data());
                    std::vector<uint8_t> h(elems * 2);
                    uint16_t* d = reinterpret_cast<uint16_t*>(h.data());
                    for (size_t i = 0; i < elems; i++) d[i] = f32_to_f16_rne(f[i]);
                    data = std::move(h);
                } else if (data.size() != elems * 2) {
                    std::fprintf(stderr, "error: input %zu size %zu != %zu*2/4 (elems=%zu)\n",
                                 k, data.size(), elems, elems);
                    return 2;
                }
            } else {
                data.assign(elems * 2, 0);

            }
            uint32_t sid = em.add_slot((uint32_t)data.size(), (uint32_t)elems, data.data());
            if (k == 0) {
                em.slots[sid].addr = WT_SLOT_EXT_IN;  // 主输入: run_io 注入
            }
            em.input_slots[in_ids[k]] = sid;
        }
    }

    // 2a2. 阶段二: spill 池槽(必须在输入段之后; 槽 0 = 主输入不可占)
    if (em.ddr_static && !em.ddr_spill_map.empty() && em.spill_pool_slot == 0) {
        uint64_t spill_total = 0;
        for (const auto& [k, v] : em.ddr_spill_map)
            spill_total = std::max<uint64_t>(spill_total, (uint64_t)v.first + v.second);
        std::vector<uint8_t> zeros((size_t)spill_total, 0);
        em.spill_pool_slot = em.add_slot((uint32_t)spill_total,
                                         (uint32_t)(spill_total / 2), zeros.data());
        std::fprintf(stderr, "[ddr] spill pool slot=%u total=%llu\n",
                     em.spill_pool_slot, (unsigned long long)spill_total);
    }

    // 2b. GGUF 供给(M2c): 读 TSV 匹配表 + GGUF 文件
    std::map<std::string, GgufEntry> gguf_map;
    std::vector<uint8_t> gguf_data;
    bool have_gguf = false;
    if (!gguf_path.empty() && !match_path.empty()) {
        std::ifstream mf(match_path);
        std::string line;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == 'n') continue;
            std::vector<std::string> cols = split_str(line, '\t');
            if (cols.size() < 6) continue;
            GgufEntry e{};
            e.type = (uint32_t)std::strtoul(cols[2].c_str(), nullptr, 10);
            e.file_offset = std::strtoull(cols[3].c_str(), nullptr, 10);
            e.nbytes = std::strtoull(cols[4].c_str(), nullptr, 10);
            for (auto& d : split_str(cols[5], ','))
                e.dims.push_back(std::strtoull(d.c_str(), nullptr, 10));
            gguf_map[cols[0]] = e;
        }
        if (!load_file(gguf_path, gguf_data)) {
            std::fprintf(stderr, "error: cannot open %s\n", gguf_path.c_str());
            return 2;
        }
        have_gguf = true;
        std::printf("[gguf] loaded %zu entries, %zu bytes\n", gguf_map.size(), gguf_data.size());
    }
    if (have_gguf) {
        em.use_gguf = true;
        em.gguf_tab = &gguf_map;
        em.gguf_bytes = &gguf_data;
    }

    // 2c. 路线B: 外置权重源(--weights-bin params.bin), ensure_weight_slot 按
    //     extent offset 从此读权重字节(const_pool 空时的替代源)。
    std::vector<uint8_t> ext_weights_bin;
    if (em.ext_weights && !ext_weights_path.empty()) {
        if (!load_file(ext_weights_path, ext_weights_bin)) {
            std::fprintf(stderr, "error: cannot open weights-bin %s\n", ext_weights_path.c_str());
            return 2;
        }
        em.ext_weights_bin = &ext_weights_bin;
        std::printf("[ext-weights] source %s = %zu bytes\n", ext_weights_path.c_str(), ext_weights_bin.size());
    }

    // 3. 通用权重槽收集(M2): 按 plan_order 遍历, 对权重消费 op 的 const 输入
    //     建 f16 槽(按 const id 去重, tie 权重共享一槽)
    std::map<uint64_t, uint32_t> wslots;  // const op_id → slot
    {
        std::vector<op_id_t> pre_order = gp.plan_order();
        for (op_id_t id : pre_order) {
            const OpDef* od = gp.get_op_at(id);
            if (!od || od->is_const() || !od->name_tag) continue;
            std::string nm = od->name_tag->name() ? od->name_tag->name() : "";
            int wi = weight_input_index(nm);
            if (wi < 0 || od->inputs.size() <= (size_t)wi) continue;
            const OpDef* w = gp.get_op_at(od->inputs[wi].src_id);
            if (!w || w->const_data_size == 0) continue;
            em.ensure_weight_slot(gp, w, wslots, od->grouping);
        }
    }
    // conv 分支的便捷引用(无 conv 图时为 0, conv 分支不会触发)
    uint32_t w_slot = 0, b_slot = 0;
    {
        const OpDef* conv = nullptr;
        for (op_id_t id : gp.plan_order()) {
            const OpDef* od = gp.get_op_at(id);
            if (od && od->name_tag && std::string(od->name_tag->name() ? od->name_tag->name() : "") == "Conv2d")
                { conv = od; break; }
        }
        if (conv) {
            if (conv->inputs.size() < 2) { std::fprintf(stderr, "error: conv missing weight input\n"); return 2; }
            const OpDef* w = gp.get_op_at(conv->inputs[1].src_id);
            if (!w) { std::fprintf(stderr, "error: weight const missing\n"); return 2; }
            w_slot = em.ensure_weight_slot(gp, w, wslots, conv->grouping);
            if (conv->inputs.size() > 2) {
                const OpDef* b = gp.get_op_at(conv->inputs[2].src_id);
                if (b) b_slot = em.ensure_weight_slot(gp, b, wslots, conv->grouping);
            }
        }
    }

    // 4. 按 plan_order 发射 op
    std::vector<op_id_t> order = gp.plan_order();
    /* M1(内存规划收编)起: plan_order 由编译器 do_prepare2_late Kahn 定稿
     * (保证拓扑), 本块不再重排。仅做拓扑性校验: 发现"消费者排在生产者前"
     * = 未经 M1 定稿的旧 bin → warn 并退回本地 Kahn 重排(保旧行为);
     * 无违反则直接用 bin 序。 */
    {
        std::unordered_map<op_id_t, size_t> pos;
        for (size_t i = 0; i < order.size(); i++) pos[order[i]] = i;
        bool violated = false;
        for (op_id_t id : order) {
            const OpDef* od = gp.get_op_at(id);
            if (!od) continue;
            for (const auto& c : od->inputs) {
                auto it = pos.find(c.src_id);
                if (it != pos.end() && it->second > pos[id] && gp.get_op_at(c.src_id)) {
                    std::fprintf(stderr,
                                 "warn: plan_order 拓扑违反: op %llu 先于其生产者 %llu (旧 bin?)\n",
                                 (unsigned long long)id, (unsigned long long)c.src_id);
                    violated = true;
                }
            }
        }
        if (violated) {
            /* 旧 bin 兜底: 原 Kahn 重排(与编译器定稿算法逐语句一致;
             * 环图回退原序) */
            std::unordered_map<op_id_t, size_t> indeg;
            std::unordered_map<op_id_t, std::vector<op_id_t>> succ;
            std::vector<op_id_t> orphans;
            for (op_id_t id : order) {
                const OpDef* od = gp.get_op_at(id);
                if (!od) { orphans.push_back(id); continue; }
                indeg[id] = 0;
                for (const auto& c : od->inputs)
                    if (indeg.count(c.src_id)) { ++indeg[id]; succ[c.src_id].push_back(id); }
            }
            std::vector<op_id_t> ready;
            for (auto& [id, d] : indeg) if (d == 0) ready.push_back(id);
            std::sort(ready.begin(), ready.end());
            std::vector<op_id_t> topo;
            while (!ready.empty()) {
                op_id_t id = ready.back();
                ready.pop_back();
                topo.push_back(id);
                for (op_id_t s : succ[id])
                    if (--indeg[s] == 0) ready.push_back(s);
            }
            for (op_id_t id : orphans) topo.push_back(id);
            if (topo.size() == order.size()) order = topo;
        }
    }
    if (order.empty()) {
        for (op_id_t id = 1; id <= 10; id++) order.push_back(id);  // 兜底(常规图 id 1..10)
    }
    em.compute_last_use(order, gp);
    // Output 引用的 temp 必须活到循环外(唯一真实跨循环引用)
    {
        const OpDef* out = gp.get_op_at(gp.get_output_node_id());
        if (out && !out->inputs.empty())
            em.last_use[Emitter::tkey(out->inputs[0].src_id, out->inputs[0].out_idx)] =
                static_cast<uint32_t>(order.size());
    }
    // spill/fill 不钉 temp: 0.8B 默认预算下全图溢出 → 上千 spill 记录,
    // 钉住会把整个活跃集抬到 16879。(M3: 旧 spill 段已删, 溢出走 TAG_MEM_PLAN 阶段二)
    WtopEmitShared sh{H, W, C, w_slot, b_slot};
    for (size_t pos = 0; pos < order.size(); pos++) {
        op_id_t id = order[pos];
        em.cur_op = id;
        if (pos < 8 || id < 100) {
            const OpDef* dbg = gp.get_op_at(id);
            std::fprintf(stderr, "[dbg] pos=%zu id=%llu name=%s const=%d\n", pos,
                         (unsigned long long)id,
                         dbg && dbg->name_tag && dbg->name_tag->name() ? dbg->name_tag->name() : "?",
                         dbg && dbg->is_const());
        }
        const OpDef* od = gp.get_op_at(id);
        if (!od || od->is_const() || !od->name_tag) continue;
        std::string nm = od->name_tag->name() ? od->name_tag->name() : "";
        if (nm == "Input" || nm == "Output" || nm.empty()) continue;
        // loader 合成的 tensor_param const(非 is_const 标志): *_shape/*_axes/
        // *_pad_amount/*_ranges 后缀, 无发射语义
        {
            static const char* suf[] = {"_shape", "_axes", "_pad_amount", "_ranges", "_perm", "_multiples", "_split_index", "_dilation", "_stride"};
            bool is_param = false;
            for (const char* sfx : suf) {
                size_t l = std::strlen(sfx);
                if (nm.size() >= l && nm.compare(nm.size() - l, l, sfx) == 0) is_param = true;
            }
            if (is_param) continue;
        }
        // 阶段二插桩(前): 消费溢出张量的 op 前插 FILL(溢出区 → scratch temp)
        if (em.ddr_static && !em.ddr_spill_map.empty()) {
            for (const auto& c : od->inputs) {
                auto sit = em.ddr_spill_map.find(c.src_id);
                if (sit == em.ddr_spill_map.end()) continue;
                auto tit = em.op_temp.find(Emitter::tkey(c.src_id, c.out_idx));
                if (tit == em.op_temp.end()) continue;  // 未物化(如 const 引用)
                if (emit_dma) {
                    /* OP_DMA: DDR→DDR 溢出搬移显式化 (src_bypass=1/dst_bypass=0 同
                     * dma_copy_ddr; fence DMA→CPU/DDR = INVALIDATE; slot_off=池内偏移) */
                    em.add_op(OP_DMA, {0x8000u | em.spill_pool_slot, tit->second,
                                       (uint32_t)sit->second.second,
                                       1, 0, 1, 0, 0, (uint32_t)sit->second.first});
                } else {
                    em.add_op(OP_FILL, {em.spill_pool_slot, (uint32_t)sit->second.first,
                                        tit->second, (uint32_t)(sit->second.second / 2)});
                }
            }
        }
        auto hit = wtop::op_registry().find(nm);
        if (hit == wtop::op_registry().end()) {
            std::fprintf(stderr, "error: unsupported op '%s' in wtop_emit\n", nm.c_str());
            return 4;
        }
        int hrc = hit->second(em, gp, od, wslots, nm, sh);
        if (hrc) return hrc;
        // 阶段二插桩(后): 溢出张量的生产 op 后插 SPILL(scratch temp → 溢出区)
        // 必须在 release 之前: release 会回收 temp 复用到后续 op。
        if (em.ddr_static && !em.ddr_spill_map.empty()) {
            auto sit = em.ddr_spill_map.find(id);
            if (sit != em.ddr_spill_map.end()) {
                auto tit = em.op_temp.find(Emitter::tkey(id, 0));
                if (tit != em.op_temp.end()) {
                    if (emit_dma) {
                        em.add_op(OP_DMA, {tit->second, 0x8000u | em.spill_pool_slot,
                                           (uint32_t)sit->second.second,
                                           1, 0, 1, 0, 0, (uint32_t)sit->second.first});
                    } else {
                        em.add_op(OP_SPILL, {tit->second, em.spill_pool_slot,
                                             (uint32_t)sit->second.first,
                                             (uint32_t)(sit->second.second / 2)});
                    }
                }
            }
        }
        // 活性: 本 op 的输入已消费, 释放其 temp
        for (const auto& c : od->inputs)
            em.release_at(Emitter::tkey(c.src_id, c.out_idx), pos);
    }


    // 5b. 第7步阶段一: 静态 temp 偏移表 → TEMPOFF 槽
    // [cap u32][reserve u32][n u32][n × {temp_id u32, offset u32, size u32}]
    // 第7步阶段二: reserve 字段拆两段(u32): [低 16 位=表外 bump 预留] |
    // [高 16 位=VTCM 池大小]。两个域都 ≤ 65535(单元是 KB 级, 128 对齐
    // 上取整 —— probe VTCM 驻留 256KB = 256 单位)。VTCM 驻留项 temp_id
    // 带 0x4000 标记(WT_REF_VTCM_FLAG)。
    if (em.ddr_static) {
        uint64_t bump_res = em.bump_reserve ? em.bump_reserve
                                            : std::max<uint64_t>(4u << 20, em.ddr_static_cap / 4);
        uint32_t reserve = ((uint32_t)((em.ddr_vtcm_cap + 1023) / 1024) << 16)
                         | ((uint32_t)((bump_res + 1023) / 1024) & 0xFFFFu);
        std::vector<uint8_t> tab;
        auto put32 = [&](uint32_t v) {
            for (int i = 0; i < 4; i++) tab.push_back((uint8_t)(v >> (8 * i)));
        };
        put32((uint32_t)em.ddr_static_cap);
        put32(reserve);
        put32((uint32_t)em.ddr_temp_tab.size());
        for (const auto& [tid, offsz] : em.ddr_temp_tab) {
            uint32_t tagged_tid = tid;
            if (em.ddr_temp_vtcm.count(tid) && em.ddr_temp_vtcm.at(tid))
                tagged_tid |= WT_REF_VTCM_FLAG;
            put32(tagged_tid);
            put32(offsz.first);
            put32(offsz.second);
        }
        uint32_t sid = em.add_slot((uint32_t)tab.size(),
                                   (uint32_t)(tab.size() / 4), tab.data());
        em.slots[sid].addr = WT_SLOT_TEMPOFF;
        std::fprintf(stderr, "[ddr] TEMPOFF slot %u: cap=%llu vtcm_cap=%llu n_temps=%zu (vtcm=%zu)\n",
                     sid, (unsigned long long)em.ddr_static_cap,
                     (unsigned long long)em.ddr_vtcm_cap,
                     em.ddr_temp_tab.size(), em.ddr_temp_vtcm.size());
    }

    // 6. 组装 blob
    {
        uint32_t n_slots = (uint32_t)em.slots.size();
        uint32_t n_ops = (uint32_t)em.ops.size();
        /* WTOP v2(C4): 权重区(内联 weight_area 或外置 ext_weight_area)逼近/超
         * v1 的 u32 字节偏移上限(4GB)时, slot 线记录 16B→24B 且 offset 升 u64。
         * 内存侧 em.slots 恒为字节偏移, 仅此处线编码变化。小模型保持 v1 →
         * 存量 byte-exact 回归零影响。--force-v2 供小模型走 v2 通路做设备回归
         * (无需 4GB 资产)。 */
        const uint64_t v1_off_max = 0xFFFFFF00ull;  // 留 256B 护栏
        bool need_v2 = force_v2 ||
                       em.weight_area.size() > v1_off_max ||
                       em.ext_weight_area.size() > v1_off_max;
        uint16_t ver = need_v2 ? WT_BLOB_VER_2 : WT_BLOB_VER;
        const uint32_t slot_rec = need_v2 ? WT_SLOT_SIZE_V2 : WT_SLOT_SIZE;
        if (need_v2)
            std::fprintf(stderr, "[wtop] v2 格式: %s (weight_area=%zu ext=%zu), offset 升 u64\n",
                         force_v2 ? "force-v2 强制" : "权重区超 v1 偏移上限",
                         em.weight_area.size(), em.ext_weight_area.size());
        em.blob.resize(16 + (size_t)n_slots * slot_rec + 1, 0);
        std::memcpy(em.blob.data(), "WTOP", 4);
        uint16_t eck = WT_ENDIAN_CHK;
        std::memcpy(em.blob.data() + 4, &ver, 2);
        std::memcpy(em.blob.data() + 6, &eck, 2);
        std::memcpy(em.blob.data() + 8, &n_slots, 4);
        std::memcpy(em.blob.data() + 12, &n_ops, 4);
        for (uint32_t i = 0; i < n_slots; i++) {
            uint8_t* s = em.blob.data() + 16 + (size_t)i * slot_rec;
            std::memcpy(s, &em.slots[i].len, 4);
            std::memcpy(s + 4, &em.slots[i].count, 4);
            uint64_t off = em.slots[i].offset;
            if (need_v2) {
                uint32_t lo = (uint32_t)(off & 0xFFFFFFFFu), hi = (uint32_t)(off >> 32);
                std::memcpy(s + 8, &lo, 4);
                std::memcpy(s + 12, &hi, 4);
                std::memcpy(s + 16, &em.slots[i].addr, 4);
                uint32_t zero = 0;
                std::memcpy(s + 20, &zero, 4);
            } else {
                uint32_t off32 = (uint32_t)off;
                std::memcpy(s + 8, &off32, 4);
                std::memcpy(s + 12, &em.slots[i].addr, 4);
            }
        }
        size_t p = 16 + (size_t)n_slots * slot_rec;
        em.blob.resize(p);
        for (const auto& o : em.ops) {
            size_t sz = 4 + o.n_args * 4;
            size_t old = em.blob.size();
            em.blob.resize(old + sz);
            std::memcpy(em.blob.data() + old, &o.opcode, 2);
            std::memcpy(em.blob.data() + old + 2, &o.n_args, 2);
            for (uint16_t a = 0; a < o.n_args; a++)
                std::memcpy(em.blob.data() + old + 4 + a * 4, &o.args[a], 4);
            p += sz;
        }
        size_t woff = (p + 127) & ~(size_t)127;
        em.blob.resize(woff + em.weight_area.size(), 0);
        std::memcpy(em.blob.data() + woff, em.weight_area.data(), em.weight_area.size());
    }

    // 7. 自校验 + 落盘
    {
        wt_blob* wb = new wt_blob{};
        int rc = wt_parse(em.blob.data(), em.blob.size(), wb);
        if (rc != WT_OK) {
            std::fprintf(stderr, "error: self-validate failed: %s (blob=%zu weight_area=%zu)\n",
                         wt_err_str(rc), em.blob.size(), em.weight_area.size());
            uint64_t max_end = 0;
            for (auto& sl : em.slots)
                max_end = std::max<uint64_t>(max_end, (uint64_t)sl.offset + sl.len);
            std::fprintf(stderr, "  max slot end=%llu n_slots=%zu\n",
                         (unsigned long long)max_end, em.slots.size());
            return 3;
        }
        // 外置槽界自校验(wt_parse 只校验内联区, 不覆盖 ext 区)
        for (auto& sl : em.slots)
            if (sl.addr == WT_SLOT_EXT_WGT &&
                (uint64_t)sl.offset + sl.len > em.ext_weight_area.size()) {
                std::fprintf(stderr, "error: ext weight slot overruns ext area "
                             "(ext_area=%zu slot_end=%llu)\n", em.ext_weight_area.size(),
                             (unsigned long long)((uint64_t)sl.offset + sl.len));
                return 3;
            }
        std::printf("WTOP OK: slots=%u ops=%u bytes=%zu (gguf hits=%u miss=%u)\n",
                    (unsigned)em.slots.size(), (unsigned)em.ops.size(), em.blob.size(),
                    em.gguf_hits, em.gguf_miss);

    }
    {
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str()); return 2; }
        std::fwrite(em.blob.data(), 1, em.blob.size(), f);
        std::fclose(f);
    }

    // 7b. 路线B: 外置权重落盘(model.weights.bin; blob 仅描述符, EXT_WGT 槽)
    if (em.ext_weights && !em.ext_weight_area.empty()) {
        std::string wpath = out_path;
        size_t dot = wpath.rfind(".wtop");
        wpath = (dot == std::string::npos) ? wpath + ".weights.bin"
                                           : wpath.substr(0, dot) + ".weights.bin";
        FILE* f = std::fopen(wpath.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "error: cannot write %s\n", wpath.c_str()); return 2; }
        std::fwrite(em.ext_weight_area.data(), 1, em.ext_weight_area.size(), f);
        std::fclose(f);
        std::printf("[ext-weights] %zu bytes -> %s\n", em.ext_weight_area.size(), wpath.c_str());
    }

    // 8. manifest
    if (!manifest_path.empty()) {
        FILE* f = std::fopen(manifest_path.c_str(), "w");
        if (f) {
            /* 输出 temp = Output 节点输入解析到的 temp id(liveness 复用下
             * 末 op 输出可能落在低编号 temp, next_temp-1 是错的) */
            uint32_t out_temp = 0;
            const OpDef* out = gp.get_op_at(gp.get_output_node_id());
            if (out && !out->inputs.empty())
                out_temp = em.src_ref(out->inputs[0], gp.get_input_node_id(), gp, wslots);
            /* PROF W-P1: manifest v2 = v1 全字段 + op_names (runlist idx →
             * net.json 节点名, 经 emitted_ids → OpDef.name_tag)。名字只在
             * emit 时手里有, blob 不带; gehtp_prof.py join 用。 */
            std::fprintf(f, "{\n");
            std::fprintf(f, "  \"manifest_ver\": 2,\n");
            std::fprintf(f, "  \"input_slot\": 0,\n");
            std::fprintf(f, "  \"input_elems\": %zu,\n", input_elems);
            std::fprintf(f, "  \"output_temp\": %u,\n", out_temp & 0x7FFFu);
            std::fprintf(f, "  \"n_slots\": %u,\n  \"n_ops\": %u,\n",
                         (unsigned)em.slots.size(), (unsigned)em.ops.size());
            std::fprintf(f, "  \"op_ids\": [");
            for (size_t i = 0; i < em.emitted_ids.size(); i++)
                std::fprintf(f, "%s%llu", i ? "," : "", (unsigned long long)em.emitted_ids[i]);
            std::fprintf(f, "],\n  \"opcodes\": [");
            for (size_t i = 0; i < em.ops.size(); i++)
                std::fprintf(f, "%s%u", i ? "," : "", (unsigned)em.ops[i].opcode);
            std::fprintf(f, "],\n  \"op_names\": [");
            for (size_t i = 0; i < em.emitted_ids.size(); i++) {
                const OpDef* nod = gp.get_op_at(em.emitted_ids[i]);
                const char* nm = (nod && nod->name_tag && nod->name_tag->name())
                                 ? nod->name_tag->name() : "?";
                std::fprintf(f, "%s\"", i ? "," : "");   /* 分隔符仅逗号; 此处的 " 是开引号 */
                for (const char* p = nm; *p; p++) {   /* JSON 转义最小集 */
                    unsigned char ch = (unsigned char)*p;
                    if (ch == '"' || ch == '\\') std::fprintf(f, "\\%c", ch);
                    else if (ch < 0x20) std::fprintf(f, "\\u%04x", ch);
                    else std::fputc(ch, f);
                }
                std::fprintf(f, "\"");
            }
            std::fprintf(f, "]\n}\n");
            std::fclose(f);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string bin_path, in_f16, out_path, manifest_path, gguf_path, match_path, ext_weights_path;
    bool ext_weights = false;  // --ext-weights: 路线B 权重外置(blob 描述符 + model.weights.bin)
    bool force_v2 = false;     // --force-v2: 小模型强制 v2 槽记录(设备回归 v2 通路用)
    bool emit_dma = false;     // --emit-dma: 溢出搬移发射真 OP_DMA(非 SPILL/FILL)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--bin") bin_path = next();
        else if (a == "--list-ops") {
            std::vector<std::string> names;
            for (const auto& kv : wtop::op_registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (const auto& n : names) std::printf("%s\n", n.c_str());
            return 0;
        }
        else if (a == "--input-f16") in_f16 = next();
        else if (a == "--out") out_path = next();
        else if (a == "--manifest") manifest_path = next();
        else if (a == "--gguf") gguf_path = next();
        else if (a == "--match") match_path = next();
        else if (a == "--ext-weights") ext_weights = true;
        else if (a == "--weights-bin") ext_weights_path = next();
        else if (a == "--force-v2") force_v2 = true;
        else if (a == "--emit-dma") emit_dma = true;
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (bin_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: wtop_emit --bin <tagged.bin> [--input-f16 <f16.raw>] --out <blob.wtop> [--manifest <json>] [--gguf <g> --match <tsv>] [--ext-weights [--weights-bin <params.bin>]] [--emit-dma]\n");
        return 2;
    }
    return emit(bin_path, in_f16, out_path, manifest_path, gguf_path, match_path,
                ext_weights, ext_weights_path, force_v2, emit_dma);
}
