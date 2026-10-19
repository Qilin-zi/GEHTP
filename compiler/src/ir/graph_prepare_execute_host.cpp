#include <cstdio>
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/scalar_params.hpp"
#include "hnnx/ir/op_extra.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/pass_view_fold.hpp"  // 005: dump_view_census
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/cost/cost_model.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/host_decode_state.hpp"
#include "hnnx/mcast/mcast_optimizer.hpp"
#include "hnnx/scheduler/dp_sequencer.hpp"
#include "hnnx/vtcm/dp_group_graph.hpp"
#include "hnnx/vtcm/supertile.hpp"
#include "hnnx/ir/graph_deps.hpp"
#include "hnnx/schedule/scheduler.hpp"  // Scheduler(ST-Cut 计划序, do_prepare2 计算)
#include "hnnx/tiling/conv_tiling.hpp"  // 阶段6: conv 空间分块(halo 推导)
#include <algorithm>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <cstring>

namespace hnnx {

// Host reference execution: execute_host()

GraphPrepare::ExecResult GraphPrepare::execute_host(
        const std::vector<float>& input,
        const std::vector<std::vector<float>>* multi_inputs) {
    ExecResult ret{};
    ops_passthru_reset();   /* A6: 每次 host 执行前清零直通计数 */
    /* 004 T018c: decode 持久状态落盘协议 — GEHTP_HOST_STATE=<path>:
     * 进程首步装载(不存在=零态新序列), 末尾覆写。多步驱动 = 逐进程串跑。 */
    static const char* hstate_path = std::getenv("GEHTP_HOST_STATE");
    if (hstate_path && *hstate_path) {
        int rc = host_decode_state_load(hstate_path);
        if (rc < 0)
            std::fprintf(stderr, "[hostdec] state file %s corrupt, starting fresh\n",
                         hstate_path);
    }

    /* ops_ 在 do_prepare2 的 prepare_op 填充后, Phase7 fixpoint 的 DCE 仍会
     * 改 opdef_map_(删 op)——ops_ 保持陈旧(probe: Slice 被删但 TypicalOp
     * 照跑, tensor_param 注入/输入链全断)。以 opdef_map_ 为准重建。 */
    ops_.clear();
    /* T018d: GEHTP_DBG_T18_IDS="153,155" 打指定 op 的执行期输入边(名字+源) */
    const char* dbg_ids = std::getenv("GEHTP_DBG_T18_IDS");
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
        if (!opdef->name_tag || !opdef->name_tag->name()) continue;
        if (dbg_ids && *dbg_ids) {
            char pat[32];
            std::snprintf(pat, sizeof pat, ",%llu,", (unsigned long long)id);
            std::string hay = std::string(",") + dbg_ids + ",";
            if (std::strstr(hay.c_str(), pat)) {
                std::fprintf(stderr, "[t18-edge] %s id=%llu n_in=%zu odata=%zu",
                             opdef->name_tag->name(), (unsigned long long)id,
                             opdef->inputs.size(), opdef->op_data.size());
                {
                    auto pv = unpack_scalar_params(opdef->op_data);
                    for (const auto& p2 : pv)
                        std::fprintf(stderr, " %s=%s", p2.name.c_str(),
                                     p2.is_numeric ? std::to_string(p2.value_num).c_str()
                                                   : p2.value_str.c_str());
                }
                for (const auto& c : opdef->inputs) {
                    OpDef* s = get_op_at(c.src_id);
                    std::fprintf(stderr, " <-%llu:%s(r%u,%u,%u,%u)",
                                 (unsigned long long)c.src_id,
                                 s && s->name_tag && s->name_tag->name()
                                     ? s->name_tag->name() : "MISSING",
                                 s ? s->output_def.dims[0] : 0,
                                 s ? s->output_def.dims[1] : 0,
                                 s ? s->output_def.dims[2] : 0,
                                 s ? s->output_def.dims[3] : 0);
                }
                std::fprintf(stderr, "\n");
            }
        }
        if (getenv("GEHTP_DBG_T18")) {
            const char* dn = opdef->name_tag->name();
            if (std::strstr(dn, "KvAppend") || std::strstr(dn, "KvGather") ||
                std::strstr(dn, "Conv1dSsmStep") || std::strstr(dn, "DeltaChunk") ||
                std::strstr(dn, "DecodeFace")) {
                std::fprintf(stderr, "[t18-ex] %s id=%llu op_data=%zu", dn,
                             (unsigned long long)id, opdef->op_data.size());
                auto spv = unpack_scalar_params(opdef->op_data);
                for (const auto& pv : spv)
                    std::fprintf(stderr, " %s=%g", pv.name.c_str(),
                                 pv.is_numeric ? pv.value_num
                                               : std::strtod(pv.value_str.c_str(), nullptr));
                for (const auto& c : opdef->inputs) {
                    OpDef* s = get_op_at(c.src_id);
                    std::fprintf(stderr, " <-%llu%s", (unsigned long long)c.src_id,
                                 s ? "" : "(MISSING)");
                }
                std::fprintf(stderr, "\n");
            }
        }
        OpIoPtrs io{};
        io.graph_prepare = this;
        io.opdef_ptr = opdef.get();
        auto op = op_factory_generate(io, id);
        if (op) {
            ops_.push_back(std::move(op));
        } else {
            /* A6: factory-null 合法类仅 $Const/Input/Output/const 节点;
             * 非 const 计算 op 未注册 → 静默跳过 = 比直通更坏的暗洞
             * (下游读空 buf, BogusOpXYZ 阳性对照实锤) — 登记治理。 */
            const char* nm = opdef->name_tag->name();
            const bool legit = opdef->is_const()
                || !std::strcmp(nm, "$Const") || !std::strcmp(nm, "Input")
                || !std::strcmp(nm, "Output");
            if (!legit)
                ops_passthru_note(std::string("factory-null:") + nm);
            if (getenv("GEHTP_EXDIAG")) {
                std::fprintf(stderr, "[skip] oid=%llu factory-null %s const=%d dead=%d out(r%u:%u,%u,%u)\n",
                             (unsigned long long)id, nm,
                             (int)opdef->is_const(), (int)opdef->is_dead(),
                             opdef->output_def.rank, opdef->output_def.dims[0],
                             opdef->output_def.dims[1], opdef->output_def.dims[2]);
            }
        }
    }

    // Input/Output are boundary nodes (not in ops_). Read their info from opdef_map_.
    op_id_t input_id = get_input_node_id();
    op_id_t output_id = get_output_node_id();
    if (input_id == 0 || output_id == 0) return ret;

    // Read input dims from Input's OpDef.
    size_t input_n = 1;
    {
        auto* input_def = get_op_at(input_id);
        if (input_def) {
            for (uint32_t i = 0; i < input_def->output_def.rank && i < 5; ++i)
                input_n *= static_cast<size_t>(input_def->output_def.dims[i]);
        }
    }

    // tensor_map: op_id -> float buffer
    std::unordered_map<op_id_t, std::vector<float>> tensor_map;
    /* 多输入: Input 槽位序 = host_input_order()(声明序优先, id 序回退);
     * multi 为空时单 input 喂第一个 Input(向后兼容)。 */
    std::vector<op_id_t> in_ids = host_input_order();
    if (in_ids.empty()) in_ids.push_back(input_id);
    if (getenv("GEHTP_EXDIAG")) {
        std::fprintf(stderr, "[mdiag] in_ids:");
        for (op_id_t iid : in_ids) std::fprintf(stderr, " %llu", (unsigned long long)iid);
        std::fprintf(stderr, " multi=%zu input=%zu\n",
                     multi_inputs ? multi_inputs->size() : 0, input.size());
        std::fprintf(stderr, "[mdiag] all nodes:");
        for (auto& [id, opdef] : opdef_map_)
            if (opdef)
                std::fprintf(stderr, " %llu:%s%s%s%s", (unsigned long long)id,
                             opdef->name_tag && opdef->name_tag->name() ? opdef->name_tag->name() : "?",
                             opdef->is_const() ? "[c]" : "",
                             opdef->is_enabled() ? "" : "[dis]",
                             opdef->is_dead() ? "[DEAD]" : "");
        std::fprintf(stderr, "\n");
    }
    for (size_t k = 0; k < in_ids.size(); k++) {
        op_id_t iid = in_ids[k];
        const std::vector<float>* src = nullptr;
        if (multi_inputs && k < multi_inputs->size()) src = &(*multi_inputs)[k];
        else if (k == 0 && !input.empty()) src = &input;
        auto* idef = get_op_at(iid);
        size_t in_n = 1;
        if (idef)
            for (uint32_t i = 0; i < idef->output_def.rank && i < 5; ++i)
                in_n *= static_cast<size_t>(idef->output_def.dims[i]);
        auto& buf = tensor_map[iid];
        buf.resize(in_n > 0 ? in_n : 1, 0.0f);
        if (src) {
            /* i64 输入文件: nbytes/4 = 2×in_n 个字 = [lo,hi] 交织 —
             * 线性取前 in_n 会把前半 tokens 的高字(恒 0)掺进来
             * (0.8B 全图实锤: EQUAL(x,0) 掩码链输出严格交替 [0,1,0,1],
             *  奇数行 causal mask 全 -inf → softmax NaN)。
             * 收缩 = 取每个 int64 的低 32 位位模式(token id < 2^31)。 */
            if (src->size() == 2 * in_n) {
                const uint32_t* w = reinterpret_cast<const uint32_t*>(src->data());
                for (size_t i = 0; i < in_n; i++) {
                    uint32_t lo = w[i * 2];
                    std::memcpy(&buf[i], &lo, 4);
                }
            } else {
                for (size_t i = 0; i < in_n && i < src->size(); ++i)
                    buf[i] = (*src)[i];
            }
        }
    }

    // Materialize const ops: copy their data from const_pool_ into tensor_map.
    // Const ops are not in ops_ (not registered), so we populate their outputs
    // here before the compute-op loop reads them.
    // 按 element_size 拓宽到 float: bool(1B)/f16(2B)/int32(4B)/f32(4B)
    // (GDN Trilu 掩码 = converter 折叠的 bool 静态张量, 4096B≠1024 float)。
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_const() || opdef->is_dead()) continue;
        if (opdef->const_data_size == 0) continue;
        size_t es = opdef->output_def.element_size;
        if (es == 0 || es > 4) es = 4;
        /* 物理宽度优先: adapter 把 f16 静态张量拓宽成 f32 存储, 但
         * output_def.element_size 仍按 JSON dt 记 2 —— 按 es=2 读 f32
         * 字节会把每个 f32 错读成 2 个 f16(0.8B causal mask 实锤:
         * 读出 2.75/NaN 垃圾)。以 const_data_size/∏dims 反推真实宽度,
         * 与声明不符时以数据为准。 */
        {
            size_t dn = 1;
            for (uint32_t d = 0; d < opdef->output_def.rank && d < 5; ++d)
                dn *= (size_t)opdef->output_def.dims[d];
            if (dn > 0) {
                if (opdef->const_data_size == dn * 4) es = 4;
                else if (opdef->const_data_size == dn * 2) es = 2;
                else if (opdef->const_data_size == dn) es = 1;
            }
        }
        size_t elem_n = opdef->const_data_size / es;
        auto& buf = tensor_map[id];
        buf.resize(elem_n, 0.0f);
        bool in_pool = opdef->const_data_offset + opdef->const_data_size <= const_pool_.size();
        const uint8_t* src = in_pool ? const_pool_.data() + opdef->const_data_offset : nullptr;
        if (es == 4) {
            if (src) std::memcpy(buf.data(), src, opdef->const_data_size);
        } else if (es == 2) {
            for (size_t i = 0; i < elem_n; i++) {
                if (!src) break;
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                // f16 → f32(次正规/Inf 保真)
                uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
                uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, u;
                if (exp == 0) {
                    if (mant == 0) u = sign;
                    else {
                        int e = -1;
                        while (!(mant & 0x400)) { mant <<= 1; e--; }
                        u = sign | ((uint32_t)(114 + e) << 23) | ((mant & 0x3FF) << 13);
                    }
                } else if (exp == 31) u = sign | 0x7F800000u | (mant << 13);
                else u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
                float f;
                std::memcpy(&f, &u, 4);
                buf[i] = f;
            }
        } else {  // es == 1: bool/uint8 → float
            for (size_t i = 0; i < elem_n && src; i++)
                buf[i] = (float)src[i];
        }
        if (getenv("GEHTP_EXDIAG")) {
            std::fprintf(stderr, "[const] id=%llu es=%zu size=%zu off=%llu elem=%zu rank=%u dims=[%u,%u,%u,%u] first4=%g,%g,%g,%g\n",
                         (unsigned long long)id, es, (size_t)opdef->const_data_size,
                         (unsigned long long)opdef->const_data_offset, elem_n,
                         opdef->output_def.rank,
                         opdef->output_def.dims[0], opdef->output_def.dims[1],
                         opdef->output_def.dims[2], opdef->output_def.dims[3],
                         buf.size() > 0 ? buf[0] : 0.f, buf.size() > 1 ? buf[1] : 0.f,
                         buf.size() > 2 ? buf[2] : 0.f, buf.size() > 3 ? buf[3] : 0.f);
        }
    }

    // 拓扑排序(通用): ops_ 源自 opdef_map_ 迭代(unordered, 序不可靠),
    // 必须按输入依赖序执行(Kahn)。环图回退到原序。
    std::vector<const TypicalOp*> topo;
    {
        std::vector<const TypicalOp*> cands;
        std::unordered_map<op_id_t, size_t> indeg;
        std::unordered_map<op_id_t, std::vector<op_id_t>> succ;
        for (auto& op_ptr : ops_) {
            auto* t = dynamic_cast<const TypicalOp*>(op_ptr.get());
            if (!t || t->op_id == 0) continue;
            cands.push_back(t);
            indeg[t->op_id] = 0;
        }
        for (auto* t : cands)
            for (const auto& c : t->exec_inputs) {
                if (indeg.count(c.src_id)) {
                    ++indeg[t->op_id];
                    succ[c.src_id].push_back(t->op_id);
                }
            }
        std::vector<op_id_t> ready;
        for (auto* t : cands)
            if (indeg[t->op_id] == 0) ready.push_back(t->op_id);
        std::sort(ready.begin(), ready.end());  /* 确定性(与 emit 对拍同序) */
        std::unordered_map<op_id_t, const TypicalOp*> by_id;
        for (auto* t : cands) by_id[t->op_id] = t;
        while (!ready.empty()) {
            op_id_t id = ready.back();
            ready.pop_back();
            topo.push_back(by_id[id]);
            for (op_id_t s : succ[id])
                if (--indeg[s] == 0) ready.push_back(s);
        }
        if (topo.size() != cands.size()) {  // 环: 回退原序(正确性由上游保证)
            topo.clear();
            topo = cands;
        }
    }

    // Iterate ops_ (compute ops only: Relu, Conv, Add, etc.).
    // Input/Output are not in ops_ (not registered).
    size_t seq_idx = 0;
    static const bool optrace = getenv("GEHTP_OPTRACE") != nullptr;
    for (const TypicalOp* top : topo) {
        op_id_t oid = top->op_id;
        if (oid == 0) continue;
        if (optrace) {
            std::fprintf(stderr, "[optrace] seq=%zu id=%llu %s ins=",
                         seq_idx, (unsigned long long)oid, top->op_type_name.c_str());
            for (const auto& c : top->exec_inputs)
                std::fprintf(stderr, "%llu ", (unsigned long long)c.src_id);
            std::fprintf(stderr, "\n");
        }
        /* 早停: GEHTP_STOP_SEQ=N 执行到 seq==N 后中断(NaN 溯源诊断用,
         * 输出不可用但 dump 已落盘) */
        static const char* stop_seq_s = getenv("GEHTP_STOP_SEQ");
        if (stop_seq_s && seq_idx >= (size_t)std::strtoull(stop_seq_s, nullptr, 10)) {
            std::fprintf(stderr, "[optrace] stop at seq=%zu\n", seq_idx);
            return ret;
        }

        // Gather input buffers and their OutputDefs from predecessors.
        std::vector<const uint8_t*> in_bufs;
        std::vector<OutputDef> in_ods;
        for (const auto& conn : top->exec_inputs) {
            auto it = tensor_map.find(conn.src_id);
            in_bufs.push_back(it != tensor_map.end() && !it->second.empty()
                ? reinterpret_cast<const uint8_t*>(it->second.data()) : nullptr);
            in_ods.push_back(conn.src_out_def);
            if (getenv("GEHTP_EXDIAG")) {
                size_t bsz = (it != tensor_map.end()) ? it->second.size() : 0;
                const OpDef* sp = get_op_at(conn.src_id);
                std::fprintf(stderr, "[exsrc] oid=%llu in%zu <- src=%llu bufsz=%zu decl(r%u:%u,%u,%u,%u) srcdef(r%u:%u,%u,%u,%u) %s\n",
                             (unsigned long long)oid, in_ods.size() - 1,
                             (unsigned long long)conn.src_id, bsz,
                             conn.src_out_def.rank, conn.src_out_def.dims[0],
                             conn.src_out_def.dims[1], conn.src_out_def.dims[2],
                             conn.src_out_def.dims[3],
                             sp ? sp->output_def.rank : 0,
                             sp ? sp->output_def.dims[0] : 0,
                             sp ? sp->output_def.dims[1] : 0,
                             sp ? sp->output_def.dims[2] : 0,
                             sp ? sp->output_def.dims[3] : 0,
                             sp && sp->name_tag && sp->name_tag->name()
                                 ? sp->name_tag->name() : "?");
            }
        }
        /* tensor_param const(ranges/perm/axes)追加到输入尾部 —
         * host 参考 StridedSlice/Transpose 等按名/按位消费
         * (inject 未把它们进 op inputs 的形态下同样可见) */
        if (const OpDef* od2 = get_op_at(oid)) {
            for (op_id_t tpid : od2->tensor_param_ids) {
                const OpDef* pc = get_op_at(tpid);
                if (!pc || pc->const_data_size == 0) continue;
                auto& tbuf = tensor_map[tpid];
                size_t en = pc->const_data_size / sizeof(float);
                if (tbuf.empty() && en > 0) {
                    tbuf.resize(en, 0.0f);
                    /* offset 0 合法(池首 const) */
                    if (pc->const_data_offset + pc->const_data_size <= const_pool_.size())
                        std::memcpy(tbuf.data(),
                                    const_pool_.data() + pc->const_data_offset,
                                    pc->const_data_size);
                }
                if (true) {
                    std::fprintf(stderr, "[hostdiag] op %llu param %llu off=%llu size=%llu first4=%g %g %g %g rank=%u dims=[%u,%u,%u,%u]\n",
                                 (unsigned long long)oid, (unsigned long long)tpid,
                                 (unsigned long long)pc->const_data_offset,
                                 (unsigned long long)pc->const_data_size,
                                 tbuf.size() > 0 ? tbuf[0] : -1.f, tbuf.size() > 1 ? tbuf[1] : -1.f,
                                 tbuf.size() > 2 ? tbuf[2] : -1.f, tbuf.size() > 3 ? tbuf[3] : -1.f,
                                 pc->output_def.rank,
                                 pc->output_def.dims[0], pc->output_def.dims[1],
                                 pc->output_def.dims[2], pc->output_def.dims[3]);
                }
                in_bufs.push_back(reinterpret_cast<const uint8_t*>(tbuf.data()));
                in_ods.push_back(pc->output_def);
            }
        }

        // Allocate output buffer.
        size_t out_n = 1;
        for (uint32_t i = 0; i < top->cached_out_def.rank && i < 5; ++i)
            out_n *= static_cast<size_t>(top->cached_out_def.dims[i]);
        if (out_n == 0) out_n = 1;

        auto& out_vec = tensor_map[oid];
        out_vec.resize(out_n, 0.0f);

        if (getenv("GEHTP_EXDIAG")) {
            std::fprintf(stderr, "[ex] seq=%zu oid=%llu %s n=%zu ins=%zu",
                         seq_idx, (unsigned long long)oid,
                         top->op_type_name.c_str(), out_n, in_bufs.size());
            for (size_t k = 0; k < in_ods.size(); k++)
                std::fprintf(stderr, " src(r%u:%u,%u,%u,%u)",
                             in_ods[k].rank, in_ods[k].dims[0],
                             in_ods[k].dims[1], in_ods[k].dims[2],
                             in_ods[k].dims[3]);
            std::fprintf(stderr, "\n");
        }
        top->execute(in_bufs, reinterpret_cast<uint8_t*>(out_vec.data()), top->cached_out_def, in_ods);
        seq_idx++;
        if (oid == 3 || oid == 14 || oid == 45 || oid == 151) {
            std::fprintf(stderr, "[hostdiag] op %llu out[0..3]=%g %g %g %g\n",
                         (unsigned long long)oid,
                         out_vec.size() > 0 ? out_vec[0] : 0.f,
                         out_vec.size() > 1 ? out_vec[1] : 0.f,
                         out_vec.size() > 2 ? out_vec[2] : 0.f,
                         out_vec.size() > 3 ? out_vec[3] : 0.f);
        }
        /* 全 op 按 op_id dump(与设备 dump 经 manifest 映射对拍)。
         * GEHTP_DUMPDIR 指定目录(多会话共享 /tmp 会串台, 诊断须独占) */
        {
            static const char* dd = getenv("GEHTP_DUMPDIR");
            static const std::string dpre = (dd && *dd) ? std::string(dd) + "/" : std::string("/tmp/");
            char pn2[512];
            std::snprintf(pn2, sizeof(pn2), "%shost_seq_%zu.f32.raw", dpre.c_str(), seq_idx);
            char pn3[512];
            std::snprintf(pn3, sizeof(pn3), "%shost_id_%llu.f32.raw", dpre.c_str(), (unsigned long long)oid);
            std::FILE* fo3 = std::fopen(pn3, "wb");
            if (fo3) {
                std::fwrite(out_vec.data(), 4, out_vec.size(), fo3);
                std::fclose(fo3);
            }
            /* host_seq 只在显式开 GEHTP_SEQDUMP 时写(大图 14821 文件 I/O
             * 是纯开销; M5 全图诊断曾因此翻倍时长) */
            if (getenv("GEHTP_SEQDUMP")) {
                std::FILE* fo2 = std::fopen(pn2, "wb");
                if (fo2) {
                    std::fwrite(out_vec.data(), 4, out_vec.size(), fo2);
                    std::fclose(fo2);
                }
            }
        }
        /* 中间对拍 dump(与 ORT 同名输出逐元素比) */
        if (oid == 22 || oid == 25 || oid == 40 || oid == 42 || oid == 43 || oid == 44 || oid == 53 || oid == 56 || oid == 58 || oid == 63 || oid == 67 || oid == 69 || oid == 70 || oid == 72 || oid == 77 || oid == 8 || oid == 10 || oid == 12 ||
            oid == 13 || oid == 29 ||
            oid == 38 || oid == 45 ||
            oid == 51 || oid == 63 || oid == 65 || oid == 81 ||
            oid == 148 || oid == 151) {
            char pn[64];
            std::snprintf(pn, sizeof(pn), "/tmp/host_op_%llu.f32.raw", (unsigned long long)oid);
            std::FILE* fo = std::fopen(pn, "wb");
            if (fo) {
                std::fwrite(out_vec.data(), 4, out_vec.size(), fo);
                std::fclose(fo);
            }
        }
    }

    // Read Output's predecessor from opdef_map_ (Output is not in ops_).
    auto* output_def = get_op_at(output_id);
    if (output_def && !output_def->inputs.empty()) {
        if (getenv("GEHTP_EXDIAG")) {
            const OpDef* sp = get_op_at(output_def->inputs[0].src_id);
            std::fprintf(stderr, "[outdiag] src=%llu %s dead=%d\n",
                         (unsigned long long)output_def->inputs[0].src_id,
                         sp && sp->name_tag && sp->name_tag->name() ? sp->name_tag->name() : "?",
                         sp ? (int)sp->is_dead() : -1);
        }
        auto src_it = tensor_map.find(output_def->inputs[0].src_id);
        if (src_it != tensor_map.end()) {
            ret.output = src_it->second;
            ret.ok = true;
        }
    }

    ops_passthru_report("execute_host");   /* A6: 直通 N≠0 即假绿报警 */
    if (hstate_path && *hstate_path) {
        if (host_decode_state_save(hstate_path) != 0)
            std::fprintf(stderr, "[hostdec] state save to %s failed\n", hstate_path);
    }
    return ret;
}
} // namespace hnnx
