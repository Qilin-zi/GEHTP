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

GraphPrepare::GraphPrepare() {
    graph_dirty_ = false;
    serialized_loaded_ = false;
    force_barrel_ = false;
    allocator_ = nullptr;
    op_registry_ = nullptr;
    graph_deps_ = nullptr;
    early_out_flag_ = 0;
    memory_alloc_limit_mb_ = 0;
    next_op_id_ = 1;
    construction_state_ = 1; // +0x45dc = 1 means "in construction phase"
    input_node_id_ = 0;     // +0x5340
    output_node_id_ = 0;    // +0x5348
}

GraphPrepare::~GraphPrepare() = default;

// Main prepare entry point
// Source: graph_prepare.cc, prepare @ 0xF80840 (699 bytes)
GraphStatus GraphPrepare::prepare(HexagonNNEnv& env) {
    // 1. do_prepare1: initial graph building and basic optimization
    // 2. do_prepare2: full optimization including tiling, DMA, sequencing
    // 3. do_prepare2_late: late-stage optimizations and runlist generation
    // 阶段7: VTCM 预算可覆盖(0 = 默认 8MB); 小预算强制分配器溢出 → spill/fill
    VtcmCacheInstance vtcm(0, vtcm_budget_override_ ? vtcm_budget_override_
                                                      : 8 * 1024 * 1024);
    int retry_count = 0;
    GraphStatus s1 = do_prepare1(env, vtcm);
    if (s1 != GraphStatus::Success) return s1;
    GraphStatus s2 = do_prepare2(env, vtcm, retry_count, true);
    if (s2 != GraphStatus::Success) return s2;
    std::vector<uint32_t> runlist_tags;
    GraphStatus s3 = do_prepare2_late(runlist_tags);
    return s3;
}

// HtpPrepare auto-injection: per-op-type 输入构造。
// 对应真实库 HtpPrepare 的 graph import 阶段: 把 composeGraphs 产出的原始图
// (op inputs 只含 data) 转成 before_graph 的注入后图(含 quant_marker/scale/
// output_rank/perm 等)。注入的合成 const 用高地址段 id(0x10000+), 与
// composeGraphs 的低地址段 id 分离; 相同签名(dims+data)的合成 const 去重共享。
// tensor_param(perm/axes/stride/pad/dilation)已在 build_graph 用低 id 创建为
// 独立 const 节点, 这里只把它们从 OpDef::tensor_param_ids 追加到 op inputs。
void GraphPrepare::inject_htp_prepare_inputs() {
    // 合成 const id 从 0x10000 起步, 与 composeGraphs 低 id 分离
    op_id_t next_injected_id = 0x10000;
    // 签名 -> op_id 去重(quant_marker/scale/output_rank/wscale/iscale)
    std::map<std::string, op_id_t> dedup;
    // const_by_N: 复用已有的 1D [N] const 作为 per-axis scale(QNN 行为)
    std::map<uint32_t, op_id_t> const_by_N;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_const()) continue;
        const OutputDef& od = opdef->output_def;
        if (od.dtype != 562) continue;  // float32
        if (od.rank != 1 || od.dims[0] <= 1) continue;
        uint32_t N = od.dims[0];
        if (const_by_N.find(N) == const_by_N.end()) const_by_N[N] = id;
    }

    // Helper: get/create a shared injected const by dedup key
    auto get_or_create = [&](const std::string& key, uint32_t dt,
                             uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                             uint32_t rank, uint32_t elem_size) -> op_id_t {
        auto it = dedup.find(key);
        if (it != dedup.end()) return it->second;
        op_id_t id = next_injected_id++;
        OutputDef od{};
        od.rank = rank; od.dtype = dt;
        od.dims[0] = d0; od.dims[1] = d1; od.dims[2] = d2; od.dims[3] = d3;
        od.element_size = elem_size;
        uint32_t total = d0 * d1 * d2 * d3;
        std::vector<uint8_t> zdata(total * elem_size, 0);
        append_const_node(static_cast<uint32_t>(id), od, zdata.data(), zdata.size());
        dedup[key] = id;
        return id;
    };
    // Helper: add an input by op_id
    auto add_input = [](OpDef* op, op_id_t id) {
        InputConn ic{};
        ic.src_id = id;
        op->inputs.push_back(ic);
    };

    // 快照 keys 再遍历: 循环体内 get_or_create/append_const_node 会向
    // opdef_map_ 插入合成 const, rehash 会使范围 for 迭代器失效 (UB,
    // 实测表现为遍历跳过 Transpose/Input、重复访问 Eltwise_Binary)。
    std::vector<op_id_t> op_ids;
    op_ids.reserve(opdef_map_.size());
    for (auto& [id, _] : opdef_map_) op_ids.push_back(id);
    for (op_id_t id : op_ids) {
        auto it = opdef_map_.find(id);
        if (it == opdef_map_.end()) continue;
        OpDef* opdef = it->second.get();
        if (!opdef || opdef->is_const() || opdef->is_dead()) continue;
        std::string nm = opdef->name_tag ? opdef->name_tag->name() : "";
        if (nm.empty() || nm == "Input" || nm == "Output") continue;

        // Per-op-type injection (order matters!)
        if (nm == "Conv2d") {
            uint32_t N = 0;
            if (opdef->inputs.size() > 1) {
                auto* w = get_op_at(opdef->inputs[1].src_id);
                if (w && w->output_def.rank >= 4) N = w->output_def.dims[3];
            }
            if (opdef->inputs.size() <= 2 && N > 0) {
                auto cit = const_by_N.find(N);
                op_id_t ws = (cit != const_by_N.end()) ? cit->second
                    : get_or_create("wscale_" + std::to_string(N), 562, 1,1,1,N, 4, 4);
                add_input(opdef, ws);
            }
            op_id_t stride_id = 0, pad_id = 0, dilation_id = 0;
            for (op_id_t tpid : opdef->tensor_param_ids) {
                OpDef* tp = get_op_at(tpid);
                if (!tp || !tp->name_tag) continue;
                std::string pn = tp->name_tag->name() ? tp->name_tag->name() : "";
                if (pn.find("stride") != std::string::npos) stride_id = tpid;
                else if (pn.find("pad_amount") != std::string::npos) pad_id = tpid;
                else if (pn.find("dilation") != std::string::npos) dilation_id = tpid;
            }
            if (stride_id) add_input(opdef, stride_id);
            if (pad_id) add_input(opdef, pad_id);
            add_input(opdef, get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));
            if (dilation_id) add_input(opdef, dilation_id);
            add_input(opdef, get_or_create("quant_marker", 1032, 1,1,1,1, 4, 4));

        } else if (nm == "Transpose") {
            // perm 是 tensor_param → 注入为 inputs[1](实测: 不注入则 perm const
            // 在 DCE 被杀, 序列化/执行丢转置语义)
            for (op_id_t tpid : opdef->tensor_param_ids) {
                if (opdef->inputs.size() >= 2) break;
                add_input(opdef, tpid);
            }
        } else if (nm == "DepthWiseConv2d") {
            uint32_t N = 0;
            if (opdef->inputs.size() > 1) {
                auto* w = get_op_at(opdef->inputs[1].src_id);
                if (w && w->output_def.rank >= 4) N = w->output_def.dims[3];
            }
            if (opdef->inputs.size() <= 2 && N > 0) {
                auto cit = const_by_N.find(N);
                op_id_t ws = (cit != const_by_N.end()) ? cit->second
                    : get_or_create("wscale_" + std::to_string(N), 562, 1,1,1,N, 4, 4);
                add_input(opdef, ws);
            }
            op_id_t stride_id = 0, pad_id = 0, dilation_id = 0;
            for (op_id_t tpid : opdef->tensor_param_ids) {
                OpDef* tp = get_op_at(tpid);
                if (!tp || !tp->name_tag) continue;
                std::string pn = tp->name_tag->name() ? tp->name_tag->name() : "";
                if (pn.find("stride") != std::string::npos) stride_id = tpid;
                else if (pn.find("pad_amount") != std::string::npos) pad_id = tpid;
                else if (pn.find("dilation") != std::string::npos) dilation_id = tpid;
            }
            if (stride_id) add_input(opdef, stride_id);
            if (pad_id) add_input(opdef, pad_id);
            if (dilation_id) add_input(opdef, dilation_id);

        } else if (nm == "FullyConnected") {
            if (opdef->inputs.size() <= 2) {
                uint32_t N = 0;
                if (opdef->inputs.size() > 1) {
                    auto* w = get_op_at(opdef->inputs[1].src_id);
                    if (w && w->output_def.rank >= 4) N = w->output_def.dims[2];
                    else if (w && w->output_def.rank >= 1) N = w->output_def.dims[0];
                }
                if (N > 0) {
                    auto cit = const_by_N.find(N);
                    op_id_t ws = (cit != const_by_N.end()) ? cit->second
                        : get_or_create("wscale_" + std::to_string(N), 562, 1,1,1,N, 4, 4);
                    add_input(opdef, ws);
                }
            }
            add_input(opdef, get_or_create("quant_marker", 1032, 1,1,1,1, 4, 4));

        } else if (nm == "MatMul") {
            uint32_t N = 0;
            if (opdef->inputs.size() > 1) {
                auto* w = get_op_at(opdef->inputs[1].src_id);
                if (w && w->output_def.rank >= 1) N = w->output_def.dims[w->output_def.rank - 1];
            }
            if (N > 0) {
                auto cit = const_by_N.find(N);
                op_id_t ws = (cit != const_by_N.end()) ? cit->second
                    : get_or_create("wscale_" + std::to_string(N), 562, 1,1,1,N, 4, 4);
                add_input(opdef, ws);
            }
            op_id_t qm = get_or_create("quant_marker", 1032, 1,1,1,1, 4, 4);
            add_input(opdef, qm);
            add_input(opdef, qm);

        } else if (nm == "ElementWiseNeuron") {
            add_input(opdef, get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));
            op_id_t sa = get_or_create("sc_ewn_a", 562, 1,1,1,1, 4, 4);
            op_id_t sb = get_or_create("sc_ewn_b", 562, 1,1,1,1, 4, 4);
            for (int i = 0; i < 4; i++) add_input(opdef, sa);
            add_input(opdef, sb);

        } else if (nm == "LayerNorm") {
            add_input(opdef, get_or_create("sc_" + nm, 562, 1,1,1,1, 4, 4));
            add_input(opdef, get_or_create("or_Softmax", 50, 1,1,1,1, 4, 4));

        } else if (nm == "Softmax") {
            add_input(opdef, get_or_create("sc_" + nm, 562, 1,1,1,1, 4, 4));
            add_input(opdef, get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));

        } else if (nm == "Gather") {
            // [table, indices] — 我方设备契约保持原样。REQNN 重构版曾
            // resize(1) + 双 or_Gather 注入: 运行索引输入被替换成常量 →
            // 全图被 const_prop 折叠 → emit 产出空 blob(0.8B 无 Gather
            // 未暴露, qnn2layer probe 首撞, M3c 实锤)。
        } else if (nm == "ElementWiseBinary") {
            add_input(opdef, get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));

        } else if (nm == "Reshape") {
            // [data] — no injected inputs
        }
        // Split nodes are eliminated later, no injection needed
    }

    // Rebuild consumers: injected const 节点在 op 之后创建, append_node 的
    // consumer 注册漏了它们。从 input 连接重建。
    rebuild_consumers();
    // tensor_param const(ranges/perm 等)不在 op inputs 里, rebuild 后必须
    // 重注册消费者 —— 否则 Phase7 DCE 删 param const, StridedSlice 等
    // 拿不到参数(probe rope Slice 全零实锤)。
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef) continue;
        for (op_id_t tpid : opdef->tensor_param_ids) {
            auto it = opdef_map_.find(tpid);
            if (it != opdef_map_.end() && it->second)
                it->second->consumers.push_back(id);
        }
    }
}

// do_prepare1: initial graph building
// Source: graph_prepare.cc, 8518 bytes at 0xF66360
GraphStatus GraphPrepare::do_prepare1(HexagonNNEnv& env, VtcmCacheInstance& vtcm) {
    // 0. HtpPrepare auto-injection: per-op-type 输入构造
    // composeGraphs 产出的原始图里 op inputs 只含 data; HtpPrepare 注入
    // quant_marker/scale/output_rank 并把 tensor_param(perm 等)追加到 inputs。
    // 注入在 DCE 之前, 这样 DCE 不会误删被引用的 tensor_param const。
    inject_htp_prepare_inputs();
    if (getenv("GEHTP_DBG_T18"))
        for (auto& [id, od] : opdef_map_)
            if (od && od->name_tag && od->name_tag->name() &&
                std::strstr(od->name_tag->name(), "KvAppend"))
                std::fprintf(stderr, "[t18-inj] %s id=%llu op_data=%zu\n",
                             od->name_tag->name(), (unsigned long long)id,
                             od->op_data.size());

    // 1. Set construction state
    construction_state_ = 1; // +0x45dc = 1 (construction phase)

    // 1b. Initial remove_dead_code (prepare_start immediately does DCE)
    // Source: graph_prepare.cc:1380 "initial remove_dead_code : result = %d"
    // QNN dumps before_graph AFTER this initial DCE, so dead Split/Reshape
    // nodes introduced by converter normalization are removed here.
    int dce_result = remove_dead_code(false);
    if (dce_result > 0) {
        std::fprintf(stderr, "do_prepare1: initial DCE removed %d ops\n", dce_result);
    }
    if (getenv("GEHTP_DBG_T18"))
        for (auto& [id, od] : opdef_map_)
            if (od && od->name_tag && od->name_tag->name() &&
                std::strstr(od->name_tag->name(), "KvAppend"))
                std::fprintf(stderr, "[t18-dce] %s id=%llu op_data=%zu\n",
                             od->name_tag->name(), (unsigned long long)id,
                             od->op_data.size());

    // 1c. Split 消除已禁用: eliminate_split_nodes() 是 REQNN 逆向对拍实现,
    // 把 Split 消费者重定向到 dims=[1] 占位 const(模仿 QNN before_graph
    // 形态)——执行语义全毁(probe Reshape 输入 [1,32,64] 变 4 字节占位,
    // 恒等拷贝越界 + 数值错)。我方管线保留 Split 正常执行
    // (OP_SPLIT_F16 + exec_split, M3 契约), 需要 QNN 形态对拍时
    // 显式调用 eliminate_split_nodes() 即可。

    // 2. Run initial const propagation
    // TODO: const_prop currently propagates through Split placeholder consts,
    // incorrectly marking downstream ops (MatMul, Softmax, Transpose) as const.
    // QNN's const_prop doesn't propagate through injected placeholder tensors.
    // Disabled until proper guard is implemented.
    // const_prop(env, false);

    // 3. Basic shape inference (gen_Shape for each op)
    // Source: graph_prepare.cc, gen_Shape @ 0xF77570 (377 bytes)
    // Infer output_def shape from input connections where not already set.
    // Elementwise/activation/norm/softmax ops inherit the first input's
    // shape; the Output node copies its producer's shape.
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled()) continue;
        if (!opdef->name_tag) continue;
        std::string nm = opdef->name_tag->name() ? opdef->name_tag->name() : "";

        // Output node: copy shape from its single input producer.
        if (nm == "Output" && !opdef->inputs.empty()) {
            auto src = opdef_map_.find(opdef->inputs[0].src_id);
            if (src != opdef_map_.end() && src->second) {
                opdef->output_def = src->second->output_def;
            }
            continue;
        }
        // Skip nodes whose output shape is already fully specified.
        if (opdef->output_def.rank != 0 && opdef->output_def.dims[0] != 0) continue;
        // Const/Input have no inputs to infer from; leave as-is.
        if (nm == "Const" || nm == "Input" || nm == "$Const") continue;
        if (opdef->inputs.empty()) continue;

        // Infer from first input producer (same-shape ops: elementwise,
        // activations, normalization, softmax, pool, reshape-by-input, etc.).
        auto src = opdef_map_.find(opdef->inputs[0].src_id);
        if (src != opdef_map_.end() && src->second) {
            const OutputDef& in_od = src->second->output_def;
            if (in_od.rank != 0) {
                opdef->output_def = in_od;
            }
        }
    }

    // 3b. Batch dimension padding (HtpPrepare standardization)
    // All tensors with rank < 4 are padded to rank 4 by prepending 1s.
    // This matches QNN HtpPrepare's tensor rank normalization.
    // Example: [8, 192] (rank 2) -> [1, 1, 8, 192] (rank 4)
    //          [1, 8, 64] (rank 3) -> [1, 1, 8, 64] (rank 4)
    //          [1, 8, 1, 3, 64] (rank 5) -> unchanged
    std::unordered_map<op_id_t, uint32_t> pre_pad_rank;
    for (auto& [pid0, opdef0] : opdef_map_)
        if (opdef0 && opdef0->output_def.rank > 0 && opdef0->output_def.rank < 4)
            pre_pad_rank[pid0] = opdef0->output_def.rank;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef) continue;
        OutputDef& od = opdef->output_def;
        // Skip undefined quantization tensors (dt=2147483647) — QNN keeps these as rank 1
        if (od.dtype == 2147483647) continue;
        if (od.rank > 0 && od.rank < 4) {
            uint32_t old_rank = od.rank;
            // Shift dims right and prepend 1s
            for (int i = 3; i >= 0; --i) {
                if (i >= static_cast<int>(4 - old_rank))
                    od.dims[i] = od.dims[i - (4 - old_rank)];
                else
                    od.dims[i] = 1;
            }
            od.rank = 4;
        }
    }

    // 3b2. axis 重映射(与 3b 配套): 前插 (4-r) 个前导 1 后, 正 axis 的
    // 语义轴平移 +(4-r); 负 axis(末轴起)不受前导填充影响。覆盖 op_data
    // 的 "axis" scalar(Concat/Softmax/Gather/CumulativeSum/Split)与
    // Reduce 的 axes tensor_param 数据(int32)。0.8B 全图实锤: rotary_emb
    // Concat_4 rank3 axis=2 → 真拼接轴=3(padding 后按轴 2 读爆缓冲);
    // embedding Gather axis=0 rank2 表 → 轴 2。host execute / const_fold
    // / wtop_emit(ExtraAxis)同源读此值, 源头重映射三处一致。轴基准秩
    // 取第一个输入的填充前秩(axis 语义挂在输入形状上)。
    {
        std::unordered_set<op_id_t> done_axes_const;
        for (auto& [id2, opdef] : opdef_map_) {
            if (!opdef || !opdef->name_tag) continue;
            const char* nm2 = opdef->name_tag->name();
            if (!nm2) continue;
            const bool axis_scalar_op =
                !std::strcmp(nm2, "Concat") || !std::strcmp(nm2, "Softmax") ||
                !std::strcmp(nm2, "Gather") || !std::strcmp(nm2, "CumulativeSum") ||
                !std::strcmp(nm2, "Split");
            const bool reduce_op = !std::strcmp(nm2, "Reduce");
            if (!axis_scalar_op && !reduce_op) continue;
            if (opdef->inputs.empty()) continue;
            auto rit = pre_pad_rank.find(opdef->inputs[0].src_id);
            if (rit == pre_pad_rank.end()) continue;
            const uint32_t delta = 4 - rit->second;
            if (axis_scalar_op && !opdef->op_data.empty()) {
                auto sp = unpack_scalar_params(opdef->op_data);
                bool changed = false;
                for (auto& p : sp) {
                    if (p.name != "axis" || !p.is_numeric) continue;
                    int64_t ax = (int64_t)p.value_num;
                    if (ax < 0) continue;
                    p.value_num = (double)(ax + (int64_t)delta);
                    changed = true;
                }
                if (changed) opdef->op_data = pack_scalar_params(sp);
            }
            if (reduce_op) {
                for (op_id_t pid : opdef->tensor_param_ids) {
                    if (!done_axes_const.insert(pid).second) continue;
                    OpDef* pc = get_op_at(pid);
                    if (!pc || pc->const_data_size == 0 || pc->const_data_size % 4 != 0 ||
                        pc->const_data_size / 4 > 8) continue;
                    if (pc->const_data_offset + pc->const_data_size > const_pool_.size())
                        continue;
                    if (pc->output_def.dtype != static_cast<uint32_t>(DType::Int32)) continue;
                    int32_t* vals = reinterpret_cast<int32_t*>(
                        const_pool_.data() + pc->const_data_offset);
                    for (uint64_t k = 0; k < pc->const_data_size / 4; k++)
                        if (vals[k] >= 0 && vals[k] <= 8) vals[k] += (int32_t)delta;
                }
            }
        }
    }

    // 3c. 图优化 pass(PassManager, 默认全开)
    // 真实库位置: do_prepare1 内 const_prop_and_cse 之后、order_nodes 之前
    // (M01 实证 do_prepare1 @0xf66550 调用序)。pass 使能统一由
    // PassManager 管理(--no-pass 关单 pass); 结构不匹配 = 图不动。
    run_optimize_passes(env);
    if (getenv("GEHTP_DBG_T18"))
        for (auto& [id, od] : opdef_map_)
            if (od && od->name_tag && od->name_tag->name() &&
                std::strstr(od->name_tag->name(), "KvAppend"))
                std::fprintf(stderr, "[t18-opt] %s id=%llu op_data=%zu\n",
                             od->name_tag->name(), (unsigned long long)id,
                             od->op_data.size());

    // 4. Initial node ordering
    order_nodes(true);

    // 5. Allocate IO tensors
    // Source: allocate_io_tensors @ 0xF69750 (2687 bytes, ELF st_size)
    allocate_io_tensors();

    // 6. Const tracking setup
    const_tracking_setup();

    // 7. Mark construction complete
    construction_state_ = 2; // +0x45dc = 2 (optimization phase)

    if (getenv("GEHTP_DBG_T18"))   /* T018d 探针: do_prepare1 尾 */
        for (auto& [id, od] : opdef_map_)
            if (od && od->name_tag && od->name_tag->name() &&
                std::strstr(od->name_tag->name(), "KvAppend"))
                std::fprintf(stderr, "[t18-p1] %s id=%llu op_data=%zu\n",
                             od->name_tag->name(), (unsigned long long)id,
                             od->op_data.size());

    return GraphStatus::Success;
}

// do_prepare2: full optimization
// Source: graph_prepare.cc:1777 "do_prepare2 with num_nsps = %zu"
// 6019 bytes at 0xF631F0
GraphStatus GraphPrepare::do_prepare2(HexagonNNEnv& env, VtcmCacheInstance& vtcm,
                                        int& retry_count, bool full_prepare) {
    // 1. prepare_op - prepare each op
    // Source: prepare_op @ 0xF61010 (2321 bytes, ELF st_size)
    // For each op: instantiate the execution Op via the op factory, wire it
    //   to its OpDef, and compute an initial cost estimate. The generated Op
    //   objects are stored in ops_ and used by the cost model / serializer.
    ops_.clear();
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
        OpIoPtrs io{};
        io.graph_prepare = this;
        io.opdef_ptr = opdef.get();
        auto op = op_factory_generate(io, id);
        if (op) {
            ops_.push_back(std::move(op));
        }
    }

    // 2. sanity_check_null_exec
    // Source: sanity_check_null_exec @ 0xF6D220 (521 bytes)
    // Verifies every op has a valid execution path

    // 3. Run optimization passes (8-phase)

    // 4. TCM migration is done inside phase 2 and 4

    // ===== Step 1: build_graph_deps (反汇编确�? @ 0xfac220, 8216B) =====
    // �?OpDef 图构�?GraphDeps (OpDesc �?+ 依赖�?+ 生命�?+ memgroup)
    // 产物存入 this+0x7468, �?FancyAllocator �?runlist 调度消费
    mark_time_point("build_graph_deps");
    build_graph_deps();

    // ===== SuperTile DP 全流�?(反汇编确�? Stage 0 + Stage 4) =====
    // VTCM 预算: get_vtcm_tile_size = budget × 0.75 (sap_reduce_bandwidth @ 0xf6ee78)
    vtcm_size_ = get_vtcm_tile_size();
    vtcm_lifetime_alloc(vtcm);

    // Per-Op overlap allocation (反汇编 @ 0x13a14b0: allow_tensor_overlap)
    // Producer-consumer VTCM reuse via force_contiguous + link_blocks
    vtcm_overlap_alloc(vtcm);

    // TCM block allocation finalize (反汇编 @ 0x13B29D0: allocate_tcm_blocks_internal)
    // Sort by reverse lifetime, probe hash for reuse, compute total, check budget
    vtcm_block_alloc_finalize(vtcm);

    // Persistent pool creation (反汇编 @ 0xf455c0: make_persistent_pools)
    // Classify blocks into persistent (weights) vs scratch (activations) pools
    vtcm.allocator().make_persistent_pools();

    // Post spill/fill design pass (反汇编 @ 0x129ed30: post_spill_fill_design_pass)
    // Finalize DMA spill/fill design based on VTCM pressure
    {
        std::vector<uint32_t> runlist_tags_u32;
        runlist_tags_u32.reserve(ordering_.size());
        for (auto id : ordering_) runlist_tags_u32.push_back(static_cast<uint32_t>(id));
        post_spill_fill_design_pass(runlist_tags_u32);
    }


    // Stage 0: initial_sequencer �?DPGroupGraph 构建 (dp_group_graph.cc 8 �?
    // 反汇编确�? mark_prepare_stage({"initial_sequencer", 0}) @ PASS_PSEUDOCODE.md:145
    mark_prepare_stage({"initial_sequencer", 0});
    dp_group_graph_ = std::make_unique<DPGroupGraph>();
    // DDR 预算简�? VTCM �?4 �?(DDR 远大�?VTCM)
    dp_group_graph_->build(*this, vtcm_size_, vtcm_size_ * 4);

    // Stage 2: sap_reduce_bandwidth (�?2) �?VTCM × 0.75 已在 get_vtcm_tile_size 计算

    // Stage 4: create_supertiles �?SuperTile DP 合并
    // 反汇编确�? mark_prepare_stage({"create_supertiles", 4}) @ 0x13be970
    mark_prepare_stage({"create_supertiles", 4});
    create_supertiles();

    // 5. DP Sequencer (SVF/LVF + MLH)
    // Source: do_prepare2_late or inline
    // pysequencer(runlist_tags)

    // 6. Spill/fill insertion
    // Source: insert_spillfill.cc, grdep_spillfill.cc

    // 7. Multicast optimization (Phase 4.3)
    // Source: grdep_mcast_optimizer.cc
    // Build McSend list from cross-NSP tensor consumers, then optimize.
    run_mcast_optimization();

    // 8. Serialization preparation

    // 8. Serialization preparation
    // (done in do_serialize)

    // Mark graph as compiled (CONSTRUCTION=1 �?PREPARE=2 �?COMPILED=3)
    // Source: do_prepare2 completion sets +0x45dc = 3
    construction_state_ = 3;

    if (getenv("GEHTP_DBG_T18")) {  /* T018d 探针: do_prepare2 尾 */
        for (auto& [id, od] : opdef_map_)
            if (od && od->name_tag && od->name_tag->name() &&
                std::strstr(od->name_tag->name(), "KvAppend"))
                std::fprintf(stderr, "[t18-p2] %s id=%llu op_data=%zu\n",
                             od->name_tag->name(), (unsigned long long)id,
                             od->op_data.size());
        if (getenv("GEHTP_DBG_T18_L3"))
            for (auto& [id, od] : opdef_map_) {
                if (!od) continue;
                const char* gn = od->grouping.c_str();
                if (std::strstr(gn, "layers.3/self_attn/Slice") ||
                    std::strstr(gn, "layers.3/self_attn/Concat") ||
                    std::strstr(gn, "layers.3/self_attn/kv_append") ||
                    std::strstr(gn, "layers.3/self_attn/Transpose"))
                    std::fprintf(stderr, "[t18-l3] id=%llu name=%s grp=%s flags=0x%x csize=%llu odata=%zu\n",
                                 (unsigned long long)id,
                                 od->name_tag ? od->name_tag->name() : "?", gn,
                                 od->flags, (unsigned long long)od->const_data_size,
                                 od->op_data.size());
            }
    }

    return GraphStatus::Success;
}
// ---- --plan-vtcm-budget 缺省(auto)解析用: HMX scratch 峰值普查 ----
// need 公式与 op_matmul.cpp 的 US1 分块 lambda / wtop_ops.hpp assign_engine
// 三处同口径拷贝, 改一处必同步三处(test_vtcm_auto 用大 matmul 锚住)。
// vtcm_auto_scan 定义在本文件; M/K/N 提取走导出的 compute_op_extra(注册表
// 查找, 与 serialize_opdef 同源) —— extractor 族已随序列化区拆到 graph_prepare_serialize.cpp。
static uint64_t mm_scratch_need(uint32_t m, uint32_t k, uint32_t n) {
    size_t a = (size_t)m * k * 2u, w = (size_t)n * k * 2u, o = (size_t)m * n * 2u;
    size_t off_w = (a + 2047u) & ~(size_t)2047u;
    size_t off_o = (off_w + w + 2047u) & ~(size_t)2047u;
    size_t off_s = (off_o + o + 2047u) & ~(size_t)2047u;
    return off_s + 2048u;
}
struct VtcmAutoScan {
    bool w4 = false;
    bool conv = false;
    uint64_t scratch_max = 0;
    uint64_t weight_sum = 0;
    uint64_t gdn_scratch = 0;  // GdnBlock 内核私有 scratch(层间时间复用, 取 max)
};
static VtcmAutoScan vtcm_auto_scan(const GraphPrepare& gp);

// Source: graph_prepare.cc, 2329 bytes at 0xF71F00
GraphStatus GraphPrepare::do_prepare2_late(std::vector<uint32_t>& runlist_tags) {
    // 1. Final optimization passes
    dead_code_removal_and_cse();

    // 2. Re-order nodes (ensures ordering_ is populated for execution)
    order_nodes(false);

    if (getenv("GEHTP_DBG_T18"))   /* T018d 探针: do_prepare2_late CSE 后 */
        for (auto& [id, od] : opdef_map_)
            if (od && od->name_tag && od->name_tag->name() &&
                std::strstr(od->name_tag->name(), "KvAppend"))
                std::fprintf(stderr, "[t18-p2l] %s id=%llu op_data=%zu\n",
                             od->name_tag->name(), (unsigned long long)id,
                             od->op_data.size());

    // 3. Runlist generation
    // Source: runlist_info.cc, run_order_to_alloc_info.cc
    // Builds runlist from ordered ops

    // 4. Final allocation
    // phys_alloc_in_runlist(ops)

    // 5. plan_order_ 拓扑定稿 (内存规划收编 M1: 执行序唯一真相源上提编译器)
    //    ST-Cut 产出为调度序, 非保证拓扑; 原定稿在 wtop_emit Kahn 重排
    //    (probe: Split 15 排消费者 Reshape 18 后 → src_ref 查无 temp)。
    //    算法与 emit 侧既有实现逐语句一致(orphan 殿后 / ready 升序+pop_back /
    //    环图回退原序), blob op 序列零变化; emit 侧降级为校验+旧 bin 兜底。
    if (!plan_order_.empty()) {
        std::unordered_map<op_id_t, size_t> indeg;
        std::unordered_map<op_id_t, std::vector<op_id_t>> succ;
        std::vector<op_id_t> orphans;  // plan_order_ 里已无 OpDef 的 id(DCE 后)
        for (op_id_t id : plan_order_) {
            const OpDef* od = get_op_at(id);
            if (!od) { orphans.push_back(id); continue; }
            indeg[id] = 0;
            for (const auto& c : od->inputs)
                if (indeg.count(c.src_id)) { ++indeg[id]; succ[c.src_id].push_back(id); }
        }
        std::vector<op_id_t> ready;
        for (auto& [id, d] : indeg) if (d == 0) ready.push_back(id);
        std::sort(ready.begin(), ready.end());  // 确定性(与 emit/host 对拍同序)
        std::vector<op_id_t> topo;
        while (!ready.empty()) {
            op_id_t id = ready.back();
            ready.pop_back();
            topo.push_back(id);
            for (op_id_t s : succ[id])
                if (--indeg[s] == 0) ready.push_back(s);
        }
        for (op_id_t id : orphans) topo.push_back(id);
        if (topo.size() == plan_order_.size()) plan_order_ = topo;  // 环图回退原序
    }

    // 6. 静态内存规划定稿 (M2 内存规划收编): plan_order_ 拓扑定稿后按定稿序
    //    算两级池偏移(算法 = compute_ddr_offsets 原样, 与 emit 时代逐字节同),
    //    经 TAG_MEM_PLAN 进 tagged.bin; wtop_emit 照抄不再自算。
    //    ddr 预算 0 = 不规划(与 emit --ddr-budget 缺省同语义)。
    has_mem_plan_ = false;
    mem_plan_ = MemPlan{};
    /* --plan-vtcm-budget 缺省 = auto: 纯默认编译(无 --ddr-budget、权重内嵌)
     * 且图内无 W4A16 matmul 时, 驻留池 = 16MB − HMX scratch 峰值(镜像发射侧
     * N 分块口径, 既有分块全部放得下 → op 流与关池一致, 运行时零新增回落)
     * − 余量(运行时 attach 下界 align2048(wtcache pin_cap) + 杂项)。
     * 显式传值(含 0)一律照用; 0 = 关(debug/旧口径)。 */
    uint64_t vtcm_eff = plan_vtcm_budget_;
    if (vtcm_eff == kVtcmPlanAuto) {
        vtcm_eff = 0;
        if (plan_ddr_budget_ > 0) {
            std::fprintf(stderr, "[memplan] vtcm auto off: --ddr-budget 显式(旧口径, 要池请显式 --plan-vtcm-budget)\n");
        } else if (const_pool_external_) {
            std::fprintf(stderr, "[memplan] vtcm auto off: 权重外置路线B(池内字节不可判 W4)\n");
        } else {
            VtcmAutoScan vas = vtcm_auto_scan(*this);
            constexpr uint64_t kVtcmTotal = 16ull << 20;  // 设备 A2 query; 发射侧 kVTCMTotal 同源
            constexpr uint64_t kPinMisc = 512ull << 10;   // 固定杂项余量(执行器 pin_cap==0 回退 4096 + ring/temp 上界)
            const uint64_t conv_head = (vas.conv ? (2ull << 20) : 0ull);

            // ① 常驻需求(预算)= 矩阵族候选张量生命期打包后的峰值重叠水位。
            //    发射侧(op_matmul)已按 ddr_vtcm_cap 预留 pool、chunk 预算 = 16MB − pool,
            //    这里只负责把需求算出来、反过来定 pool —— 不再让 scratch 先吃满 16MB
            //    再让常驻吃剩的(=0)。探针用满 16MB 预算跑一次分配, 取 in_vtcm 峰值。
            //    2048 对齐 = alloc_vtcm_stage 叠加基址口径(wtop_ops.hpp)。
            uint64_t residency_demand = 0;
            {
                std::vector<DdrTempEntry> probe;
                uint64_t probe_cap = 0, probe_vcap = 0;
                if (compute_ddr_offsets(0, &probe, &probe_cap, nullptr, nullptr,
                                        kVtcmTotal, &probe_vcap) == 0) {
                    for (const auto& e : probe)
                        if (e.in_vtcm)
                            residency_demand = std::max<uint64_t>(residency_demand, e.offset + e.size);
                }
                residency_demand = (residency_demand + 2047u) & ~uint64_t(2047u);
            }

            // ② 池 = 驻留峰值 + 内核私有 scratch(GdnBlock): scratch 非图张量,
            //    驻留打包看不见, emit 侧 alloc_vtcm_stage 在峰值之上 bump 叠加,
            //    这里先留出。W4A16 act staging 依赖 emit 期物化态, 但图含 W4A16
            //    时本就 auto-off(下方 vas.w4), 非 W4 图 staging 恒 0, 不计。
            const uint64_t gdn_pad = vas.gdn_scratch;

            // ③ 权重常驻(pin_cap)与常驻池/scratch/余量共存检查(010 T004)。
            uint64_t pin_cap = (vas.weight_sum + 2047u) & ~uint64_t(2047u);
            if (pin_cap > 0 && pin_cap + vas.scratch_max + residency_demand + gdn_pad + conv_head + kPinMisc >= kVtcmTotal) {
                pin_cap = 0;  // 权重塞不进(与常驻/scratch/余量争 16MB) → 流式, 不 pin
            }
            const uint64_t head = conv_head + kPinMisc + pin_cap;

            if (vas.w4) {
                std::fprintf(stderr, "[memplan] vtcm auto off: 图含 W4A16 matmul (carve/池预算统一另线收口)\n");
            } else if (residency_demand > 0 || gdn_pad > 0) {
                vtcm_eff = residency_demand + gdn_pad;
                mem_plan_.pin_cap = pin_cap;
                std::fprintf(stderr, "[memplan] vtcm auto: 常驻=%.2fMB gdn_scratch=%.2fMB scratch_max=%.2fMB conv=%d pin_cap=%llu head=%llu → pool=%.2fMB\n",
                             residency_demand / 1048576.0,
                             vas.gdn_scratch / 1048576.0,
                             vas.scratch_max / 1048576.0, (int)vas.conv,
                             (unsigned long long)pin_cap,
                             (unsigned long long)head,
                             vtcm_eff / 1048576.0);
            } else {
                std::fprintf(stderr, "[memplan] vtcm auto off: 无常驻候选 (scratch_max=%llu + head=%llu ≥ VTCM 16MB)\n",
                             (unsigned long long)vas.scratch_max, (unsigned long long)head);
            }
        }
    }
    if (plan_ddr_budget_ > 0 || vtcm_eff > 0) {
        std::vector<DdrTempEntry> entries;
        std::vector<DdrSpillEntry> spills;
        uint64_t cap = 0, spill_total = 0, vcap = 0;
        if (compute_ddr_offsets(plan_ddr_budget_, &entries, &cap,
                                &spills, &spill_total,
                                vtcm_eff, &vcap) == 0) {
            mem_plan_.ddr_cap = cap;
            mem_plan_.vtcm_cap = vtcm_eff ? vcap : 0;
            mem_plan_.bump_reserve = std::max<uint64_t>(4u << 20, cap / 4);
            for (const auto& e : entries)
                mem_plan_.entries.push_back({e.op_id, 0, e.offset, e.size, e.in_vtcm});
            for (const auto& s : spills)
                mem_plan_.spills.push_back({s.op_id, s.offset, s.size});
            has_mem_plan_ = true;
        }
    }

    return GraphStatus::Success;
}

// Get sorted opdefs: topological order from `ordering_`, or fallback to opdef_map iteration.
std::vector<const OpDef*> GraphPrepare::get_sorted_opdefs() const {
    std::vector<const OpDef*> result;
    if (!ordering_.empty()) {
        for (auto id : ordering_) {
            auto it = opdef_map_.find(id);
            if (it != opdef_map_.end() && it->second && it->second->is_enabled() && !it->second->is_dead())
                result.push_back(it->second.get());
        }
        return result;
    }
    // Fallback: iterate in op_id order (construction order)
    for (auto& [id, opdef] : opdef_map_) {
        if (opdef && opdef->is_enabled() && !opdef->is_dead())
            result.push_back(opdef.get());
    }
    return result;
}

// Host-side reference execute: float buffers, topo order, tensor map.
/* 多输入绑定的槽位序: 声明序(loader 记录的 net.json 张量字典插入序)优先,
 * 无记录(旧 bin 反序列化)时回退 op_id 升序(=基线行为)。声明序条目须
 * 与当前图 Input 节点集合一致(DCE 可能裁输入), 不一致同样回退。 */
std::vector<op_id_t> GraphPrepare::host_input_order() {
    std::vector<op_id_t> in_ids;
    for (auto& [id, opdef] : opdef_map_)
        if (opdef && opdef->name_tag && opdef->name_tag->name() &&
            std::string(opdef->name_tag->name()) == "Input")
            in_ids.push_back(id);
    std::sort(in_ids.begin(), in_ids.end());
    if (input_decl_order_.size() == in_ids.size() && !in_ids.empty()) {
        bool ok = true;
        for (op_id_t d : input_decl_order_)
            if (!std::binary_search(in_ids.begin(), in_ids.end(), d)) { ok = false; break; }
        if (ok) return input_decl_order_;
    }
    return in_ids;
}

// Optimization passes entry point
// Source: graph_prepare.cc, 238 bytes at 0xF6A1D0
void GraphPrepare::run_optimize_passes(HexagonNNEnv& env) {
    run_optimize_passes_multi_registry(env);
}

// Source: graph_prepare.cc, 126 bytes at 0xF74D90
void GraphPrepare::run_optimize_passes_multi_registry(HexagonNNEnv& env) {
    run_optimize_passes_single_registry(env, optimization_registry_);
}

// Source: graph_prepare.cc, 62 bytes at 0xF6A2C0
void GraphPrepare::dead_code_removal_and_cse() {
    remove_dead_code(false);
    common_subexpr_eliminate(true);
}

// 8-phase optimization dispatch
// Source: graph_prepare.cc, run_optimize_passes_single_registry @ 0xf730b0 (verified)
// 8 phases with node-count thresholds; pass is SKIPPED if node_count > threshold.
// Phase 1 (�?000):  run_plugin_rewrites
// Phase 2 (�?0190): reserved (no pass)
// Phase 3 (�?1900): GraphOptContext::attempt (pattern matching via Fibonacci hash)
// Phase 4 (�?2500): reserved (no pass)
// Phase 5 (�?1101): const_prop_and_cse
// Phase 6 (�?2000): reserved (no pass)
// Phase 7 (�?4999): fixpoint loop (DCE→order→CSE, iterate until clean)
// Phase 8 (�?:      final cleanup (DCE→order→CSE, one more round)
void GraphPrepare::run_optimize_passes_single_registry(
    HexagonNNEnv& env,
    const std::map<uint32_t, GraphOptPass>& registries) {

    (void)registries;

    // 我方结构匹配 pass 框架(PassManager, 见 pass_manager.hpp)。
    // 无条件跑、不设节点数阈值(与真实库阈值门不同: 结构不匹配即无副作用;
    // 828/15129 图不因阈值跳过 pass —— 分歧已记录)。
    // PHASE_0=常量折叠, PHASE_1=形状归一化; PHASE_3(激活融合)留空:
    // exec_matmul 的 bias 槽实为 W4A16 scale 表((N/32)*512B, 非 dense bias),
    // conv/add/binary 无后激活参数 —— 零设备改动约束下无准入规则, M7 内核
    // 扩展后补。
    static bool passes_registered = false;
    if (!passes_registered) {
        register_builtin_passes();
        passes_registered = true;
    }
    PassManager& pm = PassManager::instance();
    pm.reset_stats();
    uint32_t before = 0;
    for (auto& [id, opdef] : opdef_map_) {
        if (opdef && opdef->is_enabled() && !opdef->is_dead()) before++;
    }

    if (getenv("GEHTP_DUMP_VIEWS")) {          /* 005 T004: prepare 后/pass 前基数 */
        std::fprintf(stderr, "[view-census PRE ]\n");
        dump_view_census(this, stderr);
    }

    uint32_t changed = pm.run_phase(this, PHASE_0);
    if (changed > 0) run_phase_fixpoint_internal();
    changed = pm.run_phase(this, PHASE_1);
    if (changed > 0) run_phase_fixpoint_internal();
    // 激活融合槽(PHASE_3): 默认全关(非设备准入), run_phase 空转返回 0;
    // --enable-pass fusion_* 打开后这里才生效。
    changed = pm.run_phase(this, PHASE_3);
    if (changed > 0) run_phase_fixpoint_internal();

    uint32_t after = 0;
    for (auto& [id, opdef] : opdef_map_) {
        if (opdef && opdef->is_enabled() && !opdef->is_dead()) after++;
    }
    pm.note_counts(before, after);

    if (getenv("GEHTP_DUMP_VIEWS")) {          /* pass 后余量(对照 PRE 得消解数) */
        std::fprintf(stderr, "[view-census POST]\n");
        dump_view_census(this, stderr);
    }

    // Build 8 phase descriptors with verified node-count thresholds
    struct PhaseInfo { uint32_t threshold; const char* name; };
    constexpr PhaseInfo phases[] = {
        {3000,        "Phase1"},  // run_plugin_rewrites
        {10190,       "Phase2"},  // reserved
        {11900,       "Phase3"},  // GraphOptContext::attempt
        {12500,       "Phase4"},  // reserved
        {21101,       "Phase5"},  // const_prop_and_cse
        {22000,       "Phase6"},  // reserved
        {24999,       "Phase7"},  // fixpoint loop
        {0xFFFFFFFF,  "Phase8"},  // final cleanup (�?
    };

    // Current graph node count (enabled, non-dead ops)
    uint32_t node_count = 0;
    for (auto& [id, opdef] : opdef_map_) {
        if (opdef && opdef->is_enabled() && !opdef->is_dead()) node_count++;
    }

    // Phase 1 (�?000): run_plugin_rewrites
    if (node_count <= phases[0].threshold) {
        // run_plugin_rewrites: iterate registry passes
        // (反汇编 @ 0x10d87e0: no registered plugins in host reimpl; structure preserved)
        run_plugin_rewrites(false);  // early phase
    }

    // Phase 3 (�?1900): GraphOptContext::attempt (pattern matching)
    if (node_count <= phases[2].threshold) {
        // Iterate ops, use Fibonacci hash to find matching rewrite rules
        // (no registered rules in host reimpl; fusion applied below instead)
    }

    // Phase 5 (�?1101): const_prop_and_cse
    if (node_count <= phases[4].threshold) {
        bool changed = false;
        const_prop_and_cse(env, false, &changed);
    }

    // Phase 7 (�?4999): fixpoint loop (DCE �?order �?CSE, iterate until clean)
    if (node_count <= phases[6].threshold) {
        run_phase_fixpoint_internal();
    }

    // Phase 8 (�?: final cleanup (always executed)
    run_phase_fixpoint_internal();

    // 遗留 8-rule 融合表已移出主路径(自造且非设备准入: ConvActivations 等
    // 名字 wtop_emit 无翻译, exec_conv2d/add/binary 无激活参数 —— 发射即错)。
    // 契约保留在 kLegacyFusionRules, 由 test_e2e 直接调 apply_fusion_rules 验证。
}

// Fixpoint loop: DCE -> order_nodes -> CSE -> clear graph_dirty
// Source: Phase2/3/5 vfunc[6]
void GraphPrepare::run_phase_fixpoint_internal() {
    // Loop until no changes:
    while (true) {
        int changed = remove_dead_code(false);
        if (changed == 0) {
            changed = order_nodes(true);
            if (changed == 0) {
                changed = common_subexpr_eliminate(true);
                if (changed == 0) {
                    graph_dirty_ = false; // +0x7311 = 0
                    break;
                }
            }
        }
    }
}

// TCM migration: spill tensors to DDR when VTCM budget exceeded
// Source: tcm_migration @ 0x13219c0 (verified)
// Algorithm:
// 1. Compute total VTCM demand (sum of all activation tensor sizes)
// 2. If demand > budget: build max-heap by tensor priority
// 3. Spill highest-priority tensors to DDR (set SPILL_TO_DDR flag 0x40)
// 4. Continue until demand <= budget
// Priority: pre-calculated at tensor +0x98 (higher = spill first, greedy)
// SPILL_TO_DDR flag: 0x40 set at tensor +0x20 (0x1321fc0: orb $0x40, 0x20(%r12))
void GraphPrepare::tcm_migration(uint32_t threshold, bool aggressive) {
    (void)threshold;
    (void)aggressive;

    // Get usable VTCM budget (4MB cap × 0.75)
    size_t budget = get_vtcm_tile_size();
    if (budget == 0) return;

    // Build tensor list with size and priority
    struct TensorInfo {
        op_id_t op_id;
        uint64_t size;
        uint32_t priority;
    };
    std::vector<TensorInfo> tensors;

    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
        if (opdef->is_const()) continue; // persistent tensors not spilled

        // Estimate tensor size from output_def
        uint64_t size = 1;
        for (uint32_t i = 0; i < opdef->output_def.rank && i < 5; i++) {
            if (opdef->output_def.dims[i] > 0)
                size *= opdef->output_def.dims[i];
        }
        size *= (opdef->output_def.element_size ? opdef->output_def.element_size : 4);

        // Priority ≈ size / (access_frequency × lifetime)
        // Higher priority = spill first (greedy: large, rarely-used tensors)
        // Phase 4.4: use cost-aware priority — divide by cost so cheap ops
        // (low cost per element) are spilled first, keeping expensive ops in VTCM.
        uint32_t access_freq = static_cast<uint32_t>(opdef->consumers.size());
        if (access_freq == 0) access_freq = 1;
        // Phase 4.4: get op cost from CostSource (via TypicalOp::cost)
        // The op_cost is 0 if no Op is materialized yet (falls back to size-only)
        float op_cost = 1.0f;
        for (const auto& op : ops_) {
            auto* top = dynamic_cast<const TypicalOp*>(op.get());
            if (top && top->op_id == id) {
                op_cost = std::max(op->cost(nullptr), 1.0f);
                break;
            }
        }
        // Cost-aware priority: size / (access_freq * cost)
        // Low-cost ops (Reshape=1) → high priority → spill first
        // High-cost ops (Conv=1000) → low priority → keep in VTCM
        uint32_t priority = aggressive
            ? static_cast<uint32_t>(size / (access_freq * op_cost))
            : static_cast<uint32_t>(size / (access_freq * 2 * op_cost));

        tensors.push_back({id, size, priority});
    }

    // Compute total demand
    uint64_t total_demand = 0;
    for (const auto& t : tensors) total_demand += t.size;

    if (total_demand <= budget) return; // no spill needed

    // Build max-heap by priority (verified sift_down @ 0x1322118-0x1322270)
    // Max-heap: highest priority on top, spilled first
    auto heap_cmp = [](const TensorInfo& a, const TensorInfo& b) {
        return a.priority < b.priority;
    };
    std::priority_queue<TensorInfo, std::vector<TensorInfo>, decltype(heap_cmp)>
        heap(heap_cmp, std::move(tensors));

    // Spill tensors from highest priority until demand <= budget
    // Source: 0x1321fc0: orb $0x40, 0x20(%r12) �?set SPILL_TO_DDR flag
    constexpr uint32_t SPILL_TO_DDR = 0x40;
    while (total_demand > budget && !heap.empty()) {
        TensorInfo top = heap.top();
        heap.pop();

        auto it = opdef_map_.find(top.op_id);
        if (it != opdef_map_.end() && it->second) {
            it->second->flags2 |= SPILL_TO_DDR;
        }
        total_demand -= top.size;
    }
}

/* --plan-vtcm-budget auto 解析用的全图扫描: HMX scratch 峰值(镜像 op_matmul
 * US1 分块口径) + conv 存在性 + W4A16 存在性(auto 池与 carve footer 的
 * VTCM 布局未对齐, 混跑即拒)。M/K/N 取法镜像 serialize_opdef:
 * serialized_extra 非空复用(回读 .bin 场景), 空则调注册表 extractor 现算
 * (全新编译场景 —— serialized_extra 要到 serialize 阶段才落, step6 时恒空)。 */
static VtcmAutoScan vtcm_auto_scan(const GraphPrepare& gp) {
    constexpr uint64_t kVtcmTotal = 16ull << 20;
    VtcmAutoScan r;
    for (const OpDef* od : gp.get_sorted_opdefs()) {
        if (!od->name_tag || !od->name_tag->name()) continue;
        const char* nm = od->name_tag->name();
        if (std::strcmp(nm, "MatMul") != 0 && std::strcmp(nm, "FullyConnected") != 0) {
            if (std::strcmp(nm, "Conv2d") == 0 || std::strcmp(nm, "DepthWiseConv2d") == 0)
                r.conv = true;
            else if (std::strcmp(nm, "GdnBlock") == 0) {
                /* 内核私有 scratch 预算(extractor 闭式值, 128 对齐): 非图张量,
                 * 驻留打包看不见 —— emit 侧 alloc_vtcm_stage 在驻留峰值之上
                 * 叠加, 池预算必须把它算进来(层间时间复用同一段, 取 max)。
                 * extra 序列化时才现算注册表注入, 此处多为空 → 现调 extractor
                 * (MatMul/FC 分支同套路)。 */
                std::vector<uint8_t> xb = od->serialized_extra;
                if (xb.size() < sizeof(ExtraGdnBlock)) xb = compute_op_extra(gp, *od);
                if (xb.size() >= sizeof(ExtraGdnBlock)) {
                    ExtraGdnBlock eg{};
                    std::memcpy(&eg, xb.data(), sizeof(eg));
                    r.gdn_scratch = std::max<uint64_t>(r.gdn_scratch, eg.vtcm_budget_bytes);
                }
            }
            continue;
        }
        uint32_t m = 0, k = 0, n = 0, has_bias = 0, batched = 0;
        std::vector<uint8_t> xb = od->serialized_extra;
        if (std::strcmp(nm, "FullyConnected") == 0) {
            if (xb.empty()) xb = compute_op_extra(gp, *od);
            ExtraFc ef{};
            if (xb.size() >= sizeof(ef))
                std::memcpy(&ef, xb.data(), sizeof(ef));
            m = ef.m; k = ef.k; n = ef.n; has_bias = ef.has_bias;
        } else {
            if (xb.empty()) xb = compute_op_extra(gp, *od);
            ExtraMatMul em{};
            if (!xb.empty()) {   // min-size: 旧 bin 24B 无 has_bias/batched
                size_t c = std::min(xb.size(), sizeof(em));
                std::memcpy(&em, xb.data(), c);
            }
            m = em.m; k = em.k; n = em.n; has_bias = em.has_bias; batched = em.batched;
        }
        if (!m || !k || !n) continue;
        // W4A16 判定(发射侧 slots.len==k*n/2 的编译期同口径): Q4 权重池字节数减半
        if (od->inputs.size() > 1) {
            const OpDef* w = gp.get_op_at(od->inputs[1].src_id);
            if (w && w->const_data_size > 0 &&
                (uint64_t)w->const_data_size * 2u == (uint64_t)k * n) { r.w4 = true; continue; }
            // 权重常驻需求(010 T004): 非 W4A16 权重按 op 求和(共享权重重复计 =
            // 保守上界; 发射侧 f32→f16 洗白后实际字节更小, 此处只做 budget 上界)。
            if (w && w->const_data_size > 0) r.weight_sum += w->const_data_size;
        }
        // 可分块性镜像(op_matmul: !has_bias && !batched && 32 对齐 && 整 op 超界);
        // 二分循环逐字照抄发射侧, 保证 N_chunk 一致 → scratch 峰值一致
        uint64_t need = mm_scratch_need(m, k, n);
        if (!has_bias && !batched && (m % 32u) == 0u && (k % 32u) == 0u && (n % 32u) == 0u &&
            need > kVtcmTotal) {
            uint32_t N_chunk = 0, lo = 32u, hi = n;
            while (lo <= hi) {
                uint32_t mid = lo + ((hi - lo) / 64u) * 32u;
                if (mm_scratch_need(m, k, mid) <= kVtcmTotal) { N_chunk = mid; lo = mid + 32u; }
                else { hi = mid - 32u; }
            }
            if (N_chunk != 0u) need = mm_scratch_need(m, k, N_chunk);
            // N_chunk==0: 维持整 op(发射侧同款 warn 路径), need 不缩
        }
        r.scratch_max = std::max(r.scratch_max, need);
    }
    return r;
}


// Op management - REAL implementation
// Source: do_append_node @ 0xF75F30 (4003 bytes)
op_id_t GraphPrepare::append_node(const std::string& name, uint32_t node_type,
                                    const InputDef* inputs, size_t num_inputs,
                                    const OutputDef* outputs, size_t num_outputs,
                                    const uint8_t* ops_data, size_t ops_data_len) {
    // 1. Validate construction state
    // Source: graph_prepare.cc:2975 "append_node, not in construction phase. state %d"
    if (construction_state_ != 1) {
        return 0; // ERROR: not in construction phase
    }

    // 2. Validate name (not empty, not starting with '$')
    // Source: graph_prepare.cc:2980 "append_node, name \"%s\" not allowed"
    if (name.empty() || name[0] == '$') {
        return 0;
    }

    // 3. Validate node ID != 0
    // Source: graph_prepare.cc:2985 "can't add node with zero id"
    op_id_t new_id = node_type;
    if (new_id == 0) {
        return 0;
    }
    if (new_id >= next_op_id_) next_op_id_ = new_id + 1;

    // 4. Map op name to string_tag
    // Source: string_tag_t::map_str(param_1)
    string_tag_t* name_tag = string_tag_t::map_str(name.c_str());

    // 5. Validate output defs
    // Source: graph_prepare.cc:3016 "node %X, output %d has improper OutputDef"
    if (num_outputs > 0 && outputs) {
        for (size_t i = 0; i < num_outputs; ++i) {
            const OutputDef* od = &outputs[i];
            // FUN_01075e20 validates OutputDef:
            //   rank must be in [0,5]; dtype must be a known DType; when rank>0
            //   the leading dims must be non-zero.
            if (od->rank > 5) return 0;
            if (od->dtype > static_cast<uint32_t>(DType::MXFP4)) return 0;
            if (od->rank > 0 && od->dims[0] == 0) return 0;
        }
    }

    // 6. Check ID not already in graph
    // Source: graph_prepare.cc:3030 "id %llx already in graph"
    if (opdef_map_.count(new_id)) {
        return 0; // ID already exists
    }

    // 7. Handle special Input/Output nodes
    // Source: graph_prepare.cc:3041 "graph has second Input node"
    // Source: graph_prepare.cc:3048 "graph has second Output node"
    if (name == "Input") {
        // 真实库拒绝第二 Input; 我方多输入图每输入一个 Input 节点
        // (id 序 = net.json 声明序, execute_host 按序喂)
        if (input_node_id_ == 0) input_node_id_ = new_id;
    } else if (name == "Output") {
        if (output_node_id_ != 0) return 0; // second Output
        output_node_id_ = new_id;
    }

    // 8. Resolve inputs: each input is (src_id, output_idx) pair
    // Source: graph_prepare.cc:3063 "node %x has input with id %x"
    // Source: graph_prepare.cc:3071 "Output index too high (max %d): %X:%d"
    // Build input list: for each input, look up src opdef, connect outputs
    std::vector<InputConn> input_conns;
    if (num_inputs > 0 && inputs) {
        for (size_t i = 0; i < num_inputs; ++i) {
            uint32_t src_id = inputs[i].rank;        // first 4 bytes = src node id
            uint32_t out_idx = inputs[i].dtype;      // next 4 bytes = output index
            if (src_id == 0 || src_id == new_id) {
                return 0; // invalid input
            }
            if (out_idx > 0xFFF) {
                return 0; // output index too high
            }
            input_conns.push_back({src_id, out_idx});
        }
    }

    // 9. Construct OpDef
    // Source: FUN_01077430 constructs OpDef with:
    //   graph_prepare, op_id, name_tag, input_list, output_def_list
    auto opdef = std::make_unique<OpDef>();
    opdef->vtable = nullptr;
    opdef->flags = OP_ENABLED; // +0x08 bit0
    opdef->string_tag = 0;
    opdef->name_tag = name_tag;
    opdef->graph = this;
    opdef->op_id = new_id;
    opdef->inputs_start = nullptr;
    opdef->inputs_end = nullptr;
    opdef->extra_40 = nullptr;

    // Copy output def if present
    if (num_outputs > 0 && outputs) {
        std::memcpy(&opdef->output_def, &outputs[0], sizeof(OutputDef));
    }

    opdef->tensor_ptr = nullptr;
    opdef->flags2 = 0;
    opdef->vtable2 = nullptr;
    opdef->persistent_tensor = nullptr;

    // Store input connections
    opdef->inputs = std::move(input_conns);

    // 10b. 保存 op_data 参数 blob(loader 打包的 scalar_params 等)
    if (ops_data && ops_data_len > 0) {
        opdef->op_data.assign(ops_data, ops_data + ops_data_len);
    }

    // 10. Quantization validation
    // Source: conditionally_validate_single_quant
    // If +0x61a0 == 0 and name != "Input"/"Output":
    //   validate quant params

    // 11. Register as consumer in source ops
    for (const auto& conn : opdef->inputs) {
        auto src_it = opdef_map_.find(conn.src_id);
        if (src_it != opdef_map_.end() && src_it->second) {
            src_it->second->consumers.push_back(new_id);
        }
    }

    // 12. Store in opdef_map
    op_id_t id = opdef->op_id;
    opdef_map_[id] = std::move(opdef);

    // 13. Mark graph dirty
    graph_dirty_ = true;

    return id;
}

// Source: do_append_const_node @ 0xF776F0 (1441 bytes)
// 旧签名兼容: 盲拷 256 字节(既有调用方语义不变; 新代码用带长度版)
op_id_t GraphPrepare::append_node(const std::string& name, uint32_t node_type,
                                    const InputDef* inputs, size_t num_inputs,
                                    const OutputDef* outputs, size_t num_outputs,
                                    const uint8_t* ops_data) {
    return append_node(name, node_type, inputs, num_inputs, outputs, num_outputs,
                       ops_data, ops_data ? 256 : 0);
}

op_id_t GraphPrepare::append_const_node(uint32_t node_type, const OutputDef& od,
                                          const uint8_t* data, size_t data_len) {
    // Validate construction state。state==2(优化期)也放行: pass rewrite
    // (view_fold 合成 ranges 等)需要建新 const —— 与 fold_op_to_const 在
    // 优化期直接追加 const_pool_ 同一先例, 池追加约定(4 对齐+extents)不变。
    if (construction_state_ != 1 && construction_state_ != 2) return 0;

    op_id_t new_id = (node_type != 0) ? static_cast<op_id_t>(node_type) : next_op_id_++;
    if (opdef_map_.count(new_id)) return 0;
    if (new_id >= next_op_id_) next_op_id_ = new_id + 1;

    auto opdef = std::make_unique<OpDef_Const>(*this, new_id, od, data, data_len);

    // 把常量数据集中存�?const pool，并记录 extent�?    // 4 字节对齐以便反序列化时直�?memcpy�?    // const_pool_ 起始保留 4 字节 (offset 0 作为 "无数�? 哨兵)，所�?    // 第一个真正的常量�?offset 4 开始�?
    if (const_pool_.empty()) const_pool_.assign(4, 0);
    uint64_t aligned_len = (data_len + 3) & ~uint64_t(3);
    uint64_t offset = const_pool_.size();
    if (data && data_len > 0) {
        const_pool_.resize(const_pool_.size() + aligned_len, 0);
        std::memcpy(const_pool_.data() + offset, data, data_len);
    }
    opdef->const_data_offset = (data && data_len > 0) ? offset : 0;
    opdef->const_data_size = data_len;
    const_extents_.push_back({new_id, opdef->const_data_offset, data_len});

    opdef_map_[new_id] = std::move(opdef);

    graph_dirty_ = true;
    return new_id;
}

void GraphPrepare::insert_op(std::unique_ptr<Op> op, bool before) {
    ops_.push_back(std::move(op));
    graph_dirty_ = true;
}

void GraphPrepare::erase_op(op_id_t id) {
    opdef_map_.erase(id);
    graph_dirty_ = true;
}

OpDef* GraphPrepare::get_op_at(op_id_t id) const {
    auto it = opdef_map_.find(id);
    return it != opdef_map_.end() ? it->second.get() : nullptr;
}

// Profiling
void GraphPrepare::mark_time_point(const char* name) {
    // Source: graph_prepare.cc, 890 bytes at 0xF65D90
    // Records timestamp with name for profiling
    time_points_.push_back({name, 0}); // timestamp would be set from clock
}

void GraphPrepare::log_time_points() {
    // Source: graph_prepare.cc, 163 bytes at 0xF661C0
    // Logs all recorded time points
}

// Graph checksum: 64-bit Galois LFSR over all op IDs
// Source: graph_prepare.cc, calculate_graph_checksum @ 0xf6cc00 (verified)
// Walks all ops, feeds each op's 8-byte ID into checksum_bytes(acc, &val, 8).
// This is a 64-bit Galois LFSR (poly=0x1b), NOT CRC32.
uint64_t GraphPrepare::calculate_graph_checksum() const {
    uint64_t acc = 0;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef) continue;
        uint64_t val = id;
        acc = checksum_bytes(acc, reinterpret_cast<const uint8_t*>(&val), 8);
    }
    return acc;
}

bool GraphPrepare::check_connectivity() const {
    // Source: graph_prepare.cc, 3 bytes at 0xF69740
    // Returns true if all ops are connected (no dangling references)
    return true;
}

// allocate_io_tensors
// Source: graph_prepare.cc, 2431 bytes at 0xF69750
// Set up the boundary tensor descriptors for the Input/Output nodes. In the
// host reimplementation there is no real device allocator, so we simply make
// sure the Input/Output nodes carry a well-formed OutputDef (the runtime
// side allocates the actual buffers); the Output node mirrors its producer.
void GraphPrepare::allocate_io_tensors() {
    if (input_node_id_ != 0) {
        auto it = opdef_map_.find(input_node_id_);
        if (it != opdef_map_.end() && it->second) {
            // Input node's output_def is already supplied at append time;
            // ensure the enabled flag is set so it survives optimization.
            it->second->flags |= OP_ENABLED;
        }
    }
    if (output_node_id_ != 0) {
        auto it = opdef_map_.find(output_node_id_);
        if (it != opdef_map_.end() && it->second && !it->second->inputs.empty()) {
            auto src = opdef_map_.find(it->second->inputs[0].src_id);
            if (src != opdef_map_.end() && src->second) {
                it->second->output_def = src->second->output_def;
            }
        }
    }
}

// const_tracking_setup
// Source: graph_prepare.cc, const_tracking_setup @ 0xEE8C30 (329 bytes)
void GraphPrepare::const_tracking_setup() {
    // Initialize const tracking: scan all const ops, record their IDs
    // +0x45dc state must be in construction phase
}

void GraphPrepare::const_tracking_finalize() {
    // Source: graph_prepare.cc, const_tracking_finalize @ 0xEEA830 (136 bytes)
}

void GraphPrepare::const_tracking_after_prep(HexagonNNEnv& env) {
    // Source: graph_prepare.cc, const_tracking_after_prep @ 0xEEA8C0 (502 bytes)
}

void GraphPrepare::add_tracked_id(op_id_t id, const OpDef& opdef, bool force) {
    // Source: graph_prepare.cc, add_tracked_id @ 0xEE9680 (15 bytes)
}

void GraphPrepare::identify_updateable_quant_ops() {
    // Source: graph_prepare.cc, 417 bytes at 0xF6A300
    // Scan ops for quantization params that can be updated at runtime
}

void GraphPrepare::run_predication_pass() {
    // Source: graph_prepare.cc, 73 bytes at 0xF6CAF0
    // Analyze ops for predication opportunities
}

void GraphPrepare::mark_op_deletable(OpDef* op) {
    if (op) op->flags |= OP_DEAD;
}

void GraphPrepare::mark_op_deletable(op_id_t id) {
    auto it = opdef_map_.find(id);
    if (it != opdef_map_.end() && it->second) {
        it->second->flags |= OP_DEAD;
    }
}

void GraphPrepare::collect_deletable_nodes() {
    // Source: graph_prepare.cc, 913 bytes at 0xF78B40
    // Iterate marked-deletable ops and remove them
    for (auto it = opdef_map_.begin(); it != opdef_map_.end(); ) {
        if (it->second && (it->second->flags & OP_DEAD)) {
            it = opdef_map_.erase(it);
        } else {
            ++it;
        }
    }
}

void GraphPrepare::gen_quant_params_hash(uint64_t& hash) const {
    // Source: graph_prepare.cc, 206 bytes at 0xF661C0
    hash = 0;
    for (auto& [id, opdef] : opdef_map_) {
        if (opdef) {
            hash ^= id * 0x192E2101;
            hash = (hash << 1) | (hash >> 63);
        }
    }
}

bool GraphPrepare::needs_activation_fixup(DType dt) const {
    // Source: graph_prepare.cc, 10 bytes at 0xF774B0
    return dt == DType::Int8 || dt == DType::Int16;
}

void GraphPrepare::fixup_signed_activations(OutputDef& od) const {
    // Source: graph_prepare.cc, 42 bytes at 0xF774C0
    // Adjust signed activation parameters
}

void GraphPrepare::reapply_signed_activations(OutputDef& od) {
    // Source: graph_prepare.cc, 42 bytes at 0xF82ED0
}

size_t GraphPrepare::get_vtcm_tile_size() const {
    // Source: graph_prepare.cc @ 0xF81CB0, sap_reduce_bandwidth @ 0xf6ee78
    // Returns usable VTCM = min(budget, 4MB) × 0.75
    // 0x400000 = 4MB hard limit; 0.75 coefficient (0x3fe8000000000000 as double)
    // 1/4 reserved for system/stack
    // 008 T006 预算单源化: 显式 --plan-vtcm-budget 优先(真 VTCM=16MB 口径,
    // 两处消费 supertile 规划 + tcm_migration 同数); 缺省 0 走旧 3MB 路径
    // (惰性门: 无旗标时产物逐字节不变)。
    if (plan_vtcm_budget_) return plan_vtcm_budget_;
    constexpr uint64_t VTCM_4MB = 0x400000;
    constexpr double VTCM_FACTOR = 0.75;
    uint64_t budget = 4 * 1024 * 1024; // default 4MB
    return static_cast<size_t>(static_cast<double>(std::min(budget, VTCM_4MB)) * VTCM_FACTOR);
}

op_id_t GraphPrepare::get_pretiling_op_id(op_id_t id) const {
    // Source: graph_prepare.cc, 12 bytes at 0xF64B60
    return id;
}

op_id_t GraphPrepare::get_ct_op_id(op_id_t id) const {
    // Source: graph_prepare.cc, 12 bytes at 0xF64B50
    return id;
}

op_id_t GraphPrepare::extract_op(op_id_t id) {
    // Source: graph_prepare.cc, 290 bytes at 0xF75E00
    return id;
}

void GraphPrepare::set_node_ids(uint32_t start, uint32_t end, uint32_t base) {
    // Source: graph_prepare.cc, 1077 bytes at 0xF80150
    // Renumber op IDs from [start, end) starting at base
}

op_id_t GraphPrepare::op_def_posn(op_id_t id) const {
    // Source: graph_prepare.cc, 69 bytes at 0xF61930
    return id;
}

void GraphPrepare::change_opstr(OpDef* opdef, string_tag_t* new_tag, const char* str, uint32_t len) {
    // Source: graph_prepare.cc, 323 bytes at 0xF74B70
    if (opdef) {
        opdef->name_tag = new_tag;
    }
}

void GraphPrepare::change_input(OpDef* opdef, uint32_t idx, op_id_t new_src, const char* str, uint32_t len) {
    // Source: graph_prepare.cc, 991 bytes at 0xF80E80
    // Change input[idx] of opdef to point to new_src
}

void GraphPrepare::replace_with(OpDef* old, op_id_t new_id, const char* str, uint32_t len, bool keep) {
    // Source: graph_prepare.cc, 696 bytes at 0xF819F0
    // Replace old opdef with new_id in all consumers
    if (!keep) {
        old->flags |= OP_DEAD;
    }
}

void GraphPrepare::replace_opdef_with_opconst(OpDef& old, std::unique_ptr<OpDef> replacement) {
    // Source: graph_prepare.cc, 609 bytes at 0xF7A950
    // Replace old opdef with a const opdef
    if (replacement) {
        opdef_map_[replacement->op_id] = std::move(replacement);
        old.flags |= OP_DEAD;
    }
}

bool GraphPrepare::fold_op_to_const(OpDef* op, const uint8_t* data, size_t data_len) {
    // 折叠基础设施(图优化 pass 共用): op → 持有值的 const 节点(同 op_id,
    // 消费者 inputs 不用改线)。const_pool_ 追加 4 对齐(与 append_const_node
    // 同约定, 反序列化直接 memcpy), const_extents_ 登记。
    if (!op || !op->is_enabled() || op->is_dead() || op->is_const()) return false;
    if (!data || data_len == 0) return false;

    if (const_pool_.empty()) const_pool_.assign(4, 0);
    uint64_t aligned_len = (data_len + 3) & ~uint64_t(3);
    uint64_t offset = const_pool_.size();
    const_pool_.resize(const_pool_.size() + aligned_len, 0);
    std::memcpy(const_pool_.data() + offset, data, data_len);

    auto replacement = std::make_unique<OpDef_Const>(*this, op->op_id,
                                                     op->output_def, nullptr, 0);
    replacement->const_data_offset = offset;
    replacement->const_data_size = data_len;
    // 继承消费者关系与 grouping —— 否则新 const 的 consumers 为空,
    // DCE 当孤儿删掉 → 下游读空槽(测试[1] 实锤: after=1 只剩 Output)。
    replacement->consumers = op->consumers;
    replacement->grouping = op->grouping;
    const_extents_.push_back({op->op_id, offset, data_len});

    replace_opdef_with_opconst(*op, std::move(replacement));
    graph_dirty_ = true;
    return true;
}

void GraphPrepare::note_new_node(const OpDef& opdef, const char* str, uint32_t len) {
    // Source: graph_prepare.cc, 424 bytes at 0xF81840
    // Record new node for replacement tracking
}

void GraphPrepare::note_replace(op_id_t old, const std::vector<OpRef>& refs,
                                 op_id_t new_id, uint32_t idx, const std::string& str) {
    // Source: graph_prepare.cc, 1104 bytes at 0xF812D0
    // Record replacement of old with new_id
}

void GraphPrepare::note_outputs(void* file, const OpDef& opdef) {
    // Source: graph_prepare.cc, 501 bytes at 0xF7B1D0
    // Log opdef outputs to file
}

void GraphPrepare::supersede_op(OpDef* old, op_id_t new_id, bool keep) {
    // Source: graph_prepare.cc, 179 bytes at 0xF7B110
    replace_with(old, new_id, nullptr, 0, keep);
}

void GraphPrepare::supersede_outputless_op(OpDef* old, op_id_t new_id) {
    // Source: graph_prepare.cc, 1563 bytes at 0xF81D60
    old->flags |= OP_DEAD;
}

void GraphPrepare::opdef_delete(op_id_t id) {
    // Source: graph_prepare.cc, 1187 bytes at 0xF7B3D0
    opdef_map_.erase(id);
    graph_dirty_ = true;
}

std::unique_ptr<Op> GraphPrepare::op_factory_generate(const struct OpIoPtrs& io, op_id_t id) {
    // Source: op_registry_prepare.cc, op_factory_generate @ 0x10BE710 (1197 bytes, ELF st_size)
    // For each registered constructor matching the op name:
    //   Try to construct an Op, query Op::cost(), select the lowest cost.
    // Delegate to the global registry's name-based generator.
    return OpRegistry::instance().generate(io, id);
}

void GraphPrepare::pprint() const {
    // Source: graph_prepare.cc, 1659 bytes at 0xF7BC50
    // Pretty-print the graph
}

void GraphPrepare::graphviz_pprint(const char* filename, bool full) const {
    // Source: graph_prepare.cc, 849 bytes at 0xF68F40
    // Output graphviz DOT format
}

void GraphPrepare::python_pprint_graph_summary(const char* filename, bool full, bool late) const {
    // Source: graph_prepare.cc, 270 bytes at 0xF74A60
    // Python-readable graph summary
}

void GraphPrepare::python_pprint_detail() const {
    // Source: graph_prepare.cc, 3455 bytes at 0xF6DDB0
}

void GraphPrepare::python_pprint_runlist() const {
    // Source: graph_prepare.cc, 296 bytes at 0xF7F900
}

uint32_t GraphPrepare::num_profiling_timepoints(uint32_t* count) const {
    // Source: graph_prepare.cc, 21 bytes at 0xF66290
    if (count) *count = static_cast<uint32_t>(time_points_.size());
    return static_cast<uint32_t>(time_points_.size());
}

void GraphPrepare::clear_profiling_info() {
    // Source: graph_prepare.cc, 16 bytes at 0xF66350
    time_points_.clear();
}

void GraphPrepare::serialize_profiling_timepoints(struct profilingevent* events, uint32_t count) {
    // Source: graph_prepare.cc, 157 bytes at 0xF662B0
}

void GraphPrepare::mark_prepare_stage(std::pair<std::string, uint64_t> stage) {
    // Source: graph_prepare.cc, 101 bytes at 0xF65D20
    time_points_.push_back({stage.first.c_str(), stage.second});
}

void GraphPrepare::setup_rewrite_log() {
    // Source: graph_prepare.cc, 1 byte at 0xF64980
}

void GraphPrepare::close_rewrite_log() {
    // Source: graph_prepare.cc, 301 bytes at 0xF64990
}

void GraphPrepare::add_replacement_recorder() {
    // Source: graph_prepare.cc, 12 bytes at 0xF64AC0
}

void GraphPrepare::enable_replacement_recording() {
    // Source: graph_prepare.cc, 8 bytes at 0xF64AD0
}

void GraphPrepare::start_replacement_recorder(void** recorder) {
    // Source: graph_prepare.cc, 49 bytes at 0xF64AE0
}

void GraphPrepare::stop_replacement_recorder(void* recorder) {
    // Source: graph_prepare.cc, 30 bytes at 0xF64B20
}

op_id_t GraphPrepare::get_oldest_replaced_id(void* recorder, op_id_t id) const {
    // Source: graph_prepare.cc, 12 bytes at 0xF64B40
    return id;
}

void GraphPrepare::make_sorted_optrs() {
    // Source: graph_prepare.cc, GraphOrdering::make_sorted_optrs @ 0xF82AE0 (750 bytes)
    // Sort ops by ordering_ index
}

op_id_t GraphPrepare::lookup_op_in_ordering(OrderInfo* ordering, int idx, op_id_t id) const {
    // Source: graph_prepare.cc, 65 bytes at 0xF828B0
    return id;
}

bool GraphPrepare::serialize_file(int fd) const {
    // Source: graph_prepare.cc, 455 bytes at 0xF830D0
    return false;
}

bool GraphPrepare::serialize_patch_metadata(class FileSerializer& fs) const {
    // Source: graph_prepare.cc, 180 bytes at 0xF60A20
    return false;
}

bool GraphPrepare::serialize_replaceable_constpool(class FileSerializer& fs) const {
    // Source: graph_prepare.cc, 206 bytes at 0xF60AE0
    return false;
}

void GraphPrepare::adjust_heap_stats(Serializer& ser) const {
    // Source: graph_prepare.cc, 216 bytes at 0xF65C30
    // Adjust heap statistics after serialization prescan
}

bool GraphPrepare::is_dynamic_inputs_active() const {
    // Source: graph_prepare.cc, 11 bytes at 0xF17700
    return false;
}

bool GraphPrepare::is_dynamic_dma(op_id_t id) const {
    // Source: graph_prepare.cc, 107 bytes at 0xF17720
    return false;
}

void GraphPrepare::dynamic_inputs_pre_optimization_pass() {
    // Source: graph_prepare.cc, 264 bytes at 0xF17800
}

void GraphPrepare::dynamic_inputs_post_optimization_pass() {
    // Source: graph_prepare.cc, 98 bytes at 0xF17C00
}

void GraphPrepare::get_nsp_id_mapping(struct NspIdMap& map) const {
    // Source: graph_prepare.cc, 1077 bytes at 0xF80400
}

void GraphPrepare::phys_alloc_in_runlist(const std::vector<Op*>& ops) {
    // Source: graph_prepare.cc, 874 bytes at 0xF72910
    // Physical allocation for runlist ops
}

void GraphPrepare::force_contiguous_allocate_mcrecv_blocks(
    const VtcmCacheInstance& vtcm, const std::vector<uint32_t>& tags) {
    // Source: graph_prepare.cc, 178 bytes at 0xF823A0
}

void GraphPrepare::link_source_destructive_operands(const std::vector<uint32_t>& tags) {
    // Source: graph_prepare.cc, 752 bytes at 0xF82490
}

void GraphPrepare::allocate_for_reschedule_grdep(
    const VtcmCacheInstance& vtcm, const std::vector<uint32_t>& tags, bool) {
    // Source: graph_prepare.cc, 294 bytes at 0xF82780
}

void GraphPrepare::pysequencer(std::vector<uint32_t>& runlist_tags) {
    // Source: graph_prepare.cc, 1900 bytes at 0xF6BB40
    // Python sequencer integration
}

void GraphPrepare::show_runlist(GraphDeps& deps, const std::vector<uint32_t>& tags, int idx) const {
    // Source: graph_prepare.cc, 1 byte at 0xF6B900
}

void GraphPrepare::dump_runlist(GraphDeps& deps, const std::vector<uint32_t>& tags, const char* filename) const {
    // Source: graph_prepare.cc, 1 byte at 0xF6B920
}

bool GraphPrepare::is_moe_aggregator(op_id_t id) const {
    // Source: graph_prepare.cc, 25 bytes at 0xF17E40
    return false;
}

bool GraphPrepare::is_part_of_moe_block(op_id_t id) const {
    // Source: graph_prepare.cc, 25 bytes at 0xF17E80
    return false;
}

void GraphPrepare::get_moe_block_ops_grouped_by_branch(uint16_t block_id) const {
    // Source: graph_prepare.cc, 56 bytes at 0xF17EB0
}

void GraphPrepare::get_all_moe_block_ids() const {
    // Source: graph_prepare.cc, 53 bytes at 0xF17DD0
}

bool GraphPrepare::multi_quant_transformed_to_single_quant() const {
    // Source: graph_prepare.cc, 11 bytes at 0xF17F30
    return false;
}

void GraphPrepare::get_dynamically_switchable_blocks() const {
    // Source: graph_prepare.cc, 75 bytes at 0xF17F50
}

void GraphPrepare::sanity_check_null_exec(op_id_t id, const Op* op) const {
    // Source: graph_prepare.cc, 521 bytes at 0xF6D220
    // Verify op has valid execution path
}

void GraphPrepare::log_mux_match_fail(const GraphOptInfo& info, op_id_t a, op_id_t b, string_tag_t* tag) {
    // Source: graph_prepare.cc, 20 bytes at 0xF17FF0
}

op_id_t GraphPrepare::update_tensor_map_with_duplicates(op_id_t id, const std::vector<OpDef*>& ops) {
    // Source: graph_prepare.cc, 575 bytes at 0xF18020
    return id;
}

op_id_t GraphPrepare::update_tensor_map_with_combined(op_id_t id, const std::vector<OpDef*>& ops) {
    // Source: graph_prepare.cc, 546 bytes at 0xF18260
    return id;
}

void GraphPrepare::set_wtshare_metadata_filename(const char* filename) {
    // Source: graph_prepare.cc, 26 bytes at 0xF64D20
}

void GraphPrepare::serialize_sharing_metadata(const char* filename) const {
    // Source: graph_prepare.cc, 111 bytes at 0xF64D40
}

void GraphPrepare::remove_input_refs(op_id_t id) {
    // Source: graph_prepare.cc, 180 bytes at 0xF7A890
}

bool GraphPrepare::truegraph_is_needed(const OpDef* opdef) const {
    // Source: graph_prepare.cc, 351 bytes at 0xF7CF50
    return false;
}

op_id_t GraphPrepare::truegraph_source_id(OpRef ref) const {
    // Source: graph_prepare.cc, 180 bytes at 0xF7E210
    return ref.op_id;
}

void GraphPrepare::truegraph_outputdefs(const OpDef* opdef) const {
    // Source: graph_prepare.cc, 173 bytes at 0xF7E2D0
}

bool GraphPrepare::is_native_KV_success() const {
    // Source: graph_prepare.cc, 213 bytes at 0xF6B740
    return false;
}

void GraphPrepare::debug_grdeps(GraphDeps& deps, const char* name) {
    // Source: graph_prepare.cc, 1 byte at 0xF6B8F0
}

// ===== build_graph_deps (反汇编确�? @ 0xfac220, 8216B) =====
// 文件: grdep_main.cc, 字符�?"GRAPHDEPS_DUMP"
// 输入: GraphPrepare 自身 (opdef_map_ + 配置)
// 输出: GraphDeps 对象 (存入 this+0x7468), 内含:
//        - OpDesc[op_id] 数组 (�?op 一�?120B 描述�?
//        - op_id �?OpDesc hash �?//        - 依赖�?(predecessors/successors via memgroup)
//        - 生命�?(first_use/last_use/access_count)
//        - memgroup �?// �?FancyAllocator (VTCM/DDR 分配) �?runlist 调度消费
GraphStatus GraphPrepare::build_graph_deps() {
    // 释放�?GraphDeps (反汇�? 0xfac26b-0xfac28e 先析构旧对象)
    graph_deps_ = std::make_unique<GraphDeps>();
    GraphDeps& deps = *graph_deps_;

    // Step 1: 构�?(反汇�? 0xfb1190 GraphDeps::GraphDeps, 初始�?0x00-0x490 字段)
    deps.init(*this);

    // Step 2: 为每�?op 创建 OpDesc (反汇�? 0xfac2f8 �?opdef 容器, 0xfa3630 fibonacci hash 插入)
    std::vector<const OpDef*> sorted = get_sorted_opdefs();
    for (auto* od : sorted) {
        if (!od || !od->is_enabled() || od->is_dead()) continue;
        const Op* op_ptr = nullptr;
        for (auto& op : ops_) {
            auto* top = dynamic_cast<const TypicalOp*>(op.get());
            if (top && top->op_id == od->op_id) { op_ptr = op.get(); break; }
        }
        deps.add_op(od->op_id, od, op_ptr);
    }

    // Step 3: 构建 memgroup (反汇编字符串: "memgroup(bytes=%zu, is_tcm=%s, gen_op='#%d'")
    deps.build_memgroups(*this);

    // Step 4: 建立依赖�?(op 输入 �?�?op 输出 memgroup)
    deps.build_dependencies(*this);

    // Step 5: 计算生命�?(FancyAllocator::set_lifetimes @ 0xf49960 消费)
    deps.compute_lifetimes(*this);

    // Step 6: 构建 runlist (反汇�? 0xfb10a0, 4864B)
    deps.build_runlist();

    // Step 7: 估算 VTCM/DDR 需�?
    for (auto& desc : deps.opdescs) {
        if (!desc.opdef_ptr) continue;
        const OutputDef& od = desc.opdef_ptr->output_def;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od.rank && d < 5; ++d) {
            if (od.dims[d] > 0) sz *= od.dims[d];
        }
        uint64_t esize = od.element_size ? od.element_size : 4;
        desc.vtcm_requirement = sz * esize;
        desc.ddr_requirement = desc.vtcm_requirement / 5;
    }

    // Step 8: Scheduler 计划(ST-Cut 链)产执行序 —— 产品路径的调度决策。
    // 结果存入 plan_order_, do_serialize 按此序发射 op 记录并写出
    // TAG_PLAN_ORDER(反序列化回读, round-trip 确定性)。
    {
        Scheduler sched;
        Scheduler::Plan plan = sched.schedule(*this);
        plan_order_.clear();
        plan_order_.reserve(plan.op_order.size());
        for (uint32_t id : plan.op_order) plan_order_.push_back(static_cast<op_id_t>(id));
    }

    return GraphStatus::Success;
}
// ===== VTCM lifetime-aware allocation (反汇编路径) =====
// Source: libHtpPrepare.so x86_64 v2.48
//   allocate_tcm_blocks_internal     @ 0x13b29d0
//   sort_blocks_by_reverse_lifetime_end @ 0x13b21b0
//   allow_tensor_overlap             @ 0x13a14b0
// Collects non-const, non-spilled tensors with sizes + lifetimes from
// GraphDeps, calls FancyAllocator::allocate_with_lifetime, stores results.
void GraphPrepare::vtcm_lifetime_alloc(VtcmCacheInstance& vtcm) {
    vtcm_allocations_.clear();
    cp_active_ = false;
    cp_plans_.clear();
    cp_solution_ = cp::CPSolution{};
    cp_op_emitter_.reset();
    if (!graph_deps_) return;

    const size_t budget = vtcm.usable_size();

    // Build allocation requests from DepOpDesc entries.
    // Skip: const tensors (persistent, not in scratch pool),
    //       spilled tensors (flags2 & SPILL_TO_DDR = 0x40, set by tcm_migration).
    constexpr uint32_t SPILL_TO_DDR = 0x40;
    std::vector<fa::FancyAllocator::AllocRequest> requests;

    for (const auto& desc : graph_deps_->opdescs) {
        if (!desc.opdef_ptr) continue;
        const OpDef& od = *desc.opdef_ptr;
        if (od.is_const()) continue;             // persistent weights/constants
        if (od.flags2 & SPILL_TO_DDR) continue;   // already marked for DDR
        if (desc.vtcm_requirement == 0) continue; // zero-size output

        fa::FancyAllocator::AllocRequest req;
        req.op_id = desc.op_id;
        req.size = desc.vtcm_requirement;
        req.life_begin = desc.life_begin;
        req.life_end = desc.life_end;
        requests.push_back(req);
        if (getenv("HNNX_CP_DEBUG")) {
            const OutputDef& od2 = od.output_def;
            std::fprintf(stderr, "  req op%u size=%llu od.rank=%u dims=[%llu,%llu,%llu,%llu,%llu] esz=%llu life=[%u,%u]\n",
                         desc.op_id, (unsigned long long)desc.vtcm_requirement, od2.rank,
                         (unsigned long long)od2.dims[0], (unsigned long long)od2.dims[1],
                         (unsigned long long)od2.dims[2], (unsigned long long)od2.dims[3],
                         (unsigned long long)od2.dims[4], (unsigned long long)od2.element_size,
                         desc.life_begin, desc.life_end);
        }
    }

    if (requests.empty()) return;

    // CP oracle branch (HNNX_VTCM_ALLOCATOR=cp*): provably-optimal Eq.3
    // schedule over the SAME request set. Any failure (infeasible, guardrail
    // abort, verify mismatch) falls through to the greedy path unchanged.
    cp::CPOptions cp_opts;
    if (cp::parse_cp_options_from_env(&cp_opts) &&
        vtcm_lifetime_alloc_cp(vtcm, requests, cp_opts)) {
        return;
    }

    // Run lifetime-end sort + identical-interval grouping + event-sweep reuse.
    auto results = vtcm.allocator().allocate_with_lifetime(requests, budget);

    // Store into vtcm_allocations_ (convert AllocResult -> VtcmAllocEntry).
    for (const auto& [op_id, res] : results) {
        vtcm_allocations_[op_id] = {res.offset, res.block_id, res.spilled};
    }
}

// ===== CP oracle branch (US20240386237 ideal-solution solver) =====
// Upstream bridge: requests (identical collection to the greedy path) +
// edge set from OpDef::inputs (the producer->consumer relation
// compute_lifetimes walks) + topo hint from ordering_ + M = usable_size().
// Downstream channel 1: resident tensors -> vtcm_allocations_ (same
// AllocResult contract as the greedy path, so vtcm_overlap_alloc /
// finalize / pools consume it unchanged).
// Downstream channel 2: PAGE tensors -> cp_plans_ (topo-index positions,
// consumed by post_spill_fill_design_pass).
bool GraphPrepare::vtcm_lifetime_alloc_cp(
    VtcmCacheInstance& vtcm,
    const std::vector<fa::FancyAllocator::AllocRequest>& requests,
    const cp::CPOptions& opts) {
    std::unordered_set<op_id_t> req_ids;
    req_ids.reserve(requests.size());
    for (const auto& r : requests) req_ids.insert(r.op_id);

    std::vector<std::pair<op_id_t, op_id_t>> edges;
    for (const auto& desc : graph_deps_->opdescs) {
        if (!desc.opdef_ptr || !req_ids.count(desc.op_id)) continue;
        for (const auto& in : desc.opdef_ptr->inputs)
            if (in.src_id != desc.op_id && req_ids.count(in.src_id))
                edges.emplace_back(in.src_id, desc.op_id);
    }

    std::vector<op_id_t> topo;
    topo.reserve(ordering_.size());
    for (op_id_t id : ordering_)
        if (req_ids.count(id)) topo.push_back(id);

    cp::CPProblem problem =
        cp::build_problem_from_requests(requests, edges, vtcm.usable_size(), topo, opts);
    cp::CPProblem keep = problem;
    if (getenv("HNNX_CP_DEBUG")) {
        std::fprintf(stderr, "cp problem: n=%u M=%llu W=%llu nodes:", problem.n(),
                     (unsigned long long)problem.M, (unsigned long long)problem.W);
        for (uint32_t v = 0; v < problem.n(); v++)
            std::fprintf(stderr, " [%u]op%u m=%llu w=%llu C=%u", v, problem.nodes[v].id,
                         (unsigned long long)problem.nodes[v].m,
                         (unsigned long long)problem.nodes[v].w, problem.nodes[v].C);
        std::fprintf(stderr, "\n");
    }
    cp::CPSolver solver(std::move(problem), opts);
    cp::CPSolution sol = solver.solve();
    if (!sol.feasible) {
        std::fprintf(stderr,
                     "vtcm_lifetime_alloc: cp%s infeasible (%s); falling back to greedy\n",
                     opts.mode_name(), sol.abort_reason.c_str());
        return false;
    }

    auto results = cp::to_alloc_results(sol, keep, vtcm.usable_size());
    for (const auto& [op_id, res] : results)
        vtcm_allocations_[op_id] = {res.offset, res.block_id, res.spilled};

    cp_plans_ = cp::to_spill_fill_plans(sol, keep);
    cp_solution_ = sol;
    cp_active_ = true;
    std::fprintf(stderr,
                 "vtcm_lifetime_alloc: cp%s ddr=%llu bytes (%u spill, %u remat), "
                 "peak=%llu/%zu, optimal=%s, plans=%zu\n",
                 opts.mode_name(), (unsigned long long)sol.ddr_bytes, sol.spill_count,
                 sol.remat_count, (unsigned long long)sol.peak_resident,
                 vtcm.usable_size(), sol.optimal ? "yes" : "no", cp_plans_.size());
    return true;
}

// ===== VTCM per-Op overlap allocation (反汇编 @ 0x13a14b0) =====
// Implements the real .so's producer-consumer VTCM reuse:
// For each op in topological order, calls allow_tensor_overlap_opdef which:
//   1. Enumerates input block IDs from OpDef::inputs
//   2. Calls force_contiguous to register/allocate blocks in Fibonacci hash table
//   3. Calls link_blocks to connect first input → output for reuse
void GraphPrepare::vtcm_overlap_alloc(VtcmCacheInstance& vtcm) {
    overlap_allocations_.clear();
    if (!graph_deps_) return;

    auto& allocator = vtcm.allocator();
    allocator.set_overlap_budget(vtcm.usable_size());

    // Process ops in topological order (反汇编: per-Op loop in do_prepare1)
    std::vector<const OpDef*> sorted = get_sorted_opdefs();

    for (uint32_t topo_idx = 0; topo_idx < sorted.size(); ++topo_idx) {
        const OpDef* od = sorted[topo_idx];
        if (!od || !od->is_enabled() || od->is_dead()) continue;

        allocator.allow_tensor_overlap_opdef(*od, *graph_deps_, topo_idx);
    }

    // Copy results to overlap_allocations_
    for (const auto& [op_id, alloc] : allocator.overlap_allocs()) {
        overlap_allocations_[op_id] = alloc;
    }
}

// ===== VTCM block allocation finalize (反汇编 @ 0x13B29D0) =====
// Calls allocate_tcm_blocks_internal to sort blocks by reverse lifetime,
// probe Fibonacci hash for reuse opportunities, compute total VTCM usage,
// and check against budget (行 2425 ERROR / 行 2429 STAT).
void GraphPrepare::vtcm_block_alloc_finalize(VtcmCacheInstance& vtcm) {
    if (!graph_deps_) return;
    auto& allocator = vtcm.allocator();

    // Collect all block IDs that were registered by allow_tensor_overlap_opdef
    std::vector<uint32_t> block_ids;
    for (const auto& [bid, idx] : allocator.block_id_to_idx()) {
        block_ids.push_back(bid);
    }

    fa::FancyAllocator::TcmAllocOptions opts;
    opts.pin_persistent = true;
    opts.nsp_id = vtcm.nsp_id();
    allocator.allocate_tcm_blocks_internal(*graph_deps_, block_ids, opts);
}

// ===== Post spill/fill design pass (反汇编 @ 0x129ed30, 768B = 0x300) =====
// slc_graph_prepare.cc — called after spill/fill insertion to finalize DMA design.
//
// 反汇编确认的控制流 (sym_master: post_spill_fill_design_pass @ 0x129ed30, 768B):
//
// 1. 入口 flag 检查 (0x129ed4d): cmpb $0x0,0x611d(%rdi) — 读 [this+0x611d]
//    0x129ed54: je 0x129ef93 — flag==0 → 直接跳函数尾声返回 (无 spill/fill)
//
// 2. flag!=0 路径 (0x129ed5a-0x129eda6):
//    0x129ed5a: r15 = this
//    0x129ed5d: r8 = this + 0x54d0 (VtcmCacheInstance*)
//    0x129ed64: rax = [this+0x7468] (GraphDeps*)
//    0x129ed6b-0x129ed75: 从 runlist_tags 向量算 count = (end-begin)/4
//    0x129ed84: call 0x12934b0 — spill/fill sequencing 主阶段
//       (静态局部函数; objdump 误标为 sequencing_stage@@Base+0xb2410,
//        但 sym_master 确认 sequencing_stage 实为 0x11e10a0 — 符号边界重叠问题)
//
// 3. 配置名字符串拷贝 (0x129ed93): call basic_string C1 — 拷贝 [this+0x6128]
//    0x129eda6: call 0x1298d00 — 静态局部函数 (处理配置)
//
// 4. op count 检查 (0x129edbb-0x129edc4): 读 [this+0x6120]
//    0x129edbb: mov 0x6120(%r15),%esi
//    0x129edc2: test %esi,%esi
//    0x129edc4: jle 0x129ee5a — count<=0 → 跳过 design pass
//    count>0 路径 (0x129edca-0x129ee33):
//      0x129edca-0x129ee15: 构造 5 元素字符串 vector
//        0x129ee15: call 0x129f0b0 — 静态局部函数, 构建 5 个字符串 tag
//      0x129ee24: call 0x1299ea0 — 清理/析构 (释放 [rsp+0x78] 字符串)
//      0x129ee33: call 0x1299ee0 — 清理/析构 (释放 [rsp+0x60] 字符串)
//
// 5. 第二 flag 检查 (0x129ee5a): cmpb $0x0,0x611e(%r15) — 读 [this+0x611e]
//    0x129ee62: je 0x129ef06 — flag==0 → 跳尾声清理
//    flag!=0 路径 (0x129ee68-0x129ef01): 字符串比较 ([this+0x5fa8] 与常量)
//      0x129ef01: call 0x1295a80 — 静态局部函数
//      0x129efb3: call GetLogPriorityLevel; 0x129efb8: cmp $0x3,%eax
//      0x129efdd: call qnndsp_log (level 3, "slc_allocator_debug flag was set...")
//
// 6. 尾声清理 (0x129ef06-0x129ef93): 释放多个栈字符串/vector
//    0x129ef30: call 0x1299f20; 0x129effc: call 0x1299dc0
//    0x129ef93: 栈保护检查; 0x129efb2: ret
//
// 子函数 0x12934b0/0x1298d00/0x129f0b0/0x1299ea0/0x1299ee0/0x1299f20/0x1299dc0
// 均为无符号的静态局部函数, 内部算法需单独反汇编; 本函数内仅记录其调用点与参数。
void GraphPrepare::post_spill_fill_design_pass(const std::vector<uint32_t>& runlist_tags) {
    // 反汇编 0x129ed4d: cmpb $0x0,0x611d(%rdi) — [this+0x611d] spill/fill enabled flag
    // 反汇编 0x129ed54: je 0x129ef93 — flag==0 → 直接返回 (无 spill/fill)
    // REQNN: 无 [this+0x611d] flag; 以 graph_deps_ 是否有效代替 (空 → 提前返回, 对应 je 路径)
    if (!graph_deps_) return;

    // 反汇编 0x129ed84: call 0x12934b0 (spill/fill sequencing 主阶段, 静态局部函数)
    // 反汇编 0x129ed93: call basic_string C1 (拷贝 [this+0x6128] 配置名)
    // 反汇编 0x129eda6: call 0x1298d00 (静态局部函数, 处理配置)
    // REQNN: 无 spill/fill sequencing 实现
    (void)runlist_tags;

    // 反汇编 0x129edbb: mov 0x6120(%r15),%esi — [this+0x6120] spill/fill op count
    // 反汇编 0x129edc2: test %esi,%esi
    // 反汇编 0x129edc4: jle 0x129ee5a — count<=0 → 跳过 design pass
    // REQNN: 无 [this+0x6120] 字段 (由 0x12934b0 填充); op count = 0 → jle 路径
    uint32_t spill_fill_op_count = 0;

    // CP oracle branch (Phase B, record level): the .so's 0x12934b0 stage is
    // where real spill/fill ops are sequenced; for the CP path we replay the
    // solver's plans through the OpEmitter. Positions are already topo-index
    // space (to_spill_fill_plans maps CP slots -> consumer topo indices;
    // respect_topo_order keeps slot order a topo subsequence, so the mapping
    // is order-preserving).
    if (cp_active_ && !cp_plans_.empty()) {
        bool ok = true;
        uint64_t arena = 0;
        for (const auto& pl : cp_plans_) {
            if (pl.fill_position <= pl.spill_position) {
                std::fprintf(stderr, "post_spill_fill: plan op %llu fill (%zu) not after "
                                     "spill (%zu) — dropping CP plans\n",
                             (unsigned long long)pl.op_id, pl.fill_position, pl.spill_position);
                ok = false;
                break;
            }
            if (pl.size == 0) { ok = false; break; }
            arena += pl.size;
        }
        // Eq.9 same-address rule is structural (to_spill_fill_plans copies
        // s0.ddr_offset into both ends of the plan); the arena check bounds
        // the DDR side against the solution's spill total.
        if (ok && arena > cp_solution_.ddr_arena_used) {
            std::fprintf(stderr, "post_spill_fill: plan arena %llu exceeds solution arena "
                                 "%llu — dropping CP plans\n",
                         (unsigned long long)arena,
                         (unsigned long long)cp_solution_.ddr_arena_used);
            ok = false;
        }
        if (ok) {
            cp_op_emitter_ = std::make_unique<OpEmitter>(this);
            for (const auto& pl : cp_plans_) {
                // vtcm_offset here is the FILL destination (seg1); seg0 and
                // seg1 share it whenever first-fit could reuse it.
                cp_op_emitter_->insert_spill_fill_pair(
                    pl.vtcm_offset, pl.ddr_offset, pl.size, pl.spill_position,
                    pl.fill_position, pl.double_buffered);
                spill_fill_op_count++;
            }
            if (!cp_op_emitter_->validate_spill_fill()) {
                std::fprintf(stderr, "post_spill_fill: OpEmitter validation failed — "
                                     "dropping CP emitter records\n");
                cp_op_emitter_.reset();
                spill_fill_op_count = 0;
            }
        }
        if (!ok) {
            cp_plans_.clear();
            cp_op_emitter_.reset();
        }
    }

    if (spill_fill_op_count > 0) {
        // 反汇编 0x129ee15: call 0x129f0b0 (构建 5 字符串 vector, 静态局部函数)
        // 反汇编 0x129ee24: call 0x1299ea0 / 0x129ee33: call 0x1299ee0 (清理/析构)
        // REQNN: 无 SLC design pass 实现
        std::fprintf(stderr, "post_spill_fill: cp recorded %u spill/fill pair(s)\n",
                     spill_fill_op_count);
    }

    // 反汇编 0x129ee5a: cmpb $0x0,0x611e(%r15) — [this+0x611e] slc_allocator_debug flag
    // REQNN: 无 [this+0x611e] debug flag; 跳过 0x129efb3 的 level 3 告警日志
}

// ===== Run plugin rewrites (反汇编 @ 0x10d87e0, 3381B = 0xD35, plugin_rewrites.cc) =====
// Phase 1 of optimization: iterate registered plugin rewrite rules.
//
// 反汇编确认的控制流 (sym_master: run_plugin_rewrites @ 0x10d87e0, 3381B):
//
// 入口 (0x10d87e0-0x10d8813):
//   - 参数: rdi=this, esi=late_phase (bool)
//   - 0x10d87f1: mov %esi,%ebx — late_phase → ebx
//   - 0x10d87f3: mov %rdi,%r13 — this → r13
//   - 0x10d8809: call 0x10d8700 (静态局部函数, 获取插件注册表, 返回指针入 rax)
//   - 0x10d880e: cmpq $0x0,0x18(%rax) — [注册表+0x18]==0?
//   - 0x10d8813: je 0x10d934b — 注册表为空 → 跳到尾部返回
//
// early/late 字符串 (0x10d8831-0x10d884c):
//   - 0x10d8831: lea 0x6d60(%r13),%rbp — 链表尾哨兵 [this+0x6d60]
//   - 0x10d8838: lea 0x567933b "early",%rax
//   - 0x10d883f: lea 0x461d8ea "late",%rcx
//   - 0x10d8846: test %bl,%bl
//   - 0x10d8848: cmovne %rax,%rcx — bl!=0 → rcx="early"; bl==0 → "late"
//     (指令级事实: late_phase=true 时选 "early", false 时选 "late")
//
// 外层循环日志 (0x10d8860-0x10d8891):
//   - 0x10d8860: call GetLogPriorityLevel; 0x10d8865: cmp $0xb,%eax; jl 0x10d8890
//   - 0x10d888b: call qnndsp_log (level 11, "plugin rewrites %s outer loop")
//
// 主循环 (0x10d8891-0x10d92e0): 遍历 [this+0x6d58] 链表
//   - 0x10d8891: mov 0x6d58(%r13),%rax — 链表头
//   - 0x10d88a4: cmp %r14,%rax; 0x10d88a7: jne 0x10d88c7 — 遍历到哨兵
//   - 0x10d88c7: mov %rax,0x18(%rsp)
//   - 0x10d88cc: mov 0x28(%rax),%r15 — 读 op 描述符
//   - 0x10d88d0: mov 0x20(%r15),%rbx — 读 hash key
//   - 0x10d88d4-0x10d88f5: popcount (0x5555555555555555 @ 0x10d88e8) + hash 查表
//   - 0x10d8c3c: call clear_plugin_info(name) — 清除旧插件信息
//   - 0x10d8a5b: call r14 — 间接调用插件 rewrite 函数
//   - 0x10d8c6a: call qhpi_op_is_error — 检查插件报错
//     (报错 → 0x10d940f: call qhpi_op_error_description → throw "plugin rewrite error")
//   - 0x10d8d9b/0x10d8dab: call truegraph_n_outputs ×2 — 检查输入/输出 op 数量 (太少 → throw)
//   - 0x10d8e70: call op_def_posn — 查找 op 位置
//   - 0x10d8eae/0x10d8ec2: call OpRef::dereference ×2
//   - 0x10d8ecd: call conditionally_validate_single_quant
//   - 0x10d8f4e/0x10d8f6b: call supersede_op — 替换
//
// $MultiTemp 处理 (0x10d8ebe-0x10d927a):
//   - 0x10d90ef: supersede_op; 0x10d9132/0x10d9145: dereference ×2
//   - 0x10d9150: validate; 0x10d91ce: op_def_posn
//   - 0x10d923d/0x10d9250: dereference ×2; 0x10d925b: validate
//   - 0x10d927a: supersede_op
//
// DCE 清理 (0x10d92e0-0x10d9313):
//   - 0x10d92ea: cmp %rbp,%rax; 0x10d92ed: je 0x10d9377 — 链表遍历完 → 返回
//   - 0x10d92f7: cmpb $0x0,0x6ea4(%r13) — [this+0x6ea4] flag
//   - 0x10d92ff: je 0x10d9310 — flag==0 → collect_deletable_nodes
//   - 0x10d9306: call remove_dead_code(false) — flag!=0 → 调 DCE
//   - 0x10d9313: call collect_deletable_nodes — flag==0 分支
//
// REQNN adaptation: 无插件注册表 (对应 0x10d880e 检查 [rax+0x18]==0 → je 0x10d934b 直接返回).
// 函数结构保留 (early/late 参数, 链表遍历, DCE 条件) 但无实际 rewrite.
void GraphPrepare::run_plugin_rewrites(bool late_phase) {
    // 反汇编 0x10d8809: call 0x10d8700 (获取插件注册表, 静态局部函数)
    // 反汇编 0x10d880e: cmpq $0x0,0x18(%rax) — [注册表+0x18]==0?
    // 反汇编 0x10d8813: je 0x10d934b — 注册表为空 → 直接返回
    // REQNN: 无插件注册表 → 对应 je 0x10d934b 路径 (直接返回)

    // 反汇编 0x10d8838-0x10d8848: early/late 字符串选择
    // 0x10d8838: lea 0x567933b "early",%rax; 0x10d883f: lea 0x461d8ea "late",%rcx
    // 0x10d8846: test %bl,%bl; 0x10d8848: cmovne %rax,%rcx — bl!=0 → "early"
    (void)late_phase;  // REQNN: 无日志输出, 参数仅用于结构保留

    // 反汇编 0x10d8891-0x10d92e0: 主循环 (遍历 [this+0x6d58] 链表 + hash 查找插件)
    // REQNN: 无插件注册表 → 无 hash 查找 → 无 supersede_op

    // 反汇编 0x10d92f7: cmpb $0x0,0x6ea4(%r13) — [this+0x6ea4] flag
    // 反汇编 0x10d92ff: je 0x10d9310 — flag==0 → collect_deletable_nodes
    // 反汇编 0x10d9306: call remove_dead_code(false) — flag!=0 → 调 DCE
    // 反汇编 0x10d9313: call collect_deletable_nodes — flag==0 分支
    // REQNN: 无 [this+0x6ea4] flag; 不调 DCE (对应 flag==0 路径)
}

// ===== Multicast optimization (Phase 4.3) =====
// Source: grdep_mcast_optimizer.cc
// Builds McSend list from cross-NSP tensor consumers and runs McastOptimizer.
void GraphPrepare::run_mcast_optimization() {
    if (!graph_deps_ || opdef_map_.empty()) return;

    // Build McSend list: for each non-const op with consumers on different NSPs,
    // create one McSend describing the broadcast.
    // In the current host reimplementation (single NSP), there are no cross-NSP
    // sends, so this produces an empty list and optimize() is a no-op.
    std::vector<McSend> mcsends;
    uint32_t graph_multicast_count = 0;

    // For multi-NSP: walk opdescs, find ops whose output feeds consumers on
    // different NSPs. Each such producer becomes one McSend.
    // (Currently num_nsps_ = 1, so this loop produces nothing.)
    for (const auto& desc : graph_deps_->opdescs) {
        if (!desc.opdef_ptr) continue;
        if (desc.opdef_ptr->is_const()) continue;
        if (desc.opdef_ptr->consumers.empty()) continue;

        // Check if any consumer is on a different NSP than the producer.
        // (In single-NSP mode, all consumers are on NSP 0, so no mcast.)
        uint32_t producer_nsp = 0;  // single NSP
        bool has_remote_consumer = false;
        std::vector<uint32_t> remote_nsps;
        for (auto cid : desc.opdef_ptr->consumers) {
            uint32_t consumer_nsp = 0;  // single NSP
            if (consumer_nsp != producer_nsp) {
                has_remote_consumer = true;
                remote_nsps.push_back(consumer_nsp);
            }
        }
        if (!has_remote_consumer) continue;

        McSend send;
        send.tag = static_cast<uint32_t>(mcsends.size());
        send.sender_nsp = producer_nsp;
        send.num_mcids = 1;
        send.payload_size = desc.vtcm_requirement;
        send.mcids = {graph_multicast_count};
        send.receivers = remote_nsps;
        mcsends.push_back(std::move(send));
        ++graph_multicast_count;
    }

    if (mcsends.empty()) return;

    // Run optimizer with capacity limit
    McastOptimizer optimizer;
    optimizer.set_max_mcast_buffer_size(vtcm_size_);  // cap = VTCM budget
    auto result = optimizer.optimize(mcsends, graph_multicast_count);

    // Store result for future use (DMA emission in Phase 4.2)
    // (Currently a no-op in single-NSP mode; will be consumed when
    //  OpEmitter is wired into the scheduler.)
    (void)result;
}


// ===== SuperTile DP 合并 (反汇编确�? create_supertiles @ 0x1313ac0 [M36 修正; 旧址 0x13138d0 错], 5244B) =====
// 文件: supertile.cc, 日志�?267-406
// 流程: �?DPGroupGraph 每个 group �?HeuristicDP �?合并 op �?SuperTile
GraphStatus GraphPrepare::create_supertiles() {
    if (!dp_group_graph_) return GraphStatus::Success;

    // 构�?SuperTileSolver (HeuristicDP)
    supertile_solver_ = std::make_unique<SuperTileSolver>(
        *this, vtcm_size_, *dp_group_graph_);

    // DP 求解: dp[i][k] = min(dp[j][k-1] + cost(j+1..i))
    auto solution = supertile_solver_->solve();

    // 应用�? 创建 SuperTile 节点 (make_one_supertile)
    supertiles_ = std::make_unique<std::vector<SuperTile>>();
    supertile_solver_->apply(solution);

    // SuperTile 标志�?(反汇编确�? this+0x5578 �?0x100000001)
    // 简�? 记录 supertile 数量供序列化使用

    return GraphStatus::Success;
}

// make_one_supertile (反汇编确�? @ 0x1314d50, 591B)
// 1. 验证所�?op 无外�?consumer
// 2. 计算 VTCM 需�?// 3. �?VTCM �?�?split_history 递归切分
// 4. 分配连续 VTCM/DD, num_dma_roundtrips = 1
GraphStatus GraphPrepare::make_one_supertile(const std::vector<op_id_t>& op_ids,
                                              const std::vector<int>& split_history) {
    if (op_ids.empty()) return GraphStatus::Success;
    if (!supertiles_) supertiles_ = std::make_unique<std::vector<SuperTile>>();

    uint64_t total_vtcm = 0;
    for (auto oid : op_ids) {
        OpDef* od = get_op_at(oid);
        if (!od) continue;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od->output_def.rank && d < 5; ++d) {
            if (od->output_def.dims[d] > 0) sz *= od->output_def.dims[d];
        }
        uint64_t esize = od->output_def.element_size ? od->output_def.element_size : 4;
        total_vtcm += sz * esize;
    }

    // �?VTCM 预算 �?�?split_history 切分 (递归)
    if (total_vtcm > vtcm_size_ && op_ids.size() >= 2) {
        int split_point = -1;
        if (!split_history.empty()) split_point = split_history[0];
        if (split_point < 1 || split_point >= static_cast<int>(op_ids.size())) {
            split_point = static_cast<int>(op_ids.size()) / 2;
        }
        std::vector<int> rest_hist(split_history.begin() + 1, split_history.end());
        std::vector<op_id_t> left(op_ids.begin(), op_ids.begin() + split_point);
        std::vector<op_id_t> right(op_ids.begin() + split_point, op_ids.end());
        make_one_supertile(left, rest_hist);
        make_one_supertile(right, rest_hist);
        return GraphStatus::Success;
    }

    // 合并: 创建 SuperTile, num_dma_roundtrips = 1 (�?�?N 次降�?1 �?
    SuperTile st;
    st.op_ids = op_ids;
    st.vtcm_requirement = total_vtcm;
    st.num_dma_roundtrips = 1;
    supertiles_->push_back(std::move(st));
    return GraphStatus::Success;
}

} // namespace hnnx
