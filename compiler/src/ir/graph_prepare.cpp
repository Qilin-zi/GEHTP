#include <cstdio>
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/cost/cost_model.hpp"
#include "hnnx/ops/ops.hpp"
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

    for (auto& [id, opdef] : opdef_map_) {
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
                add_input(opdef.get(), ws);
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
            if (stride_id) add_input(opdef.get(), stride_id);
            if (pad_id) add_input(opdef.get(), pad_id);
            add_input(opdef.get(), get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));
            if (dilation_id) add_input(opdef.get(), dilation_id);
            add_input(opdef.get(), get_or_create("quant_marker", 1032, 1,1,1,1, 4, 4));

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
                add_input(opdef.get(), ws);
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
            if (stride_id) add_input(opdef.get(), stride_id);
            if (pad_id) add_input(opdef.get(), pad_id);
            if (dilation_id) add_input(opdef.get(), dilation_id);

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
                    add_input(opdef.get(), ws);
                }
            }
            add_input(opdef.get(), get_or_create("quant_marker", 1032, 1,1,1,1, 4, 4));

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
                add_input(opdef.get(), ws);
            }
            op_id_t qm = get_or_create("quant_marker", 1032, 1,1,1,1, 4, 4);
            add_input(opdef.get(), qm);
            add_input(opdef.get(), qm);

        } else if (nm == "ElementWiseNeuron") {
            add_input(opdef.get(), get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));
            op_id_t sa = get_or_create("sc_ewn_a", 562, 1,1,1,1, 4, 4);
            op_id_t sb = get_or_create("sc_ewn_b", 562, 1,1,1,1, 4, 4);
            for (int i = 0; i < 4; i++) add_input(opdef.get(), sa);
            add_input(opdef.get(), sb);

        } else if (nm == "LayerNorm") {
            add_input(opdef.get(), get_or_create("sc_" + nm, 562, 1,1,1,1, 4, 4));
            add_input(opdef.get(), get_or_create("or_Softmax", 50, 1,1,1,1, 4, 4));

        } else if (nm == "Softmax") {
            add_input(opdef.get(), get_or_create("sc_" + nm, 562, 1,1,1,1, 4, 4));
            add_input(opdef.get(), get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));

        } else if (nm == "Gather") {
            if (opdef->inputs.size() > 1) opdef->inputs.resize(1);
            op_id_t or_id = get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4);
            add_input(opdef.get(), or_id);
            add_input(opdef.get(), or_id);

        } else if (nm == "ElementWiseBinary") {
            add_input(opdef.get(), get_or_create("or_" + nm, 50, 1,1,1,1, 4, 4));

        } else if (nm == "Transpose") {
            for (op_id_t tpid : opdef->tensor_param_ids)
                add_input(opdef.get(), tpid);

        } else if (nm == "Reshape") {
            // [data] — no injected inputs
        }
        // Split nodes are eliminated later, no injection needed
    }

    // Rebuild consumers: injected const 节点在 op 之后创建, append_node 的
    // consumer 注册漏了它们。从 input 连接重建。
    rebuild_consumers();
}

// do_prepare1: initial graph building
// Source: graph_prepare.cc, 8518 bytes at 0xF66360
GraphStatus GraphPrepare::do_prepare1(HexagonNNEnv& env, VtcmCacheInstance& vtcm) {
    // 0. HtpPrepare auto-injection: per-op-type 输入构造
    // composeGraphs 产出的原始图里 op inputs 只含 data; HtpPrepare 注入
    // quant_marker/scale/output_rank 并把 tensor_param(perm 等)追加到 inputs。
    // 注入在 DCE 之前, 这样 DCE 不会误删被引用的 tensor_param const。
    inject_htp_prepare_inputs();

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

    // 1c. Eliminate Split nodes (HtpPrepare graph import folds Split)
    // Split outputs are replaced by direct references to Split's input producer.
    eliminate_split_nodes();

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

    // 4. Initial node ordering
    order_nodes(true);

    // 5. Allocate IO tensors
    // Source: allocate_io_tensors @ 0xF69750 (2687 bytes, ELF st_size)
    allocate_io_tensors();

    // 6. Const tracking setup
    const_tracking_setup();

    // 7. Mark construction complete
    construction_state_ = 2; // +0x45dc = 2 (optimization phase)

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
    run_optimize_passes(env);

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

    return GraphStatus::Success;
}
// Source: graph_prepare.cc, 2329 bytes at 0xF71F00
GraphStatus GraphPrepare::do_prepare2_late(std::vector<uint32_t>& runlist_tags) {
    // 1. Final optimization passes
    dead_code_removal_and_cse();

    // 2. Re-order nodes (ensures ordering_ is populated for execution)
    order_nodes(false);

    // 3. Runlist generation
    // Source: runlist_info.cc, run_order_to_alloc_info.cc
    // Builds runlist from ordered ops

    // 4. Final allocation
    // phys_alloc_in_runlist(ops)

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
GraphPrepare::ExecResult GraphPrepare::execute_host(const std::vector<float>& input) const {
    ExecResult ret{};

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
    tensor_map[input_id].resize(input_n > 0 ? input_n : 1);
    for (size_t i = 0; i < input_n && i < input.size(); ++i)
        tensor_map[input_id][i] = input[i];

    // Materialize const ops: copy their data from const_pool_ into tensor_map.
    // Const ops are not in ops_ (not registered), so we populate their outputs
    // here before the compute-op loop reads them.
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_const() || opdef->is_dead()) continue;
        if (opdef->const_data_size == 0) continue;
        size_t elem_n = opdef->const_data_size / sizeof(float);
        auto& buf = tensor_map[id];
        buf.resize(elem_n, 0.0f);
        if (opdef->const_data_offset > 0 && opdef->const_data_offset + opdef->const_data_size <= const_pool_.size()) {
            std::memcpy(buf.data(), const_pool_.data() + opdef->const_data_offset, opdef->const_data_size);
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
    for (const TypicalOp* top : topo) {
        op_id_t oid = top->op_id;
        if (oid == 0) continue;

        // Gather input buffers and their OutputDefs from predecessors.
        std::vector<const uint8_t*> in_bufs;
        std::vector<OutputDef> in_ods;
        for (const auto& conn : top->exec_inputs) {
            auto it = tensor_map.find(conn.src_id);
            in_bufs.push_back(it != tensor_map.end() && !it->second.empty()
                ? reinterpret_cast<const uint8_t*>(it->second.data()) : nullptr);
            in_ods.push_back(conn.src_out_def);
        }

        // Allocate output buffer.
        size_t out_n = 1;
        for (uint32_t i = 0; i < top->cached_out_def.rank && i < 5; ++i)
            out_n *= static_cast<size_t>(top->cached_out_def.dims[i]);
        if (out_n == 0) out_n = 1;

        auto& out_vec = tensor_map[oid];
        out_vec.resize(out_n, 0.0f);

        top->execute(in_bufs, reinterpret_cast<uint8_t*>(out_vec.data()), top->cached_out_def, in_ods);
    }

    // Read Output's predecessor from opdef_map_ (Output is not in ops_).
    auto* output_def = get_op_at(output_id);
    if (output_def && !output_def->inputs.empty()) {
        auto src_it = tensor_map.find(output_def->inputs[0].src_id);
        if (src_it != tensor_map.end()) {
            ret.output = src_it->second;
            ret.ok = true;
        }
    }

    return ret;
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

    // Fusion rules: applied after all phase fixpoints
    static const std::vector<FusionRule> fusion_rules = {
        {"Conv",    "Relu",    "ConvActivations"},
        {"Conv",    "Clamp",   "ConvActivations"},
        {"MatMul",  "Add",     "MatMul"},
        {"MatMul",  "Gelu",    "MatMul"},
        {"MatMul",  "Relu",    "MatMul"},
        {"Add",     "Relu",    "Add"},
        {"Add",     "Sigmoid", "Add"},
        {"Dense",   "Add",     "Dense"},
    };
    int fused = apply_fusion_rules(this, fusion_rules);
    if (fused > 0) {
        run_phase_fixpoint_internal();
    }
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

// Serialization
// Source: graph_prepare.cc:6751
bool GraphPrepare::serialize(uint8_t* buf, size_t buf_size, size_t& out_size) const {
    // Check: serialized_loaded_ (+0x45ac) != 0 -> error
    // "Cannot serialize a graph loaded from serialization" (graph_prepare.cc:6751)
    if (serialized_loaded_) return false;

    // Check: force_barrel_ (+0x6008) != 0 -> error
    // "Cannot apply serialize_force_barrel in serialize-to-memory" (graph_prepare.cc:6755)
    if (force_barrel_) return false;

    // If buf_size == 0, use internal buffer
    // If buf_size < 0x40 or not 4-byte aligned, error
    if (buf_size != 0 && (buf_size < 0x40 || (reinterpret_cast<uintptr_t>(buf) & 3) != 0)) {
        return false;
    }

    // 真实 .so 的 GraphPrepare ctor 必持 Allocator& (FancyAllocator ctor
    // @0xf3f2c0 ← fa::RuntimeAllocator(hnnx::Allocator::Mode, Graph&)), 此处不为
    // 空; 本仓近似 GraphPrepare 的 allocator_ 无供给路径 (恒 nullptr) —— 取
    // serializer.cpp 的兜底分配器 (空实现, 仅满足 ctor @0x12f14ab 的
    // dynamic_cast<FancyAllocator&>(*alloc))。M35/M36 接真供给后移除。
    hnnx::Allocator *ser_alloc = static_cast<hnnx::Allocator *>(allocator_);
    if (ser_alloc == nullptr) ser_alloc = hnnx::default_serializer_allocator();

    // Create Serializer (真实签名 @0x12f1320: (GraphPrepare const&, Allocator*, char*, size_t))
    Serializer ser(*const_cast<GraphPrepare*>(this), ser_alloc, reinterpret_cast<char*>(buf), buf_size);

    // Call do_serialize(ser)
    bool ok = do_serialize(ser);

    if (ok) {
        out_size = ser.current_position();
        // Align to 8 bytes: (out_size + 7) & ~7
        out_size = (out_size + 7) & ~size_t(7);
        // Fill remaining with 0x55
        if (out_size < buf_size) {
            std::memset(buf + out_size, 0x55, buf_size - out_size);
        }
    }
    return ok;
}

// do_serialize: full serialization
// Source: graph_prepare.cc, 3629 bytes at 0xF64DB0
bool GraphPrepare::do_serialize(Serializer& ser) const {
    // Check: serialized_loaded_ (+0x45ac) != 0 -> error
    // "Cannot serialize graph loaded from serialization!" (graph_prepare.cc:517)
    if (serialized_loaded_) {
        ser.set_error("Cannot serialize graph loaded from serialization!");
        return false;
    }

    // Check multi-NSP: graph_deps +0x7468, +0x160 != 0 and ser +0x10c == 0
    // "cannot use serialize method on multi-nsp graph" (graph_prepare.cc:526)

    // Write BinHeader (大端! 从真�?.bin 样本确认: 计数+偏移表格�?
    // Source: hexagon_nn_deserialize_graph 反汇�?+ test_minimal.serialized.bin
    // 无文件级 BEEF/FA00 magic; 头部�?num_graphs + num_records + 偏移
    {
        BinHeader hdr{};
        hdr.num_graphs = 1;
        hdr.num_records = static_cast<uint32_t>(const_extents_.size());
        hdr.reserved_08 = 0;
        hdr.flags = 1;
        hdr.offset_table = 0;
        hdr.block_size = const_pool_.size();
        hdr.base_address = 0;
        ser.write_tagged_record(0x4845, &hdr, sizeof(hdr));  // 'HE' header
    }

    // Write config records (tagged records):
    // Each record: write_tagged_record(tag, data, data_size)
    // Source: FUN_00f82370 (tagged_record_writer)

    // io_dma_bypass (0xEF4D) if +0x5c8c != 0
    {
        uint32_t dma_bypass = 0; // *(uint32_t*)(this + 0x5c8c)
        if (dma_bypass != 0 && dma_bypass <= 3) {
            ser.write_tagged_record(TAG_IO_DMA_BYPASS, &dma_bypass, 4);
        }
    }

    // spill_fill_instead (0x4453) if +0x58d8 != 0 and +0x62c0 >= 2
    {
        uint32_t spill_fill = 0; // *(uint32_t*)(this + 0x58d8)
        uint64_t multicast_count = 0; // *(uint64_t*)(this + 0x62c0)
        if (spill_fill != 0 && multicast_count >= 2) {
            uint32_t val = 0;
            ser.write_tagged_record(TAG_SPILL_FILL_INSTEAD, &val, 4);
        }
    }

    // extended_udma (0xD446) if +0x6144 != 0
    {
        uint8_t extended_udma = 0; // *(uint8_t*)(this + 0x6144)
        if (extended_udma != 0) {
            uint32_t val = 0;
            ser.write_tagged_record(TAG_EXTENDED_UDMA, &val, 4);
        }
    }

    // io_tensors_config (0xE347)
    {
        uint32_t io_config = 0; // *(uint32_t*)(this + 0x7470)
        ser.write_tagged_record(TAG_IO_TENSORS_CONFIG, &io_config, 4);
    }

    // extra_config (0xD352) - 0x2c bytes of extra config data
    {
        uint8_t extra_config[0x2c] = {};
        ser.write_tagged_record(TAG_EXTRA_CONFIG, extra_config, 0x2c);
    }

    // multicast_config (0xD349)
    {
        uint32_t mcast_count = 0; // calculated from op count
        ser.write_tagged_record(TAG_MULTICAST_CONFIG, &mcast_count, 4);
    }

    // serialize_io (prescan mode) - first pass to compute sizes
    uint64_t counter = 0;
    ser.set_mode(Serializer::Mode::Prescan);
    const_cast<GraphPrepare*>(this)->serialize_io(ser, counter, true);
    // Restore Write mode so config records below are actually emitted.
    // (Prescan only measures sizes; all subsequent writes must hit the buffer.)
    ser.set_mode(Serializer::Mode::Write);

    // make_plan_for_deser_by_segments if segment count > 0
    {
        uint32_t num_segments = 0; // *(uint32_t*)(this + 0x6028)
        if (num_segments > 0) {
            make_plan_for_deser_by_segments(&ser, num_segments, 0);
        }
    }

    // Write runlist counts:
    // 0x5248: vec runlist count (lVar22 = ops count)
    {
        uint32_t vec_count = 0;
        ser.write_tagged_record(TAG_NUM_SEGMENTS, &vec_count, 4);
    }
    // 0x524C: mtx runlist count (iVar24)
    {
        uint32_t mtx_count = 0;
        ser.write_tagged_record(0x524C, &mtx_count, 4);
    }
    // 0x5647: runlist segment desc (always 0x200000000)
    {
        uint64_t seg_desc = 0x200000000ULL;
        ser.write_tagged_record(TAG_RUNLIST_SEGMENT_DESC, &seg_desc, 8);
    }

    // Self-slicing config (0xC953) if +0x45ad != 0
    {
        uint8_t self_slice_loaded = 0; // *(uint8_t*)(this + 0x45ad)
        if (self_slice_loaded != 0) {
            uint32_t val = 1;
            ser.write_tagged_record(TAG_SELF_SLICING, &val, 4);
        }
    }
    // 18000 (0x4650): self-slicing mode if +0x61ed != 0
    {
        uint8_t slicing_mode = 0; // *(uint8_t*)(this + 0x61ed)
        if (slicing_mode != 0) {
            uint32_t val = slicing_mode;
            ser.write_tagged_record(18000, &val, 4);
        }
    }

    // Pass registry config (0xC955, 0xCF55) �?verified tag IDs
    // 0xC955: pass registry size (variable length)
    {
        // *(uint64_t*)(this + 0x5db0) to *(uint64_t*)(this + 0x5db8) range
        ser.write_tagged_record(TAG_PASS_REGISTRY, nullptr, 0);
    }
    // 0xCF55: another registry (variable length)
    {
        ser.write_tagged_record(TAG_ANOTHER_REGISTRY, nullptr, 0);
    }

    // Profiling config �?not using 0x5350 (that's IO counts in real .bin)
    // Profiling uses a different tag in real lib; omitted here (prof_count=0)
    {
        uint32_t prof_count = 0; // *(uint32_t*)(this + 0x6508)
        if (prof_count != 0) {
            uint64_t prof_data[2] = {prof_count, 0};
            ser.write_tagged_record(0x6508, prof_data, 8);
        }
    }

    // MC cacheable shared (0x5453) if +0x62d6 != 0
    {
        uint8_t mc_cacheable = 0; // *(uint8_t*)(this + 0x62d6)
        if (mc_cacheable != 0) {
            uint32_t mc_size = 0; // calculated from cacheable blocks
            ser.write_tagged_record(TAG_RUNLIST_AUX, &mc_size, 4);
        }
    }

    // IO counts (0x5350): [uint32 input_count][uint32 output_count] = 8 bytes
    // Source: do_serialize @ 0xf654b0 (verified tag)
    {
        uint32_t in_count = (input_node_id_ != 0) ? 1u : 0u;
        uint32_t out_count = (output_node_id_ != 0) ? 1u : 0u;
        uint32_t io_counts[2] = {in_count, out_count};
        ser.write_tagged_record(TAG_IO_COUNTS, io_counts, sizeof(io_counts));
    }

    // Graph header: counts so the deserializer can pre-allocate tables.
    // Record payload: [uint32 op_count][uint32 input_count][uint32 output_count]
    {
        uint32_t op_count = 0;
        for (auto& [id, opdef] : opdef_map_) {
            if (opdef && opdef->is_enabled() && !opdef->is_dead()) op_count++;
        }
        uint32_t in_count = (input_node_id_ != 0) ? 1u : 0u;
        uint32_t out_count = (output_node_id_ != 0) ? 1u : 0u;
        uint32_t hdr[3] = {op_count, in_count, out_count};
        ser.write_tagged_record(TAG_GRAPH_HEADER, hdr, sizeof(hdr));
    }

    // Separator 0xFA0000FA
    ser.write_uint32(0xFA0000FA);

    // Const pool: 先写 extent 描述符表 (TAG_CONST_EXTENT)，再写常量数据块�?    // �?ConstExtentDesc 结构: [op_id][offset][size][tensor_type][reserved] = 32 字节
    // Source: Deserializer::extract_const_extent_table / auxdata_read_const_extent_descriptor
    if (!const_extents_.empty()) {
        std::vector<uint8_t> ext_buf(const_extents_.size() * sizeof(ConstExtentDesc), 0);
        size_t off = 0;
        for (const auto& e : const_extents_) {
            ConstExtentDesc desc{};
            desc.op_id = e.op_id;
            desc.offset = e.offset;
            desc.size = e.size;
            desc.tensor_type = 0;  // PlainFloat 占位 (真实�?tensor 类型�?
            desc.reserved = 0;
            std::memcpy(ext_buf.data() + off, &desc, sizeof(desc));
            off += sizeof(desc);
        }
        ser.write_tagged_record(TAG_CONST_EXTENT, ext_buf.data(),
                                static_cast<int>(ext_buf.size()));
        // 常量数据�?(整块 4 字节对齐)
        if (!const_pool_.empty()) {
            ser.write_tagged_record(0xCF56, const_pool_.data(),
                                    static_cast<int>(const_pool_.size()));
        }
    }

    // Block table: VTCM block_id -> (pool_id, offset, size) 映射�?    // 每个 entry: [uint32 block_id][uint32 pool_id][uint64 offset][uint64 size]
    if (!block_table_.empty()) {
        std::vector<uint8_t> bt_buf(block_table_.size() * 24, 0);
        size_t off = 0;
        for (const auto& b : block_table_) {
            std::memcpy(bt_buf.data() + off, &b.block_id, 4); off += 4;
            std::memcpy(bt_buf.data() + off, &b.pool_id, 4); off += 4;
            std::memcpy(bt_buf.data() + off, &b.offset, 8); off += 8;
            std::memcpy(bt_buf.data() + off, &b.size, 8); off += 8;
        }
        ser.write_tagged_record(0x4254, bt_buf.data(),  // 'BT' block table
                                static_cast<int>(bt_buf.size()));
    }

    // Segment plan: record segment offsets for incremental loading.
    if (!segment_plans_.empty()) {
        std::vector<uint8_t> sp_buf(segment_plans_.size() * 24, 0);
        size_t off = 0;
        for (const auto& s : segment_plans_) {
            std::memcpy(sp_buf.data() + off, &s.segment_index, 4); off += 4;
            std::memcpy(sp_buf.data() + off, &s.op_count, 4); off += 4;
            std::memcpy(sp_buf.data() + off, &s.byte_offset, 8); off += 8;
            off += 8;  // reserved
        }
        ser.write_tagged_record(0x5347, sp_buf.data(),  // 'SG' segment plan
                                static_cast<int>(sp_buf.size()));
    }

    // serialize_io (write mode) - actual write
    // Mode already restored to Write after the prescan above.
    counter = 0;
    const_cast<GraphPrepare*>(this)->serialize_io(ser, counter, false);

    // Write runlist ops
    // For each op in runlist: serialize_internal
    // Source: serialize_oplist.cc, Op::serialize_internal
    // We serialize from the OpDef (graph structure); the execution Op is
    // regenerated by the op factory at load time. Iterate in topological
    // order when available, otherwise fall back to map order.
    // 确定性发射: 按 op_id 升序(unordered_map 迭代序随进程 ASLR 变化,
    // deserialize 后重建的 map 序也可能不同 → round-trip 字节确定性
    // 必须显式排序; 设备侧执行序由 wtop_emit/引擎的拓扑处理保证)。
    // 发射序 = Scheduler 计划序(ST-Cut, do_prepare2 计算); 未调度(空)则
    // 回退 op_id 升序(确定性)。计划序随后以 TAG_PLAN_ORDER 记录写出,
    // deserialize 回读 → re-serialize 字节确定性。
    std::vector<op_id_t> emit_order = plan_order_;
    if (emit_order.empty()) {
        for (auto& [id, opdef] : opdef_map_) {
            if (opdef && opdef->is_enabled() && !opdef->is_dead())
                emit_order.push_back(id);
        }
        std::sort(emit_order.begin(), emit_order.end());
    }
    for (op_id_t id : emit_order) {
        auto it = opdef_map_.find(id);
        if (it == opdef_map_.end() || !it->second) continue;
        if (!it->second->is_enabled() || it->second->is_dead()) continue;
        serialize_opdef(ser, *it->second);
    }
    // TAG_PLAN_ORDER: [u32 count][u32 ids...](count=0 也写出, 显式"未调度")
    {
        std::vector<uint8_t> pl(4 + emit_order.size() * 4, 0);
        uint32_t cnt = static_cast<uint32_t>(emit_order.size());
        std::memcpy(pl.data(), &cnt, 4);
        for (size_t i = 0; i < emit_order.size(); i++) {
            uint32_t v = static_cast<uint32_t>(emit_order[i]);
            std::memcpy(pl.data() + 4 + i * 4, &v, 4);
        }
        ser.write_tagged_record(TAG_PLAN_ORDER, pl.data(), static_cast<int>(pl.size()));
    }

    // Spill/fill(阶段7): 溢出张量 → 0x4453 配置记录 + 每张量 0x5346 DMA 记录。
    // 收集来源: ①allocator spilled(vtcm_allocations_)②tcm_migration 标记
    // (flags2 & SPILL_TO_DDR=0x40, 未在①中)。DDR 池偏移按 op_id 升序确定性
    // 分配(128B 对齐, 起 0x1000)。反序列化回读 spill_fill_recs_, 优先复用
    // (round-trip 确定性)。
    {
        std::vector<SpillFillRec> recs;
        if (!spill_fill_recs_.empty()) {
            recs = spill_fill_recs_;
        } else {
            auto tensor_size = [&](op_id_t id) -> uint64_t {
                const OpDef* od = get_op_at(id);
                if (!od) return 0;
                uint64_t sz = 1;
                for (uint32_t d = 0; d < od->output_def.rank && d < 5; ++d)
                    if (od->output_def.dims[d] > 0) sz *= od->output_def.dims[d];
                return sz * (od->output_def.element_size ? od->output_def.element_size : 4);
            };
            for (const auto& [id, a] : vtcm_allocations_) {
                if (a.spilled) recs.push_back({id, a.block_id, 0, tensor_size(id)});
            }
            constexpr uint32_t SPILL_TO_DDR = 0x40;
            for (auto& [id, opdef] : opdef_map_) {
                if (!opdef || !(opdef->flags2 & SPILL_TO_DDR)) continue;
                bool dup = false;
                for (const auto& r : recs) dup |= (r.op_id == id);
                if (!dup) recs.push_back({id, 0, 0, tensor_size(id)});
            }
            std::sort(recs.begin(), recs.end(),
                      [](const SpillFillRec& a, const SpillFillRec& b) { return a.op_id < b.op_id; });
            uint64_t cur = 0x1000;
            for (auto& r : recs) {
                r.ddr_offset = cur;
                cur += (r.size + 127) & ~uint64_t(127);
            }
        }
        if (!recs.empty()) {
            uint32_t cnt = static_cast<uint32_t>(recs.size());
            ser.write_tagged_record(TAG_SPILL_FILL_INSTEAD, &cnt, 4);
            // 28B 固定布局(显式逐字段写入, 避免 struct padding 歧义;
            // 设备侧同构解析): [u64 op_id][u32 block_id][u64 ddr_offset][u64 size]
            for (const auto& r : recs) {
                uint8_t payload[28];
                size_t o = 0;
                auto w64 = [&](uint64_t v) { std::memcpy(payload + o, &v, 8); o += 8; };
                auto w32 = [&](uint32_t v) { std::memcpy(payload + o, &v, 4); o += 4; };
                w64(r.op_id); w32(r.block_id); w64(r.ddr_offset); w64(r.size);
                ser.write_tagged_record(TAG_SPILL_FILL_OP, payload, 28);
            }
        }
    }

    // End separator 0xBEEFF00D
    ser.write_uint32(0xBEEFF00D);

    // Log: "RUNLIST: total %d ops, %d in runlist, %d in vec runlist, %d in mtx runlist, %d in elt runlist"
    // (graph_prepare.cc:736)

    // Allocator setup -> setup_heap_info
    // adjust_heap_stats
    // check_total_allocation (memory limit check)

    return true;
}

// ---------------------------------------------------------------------------
// per-op 参数 schema 注册(通用性落地点): 新算子 = 注册 extractor 函数。
// serialize_opdef 在固定结构字段后尾随 [u32 extra_len][extra bytes];
// deserialize 按 length 回读并原样保存(serialized_extra), re-serialize 原样
// 重发 → round-trip 字节确定性。旧流(无尾随字段)天然兼容(remaining < 4)。
// extractor 从图中推导参数(权重 const 形状 + 注入的 stride/pad/dilation
// const 数据), 不依赖任何模型特判。
// ---------------------------------------------------------------------------
namespace {
using ExtraInfoFn = std::vector<uint8_t>(*)(const GraphPrepare& gp, const OpDef& opdef);

// ConvExtraInfo 固定二进制布局(全小端 u32/u64, 设备侧同构解析):
//   [sh][sw][ph_begin][ph_end][pw_begin][pw_end][dh][dw][group][kh][kw]
//   [u64 weight_src][u64 bias_src]
struct ConvExtraInfo {
    uint32_t sh = 1, sw = 1;
    uint32_t ph_begin = 0, ph_end = 0, pw_begin = 0, pw_end = 0;
    uint32_t dh = 1, dw = 1, group = 1, kh = 1, kw = 1;
    uint64_t weight_src = 0, bias_src = 0;
};

const OpDef* find_param_const(const GraphPrepare& gp, const OpDef& opdef, const char* key) {
    for (const auto& c : opdef.inputs) {
        const OpDef* s = gp.get_op_at(c.src_id);
        if (!s || !s->name_tag) continue;
        const char* n = s->name_tag->name();
        if (n && std::strstr(n, key)) return s;
    }
    return nullptr;
}

// 从注入 const(在 const_pool_ 中)读 u32 数组
std::vector<uint32_t> read_param_u32(const GraphPrepare& gp, const OpDef* p, size_t max_n) {
    std::vector<uint32_t> v;
    if (!p || p->const_data_size < 4) return v;
    size_t n = std::min(max_n, p->const_data_size / 4);
    const auto& pool = gp.const_pool();
    if (p->const_data_offset == 0 || p->const_data_offset + p->const_data_size > pool.size())
        return v;
    v.resize(n);
    std::memcpy(v.data(), pool.data() + p->const_data_offset, n * 4);
    return v;
}

std::vector<uint8_t> extract_conv_extra(const GraphPrepare& gp, const OpDef& opdef) {
    ConvExtraInfo e;
    // kh/kw 来自权重 const 的 output_def (inputs[1] = W, dims [Kh,Kw,Cin,Cout])
    if (opdef.inputs.size() > 1) {
        e.weight_src = opdef.inputs[1].src_id;
        const OpDef* w = gp.get_op_at(opdef.inputs[1].src_id);
        if (w && w->output_def.rank >= 4) {
            e.kh = static_cast<uint32_t>(w->output_def.dims[0]);
            e.kw = static_cast<uint32_t>(w->output_def.dims[1]);
        }
    }
    if (opdef.inputs.size() > 2) e.bias_src = opdef.inputs[2].src_id;

    if (const OpDef* st = find_param_const(gp, opdef, "stride")) {
        auto v = read_param_u32(gp, st, 2);
        if (v.size() == 2) { e.sh = v[0]; e.sw = v[1]; }
    }
    if (const OpDef* pd = find_param_const(gp, opdef, "pad_amount")) {
        auto v = read_param_u32(gp, pd, 4);
        if (v.size() == 4) { e.ph_begin = v[0]; e.ph_end = v[1]; e.pw_begin = v[2]; e.pw_end = v[3]; }
    }
    if (const OpDef* dl = find_param_const(gp, opdef, "dilation")) {
        auto v = read_param_u32(gp, dl, 2);
        if (v.size() == 2) { e.dh = v[0]; e.dw = v[1]; }
    }

    std::vector<uint8_t> out(sizeof(ConvExtraInfo));
    std::memcpy(out.data(), &e, sizeof(ConvExtraInfo));

    // Tiling 段(阶段6, 通用): [u32 tile_h][u32 tile_w][u32 co_per_tile]
    //                          [u32 num_tiles][ConvTileDesc × num_tiles]
    // 配置来源: gp.tiling_config()(conv_height/width/channel_tiling = 切分数)
    {
        uint32_t in_h = 0, in_w = 0, cin = 0;
        if (!opdef.inputs.empty()) {
            const OpDef* act = gp.get_op_at(opdef.inputs[0].src_id);
            if (act && act->output_def.rank >= 4) {
                in_h = static_cast<uint32_t>(act->output_def.dims[1]);
                in_w = static_cast<uint32_t>(act->output_def.dims[2]);
                cin = static_cast<uint32_t>(act->output_def.dims[3]);
            }
        }
        uint32_t out_h = (opdef.output_def.rank >= 4)
            ? static_cast<uint32_t>(opdef.output_def.dims[1]) : 0;
        uint32_t out_w = (opdef.output_def.rank >= 4)
            ? static_cast<uint32_t>(opdef.output_def.dims[2]) : 0;
        uint32_t cout = (opdef.output_def.rank >= 4)
            ? static_cast<uint32_t>(opdef.output_def.dims[3]) : 0;

        const TilingConfig& cfg = gp.tiling_config();
        uint32_t ht = cfg.conv_height_tiling > 0 ? cfg.conv_height_tiling : 1;
        uint32_t wt = cfg.conv_width_tiling > 0 ? cfg.conv_width_tiling : 1;
        uint32_t ct = cfg.conv_channel_tiling > 0 ? cfg.conv_channel_tiling : 1;
        uint32_t tile_h = out_h > 0 ? (out_h + ht - 1) / ht : 0;
        uint32_t tile_w = out_w > 0 ? (out_w + wt - 1) / wt : 0;
        uint32_t co_per_tile = cout > 0 ? (cout + ct - 1) / ct : 0;

        std::vector<ConvTileDesc> tiles = compute_conv_tiles(
            in_h, in_w, cin, out_h, out_w, cout,
            e.kh, e.kw, e.sh, e.sw, e.ph_begin, e.pw_begin,
            tile_h, tile_w, co_per_tile);

        std::vector<uint8_t> sec(16 + tiles.size() * sizeof(ConvTileDesc));
        uint32_t hdr[4] = {tile_h, tile_w, co_per_tile,
                           static_cast<uint32_t>(tiles.size())};
        std::memcpy(sec.data(), hdr, 16);
        std::memcpy(sec.data() + 16, tiles.data(), tiles.size() * sizeof(ConvTileDesc));
        out.insert(out.end(), sec.begin(), sec.end());
    }
    return out;
}

std::map<std::string, ExtraInfoFn>& extra_info_registry() {
    static std::map<std::string, ExtraInfoFn> reg = [] {
        std::map<std::string, ExtraInfoFn> m;
        m["Conv2d"] = extract_conv_extra;
        return m;
    }();
    return reg;
}
} // namespace

// serialize_opdef: write a single op's structural record (TAG_OP_RECORD).
//   [uint32 name_len][char[] name, 4-aligned][uint64 op_id][uint16 flags]
//   [uint16 reserved][uint32 num_inputs]
//   per input: [uint64 src_id][uint32 out_idx][uint32 pad]
//   [OutputDef output_def]
// The execution Op is regenerated by the op factory at load time, so only
// the graph-structure data (type name, id, connections, output shape, flags)
// needs to persist.
//
// [反汇编证�?libQnnHtp.so v2.48] 真实 serialize_op @0x12ec630 写入格式:
//   [0x1303EE{XX}] (4B 标记, XX=记录ID, 硬编�?0x1303ee71 起始)
//   [counter:u32]  (= tensor id, 来自 net.json 张量 id)
//   [type:u32<<24] (0x01=compute, 0x02=memory, 0x03=sync, 0x04=DMA, 0x0b=HMX)
//                  (type = (extra_info[0x18]>>6 & 0xf) << 24, HMX 修正)
//   [block_ref:u32] (0x10{idx}, VTCM 块引�? extra_info[0x24]&0xFFFFFF|(3<<24))
//   [tensor_ids...] (变长, 输入/输出张量 id 列表)
// 注意: SET/WAIT (make_dma_checkpoint_op @0xd958d0) 共享 serialize_internal @0xd969a0
//       �?context binary 字节不可区分 SET vs WAIT
// 详见 reference/docs/schedule_analysis.md
void GraphPrepare::serialize_opdef(Serializer& ser, const OpDef& opdef) const {
    const char* name = opdef.name_tag ? opdef.name_tag->name() : "";
    uint32_t name_len = static_cast<uint32_t>(std::strlen(name));
    uint32_t name_padded = (name_len + 3) & ~uint32_t(3);

    // 尾随 per-op 参数 schema(通用): 优先复用反序列化回读的字节(round-trip
    // 确定性), 否则按注册表 extractor 现算。
    std::vector<uint8_t> extra;
    if (!opdef.serialized_extra.empty()) {
        extra = opdef.serialized_extra;
    } else {
        auto it = extra_info_registry().find(name);
        if (it != extra_info_registry().end()) extra = it->second(*this, opdef);
    }

    // Compute total payload size for the tagged record header word count.
    size_t payload =
        4 + name_padded +            // name_len + name
        8 + 2 + 2 + 4 +              // op_id, flags, reserved, num_inputs
        opdef.inputs.size() * 16 +   // (src_id, out_idx, pad) per input
        sizeof(OutputDef) +           // output_def
        8 + 8 +                       // const_data_offset, const_data_size
        4 + extra.size();             // 尾随 [u32 extra_len][extra bytes]

    // Build the payload in a local buffer, then emit as one tagged record.
    std::vector<uint8_t> buf(payload, 0);
    size_t off = 0;
    auto w32 = [&](uint32_t v) { std::memcpy(buf.data() + off, &v, 4); off += 4; };
    auto w16 = [&](uint16_t v) { std::memcpy(buf.data() + off, &v, 2); off += 2; };
    auto w64 = [&](uint64_t v) { std::memcpy(buf.data() + off, &v, 8); off += 8; };

    w32(name_len);
    std::memcpy(buf.data() + off, name, name_len);
    off += name_padded;
    w64(opdef.op_id);
    w16(opdef.flags);
    w16(0);  // reserved
    w32(static_cast<uint32_t>(opdef.inputs.size()));
    for (const auto& conn : opdef.inputs) {
        w64(conn.src_id);
        w32(conn.out_idx);
        w32(0);  // pad
    }
    std::memcpy(buf.data() + off, &opdef.output_def, sizeof(OutputDef));
    off += sizeof(OutputDef);
    w64(opdef.const_data_offset);
    w64(opdef.const_data_size);
    w32(static_cast<uint32_t>(extra.size()));
    if (!extra.empty()) {
        std::memcpy(buf.data() + off, extra.data(), extra.size());
        off += extra.size();
    }

    ser.write_tagged_record(TAG_OP_RECORD, buf.data(), static_cast<int>(off));
}

// Cursor for reading the tagged-record stream back from a flat buffer.
struct BinReader {
    const uint8_t* p;
    const uint8_t* end;
    explicit BinReader(const uint8_t* base, size_t n) : p(base), end(base + n) {}
    bool eof() const { return p >= end; }
    size_t remaining() const { return end - p; }
    uint32_t r32() { uint32_t v = 0; std::memcpy(&v, p, 4); p += 4; return v; }
    uint64_t r64() { uint64_t v = 0; std::memcpy(&v, p, 8); p += 8; return v; }
    uint16_t r16() { uint16_t v = 0; std::memcpy(&v, p, 2); p += 2; return v; }
    void read(void* dst, size_t n) { if (dst) std::memcpy(dst, p, n); p += n; }
};

// deserialize: rebuild the graph from a buffer produced by serialize().
// Round-trips the structural records written by do_serialize/serialize_opdef.
bool GraphPrepare::deserialize(const uint8_t* buf, size_t buf_size) {
    if (!buf || buf_size < 4) return false;

    // Reset to a fresh construction-phase graph.
    opdef_map_.clear();
    ordering_.clear();
    ops_.clear();
    const_pool_.clear();
    const_extents_.clear();
    block_table_.clear();
    segment_plans_.clear();
    input_node_id_ = 0;
    output_node_id_ = 0;
    next_op_id_ = 1;
    construction_state_ = 1;  // allow direct opdef insertion
    serialized_loaded_ = true;

    BinReader r(buf, buf_size);
    uint32_t expected_ops = 0;

    while (!r.eof()) {
        uint32_t word0 = r.r32();

        // Single-word separators / end marker are not tagged records.
        if (word0 == SEPARATOR_NORMAL || word0 == SEPARATOR_AUX) {
            continue;  // segment / aux separator
        }
        if (word0 == SEPARATOR_END) {
            break;  // 0xBEEFF00D end-of-stream
        }

        // Otherwise word0 is an encoded tag: read word_count + third field.
        uint32_t word_count = r.r32();
        uint32_t third = r.r32();  // third field (== word_count for our writer)
        (void)third;
        size_t data_size = static_cast<size_t>(word_count) * 4;
        if (data_size > r.remaining()) return false;
        const uint8_t* rec = r.p;
        r.p += data_size;

        // Decode the tag: inverse of ((tag & 0xFFFF) | (tag << 16)) ^ 0xFFFF.
        uint32_t decoded = word0 ^ 0xFFFF;
        uint32_t tag = decoded & 0xFFFF;

        if (tag == 0x4845) {  // 'HE' BinHeader (大端计数+偏移�?
            BinHeader hdr;
            std::memcpy(&hdr, rec, sizeof(BinHeader));
            // 真实 .bin 无文件级 magic; �?num_graphs 合理性校�?
    if (hdr.num_graphs == 0 || hdr.num_graphs > 100) {
                return false;
            }
            continue;
        }
        if (tag == TAG_GRAPH_HEADER) {
            expected_ops = rec[0] | (rec[1] << 8) | (rec[2] << 16) | (rec[3] << 24);
            (void)expected_ops;
            continue;
        }
        if (tag == TAG_SPILL_FILL_OP) {
            BinReader rr(rec, data_size);
            if (rr.remaining() >= 28) {
                SpillFillRec r{};
                r.op_id = rr.r64();
                r.block_id = rr.r32();
                r.ddr_offset = rr.r64();
                r.size = rr.r64();
                spill_fill_recs_.push_back(r);
            }
            continue;
        }
        if (tag == TAG_SPILL_FILL_INSTEAD && data_size == 4) {
            // 0x4453 现在携带 spill 计数(阶段7); 旧语义(仅 flag+multicast)已被
            // 此计数形式取代 —— 记录本身无副作用, 计数由 0x5346 记录实际决定。
            continue;
        }
        if (tag == TAG_PLAN_ORDER) {
            BinReader rr(rec, data_size);
            if (rr.remaining() >= 4) {
                uint32_t cnt = rr.r32();
                plan_order_.clear();
                for (uint32_t i = 0; i < cnt && rr.remaining() >= 4; i++)
                    plan_order_.push_back(static_cast<op_id_t>(rr.r32()));
            }
            continue;
        }
        if (tag == TAG_CONST_EXTENT) {
            // extent �? 每条 ConstExtentDesc = 32 字节
            // [op_id 8B][offset 8B][size 8B][tensor_type 4B][reserved 4B]
            size_t n = data_size / sizeof(ConstExtentDesc);
            const_extents_.clear();
            const uint8_t* pp = rec;
            for (size_t i = 0; i < n; ++i) {
                ConstExtentDesc desc;
                std::memcpy(&desc, pp, sizeof(desc));
                pp += sizeof(desc);
                const_extents_.push_back({desc.op_id, desc.offset, desc.size});
            }
            continue;
        }
        if (tag == 0xCF56) {
            // 常量数据�? 直接拷回 const_pool_
            const_pool_.assign(rec, rec + data_size);
            continue;
        }
        if (tag == 0x4254) {
            // block table: �?24 字节 [block_id][pool_id][offset][size]
            size_t n = data_size / 24;
            block_table_.clear();
            const uint8_t* pp = rec;
            for (size_t i = 0; i < n; ++i) {
                BlockEntry b{};
                std::memcpy(&b.block_id, pp, 4); pp += 4;
                std::memcpy(&b.pool_id, pp, 4); pp += 4;
                std::memcpy(&b.offset, pp, 8); pp += 8;
                std::memcpy(&b.size, pp, 8); pp += 8;
                block_table_.push_back(b);
            }
            continue;
        }
        if (tag == 0x5347) {
            // segment plan: �?24 字节 [seg_idx][op_count][byte_offset][reserved8]
            size_t n = data_size / 24;
            segment_plans_.clear();
            const uint8_t* pp = rec;
            for (size_t i = 0; i < n; ++i) {
                SegmentPlan s{};
                std::memcpy(&s.segment_index, pp, 4); pp += 4;
                std::memcpy(&s.op_count, pp, 4); pp += 4;
                std::memcpy(&s.byte_offset, pp, 8); pp += 8;
                pp += 8;  // reserved
                segment_plans_.push_back(s);
            }
            continue;
        }
        if (tag == TAG_IO_TENSOR_DESC) {
            // [uint32 kind][uint32 pad][uint64 op_id][OutputDef]
            BinReader rr(rec, data_size);
            uint32_t kind = rr.r32();
            (void)rr.r32();  // pad
            uint64_t op_id = rr.r64();
            // Register the Input/Output node by id; the producer/consumer
            // links are re-established when reading the op records below.
            auto opdef = std::make_unique<OpDef>();
            opdef->flags = OP_ENABLED;
            opdef->graph = this;
            opdef->op_id = op_id;
            opdef->name_tag = string_tag_t::map_str(kind == 0 ? "Input" : "Output");
            std::memcpy(&opdef->output_def, rr.p, sizeof(OutputDef));
            if (kind == 0) input_node_id_ = op_id;
            else output_node_id_ = op_id;
            opdef_map_[op_id] = std::move(opdef);
            continue;
        }
        if (tag == TAG_OP_RECORD) {
            BinReader rr(rec, data_size);
            uint32_t name_len = rr.r32();
            uint32_t name_padded = (name_len + 3) & ~uint32_t(3);
            std::string name(reinterpret_cast<const char*>(rr.p), name_len);
            rr.p += name_padded;
            uint64_t op_id = rr.r64();
            uint16_t flags = rr.r16();
            (void)rr.r16();  // reserved
            uint32_t num_inputs = rr.r32();

            auto opdef = std::make_unique<OpDef>();
            opdef->flags = flags;
            opdef->graph = this;
            opdef->op_id = op_id;
            opdef->name_tag = string_tag_t::map_str(name.c_str());

            opdef->inputs.reserve(num_inputs);
            for (uint32_t i = 0; i < num_inputs; ++i) {
                uint64_t src_id = rr.r64();
                uint32_t out_idx = rr.r32();
                (void)rr.r32();  // pad
                opdef->inputs.push_back({src_id, out_idx});
            }
            std::memcpy(&opdef->output_def, rr.p, sizeof(OutputDef));
            rr.p += sizeof(OutputDef);
            opdef->const_data_offset = rr.r64();
            opdef->const_data_size = rr.r64();
            // 尾随 per-op 参数 schema(向后兼容: 旧流无此字段 → remaining < 4)
            if (rr.remaining() >= 4) {
                uint32_t elen = rr.r32();
                if (elen > 0 && rr.remaining() >= elen) {
                    opdef->serialized_extra.resize(elen);
                    rr.read(opdef->serialized_extra.data(), elen);
                }
            }

            if (name == "Input") input_node_id_ = op_id;
            else if (name == "Output") output_node_id_ = op_id;

            opdef_map_[op_id] = std::move(opdef);
            if (op_id >= next_op_id_) next_op_id_ = op_id + 1;
            continue;
        }
        // Unknown config record: already skipped via r.p advance above.
    }

    // Rebuild consumer lists from the restored input connections.
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef) continue;
        for (const auto& conn : opdef->inputs) {
            auto src_it = opdef_map_.find(conn.src_id);
            if (src_it != opdef_map_.end() && src_it->second) {
                src_it->second->consumers.push_back(id);
            }
        }
    }

    construction_state_ = 2;  // graph now in "loaded" state
    // The reconstructed graph is a clean, self-contained structure (not tied
    // to any external loaded buffer), so allow it to be re-serialized. This
    // lets the round-trip test verify deterministic layout.
    serialized_loaded_ = false;

    // Rebuild topological ordering so ops_ are in valid execution order.
    // Without this, iterating opdef_map_ (unordered) may produce wrong order
    // (e.g. Add before MatMul).
    order_nodes(false);

    // Build ops_ from the restored opdef_map_ so execute_host can run
    // without a separate prepare() call.  Use ordering_ for correct
    // topological execution order.
    ops_.clear();
    if (!ordering_.empty()) {
        for (op_id_t id : ordering_) {
            auto it = opdef_map_.find(id);
            if (it == opdef_map_.end() || !it->second) continue;
            if (!it->second->is_enabled() || it->second->is_dead()) continue;
            OpIoPtrs io{};
            io.graph_prepare = this;
            io.opdef_ptr = it->second.get();
            auto op = op_factory_generate(io, id);
            if (op) { ops_.push_back(std::move(op)); }
        }
    } else {
        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
            OpIoPtrs io{};
            io.graph_prepare = this;
            io.opdef_ptr = opdef.get();
            auto op = op_factory_generate(io, id);
            if (op) { ops_.push_back(std::move(op)); }
        }
    }

    return true;
}

void GraphPrepare::serialize_io(Serializer& ser, uint64_t& counter, bool is_prescan) const {
    (void)is_prescan;
    // Input node: its single output_def describes the input tensor fed at runtime
    if (input_node_id_ != 0) {
        auto it = opdef_map_.find(input_node_id_);
        if (it != opdef_map_.end() && it->second) {
            const OpDef& iod = *it->second;
            // payload: [uint32 kind=0(input)][uint64 op_id][OutputDef]
            struct IoDesc {
                uint32_t kind;     // 0 = input, 1 = output
                uint32_t pad;
                uint64_t op_id;
                OutputDef od;
            } desc{};
            desc.kind = 0;
            desc.op_id = iod.op_id;
            desc.od = iod.output_def;
            ser.write_tagged_record(TAG_IO_TENSOR_DESC, &desc, sizeof(desc));
            counter++;
        }
    }
    // Output node: its single input references the tensor produced for the host
    if (output_node_id_ != 0) {
        auto it = opdef_map_.find(output_node_id_);
        if (it != opdef_map_.end() && it->second) {
            const OpDef& iod = *it->second;
            // Resolve the producer op to grab the real output_def
            OutputDef od{};
            if (!iod.inputs.empty()) {
                auto src = opdef_map_.find(iod.inputs[0].src_id);
                if (src != opdef_map_.end() && src->second) {
                    od = src->second->output_def;
                }
            }
            struct IoDesc {
                uint32_t kind;     // 1 = output
                uint32_t pad;
                uint64_t op_id;
                OutputDef od;
            } desc{};
            desc.kind = 1;
            desc.op_id = iod.op_id;
            desc.od = od;
            ser.write_tagged_record(TAG_IO_TENSOR_DESC, &desc, sizeof(desc));
            counter++;
        }
    }
}

// Op management - REAL implementation
// Source: do_append_node @ 0xF75F30 (4003 bytes)
op_id_t GraphPrepare::append_node(const std::string& name, uint32_t node_type,
                                    const InputDef* inputs, size_t num_inputs,
                                    const OutputDef* outputs, size_t num_outputs,
                                    const uint8_t* ops_data) {
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
        if (input_node_id_ != 0) return 0; // second Input
        input_node_id_ = new_id;
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

    // 10b. 保存 op_data 参数 blob �?op 构造时解析
    if (ops_data && num_outputs > 0) {
        // ops_data 长度未知; 真实库按 op 类型读固定字段。这里存�?256 字节
        // 作为参数 blob 上限 (足够 stride/padding/dilation/axis 等常见参�?�?
    size_t blob_len = 256;
        opdef->op_data.assign(ops_data, ops_data + blob_len);
    } else if (ops_data) {
        opdef->op_data.assign(ops_data, ops_data + 256);
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
op_id_t GraphPrepare::append_const_node(uint32_t node_type, const OutputDef& od,
                                          const uint8_t* data, size_t data_len) {
    // Validate construction state
    if (construction_state_ != 1) return 0;

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

// remove_dead_code: REAL implementation
// Source: graph_prepare.cc, 402 bytes at 0xF68D90
int GraphPrepare::remove_dead_code(bool) {
    // Source: graph_prepare.cc, 402 bytes at 0xF68D90
    // Algorithm (from decompilation):
    // 1. Iterate all ops, find those with no consumers (dead)
    // 2. Mark them dead (flags |= 3)
    // 3. Collect and delete
    // 4. Repeat for newly-dead ops (cascade)
    // +0x6ea4 flag tracks if any dead code was found

    int deleted_count = 0;
    bool changed = true;

    // Iterate to handle cascading dead code
    while (changed) {
        changed = false;
        std::vector<op_id_t> to_delete;

        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled()) continue;
            if (opdef->is_dead()) continue;
            if (id == output_node_id_) continue; // Don't delete Output node
            if (id == input_node_id_) continue;  // Don't delete Input node

            // Check if this op has any live consumers
            bool has_live_consumer = false;
            for (op_id_t consumer_id : opdef->consumers) {
                auto it = opdef_map_.find(consumer_id);
                if (it != opdef_map_.end() && it->second &&
                    it->second->is_enabled() && !it->second->is_dead()) {
                    has_live_consumer = true;
                    break;
                }
            }

            if (!has_live_consumer) {
                // Mark as dead: flags |= 3 (OP_ENABLED | OP_DEAD)
                opdef->flags |= OP_DEAD;
                to_delete.push_back(id);
                changed = true;
            }
        }

        // Delete dead ops
        for (op_id_t id : to_delete) {
            // Remove this op from its sources' consumer lists
            auto it = opdef_map_.find(id);
            if (it != opdef_map_.end() && it->second) {
                for (const auto& conn : it->second->inputs) {
                    auto src_it = opdef_map_.find(conn.src_id);
                    if (src_it != opdef_map_.end() && src_it->second) {
                        auto& consumers = src_it->second->consumers;
                        consumers.erase(
                            std::remove(consumers.begin(), consumers.end(), id),
                            consumers.end());
                    }
                }
            }
            opdef_map_.erase(id);
            deleted_count++;
        }
    }

    if (deleted_count > 0) {
        graph_dirty_ = true;
    }

    return deleted_count;
}

// order_nodes: topological sort via iterative DFS (stupid_fast_topo_sort)
// Source: stupid_fast_topo_sort @ 0x1301c70 (verified)
// Algorithm: iterative DFS with bitmap visited, explicit stack, post-order output.
// "stupid fast" = no recursion, no priority queue, bitmap O(1) visited check.
// 本函数只做 post-order 反转得到拓扑序，无随机打散 / tie-break。
// (PCG32 常数 0x5851f42d4c957f2d / 0x14057b7ef767814f 属 schedule_for_alloc @ 0x1302900
//  的调度键哈希，非 order_nodes 的同层 tie-break —— M17 已纠正。)
int GraphPrepare::order_nodes(bool) {
    if (opdef_map_.empty()) return 0;

    // Build node list and id->index map
    std::vector<op_id_t> nodes;
    std::unordered_map<op_id_t, size_t> id_to_idx;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || opdef->is_dead()) continue;
        id_to_idx[id] = nodes.size();
        nodes.push_back(id);
    }
    size_t node_count = nodes.size();
    if (node_count == 0) return 0;

    // Build successor adjacency list: edge from input source -> consumer
    std::vector<std::vector<size_t>> adj(node_count);
    for (size_t i = 0; i < node_count; i++) {
        auto it = opdef_map_.find(nodes[i]);
        if (it == opdef_map_.end() || !it->second) continue;
        for (const auto& conn : it->second->inputs) {
            auto src_it = id_to_idx.find(conn.src_id);
            if (src_it != id_to_idx.end()) {
                adj[src_it->second].push_back(i);
            }
        }
    }

    // Bitmap visited: (node_count + 31) / 32 uint32 words, zero-initialized
    // Source: 0x1301ce8-0x1301f74, movdqu批量�?
    std::vector<uint32_t> visited((node_count + 31) / 32, 0);
    auto test_bit = [&](size_t i) -> bool {
        return (visited[i / 32] >> (i % 32)) & 1u;
    };
    auto set_bit = [&](size_t i) {
        visited[i / 32] |= (1u << (i % 32));
    };

    // DFS explicit stack (linked-list stack in real binary, vector here)
    // Source: 0x60(%rsp) and 0x78(%rsp) two linked-list heads
    // Each stack node 24 bytes: +0x00 next, +0x08 prev, +0x10 node_id
    struct StackEntry { size_t node_idx; size_t neighbor_pos; };
    std::vector<StackEntry> dfs_stack;
    std::vector<op_id_t> post_order;
    post_order.reserve(node_count);

    for (size_t start = 0; start < node_count; start++) {
        if (test_bit(start)) continue;
        set_bit(start);
        dfs_stack.push_back({start, 0});

        while (!dfs_stack.empty()) {
            auto& top = dfs_stack.back();
            size_t idx = top.node_idx;

            // Find next unvisited neighbor (depth-first: break on first found)
            bool pushed = false;
            while (top.neighbor_pos < adj[idx].size()) {
                size_t neighbor = adj[idx][top.neighbor_pos++];
                if (!test_bit(neighbor)) {
                    set_bit(neighbor);
                    dfs_stack.push_back({neighbor, 0});
                    pushed = true;
                    break;
                }
            }

            if (!pushed) {
                // No unvisited neighbors: post-order output
                post_order.push_back(nodes[idx]);
                dfs_stack.pop_back();
            }
        }
    }

    // Reverse post-order = topological order
    std::reverse(post_order.begin(), post_order.end());

    int changed = (post_order != ordering_) ? static_cast<int>(post_order.size()) : 0;
    if (changed > 0) {
        ordering_ = std::move(post_order);
        graph_dirty_ = true;
    }
    return changed;
}

// ===== schedule_for_alloc 的 17 个文件内 helper（反汇编确证，M21+M22+M23+M24）=====
// 调度器本体 schedule_for_alloc @ 0x1302900（10286B，M17 已定位），内含若干无符号
// 静态局部 helper（落在 dynsym 间隙）。以下 10 个已逐指令反汇编确证，供后续重实现
// schedule_for_alloc 时按地址回填。ctx 结构布局见 M18–M23。
//
// 【M25 原名恢复】源文件 = st_cut.cc（.rodata 0x55b3db9 日志宏字符串证实）。
//   二进制 strip 掉 .symtab（仅 5758 个 .dynsym 导出），但 qnndsp 日志宏把 __func__
//   字符串嵌进 .rodata（0x55b3de5–0x55b402c，18 个 stcut_* 名）。计时器 0x130ea30
//   已反汇编证实为 vector<pair<const char*,u64>>::push_back({name, rdtsc>>4})：
//   名字标注其**后**即将执行的 stage 调用（SCHED_CYCLS 差分的数据源）。
//   已配对原名（名字 lea → 计时 call → 下一个非计时 call；M26 修正：配对需
//   距离≤0x30 且目标在 helper 簇 0x12fc000–0x1307000，12 个成立）：
//     stcut_read_nodes=0x12fc820  stcut_collate_sibs=0x12fd0e0
//     stcut_node_hash=0x1300210   stcut_quick_early_sort=0x1300600
//     stcut_block_relate=0x1300a40  stcut_connect_nodes=0x1300cc0 ←（M23 误标更正）
//     stcut_add_dependencies=0x1305130  stcut_strong_relevel=0x12fd600
//     stcut_arrange_sibs=0x12fffd0  stcut_clean_sibs=0x13056f0
//     stcut_delay_dma=0x13065f0   stcut_delay_dma_again=0x12fc740（内部已解，见
//       下方 M24/M25 收尾块：清出向量→按 flags&0xB==0 过滤入参向量→投影
//       [ctx+0x38 向量][id] 压回）
//   内联/未定：stcut_relate_by_tensor / stcut_topo_presort / stcut_measure_peak
//     （初版误配 0x130ebe0，实为 CSV 整数解析器：',' 跳字段→strtol→cmova 取值/0；
//     同族 0x130eb50=CSV 浮点解析 strtod 失败取默认 [rsi+0x128]）/ stcut_full_schedule
//     （SCHED_ITERS 重试循环，schedule_for_alloc 体内）/ stcut_convert_to_ids。
//     SCHED_OPTIONS 的 OT/IT/... 14 参数即由解析器家族在 1303ee9–1304060 从 CSV 读入
//     （条件：GraphPrepare+0x5638..0x56b0 optional 字段步长 0x18 已置位）。
//   其余叶子 helper（0x1309230/0x13096b0/0x1307820/0x1307960/0x1307f60/0x1307aa0/
//   0x1301040/0x12fba20/0x1307780/0x1307c90/0x1307d90 等）无日志调用 → 二进制内
//   无原名，保留本文件工作名。
//
// 【M26 schedule_for_alloc 内联准备段（13039e5–1303ee9，进重试循环前）】
//   1303a04 节点数=(end-begin)>>4/4；1303a27 NODE_ID_LIST BEGIN 日志（行927，优先级3）。
//   1303a2c 三个空 vector：rsp+0x10e0/0x1100/0x1120（各 24B）。
//   1303a5a rdtsc 原始 TSC（r14d 低/[rsp] 高）；1303a63-1303a8e 预算 double =
//     u64(this+0x6830) × double(this+0x55f8)（punpcklps/subpd@0x3970340/0x3970350
//     为 u64→double 改写）→ [rsp+0x40]；1303e80 常量 0xB2D05E00=3e9（3GHz TSC 基准）。
//   1303a94 节点序快照#1：_Znwm n×4B → vector@rsp+0x10e0，movups 64B/轮 拷自
//     [rsp+0x1140/0x1148]（有序节点 u32 列表）。
//   1303c4a call 0xf6f1f0(&rsp+0xa8, [rsp+0x38]) = 配置对象深拷贝（+0 u64、六个
//     24B string @+0x8/0x20/0x38/0x50/0x68/0x90、+0x80 u64、+0x88 u32、+0xb0 byte）。
//     ⚠ objdump 标签 sap_reduce_bandwidth 是回退标签（真身 0xf6ed80 止于 0xf6efdc）。
//   1303c5c 节点序快照#2 → vector@rsp+0x90（同源同法）。
//   1303e0f 门1：this+0x55dc(i32) ≤ 0 → 跳 13044cb 绕过循环；1303e1c-1303e58 门2：
//     [rsp+0x18]<<0xb ≤ double(r15)×[rsp+0x10] → 跳 13044cb（r15/[rsp+0x10] 来源未解）。
//   1303e5e TSC/16 起点 → [rsp]；1303e74 预算 int64 → [rsp+0x68]（deadline）。
//   1303e86 预取 6 个 optional 地址（this+0x5639/0x5651/0x5669/0x5681/0x5699/0x56b1）。
//   1303ee9- 选项解析循环：optional{flag@+0x0, csv 指针@+0x10}（0x18 步长）；
//     call 0x130ebe0 csv_int(idx) 后 test $0x1,%al 当布尔；假/未置位 → 取默认
//     this+0x55d4→[rsp+0x1ac]、this+0x55d8→[rsp+0x1b0]。1303fb2 另有 csv float
//     （0x130eb50）。即 SCHED_OPTIONS 14 参数的装载（OT/IT/…/DB，打印@1304642）。
//
// 【M27 内联 full_schedule 重试主循环（1303ee0–13044e3，逐指令实证）】
//   多轮试探—择优：回边 13044c5 jne→1303ee0。r14=轮次(1 起)，r15=r14−1=CSV 字段号。
//   三份节点序分工闭环：[rsp+0x10e0]=最优序（1304325 本轮 r15 更优时 copy 工作序入，
//     [rsp+0x18]=r15 历史最优 grain 数；退出后 13044cb-13044e3 copy 回工作序=最终输出）；
//     [rsp+0x90]=原始序（进循环前 1303c5c 拍；坏调度 130413f 与每轮尾 13043ff 重置用）；
//     [rsp+0x1140]=工作序（每轮被调度修改）。
//   每轮序列：
//   ① 1303ee9-1304060 按 (r14−1) 从各选项 CSV 重载参数（[rsp+0x1ac/0x1b0/0x1bc]，
//     默认 this+0x55d4/0x55d8/0x55dc）；1304062 记录区 [rsp+0x1500/0x1538=0x330e tag/
//     0x153a/0x153c=轮次两半/0x1540=本轮耗时]。
//   ② 130408b 迭代预算 = (opt2×1e6 + 3e9 − n×530000) / ((n×0x1450f0)>>32 + 26)，
//     n=([rsp+0x1300]−[rsp+0x12f8])>>3（工作表#1 元素数）；+ [rsp+0x1530] 已用量 →
//     作 r8 传 0x1306750。
//   ③ 1304104 call 0x1306750(ctx, [rsp+0x20], [rsp+0x30], &[rsp+0x1148], 预算,
//     [rsp+0x1ac], [rsp+0x30])＝本轮准备（内部 1306782 call 0x1307780 首轮五表快照）。
//   ④ 1304136 call 0x1306a20(ctx)→al＝本轮调度执行体（M21-M24 helper 链驱动者）。
//     失败(130413f)：恢复原始序+「Bad Schedule Detected (seed=%u)」(行 948,seed=r14d)；
//     1304186 this+0x55e8≠0 → 1304b48 Fatal(行 950)。
//   ⑤ 1304193 call 0x13065f0(ctx)→r15＝本轮 flow（grain 数；×2048=字节）。
//     13041a5 push 到 [rsp+0x1120] 向量(SCHED_FLOWS)；13042ad rdtsc/16−[rsp]→
//     [rsp+0x1540]，13042ce push 到 [rsp+0x1100] 向量(SCHED_CYCLES)；13042e9
//     FlowRetry 日志(行 959,优先级 11,%llx=r15<<0xb)。
//   ⑥ 130431e r15<[rsp+0x18] → 1304325 copy(工作序→最优序)、[rsp+0x18]=r15。
//   ⑦ 判定：1304347 cl=(r15<<0xb > [rsp+0x68] deadline)；130434f sil=([rsp+0x1f0]==0)；
//     1304363 预算公式第二遍（opt3=[rsp+0x1bc]）；13043b3 [rsp+0x1530]≤预算 →
//     13043f9 ebp=cl|sil；否则 13043b8 TIMEOUT 日志(行 971,%llu=opt3 原值,
//     #%d=[rsp+0x10] 0基轮次)+ebp=0 停。
//   ⑧ 13043ff 恢复原始序→工作序；1304434-13044a8 五表恢复（快照表[+0x190..0x208]
//     =栈 rsp+0x1420..0x14a0 → 工作表 rsp+0x12f8/0x1310/0x1340/0x1370/0x1388；
//     call ef9d90/13111b0/13112e0/8f02c0×2）。
//   ⑨ 13044b2 r14d ≥ this+0x55dc(Rt=最大轮数) → 退出；否则 r14d++、[rsp]=r12、
//     13044c5 ebp≠0 → 回①。
//   后循环：13044cb 恢复最优序；1304510 push 名"stcut_full_schedule"（⚠计时归属
//     异常见 M27 报告 §5a——push 在循环之后）；130453d push"stcut_delay_dma_again"；
//     1304557 call 0x12fc740(ctx,[rsp+0x50],&工作序)；1304591 push"stcut_convert_to_ids"
//     （无独立 call 可配对——完全内联于其后的统计/收尾段，区间未划出）。
//   统计块 1304596-1304ab9 受 this+0x5634 总闸：1304642 SCHED_OPTIONS 打印（行 999，
//     实参逐一重建出字段映射：OT=+0x55e4, IT=+0x55d8, Rg=+0x55d4, Rt=+0x55dc,
//     AB=+0x55e0, AM=+0x55e8(byte), EO=+0x5618, HD=+0x5619, DD=+0x561a(byte),
//     TR=+0x55f8(double,即预算乘数), LT=+0x5620, RC=+0x5628, RP=+0x562c,
//     DB=+0x5630(byte)；第 10 栈参=0x4628a0e 空串""宏尾参）；13046a2/13046d4/1304706
//     GRAINS_CUT/FLOW/RELEVEL ← [rsp+0x1510/0x1518/0x1520]（行 1000-1002）；
//     1304770 差分 STAT 循环（行 694,"%s=%lld",16B 元素表）；13047c3 call 0x13065f0
//     最终 delay_dma 复算（最优序上）→ rbx 再打印。
//   未完全理解：[rsp+0x1530]/[rsp+0x1f0] 写入点；0x1450f0 等效除数；CSV 值是否写回
//     this；统计块尾部 13047e0-1304900 打印未逐条对齐。详见
//     audit_verify/reports/M27_full_schedule_retry_loop.md。
//
// 【M28 循环体三大件（全解，见 M28_loop_body_three_functions.md）】
// 0x13065f0 — stcut_delay_dma = 目标函数（337B 全解）：
//   1306601 懒建权重树（cmpb ctx+0x18 → 0x130ccd0，与 0x1307200 同款）；
//   [ctx+0x158] = 工作序 vector 对象指针（★0x1306750:130677b 存入）。
//   遍历 u32 id：权重树 find-or-insert（48B 节点 key@+0x20/count@+0x28，
//   0x868c50 再平衡，树大小 ctx+0x30）；1306664 r13+=count；1306668 rax=max(rax,r13)。
//   返回 = 前缀和运行最大 = 内存峰值（grain 数，×2048=字节）。
//   对照 0x1307200（M22）：同一遍历，Σ前缀和（调试用）；本函数 max（调度目标）。
// 0x1306750 — 本轮准备（0x272B 全解）：[ctx+0x158]=&工作序；call 0x1307780 首轮
//   五表快照；v30=vector<u64>(n) 清零，call 0x130cea0(ctx,&v30,0,n) 填充；
//   call 0x130d3e0(ctx,arg2,arg3,&v10空,&v30,预算,opt1,arg7) = ★真引擎（副作用
//   写 ctx，两局部向量随后释放）。
// 0x1306a20 — 排序合法性验证器（0x750B 全解，st_cut.cc:1782）：
//   局部 BFS 工作表 v30@rsp+0x30 + visited 集合@rsp+0x48（32B 节点 key@+0x1c）。
//   倒序扫工作序：id 已在集合 → 130708c node=[[ctx+0x68]][id]，log(1,
//   "%s:1782:WARNING:invalid in this ordering: %u %llx") → 返 0（=Bad Schedule）；
//   否则入表入集，BFS 展开：邻接=[ctx+0x1f0]（快照#4=add-set 每 id 24B 记录
//   {u32 边表}）；过滤 [ctx+0x1c0]（快照#3=+0xb0 记录 flags&4）与 [ctx+0x1a8]
//   （快照#2=+0x80 字节表 &2）；canon=记录.id；未访问则入表入集。扫完返 1。
//   语义：替换/吸收关系传递闭包，被吸收 id 出现于工作序 → 非法。
// 0x1307170 — ctx 五表恢复（新识别，35B）：(+0x68,+0x80,+0xb0,+0xe0,+0xf8) ←
//   (+0x190,+0x1a8,+0x1c0,+0x1f0,+0x208)。循环尾 1304434-13044a8 即其内联展开。
//   ★更正：五张工作表 = +0x68/+0x80/+0xb0/+0xe0/+0xf8（M27 报告括注
//   +0x98/+0x110/+0x128 系换算笔误，栈槽没错）。
// 0x130d3e0 — develop_schedule()（M29 全解，st_cut.cc:3101/3331 日志自证；
//   0x130d3e0-0x130e87f，0x168 帧，唯一 retq@130e87f）：
//   签名 (ctx, arg2, arg3, out_vec=&v10, aux=&v30, inner_budget=r9, max_iters=arg7)。
//   序言：深拷工作序→[rsp+0x140]；[ctx+8]=u16 事件 id 计数；journal（rsp+0x120，
//   64B/条：kind@0/u32@0x10/u32@0x14/区间长@0x18/byte@0x20/u32@0x24/区间起@0x28/
//   u16 id@0x38）压 3 条初始记录（#2 kind=2 覆盖全 [0,n)）；ctx+0x98 逐节点 u16
//   归属表扩到节点数并全填记录#2 id（0x13114b0 扩容）。
//   主循环 130d820-130e0c6：门=[ctx+0x2a0](inner 计数)≥r9 预算 → 130e117
//   log "st_cut.cc:3331:WARNING:TIMEOUT: develop_schedule() hit inner-loop
//   timeout with %d iterations remaining"（★M24 遗留 [ctx+0x2a0] 读方关闭）；
//   arg7≤0 跳主循环；每 5 轮（模 5 魔数）log "…:3101:Schedule-cutting has %d
//   remaining iterations or %zu inner-loops"。
//   ①评分扫描 130d890-130daa8：阈值 r12 = ((u64)[ctx+0x278] 位技巧转 double) ×
//   [[ctx+0x270]+0x128] ÷ 2048（grain）；score = Σ_{k∈记录区间}(aux[k]+3·aux[k]·
//   [aux[k]>r12])（超阈值 3 倍惩罚面积），取最大记录。out_vec 数≠游标时走 130da40
//   历史切点复用（沿记录链 field0 步进，==2 停）。
//   ②130db08 0x130ab30 提案（切点 u32→[rsp+0x8]）；rdtsc×3 夹 130db76 0x1309a60
//   执行（M24/M25 迭代驱动）；[ctx+0x280] += 执行耗时（/16 拍粒度累计）。
//   ③落地：输出表条数==记录长 → 逐 id 写 ctx+0x98[id]=记录 id；否则 0x130b9a0
//   修正 journal 再写；130dc7f 0x130cea0(ctx,aux,区间起,区间长) 重填被切区间 aux；
//   门 [[ctx+0x270]+0x160]：扫 aux 非降前缀末点，越过水位 [rsp+0x38] 则 0x8dc340
//   把 [ctx+0x158] 拷回局部同步；130dddc 0x130d050 分裂判定(u8) → 130de22
//   0x130b600 分裂记录压回 journal（新 id=[ctx+8]++）；out_vec.push_back(切点)。
//   ④随机重启 130df50：门 [[ctx+0x270]+0x158]≥2 且 0xd79830(&ctx+0x2a8)取模==0
//   → 链上等分布选活记录 → 0x130bc70 扰动。
//   终段 130e143-130e801：门 [[ctx+0x270]+0x110]；130e15a 0x130a560 收尾；
//   逐记录：id=工作序[k]、node=[ctx+0x140][id]，收集 (id,node*) 批 + set<u32>
//   （32B 节点同验证器）；130e600 0x13125e0 按比较器 0x12fffb0 = (node指针,id)
//   字典序（=节点创建序）std::sort；多趟 Kahn：ctx+0xe0 邻接中 flags&4 关系的
//   canon（ctx+0xb0[rel].id）仍在 set → 本趟跳过；可发射则删出 set 并
//   130e79b ★写回 [[ctx+0x158]].begin[range_start++] = id —— 工作序原地重排。
//   130e810 eb fe = unreachable 自旋（编译器生成）。
//   未解：六 helper 内部（M30）、[ctx+0x2a0] 递增指令、[ctx+0x278] 语义、
//   field0==2 常量两用性。
// 0x130cea0 — aux 前缀和填充器（M29 全解，0x1a1B）：(ctx, aux_vec, start, len)；
//   130ceb1 懒建权重树（守卫 ctx+0x18→0x130ccd0）；r13 = start? aux[start-1] : 0；
//   for k∈[start,start+len)：id=工作序[k]，权重树 find-or-insert（48B 节点，
//   0x868c50 再平衡，树大小 ctx+0x30；防御插入 count=0），r13 += count，
//   aux[k] = r13。★aux = 内存累计曲线（grain）；M28 记的第 4 参"终点 n"应更正为
//   "长度 n"（130cece addq：上界=start+len；终段 130e1c6 同式互证）。
// 0x130ccd0 — 权重树懒初始化 = ★count 写入方（M28 遗留#1 关闭）：guard ctx+0x18
//   置 1；遍历 [ctx+0x68] 节点表建树节点（key=节点号，0x868c50，[ctx+0x30]++），
//   count = Σ_{rel∈ctx+0xe0[id] 邻接} ctx+0xb0[rel].field@+8（grain 量纲权重，
//   spill/DMA 内存量）。写一次后只读。
//
// 【M30 切分六件套内部（见 audit_verify/reports/M30_cut_helpers.md）】
// 0x130ab30 — 切点提案器（M30 全解）：外部候选路（journal[min(best,链next)].@0x20
//   置位 → ctx+0x240[rec.@0x24] 候选表过滤到 min 记录归属（ctx+0x98[node]==min_id），
//   切点=中位元素）；通用路 0x130a330(ctx,best@0x14,next@0x10,&局部树) 后扫区间
//   （跳 ctx+0x80[id]&4），weight=候选数×子树.count，jitter 门 [[ctx+0x270]+0x15c]
//   ≠0 时 +=(rand%(cfg+1)×weight)>>6；排除集（32B 节点树）校验；无候选→随机兜底
//   working_order[rand%区间长+区间起]；候选表在切点劈两半（前段→rsp+0x68、
//   后段→rsp+0x88 交执行器），返回切点 u32。
// 0x1309a60 — 迁移执行器（M30 全解）：0x1309940 估量 ≥0x5f5e0ff(1e8) → 先
//   0x13080a0 迭代慢路径再带输出向量重调；out.clear()；memset([ctx+0x128])；
//   工作列表={pivot(edx)}，换缓冲逐层泛洪：非 pivot 节点写 ctx+0x98[id]=u16
//   新记录号（归属改写）+ push 进 out；邻接展开门 = ctx+0x80[canon]&3==0 且
//   ctx+0x128 未标记，flags&4 边另需 ctx+0xb0[rel.@4].@8 > ctx+0xc8[rel.@4]；
//   收尾两遍把「对方不属本记录」的 flags&4 边 @4 字段导出到导出向量
//   （pivot 邻接一遍 + 全部已迁节点邻接一遍）；返回 0x1309940 估量。
// 0x130b9a0 — 工作序稳定划分修正器（M30 全解）：boundary = 区间起+区间长−|out|；
//   跳过前缀已属本记录节点；自首个属本记录位起压实（属→留、不属→存临时向量、
//   被换入源位写 0 打洞）；再单遍将临时向量内容填回尾部空洞（非洞先收入临时
//   向量再后填）= 区间内 [父保留前缀 | 迁移尾巴] 稳定划分，与 0x130b600 的
//   parent.@0x18 -= cut_size 完全自洽。
// 0x130d050 — 分裂判定器（M30 全解，返回 u8）：0x1305130(ctx, journal[best].@0x14)
//   打层标记进 ctx+0x128；对切集每条非 flags&4 关系取远端标记值做直方图计数；
//   <8 桶 → 返 0（不切）；≥8 桶 → 清零前 ≤5 桶 → 前缀和 ×0x2AAAAAAA（1/6 定点）
//   → verdict = (桶数×最大前缀和)/总数 < 0x340000000（≈最大前缀占比 < 19/桶数，
//   分布够平才切）。
// 0x130b600 — 记录分裂器（M30 全解）：计数器++×2 经 0x1301040 建 kind=2/1 两新
//   槽；journal[best].@0x18(区间长) -= cut_size；子记录=父区间尾段（@0x28 终点
//   继承、长=cut_size、判定 u8@0x50、id@0x68）经 0x130b840 压入 journal；
//   journal[best].@0(链后继)=新记录索引、子.@0x8=自身索引（链表式链接）；
//   对切集每 id：0x1300cc0(ctx, related→A, weight,0) + 0x1300cc0(ctx,B→node,
//   weight,0) 双侧重接线（0x13087f0 逐 id 清理）；journal[A].@0x30=元素计数。
// 0x130a560 — ★净重计算器（M30 全解；M29 曾注"收尾"，实为终段排序键来源）：
//   ctx+0x140 表 resize 到节点数 N（0x1311b50 扩/截），每项 =
//   Σ(add集 非 flags&4 的 @8) − Σ(subtract集 非 flags&4 的 @8) = 节点净内存
//   增量（grain）。⇒ M29 终段批排序比较器 0x12fffb0 的第一键 [ctx+0x140][id]
//   是净重而非"节点指针/创建序"（★M29 修正）。
// 0x130a330 — 局部权重树构建器（提案器用，M30 全解）：0x1305130 打标前快照
//   ctx+0x128 副本；0x1308c50 重打标（=层级标记 BFS，见下方收尾块，标记=层级
//   号≥2）；双标记（旧≠0 且新≠0）且 ctx+0x80&3==0 的节点插入调用方树，
//   count = (新标记−1)×旧标记（新标记为层级号故 ≥2，公式自洽）。
// 0x130bc70 — 随机重启（0x130bc70-0x130ccbf，~4.3KB）：记录从链摘除（前驱.@0 =
//   本记录.@0）、@0/@8 清零判死；拷贝其 @0x10→subtract 集与 @0x14→add 集；
//   局部关联树（40B 节点 key@0x1c/count@0x20）收 add 集非 flags&4 关系；
//   130c0a0 在 subtract 副本找 canon==@0x10 槽并双向修剪；重合并循环
//   （130c150-130c74e）快照对账：远端快照 subtract 集（ctx+0x208）逐成员入树，
//   130c3e7/130c3f1 活跃权重表 ctx+0xb0 两处减账，130c3f6 建 24B 重放记录
//   {op,0,type=4} 经 0x13080a0 重派发（type4=重新加入 add 集，M22），
//   再对远端活跃 add 集（ctx+0xe0）/快照 add 集（ctx+0x1f0）数非 flags&4 成员
//   （130c490/130c4f0 adcq 同构计数）——把该记录的关系编辑回滚向快照态
//   （撤销这刀重来）。
// 未完全理解：0x130bc70 中后段 0x130c519-0x130cb4d（同构对账/删除/重放循环
//   展开，未逐条注释）；0x1305130(stcut_add_dependencies) 内部与 0x1308c50
//   收尾段(1308f80-130922f)；[ctx+0x278] 语义；0x12fffb0 升/降序方向。
//
// 【M24/M25 遗留收尾（本日，转储 asm/M25_parser_variants_*.asm、
//   asm/M25_ctx_dtor_probe_*.asm、asm/M24_slowpath_*.asm、asm/CU_range_*.asm）】
// 0x130ec60/0x130ec65 — ★修正：非 CSV 解析器变体（M25/M26 误判），是树节点
//   递归析构（后序：call([node])×2 → delete node）：节点 {left@0, right@8, POD}，
//   即 ctx 权重树节点析构（0x30B：key@0x20、count@0x28）。
// 0x130eca0/0x130eca5 — 同族但节点带 vector 载荷 {left@0, right@8, …, buf@0x28,
//   end@0x30}：每节点先 end=buf、delete[] buf 再 delete 节点 = map<key,vector> 析构。
// 0x13073c0 — ctx 析构函数（0x13073c0–0x13076de）：依次释放 ctx+0x258/0x240/0x228
//   (树)/0x208(vector<vector>)/0x1f0/0x1d8/0x1c0/0x1a8/0x190/0x178/0x160/0x140/
//   0x128/0xe0/0xf8/0x110/…/0x98/0x68/0x50/0x38，末尾 this=ctx+0x20、root=[ctx+0x28]
//   尾跳 0x130ec60 销毁权重树。其异常孪生内联在 0x12fba20 尾部 12fc07b-12fc229
//   （同序，_Unwind_Resume@12fc229）→ 0x12fba20 全函数=0x12fba20-12fc046。
//   ctx 新字段：+0x228/0x230 map<key,vector> 树；+0x240/0x248、+0x258/0x260、
//   +0x1d8 = vector；加/减集与其快照(0xe0/0xf8/0x1f0/0x208) = vector<vector<u32>>
//   （析构逐 24B 元素删各自缓冲区）。
// 0x12fc740 — stcut_delay_dma_again 内部（调用点 1304557）：清出向量(vecA)、
//   按入参 vecB 大小 reserve，遍历 vecB 每个 id：[ctx+0x80][id]&0xB≠0 跳过；
//   否则经 0x105bd40 把 [ctx+0x38 向量][id]（u32）压入 vecA。
// 0x1308c50 — ★层级标记 BFS（M30 未解项半关）：memset(rcx 标记表)；worklist=
//   {edx 种子}；层级计数 ebp 自 2 递增（1308cdf）；每层每节点 idx：扫
//   subtract集[idx] 找「伙伴(表[rel].@4→表[far].@0) 标记≤1 且 表[rel].flags&4」
//   的关系 → idx 入输出向量；再扫 add集[idx]：flags&4 边的 canon 未标记 →
//   下一层 push + 打标。收尾段 1308f80-130922f 未逐条。
//   ⇒ 标记值=层级号(≥2)，解释 0x130a330 的 (新−1)×旧 公式。
// 0x1305130 = stcut_add_dependencies（M25 配名）全部 4 调用方：1303766
//   (schedule_for_alloc 阶段调用)、1305738(stcut_clean_sibs 内)、130a354
//   (0x130a330)、130d07c(0x130d050)；内部仍未拆。
// 0x130ed00/0x130ed10 — __cxa_throw 小包装（throw length_error/reserve 溢出，
//   vector push_back 家族的越界路径共用）。
//
// 0x1307200 — 权重树查找/插入 + 前缀和累加（rdi=ctx，返回 rax=u64 累加值）
//   1307211 cmpb $0,0x18(%rdi) / jne 130721f：未初始化 → call 130ccd0 建树。
//   1307222 起遍历 [ctx+0x158] vector<u32>，对每个索引在红黑树（root 在 ctx+0x28）
//   查 node->count（节点 +0x28，u64 权重）；命中 1307274 add 0x28(%rbx),%r13 /
//   add %r13,%rax。未命中（13072de）→ new 0x30 字节节点：+0x20=key、+0x28=count=0、
//   +0x10=parent、1307302 链接；130725b call 868c50（__tree_balance_after_insert）；
//   1307260 addq $1,0x30(%rbp) 树大小+1（+0x30 是 ctx 树大小计数，非节点域）。
//   返回 rax = Σ_k (Σ_{j≤k} weight_j)（前缀和之和）。消费方（M22 已证）：schedule_for_alloc
//   本体 1304801 / 13049f3 两处 call 0x1307200，返回值经 rbx 作 r8/rcx 传入 qnndsp_log
//   打调试统计（优先级 2 / 0），非 0x130d3e0 消费，不参与调度计算。ctx = &[rsp+0x1290]
//   （栈局部结构体，13047f9/13049eb lea 0x1290(%rsp),%rdi）。
//
// 0x1307780 — 一次性物化 5 个源 vector 到工作副本（rdi=ctx，void）
//   1307781 cmpb $0,0x220(%rdi) / je 130778c：一次性门，已执行则 ret；
//   130778f movb $1,0x220(%rdi) 置门。随后 5 次深拷贝（源 → 工作区）：
//     [ctx+0x68]→[ctx+0x190]（helper ef9d90）、[ctx+0x80]→[ctx+0x1a8]（13111b0）、
//     [ctx+0xb0]→[ctx+0x1c0]（13112e0）、[ctx+0xe0]→[ctx+0x1f0]（8f02c0）、
//     [ctx+0xf8]→[ctx+0x208]（8f02c0）。1307812 movl $0,0xc(%rbx) 清零 [ctx+0xc]。
//   说明：存在「源表 vs 工作表」两套权重表，由本函数一次性同步。
//
// 0x13087f0 — 从 add/subtract 集移除 op + 置 filter=0x80 + 追加输出（rdi=ctx, esi=op索引, rdx=回调）
//   1308821 起 r12=esi*3（24B/slot）；经权重表[esi].field0(+0x0) 与 .idx(+0x4) 链式定位
//   到 add 集与 subtract 集，swap-with-last 移除该 op（1308864–130887a 等）。
//   130895d test %rdx,%rdx：回调非空 → call 1307f60；否则 13089a7/13089b7
//   movl $0x80,0x10(权重表) 置 filter 位 0x80（已移除），并 13089d7 push op 索引到
//   [ctx+0x178] 输出 vector。1308ad6 call 853eb0 = vector<u32>::push_back
//   （1308acf add $4,%rsi 指向 weight_table[esi].idx，push 该 u32，M22 已证）。
//
// 0x1309a60 — spill 候选筛选（M20 遗留 [ctx+0x128] u16 三态表的写入方）
//   1309b38–1309b4b memset([ctx+0x128], 0, size) 全表清零；
//   1309bf8–1309bff 写 tag 数组 [ctx+0x98]（u16，tag[ebp]=cx）；
//   第二遍 1309d30 起遍历 add 集：1309d8e testb $0x4,0x10(权重表)（纳入位）、
//   1309da9 weight(+0x8) > [ctx+0xc8] 阈值、1309db6 cmpw $0,0x128(表)==0 且
//   1309d67 testb $0x3,[ctx+0x80]&0x3==0，则 1309dcf movw $1,(%rax,%r14,2) 置 1 并收集。
//   三态：0=未访问、1=本函数标记候选、≥2=「已最终处理」。M22 已证：全区域（0x1302900–
//   0x1313ac0）对 u16 表只有 movw $0x1（1305427/1308f3d/13092e9/130939a/1309dcf）与
//   memset 清零（1309b4b），无 incw/addw；cmpw $0x1;ja（>1 跳过，130529b/1308dbb/1309352）
//   的 ≥2 状态无生产者，属死/防御性代码（ctx 为栈局部，外部无法写）。
//
// 0x130ebe0 — 逗号分隔整数域解析器（rdi=域索引, rsi=逗号串，返回 eax=bool）
//   130ec00–130ec19 跳过 edi 个逗号；130ec23 mov $0xa,%edx / 130ec28 call strtol 基数 10；
//   130ec2f cmp %rbx,(%rsp)（endptr vs 原指针）→ seta 得 bool。非「6 选项 flag 检查器」
//   （M17 已纠正）；真 flag 检查是调用点前的 testb $0x1,[this+0x5638/0x5650/0x5668/0x5680/0x5698/0x56b0]。
//
// 权重表 [ctx+0xb0] 24B/slot 全字段：+0x0 field0(组id) +0x4 idx +0x8 weight(u64)
//   +0x10 filter（bit0x4=纳入、bit0x80=已移除）。新增 ctx 字段：+0xc8 阈值数组(u64)、
//   +0x80 组标志字节数组、+0x98 u16 tag 数组、+0x128 u16 三态表。
//
// 0x1309940 — 替换对批量记账 wrapper（M22；M23 改名：非 load_replacement_plan 包装器）
//   1309969 mov $0x5f5e0ff,%ecx 哨兵阈值（无阈值/恒接受）。4 次 call 1300cc0：
//     1309974(ctx,arg2,arg4,99999999,arg7)、13099d3(ctx,field0,arg3,99999999,arg7)、
//     1309a00(ctx,arg2,元素,99999999,arg7)、1309a30(ctx,元素,arg3,99999999,arg7)。
//   循环1 1309990–13099d8 遍历 add集[arg4] 跳过 filter0x4 或 field0==arg3 者；
//   循环2 13099da–1309a0c 遍历 arg5 向量；循环3 1309a0e–1309a3c 遍历 arg6 向量。
//   1309a55 jmp 1309810 尾调用（ctx,arg2,arg3）。被 0x1309a60 调用 2 次（1309ab1/1309afd）。
//
// 0x1300cc0 — 替换对记录核心（M23，无符号，gap 0x12f9a9e–0x1301c70）
//   【M25 原名 = stcut_connect_nodes】0x130368b lea "stcut_connect_nodes" → 130369a
//   计时 call 130ea30 → 130371d call 0x1300cc0（另 1303866/1303887 重试点）。
//   名字与 M23 功能解读一致：连接节点 arg2→arg3 并设阈值权重（建/查替换对）。
//   签名 u32(ctx*, u32 arg2, u32 arg3, u64 threshold, void* callback)，返回映射到 arg3 的槽索引。
//   objdump 标签 load_replacement_plan@@Base+0x72b0 是回退标签（非调用关系）：M23 已证
//   load_replacement_plan(DataReader&) @0x12f9820 不调用本函数，其反序列化进 this->[0x72e0]。
//   1300cec arg2==arg3 时 1300cf4 取 [ctx+0x68][arg2] 打日志（qnndsp_log 优先级1）。
//   1300d31–1300d77 在 add集[arg2] 遍历，找 权重表[idx].field0==arg3：
//     命中 1300fc9：callback 非空记 {idx,旧weight,type=0}（1300ff9 call 0x1307f60），
//       1301008–130101d 权重表[idx].weight = min(threshold+旧weight, 99999999)，返回 idx。
//     未命中：按 [ctx+0xc] 分路——快路径(==0) 1300da7/1300dab 从 [ctx+0x178] 空闲槽栈
//       pop 2 索引(ebp=slotB, ebx=slotA)，1300dc5/1300e02 复用写两槽；慢路径(!=0)
//       1300e6f/1300eb8 call 0x1308b10 push_back 2 个 24B 槽 + 1300e8c/1300ed3 call
//       0x12fc610 push 2 个 u64 0 到 [ctx+0xc8]。（两 callee 已解，M24 收尾：
//       0x1308b10=vector<24B>::push_back 缺省扩容版——快路径 24B 原地拷、满则
//       max(2n,n+1)×0x18 扩容 memcpy，元素数按字节>>3×0xAAAAAAAAAAAAAAAB ÷3；
//       0x12fc610=vector<u64>::push_back 同构（8B 元素，上限 0x1fffffffffffffff）；
//       另 0x12fc4d0 为第三份 24B 拷贝，仅 0x12fba20 尾部 @12fc00d 调用。）
//       两路结果一致：
//         slotA(idx=ebx): field0=arg3, idx=ebp, weight=threshold, filter=0
//         slotB(idx=ebp): field0=arg2, idx=ebx, weight=99999999, filter=[ctx+0xc]|4
//       （slotA/B 的 +0x4 idx 互指，构成双向替换对；[ctx+0xc8] 对应两项清零）
//   1300ee0/1300eed call 0x1307820(ctx,arg2,slotA)/(ctx,arg3,slotB)；
//   1300efa/1300f07 call 0x1307960(ctx,arg2,slotB)/(ctx,arg3,slotA)（建双向链接）。
//   1300f0c–1300f75 同步 [ctx+0x110] 每槽 u32 数组到槽数（扩充 call 0xcf27b0 / 收缩改 end）。
//   1300f7c callback 非空记 {slotA,0,type=2}（1300fa3 call 0x1307f60）；1300fb8 返回 slotA。
//
// 0x1309810 — 调度迭代驱动（M23，rdi=ctx, esi=arg2, edx=arg3，返回 u64 累计处理数）
//   130981e addq $1,0x10(%rdi) 计数；1309829–130983f memset([ctx+0xc8],0,size) 清阈值数组。
//   130984d call 1309230(ctx,arg2,arg3) 首次探测，1309852 test/je 130992d（0 → 返回 0）。
//   循环A 1309860–13098c8：1309897 call 13096b0(ctx,arg2,arg3,99999999,0) 批量处理，
//     13098c2 r12+=rax；rdtsc 计时 13098bb [ctx+0x288]+=(TSC_end/16-TSC_start/16)；
//     13098c8 jne 1309860（处理数≠0 续）。
//   循环B 13098ca–1309927：13098f9 call 1309230 探测，130991d [ctx+0x290]+=耗时，
//     1309927 jne 1309860（非0 回循环A）。130992d 返回 r12。
//   （rdtsc 双采样 TSC>>4 与 (TSC>>3)&~1 净效果为 /16，属计时改写，非调度计算。）
//
// 0x1309230 — 组可达传播/收集（M24；rdi=ctx, esi=seed, edx=target，返回 u64 收集计数）
//   130926d memset([ctx+0x128] u16 三态表,0)；1309285 memset([ctx+0x110] u32 位数组,0)。
//   13092b4 _Znwm(4) 建 frontier vector<u32>=[seed]（13092c0 mov %ebp,(%r10)）；
//   13092e9 movw $0x1,(%rax,%rcx,2) 给 seed 打 tag=1。代计数 [rsp+0x14] 初值 2（13092fc）。
//   外层循环（1309330–13095ea）逐代处理 frontier：130934f ecx=frontier[i]，
//   1309352 cmpw $0x1（tag!=1 跳过，含 M22 的 >1 跳过）；130935d 写本代 tag=[rsp+0x14]。
//   内层循环（13093ed–13093e7）遍历 add集[ecx]（130936c r12/r13=[ctx+0xe0][ecx*3]）：
//     ebp=成员；13093f1 [ctx+0x2a0]++ 访问计数；r15d=权重表[ebp].field0；
//     1309408 cmpw $0,(%rax,%r15,2)==0（未访问才进）；1309417 rdx=[ctx+0xc8][ebp] 与
//     权重表[ebp].weight(+0x8) 比较，jge 跳过（阈值已≥权重，无可取）；1309429
//     testb $0x2,[ctx+0x80][r15]（组标志 bit2：置位时须 r15==target，1309430 cmp/jne）；
//     通过则 1309388 推入结果 vector（[r9]=[rsp+0x60]，扩容 1309443–130958e 内联 2x）、
//     130939a movw $1 给 r15 打 tag=1（下代 frontier）、13093b8 置位 [ctx+0x110] 位图
//     （rdx=ebp>>5, or 1<<(ebp&31)）、13093bb r8++、13093d3 r11d|=（field0==target）。
//   本代结束（13095a0）：13095a0 testb $0x1,[rsp+0x38]（已触达 target 标志）非 0 →收尾；
//   否则 13095c8 代计数++、结果/新frontier 向量互换（13095c8–13095db），回 1309330。
//   13095ef–1309643 释放两 vector；130962f mov %r8,%rax 返回收集计数。
//   语义：从 seed 沿 add 集逐代洪泛，收集「权重>阈值」的可达槽，触达 target 即停。
//
// 0x13096b0 — 递归容量受限权重搬运（M24；rdi=ctx, esi=seed, edx=arg3, rcx=cap, r8d=flags，
//   返回 u64 实际搬运量）
//   13096ca seed==arg3 → 13097e6 返回 0。13096dd 取 add集[seed]（[ctx+0xe0][seed*3]），
//   空 → 返回 0。13096ff 深度计数 [rsp+0xc]++。130970d 累计器 [rsp+0x10]=0。
//   循环（1309742–13097e1）遍历 add集[seed]：ebx=成员；130974c r12=ebx>>5 取
//   [ctx+0x110] 位图字；130975b bt %ebx,%edx 测试 pending 位——
//     位 0 → 1309733 not/and 清位跳过（已处理）；位 1 → 1309760：
//     r14=权重表[ebx].weight(+0x8) − [ctx+0xc8][ebx]（可用量=权重−阈值）；
//     130977b cmp/cmovge %r8,%r14 → r14=min(可用量,cap)；
//     130978f–130979e 递归 call 13096b0(ctx, 权重表[ebx].field0, arg3, r14, [rsp+0xc])；
//     返回 0 → 1309720 清位继续；非 0 → 13097ac [ctx+0xc8][ebx]+=rax、
//     13097b7 累计器+=rax、13097c8 [ctx+0xc8][权重表[ebx].idx]-=rax（对槽扣减）；
//     13097d0 cmp $0x2710(10000),%r14：<10000 → 13097ed 返回累计（小量即停）；
//     否则 13097d9 cap-=rax 续。
//   13097ed 返回 [rsp+0x10]。
//   语义：自 seed 向下游递归「抽取」至多 cap 的权重，写入阈值账本 [ctx+0xc8]；
//   单步 <10000 提前返回（防长链微搬运）。
//
// 0x1307820 — add 集追加（M24；rdi=ctx, esi=slot, edx=val，void）
//   1307830 rbx=[ctx+0xe0]；1307843 rdx=add集[slot*3].end；1307848 cmp 0x10(容量)：
//   快路径 130784f–1307856 *end=val, end+=4；扩容 1307868–1307930（2x+memcpy+free，
//   溢出 130794f/1307954 length_error/bad_array_new_length）。
//
// 0x1307960 — subtract 集追加（M24；rdi=ctx, esi=slot, edx=val，void）
//   与 0x1307820 同构，作用表 [ctx+0xf8]：1307970 rbx=[ctx+0xf8]，快路径 130798f–1307996，
//   扩容 13079a8–1307a70。
//
// 0x1307f60 — 24B 账本记录 push_back（M24；rdi=vector<24B>* , rsi=记录*，返回新记录指针）
//   快路径 1307f7c–1307f96：end<cap 时拷 24B（movups 16B+u64），end+=0x18。
//   扩容 1307f9b–1308076：1307fb2 imul 0xAAAAAAAAAAAAAAAB（无符号 /3 魔数）算旧容量，
//   1307fd9–1307fe0 新容量=max(旧+1, 2x)，上限 0x555555555555555；1307ff5 容量 0 →
//   1308020；130800f _Znwm(24*cap)；memcpy 旧数据；free 旧。
//   130807a rbx=end−0x18 → rax 返回 pushed 记录指针（非 void，供调用方就地改写）。
//   溢出 130808b call 13114a0 / 1308093 call 855470。
//
// 0x1307aa0 — 节点槽退役（M24；rdi=ctx, esi=slot，void）
//   1307ab3/1307aca 清空 slot 的 add 集（[ctx+0xe0][slot*3]）与 subtract 集
//   （[ctx+0xf8][slot*3]）：mov (%rcx,%rdx),%rsi; mov %rsi,0x8(...) 即 end=begin（不释放）。
//   1307adf [ctx+0x68][slot*8]=0（节点指针清空）；1307af2 movb $0x83,[ctx+0x80][slot]
//   （组标志=0x83）；1307b06 push slot 进空闲槽向量 [ctx+0x160/0x168/0x170]（快路径
//   1307b06–1307b0d，扩容 1307b23–1307bf3）。注意：这是节点槽空闲表，与权重槽空闲表
//   [ctx+0x178/0x180]（0x1300cc0 pop）是两个不同 vector。
//
// 0x1301040 — 节点注册（M24；rdi=ctx, rsi=节点指针, dl=flags；返回 eax=槽索引；
//   [ctx+0xc] 的唯一「读路」消费者）
//   1301068 cmpl $0,[ctx+0xc]：!=0（构造期 0x40）→ 追加路 130106e：节点指针 push 进
//   [ctx+0x68/0x70/0x78] vector<u64>（扩容 13010e2–13011a7），槽=旧 size；
//   ==0 → 复用路 130109d：从 [ctx+0x160/0x168] pop 槽号 r13d（13010ad mov -0x4(%rax)，
//   end-=4），13010c4 [ctx+0x68][r13]=节点，13010cb [ctx+0x80][r13]=flags 字节，
//   13010d6 movw $0,[ctx+0x128][r13*2]（三态表复位）。
//   13011af orl [ctx+0xc],%r12d（flags|当前模式字）→ 13011c6 把该字节 push 进
//   [ctx+0x88/0x90] 字节 vector（flags 账本）。返回槽索引。
//
// 0x12fba20 — ctx 构造函数（M24；rdi=ctx, rsi=GraphPrepare*，schedule_for_alloc
//   1302a3b 调用）—— [ctx+0xc] 的初始化方
//   12fba44 [ctx+0x0]=[rsi+0x7468]（自 GraphPrepare 取初值）；12fba4e [ctx+0x8]=0(u16)；
//   12fba54 movl $0x40,0xc(%rdi) ← [ctx+0xc]=0x40（构造即非 0！bit0x40=建表/追加期）；
//   12fba5b [ctx+0x10]=0；12fba63 [ctx+0x18]=0；随后 0x20–0x2a8 全部向量/数组清零。
//   [ctx+0xc] 全部写方（0x12f9a10–0x1312b85 范围 grep 确证）：初始化 12fba54=0x40、
//   清零 1307812（0x1307780 快照函数内）与 130d1c2；无其他写指令。
//
// 0x1307780 — ctx 阶段快照（M24；rdi=ctx；[ctx+0x220] 一次性守卫）
//   1307781 cmpb $0,0x220(%rdi) 已做过→ret；130778f 置 1。随后 5 次快照调用：
//   13077a5 call ef9d90（[ctx+0x68/0x70] 节点指针 vector → [ctx+0x190]）、
//   13077bf call 13111b0（[ctx+0x80/0x88] flags 字节 vector → [ctx+0x1a8]）、
//   13077d9 call 13112e0（[ctx+0xb0/0xb8] 权重表 → [ctx+0x1c0]）、
//   13077f3 call 8f02c0（[ctx+0xe0/0xe8] add 集表 → [ctx+0x1f0]）、
//   130780d call 8f02c0（[ctx+0xf8/0x100] subtract 集表 → [ctx+0x208]）。
//   1307812 movl $0,0xc(%rbx) 清模式字 → 进入「复用空闲槽」阶段（0x1301040/0x1300cc0
//   的快路径自此生效）。调用方：0x1306750 helper @1306782。
//
// 0x1307c90 — 替换对重指向 A 侧（M24；rdi=ctx, esi=slot, edx=newPartner，void）
//   eax=权重表[slot].field0（旧伙伴）；1307cd0 在 subtract集[旧伙伴] 线性找 slot，
//   命中 1307ce4–1307cfa swap-with-last 移除；1307d20 在 add集[旧伙伴] 找
//   权重表[slot].idx(+0x4)，命中 1307d34–1307d4a 同法移除；1307d53 call 0x1307960
//   (ctx,newPartner,slot)（subtract集[newPartner] push slot）；1307d71 call 0x1307820
//   (ctx,newPartner,权重表[slot].idx)（add集[newPartner] push idx）；
//   1307d7d 权重表[slot].field0=newPartner。
//   调用方（M24 遗留收尾，全簇 grep 确证）：仅 0x130bc70 随机重启 @130c797/130c7f8
//   ——重指向是重启回滚的接线工具（与 M30「撤销这刀重来」互证）。
//
// 0x1307d90 — 替换对重指向 B 侧（M24；rdi=ctx, esi=slot, edx=newPartner，void；
//   0x1307c90 的镜像，作用于 idx 侧）
//   ecx=权重表[slot].idx(+4)；eax=权重表[idx].field0（对侧旧伙伴）；1307de0 在
//   add集[对侧旧伙伴] 找 slot，命中 1307df4–1307e0f swap 移除；1307e30 在
//   subtract集[对侧旧伙伴] 找 idx，命中 1307e44–1307e5d swap 移除；1307e66 call
//   0x1307820(ctx,newPartner,slot)（add集[newPartner] push slot）；1307e84 call
//   0x1307960(ctx,newPartner,权重表[newPartner].idx)（subtract集[newPartner] push）；
//   1307e98 权重表[权重表[newPartner].idx].field0=newPartner。
//   调用方（M24 遗留收尾）：仅 0x130bc70 随机重启 @130c18a。
//
// 0x13080a0 — 6-case 记录重放 dispatch（M22，rdi=ctx, rsi=24B 记录 vector，逆序 pop）
//   记录 24B：+0x0 op索引(u32,r14d) +0x8 value(u64) +0x10 type(u32,ecx)。
//   13080f7 end-=0x18 逆序 pop；130810b cmp $0x5 / ja 1308136（>5 错误）；
//   1308115 movslq 0x0(%r13,%rcx,4) 跳转表（r13=0x55b3d04，6 个 32-bit 相对偏移）→ jmp *%rcx。
//   type→case（跳转表已提取）：
//     0→130811f 置权重 weight_table[op].weight(+0x8)=value（130812f）；
//     1→1308136 日志(优先级100, fmt@0x55b42b0)；2→1308151 从 add 集移除(swap-with-last)；
//     3→13080d4 [ctx+0x70]-=8 且 [ctx+0x88]-=1（pop 计数）；4→13081a8 加入 add 集(push)；
//     5→13081fe 日志(优先级100, fmt@0x55b4288)。type 1/5=日志记录、3=pop 记录、0/2/4=权重/add 更新。
//
// 0x853eb0 — vector<u32>::push_back 冷路径（M22，rdi=vector*, rsi=const u32*）
//   853ec1 mov 0x8(%rdi),%rdx（end）/ 853ec5 cmp 0x10(%rdi)（capacity）/ 853ec9 je 扩容：
//   快路径 853ecb–853ed4 *end=*element, end+=4；扩容路径 853ee7–853faa 2x 增长+memcpy+free，
//   溢出 853fbd call 8553f0(length_error)/853fc5 call 855470(bad_array_new_length)。
//   7 处调用点（13057e5/1308ad6/1309f6b/130b311/130b328/130bb50/130bc01），均 push 一个 u32
//   （典型 weight_table[].idx）。
//
// common_subexpr_eliminate: CSE
// Source: graph_prepare.cc, 4156 bytes at 0xF6A5B0
int GraphPrepare::common_subexpr_eliminate(bool) {
    // Source: graph_prepare.cc, 4156 bytes at 0xF6A5B0
    // Algorithm:
    // 1. For each enabled op, build a signature:
    //    signature = (name_tag_hash, sorted_input (src_id, out_idx) pairs)
    // 2. Hash the signature
    // 3. Look up in hash table
    // 4. If match found with identical signature: supersede (replace) this op
    // 5. Return number of ops eliminated

    int eliminated = 0;

    // signature -> existing opdef
    std::unordered_map<uint64_t, OpDef*> cse_map;

    // Helper: compute CSE signature for an op
    auto compute_signature = [](const OpDef* opdef) -> uint64_t {
        uint64_t sig = opdef->hash_key();
        // Mix in input connections (sorted for order-independence)
        std::vector<InputConn> sorted_inputs = opdef->inputs;
        std::sort(sorted_inputs.begin(), sorted_inputs.end(),
            [](const InputConn& a, const InputConn& b) {
                if (a.src_id != b.src_id) return a.src_id < b.src_id;
                return a.out_idx < b.out_idx;
            });
        for (const auto& conn : sorted_inputs) {
            sig ^= (conn.src_id * 0x9E3779B97F4A7C15ULL);
            sig ^= (static_cast<uint64_t>(conn.out_idx) << 32);
            sig = (sig << 1) | (sig >> 63);
        }
        return sig;
    };

    std::vector<op_id_t> to_remove;

    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled() || opdef->is_const() || opdef->is_dead())
            continue;
        if (id == input_node_id_ || id == output_node_id_) continue;

        uint64_t sig = compute_signature(opdef.get());

        auto it = cse_map.find(sig);
        if (it != cse_map.end()) {
            // Found potential match - verify inputs are identical
            OpDef* existing = it->second;
            bool match = (existing->inputs.size() == opdef->inputs.size());
            if (match) {
                for (size_t i = 0; i < opdef->inputs.size(); ++i) {
                    if (existing->inputs[i].src_id != opdef->inputs[i].src_id ||
                        existing->inputs[i].out_idx != opdef->inputs[i].out_idx) {
                        match = false;
                        break;
                    }
                }
            }

            if (match) {
                // Replace this op's consumers to point to existing op
                // Update consumers of this op to use existing->op_id instead
                for (op_id_t consumer_id : opdef->consumers) {
                    auto cit = opdef_map_.find(consumer_id);
                    if (cit != opdef_map_.end() && cit->second) {
                        for (auto& conn : cit->second->inputs) {
                            if (conn.src_id == id) {
                                conn.src_id = existing->op_id;
                            }
                        }
                        // Add to existing op's consumer list
                        existing->consumers.push_back(consumer_id);
                    }
                }
                to_remove.push_back(id);
                eliminated++;
            }
        } else {
            cse_map[sig] = opdef.get();
        }
    }

    // Remove eliminated ops
    for (op_id_t id : to_remove) {
        opdef_map_.erase(id);
    }

    if (eliminated > 0) graph_dirty_ = true;
    return eliminated;
}

// const_prop: constant propagation
// Source: graph_prepare.cc, 1667 bytes at 0xF74E30
void GraphPrepare::const_prop(HexagonNNEnv& env, bool aggressive) {
    // Source: graph_prepare.cc, 1667 bytes at 0xF74E30
    // Algorithm:
    // 1. Find all const ops (OpDef_Const)
    // 2. For each consumer of a const op:
    //    - If all inputs are const: the op can be constant-folded
    //    - replace_opdef_with_opconst: replace with OpDef_Const
    // 3. aggressive=true: fold through more op types (e.g., Concat, Reshape)
    // 4. const_prop_extract_outputs: evaluate the op with const inputs
    // 5. zero_fill_outputs_for_const_prop: if can't evaluate, zero-fill
    // 6. try_reduce_const: simplify const expressions

    bool changed = true;
    while (changed) {
        changed = false;

        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled() || opdef->is_const() || opdef->is_dead())
                continue;
            if (id == input_node_id_ || id == output_node_id_) continue;

            // Check if all inputs are const
            bool all_const = !opdef->inputs.empty();
            for (const auto& conn : opdef->inputs) {
                auto src_it = opdef_map_.find(conn.src_id);
                if (src_it == opdef_map_.end() || !src_it->second ||
                    !src_it->second->is_const()) {
                    all_const = false;
                    break;
                }
            }

            if (all_const) {
                // All inputs are const -> this op can be folded
                // Source: const_prop_extract_outputs @ 0xF7A090 (2043 bytes)
                // Evaluate the op with constant inputs to produce constant output
                //
                // For simple ops (Add, Mul, etc.): compute result directly
                // For complex ops (Conv, MatMul): would need full kernel
                //
                // Source: replace_opdef_with_opconst @ 0xF7A950 (609 bytes)
                // Replace this opdef with an OpDef_Const containing the folded value

                // Mark as const (simplified: real impl would compute value)
                opdef->flags |= OP_CONST;
                changed = true;
            }
        }
    }

    if (changed) graph_dirty_ = true;
}

void GraphPrepare::const_prop_and_cse(HexagonNNEnv& env, bool aggressive, bool* changed) {
    // Source: graph_prepare.cc, 244 bytes at 0xF6A4B0
    const_prop(env, aggressive);
    int cse_changed = common_subexpr_eliminate(true);
    if (changed) *changed = (cse_changed != 0);
}

// eliminate_split_nodes: fold Split ops by replacing Split outputs with
// injected placeholder const tensors (type=4, dims=[1]).
// HtpPrepare's graph import eliminates Split nodes: each Split output is
// replaced by an injected placeholder tensor (no real producer). Downstream
// ops reference this placeholder, which appears as type=4 dims=[1] in
// before_graph.json.
void GraphPrepare::eliminate_split_nodes() {
    // Create one shared placeholder per Split GROUP (identified by 'grouping').
    // QNN uses dt=2147483647 (undefined quantization) with dims=[1] for
    // Split output placeholders (e.g., k/v tensors in attention).
    // All Split outputs from the same original Split share one placeholder.
    
    // First pass: create one placeholder per unique grouping
    std::map<std::string, op_id_t> group_placeholders;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
        if (!opdef->name_tag) continue;
        std::string nm = opdef->name_tag->name() ? opdef->name_tag->name() : "";
        if (nm != "Split") continue;
        std::string grp = opdef->grouping;
        if (group_placeholders.count(grp)) continue;
        
        op_id_t placeholder_id = next_op_id_++;
        OutputDef pod{};
        pod.rank = 1;
        pod.dtype = 2147483647;  // QNN undefined quantization datatype
        pod.dims[0] = 1;
        pod.element_size = 4;
        uint8_t zero[4] = {0, 0, 0, 0};
        append_const_node(placeholder_id, pod, zero, 4);
        group_placeholders[grp] = placeholder_id;
    }
    
    // Second pass: redirect all Split outputs to their group's placeholder
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& [id, opdef] : opdef_map_) {
            if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;
            if (!opdef->name_tag) continue;
            std::string nm = opdef->name_tag->name() ? opdef->name_tag->name() : "";
            if (nm != "Split") continue;
            std::string grp = opdef->grouping;
            auto pit = group_placeholders.find(grp);
            if (pit == group_placeholders.end()) continue;
            op_id_t placeholder_id = pit->second;

            // Redirect all consumers: replace input pointing to this Split
            // with the placeholder const.
            for (auto& [cid, cdef] : opdef_map_) {
                if (!cdef || !cdef->is_enabled() || cdef->is_dead()) continue;
                for (auto& conn : cdef->inputs) {
                    if (conn.src_id == id) {
                        conn.src_id = placeholder_id;
                    }
                }
                auto& cs = cdef->consumers;
                cs.erase(std::remove(cs.begin(), cs.end(), id), cs.end());
            }

            // Mark Split as dead
            opdef->flags |= OP_DEAD;
            changed = true;
            break; // restart iteration after modification
        }
    }
    // Rebuild consumer lists after redirecting inputs to placeholders
    rebuild_consumers();
    // Directly erase dead Split nodes from opdef_map_ (no cascade DCE).
    // QNN's before_graph keeps upstream nodes even after Split elimination,
    // so we must NOT cascade-delete the Split's input producer.
    for (auto it = opdef_map_.begin(); it != opdef_map_.end(); ) {
        if (it->second && it->second->is_dead()) {
            it = opdef_map_.erase(it);
        } else {
            ++it;
        }
    }
}

// rebuild_consumers: reconstruct consumer lists from input connections.
// Needed after build_graph because param const nodes (created with high IDs)
// may be created AFTER the op that references them, so append_node's consumer
// registration misses them. This walks all ops' inputs and rebuilds consumers.
void GraphPrepare::rebuild_consumers() {
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef) continue;
        opdef->consumers.clear();
    }
    int links = 0;
    for (auto& [id, opdef] : opdef_map_) {
        if (!opdef || !opdef->is_enabled()) continue;
        for (const auto& conn : opdef->inputs) {
            auto src_it = opdef_map_.find(conn.src_id);
            if (src_it != opdef_map_.end() && src_it->second) {
                auto& cs = src_it->second->consumers;
                if (std::find(cs.begin(), cs.end(), id) == cs.end()) {
                    cs.push_back(id);
                    links++;
                }
            }
        }
    }
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
