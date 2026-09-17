#include "hnnx/schedule/scheduler.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/schedule/e2e_bridge.hpp"  // build_stcut_input_from_graph / run_stcut_schedule
#include <algorithm>
#include <cstring>
#include <set>

namespace hnnx {

// ========== KernelSelector ==========

KernelSelector::KernelSelector() = default;

std::string KernelSelector::dtype_suffix(DType dt) const {
    switch (dt) {
        case DType::Float32:  return "ff";
        case DType::Float16:  return "ff16";
        case DType::Int8:     return "i8";
        case DType::Int16:    return "i16";
        case DType::Int32:    return "i32";
        default:              return "ff";
    }
}

KernelSelector::KernelResult
KernelSelector::select(const std::string& op_name, DType dtype,
                       bool is_input_side, bool is_output_side) const {
    KernelResult result;
    std::string suffix = dtype_suffix(dtype);

    if (op_name == "Transpose" || op_name == "transpose") {
        if (is_input_side) {
            result.kernel_name = "*InputSlice@Ff.s4*6.";
            result.f2 = 0x02;
        } else {
            result.kernel_name = "Transpose_impl@" + suffix + "*2.fi.t";
            result.f2 = 0x03;
        }
    } else if (op_name == "Reshape" || op_name == "reshape") {
        result.kernel_name = "flat_from_vtcm@" + suffix + ".Ff.";
        result.f2 = 0x11;
    } else if (op_name == "FullyConnected" || op_name == "MatMul" ||
               op_name == "MatMul_bias" || op_name == "FCElementwiseAdd") {
        result.kernel_name = "MatMul_bias@" + suffix + "*4";
        result.f2 = 0x14;
    } else if (op_name == "Input" || op_name == "InputNodeSetup") {
        result.kernel_name = "";
        result.f2 = 0;
    } else if (op_name == "Output" || op_name == "OutputNodeSetup") {
        result.kernel_name = "";
        result.f2 = 0;
    } else if (op_name == "Const") {
        result.kernel_name = "Const";
        result.f2 = 0;
    } else {
        result.kernel_name = op_name;
        result.f2 = 0;
    }

    return result;
}

std::vector<std::string> KernelSelector::get_kernel_table() const {
    return {
        "Shape",
        "*InputSlice@Ff.s4*6.",
        "@DmaCheckpointSet",
        "@DmaCheckpointWait",
        "Const",
        "Transpose_impl@Ff*2.fi.t",
        "flat_from_vtcm@ff.Ff.",
        "MatMul_bias@ff*4",
        "Transpose_impl@ff*2.fi.t",
        "*OutputSlice@ff.s4*3.",
        "@SyncOp",
    };
}

// ========== Scheduler ==========

Scheduler::Scheduler() : next_step_(0), next_dma_tag_(0x11), next_vtcm_block_(0x19) {}

ScheduledOp Scheduler::make_op(uint32_t tensor_id, OpRecordType type,
                               uint32_t f2, uint32_t block_ref,
                               const std::vector<uint32_t>& extras,
                               const std::string& kernel_name,
                               const std::string& step_name) {
    ScheduledOp op;
    op.record_id = compute_record_id(next_step_++);
    op.tensor_id = tensor_id;
    op.type = type;
    op.f2 = f2;
    op.block_ref = block_ref;
    op.extras = extras;
    op.kernel_name = kernel_name;
    op.step_name = step_name;
    return op;
}

uint32_t Scheduler::alloc_vtcm_block(uint32_t tensor_id) {
    auto it = tensor_block_map_.find(tensor_id);
    if (it != tensor_block_map_.end()) return it->second;
    uint32_t block = next_vtcm_block_++;
    tensor_block_map_[tensor_id] = block;
    return block;
}

// === Step generators (pattern-based, verified from real .bin) ===

void Scheduler::gen_input_node_setup(Plan& plan, const TensorInfo& input_tensor, bool is_first) {
    // Input node setup: counter=0 (graph_node), type=compute, f2=hash
    // Extras: [1, shape_padded_to_5]
    // For first input: [1, 0, 0, 0, 0]
    // For actual input tensor: [1, d0, d1, d2, d3] where dims are padded to rank 5
    std::vector<uint32_t> extras(5, 0);
    extras[0] = 1;
    if (!is_first) {
        for (uint32_t i = 0; i < 5 && i < input_tensor.rank; i++) {
            extras[1 + i] = static_cast<uint32_t>(input_tensor.dims[i]);
        }
    }

    uint32_t hash = 0x82D28377;
    if (!is_first) hash = 0x9DCAC54A;

    plan.ops.push_back(make_op(
        0, OP_TYPE_COMPUTE, hash, 0x00001003,
        extras, "", "InputNodeSetup"));
}

void Scheduler::gen_input_slice(Plan& plan, const TensorInfo& input_tensor,
                                const TensorInfo& output_tensor) {
    // InputSlice: counter=input_tensor_id, type=compute, f2=2
    // block_ref = 0x1000 | alloc_vtcm_block(input_tensor.tensor_id)
    // Extras: [out_rank, in_rank, out_dim0, in_dim0, out_dim1, in_dim1, out_dim2, in_dim2, 0x00010020]
    uint32_t in_id = input_tensor.tensor_id;
    uint32_t block = alloc_vtcm_block(in_id);

    std::vector<uint32_t> extras;
    uint32_t in_rank = input_tensor.rank;
    uint32_t out_rank = output_tensor.rank;
    extras.push_back(out_rank);
    extras.push_back(in_rank);
    for (uint32_t i = 0; i < out_rank && i < 3; i++) {
        extras.push_back(static_cast<uint32_t>(output_tensor.dims[i]));
        extras.push_back(static_cast<uint32_t>(input_tensor.dims[i]));
    }
    extras.push_back(0x00010020);

    plan.ops.push_back(make_op(
        in_id, OP_TYPE_COMPUTE, 0x02, 0x00001000 | block,
        extras, "*InputSlice@Ff.s4*6.", "InputSlice"));
}

void Scheduler::gen_perm_load(Plan& plan, const TensorInfo& perm_tensor) {
    // Perm load: counter=perm_tensor_id, type=memory, f2=2 or 3
    // block_ref = 0x1000 | alloc_vtcm_block(perm_tensor.tensor_id)
    // Extras: [0, 1]
    uint32_t block = alloc_vtcm_block(perm_tensor.tensor_id);
    uint32_t f2 = (next_step_ < 10) ? 0x02 : 0x03;

    plan.ops.push_back(make_op(
        perm_tensor.tensor_id, OP_TYPE_MEMORY, f2, 0x00001000 | block,
        {0, 1}, "", "perm_load"));
}

void Scheduler::gen_layout_convert(Plan& plan, const TensorInfo& in_tensor,
                                   const TensorInfo& out_tensor) {
    // Layout convert: counter=out_tensor_id, type=memory, f2=2
    // block_ref = 0x1000 | alloc_vtcm_block(out_tensor.tensor_id)
    // Extras: [2, 1]
    uint32_t block = alloc_vtcm_block(out_tensor.tensor_id);

    plan.ops.push_back(make_op(
        out_tensor.tensor_id, OP_TYPE_MEMORY, 0x02, 0x00001000 | block,
        {2, 1}, "", "layout_convert"));
}

void Scheduler::gen_flat_from_vtcm(Plan& plan, const TensorInfo& tensor,
                                   uint32_t f2_phase) {
    // flat_from_vtcm: counter=tensor_id, type=compute, f2=phase
    // block_ref = 0x1000 | alloc_vtcm_block(tensor.tensor_id)
    // Extras depend on phase:
    //   phase 0x11: [3, OUT_TENSOR[tid], 0xCCCC0001, 0x40001000, shape_val, 0x0003xxxx]
    //   phase 0x1A: [3, 3, 0x0003xxxx]
    uint32_t tid = tensor.tensor_id;
    uint32_t block = alloc_vtcm_block(tid);

    std::vector<uint32_t> extras;
    if (f2_phase == 0x11) {
        extras = {3, 0x80000000 | tid, 0xCCCC0001, 0x40001000,
                  static_cast<uint32_t>(tensor.dims[0] * tensor.dims[1] * 0x19),
                  0x00030000 | static_cast<uint32_t>(tensor.dims[1] * 3)};
    } else if (f2_phase == 0x1A) {
        extras = {3, 3, 0x00030010};
    }

    plan.ops.push_back(make_op(
        tid, OP_TYPE_COMPUTE, f2_phase, 0x00001000 | block,
        extras, "flat_from_vtcm@ff.Ff.", "flat_from_vtcm"));
}

void Scheduler::gen_fc_prep(Plan& plan, const TensorInfo& tensor, uint32_t f2_phase) {
    // FC prep: counter=tensor_id, type=compute, f2=phase
    // Extras: [3, OUT_TENSOR[tid+1], 0xCCCC0001, 0x40001000, dim_val, 0x0003xxxx]
    uint32_t tid = tensor.tensor_id;
    uint32_t block = alloc_vtcm_block(tid);

    std::vector<uint32_t> extras = {
        3,
        0x80000000 | (tid + 1),
        0xCCCC0001,
        0x40001000,
        4,
        0x00030008
    };

    plan.ops.push_back(make_op(
        tid, OP_TYPE_COMPUTE, f2_phase, 0x00001000 | block,
        extras, "", "fc_prep"));
}

void Scheduler::gen_dma_set(Plan& plan, const TensorInfo& weight_tensor,
                            uint32_t dma_tag,
                            const std::vector<uint32_t>& dma_extras) {
    // DMA SET: counter=weight_tensor_id, type=DMA, f2=dma_tag
    // block_ref = 0x1000 | alloc_vtcm_block(weight_tensor.tensor_id)
    // Extras: type-specific (see real .bin patterns)
    uint32_t tid = weight_tensor.tensor_id;
    uint32_t block = alloc_vtcm_block(tid);

    plan.ops.push_back(make_op(
        tid, OP_TYPE_DMA, dma_tag, 0x00001000 | block,
        dma_extras, "@DmaCheckpointSet", "DmaCheckpointSet"));
}

void Scheduler::gen_matmul(Plan& plan, const TensorInfo& out_tensor,
                           uint32_t f2_phase, bool with_bias) {
    // MatMul: counter=out_tensor_id (or related), type=compute, f2=phase
    // block_ref = 0 (direct compute, no VTCM block)
    // Extras:
    //   with_bias=false (x·W): [0, OUT_TENSOR[tid], 0xCCCC0001, 0x40001100, 2, 4, 0x00030000]
    //   with_bias=true (+b):   [0, OUT_TENSOR[tid], 0xCCCC0001, 0x40001000, 2, 0x00030004]
    uint32_t tid = out_tensor.tensor_id;

    std::vector<uint32_t> extras;
    if (!with_bias) {
        extras = {0, 0x80000000 | tid, 0xCCCC0001, 0x40001100, 2, 4, 0x00030000};
    } else {
        extras = {0, 0x80000000 | tid, 0xCCCC0001, 0x40001000, 2, 0x00030004};
    }

    plan.ops.push_back(make_op(
        tid, OP_TYPE_COMPUTE, f2_phase, 0x00000000,
        extras, "MatMul_bias@ff*4", with_bias ? "MatMul_bias(+b)" : "MatMul_bias(xW)"));
}

void Scheduler::gen_output_node_setup(Plan& plan, const TensorInfo& output_tensor) {
    // Output node setup: counter=0 (graph_node), type=compute, f2=hash
    // Extras: [1, shape_padded_to_5]
    std::vector<uint32_t> extras(5, 0);
    extras[0] = 1;
    for (uint32_t i = 0; i < 5 && i < output_tensor.rank; i++) {
        extras[1 + i] = static_cast<uint32_t>(output_tensor.dims[i]);
    }

    plan.ops.push_back(make_op(
        0, OP_TYPE_COMPUTE, 0x9BCA6304, 0x00001003,
        extras, "", "OutputNodeSetup"));
}

void Scheduler::gen_sync(Plan& plan, const TensorInfo& sync_tensor,
                         const std::vector<uint32_t>& sync_extras) {
    // SyncOp: counter=sync_tensor_id, type=sync, f2=3
    uint32_t block = alloc_vtcm_block(sync_tensor.tensor_id);

    plan.ops.push_back(make_op(
        sync_tensor.tensor_id, OP_TYPE_SYNC, 0x03, 0x00001000 | block,
        sync_extras, "@SyncOp", "SyncOp"));
}

void Scheduler::gen_transpose_impl(Plan& plan, const TensorInfo& in_tensor,
                                   const TensorInfo& out_tensor) {
    // Transpose_impl: counter=in_tensor_id, type=compute, f2=3
    uint32_t block = alloc_vtcm_block(in_tensor.tensor_id);

    plan.ops.push_back(make_op(
        in_tensor.tensor_id, OP_TYPE_COMPUTE, 0x03, 0x00001000 | block,
        {1}, "Transpose_impl@ff*2.fi.t", "Transpose_impl"));
}

void Scheduler::gen_output_slice(Plan& plan, const TensorInfo& output_tensor) {
    // OutputSlice: counter=output_tensor_id, type=compute, f2=0
    uint32_t block = alloc_vtcm_block(output_tensor.tensor_id);

    plan.ops.push_back(make_op(
        output_tensor.tensor_id, OP_TYPE_COMPUTE, 0x00, 0x00001000 | block,
        {0}, "*OutputSlice@ff.s4*3.", "OutputSlice"));
}

// === General scheduling from ModelDesc ===

Scheduler::Plan Scheduler::schedule_general(const ModelDesc& model) {
    Plan plan;
    next_step_ = 0;
    next_dma_tag_ = 0x11;
    next_vtcm_block_ = 0x19;
    tensor_block_map_.clear();

    plan.op_names = model.op_names;
    plan.tensor_names = model.tensor_names;
    plan.kernel_names = kernel_sel_.get_kernel_table();

    // Classify model type
    bool is_standalone_transpose = (model.ops.size() == 1 &&
                                   model.ops[0].type == HtpOpType::Transpose);
    bool is_standalone_reshape = (model.ops.size() == 1 &&
                                  model.ops[0].type == HtpOpType::Reshape);
    uint32_t fc_count = 0;
    for (const auto& op : model.ops) {
        if (op.type == HtpOpType::FullyConnected) fc_count++;
    }

    // Check if Transpose precedes FC
    bool has_transpose_before_fc = false;
    for (size_t i = 0; i < model.ops.size(); i++) {
        if (model.ops[i].type == HtpOpType::Transpose) {
            for (size_t j = i + 1; j < model.ops.size(); j++) {
                if (model.ops[j].type == HtpOpType::FullyConnected) {
                    has_transpose_before_fc = true;
                    break;
                }
            }
        }
    }
    if (is_standalone_transpose) has_transpose_before_fc = false;

    auto mk = [&](uint32_t tid, OpRecordType type, uint32_t f2,
                  uint32_t block_ref, std::vector<uint32_t> extras,
                  const char* kernel, const char* step) {
        ScheduledOp op;
        op.record_id = compute_record_id(next_step_++);
        op.tensor_id = tid;
        op.type = type;
        op.f2 = f2;
        op.block_ref = block_ref;
        op.extras = std::move(extras);
        op.kernel_name = kernel;
        op.step_name = step;
        plan.ops.push_back(std::move(op));
    };

    // Calculate InputSlice base block
    // Formula: IS = 0x15 + 5 * ((n_dma + 1) // 2) + 2 * has_transpose_before_fc
    // n_dma = actual number of DMA operations in the plan
    uint32_t n_dma = 0;
    bool prev_was_fc = false;
    for (size_t i = 0; i < model.ops.size(); i++) {
        const auto& op = model.ops[i];
        if (op.type == HtpOpType::FullyConnected) {
            if (!prev_was_fc) n_dma += 1;  // W DMA (first FC or after non-FC)
            bool has_reshape_after = (i + 1 < model.ops.size() &&
                                     model.ops[i+1].type == HtpOpType::Reshape);
            bool has_fc_after = (i + 1 < model.ops.size() &&
                                model.ops[i+1].type == HtpOpType::FullyConnected);
            if (has_reshape_after) {
                n_dma += 2;  // out_fc + out_ncf
            } else if (has_fc_after) {
                n_dma += 1;  // inter DMA (loads W2)
            } else {
                n_dma += 1;  // out DMA
            }
            if (has_transpose_before_fc && !prev_was_fc) n_dma += 1;  // b DMA
            prev_was_fc = true;
        } else {
            if (op.type == HtpOpType::Transpose && is_standalone_transpose) {
                n_dma += 1;  // trans output DMA
            }
            prev_was_fc = false;
        }
    }

    uint32_t is_base = 0x15 + 5 * ((n_dma + 1) / 2) + 2 * (has_transpose_before_fc ? 1 : 0);

    // perm_in_delta: 6(0 DMA), 9(1-2 DMA), 10(3 DMA), 15(4+ DMA)
    uint32_t perm_in_delta = 6;
    if (n_dma > 0) perm_in_delta += 3;
    if (n_dma >= 3) perm_in_delta += 1;
    if (n_dma >= 4) perm_in_delta += 5;

    // Helper: block from IS offset
    auto blk_off = [&](int32_t offset) -> uint32_t {
        return 0x00001000 | static_cast<uint32_t>(static_cast<int32_t>(is_base) + offset);
    };

    uint32_t blk_is    = blk_off(0);            // InputSlice
    uint32_t blk_perm_in = blk_off(perm_in_delta);  // perm_load_in
    uint32_t blk_sync  = blk_off(1);             // SyncOp = IS + 1
    uint32_t blk_perm_out = blk_off(perm_in_delta + 1);  // perm_load_out = max
    uint32_t blk_trans_out = blk_off(perm_in_delta + 1 - 3);  // Transpose_impl = perm_out - 3
    uint32_t blk_out_slice = blk_off(perm_in_delta + 1 - 3 + 1);  // OutputSlice = trans_out + 1

    bool has_input_transpose = false;
    bool has_dma = false;
    for (const auto& op : model.ops) {
        if (op.type == HtpOpType::Transpose) has_input_transpose = true;
        if (op.type == HtpOpType::FullyConnected) has_dma = true;
        if (op.type == HtpOpType::Transpose) has_dma = true;
    }

    uint32_t input_flag = has_input_transpose ? 0x00010020 : 0x00010000;

    // Step 0: InputNodeSetup null
    mk(0, OP_TYPE_COMPUTE, 0x82D28377, 0x00001003, {1, 0, 0, 0, 0}, "", "InputNodeSetup");

    // Step 1: InputNodeSetup shape
    // extras = [1, 1, 1, input_dim0, input_dim1] (NCHW format with channel=1)
    uint32_t in_hash = htp_shape_hash({1, model.input_batch, model.input_dim0, model.input_dim1});
    mk(0, OP_TYPE_COMPUTE, in_hash, 0x00001003,
       {1, 1, 1, model.input_dim0, model.input_dim1}, "", "InputNodeSetup");

    // Step 2: InputSlice (fixed extras)
    mk(model.input_tid, OP_TYPE_COMPUTE, 0x02, blk_is,
       {3, 3, 4, 3, 4, 4, 2, 1, input_flag},
       "*InputSlice@Ff.s4*6.", "InputSlice");

    // Step 3: perm_load (input side)
    mk(model.perm_in_tid, OP_TYPE_MEMORY, 0x02, blk_perm_in,
       {0, 1}, "", "perm_load");

    // Determine if we need layout_convert (Transpose or FC present)
    bool need_layout_convert = false;
    for (const auto& op : model.ops) {
        if (op.type == HtpOpType::Transpose || op.type == HtpOpType::FullyConnected) {
            need_layout_convert = true;
            break;
        }
    }

    // Step 4: layout_convert (if needed)
    // layout_convert offset: IS + (perm_in_delta - 1) for reshape,
    //                         IS + (perm_in_delta - 4) for trans/fc, etc.
    // Pattern: layout = perm_in - (perm_in_delta - layout_offset)
    // layout_offset: reshape=3, trans=5, fc=4, two_fc=4, simple=8
    uint32_t layout_tid = 3;
    if (need_layout_convert) {
        // layout_convert block = IS + (perm_in_delta - delta2)
        // delta2 varies: trans=4, fc=5, two_fc=6, simple=7
        int32_t layout_off;
        if (is_standalone_transpose) layout_off = 5;
        else if (has_transpose_before_fc) layout_off = 8;
        else if (fc_count >= 2) layout_off = 4;
        else layout_off = 4;  // fc_only
        mk(layout_tid, OP_TYPE_MEMORY, 0x02, blk_off(layout_off), {2, 1}, "", "layout_convert");
    }

    // Op-specific steps
    // (has_transpose_before_fc, fc_count, is_standalone_* already computed above)

    uint32_t compute_base = has_transpose_before_fc ? 0x14 : 0x10;
    uint32_t dma_base = compute_base + fc_count * 2;
    uint32_t dma_counter = 2;  // First field in DMA extras, increments per DMA

    uint32_t fc_index = 0;

    for (size_t i = 0; i < model.ops.size(); i++) {
        const auto& op = model.ops[i];

        // Check context for this op
        bool has_reshape_after = (i + 1 < model.ops.size() &&
                                 model.ops[i+1].type == HtpOpType::Reshape);
        bool has_fc_after = (i + 1 < model.ops.size() &&
                             model.ops[i+1].type == HtpOpType::FullyConnected);

        if (op.type == HtpOpType::Reshape) {
            if (is_standalone_reshape) {
                // reshape_only: simple reshape (2 steps)
                // reshape_prep at IS+3, reshape_compute at IS+1
                // reshape_compute extras[0] = model.output_tid (final output)
                mk(op.input_tid, OP_TYPE_COMPUTE, 0x02, blk_off(3), {1}, "", "reshape_prep");
                mk(op.output_tid, OP_TYPE_COMPUTE, 0x03, blk_off(1),
                   {model.output_tid, (uint32_t)4, (uint32_t)3, (uint32_t)3}, "", "reshape_compute");
            } else if (has_fc_after) {
                // Reshape before FC (pre_reshape): flat_from_vtcm + fc_prep (2 steps)
                // flat at IS+2, fc_prep at IS-7 (for simple) or IS-5 (for trans)
                int32_t fc_prep_off = has_transpose_before_fc ? -7 : -5;
                mk(op.output_tid, OP_TYPE_COMPUTE, 0x11, blk_off(2),
                   {3, 0x80000000u | (op.output_tid - 1), 0xCCCC0001, 0x40001000, 0x19, 0x0003000C},
                   "flat_from_vtcm@ff.Ff.", "flat_from_vtcm");
                mk(op.output_tid, OP_TYPE_COMPUTE, 0x12, blk_off(fc_prep_off),
                   {3, 0x80000000u | op.output_tid, 0xCCCC0001, 0x40001000, 4, 0x00030008},
                   "", "fc_prep");
            } else {
                // Reshape after FC (post_reshape): flat_from_vtcm + DMA out_ncf (2 steps)
                // flat_out at IS+4, DMA_out_ncf at IS+5
                mk(op.input_tid, OP_TYPE_COMPUTE, 0x1A, blk_off(4),
                   {3, 3, 0x00030010}, "flat_from_vtcm@ff.Ff.", "flat_from_vtcm");
                uint32_t dc = dma_counter++;
                uint32_t mot = model.output_tid;
                mk(op.output_tid, OP_TYPE_DMA, 0x1A, blk_off(5),
                   {dc, dc + 1, mot + 2, mot + 3, mot - 3, 0, 2, 0x00020004},
                   "@DmaCheckpointSet", "DmaCheckpointSet(out_ncf)");
            }
        }
        else if (op.type == HtpOpType::Transpose) {
            if (is_standalone_transpose) {
                // Transpose-only: flat_from_vtcm + fc_prep + DMA (3 steps)
                // flat at IS+2, fc_prep at IS-5, DMA at IS+3
                // extras reference model.input_ncf_tid (not op.output_tid)
                mk(op.output_tid, OP_TYPE_COMPUTE, 0x11, blk_off(2),
                   {3, 0x80000000u | model.input_ncf_tid, 0xCCCC0001, 0x40001000, 0x19, 0x00020004},
                   "flat_from_vtcm@ff.Ff.", "flat_from_vtcm");
                mk(op.output_tid, OP_TYPE_COMPUTE, 0x12, blk_off(-5),
                   {3, 0x80000000u | op.output_tid, 0xCCCC0001, 0x40001000, 4, 0x00020000},
                   "", "fc_prep");
                uint32_t dc = dma_counter++;
                uint32_t dma_tid = op.output_tid + 1;
                mk(dma_tid, OP_TYPE_DMA, 0x11, blk_off(3),
                   {dc, dc + 1, dma_tid, dma_tid + 1, dma_tid + 2, 2, 2, 0x00010000},
                   "@DmaCheckpointSet", "DmaCheckpointSet");
            }
            // If not standalone, Transpose's layout_convert was already added above
        }
        else if (op.type == HtpOpType::FullyConnected) {
            uint32_t w_tid = op.weight_tid;
            uint32_t b_tid = op.bias_tid;
            uint32_t out_tid = op.output_tid;
            uint32_t w_m = op.weight_m;
            uint32_t w_k = op.weight_k;

            bool prev_is_fc = (i > 0 && model.ops[i-1].type == HtpOpType::FullyConnected);

            uint32_t matmul_phase = compute_base + fc_index * 2;

            // DMA W: skip if previous op was FC (W loaded with inter DMA)
            if (!prev_is_fc) {
                uint32_t dc = dma_counter++;
                if (has_transpose_before_fc) {
                    // Large DMA format for W (at IS+3) + DMA for b (at IS+6)
                    // extras[8,9] = input_dim1, input_dim0 (shape-dependent)
                    mk(w_tid, OP_TYPE_DMA, 0x11, blk_off(3),
                       {dc, dc + 1, w_tid, b_tid, out_tid, 2, 0x80000000u | w_tid, 0x40001100,
                        model.input_dim1, model.input_dim0, 0x00010000},
                       "@DmaCheckpointSet", "DmaCheckpointSet(W)");
                    uint32_t dc2 = dma_counter++;
                    mk(b_tid, OP_TYPE_DMA, 0x11, blk_off(6),
                       {dc2, dc2 + 1, b_tid + 2, 0, 5, 0x00020004},
                       "@DmaCheckpointSet", "DmaCheckpointSet(b)");
                } else {
                    // Small DMA format for W only (fc_only at IS+2, two_fc at IS+2)
                    mk(w_tid, OP_TYPE_DMA, 0x02, blk_off(2),
                       {dc, dc + 1, w_tid + 1, 0, 1, 0x00020004},
                       "@DmaCheckpointSet", "DmaCheckpointSet(W)");
                }
            }

            // MatMul x·W — no block (compute in-place)
            // extras[1] = 0x80000000 | activation_tid (the tensor that holds the matmul input)
            // For simple_linear: activation_tid = out_tid - 1 (pre_reshape)
            // For fc_only/two_fc 1st: activation_tid = model.input_ncf_tid
            // For two_fc 2nd: activation_tid = op.act_ref_tid (inter DMA output)
            uint32_t act_tid;
            if (op.act_ref_tid != 0) {
                act_tid = op.act_ref_tid;
            } else if (has_transpose_before_fc) {
                act_tid = out_tid - 1;
            } else {
                act_tid = model.input_ncf_tid;
            }
            // MatMul x·W — last extra: 0x00030000 (1st FC), 0x00030008 (2nd FC)
            // extras[4] = w_m (output dim), extras[5] = K dim
            // K dim = input_dim0 (with transpose) or w_k (without transpose)
            uint32_t k_dim = has_transpose_before_fc ? model.input_dim0 : w_k;
            uint32_t matmul_last = prev_is_fc ? 0x00030008 : 0x00030000;
            mk(op.input_tid, OP_TYPE_COMPUTE, matmul_phase, 0x00000000,
               {0, 0x80000000u | act_tid, 0xCCCC0001, 0x40001100, w_m, k_dim, matmul_last},
               "MatMul_bias@ff*4", "MatMul_bias(xW)");

            // MatMul +b — no block
            uint32_t bias_tid_extras;
            if (has_transpose_before_fc) {
                bias_tid_extras = out_tid;
            } else if (prev_is_fc) {
                bias_tid_extras = out_tid + 1;  // 2nd FC: result goes to out_tid+1
            } else {
                bias_tid_extras = op.weight_tid;  // 1st FC: result reuses W slot
            }
            // Last extra: 0x0003000C (2nd FC), 0x00030004 (1st FC/simple)
            uint32_t bias_last = prev_is_fc ? 0x0003000C : 0x00030004;
            mk(op.input_tid, OP_TYPE_COMPUTE, matmul_phase + 1, 0x00000000,
               {0, 0x80000000u | bias_tid_extras, 0xCCCC0001, 0x40001000, w_m, bias_last},
               "MatMul_bias@ff*4", "MatMul_bias(+b)");

            // DMA output
            uint32_t dma_tag = dma_base + fc_index * 2;
            if (has_reshape_after) {
                // Large DMA output (out_fc) at IS-8
                // extras reference model.output_tid: [dc, dc+1, out_tid-1, out_tid, out_tid+1, 0, 0x80000000|(out_tid-2), ...]
                // extras[8] = input_dim1, extras[9] = w_m (output dim)
                uint32_t dc = dma_counter++;
                uint32_t mot = model.output_tid;
                mk(out_tid, OP_TYPE_DMA, dma_tag, blk_off(-8),
                   {dc, dc + 1, mot - 1, mot, mot + 1, 0, 0x80000000u | (mot - 2), 0x40001100,
                    model.input_dim1, w_m, 0x00020008},
                   "@DmaCheckpointSet", "DmaCheckpointSet(out_fc)");
            } else if (has_fc_after) {
                // DMA inter at IS-2
                // extras reference op.input_tid (fc_act, the intermediate result)
                uint32_t dc = dma_counter++;
                mk(out_tid, OP_TYPE_DMA, dma_tag, blk_off(-2),
                   {dc, dc + 1, out_tid, out_tid + 1, out_tid + 2, 0, 0x80000000u | op.input_tid, 0x40001100, 3, 2, 0x00020008},
                   "@DmaCheckpointSet", "DmaCheckpointSet(inter)");
            } else {
                // Small DMA output at IS-1
                uint32_t dc = dma_counter++;
                uint32_t dma_extras_tid = prev_is_fc ? model.output_tid + 1 : out_tid;
                // Last extra: 0x00020004 (2nd FC), 0x00020008 (1st FC)
                uint32_t dma_last = prev_is_fc ? 0x00020004 : 0x00020008;
                mk(out_tid, OP_TYPE_DMA, dma_tag, blk_off(-1),
                   {dc, dc + 1, dma_extras_tid, dma_extras_tid + 1, dma_extras_tid + 2, 0, 2, dma_last},
                   "@DmaCheckpointSet", "DmaCheckpointSet(out)");
            }

            fc_index++;
        }
    }

    // Output tail
    if (has_dma) {
        // OutputNodeSetup
        // extras = [1, 1, 1, output_dim0, output_dim1] (NCHW format with channel=1)
        uint32_t out_hash = htp_shape_hash({1, model.output_batch, model.output_dim0, model.output_dim1});
        mk(0, OP_TYPE_COMPUTE, out_hash, 0x00001003,
           {1, 1, 1, model.output_dim0, model.output_dim1}, "", "OutputNodeSetup");

        // SyncOp at IS+1
        // extras = [n_dma + 2, n_steps - 5, n_steps - 4, 3, 3]
        uint32_t n_steps = static_cast<uint32_t>(plan.ops.size()) + 4;  // +4 for remaining steps
        mk(model.sync_tid, OP_TYPE_SYNC, 0x03, blk_sync,
           {n_dma + 2, n_steps - 5, n_steps - 4, 3, 3}, "@SyncOp", "SyncOp");
    }

    // perm_load (output side) at perm_out = max_block
    mk(model.perm_in_tid, OP_TYPE_MEMORY, 0x03, blk_perm_out,
       {0, 1}, "", "perm_load");

    // Transpose_impl (output side) at perm_out - 3
    uint32_t trans_tid = 3;
    mk(trans_tid, OP_TYPE_COMPUTE, 0x03, blk_trans_out,
       {1}, "Transpose_impl@ff*2.fi.t", "Transpose_impl");

    // OutputSlice at trans_out + 1
    mk(model.output_tid, OP_TYPE_COMPUTE, 0x00, blk_out_slice,
       {0}, "*OutputSlice@ff.s4*3.", "OutputSlice");

    return plan;
}

// === Main scheduling for simple_linear ===

Scheduler::Plan Scheduler::schedule_simple_linear(
    const std::vector<TensorInfo>& tensors,
    const std::vector<std::string>& op_names) {
    Plan plan;
    next_step_ = 0;
    next_dma_tag_ = 0x11;
    next_vtcm_block_ = 0x19;
    tensor_block_map_.clear();

    plan.tensors = tensors;
    plan.op_names = op_names;
    plan.kernel_names = kernel_sel_.get_kernel_table();

    // Block indices from real .bin VTCM allocation (verified by byte analysis).
    // These are model-specific; will be computed by VTCM allocator in future.
    auto blk = [](uint32_t idx) { return 0x00001000 | idx; };

    // Helper to create op with exact values
    auto mk = [&](uint32_t tid, OpRecordType type, uint32_t f2,
                  uint32_t block_ref, std::vector<uint32_t> extras,
                  const char* kernel, const char* step) {
        ScheduledOp op;
        op.record_id = compute_record_id(next_step_++);
        op.tensor_id = tid;
        op.type = type;
        op.f2 = f2;
        op.block_ref = block_ref;
        op.extras = std::move(extras);
        op.kernel_name = kernel;
        op.step_name = step;
        plan.ops.push_back(std::move(op));
    };

    // 19 steps from real simple_linear_context.bin byte analysis:
    //
    // Step  cnt  type     f2         block_ref   extras
    //   0    0   compute  0x82D28377 0x00001003  [1,0,0,0,0]
    //   1    0   compute  0x9DCAC54A 0x00001003  [1,1,1,4,3]
    //   2    1   compute  0x02       0x00001021  [3,3,4,3,4,4,2,1,0x00010020]
    //   3    2   memory   0x02       0x00001030  [0,1]
    //   4    3   memory   0x02       0x00001029  [2,1]
    //   5    4   compute  0x11       0x00001023  [3,0x80000003,0xCCCC0001,0x40001000,0x19,0x0003000C]
    //   6    4   compute  0x12       0x0000101A  [3,0x80000004,0xCCCC0001,0x40001000,4,0x00030008]
    //   7    5   DMA      0x11       0x00001024  [2,3,5,6,7,2,0x80000005,0x40001100,3,4,0x00010000]
    //   8    6   DMA      0x11       0x00001027  [3,4,8,0,5,0x00020004]
    //   9    4   compute  0x14       0x00000000  [0,0x80000006,0xCCCC0001,0x40001100,2,4,0x00030000]
    //  10    4   compute  0x15       0x00000000  [0,0x80000007,0xCCCC0001,0x40001000,2,0x00030004]
    //  11    7   DMA      0x16       0x00001019  [4,5,9,10,11,0,0x80000008,0x40001100,3,2,0x00020008]
    //  12    4   compute  0x1A       0x00001025  [3,3,0x00030010]
    //  13    8   DMA      0x1A       0x00001026  [5,6,12,13,7,0,2,0x00020004]
    //  14    0   compute  0x9BCA6304 0x00001003  [1,1,1,2,3]
    //  15    9   sync     0x03       0x00001022  [6,14,15,3,3]
    //  16    2   memory   0x03       0x00001031  [0,1]
    //  17    3   compute  0x03       0x0000102E  [1]
    //  18   10   compute  0x00       0x0000102F  [0]

    mk(0, OP_TYPE_COMPUTE, 0x82D28377, 0x00001003, {1,0,0,0,0}, "", "InputNodeSetup");
    mk(0, OP_TYPE_COMPUTE, 0x9DCAC54A, 0x00001003, {1,1,1,4,3}, "", "InputNodeSetup");
    mk(1, OP_TYPE_COMPUTE, 0x02, blk(0x21), {3,3,4,3,4,4,2,1,0x00010020}, "*InputSlice@Ff.s4*6.", "InputSlice");
    mk(2, OP_TYPE_MEMORY, 0x02, blk(0x30), {0,1}, "", "perm_load");
    mk(3, OP_TYPE_MEMORY, 0x02, blk(0x29), {2,1}, "", "layout_convert");
    mk(4, OP_TYPE_COMPUTE, 0x11, blk(0x23), {3,0x80000003,0xCCCC0001,0x40001000,0x19,0x0003000C}, "flat_from_vtcm@ff.Ff.", "flat_from_vtcm");
    mk(4, OP_TYPE_COMPUTE, 0x12, blk(0x1A), {3,0x80000004,0xCCCC0001,0x40001000,4,0x00030008}, "", "fc_prep");
    mk(5, OP_TYPE_DMA, 0x11, blk(0x24), {2,3,5,6,7,2,0x80000005,0x40001100,3,4,0x00010000}, "@DmaCheckpointSet", "DmaCheckpointSet(W)");
    mk(6, OP_TYPE_DMA, 0x11, blk(0x27), {3,4,8,0,5,0x00020004}, "@DmaCheckpointSet", "DmaCheckpointSet(b)");
    mk(4, OP_TYPE_COMPUTE, 0x14, 0x00000000, {0,0x80000006,0xCCCC0001,0x40001100,2,4,0x00030000}, "MatMul_bias@ff*4", "MatMul_bias(xW)");
    mk(4, OP_TYPE_COMPUTE, 0x15, 0x00000000, {0,0x80000007,0xCCCC0001,0x40001000,2,0x00030004}, "MatMul_bias@ff*4", "MatMul_bias(+b)");
    mk(7, OP_TYPE_DMA, 0x16, blk(0x19), {4,5,9,10,11,0,0x80000008,0x40001100,3,2,0x00020008}, "@DmaCheckpointSet", "DmaCheckpointSet(out_fc)");
    mk(4, OP_TYPE_COMPUTE, 0x1A, blk(0x25), {3,3,0x00030010}, "flat_from_vtcm@ff.Ff.", "flat_from_vtcm");
    mk(8, OP_TYPE_DMA, 0x1A, blk(0x26), {5,6,12,13,7,0,2,0x00020004}, "@DmaCheckpointSet", "DmaCheckpointSet(out_ncf)");
    mk(0, OP_TYPE_COMPUTE, 0x9BCA6304, 0x00001003, {1,1,1,2,3}, "", "OutputNodeSetup");
    mk(9, OP_TYPE_SYNC, 0x03, blk(0x22), {6,14,15,3,3}, "@SyncOp", "SyncOp");
    mk(2, OP_TYPE_MEMORY, 0x03, blk(0x31), {0,1}, "", "perm_load");
    mk(3, OP_TYPE_COMPUTE, 0x03, blk(0x2E), {1}, "Transpose_impl@ff*2.fi.t", "Transpose_impl");
    mk(10, OP_TYPE_COMPUTE, 0x00, blk(0x2F), {0}, "*OutputSlice@ff.s4*3.", "OutputSlice");

    plan.tensor_names = {"output", "input"};
    return plan;
}

Scheduler::Plan Scheduler::schedule(const GraphPrepare& gp) {
    // 通用路径: ST-Cut 调度链(M31)产执行序; plan.ops 每条 = 一个图 op。
    Plan plan;

    // 1. 图结构清单(tensors 沿 gp 序 —— CP spill/fill 对齐依赖此走序)
    std::vector<TensorInfo> tensors;
    std::vector<std::string> op_names;
    const auto& ordering = gp.get_ordering();
    for (size_t i = 0; i < ordering.size(); i++) {
        auto* opdef = gp.get_op_at(ordering[i]);
        if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;

        std::string name = opdef->name_tag ? (opdef->name_tag->name() ? opdef->name_tag->name() : "") : "";
        if (!name.empty()) op_names.push_back(name);

        TensorInfo ti{};
        ti.tensor_id = static_cast<uint32_t>(opdef->op_id);
        ti.name = name;
        ti.rank = opdef->output_def.rank;
        for (uint32_t j = 0; j < 5 && j < ti.rank; j++) {
            ti.dims[j] = opdef->output_def.dims[j];
        }
        ti.dtype = static_cast<DType>(opdef->output_def.dtype);
        ti.is_const = opdef->is_const();
        ti.is_graph_io = (name == "Input" || name == "Output");
        tensors.push_back(ti);
    }

    // 2. dense→op_id 逆映射: 与 build_stcut_input_from_graph 完全相同的走序
    //    (ordering 中 enabled 且未死的 op 依次编号 0..N-1)
    std::vector<op_id_t> dense_to_id;
    for (op_id_t id : ordering) {
        const OpDef* od = gp.get_op_at(id);
        if (!od || !od->is_enabled() || od->is_dead()) continue;
        dense_to_id.push_back(id);
    }

    // 3. ST-Cut 链(M31)产 dense 执行序 → 译回 op_id
    StCutGraphInput stin;
    std::vector<uint32_t> best_dense;
    std::vector<uint64_t> flows, cycles;
    bool have_stcut = build_stcut_input_from_graph(gp, stin) && stin.node_count > 0;
    if (have_stcut) {
        StCutOptions opt{};
        opt.rt = 3; opt.it = 50; opt.rg = 20; opt.am = 0;
        opt.budget_base = 1ull << 40; opt.tr = 1.0;
        run_stcut_schedule(gp, opt, best_dense, flows, cycles);
    }
    std::vector<uint32_t> order_ids;
    for (uint32_t d : best_dense)
        if (d < dense_to_id.size()) order_ids.push_back(static_cast<uint32_t>(dense_to_id[d]));
    // 4. 校验/兜底: 必须 1:1 覆盖全部节点; 否则确定序(ordering)兜底
    {
        std::set<uint32_t> uniq(order_ids.begin(), order_ids.end());
        if (!have_stcut || best_dense.empty() || order_ids.size() != dense_to_id.size() ||
            uniq.size() != dense_to_id.size()) {
            order_ids.clear();
            for (op_id_t id : dense_to_id) order_ids.push_back(static_cast<uint32_t>(id));
        }
    }
    plan.op_order = order_ids;

    // 4. plan.ops: 每条 = 一个图 op(按 tensors 走序; record_id 按位次;
    //    block_ref 暂 0 —— M37 phys_alloc 接真后回填)
    for (size_t i = 0; i < tensors.size(); i++) {
        ScheduledOp op;
        op.record_id = compute_record_id(static_cast<uint32_t>(i));
        op.tensor_id = tensors[i].tensor_id;
        op.type = tensors[i].is_const ? OP_TYPE_MEMORY : OP_TYPE_COMPUTE;
        op.f2 = 0;
        op.block_ref = 0;
        op.step_name = tensors[i].name;
        plan.ops.push_back(op);
    }
    plan.op_names = op_names;
    for (const auto& t : tensors) plan.tensor_names.push_back(t.name);

    // 5. Phase B: CP spill/fill
    apply_cp_spill_fill(plan, gp);
    return plan;
}

// 路径A 金样重放(冻结): 19 步硬编码重放, 仅服务 ContextBinaryWriter 字节对拍。
Scheduler::Plan Scheduler::schedule_path_a_replay(const GraphPrepare& gp) {
    // Extract tensor info from the graph
    std::vector<TensorInfo> tensors;
    std::vector<std::string> op_names;

    // Walk the graph's opdef map to build tensor info
    // For now, use the simple_linear pattern directly
    // (will be generalized later)

    // Build tensor list from graph
    const auto& ordering = gp.get_ordering();
    for (size_t i = 0; i < ordering.size(); i++) {
        auto* opdef = gp.get_op_at(ordering[i]);
        if (!opdef || !opdef->is_enabled() || opdef->is_dead()) continue;

        std::string name = opdef->name_tag ? (opdef->name_tag->name() ? opdef->name_tag->name() : "") : "";
        if (!name.empty()) op_names.push_back(name);

        TensorInfo ti{};
        ti.tensor_id = static_cast<uint32_t>(opdef->op_id);
        ti.name = name;
        ti.rank = opdef->output_def.rank;
        for (uint32_t j = 0; j < 5 && j < ti.rank; j++) {
            ti.dims[j] = opdef->output_def.dims[j];
        }
        ti.dtype = static_cast<DType>(opdef->output_def.dtype);
        ti.is_const = opdef->is_const();
        ti.is_graph_io = (name == "Input" || name == "Output");
        tensors.push_back(ti);
    }

    Plan plan = schedule_simple_linear(tensors, op_names);
    // Phase B: CP-mode plans gain their spill/fill DMA records here;
    // non-CP graphs are a no-op (zero insertion).
    apply_cp_spill_fill(plan, gp);
    return plan;
}

// ── Phase B (serialization level): CP spill/fill → DMA ScheduledOps ──
//
// For each GraphPrepare CP spill/fill plan insert two OP_TYPE_DMA records:
//   spill: right after the last step of the last seg0 consumer
//   fill:  right before the first step of the first seg1 consumer
// (positions come from CP slot→topo mapping; consumers are identified by
// tensor_id — the same id space Plan.ops and graph ops share).
// extras layout is TENTATIVE: {ddr_offset, vtcm_offset, size} as three LE
// u32 words, pending golden .bin verification (G-series experiments). The
// DDR word is shared by both ends (Eq.9 same-address contract).
// After all insertions every record_id is recomputed from its final step
// index, keeping the (odd,even)-pair pattern valid.
void Scheduler::apply_cp_spill_fill(Plan& plan, const GraphPrepare& gp) {
    if (!gp.is_cp_allocator_active()) return;
    const auto& cp_plans = gp.get_cp_spill_fill_plans();
    if (cp_plans.empty()) return;

    // plan.tensors was built by walking gp's ordering, so index t addresses
    // the same node as CP's topo index t. Disabled/dead nodes skipped by the
    // walk would shift the alignment — guarded by the bounds checks below
    // (out-of-range anchors fall back to plan head/tail deterministically).
    const size_t n_tensors = plan.tensors.size();

    // Step-index anchors: first/last step currently referencing a tensor.
    auto first_step_of = [&plan](uint32_t tid) -> int {
        for (size_t j = 0; j < plan.ops.size(); j++)
            if (plan.ops[j].tensor_id == tid) return static_cast<int>(j);
        return -1;
    };
    auto last_step_of = [&plan](uint32_t tid) -> int {
        for (int j = static_cast<int>(plan.ops.size()) - 1; j >= 0; j--)
            if (plan.ops[j].tensor_id == tid) return j;
        return -1;
    };

    struct Insertion {
        size_t at;
        bool is_fill;
        ScheduledOp op;
    };
    std::vector<Insertion> ins;
    const auto& allocs = gp.get_vtcm_allocations();

    for (const auto& pl : cp_plans) {
        if (pl.size == 0) continue;
        // VTCM block for the paged tensor: CP AllocResult block_id when the
        // allocation channel carries one, else a fresh scheduler block.
        uint32_t block;
        auto it = allocs.find(pl.op_id);
        if (it != allocs.end() && !it->second.spilled)
            block = it->second.block_id;
        else
            block = alloc_vtcm_block(static_cast<uint32_t>(pl.op_id));
        const uint32_t block_ref = 0x00001000u | (block & 0xFFFu);

        // DMA tags: the hardcoded simple_linear pattern uses tags up to 0x1A
        // without advancing next_dma_tag_, so keep CP tags clear of that
        // range even right after a counter reset.
        uint32_t tag = std::max(next_dma_tag_, 0x20u);
        next_dma_tag_ = tag + 1;
        uint32_t tag2 = std::max(next_dma_tag_, 0x20u);
        next_dma_tag_ = tag2 + 1;

        // TENTATIVE extras (see comment above): {ddr, vtcm, size}.
        // Spill reads VTCM at the seg0 offset, fill writes the seg1 offset;
        // both ends share the DDR address.
        ScheduledOp spill;
        spill.record_id = 0; // renumbered below
        spill.tensor_id = static_cast<uint32_t>(pl.op_id);
        spill.type = OP_TYPE_DMA;
        spill.f2 = tag;
        spill.block_ref = block_ref;
        spill.extras = {static_cast<uint32_t>(pl.ddr_offset),
                        static_cast<uint32_t>(pl.vtcm_offset_spill),
                        static_cast<uint32_t>(pl.size)};
        spill.kernel_name = "@DmaCheckpointSet";
        spill.step_name = "CpSpill";

        ScheduledOp fill = spill;
        fill.f2 = tag2;
        fill.extras = {static_cast<uint32_t>(pl.ddr_offset),
                       static_cast<uint32_t>(pl.vtcm_offset),
                       static_cast<uint32_t>(pl.size)};
        fill.step_name = "CpFill";

        // Anchor to the consumer tensors' step ranges (fill consumer is
        // strictly later in topo, so fill_at > spill_at holds whenever both
        // anchors resolve; degenerate anchors keep the pair ordered anyway).
        int spill_at = (pl.spill_position < n_tensors)
                           ? last_step_of(plan.tensors[pl.spill_position].tensor_id)
                           : -1;
        int fill_at = (pl.fill_position < n_tensors)
                          ? first_step_of(plan.tensors[pl.fill_position].tensor_id)
                          : -1;
        if (spill_at < 0) spill_at = 0; // degenerate anchor: plan head
        if (fill_at < 0) fill_at = static_cast<int>(plan.ops.size()); // tail
        if (fill_at <= spill_at) fill_at = spill_at + 1;

        ins.push_back({static_cast<size_t>(spill_at) + 1, false, spill});
        ins.push_back({static_cast<size_t>(fill_at), true, fill});
    }

    // Apply back-to-front so earlier indices stay valid. On equal positions
    // (adjacent spill/fill pair) apply the fill first: inserting the spill
    // at the same index afterwards lands it before the fill, which is the
    // required [spill, fill] order.
    std::sort(ins.begin(), ins.end(), [](const Insertion& a, const Insertion& b) {
        if (a.at != b.at) return a.at > b.at;
        return a.is_fill && !b.is_fill;
    });
    for (const auto& e : ins)
        plan.ops.insert(plan.ops.begin() + e.at, e.op);

    // Renumber every record_id from the final step order.
    for (size_t i = 0; i < plan.ops.size(); i++)
        plan.ops[i].record_id = compute_record_id(static_cast<uint32_t>(i));
}

} // namespace hnnx
