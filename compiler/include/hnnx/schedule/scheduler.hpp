#pragma once
#include "hnnx/ir/types.hpp"
#include "hnnx/serialize/context_binary_writer.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace hnnx {

class GraphPrepare;

// Tensor info for scheduling
struct TensorInfo {
    uint32_t tensor_id;       // 0=graph_node, 1-N=tensors
    std::string name;          // e.g. "input", "W", "output_fc"
    uint32_t rank;             // number of dimensions
    uint64_t dims[5];          // shape padded to 5
    DType dtype;               // data type
    bool is_const;             // is this a const tensor (weight/bias)?
    bool is_graph_io;          // is this a graph input/output?
};

// Op types for step expansion
enum class HtpOpType {
    Transpose,
    Reshape,
    FullyConnected,
};

// Descriptor for a single QNN op in the model
struct OpDesc {
    HtpOpType type;
    std::string name;
    uint32_t input_tid;     // primary input tensor id
    uint32_t output_tid;   // primary output tensor id
    uint32_t weight_tid;    // W tensor id (FC only)
    uint32_t bias_tid;      // b tensor id (FC only)
    uint32_t perm_tid;      // perm tensor id (Transpose only)
    uint32_t weight_m;      // W rows (FC only)
    uint32_t weight_k;      // W cols (FC only)
    uint32_t act_ref_tid;   // activation reference tid for MatMul extras (0 = use default)
};

// Model descriptor for general scheduling
struct ModelDesc {
    uint32_t input_tid;
    uint32_t input_batch;
    uint32_t input_dim0;
    uint32_t input_dim1;
    uint32_t input_dim2;   // 0 for rank-2

    uint32_t output_tid;
    uint32_t output_batch;
    uint32_t output_dim0;
    uint32_t output_dim1;
    uint32_t output_dim2;  // 0 for rank-2

    uint32_t perm_in_tid;
    uint32_t perm_out_tid;
    uint32_t sync_tid;       // tensor ID for SyncOp (model-specific)
    uint32_t input_ncf_tid;  // tensor ID for input_ncf (MatMul input reference)

    std::vector<OpDesc> ops;
    std::vector<std::string> op_names;
    std::vector<std::string> tensor_names;
};

// Kernel selector: maps QNN op names to HTP kernel names.
// The kernel name encodes data type, tiling, and parallelism:
//   MatMul_bias@ff*4  = float32, vector width 4
//   Transpose_impl@ff*2.fi.t = float32, 2D, flat+isolated, transpose
//   *InputSlice@Ff.s4*6. = float32, HVX width 4, 6-way parallel
//   flat_from_vtcm@ff.Ff. = float32, layout conversion
class KernelSelector {
public:
    KernelSelector();

    // Select kernel for a QNN op. Returns kernel name and f2 (phase index).
    struct KernelResult {
        std::string kernel_name;
        uint32_t f2;  // phase index / DMA tag
    };

    // Main selection: given op name, data type, and context, pick kernel
    KernelResult select(const std::string& op_name, DType dtype,
                        bool is_input_side, bool is_output_side) const;

    // Get all unique kernel names used (for kernel name table)
    std::vector<std::string> get_kernel_table() const;

    // DMA and sync kernel names (fixed)
    static constexpr const char* DMA_SET   = "@DmaCheckpointSet";
    static constexpr const char* DMA_WAIT  = "@DmaCheckpointWait";
    static constexpr const char* SYNC_OP   = "@SyncOp";
    static constexpr const char* CONST     = "Const";
    static constexpr const char* SHAPE     = "Shape";

private:
    // Data type suffix encoding
    std::string dtype_suffix(DType dt) const;
};

// Scheduler: transforms a GraphPrepare's optimized graph into a
// list of ScheduledOps (the HTP execution plan).
//
// For simple_linear (5 QNN ops → 19 HTP steps):
//   Input node → 2 InputNodeSetup steps
//   Transpose(input) → InputSlice + perm_load + layout_convert (3 steps)
//   Reshape(pre-FC) → flat_from_vtcm + fc_prep (2 steps)
//   FC weights → DMA SET(W) + DMA SET(b) (2 steps)
//   FullyConnected → MatMul(x·W) + MatMul(+b) (2 steps)
//   FC output → DMA SET(out_fc) + flat_from_vtcm + DMA SET(out_ncf) (3 steps)
//   Output node → OutputNodeSetup (1 step)
//   Sync → SyncOp (1 step)
//   Transpose(output) → perm_load + Transpose_impl + OutputSlice (3 steps)
class Scheduler {
public:
    struct Plan {
        std::vector<ScheduledOp> ops;           // 19 scheduled ops
        std::vector<std::string> kernel_names;  // unique kernel name table
        std::vector<std::string> op_names;       // op names for trailer
        std::vector<std::string> tensor_names;   // tensor names for trailer
        std::vector<TensorInfo> tensors;        // tensor metadata
        std::vector<uint32_t> op_order;          // st-cut compact execution order
    };

    Scheduler();

    // Build execution plan from a prepared graph
    Plan schedule(const GraphPrepare& gp);

    // Build execution plan from explicit model descriptor (general)
    Plan schedule_general(const ModelDesc& model);

    // Build execution plan from explicit model definition (for testing)
    Plan schedule_simple_linear(
        const std::vector<TensorInfo>& tensors,
        const std::vector<std::string>& op_names);

    // Phase B (serialization level): for each CP spill/fill plan insert one
    // OP_TYPE_DMA pair (spill + fill) into the plan, then renumber every
    // record_id via compute_record_id. No-op unless the graph ran the CP
    // allocator branch — non-CP mode inserts nothing (byte-identical plan).
    void apply_cp_spill_fill(Plan& plan, const GraphPrepare& gp);

private:
    KernelSelector kernel_sel_;
    uint32_t next_step_ = 0;     // current step index (for record_id computation)
    uint32_t next_dma_tag_ = 0x11;  // DMA tags start at 0x11

    // VTCM block index assignment
    uint32_t next_vtcm_block_ = 0x19;  // VTCM blocks start at 0x19
    std::unordered_map<uint32_t, uint32_t> tensor_block_map_;  // tensor_id → vtcm_block

    // Helpers
    ScheduledOp make_op(uint32_t tensor_id, OpRecordType type,
                        uint32_t f2, uint32_t block_ref,
                        const std::vector<uint32_t>& extras,
                        const std::string& kernel_name,
                        const std::string& step_name);

    // Step generators for simple_linear pattern
    void gen_input_node_setup(Plan& plan, const TensorInfo& input_tensor, bool is_first);
    void gen_input_slice(Plan& plan, const TensorInfo& input_tensor,
                         const TensorInfo& output_tensor);
    void gen_perm_load(Plan& plan, const TensorInfo& perm_tensor);
    void gen_layout_convert(Plan& plan, const TensorInfo& in_tensor,
                            const TensorInfo& out_tensor);
    void gen_flat_from_vtcm(Plan& plan, const TensorInfo& tensor,
                            uint32_t f2_phase);
    void gen_fc_prep(Plan& plan, const TensorInfo& tensor, uint32_t f2_phase);
    void gen_dma_set(Plan& plan, const TensorInfo& weight_tensor,
                     uint32_t dma_tag, const std::vector<uint32_t>& dma_extras);
    void gen_matmul(Plan& plan, const TensorInfo& out_tensor,
                    uint32_t f2_phase, bool with_bias);
    void gen_output_node_setup(Plan& plan, const TensorInfo& output_tensor);
    void gen_sync(Plan& plan, const TensorInfo& sync_tensor,
                  const std::vector<uint32_t>& sync_extras);
    void gen_transpose_impl(Plan& plan, const TensorInfo& in_tensor,
                            const TensorInfo& out_tensor);
    void gen_output_slice(Plan& plan, const TensorInfo& output_tensor);

    // VTCM block allocation
    uint32_t alloc_vtcm_block(uint32_t tensor_id);
};

} // namespace hnnx
