#pragma once
#include "hnnx/ir/types.hpp"
#include "hnnx/serialize/serializer.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hnnx {

// Forward declarations
class GraphPrepare;

// Real QNN HTP context binary format (reverse-engineered from
// simple_linear_context.bin, v2.48.0.260626).
//
// File layout:
//   [段1] 系统信息头 [0x0000, 0x1000)  4096 B   LE+BE mixed
//     [0x18] contextBlob offset (LE u64) = 0x1000
//     [0x20] contextBlob size   (LE u64)
//     [0x130] ioTensorSize (BE u32)
//     [0x177] constSize     (BE u32)
//     [0x1D5] dspArch       (BE u16)
//     [0x1E8] graph name    (null-terminated)
//     [0x334] buildId       (null-terminated)
//     [0x3C0,0x1000) zero padding (4KB aligned)
//
//   [段2] contextBlob [0x1000, end)  big-endian
//     ├─ header descriptor [0x1000, ~0x10A0)
//     ├─ ROOT record       [0x1098, ~0x10D8)
//     ├─ PICKLE record     [0x10D8, ~0x1100) "PICKLE/256/<graph_name>"
//     ├─ ioTensor segment  [0x2000, ~0x6000) tensor descriptors + graph config
//     ├─ op config strings [~0x6000, ~0x73C0) optimization flags as text
//     ├─ kernel name table [0x73C0, ~0x7490) 11 kernel names
//     ├─ opData segment    [0x7534, ~0x7B00)
//     │   FA0000FA separator → 19 op records → BEEFF00D end
//     ├─ const segment     [~0x9000, ~0xB000) weight extent table + data
//     ├─ ddrTensor segment [~0xA000, ~0xA400)
//     └─ trailer           [~0xB000, end) graph name + op names back-ref

// Op record type field (extra_info[0x18]>>6 & 0xf) << 24
enum OpRecordType : uint32_t {
    OP_TYPE_COMPUTE = 0x10,  // compute op (MatMul, Conv, etc.)
    OP_TYPE_MEMORY  = 0x20,  // memory op (perm load, layout convert)
    OP_TYPE_SYNC    = 0x30,  // sync barrier (@SyncOp)
    OP_TYPE_DMA     = 0x40,  // DMA checkpoint (@DmaCheckpointSet/Wait)
};

// Op record marker: 0x1303EE{XX} where XX = record id (incrementing from 0x71)
// Source: serialize_op @0x12ec630, 0x12ec688 (mov r13d, 0x1303ee71)
constexpr uint32_t OP_MARKER_BASE = 0x1303EE00;

// Separators and end markers (DSP in-memory, appear in contextBlob opData)
constexpr uint32_t SEP_FA0000FA = 0xFA0000FA;  // opData start separator
constexpr uint32_t END_BEEFF00D = 0xBEEFF00D;  // opData end marker

// System header field offsets (from real sample)
constexpr uint32_t SYS_HDR_CONTEXT_BLOB_OFFSET = 0x18;  // LE u64
constexpr uint32_t SYS_HDR_CONTEXT_BLOB_SIZE   = 0x20;  // LE u64
constexpr uint32_t SYS_HDR_IOTENSOR_SIZE       = 0x130; // BE u32
constexpr uint32_t SYS_HDR_CONST_SIZE          = 0x177; // BE u32
constexpr uint32_t SYS_HDR_DSP_ARCH            = 0x1D5; // BE u16
constexpr uint32_t SYS_HDR_GRAPH_NAME          = 0x1E8; // null-terminated
constexpr uint32_t SYS_HDR_BUILD_ID            = 0x334; // null-terminated
constexpr uint32_t SYS_HDR_TOTAL_SIZE          = 0x1000; // 4KB

// ContextBlob internal segment offsets (from simple_linear sample)
constexpr uint32_t CTX_BLOB_BASE       = 0x1000;
constexpr uint32_t IOTENSOR_SEG_BASE   = 0x2000;  // relative to file start
constexpr uint32_t KERNEL_TABLE_BASE  = 0x73C0;  // relative to file start
constexpr uint32_t OPDATA_SEG_BASE    = 0x7534;  // relative to file start (FA0000FA)
constexpr uint32_t CONST_SEG_BASE      = 0x9000;  // relative to file start
constexpr uint32_t TRAILER_BASE       = 0xB000;  // relative to file start

// Const tensor descriptor parameters.
// The 0x68-byte const extent descriptor is parameterized by (tensor_id, X, Y, elem_size).
// For FC weight (W): X = N (FC input rows), Y = M (FC output count)
// For FC bias  (b): X = K (FC input cols), Y = N (FC input rows)
// All 26 u32 fields of the descriptor are computed from these 4 values.
struct ConstDescriptorParam {
    uint32_t tensor_id;   // tensor id (e.g. 5 for W, 6 for b)
    uint32_t X;           // first shape parameter
    uint32_t Y;           // second shape parameter
    uint32_t elem_size;   // element size in bytes (4 for float32)
};

// OpData preamble parameters. The preamble encodes input/output tensor
// descriptors and graph configuration. Its length varies with QNN op count:
//   preamble_length = 33 + qnn_op_count (for FC models)
// All fields are computed from graph state — no model-specific hardcoding.
struct PreambleParam {
    uint32_t input_dim0;     // input tensor dim0 (e.g. 4 for [1,4,3])
    uint32_t input_dim1;     // input tensor dim1 (e.g. 3 for [1,4,3])
    uint32_t output_dim0;    // output tensor dim0 (e.g. 2 for [1,2,3])
    uint32_t output_dim1;    // output tensor dim1 (e.g. 3 for [1,2,3])
    uint32_t nonconst_count; // number of non-const tensors in graph
    uint32_t max_tid;        // maximum tensor id in graph
    uint32_t qnn_op_count;   // number of original QNN ops (before HtpPrepare injection)
};

// Trailer parameters for dynamic trailer generation.
// The trailer encodes: header constants, graph name, op name table with
// back-references, tensor name blocks, and a config block.
// All computed from graph state — no model-specific hardcoding.
struct TrailerParam {
    std::vector<std::string> output_op_names;  // ops that produce graph output (in QNN order)
    std::vector<std::string> input_op_names;   // ops that consume graph input (reversed)
    std::string output_tensor_name;            // graph output tensor name (e.g. "output")
    std::string input_tensor_name;             // graph input tensor name (e.g. "input")
    uint32_t output_tid;                        // output tensor id
    uint32_t input_tid;                         // input tensor id
    uint32_t input_rank;                        // input tensor rank
    uint32_t output_rank;                       // output tensor rank
    std::vector<uint32_t> output_dims;          // output tensor dims (rank entries)
    std::vector<uint32_t> input_dims;           // input tensor dims (rank entries)
};

// Kernel name entry (offset + name in kernel name table)
struct KernelNameEntry {
    uint32_t offset;  // offset within contextBlob
    std::string name;  // e.g. "MatMul_bias@ff*4"
};

// Op record in the opData segment (after scheduling)
// Byte layout (all LITTLE-ENDIAN, verified from real .bin):
//   [0-3]   marker    = 0x1303EE{record_id} (LE u32)
//   [4]     counter   = tensor_id (1 byte)
//   [5-6]   padding   = 0x0000
//   [7]     type      = 0x10/0x20/0x30/0x40 (1 byte)
//   [8-11]  f2        = DMA tag / kernel index / node hash (LE u32)
//   [12-15] block_ref = 0x1000 | vtcm_block_idx, or 0 (LE u32)
//   [16+]   extras[]  = variable-length, type-specific (LE u32 array)
struct ScheduledOp {
    uint32_t record_id;        // XX in 0x1303EE{XX}, computed by ID algorithm
    uint32_t tensor_id;        // counter = tensor id (0=graph node, 1-10=tensors)
    OpRecordType type;         // compute/memory/sync/dma
    uint32_t f2;               // DMA tag / kernel phase index / node hash
    uint32_t block_ref;        // 0x1000 | vtcm_block_idx, or 0 for direct compute
    std::vector<uint32_t> extras;  // variable-length type-specific data
    std::string kernel_name;   // HTP kernel name (for kernel name table)
    std::string step_name;     // human-readable step description
};

// Compute op record ID from step index.
// Pattern (verified from real .bin): pairs (odd, even), incrementing by 2.
//   step 0 → 0x71, step 1 → 0x70, step 2 → 0x73, step 3 → 0x72, ...
//   Formula: pair = step / 2
//   base = (0x11 + 2 * pair) & 0x1F
//   even step: id = base | 0x60
//   odd step:  id = (base - 1) | 0x60
inline uint32_t compute_record_id(uint32_t step_index) {
    uint32_t pair = step_index / 2;
    uint32_t base = (0x11 + 2 * pair) & 0x1F;
    if (step_index % 2 == 0)
        return base | 0x60;
    else
        return ((base - 1) & 0x1F) | 0x60;
}

// ContextBinaryWriter: produces a real QNN HTP context binary
// that the on-device libQnnHtpV73Skel.so can deserialize and execute.
class ContextBinaryWriter {
public:
    ContextBinaryWriter();
    ~ContextBinaryWriter();

    // Configuration setters (must be called before write())
    void set_graph_name(const std::string& name) { graph_name_ = name; }
    void set_build_id(const std::string& id) { build_id_ = id; }
    void set_dsp_arch(uint16_t arch) { dsp_arch_ = arch; }
    void set_io_tensor_size(uint32_t sz) { io_tensor_size_ = sz; }
    void set_const_size(uint32_t sz) { const_size_ = sz; }

    // Content setters
    void set_kernel_names(const std::vector<std::string>& names) {
        kernel_names_ = names;
    }
    void set_scheduled_ops(const std::vector<ScheduledOp>& ops) {
        scheduled_ops_ = ops;
    }
    void set_const_extents(const std::vector<ConstExtentDesc>& exts) {
        const_extents_ = exts;
    }
    void set_const_descriptor_params(const std::vector<ConstDescriptorParam>& params) {
        const_desc_params_ = params;
    }
    void set_preamble_param(const PreambleParam& p) { preamble_param_ = p; }
    void set_trailer_param(const TrailerParam& p) { trailer_param_ = p; }
    void set_const_pool(const std::vector<uint8_t>& pool) {
        const_pool_ = pool;
    }
    void set_op_names(const std::vector<std::string>& names) {
        op_names_ = names;
    }
    void set_tensor_names(const std::vector<std::string>& names) {
        tensor_names_ = names;
    }
    void set_sz_record_value(uint32_t v) { sz_record_value_ = v; }

    // Write the complete context binary to a buffer.
    // Returns total size written; 0 on error.
    size_t write(std::vector<uint8_t>& out);

    // Write context binary using a real .bin as template.
    // Extracts everything except opData records from template, then inserts
    // our scheduled ops. This allows byte-level comparison with real SDK output.
    // Returns total size; 0 on error.
    size_t write_from_template(const std::vector<uint8_t>& template_bin,
                               std::vector<uint8_t>& out);


private:
    // --- Configuration ---
    std::string graph_name_ = "simple_linear";
    std::string build_id_ = "v2.48.0.260626120635";
    uint16_t dsp_arch_ = 0;       // 0 for this sample
    uint32_t io_tensor_size_ = 0x00400000;
    uint32_t const_size_ = 0x00200000;

    // --- Content ---
    std::vector<std::string> kernel_names_;      // 11 kernel names
    std::vector<ScheduledOp> scheduled_ops_;      // 19 scheduled ops
    std::vector<ConstExtentDesc> const_extents_;  // weight extent table
    std::vector<ConstDescriptorParam> const_desc_params_; // descriptor params per const
    PreambleParam preamble_param_{};             // opData preamble parameters
    TrailerParam trailer_param_{};               // trailer parameters
    std::vector<uint8_t> const_pool_;            // weight data block
    std::vector<std::string> op_names_;           // trailer op names
    std::vector<std::string> tensor_names_;      // trailer tensor names
    uint32_t sz_record_value_ = 0;               // Sz record payload[5] (0x50EC)

    // --- Internal write helpers (big-endian for contextBlob) ---
    // Output buffer and current position
    std::vector<uint8_t> buf_;
    size_t pos_ = 0;

    void ensure(size_t extra);
    void put_u8(uint8_t v);
    void put_u16_le(uint16_t v);
    void put_u32_le(uint32_t v);
    void put_u64_le(uint64_t v);
    void put_u16_be(uint16_t v);
    void put_u32_be(uint32_t v);
    void put_u64_be(uint64_t v);
    void put_bytes(const void* data, size_t len);
    void put_string_null(const std::string& s);
    void put_zero(size_t count);
    void align_to(size_t alignment);

    // Section writers
    void write_system_header();
    size_t write_context_blob();
    void write_ctx_header_descriptor();
    void write_root_record();
    void write_pickle_record();
    void write_io_tensor_segment();
    void write_kernel_name_table();
    void write_opdata_segment();
    void write_post_opdata();
    void write_const_segment();
    void write_const_descriptor(const ConstDescriptorParam& param);
    void write_trailer();
    void write_trailer_tensor_block(const std::string& name, uint32_t tid, uint32_t rank,
                                     const std::vector<uint32_t>& dims, bool is_output);
    void write_u32_at(size_t offset, uint32_t val);
    void patch_pre_opdata();
    bool kernel_names_has(const std::string& substr) const;
};

} // namespace hnnx
