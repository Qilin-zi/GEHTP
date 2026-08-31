#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <unordered_map>

#include "status.hpp" // hnnx::GraphStatus (唯一定义在此; GCP 旧族与 ir 精确族共用)
#include "hnnx/ir/op_id.hpp" // hnnx::op_id_t (唯一声明点)

// 全局真身 Tensor 的前置 (定义在 tensor_base.hpp; vtable _ZTV6Tensor)。
// hnnx 命名空间内的非限定 Tensor 统一解析至此 —— 不再有 hnnx::Tensor 占位
// (占位与 ::Tensor 是两个不同类型, 会在签名交界处静默错位)。
class Tensor;

namespace hnnx {

// op_id_t 见 hnnx/ir/op_id.hpp (唯一声明点)
using op_hash_t = uint64_t;

// GCP 2.2: Storage class for memory planning
enum StorageClass : uint32_t {
    DDR             = 0,   // external memory, large capacity, low bandwidth
    VTCM            = 1,   // on-chip tightly coupled memory, needs scheduling
    CONST           = 2,   // constant weights, read-only cache
    VTCM_PERSISTENT = 3,   // resident VTCM, not eligible for spill
};

// GraphStatus 已移至 status.hpp (ir 精确族只依赖该枚举, 避免与本文件
// GCP 占位结构 hnnx::Op/OpDef/OutputDef/DType 的重定义/遮蔽冲突)

constexpr uint32_t FIB_MULT_1 = 0x192E2101;
constexpr uint32_t FIB_MULT_2 = 0x740F1DE9;

// Fibonacci hash for optimization rule lookup
uint64_t fibonacci_hash(uint64_t key);

// HTP shape hash: computes f2 field for InputNodeSetup/OutputNodeSetup ops.
// Source: libHtpPrepare.so gen_Const_int32_common @ 0xF866A0
//   mov $0x51BE1035, %r13d      ; initial hash
//   add dims[0], %r13d          ; hash += dims[0]
//   imul $0x01003123, %r13d     ; hash *= 0x01003123
//   add dims[1], %r13d         ; hash += dims[1]
//   ... (repeat for each dim, up to 8)
// Verified against 7 context .bin samples (simple_linear + 6 variants).
constexpr uint32_t HTP_HASH_INIT = 0x51BE1035;
constexpr uint32_t HTP_HASH_MULT = 0x01003123;

static inline uint32_t htp_shape_hash(const uint32_t* dims, uint32_t rank) {
    uint32_t h = HTP_HASH_INIT;
    h += dims[0];
    for (uint32_t i = 1; i < rank && i < 8; i++) {
        h *= HTP_HASH_MULT;
        h += dims[i];
    }
    return h;
}

static inline uint32_t htp_shape_hash(const std::vector<uint32_t>& dims) {
    if (dims.empty()) {
        uint32_t zero = 0;
        return htp_shape_hash(&zero, 0);
    }
    return htp_shape_hash(dims.data(), static_cast<uint32_t>(dims.size()));
}

// 64-bit Galois LFSR checksum (hnnx::checksum_bytes)
// Verified: poly=0x1b, two LFSR steps per byte then XOR with data byte.
// Source: libHtpPrepare.so @0xdacd20, helper @0x35a5fd0
// Test vectors confirmed against real .so:
//   (0, "Hello, HTP!", 11) -> 0x6974500000000031
//   (0xDEADBEEFCAFEBABE, "Hello, HTP!", 11) -> 0xae18277ab6fbbf1a
//   (0, "", 0) -> 0
static inline uint64_t crc_step_lfsr(uint64_t x) {
    uint64_t v = x & 0x1b;
    int cnt = 0;
    while (v) { v &= v - 1; cnt++; }
    return ((uint64_t)(cnt & 1) << 63) | (x >> 1);
}

static inline uint64_t checksum_bytes(uint64_t init, const uint8_t* data, uint32_t len) {
    uint64_t acc = init;
    for (uint32_t i = 0; i < len; i++) {
        acc = crc_step_lfsr(crc_step_lfsr(acc));
        acc ^= data[i];
    }
    return acc;
}

struct string_tag_t {
    const char* name_ = nullptr;
    uint64_t hash_key_ = 0;
    static string_tag_t* map_str(const char* name);
    const char* name() const { return name_; }
    uint64_t hash_key() const { return hash_key_; }
};

enum class DType : uint32_t {
    Float32 = 0,
    Float16 = 1,
    Int8    = 2,
    Int16   = 3,
    Int32   = 4,
    UInt8   = 5,
    Bool    = 6,
    BFloat16= 7,
    Int4    = 8,
    UInt4   = 9,
    FP8_E4M3= 10,
    FP8_E5M2= 11,
    MXFP4   = 12,
};

struct InputDef {
    uint32_t rank;
    uint32_t dtype;
    uint32_t flags;
    uint32_t reserved;
    uint64_t dims[5];
    uint64_t element_size;
};

struct OutputDef {
    uint32_t rank;
    uint32_t dtype;
    uint32_t flags;
    uint32_t quant_params;
    uint64_t dims[5];
    uint64_t element_size;
    uint64_t quant_scale;
    uint64_t quant_offset;
    uint64_t extra[3];

    OutputDef() = default;
};

enum OpDefFlags : uint16_t {
    OP_ENABLED  = 0x01,
    OP_CONST    = 0x02,
    OP_DEAD     = 0x04,
    OP_SHAPE    = 0x08,
    OP_DYNAMIC  = 0x10,
    OP_SWITCHED = 0x20,
    OP_SLICED   = 0x40,
    // GCP 2.1: bit 0x40 = migrated (TCM migration marker, same bit as SLICED)
    OP_MIGRATED = 0x40,
};

// Input connection: (source op id, output index)
// Carries the source op's OutputDef so the host reference executor can
// infer input shapes for ops like MatMul that need to read their kernels.
struct InputConn {
    op_id_t src_id;
    uint32_t out_idx;
    OutputDef src_out_def{};
};

struct OpDef {
    void* vtable = nullptr;
    uint16_t flags = 0;
    uint16_t string_tag = 0;
    uint64_t reserved_10 = 0;
    void* graph = nullptr;
    op_id_t op_id = 0;
    string_tag_t* name_tag = nullptr;
    void* inputs_start = nullptr;
    void* inputs_end = nullptr;
    void* extra_40 = nullptr;
    OutputDef output_def;
    void* tensor_ptr = nullptr;
    uint32_t flags2 = 0;
    void* vtable2 = nullptr;
    void* persistent_tensor = nullptr;

    // C++-side input connections (mirrors inputs_start..inputs_end in binary)
    std::vector<InputConn> inputs;

    // Consumer list: ops that consume this op's output
    std::vector<op_id_t> consumers;

    // Const data location in the graph's const pool (OpDef_Const only).
    // const_data_offset == 0 && const_data_size == 0 means "no inline data".
    uint64_t const_data_offset = 0;
    uint64_t const_data_size = 0;

    // op_data: per-op 参数 blob (由 append_node 的 ops_data 传入)。
    // 真实库在 Op 构造时解析这些字节得到 stride/padding/dilation/axis 等。
    std::vector<uint8_t> op_data;

    // serialized_extra: 序列化侧 per-op 参数 schema 的尾随字段字节
    // (serialize_opdef 尾随 [u32 len][bytes])。deserialize 回读存于此,
    // re-serialize 时原样重发(round-trip 确定性, 不重跑 extractor)。
    std::vector<uint8_t> serialized_extra;

    // grouping: 原始 op 名（来自 QNN IR addNode 的 node_name 参数），
    // 用于 dump before/after graph JSON 时和 QNN before_graph.json 对齐。
    // 例如 "input_ncf"、"MatMul_0_pre_reshape"。
    std::string grouping;

    // tensor_param_ids: QNN tensor_param 常量(perm/axes/stride/pad/dilation)的 op_id 列表。
    // composeGraphs 阶段这些是独立 const 节点, 不放入 op inputs;
    // HtpPrepare(do_prepare1) 注入阶段把它们追加到 inputs。
    std::vector<op_id_t> tensor_param_ids;

    // GCP 2.1 supplementary fields
    uint32_t op_type = 0;            // +0x08: MatMul=0, RMSNorm=1, GELU=2, ...
    uint32_t sub_type = 0;           // +0x0c
    uint64_t phase_id = 0;           // +0x28: optimization phase this op belongs to
    std::vector<void*> inputs_vec;   // +0x10: std::vector<Tensor*>
    std::vector<void*> outputs_vec;  // +0x30: std::vector<Tensor*>
    uint32_t quant_count = 0;        // +0x48: number of quantization parameters
    void*    quant_array = nullptr;  // +0x50: quantization parameter array
    uint8_t  crouton_from_vtcm = 0;  // +0x5d: input expected already in VTCM
    uint8_t  crouton_to_vtcm = 0;    // +0x5f: output will reside in VTCM
    uint64_t tag_bitmap = 0;         // +0x68: tag bitmap
    uint32_t priority = 0;           // +0x98: TcmMigration heap sort priority

    bool is_enabled() const { return (flags & OP_ENABLED) != 0; }
    bool is_const() const { return (flags & OP_CONST) != 0; }
    bool is_dead() const { return (flags & OP_DEAD) != 0; }
    bool is_migrated() const { return (flags & OP_MIGRATED) != 0; }
    op_hash_t hash_key() const;
    size_t input_count() const;

    virtual ~OpDef() = default;
};

struct OpDef_Const : OpDef {
    OpDef_Const(class GraphPrepare& gp, op_id_t id, const OutputDef& od,
                const uint8_t* data, size_t data_len);
};

struct Op {
    void* vtable;
    uint64_t reserved_08;
    uint64_t reserved_10;
    void* graph;
    uint64_t reserved_28;
    uint64_t output_info;

    virtual ~Op() = default;
    virtual float cost(const struct Graph*) const = 0;
    virtual void serialize_internal(class Serializer&, int chkpt_type) const = 0;
};

struct Graph {
    void* vtable;
    uint64_t reserved[31];

    // GCP 2.3 supplementary fields
    std::vector<OpDef*> ops;          // operators in execution order
    std::vector<Tensor*> tensors;     // tensor list
    void* graph_deps_ptr = nullptr;   // +0x7468: GraphDeps* (see graph_deps.hpp)
    uint8_t state_machine = 1;        // +0x45dc: 0=ERROR,1=CONSTRUCTION,2=PREPARE,3=COMPILED
    bool graph_dirty = false;         // +0x7311: Fixpoint convergence flag

    const void* get_extra_info(const Op* op) const;
};

} // namespace hnnx
