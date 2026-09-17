#pragma once
// ============================================================================
// REQNN — 中层 IR 完整数据结构整合头文件
// ----------------------------------------------------------------------------
// 整合来源:
//   [REQNN]  REQNN 工程 include/hnnx/** 全部 18 个头文件（逆向重实现现状）
//   [GCP]    GRAPH_COMPILER_PRINCIPLES.md（libHtpPrepare 反编译原理，8-bank CBS 等）
//   [GUIDE]  SERIALIZED_BIN_ANALYSIS_GUIDE.md（真实 .bin 字节格式）
//   [BFA]    reference/docs/bin_format_analysis.md（真实 .bin 大端 count+offset）
//
// 标注约定:
//   ✅ = 已在 REQNN 中实现（附反汇编来源注释）
//   ⚠️ = 结构存在但实现降级/简化（见注释说明）
//   🔶 = 缺口补充：真实库存在、REQNN 缺失，按 GCP/GUIDE 补齐，待实现
//
// 本文件为自包含整合视图（不 include 子头文件），可作为联合调度 + 内存
// 规划优化框架的单一数据结构基准。
// ============================================================================

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <unordered_map>
#include <atomic>

// Tensor 真身是全局类 (tensor_base.hpp; _ZTV6Tensor)。此处全局前置,
// 使下方 namespace hnnx 内的非限定 Tensor 解析到 ::Tensor (不再有
// hnnx::Tensor 占位 —— 与 ::Tensor 是两个类型, 交界处会静默错位)。
class Tensor;

namespace hnnx {

// ============================================================================
// 0. 前向声明（使本头文件可独立编译）
// ============================================================================

class GraphPrepare;
class Serializer;
class Deserializer;
class Allocator;
class FileSerializer;
class HexagonNNEnv;
struct Op;
struct OpDef;
struct OpRef;
struct OpIoPtrs;
struct NspIdMap;
struct profilingevent;
struct runlist_auxdata_seg_desc;
struct Graph;
struct OutputDef;
class VtcmCacheInstance;

template<typename T> struct uptr_DWDI { T* ptr; };

// ============================================================================
// 1. 基础类型
// ============================================================================

using op_id_t   = uint64_t;
using op_hash_t = uint64_t;

// ✅ GCP 3.3: Fibonacci hash 乘数（与真实库一致）
constexpr uint32_t FIB_MULT_1 = 0x192E2101;
constexpr uint32_t FIB_MULT_2 = 0x740F1DE9;

uint64_t fibonacci_hash(uint64_t key);

// ✅ 内部字符串标签（hash 化，替代全串比较）
struct string_tag_t {
    const char* name_ = nullptr;
    uint64_t hash_key_ = 0;
    static string_tag_t* map_str(const char* name);
    const char* name() const { return name_; }
    uint64_t hash_key() const { return hash_key_; }
};

// ✅ 数据类型枚举
enum class DType : uint32_t {
    Float32 = 0, Float16 = 1, Int8 = 2, Int16 = 3, Int32 = 4,
    UInt8 = 5, Bool = 6, BFloat16 = 7, Int4 = 8, UInt4 = 9,
    FP8_E4M3 = 10, FP8_E5M2 = 11, MXFP4 = 12,
};

// ============================================================================
// 2. Tensor 元数据
// ============================================================================

// 🔶 GCP 2.2 存储类别（REQNN 的 Tensor 仅 data[14]，缺此三字段）
enum StorageClass : uint32_t {
    DDR           = 0,   // 外部内存, 容量大带宽低
    VTCM          = 1,   // 片上紧耦合内存, 需规划
    CONST         = 2,   // 常量权重, 只读缓存
    VTCM_PERSISTENT = 3, // 常驻 VTCM, 不参与 spill
};

// ✅ REQNN 现有 Tensor: 极简布局（+0x00 vtable, +0x08~0x98 tensor data）
// 🔶 缺口补充字段按 GCP 2.2 注释，供内存规划/生命周期分析使用
struct Tensor {
    void* vtable;             // +0x00
    uint64_t data[14];        // +0x08-0x98: dims/dtype/quant/buffer ptr

    // 🔶 GCP 2.2 补充字段（真实库 Tensor 元数据）
    uint32_t id = 0;                    // tensor ID
    std::vector<uint32_t> dims;         // 形状
    uint32_t dtype = 0;                 // DType
    StorageClass storage_class = DDR;   // 存储类别
    uint8_t  bank_mask = 0;             // 8-bit bank 占用掩码
    uint32_t vtcm_offset = 0;           // VTCM 内偏移 (2KB 对齐)
    uint32_t lifetime_start = 0;        // 首次产生的算子索引
    uint32_t lifetime_end = 0;          // 最后使用的算子索引
    std::vector<op_id_t> producers;     // 生产者 op
    std::vector<op_id_t> consumers;     // 消费者 op
    bool is_output = false;
    bool has_side_effect = false;

    // ✅ REQNN
    static void persistent_clone(class Allocator* alloc, Tensor* src);
    uint64_t memory_cost() const;   // ⚠️ 未真正实现
    uint64_t num_elements() const;  // ⚠️ 未真正实现
    uint64_t getSize() const;       // 🔶 元素数*元素字节
    bool isVTCM() const { return storage_class == VTCM || storage_class == VTCM_PERSISTENT; }
};

// ✅ REQNN: 标量 tensor 生成器（⚠️ 当前返回 nullptr, 未实现）
Tensor* tensor_generator_scalar(OutputDef* od_override, const OutputDef* od, const uint8_t* data);

// ============================================================================
// 3. 输入/输出定义
// ============================================================================

// ✅ REQNN: 输入定义（rank/dtype/dims/element_size）
struct InputDef {
    uint32_t rank;
    uint32_t dtype;
    uint32_t flags;
    uint32_t reserved;
    uint64_t dims[5];
    uint64_t element_size;
};

// ✅ REQNN: 输出定义（含量化参数）
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

// ✅ REQNN: OpDef flags（注意：真实库 +0x04 bit0x40=migrated 见 OpDefFlags 补）
enum OpDefFlags : uint16_t {
    OP_ENABLED  = 0x01,
    OP_CONST    = 0x02,
    OP_DEAD     = 0x04,
    OP_SHAPE    = 0x08,
    OP_DYNAMIC  = 0x10,
    OP_SWITCHED = 0x20,
    OP_SLICED   = 0x40,
    // 🔶 GCP 2.1: 真实库 flags bit0x40 = migrated（TCM 迁移标记）
    OP_MIGRATED = 0x40,
};

// ✅ REQNN: 输入连接 (source op id, output index)
struct InputConn {
    op_id_t src_id;
    uint32_t out_idx;
};

// ============================================================================
// 4. OpDef（算子定义）
// ============================================================================

// ✅ REQNN 现有字段 + 🔶 GCP 2.1 缺口补充（crouton/priority/quant/op_type）
// 真实库布局约 256 字节。REQNN 以 C++ 字段承载；GCP 字段偏移为参考。
struct OpDef {
    void* vtable = nullptr;          // +0x00
    uint16_t flags = 0;              // +0x04 (真实库 uint8 @ +0x04)
    uint16_t string_tag = 0;
    uint64_t reserved_10 = 0;
    void* graph = nullptr;
    op_id_t op_id = 0;
    string_tag_t* name_tag = nullptr;
    void* inputs_start = nullptr;    // +0x10: std::vector<Tensor*> inputs
    void* inputs_end = nullptr;
    void* extra_40 = nullptr;
    OutputDef output_def;
    void* tensor_ptr = nullptr;
    uint32_t flags2 = 0;
    void* vtable2 = nullptr;
    void* persistent_tensor = nullptr;

    // 🔶 GCP 2.1 缺口补充字段
    uint32_t op_type = 0;            // +0x08: MatMul=0, RMSNorm=1, GELU=2, ...
    uint32_t sub_type = 0;           // +0x0c
    uint64_t phase_id = 0;           // +0x28: 所属优化 phase
    std::vector<void*> inputs_vec;   // +0x10: std::vector<Tensor*>
    std::vector<void*> outputs_vec;  // +0x30: std::vector<Tensor*>
    uint32_t quant_count = 0;        // +0x48: 量化参数个数
    void*    quant_array = nullptr;  // +0x50: 量化参数数组
    uint8_t  crouton_from_vtcm = 0;  // +0x5d: 输入预期已在 VTCM
    uint8_t  crouton_to_vtcm = 0;    // +0x5f: 输出将驻留 VTCM
    uint64_t tag_bitmap = 0;         // +0x68: 标签位图
    uint32_t priority = 0;           // +0x98: TcmMigration 堆排序优先级

    // ✅ REQNN C++-side
    std::vector<InputConn> inputs;   // 镜像 inputs_start..inputs_end
    std::vector<op_id_t> consumers;  // 消费本 op 输出的 op 列表
    uint64_t const_data_offset = 0;  // const pool 偏移 (OpDef_Const)
    uint64_t const_data_size = 0;    // const 数据大小
    std::vector<uint8_t> op_data;    // append_node 传入的 per-op 参数 blob

    bool is_enabled() const { return (flags & OP_ENABLED) != 0; }
    bool is_const() const   { return (flags & OP_CONST) != 0; }
    bool is_dead() const    { return (flags & OP_DEAD) != 0; }
    bool is_migrated() const { return (flags & OP_MIGRATED) != 0; }  // 🔶
    op_hash_t hash_key() const;
    size_t input_count() const;

    virtual ~OpDef() = default;
};

// ✅ REQNN: 常量 op（内联 const data）
struct OpDef_Const : OpDef {
    OpDef_Const(class GraphPrepare& gp, op_id_t id, const OutputDef& od,
                const uint8_t* data, size_t data_len);
};

// ============================================================================
// 5. GraphDeps（依赖图 / 冲突图）—— 内存规划核心
// ============================================================================

// ✅ REQNN 前向声明（⚠️ 未实现完整结构）
// 🔶 GCP 2.4 缺口补充: 位图邻接矩阵 + 生命周期数组 + 着色结果
struct GraphDeps {
    uint32_t num_tensors = 0;

    // 🔶 位图优化的邻接矩阵: conflict_matrix[i][j/64] bit(j%64) = i,j 冲突
    std::vector<std::vector<uint64_t>> conflict_matrix;

    // 🔶 生命周期数组（扁平化, CBS 快速访问）
    std::vector<uint32_t> lifetime_starts;
    std::vector<uint32_t> lifetime_ends;

    // 🔶 CBS 分配结果
    std::vector<struct BlockTableEntry> entries;

    // 🔶 冲突检查/添加
    bool hasConflict(uint32_t i, uint32_t j) const {
        uint64_t mask = 1ULL << (j % 64);
        return conflict_matrix[i][j / 64] & mask;
    }
    void addConflict(uint32_t i, uint32_t j) {
        conflict_matrix[i][j / 64] |= (1ULL << (j % 64));
        conflict_matrix[j][i / 64] |= (1ULL << (i % 64));
    }
};

// 🔶 GCP 2.4 / 6.2: BlockTableEntry —— 每个张量的存储决策
// (bank, 偏移, spill 标记)。序列化 BlockTable 段按此结构逐条写出。
struct BlockTableEntry {
    uint32_t tensor_id;
    uint8_t  color;              // bank 颜色 (0-7) / 8 = spill 到 DDR
    uint32_t vtcm_offset;        // VTCM 内偏移 (2KB 对齐)
    uint32_t ddr_offset;         // DDR 内偏移
    uint32_t size;               // 字节数
    StorageClass storage_class;  // 存储类别
    bool     is_spilled;         // 是否被 spill
    uint8_t  bank_mask;          // 占用 bank 掩码
    uint32_t pool_id;            // 内存池 ID
};

// ============================================================================
// 6. Graph（顶层容器）
// ============================================================================

// ✅ REQNN: Graph 顶层（⚠️ 当前为 vtable + reserved 壳）
// 🔶 GCP 2.3: 真实字段 ops/tensors/deps/state_machine/graph_dirty
struct Graph {
    void* vtable;
    uint64_t reserved[31];

    // 🔶 GCP 2.3 缺口补充字段
    std::vector<OpDef*> ops;            // 算子列表（按执行顺序）
    std::vector<Tensor*> tensors;       // 张量列表
    GraphDeps* deps = nullptr;          // +0x7468 依赖图指针
    uint8_t state_machine = 1;          // +0x45dc: 0=ERROR,1=CONSTRUCTION,2=PREPARE,3=COMPILED
    bool graph_dirty = false;           // +0x7311: Fixpoint 收敛标志

    const void* get_extra_info(const Op* op) const;
};

// ============================================================================
// 7. Tiling（分块）
// ============================================================================

// ✅ REQNN: tile 形状 / tile 信息 / tiling 配置
struct TileShape {
    uint32_t dims[5];
    uint32_t rank;
};

struct TileInfo {
    TileShape shape;
    uint32_t nsp_id;        // 归属 NSP
    uint32_t vtcm_offset;   // VTCM 内偏移
    uint64_t ddr_offset;    // DDR 内偏移
    bool is_contiguous;     // 是否连续
    bool needs_dma;         // 是否需要 DMA 搬运
};

struct TilingConfig {
    uint32_t conv_batch_tiling = 1;
    uint32_t conv_width_tiling = 1;
    uint32_t conv_height_tiling = 1;
    uint32_t conv_channel_tiling = 1;
    bool central_tiling = false;
    uint32_t central_tiler_conv_batch = 0;
    uint32_t central_tiler_conv_width = 0;
    bool force_conv_fusion = false;
    bool conv_output_dynamic_rescaling = false;
};

// ✅ REQNN: Tiler 基类与变体（⚠️ 实现为简化版）
class Tiler {
public:
    virtual ~Tiler() = default;
    virtual std::vector<TileInfo> generate_tiles(
        const Op* op, const Graph* graph, const TilingConfig& config) = 0;
    virtual uint64_t estimate_cost(
        const Op* op, const std::vector<TileInfo>& tiles) const = 0;
    virtual const char* name() const = 0;
};

class SimpleTiler  : public Tiler { /* ✅ */ };
class Supertiler   : public Tiler { /* ✅ 合并相邻同维 tile */ };
class ConvTiler    : public Tiler { /* ✅ 读真实 OutputDef N,H,W,C */ };
class MatMulTiler  : public Tiler { /* ✅ M,K,N 按 VTCM 友好切块 */ };

// ✅ REQNN: tile 分配 / 提取 / conform / 注册
class TileDistributor {
public:
    std::vector<std::vector<TileInfo>> distribute(
        const std::vector<TileInfo>& tiles, uint32_t num_nsps) const;
    void print_stats(const std::vector<std::vector<TileInfo>>& distribution) const;
};

class TileExtractor {
public:
    void extract(const void* src_data, const TileShape& src_shape,
                 void* dst_data, const TileInfo& tile);
};

class TileConformer {
public:
    TileShape conform(const TileShape& shape, DType dtype, uint32_t nsp_id) const;
};

class TilingRegistry {
public:
    static TilingRegistry& instance();
    void register_tiler(const std::string& op_name, std::unique_ptr<Tiler> tiler);
    Tiler* get_tiler(const std::string& op_name) const;
    void register_callback(const std::string& event,
                           std::function<void(const TileInfo&)> callback);
};

uint64_t compute_conv_tile_cost(
    const TileShape& input_shape, const TileShape& output_shape,
    const TileShape& weight_shape, uint32_t stride, uint32_t padding);

// ============================================================================
// 8. 内存规划（VTCM / FancyAllocator）
// ============================================================================

namespace fa {

// ✅ REQNN: 内存块 / 内存池
struct MemBlock {
    uint64_t offset;
    uint64_t size;
    uint32_t pool_id;
    bool is_free;
    bool is_mc_cacheable_shared;
};

struct Pool {
    uint64_t base_offset;
    uint64_t size;
    std::vector<MemBlock> blocks;
};

// ✅ REQNN: FancyAllocator
// ⚠️ 头注释: 真实库是 linear scan / bin packing（非图着色非 ILP）
// ⚠️ 实现: 线性 bump（pool.size 前进）, allow_tensor_overlap 简化
// 🔶 真实库另有 lifetime-overlap 复用（TakenRange 仅雏形）
struct TakenRange {
    uint64_t start;
    uint64_t end;
    int op_index;
};

class FancyAllocator {
public:
    FancyAllocator();
    ~FancyAllocator();

    void* allocate(size_t size, size_t alignment);
    void deallocate(void* ptr);

    void allow_tensor_overlap(const hnnx::Op* op);
    void setup_heap_info(hnnx::Serializer& ser, uint64_t total_size);
    void* get_ws_metadata();
    int check_total_allocation(uint64_t* limit, uint64_t used, uint64_t extra,
                               uint64_t const_size);

    void force_contiguous_allocate_mcrecv_blocks(
        const hnnx::VtcmCacheInstance& vtcm, const std::vector<uint32_t>& tags);

    void make_persistent_pools();
    void make_replaceable_persistent_pool();
    void serialize_pools(hnnx::Serializer& ser);
    void serialize_replaceable_mempool(hnnx::Serializer& ser);

    void map_plain_block(const void* block);
    void map_plain_block_to_pool_offs(const void* block, uint32_t* pool_id, uint64_t* offset);
    void* rewrite_to_physical_offset(uint64_t virtual_offset);
    void set_placeholder_mapping(const void* placeholder, void* actual);
    void get_blocks_pool_and_location(const void* block, uint32_t* pool, uint64_t* location);
    void pointers_to_poolid_offs();
    void* get_gather_desc_for_const_mempool(uint32_t pool_id);
    void set_mode(int mode);

private:
    std::vector<Pool> pools_;
    std::unordered_map<const void*, MemBlock*> block_map_;
    int mode_ = 0;
    std::vector<TakenRange> taken_ranges_;
    void find_live_taken_ranges(int start_idx, int end_idx);
};

} // namespace fa

// ✅ REQNN: 每个 NSP 的 VTCM 实例
class VtcmCacheInstance {
public:
    VtcmCacheInstance(uint32_t nsp_id, size_t vtcm_size);
    ~VtcmCacheInstance();
    size_t size() const { return vtcm_size_; }
    uint32_t nsp_id() const { return nsp_id_; }
    fa::FancyAllocator& allocator() { return allocator_; }
private:
    uint32_t nsp_id_;
    size_t vtcm_size_;
    fa::FancyAllocator allocator_;
};

// 🔶 GCP 4.x CBS: 多 NSP bank 划分参数（8 bank 均分给 num_nsps）
struct NspBankMap {
    uint32_t num_nsps;
    uint32_t banks_per_nsp;      // 8 / num_nsps
    uint32_t bank_size;          // vtcm_size / 8
    std::vector<uint8_t> nsp_of_bank;  // bank -> nsp
};

// ============================================================================
// 9. 调度（DP Sequencer / SVF / LVF / MLH）
// ============================================================================

// ✅ REQNN: 调度层级
enum class SequencerLevel { SVF0, SVF1, SVF2, LVF };

// ✅ REQNN: DP Sequencer 配置（50+ seq_sf_* 参数）
struct SequencerConfig {
    std::string algo_selector;
    std::string resequencer;
    bool svf_en = true;
    bool svf_abort_en = false;
    uint32_t lvf_parallelism_cfg = 0;
    uint32_t svf0_parallelism_cfg = 0;
    uint32_t svf1_parallelism_cfg = 0;
    uint32_t svf2_parallelism_cfg = 0;
    uint32_t parallelism_pull_limit = 0;
    bool mitigate_tcm_pressure = false;
    std::string tcm_pressure_dist;
    float mlh_svf_max_tcm_ratio = 0.0f;
    float mlh_lvf_max_tcm_ratio = 0.0f;
    std::vector<float> mlh_lvf_tcm_reduction_list;
    float mlh_ddr_ratio = 0.0f;
    std::string mlh_svf0_dma_cfg;
    std::string mlh_svf1_dma_cfg;
    std::string mlh_svf2_dma_cfg;
    std::string mlh_lvf_dma_cfg;
    bool mlh_parallelism_en = false;
    bool mlh_training_mode = false;
    bool mlh_training_feature_dump = false;
    std::string mlh_training_csv = "network_and_device_ml_features.csv";
    bool mlh_update_precomputed_keys = false;
    bool debug_mlh_verify = false;
    bool debug_mlh_terminate = false;
    std::string debug_mlh_model;
    uint32_t dp_greedy_fallback_threshold = 0;
    bool dp_popular_groups_en = false;
    bool dp_splithist_based_offsets_en = false;
    uint32_t dp_sg_mapping_cost = 0;
    bool dp_sg_reorder = false;
    uint32_t dp_sg_reorder_set_sel = 0;
    uint32_t dp_reorder_cost = 0;
    uint32_t dp_sg_reorder_threshold = 0;
    bool dp_early_exit_en = false;
    float heuristic_select_confidence_threshold = 0.0f;
    bool external_sequencer = false;
    std::string selected_sequencer;
    std::string sequencer_py_path = "scripts/sequencer.py";
    float sched_threshold_ratio = 1.0f;
    float sched_lower_threshold_ratio = 0.0f;
    uint32_t sched_timeout = 0;
    uint32_t sched_outer_timeout = 0;
    uint32_t sched_full_retries = 0;
    bool sched_afterburner = false;
    bool sched_abort_on_mistake = false;
    bool sched_early_out = false;
    bool sched_hint_depthwise = false;
    bool sched_delay_dma = false;
    uint32_t vtcm_retention = 0;
    uint32_t spill_fill_buffer_sizes = 0;
};

// ✅ REQNN: DP op 图
struct DPOpNode {
    op_id_t op_id;
    std::vector<DPOpNode*> predecessors;
    std::vector<DPOpNode*> successors;
    uint64_t vtcm_requirement;
    uint64_t ddr_requirement;
    uint32_t nsp_assignment;
    SequencerLevel level;
};

struct DPOpGraph {
    std::vector<DPOpNode> nodes;
    std::vector<std::vector<DPOpNode*>> groups; // subgroups
    std::vector<float> get_op_stats(const std::vector<DPOpNode*>& nodes) const;
};

// ✅ REQNN: XGBoost 决策树（MLH 启发式）
struct XGBDecisionTreeNode {
    int feature_index = -1;
    float threshold = 0.0f;
    int left_child = -1;
    int right_child = -1;
    float leaf_value = 0.0f;
    bool is_leaf = false;
};

class XGBDecisionTree {
public:
    XGBDecisionTree();
    ~XGBDecisionTree();
    float predict(const std::vector<float>& features) const;
    void load(const std::vector<XGBDecisionTreeNode>& nodes);
private:
    std::vector<XGBDecisionTreeNode> nodes_;
};

// ✅ REQNN: MLH 模型（⚠️ 模型权重未嵌入, 缺失时 greedy fallback）
class MLHModel {
public:
    MLHModel();
    ~MLHModel();
    struct MLHDecision {
        SequencerLevel level;
        std::string dma_config;
        uint32_t parallelism;
    };
    MLHDecision select(const std::vector<float>& features) const;
    MLHDecision greedy_fallback(const DPOpGraph& graph, uint32_t threshold) const;
    void dump_features(const std::vector<float>& features, const std::string& csv_path) const;
private:
    std::vector<XGBDecisionTree> trees_;
    bool model_loaded_ = false;
};

// ✅ REQNN: DP Sequencer
class DPSequencer {
public:
    DPSequencer();
    ~DPSequencer();
    std::vector<op_id_t> sequence(const DPOpGraph& graph,
                                  const SequencerConfig& config,
                                  const MLHModel& mlh_model);
    std::vector<op_id_t> external_sequence(const DPOpGraph& graph,
                                           const std::string& script_path);
    enum class ReorderMode { MODE_1_1_2_1_0, MODE_1_1_1_1_1, MODE_2_4_4_4_0 };
    ReorderMode predict_reorder_mode(const DPOpGraph& graph) const;
private:
    SequencerConfig config_;
    MLHModel mlh_model_;
    std::vector<op_id_t> run_svf(const DPOpGraph& graph, SequencerLevel level,
                                 uint32_t parallelism, const std::string& dma_cfg);
    std::vector<op_id_t> run_lvf(const DPOpGraph& graph, uint32_t parallelism,
                                 const std::string& dma_cfg);
};

// ============================================================================
// 10. DMA（Spill / Fill / 多核广播 / 同步）
// ============================================================================

// ✅ REQNN: DMA op 类型（与真实 runlist 字符串一致）
enum class DmaOpType : uint32_t {
    Spill,
    Fill,
    SpillWithDB,    // 双缓冲 spill
    FillWithDB,     // 双缓冲 fill
    MCSend,
    MCRecvRdy,
    MCRecvDone,
    MCRecvDone2,
    ChunkPreload,   // DMA 预加载
    MSyncPost,
    MSyncWait,
    HVXSpawn,       // HVX fork/join
};

// ✅ REQNN: DMA op 信息
// 🔶 GCP 5.2: 补充 synctoken_id / 2D stride / flags 优先级
struct DmaOpInfo {
    DmaOpType type;
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t size;
    uint32_t src_nsp;
    uint32_t dst_nsp;
    uint32_t mcid;
    uint64_t payload_size;
    bool is_multicast;
    bool double_buffered;

    // 🔶 GCP 5.2 补充
    uint32_t synctoken_id = 0;  // 同步标记 ID
};

// 🔶 GCP 5.2: PortableDMA（约 64 字节, DMA 描述符标准表示）
struct PortableDMA {
    uint64_t src_ptr;        // 源地址（DDR 或 VTCM 偏移）
    uint64_t dst_ptr;        // 目标地址
    uint32_t size;           // 传输字节数
    uint16_t flags;          // 1D/2D 模式等
    uint16_t config;         // 优先级等配置
    uint32_t synctoken_id;   // 同步标记 ID
    // 2D 传输专用
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t width;
    uint32_t height;
};

// DMA 模式标志（GCP 5.2）
constexpr uint16_t DMA_MODE_1D = 0x0000;
constexpr uint16_t DMA_MODE_2D = 0x0001;
constexpr uint16_t DMA_MODE_FILL = 0x0010;   // DDR -> VTCM
constexpr uint16_t DMA_MODE_SPILL = 0x0020;  // VTCM -> DDR
constexpr uint16_t DMA_PRIORITY_HIGH = 0x0100;

// ✅ REQNN: OpEmitter（反汇编锚点 op_emitter @0x1048320, 28264B）
class OpEmitter {
public:
    OpEmitter(class GraphPrepare* gp);
    ~OpEmitter();
    void emit_dma_op(const DmaOpInfo& info, size_t position);
    void insert_preload_op(size_t position, size_t prev_position);
    void insert_spill_fill_pair(uint64_t vtcm_offset, uint64_t ddr_offset, uint64_t size,
                                size_t spill_pos, size_t fill_pos, bool double_buffered);
    void insert_mcast_pair(uint32_t src_nsp, uint32_t dst_nsp, uint32_t mcid,
                           uint64_t payload_size, size_t send_pos, size_t recv_pos);
    void insert_msync_post(size_t pos);
    void insert_msync_wait(size_t pos);
    void insert_hvx_spawn(size_t fork_pos, size_t join_pos);
    bool validate_spill_fill() const;
    void combine_fills();
    void link_source_destructive_operands(const std::vector<uint32_t>& tags);
private:
    class GraphPrepare* gp_;
    std::vector<DmaOpInfo> emitted_ops_;
    uint64_t spill_fill_buffer_size_ = 0;
    uint64_t min_spill_fill_buffer_size() const;
};

// ✅ REQNN: Spill/Fill 调度器（⚠️ 压力曲线计算简化）
class SpillFillScheduler {
public:
    SpillFillScheduler();
    ~SpillFillScheduler();
    struct SpillFillPlan {
        size_t spill_position;
        size_t fill_position;
        uint64_t vtcm_offset;
        uint64_t ddr_offset;
        uint64_t size;
        bool double_buffered;
    };
    std::vector<SpillFillPlan> plan(const std::vector<Op*>& runlist,
                                    const fa::FancyAllocator& allocator,
                                    size_t vtcm_size);
    uint64_t min_buffer_size(const std::vector<Op*>& runlist,
                             const fa::FancyAllocator& allocator) const;
private:
    struct VtcmPressurePoint {
        size_t op_index;
        uint64_t current_usage;
        uint64_t peak_usage;
    };
    std::vector<VtcmPressurePoint> compute_pressure_curve(
        const std::vector<Op*>& runlist, const fa::FancyAllocator& allocator) const;
};

// ============================================================================
// 11. 同步原语（Synctoken / Barrier）
// ============================================================================

// 🔶 GCP 5.7: Synctoken 管理器（REQNN 缺失, 联合调度的关键变量）
class SynctokenManager {
public:
    uint32_t allocate() { return next_id_++; }
    void signal(uint32_t id) { tokens_[id] = true; }
    void wait(uint32_t id) { while (!tokens_[id]) {} }
    bool poll(uint32_t id) { return tokens_[id]; }
    void barrier(const std::vector<uint32_t>& tokens) {
        uint32_t bt = allocate();
        for (auto t : tokens) wait(t);
        signal(bt);
    }
private:
    std::atomic<uint32_t> next_id_{0};
    std::unordered_map<uint32_t, std::atomic<bool>> tokens_;
};

// ============================================================================
// 12. 多核广播（Multicast）
// ============================================================================

// ✅ REQNN: McSend / SuperCast / ILP 输入
struct McSend {
    uint32_t tag;
    uint32_t sender_nsp;
    uint32_t num_mcids;
    uint64_t payload_size;
    std::vector<uint32_t> mcids;
    std::vector<uint32_t> receivers;
};

struct SuperCast {
    std::vector<uint32_t> mcsend_tags;
    std::vector<uint32_t> mcids;
    uint32_t sender_nsp;
    uint64_t total_payload;
};

struct MCastLPInput {
    std::vector<McSend> mcsends;
    std::vector<SuperCast> supercasts;
    uint32_t max_mcsend_tag;
    uint32_t graph_multicast_count;
};

// ✅ REQNN: McastOptimizer（⚠️ 实际为贪心两两合并, HiGHS ILP 未接）
// 反汇编锚点: top @0x1075FF0, LP builder @0x1176340
class McastOptimizer {
public:
    McastOptimizer();
    ~McastOptimizer();
    std::vector<McSend> optimize(const std::vector<McSend>& mcsends,
                                 uint32_t graph_multicast_count);
    MCastLPInput build_lp_input(const std::vector<McSend>& mcsends,
                                uint32_t graph_multicast_count);
    bool solve_ilp(const MCastLPInput& input);
    std::vector<McSend> apply_results(const std::vector<McSend>& original,
                                      const std::vector<SuperCast>& supercasts);
    void dump_mps(const std::string& filename) const;
    void set_run_crossover(bool enable) { run_crossover_ = enable; }
    void set_dump_mps(bool enable) { dump_mps_ = enable; }
private:
    bool run_crossover_ = false;
    bool dump_mps_ = false;
    struct SimplexSolver {
        bool create_and_populate(const MCastLPInput& input);
        bool solve();
        void get_solution(std::vector<double>& primal, std::vector<double>& dual,
                          int& status, int& iterations, double& objective) const;
    };
    SimplexSolver solver_;
    int num_variables_ = 0;
    int num_constraints_ = 0;
    std::vector<double> objective_coeffs_;
    std::vector<std::vector<double>> constraint_matrix_;
    std::vector<double> constraint_lb_;
    std::vector<double> constraint_ub_;
    std::vector<double> var_lb_;
    std::vector<double> var_ub_;
};

// ============================================================================
// 13. RunList（执行计划）
// ============================================================================

// ✅ REQNN: runlist tags（调度器输出, 反汇编锚点 do_prepare2_late）
// 🔶 GCP 6.5: RunListEntry 执行条目（REQNN 缺失）
struct RunListEntry {
    enum Type : uint32_t {
        OP = 0,        // 执行算子
        DMA_WAIT = 1,  // 等待 DMA 完成
        BARRIER = 2,   // 同步屏障
        NSP_SYNC = 3,  // 多 NSP 同步
        PROFILING = 4, // 性能分析点
    };
    uint32_t type;       // OP / DMA_WAIT / BARRIER / NSP_SYNC / PROFILING
    uint32_t data;       // Op 索引或 DMA token
    uint32_t synctoken;  // 用于同步
    uint32_t padding;    // 对齐到 16 字节
};

// ============================================================================
// 14. 成本模型（Cost Model）
// ============================================================================

// ✅ REQNN: 推理模式
struct InferenceMode {
    enum Type { Performance, Power, Bandwidth } type = Performance;
};

// ✅ REQNN: grdep::OpDesc
namespace grdep {
    struct OpDesc {
        std::string op_name;
        uint32_t op_type;
        std::vector<uint64_t> input_dims;
        std::vector<uint64_t> output_dims;
        uint32_t nsp_count;
        uint32_t vtcm_budget;
    };
}

// ✅ REQNN: CostSource（⚠️ 每元素常数表, MLP 未真正加载）
namespace costbased {
    class CostSource {
    public:
        CostSource();
        ~CostSource();
        bool init_for_soc(const std::string& soc_type);
        float get_prediction_from_cost_model(
            const std::string& op_name, const Op* op,
            const grdep::OpDesc* desc, InferenceMode mode) const;
        struct MLP {
            std::vector<std::vector<float>> weights;
            std::vector<std::vector<float>> biases;
            std::vector<float> predict(const std::vector<float>& features) const;
        };
        std::vector<float> extract_features(const Op* op,
                                            const grdep::OpDesc* desc,
                                            InferenceMode mode) const;
    private:
        bool initialized_ = false;
        std::string soc_type_;
        MLP mlp_model_;
        std::unordered_map<std::string, float> cost_table_;
        void load_cost_table();
        void load_mlp_model();
    };

    class CostBasedScheduler {
    public:
        CostBasedScheduler(CostSource& cost_source);
        ~CostBasedScheduler();
        std::vector<Op*> schedule(const std::vector<Op*>& ops,
                                  const Graph* graph, InferenceMode mode) const;
    private:
        CostSource& cost_source_;
    };

    class HextimateSimulator {
    public:
        HextimateSimulator();
        ~HextimateSimulator();
        uint64_t simulate(const std::vector<Op*>& runlist,
                          const Graph* graph, uint32_t num_nsps) const;
        void enable_chrome_trace(const std::string& filename);
        void disable_chrome_trace();
    private:
        bool trace_enabled_ = false;
        std::string trace_filename_;
    };
}

// ============================================================================
// 15. 优化 Pipeline（八阶段 / 融合 / Fixpoint）
// ============================================================================

// ✅ REQNN: 优化 phase 阈值（GCP 3.2 一致）
// 🔶 补充 PHASE_6 = 24999（GCP 有 8 相, REQNN 只到 6 相）
enum OptPhase : uint32_t {
    PHASE_0    = 3000,     // 0xBB8  常量折叠
    PHASE_1    = 10190,    // 0x27CE 形状归一化
    PHASE_2    = 11892,    // 0x2E7C 量化折叠 + tcm_migration
    PHASE_3    = 12492,    // 0x30D4 激活融合
    PHASE_4    = 21101,    // 0x526D TCM 重写
    PHASE_5    = 22000,    // 0x55F0 布局转换
    PHASE_6    = 24999,    // 0x61A7 窥孔优化 (🔶 REQNN 缺失)
    PHASE_TERM = 0xFFFFFFFF,
};

// ✅ REQNN: phase 描述符（0x40 字节） / GraphOptInfo（0x80 字节）
struct PhaseDescriptor {
    uint32_t threshold;
    void* vtable;
    void* graph;
    uint64_t reserved;
    void* list_self;
    uint64_t reserved2[3];
};

struct GraphOptInfo {
    void* vtable;        // +0x00
    uint32_t phase;      // +0x08
    uint32_t id;         // +0x0C
    void* defopt_fn;     // +0x10
    void* matcher_desc;  // +0x18 -- matcher 数组
    // ... 总 0x80 字节
    GraphOptInfo(uint32_t phase, uint32_t id, void* defopt_fn, const char* name);
    ~GraphOptInfo();
    op_hash_t get_hash_key() const;
    uint16_t get_min_inputs() const;
    uint16_t get_max_inputs() const;
};

struct GraphOptPass {
    struct MatchIterator {
        void* vtable;
        void* match_state;
        const GraphOptInfo* current_rule;
        bool has_match;
        void advance();
        void advance_select();
        bool next();
    };
    struct MatcherState {
        std::vector<void*> matched_ops;
    };
    std::vector<GraphOptInfo*> rules_;
    void* hash_table_ = nullptr;
    size_t hash_table_size_ = 0;
    void add_optim(GraphOptInfo* info);
    void build_matchers();
    MatchIterator begin_match(const OpDef* opdef) const;
};

// ✅ REQNN: 融合规则（8 条结构级, 真实库有几十条带 op 语义 matcher）
struct FusionRule {
    const char* producer;
    const char* consumer;
    const char* fused;
};

int apply_fusion_rules(class GraphPrepare* gp, const std::vector<FusionRule>& rules);

// ============================================================================
// 16. 指令选择（HwWrapper）
// ============================================================================

// ✅ REQNN: 硬件 wrapper 选择
enum class HwWrapper : uint8_t {
    HVX_Vector    = 0,
    HVX_Scalar    = 1,
    HMX_Matrix    = 2,
    HMX_MatrixInt4 = 3,
    Ref_Host      = 0xFE,
};
HwWrapper select_wrapper(const std::string& op_type, DType dtype, uint32_t soc_type);

// ============================================================================
// 17. 量化（Quantization）
// ============================================================================

// ✅ REQNN: 量化类型与参数
enum class QuantType : uint32_t {
    None = 0, PerTensor = 1, PerChannel = 2, PerAxis = 3,
    Block = 4, BwAxis = 5,
};

struct QuantParams {
    QuantType type = QuantType::None;
    float scale = 1.0f;
    int32_t offset = 0;
    uint32_t bitwidth = 8;
    uint32_t axis = 0;
    std::vector<float> per_channel_scales;
    std::vector<int32_t> per_channel_offsets;
    bool is_signed = true;
};

class Quantizer {
public:
    std::vector<int8_t>  quantize_int8(const float* data, size_t count,
                                       const QuantParams& params) const;
    std::vector<int16_t> quantize_int16(const float* data, size_t count,
                                        const QuantParams& params) const;
    std::vector<int32_t> quantize_int32(const float* data, size_t count,
                                        const QuantParams& params) const;
    std::vector<float> dequantize(const void* data, size_t count,
                                  DType dtype, const QuantParams& params) const;
    std::vector<int8_t> requantize(const void* input, size_t count,
                                   DType in_dtype, const QuantParams& in_params,
                                   const QuantParams& out_params) const;
    QuantParams compute_params(const float* data, size_t count,
                               DType target_dtype, QuantType type,
                               uint32_t axis = 0) const;
    bool validate(const QuantParams& params) const;
};

// ============================================================================
// 18. 权重处理（Weights / 共享）
// ============================================================================

// ✅ REQNN: 权重描述符与处理器
struct WeightDescriptor {
    DType dtype;
    uint32_t rank;
    uint64_t dims[5];
    uint64_t element_size;
    uint64_t total_size;
    bool is_const;
    bool is_shared;
    bool is_composed;
    uint32_t wtshare_tag;
};

class WeightProcessor {
public:
    struct WtshareMetadata {
        uint32_t tag;
        uint32_t nsp_id;
        uint64_t offset;
        uint64_t size;
        std::string blockref;
    };
    std::vector<uint8_t> compose(const std::vector<std::vector<uint8_t>>& parts,
                                 const WeightDescriptor& desc) const;
    std::vector<uint8_t> convert_conv_weights(const void* weights,
                                              const WeightDescriptor& src_desc,
                                              const WeightDescriptor& dst_desc) const;
    std::vector<std::vector<uint8_t>> scatter_conv_weights(
        const void* weights, size_t total_size, uint32_t num_nsps) const;
    std::vector<WtshareMetadata> compute_sharing_plan(
        const std::vector<WeightDescriptor>& weights, uint32_t num_nsps) const;
    std::vector<uint8_t> serialize_for_pickle(const void* weights, size_t size,
                                              const WeightDescriptor& desc) const;
    std::vector<uint8_t> deserialize_from_pickle(const void* data, size_t size,
                                                 WeightDescriptor& out_desc) const;
    void dump(const std::string& filename, const void* weights,
              size_t size, const WeightDescriptor& desc) const;
};

// ============================================================================
// 19. Op 基类（TypicalOp）
// ============================================================================

// ✅ REQNN: 基类 Op 与 TypicalOp
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

class TypicalOp : public Op {
public:
    TypicalOp() { graph = nullptr; }
    float cost(const Graph* g) const override;                 // ⚠️ 简化
    void serialize_internal(class Serializer&, int) const override;
    std::string op_type_name;
    std::vector<uint8_t> params;
    OutputDef cached_out_def{};
    HwWrapper wrapper = HwWrapper::Ref_Host;
    virtual void execute(const std::vector<const uint8_t*>& inputs,
                         uint8_t* output,
                         const struct OutputDef& out_def) const;  // ⚠️ host 简化直通
};

// ============================================================================
// 20. 序列化（.bin 格式）
// ============================================================================

// ✅ REQNN: tag 编码 (tag&0xFFFF | tag<<16) ^ 0xFFFF
uint32_t encode_bin_tag(uint32_t tag);
uint32_t decode_bin_tag(uint32_t encoded);

// ✅ REQNN: 已知 tag ID（来自 do_serialize 反编译）
enum BinFormatTag : uint32_t {
    TAG_IO_DMA_BYPASS        = 0xEF4D,
    TAG_SPILL_FILL_INSTEAD   = 0x4453,
    TAG_EXTENDED_UDMA        = 0xD446,
    TAG_IO_TENSORS_CONFIG    = 0xE347,
    TAG_EXTRA_CONFIG         = 0xD352,
    TAG_MULTICAST_CONFIG     = 0xD349,
    TAG_NUM_SEGMENTS         = 0x5248,
    TAG_VEC_RUNLIST_COUNT    = 0x524C,
    TAG_RUNLIST_SEGMENT_DESC = 0x5647,
    TAG_SELF_SLICING         = 0xC953,
    TAG_SLICING_CONFIG       = 18000,  // 0x4650
    TAG_DEHYDRATION          = 0xC955,
    TAG_CONST_EXTENT         = 0xCF55,
    TAG_MC_CACHEABLE         = 0x5350,
    TAG_RUNLIST_AUX          = 0x5453,
    TAG_INTERFACE_BASELINE   = 0xE358,
    // REQNN 自定义 tag（非真实 QNN 兼容）
    TAG_OP_RECORD            = 0x4F50,
    TAG_IO_TENSOR_DESC       = 0x494F,
    TAG_GRAPH_HEADER         = 0x4748,
};

// ✅ REQNN: 内存内分隔符（DSP 内存内, 不在 .bin 文件里 —— BFA 证实）
// serializer 用 tagged-record 流写 contextBlob 内部, 每一条记录前后用这些标记。 
constexpr uint32_t SEPARATOR_NORMAL = 0xFA0000FA;
constexpr uint32_t SEPARATOR_AUX    = 0xFA0000FE;
constexpr uint32_t SEPARATOR_END    = 0xBEEFF00D;

// ========================================================================
// .bin 文件结构（两层: host 小端系统信息头 → DSP 大端 context blob）
//
// ┌─ .bin 文件 ───────────────────────────────────────────────────┐
// │ [小端] SystemInfoHeader (64B)                                 │
// │   version_major/minor, num_graphs, graph_data_size(╱偏移表大小), │
// │   weights_offset → 指向实际权重数据                              │
// │                                                                │
// │ [大端] contextBlob (从 graph_data_offset 开始)                  │
// │   ┌─ BinHeader (28B): count + offset 表 ───────────────────┐  │
// │   │   num_graphs(大端=2), num_records(大端=3), flags,       │  │
// │   │   offset_table → 指向 graphBlobInfoV2 数组               │  │
// │   │   block_size(=1MB), base_address(=DSP地址)              │  │
// │   ├─ graphBlobInfoV2[]: 每条 {偏移对, 绝对基址} ──────────┤  │
// │   ├─ 各段数据 (大端):                                       │  │
// │   │   opData / ioTensor / const  / spillfill / ...         │  │
// │   └─────────────────────────────────────────────────────────┘  │
// └────────────────────────────────────────────────────────────────┘
// ========================================================================

// ✅ 【外层】系统信息头（小端, host 侧加载用）
// BFA 描述的 64B 头部已确认就是这一层, 是 host 加载器定位 contextBlob + weights 的入口。
// * version_major=2, version_minor=3 （Qwen3.5-4B 样本）
// * num_graphs=1, graph_data_size=0x5630 (=偏移表大小)
// * weights_offset=0x6000 （指向权重数据在文件中的偏移）
// 参考: SERIALIZED_BIN_ANALYSIS_GUIDE §3.1 + 用户确认
struct SystemInfoHeader {
    uint32_t version_major;  // +0x00 小端, e.g. 2
    uint32_t version_minor;  // +0x04 小端, e.g. 3
    uint32_t platform_id;    // +0x08 保留 / 平台标识
    uint32_t num_graphs;     // +0x0C 小端, e.g. 1
    uint32_t graph_data_offset; // +0x10 contextBlob 在文件内的偏移
    uint32_t graph_data_size;   // +0x14 contextBlob 字节数
    uint64_t weights_offset;    // +0x18 权重数据起始偏移
    uint32_t reserved2[4];      // +0x20
};

// ✅ 【内层】contextBlob 头部（大端, DSP 原生格式）
// BFA 从 test_minimal.serialized.bin 头部确认:
//   num_graphs(大端)=2, num_records(大端)=3, block_size=1MB, base_address=0x2E5E9000
struct BinHeader {
    uint32_t num_graphs;    // [0x00] 大端, 图的个数
    uint32_t num_records;   // [0x04] 大端, graphBlobInfoV2 条目数
    uint32_t flags;         // [0x08] 标志位
    uint32_t version;       // [0x0C] 版本
    uint64_t offset_table;  // [0x10] 大端: graphBlobInfoV2 数组偏移 (相对于 contextBlob 基址)
    uint64_t block_size;    // [0x18] 大端: 块大小 (样本=0x100000=1MB)
    uint64_t base_address;  // [0x20] 大端: DSP 映射基址 (样本=0x2E5E9000)
};

// 🔶 graphBlobInfoV2: 偏移描述符（记录内的子段偏移, BFA switch(0-11) 跳转表按此解析）
struct GraphBlobInfoV2 {
    uint64_t data_offset;   // 大端: 段偏移 (相对于 contextBlob 基址)
    uint64_t data_size;     // 大端: 段大小
    // 每个 num_records 对应一条 graphBlobInfoV2
    // hexagon_nn_deserialize_graph @0x5ef880 switch(0-11) 据此分派
};

// 🔶 GUIDE 4.1: 字符串表（真实格式用 str_idx 引用, REQNN 无此结构）
struct StringTable {
    std::vector<std::string> strings;      // 反序列化后索引
    std::unordered_map<std::string, uint32_t> idx_of;  // 序列化时去重
    uint32_t intern(const std::string& s) {
        auto it = idx_of.find(s);
        if (it != idx_of.end()) return it->second;
        uint32_t i = (uint32_t)strings.size();
        strings.push_back(s);
        idx_of[s] = i;
        return i;
    }
};

// ✅ REQNN: 类型注册表（反序列化时找到 tensor/op 构造函数）
constexpr uint32_t TYPE_MAGIC_1 = 0x71A6009B;
constexpr uint32_t TYPE_MAGIC_2 = 0xEBC0FEFE;

struct ClassIndexEntry {
    uint32_t class_id;
    uint32_t name_len;
    // 后跟 name_len 字节类型名 (4 字节对齐)
};

// ✅ REQNN: 段 span / const extent 描述符
struct DeserSegmentSpan {
    uint32_t segment_index;
    uint32_t op_count;
    uint64_t byte_offset;
    uint64_t byte_size;
};

struct ConstExtentDesc {
    uint64_t op_id;
    uint64_t offset;      // const pool 偏移
    uint64_t size;
    uint32_t tensor_type; // ClassIndexEntry 类型 ID
    uint32_t reserved;
};

// ✅ REQNN: Serializer（双模式: prescan 计数 / write 写出）
class Serializer {
public:
    enum class Mode : int { Prescan = 0, Write = 1 };
    Serializer(class GraphPrepare* gp, void* allocator, char* buf, size_t buf_size);
    ~Serializer();

    void serialize_fwrite(const void* data, size_t size, bool align4);
    void serialize_uint32(uint32_t a, uint32_t b, uint32_t c);
    void write_uint32(uint32_t val) {
        if (mode_ == Mode::Write) {
            *reinterpret_cast<uint32_t*>(buf_cur_) = val;
            buf_cur_ += 4;
        }
        pos_ += 4;
    }
    void set_error(const char* msg) { error_msg_ = msg; }
    bool has_error() const { return error_msg_ != nullptr; }
    size_t current_position() const { return pos_; }
    void set_mode(Mode m) { mode_ = m; }
    void write_tagged_record(uint32_t tag, const void* data, int data_size);
    void prescan_tensor(const Tensor* tensor);
    void serialize_single_tensor_pointer(const void* tensor_ptr);
    void graph_io_tensors(const struct uptr_DWDI<Tensor>* inputs, uint32_t num_inputs,
                          const struct uptr_DWDI<Tensor>* outputs, uint32_t num_outputs, bool);
    void prescan_ops_func(Op* const* ops, uint32_t count, bool);
    void do_insert_preload_op();
    void make_class_index_aux_record(
        const std::vector<std::pair<std::string_view, uint32_t>>& classes, bool);
    void make_runlist_segment_descs_aux_record(bool, int);
    void make_auxdata_for_self_slicing();
    void rewrite_auxdata(uint64_t offset, uint32_t size, const void* data, bool, bool);
    void supply_segment_index_positions(uint32_t);
    void setup_heap_info(class FancyAllocator* alloc, uint64_t total_size);
    void do_insert_overall_len();

private:
    class GraphPrepare* gp_;
    void* allocator_;
    char* buf_start_;
    char* buf_end_;
    char* buf_cur_;
    size_t pos_;
    size_t prev_pos_;
    int preload_count_;
    int state_;
    Mode mode_;
    bool in_runlist_;
    bool self_slicing_;
    const char* error_msg_ = nullptr;
};

// ✅ REQNN: 低层读取器
class Deserz {
public:
    Deserz(class Deserializer* parent, const char* data, size_t size, Graph* graph);
    ~Deserz();
    void deserialize_fread(void* dst, uint32_t size, bool align4);
    std::string deserialize_str();
    void deserialize_uint32_arr(uint32_t* arr, uint32_t count);
    struct Uint32x2 { uint32_t a, b; };
    struct Uint32x3 { uint32_t a, b, c; };
    struct Uint32x4 { uint32_t a, b, c, d; };
    Uint32x2 deserialize_uint32_x2();
    Uint32x3 deserialize_uint32_x3();
    Uint32x4 deserialize_uint32_x4();
    void deserialize_buf(uint32_t size, void* dst);
    void deserialize_buf_withlen(uint32_t size, void* dst);
    void* deserialize_shared_obj_func(void** out);
    const char* current() const { return cur_; }
    size_t remaining() const { return data_end_ - cur_; }
private:
    void* vtable;
    class Deserializer* parent_;
    Graph* graph_;
    const char* data_start_;
    const char* data_end_;
    const char* cur_;
    const char* page_end_;
};

// ✅ REQNN: Deserializer（真实主入口 hexagon_nn_deserialize_graph @0x5ef880）
class Deserializer {
public:
    Deserializer(const char* data, size_t size, Graph* graph);
    ~Deserializer();

    void load_header(size_t offset);
    void auxdata_deserialize_segments(uint32_t param);
    void auxdata_class_index(uint32_t tag, bool flag);
    void auxdata_temparr_sizes(uint32_t tag);
    void auxdata_read_const_extent_descriptor(uint32_t tag);
    void extract_const_extent_table(uint32_t tag);
    void extract_const_extent_data(uint64_t offset, uint32_t size, void* dst,
                                   uint64_t a, uint64_t b);
    void resize_object_tables(const struct runlist_auxdata_seg_desc&);
    void segmentjob_deserialize_ops(uint32_t seg_idx, uint32_t job_idx);
    void handle_auxdata_deser(uint32_t tag, uint32_t size);
    uint32_t crate_size_according_to_segments();
    void get_forward_span(uint32_t seg, uint32_t a, uint32_t b);
    void skip_to_after_span(const DeserSegmentSpan& span);
    void set_graph(Graph* g);
    const char* get_name(uint64_t a, uint64_t b);
    const BinHeader& header() const { return header_; }
    bool header_ok() const { return header_.num_graphs > 0 && header_.num_graphs < 100; }

private:
    void* vtable;
    BinHeader header_{};
    std::vector<ClassIndexEntry> class_index_;
    std::vector<DeserSegmentSpan> segment_spans_;
};

// ============================================================================
// 21. GraphPrepare（编译主控）
// ============================================================================

// ✅ REQNN: GraphPrepare（反汇编锚点 prepare @0xF80840, 字段偏移已确认）
class GraphPrepare {
public:
    GraphPrepare();
    ~GraphPrepare();

    enum class GraphStatus : int { OK = 0, FAIL = 1, SKIP = 2 };

    struct OrderInfo {
        void* vtable;
        op_id_t op_id;
        int ordering_index;
    };

    GraphStatus prepare(class HexagonNNEnv& env);
    GraphStatus do_prepare1(class HexagonNNEnv& env, VtcmCacheInstance& vtcm);
    GraphStatus do_prepare2(class HexagonNNEnv& env, VtcmCacheInstance& vtcm,
                            int& retry_count, bool full_prepare);
    GraphStatus do_prepare2_late(std::vector<uint32_t>& runlist_tags);

    op_id_t append_node(const std::string& name, uint32_t node_type,
                        const InputDef* inputs, size_t num_inputs,
                        const OutputDef* outputs, size_t num_outputs,
                        const uint8_t* ops_data);
    op_id_t append_const_node(uint32_t node_type, const OutputDef& od,
                              const uint8_t* data, size_t data_len);
    void insert_op(std::unique_ptr<Op> op, bool before);
    void erase_op(op_id_t id);
    void supersede_op(OpDef* old, op_id_t new_id, bool keep);
    void opdef_delete(op_id_t id);
    OpDef* get_op_at(op_id_t id) const;
    size_t op_count() const { return opdef_map_.size(); }
    void mark_op_deletable(OpDef* op);
    void mark_op_deletable(op_id_t id);
    void collect_deletable_nodes();

    void run_optimize_passes(class HexagonNNEnv& env);
    void run_optimize_passes_single_registry(
        class HexagonNNEnv& env, const std::map<uint32_t, GraphOptPass>& registries);
    void run_optimize_passes_multi_registry(class HexagonNNEnv& env);
    void dead_code_removal_and_cse();
    int remove_dead_code(bool);
    int order_nodes(bool);
    int common_subexpr_eliminate(bool);
    void const_prop(class HexagonNNEnv& env, bool aggressive);
    void const_prop_and_cse(class HexagonNNEnv& env, bool aggressive, bool* changed);
    void identify_updateable_quant_ops();
    void run_predication_pass();
    void tcm_migration(uint32_t threshold, bool aggressive);

    bool serialize(uint8_t* buf, size_t buf_size, size_t& out_size) const;
    bool serialize_file(int fd) const;
    bool do_serialize(Serializer& ser) const;
    void serialize_io(Serializer& ser, uint64_t& counter, bool is_prescan) const;
    void serialize_opdef(Serializer& ser, const OpDef& opdef) const;
    void adjust_heap_stats(Serializer& ser) const;
    bool serialize_patch_metadata(class FileSerializer& fs) const;
    bool serialize_replaceable_constpool(class FileSerializer& fs) const;
    bool deserialize(const uint8_t* buf, size_t buf_size);

    void pprint() const;
    void graphviz_pprint(const char* filename, bool full) const;
    void python_pprint_graph_summary(const char* filename, bool full, bool late) const;
    void python_pprint_detail() const;
    void python_pprint_runlist() const;
    uint32_t calculate_graph_checksum() const;
    bool check_connectivity() const;
    void sanity_check_null_exec(op_id_t id, const Op* op) const;
    void make_sorted_optrs();
    op_id_t lookup_op_in_ordering(OrderInfo* ordering, int idx, op_id_t id) const;

    void const_tracking_setup();
    void const_tracking_finalize();
    void const_tracking_after_prep(class HexagonNNEnv& env);
    void add_tracked_id(op_id_t id, const OpDef& opdef, bool force);

    bool is_dynamic_inputs_active() const;
    bool is_dynamic_dma(op_id_t id) const;
    void dynamic_inputs_pre_optimization_pass();
    void dynamic_inputs_post_optimization_pass();

    void gen_quant_params_hash(uint64_t& hash) const;
    void fixup_signed_activations(OutputDef& od) const;
    bool needs_activation_fixup(DType dt) const;
    std::unique_ptr<Op> op_factory_generate(const struct OpIoPtrs& io, op_id_t id);

    void allocate_io_tensors();
    void phys_alloc_in_runlist(const std::vector<Op*>& ops);
    void force_contiguous_allocate_mcrecv_blocks(const VtcmCacheInstance& vtcm,
                                                 const std::vector<uint32_t>& tags);
    void link_source_destructive_operands(const std::vector<uint32_t>& tags);
    void allocate_for_reschedule_grdep(const VtcmCacheInstance& vtcm,
                                       const std::vector<uint32_t>& tags, bool);

    void mark_time_point(const char* name);
    void log_time_points();
    void mark_prepare_stage(std::pair<std::string, uint64_t> stage);
    void clear_profiling_info();
    uint32_t num_profiling_timepoints(uint32_t* count) const;
    void serialize_profiling_timepoints(struct profilingevent* events, uint32_t count);

    void setup_rewrite_log();
    void close_rewrite_log();
    void add_replacement_recorder();
    void enable_replacement_recording();
    void start_replacement_recorder(void** recorder);
    void stop_replacement_recorder(void* recorder);
    op_id_t get_oldest_replaced_id(void* recorder, op_id_t id) const;

    size_t get_vtcm_tile_size() const;
    op_id_t get_pretiling_op_id(op_id_t id) const;
    op_id_t get_ct_op_id(op_id_t id) const;
    op_id_t extract_op(op_id_t id);
    void set_node_ids(uint32_t start, uint32_t end, uint32_t base);
    op_id_t op_def_posn(op_id_t id) const;
    void change_opstr(OpDef* opdef, string_tag_t* new_tag, const char* str, uint32_t len);
    void change_input(OpDef* opdef, uint32_t idx, op_id_t new_src, const char* str, uint32_t len);
    void replace_with(OpDef* old, op_id_t new_id, const char* str, uint32_t len, bool keep);
    void replace_opdef_with_opconst(OpDef& old, std::unique_ptr<OpDef> replacement);
    void note_new_node(const OpDef& opdef, const char* str, uint32_t len);
    void note_replace(op_id_t old, const std::vector<struct OpRef>& refs,
                      op_id_t new_id, uint32_t idx, const std::string& str);
    void note_outputs(void* file, const OpDef& opdef);
    void reapply_signed_activations(OutputDef& od);
    void supersede_outputless_op(OpDef* old, op_id_t new_id);
    void log_mux_match_fail(const GraphOptInfo& info, op_id_t a, op_id_t b, string_tag_t* tag);
    op_id_t update_tensor_map_with_duplicates(op_id_t id, const std::vector<OpDef*>& ops);
    op_id_t update_tensor_map_with_combined(op_id_t id, const std::vector<OpDef*>& ops);
    void set_wtshare_metadata_filename(const char* filename);
    void serialize_sharing_metadata(const char* filename) const;
    void remove_input_refs(op_id_t id);
    bool truegraph_is_needed(const OpDef* opdef) const;
    op_id_t truegraph_source_id(struct OpRef ref) const;
    void truegraph_outputdefs(const OpDef* opdef) const;
    bool is_native_KV_success() const;
    void debug_grdeps(GraphDeps& deps, const char* name);

    void pysequencer(std::vector<uint32_t>& runlist_tags);
    void show_runlist(GraphDeps& deps, const std::vector<uint32_t>& tags, int idx) const;
    void dump_runlist(GraphDeps& deps, const std::vector<uint32_t>& tags, const char* filename) const;
    void get_nsp_id_mapping(struct NspIdMap& map) const;

    bool is_moe_aggregator(op_id_t id) const;
    bool is_part_of_moe_block(op_id_t id) const;
    void get_moe_block_ops_grouped_by_branch(uint16_t block_id) const;
    void get_all_moe_block_ids() const;
    bool multi_quant_transformed_to_single_quant() const;
    void get_dynamically_switchable_blocks() const;

private:
    // 字段偏移（反汇编确认）
    // +0x45dc: construction_state (0=init,1=construction,2=optimization)
    // +0x7311: graph_dirty
    // +0x7468: graph_deps
    // +0x5d18: memory_alloc_limit (MB)

    std::unordered_map<op_id_t, std::unique_ptr<OpDef>> opdef_map_;
    std::vector<std::unique_ptr<Op>> ops_;
    std::map<uint32_t, GraphOptPass> optimization_registry_;

    std::vector<uint8_t> const_pool_;
    struct ConstExtent {
        op_id_t op_id;
        uint64_t offset;
        uint64_t size;
    };
    std::vector<ConstExtent> const_extents_;

    // ✅ BlockEntry（⚠️ 真实 BlockTableEntry 需含 bank/storage_class/spill, 见 §5）
    struct BlockEntry {
        uint32_t block_id;
        uint32_t pool_id;
        uint64_t offset;
        uint64_t size;
    };
    std::vector<BlockEntry> block_table_;

    struct SegmentPlan {
        uint32_t segment_index;
        uint32_t op_count;
        uint64_t byte_offset;
    };
    std::vector<SegmentPlan> segment_plans_;

    bool graph_dirty_ = false;
    bool serialized_loaded_ = false;
    bool force_barrel_ = false;
    void* allocator_ = nullptr;
    void* op_registry_ = nullptr;
    void* graph_deps_ = nullptr;
    uint32_t early_out_flag_ = 0;
    uint32_t memory_alloc_limit_mb_ = 0;
    op_id_t next_op_id_ = 1;
    int construction_state_ = 0;
    op_id_t input_node_id_ = 0;
    op_id_t output_node_id_ = 0;
    std::vector<op_id_t> ordering_;
    struct TimePoint { const char* name; uint64_t timestamp; };
    std::vector<TimePoint> time_points_;
    void run_phase_fixpoint_internal();
};

} // namespace hnnx
