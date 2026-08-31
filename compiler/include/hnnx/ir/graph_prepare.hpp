#pragma once
#include "hnnx/ir/types.hpp"
#include "hnnx/tiling/tiler.hpp"  // TilingConfig(阶段6 单算子分块)
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/vtcm/cp_solver.hpp"
#include "hnnx/mcast/mcast_optimizer.hpp"
#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace hnnx {

class HexagonNNEnv;
class VtcmCacheInstance;
class GraphDeps;
class DPGroupGraph;
class SuperTileSolver;
class OpEmitter;
struct SuperTile;

struct GraphOptInfo;
struct GraphOptPass;
struct GraphOptContext;

struct OrderInfo {
    void* vtable;
    op_id_t op_id;
    int ordering_index;
};

// Const 池 extent 描述(从 GraphPrepare 私有区提升到命名空间级,供公开访问器使用)
struct ConstExtent {
    op_id_t op_id;
    uint64_t offset;  // 在 const_pool_ 中的字节偏移
    uint64_t size;    // 字节数
};

class GraphPrepare {
public:
    GraphPrepare();
    ~GraphPrepare();

    // Core lifecycle
    GraphStatus prepare(HexagonNNEnv& env);
    GraphStatus do_prepare1(HexagonNNEnv& env, VtcmCacheInstance& vtcm);
    GraphStatus do_prepare2(HexagonNNEnv& env, VtcmCacheInstance& vtcm,
                           int& retry_count, bool full_prepare);
    GraphStatus do_prepare2_late(std::vector<uint32_t>& runlist_tags);

    // Accessors for execution
    op_id_t get_input_node_id() const { return input_node_id_; }
    op_id_t get_output_node_id() const { return output_node_id_; }
    const std::vector<op_id_t>& get_ordering() const { return ordering_; }
    // Scheduler(ST-Cut)计划执行序: do_prepare2 计算, do_serialize 按此发射 op 记录
    const std::vector<op_id_t>& plan_order() const { return plan_order_; }
    // 单算子分块配置(阶段6): Conv2d extra_info tiling 段消费
    void set_tiling_config(const TilingConfig& cfg) { tiling_cfg_ = cfg; }
    const TilingConfig& tiling_config() const { return tiling_cfg_; }
    // VTCM 预算覆盖(阶段7 测试用): 0 = 默认 8MB; 小预算可强制分配器溢出
    void set_vtcm_budget(uint64_t bytes) { vtcm_budget_override_ = bytes; }
    // spill/fill 记录(阶段7): do_serialize 发射, deserialize 回读
    struct SpillFillRec {
        uint64_t op_id;
        uint32_t block_id;
        uint64_t ddr_offset;  // DDR 池偏移(序列化时确定性分配)
        uint64_t size;        // 字节数
    };
    const std::vector<SpillFillRec>& spill_fill_recs() const { return spill_fill_recs_; }

    // Const 池只读访问(阶段4 P0 验证用: prepare 后权重/参数字节均在池中)
    const std::vector<uint8_t>& const_pool() const { return const_pool_; }
    const std::vector<ConstExtent>& const_extents() const { return const_extents_; }

    // Get opdefs in topological execution order
    std::vector<const OpDef*> get_sorted_opdefs() const;
    // Host-side execution: walk sorted ops, call each op's execute() with float buffers
    // Returns the output buffer (caller frees) and its element count.
    // 注意: 这是 host reference path, 非 DSP 执行。
    struct ExecResult {
        std::vector<float> output;
        bool ok = false;
    };
    ExecResult execute_host(const std::vector<float>& input) const;

    // Op management
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
    // 遍历所有 opdef (供优化 pass 使用)
    template <typename Fn> void for_each_op(Fn fn) {
        for (auto& [id, opdef] : opdef_map_) fn(opdef.get());
    }
    size_t op_count() const { return opdef_map_.size(); }
    void mark_op_deletable(OpDef* op);
    void mark_op_deletable(op_id_t id);
    void collect_deletable_nodes();

    // Optimization passes
    void run_optimize_passes(HexagonNNEnv& env);
    void run_optimize_passes_single_registry(HexagonNNEnv& env,
                                              const std::map<uint32_t, GraphOptPass>& registries);
    void run_optimize_passes_multi_registry(HexagonNNEnv& env);
    void dead_code_removal_and_cse();
    int remove_dead_code(bool);
    int order_nodes(bool);
    int common_subexpr_eliminate(bool);
    void const_prop(HexagonNNEnv& env, bool aggressive);
    void const_prop_and_cse(HexagonNNEnv& env, bool aggressive, bool* changed);
    void eliminate_split_nodes();
    void rebuild_consumers();
    void identify_updateable_quant_ops();
    void run_predication_pass();

    // HtpPrepare auto-injection: per-op-type 输入构造。
    // composeGraphs 产出原始图(op inputs 只含 data); 此方法注入
    // quant_marker/scale/output_rank 并把 tensor_param(perm 等)追加到 inputs。
    // 在 do_prepare1 开头、DCE 之前调用。
    void inject_htp_prepare_inputs();

    // TCM migration
    void tcm_migration(uint32_t threshold, bool aggressive);

    // build_graph_deps (反汇编确认: @ 0xfac220, 8216B)
    // 从 OpDef 图构建 GraphDeps (OpDesc 表 + 依赖链 + 生命期 + memgroup)
    // 产物存入 this+0x7468, 供 FancyAllocator 与 runlist 调度消费
    GraphStatus build_graph_deps();
    GraphDeps* get_graph_deps() const { return graph_deps_.get(); }

    // VTCM lifetime-aware allocation (Phase 4.1)
    // Source: fa_alloc.cc First-Fit graph coloring
    // (libQnnHtpPrepare_tiling_analysis.md §3.2)
    // Runs after build_graph_deps(): collects non-const, non-spilled tensors
    // with their sizes and lifetimes, calls FancyAllocator::allocate_with_lifetime,
    // stores results in vtcm_allocations_.
    void vtcm_lifetime_alloc(VtcmCacheInstance& vtcm);

    // CP oracle branch of vtcm_lifetime_alloc (US20240386237 ideal solver).
    // Selected by HNNX_VTCM_ALLOCATOR=cp/cp-paging/cp-remat/cp-seq/cp-reorder;
    // runs over the SAME request set as the greedy path. Returns true when a
    // verified solution was stored (vtcm_allocations_ + cp_plans_/cp_solution_);
    // false means the caller should fall back to the greedy path unchanged.
    bool vtcm_lifetime_alloc_cp(VtcmCacheInstance& vtcm,
                                const std::vector<fa::FancyAllocator::AllocRequest>& requests,
                                const cp::CPOptions& opts);

    // VTCM per-Op overlap allocation (反汇编 @ 0x13a14b0)
    // Calls FancyAllocator::allow_tensor_overlap_opdef for each op in
    // topological order, implementing the real .so's producer-consumer
    // VTCM reuse via force_contiguous + link_blocks.
    // Stores results in overlap_allocations_.
    void vtcm_overlap_alloc(VtcmCacheInstance& vtcm);

    // VTCM block allocation finalize (反汇编 @ 0x13B29D0)
    // Calls FancyAllocator::allocate_tcm_blocks_internal to sort blocks
    // by reverse lifetime, probe hash for reuse, compute total, check budget.
    void vtcm_block_alloc_finalize(VtcmCacheInstance& vtcm);

    // Post spill/fill design pass (反汇编 @ 0x129EB40, 768B, slc_graph_prepare.cc)
    // Called after spill/fill insertion to finalize the DMA design:
    //   1. Check spill/fill enabled flag ([this+0x611d])
    //   2. If enabled: run spill/fill analyzer on GraphDeps + runlist
    //   3. Read config string ([this+0x6128]) + run design pass (0x129eec0)
    //   4. If spill/fill ops count ([this+0x6120]) > 0: finalize + cleanup
    //   5. Check slc_allocator_debug flag (行 27 warning if no sim_trace_final)
    // REQNN adaptation: takes runlist, processes spill/fill plans
    void post_spill_fill_design_pass(const std::vector<uint32_t>& runlist_tags);

    // Run plugin rewrites (反汇编 @ 0x10D85F0, 3381B, plugin_rewrites.cc)
    // Phase 1 of optimization: iterate registered plugin rewrite rules.
    // Algorithm (from disassembly):
    //   1. Log "plugin rewrites %s outer loop" (行 104) — "early" or "late" phase
    //   2. For each op in topological order:
    //      a. Dereference OpRef → OpDef
    //      b. conditionally_validate_single_quant (×3)
    //      c. For each registered plugin:
    //         - Log "plugin attempt 0x%llx %s" (行 115)
    //         - Try to rewrite the op
    //         - If success: supersede_op (×4) to replace old op with new
    //         - Log "plugin rewrite, inputs:" / "plugin rewrite, results:"
    //      d. Handle multi-out ops ($MultiTemp)
    //   3. Error checks: 行 61 "too few outputs", 行 123 "plugin rewrite error",
    //      行 147 "Graph prepare failed during optimization"
    //   4. qhpi_op_is_error / qhpi_op_error_description for plugin errors
    //   5. remove_dead_code + collect_deletable_nodes cleanup
    // REQNN adaptation: no registered plugins in host reimpl; structure preserved
    void run_plugin_rewrites(bool late_phase);

    struct VtcmAllocEntry {
        uint64_t offset;
        uint32_t block_id;
        bool spilled;
    };
    const std::unordered_map<op_id_t, VtcmAllocEntry>& get_vtcm_allocations() const {
        return vtcm_allocations_;
    }

    // CP oracle outputs (populated iff HNNX_VTCM_ALLOCATOR selected a cp*
    // mode AND vtcm_lifetime_alloc_cp succeeded; otherwise empty/default).
    bool is_cp_allocator_active() const { return cp_active_; }
    const std::vector<cp::CPSpillFillPlan>& get_cp_spill_fill_plans() const { return cp_plans_; }
    const cp::CPSolution& get_cp_solution() const { return cp_solution_; }
    // OpEmitter records from post_spill_fill_design_pass's CP branch
    // (one insert_spill_fill_pair per plan; record-level only — Phase B).
    const OpEmitter* get_cp_op_emitter() const { return cp_op_emitter_.get(); }

    // Multicast optimization (Phase 4.3)
    // Source: grdep_mcast_optimizer.cc
    // Builds McSend list from cross-NSP tensor consumers and runs McastOptimizer.
    void run_mcast_optimization();

    // SuperTile (反汇编确认: create_supertiles @ 0x1313ac0 [M36 修正; 旧址 0x13138d0 错], make_one_supertile @ 0x1314d50)
    GraphStatus create_supertiles();
    GraphStatus make_one_supertile(const std::vector<op_id_t>& op_ids,
                                    const std::vector<int>& split_history);
    const std::vector<SuperTile>& get_supertiles() const { return *supertiles_; }

    // Serialization
    bool serialize(uint8_t* buf, size_t buf_size, size_t& out_size) const;
    bool serialize_file(int fd) const;
    bool do_serialize(Serializer& ser) const;
    void serialize_io(Serializer& ser, uint64_t& counter, bool is_prescan) const;
    void serialize_opdef(Serializer& ser, const OpDef& opdef) const;
    void adjust_heap_stats(Serializer& ser) const;
    bool serialize_patch_metadata(class FileSerializer& fs) const;
    bool serialize_replaceable_constpool(class FileSerializer& fs) const;
    // Deserialization round-trip: rebuild graph from a serialized buffer
    bool deserialize(const uint8_t* buf, size_t buf_size);

    // Graph analysis
    void pprint() const;
    void graphviz_pprint(const char* filename, bool full) const;
    void python_pprint_graph_summary(const char* filename, bool full, bool late) const;
    void python_pprint_detail() const;
    void python_pprint_runlist() const;
    uint64_t calculate_graph_checksum() const;
    bool check_connectivity() const;
    void sanity_check_null_exec(op_id_t id, const Op* op) const;

    // Node ordering
    void make_sorted_optrs();
    op_id_t lookup_op_in_ordering(OrderInfo* ordering, int idx, op_id_t id) const;

    // Const tracking
    void const_tracking_setup();
    void const_tracking_finalize();
    void const_tracking_after_prep(HexagonNNEnv& env);
    void add_tracked_id(op_id_t id, const OpDef& opdef, bool force);

    // Dynamic inputs
    bool is_dynamic_inputs_active() const;
    bool is_dynamic_dma(op_id_t id) const;
    void dynamic_inputs_pre_optimization_pass();
    void dynamic_inputs_post_optimization_pass();

    // Quantization
    void gen_quant_params_hash(uint64_t& hash) const;
    void fixup_signed_activations(OutputDef& od) const;
    bool needs_activation_fixup(DType dt) const;

    // Op factory
    std::unique_ptr<Op> op_factory_generate(const struct OpIoPtrs& io, op_id_t id);

    // VTCM/Allocation
    void allocate_io_tensors();
    void phys_alloc_in_runlist(const std::vector<Op*>& ops);
    void force_contiguous_allocate_mcrecv_blocks(const VtcmCacheInstance& vtcm,
                                                  const std::vector<uint32_t>& tags);
    void link_source_destructive_operands(const std::vector<uint32_t>& tags);
    void allocate_for_reschedule_grdep(const VtcmCacheInstance& vtcm,
                                       const std::vector<uint32_t>& tags, bool);

    // Timing/profiling
    void mark_time_point(const char* name);
    void log_time_points();
    void mark_prepare_stage(std::pair<std::string, uint64_t> stage);
    void clear_profiling_info();
    uint32_t num_profiling_timepoints(uint32_t* count) const;
    void serialize_profiling_timepoints(struct profilingevent* events, uint32_t count);

    // Replacement recording
    void setup_rewrite_log();
    void close_rewrite_log();
    void add_replacement_recorder();
    void enable_replacement_recording();
    void start_replacement_recorder(void** recorder);
    void stop_replacement_recorder(void* recorder);
    op_id_t get_oldest_replaced_id(void* recorder, op_id_t id) const;

    // Helpers
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
    op_id_t truegraph_source_id(OpRef ref) const;
    void truegraph_outputdefs(const OpDef* opdef) const;
    bool is_native_KV_success() const;
    void debug_grdeps(class GraphDeps& deps, const char* name);

    // Runlist
    void pysequencer(std::vector<uint32_t>& runlist_tags);
    void show_runlist(GraphDeps& deps, const std::vector<uint32_t>& tags, int idx) const;
    void dump_runlist(GraphDeps& deps, const std::vector<uint32_t>& tags, const char* filename) const;

    // Multi-NSP
    void get_nsp_id_mapping(struct NspIdMap& map) const;

    // MoE
    bool is_moe_aggregator(op_id_t id) const;
    bool is_part_of_moe_block(op_id_t id) const;
    void get_moe_block_ops_grouped_by_branch(uint16_t block_id) const;
    void get_all_moe_block_ids() const;
    bool multi_quant_transformed_to_single_quant() const;
    void get_dynamically_switchable_blocks() const;

private:
    // Field offsets (all confirmed by decompilation)
    // +0x1d8: Allocator*
    // +0x45ac: serialization state (non-zero = loaded from serialization)
    // +0x5c74-0x61a0: config flags
    // +0x5c8c: io_dma_bypass config
    // +0x5c90: force_barrel flag
    // +0x5d18: memory_alloc_limit (MB)
    // +0x5db0-0x5db8: extra config vector
    // +0x6008: serialize_force_barrel
    // +0x6024: segment count
    // +0x6028: segment count (for runlist)
    // +0x6070-0x6071: sharing metadata flags
    // +0x6143-0x6144: extended_udma flags
    // +0x6155-0x6158: tcm config flags
    // +0x61a0: early-out switch
    // +0x61ed: self-slicing flag
    // +0x62c0-0x62d0: multicast config
    // +0x6508-0x6510: profiling config
    // +0x6c88-0x6c98: sharing metadata string
    // +0x6d58: pass list head
    // +0x6d68: pass count
    // +0x7230-0x72d0: extra config
    // +0x7310-0x7311: graph_dirty flag
    // +0x7458: op registry pointer
    // +0x7468: graph_deps pointer

    std::unordered_map<op_id_t, std::unique_ptr<OpDef>> opdef_map_;
    std::vector<std::unique_ptr<Op>> ops_;
    std::map<uint32_t, GraphOptPass> optimization_registry_;

    // Const pool:集中存储所有 OpDef_Const 的权重/常量数据。
    // 序列化时作为单一数据块写出，每个 const op 通过 (offset,size) 引用。
    std::vector<uint8_t> const_pool_;
    std::vector<ConstExtent> const_extents_;

    // Block table: VTCM block_id -> (pool_id, offset, size) 映射。
    // 序列化时写出，反序列化时重建，供运行期 DMA 寻址。
    struct BlockEntry {
        uint32_t block_id;
        uint32_t pool_id;
        uint64_t offset;
        uint64_t size;
    };
    std::vector<BlockEntry> block_table_;

    // 分段计划: 把 runlist 按 op 数切分成段，记录每段在 .bin 中的偏移，
    // 供大图增量加载。
    struct SegmentPlan {
        uint32_t segment_index;
        uint32_t op_count;
        uint64_t byte_offset;   // 该段在 .bin 中的起始偏移
    };
    std::vector<SegmentPlan> segment_plans_;

    bool graph_dirty_ = false;
    bool serialized_loaded_ = false;
    bool force_barrel_ = false;
    void* allocator_ = nullptr;
    void* op_registry_ = nullptr;
    std::unique_ptr<GraphDeps> graph_deps_;
    uint32_t early_out_flag_ = 0;
    uint32_t memory_alloc_limit_mb_ = 0;
    op_id_t next_op_id_ = 1;
    int construction_state_ = 0;  // +0x45dc: 0=init, 1=CONSTRUCTION, 2=PREPARE, 3=COMPILED
    op_id_t input_node_id_ = 0;   // +0x5340
    op_id_t output_node_id_ = 0;  // +0x5348
    std::vector<op_id_t> ordering_;  // topological ordering of ops
    std::vector<op_id_t> plan_order_;  // ST-Cut 计划执行序(do_prepare2; 空 = 未调度)
    TilingConfig tiling_cfg_;          // 单算子分块配置(阶段6)
    uint64_t vtcm_budget_override_ = 0;  // 阶段7: 0=默认 8MB
    std::vector<SpillFillRec> spill_fill_recs_;  // 阶段7: spill/fill 记录
    struct TimePoint { const char* name; uint64_t timestamp; };
    std::vector<TimePoint> time_points_;

    void run_phase_fixpoint_internal();

    // SuperTile / DPGroupGraph 状态 (反汇编确认: this+0x5578 SuperTile 标志)
    std::unique_ptr<DPGroupGraph> dp_group_graph_;
    std::unique_ptr<SuperTileSolver> supertile_solver_;
    std::unique_ptr<std::vector<SuperTile>> supertiles_;
    uint64_t vtcm_size_ = 0;        // 实际可用 VTCM = budget × 0.75
    uint64_t ddr_bandwidth_ = 48000000000ULL;  // 48 GB/s

    // VTCM lifetime-aware allocation results (Phase 4.1)
    // Populated by vtcm_lifetime_alloc(); keyed by op_id.
    std::unordered_map<op_id_t, VtcmAllocEntry> vtcm_allocations_;

    // VTCM overlap allocation results (反汇编 @ 0x13a14b0)
    // Populated by vtcm_overlap_alloc(); keyed by op_id.
    std::unordered_map<op_id_t, fa::FancyAllocator::OverlapAlloc> overlap_allocations_;

    // CP oracle state (US20240386237): plans/solution from the cp branch of
    // vtcm_lifetime_alloc; emitter records from post_spill_fill_design_pass.
    bool cp_active_ = false;
    std::vector<cp::CPSpillFillPlan> cp_plans_;
    cp::CPSolution cp_solution_;
    std::unique_ptr<OpEmitter> cp_op_emitter_;
};

} // namespace hnnx
