#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <cstring>
#include <vector>

namespace hnnx {

// Optimization phase thresholds (confirmed by decompilation)
// Phase descriptors are 0x40 (64) bytes each on stack
enum OptPhase : uint32_t {
    PHASE_0     = 3000,    // 0xBB8
    PHASE_1     = 10190,   // 0x27CE
    PHASE_2     = 11892,   // 0x2E7C
    PHASE_3     = 12492,   // 0x30D4
    PHASE_4     = 21101,   // 0x526D
    PHASE_5     = 22000,   // 0x55F0
    PHASE_TERM  = 0xFFFFFFFF,
};

struct PhaseDescriptor {
    uint32_t threshold;
    void* vtable;       // .data.rel.ro vtable pointer
    void* graph;         // GraphPrepare*
    uint64_t reserved;
    void* list_self;     // linked list self-pointer
    uint64_t reserved2[3];
};

// GraphOptInfo: 128 (0x80) bytes, allocated via operator new(0x80)
// +0x08: phase number (int)
// +0x18: matcher descriptor ([0]=match fn vtable, [1]=hash key, [2..0x12]=input count min/max)
struct GraphOptInfo {
    void* vtable;            // +0x00
    uint32_t phase;          // +0x08
    uint32_t id;             // +0x0C
    void* defopt_fn;         // +0x10
    void* matcher_desc;      // +0x18 -- points to matcher array
    // ... total 0x80 bytes

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
        // Internal matching state
        std::vector<void*> matched_ops;
    };

    std::vector<GraphOptInfo*> rules_;
    void* hash_table_ = nullptr;
    size_t hash_table_size_ = 0;

    void add_optim(GraphOptInfo* info);
    void build_matchers();
    MatchIterator begin_match(const OpDef* opdef) const;
};

struct GraphOptContext {
    GraphPrepare* graph_prepare_;
    const GraphOptInfo* rule_;
    OpDef* target_opdef_;

    // Attempt to apply a rule to an opdef
    // Returns true if the rule was applied
    bool attempt(const GraphOptInfo* rule, OpDef* opdef);

    // Build a new op from the rule
    std::unique_ptr<Op> build_new_op(const GraphOptInfo& rule, OpDef& opdef);
};

// Merge multiple optimization registries into one
std::map<uint32_t, GraphOptPass> merge_optimization_passes(
    std::map<uint32_t, GraphOptPass>& dst,
    const std::vector<std::string>& registry_names);

// Phase vtable layout (7 phases, Itanium ABI)
// vfunc[0-1]: trivial destructors
// vfunc[2]: construct/clone
// vfunc[3]: generic method
// vfunc[4-5]: trivial
// vfunc[6]: CORE EXECUTION (calls optimization passes)
// vfunc[7]: type identification
// vfunc[8]: trivial
// vfunc[9]: mutex init

// Phase vfunc[6] behavior per phase:
// Phase0/1: trivial (forwarding, filtered)
// Phase2: Fixpoint -> tcm_migration(0x2E7C, true) -> Fixpoint -> optional dump
// Phase3: Fixpoint (DCE -> order -> CSE -> clear dirty)
// Phase4: tcm_migration(0x526D, false)
// Phase5: Fixpoint (DCE -> order -> CSE -> clear dirty)
// Terminal: trivial

void run_phase_fixpoint(GraphPrepare* gp);
// Fixpoint = remove_dead_code(false) -> order_nodes(true) -> common_subexpr_eliminate(true) -> clear graph_dirty

// Fusion 规则: 把 (producer_type, consumer_type) 融合成 fused_type。
// 在 run_optimize_passes 的 fixpoint 后扫描图应用。
struct FusionRule {
    const char* producer;   // 上游 op 类型名 (如 "Conv")
    const char* consumer;   // 下游 op 类型名 (如 "Relu")
    const char* fused;      // 融合后 op 类型名 (如 "ConvActivations")
};
// 应用所有 fusion 规则，返回融合的 op 数。
int apply_fusion_rules(GraphPrepare* gp, const std::vector<FusionRule>& rules);

} // namespace hnnx
