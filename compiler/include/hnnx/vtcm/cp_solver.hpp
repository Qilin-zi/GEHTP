// @file cp_solver.hpp
//
// CP (Constraint Programming) ideal-solution solver for VTCM scheduling —
// the oracle reference for US20240386237 A1
// "Efficient Optimization of Tensor Rematerialization and Paging for Neural
// Networks" (Qualcomm).
//
// Formalization (equation numbers follow the patent, see
// patents_qnn/CP_SOLVER_SPEC.md):
//   Eq.3  objective: minimize sum over active segments of (p_vi + q_vi) * m_v
//         (DDR bytes = page-in + page-out)
//   Eq.6  capacity:   sum of resident m_v at any event slot <= M
//   Eq.7  dependency: a COMPUTE/REMAT of v requires every pred u open
//         (page-in exempt: (1-p_vi) multiplier in the patent)
//   Eq.8  all-different start slots (serial event sequence)
//   Eq.9  first segment starts with compute; fill requires prior spill
//   Eq.10 compute budget: sum of w over compute-started segments <= W
//   Eq.12 every node computed at least once (a_v1 = 1)
//   Eq.14 at most one page-out per tensor
//
// Search: DFS branch-and-bound over event slots. Structural constraints
// (Eq.4/5/8/9/11/12/14) hold by construction:
//   - tasks placed one per slot in increasing slot order (Eq.8)
//   - segments created in order, first always compute (Eq.4/5/12)
//   - spill close before fill start on the same tensor (Eq.9/14)
// Only Eq.6/Eq.7/Eq.10 are checked, all incrementally.
//
// Dominance trichotomy (derived from Eq.14 + Eq.9 + cost domination of
// spill-without-fill by discard+remat): each tensor's meaningful choice is
//   SINGLE (one segment, discard, DDR 0) / REMAT (recompute, DDR 0,
//   compute +w_v) / PAGE (spill+fill, DDR 2*m_v) — so Eq.3 reduces to
//   2 * sum(m_v over PAGE tensors), a coarse subset-sum that bounds well.
//
// Canonical form (lossless, by weak domination): a segment closes only at
// the slot boundary right after the tensor was consumed; closing later
// without an intervening consumption is dominated by closing earlier.
//
// Degenerate knobs (patent L1144):
//   PAGING_ONLY: REMAT candidates removed (paging only)
//   REMAT_ONLY : PAGE_IN/spill-close removed, objective -> total compute
//   SEQ_ONLY   : C_v == 1, objective -> peak resident memory
//
// Relation to the .so (reverse-engineering ground truth):
//   FancyAllocator::allocate_with_lifetime is the *greedy approximation* of
//   this problem (life_end desc sort + identical-interval grouping + event
//   sweep + spill on overflow). This solver is the oracle it approximates.
//   NOTE: the real .so's CBS is cost-based scheduling (sched_threshold_ratio
//   ladder), NOT 8-bank graph coloring — that earlier RE claim is disproven;
//   do not cite 8-bank/spill_count<10/retcode-0xe parameters as baseline.
#pragma once

#include "hnnx/ir/types.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hnnx {
namespace cp {

// ── L1: pure problem structures (no GraphPrepare/OpDef/GraphDeps dep) ──

struct CPTensor {
    op_id_t id = 0;        // external op_id (bridging key back to AllocResult)
    uint64_t m = 0;        // output tensor size in bytes (aligned)
    uint64_t w = 1;        // compute cost (cycles; cost model fallback)
    uint32_t C = 2;        // max retention segments (1 or 2)
    bool no_remat = false; // lock C to 1 (IO/side-effect nodes)
};

struct CPProblem {
    std::vector<CPTensor> nodes;                     // internal index 0..n-1
    std::vector<std::pair<uint32_t, uint32_t>> edges; // (u,v): v consumes u
    uint64_t M = 0;                                  // VTCM capacity (pinned deducted)
    uint64_t W = 0;                                  // compute budget (Eq.10)
    std::vector<op_id_t> topo_hint;                  // topo order (ids); may be empty

    // derived helpers (filled by CPSolver ctor)
    uint32_t n() const { return static_cast<uint32_t>(nodes.size()); }
};

struct CPOptions {
    enum class Mode { FULL, PAGING_ONLY, REMAT_ONLY, SEQ_ONLY };
    enum class Objective { DDR_BYTES, PEAK_RESIDENT, TOTAL_COMPUTE };
    enum class CostModel { UNIFORM, BYTES, COST_TABLE }; // w_v fallback ladder

    Mode mode = Mode::FULL;
    Objective objective = Objective::DDR_BYTES; // auto-switched by mode
    CostModel cost_model = CostModel::BYTES;
    bool respect_topo_order = true; // Eq.8 reorder freedom vs topo lifetimes
    uint64_t node_limit = 5000000;  // deterministic guardrail (tests use this)
    uint32_t time_limit_ms = 5000;  // wall-clock guardrail (checked per 4096 nodes)
    uint32_t max_exact_nodes = 50;  // n above this -> degenerate greedy
    uint32_t C_default = 2;

    bool mode_active() const { return true; } // env layer decides path; opts always on here

    const char* mode_name() const {
        switch (mode) {
            case Mode::PAGING_ONLY: return "cp-paging";
            case Mode::REMAT_ONLY:  return "cp-remat";
            case Mode::SEQ_ONLY:    return "cp-seq";
            default:                return "cp";
        }
    }
};

struct CPSegment {
    uint32_t s = 0, e = 0;    // event slots (1-based, inclusive)
    bool active = false;
    bool fill_start = false;  // p_vi: 1 = page-in start
    bool spill_end = false;   // q_vi: 1 = page-out end
    uint64_t vtcm_offset = 0; // backfilled by extraction (first-fit)
    uint64_t ddr_offset = 0;  // backfilled for spill segments (arena bump)
};

struct CPTask {
    enum class Kind : uint8_t { COMPUTE, REMAT, PAGE_IN, PAGE_OUT };
    Kind kind = Kind::COMPUTE;
    op_id_t v = 0;
    uint64_t bytes = 0;   // m_v (DMA bytes for PAGE_IN/OUT)
    uint32_t slot = 0;    // event slot (PAGE_OUT: gap after this slot)
};

struct CPSolution {
    bool feasible = false;
    std::vector<std::array<CPSegment, 2>> segments; // per internal node index
    std::vector<CPTask> seq;                        // total order (P6 extraction)
    uint64_t ddr_bytes = 0;       // Eq.3 objective value
    uint64_t peak_resident = 0;   // max slot residency (<= M)
    uint64_t total_compute = 0;   // Eq.10 left-hand side
    uint32_t remat_count = 0;
    uint32_t spill_count = 0;     // PAGE tensors (spill+fill pairs)
    uint32_t placement_degraded = 0; // tensors whose first-fit failed -> spilled
    uint64_t vtcm_arena_used = 0;    // high-water of first-fit placement
    uint64_t ddr_arena_used = 0;     // sum of spilled tensor sizes (Eq.14: single version)
    bool optimal = false;            // true only if search tree exhausted
    double bound_gap = 0.0;          // (incumbent - LB) / incumbent; 0 when optimal
    uint64_t nodes_explored = 0;
    std::string abort_reason;        // timeout/nodelimit/infeasible (fallback trigger)
};

class CPSolver {
public:
    CPSolver(CPProblem problem, CPOptions options);
    ~CPSolver(); // out-of-line: Impl is incomplete in this header

    // Main entry: guardrails -> exact DFS branch-and-bound / degenerate greedy.
    CPSolution solve();

    // P7: independent recomputation of Eq.4-14 + objective consistency from
    // segments/seq alone (no solver internals). On failure returns false and,
    // if `why` is non-null, writes the first violated check.
    static bool verify(const CPProblem& p, const CPSolution& s, std::string* why = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── L2: bridging adapters (fa::FancyAllocator / runlist contracts) ──

// Env-driven option parsing: HNNX_VTCM_ALLOCATOR (fancy/cp/cp-paging/
// cp-remat/cp-seq/cp-reorder), HNNX_CP_NODE_LIMIT, HNNX_CP_TIME_LIMIT_MS,
// HNNX_CP_MAX_NODES, HNNX_CP_W_SLACK, HNNX_CP_COST_MODEL.
// Returns false when the path selection is the default greedy (fancy).
bool parse_cp_options_from_env(CPOptions* out);

// requests+edges (op_id level) -> CPProblem. w_v per cost_model ladder,
// W = ceil((1+slack) * sum(w)) with slack from env (W has no .so source).
CPProblem build_problem_from_requests(
    const std::vector<fa::FancyAllocator::AllocRequest>& requests,
    const std::vector<std::pair<op_id_t, op_id_t>>& edges,
    size_t vtcm_budget, const std::vector<op_id_t>& topo_order,
    const CPOptions& opts);

// Channel 1: resident tensors -> AllocResult (same type as greedy path).
// First-fit offsets into [0, budget); failed placements -> spilled=true.
std::unordered_map<op_id_t, fa::FancyAllocator::AllocResult>
to_alloc_results(const CPSolution& sol, const CPProblem& p,
                 size_t budget, size_t alignment = 128);

// Channel 2: PAGE tensors -> spill/fill position plans (runlist index space
// via consumers: spill after last seg0 consumer, fill before first seg1
// consumer). Output type mirrors SpillFillScheduler::SpillFillPlan to stay
// decoupled from the dma header.
struct CPSpillFillPlan {
    size_t spill_position = 0;      // topo index of last seg0-served consumer
    size_t fill_position = 0;       // topo index of first seg1-served consumer
    uint64_t vtcm_offset = 0;       // fill destination (seg1)
    uint64_t vtcm_offset_spill = 0; // spill source (seg0; may differ)
    uint64_t ddr_offset = 0;        // == for both ends (Eq.9 same address)
    uint64_t size = 0;
    bool double_buffered = false;
    op_id_t op_id = 0;
};
std::vector<CPSpillFillPlan> to_spill_fill_plans(const CPSolution& sol,
                                                 const CPProblem& p);

} // namespace cp
} // namespace hnnx
