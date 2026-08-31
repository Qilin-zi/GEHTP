#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <algorithm>
#include <limits>

namespace hnnx {

SpillFillScheduler::SpillFillScheduler() = default;
SpillFillScheduler::~SpillFillScheduler() = default;

// SpillFillScheduler: calculates when to spill/fill tensors
// between VTCM and DDR based on memory pressure.
//
// Algorithm:
// 1. Walk through the runlist in execution order
// 2. Track VTCM usage: each op allocates tensors, deallocates when done
// 3. When VTCM usage exceeds budget: find a tensor to spill to DDR
// 4. When a spilled tensor is needed again: insert a fill (DDR -> VTCM)
// 5. Use double buffering when possible to hide DMA latency
//
// Source: insert_spillfill.cc, grdep_spillfill.cc
// "Minimum Required Spill-Fill Buffer Size is %llu"

std::vector<SpillFillScheduler::SpillFillPlan> SpillFillScheduler::plan(
    const std::vector<Op*>& runlist,
    const fa::FancyAllocator& allocator,
    size_t vtcm_size) {

    // Compute VTCM pressure curve
    auto pressure_curve = compute_pressure_curve(runlist, allocator);

    std::vector<SpillFillPlan> plans;

    // Walk through pressure curve and find points where VTCM is exceeded
    size_t vtcm_budget = vtcm_size;
    // Reserve some VTCM for workspace (typically 10-20%)
    size_t usable_vtcm = vtcm_budget * 8 / 10;

    // Track which tensors are currently in VTCM vs DDR
    struct TensorLocation {
        uint64_t vtcm_offset;
        uint64_t ddr_offset;
        size_t first_use;
        size_t last_use;
        size_t size;
        bool in_vtcm;
    };
    std::vector<TensorLocation> tensor_locs;

    for (size_t i = 0; i < pressure_curve.size(); ++i) {
        const auto& pp = pressure_curve[i];

        // If VTCM pressure exceeds budget, need to spill
        if (pp.current_usage > usable_vtcm) {
            // Find the tensor with the furthest last use to spill
            // (LRU-like policy: spill the one not needed for longest)
            size_t best_spill_idx = 0;
            size_t max_distance = 0;

            for (size_t j = 0; j < tensor_locs.size(); ++j) {
                if (!tensor_locs[j].in_vtcm) continue;
                size_t distance = tensor_locs[j].last_use - i;
                if (distance > max_distance) {
                    max_distance = distance;
                    best_spill_idx = j;
                }
            }

            if (max_distance > 0) {
                SpillFillPlan plan;
                plan.spill_position = i;
                plan.fill_position = tensor_locs[best_spill_idx].last_use;
                plan.vtcm_offset = tensor_locs[best_spill_idx].vtcm_offset;
                plan.ddr_offset = tensor_locs[best_spill_idx].ddr_offset;
                plan.size = tensor_locs[best_spill_idx].size;
                plan.double_buffered = false; // can enable if buffer available
                plans.push_back(plan);

                tensor_locs[best_spill_idx].in_vtcm = false;
            }
        }
    }

    return plans;
}

uint64_t SpillFillScheduler::min_buffer_size(
    const std::vector<Op*>& runlist,
    const fa::FancyAllocator& allocator) const {

    // Source: "Minimum Required Spill-Fill Buffer Size is %llu"
    // The minimum buffer size is the maximum concurrent spill+fill size
    //
    // For each point in the runlist:
    //   Count active spills (spilled but not yet filled)
    //   Sum their sizes
    //   Track the maximum concurrent total

    auto pressure_curve = compute_pressure_curve(runlist, allocator);

    uint64_t max_concurrent = 0;
    uint64_t current_concurrent = 0;

    for (const auto& pp : pressure_curve) {
        // If pressure is high, we're spilling
        if (pp.current_usage > pp.peak_usage / 2) {
            current_concurrent += 1024; // approximate per-op spill size
        } else {
            current_concurrent = std::max(uint64_t(0), current_concurrent - 1024);
        }
        max_concurrent = std::max(max_concurrent, current_concurrent);
    }

    // Minimum buffer = max concurrent spill size + alignment padding
    return (max_concurrent + 127) & ~uint64_t(127); // 128-byte aligned
}

std::vector<SpillFillScheduler::VtcmPressurePoint>
SpillFillScheduler::compute_pressure_curve(
    const std::vector<Op*>& runlist,
    const fa::FancyAllocator& allocator) const {

    // Compute VTCM usage at each point in the runlist
    // For each op:
    //   Add output tensor sizes (allocation)
    //   Subtract freed tensor sizes (deallocation when last consumer done)

    std::vector<VtcmPressurePoint> curve;
    uint64_t current_usage = 0;
    uint64_t peak_usage = 0;

    for (size_t i = 0; i < runlist.size(); ++i) {
        // Each op produces output tensors (add to VTCM)
        // Each op consumes input tensors (don't free yet, they may be needed)
        // When an op's output is no longer needed by any later op: free it

        // Approximate: each op uses ~1KB of VTCM for its output
        uint64_t op_vtcm = 1024; // would be computed from OutputDef in real impl
        current_usage += op_vtcm;
        peak_usage = std::max(peak_usage, current_usage);

        curve.push_back({i, current_usage, peak_usage});

        // Free tensors whose last use was this op
        // (In real impl: track tensor lifetimes)
        current_usage = std::max(uint64_t(0), current_usage - op_vtcm / 2);
    }

    return curve;
}

// ===== SynctokenManager (Phase 4.2) =====
// Source: synctoken_manager.cc, make_dma_checkpoint_op @0xd958d0
// Compile-time DMA sync token planner.

SynctokenManager::SynctokenManager() = default;

uint32_t SynctokenManager::allocate() {
    uint32_t id = next_id_;
    next_id_ += 2;  // tags increment by 2 (verified from .bin: 0x11, 0x16, 0x1A...)
    TokenEntry entry;
    entry.token_id = id;
    tokens_.push_back(std::move(entry));
    return id;
}

void SynctokenManager::reuse(uint32_t id) {
    // Group a DMA op under an existing token (e.g., W+b share 0x11).
    // No-op if the token doesn't exist yet.
    if (find_token(id)) return;
    // If not found, register it without advancing next_id_.
    TokenEntry entry;
    entry.token_id = id;
    tokens_.push_back(std::move(entry));
}

void SynctokenManager::signal(uint32_t id, size_t pos, const std::string& name) {
    TokenEntry* tok = find_token(id);
    if (!tok) {
        // Auto-create token if not pre-allocated
        reuse(id);
        tok = find_token(id);
    }
    if (tok) {
        tok->signal_pos = pos;
        tok->signal_name = name;
    }
}

void SynctokenManager::wait(uint32_t id, size_t pos) {
    TokenEntry* tok = find_token(id);
    if (!tok) {
        // Auto-create token if not pre-allocated
        reuse(id);
        tok = find_token(id);
    }
    if (tok) {
        tok->wait_positions.push_back(pos);
    }
}

bool SynctokenManager::validate() const {
    for (const auto& tok : tokens_) {
        // Every token with waiters must have a signal.
        if (!tok.wait_positions.empty() && tok.signal_pos == SIZE_MAX) {
            return false;  // wait without signal
        }
        // Signal must come before every wait.
        for (size_t wpos : tok.wait_positions) {
            if (tok.signal_pos == SIZE_MAX || wpos <= tok.signal_pos) {
                return false;  // wait at or before signal
            }
        }
    }
    return true;
}

void SynctokenManager::reset() {
    next_id_ = 0x11;
    tokens_.clear();
}

SynctokenManager::TokenEntry* SynctokenManager::find_token(uint32_t id) {
    for (auto& tok : tokens_) {
        if (tok.token_id == id) return &tok;
    }
    return nullptr;
}

} // namespace hnnx
