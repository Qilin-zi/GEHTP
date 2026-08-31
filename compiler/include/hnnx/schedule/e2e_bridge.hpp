#pragma once

#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/scheduler/st_cut.hpp"

#include <cstdint>
#include <vector>

namespace hnnx {

// Build the compact st-cut graph from an already prepared GraphPrepare.
//
// Node ids in StCutGraphInput are dense indices [0, node_count). The returned
// initial_order preserves GraphPrepare::get_ordering(), and relation weights
// are producer output sizes in bytes (a conservative unit; st-cut treats them
// as opaque weights and only compares/relaxes them).
bool build_stcut_input_from_graph(const GraphPrepare& gp, StCutGraphInput& out);

// Run stcut_full_schedule over the graph and return its best compact order,
// plus the flow/cycle histories from the retry loop.
void run_stcut_schedule(const GraphPrepare& gp,
                        const StCutOptions& opt,
                        std::vector<uint32_t>& best_order,
                        std::vector<uint64_t>& flows,
                        std::vector<uint64_t>& cycles);

} // namespace hnnx
