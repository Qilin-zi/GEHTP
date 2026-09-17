#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"

namespace hnnx {

// Fixpoint loop implementation
// Source: Phase2/3/5 vfunc[6] in graph_prepare.cc
// Also implemented as run_phase_fixpoint_internal in graph_prepare.cpp

// run_phase_fixpoint is implemented in optimization_passes.cpp (real loop:
// remove_dead_code / order_nodes / common_subexpr_eliminate until clean).
// This file previously carried a second, no-op definition of the same symbol,
// which collided at link time (both TUs are in htp_core) and, depending on
// archive extraction order, could silently shadow the real implementation.
// graph_prepare itself uses its own member run_phase_fixpoint_internal().

} // namespace hnnx
