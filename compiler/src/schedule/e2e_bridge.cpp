#include "hnnx/schedule/e2e_bridge.hpp"

#include <map>

namespace hnnx {

bool build_stcut_input_from_graph(const GraphPrepare& gp, StCutGraphInput& out) {
    out = StCutGraphInput{};
    const auto& ordering = gp.get_ordering();
    if (ordering.empty()) return false;

    std::map<op_id_t, uint32_t> compact;
    for (op_id_t id : ordering) {
        const OpDef* od = gp.get_op_at(id);
        if (!od || !od->is_enabled() || od->is_dead()) continue;
        uint32_t idx = static_cast<uint32_t>(out.initial_order.size());
        compact[id] = idx;
        out.initial_order.push_back(idx);
    }
    out.node_count = static_cast<uint32_t>(out.initial_order.size());
    if (out.node_count == 0) return false;

    for (op_id_t id : ordering) {
        const OpDef* od = gp.get_op_at(id);
        if (!od || !od->is_enabled() || od->is_dead()) continue;
        auto dst = compact.find(od->op_id);
        if (dst == compact.end()) continue;

        for (const auto& conn : od->inputs) {
            auto src = compact.find(conn.src_id);
            if (src == compact.end()) continue;

            uint64_t weight = 1;
            const OpDef* src_def = gp.get_op_at(conn.src_id);
            if (src_def) {
                uint64_t elems = 1;
                for (uint32_t d = 0; d < src_def->output_def.rank && d < 5; ++d) {
                    if (src_def->output_def.dims[d] == 0) continue;
                    elems *= src_def->output_def.dims[d];
                }
                uint64_t esz = src_def->output_def.element_size
                                   ? src_def->output_def.element_size
                                   : 4;
                weight = elems * esz;
            }
            out.relations.push_back({src->second, dst->second, weight, false});
        }
    }
    return true;
}

void run_stcut_schedule(const GraphPrepare& gp,
                        const StCutOptions& opt,
                        std::vector<uint32_t>& best_order,
                        std::vector<uint64_t>& flows,
                        std::vector<uint64_t>& cycles) {
    StCutGraphInput input;
    if (!build_stcut_input_from_graph(gp, input)) {
        best_order.clear();
        flows.clear();
        cycles.clear();
        return;
    }

    StCutContext ctx;
    stcut_build_initial_state(ctx, input);
    stcut_full_schedule(ctx, opt, input.initial_order, best_order, flows, cycles, 0, 0);
}

} // namespace hnnx
