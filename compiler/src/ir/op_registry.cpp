#include "hnnx/ir/op_registry.hpp"
#include "hnnx/ops/ops.hpp"
#include <limits>

namespace hnnx {

OpRegistry& OpRegistry::instance() {
    static OpRegistry reg;
    return reg;
}

void OpRegistry::register_op(const std::string& name, ConstructorFn ctor) {
    candidates_[name].push_back(std::move(ctor));
    constructors_[name] = nullptr; // mark as registered
}

std::unique_ptr<Op> OpRegistry::generate(const OpIoPtrs& io, op_id_t id) const {
    // Source: op_registry_prepare.cc:97-99
    // Resolve the op type name from the OpDef carried in io.opdef_ptr, then
    // dispatch to generate_by_name which selects the lowest-cost candidate.
    if (!io.opdef_ptr) return nullptr;
    const OpDef* opdef = static_cast<const OpDef*>(io.opdef_ptr);
    const char* name = opdef->name_tag ? opdef->name_tag->name() : nullptr;
    if (!name) return nullptr;
    return generate_by_name(name, io, id);
}

std::unique_ptr<Op> OpRegistry::generate_by_name(const std::string& op_name,
                                                    const OpIoPtrs& io, op_id_t id) const {
    // Find constructor by op name and create Op
    auto it = candidates_.find(op_name);
    if (it == candidates_.end() || it->second.empty()) return nullptr;

    // Try each candidate, pick lowest cost
    // Source: op_factory_generate @ 0x10BE710
    std::unique_ptr<Op> best_op;
    float best_cost = std::numeric_limits<float>::max();

    for (const auto& ctor : it->second) {
        auto op = ctor(io, id);
        if (op) {
            // Set op type name for cost model
            auto* typical = dynamic_cast<TypicalOp*>(op.get());
            if (typical) typical->op_type_name = op_name;

            float cost = op->cost(nullptr);
            if (cost < best_cost) {
                best_cost = cost;
                best_op = std::move(op);
            }
        }
    }

    return best_op;
}

} // namespace hnnx
