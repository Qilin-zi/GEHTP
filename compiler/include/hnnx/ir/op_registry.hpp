#include "hnnx/ir/types.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include <functional>
#include <unordered_map>

namespace hnnx {

// Op registry: maps op names to constructors
// Source: op_registry_prepare.cc, op_def_map.cc
class OpRegistry {
public:
    static OpRegistry& instance();

    using ConstructorFn = std::function<std::unique_ptr<Op>(const OpIoPtrs&, op_id_t)>;

    void register_op(const std::string& name, ConstructorFn ctor);
    void register_op_fn(const std::string& name, ConstructorFn ctor) { register_op(name, std::move(ctor)); }
    std::unique_ptr<Op> generate(const OpIoPtrs& io, op_id_t id) const;
    std::unique_ptr<Op> generate_by_name(const std::string& op_name, const OpIoPtrs& io, op_id_t id) const;
    size_t get_tiler_count() const { return constructors_.size(); }
    size_t get_op_count() const { return constructors_.size(); }

private:
    std::unordered_map<std::string, std::vector<ConstructorFn>> candidates_;
    std::unordered_map<std::string, ConstructorFn> constructors_;
};

} // namespace hnnx
