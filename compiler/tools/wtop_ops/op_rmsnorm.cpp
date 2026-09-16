// op_rmsnorm.cpp — RmsNorm 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_rmsnorm(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
    uint32_t w_s = w ? em.ensure_weight_slot(gp, w, wslots, od->grouping) : em.dummy_slot_id;
    const OpDef* bs = od->inputs.size() > 2 ? gp.get_op_at(od->inputs[2].src_id) : nullptr;
    uint32_t b_s = bs ? em.ensure_weight_slot(gp, bs, wslots, od->grouping) : em.dummy_slot_id;
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_RMSNORM2_F16, {x_t, w_s, b_s, out_t, (uint32_t)elems_of(od)});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_rmsnorm{
    {"RmsNorm", op_rmsnorm},
};
