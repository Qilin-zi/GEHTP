// op_eltwise_ternary.cpp — Eltwise_Ternary 发射(算子三件套·host 发射侧; SELECT)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_eltwise_ternary(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    // SELECT 语义: out = in0(cond) ? in1 : in2
    uint32_t c_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t a_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t b_t = em.src_ref(od->inputs[2], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_BINARY_F16, {c_t, a_t, b_t, out_t, 8});  // 8=SELECT(cond,a,b)
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_eltwise_ternary{
    {"Eltwise_Ternary", op_eltwise_ternary},
};
