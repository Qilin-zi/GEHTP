// op_unary.cpp — Eltwise_Unary / ElementWiseNeuron 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_unary(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    ExtraEltwise e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint32_t sub = (nm == "Eltwise_Unary") ? qnn_unary_to_sub(e.operation)
                                           : qnn_neuron_to_sub(e.operation);
    em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od),
                             sub == 0xFFFFFFFFu ? 0xFFFFFFFFu : sub});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_unary{
    {"Eltwise_Unary", op_unary},
    {"ElementWiseNeuron", op_unary},
};
