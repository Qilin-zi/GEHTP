// op_softmax.cpp — Softmax 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_softmax(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraAxis e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint64_t n_elems = elems_of(od);
    uint32_t ax = (e.axis < 0) ? od->output_def.rank - 1 : (uint32_t)e.axis;
    uint64_t row_w = 1;
    if (ax < od->output_def.rank) row_w = od->output_def.dims[ax];
    em.add_op(OP_SOFTMAX_F16, {x_t, out_t, (uint32_t)(n_elems / row_w), (uint32_t)row_w});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_softmax{
    {"Softmax", op_softmax},
};
