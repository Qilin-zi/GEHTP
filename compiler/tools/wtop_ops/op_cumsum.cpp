// op_cumsum.cpp — CumulativeSum 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_cumsum(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraAxis e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint64_t ne = elems_of(od);
    uint32_t ax = (e.axis < 0) ? od->output_def.rank - 1 : (uint32_t)e.axis;
    uint32_t rows = 1, n_ax = 1;
    if (ax < od->output_def.rank) {
        rows = (uint32_t)(ne / od->output_def.dims[ax]);
        n_ax = od->output_def.dims[ax];
    }
    em.add_op(OP_CUMSUM_F32, {x_t, out_t, rows, n_ax, ax, e.exclusive, e.reverse});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_cumsum{
    {"CumulativeSum", op_cumsum},
};
