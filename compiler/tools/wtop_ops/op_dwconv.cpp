// op_dwconv.cpp — DepthWiseConv2d 发射(算子三件套·host 发射侧; SSM 一维卷积)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_dwconv(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraDepthwiseConv e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
    uint32_t w_s = w ? em.ensure_weight_slot(gp, w, wslots, od->grouping) : em.dummy_slot_id;
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint64_t ne = elems_of(od);
    uint32_t seq = 1, ch = (uint32_t)ne;
    if (od->output_def.rank >= 2) {
        seq = od->output_def.dims[od->output_def.rank - 2];
        ch = od->output_def.dims[od->output_def.rank - 1];
    }
    em.add_op(OP_CONV1D_SSM_F16, {x_t, w_s, out_t, seq, ch, e.kw});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_dwconv{
    {"DepthWiseConv2d", op_dwconv},
};
