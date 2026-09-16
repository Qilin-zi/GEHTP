// op_gather.cpp — Gather 发射(算子三件套·host 发射侧; embedding 表)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_gather(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    const OpDef* tbl = od->inputs.size() > 0 ? gp.get_op_at(od->inputs[0].src_id) : nullptr;
    uint32_t tbl_s = tbl ? em.ensure_weight_slot(gp, tbl, wslots, od->grouping) : em.dummy_slot_id;
    uint32_t idx_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint32_t row_bytes = 0;
    if (tbl && tbl->output_def.rank >= 1)
        row_bytes = (uint32_t)tbl->output_def.dims[tbl->output_def.rank - 1] * 2;
    em.add_op(OP_GATHER_F16, {tbl_s, idx_t, out_t, (uint32_t)elems_of(od), row_bytes});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_gather{
    {"Gather", op_gather},
};
