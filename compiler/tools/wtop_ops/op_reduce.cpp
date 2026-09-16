// op_reduce.cpp — Reduce 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_reduce(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraAxis e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    {
        const OpDef* src_r = gp.get_op_at(od->inputs[0].src_id);
        uint64_t n_in = src_r ? elems_of(src_r) : elems_of(od);
        std::vector<uint32_t> args{x_t, out_t, (uint32_t)n_in,
                                   (uint32_t)e.axis, e.reduce_type};
        for (uint32_t i = 0; i < 4; i++)
            args.push_back(src_r && i < src_r->output_def.rank
                               ? src_r->output_def.dims[i] : 1u);
        em.add_op(OP_REDUCE_F16, args);
    }
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_reduce{
    {"Reduce", op_reduce},
};
