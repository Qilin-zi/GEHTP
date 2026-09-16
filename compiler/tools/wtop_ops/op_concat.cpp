// op_concat.cpp — Concat 发射(算子三件套·host 发射侧; last-dim 交错)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_concat(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraAxis e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    std::vector<uint32_t> args;
    for (size_t i = 0; i < 8; i++)
        args.push_back(i < od->inputs.size()
            ? em.src_ref(od->inputs[i], gp.get_input_node_id(), gp, wslots) : 0u);
    args.push_back(out_t);
    uint32_t ax_c = (uint32_t)(e.axis < 0 ? od->output_def.rank - 1 : e.axis);
    args.push_back(ax_c);
    args.push_back((uint32_t)od->inputs.size());
    args.push_back((uint32_t)elems_of(od));
    // 每段 axis 维尺寸(≤4 段): 从各输入生产者 output_def 取
    for (size_t i = 0; i < 4; i++) {
        uint32_t sz = 0;
        if (i < od->inputs.size()) {
            const OpDef* src = gp.get_op_at(od->inputs[i].src_id);
            if (src && ax_c < src->output_def.rank)
                sz = src->output_def.dims[ax_c];
        }
        args.push_back(sz);
    }
    em.add_op(OP_CONCAT_F16, args);
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_concat{
    {"Concat", op_concat},
};
