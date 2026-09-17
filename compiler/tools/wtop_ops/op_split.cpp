// op_split.cpp — Split 发射(算子三件套·host 发射侧; 副本语义只建 out0)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_split(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraSplit e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    std::vector<uint32_t> args{x_t};
    for (size_t i = 0; i < 8; i++) {
        /* 副本语义: 只建 out0(exec 只写 args[1]); 多余 temp 会
         * 污染 free list(probe: Reshape 错拿 split 的 out2) */
        uint32_t t = (i == 0 && e.num_splits > 0)
                         ? em.fresh_temp(gp, od->op_id, 0) : 0u;
        args.push_back(t);
    }
    args.push_back((uint32_t)(e.axis < 0 ? 0 : e.axis));
    args.push_back(e.num_splits);
    for (size_t i = 0; i < 4; i++)
        args.push_back(i < (size_t)e.num_splits ? e.sizes[i] : 0u);
    args.push_back(e.split_index);  /* 副本 op 取自己的段写 out0 */
    em.add_op(OP_SPLIT_F16, args);
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_split{
    {"Split", op_split},
};
