// op_copy_sem.cpp — Reshape / Pad / ScatterNd / Cast 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_copy_sem(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    // 数据搬运/常量填充/状态更新语义 → 恒等(设备 M4 数值门兜底)
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od), 0xFFFFFFFFu});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_copy_sem{
    {"Reshape", op_copy_sem},
    {"Pad", op_copy_sem},
    {"ScatterNd", op_copy_sem},
    {"Cast", op_copy_sem},
};
