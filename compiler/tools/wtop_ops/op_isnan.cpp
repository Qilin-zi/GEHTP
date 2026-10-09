// op_isnan.cpp — IsNan / IsInf 发射(算子三件套·host 发射侧)
//
// QNN 2.48 的 IsNan/IsInf 无 scalar "operation"(graph_prepare 未注册
// extractor → serialized_extra 为空), 故按算子名分派设备 subtype。
// 输出是 bool 0/1, 与 EQ/AND 一样落在 f16 槽(1.0f/0.0f)。
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_isnan(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    uint32_t sub;
    if (nm == "IsNan") sub = 15;   /* OP_UNARY_F16 subtype 15 */
    else if (nm == "IsInf") sub = 16;
    else {
        std::fprintf(stderr, "error: %s 无设备 subtype 映射\n", nm.c_str());
        return 5;
    }
    if (od->inputs.size() < 1) {
        std::fprintf(stderr, "error: %s op %llu 缺输入\n", nm.c_str(), (unsigned long long)od->op_id);
        return 5;
    }
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od), sub});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_isnan{
    {"IsNan", op_isnan},
    {"IsNaN", op_isnan},
    {"IsInf", op_isnan},
};
