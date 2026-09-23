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
    if (sub == 0xFFFFFFFFu) {
        /* 硬错误(P1 消除静默兜底): 未知 operation 不得进 blob —
         * 此前 0xFFFFFFFF 原样入 blob, exec_unary default 静默直通,
         * 0.8B GDN SOFTPLUS ×18 全被拷成恒等(战役日志 2026-09-22) */
        std::fprintf(stderr,
                     "error: %s op %llu (%s) operation=%u 无设备映射\n",
                     nm.c_str(), (unsigned long long)od->op_id,
                     od->name_tag && od->name_tag->name() ? od->name_tag->name() : "?",
                     e.operation);
        return 5;
    }
    em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od), sub});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_unary{
    {"Eltwise_Unary", op_unary},
    {"ElementWiseNeuron", op_unary},
};
