// op_matmul.cpp — FullyConnected / MatMul 发射(算子三件套·host 发射侧;
// W4A16 权重判定 + batched BMM flags)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_matmul(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    uint32_t a_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
    /* B 为 const(权重)→ 槽; 运行时(q·kᵀ 的 k)→ temp 引用 */
    bool w_is_const = w && (w->is_const() || w->const_data_size > 0);
    uint32_t w_s = w_is_const
        ? (0x8000u | em.ensure_weight_slot(gp, w, wslots, od->grouping))
        : (od->inputs.size() > 1
               ? em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots)
               : em.dummy_slot_id);
    ExtraFc ef{};
    ExtraMatMul em2{};
    uint32_t m = 0, k = 0, nn = 0;
    if (nm == "FullyConnected") {
        if (od->serialized_extra.size() >= sizeof(ef))
            std::memcpy(&ef, od->serialized_extra.data(), sizeof(ef));
        m = ef.m; k = ef.k; nn = ef.n;
    } else {
        if (od->serialized_extra.size() >= sizeof(em2))
            std::memcpy(&em2, od->serialized_extra.data(), sizeof(em2));
        m = em2.m; k = em2.k; nn = em2.n;
    }
    if (m == 0 || k == 0 || nn == 0) {
        std::fprintf(stderr, "error: %s extra 未提取 (M/K/N=0)\n", nm.c_str());
        return 4;
    }
    /* Q4_0 打包权重(K*N/2 字节)→ W4A16;f16 池权重 → MATMUL_F16
     * (float 图; probe 首撞: f16 权重被按 W4A16 执行直接失败) */
    bool w4 = w_is_const && (w_s != em.dummy_slot_id) &&
              (em.slots[w_s & 0x7FFFu].len == (k * nn / 2u));
    if (w4) {
        em.add_op(OP_MATMUL_W4A16, {a_t, w_s, out_t, m, k, nn});
    } else {
        uint32_t flags = 0;
        if (nm == "MatMul")
            flags = (em2.transpose_in0 & 1u) | ((em2.transpose_in1 & 1u) << 1);
        else
            flags = 2u;  /* FC: 权重存 [N,K] */
        /* batched BMM: bit2 置位, 批数入高 16 位 */
        if (em2.batched) {
            flags |= 4u;
            uint32_t rk = od->output_def.rank;
            while (rk > 0 && od->output_def.dims[rk - 1] == 1) rk--;
            uint64_t bn = 1;
            for (uint32_t i = 0; i + 2 < rk; i++) bn *= od->output_def.dims[i];
            flags |= ((uint32_t)bn & 0xFFFFu) << 16;
        }
        em.add_op(OP_MATMUL_F16, {a_t, w_s, out_t, m, k, nn, flags});
    }
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_matmul{
    {"FullyConnected", op_matmul},
    {"MatMul", op_matmul},
};
