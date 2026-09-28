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
        /* 002 US1: 超 VTCM 界的 f16 matmul 编译期 N 分块 (OP_MATMUL_F16_SPLIT)。
         * need 公式与 exec_matmul_f16_hmx 逐字节同口径; US2 前预算 = 运行时
         * VTCM total (16MB = 16777216, 设备 A2 query_VTCM 实锤), US2 后换
         * TAG_MEM_PLAN v2 的 scratch_size。
         * 条件: M/K/N 全 32 对齐 + 非 batched; 不满足则维持整 op (运行时
         * 超界回落标量, 现状行为零回归)。act/wgt/out 三引用跨 chunk 共享,
         * 仅 n0/N_chunk 两标量不同; 逐字节恒等五纪律见 spec FR-013。 */
        auto need_mm = [](uint32_t mm, uint32_t kk, uint32_t nn2) -> size_t {
            size_t a = (size_t)mm * kk * 2u;
            size_t w = (size_t)nn2 * kk * 2u;
            size_t o = (size_t)mm * nn2 * 2u;
            size_t off_w = (a + 2047u) & ~(size_t)2047u;
            size_t off_o = (off_w + w + 2047u) & ~(size_t)2047u;
            size_t off_s = (off_o + o + 2047u) & ~(size_t)2047u;
            return off_s + 2048u;
        };
        const size_t kVTCMTotal = 16u * 1024u * 1024u;
        if (!em2.batched && (m % 32u) == 0u && (k % 32u) == 0u && (nn % 32u) == 0u &&
            need_mm(m, k, nn) > kVTCMTotal) {
            /* 二分找最大 32 倍数 N_c 使 need(m,k,N_c) ≤ 预算 (need 随 N 单调) */
            uint32_t N_chunk = 0, lo = 32u, hi = nn;
            while (lo <= hi) {
                uint32_t mid = lo + ((hi - lo) / 64u) * 32u;
                if (need_mm(m, k, mid) <= kVTCMTotal) { N_chunk = mid; lo = mid + 32u; }
                else { hi = mid - 32u; }
            }
            if (N_chunk == 0) {
                /* 连最小 32 列 chunk 都超界 (act 自身超 VTCM 的极端形状) */
                std::fprintf(stderr, "warn: matmul M=%u K=%u N=%u 最小 chunk 也超 VTCM, 维持整 op\n",
                             m, k, nn);
            } else {
                for (uint32_t n0 = 0; n0 < nn; n0 += N_chunk) {
                    uint32_t nc = (N_chunk < nn - n0) ? N_chunk : (nn - n0);
                    em.add_op(OP_MATMUL_F16_SPLIT, {a_t, w_s, out_t, m, k, nn, n0, nc, flags});
                }
                return 0;
            }
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
