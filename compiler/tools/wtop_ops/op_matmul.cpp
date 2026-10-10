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
        ? (0x8000u | em.ensure_weight_slot(gp, w, wslots, od))
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
        if (!od->serialized_extra.empty()) {
            size_t c = std::min(od->serialized_extra.size(), sizeof(em2));
            std::memcpy(&em2, od->serialized_extra.data(), c);
        }
        m = em2.m; k = em2.k; nn = em2.n;
    }
    if (m == 0 || k == 0 || nn == 0) {
        std::fprintf(stderr, "error: %s extra 未提取 (M/K/N=0)\n", nm.c_str());
        return 4;
    }
    /* FC bias 折叠(pass_fc_bias)配套: ExtraFc.has_bias 只认 pass 写入的
     * op_data scalar 标记(extract_fc_extra 同源口径; 注入的 N 元素 wscale
     * const 不再误判)。此处再核实 inputs[2] 确为 const 才发 BIAS opcode。
     * 005 US3a: MatMul 链头同标记同通道(em2.has_bias, 旧 bin 24B 无此
     * 字段按 0)。 */
    const OpDef* bias_op = nullptr;
    bool mm_mark = (nm == "MatMul") && em2.has_bias;
    bool has_bias = ((nm == "FullyConnected") ? (ef.has_bias != 0) : mm_mark) &&
                    od->inputs.size() > 2;
    if (nm == "MatMul" && od->serialized_extra.size() == 24u)
        has_bias = false;                     // 旧 bin: 无 has_bias 字段
    if (has_bias) {
        const OpDef* bp = gp.get_op_at(od->inputs[2].src_id);
        if (bp && (bp->is_const() || bp->const_data_size > 0)) bias_op = bp;
        else has_bias = false;
    }
    /* Q4_0 打包权重(K*N/2 字节)→ W4A16;f16 池权重 → MATMUL_F16
     * (float 图; probe 首撞: f16 权重被按 W4A16 执行直接失败) */
    bool w4 = w_is_const && (w_s != em.dummy_slot_id) &&
              (em.slots[w_s & 0x7FFFu].len == (k * nn / 2u));
    if (w4) {
        if (has_bias) {
            /* W4A16 的 bias 走权重内嵌表(exec_matmul 扫槽), 独立 bias 槽
             * 无通道, 静默丢 bias = 数值错 —— 硬报错, 待 W4A16+bias 通道 */
            std::fprintf(stderr, "error: FC has_bias + W4A16 权重组合暂不支持 (M=%u K=%u N=%u)\n",
                         m, k, nn);
            return 4;
        }
        /* 003 T015: W4A16 引用规范化。运行期 exec_matmul 的 act/w 均按裸
         * slot 索引取数 (b->slots[act_s]), act 另认 0x4000|temp 驻留形态;
         * 旧发射把 0x8000|slot / 裸 temp 原样传入 = slots 表越界读, 真实图
         * W4A16 从未走通 (仅合成 blob 用裸 slot 验证过)。act 三态:
         *   const 槽 → 裸 slot (烘焙字节, 零新 op)
         *   VTCM 驻留 temp → 0x4000|temp (生产者已直写池, 零新 op)
         *   DDR temp / 主输入(EXT_IN 注入) → OP_DMA 预置到池内 staging (--dma-runlist) */
        uint32_t w_raw = w_s & 0x7FFFu;
        uint32_t a_slot = (a_t & 0x8000u) ? (a_t & 0x7FFFu) : 0xFFFFFFFFu;
        bool act_is_const_slot = (a_slot != 0xFFFFFFFFu) && a_slot != 0u &&
                                 a_slot != em.dummy_slot_id;
        uint32_t act_arg;
        if (act_is_const_slot) {
            act_arg = a_slot;
        } else if (!(a_t & 0x8000u) && em.ddr_static &&
                   em.ddr_temp_vtcm.count(a_t) && em.ddr_temp_vtcm.at(a_t)) {
            act_arg = 0x4000u | a_t;
        } else {
            uint32_t act_bytes = m * k * 2u;
            uint32_t off = 0;
            int stg = em.alloc_vtcm_stage(act_bytes, &off);
            if (stg < 0 || !em.dma_runlist) {
                std::fprintf(stderr,
                    "error: W4A16 act 引用不可表达 (a_t=0x%X M=%u K=%u): 运行期仅认"
                    " 裸 slot / 0x4000|temp; DDR 源需 --dma-runlist + VTCM 池余量"
                    " (staging rc=%d dma_runlist=%d)\n",
                    a_t, m, k, stg, (int)em.dma_runlist);
                return 4;
            }
            uint32_t prev_eng = em.cur_engine;
            em.cur_engine = wt_annot_pack(0u, 1u, 0u);  /* 预置 DMA = 内存 op */
            em.add_op(OP_DMA, {a_t, 0x4000u | (uint32_t)stg, act_bytes,
                               0u, 0u, 1u, WT_DMA_FENCE_NONE, WT_DMA_FENCE_NONE});
            em.cur_engine = prev_eng;
            em.pending_stages.push_back({(uint32_t)(em.ops.size() - 1u), 0u,
                                         off, act_bytes});
            act_arg = 0x4000u | (uint32_t)stg;
        }
        em.add_op(OP_MATMUL_W4A16, {act_arg, w_raw, out_t, m, k, nn});
    } else {
        uint32_t flags = 0;
        if (nm == "MatMul")
            flags = (em2.transpose_in0 & 1u) | ((em2.transpose_in1 & 1u) << 1);
        else
            flags = 2u;  /* FC: 权重存 [N,K] */
        /* batched BMM: bit2 置位, 批数入高 16 位。批维 = 除末两维(M,N)外的
         * 全部前导维 —— 不得裁剪尾部 1 维: [1,8,1,1](attn q·kᵀ, M=N=1)会被
         * 剪掉真实批维 8 → Bn=1 → 执行器只写 1 元素, 其余 7 头读陈旧 temp
         * (seq=1 实锤: gid 4560 首发, 顺流污染全部 full-attn 层)。 */
        if (em2.batched) {
            flags |= 4u;
            uint32_t rk = od->output_def.rank;
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
        if (!has_bias && !em2.batched && (m % 32u) == 0u && (k % 32u) == 0u && (nn % 32u) == 0u &&
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
                /* 分块后每个 chunk 都满足 need_mm <= 16MB，必须保留
                 * HMX 计划；assign_engine 只看整 op 的形状，看不见这里的
                 * emit-side N 分块，否则 v3 会把所有 chunk 错标为 scalar。 */
                em.cur_engine = wt_annot_pack(2u, 1u, 0u);  /* HMX, 可回落 */
                for (uint32_t n0 = 0; n0 < nn; n0 += N_chunk) {
                    uint32_t nc = (N_chunk < nn - n0) ? N_chunk : (nn - n0);
                    em.add_op(OP_MATMUL_F16_SPLIT, {a_t, w_s, out_t, m, k, nn, n0, nc, flags});
                }
                return 0;
            }
        }
        if (has_bias) {
            /* OP_MATMUL_F16_BIAS (opcode 36): 尾部追加 bias 槽。
             * 005 US3b: 设备 HMX 路径已有 bias 通道(scale 区 out_bias,
             * SC-005 三判据 0 违例), 32 对齐形默认走 HMX —— US3a 的
             * 对齐硬报错已随 pass 门一并移除。 */
            uint32_t bias_s = 0x8000u | em.ensure_weight_slot(gp, bias_op, wslots, od, false);
            em.add_op(OP_MATMUL_F16_BIAS, {a_t, w_s, out_t, m, k, nn, flags, bias_s});
            return 0;
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
