/*
 * hvxhmx_v2_flash_attn.h — V2 Flash Attention 内核 (HVX + HMX)
 * =====================================================================
 * 源自 ggmlHTPV3E htp/hvx-fa-kernels.h + hmx-fa-kernels.h + hvx-flash-attn.h.
 *
 * 本头暴露 FA 的 tile 级 building blocks, 调用方自行编排 online-softmax 外层
 * 循环 (KV 分块 → qk_dot → online softmax → attn·V 累加 → 收尾 norm).
 * 完整驱动 (含 KV cache 遍历/掩码/cross-chunk 流水) 在 ggmlHTPV3E flash-attn-ops.c
 * 里与 tensor 模型耦合, 此处仅取其已验证的数学内核.
 *
 * 数学 (标准 flash attention):
 *   S = Q·K^T * scale             (qk_dot)
 *   P = softmax(S, mask)          (外层循环, 见 hvx_v2_softmax)
 *   O = P·V                       (attn_v_mad / o_update_tile)
 *   O_final = O / rowmax_exp_sum  (o_norm_tile)
 *
 * 对齐: 所有 tile 2KB(VTCM)/128B 对齐. head_dim 通常 64/128.
 */
#ifndef HVXHMX_V2_FLASH_ATTN_H
#define HVXHMX_V2_FLASH_ATTN_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  HVX FA 内核 (fp16 Q/K/V, f32 累加)
 * ============================================================ */

/**
 * @brief 单个 Q·K^T 点积 → 缩放后的标量 (f32).
 * @param q   Q 行, fp16 [head_dim], 128B 对齐
 * @param k   K 行, fp16 [head_dim], 128B 对齐
 * @param head_dim  D (≤128 走单向量量; >128 自动多向量)
 * @param scale     1/sqrt(head_dim)
 * @param out       输出 f32 标量 (1 个 float)
 * @note  对齐版 (q/k 均 128B 对齐).
 */
void hvhx_v2_fa_qk_dot_f16(const void * __restrict__ q,
                            const void * __restrict__ k,
                            uint32_t head_dim, float scale,
                            float * __restrict__ out);

/**
 * @brief attn·V 累加: y(f32) += x(fp16) * s(fp16 标量).
 *        FA 里 P_row · V_chunk 的逐 chunk 累加内核.
 * @param y    f32 累加缓冲 [head_dim], 128B 对齐, in/out
 * @param x    V 行 fp16 [head_dim], 128B 对齐
 * @param s    注意力权重 fp16 标量 (softmax(P)[i])
 * @param head_dim  D
 */
void hvhx_v2_fa_attn_v_mad_f16(float * __restrict__ y,
                                const void * __restrict__ x,
                                const __fp16 * __restrict__ s,
                                uint32_t head_dim);

/**
 * @brief ALiBi 偏置斜率向量 (32 个 head 的 slope).
 * @param kv_head     当前 KV head 索引
 * @param G           GQA group size (n_head / n_kv_head)
 * @param n_head_log2 log2(n_head)
 * @param m0, m1      ALiBi 斜率参数
 * @param out         输出 f32 [32], 128B 对齐
 */
void hvhx_v2_alibi_slopes(uint32_t kv_head, uint32_t G,
                          uint32_t n_head_log2, float m0, float m1,
                          float * __restrict__ out);

/* ============================================================
 *  HMX FA 内核 (32×32 tile, fp16)
 *  Q/K/V 需先用 hmx_interleave_* 转成 tile-major interleaved.
 *
 *  online-softmax 语义 (与 ggmlHTPV3E hmx-fa-kernels.h 一致):
 *    d_diag = 1/row_exp_sum (broadcast tile, 每行一个值铺 32 列)
 *    o_rc   = 上一 KV chunk 的输出 tile (或初值 0)
 *    qk_dot_tile:   out = Σ_d Q[r,d]·K[c,d]
 *    o_update_tile: o_out = d_diag ⊗ o_rc + Σ_k P[r,k]·V[k,c]
 *    o_norm_tile:   o_out = d_diag ⊗ o_rc        (收尾归一化)
 *
 *  注: 本头只暴露已验证的 tile 数学内核, 调用方负责 online-softmax 外层
 *  循环 (rowmax/row_exp_sum 更新, mask, KV chunk 遍历, d_diag 计算).
 * ============================================================ */

/**
 * @brief HMX QK 点积 tile: out[r,c] = Σ_d Q[r,d]·K[c,d], 32×32 tile.
 * @param row_tiles   Q tile 序列 fp16 [n_dot_tiles × 1024], 2KB 对齐
 * @param col_tiles   K tile 序列 fp16 [n_dot_tiles × 1024], 2KB 对齐
 * @param out_tile    输出 fp16 [1024] (32×32), 2KB 对齐
 * @param n_dot_tiles head_dim/32 (2/4/8 走展开快路径, 其他走循环)
 * @note  内部不调 hmx_enable_execution — 由外层统一管理 HMX 锁.
 */
void hvhx_v2_hmx_fa_qk_dot_tile(const __fp16 * __restrict__ row_tiles,
                                 const __fp16 * __restrict__ col_tiles,
                                 __fp16 * __restrict__ out_tile,
                                 uint32_t n_dot_tiles);

/**
 * @brief HMX 输出 tile 更新: o_out = d_diag ⊗ o_rc + Σ P·V.
 * @param d_diag      1/exp_sum 归一化 tile fp16 [1024], 2KB 对齐
 * @param o_rc        当前 O tile fp16 [1024] (累加基), 2KB 对齐
 * @param p_tile_in   注意力权重 P tile 序列 fp16 [n_col_tiles × 1024]
 * @param v_tile_in   V tile 序列 fp16 [n_col_tiles × 1024]
 * @param o_tile_out  输出 O tile fp16 [1024], 2KB 对齐
 * @param n_col_tiles head_dim/32 (2/4/8 展开快路径)
 */
void hvhx_v2_hmx_fa_o_update_tile(const __fp16 * __restrict__ d_diag,
                                   const __fp16 * __restrict__ o_rc,
                                   const __fp16 * __restrict__ p_tile_in,
                                   const __fp16 * __restrict__ v_tile_in,
                                   __fp16 * __restrict__ o_tile_out,
                                   uint32_t n_col_tiles);

/**
 * @brief HMX 输出 tile 归一化: o_out = d_diag ⊗ o_rc (收尾).
 * @param d_diag  1/exp_sum tile fp16 [1024], 2KB 对齐
 * @param o_rc    待归一化 O tile fp16 [1024], 2KB 对齐
 * @param o_out   输出 fp16 [1024], 2KB 对齐
 */
void hvhx_v2_hmx_fa_o_norm_tile(const __fp16 * __restrict__ d_diag,
                                 const __fp16 * __restrict__ o_rc,
                                 __fp16 * __restrict__ o_out);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_FLASH_ATTN_H */
