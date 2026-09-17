/*
 * hvxhmx_v2_ssm.h — V2 状态空间模型内核 (solve_tri + gated delta net)
 * =====================================================================
 * 源自 ggmlHTPV3E htp/:
 *   - solve-tri-ops.c       (solve_tri_row_{scalar,hvx}: 下三角逐行前代)
 *   - gated-delta-net-ops.c (gdn_mul_dot{4,8} / gdn_add_scaled_dot{4,8})
 *   - ssm-conv.c            (hvx_transpose_32x32_f32 + ssm_conv_dot32_hvx +
 *                            标量 depthwise 1D causal conv)
 *
 * === solve_tri (下三角前代解) ===
 *   解 L·X = B, L 为下三角 (unit 或给定对角逆). 逐行前代:
 *     X[row, col] = (B[row, col] - Σ_{t<row} L[row,t]·X[t,col]) * inv_diag
 *   HVX 版: 同一 row 的 col 向量化 (32 col 一次), 依赖 X 已算的前 row.
 *
 * === gated delta net (线性注意力 + delta 规则状态更新) ===
 *   dot4/dot8 = 并行状态向量数 (4 或 8 个 dst 缓冲).
 *   gdn_mul_dot{N}:    dst_k[i] *= mul[i];    sums[k] += Σ_i dst_k[i]*dot[i]
 *   gdn_add_scaled_{N}: dst_k[i] += src[i]*scale[0]; sums[k] += Σ_i dst_k[i]*dot[i]
 *   sums[k] 是每个状态向量的 dot 积, 用于后续 state 读写门控.
 *
 * 约束: n 为 32 倍数 (HVX 向量边界); 状态缓冲 dst_k 建议在 VTCM.
 */
#ifndef HVXHMX_V2_SSM_H
#define HVXHMX_V2_SSM_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  下三角前代解 (逐行)
 * ============================================================ */

/**
 * @brief 解下三角系统 L·X = B 的第 row 行 (前代).
 * @param A_row     L 的第 row 行, f32 [row] (L[row, 0..row-1])
 * @param B_row     B 的第 row 行, f32 [col0+coln]
 * @param X         解矩阵 X, f32 [k × (col0+coln)], 前面 row 已算好
 * @param row       当前行索引 (0-based)
 * @param k         X 的行步长 (leading dim)
 * @param col0      列起始偏移
 * @param coln      列数 (HVX 版 ≤32; >32 调用方分块)
 * @param inv_diag  1 / L[row,row] (对角逆, 预计算)
 * @note HVX 路径要求 coln ≤ 32; 否则自动走标量.
 */
void hvhx_v2_solve_tri_row_f32(const float * __restrict__ A_row,
                                const float * __restrict__ B_row,
                                float * __restrict__ X,
                                uint32_t row, uint32_t k,
                                uint32_t col0, uint32_t coln,
                                float inv_diag);

/* ============================================================
 *  Gated Delta Net — 4 状态
 * ============================================================ */

/**
 * @brief 4 状态门控更新 + dot 积: dst_k[i] *= mul[i]; sums[k] = Σ dst_k[i]·dot[i].
 * @param dst0..dst3  状态向量 f32 [n], in/out (VTCM 性能最佳)
 * @param mul         门控向量 f32 [n] (forget gate, 所有状态共享)
 * @param dot         查询向量 f32 [n]
 * @param n           元素数 (32 倍数)
 * @param sums        输出 4 个 f32 [4] (每状态的 dot 积)
 */
void hvhx_v2_gdn_mul_dot4_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                               float * __restrict__ dst2, float * __restrict__ dst3,
                               const float * __restrict__ mul,
                               const float * __restrict__ dot,
                               uint32_t n, float * __restrict__ sums);

/**
 * @brief 4 状态累加缩放 + dot 积: dst_k[i] += src[i]*scale[0]; sums[k] = Σ dst_k[i]·dot[i].
 * @param src    输入向量 f32 [n]
 * @param scale  f32 [1] (标量缩放, 存为数组以对齐 ABI)
 */
void hvhx_v2_gdn_add_scaled_dot4_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                                      float * __restrict__ dst2, float * __restrict__ dst3,
                                      const float * __restrict__ src,
                                      const float * __restrict__ scale,
                                      const float * __restrict__ dot,
                                      uint32_t n, float * __restrict__ sums);

/* ============================================================
 *  Gated Delta Net — 8 状态
 * ============================================================ */
void hvhx_v2_gdn_mul_dot8_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                               float * __restrict__ dst2, float * __restrict__ dst3,
                               float * __restrict__ dst4, float * __restrict__ dst5,
                               float * __restrict__ dst6, float * __restrict__ dst7,
                               const float * __restrict__ mul,
                               const float * __restrict__ dot,
                               uint32_t n, float * __restrict__ sums);

void hvhx_v2_gdn_add_scaled_dot8_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                                      float * __restrict__ dst2, float * __restrict__ dst3,
                                      float * __restrict__ dst4, float * __restrict__ dst5,
                                      float * __restrict__ dst6, float * __restrict__ dst7,
                                      const float * __restrict__ src,
                                      const float * __restrict__ scale,
                                      const float * __restrict__ dot,
                                      uint32_t n, float * __restrict__ sums);

/* ============================================================
 *  SSM depthwise 1D causal conv (Mamba/Jamba 前导卷积)
 *
 *  数学: dst[i1, i2, i3] = Σ_{j=0}^{d_conv-1} src0[i2+j, i1, i3] * src1[j, i1]
 *
 *  提供两级 building block:
 *    - hvhx_v2_ssm_conv_f32:       全量 row-major (标量, 任意尺寸, 正确性基准)
 *    - hvhx_v2_ssm_conv_dot32_f32: HVX 32-channel tile 级 (已转置布局, 高吞吐)
 *  + hvhx_v2_transpose_32x32_f32:  32×32 f32 转置布局工具 (dot32 的前置)
 * ============================================================ */

/**
 * @brief 全量 depthwise 1D causal conv (row-major, 标量).
 *
 * @param src0       [n_s][d_inner][ncs] f32 — 左 padding 后的输入 (ncs = n_t + d_conv - 1)
 * @param src1       [d_inner][d_conv]   f32 — 每 channel 一个 d_conv 抽头滤波器
 * @param dst        [n_s][n_t][d_inner] f32 — conv 输出
 * @param d_inner    channel 数 (src0 dim 1 / src1 dim 1 / dst dim 2)
 * @param n_t        输出时间步数 (dst dim 1)
 * @param n_s        sequence 数 (src0/dst dim 0)
 * @param ncs        padding 后时间维 (src0 dim 2 = n_t + d_conv - 1)
 * @param d_conv     卷积核长 (src1 dim 1)
 * @param apply_silu 1 = 写 silu(conv_out), 0 = 写 conv_out
 *
 * @note 三个数组均 row-major 连续. 标量实现 (DSP + host 一致), 任意尺寸可用.
 *       高吞吐路径见 hvhx_v2_ssm_conv_dot32_f32.
 */
void hvhx_v2_ssm_conv_f32(const float * __restrict__ src0,
                          const float * __restrict__ src1,
                          float       * __restrict__ dst,
                          uint32_t d_inner, uint32_t n_t, uint32_t n_s,
                          uint32_t ncs, uint32_t d_conv,
                          int apply_silu);

/**
 * @brief HVX 32-channel depthwise conv dot (已转置 channel-contiguous 布局).
 *
 * 前置条件: 调用方已把 src0/src1 转置 (用 hvx_v2_transpose_32x32_f32 或等价),
 *   使每个 128B 向量 = 32 个连续 channel 在给定 (time, conv_tap) 处的值.
 *
 * @param dst32      输出 32 个 f32 (1 个 HVX_Vector: 32 channel 的 conv 结果)
 * @param x_row      转置后 src0 基址: x_row[j*x_stride] = 时间偏移 (t+j) 的 32-channel 向量
 * @param x_stride   连续时间偏移间距 (float 单位)
 * @param w_row      转置后 src1 基址: w_row[j*w_stride] = 第 j 抽头的 32-channel 权重
 * @param w_stride   连续抽头间距 (float 单位)
 * @param d_conv     卷积核长
 * @param apply_silu 1 = 写 silu(conv_out)
 *
 * @note src: ssm-conv.c:318-345 内核体. host 路径标量等价 (32 channel 循环).
 */
void hvhx_v2_ssm_conv_dot32_f32(float       * __restrict__ dst32,
                                const float * __restrict__ x_row, uint32_t x_stride,
                                const float * __restrict__ w_row, uint32_t w_stride,
                                uint32_t d_conv, int apply_silu);

/**
 * @brief 32×32 f32 矩阵转置 (5-stage HVX vshuff butterfly, 寄存器内).
 * @param m32x32  f32 [32][32] (= 1024 floats, 行主序), in-place 转置.
 *
 * @note src: ssm-conv.c:140-186. 纯布局工具, dot32 的前置变换.
 *       host 路径标量转置. m32x32 必须 128B 对齐 (DSP).
 */
void hvhx_v2_transpose_32x32_f32(float * __restrict__ m32x32);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_SSM_H */
