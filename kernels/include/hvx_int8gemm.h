/*
 * hvx_int8gemm.h — V81 HVX int8 GEMM (uint8 × int8 → uint8)
 */
#ifndef HVXHMX_HVX_INT8GEMM_H
#define HVXHMX_HVX_INT8GEMM_H

#include <stdint.h>
#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* dense [32][32] wgt → padded [32][128] wgt (32B data + 96B zero per row).
 * HVX core 要求 padded 布局 (128B stride, 非重叠 load). */
void hvx_int8gemm_pack_wgt(const int8_t * __restrict__ wgt_dense,
                           int8_t        * __restrict__ wgt_padded);

/* 单 tile 32×32×32 GEMM, VTCM 指针入参.
 *   act_v:  [32][32] row-major, 1KB
 *   wgt_v:  [32][128] padded (每行 32B data + 96B zero), 4KB, 128B 对齐
 *   out_v:  [32][32] row-major, 1KB
 * out[m][n] = sat_u8( Σ_k act_u8[m][k] * wgt_i8[k][n] ).
 * 精度: int16 累加器, 要求 |Σ| < 32768. */
void hvx_int8gemm_32x32x32_core(const uint8_t * __restrict__ act_v,
                                const int8_t  * __restrict__ wgt_v,
                                uint8_t       * __restrict__ out_v);

/* bias-aware 单 tile 32×32×32 GEMM (VTCM 指针入参).
 *   act_v:  [32][32] row-major, 1KB
 *   wgt_v:  [32][128] padded, 4KB, 128B 对齐
 *   bias_v: [32] int16 (截断自 int32; 须 fit int16)
 *   out_v:  [32][32] row-major, 1KB
 * out[m][n] = sat_u8( bias[n] + Σ_k act_u8[m][k] * wgt_i8[k][n] ).
 * bias 在 vasr 之前以 int16 加到累加器 lane (偶 col→lo[j], 奇 col→hi[j]). */
void hvx_int8gemm_bias_32x32x32_core(const uint8_t * __restrict__ act_v,
                                     const int8_t  * __restrict__ wgt_v,
                                     const int16_t * __restrict__ bias_v,
                                     uint8_t       * __restrict__ out_v);

/* bias-aware 单 tile 32×32×32 GEMM, uint16 写回 (convbbh 用).
 *   out[m][n] = sat_u16( bias[n] + Σ_k act_u8[m][k] * wgt_i8[k][n] )
 * int16 累加器: 正值直接按 u16 写, 负值钳 0. */
void hvx_int8gemm_bias_u16_32x32x32_core(const uint8_t  * __restrict__ act_v,
                                         const int8_t   * __restrict__ wgt_v,
                                         const int16_t  * __restrict__ bias_v,
                                         uint16_t       * __restrict__ out_v);

/* 通用接口: M=K=N=32 走 HVX tile (内部 pack), 否则退标量. */
void hvx_int8gemm(const uint8_t *act, const int8_t *wgt,
                  uint8_t *out, uint32_t M, uint32_t K, uint32_t N);

/* 多 tile M/N/K GEMM (uint8 写回). M/K/N 必须是 32 倍数, 否则退标量.
 *   act:  [M][K] row-major
 *   wgt:  [K][N] row-major
 *   bias: [N] int16
 *   out:  [M][N] row-major uint8
 * K 维在 int16 acc 跨 kt 累加 (小值/中 K 安全; 大 K 大值溢出退标量). */
void hvx_int8gemm_bias_multi(const uint8_t * __restrict__ act,
                             const int8_t  * __restrict__ wgt,
                             const int16_t * __restrict__ bias,
                             uint8_t       * __restrict__ out,
                             uint32_t M, uint32_t K, uint32_t N);

/* 多 tile M/N/K GEMM (uint16 写回, convbbh 用). 同上, 输出 sat_u16. */
void hvx_int8gemm_bias_u16_multi(const uint8_t  * __restrict__ act,
                                 const int8_t   * __restrict__ wgt,
                                 const int16_t  * __restrict__ bias,
                                 uint16_t       * __restrict__ out,
                                 uint32_t M, uint32_t K, uint32_t N);

/* 诊断版: dump acc pair raw int16 (256B). wgt_v_padded = [32][128] padded. */
void hvx_int8gemm_diag(const uint8_t *act_v, const int8_t *wgt_v_padded, int16_t *dump);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_HVX_INT8GEMM_H */
