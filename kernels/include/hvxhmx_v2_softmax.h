/*
 * hvxhmx_v2_softmax.h — V2 Softmax (HVX 向量化 online softmax)
 * =====================================================================
 * 源自 ggmlHTPV3E htp/softmax-ops.c:
 *   - hvx_fast_softmax_f32       (fused max→exp→sum→scale, 32 倍数 n)
 *   - hvx_fast_softmax_prep_f32  (score*scale + mask*slope, FA mask 预处理)
 *
 * 数学: softmax(x)[i] = exp(x[i] - max) / Σ exp(x[j] - max)
 *
 * 约束: fused 版要求 n 为 32 的倍数 (HVX vector 边界), 且 src/dst/pad 均
 * 128B 对齐 (VTCM 性能最佳). 非对齐/非 32 倍数时走标量 fallback.
 */
#ifndef HVXHMX_V2_SOFTMAX_H
#define HVXHMX_V2_SOFTMAX_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fused online softmax (max → exp → sum → scale), 单趟完成.
 * @param src  输入 f32 [n], 128B 对齐, n 为 32 倍数
 * @param dst  输出 f32 [n], 128B 对齐
 * @param pad  scratch f32 [n], VTCM (存放 exp 中间结果)
 * @param n    元素数 (必须 32 倍数; 否则走标量 fallback)
 */
void hvhx_v2_softmax_f32(const float * __restrict__ src,
                          float * __restrict__ dst,
                          float * __restrict__ pad,
                          uint32_t n);

/**
 * @brief FA mask 预处理: dst[i] = src[i]*scale + mask[i]*slope.
 *        Flash Attention 里 score·scale + ALiBi/causal-mask·slope 的合并步骤.
 * @param src   score f32 [n], 128B 对齐
 * @param dst   输出 f32 [n], 128B 对齐
 * @param n     元素数 (32 倍数)
 * @param scale attention scale = 1/sqrt(head_dim)
 * @param mask  位置偏置 f32 [n] (ALiBi slope / causal -inf)
 * @param slope mask 系数 (通常 1.0; ALiBi 时为 slope)
 */
void hvhx_v2_softmax_mask_f32(const float * __restrict__ src,
                               float * __restrict__ dst,
                               uint32_t n, float scale,
                               const float * __restrict__ mask, float slope);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_SOFTMAX_H */
