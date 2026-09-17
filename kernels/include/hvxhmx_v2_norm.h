/*
 * hvxhmx_v2_norm.h — V2 归一化算子 (RMSNorm / LayerNorm / L2Norm)
 * =====================================================================
 * 源自 ggmlHTPV3E htp/hvx-norm.h, HVX 128B 向量化, 纯 f32 路径.
 * 单行 kernel 已极致优化 (一次 pass 算 sum-of-sq + 第二 pass scale), 这里
 * 提供面向 LLM 的 batched 入口 (n_rows × row_size), 128B 对齐由调用方保证.
 *
 * 数学:
 *   rms_norm:    dst = x / sqrt(mean(x^2) + eps)
 *   rms_norm_mul:dst = w * x / sqrt(mean(x^2) + eps)
 *   norm:        dst = (x - mean) / sqrt(var + eps)
 *   l2_norm:     dst = x / fmax(sqrt(sum(x^2)), eps)
 *
 * 对齐: src/dst/weight 必须 128B 对齐; row_size 任意 (尾部用 predicate 掩码).
 */
#ifndef HVXHMX_V2_NORM_H
#define HVXHMX_V2_NORM_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 单行 (row_size 元素) ---- */

/** @brief RMSNorm, 单行 f32. dst[i] = x[i] / sqrt(mean(x^2)+eps). */
void hvhx_v2_rms_norm_f32(const float * __restrict__ src,
                          float       * __restrict__ dst,
                          uint32_t num_elems, float eps);

/** @brief RMSNorm + 逐元素权重, 单行 f32. (LLM RMSNorm 标准形态) */
void hvhx_v2_rms_norm_mul_f32(const float * __restrict__ src,
                              const float * __restrict__ weight,
                              float       * __restrict__ dst,
                              uint32_t num_elems, float eps);

/** @brief LayerNorm, 单行 f32. dst = (x-mean)/sqrt(var+eps). */
void hvhx_v2_norm_f32(const float * __restrict__ src,
                      float       * __restrict__ dst,
                      uint32_t num_elems, float eps);

/** @brief L2Norm, 单行 f32. dst = x / fmax(sqrt(sum(x^2)),eps). */
void hvhx_v2_l2_norm_f32(const float * __restrict__ src,
                         float       * __restrict__ dst,
                         uint32_t num_elems, float eps);

/* ---- 批量 (n_rows 行, 每行 row_size) —— LLM 推理主路径 ---- */

/**
 * @brief 批量 RMSNorm + 权重. src/dst [n_rows, row_size], weight [row_size].
 * @note  这是 transformer block 里 RMSNorm 的标准调用形态.
 *        每行独立归一化. 行间数据连续 (row-major).
 */
void hvhx_v2_rms_norm_mul_f32_rows(const float * __restrict__ src,
                                   const float * __restrict__ weight,
                                   float       * __restrict__ dst,
                                   uint32_t n_rows, uint32_t row_size, float eps);

/**
 * @brief 批量 RMSNorm (无权重). src/dst [n_rows, row_size].
 */
void hvhx_v2_rms_norm_f32_rows(const float * __restrict__ src,
                               float       * __restrict__ dst,
                               uint32_t n_rows, uint32_t row_size, float eps);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_NORM_H */
