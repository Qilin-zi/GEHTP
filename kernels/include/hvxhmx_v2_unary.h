/*
 * hvxhmx_v2_unary.h — V2 一元数学函数族 (HVX 向量化)
 * =====================================================================
 * 源自 ggmlHTPV3E htp/hvx-{exp,log,sqrt,sigmoid,sin-cos,pow,inverse,
 * floor,scale,arith,reduce}.h 的 buffer-level 封装.
 *
 * 所有函数: dst[i] = f(src[i]), i ∈ [0, n), f32 in/out.
 * 对齐: 128B 对齐性能最佳; 内部自动处理 unaligned 尾部.
 *
 * 精度: HVX 路径用向量近似 (exp/log/sin/cos 等), 与标量 libm 差异
 * < 1 ULP 量级, 适合 LLM 推理. host 路径用 libm 标量.
 */
#ifndef HVXHMX_V2_UNARY_H
#define HVXHMX_V2_UNARY_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 指数 / 对数 ---- */
void hvhx_v2_exp_f32      (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_neg_exp_f32  (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);  /* exp(-x) */
void hvhx_v2_log_f32      (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);

/* ---- 平方根 / 逆平方根 ---- */
void hvhx_v2_sqrt_f32     (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_rsqrt_f32    (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);

/* ---- 激活: sigmoid / tanh ---- */
void hvhx_v2_sigmoid_f32  (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_tanh_f32     (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);

/* ---- 激活: silu / gelu / softplus (act-ops.c 数学) ----
 * silu(x)     = x * sigmoid(x)                              (act-ops.c:479)
 * gelu(x)     = x * sigmoid(1.702 * x)                      (act-ops.c:381, quick 近似)
 * gelu_tanh(x)= 0.5*x*(1+tanh(√(2/π)·x·(1+0.044715·x²)))   (act-ops.c:589)
 * softplus(x) = log(1 + exp(x))                             (简单式, x>20 精度下降)
 */
void hvhx_v2_silu_f32       (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_gelu_f32       (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_gelu_tanh_f32  (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_softplus_f32   (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);

/* ---- 三角: sin / cos ---- */
void hvhx_v2_sin_f32      (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_cos_f32      (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);

/* ---- 幂: element-wise 与 const-base ---- */
void hvhx_v2_pow_f32             (float * __restrict__ dst, const float * __restrict__ base,
                                  const float * __restrict__ exponent, uint32_t n);
void hvhx_v2_pow_const_base_f32  (float * __restrict__ dst, float base,
                                  const float * __restrict__ exponent_src, uint32_t n);

/* ---- 倒数 / 平方 ---- */
void hvhx_v2_inverse_f32  (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_sqr_f32      (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);

/* ---- 取整 ---- */
void hvhx_v2_floor_f32    (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);
void hvhx_v2_truncate_f32 (float * __restrict__ dst, const float * __restrict__ src, uint32_t n);

/* ---- 缩放 / 缩放+偏移 ---- */
void hvhx_v2_scale_f32        (float * __restrict__ dst, const float * __restrict__ src,
                               uint32_t n, float scale);
void hvhx_v2_scale_offset_f32 (float * __restrict__ dst, const float * __restrict__ src,
                               uint32_t n, float scale, float offset);

/* ---- 归约: sum → 标量 ---- */
float hvhx_v2_reduce_sum_f32(const float * __restrict__ src, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_UNARY_H */
