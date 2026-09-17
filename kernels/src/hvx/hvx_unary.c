/*
 * hvx_unary.c — V2 一元数学函数封装
 * Module: v2-unary
 * 源:    ggmlHTPV3E htp/hvx-{exp,log,sqrt,sigmoid,sin-cos,pow,inverse,
 *        floor,scale,arith,reduce}.h
 *
 * 策略:
 *   - 有 buffer-level wrapper 的 (exp/sqrt/sigmoid/scale/sqr/reduce_sum):
 *     直接调.
 *   - 仅有 vec-level 的 (log/rsqrt/tanh/sin/cos/pow/inverse/floor/truncate):
 *     全向量循环 + 标量尾.
 */
#include "hvxhmx_v2_unary.h"
#include "hvxhmx_types.h"

#include <math.h>

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-utils.h"   /* 拉入全部 hvx-* math 头 */
#endif

#define VLEN_F32 32u   /* f32 per HVX_Vector (128B / 4B) */

#if HVX_V2_KERNELS_ENABLED
/* 通用 vec-only buffer wrapper: 全向量走 VEC_OP, 尾部标量 SCALAR_EXPR (用 x). */
#define V2_VEC_LOOP(dst, src, n, VEC_OP, SCALAR_EXPR)                      \
    do {                                                                    \
        const uint32_t _nvec = (n) / VLEN_F32;                              \
        const uint32_t _tail = (n) - _nvec * VLEN_F32;                      \
        HVX_Vector       * __restrict__ _dv = (HVX_Vector *)(dst);          \
        const HVX_Vector * __restrict__ _sv = (const HVX_Vector *)(src);    \
        for (uint32_t _i = 0; _i < _nvec; ++_i) _dv[_i] = (VEC_OP)(_sv[_i]);\
        for (uint32_t _i = 0; _i < _tail; ++_i) {                           \
            float x = ((const float *)(src))[_nvec * VLEN_F32 + _i];        \
            ((float *)(dst))[_nvec * VLEN_F32 + _i] = (SCALAR_EXPR);        \
        }                                                                   \
    } while (0)
#endif

/* ============================================================
 *  指数 / 对数
 * ============================================================ */
void hvhx_v2_exp_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_exp_f32((uint8_t *)dst, (const uint8_t *)src, (int)n, false);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = expf(src[i]);
#endif
}

void hvhx_v2_neg_exp_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_exp_f32((uint8_t *)dst, (const uint8_t *)src, (int)n, true);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = expf(-src[i]);
#endif
}

void hvhx_v2_log_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_log_f32, logf(x));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = logf(src[i]);
#endif
}

/* ============================================================
 *  平方根 / 逆平方根
 * ============================================================ */
void hvhx_v2_sqrt_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_sqrt_f32((uint8_t *)dst, (const uint8_t *)src, (int)n);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = sqrtf(src[i]);
#endif
}

void hvhx_v2_rsqrt_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_rsqrt_f32, (1.0f / sqrtf(x)));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = 1.0f / sqrtf(src[i]);
#endif
}

/* ============================================================
 *  激活
 * ============================================================ */
void hvhx_v2_sigmoid_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    /* sigmoid 无 dispatcher, 内联 aa/au/ua/uu 选择 */
    const int da = ((uintptr_t)dst % 128u) == 0;
    const int sa = ((uintptr_t)src % 128u) == 0;
    if (da && sa)       hvx_sigmoid_f32_aa((uint8_t *)dst, (const uint8_t *)src, n);
    else if (da)        hvx_sigmoid_f32_au((uint8_t *)dst, (const uint8_t *)src, n);
    else if (sa)        hvx_sigmoid_f32_ua((uint8_t *)dst, (const uint8_t *)src, n);
    else                hvx_sigmoid_f32_uu((uint8_t *)dst, (const uint8_t *)src, n);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = 1.0f / (1.0f + expf(-src[i]));
#endif
}

void hvhx_v2_tanh_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_tanh_f32, tanhf(x));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = tanhf(src[i]);
#endif
}

/* ---- silu / gelu / softplus: vec 级 building block 组合 (act-ops.c 数学) ---- */
#if HVX_V2_KERNELS_ENABLED
/* in-register sigmoid = 1/(1+exp(-x)), exp 经 guard 防溢出. */
static inline HVX_Vector hvx_v2_vec_sigmoid_f32(HVX_Vector x, HVX_Vector one,
                                                 HVX_Vector zero, HVX_Vector max_exp,
                                                 HVX_Vector inf) {
    HVX_Vector neg_x = hvx_vec_sub_f32_f32(zero, x);
    HVX_Vector ex    = hvx_vec_exp_f32_guard(neg_x, max_exp, inf);
    return hvx_vec_inverse_f32(hvx_vec_add_f32_f32(one, ex));
}
#endif

void hvhx_v2_silu_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector zero    = Q6_V_vzero();
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector inf     = hvx_vec_splat_f32(INFINITY);
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ sv = (const HVX_Vector *)src;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector x = sv[i];
        HVX_Vector s = hvx_v2_vec_sigmoid_f32(x, one, zero, max_exp, inf);
        dv[i] = hvx_vec_mul_f32_f32(x, s);
    }
    for (uint32_t i = 0; i < tail; ++i) {
        float x = src[nvec * VLEN_F32 + i];
        dst[nvec * VLEN_F32 + i] = x / (1.0f + expf(-x));
    }
#else
    for (uint32_t i = 0; i < n; ++i) {
        float x = src[i];
        dst[i] = x / (1.0f + expf(-x));
    }
#endif
}

void hvhx_v2_gelu_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector zero    = Q6_V_vzero();
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector inf     = hvx_vec_splat_f32(INFINITY);
    const HVX_Vector scale   = hvx_vec_splat_f32(1.702f);
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ sv = (const HVX_Vector *)src;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector x = sv[i];
        HVX_Vector s = hvx_v2_vec_sigmoid_f32(hvx_vec_mul_f32_f32(x, scale),
                                               one, zero, max_exp, inf);
        dv[i] = hvx_vec_mul_f32_f32(x, s);
    }
    for (uint32_t i = 0; i < tail; ++i) {
        float x = src[nvec * VLEN_F32 + i];
        dst[nvec * VLEN_F32 + i] = x / (1.0f + expf(-1.702f * x));
    }
#else
    for (uint32_t i = 0; i < n; ++i) {
        float x = src[i];
        dst[i] = x / (1.0f + expf(-1.702f * x));
    }
#endif
}

void hvhx_v2_gelu_tanh_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    const HVX_Vector one        = hvx_vec_splat_f32(1.0f);
    const HVX_Vector half       = hvx_vec_splat_f32(0.5f);
    const HVX_Vector coef_a     = hvx_vec_splat_f32(0.044715f);
    const HVX_Vector sqrt_2_pi  = hvx_vec_splat_f32(0.79788456080286535587989211986876f);
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ sv = (const HVX_Vector *)src;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector x = sv[i];
        HVX_Vector inner = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, x), coef_a);
        inner = hvx_vec_add_f32_f32(inner, one);
        inner = hvx_vec_mul_f32_f32(inner, x);
        inner = hvx_vec_mul_f32_f32(inner, sqrt_2_pi);
        HVX_Vector t = hvx_vec_add_f32_f32(hvx_vec_tanh_f32(inner), one);
        dv[i] = hvx_vec_mul_f32_f32(hvx_vec_mul_f32_f32(x, t), half);
    }
    for (uint32_t i = 0; i < tail; ++i) {
        float x = src[nvec * VLEN_F32 + i];
        float inner = 0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x);
        dst[nvec * VLEN_F32 + i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#else
    for (uint32_t i = 0; i < n; ++i) {
        float x = src[i];
        float inner = 0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x);
        dst[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
#endif
}

void hvhx_v2_softplus_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    const HVX_Vector one     = hvx_vec_splat_f32(1.0f);
    const HVX_Vector max_exp = hvx_vec_splat_f32(87.0f);
    const HVX_Vector inf     = hvx_vec_splat_f32(INFINITY);
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ sv = (const HVX_Vector *)src;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector ex = hvx_vec_exp_f32_guard(sv[i], max_exp, inf);
        dv[i] = hvx_vec_log_f32(hvx_vec_add_f32_f32(one, ex));
    }
    for (uint32_t i = 0; i < tail; ++i) {
        float x = src[nvec * VLEN_F32 + i];
        dst[nvec * VLEN_F32 + i] = log1pf(expf(x));
    }
#else
    for (uint32_t i = 0; i < n; ++i)
        dst[i] = log1pf(expf(src[i]));
#endif
}

/* ============================================================
 *  三角
 * ============================================================ */
void hvhx_v2_sin_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_sin_f32, sinf(x));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = sinf(src[i]);
#endif
}

void hvhx_v2_cos_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_cos_f32, cosf(x));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = cosf(src[i]);
#endif
}

/* ============================================================
 *  幂
 * ============================================================ */
void hvhx_v2_pow_f32(float * __restrict__ dst, const float * __restrict__ base,
                     const float * __restrict__ exponent, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ bv = (const HVX_Vector *)base;
    const HVX_Vector * __restrict__ ev = (const HVX_Vector *)exponent;
    for (uint32_t i = 0; i < nvec; ++i) dv[i] = hvx_vec_pow_f32(bv[i], ev[i]);
    for (uint32_t i = 0; i < tail; ++i) {
        uint32_t j = nvec * VLEN_F32 + i;
        dst[j] = powf(base[j], exponent[j]);
    }
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = powf(base[i], exponent[i]);
#endif
}

void hvhx_v2_pow_const_base_f32(float * __restrict__ dst, float base,
                                const float * __restrict__ exponent_src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ ev = (const HVX_Vector *)exponent_src;
    for (uint32_t i = 0; i < nvec; ++i) dv[i] = hvx_vec_pow_const_base_f32(base, ev[i]);
    for (uint32_t i = 0; i < tail; ++i) {
        uint32_t j = nvec * VLEN_F32 + i;
        dst[j] = powf(base, exponent_src[j]);
    }
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = powf(base, exponent_src[i]);
#endif
}

/* ============================================================
 *  倒数 / 平方
 * ============================================================ */
void hvhx_v2_inverse_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_inverse_f32, (1.0f / x));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = 1.0f / src[i];
#endif
}

void hvhx_v2_sqr_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_sqr_f32((uint8_t *)dst, (const uint8_t *)src, n);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = src[i] * src[i];
#endif
}

/* ============================================================
 *  取整
 * ============================================================ */
void hvhx_v2_floor_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_floor_f32, floorf(x));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = floorf(src[i]);
#endif
}

void hvhx_v2_truncate_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_VEC_LOOP(dst, src, n, hvx_vec_truncate_f32, truncf(x));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = truncf(src[i]);
#endif
}

/* ============================================================
 *  缩放
 * ============================================================ */
void hvhx_v2_scale_f32(float * __restrict__ dst, const float * __restrict__ src,
                       uint32_t n, float scale)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_scale_f32((uint8_t *)dst, (const uint8_t *)src, (int)n, scale);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = src[i] * scale;
#endif
}

void hvhx_v2_scale_offset_f32(float * __restrict__ dst, const float * __restrict__ src,
                               uint32_t n, float scale, float offset)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_scale_offset_f32((uint8_t *)dst, (const uint8_t *)src, (int)n, scale, offset);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = src[i] * scale + offset;
#endif
}

/* ============================================================
 *  归约
 * ============================================================ */
float hvhx_v2_reduce_sum_f32(const float * __restrict__ src, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    return hvx_reduce_sum_f32((const uint8_t *)src, (int)n);
#else
    double acc = 0.0;
    for (uint32_t i = 0; i < n; ++i) acc += src[i];
    return (float)acc;
#endif
}
