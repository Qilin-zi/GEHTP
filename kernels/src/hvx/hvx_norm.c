/*
 * hvx_norm.c — V2 归一化算子封装 (RMSNorm / LayerNorm / L2Norm)
 * Module: v2-norm
 * 源:    ggmlHTPV3E htp/hvx-norm.h (static inline HVX kernel)
 * 封装:   把单行 kernel 套上 hvhx_v2_* 公共符号 + host 标量 fallback +
 *         n_rows 批量循环.
 */
#include "hvxhmx_v2_norm.h"
#include "hvxhmx_types.h"

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-norm.h"
#endif

#include <math.h>

/* ============================================================
 *  host 标量 fallback (与 DSP 路径数值一致, 供 host 编译/回归对比)
 * ============================================================ */
#if !HVX_V2_KERNELS_ENABLED
static void scalar_rms_norm_f32(const float *src, float *dst,
                                uint32_t n, float eps)
{
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)src[i] * src[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; ++i) dst[i] = src[i] * scale;
}

static void scalar_rms_norm_mul_f32(const float *src, const float *w,
                                    float *dst, uint32_t n, float eps)
{
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)src[i] * src[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; ++i) dst[i] = src[i] * scale * w[i];
}

static void scalar_norm_f32(const float *src, float *dst, uint32_t n, float eps)
{
    double sx = 0.0, sxx = 0.0;
    for (uint32_t i = 0; i < n; ++i) { sx += src[i]; sxx += (double)src[i]*src[i]; }
    double mean = sx / n;
    double var  = sxx / n - mean * mean;
    float scale = 1.0f / sqrtf((float)var + eps);
    for (uint32_t i = 0; i < n; ++i) dst[i] = (float)((src[i] - mean) * scale);
}

static void scalar_l2_norm_f32(const float *src, float *dst, uint32_t n, float eps)
{
    double ss = 0.0;
    for (uint32_t i = 0; i < n; ++i) ss += (double)src[i] * src[i];
    float denom = fmaxf(sqrtf((float)ss), eps);
    float scale = 1.0f / denom;
    for (uint32_t i = 0; i < n; ++i) dst[i] = src[i] * scale;
}
#endif /* !HVX_V2_KERNELS_ENABLED */

/* ============================================================
 *  单行公共 API
 * ============================================================ */
void hvhx_v2_rms_norm_f32(const float * __restrict__ src,
                          float       * __restrict__ dst,
                          uint32_t num_elems, float eps)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_fast_rms_norm_f32((const uint8_t *)src, (uint8_t *)dst,
                          (int)num_elems, eps);
#else
    scalar_rms_norm_f32(src, dst, num_elems, eps);
#endif
}

void hvhx_v2_rms_norm_mul_f32(const float * __restrict__ src,
                              const float * __restrict__ weight,
                              float       * __restrict__ dst,
                              uint32_t num_elems, float eps)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_fast_rms_norm_mul_f32((const uint8_t *)src, (const uint8_t *)weight,
                              (uint8_t *)dst, (int)num_elems, eps);
#else
    scalar_rms_norm_mul_f32(src, weight, dst, num_elems, eps);
#endif
}

void hvhx_v2_norm_f32(const float * __restrict__ src,
                      float       * __restrict__ dst,
                      uint32_t num_elems, float eps)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_fast_norm_f32((const uint8_t *)src, (uint8_t *)dst,
                      (int)num_elems, eps);
#else
    scalar_norm_f32(src, dst, num_elems, eps);
#endif
}

void hvhx_v2_l2_norm_f32(const float * __restrict__ src,
                         float       * __restrict__ dst,
                         uint32_t num_elems, float eps)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_fast_l2_norm_f32((const uint8_t *)src, (uint8_t *)dst,
                         (int)num_elems, eps);
#else
    scalar_l2_norm_f32(src, dst, num_elems, eps);
#endif
}

/* ============================================================
 *  批量 (n_rows 行) —— LLM 推理主路径
 * ============================================================ */
void hvhx_v2_rms_norm_mul_f32_rows(const float * __restrict__ src,
                                   const float * __restrict__ weight,
                                   float       * __restrict__ dst,
                                   uint32_t n_rows, uint32_t row_size, float eps)
{
    for (uint32_t r = 0; r < n_rows; ++r) {
        hvhx_v2_rms_norm_mul_f32(src + (size_t)r * row_size, weight,
                                 dst + (size_t)r * row_size, row_size, eps);
    }
}

void hvhx_v2_rms_norm_f32_rows(const float * __restrict__ src,
                               float       * __restrict__ dst,
                               uint32_t n_rows, uint32_t row_size, float eps)
{
    for (uint32_t r = 0; r < n_rows; ++r) {
        hvhx_v2_rms_norm_f32(src + (size_t)r * row_size,
                             dst + (size_t)r * row_size, row_size, eps);
    }
}
