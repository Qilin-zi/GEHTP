/*
 * hvx_ssm.c — V2 状态空间模型内核封装 (solve_tri + gated delta net)
 * Module: v2-ssm
 * 源:    ggmlHTPV3E htp/solve-tri-ops.c + gated-delta-net-ops.c
 *        (内核体原样提取到 internal/ssm-kernels.h)
 */
#include "hvxhmx_v2_ssm.h"
#include "hvxhmx_types.h"

#include <string.h>

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-utils.h"
#include "internal/ssm-kernels.h"
#endif

/* ============================================================
 *  下三角前代解 (逐行)
 * ============================================================ */
void hvhx_v2_solve_tri_row_f32(const float * __restrict__ A_row,
                                const float * __restrict__ B_row,
                                float * __restrict__ X,
                                uint32_t row, uint32_t k,
                                uint32_t col0, uint32_t coln,
                                float inv_diag)
{
#if HVX_V2_KERNELS_ENABLED
    if (coln <= 32)
        solve_tri_row_hvx(A_row, B_row, X, row, k, col0, coln, inv_diag);
    else
        solve_tri_row_scalar(A_row, B_row, X, row, k, col0, coln, inv_diag);
#else
    for (uint32_t col = col0; col < col0 + coln; ++col) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < row; ++t)
            sum += A_row[t] * X[t * k + col];
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
#endif
}

/* ============================================================
 *  Gated Delta Net — 4 状态
 * ============================================================ */
void hvhx_v2_gdn_mul_dot4_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                               float * __restrict__ dst2, float * __restrict__ dst3,
                               const float * __restrict__ mul,
                               const float * __restrict__ dot,
                               uint32_t n, float * __restrict__ sums)
{
#if HVX_V2_KERNELS_ENABLED
    gdn_mul_dot4_f32(dst0, dst1, dst2, dst3, mul, dot, n, sums);
#else
    sums[0] = sums[1] = sums[2] = sums[3] = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        dst0[i] *= mul[i];  dst1[i] *= mul[i];
        dst2[i] *= mul[i];  dst3[i] *= mul[i];
        sums[0] += dst0[i] * dot[i];
        sums[1] += dst1[i] * dot[i];
        sums[2] += dst2[i] * dot[i];
        sums[3] += dst3[i] * dot[i];
    }
#endif
}

void hvhx_v2_gdn_add_scaled_dot4_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                                      float * __restrict__ dst2, float * __restrict__ dst3,
                                      const float * __restrict__ src,
                                      const float * __restrict__ scale,
                                      const float * __restrict__ dot,
                                      uint32_t n, float * __restrict__ sums)
{
#if HVX_V2_KERNELS_ENABLED
    gdn_add_scaled_dot4_f32(dst0, dst1, dst2, dst3, src, scale, dot, n, sums);
#else
    sums[0] = sums[1] = sums[2] = sums[3] = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        dst0[i] += src[i] * scale[0];
        dst1[i] += src[i] * scale[1];
        dst2[i] += src[i] * scale[2];
        dst3[i] += src[i] * scale[3];
        sums[0] += dst0[i] * dot[i];
        sums[1] += dst1[i] * dot[i];
        sums[2] += dst2[i] * dot[i];
        sums[3] += dst3[i] * dot[i];
    }
#endif
}

/* ============================================================
 *  Gated Delta Net — 8 状态
 * ============================================================ */
void hvhx_v2_gdn_mul_dot8_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                               float * __restrict__ dst2, float * __restrict__ dst3,
                               float * __restrict__ dst4, float * __restrict__ dst5,
                               float * __restrict__ dst6, float * __restrict__ dst7,
                               const float * __restrict__ mul,
                               const float * __restrict__ dot,
                               uint32_t n, float * __restrict__ sums)
{
#if HVX_V2_KERNELS_ENABLED
    gdn_mul_dot8_f32(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7,
                     mul, dot, n, sums);
#else
    for (int k = 0; k < 8; ++k) sums[k] = 0.0f;
    float *d[8] = { dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7 };
    for (uint32_t i = 0; i < n; ++i) {
        for (int s = 0; s < 8; ++s) {
            d[s][i] *= mul[i];
            sums[s] += d[s][i] * dot[i];
        }
    }
#endif
}

void hvhx_v2_gdn_add_scaled_dot8_f32(float * __restrict__ dst0, float * __restrict__ dst1,
                                      float * __restrict__ dst2, float * __restrict__ dst3,
                                      float * __restrict__ dst4, float * __restrict__ dst5,
                                      float * __restrict__ dst6, float * __restrict__ dst7,
                                      const float * __restrict__ src,
                                      const float * __restrict__ scale,
                                      const float * __restrict__ dot,
                                      uint32_t n, float * __restrict__ sums)
{
#if HVX_V2_KERNELS_ENABLED
    gdn_add_scaled_dot8_f32(dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7,
                            src, scale, dot, n, sums);
#else
    for (int k = 0; k < 8; ++k) sums[k] = 0.0f;
    float *d[8] = { dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7 };
    for (uint32_t i = 0; i < n; ++i) {
        for (int s = 0; s < 8; ++s) {
            d[s][i] += src[i] * scale[s];
            sums[s] += d[s][i] * dot[i];
        }
    }
#endif
}

/* ============================================================
 *  SSM depthwise 1D causal conv (ssm-conv.c)
 * ============================================================ */
void hvhx_v2_ssm_conv_f32(const float * __restrict__ src0,
                          const float * __restrict__ src1,
                          float       * __restrict__ dst,
                          uint32_t d_inner, uint32_t n_t, uint32_t n_s,
                          uint32_t ncs, uint32_t d_conv,
                          int apply_silu)
{
    for (uint32_t i3 = 0; i3 < n_s; ++i3) {
        for (uint32_t i2 = 0; i2 < n_t; ++i2) {
            for (uint32_t i1 = 0; i1 < d_inner; ++i1) {
                float sumf = 0.0f;
                for (uint32_t j = 0; j < d_conv; ++j) {
                    sumf += src0[(i2 + j) + i1 * ncs + i3 * ncs * d_inner]
                          * src1[j + i1 * d_conv];
                }
                if (apply_silu) {
                    sumf = sumf / (1.0f + expf(-sumf));
                }
                dst[i1 + i2 * d_inner + i3 * d_inner * n_t] = sumf;
            }
        }
    }
}

void hvhx_v2_ssm_conv_dot32_f32(float       * __restrict__ dst32,
                                const float * __restrict__ x_row, uint32_t x_stride,
                                const float * __restrict__ w_row, uint32_t w_stride,
                                uint32_t d_conv, int apply_silu)
{
#if HVX_V2_KERNELS_ENABLED
    HVX_Vector res;
    ssm_conv_dot32_hvx(&res, x_row, x_stride, w_row, w_stride, d_conv, apply_silu);
    *((HVX_UVector *)dst32) = res;
#else
    for (uint32_t ch = 0; ch < 32; ++ch) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < d_conv; ++j) {
            acc += x_row[j * x_stride + ch] * w_row[j * w_stride + ch];
        }
        if (apply_silu) {
            acc = acc / (1.0f + expf(-acc));
        }
        dst32[ch] = acc;
    }
#endif
}

void hvhx_v2_transpose_32x32_f32(float * __restrict__ m32x32)
{
#if HVX_V2_KERNELS_ENABLED
    HVX_Vector *m = (HVX_Vector *)m32x32;
    hvx_transpose_32x32_f32(m);
#else
    float tmp[32 * 32];
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c)
            tmp[c * 32 + r] = m32x32[r * 32 + c];
    memcpy(m32x32, tmp, sizeof(tmp));
#endif
}
