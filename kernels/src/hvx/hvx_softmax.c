/*
 * hvx_softmax.c — V2 Softmax 封装
 * Module: v2-softmax
 * 源:    ggmlHTPV3E htp/softmax-ops.c
 *        (hvx_fast_softmax_f32, hvx_fast_softmax_prep_f32)
 */
#include "hvxhmx_v2_softmax.h"
#include "hvxhmx_types.h"

#include <math.h>
#include <float.h>

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-utils.h"
#endif

/* ============================================================
 *  Fused online softmax (源 softmax-ops.c hvx_fast_softmax_f32)
 * ============================================================ */
#if HVX_V2_KERNELS_ENABLED
static void hvx_fast_softmax_f32(const uint8_t * __restrict__ src,
                                 uint8_t * __restrict__ dst,
                                 uint8_t * __restrict__ pad, int num_elems)
{
    const HVX_Vector * __restrict__ v_src = (const HVX_Vector *)src;
    HVX_Vector       * __restrict__ v_pad = (HVX_Vector *)pad;
    HVX_Vector       * __restrict__ v_dst = (HVX_Vector *)dst;

    HVX_Vector sum_vec = Q6_V_vsplat_R(0x00000000);
    HVX_Vector max_vec = hvx_vec_splat_f32(((const float *)src)[0]);
    HVX_Vector zero_v  = Q6_V_vzero();
    HVX_Vector one_v   = hvx_vec_splat_f32(1.0f);

    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        max_vec = Q6_Vsf_vmax_VsfVsf(max_vec, v1);
    }
    max_vec = hvx_vec_reduce_max_f32(max_vec);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vsub_VsfVsf(v1, max_vec);
        HVX_Vector v3 = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(v2));
        sum_vec = Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(sum_vec), v3);
        v_pad[i] = v3;
    }
    sum_vec = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_vec));

    HVX_VectorPred pos_sum = Q6_Q_vcmp_gt_VwVw(sum_vec, zero_v);
    HVX_Vector     v4      = hvx_vec_inverse_f32(sum_vec);
    HVX_Vector     scale_v = Q6_V_vmux_QVV(pos_sum, v4, one_v);

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = v_pad[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }
}
#endif

void hvhx_v2_softmax_f32(const float * __restrict__ src,
                          float * __restrict__ dst,
                          float * __restrict__ pad,
                          uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    if ((n % 32u) == 0 && pad != NULL &&
        ((uintptr_t)src % 128u == 0) && ((uintptr_t)dst % 128u == 0) &&
        ((uintptr_t)pad % 128u == 0)) {
        hvx_fast_softmax_f32((const uint8_t *)src, (uint8_t *)dst,
                             (uint8_t *)pad, (int)n);
        return;
    }
#endif
    /* 标量 fallback (任意 n, 无对齐要求) */
    float mx = -FLT_MAX;
    for (uint32_t i = 0; i < n; ++i) if (src[i] > mx) mx = src[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        float e = expf(src[i] - mx);
        if (pad) ((float *)pad)[i] = e;
        dst[i] = e;
        sum += e;
    }
    float inv = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (uint32_t i = 0; i < n; ++i) dst[i] *= inv;
}

/* ============================================================
 *  FA mask 预处理 (源 softmax-ops.c hvx_fast_softmax_prep_f32)
 *  dst[i] = src[i]*scale + mask[i]*slope
 * ============================================================ */
#if HVX_V2_KERNELS_ENABLED
static void hvx_fast_softmax_prep_f32(const uint8_t * __restrict__ src,
                                      uint8_t * __restrict__ dst,
                                      int num_elems, float scale,
                                      const uint8_t * __restrict__ mask, float slope)
{
    const uint8_t * __restrict__ src_curr  = src;
    uint8_t       * __restrict__ dst_curr  = dst;
    const uint8_t * __restrict__ mask_curr = mask;

    HVX_Vector scale_vec = hvx_vec_splat_f32(scale);
    HVX_Vector slope_vec = hvx_vec_splat_f32(slope);
    int step_of_1 = num_elems >> 5;

    #pragma unroll(4)
    for (int i = 0; i < step_of_1; i++) {
        HVX_Vector v1 = *(const HVX_Vector *)src_curr;
        HVX_Vector v3 = *(const HVX_Vector *)mask_curr;
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_vec);
        HVX_Vector v4 = Q6_Vqf32_vmpy_VsfVsf(v3, slope_vec);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(v2, v4);
        *(HVX_Vector *)dst_curr = Q6_Vsf_equals_Vqf32(v5);
        src_curr  += 128;
        dst_curr  += 128;
        mask_curr += 128;
    }
}
#endif

void hvhx_v2_softmax_mask_f32(const float * __restrict__ src,
                               float * __restrict__ dst,
                               uint32_t n, float scale,
                               const float * __restrict__ mask, float slope)
{
#if HVX_V2_KERNELS_ENABLED
    if ((n % 32u) == 0 && mask != NULL &&
        ((uintptr_t)src % 128u == 0) && ((uintptr_t)dst % 128u == 0) &&
        ((uintptr_t)mask % 128u == 0)) {
        hvx_fast_softmax_prep_f32((const uint8_t *)src, (uint8_t *)dst,
                                  (int)n, scale, (const uint8_t *)mask, slope);
        return;
    }
#endif
    for (uint32_t i = 0; i < n; ++i)
        dst[i] = src[i] * scale + (mask ? mask[i] * slope : 0.0f);
}
