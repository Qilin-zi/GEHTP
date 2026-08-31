/*
 * hvx_rope.c — V2 RoPE (Rotary Position Embedding) 内核封装
 * Module: v2-rope
 * 源:    ggmlHTPV3E htp/rope-ops.c 的核心数学内核
 *        (rope_cache_init, hvx_rope_neox_f32_aa, hvx_rope_f32_aa, YaRN helpers)
 */
#include "hvxhmx_v2_rope.h"
#include "hvxhmx_types.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-utils.h"   /* 拉入 vec cos/sin/inverse + 全 HVX 原语 */
#endif

/* ============================================================
 *  YaRN 标量 helpers (host + DSP 共用, 纯标量)
 * ============================================================ */
static inline float rope_yarn_ramp(float low, float high, int i0)
{
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static inline void rope_yarn_one(float theta, float freq_scale, const float *corr_dims,
                                 uint32_t i0, float ext_factor, float mscale,
                                 float *cache)
{
    float theta_extrap = theta;
    float theta_interp = freq_scale * theta_extrap;
    float theta_final  = theta_interp;
    float mscale_final = mscale;

    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i0) * ext_factor;
        theta_final    = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    cache[i0 + 0] = cosf(theta_final) * mscale_final;
    cache[i0 + 1] = sinf(theta_final) * mscale_final;
}

void hvhx_v2_rope_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                            float beta_fast, float beta_slow, float dims[2])
{
    float start = floorf((float)n_dims * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / (2.0f * logf(freq_base)));
    float end   = ceilf((float)n_dims * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / (2.0f * logf(freq_base)));
    dims[0]     = fmaxf(0.0f, start);
    dims[1]     = fminf((float)(n_dims - 1), end);
}

/* ============================================================
 *  theta_cache 构建
 * ============================================================ */
#if HVX_V2_KERNELS_ENABLED
/* DSP 向量化版 — 源自 rope-ops.c rope_cache_init (V79+ fast path).
 * 依赖 hvx_vec_cos/sin/inverse + hvx_vmem/vmemu + Q6 intrinsics. */
static void rope_cache_init_vec(float theta_base, float freq_scale,
                                const float *freq_factors, const float *corr_dims,
                                uint32_t ne0, float ext_factor, float mscale,
                                float *cache, float theta_scale)
{
    const bool fast_path = (__HVX_ARCH__ >= 79) && (ext_factor == 0.0f);

    if (fast_path) {
        const uint32_t n_blocks = ne0 / 64;

        float __attribute__((aligned(128))) theta_powers[32];
        theta_powers[0] = 1.0f;
        for (int j = 1; j < 32; j++)
            theta_powers[j] = theta_powers[j - 1] * theta_scale;
        HVX_Vector v_theta_powers = hvx_vmem(theta_powers);

        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);

        float theta_block    = theta_base;
        float theta_scale_32 = 1.0f;
        for (int j = 0; j < 32; j++) theta_scale_32 *= theta_scale;

        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t i0 = b * 64;
            HVX_Vector v_theta_base = hvx_vec_splat_f32(theta_block);
            HVX_Vector v_theta      = hvx_vec_mul_f32_f32(v_theta_base, v_theta_powers);

            if (freq_factors) {
                HVX_Vector v_ff     = hvx_vmemu(freq_factors + i0 / 2);
                HVX_Vector v_inv_ff = hvx_vec_inverse_f32(v_ff);
                v_theta = hvx_vec_mul_f32_f32(v_theta, v_inv_ff);
            }

            HVX_Vector v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
            HVX_Vector vcos = hvx_vec_cos_f32(v_theta_final);
            HVX_Vector vsin = hvx_vec_sin_f32(v_theta_final);
            vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
            vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(vsin, vcos, -4);
            if (((uintptr_t)cache) % 128 == 0) {
                hvx_vmem(cache + i0 + 0)  = Q6_V_lo_W(vstore);
                hvx_vmem(cache + i0 + 32) = Q6_V_hi_W(vstore);
            } else {
                hvx_vec_store_u(cache + i0 + 0,  32 * sizeof(float), Q6_V_lo_W(vstore));
                hvx_vec_store_u(cache + i0 + 32, 32 * sizeof(float), Q6_V_hi_W(vstore));
            }
            theta_block *= theta_scale_32;
        }

        float theta = theta_block;
        for (uint32_t i0 = n_blocks * 64; i0 < ne0; i0 += 2) {
            float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    } else {
        float theta = theta_base;
        for (uint32_t i0 = 0; i0 < ne0; i0 += 2) {
            float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    }
}
#endif /* HVX_V2_KERNELS_ENABLED */

void hvhx_v2_rope_cache_init(float theta_base, float freq_scale,
                             const float *freq_factors, const float corr_dims[2],
                             uint32_t ne0, float ext_factor, float mscale,
                             float *cache, float theta_scale)
{
#if HVX_V2_KERNELS_ENABLED
    rope_cache_init_vec(theta_base, freq_scale, freq_factors, corr_dims,
                        ne0, ext_factor, mscale, cache, theta_scale);
#else
    (void)corr_dims; /* ext_factor==0 时 host 路径不用 corr_dims */
    float theta = theta_base;
    for (uint32_t i0 = 0; i0 < ne0; i0 += 2) {
        float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
        theta *= theta_scale;
    }
#endif
}

/* ============================================================
 *  单行旋转 — DSP HVX 版 (源 rope-ops.c hvx_rope_neox_f32_aa / hvx_rope_f32_aa)
 * ============================================================ */
#if HVX_V2_KERNELS_ENABLED
static void hvx_rope_neox_f32_aa(float * restrict dst, const float * restrict src,
                                 uint32_t ne, const float * restrict theta_cache)
{
    const uint32_t he   = ne / 2;
    const uint32_t nvec = he / 32;
    const uint32_t nloe = he % 32;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((const HVX_Vector *) src)[i];
        HVX_Vector v1 = hvx_vmemu(src + he + i * 32);
        HVX_Vector v2 = ((const HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector v3 = ((const HVX_Vector *) theta_cache)[i * 2 + 1];
        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);
        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_hi_W(vcos_sin));
        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);
        ((HVX_Vector *) dst)[i] = Q6_Vsf_equals_Vqf32(v4);
        hvx_vmemu(dst + he + i * 32) = Q6_Vsf_equals_Vqf32(v5);
    }
    if (nloe > 0) {
        HVX_Vector v0 = hvx_vmemu(src + nvec * 32);
        HVX_Vector v1 = hvx_vmemu(src + he + nvec * 32);
        HVX_Vector v2 = ((const HVX_Vector *) theta_cache)[nvec * 2 + 0];
        HVX_Vector v3 = ((const HVX_Vector *) theta_cache)[nvec * 2 + 1];
        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);
        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, Q6_V_hi_W(vcos_sin));
        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);
        hvx_vec_store_u(dst + nvec * 32, nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v4));
        hvx_vec_store_u(dst + he + nvec * 32, nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v5));
    }
}

static void hvx_rope_normal_f32_aa(float * restrict dst, const float * restrict src,
                                   uint32_t ne, const float * restrict theta_cache)
{
    const uint32_t nvec = ne / 64;
    const uint32_t nloe = ne % 64;

    for (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((const HVX_Vector *) src)[i * 2 + 0];
        HVX_Vector v1 = ((const HVX_Vector *) src)[i * 2 + 1];
        HVX_Vector v2 = ((const HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector v3 = ((const HVX_Vector *) theta_cache)[i * 2 + 1];
        HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(v1, v0, -4);
        HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);
        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));
        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);
        HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);
        ((HVX_Vector *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_Vector *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }
    if (nloe > 0) {
        if (nloe <= 32) {
            HVX_Vector v0 = hvx_vmemu(src + nvec * 64);
            HVX_Vector v2 = hvx_vmemu(theta_cache + nvec * 64);
            HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);
            HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(Q6_V_vzero(), v2, -4);
            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));
            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);
            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);
            hvx_vec_store_u(dst + nvec * 64, nloe * sizeof(float), Q6_V_lo_W(vstore));
        } else {
            HVX_Vector v0 = hvx_vmemu(src + nvec * 64);
            HVX_Vector v1 = hvx_vmemu(src + nvec * 64 + 32);
            HVX_Vector v2 = hvx_vmemu(theta_cache + nvec * 64);
            HVX_Vector v3 = hvx_vmemu(theta_cache + nvec * 64 + 32);
            HVX_VectorPair vx0_x1   = Q6_W_vdeal_VVR(v1, v0, -4);
            HVX_VectorPair vcos_sin = Q6_W_vdeal_VVR(v3, v2, -4);
            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), Q6_V_hi_W(vcos_sin));
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_lo_W(vcos_sin));
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), Q6_V_hi_W(vcos_sin));
            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);
            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);
            ((HVX_Vector *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * sizeof(float), Q6_V_hi_W(vstore));
        }
    }
}
#endif /* HVX_V2_KERNELS_ENABLED */

/* ============================================================
 *  host 标量 fallback
 * ============================================================ */
#if !HVX_V2_KERNELS_ENABLED
static void scalar_rope_neox_f32(float *dst, const float *src, uint32_t n_dims,
                                 const float *theta_cache)
{
    uint32_t he = n_dims / 2;
    for (uint32_t i = 0; i < he; i++) {
        float c = theta_cache[2 * i + 0];
        float s = theta_cache[2 * i + 1];
        float x0 = src[i];
        float x1 = src[i + he];
        dst[i]      = x0 * c - x1 * s;
        dst[i + he] = x0 * s + x1 * c;
    }
}

static void scalar_rope_normal_f32(float *dst, const float *src, uint32_t n_dims,
                                   const float *theta_cache)
{
    for (uint32_t i = 0; i < n_dims; i += 2) {
        float c = theta_cache[i + 0];
        float s = theta_cache[i + 1];
        float x0 = src[i + 0];
        float x1 = src[i + 1];
        dst[i + 0] = x0 * c - x1 * s;
        dst[i + 1] = x0 * s + x1 * c;
    }
}
#endif /* !HVX_V2_KERNELS_ENABLED */

/* ============================================================
 *  公共 API
 * ============================================================ */
void hvhx_v2_rope_neox_f32(float * __restrict__ dst,
                            const float * __restrict__ src,
                            uint32_t n_dims,
                            const float * __restrict__ theta_cache)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_rope_neox_f32_aa(dst, src, n_dims, theta_cache);
#else
    scalar_rope_neox_f32(dst, src, n_dims, theta_cache);
#endif
}

void hvhx_v2_rope_normal_f32(float * __restrict__ dst,
                              const float * __restrict__ src,
                              uint32_t n_dims,
                              const float * __restrict__ theta_cache)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_rope_normal_f32_aa(dst, src, n_dims, theta_cache);
#else
    scalar_rope_normal_f32(dst, src, n_dims, theta_cache);
#endif
}

void hvhx_v2_rope_row_f32(float * __restrict__ dst,
                           const float * __restrict__ src,
                           uint32_t n_dims, uint32_t ne0,
                           const float * __restrict__ theta_cache,
                           hvhx_v2_rope_mode_t mode)
{
    if (mode == HVHX_V2_ROPE_NEOX)
        hvhx_v2_rope_neox_f32(dst, src, n_dims, theta_cache);
    else
        hvhx_v2_rope_normal_f32(dst, src, n_dims, theta_cache);

    if (n_dims < ne0)
        memcpy(dst + n_dims, src + n_dims, (ne0 - n_dims) * sizeof(float));
}
