/*
 * hvx_flash_attn.c — V2 Flash Attention 内核封装 (HVX + HMX)
 * Module: v2-flash-attn
 * 源:    ggmlHTPV3E htp/hvx-fa-kernels.h (HVX dot/mad)
 *        + htp/hvx-flash-attn.h  (ALiBi slopes)
 *        + htp/hmx-fa-kernels.h  (HMX tile qk/o_update/o_norm)
 *
 * 本文件只暴露 tile 级数学内核; 外层 online-softmax 驱动由调用方实现.
 */
#include "hvxhmx_v2_flash_attn.h"
#include "hvxhmx_types.h"

#include <math.h>

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-fa-kernels.h"
#include "internal/hvx-flash-attn.h"
#include "internal/hmx-fa-kernels.h"
#endif

/* ============================================================
 *  host 标量 fallback
 * ============================================================ */
#if !HVX_V2_KERNELS_ENABLED
static void scalar_qk_dot_f16(const void *q, const void *k,
                              uint32_t n, float scale, float *out)
{
    const __fp16 *qx = (const __fp16 *)q;
    const __fp16 *kx = (const __fp16 *)k;
    double acc = 0.0;
    for (uint32_t i = 0; i < n; ++i) acc += (double)qx[i] * (double)kx[i];
    *out = (float)(acc * scale);
}

static void scalar_attn_v_mad_f16(float *y, const void *x,
                                  const __fp16 *s, uint32_t n)
{
    const __fp16 *xx = (const __fp16 *)x;
    float sv = (float)*s;
    for (uint32_t i = 0; i < n; ++i) y[i] += (float)xx[i] * sv;
}

static float scalar_alibi_slope(uint32_t h, uint32_t n_head_log2, float m0, float m1)
{
    return (h < n_head_log2) ? powf(m0, (float)(h + 1))
                             : powf(m1, (float)(2 * (h - n_head_log2) + 1));
}

static void scalar_alibi_slopes(uint32_t kv_head, uint32_t G,
                                uint32_t n_head_log2, float m0, float m1,
                                float *out)
{
    for (uint32_t i = 0; i < 32; ++i)
        out[i] = scalar_alibi_slope(kv_head * G + i, n_head_log2, m0, m1);
}
#endif /* !HVX_V2_KERNELS_ENABLED */

/* ============================================================
 *  HVX FA 内核
 * ============================================================ */
void hvhx_v2_fa_qk_dot_f16(const void * __restrict__ q,
                            const void * __restrict__ k,
                            uint32_t head_dim, float scale,
                            float * __restrict__ out)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_dot_f16_f16_aa(out, q, k, head_dim, scale);
#else
    scalar_qk_dot_f16(q, k, head_dim, scale, out);
#endif
}

void hvhx_v2_fa_attn_v_mad_f16(float * __restrict__ y,
                                const void * __restrict__ x,
                                const __fp16 * __restrict__ s,
                                uint32_t head_dim)
{
#if HVX_V2_KERNELS_ENABLED
    hvx_mad_f32_f16_aa(y, x, s, head_dim);
#else
    scalar_attn_v_mad_f16(y, x, s, head_dim);
#endif
}

void hvhx_v2_alibi_slopes(uint32_t kv_head, uint32_t G,
                          uint32_t n_head_log2, float m0, float m1,
                          float * __restrict__ out)
{
#if HVX_V2_KERNELS_ENABLED
    /* hvx_alibi_slopes 返回一个 HVX_Vector (32 个 f32). 直接 store 进 out. */
    *(HVX_Vector *)out = hvx_alibi_slopes(kv_head, G, n_head_log2, m0, m1);
#else
    scalar_alibi_slopes(kv_head, G, n_head_log2, m0, m1, out);
#endif
}

/* ============================================================
 *  HMX FA tile 内核
 *  注: 不在此处调 hmx_enable_execution — 外层统一管理 HMX 锁, 避免
 *  在 KV chunk 循环里重复使能/禁止的开销.
 * ============================================================ */
void hvhx_v2_hmx_fa_qk_dot_tile(const __fp16 * __restrict__ row_tiles,
                                 const __fp16 * __restrict__ col_tiles,
                                 __fp16 * __restrict__ out_tile,
                                 uint32_t n_dot_tiles)
{
#if HVX_V2_KERNELS_ENABLED
    hmx_fa_qk_dot_tile(row_tiles, col_tiles, out_tile, (size_t)n_dot_tiles);
#else
    (void)row_tiles; (void)col_tiles; (void)out_tile; (void)n_dot_tiles;
#endif
}

void hvhx_v2_hmx_fa_o_update_tile(const __fp16 * __restrict__ d_diag,
                                   const __fp16 * __restrict__ o_rc,
                                   const __fp16 * __restrict__ p_tile_in,
                                   const __fp16 * __restrict__ v_tile_in,
                                   __fp16 * __restrict__ o_tile_out,
                                   uint32_t n_col_tiles)
{
#if HVX_V2_KERNELS_ENABLED
    hmx_fa_o_update_tile(d_diag, o_rc, p_tile_in, v_tile_in,
                         o_tile_out, (size_t)n_col_tiles);
#else
    (void)d_diag; (void)o_rc; (void)p_tile_in; (void)v_tile_in;
    (void)o_tile_out; (void)n_col_tiles;
#endif
}

void hvhx_v2_hmx_fa_o_norm_tile(const __fp16 * __restrict__ d_diag,
                                 const __fp16 * __restrict__ o_rc,
                                 __fp16 * __restrict__ o_out)
{
#if HVX_V2_KERNELS_ENABLED
    hmx_fa_o_norm_tile(d_diag, o_rc, o_out);
#else
    (void)d_diag; (void)o_rc; (void)o_out;
#endif
}
