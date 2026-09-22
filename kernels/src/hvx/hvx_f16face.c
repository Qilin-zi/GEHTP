/*
 * hvx_f16face.c — GEHTP oplist 执行器 f16 面 HVX 接线族 (W1 应用尽用)
 * Module: gehtp-f16face
 * 契约: 与 oplist_exec 标量体同数学 (f32 中间精度, RNE 回 f16), 直收直发 f16。
 *   - mul_f16: 纯原生 f16 向量 (镜像 hvhx_v2_add_f16 结构)
 *   - unary/silu/softmax/rmsnorm: f16→f32 转换 → v2 f32 内核 → f16 回写
 *   - 全部 scratch 由调用者给 (库内零 malloc, 运行时零 malloc 铁律)
 * 精度注记: softmax/rmsnorm 标量体在中间步骤有 f16 落盘舍入, 本族全程 f32
 *   更接近 host 金标; A/B 门判据 = 对 golden 值差不劣化 (非与标量逐位同)。
 */
#include "hvxhmx_v2_f16face.h"
#include "hvxhmx_v2_binary.h"
#include "hvxhmx_v2_unary.h"
#include "hvxhmx_v2_norm.h"
#include "hvxhmx_v2_softmax.h"
#include "hvxhmx_types.h"

#include <math.h>
#include <string.h>

#if defined(__HVX__) || defined(__hexagon__)
#define F16FACE_HVX 1
#include "internal/hvx-utils.h"
#endif

#define VLEN_F16 64u   /* 128B / 2B */
#define VLEN_F32 32u   /* 128B / 4B */

/* ---- 自包含标量转换 (与 hvx_binary.c 同算法, IEEE RNE; 前导/尾/降级用) ---- */
static float ff_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            uint32_t m = man; int e = -1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            bits = sign | ((uint32_t)(127 - 15 + 1 - e - 1) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}
static uint16_t ff_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> (shift + 1);
        uint32_t rest = man & ((1u << shift) - 1u);
        if ((man >> shift) & 1u) half += (rest || (half & 1u)) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man >> 12) & 1u) half += ((man & 0xfffu) || (half & 1u)) ? 1u : 0u;
    return (uint16_t)half;
}

/* ---- f16 → f32 向量转换 (dst f32 前导对齐, src 未对齐读) ---- */
void gehtp_hvx_cvt_f16_to_f32(float * __restrict__ dst,
                              const uint16_t * __restrict__ src, uint32_t n) {
#if F16FACE_HVX
    uint32_t i = 0;
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 4u);
    if (lead > n) lead = n;
    for (; i < lead; ++i) dst[i] = ff_f16_to_f32(src[i]);
    uint32_t body = (n - i) / VLEN_F16;
    for (uint32_t v = 0; v < body; ++v) {
        uint32_t base = i + v * VLEN_F16;
        HVX_VectorPair fp = hvx_vec_f16_to_f32(hvx_vmemu(src + base));
        float* d = dst + base;
        *(HVX_Vector *)d       = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(fp));
        *(HVX_Vector *)(d + 32) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(fp));
    }
    i += body * VLEN_F16;
    for (; i < n; ++i) dst[i] = ff_f16_to_f32(src[i]);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = ff_f16_to_f32(src[i]);
#endif
}

/* ---- f32 → f16 向量转换 (RNE) ---- */
void gehtp_hvx_cvt_f32_to_f16(uint16_t * __restrict__ dst,
                              const float * __restrict__ src, uint32_t n) {
#if F16FACE_HVX
    uint32_t i = 0;
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 2u);
    if (lead > n) lead = n;
    for (; i < lead; ++i) dst[i] = ff_f32_to_f16(src[i]);
    uint32_t body = (n - i) / VLEN_F16;
    for (uint32_t v = 0; v < body; ++v) {
        uint32_t base = i + v * VLEN_F16;
        const float* s = src + base;
        HVX_Vector v0 = hvx_vmemu(s);        /* 前 32 f32 */
        HVX_Vector v1 = hvx_vmemu(s + 32);   /* 后 32 f32 */
        *(HVX_Vector *)(dst + base) = hvx_vec_f32_to_f16(v0, v1);
    }
    i += body * VLEN_F16;
    for (; i < n; ++i) dst[i] = ff_f32_to_f16(src[i]);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = ff_f32_to_f16(src[i]);
#endif
}

/* ---- 原生 f16 乘 (镜像 hvhx_v2_add_f16) ---- */
void gehtp_hvx_mul_f16(uint16_t * __restrict__ dst,
                       const uint16_t * __restrict__ a,
                       const uint16_t * __restrict__ b, uint32_t n) {
#if F16FACE_HVX
    uint32_t i = 0;
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 2u);
    if (lead > n) lead = n;
    for (; i < lead; ++i)
        dst[i] = ff_f32_to_f16(ff_f16_to_f32(a[i]) * ff_f16_to_f32(b[i]));
    uint32_t body = (n - i) / VLEN_F16;
    for (uint32_t v = 0; v < body; ++v) {
        uint32_t base = i + v * VLEN_F16;
        HVX_VectorPair fa = hvx_vec_f16_to_f32(hvx_vmemu(a + base));
        HVX_VectorPair fb = hvx_vec_f16_to_f32(hvx_vmemu(b + base));
        HVX_Vector p_lo = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(fa), Q6_V_lo_W(fb)));
        HVX_Vector p_hi = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(fa), Q6_V_hi_W(fb)));
        *(HVX_Vector *)(dst + base) = hvx_vec_f32_to_f16(p_lo, p_hi);
    }
    i += body * VLEN_F16;
    for (; i < n; ++i)
        dst[i] = ff_f32_to_f16(ff_f16_to_f32(a[i]) * ff_f16_to_f32(b[i]));
#else
    for (uint32_t i = 0; i < n; ++i)
        dst[i] = ff_f32_to_f16(ff_f16_to_f32(a[i]) * ff_f16_to_f32(b[i]));
#endif
}

/* ---- 一元族 f16 面: subtype 与 oplist_exec exec_unary 同编号 ----
 * 返回 0 成功; -1 = 该 subtype 无 HVX 核 (调用者回落标量)。
 * scratch: si/so 各 n 个 f32, 128B 对齐。 */
int gehtp_hvx_unary_f16(uint16_t * __restrict__ dst, const uint16_t * __restrict__ src,
                        uint32_t n, uint32_t subtype,
                        float * __restrict__ si, float * __restrict__ so) {
    if (!si || !so) return -1;
    gehtp_hvx_cvt_f16_to_f32(si, src, n);
    switch (subtype) {
    case 0:  hvhx_v2_mul_scalar_f32(so, si, -1.0f, n); break;
    case 1:  hvhx_v2_exp_f32(so, si, n); break;
    case 2:  hvhx_v2_sqrt_f32(so, si, n); break;
    case 3:  hvhx_v2_rsqrt_f32(so, si, n); break;
    case 4:  hvhx_v2_log_f32(so, si, n); break;
    case 8:  hvhx_v2_sigmoid_f32(so, si, n); break;
    case 9:  hvhx_v2_tanh_f32(so, si, n); break;
    case 10: hvhx_v2_gelu_tanh_f32(so, si, n); break;
    case 12: hvhx_v2_silu_f32(so, si, n); break;
    default: return -1;   /* 5 abs / 6 sin / 7 cos / 11 relu 等回落标量 */
    }
    gehtp_hvx_cvt_f32_to_f16(dst, so, n);
    return 0;
}

/* ---- SiLU f16 面 (exec_silu 用) ---- */
void gehtp_hvx_silu_f16(uint16_t * __restrict__ dst, const uint16_t * __restrict__ src,
                        uint32_t n, float * __restrict__ si, float * __restrict__ so) {
    gehtp_hvx_cvt_f16_to_f32(si, src, n);
    hvhx_v2_silu_f32(so, si, n);
    gehtp_hvx_cvt_f32_to_f16(dst, so, n);
}

/* ---- Softmax f16 面 (rows × n; n 须 32 倍数, 否则调用者回落) ----
 * pad 用 si 尾部 (softmax_f32 的 pad 参数: 行内 scratch)。 */
void gehtp_hvx_softmax_f16(uint16_t * __restrict__ dst, const uint16_t * __restrict__ src,
                           uint32_t rows, uint32_t n,
                           float * __restrict__ si, float * __restrict__ so,
                           float * __restrict__ pad) {
    for (uint32_t r = 0; r < rows; r++) {
        gehtp_hvx_cvt_f16_to_f32(si, src + (size_t)r * n, n);
        hvhx_v2_softmax_f32(si, so, pad, n);
        gehtp_hvx_cvt_f32_to_f16(dst + (size_t)r * n, so, n);
    }
}

/* ---- RMSNorm×gamma f16 面 (exec_rmsnorm2 用; bias 可选) ----
 * 标量体: y = x / sqrt(mean(x²)+eps) * w + b; 本族: x*rsqrt(mean(x²)+eps)*w,
 * 再 (可选) +b。bias 存在时用 mul_scalar 语义外的逐元素加 — 走 v2 add_f32。 */
void gehtp_hvx_rmsnorm_mul_f16(uint16_t * __restrict__ dst,
                               const uint16_t * __restrict__ src,
                               const uint16_t * __restrict__ w,
                               const uint16_t * __restrict__ bias,
                               uint32_t m, uint32_t rw, float eps,
                               float * __restrict__ si, float * __restrict__ sw,
                               float * __restrict__ so) {
    gehtp_hvx_cvt_f16_to_f32(si, src, m * rw);
    gehtp_hvx_cvt_f16_to_f32(sw, w, rw);
    hvhx_v2_rms_norm_mul_f32_rows(si, sw, so, m, rw, eps);
    if (bias) {
        /* 逐行加 bias: so[r*rw+i] += bias_f32[i] —— 转一次 bias 复用 */
        gehtp_hvx_cvt_f16_to_f32(si, bias, rw);       /* si 复用为 bias 缓冲 */
        for (uint32_t r = 0; r < m; r++)
            hvhx_v2_add_f32(so + (size_t)r * rw, so + (size_t)r * rw, si, rw);
    }
    gehtp_hvx_cvt_f32_to_f16(dst, so, m * rw);
}
