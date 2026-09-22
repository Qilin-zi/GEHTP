/*
 * hvx_binary.c — V2 逐元素二元运算封装
 * Module: v2-binary
 * 源:    ggmlHTPV3E htp/hvx-base.h (hvx_vec_{add,sub,mul}_f32_f32)
 *        + htp/hvx-inverse.h (hvx_vec_inverse_f32, for div)
 */
#include "hvxhmx_v2_binary.h"
#include "hvxhmx_types.h"

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-utils.h"
#endif

#define VLEN_F32 32u

#if HVX_V2_KERNELS_ENABLED
#define V2_BIN_LOOP(dst, a, b, n, VEC_OP, SCALAR_EXPR)                       \
    do {                                                                      \
        const uint32_t _nvec = (n) / VLEN_F32;                                \
        const uint32_t _tail = (n) - _nvec * VLEN_F32;                        \
        HVX_Vector       * __restrict__ _dv = (HVX_Vector *)(dst);            \
        const HVX_Vector * __restrict__ _av = (const HVX_Vector *)(a);        \
        const HVX_Vector * __restrict__ _bv = (const HVX_Vector *)(b);        \
        for (uint32_t _i = 0; _i < _nvec; ++_i)                               \
            _dv[_i] = VEC_OP(_av[_i], _bv[_i]);                               \
        for (uint32_t _i = 0; _i < _tail; ++_i) {                             \
            uint32_t _j = _nvec * VLEN_F32 + _i;                              \
            float x = ((const float *)(a))[_j];                               \
            float y = ((const float *)(b))[_j];                               \
            ((float *)(dst))[_j] = (SCALAR_EXPR);                             \
        }                                                                     \
    } while (0)
#endif

void hvhx_v2_add_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_BIN_LOOP(dst, a, b, n, hvx_vec_add_f32_f32, (x + y));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] + b[i];
#endif
}

void hvhx_v2_sub_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_BIN_LOOP(dst, a, b, n, hvx_vec_sub_f32_f32, (x - y));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] - b[i];
#endif
}

void hvhx_v2_mul_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_BIN_LOOP(dst, a, b, n, hvx_vec_mul_f32_f32, (x * y));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] * b[i];
#endif
}

void hvhx_v2_div_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
    /* div = a * inverse(b); inverse 是两步, 不套 V2_BIN_LOOP. */
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ av = (const HVX_Vector *)a;
    const HVX_Vector * __restrict__ bv = (const HVX_Vector *)b;
    for (uint32_t i = 0; i < nvec; ++i)
        dv[i] = hvx_vec_mul_f32_f32(av[i], hvx_vec_inverse_f32(bv[i]));
    for (uint32_t i = 0; i < tail; ++i) {
        uint32_t j = nvec * VLEN_F32 + i;
        dst[j] = a[j] / b[j];
    }
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] / b[i];
#endif
}

void hvhx_v2_add_scalar_f32(float * __restrict__ dst, const float * __restrict__ a,
                             float s, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    HVX_Vector vs = hvx_vec_splat_f32(s);
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ av = (const HVX_Vector *)a;
    for (uint32_t i = 0; i < nvec; ++i) dv[i] = hvx_vec_add_f32_f32(av[i], vs);
    for (uint32_t i = 0; i < tail; ++i) dst[nvec * VLEN_F32 + i] = a[nvec * VLEN_F32 + i] + s;
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] + s;
#endif
}

void hvhx_v2_mul_scalar_f32(float * __restrict__ dst, const float * __restrict__ a,
                             float s, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    HVX_Vector vs = hvx_vec_splat_f32(s);
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ av = (const HVX_Vector *)a;
    for (uint32_t i = 0; i < nvec; ++i) dv[i] = hvx_vec_mul_f32_f32(av[i], vs);
    for (uint32_t i = 0; i < tail; ++i) dst[nvec * VLEN_F32 + i] = a[nvec * VLEN_F32 + i] * s;
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] * s;
#endif
}

/* ---- f16 面直收直发 (oplist 执行器契约) ----
 * 数学: dst[i] = f32_to_f16(f16_to_f32(a[i]) + f16_to_f32(b[i])) —— 与
 * oplist_exec 标量逐字节同(f32 中间精度, RNE 舍入)。
 * 向量主体: hvx_vec_f16_to_f32 (vmpy 位级保真) → hvx_vec_add_f32_f32 →
 *           hvx_vec_f32_to_f16 (RNE)。
 * 对齐策略: 三指针任意地址安全。
 *   - dst 未对齐前导 + 末尾余量走标量(与标量版逐字节同);
 *   - 主体要求 dst 128B 对齐(写安全), src 用 Q6_V_vloadu 容忍未对齐读。
 * VLEN_F16 = 64 (128B / 2B)。 */
#define VLEN_F16 64u

/* 自包含 f16↔f32 标量转换(与 oplist_exec 同算法, IEEE RNE) —— 前导/尾部
 * 用, 避免 hvx 库反向依赖 runtime 的 static 函数。 */
static float hb_f16_to_f32(uint16_t h) {
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
static uint16_t hb_f32_to_f16(float f) {
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

void hvhx_v2_add_f16(uint16_t * __restrict__ dst,
                     const uint16_t * __restrict__ a, const uint16_t * __restrict__ b,
                     uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    /* 标量参考(与 oplist_exec 同源, 避免循环依赖) */
    uint32_t i = 0;

    /* 1) dst 前导至 128B 对齐 */
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 2u);
    if (lead > n) lead = n;
    for (; i < lead; ++i)
        dst[i] = hb_f32_to_f16(hb_f16_to_f32(a[i]) + hb_f16_to_f32(b[i]));

    /* 2) 向量主体(dst 已 128B 对齐) */
    uint32_t body = (n - i) / VLEN_F16;
    for (uint32_t v = 0; v < body; ++v) {
        uint32_t base = i + v * VLEN_F16;
        /* f16 未对齐读(hvx_vmemu 容忍; a/b 对齐不定) */
        HVX_Vector va = hvx_vmemu(a + base);
        HVX_Vector vb = hvx_vmemu(b + base);
        /* f16 → f32 (每 HVX_Vector f16 = 64 元素 → 2× f32 向量, lo=前 32, hi=后 32) */
        HVX_VectorPair fa = hvx_vec_f16_to_f32(va);
        HVX_VectorPair fb = hvx_vec_f16_to_f32(vb);
        /* f32 加(qf32 原语, V81 无 hvx_vec_add_f32_f32 定义) */
        HVX_Vector sum_lo = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(fa), Q6_V_lo_W(fb)));
        HVX_Vector sum_hi = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_VsfVsf(Q6_V_hi_W(fa), Q6_V_hi_W(fb)));
        /* f32 → f16 (RNE); f32_to_f16(v0,v1) 把 v0 放低 32 元素 → lo 在前保 C 序 */
        HVX_Vector vr = hvx_vec_f32_to_f16(sum_lo, sum_hi);
        *(HVX_Vector *)(dst + base) = vr;
    }
    i += body * VLEN_F16;

    /* 3) 末尾余量标量 */
    for (; i < n; ++i)
        dst[i] = hb_f32_to_f16(hb_f16_to_f32(a[i]) + hb_f16_to_f32(b[i]));
#else
    for (uint32_t i = 0; i < n; ++i)
        dst[i] = hb_f32_to_f16(hb_f16_to_f32(a[i]) + hb_f16_to_f32(b[i]));
#endif
}

void hvhx_v2_cvt_f16_to_f32(float * __restrict__ dst,
                            const uint16_t * __restrict__ src,
                            uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    uint32_t i = 0;
    /* dst 是 f32 (4B/元素): 前导至 128B 对齐 */
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 4u);
    if (lead > n) lead = n;
    for (; i < lead; ++i) dst[i] = hb_f16_to_f32(src[i]);
    /* 主体: f16 (64 元素) -> 2x f32 (lo=前 32, hi=后 32), 保 C 序 */
    uint32_t body = (n - i) / VLEN_F16;
    for (uint32_t v = 0; v < body; ++v) {
        uint32_t base = i + v * VLEN_F16;
        HVX_Vector va = hvx_vmemu(src + base);
        HVX_VectorPair fa = hvx_vec_f16_to_f32(va);
        *(HVX_Vector *)(dst + base)            = Q6_V_lo_W(fa);
        *(HVX_Vector *)(dst + base + VLEN_F32) = Q6_V_hi_W(fa);
    }
    i += body * VLEN_F16;
    for (; i < n; ++i) dst[i] = hb_f16_to_f32(src[i]);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = hb_f16_to_f32(src[i]);
#endif
}

void hvhx_v2_cvt_f32_to_f16(uint16_t * __restrict__ dst,
                            const float * __restrict__ src,
                            uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    uint32_t i = 0;
    /* dst 是 f16 (2B/元素): 前导至 128B 对齐 */
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 2u);
    if (lead > n) lead = n;
    for (; i < lead; ++i) dst[i] = hb_f32_to_f16(src[i]);
    /* 主体: 2x f32 -> f16 (64 元素) */
    uint32_t body = (n - i) / VLEN_F16;
    for (uint32_t v = 0; v < body; ++v) {
        uint32_t base = i + v * VLEN_F16;
        HVX_Vector lo = hvx_vmemu(src + base);
        HVX_Vector hi = hvx_vmemu(src + base + VLEN_F32);
        HVX_Vector vr = hvx_vec_f32_to_f16(lo, hi);
        *(HVX_Vector *)(dst + base) = vr;
    }
    i += body * VLEN_F16;
    for (; i < n; ++i) dst[i] = hb_f32_to_f16(src[i]);
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = hb_f32_to_f16(src[i]);
#endif
}

