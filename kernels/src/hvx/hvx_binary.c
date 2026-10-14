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
            while (!(m & 0x400)) { m <<= 1; e++; }
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
        /* subnormal RNE (与 gdn_ref.c gdn_f32_to_f16 逐字同款): 旧版 man >> (shift+1)
         * 把 f16 subnormal 输出砍半, 与 oplist_exec.c f32_to_f16 同根因一起修。 */
        uint32_t sub = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        sub += (rem > (1u << (shift - 1u)) || (rem == (1u << (shift - 1u)) && (sub & 1u))) ? 1u : 0u;
        return (uint16_t)(sign | sub);
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

/* ---- broadcast 纯搬运原语 (007 US1) ----
 * 位级 = 标量逐元素拷贝 (无数值语义), 形态 1/2 执行体的唯一底层依赖。 */
void hvhx_v2_fill_f16(uint16_t * __restrict__ dst, uint16_t s, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    uint32_t i = 0;
    /* dst 前导至 128B 对齐 */
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 2u);
    if (lead > n) lead = n;
    for (; i < lead; ++i) dst[i] = s;
    uint32_t body = (n - i) / VLEN_F16;
    if (body) {
        HVX_Vector vs = Q6_Vh_vsplat_R((int32_t)s);
        HVX_Vector * __restrict__ dv = (HVX_Vector *)(dst + i);
        for (uint32_t v = 0; v < body; ++v) dv[v] = vs;
        i += body * VLEN_F16;
    }
    for (; i < n; ++i) dst[i] = s;
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = s;
#endif
}

void hvhx_v2_copy_f16(uint16_t * __restrict__ dst,
                      const uint16_t * __restrict__ src,
                      uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    uint32_t i = 0;
    /* dst 前导至 128B 对齐; src 侧 hvx_vmemu 容忍未对齐 */
    uintptr_t daddr = (uintptr_t)dst;
    uint32_t lead = (uint32_t)(((128u - (daddr & 127u)) & 127u) / 2u);
    if (lead > n) lead = n;
    for (; i < lead; ++i) dst[i] = src[i];
    uint32_t body = (n - i) / VLEN_F16;
    for (uint32_t v = 0; v < body; ++v) {
        uint32_t base = i + v * VLEN_F16;
        HVX_Vector vs = hvx_vmemu(src + base);
        *(HVX_Vector *)(dst + base) = vs;
    }
    i += body * VLEN_F16;
    for (; i < n; ++i) dst[i] = src[i];
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = src[i];
#endif
}

/* ---- Q4_0 反量化 + crouton pack (013; 契约见 hvxhmx_v2_binary.h) ----
 * ref = oplist_exec T017 LUT 分支的逐位同款: d*(q-8) 在 f32 精确
 * (≤15 位有效数 < 24), 唯一舍入点 = f32→f16 RNE; u32 对写
 * (hi<<16)|lo 即 crouton pair-interleave 面 pos(k,n)=(k>>1)*64+2n+(k&1)。 */
static void hb_dequant_q4_0_crouton_ref(uint16_t * __restrict__ dst,
                                        const uint8_t * __restrict__ src,
                                        uint32_t n_tiles)
{
    for (uint32_t t = 0; t < n_tiles; ++t) {
        const uint8_t * qt = src + (size_t)t * 640u;
        const uint16_t * dv = (const uint16_t *)(const void *)(qt + 512u);
        uint32_t * tile32 = (uint32_t *)(void *)(dst + (size_t)t * 1024u);
        for (uint32_t cp = 0; cp < 16u; ++cp) {
            const uint8_t * qb = qt + cp * 32u;
            for (uint32_t n = 0; n < 32u; ++n) {
                float d = hb_f16_to_f32(dv[n]);
                uint32_t b = qb[n];
                tile32[cp * 32u + n] =
                    ((uint32_t)hb_f32_to_f16(d * (float)((int)(b >> 4) - 8)) << 16)
                    | hb_f32_to_f16(d * (float)((int)(b & 0xFu) - 8));
            }
        }
    }
}

void hvhx_v2_dequant_q4_0_crouton(uint16_t * __restrict__ dst,
                                  const uint8_t * __restrict__ src,
                                  uint32_t n_tiles)
{
#if HVX_V2_KERNELS_ENABLED
    /* vmem 对齐存储要求; 未对齐 dst 回落 ref (调用点 wgt_cr 由 HMX carve
     * 128B 对齐, 此为防御) */
    if (((uintptr_t)dst & 127u) != 0u) {
        hb_dequant_q4_0_crouton_ref(dst, src, n_tiles);
        return;
    }
    const HVX_Vector nib_mask = Q6_Vb_vsplat_R(0x0F);
    /* 整数→f32 无舍入: 位拼 2^23+q (vor, q≤15 与尾数位不相交) 再减
     * (2^23+8) = f32(q-8) 精确 */
    const HVX_Vector magic = Q6_V_vsplat_R(0x4B000000u);
    const HVX_Vector base  = hvx_vec_splat_f32(8388616.0f);
    for (uint32_t t = 0; t < n_tiles; ++t) {
        const uint8_t * qt = src + (size_t)t * 640u;
        /* 缩放: v_sc = [d0..d31 | 垃圾], f16→f32 (C 序) 的 lo 半即
         * f32 lane n = d_n, 一次乘法覆盖 n=0..31 (垃圾全落 hi 弃用) */
        HVX_Vector v_sc = hvx_vmemu(qt + 512);
        HVX_Vector scale = Q6_V_lo_W(hvx_vec_f16_to_f32(v_sc));
        for (uint32_t g = 0; g < 4u; ++g) {
            HVX_Vector vq = hvx_vmemu(qt + g * 128);
            /* 低 nibble = K 偶行, 高 nibble = K 奇行 (板探针 S1 证实);
             * vshuff(Vh,Vl,-1) 字节交织: mix.lo = 前半 [e,o,e,o..],
             * mix.hi = 后半 → 交织序即 K 偶/奇逐列对 */
            HVX_Vector v_e = Q6_V_vand_VV(vq, nib_mask);
            HVX_Vector v_o = Q6_Vub_vlsr_VubR(vq, 4);
            HVX_VectorPair mix = Q6_W_vshuff_VVR(v_o, v_e, -1);
            for (uint32_t h = 0; h < 2u; ++h) {
                HVX_Vector half = h ? Q6_V_hi_W(mix) : Q6_V_lo_W(mix);
                /* vzxt = vunpack 奇偶拆分 (板探针 S2 证实, 非位置序):
                 * w16.lo = K 偶行 u16, w16.hi = K 奇行 u16;
                 * 每 64 lane = 2 个 cp 各 32 列 (n 序) */
                HVX_VectorPair w16 = Q6_Wuh_vzxt_Vub(half);
                /* u16→u32 又按 n 奇偶拆 → vshuff -4 (u32 粒度, 板探针实测) 交织复原
                 * lo=cpA n 序 / hi=cpB n 序 */
                HVX_VectorPair w32e = Q6_Wuw_vzxt_Vuh(Q6_V_lo_W(w16));
                HVX_VectorPair w32o = Q6_Wuw_vzxt_Vuh(Q6_V_hi_W(w16));
                HVX_VectorPair rec_e =
                    Q6_W_vshuff_VVR(Q6_V_hi_W(w32e), Q6_V_lo_W(w32e), -4);
                HVX_VectorPair rec_o =
                    Q6_W_vshuff_VVR(Q6_V_hi_W(w32o), Q6_V_lo_W(w32o), -4);
                for (uint32_t c = 0; c < 2u; ++c) {
                    HVX_Vector qe = hvx_vec_sub_f32_f32(Q6_V_vor_VV(
                        c ? Q6_V_hi_W(rec_e) : Q6_V_lo_W(rec_e), magic), base);
                    HVX_Vector qo = hvx_vec_sub_f32_f32(Q6_V_vor_VV(
                        c ? Q6_V_hi_W(rec_o) : Q6_V_lo_W(rec_o), magic), base);
                    HVX_Vector pe = hvx_vec_mul_f32_f32(qe, scale);
                    HVX_Vector po = hvx_vec_mul_f32_f32(qo, scale);
                    /* f32→f16 RNE 交织版 (_shuff): 偶 lane=pe(K 偶行), 奇=po —— 板探针 SC 实证即 crouton 序 */
                    hvx_vmem(dst + ((size_t)t * 16u + g * 4u + h * 2u + c) * 64u) =
                        hvx_vec_f32_to_f16_shuff(pe, po);
                }
            }
        }
    }
#else
    hb_dequant_q4_0_crouton_ref(dst, src, n_tiles);
#endif
}

