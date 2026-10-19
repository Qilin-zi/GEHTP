/* gdn_kern.c — B1/B2/B3 f16 kernel (内部 f32, 状态 f32 外置)
 * chunk 公式推导见 gdn_sm.h 头注; host numpy 对拍 1e-14 (proto_chunk.py) */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef __hexagon__
#include <HAP_perf.h>
#endif
#ifndef __hexagon__
#include <malloc.h>   /* memalign 原型 (host x86 缺则按 int 截断指针 → SIGSEGV) */
#endif
#include "gdn_sm.h"

#if defined(__HVX__)
#define HVX_GDN_ENABLED 1
#include "internal/hvx-base.h"
#include "internal/hvx-reduce.h"
#include "internal/hvx-exp.h"
#include "internal/hvx-inverse.h"
#endif

/* ================= HVX f32 原语 (009 T022d) =================
 * GEMM (S·k_j / S·q_t 批量算) 与外积是 chunk delta 的三处 O(c·D·D) 热点。
 * 全 f32 精度 (qf32 累加; dot 走 qf32+sf 树归约, 镜像 hvx-norm.h sum-of-squares, v81 验证)。
 * 前置: S/kf/qf/vf/u 均 128B 对齐 (memalign), D 为 32 整数倍。 */
#if HVX_GDN_ENABLED
/* S[m][n] += scale·u[m]·k[n] 外积 rank-1 更新 (逐元素, 与标量位级同) */
static void gdn_hvx_outer(float* __restrict__ S, const float* __restrict__ u,
                          const float* __restrict__ k, float scale, int D, int nvec) {
    const HVX_Vector* ka = (const HVX_Vector*)k;
    for (int m = 0; m < D; m++) {
        float wu = scale * u[m];
        if (wu == 0.0f) continue;
        HVX_Vector cu = hvx_vec_splat_f32(wu);
        HVX_Vector* Sm = (HVX_Vector*)(S + (size_t)m * D);
        for (int b = 0; b < nvec; b++)
            Sm[b] = Q6_Vsf_vadd_VsfVsf(Sm[b], Q6_Vsf_vmpy_VsfVsf(cu, ka[b]));
    }
}
/* a·b 点积 (D 元素, D 为 32 整数倍), 全 f32 qf32 累加 + sf 树归约 (镜像 matvec) */
static float gdn_hvx_dot(const float* __restrict__ a, const float* __restrict__ b,
                         int D, int nvec) {
    const HVX_Vector* aa = (const HVX_Vector*)a;
    const HVX_Vector* ba = (const HVX_Vector*)b;
    HVX_Vector acc = Q6_V_vsplat_R(0);
    for (int bb = 0; bb < nvec; bb++)
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(aa[bb], ba[bb]));
    return hvx_vec_get_f32(hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(acc)));
}
/* y += scale·x (D 元素); scale==0 早退; 逐元素与标量位级同 */
static void gdn_hvx_axpy(float* __restrict__ y, const float* __restrict__ x,
                         float scale, int D, int nvec) {
    if (scale == 0.0f) return;
    HVX_Vector cs = hvx_vec_splat_f32(scale);
    HVX_Vector* ya = (HVX_Vector*)y;
    const HVX_Vector* xa = (const HVX_Vector*)x;
    for (int bb = 0; bb < nvec; bb++)
        ya[bb] = Q6_Vsf_vadd_VsfVsf(ya[bb], Q6_Vsf_vmpy_VsfVsf(cs, xa[bb]));
}
/* u[j] = Σ_{i≤j} T[j][i]·b[i]  (三角回代, 寄存器分块累加)
 * 与逐次 gdn_hvx_axpy 位级同: 累加序 i=0..j 逐 lane 左折 (Vsf vadd+Vsf vmpy),
 * 仅把累加器从内存搬到寄存器, 免每次 axpy 的 u[j] 读改写内存往返。
 * 前置: nvec ≤ GDN_VEC_MAX (D=128 → nvec=4)。 */
#define GDN_VEC_MAX 8
static void gdn_hvx_tri_accum(float* __restrict__ u, const float* __restrict__ b,
                              const float* __restrict__ T, int j, int c, int D, int nvec) {
    HVX_Vector acc[GDN_VEC_MAX];
    for (int bb = 0; bb < nvec; bb++) acc[bb] = Q6_V_vsplat_R(0);
    for (int i = 0; i <= j; i++) {
        float tij = T[(size_t)j * c + i];
        if (tij == 0.0f) continue;
        HVX_Vector cs = hvx_vec_splat_f32(tij);
        const HVX_Vector* bi = (const HVX_Vector*)(b + (size_t)i * D);
        for (int bb = 0; bb < nvec; bb++)
            acc[bb] = Q6_Vsf_vadd_VsfVsf(acc[bb], Q6_Vsf_vmpy_VsfVsf(cs, bi[bb]));
    }
    HVX_Vector* uj = (HVX_Vector*)(u + (size_t)j * D);
    for (int bb = 0; bb < nvec; bb++) uj[bb] = acc[bb];
}
/* y *= scale (n 元素, n 为 32 整数倍); 单次乘与标量位级同 */
static void gdn_hvx_scale(float* __restrict__ y, int n, float scale) {
    HVX_Vector cs = hvx_vec_splat_f32(scale);
    HVX_Vector* ya = (HVX_Vector*)y;
    for (int i = 0; i < n / 32; i++)
        ya[i] = Q6_Vsf_vmpy_VsfVsf(ya[i], cs);
}
/* B1 conv+SiLU 向量化 (2026-10-15 板上阶段计时: conv 22.2ms/层 = 61%)。
 * FIR 逐乘加 RNE 与标量位级同 (每 lane 即一通道, i 升序同式);
 * SiLU = s·(1+exp(-s))^{-1}: exp 走 hvx_vec_exp_f32 (softmax 线已上板),
 * 倒数 NR (hvx_vec_inverse_f32_guard) — 与 expf/IEEE div 有 ulp 级差,
 * yc 非位级同; s 大负→d 大→inv→0 ≈ silu≈0, s 大正→d≈1 ≈ s, 端点同式。
 * 前置: DI 32 倍数, yr/win/cw/xf 行 128B 对齐 (GDN_ALLOC align 128)。 */
static void gdn_hvx_conv_silu(float* yr, const float* cw, const float* win,
                              const float* xf, int DI, int dc, int nv) {
    const HVX_Vector nan_mask = Q6_V_vsplat_R(0x7f800000);
    const HVX_Vector one = hvx_vec_splat_f32(1.0f);
    for (int v = 0; v < nv; v++) {
        const float* xfv = xf + (size_t)v * 32;
        HVX_Vector acc = Q6_V_vsplat_R(0);
        for (int i = 0; i < dc - 1; i++) {
            const float* wv = cw + (size_t)i * DI + (size_t)v * 32;
            const float* xv = win + (size_t)i * DI + (size_t)v * 32;
            acc = Q6_Vsf_vadd_VsfVsf(acc,
                  Q6_Vsf_vmpy_VsfVsf(*(const HVX_Vector*)wv, *(const HVX_Vector*)xv));
        }
        const float* cwv = cw + (size_t)(dc - 1) * DI + (size_t)v * 32;
        acc = Q6_Vsf_vadd_VsfVsf(acc,
              Q6_Vsf_vmpy_VsfVsf(*(const HVX_Vector*)cwv, *(const HVX_Vector*)xfv));
        /* silu: acc * 1/(1+exp(-acc)); neg = vasr(acc, 1) 非精确负, 用 vmux(-)? 无 vneg sf → 0-acc */
        HVX_Vector e = hvx_vec_exp_f32(Q6_Vsf_vsub_VsfVsf(Q6_V_vsplat_R(0), acc));
        HVX_Vector d = Q6_Vsf_vadd_VsfVsf(one, e);
        HVX_Vector inv = hvx_vec_inverse_f32_guard(d, nan_mask);
        *(HVX_Vector*)(yr + (size_t)v * 32) = Q6_Vsf_vmpy_VsfVsf(acc, inv);
    }
}
/* f16→f32 整段向量化 (cw 预转 / 每 token xqkv 转): lo=前 32 hi=后 32 (binary.c 同式) */
static void gdn_hvx_cvt_f32(float* dst, const int16_t* src, int n, int nv16) {
    for (int v = 0; v < nv16; v++) {
        HVX_Vector h = hvx_vmemu(src + (size_t)v * 64);
        HVX_VectorPair p = hvx_vec_f16_to_f32(h);
        *(HVX_Vector*)(dst + (size_t)v * 64)        = Q6_V_lo_W(p);
        *(HVX_Vector*)(dst + (size_t)v * 64 + 32)   = Q6_V_hi_W(p);
    }
    (void)n;
}
/* C[m][j] = Σ_n S[m][n]·KT[n][j]  (D×c GEMM, c=32 恰一向量宽)
 * S [D][D] 行主序, KT [D][c] 行主序, C [D][c] 行主序; 无树归约 (每 lane 即一 j)。
 * 累加序 n=0..D-1 纯 qf32 (比 matvec 的 qf32+SF 树归约更精确, 数值差异 <1ulp)。 */
static void gdn_hvx_gemm(const float* __restrict__ S, const float* __restrict__ KT,
                         float* __restrict__ C, int D, int c) {
    for (int m = 0; m < D; m++) {
        const float* Sm = S + (size_t)m * D;
        HVX_Vector acc = Q6_V_vsplat_R(0);
        for (int n = 0; n < D; n++) {
            HVX_Vector sn = hvx_vec_splat_f32(Sm[n]);
            HVX_Vector ktrow = *(const HVX_Vector*)(KT + (size_t)n * c);
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(sn, ktrow));
        }
        *(HVX_Vector*)(C + (size_t)m * c) = Q6_Vsf_equals_Vqf32(acc);
    }
}
/* G[j][i] = Σ_n kf_T[n][j]·kf_T[n][i]  (K·K^T, 批量替代 496 个独立 dot)
 * kf_T [D][c] 行主序 (c=32 恰一向量宽); 累加序 n=0..D-1 纯 qf32
 * (与 gdn_hvx_gemm 同式, <1ulp vs 标量 n 序)。前置: c 为 32 整数倍。 */
static void gdn_hvx_gramm(const float* __restrict__ kf_T, float* __restrict__ G,
                          int D, int c) {
    for (int j = 0; j < c; j++) {
        HVX_Vector acc = Q6_V_vsplat_R(0);
        for (int n = 0; n < D; n++) {
            HVX_Vector cs = hvx_vec_splat_f32(kf_T[(size_t)n * c + j]);
            HVX_Vector row = *(const HVX_Vector*)(kf_T + (size_t)n * c);
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(cs, row));
        }
        *(HVX_Vector*)(G + (size_t)j * c) = Q6_Vsf_equals_Vqf32(acc);
    }
}
/* C[j][i] = Σ_n A_T[n][j]·B_T[n][i]  (A·B^T, 双输入 Gram, 批量替代 c²/2 独立 dot)
 * A_T/B_T [D][c] 行主序 (c=32 恰一向量宽); 累加序 n=0..D-1 纯 qf32。
 * 前置: c 为 32 整数倍。 */
static void gdn_hvx_gramm2(const float* __restrict__ A_T, const float* __restrict__ B_T,
                           float* __restrict__ C, int D, int c) {
    for (int j = 0; j < c; j++) {
        HVX_Vector acc = Q6_V_vsplat_R(0);
        for (int n = 0; n < D; n++) {
            HVX_Vector cs = hvx_vec_splat_f32(A_T[(size_t)n * c + j]);
            HVX_Vector row = *(const HVX_Vector*)(B_T + (size_t)n * c);
            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(cs, row));
        }
        *(HVX_Vector*)(C + (size_t)j * c) = Q6_Vsf_equals_Vqf32(acc);
    }
}
#else
/* host 标量回落 (x86 无 __HVX__): 与既有标量体逐字同式 */
static void gdn_hvx_outer(float* __restrict__ S, const float* __restrict__ u,
                          const float* __restrict__ k, float scale, int D, int nvec) {
    (void)nvec;
    for (int m = 0; m < D; m++) {
        float wu = scale * u[m];
        if (wu == 0.0f) continue;
        float* row = S + (size_t)m * D;
        for (int n = 0; n < D; n++) row[n] += wu * k[n];
    }
}
static float gdn_hvx_dot(const float* __restrict__ a, const float* __restrict__ b,
                         int D, int nvec) {
    (void)nvec;
    float s = 0.0f;
    for (int n = 0; n < D; n++) s += a[n] * b[n];
    return s;
}
static void gdn_hvx_axpy(float* __restrict__ y, const float* __restrict__ x,
                         float scale, int D, int nvec) {
    (void)nvec;
    if (scale == 0.0f) return;
    for (int n = 0; n < D; n++) y[n] += scale * x[n];
}
static void gdn_hvx_tri_accum(float* __restrict__ u, const float* __restrict__ b,
                              const float* __restrict__ T, int j, int c, int D, int nvec) {
    (void)nvec;
    for (int m = 0; m < D; m++) {
        float s = 0.0f;
        for (int i = 0; i <= j; i++) {
            float tij = T[(size_t)j * c + i];
            if (tij == 0.0f) continue;
            s += tij * b[(size_t)i * D + m];
        }
        u[(size_t)j * D + m] = s;
    }
}
static void gdn_hvx_scale(float* __restrict__ y, int n, float scale) {
    for (int i = 0; i < n; i++) y[i] *= scale;
}
static void gdn_hvx_gemm(const float* __restrict__ S, const float* __restrict__ KT,
                         float* __restrict__ C, int D, int c) {
    for (int m = 0; m < D; m++)
        for (int j = 0; j < c; j++) {
            float s = 0.0f;
            for (int n = 0; n < D; n++) s += S[(size_t)m * D + n] * KT[(size_t)n * c + j];
            C[(size_t)m * c + j] = s;
        }
}
static void gdn_hvx_gramm(const float* __restrict__ kf_T, float* __restrict__ G,
                          int D, int c) {
    for (int j = 0; j < c; j++)
        for (int i = 0; i < c; i++) {
            float s = 0.0f;
            for (int n = 0; n < D; n++) s += kf_T[(size_t)n * c + j] * kf_T[(size_t)n * c + i];
            G[(size_t)j * c + i] = s;
        }
}
static void gdn_hvx_gramm2(const float* __restrict__ A_T, const float* __restrict__ B_T,
                           float* __restrict__ C, int D, int c) {
    for (int j = 0; j < c; j++)
        for (int i = 0; i < c; i++) {
            float s = 0.0f;
            for (int n = 0; n < D; n++) s += A_T[(size_t)n * c + j] * B_T[(size_t)n * c + i];
            C[(size_t)j * c + i] = s;
        }
}
#endif

#define D_INNER 9216
#define D_CONV  4
#define NHEAD   32
#define HD      128

/* ================= B1: conv1d + SiLU ================= */
/* SiLU 向量化: y = s/(1+e^{-s})。exp = degree-7 Taylor (~1e-6), 倒数 = Newton (~1e-7),
 * 与标量 expf/div 差异 ~1e-6 (与 prefill gdn_block HVX 同量级, 12/12 已验)。
 * 前置: di 为 32 整数倍, yf 128B 对齐。s 已由标量点积求出并暂存 yf。 */
#if HVX_GDN_ENABLED
static void conv_silu_f32(float* __restrict__ yf, int di) {
    const HVX_Vector one = hvx_vec_splat_f32(1.0f);
    HVX_Vector* y = (HVX_Vector*)yf;
    for (int b = 0, nvec = di / 32; b < nvec; b++) {
        HVX_Vector s = y[b];
        HVX_Vector e = hvx_vec_exp_f32(hvx_vec_neg_f32(s));
        HVX_Vector r = hvx_vec_inverse_f32(Q6_Vsf_vadd_VsfVsf(one, e));
        y[b] = Q6_Vsf_vmpy_VsfVsf(s, r);
    }
}
#endif

#if HVX_GDN_ENABLED
/* f16(int16_t raw half) -> f32 向量转换 (正常数范围位级同 gdn_f16_to_f32)。
 * n 为 64 整数倍, dst 128B 对齐, src 未对齐经 vmemu。 */
static void gdn_cvt_f16_f32_vec(float* __restrict__ dst, const int16_t* __restrict__ src, int n) {
    HVX_Vector* dv = (HVX_Vector*)dst;
    for (int i = 0, nv = n / 64; i < nv; i++) {
        HVX_Vector va = hvx_vmemu((const void*)(src + i * 64));
        HVX_VectorPair fa = hvx_vec_f16_to_f32(va);
        dv[2 * i]     = Q6_V_lo_W(fa);
        dv[2 * i + 1] = Q6_V_hi_W(fa);
    }
}
/* f32 -> f16(int16_t raw half) 向量转换 (RNE, 正常数范围位级同 gdn_f32_to_f16)。
 * src 128B 对齐, dst 未对齐经前导/余量标量。 */
static void gdn_cvt_f32_f16_vec(int16_t* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = 0;
    int lead = (int)(((128u - ((uintptr_t)dst & 127u)) & 127u) / 2u);
    if (lead > n) lead = n;
    for (; i < lead; i++) dst[i] = gdn_f32_to_f16(src[i]);
    int body = (n - i) / 64;
    for (int v = 0; v < body; v++) {
        int base = i + v * 64;
        HVX_Vector lo = *(const HVX_Vector*)(src + base);
        HVX_Vector hi = *(const HVX_Vector*)(src + base + 32);
        *(HVX_Vector*)(dst + base) = hvx_vec_f32_to_f16(lo, hi);
    }
    i += body * 64;
    for (; i < n; i++) dst[i] = gdn_f32_to_f16(src[i]);
}
#endif

static void conv_f16(conv_state_t* st, const int16_t* w_f16, const float* xf, float* yf) {
    int dc = st->d_conv, di = st->d_inner;
    const float* win = st->win;
#if HVX_GDN_ENABLED
    /* w f16 -> f32 一次向量转 (dc*di 个); 点积仍标量 (位级同原式, 免 qf32 累加噪声) */
    float* wf = memalign(128, (size_t)dc * di * 4);
    for (int i = 0; i < dc; i++)
        gdn_cvt_f16_f32_vec(wf + (size_t)i * di, w_f16 + (size_t)i * di, di);
    for (int ch = 0; ch < di; ch++) {
        float s = 0.0f;
        for (int i = 0; i < dc - 1; i++)
            s += wf[(size_t)i * di + ch] * win[(size_t)i * di + ch];
        s += wf[(size_t)(dc - 1) * di + ch] * xf[ch];
        yf[ch] = s;
    }
    free(wf);
    conv_silu_f32(yf, di);
#else
    for (int ch = 0; ch < di; ch++) {
        float s = 0.0f;
        for (int i = 0; i < dc - 1; i++)
            s += gdn_f16_to_f32(w_f16[(size_t)i * di + ch]) * win[(size_t)i * di + ch];
        s += gdn_f16_to_f32(w_f16[(size_t)(dc - 1) * di + ch]) * xf[ch];
        yf[ch] = s / (1.0f + expf(-s));
    }
#endif
    memmove(st->win, st->win + di, (size_t)(dc - 2) * di * sizeof(float));
    memcpy(st->win + (size_t)(dc - 2) * di, xf, (size_t)di * sizeof(float));
}

int conv1d_step_f16(conv_state_t* st, const int16_t* w_f16, const int16_t* x, int16_t* y) {
    int di = st->d_inner;
    float *xf = memalign(128, (size_t)di * 4), *yf = memalign(128, (size_t)di * 4);
    if (!xf || !yf) { free(xf); free(yf); return 1; }
#if HVX_GDN_ENABLED
    gdn_cvt_f16_f32_vec(xf, x, di);
#else
    for (int i = 0; i < di; i++) xf[i] = gdn_f16_to_f32(x[i]);
#endif
    conv_f16(st, w_f16, xf, yf);
#if HVX_GDN_ENABLED
    gdn_cvt_f32_f16_vec(y, yf, di);
#else
    for (int i = 0; i < di; i++) y[i] = gdn_f32_to_f16(yf[i]);
#endif
    free(xf); free(yf);
    return 0;
}

int conv1d_block_f16(conv_state_t* st, const int16_t* w_f16, const int16_t* x, int16_t* y, int m) {
    int di = st->d_inner;
    float *xf = memalign(128, (size_t)di * 4), *yf = memalign(128, (size_t)di * 4);
    if (!xf || !yf) { free(xf); free(yf); return 1; }
    for (int t = 0; t < m; t++) {
        for (int i = 0; i < di; i++) xf[i] = gdn_f16_to_f32(x[(size_t)t * di + i]);
        conv_f16(st, w_f16, xf, yf);
        for (int i = 0; i < di; i++) y[(size_t)t * di + i] = gdn_f32_to_f16(yf[i]);
    }
    free(xf); free(yf);
    return 0;
}

/* ================= B3: T = (I+L)^{-1} (unitriangular, f16 in/out) ================= */
int solve_tri_f16(const int16_t* L, int16_t* T, int c) {
    /* T[r][j] = δ(r,j) − Σ_{i<r} L[r][i]·T[i][j]  (f32 累加) */
    for (int r = 0; r < c; r++) {
        for (int j = 0; j <= r; j++) {              /* 上半严格三角恒 0, 只算下半+对角 */
            float s = (r == j) ? 1.0f : 0.0f;
            const int16_t* Lr = L + (size_t)r * c;
            for (int i = j; i < r; i++)             /* i≥j (T[i][j]=0 for i<j) */
                s -= gdn_f16_to_f32(Lr[i]) * gdn_f16_to_f32(T[(size_t)i * c + j]);
            T[(size_t)r * c + j] = gdn_f32_to_f16(s);
        }
        for (int j = r + 1; j < c; j++) T[(size_t)r * c + j] = 0;
    }
    return 0;
}

/* ================= B2: delta-rule chunk (per head 批) ================= */
int delta_chunk_f16(rec_state_t* st, const int16_t* k, const int16_t* v,
                    const int16_t* q, const int16_t* beta, const int16_t* g,
                    int16_t* y, int c, int ntok, int q_raw) {
    if (c < 1 || c > GDN_C_MAX) return 1;
    int H = st->h, d = st->d;
    float scale = q_raw ? 1.0f : 1.0f / sqrtf((float)d);
    int16_t* T = malloc((size_t)c * c * 2);
    /* kf/vf/qf/u/bmat/bf 走 HVX 原语 (qf32 累加 / 向量乘), memalign 128B 对齐
     * (镜像 gdn_block_f16 供给面); G/T/Lh 仍标量 malloc。host 侧 gdn_hvx_* 落
     * 标量体 (与旧式逐字同), 数值路径不变; 设备侧 d 须 32 整数倍 (GDN D=128)。 */
    float *kf = memalign(128, (size_t)c * d * 4), *vf = memalign(128, (size_t)c * d * 4),
          *qf = memalign(128, (size_t)c * d * 4), *G = malloc((size_t)c * 4),
          *bf = memalign(128, (size_t)d * 4), *u = memalign(128, (size_t)c * d * 4),
          *bmat = memalign(128, (size_t)c * d * 4);
    int16_t* Lh = malloc((size_t)c * c * 2);
    if (!T || !kf || !vf || !qf || !G || !bf || !u || !bmat || !Lh) {
        free(T); free(kf); free(vf); free(qf); free(G); free(bf); free(u); free(bmat); free(Lh);
        return 2;
    }
    const int nvec = d / 32;   /* HVX f32 每向量 32 lane (d=128 → 4) */
    for (int h = 0; h < H; h++) {
        const int16_t *kh = k + (size_t)h * ntok * d, *vh = v + (size_t)h * ntok * d,
                      *qh = q + (size_t)h * ntok * d;
        const int16_t *bh = beta + (size_t)h * ntok, *gh = g + (size_t)h * ntok;
        float* S = st->s + (size_t)h * d * d;
        /* G inclusive cumsum; f16 化三矩阵 */
        float acc = 0.0f;
        for (int t = 0; t < c; t++) {
            acc += gdn_f16_to_f32(gh[t]);
            G[t] = acc;
            for (int n = 0; n < d; n++) {
                kf[(size_t)t * d + n] = gdn_f16_to_f32(kh[(size_t)t * d + n]);
                vf[(size_t)t * d + n] = gdn_f16_to_f32(vh[(size_t)t * d + n]);
                qf[(size_t)t * d + n] = gdn_f16_to_f32(qh[(size_t)t * d + n]) * scale;
            }
        }
        /* L[j][i] = exp(G_j-G_i)·(k_j·k_i)·β_j, 严格下三角 → f16 */
        for (int j = 0; j < c; j++) {
            float bj = gdn_f16_to_f32(bh[j]);
            for (int i = 0; i < j; i++) {
                float s = gdn_hvx_dot(kf + (size_t)j * d, kf + (size_t)i * d, d, nvec);
                Lh[(size_t)j * c + i] = gdn_f32_to_f16(expf(G[j] - G[i]) * s * bj);
            }
            for (int i = j; i < c; i++) Lh[(size_t)j * c + i] = 0;
        }
        /* B3 显式调用: T = (I+L)^{-1} */
        if (solve_tri_f16(Lh, T, c)) { /* unreachable */ }
        /* bmat_j = β_j(v_j − e^{G_j} S0 k_j);  u = T bmat (行回代) */
        for (int j = 0; j < c; j++) {
            float ej = expf(G[j]), bj = gdn_f16_to_f32(bh[j]);
            for (int m = 0; m < d; m++) {
                float sk = gdn_hvx_dot(S + (size_t)m * d, kf + (size_t)j * d, d, nvec);
                bmat[(size_t)j * d + m] = bj * (vf[(size_t)j * d + m] - ej * sk);
                u[(size_t)j * d + m] = 0.0f;
            }
            for (int i = 0; i <= j; i++) {
                float tij = gdn_f16_to_f32(T[(size_t)j * c + i]);
                if (tij == 0.0f) continue;
                gdn_hvx_axpy(u + (size_t)j * d, bmat + (size_t)i * d, tij, d, nvec);
            }
        }
        /* y_t = e^{G_t}(S0 q_t) + Σ_{i<=t} e^{G_t-G_i}(q_t·k_i)·u_i */
        for (int t = 0; t < c; t++) {
            float et = expf(G[t]);
            for (int m = 0; m < d; m++) {
                float s = gdn_hvx_dot(S + (size_t)m * d, qf + (size_t)t * d, d, nvec);
                bf[m] = et * s;                              /* 复用 bf */
                y[(size_t)h * ntok * d + (size_t)t * d + m] = 0;
            }
            for (int i = 0; i <= t; i++) {
                float w = expf(G[t] - G[i]);
                float qk = gdn_hvx_dot(qf + (size_t)t * d, kf + (size_t)i * d, d, nvec) * w;
                if (qk == 0.0f) continue;
                gdn_hvx_axpy(bf, u + (size_t)i * d, qk, d, nvec);
            }
            for (int m = 0; m < d; m++)
                y[(size_t)h * ntok * d + (size_t)t * d + m] = gdn_f32_to_f16(bf[m]);
        }
        /* S' = e^{G_C}S0 + Σ_i e^{G_C-G_i} u_i k_i^T  (原地: 先缩放, 再累加) */
        float eC = expf(G[c - 1]);
        gdn_hvx_scale(S, d * d, eC);
        for (int i = 0; i < c; i++) {
            float w = expf(G[c - 1] - G[i]);
            gdn_hvx_outer(S, u + (size_t)i * d, kf + (size_t)i * d, w, d, nvec);
        }
    }
    free(T); free(kf); free(vf); free(qf); free(G); free(bf); free(u); free(bmat); free(Lh);
    return 0;
}

/* ================= 009 整块单发射: B1 conv+SiLU → B2 delta chunk 级联 =================
 * 逐字镜像 compiler ops.cpp GdnBlock 分支 (numerics-contract ver=1, B1-B5):
 *   B2 conv 门 SiLU 后 yc 保 f32 (不落 f16); B3 delta 块内全程 f32, S 原地;
 *   B4 出口 y 取 f16 RNE (gdn_f32_to_f16 与 host 位级同)。
 * research D5: work != NULL 时 16 个中间缓冲全部自 work 区 bump (VTCM 池内
 * carve 窗口, emit 闭式预算), 免逐层 DDR malloc/free; work == NULL 走原
 * malloc 路径 (host 调试/旧 blob 回退), 数值两条路径逐字节同。 */
/* work 区结束标记 base+cap 可能越过 4GB 回绕 (池钉在 VTCM 顶, gdn 窗顶=4GB),
 * 故不能对结束指针做比较 —— 改以「已消费偏移 used + 本次 bytes > cap」判界,
 * used 是小量不回绕。 */
static void* gdn_wk_take(uint8_t** wp, const uint8_t* base, size_t cap,
                         size_t bytes, size_t align, int zero) {
    uintptr_t b = ((uintptr_t)*wp + (align - 1)) & ~(uintptr_t)(align - 1);
    size_t used = (size_t)(b - (uintptr_t)base);
    if (used + bytes > cap) return NULL;
    void* r = (void*)b;
    *wp = (uint8_t*)(b + bytes);
    if (zero) memset(r, 0, bytes);
    return r;
}
/* 阶段计时 (板上成本定位, 累计全调用): runner 走查后打印一行。
 * [0]alloc+cw转 [1]conv B1 [2]norm/转置 [3]Gram L [4]solve_tri
 * [5]GEMM1 S·k [6]bmat/u [7]GEMM2 S·q [8]y 累加 [9]S 更新 */
uint64_t g_gdn_phase_us[10];
uint64_t g_gdn_calls;
#ifdef __hexagon__
#define GDN_T(phase) g_gdn_phase_us[phase] += (uint64_t)HAP_perf_get_time_us()
#define GDN_T0() int64_t _gt = HAP_perf_get_time_us()
#define GDN_T1(phase) g_gdn_phase_us[phase] += (uint64_t)(HAP_perf_get_time_us() - _gt)
#else
#define GDN_T(phase) (void)0
#define GDN_T0() (void)0
#define GDN_T1(phase) (void)0
#endif
int gdn_block_f16(const int16_t* xqkv, const int16_t* cw, const int16_t* beta,
                  const int16_t* g, int16_t* y, float* S,
                  uint8_t* work, size_t work_bytes,
                  int T, int H, int D, int d_conv, int chunk) {
    if (T <= 0 || H <= 0 || D <= 0 || d_conv < 2 || chunk <= 0 || chunk > GDN_C_MAX)
        return 1;
    const int hd = H * D, DI = 3 * hd;   /* 局部 hd 避开文件顶 HD=128 宏 */
    const float scale = 1.0f / sqrtf((float)D);

    uint8_t* wp = work;
    /* kf/vf/qf/u 走 HVX 对齐 (matvec/外积热点); 其余标量 malloc 不动。
     * 修 (2026-10-16): 原宏在 zero=1 时走 calloc 吞掉 align (calloc 不保证
     * 128B 对齐), 但 L/win 走 HVX 128B 对齐存/取 → 板上 misaligned vector
     * store 崩 (-2147482611)。现 zero=1 且 align>8 改走 memalign+memset,
     * 语义(零初始化)不变、对齐到位。 */
#define GDN_ALLOC(p, bytes, align, zero)                                        \
    do {                                                                        \
        (p) = work ? gdn_wk_take(&wp, work, work_bytes, (bytes), (align), (zero)) \
                   : ((align) > 8 ? memalign((align), (bytes))                  \
                                  : ((zero) ? calloc(1, (bytes)) : malloc((bytes)))); \
        if (!(work) && (zero) && (align) > 8 && (p)) memset((p), 0, (bytes));   \
    } while (0)
    float* yc;   GDN_ALLOC(yc,  (size_t)T * DI * 4, 128, 0);
    float* win;  GDN_ALLOC(win, (size_t)(d_conv - 1) * DI * 4, 128, 1);
    float* kf;   GDN_ALLOC(kf,  (size_t)chunk * D * 4, 128, 0);
    float* vf;   GDN_ALLOC(vf,  (size_t)chunk * D * 4, 128, 0);
    float* qf;   GDN_ALLOC(qf,  (size_t)chunk * D * 4, 128, 0);
    float* Gc;   GDN_ALLOC(Gc,  (size_t)chunk * 4, 8, 0);
    float* L;    GDN_ALLOC(L,   (size_t)chunk * chunk * 4, 128, 1);
    float* Tm;   GDN_ALLOC(Tm,  (size_t)chunk * chunk * 4, 128, 1);
    float* bmat; GDN_ALLOC(bmat, (size_t)chunk * D * 4, 128, 0);
    float* u;    GDN_ALLOC(u,    (size_t)chunk * D * 4, 128, 0);
    float* bf;   GDN_ALLOC(bf,   (size_t)D * 4, 128, 0);
    float* kf_T; GDN_ALLOC(kf_T, (size_t)D * chunk * 4, 128, 0);   /* kf 转置 [D][c] 供 GEMM (T022d) */
    float* qf_T; GDN_ALLOC(qf_T, (size_t)D * chunk * 4, 128, 0);   /* qf 转置 [D][c] */
    float* gemm_buf; GDN_ALLOC(gemm_buf, (size_t)D * chunk * 4, 128, 0); /* GEMM 输出 [D][c] scratch */
    float* QK; GDN_ALLOC(QK, (size_t)chunk * chunk * 4, 128, 0);   /* qf·kf^T [c][c] 供 yacc 批量 dot (015 P2) */
    float* cw_f32; GDN_ALLOC(cw_f32, (size_t)d_conv * DI * 4, 128, 0);  /* cw f16→f32 预转一次 (T022d) */
    float* xf_buf; GDN_ALLOC(xf_buf, (size_t)DI * 4, 128, 0);           /* 每 token xqkv f16→f32 一次 */
#undef GDN_ALLOC
    if (!yc || !win || !kf || !vf || !qf || !Gc || !L || !Tm || !bmat || !u || !bf ||
        !kf_T || !qf_T || !gemm_buf || !QK || !cw_f32 || !xf_buf) {
        if (!work) {
            free(yc); free(win); free(kf); free(vf); free(qf); free(Gc);
            free(L); free(Tm); free(bmat); free(u); free(bf);
            free(kf_T); free(qf_T); free(gemm_buf); free(QK);
            free(cw_f32); free(xf_buf);
        }
        return 2;
    }

    g_gdn_calls++;
    GDN_T0();
    /* B2: conv1d + SiLU, 滑窗零起, 推窗在算后 (conv_f16 同式, 出 f32 yc) */
    /* cw f16→f32 预转一次 (原内层 786K 冗余转换); xqkv 每 token 转一次复用 (T022d) */
#if HVX_GDN_ENABLED
    if ((DI & 63) == 0)
        gdn_hvx_cvt_f32(cw_f32, cw, d_conv * DI, (d_conv * DI) / 64);
    else
#endif
    for (int i = 0; i < d_conv * DI; i++) cw_f32[i] = gdn_f16_to_f32(cw[i]);
    GDN_T1(0);
    {
    GDN_T0();
#if HVX_GDN_ENABLED
    if ((DI & 63) == 0)
        for (int t = 0; t < T; t++) {
            const int16_t* xr = xqkv + (size_t)t * DI;
            float* yr = yc + (size_t)t * DI;
            const int nv = DI / 32;
            gdn_hvx_cvt_f32(xf_buf, xr, DI, DI / 64);
            gdn_hvx_conv_silu(yr, cw_f32, win, xf_buf, DI, d_conv, nv);
            memmove(win, win + DI, (size_t)(d_conv - 2) * DI * sizeof(float));
            float* wtail = win + (size_t)(d_conv - 2) * DI;
            for (int v = 0; v < nv; v++)
                *(HVX_Vector*)(wtail + (size_t)v * 32) = *(const HVX_Vector*)(xf_buf + (size_t)v * 32);
        }
    else
#endif
    for (int t = 0; t < T; t++) {
        const int16_t* xr = xqkv + (size_t)t * DI;
        float* yr = yc + (size_t)t * DI;
        for (int ch = 0; ch < DI; ch++) xf_buf[ch] = gdn_f16_to_f32(xr[ch]);
        for (int ch = 0; ch < DI; ch++) {
            float s = 0.0f;
            for (int i = 0; i < d_conv - 1; i++)
                s += cw_f32[(size_t)i * DI + ch] * win[(size_t)i * DI + ch];
            s += cw_f32[(size_t)(d_conv - 1) * DI + ch] * xf_buf[ch];
            yr[ch] = s / (1.0f + expf(-s));
        }
        memmove(win, win + DI, (size_t)(d_conv - 2) * DI * sizeof(float));
        /* 新样本入窗复用 xf_buf (已 f32) */
        for (int ch = 0; ch < DI; ch++)
            win[(size_t)(d_conv - 2) * DI + ch] = xf_buf[ch];
    }
    GDN_T1(1);
    }

    /* B3: delta 块内全 f32; state S f32 原地 (prefill 零起), chunk 跨块携带 */
    const int nvec = D / 32;   /* HVX f32 每向量 32 lane (D=128 → 4 向量) */
    for (int h = 0; h < H; h++) {
        float* Sh = S + (size_t)h * D * D;
        for (int base = 0; base < T; base += chunk) {
            const int c = (T - base < chunk) ? (T - base) : chunk;
            float acc = 0.0f;
            for (int t = 0; t < c; t++) {
                acc += gdn_f16_to_f32(g[(base + t) * H + h]);
                Gc[t] = acc;
            }
            GDN_T0();
            for (int t = 0; t < c; t++) {
                const float* yrow = yc + (size_t)(base + t) * DI + (size_t)h * D;
                /* q/k l2norm (HF use_qk_l2norm_in_kernel=True, 镜像 ops.cpp GdnBlock):
                 * x·rsqrt(Σx²+1e-6); q 再乘 1/√D, k 不乘; v 原样。缺失则
                 * k·k 点积无界 → (I+L)^{-1} 前代回代溢出 → Inf/NaN (全链门败因)。 */
                float qsq = 0.0f, ksq = 0.0f;
                for (int nn = 0; nn < D; nn++) {
                    float qv = yrow[nn], kv = yrow[hd + nn];
                    qsq += qv * qv;
                    ksq += kv * kv;
                    vf[(size_t)t * D + nn] = yrow[2 * hd + nn];
                }
                const float iq = 1.0f / sqrtf(qsq + 1e-6f);
                const float ik = 1.0f / sqrtf(ksq + 1e-6f);
                for (int nn = 0; nn < D; nn++) {
                    kf[(size_t)t * D + nn] = yrow[hd + nn] * ik;
                    qf[(size_t)t * D + nn] = yrow[nn] * (iq * scale);
                }
                for (int nn = 0; nn < D; nn++) {                             /* 转置 [D][c] 供 GEMM (T022d) */
                    kf_T[(size_t)nn * c + t] = kf[(size_t)t * D + nn];
                    qf_T[(size_t)nn * c + t] = qf[(size_t)t * D + nn];
                }
            }
            GDN_T1(2);
            {
            GDN_T0();
            if (c == 32) {   /* 整 chunk: 批量 Gram (kf_T 已转置), 替代 c²/2 个独立 dot */
                gdn_hvx_gramm(kf_T, L, D, c);
                for (int j = 0; j < c; j++) {
                    float bj = gdn_f16_to_f32(beta[(base + j) * H + h]);
                    for (int i = 0; i < j; i++)
                        L[(size_t)j * c + i] = expf(Gc[j] - Gc[i]) * L[(size_t)j * c + i] * bj;
                    for (int i = j; i < c; i++) L[(size_t)j * c + i] = 0.0f;
                }
            } else {   /* 尾 chunk (<32): 标量 dot, 位级同旧式 */
                for (int j = 0; j < c; j++) {
                    float bj = gdn_f16_to_f32(beta[(base + j) * H + h]);
                    for (int i = 0; i < j; i++) {
                        float s = gdn_hvx_dot(kf + (size_t)j * D, kf + (size_t)i * D, D, nvec);
                        L[(size_t)j * c + i] = expf(Gc[j] - Gc[i]) * s * bj;
                    }
                    for (int i = j; i < c; i++) L[(size_t)j * c + i] = 0.0f;
                }
            }
            GDN_T1(3);
            }
            {
            GDN_T0();
            for (int r = 0; r < c; r++)                            /* Tm=(I+L)^{-1}, ref_solve_tri 同式 */
                for (int j = 0; j < c; j++) {
                    float s = (r == j) ? 1.0f : 0.0f;
                    for (int i = 0; i < r; i++)
                        s -= L[(size_t)r * c + i] * Tm[(size_t)i * c + j];
                    Tm[(size_t)r * c + j] = s;
                }
            GDN_T1(4);
            }
            {
            GDN_T0();
            gdn_hvx_gemm(Sh, kf_T, gemm_buf, D, c);   /* gemm_buf[m][j] = (S·k_j)[m], 一次算全 j */
            GDN_T1(5);
            for (int j = 0; j < c; j++) {
                float ej = expf(Gc[j]);
                float bj = gdn_f16_to_f32(beta[(base + j) * H + h]);
                for (int m = 0; m < D; m++)
                    bmat[(size_t)j * D + m] = bj * (vf[(size_t)j * D + m] - ej * gemm_buf[(size_t)m * c + j]);
                gdn_hvx_tri_accum(u, bmat, Tm, j, c, D, nvec);   /* u[j] = Σ_{i≤j} Tm[j][i]·bmat[i] */
            }
            GDN_T1(6);
            }
            {
            GDN_T0();
            gdn_hvx_gemm(Sh, qf_T, gemm_buf, D, c);   /* gemm_buf[m][t] = (S·q_t)[m], 一次算全 t */
            GDN_T1(7);
            {
            GDN_T0();
            if (c == 32) {   /* 整 chunk: 批量 Q·K^T 替代 c²/2 独立 dot (015 P2) */
                gdn_hvx_gramm2(qf_T, kf_T, QK, D, c);
            } else {         /* 尾 chunk (<32): 标量 dot 填 QK, 位级同旧式 (decode 安全) */
                for (int t = 0; t < c; t++)
                    for (int i = 0; i <= t; i++)
                        QK[(size_t)t * c + i] = gdn_hvx_dot(qf + (size_t)t * D, kf + (size_t)i * D, D, nvec);
            }
            for (int t = 0; t < c; t++) {
                float et = expf(Gc[t]);
                for (int m = 0; m < D; m++) bf[m] = gemm_buf[(size_t)m * c + t] * et;
                for (int i = 0; i <= t; i++) {
                    float w = expf(Gc[t] - Gc[i]);
                    float qk = QK[(size_t)t * c + i] * w;
                    if (qk == 0.0f) continue;
                    gdn_hvx_axpy(bf, u + (size_t)i * D, qk, D, nvec);
                }
                for (int m = 0; m < D; m++)                         /* B4: 出口 f16 RNE */
                    y[(size_t)(base + t) * hd + (size_t)h * D + m] = gdn_f32_to_f16(bf[m]);
            }
            GDN_T1(8);
            }
            }
            {
            GDN_T0();
            float eC = expf(Gc[c - 1]);
            gdn_hvx_scale(Sh, D * D, eC);
            for (int i = 0; i < c; i++) {
                float w = expf(Gc[c - 1] - Gc[i]);
                /* S += w·u_i·k_i^T (HVX rank-1 外积) */
                gdn_hvx_outer(Sh, u + (size_t)i * D, kf + (size_t)i * D, w, D, nvec);
            }
            GDN_T1(9);
            }
        }
    }

    if (!work) {
        free(yc); free(win); free(kf); free(vf); free(qf); free(Gc);
        free(L); free(Tm); free(bmat); free(u); free(bf);
        free(kf_T); free(qf_T); free(gemm_buf); free(QK);
        free(cw_f32); free(xf_buf);
    }
    return 0;
}
