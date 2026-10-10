/* gdn_kern.c — B1/B2/B3 f16 kernel (内部 f32, 状态 f32 外置)
 * chunk 公式推导见 gdn_sm.h 头注; host numpy 对拍 1e-14 (proto_chunk.py) */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifndef __hexagon__
#include <malloc.h>   /* memalign 原型 (host x86 缺则按 int 截断指针 → SIGSEGV) */
#endif
#include "gdn_sm.h"

#if defined(__HVX__)
#define HVX_GDN_ENABLED 1
#include "internal/hvx-base.h"
#include "internal/hvx-reduce.h"
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
/* y *= scale (n 元素, n 为 32 整数倍); 单次乘与标量位级同 */
static void gdn_hvx_scale(float* __restrict__ y, int n, float scale) {
    HVX_Vector cs = hvx_vec_splat_f32(scale);
    HVX_Vector* ya = (HVX_Vector*)y;
    for (int i = 0; i < n / 32; i++)
        ya[i] = Q6_Vsf_vmpy_VsfVsf(ya[i], cs);
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
#endif

#define D_INNER 9216
#define D_CONV  4
#define NHEAD   32
#define HD      128

/* ================= B1: conv1d + SiLU ================= */
static void conv_f16(conv_state_t* st, const int16_t* w_f16, const float* xf, float* yf) {
    int dc = st->d_conv, di = st->d_inner;
    const float* win = st->win;
    for (int ch = 0; ch < di; ch++) {
        float s = 0.0f;
        for (int i = 0; i < dc - 1; i++)
            s += gdn_f16_to_f32(w_f16[(size_t)i * di + ch]) * win[(size_t)i * di + ch];
        s += gdn_f16_to_f32(w_f16[(size_t)(dc - 1) * di + ch]) * xf[ch];
        yf[ch] = s / (1.0f + expf(-s));
    }
    memmove(st->win, st->win + di, (size_t)(dc - 2) * di * sizeof(float));
    memcpy(st->win + (size_t)(dc - 2) * di, xf, (size_t)di * sizeof(float));
}

int conv1d_step_f16(conv_state_t* st, const int16_t* w_f16, const int16_t* x, int16_t* y) {
    int di = st->d_inner;
    float *xf = malloc((size_t)di * 4), *yf = malloc((size_t)di * 4);
    if (!xf || !yf) { free(xf); free(yf); return 1; }
    for (int i = 0; i < di; i++) xf[i] = gdn_f16_to_f32(x[i]);
    conv_f16(st, w_f16, xf, yf);
    for (int i = 0; i < di; i++) y[i] = gdn_f32_to_f16(yf[i]);
    free(xf); free(yf);
    return 0;
}

int conv1d_block_f16(conv_state_t* st, const int16_t* w_f16, const int16_t* x, int16_t* y, int m) {
    int di = st->d_inner;
    float *xf = malloc((size_t)di * 4), *yf = malloc((size_t)di * 4);
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
    float *kf = malloc((size_t)c * d * 4), *vf = malloc((size_t)c * d * 4),
          *qf = malloc((size_t)c * d * 4), *G = malloc((size_t)c * 4),
          *bf = malloc((size_t)d * 4), *u = malloc((size_t)c * d * 4),
          *bmat = malloc((size_t)c * d * 4);
    int16_t* Lh = malloc((size_t)c * c * 2);
    if (!T || !kf || !vf || !qf || !G || !bf || !u || !bmat || !Lh) {
        free(T); free(kf); free(vf); free(qf); free(G); free(bf); free(u); free(bmat); free(Lh);
        return 2;
    }
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
                float s = 0.0f;
                for (int n = 0; n < d; n++)
                    s += kf[(size_t)j * d + n] * kf[(size_t)i * d + n];
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
                float sk = 0.0f;
                for (int n = 0; n < d; n++) sk += S[(size_t)m * d + n] * kf[(size_t)j * d + n];
                bmat[(size_t)j * d + m] = bj * (vf[(size_t)j * d + m] - ej * sk);
                u[(size_t)j * d + m] = 0.0f;
            }
            for (int i = 0; i <= j; i++) {
                float tij = gdn_f16_to_f32(T[(size_t)j * c + i]);
                if (tij == 0.0f) continue;
                for (int m = 0; m < d; m++)
                    u[(size_t)j * d + m] += tij * bmat[(size_t)i * d + m];
            }
        }
        /* y_t = e^{G_t}(S0 q_t) + Σ_{i<=t} e^{G_t-G_i}(q_t·k_i)·u_i */
        for (int t = 0; t < c; t++) {
            float et = expf(G[t]);
            for (int m = 0; m < d; m++) {
                float s = 0.0f;
                for (int n = 0; n < d; n++) s += S[(size_t)m * d + n] * qf[(size_t)t * d + n];
                bf[m] = et * s;                              /* 复用 bf */
                y[(size_t)h * ntok * d + (size_t)t * d + m] = 0;
            }
            for (int i = 0; i <= t; i++) {
                float qk = 0.0f, w = expf(G[t] - G[i]);
                for (int n = 0; n < d; n++)
                    qk += qf[(size_t)t * d + n] * kf[(size_t)i * d + n];
                qk *= w;
                if (qk == 0.0f) continue;
                for (int m = 0; m < d; m++) bf[m] += qk * u[(size_t)i * d + m];
            }
            for (int m = 0; m < d; m++)
                y[(size_t)h * ntok * d + (size_t)t * d + m] = gdn_f32_to_f16(bf[m]);
        }
        /* S' = e^{G_C}S0 + Σ_i e^{G_C-G_i} u_i k_i^T  (原地: 先缩放, 再累加) */
        float eC = expf(G[c - 1]);
        for (size_t z = 0; z < (size_t)d * d; z++) S[z] *= eC;
        for (int i = 0; i < c; i++) {
            float w = expf(G[c - 1] - G[i]);
            const float* ui = u + (size_t)i * d;
            const float* ki = kf + (size_t)i * d;
            for (int m = 0; m < d; m++) {
                float wu = w * ui[m];
                if (wu == 0.0f) continue;
                float* row = S + (size_t)m * d;
                for (int n = 0; n < d; n++) row[n] += wu * ki[n];
            }
        }
    }
    free(T); free(kf); free(vf); free(qf); free(G); free(bf); free(u); free(bmat); free(Lh);
    return 0;
}

/* ================= 009 整块单发射: B1 conv+SiLU → B2 delta chunk 级联 =================
 * 逐字镜像 compiler ops.cpp GdnBlock 分支 (numerics-contract ver=1, B1-B5):
 *   B2 conv 门 SiLU 后 yc 保 f32 (不落 f16); B3 delta 块内全程 f32, S 原地;
 *   B4 出口 y 取 f16 RNE (gdn_f32_to_f16 与 host 位级同)。 */
int gdn_block_f16(const int16_t* xqkv, const int16_t* cw, const int16_t* beta,
                  const int16_t* g, int16_t* y, float* S,
                  int T, int H, int D, int d_conv, int chunk) {
    if (T <= 0 || H <= 0 || D <= 0 || d_conv < 2 || chunk <= 0 || chunk > GDN_C_MAX)
        return 1;
    const int hd = H * D, DI = 3 * hd;   /* 局部 hd 避开文件顶 HD=128 宏 */
    const float scale = 1.0f / sqrtf((float)D);

    float* yc  = malloc((size_t)T * DI * 4);
    float* win = calloc((size_t)(d_conv - 1) * DI, 4);
    /* kf/vf/qf/u 走 HVX 对齐 (matvec/外积热点); 其余标量 malloc 不动 */
    float* kf  = memalign(128, (size_t)chunk * D * 4);
    float* vf  = memalign(128, (size_t)chunk * D * 4);
    float* qf  = memalign(128, (size_t)chunk * D * 4);
    float* Gc  = malloc((size_t)chunk * 4);
    float* L   = calloc((size_t)chunk * chunk, 4);
    float* Tm  = calloc((size_t)chunk * chunk, 4);
    float* bmat = memalign(128, (size_t)chunk * D * 4);
    float* u    = memalign(128, (size_t)chunk * D * 4);
    float* bf   = memalign(128, (size_t)D * 4);
    float* kf_T = memalign(128, (size_t)D * chunk * 4);   /* kf 转置 [D][c] 供 GEMM (T022d) */
    float* qf_T = memalign(128, (size_t)D * chunk * 4);   /* qf 转置 [D][c] */
    float* gemm_buf = memalign(128, (size_t)D * chunk * 4); /* GEMM 输出 [D][c] scratch */
    float* cw_f32 = malloc((size_t)d_conv * DI * 4);   /* cw f16→f32 预转一次 (T022d) */
    float* xf_buf = malloc((size_t)DI * 4);            /* 每 token xqkv f16→f32 一次 */
    if (!yc || !win || !kf || !vf || !qf || !Gc || !L || !Tm || !bmat || !u || !bf ||
        !kf_T || !qf_T || !gemm_buf || !cw_f32 || !xf_buf) {
        free(yc); free(win); free(kf); free(vf); free(qf); free(Gc);
        free(L); free(Tm); free(bmat); free(u); free(bf);
        free(kf_T); free(qf_T); free(gemm_buf);
        free(cw_f32); free(xf_buf);
        return 2;
    }

    /* B2: conv1d + SiLU, 滑窗零起, 推窗在算后 (conv_f16 同式, 出 f32 yc) */
    /* cw f16→f32 预转一次 (原内层 786K 冗余转换); xqkv 每 token 转一次复用 (T022d) */
    for (int i = 0; i < d_conv * DI; i++) cw_f32[i] = gdn_f16_to_f32(cw[i]);
    {
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
            for (int t = 0; t < c; t++) {
                const float* yrow = yc + (size_t)(base + t) * DI + (size_t)h * D;
                memcpy(kf + (size_t)t * D, yrow + hd, (size_t)D * 4);        /* q|k|v 三段 (c1b 口径) */
                memcpy(vf + (size_t)t * D, yrow + 2 * hd, (size_t)D * 4);
                memcpy(qf + (size_t)t * D, yrow, (size_t)D * 4);
                gdn_hvx_scale(qf + (size_t)t * D, D, scale);
                for (int nn = 0; nn < D; nn++) {                             /* 转置 [D][c] 供 GEMM (T022d) */
                    kf_T[(size_t)nn * c + t] = kf[(size_t)t * D + nn];
                    qf_T[(size_t)nn * c + t] = qf[(size_t)t * D + nn];
                }
            }
            {
            for (int j = 0; j < c; j++) {
                float bj = gdn_f16_to_f32(beta[(base + j) * H + h]);
                for (int i = 0; i < j; i++) {
                    float s = gdn_hvx_dot(kf + (size_t)j * D, kf + (size_t)i * D, D, nvec);
                    L[(size_t)j * c + i] = expf(Gc[j] - Gc[i]) * s * bj;
                }
                for (int i = j; i < c; i++) L[(size_t)j * c + i] = 0.0f;
            }
            }
            {
            for (int r = 0; r < c; r++)                            /* Tm=(I+L)^{-1}, ref_solve_tri 同式 */
                for (int j = 0; j < c; j++) {
                    float s = (r == j) ? 1.0f : 0.0f;
                    for (int i = 0; i < r; i++)
                        s -= L[(size_t)r * c + i] * Tm[(size_t)i * c + j];
                    Tm[(size_t)r * c + j] = s;
                }
            }
            {
            gdn_hvx_gemm(Sh, kf_T, gemm_buf, D, c);   /* gemm_buf[m][j] = (S·k_j)[m], 一次算全 j */
            for (int j = 0; j < c; j++) {
                float ej = expf(Gc[j]);
                float bj = gdn_f16_to_f32(beta[(base + j) * H + h]);
                for (int m = 0; m < D; m++) {
                    bmat[(size_t)j * D + m] = bj * (vf[(size_t)j * D + m] - ej * gemm_buf[(size_t)m * c + j]);
                    u[(size_t)j * D + m] = 0.0f;
                }
                for (int i = 0; i <= j; i++) {
                    float tij = Tm[(size_t)j * c + i];
                    if (tij == 0.0f) continue;
                    gdn_hvx_axpy(u + (size_t)j * D, bmat + (size_t)i * D, tij, D, nvec);
                }
            }
            }
            {
            gdn_hvx_gemm(Sh, qf_T, gemm_buf, D, c);   /* gemm_buf[m][t] = (S·q_t)[m], 一次算全 t */
            for (int t = 0; t < c; t++) {
                float et = expf(Gc[t]);
                for (int m = 0; m < D; m++) bf[m] = gemm_buf[(size_t)m * c + t] * et;
                for (int i = 0; i <= t; i++) {
                    float w = expf(Gc[t] - Gc[i]);
                    float qk = gdn_hvx_dot(qf + (size_t)t * D, kf + (size_t)i * D, D, nvec) * w;
                    if (qk == 0.0f) continue;
                    gdn_hvx_axpy(bf, u + (size_t)i * D, qk, D, nvec);
                }
                for (int m = 0; m < D; m++)                         /* B4: 出口 f16 RNE */
                    y[(size_t)(base + t) * hd + (size_t)h * D + m] = gdn_f32_to_f16(bf[m]);
            }
            }
            {
            float eC = expf(Gc[c - 1]);
            gdn_hvx_scale(Sh, D * D, eC);
            for (int i = 0; i < c; i++) {
                float w = expf(Gc[c - 1] - Gc[i]);
                /* S += w·u_i·k_i^T (HVX rank-1 外积) */
                gdn_hvx_outer(Sh, u + (size_t)i * D, kf + (size_t)i * D, w, D, nvec);
            }
            }
        }
    }

    free(yc); free(win); free(kf); free(vf); free(qf); free(Gc);
    free(L); free(Tm); free(bmat); free(u); free(bf);
    free(kf_T); free(qf_T); free(gemm_buf);
    free(cw_f32); free(xf_buf);
    return 0;
}
