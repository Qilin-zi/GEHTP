/* gdn_kern.c — B1/B2/B3 f16 kernel (内部 f32, 状态 f32 外置)
 * chunk 公式推导见 gdn_sm.h 头注; host numpy 对拍 1e-14 (proto_chunk.py) */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "gdn_sm.h"

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
                    int16_t* y, int c, int ntok) {
    if (c < 1 || c > GDN_C_MAX) return 1;
    int H = st->h, d = st->d;
    float scale = 1.0f / sqrtf((float)d);
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
