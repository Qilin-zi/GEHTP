/* gdn_ref.c — 标量 f32 oracle + f16 工具 + LCG (host/device 同源双编译) */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "gdn_sm.h"

/* ---------- f16 (round-to-nearest-even, 无 subnormal 特判 → 范围外截断) ---------- */
static float half_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else { /* subnormal: 归一化 */ int e = 0; uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400)); m &= 0x3FF;
            bits = sign | ((uint32_t)(127 - 15 - e + 1) << 23) | (m << 13); }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}
float gdn_f16_to_f32(int16_t h) { return half_to_f32((uint16_t)h); }

int16_t gdn_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFF;
    uint16_t h;
    if (((x >> 23) & 0xFF) == 0xFF) { h = (uint16_t)(sign | 0x7C00 | (man ? 0x200 : 0)); }
    else if (exp >= 31) { h = (uint16_t)(sign | 0x7C00); }              /* overflow → inf */
    else if (exp <= 0) {
        if (exp < -10) { h = (uint16_t)sign; }                          /* < 2^-25 → 0 */
        else {                                                          /* subnormal RNE */
            uint32_t m = 0x800000u | man;
            uint32_t sh = (uint32_t)(14 - exp);                         /* exp∈[-10,0] → sh∈[14,24] */
            uint32_t sub = m >> sh, rem = m & ((1u << sh) - 1);
            sub += (rem > (1u << (sh - 1)) ||
                    (rem == (1u << (sh - 1)) && (sub & 1))) ? 1u : 0u;
            h = (uint16_t)(sign | sub);
        }
    }
    else {
        uint32_t m = man >> 13, r = man & 0x1FFF, half = 0x1000;
        m += (r > half || (r == half && (m & 1))) ? 1u : 0u;            /* RNE */
        if (m >> 10) { m = 0; exp++; }
        h = (uint16_t)(sign | ((uint32_t)exp << 10) | m);
    }
    return (int16_t)h;
}

/* ---------- LCG (与 host/gen_vec.py gdn_lcg 完全一致) ---------- */
uint32_t gdn_lcg_next(uint32_t* st) {
    *st = (uint32_t)((uint64_t)(*st) * 1664525u + 1013904223u);
    return *st;
}
float gdn_lcg_norm(uint32_t* st) {
    /* 12 均匀和 − 6 ≈ N(0,1), 再 ×0.5; 与 python 同式 */
    float s = 0.0f;
    for (int i = 0; i < 12; i++) s += (float)(gdn_lcg_next(st) >> 8) / 16777216.0f;
    return (s - 6.0f) * 0.5f;
}

/* ---------- B1 oracle: y = silu(Σ w[i,ch]·win_full[i,ch]) ---------- */
static float silu_f(float x) { return x / (1.0f + expf(-x)); }

static void conv_push_win(conv_state_t* st, const float* x) {
    int w = st->d_conv - 1, di = st->d_inner;
    memmove(st->win, st->win + di, (size_t)(w - 1) * di * sizeof(float));
    memcpy(st->win + (size_t)(w - 1) * di, x, (size_t)di * sizeof(float));
}
void ref_conv_step(conv_state_t* st, const float* w, const float* x, float* y) {
    int di = st->d_inner;
    /* full = win(前 d_conv-1) + x(1); 先算再推窗 */
    int dc = st->d_conv;
    for (int ch = 0; ch < di; ch++) {
        float s = 0.0f;
        for (int i = 0; i < dc - 1; i++) s += w[(size_t)i * di + ch] * st->win[(size_t)i * di + ch];
        s += w[(size_t)(dc - 1) * di + ch] * x[ch];
        y[ch] = silu_f(s);
    }
    conv_push_win(st, x);
}
void ref_conv_block(conv_state_t* st, const float* w, const float* x, float* y, int m) {
    for (int t = 0; t < m; t++)
        ref_conv_step(st, w, x + (size_t)t * st->d_inner, y + (size_t)t * st->d_inner);
}

/* ---------- B2 oracle: per-token 递推 (ggmlHTPV3E pp-thread 同式) ---------- */
void ref_delta_token(rec_state_t* st, const float* k, const float* v,
                     const float* q, float beta, float g, float* y) {
    int d = st->d;
    float* S = st->s;
    float a = expf(g), scale = 1.0f / sqrtf((float)d);
    float* yk = (float*)malloc((size_t)d * sizeof(float));
    for (int j = 0; j < d; j++) {                 /* S *= exp(g); yk = S k */
        float* row = S + (size_t)j * d, s = 0.0f;
        for (int n = 0; n < d; n++) { row[n] *= a; s += row[n] * k[n]; }
        yk[j] = s;
    }
    for (int j = 0; j < d; j++) {                 /* S += β(v-yk)⊗k; y = scale S q */
        float dj = beta * (v[j] - yk[j]);
        float* row = S + (size_t)j * d, s = 0.0f;
        for (int n = 0; n < d; n++) { row[n] += dj * k[n]; s += row[n] * q[n]; }
        y[j] = s * scale;
    }
    free(yk);
}

/* ---------- B3 oracle: T = (I+L)^{-1} 列回代 ---------- */
void ref_solve_tri(const float* L, float* T, int c) {
    /* (I+L)T = I → T[r][j] = δ(r,j) − Σ_{i<r} L[r][i]·T[i][j] */
    for (int r = 0; r < c; r++) {
        for (int j = 0; j < c; j++) {
            float s = (r == j) ? 1.0f : 0.0f;
            for (int i = 0; i < r; i++) s -= L[(size_t)r * c + i] * T[(size_t)i * c + j];
            T[(size_t)r * c + j] = s;
        }
    }
}

/* ---------- U14 树形 oracle (gdn_tree.h) ---------- */
void ref_delta_tree(const float* S0, int d, int t, const int* parent,
                    const float* k, const float* v, const float* q,
                    const float* beta, const float* g,
                    float* y, float* states) {
    for (int i = 0; i < t; i++) {
        const float* sp = (parent[i] < 0) ? S0 : states + (size_t)parent[i] * d * d;
        rec_state_t st = { 1, d, states + (size_t)i * d * d };
        memcpy(st.s, sp, (size_t)d * d * sizeof(float));
        ref_delta_token(&st, k + (size_t)i * d, v + (size_t)i * d,
                        q + (size_t)i * d, beta[i], g[i], y + (size_t)i * d);
    }
}

void ref_tree_closed(const float* S0, int d, int t, const int* parent,
                     const float* k, const float* v, const float* q,
                     const float* beta, const float* g,
                     float* y, float* states) {
    /* Pg / e^{Pg} */
    float* pg = (float*)malloc((size_t)t * sizeof(float));
    float* a  = (float*)malloc((size_t)t * sizeof(float));
    for (int i = 0; i < t; i++) {
        pg[i] = g[i] + ((parent[i] < 0) ? 0.0f : pg[parent[i]]);
        a[i]  = expf(pg[i]);
    }
    /* L[j][i] = e^{Pg_j-Pg_i} (k_j·k_i) β_j, i ∈ Anc(j) */
    float* L = (float*)calloc((size_t)t * t, sizeof(float));
    for (int j = 1; j < t; j++) {
        int p = parent[j];
        while (p >= 0) {
            float dot = 0.0f;
            for (int n = 0; n < d; n++)
                dot += k[(size_t)j * d + n] * k[(size_t)p * d + n];
            L[(size_t)j * t + p] = (a[j] / a[p]) * dot * beta[j];
            p = parent[p];
        }
    }
    /* B[j][·] = β_j (v_j − e^{Pg_j} S0 k_j)  (d 维) */
    float* s0k = (float*)malloc((size_t)d * sizeof(float));
    float* B   = (float*)malloc((size_t)t * d * sizeof(float));
    for (int j = 0; j < t; j++) {
        for (int r = 0; r < d; r++) {                 /* (S0 k_j)[行 r] */
            float s = 0.0f;
            for (int n = 0; n < d; n++) s += S0[(size_t)r * d + n] * k[(size_t)j * d + n];
            s0k[r] = s;
        }
        for (int n = 0; n < d; n++)
            B[(size_t)j * d + n] = beta[j] * (v[(size_t)j * d + n] - a[j] * s0k[n]);
    }
    /* 前代换 (I+L)U = B: L 行只含祖先 (值非零即祖先) */
    float* U = (float*)malloc((size_t)t * d * sizeof(float));
    for (int j = 0; j < t; j++)
        for (int n = 0; n < d; n++) {
            float s = B[(size_t)j * d + n];
            for (int i = 0; i < j; i++)
                if (L[(size_t)j * t + i] != 0.0f)
                    s -= L[(size_t)j * t + i] * U[(size_t)i * d + n];
            U[(size_t)j * d + n] = s;
        }
    /* y_j = s·[e^{Pg_j} S0 q_j + Σ_{i∈Anc(j)∪{j}} e^{Pg_j-Pg_i} (q_j·k_i) u_i] */
    float scale = 1.0f / sqrtf((float)d);
    float* s0q = (float*)malloc((size_t)d * sizeof(float));
    for (int j = 0; j < t; j++) {
        for (int r = 0; r < d; r++) {
            float s = 0.0f;
            for (int n = 0; n < d; n++) s += S0[(size_t)r * d + n] * q[(size_t)j * d + n];
            s0q[r] = s;
        }
        for (int r = 0; r < d; r++) {
            float acc = a[j] * s0q[r];
            for (int i = j; ; i = parent[i]) {         /* j 自身 + 祖先链 */
                float qk = 0.0f;
                for (int n = 0; n < d; n++)
                    qk += q[(size_t)j * d + n] * k[(size_t)i * d + n];
                acc += (a[j] / a[i]) * qk * U[(size_t)i * d + r];
                if (parent[i] < 0) break;
            }
            y[(size_t)j * d + r] = acc * scale;
        }
    }
    /* S_a = e^{Pg_a} S0 + Σ_{j∈Anc(a)∪{a}} e^{Pg_a-Pg_j} u_j k_j^T */
    if (states) {
        for (int ja = 0; ja < t; ja++) {
            float* Sa = states + (size_t)ja * d * d;
            for (size_t z = 0; z < (size_t)d * d; z++) Sa[z] = a[ja] * S0[z];
            for (int i = ja; ; i = parent[i]) {
                float w = a[ja] / a[i];
                for (int r = 0; r < d; r++)
                    for (int n = 0; n < d; n++)
                        Sa[(size_t)r * d + n] += w * U[(size_t)i * d + r] * k[(size_t)i * d + n];
                if (parent[i] < 0) break;
            }
        }
    }
    free(pg); free(a); free(L); free(s0k); free(B); free(U); free(s0q);
}
