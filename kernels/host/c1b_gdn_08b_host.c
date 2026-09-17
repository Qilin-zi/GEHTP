/* c1b_gdn_08b_host.c — gdn_sm kernel 族 0.8B 生产形状 host 数值验证
 * =====================================================================
 * 0.8B Qwen3.5 GDN: conv d_inner=6144 (qkv concat 3×2048), d_conv=4;
 *                   delta h=16, d=128, chunk=32 (seq) / 64 (C_MAX 边界)
 * 门 (同例19 口径): conv/delta kernel vs 标量 oracle cos; delta 终态 S bit 级;
 * chunk 拆分一致 (32 = 16+16); 跨 chunk 状态接续 (32+32 对 64 一次性)。
 * 编译: gcc -O2 -I kernels/include c1b_gdn_08b_host.c \
 *        kernels/src/hvx/gdn_kern.c kernels/src/hvx/gdn_ref.c -lm -o /tmp/c1b
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gdn_sm.h"

#define DI   6144   /* conv d_inner (0.8B qkv) */
#define DCNV 4
#define H    16
#define D    128
#define NT   64     /* 两 chunk × 32 */
#define CK   32

static double cos_sim(const float* a, const float* b, size_t n) {
    double p = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        p += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}

static int16_t* gen_f16(uint32_t* lcg, size_t n) {
    int16_t* b = malloc(n * 2);
    for (size_t i = 0; i < n; i++) b[i] = gdn_f32_to_f16(gdn_lcg_norm(lcg) * 0.5f);
    return b;
}
static int16_t* gen_beta(uint32_t* lcg, size_t n) {   /* β = |N|+0.2 (例19 域) */
    int16_t* b = malloc(n * 2);
    for (size_t i = 0; i < n; i++) b[i] = gdn_f32_to_f16(fabsf(gdn_lcg_norm(lcg)) + 0.2f);
    return b;
}
static int16_t* gen_decay(uint32_t* lcg, size_t n) {  /* g = -|N|*0.5 (负 log-decay) */
    int16_t* b = malloc(n * 2);
    for (size_t i = 0; i < n; i++) b[i] = gdn_f32_to_f16(-fabsf(gdn_lcg_norm(lcg)) * 0.5f);
    return b;
}

int main(void) {
    uint32_t lcg = 20260917u;
    int fails = 0;

    /* ---- G1: conv1d_block @ d_inner=6144, d_conv=4, m=NT ---- */
    {
        int16_t* w16 = gen_f16(&lcg, DCNV * DI);
        int16_t* x16 = gen_f16(&lcg, NT * DI);
        int16_t* y16 = malloc(NT * DI * 2);
        float* win = calloc((DCNV - 1) * DI, 4);
        conv_state_t st = { DI, DCNV, win };
        conv1d_block_f16(&st, w16, x16, y16, NT);

        /* oracle: 同初始零窗, f32 逐步 */
        float* wf = malloc(DCNV * DI * 4), *xf = malloc(NT * DI * 4), *yof = malloc(NT * DI * 4);
        for (int i = 0; i < DCNV * DI; i++) wf[i] = gdn_f16_to_f32(w16[i]);
        for (int i = 0; i < NT * DI; i++) xf[i] = gdn_f16_to_f32(x16[i]);
        float* win2 = calloc((DCNV - 1) * DI, 4);
        conv_state_t so = { DI, DCNV, win2 };
        ref_conv_block(&so, wf, xf, yof, NT);

        float* yf = malloc(NT * DI * 4);
        for (int i = 0; i < NT * DI; i++) yf[i] = gdn_f16_to_f32(y16[i]);
        double c = cos_sim(yf, yof, (size_t)NT * DI);
        printf("G1 conv_block DI=%d m=%d: cos=%.9f %s\n", DI, NT, c, c > 0.9999 ? "PASS" : "FAIL");
        if (c <= 0.999999) fails++;
        free(w16); free(x16); free(y16); free(win); free(wf); free(xf); free(yof); free(win2); free(yf);
    }


    /* ---- G3: 多头独立流 chunk vs oracle (真 0.8B 口径: h=16 独立状态) ---- */
    {
        const int TOK = 32;
        int16_t *k16 = gen_f16(&lcg, TOK * D), *v16 = gen_f16(&lcg, TOK * D),
                *q16 = gen_f16(&lcg, TOK * D), *be16 = gen_beta(&lcg, TOK),
                *g16  = gen_decay(&lcg, TOK);
        float* yf = malloc(TOK * D * 4), *yof = malloc(TOK * D * 4);
        double worst = 1.0;
        int state_diff = 0;
        for (int h = 0; h < H; h++) {
            float* S1 = calloc(D * D, 4);
            rec_state_t rk = { 1, D, S1 };
            int16_t* y16 = malloc(TOK * D * 2);
            delta_chunk_f16(&rk, k16, v16, q16, be16, g16, y16, TOK, TOK);
            for (int i = 0; i < TOK * D; i++) yf[i] = gdn_f16_to_f32(y16[i]);

            float* S2 = calloc(D * D, 4);
            rec_state_t ro = { 1, D, S2 };
            float *kf = malloc(D * 4), *vf = malloc(D * 4), *qf = malloc(D * 4);
            for (int t = 0; t < TOK; t++) {
                for (int i = 0; i < D; i++) {
                    kf[i] = gdn_f16_to_f32(k16[(size_t)t * D + i]);
                    vf[i] = gdn_f16_to_f32(v16[(size_t)t * D + i]);
                    qf[i] = gdn_f16_to_f32(q16[(size_t)t * D + i]);
                }
                ref_delta_token(&ro, kf, vf, qf,
                                gdn_f16_to_f32(be16[t]), gdn_f16_to_f32(g16[t]),
                                yof + (size_t)t * D);
            }
            double c = cos_sim(yf, yof, (size_t)TOK * D);
            if (c < worst) worst = c;
            /* 终态: f16 kernel 内部 f32, 与 oracle 比 cos */
            double cs = cos_sim(S1, S2, (size_t)D * D);
            if (cs < 0.9999) state_diff++;
            free(S1); free(S2); free(y16); free(kf); free(vf); free(qf);
        }
        printf("G3 delta_chunk h=%d d=%d c=%d: 最差头 cos=%.9f %s; 终态异头=%d %s\n",
               H, D, TOK, worst, worst > 0.9999 ? "PASS" : "FAIL",
               state_diff, state_diff == 0 ? "PASS" : "FAIL");
        if (worst <= 0.9999 || state_diff) fails++;
        free(k16); free(v16); free(q16); free(be16); free(g16); free(yf); free(yof);
    }

    /* ---- G4: chunk 拆分一致 (c=64 一次性 vs 2×32) ---- */
    {
        const int TOK = 64;
        int16_t *k16 = gen_f16(&lcg, TOK * D), *v16 = gen_f16(&lcg, TOK * D),
                *q16 = gen_f16(&lcg, TOK * D), *be16 = gen_beta(&lcg, TOK),
                *g16  = gen_decay(&lcg, TOK);
        float* Sa = calloc(D * D, 4);
        rec_state_t ra = { 1, D, Sa };
        int16_t* ya = malloc(TOK * D * 2);
        delta_chunk_f16(&ra, k16, v16, q16, be16, g16, ya, TOK, TOK);
        float* Sb = calloc(D * D, 4);
        rec_state_t rb = { 1, D, Sb };
        int16_t* yb = malloc(TOK * D * 2);
        delta_chunk_f16(&rb, k16, v16, q16, be16, g16, yb, CK, TOK);
        delta_chunk_f16(&rb, k16 + CK * D, v16 + CK * D, q16 + CK * D,
                        be16 + CK, g16 + CK, yb + CK * D, CK, TOK - CK);
        float *yaf = malloc(TOK * D * 4), *ybf = malloc(TOK * D * 4);
        for (int i = 0; i < TOK * D; i++) {
            yaf[i] = gdn_f16_to_f32(ya[i]); ybf[i] = gdn_f16_to_f32(yb[i]);
        }
        double c = cos_sim(yaf, ybf, (size_t)TOK * D);
        double cs = cos_sim(Sa, Sb, (size_t)D * D);
        printf("G4 chunk 拆分 64=32+32: y cos=%.9f S cos=%.9f %s\n",
               c, cs, (c > 0.9999 && cs > 0.9999) ? "PASS" : "FAIL");
        if (c <= 0.9999 || cs <= 0.9999) fails++;
        free(k16); free(v16); free(q16); free(be16); free(g16);
        free(Sa); free(Sb); free(ya); free(yb); free(yaf); free(ybf);
    }

    printf("=== %s (fails=%d) ===\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails;
}
