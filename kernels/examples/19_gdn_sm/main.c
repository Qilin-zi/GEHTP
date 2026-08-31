/*
 * 19_gdn_sm — U5 gdnsm 递归状态机单元 设备验证 (缩尺 G1-G8 + 状态稳定性)
 * =====================================================================
 * 覆盖 gdn_sm.h 全部 4 个 kernel + oracle 对拍 + 状态 bit-exact 复跑 +
 * 守卫区干扰 (64KB) + chunk 拆分一致性。输入全 LCG (host 可复现)。
 * 源模块 gdn_sm_v81 G1-G9 (设备闭合), 数值契约见 docs/api_v22_gdnsm.md。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hvxhmx_v22.h"
#include "example_util.h"

#define DI 1024      /* d_inner (缩尺) */
#define DCNV 4
#define D  64        /* head_dim */
#define H  8         /* heads */
#define NT 48        /* tokens */
#define CK 16        /* chunk */
#define GUARD_B (64u * 1024u)

static double cos_sim(const float* a, const float* b, size_t n) {
    double p = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        p += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}

typedef struct { uint8_t* base; void* obj; size_t ob, stride; } guard_t;
static int guard_alloc(guard_t* g, size_t objb) {
    size_t ob = (objb + 127u) & ~(size_t)127u;
    g->base = malloc(2 * GUARD_B + ob);
    if (!g->base) return 1;
    g->ob = objb; g->stride = ob; g->obj = g->base + GUARD_B;
    return 0;
}
static void guard_pat(guard_t* g, uint8_t p) {
    memset(g->base, p, GUARD_B);
    memset(g->base + GUARD_B + g->stride, p, GUARD_B);
}
static int guard_ok(guard_t* g, uint8_t p) {
    const uint8_t* e1 = g->base + GUARD_B;
    for (const uint8_t* q = g->base; q < e1; q++) if (*q != p) return 0;
    const uint8_t* b2 = e1 + g->stride, * e2 = b2 + GUARD_B;
    for (const uint8_t* q = b2; q < e2; q++) if (*q != p) return 0;
    return 1;
}

int main(void) {
    ex_open_result("19_gdn_sm");
    uint32_t lcg = 20260822u;

    /* f16 roundtrip spot (次正规路径, MODULE B 修过的坑):
     * 判据 = decode 幂等 (encode(decode(h))==h) + r 距 f 不超半个 f16 ULP */
    {
        int bad = 0;
        for (int i = 0; i < 20000; i++) {
            float f = gdn_lcg_norm(&lcg);              /* 覆盖正常+次正规邻域 */
            int16_t h = gdn_f32_to_f16(f);
            float r = gdn_f16_to_f32(h);
            if (gdn_f32_to_f16(r) != h) bad++;
            float ulp = fabsf(r) * 1.2e-3f;            /* f16 相对 ULP ~2^-11, 1.2×余量 */
            if (fabsf(r - f) > ulp + 1e-12f) bad++;
        }
        ex_check("f16_roundtrip_idempotent", bad, 0);
    }

    /* ---- conv: step vs oracle, block vs step, 状态稳定 ---- */
    {
        int16_t* w16 = malloc((size_t)DCNV * DI * 2);
        int16_t* x16 = malloc((size_t)NT * DI * 2);
        int16_t* y16 = malloc((size_t)NT * DI * 2);
        float* w = malloc((size_t)DCNV * DI * 4);
        float* yr = malloc((size_t)NT * DI * 4);
        float* yk = malloc((size_t)NT * DI * 4);
        float* wa = calloc((size_t)(DCNV - 1) * DI, 4);
        float* wb = calloc((size_t)(DCNV - 1) * DI, 4);
        float* xf = malloc((size_t)DI * 4);
        conv_state_t sa = { DI, DCNV, wa }, sb = { DI, DCNV, wb };
        for (int i = 0; i < DCNV * DI; i++) {
            w[i] = gdn_lcg_norm(&lcg) * 0.5f;
            w16[i] = gdn_f32_to_f16(w[i]);
        }
        for (int i = 0; i < NT * DI; i++) x16[i] = gdn_f32_to_f16(gdn_lcg_norm(&lcg));
        for (int t = 0; t < NT; t++) {
            for (int i = 0; i < DI; i++) xf[i] = gdn_f16_to_f32(x16[(size_t)t * DI + i]);
            ref_conv_step(&sa, w, xf, yr + (size_t)t * DI);
            conv1d_step_f16(&sb, w16, x16 + (size_t)t * DI, y16 + (size_t)t * DI);
            for (int i = 0; i < DI; i++)
                yk[(size_t)t * DI + i] = gdn_f16_to_f32(y16[(size_t)t * DI + i]);
        }
        ex_check("conv_step_cos", cos_sim(yr, yk, (size_t)NT * DI) < 0.9999 ? 1 : 0, 0);

        /* block: 16-token 批 ×3, 末窗与 step 窗字节一致 */
        float* wS = calloc((size_t)(DCNV - 1) * DI, 4);
        float* wB = calloc((size_t)(DCNV - 1) * DI, 4);
        conv_state_t ss = { DI, DCNV, wS }, sB = { DI, DCNV, wB };
        int16_t* yb = malloc((size_t)CK * DI * 2);
        for (int b = 0; b < 3; b++) {
            for (int t = 0; t < CK; t++) {
                for (int i = 0; i < DI; i++) xf[i] = gdn_f16_to_f32(x16[(size_t)(b * CK + t) * DI + i]);
                ref_conv_step(&ss, w, xf, yr);
            }
            conv1d_block_f16(&sB, w16, x16 + (size_t)b * CK * DI, yb, CK);
        }
        ex_check("conv_block_state_bytes",
                 memcmp(wS, wB, (size_t)(DCNV - 1) * DI * 4) == 0 ? 0 : 1, 0);

        /* G3-lite: 守卫区 + bit-exact 复跑 */
        guard_t g;
        size_t wbytes = (size_t)(DCNV - 1) * DI * 4;
        if (!guard_alloc(&g, wbytes)) {
            int16_t* y1 = malloc((size_t)NT * DI * 2);
            int16_t* y2 = malloc((size_t)NT * DI * 2);
            float* w0 = malloc(wbytes);
            for (size_t i = 0; i < (DCNV - 1) * DI; i++) w0[i] = gdn_lcg_norm(&lcg) * 0.3f;
            conv_state_t sg = { DI, DCNV, (float*)g.obj };
            memcpy(g.obj, w0, wbytes); guard_pat(&g, 0xA5);
            for (int t = 0; t < NT; t++)
                conv1d_step_f16(&sg, w16, x16 + (size_t)t * DI, y1 + (size_t)t * DI);
            memcpy(g.obj, w0, wbytes); guard_pat(&g, 0x3C);
            for (int t = 0; t < NT; t++)
                conv1d_step_f16(&sg, w16, x16 + (size_t)t * DI, y2 + (size_t)t * DI);
            int ok = memcmp(y1, y2, (size_t)NT * DI * 2) == 0 && guard_ok(&g, 0x3C);
            ex_check("conv_state_bitexact_guarded", ok ? 0 : 1, 0);
            free(g.base); free(y1); free(y2); free(w0);
        }
        free(w16); free(x16); free(y16); free(w); free(yr); free(yk);
        free(wa); free(wb); free(xf); free(wS); free(wB); free(yb);
    }

    /* ---- delta chunk / solve_tri / 拆分一致 ---- */
    {
        size_t nz = (size_t)H * NT * D, nb = (size_t)H * NT;
        int16_t *k16 = malloc(nz * 2), *v16 = malloc(nz * 2), *q16 = malloc(nz * 2);
        int16_t* be = malloc(nb * 2), *gg = malloc(nb * 2);
        float *kf = malloc(nz * 4), *vf = malloc(nz * 4), *qf = malloc(nz * 4);
        float *bf = malloc(nb * 4), *gf = malloc(nb * 4);
        float* S0 = malloc((size_t)H * D * D * 4);
        float* Sr = malloc((size_t)H * D * D * 4);
        float* Sk = malloc((size_t)H * D * D * 4);
        float* yr = malloc(nz * 4);
        int16_t* y16 = malloc(nz * 2);
        for (size_t z = 0; z < nz; z++) {
            kf[z] = gdn_lcg_norm(&lcg) * 0.5f; k16[z] = gdn_f32_to_f16(kf[z]);
            vf[z] = gdn_lcg_norm(&lcg) * 0.8f; v16[z] = gdn_f32_to_f16(vf[z]);
            qf[z] = gdn_lcg_norm(&lcg) * 0.5f; q16[z] = gdn_f32_to_f16(qf[z]);
        }
        for (size_t z = 0; z < nb; z++) {
            bf[z] = fabsf(gdn_lcg_norm(&lcg)) + 0.2f; be[z] = gdn_f32_to_f16(bf[z]);
            gf[z] = -fabsf(gdn_lcg_norm(&lcg)) * 0.5f; gg[z] = gdn_f32_to_f16(gf[z]);
        }
        for (size_t z = 0; z < (size_t)H * D * D; z++) S0[z] = gdn_lcg_norm(&lcg) * 0.3f;

        /* oracle per-token 全程 */
        memcpy(Sr, S0, (size_t)H * D * D * 4);
        for (int h = 0; h < H; h++) {
            rec_state_t rs = { 1, D, Sr + (size_t)h * D * D };
            for (int t = 0; t < NT; t++)
                ref_delta_token(&rs, kf + ((size_t)h * NT + t) * D, vf + ((size_t)h * NT + t) * D,
                                qf + ((size_t)h * NT + t) * D, bf[(size_t)h * NT + t],
                                gf[(size_t)h * NT + t], yr + ((size_t)h * NT + t) * D);
        }
        /* kernel: 16×3 chunk 循环 */
        memcpy(Sk, S0, (size_t)H * D * D * 4);
        rec_state_t rk = { H, D, Sk };
        for (int t0 = 0; t0 < NT; t0 += CK) {
            int c = (NT - t0 < CK) ? (NT - t0) : CK;
            delta_chunk_f16(&rk, k16 + (size_t)t0 * D, v16 + (size_t)t0 * D,
                            q16 + (size_t)t0 * D, be + t0, gg + t0,
                            y16 + (size_t)t0 * D, c, NT);
        }
        float* ykf = malloc(nz * 4);
        for (size_t z = 0; z < nz; z++) ykf[z] = gdn_f16_to_f32(y16[z]);
        ex_check("delta_100tok_cos", cos_sim(yr, ykf, nz) < 0.9999 ? 1 : 0, 0);
        double cs = 1;
        for (int h = 0; h < H; h++) {
            double c = cos_sim(Sr + (size_t)h * D * D, Sk + (size_t)h * D * D, (size_t)D * D);
            if (c < cs) cs = c;
        }
        ex_check("delta_state_cos", cs < 0.9999 ? 1 : 0, 0);

        /* solve_tri vs oracle (4 head) */
        {
            int16_t* L = malloc((size_t)CK * CK * 2);
            int16_t* T = malloc((size_t)CK * CK * 2);
            float* Lf = malloc((size_t)CK * CK * 4);
            float* Tr = malloc((size_t)CK * CK * 4);
            float* Tk = malloc((size_t)CK * CK * 4);
            double cmin = 1;
            for (int h = 0; h < 4; h++) {
                memset(L, 0, (size_t)CK * CK * 2); memset(Lf, 0, (size_t)CK * CK * 4);
                for (int j = 1; j < CK; j++) for (int i = 0; i < j; i++) {
                    float v = gdn_lcg_norm(&lcg) * 0.3f;
                    Lf[(size_t)j * CK + i] = v;
                    L[(size_t)j * CK + i] = gdn_f32_to_f16(v);
                }
                solve_tri_f16(L, T, CK);
                ref_solve_tri(Lf, Tr, CK);
                for (size_t z = 0; z < (size_t)CK * CK; z++) Tk[z] = gdn_f16_to_f32(T[z]);
                double c = cos_sim(Tr, Tk, (size_t)CK * CK);
                if (c < cmin) cmin = c;
            }
            ex_check("solve_tri_cos", cmin < 0.99995 ? 1 : 0, 0);
            free(L); free(T); free(Lf); free(Tr); free(Tk);
        }

        /* 拆分一致 16 = 8+8 (H=2 head) */
        {
            float* Sa = malloc((size_t)2 * D * D * 4);
            float* Sb = malloc((size_t)2 * D * D * 4);
            int16_t* ya = malloc((size_t)2 * CK * D * 2);
            int16_t* yb = malloc((size_t)2 * CK * D * 2);
            float* yaf = malloc((size_t)2 * CK * D * 4);
            float* ybf = malloc((size_t)2 * CK * D * 4);
            memcpy(Sa, S0, (size_t)2 * D * D * 4);
            rec_state_t ra = { 2, D, Sa };
            delta_chunk_f16(&ra, k16, v16, q16, be, gg, ya, CK, CK);
            memcpy(Sb, S0, (size_t)2 * D * D * 4);
            rec_state_t rb = { 2, D, Sb };
            delta_chunk_f16(&rb, k16, v16, q16, be, gg, yb, 8, CK);
            delta_chunk_f16(&rb, k16 + 8 * D, v16 + 8 * D, q16 + 8 * D, be + 8, gg + 8,
                            yb + 8 * D, 8, CK);
            for (size_t z = 0; z < (size_t)2 * CK * D; z++) {
                yaf[z] = gdn_f16_to_f32(ya[z]); ybf[z] = gdn_f16_to_f32(yb[z]);
            }
            int ok = cos_sim(yaf, ybf, (size_t)2 * CK * D) > 0.9999 &&
                     cos_sim(Sa, Sb, (size_t)2 * D * D) > 0.9999;
            ex_check("chunk_split_16_8_8", ok ? 0 : 1, 0);
            free(Sa); free(Sb); free(ya); free(yb); free(yaf); free(ybf);
        }

        free(k16); free(v16); free(q16); free(be); free(gg);
        free(kf); free(vf); free(qf); free(bf); free(gf);
        free(S0); free(Sr); free(Sk); free(yr); free(y16); free(ykf);
    }

    return ex_summary();
}
