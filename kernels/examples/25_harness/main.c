/*
 * 25_harness — U11 金标对拍框架 设备验证
 * =====================================================================
 * 用框架重跑两类已闭合手写用例, 判据:
 *   1) case gdn_sm (例 19 核心): oracle 对拍 cos + 框架内直跑逐字节一致
 *   2) case w4a16_gold (例 17 核心): act_surface → 解码 vs Y_gold_2563
 *      (N,M) 金标 65536/65536 byte-exact
 *   3) sha 确定性: 两轮运行同 case, harn_last_sha 完全一致 (逐字节钉住)
 *   4) case solve_tri (纯标量): 框架不依赖引擎也能跑
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"
#include "harness.h"

#define A  "/data/local/tmp/hvxhmx23/assets/s256"
#define M 256u
#define D 64
#define H 8
#define NT 48
#define CK 16

static double cos_sim(const float* a, const float* b, size_t n) {
    double p = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        p += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}
static void minv_crouton(const uint16_t* surf, uint16_t* dst, uint32_t rows, uint32_t cols) {
    uint32_t n_m32 = rows / 32, n_kt = cols / 32, out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++)
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    const uint16_t* p = surf + out;
                    for (uint32_t c = kt * 32; c < kt * 32 + 32; c++) {
                        dst[(size_t)row0 * cols + c] = p[0];
                        dst[(size_t)(row0 + 1) * cols + c] = p[1];
                        p += 2;
                    }
                    out += 64;
                }
}

/* ---- case 1: gdn_sm (例 19 核心, 同 LCG 种子) ---- */
static int case_gdnsm(void* ud) {
    (void)ud;
    uint32_t lcg = 20260822u;
    size_t nz = (size_t)H * NT * D, nb = (size_t)H * NT;
    int16_t *k16 = malloc(nz*2), *v16 = malloc(nz*2), *q16 = malloc(nz*2);
    int16_t *be = malloc(nb*2), *gg = malloc(nb*2);
    float *kf = malloc(nz*4), *vf = malloc(nz*4), *qf = malloc(nz*4);
    float *bf = malloc(nb*4), *gf = malloc(nb*4);
    float *S0 = malloc((size_t)H*D*D*4), *Sr = malloc((size_t)H*D*D*4), *Sk = malloc((size_t)H*D*D*4);
    float* yr = malloc(nz*4);
    int16_t *y16 = malloc(nz*2), *ydirect = malloc(nz*2);
    for (size_t z = 0; z < nz; z++) {
        kf[z] = gdn_lcg_norm(&lcg)*0.5f; k16[z] = gdn_f32_to_f16(kf[z]);
        vf[z] = gdn_lcg_norm(&lcg)*0.8f; v16[z] = gdn_f32_to_f16(vf[z]);
        qf[z] = gdn_lcg_norm(&lcg)*0.5f; q16[z] = gdn_f32_to_f16(qf[z]);
    }
    for (size_t z = 0; z < nb; z++) {
        bf[z] = fabsf(gdn_lcg_norm(&lcg)) + 0.2f; be[z] = gdn_f32_to_f16(bf[z]);
        gf[z] = -fabsf(gdn_lcg_norm(&lcg)) * 0.5f; gg[z] = gdn_f32_to_f16(gf[z]);
    }
    for (size_t z = 0; z < (size_t)H*D*D; z++) S0[z] = gdn_lcg_norm(&lcg)*0.3f;

    memcpy(Sr, S0, (size_t)H*D*D*4);
    for (int h = 0; h < H; h++) {
        rec_state_t rs = { 1, D, Sr + (size_t)h*D*D };
        for (int t = 0; t < NT; t++)
            ref_delta_token(&rs, kf + ((size_t)h*NT+t)*D, vf + ((size_t)h*NT+t)*D,
                            qf + ((size_t)h*NT+t)*D, bf[(size_t)h*NT+t],
                            gf[(size_t)h*NT+t], yr + ((size_t)h*NT+t)*D);
    }
    /* 框架路径 */
    memcpy(Sk, S0, (size_t)H*D*D*4);
    rec_state_t rk = { H, D, Sk };
    for (int t0 = 0; t0 < NT; t0 += CK) {
        int c = (NT - t0 < CK) ? NT - t0 : CK;
        delta_chunk_f16(&rk, k16 + (size_t)t0*D, v16 + (size_t)t0*D,
                        q16 + (size_t)t0*D, be + t0, gg + t0, y16 + (size_t)t0*D, c, NT);
    }
    harn_emit(y16, nz * 2);
    /* 手写直跑 (同调用序列, 独立缓冲) — 逐字节一致 */
    int16_t* yd = ydirect;
    float* Sd = malloc((size_t)H*D*D*4);
    memcpy(Sd, S0, (size_t)H*D*D*4);
    rec_state_t rd = { H, D, Sd };
    for (int t0 = 0; t0 < NT; t0 += CK) {
        int c = (NT - t0 < CK) ? NT - t0 : CK;
        delta_chunk_f16(&rd, k16 + (size_t)t0*D, v16 + (size_t)t0*D,
                        q16 + (size_t)t0*D, be + t0, gg + t0, yd + (size_t)t0*D, c, NT);
    }
    float* ykf = malloc(nz*4);
    for (size_t z = 0; z < nz; z++) ykf[z] = gdn_f16_to_f32(y16[z]);
    int rc = 0;
    if (cos_sim(yr, ykf, nz) < 0.9999) rc |= 1;
    if (memcmp(y16, ydirect, nz*2) != 0) rc |= 2;
    harn_expect("gdnsm_oracle_cos", cos_sim(yr, ykf, nz) < 0.9999 ? 1 : 0, 0);
    harn_expect("gdnsm_direct_bytes_identical", memcmp(y16, ydirect, nz*2) == 0 ? 0 : 1, 0);
    free(k16);free(v16);free(q16);free(be);free(gg);free(kf);free(vf);free(qf);
    free(bf);free(gf);free(S0);free(Sr);free(Sk);free(Sd);free(yr);free(y16);free(ydirect);free(ykf);
    return rc;
}

/* ---- case 2: w4a16 金标 (例 17 核心) ---- */
static struct { struct wtcache_ctx* wc; struct dc_w4 e; int ready; } g_w4;

static int case_w4gold(void* ud) {
    (void)ud;
    uint32_t bw=0,bb=0,ba=0,bo=0,bs=0,byg=0;
    uint8_t* wt  = dc_read_file(A "/packed_weight.raw", &bw);
    uint8_t* bis = dc_read_file(A "/folded_bias.raw", &bb);
    uint8_t* at  = dc_read_file(A "/act_table.raw", &ba);
    uint8_t* ot  = dc_read_file(A "/out_table.raw", &bo);
    uint8_t* act = dc_read_file(A "/act_surface.raw", &bs);
    uint8_t* yg  = dc_read_file(A "/Y_gold_2563.raw", &byg);
    if (!wt || !bis || !at || !ot || !act || !yg) { free(wt);free(bis);free(at);free(ot);free(act);free(yg); return 1; }
    if (!g_w4.ready) {
        dc_clean_ddr(wt, bw); dc_clean_ddr(bis, bb);
        if (wtcache_open(&g_w4.wc, 4096) != WTC_OK) { return 2; }
        void* vb; uint32_t vs; void* pb; uint32_t pc;
        wtcache_layout(g_w4.wc, &vb, &vs, &pb, &pc);
        struct dc_arena ar;
        dc_arena_init(&ar, (uint8_t*)vb + ((pc + 2047u) & ~2047u), vs - ((pc + 2047u) & ~2047u));
        if (dc_w4_carve(&g_w4.e, &ar, M, M, M, at, ot)) return 3;
        memcpy(g_w4.e.wt, wt, bw);
        memcpy(g_w4.e.bias, bis, bb);
        qurt_mem_cache_clean((qurt_addr_t)g_w4.e.wt, bw,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        qurt_mem_cache_clean((qurt_addr_t)g_w4.e.bias, bb,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        g_w4.ready = 1;
    }
    memcpy(g_w4.e.act, act, bs);
    qurt_mem_cache_clean((qurt_addr_t)g_w4.e.act, bs,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    uint8_t* out1 = memalign(128, M*M*2);
    uint16_t* dec = malloc(M*M*2);
    if (dc_w4_invoke(&g_w4.e)) { free(out1); free(dec); return 4; }
    dc_w4_read_out(&g_w4.e, out1);
    minv_crouton((const uint16_t*)out1, dec, M, M);
    harn_emit(dec, M * M * 2);
    const uint16_t* g = (const uint16_t*)yg;
    uint32_t nbad = 0;
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < M; n++)
            if (dec[(size_t)m*M+n] != g[(size_t)n*M+m]) nbad++;
    harn_expect("w4a16_gold_65536_byteexact", nbad == 0 ? 0 : 1, 0);
    free(out1); free(dec);
    free(wt);free(bis);free(at);free(ot);free(act);free(yg);
    return nbad ? 5 : 0;
}

/* ---- case 3: solve_tri (纯标量, 不依赖引擎) ---- */
static int case_solve_tri(void* ud) {
    (void)ud;
    uint32_t lcg = 777u;
    int16_t* L = malloc(CK*CK*2); int16_t* T = malloc(CK*CK*2);
    float* Lf = malloc(CK*CK*4); float* Tr = malloc(CK*CK*4); float* Tk = malloc(CK*CK*4);
    double cmin = 1;
    for (int h = 0; h < 4; h++) {
        memset(L, 0, CK*CK*2); memset(Lf, 0, CK*CK*4);
        for (int j = 1; j < CK; j++) for (int i = 0; i < j; i++) {
            float v = gdn_lcg_norm(&lcg) * 0.3f;
            Lf[(size_t)j*CK+i] = v; L[(size_t)j*CK+i] = gdn_f32_to_f16(v);
        }
        solve_tri_f16(L, T, CK);
        ref_solve_tri(Lf, Tr, CK);
        for (size_t z = 0; z < (size_t)CK*CK; z++) Tk[z] = gdn_f16_to_f32(T[z]);
        double c = cos_sim(Tr, Tk, (size_t)CK*CK);
        if (c < cmin) cmin = c;
    }
    harn_emit(T, CK*CK*2);
    harn_expect("solve_tri_cos", cmin < 0.99995 ? 1 : 0, 0);
    free(L);free(T);free(Lf);free(Tr);free(Tk);
    return cmin < 0.99995 ? 1 : 0;
}

int main(void) {
    ex_open_result("25_harness");
    harn_begin();

    harn_case("solve_tri", case_solve_tri, NULL);
    char sha_g[65]; 
    harn_case("gdn_sm", case_gdnsm, NULL);
    strncpy(sha_g, harn_last_sha(), 64); sha_g[64] = 0;
    char sha_w[65];
    harn_case("w4a16_gold", case_w4gold, NULL);
    strncpy(sha_w, harn_last_sha(), 64); sha_w[64] = 0;

    /* sha 确定性: 重跑两 case, sha 必须一致 */
    harn_case("gdn_sm_rerun", case_gdnsm, NULL);
    int det_g = strcmp(sha_g, harn_last_sha()) != 0;
    harn_case("w4a16_gold_rerun", case_w4gold, NULL);
    int det_w = strcmp(sha_w, harn_last_sha()) != 0;
    ex_check("harness_sha_deterministic", det_g + det_w, 0);

    if (g_w4.ready) wtcache_close(g_w4.wc);
    return harn_summary();
}
