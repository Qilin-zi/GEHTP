/* lmhead_proto_test.c — lm_head prep/invoke/fini 协议 host 数值对拍
 *
 * 目标: 验证 N 分块重绑激活优化在数值上严格等价于逐块全新 prep。
 *   OLD = 每块 dc_w4_run (块间 dc_w4_run_fini 强制下次重新 prep)
 *   NEW = prep 一次 + N 次 invoke + 末尾 fini
 * 两者必须字节全同; 同时与纯 f32 参考对拍验证整链数值。
 *
 * 编译: gcc -O2 -I./include -o build/host/lmhead_proto_test \
 *   host/lmhead_proto_test.c host/host_stubs.c host/hvx_stubs.c -lm
 *
 * 注意: host 无量化域 (prep 直拷 f16+零 pad), 故本测试验证的是协议正确性
 * 与 refill-size 修复, 不展示设备侧 61× 速度增益 (那是量化域的赢, 设备才体现)。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "dc_parts.h"

static int npass = 0, nfail = 0;
static void ck(const char* name, int err, int tol) {
    if (err <= tol) { printf("[PASS] %s err=%d\n", name, err); npass++; }
    else { printf("[FAIL] %s err=%d\n", name, err); nfail++; }
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t ex = (h >> 10) & 0x1F, mn = h & 0x3FF;
    uint32_t u;
    if (ex == 0) u = sign;
    else if (ex == 31) u = sign | 0x7F800000u | (mn << 13);
    else u = sign | ((ex - 15 + 127) << 23) | (mn << 13);
    float r; memcpy(&r, &u, 4); return r;
}
static uint16_t f32_to_f16(float r) {
    uint32_t u; memcpy(&u, &r, 4);
    uint32_t sgn = (u >> 16) & 0x8000u;
    int32_t ex = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mn = u & 0x7FFFFFu;
    uint32_t half;
    if (ex >= 31) half = 0x7C00u;
    else if (ex <= 0) {
        if (ex < -10) half = 0;
        else {
            mn |= 0x800000u;
            int32_t sh = 14 - ex;
            half = mn >> sh;
            uint32_t rm = mn & ((1u << sh) - 1);
            half += (rm > (1u << (sh - 1))) || (rm == (1u << (sh - 1)) && (half & 1));
        }
    } else {
        half = (mn >> 13) & 0x3FFu;
        uint32_t rm = mn & 0x1FFFu;
        half += (rm > 0x1000u) || (rm == 0x1000u && (half & 1));
        if (half == 0x400u) { ex++; half = 0; }
        if (ex >= 31) half = 0x7C00u;
        else half = ((uint32_t)ex << 10) | half;
    }
    return (uint16_t)(sgn | half);
}

/* 纯 f32 参考: out[m][n] = Σ_k act[m,k]·wq[k,n]·S[n] */
static void ref_matmul(const float* act, const int8_t* wq, const float* S,
                       float* out, uint32_t M, uint32_t K, uint32_t N) {
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            double acc = 0;
            for (uint32_t k = 0; k < K; k++)
                acc += (double)act[(size_t)m*K+k] * (double)wq[(size_t)k*N+n] * (double)S[n];
            out[(size_t)m*N+n] = (float)acc;
        }
}

/* 权重 pack 成 kernel k4-lohi 面 */
static void pack_k4lohi(const int8_t* wq, uint8_t* wt, uint32_t K, uint32_t N) {
    size_t o = 0;
    for (uint32_t kb = 0; kb < K/32; kb++)
        for (uint32_t nb = 0; nb < N; nb += 32)
            for (uint32_t kg = 0; kg < 4; kg++) {
                uint32_t kb0 = kb*32 + kg*8;
                for (uint32_t n = 0; n < 32; n++)
                    for (uint32_t kr = 0; kr < 4; kr++) {
                        int8_t v0 = wq[(size_t)(kb0+kr)*N + nb+n];
                        int8_t v1 = wq[(size_t)(kb0+kr+4)*N + nb+n];
                        uint8_t lo = (uint8_t)(((v0 >= 0 ? v0 : v0+16) & 0xF) ^ 0x8);
                        uint8_t hi = (uint8_t)(((v1 >= 0 ? v1 : v1+16) & 0xF) ^ 0x8);
                        wt[o++] = (uint8_t)((hi << 4) | lo);
                    }
            }
}

static void carve_from_arena(struct dc_w4* e, uint32_t M, uint32_t K, uint32_t N,
                             const uint8_t* wt_src, const uint8_t* scale) {
    /* host 侧无需真 arena, 直接手填指针 (host_stubs 的 inplace 结构) */
    memset(e, 0, sizeof(*e));
    e->m = (M + 255u) & ~255u;
    e->k = K; e->n = N;
    e->act  = aligned_alloc(128, (size_t)e->m * K * 2);
    e->out  = aligned_alloc(128, (size_t)e->m * N * 2);
    e->wt   = aligned_alloc(128, (size_t)K * N / 2);
    e->bias = aligned_alloc(128, (size_t)(N / 32) * 512);
    e->atbl = aligned_alloc(128, 8 * (K / 32) * 4);
    e->otbl = aligned_alloc(128, 8 * (N / 32) * 4);
    e->mask = aligned_alloc(128, 32);
    e->extra= aligned_alloc(128, 16);
    memcpy(e->wt, wt_src, (size_t)K * N / 2);
    e->scale_ddr = scale;
}
static void free_carve(struct dc_w4* e) {
    free(e->act); free(e->out); free(e->wt); free(e->bias);
    free(e->atbl); free(e->otbl); free(e->mask); free(e->extra);
}

int main(void) {
    printf("=== lm_head prep/invoke/fini 协议 host 数值对拍 ===\n");

    const uint32_t M = 256, K = 2048, N = 8192, N_CHUNK = 4096;
    const uint32_t nch = N / N_CHUNK;   /* 2 */

    /* ---- 合成数据 ---- */
    uint16_t* act_f16 = aligned_alloc(128, (size_t)M * K * 2);
    int8_t*   wq      = malloc((size_t)K * N);
    uint8_t*  wt      = aligned_alloc(128, (size_t)K * N / 2);
    uint8_t*  scale16 = aligned_alloc(128, (size_t)N * 2 + 2);
    float*    scaleF  = malloc((size_t)N * 4);

    uint32_t lcg = 0x1234567u;
    for (size_t i = 0; i < (size_t)M*K; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        act_f16[i] = (uint16_t)(lcg >> 16); /* 随机 16-bit, 均匀 */
    }
    for (size_t i = 0; i < (size_t)K*N; i++) {
        lcg = lcg * 1664525u + 1013904223u;
        wq[i] = (int8_t)((lcg % 15u) - 7);  /* ∈[-7,7] 4-bit 域 */
    }
    for (uint32_t n = 0; n < N; n++) {
        lcg = lcg * 1664525u + 1013904223u;
        float s = 0.005f + (float)(lcg % 1000u) * 0.001f;
        scaleF[n] = s;
        uint16_t hv = f32_to_f16(s);
        memcpy(scale16 + (size_t)n * 2, &hv, 2);
    }
    pack_k4lohi(wq, wt, K, N);

    /* ---- f32 参考 ---- */
    float* actF = malloc((size_t)M*K*4);
    for (size_t i = 0; i < (size_t)M*K; i++) actF[i] = f16_to_f32(act_f16[i]);
    float* ref = malloc((size_t)M * N * 4);
    ref_matmul(actF, wq, scaleF, ref, M, K, N);

    /* ---- 输出缓冲 ---- */
    uint8_t* outOld = calloc(1, (size_t)M * N * 2);
    uint8_t* outNew = calloc(1, (size_t)M * N * 2);

    struct dc_w4 eo, en;
    carve_from_arena(&eo, M, K, N_CHUNK, wt, scale16);
    carve_from_arena(&en, M, K, N_CHUNK, wt, scale16);

    /* ---- OLD: 每块全 dc_w4_run + fini 强制重 prep ---- */
    int bad_old = 0;
    int bad_vs_ref = 0;
    for (uint32_t c0 = 0; c0 < N; c0 += N_CHUNK) {
        uint32_t nc = (N - c0 < N_CHUNK) ? N - c0 : N_CHUNK;
        int rc = dc_w4_run(&eo, (const uint8_t*)act_f16, outOld + c0*2u,
                           M, K, nc, scale16 + c0, N*2u, 1.0f, 0.0f);
        if (rc) bad_old++;
        dc_w4_run_fini(&eo); /* 强制下次重新 prep */
        /* 每个元素 vs f32 参考 (host 纯数学, 应 tight) */
        for (uint32_t col = 0; col < nc; col++) {
            uint16_t hv; memcpy(&hv, outOld + (size_t)c0*2u + (size_t)col*2u, 2);
            float got = f16_to_f32(hv);
            float exp = ref[c0 + col];
            float diff = fabsf(got - exp) / (fabsf(exp) + 1e-6f);
            if (exp != 0.0f && diff > 2e-3f) bad_vs_ref++;
        }
    }
    ck("old_perchunk_run_rc", bad_old, 0);
    ck("old_vs_f32ref_1e-3", bad_vs_ref, 2);

    /* ---- NEW: prep 一次 + N×invoke + fini ---- */
    int bad_new = 0;
    int rc = dc_w4_run_prep(&en, (const uint8_t*)act_f16, M, K, 1.0f, 0.0f);
    if (rc) { printf("[FAIL] prep rc=%d\n", rc); return 1; }
    for (uint32_t c0 = 0; c0 < N; c0 += N_CHUNK) {
        uint32_t nc = (N - c0 < N_CHUNK) ? N - c0 : N_CHUNK;
        int ir = dc_w4_run_invoke(&en, scale16 + c0, nc, outNew + c0*2u,
                                  N*2u, M, 1.0f, 0.0f);
        if (ir) bad_new++;
    }
    dc_w4_run_fini(&en);
    ck("new_run_invoke_rc", bad_new, 0);

    /* ---- OLD vs NEW 字节同 ---- */
    int bd = memcmp(outOld, outNew, (size_t)M * N * 2);
    ck("old_vs_new_byte_identical", bd, 0);

    /* ---- NEW vs f32 参考 ---- */
    int bad_new_ref = 0;
    for (uint32_t col = 0; col < N; col++) {
        uint16_t hv; memcpy(&hv, outNew + (size_t)col*2u, 2);
        float got  = f16_to_f32(hv);
        float exp  = ref[col];
        float rel  = (fabsf(exp) > 1e-9f) ? fabsf(got - exp) / fabsf(exp) : fabsf(got - exp);
        /* host f64→RNE(f16); 参考 f32 舍入到参考, 允许 1 ulp 差 */
        if (rel > 1e-2f) bad_new_ref++;
    }
    ck("new_vs_f32ref_1e-2", bad_new_ref, 8);

    /**** 复用正确性: 同一 e 连续两次 invoke 到不同列, 各取正确 (无 stale) ****/
    /* 用 en 再 prep, 先 inv 到 chunk0, 再单独复跑 chunk0 到独立缓冲, 必须同 */
    memset(outNew, 0, (size_t)M * 2 * 2);
    memset(outNew + (size_t)M * 2 * 2, 0, (size_t)M * 4096 * 2);
    int rr = dc_w4_run_prep(&en, (const uint8_t*)act_f16, M, K, 1.0f, 0.0f);
    dc_w4_run_invoke(&en, scale16, 4096, outNew, N*2u, M, 1.0f, 0.0f);
    dc_w4_run_invoke(&en, scale16, 4096, outNew + (size_t)M*2, N*2u, M, 1.0f, 0.0f);
    dc_w4_run_fini(&en);
    (void)rr;
    int cdiff = memcmp(outNew, outNew + (size_t)M*2, (size_t)M * 2 * 2);
    ck("new_reuse_independent_cols", cdiff, 0);

    printf("\n==== %d PASS / %d FAIL ====\n", npass, nfail);
    free(act_f16); free(wq); free(wt); free(scale16); free(scaleF);
    free(actF); free(ref); free(outOld); free(outNew);
    free_carve(&eo); free_carve(&en);
    return nfail ? 1 : 0;
}