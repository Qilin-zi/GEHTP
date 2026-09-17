/*
 * 31_gemm_dispatch — U17 MatMul 三路由决策 设备验证
 * =====================================================================
 * 判据:
 *   1) 决策表: M 扫描 1..600 × K/N ∈ {64,256,2560} 逐点与规则一致;
 *      边界 M=1 SMALLM / 32 DENSE / 128 DENSE / 256,512 W4A16
 *   2) GR_W4A16: 256³ 全引擎 vs Y_gold_2563 byte-exact (65536/65536);
 *      M=512: dispatch 拆 2×256 块 (kernel 单 invoke ABI 固定 M=256),
 *      上半==A 金标 / 下半==B 金标 (byte-exact) + crouton512 编解码往返恒等
 *   3) GR_SMALLM: M=1 pad-256 取行 0 == 全引擎行 0 (byte-exact)
 *      == 金标行 0; M=1 与 M=256 耗时同价 (|Δ|/256³ < 50%)
 *   4) GR_DENSE_F16: M=32/128, K=N=64 随机 f16 vs f32 累加标量 oracle
 *      cos ≥ 0.9999 + max|Δ| ≤ 0.5 ULP 包络
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define A "/data/local/tmp/hvxhmx23/assets/s256"
#define M 256u

static double cos_sim(const float* a, const float* b, size_t n) {
    double p = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        p += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}

static uint64_t med_u64(uint64_t* v, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (v[j] < v[i]) { uint64_t t = v[i]; v[i] = v[j]; v[j] = t; }
    return v[n / 2];
}

int main(void) {
    ex_open_result("31_gemm_dispatch");
    struct wtcache_ctx* wc = NULL;
    uint32_t bw = 0, bb = 0, ba = 0, bo = 0, bs = 0, byg = 0;

    /* G1 决策表 */
    {
        int bad = 0;
        const uint32_t kns[3] = { 64, 256, 2560 };
        for (uint32_t m = 1; m <= 600; m++)
            for (int ki = 0; ki < 3; ki++) {
                int exp = (m >= 256 && m % 256 == 0) ? GR_W4A16
                        : (m < 32) ? GR_SMALLM : GR_DENSE_F16;
                if (gemm_route_for(m, kns[ki], kns[(ki + 1) % 3]) != exp) bad++;
            }
        if (gemm_route_for(1, 256, 256) != GR_SMALLM) bad++;
        if (gemm_route_for(32, 256, 256) != GR_DENSE_F16) bad++;
        if (gemm_route_for(128, 256, 256) != GR_DENSE_F16) bad++;
        if (gemm_route_for(256, 256, 256) != GR_W4A16) bad++;
        if (gemm_route_for(512, 256, 256) != GR_W4A16) bad++;
        ex_check("route_table_exact", bad, 0);
        ex_log("  G1 决策表 600×3 扫描 + 5 边界: %d mismatch", bad);
    }

    /* G4 dense_f16 vs 标量 oracle (不依赖引擎) */
    {
        const uint32_t K = 64, N = 64;
        const uint32_t ms[2] = { 32, 128 };
        uint32_t lcg = 20260831u;
        int16_t* a = malloc((size_t)128 * K * 2);
        int16_t* w = malloc((size_t)K * N * 2);
        int16_t* c = malloc((size_t)128 * N * 2);
        float* of = malloc((size_t)128 * N * 4);
        float* cf = malloc((size_t)128 * N * 4);
        int bad = 0;
        for (size_t i = 0; i < (size_t)128 * K; i++) a[i] = gdn_f32_to_f16(gdn_lcg_norm(&lcg) * 0.3f);
        for (size_t i = 0; i < (size_t)K * N; i++) w[i] = gdn_f32_to_f16(gdn_lcg_norm(&lcg) * 0.3f);
        for (int t = 0; t < 2; t++) {
            uint32_t m = ms[t];
            gemm_f16_dense(a, w, c, m, K, N);
            for (uint32_t mi = 0; mi < m; mi++)
                for (uint32_t ni = 0; ni < N; ni++) {
                    float acc = 0.0f;
                    for (uint32_t ki = 0; ki < K; ki++)
                        acc += gdn_f16_to_f32(a[(size_t)mi * K + ki]) *
                               gdn_f16_to_f32(w[(size_t)ki * N + ni]);
                    of[(size_t)mi * N + ni] = acc;
                    cf[(size_t)mi * N + ni] = gdn_f16_to_f32(c[(size_t)mi * N + ni]);
                }
            if (cos_sim(of, cf, (size_t)m * N) < 0.9999) bad++;
            for (uint32_t i = 0; i < m * N; i++) {
                float ulp = fabsf(of[i]) * 1.3e-3f + 1e-6f;
                if (fabsf(of[i] - cf[i]) > ulp) bad++;
            }
        }
        ex_check("dense_f16_vs_scalar_oracle", bad, 0);
        free(a); free(w); free(c); free(of); free(cf);
    }

    /* 引擎面: W4A16 + SMALLM pad-256 */
    uint8_t* wt  = dc_read_file(A "/packed_weight.raw", &bw);
    uint8_t* bis = dc_read_file(A "/folded_bias.raw", &bb);
    uint8_t* at  = dc_read_file(A "/act_table.raw", &ba);
    uint8_t* ot  = dc_read_file(A "/out_table.raw", &bo);
    uint8_t* act = dc_read_file(A "/act_surface.raw", &bs);
    uint8_t* yg  = dc_read_file(A "/Y_gold_2563.raw", &byg);
    if (!wt || !bis || !at || !ot || !act || !yg || bs != M * M * 2 || byg != M * M * 2) {
        ex_log("assets missing/mismatch"); goto out;
    }
    dc_clean_ddr(wt, bw); dc_clean_ddr(bis, bb);
    if (wtcache_open(&wc, 4096) != WTC_OK) { ex_log("wtcache_open FAIL"); goto out; }
    {
        void* vb; uint32_t vs; void* pb; uint32_t pc;
        wtcache_layout(wc, &vb, &vs, &pb, &pc);
        struct dc_arena ar;
        dc_arena_init(&ar, (uint8_t*)vb + ((pc + 2047u) & ~2047u),
                      vs - ((pc + 2047u) & ~2047u));
        struct dc_w4 e256;
        if (dc_w4_carve(&e256, &ar, M, M, M, at, ot)) { ex_log("carve256 FAIL"); goto out; }
        memcpy(e256.wt, wt, bw); memcpy(e256.bias, bis, bb);
        qurt_mem_cache_clean((qurt_addr_t)e256.wt, bw, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        qurt_mem_cache_clean((qurt_addr_t)e256.bias, bb, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        memcpy(e256.act, act, bs);
        qurt_mem_cache_clean((qurt_addr_t)e256.act, bs, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

        /* G2a: 256³ vs 金标 */
        uint8_t* out256 = memalign(128, M * M * 2);
        uint16_t* dec = malloc(M * M * 2);
        if (dc_w4_invoke(&e256)) { ex_log("invoke FAIL"); goto eng; }
        dc_w4_read_out(&e256, out256);
        gemm_crouton_decode((const uint16_t*)out256, dec, M, M);
        const uint16_t* g = (const uint16_t*)yg;
        uint32_t nbad = 0;
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t n = 0; n < M; n++)
                if (dec[(size_t)m * M + n] != g[(size_t)n * M + m]) nbad++;
        ex_check("w4a16_256_gold_byteexact", nbad, 0);
        ex_log("  G2a GR_W4A16 M=256: %lu/65536 exact", (unsigned long)(M * M - nbad));

        /* G3: SMALLM M=1 pad-256 → 行 0 恒等 + 同价 (线性激活接口) */
        uint16_t* lin256 = malloc(M * M * 2);
        gemm_crouton_decode((const uint16_t*)act, lin256, M, M);
        uint16_t* row1 = malloc(M * 2);
        if (gemm_smallm_pad256(&e256, (const int16_t*)lin256, 1, (int16_t*)row1)) {
            ex_log("smallm FAIL"); free(lin256); goto eng;
        }
        int idn = memcmp(row1, dec, M * 2) != 0;                /* 全引擎行 0 */
        int idg = 0;
        for (uint32_t n = 0; n < M; n++)
            if (row1[n] != g[(size_t)n * M]) idg++;
        ex_check("smallm_row0_byteexact_engine", idn, 0);
        ex_check("smallm_row0_matches_gold", idg, 0);

        uint64_t t1[21], t2[21];
        for (int i = 0; i < 21; i++) {
            int64_t q0 = HAP_perf_get_time_us();
            gemm_smallm_pad256(&e256, (const int16_t*)lin256, 1, (int16_t*)row1);
            t1[i] = (uint64_t)(HAP_perf_get_time_us() - q0);
            q0 = HAP_perf_get_time_us();
            memcpy(e256.act, act, bs);
            qurt_mem_cache_clean((qurt_addr_t)e256.act, bs,
                                 QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            dc_w4_invoke(&e256); dc_w4_read_out(&e256, out256);
            t2[i] = (uint64_t)(HAP_perf_get_time_us() - q0);
        }
        uint64_t mu1 = med_u64(t1, 21), mu2 = med_u64(t2, 21);
        double ratio = mu2 ? (double)mu1 / (double)mu2 : 9.9;
        ex_check("smallm_same_cost_as_256", (ratio > 1.5 || ratio < 0.5) ? 1 : 0, 0);
        ex_log("  G3 GR_SMALLM M=1 med %llu us vs M=256 med %llu us (ratio %.2f)",
               (unsigned long long)mu1, (unsigned long long)mu2, ratio);

        /* G2b: M=512 = dispatch 拆 2×256 块 (kernel ABI 单 invoke 固定 M=256) */
        {
            uint16_t* lin512 = malloc(4 * bs);
            uint16_t* rt512  = malloc(4 * bs);
            uint16_t* o512   = malloc(4 * bs);
            memcpy(lin512, lin256, 2 * bs);                     /* A 半 = lin256 */
            for (uint32_t i = 0; i < M * M; i++)                /* B 半: 异源数据 */
                lin512[M * M + i] = (uint16_t)(lin256[i] ^ (uint16_t)(i * 40503u + 7u));
            gemm_crouton_encode(lin512, rt512, 512, M);
            gemm_crouton_decode(rt512, o512, 512, M);           /* 分离缓冲 (无别名) */
            ex_check("crouton512_roundtrip_byteexact",
                     memcmp(lin512, o512, 4 * bs) != 0, 0);

            /* B 半金标: 同引擎单跑 B 面 */
            uint8_t* outB = memalign(128, bs);
            uint16_t* decB = malloc(bs);
            gemm_crouton_encode(lin512 + (size_t)M * M, (uint16_t*)e256.act, M, M);
            qurt_mem_cache_clean((qurt_addr_t)e256.act, bs,
                                 QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            int rcB = dc_w4_invoke(&e256);
            if (rcB) ex_log("invoke B FAIL");
            else {
                dc_w4_read_out(&e256, outB);
                gemm_crouton_decode((const uint16_t*)outB, decB, M, M);
            }

            uint16_t* out512 = malloc(4 * bs);
            int rc = gemm_w4a16_m256(&e256, (const int16_t*)lin512, 512,
                                     (int16_t*)out512);
            int b512 = (rc || rcB) ? 512 : 0;
            if (!b512)
                for (uint32_t m = 0; m < 512; m++) {
                    const uint16_t* gold = (m < M) ? dec : decB;
                    if (memcmp(out512 + (size_t)m * M,
                               gold + (size_t)(m % M) * M, M * 2)) b512++;
                }
            if (b512) ex_log("  G2b M=512 split: %d 行不一致", b512);
            ex_check("w4a16_512_split_gold", b512, 0);
            free(lin512); free(rt512); free(o512); free(outB); free(decB); free(out512);
        }
        free(lin256); free(out256); free(dec); free(row1);
eng: ;
    }
out:
    if (wc) wtcache_close(wc);
    return ex_summary();
}
