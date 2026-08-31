/*
 * 17_w4a16_gemm — U2+U4 W4A16 HMX 引擎 设备验证 (256³ 正确性 + 性能)
 * =====================================================================
 * 判据 (输出面是 crouton16_row4 surface, 必须先逆布局再比 — 同 t10-a):
 *   1) act_surface.raw (金标对应的基础激活, 非 act_variants/v0 随机变体!) 激活,
 *      解码后 vs oracle/Y_gold_2563.raw (W4A16 256³ 闭合位恒等金标, (N,M) 线性)
 *      byte-exact — 65536/65536
 *   2) 复跑 byte-exact (确定性, surface 级)
 *   3) 100 次中位 invoke 计时 → GFLOPS
 * 注: act_variants/Y_ref_v0.raw 是 v0 随机变体的 int8 全精度标量金标, 与 w4 引擎
 * 无对应关系 (t10 闭合从未拿它做门) — 不作判据。K2560 金标 (37 LSB) 在 21 用例。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v22.h"
#include "example_util.h"

#define A "/data/local/tmp/hvxhmx23/assets/s256"
#define M 256u
#define K 256u
#define N 256u
#define TRIALS 100

static uint8_t* rd(const char* p, uint32_t* b) { return dc_read_file(p, b); }

static void cpu_to_vtcm(uint8_t* d, const uint8_t* s, uint32_t n) {
    memcpy(d, s, n);
    qurt_mem_cache_clean((qurt_addr_t)d, n, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

static uint64_t med_u64(uint64_t* a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[n / 2];
}

/* crouton16_row4 面 (M,cols) → row-major (与例 21/模块 pack 同一布局镜像) */
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

int main(int argc, char** argv) {
    ex_open_result("17_w4a16_gemm");
    struct wtcache_ctx* wc = NULL;
    uint32_t bw = 0, bb = 0, ba = 0, bo = 0, bs = 0, byg = 0;
    uint8_t *out1 = NULL, *out2 = NULL, *ygold = NULL;
    uint16_t* dec = NULL;

    uint8_t* wt  = rd(A "/packed_weight.raw", &bw);
    uint8_t* bis = rd(A "/folded_bias.raw", &bb);
    uint8_t* at  = rd(A "/act_table.raw", &ba);
    uint8_t* ot  = rd(A "/out_table.raw", &bo);
    uint8_t* act = rd(A "/act_surface.raw", &bs);   /* 金标对应的基础激活 */
    ygold = rd(A "/Y_gold_2563.raw", &byg);
    if (!wt || !bis || !at || !ot || !act || !ygold ||
        bw != K * N / 2 || bs != M * K * 2 || byg != M * N * 2) {
        ex_log("assets missing/mismatch (wt=%p bis=%p at=%p ot=%p act=%p gold=%p)",
               wt, bis, at, ot, act, ygold);
        goto out;
    }
    dc_clean_ddr(wt, bw); dc_clean_ddr(bis, bb);   /* 铁律① */

    if (wtcache_open(&wc, 4096) != WTC_OK) { ex_log("wtcache_open FAIL"); goto out; }
    {
        void* vb; uint32_t vs; void* pb; uint32_t pc;
        wtcache_layout(wc, &vb, &vs, &pb, &pc);
        struct dc_arena ar;
        dc_arena_init(&ar, (uint8_t*)vb + ((pc + 2047u) & ~2047u),
                      vs - ((pc + 2047u) & ~2047u));
        struct dc_w4 e;
        if (dc_w4_carve(&e, &ar, M, K, N, at, ot)) { ex_log("carve FAIL"); goto out; }
        cpu_to_vtcm(e.wt, wt, bw);
        cpu_to_vtcm(e.bias, bis, bb);
        cpu_to_vtcm(e.act, act, bs);

        out1 = memalign(128, M * N * 2);
        out2 = memalign(128, M * N * 2);
        dec  = malloc(M * N * 2);
        if (!dec) { ex_log("alloc FAIL"); goto out; }
        if (dc_w4_invoke(&e)) { ex_log("invoke FAIL"); goto out; }
        dc_w4_read_out(&e, out1);

        /* 输出面 crouton 布局 → 线性 (M,N); 金标是 (N,M) 线性 */
        minv_crouton((const uint16_t*)out1, dec, M, N);
        const uint16_t* g = (const uint16_t*)ygold;
        uint32_t nbad = 0; uint32_t first_bad = 0xFFFFFFFFu;
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t n = 0; n < N; n++)
                if (dec[(size_t)m * N + n] != g[(size_t)n * M + m]) {
                    nbad++;
                    if (first_bad == 0xFFFFFFFFu) first_bad = m * N + n;
                }
        ex_log("  vs Y_gold_2563: %lu/%lu exact", (unsigned long)(M * N - nbad),
               (unsigned long)(M * N));
        if (nbad) ex_log("  first_bad @ idx %lu", (unsigned long)first_bad);
        ex_check("vs_gold2563_byteexact", nbad == 0 ? 0 : 1, 0);

        int rc2 = dc_w4_invoke(&e);
        dc_w4_read_out(&e, out2);
        ex_check("rerun_byteexact", (rc2 == 0 && memcmp(out1, out2, M * N * 2) == 0) ? 0 : 1, 0);

        uint64_t t[TRIALS];
        for (int i = 0; i < TRIALS; i++) {
            int64_t t0 = HAP_perf_get_time_us();
            dc_w4_invoke(&e);
            t[i] = (uint64_t)(HAP_perf_get_time_us() - t0);
        }
        uint64_t mu = med_u64(t, TRIALS);
        double gflops = 2.0 * M * K * N / (mu ? (double)mu : 1.0) / 1000.0;
        ex_log("invoke med %llu us (%.1f us 最小), 256^3 W4A16 = %.2f GFLOPS",
               (unsigned long long)mu, (double)t[0], gflops);
        ex_check("invoke_sanity_us", mu > 0 && mu < 100000 ? 0 : 1, 0);
    }

out:
    free(out1); free(out2); free(ygold); free(dec);
    if (wc) wtcache_close(wc);
    (void)argc; (void)argv;
    return ex_summary();
}
