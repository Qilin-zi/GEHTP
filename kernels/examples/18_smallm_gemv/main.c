/*
 * 18_smallm_gemv — U8 小-M GEMV pad-256 路线 设备验证
 * =====================================================================
 * 判据 (MODULE A A1 结论的单元化复证, 资产 host/gen_smallm_assets.py;
 * 输出面是 crouton16_row4 surface — 行偏移在 surface 布局下无意义,
 * 必须先解码成线性 (M,N) 再按行比对):
 *   1) M=1 pad 行0 == full-256 (v0) 行0, byte-exact (同一 kernel tile 走法)
 *   2) M=16 pad 行0..15 == full-256 行0..15, byte-exact
 *   3) 两次 M=1 (v0/v1 行0): pad 行 1..255 byte-equal (pad 不变性), 行0 不同
 *   4) M=1 与 M=256 invoke 同价 (tile-walk bound, 成本与 M 无关)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v22.h"
#include "example_util.h"

#define A    "/data/local/tmp/hvxhmx23/assets/s256"
#define SM   "/data/local/tmp/hvxhmx23/assets/smallm"
#define M 256u
#define K 256u
#define N 256u
#define TRIALS 50
#define ROW_B (N * 2u)

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

/* crouton16_row4 面 (M,cols) → row-major (与例 17/21 同一布局镜像) */
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

int main(void) {
    ex_open_result("18_smallm_gemv");
    struct wtcache_ctx* wc = NULL;
    uint32_t bw = 0, bb = 0, ba = 0, bo = 0, b0 = 0, b1 = 0, b16 = 0, bs = 0;
    uint8_t *full = NULL, *p1v0 = NULL, *p1v1 = NULL, *p16 = NULL;
    uint16_t *dfl = NULL, *dv0 = NULL, *dv1 = NULL, *dv16 = NULL;

    uint8_t* wt  = dc_read_file(A "/packed_weight.raw", &bw);
    uint8_t* bis = dc_read_file(A "/folded_bias.raw", &bb);
    uint8_t* at  = dc_read_file(A "/act_table.raw", &ba);
    uint8_t* ot  = dc_read_file(A "/out_table.raw", &bo);
    uint8_t* av0 = dc_read_file(SM "/act_p1_v0.raw", &b0);
    uint8_t* av1 = dc_read_file(SM "/act_p1_v1.raw", &b1);
    uint8_t* a16 = dc_read_file(SM "/act_p16_v0.raw", &b16);
    uint8_t* afl = dc_read_file(A "/act_variants/v0.raw", &bs);
    if (!wt || !bis || !at || !ot || !av0 || !av1 || !a16 || !afl ||
        b0 != M * K * 2 || b16 != M * K * 2 || bs != M * K * 2) {
        ex_log("assets missing/mismatch"); goto out;
    }
    dc_clean_ddr(wt, bw); dc_clean_ddr(bis, bb); dc_clean_ddr(av0, b0);
    dc_clean_ddr(av1, b1); dc_clean_ddr(a16, b16); dc_clean_ddr(afl, bs);

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

        full = memalign(128, M * N * 2);
        p1v0 = memalign(128, M * N * 2);
        p1v1 = memalign(128, M * N * 2);
        p16  = memalign(128, M * N * 2);
        dfl = malloc(M * N * 2); dv0 = malloc(M * N * 2);
        dv1 = malloc(M * N * 2); dv16 = malloc(M * N * 2);
        if (!dfl || !dv0 || !dv1 || !dv16) { ex_log("alloc FAIL"); goto out; }

        uint64_t tm[TRIALS];
        struct { const uint8_t* act; uint8_t* out; uint16_t* dec; } runs[4] = {
            { afl, full, dfl }, { av0, p1v0, dv0 }, { av1, p1v1, dv1 }, { a16, p16, dv16 },
        };
        for (int i = 0; i < 4; i++) {
            cpu_to_vtcm(e.act, runs[i].act, bs);
            int64_t t0 = HAP_perf_get_time_us();
            if (dc_w4_invoke(&e)) { ex_log("invoke %d FAIL", i); goto out; }
            tm[i] = (uint64_t)(HAP_perf_get_time_us() - t0);
            dc_w4_read_out(&e, runs[i].out);
            minv_crouton((const uint16_t*)runs[i].out, runs[i].dec, M, N);
        }
        const uint16_t* r0f = dfl;      /* 行 m = dfl + m*N */
        ex_check("p1_row0_eq_full_row0",
                 memcmp(dv0, r0f, ROW_B) == 0 ? 0 : 1, 0);
        ex_check("p16_rows0_15_eq_full",
                 memcmp(dv16, r0f, 16 * ROW_B) == 0 ? 0 : 1, 0);
        ex_check("pad_rows_invariant",
                 memcmp(dv0 + N, dv1 + N, (M - 1) * N * 2) == 0 ? 0 : 1, 0);
        ex_check("row0_tracks_input", memcmp(dv0, dv1, ROW_B) != 0 ? 0 : 1, 0);

        /* 计时: M=256 full vs M=1 pad (同一引擎面, 只换 act) */
        uint64_t tf[TRIALS], t1[TRIALS];
        for (int i = 0; i < TRIALS; i++) {
            int64_t t0 = HAP_perf_get_time_us();
            dc_w4_invoke(&e);
            tf[i] = (uint64_t)(HAP_perf_get_time_us() - t0);
        }
        cpu_to_vtcm(e.act, av0, bs);
        for (int i = 0; i < TRIALS; i++) {
            int64_t t0 = HAP_perf_get_time_us();
            dc_w4_invoke(&e);
            t1[i] = (uint64_t)(HAP_perf_get_time_us() - t0);
        }
        uint64_t mf = med_u64(tf, TRIALS), m1 = med_u64(t1, TRIALS);
        ex_log("M=256 med %llu us vs M=1(pad) med %llu us (delta %.2f%%)",
               (unsigned long long)mf, (unsigned long long)m1,
               mf ? 100.0 * (double)(m1 > mf ? m1 - mf : mf - m1) / mf : 0.0);
        ex_check("cost_M_invariant", mf && m1 <= mf + (mf >> 4) ? 0 : 1, 0);
    }

out:
    free(full); free(p1v0); free(p1v1); free(p16);
    free(dfl); free(dv0); free(dv1); free(dv16);
    if (wc) wtcache_close(wc);
    return ex_summary();
}
