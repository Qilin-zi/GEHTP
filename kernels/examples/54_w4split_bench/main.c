/*
 * 54_w4split_bench — 018 双域 N 维分片 W4TS 单点 benchmark
 * =====================================================================
 * 裁决问题: W4TS (exec_matmul_w4_stream) 是 MLP/latency-bound, 双核 N 维
 * 分片应 ~2× (每核各算一半列, 各跑一套 DMA ring, 翻倍 MLP)。本 .so 跑
 * 单 op OP_MATMUL_F16_SPLIT (35) blob, 复用 executor 原生 n0/N_chunk 分片。
 *
 * 用法 (host 编排):
 *   ./run_main_on_hexagon 3 test_54.so 16384 full    # 单域基线 (N=16384)
 *   ./run_main_on_hexagon 3 test_54.so 16384 c0 &    # 双域 (dom3 左半)
 *   ./run_main_on_hexagon 4 test_54.so 16384 c1 &    #       (dom4 右半)
 *
 * argv: [0]=N_full [1]=role(full|c0|c1) [2]=skip_gold(0|1)
 * 判据: ① 引擎门 engine==1 (HVX 无静默回落)
 *       ② 数值门 cos(dev 列区间, gold 列区间) ≥ 0.9999 (skip_gold 跳过)
 *       ③ 报告: dur med/min/max (us)
 * 输出: result_54_w4split_n<N>_<role>.txt
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <HAP_perf.h>
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include "dc_parts.h"

#define D    "/data/local/tmp/hvxhmx23/g54"
#define K    1024u
#define WARM 2
#define REPS 8

static uint8_t* read_file(const char* p, size_t* out_len) {
    FILE* f = fopen(p, "rb");
    if (!f) { ex_log("open %s FAIL", p); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t* buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t sh = (uint32_t)(14 - e);
        uint32_t half = m >> (sh + 1);
        uint32_t rest = m & ((1u << sh) - 1u);
        if ((m >> sh) & 1u) half += (rest != 0u) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
    if ((m >> 12) & 1u) h += (((m & 0xfffu) || (h & 1u)) ? 1u : 0u);
    return h;
}
static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (!e && !m) bits = s;
    else if (e == 0x1Fu) bits = s | 0x7F800000u | (m << 13);
    else if (!e) { int8_t sh = 1; while (!(m & 0x400u)) { m <<= 1; sh++; }
        bits = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | ((m & 0x3FFu) << 13); }
    else bits = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* 与 pack_w4split.cc 逐位同 */
static void gen_act(uint16_t* a, uint32_t k_) {
    for (uint32_t k = 0; k < k_; k++)
        a[k] = f32_to_f16((float)((int)(k % 29u) - 14) * 0.25f);
}
static void gen_w_q4(uint8_t* w, uint32_t k_, uint32_t n_) {
    uint32_t n_dot = k_ / 32u, n_ct = n_ / 32u;
    for (uint32_t ct = 0; ct < n_ct; ct++)
        for (uint32_t kt = 0; kt < n_dot; kt++) {
            uint8_t* tile = w + ((size_t)ct * n_dot + kt) * 640u;
            for (uint32_t i = 0; i < 16u; i++)
                for (uint32_t r = 0; r < 32u; r++)
                    tile[i * 32u + r] = (uint8_t)(ct * 31u + kt * 17u + i * 7u + r * 11u + i * r);
            for (uint32_t r = 0; r < 32u; r++) {
                uint16_t d = f32_to_f16(0.02f + (float)((ct + r) % 7u) * 0.005f);
                tile[512u + 2u * r] = (uint8_t)(d & 0xffu);
                tile[512u + 2u * r + 1u] = (uint8_t)(d >> 8);
            }
            for (uint32_t p = 512u + 64u; p < 640u; p++) tile[p] = 0;
        }
}

/* f32 金标 (列区间 [n0, n0+N_chunk)): y[nn] = Σ_kt d_w × Σ_j (v-8)×af */
static void gold_f32_range(const uint16_t* a, const uint8_t* w, float* y,
                           uint32_t k_, uint32_t n0, uint32_t n_chunk) {
    uint32_t n_dot = k_ / 32u;
    for (uint32_t nn = 0; nn < n_chunk; nn++) {
        uint32_t n = n0 + nn;
        uint32_t ct = n / 32u, r = n % 32u;
        float acc = 0.f;
        for (uint32_t kt = 0; kt < n_dot; kt++) {
            const uint8_t* tile = w + ((size_t)ct * n_dot + kt) * 640u;
            const uint16_t* drow = (const uint16_t*)(const void*)(tile + 512u);
            float d = f16_to_f32(drow[r]);
            for (uint32_t i = 0; i < 16u; i++) {
                uint8_t b = tile[i * 32u + r];
                uint32_t k0 = kt * 32u + 2u * i, k1 = k0 + 1u;
                acc += f16_to_f32(a[k0]) * d * (float)((int)(b & 0xFu) - 8);
                acc += f16_to_f32(a[k1]) * d * (float)((int)(b >> 4) - 8);
            }
        }
        y[nn] = acc;
    }
}

static double cos_f16_f32(const uint16_t* a, const float* b, uint32_t n) {
    double p = 0, na = 0, nb = 0;
    for (uint32_t i = 0; i < n; i++) {
        double av = (double)f16_to_f32(a[i]);
        p += av * (double)b[i]; na += av * av; nb += (double)b[i] * b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}

static uint64_t med_u64(uint64_t* a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[n / 2];
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: <N_full> <role> [skip_gold]\n"); return 0xFE; }
    uint32_t N_full = (uint32_t)strtoul(argv[1], NULL, 10);
    const char* role = argv[2];
    int skip_gold = (argc > 3) ? atoi(argv[3]) : 0;

    uint32_t n0 = 0u, N_chunk = N_full;
    if (!strcmp(role, "c0")) { N_chunk = N_full / 2u; }
    else if (!strcmp(role, "c1")) { n0 = N_full / 2u; N_chunk = N_full / 2u; }
    else if (strcmp(role, "full") != 0) { fprintf(stderr, "bad role %s\n", role); return 0xFE; }

    char rname[64];
    snprintf(rname, sizeof(rname), "54_w4split_n%u_%s", N_full, role);
    ex_open_result(rname);
    ex_log("=== 54_w4split_bench N_full=%u role=%s n0=%u chunk=%u ===", N_full, role, n0, N_chunk);

    char blob[160];
    snprintf(blob, sizeof(blob), "%s/split_n%u_%s.wtop", D, N_full, role);

    size_t blen = 0;
    uint8_t* b = read_file(blob, &blen);
    if (!b) { ex_log("read FAIL %s", blob); return ex_summary(); }
    dc_clean_ddr(b, (uint32_t)blen);
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(b, blen, w) != WT_OK) { ex_log("wt_parse FAIL %s", blob); free(b); free(w); return ex_summary(); }
    struct wt_op_ts* ts = calloc(w->n_ops, sizeof(*ts));
    wt_exec_set_ts(ts, w->n_ops);
    /* 018 566µs 解剖: 关 per-op trace。g_trace_on 默认 1, 每 op 两条
     * fprintf+fflush 落 optrace.txt (pre+post 各 ~500µs)。真 runner(42)
     * 默认关, 本探针须同关方测到真实固定开销而非 trace 落盘伪开销。 */
    wt_exec_set_trace(0);

    uint64_t dur[REPS];
    char err[128];
    int rc = 0;
    for (int rep = 0; rep < WARM + REPS; rep++) {
        uint32_t em = 0;
        rc = wt_exec_run(w, &em, NULL, err, sizeof(err));
        if (rc) { ex_log("wt_exec_run FAIL rc=%d: %s", rc, err); break; }
        if (rep >= WARM) dur[rep - WARM] = ts[0].dur_us;
    }
    uint64_t dmin = 0, dmax = 0, dmed = 0;
    uint32_t engine = 0;
    if (!rc) {
        engine = ts[0].engine;
        dmin = dmax = dur[0];
        for (int i = 1; i < REPS; i++) {
            if (dur[i] < dmin) dmin = dur[i];
            if (dur[i] > dmax) dmax = dur[i];
        }
        dmed = med_u64(dur, REPS);
        ex_log("dur_med=%lluus min=%llu max=%llu engine=%u",
               (unsigned long long)dmed, (unsigned long long)dmin,
               (unsigned long long)dmax, engine);
    }

    /* 数值门: cos(dev 列区间 [n0,n0+chunk), gold) */
    if (!rc && !skip_gold) {
        uint8_t* t0p = (uint8_t*)wt_exec_temp(0);
        uint32_t want = N_full * 2u;
        if (!t0p || wt_exec_temp_bytes(0) != want)
            ex_log("[warn] temp0 bytes=%lu want=%u", (unsigned long)wt_exec_temp_bytes(0), want);
        uint16_t* dev = (uint16_t*)(void*)t0p + n0;   /* 列区间 [n0, n0+chunk) */
        float* gold = malloc((size_t)N_chunk * 4u);
        uint16_t* a = malloc((size_t)K * 2u);
        uint8_t* wq = malloc((size_t)(K / 32u) * (N_full / 32u) * 640u);
        gen_act(a, K);
        gen_w_q4(wq, K, N_full);
        gold_f32_range(a, wq, gold, K, n0, N_chunk);
        double c = cos_f16_f32(dev, gold, N_chunk);
        ex_log("cos_gold=%.6f (cols [%u,%u))", c, n0, n0 + N_chunk);
        ex_check("cos_gold", c < 0.9999 ? 1 : 0, 0);
        free(gold); free(a); free(wq);
    }
    ex_check("engine_hvx", engine != 1u ? 1 : 0, 0);
    wt_exec_shutdown();
    free(ts); free(w); free(b);
    return ex_summary();
}
