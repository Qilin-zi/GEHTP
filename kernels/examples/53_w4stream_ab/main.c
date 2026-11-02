/*
 * 53_w4stream_ab — 018 W4TS 生死门: DMA 流式 + rmpy matvec vs 基线 q4_0_gemv
 * =====================================================================
 * 两臂 (pack_w4stream_ab 组包, 同确定性权重/激活, 仅 flags 差 bit5):
 *   A: flags=2      → exec_matmul_q4_0_gemv (f16 激活, HVX l2fetch 散读 DDR)
 *   B: flags=2|0x20 → exec_matmul_w4_stream (q8 激活, DMA 菊花链 + rmpy)
 * 形状: K=1024/N=2048 (64 ct) + K=1024/N=16384 (512 ct 长链)。
 *
 * 判据 (specs/018 G1):
 *   ① 引擎门: 两臂 engine==1 (HVX, 无静默回落)
 *   ② 数值门: cos(arm, gold) ≥ 0.9999 (gold = f32 全精度参考)
 *   ③ 生死门: B 臂 eff_bw ≥ 30 GB/s (Q4 tile 字节/dur)
 *   ④ 报告: dur med/min/max + wall + eff_bw + B/A 加速比
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

#define D    "/data/local/tmp/hvxhmx23/g53"
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

/* 与 pack_w4stream_ab.cc 逐位同 */
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

/* f32 金标: y[n] = Σ_kt d_w × Σ_j (v-8)×af (f16 激活精确, 设备标量一次) */
static void gold_f32(const uint16_t* a, const uint8_t* w, float* y,
                     uint32_t k_, uint32_t n_) {
    uint32_t n_dot = k_ / 32u;
    for (uint32_t n = 0; n < n_; n++) {
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
        y[n] = acc;
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

struct arm {
    const char* name, *blob;
    uint32_t N;
    uint64_t dur_med, dur_min, dur_max;
    uint32_t engine;
    uint8_t* out;
};

static uint64_t med_u64(uint64_t* a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[n / 2];
}

static int run_arm(struct arm* A) {
    size_t blen = 0;
    uint8_t* blob = read_file(A->blob, &blen);
    if (!blob) return 1;
    dc_clean_ddr(blob, (uint32_t)blen);
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(blob, blen, w) != WT_OK) { ex_log("wt_parse FAIL %s", A->blob); free(blob); free(w); return 1; }
    struct wt_op_ts* ts = calloc(w->n_ops, sizeof(*ts));
    wt_exec_set_ts(ts, w->n_ops);

    uint64_t dur[REPS];
    char err[128];
    int rc = 0;
    for (int rep = 0; rep < WARM + REPS; rep++) {
        uint32_t em = 0;
        rc = wt_exec_run(w, &em, NULL, err, sizeof(err));
        if (rc) { ex_log("wt_exec_run FAIL rc=%d: %s", rc, err); break; }
        if (rep >= WARM) dur[rep - WARM] = ts[0].dur_us;
    }
    if (!rc) {
        uint8_t* t0p = (uint8_t*)wt_exec_temp(0);
        uint32_t want = A->N * 2u;
        if (!t0p || wt_exec_temp_bytes(0) != want)
            ex_log("[warn] temp0 bytes=%lu want=%u", (unsigned long)wt_exec_temp_bytes(0), want);
        memcpy(A->out, t0p, want);
        A->engine = ts[0].engine;
        A->dur_min = A->dur_max = dur[0];
        for (int i = 1; i < REPS; i++) {
            if (dur[i] < A->dur_min) A->dur_min = dur[i];
            if (dur[i] > A->dur_max) A->dur_max = dur[i];
        }
        A->dur_med = med_u64(dur, REPS);
    }
    wt_exec_shutdown();
    free(ts); free(w); free(blob);
    return rc ? 1 : 0;
}

int main(void) {
    ex_open_result("53_w4stream_ab");
    ex_log("=== 53_w4stream_ab (018 W4TS 生死门: DMA 流式 rmpy vs q4_0_gemv) ===");

    const uint32_t Ns[2] = {2048u, 16384u};
    static uint16_t a1[K];
    gen_act(a1, K);

    static uint8_t out_a[16384u * 2], out_b[16384u * 2];
    static float gold[16384u];
    int fail = 0;

    for (int s = 0; s < 2; s++) {
        uint32_t N = Ns[s];
        uint32_t w_bytes = (K / 32u) * (N / 32u) * 640u;
        uint8_t* wq4 = malloc(w_bytes);
        gen_w_q4(wq4, K, N);
        gold_f32(a1, wq4, gold, K, N);

        char pa[160], pb[160];
        snprintf(pa, sizeof(pa), "%s/blob_w4s_k1024n%lu_base.wtop", D, N);
        snprintf(pb, sizeof(pb), "%s/blob_w4s_k1024n%lu_stream.wtop", D, N);
        struct arm A = {"base",     pa, N, 0,0,0,0, out_a};
        struct arm B = {"w4stream", pb, N, 0,0,0,0, out_b};
        if (run_arm(&A)) { ex_log("[FAIL] arm base N=%u", N); free(wq4); fail = 1; continue; }
        if (run_arm(&B)) { ex_log("[FAIL] arm stream N=%u", N); free(wq4); fail = 1; continue; }

        double wmb = (double)w_bytes / (1024.0 * 1024.0);
        double bw_a = A.dur_med ? wmb / ((double)A.dur_med / 1e6) / 1024.0 : 0;
        double bw_b = B.dur_med ? wmb / ((double)B.dur_med / 1e6) / 1024.0 : 0;
        double cos_a = cos_f16_f32((const uint16_t*)out_a, gold, N);
        double cos_b = cos_f16_f32((const uint16_t*)out_b, gold, N);
        double cos_ab;
        {
            double p = 0, na = 0, nb = 0;
            for (uint32_t i = 0; i < N; i++) {
                double av = f16_to_f32(((const uint16_t*)out_a)[i]);
                double bv = f16_to_f32(((const uint16_t*)out_b)[i]);
                p += av * bv; na += av * av; nb += bv * bv;
            }
            cos_ab = p / (sqrt(na) * sqrt(nb) + 1e-30);
        }
        ex_log("[N=%lu] base:    %lluus (min %llu max %llu) %6.2f GB/s cos_gold=%.6f engine=%u",
               N, (unsigned long long)A.dur_med, (unsigned long long)A.dur_min,
               (unsigned long long)A.dur_max, bw_a, cos_a, A.engine);
        ex_log("[N=%lu] w4stream:%lluus (min %llu max %llu) %6.2f GB/s cos_gold=%.6f cos_vs_base=%.6f engine=%u speedup=%.2fx",
               N, (unsigned long long)B.dur_med, (unsigned long long)B.dur_min,
               (unsigned long long)B.dur_max, bw_b, cos_b, cos_ab, B.engine,
               (A.dur_med && B.dur_med) ? (double)A.dur_med / (double)B.dur_med : 0.0);

        char nm[64];
        snprintf(nm, sizeof(nm), "N%lu_engine_base_hvx", N);
        ex_check(nm, A.engine != 1u, 0);
        snprintf(nm, sizeof(nm), "N%lu_engine_stream_hvx", N);
        ex_check(nm, B.engine != 1u, 0);
        snprintf(nm, sizeof(nm), "N%lu_cos_base", N);
        ex_check(nm, cos_a < 0.9999, 0);
        snprintf(nm, sizeof(nm), "N%lu_cos_stream", N);
        ex_check(nm, cos_b < 0.9999, 0);
        snprintf(nm, sizeof(nm), "N%lu_bw_stream_ge30", N);
        ex_check(nm, bw_b < 30.0, 0);
        free(wq4);
    }
    ex_summary();
    return fail ? 1 : 0;
}
