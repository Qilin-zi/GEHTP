/*
 * 50_q6b_gemv_ab — M=1 GEMV 三臂引擎 A/B (VTCM/tiling 规划裁决)
 * =====================================================================
 * 背景: 0.6B decode op_ts 91.5% wall 在 253 个 M=1 t1 FC (HVX ~2GB/s,
 * 标定 UserDRAM 59.9GB/s)。候选药方 = 编译期 crouton 布局 + HMX + 线性
 * DMA 驻场 (exec 层已预留 w_pp=flags bit3 路径, 008 T012)。
 *
 * 三臂 (host pack_gemv_ab.cc 生成, 同数据同形状 K=1024/N=2048/W=4MB):
 *   A blob_gemv_m1.wtop    M=1  flags=2   → 现状 M<32 HVX t1 gemv
 *   B blob_gemv_m32.wtop   M=32 flags=2   → HMX + 运行时标量散 pack
 *   C blob_gemv_m32pp.wtop M=32 flags=10  → HMX + 预打包线性 DMA (规划终态)
 *
 * 判据:
 *   ① 引擎门: A=HVX(1), B/C=HMX(2) (ts.engine 实测, 排除静默回落)
 *   ② 数值门: 三臂输出行 0 == 设备标量 f32 金标, byte-exact
 *      (整数值取值, f32 精确累加, 跨引擎逐位可比)
 *   ③ pad 门: B/C 行 1..31 全零 (0·W=0, M-pad 不扰行 0)
 *   ④ 性能: warm 2 + 计时 8 取中位 (绝不取 min), ts.dur_us 与外层 wall 双报
 *      + 有效带宽 (W=4MB) + C/A、B/A 比
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include "dc_parts.h"   /* dc_clean_ddr (blob DMA bypass 源) */

#define D    "/data/local/tmp/hvxhmx23/g50"
#define K    1024u
#define N    2048u
#define M32  32u
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
        if ((m >> sh) & 1u) half += ((m & ((1u << sh) - 1u)) || (half & 1u)) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint16_t half = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
    if ((m >> 12) & 1u) half += (uint16_t)(((m & 0xfffu) || (half & 1u)) ? 1u : 0u);
    return half;
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (!e && !m) bits = s;
    else if (e == 0x1Fu) bits = s | 0x7F800000u | (m << 13);
    else if (!e) { /* subnormal */ int8_t sh = 1; while (!(m & 0x400u)) { m <<= 1; sh++; }
        bits = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | ((m & 0x3FFu) << 13); }
    else bits = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

struct arm {
    const char* name, *blob;
    uint32_t M;
    uint64_t dur_med, dur_min, dur_max, wall_med;
    uint32_t engine;
    uint8_t* out;          /* M*N*2 快照 (最后一跑后) */
};

static uint64_t med_u64(uint64_t* a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[n / 2];
}

/* 单臂: parse → warm+计时 → 快照 → shutdown。返回 0 成功 */
static int run_arm(struct arm* A) {
    size_t blen = 0;
    uint8_t* blob = read_file(A->blob, &blen);
    if (!blob) return 1;
    dc_clean_ddr(blob, (uint32_t)blen);
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(blob, blen, w) != WT_OK) { ex_log("wt_parse FAIL %s", A->blob); free(blob); free(w); return 1; }
    struct wt_op_ts* ts = calloc(w->n_ops, sizeof(*ts));
    wt_exec_set_ts(ts, w->n_ops);

    uint64_t dur[REPS], wall[REPS];
    char err[128];
    int rc = 0;
    for (int rep = 0; rep < WARM + REPS; rep++) {
        int64_t t0 = HAP_perf_get_time_us();
        uint32_t em = 0;
        rc = wt_exec_run(w, &em, NULL, err, sizeof(err));
        int64_t us = HAP_perf_get_time_us() - t0;
        if (rc) { ex_log("wt_exec_run FAIL rc=%d: %s", rc, err); break; }
        if (rep >= WARM) { dur[rep - WARM] = ts[0].dur_us; wall[rep - WARM] = (uint64_t)us; }
    }
    if (!rc) {
        uint8_t* t0p = (uint8_t*)wt_exec_temp(0);
        uint32_t want = A->M * N * 2u;
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
        A->wall_med = med_u64(wall, REPS);
    }
    wt_exec_shutdown();
    free(ts); free(w); free(blob);
    return rc ? 1 : 0;
}

int main(void) {
    ex_open_result("50_q6b_gemv_ab");
    ex_log("=== 50_q6b_gemv_ab (M=1 GEMV 三臂: HVX 现状 vs HMX 散pack vs HMX 预打包DMA) ===");

    /* 设备标量金标: 与 pack_gemv_ab 同确定性取值 */
    static uint16_t a1[K], w[N * K / 2 * 2];  /* w 大小 = N*K */
    for (uint32_t i = 0; i < K; i++) a1[i] = f32_to_f16((float)((int)(i % 29) - 14));
    for (size_t i = 0; i < (size_t)N * K; i++) w[i] = f32_to_f16((float)((int)(i % 37) - 18));
    static uint16_t gold[N];
    for (uint32_t n = 0; n < N; n++) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; k++)
            acc += f16_to_f32(a1[k]) * f16_to_f32(w[(size_t)n * K + k]);
        gold[n] = f32_to_f16(acc);
    }

    static uint8_t out_a[N * 2], out_b[M32 * N * 2], out_c[M32 * N * 2];
    struct arm arms[3] = {
        {"A_m1_hvx",  D "/blob_gemv_m1.wtop",    1u,  0,0,0,0,0, out_a},
        {"B_m32_hmx", D "/blob_gemv_m32.wtop",   M32, 0,0,0,0,0, out_b},
        {"C_m32_pp",  D "/blob_gemv_m32pp.wtop", M32, 0,0,0,0,0, out_c},
    };
    int fail = 0;
    for (int i = 0; i < 3; i++)
        if (run_arm(&arms[i])) { ex_log("[FAIL] arm %s", arms[i].name); fail = 1; }
    if (fail) return ex_summary() || 1;

    const double wmb = (double)N * K * 2u / (1024.0 * 1024.0);
    for (int i = 0; i < 3; i++) {
        struct arm* A = &arms[i];
        double bw = A->dur_med ? wmb / ((double)A->dur_med / 1e6) / 1024.0 : 0;
        ex_log("[%s] engine=%u dur med=%lluus min=%llu max=%llu wall_med=%lluus "
               "eff_bw=%.2fGB/s  vsA=%.2fx",
               A->name, A->engine,
               (unsigned long long)A->dur_med, (unsigned long long)A->dur_min,
               (unsigned long long)A->dur_max, (unsigned long long)A->wall_med,
               bw, arms[0].dur_med ? (double)arms[0].dur_med / (double)A->dur_med : 0);
    }

    /* ① 引擎门 */
    ex_check("A_engine_hvx", arms[0].engine != 1u, 0);
    ex_check("B_engine_hmx", arms[1].engine != 2u, 0);
    ex_check("C_engine_hmx", arms[2].engine != 2u, 0);
    /* ② 数值门: 行 0 == 金标 (byte-exact) */
    for (int i = 0; i < 3; i++) {
        size_t nd = 0; long first = -1;
        const uint16_t* r0 = (const uint16_t*)arms[i].out;
        for (uint32_t n = 0; n < N; n++)
            if (r0[n] != gold[n]) { nd++; if (first < 0) first = (long)n; }
        if (nd) ex_log("[gold-mismatch] %s nd=%zu first_n=%ld got=0x%04X want=0x%04X",
                       arms[i].name, nd, first, r0[first < 0 ? 0 : first], gold[first < 0 ? 0 : first]);
        char gn[48];
        snprintf(gn, sizeof(gn), "%s_row0_golden", arms[i].name);
        ex_check(gn, nd ? 1 : 0, 0);
    }
    /* ③ pad 门: B/C 行 1..31 全零 */
    for (int i = 1; i < 3; i++) {
        const uint16_t* r = (const uint16_t*)arms[i].out;
        int nz = 0;
        for (uint32_t m = 1; m < M32 && !nz; m++)
            for (uint32_t n = 0; n < N; n++)
                if (r[(size_t)m * N + n]) { nz = 1; break; }
        char gn[48];
        snprintf(gn, sizeof(gn), "%s_pad_rows_zero", arms[i].name);
        ex_check(gn, nz, 0);
    }

    ex_check("suite_complete", 0, 0);
    return ex_summary();
}
