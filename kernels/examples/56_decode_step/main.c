/*
 * 56_decode_step — 018 真图 decode 整步计时 + fence flush 成本直接测量
 * =====================================================================
 * 裁决问题: fence_handoff 单静态对(s_w_cleaned)只记一个权重 → 194 个 W4TS
 * 权重每 op 每 step 重 flush(366.6MB/step)。本 .so 两段:
 *   A) 直接对 weight 区跑 fence_handoff 两次: ① 脏(CUP 刚写, FLUSH 写回)
 *      ② 净(已 flush, 仅 invalidate 地址走查)。报告各自 GB/s。
 *   B) 整图 wt_exec_run 循环 N 步, 逐步报时。步 0 vs 稳态(步 1+)差 = 脏/净
 *      flush 差; 修复后应全步压低。
 *
 * 用法: ./run_main_on_hexagon 3 test_56.so
 * 输出: result_56_decode_step.txt
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
#include "fence.h"

#define BLOB "/data/local/tmp/hvxhmx23/gehtp_q6b_018/decode_q4_018.wtop"
#define STEPS 8

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

static uint64_t med_u64(uint64_t* a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[n / 2];
}

int main(int argc, char** argv) {
    ex_open_result("56_decode_step");
    ex_log("=== 56_decode_step blob=%s steps=%d ===", BLOB, STEPS);

    size_t blen = 0;
    uint8_t* b = read_file(BLOB, &blen);
    if (!b) { ex_log("read FAIL %s", BLOB); return ex_summary(); }
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(b, blen, w) != WT_OK) { ex_log("wt_parse FAIL"); free(b); free(w); return ex_summary(); }
    ex_log("wt_parse ok n_ops=%u n_slots=%u wbytes=%zu",
           w->n_ops, w->n_slots, w->weight_bytes);
    wt_exec_set_trace(0);

    /* ---- Phase A: fence flush 成本直接测量 (脏 vs 净) ---- */
    {
        const uint8_t* wb = w->weight_base;
        uint32_t wbytes = (uint32_t)w->weight_bytes;
        /* ① 脏: 权重由 CPU fread 写入 DDR, 尚未 flush (首次 writeback) */
        int64_t t0 = HAP_perf_get_time_us();
        fence_handoff((void*)wb, wbytes, FC_CPU, FC_DMA, FM_DDR);
        int64_t t1 = HAP_perf_get_time_us();
        /* ② 净: 同一地址再 flush (已 writeback+invalidate, 仅地址走查) */
        int64_t t2 = HAP_perf_get_time_us();
        fence_handoff((void*)wb, wbytes, FC_CPU, FC_DMA, FM_DDR);
        int64_t t3 = HAP_perf_get_time_us();
        double gb = (double)wbytes / 1e9;
        ex_log("fence dirty=%lldus (%.2f GB/s) clean=%lldus (%.2f GB/s)",
               (long long)(t1 - t0), gb / ((double)(t1 - t0) / 1e6),
               (long long)(t3 - t2), gb / ((double)(t3 - t2) / 1e6));
    }

    /* ---- Phase B: 整图 decode 步逐步报时 ---- */
    char err[128];
    uint64_t dur[STEPS];
    int rc = 0;
    for (int step = 0; step < STEPS; step++) {
        uint32_t em = 0;
        int64_t t0 = HAP_perf_get_time_us();
        rc = wt_exec_run(w, &em, NULL, err, sizeof(err));
        int64_t t1 = HAP_perf_get_time_us();
        if (rc) { ex_log("step %d run FAIL rc=%d: %s", step, rc, err); break; }
        dur[step] = (uint64_t)(t1 - t0);
        ex_log("step %d = %lluus", step, (unsigned long long)dur[step]);
    }
    if (!rc) {
        uint64_t dmin = dur[0], dmax = dur[0], dsum = 0;
        for (int i = 0; i < STEPS; i++) {
            if (dur[i] < dmin) dmin = dur[i];
            if (dur[i] > dmax) dmax = dur[i];
            dsum += dur[i];
        }
        ex_log("step_med=%lluus min=%llu max=%llu avg=%llu tok/s=%.2f",
               (unsigned long long)med_u64(dur, STEPS),
               (unsigned long long)dmin, (unsigned long long)dmax,
               (unsigned long long)(dsum / STEPS),
               1e6 / (double)med_u64(dur, STEPS));
    }
    wt_exec_shutdown();
    free(w); free(b);
    return ex_summary();
}
