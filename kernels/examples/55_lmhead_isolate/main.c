/*
 * 55_lmhead_isolate — 018 真图 lm_head 单 op 隔离计时
 * =====================================================================
 * 裁决问题: 真图 decode_q4_018.wtop 的 lm_head(idx1349, M=1 K=1024 N=151936,
 * W4TS 0x22) 在 op_ts 里 27ms@3.6GB/s, 而合成同形探针(54) 6.09ms@16GB/s。
 * 4.4× 缺口是「全图上下文(DRAM/DMA 争抢)」还是「该 op 本身(权重/rpcmem 慢)」?
 * 本 .so 只跑 op idx=1349 单 op 循环, 隔离上下文后计时。
 *
 * 用法: ./run_main_on_hexagon 3 test_55.so
 * 判据: engine==1 (HVX, W4TS 未回落) + dur 报告
 * 输出: result_55_lmhead_isolate.txt
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

#define BLOB "/data/local/tmp/hvxhmx23/gehtp_q6b_018/decode_q4_018.wtop"
#define OP_IDX 1349u
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

static uint64_t med_u64(uint64_t* a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[n / 2];
}

int main(int argc, char** argv) {
    ex_open_result("55_lmhead_isolate");
    ex_log("=== 55_lmhead_isolate blob=%s op=%u ===", BLOB, OP_IDX);

    size_t blen = 0;
    uint8_t* b = read_file(BLOB, &blen);
    if (!b) { ex_log("read FAIL %s", BLOB); return ex_summary(); }
    ex_log("blob %zu bytes", blen);
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(b, blen, w) != WT_OK) { ex_log("wt_parse FAIL"); free(b); free(w); return ex_summary(); }
    ex_log("wt_parse ok n_ops=%u n_slots=%u", w->n_ops, w->n_slots);
    /* 关 trace (同真 runner 默认), 只测内核 */
    wt_exec_set_trace(0);

    char err[128];
    int rc = 0;
    /* 先整图跑一遍: 物化 temp 池(输入 a=temp[2] 由前序 op 写) + 触发
     * W4TS 权重 fence_handoff flush(s_w_cleaned 置位), 后续隔离循环测稳态。 */
    {
        uint32_t em = 0;
        rc = wt_exec_run(w, &em, NULL, err, sizeof(err));
        if (rc) { ex_log("full-graph setup FAIL rc=%d: %s", rc, err); return ex_summary(); }
        ex_log("full-graph setup ok");
    }

    uint64_t dur[REPS];
    for (int rep = 0; rep < WARM + REPS; rep++) {
        uint32_t em = 0;
        int64_t t0 = HAP_perf_get_time_us();
        rc = wt_exec_run_range(w, OP_IDX, 1, &em, NULL, err, sizeof(err));
        int64_t t1 = HAP_perf_get_time_us();
        if (rc) { ex_log("run FAIL rc=%d: %s", rc, err); break; }
        if (rep >= WARM) dur[rep - WARM] = (uint64_t)(t1 - t0);
    }
    if (!rc) {
        uint64_t dmin = dur[0], dmax = dur[0];
        for (int i = 1; i < REPS; i++) {
            if (dur[i] < dmin) dmin = dur[i];
            if (dur[i] > dmax) dmax = dur[i];
        }
        uint64_t dmed = med_u64(dur, REPS);
        ex_log("dur_med=%lluus min=%llu max=%llu (isolated lm_head, op=%u)",
               (unsigned long long)dmed, (unsigned long long)dmin,
               (unsigned long long)dmax, OP_IDX);
        /* 权重 97.24MB → GB/s */
        ex_log("bw=%.2f GB/s", 97239040.0 / (double)dmed / 1e3);
    }
    wt_exec_shutdown();
    free(w); free(b);
    return ex_summary();
}
