/*
 * 57_opts_breakdown — 018 真图 decode 单步 op_ts 归因
 * =====================================================================
 * fence 修复后重采真图 matmul 归因: lm_head 全图时延 + matmul 总量 + top10。
 * 对照 56 的整步计时, 回答「lm_head 3.6GB/s vs 合成 16GB/s 缺口」残余多少。
 *
 * 用法: ./run_main_on_hexagon 3 test_57.so
 * 输出: result_57_opts_breakdown.txt
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
#define LM_HEAD_IDX 1349u
#define TOPN 10

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

int main(int argc, char** argv) {
    ex_open_result("57_opts_breakdown");
    ex_log("=== 57_opts_breakdown blob=%s ===", BLOB);

    size_t blen = 0;
    uint8_t* b = read_file(BLOB, &blen);
    if (!b) { ex_log("read FAIL %s", BLOB); return ex_summary(); }
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(b, blen, w) != WT_OK) { ex_log("wt_parse FAIL"); free(b); free(w); return ex_summary(); }
    ex_log("wt_parse ok n_ops=%u n_slots=%u", w->n_ops, w->n_slots);
    wt_exec_set_trace(0);

    struct wt_op_ts* ts = calloc(w->n_ops, sizeof(*ts));
    wt_exec_set_ts(ts, w->n_ops);

    char err[128];
    uint32_t em = 0;
    int64_t t0 = HAP_perf_get_time_us();
    int rc = wt_exec_run(w, &em, NULL, err, sizeof(err));
    int64_t t1 = HAP_perf_get_time_us();
    if (rc) { ex_log("run FAIL rc=%d: %s", rc, err); free(ts); free(w); free(b); return ex_summary(); }
    ex_log("step0_wall=%lluus", (unsigned long long)(t1 - t0));

    uint64_t mat_total = 0, lm_dur = 0, lm_dma = 0;
    uint32_t n_mat = 0;
    for (uint32_t i = 0; i < w->n_ops; i++) {
        uint32_t oc = w->ops[i].opcode;
        if (oc == OP_MATMUL_F16 || oc == OP_MATMUL_F16_SPLIT) {
            mat_total += ts[i].dur_us;
            n_mat++;
        }
        if (i == LM_HEAD_IDX) { lm_dur = ts[i].dur_us; lm_dma = ts[i].dma_bytes; }
    }
    ex_log("n_mat=%u mat_total=%lluus (%.1f%% of step)",
           n_mat, (unsigned long long)mat_total,
           100.0 * (double)mat_total / (double)(t1 - t0));
    /* lm_head 权重 97.24MB (K=1024 N=151936 Q4 640B/tile: 48*32*640=983040B*96.86)
     * 实取 97239040 字节 → GB/s */
    ex_log("lm_head(idx%u) dur=%lluus dma=%u bytes bw=%.2f GB/s",
           (unsigned)LM_HEAD_IDX, (unsigned long long)lm_dur, lm_dma,
           97239040.0 / (double)lm_dur / 1e3);

    /* top-N matmul by dur */
    uint32_t top_i[TOPN];
    uint64_t top_d[TOPN];
    for (int k = 0; k < TOPN; k++) { top_i[k] = 0; top_d[k] = 0; }
    for (uint32_t i = 0; i < w->n_ops; i++) {
        uint32_t oc = w->ops[i].opcode;
        if (oc != OP_MATMUL_F16 && oc != OP_MATMUL_F16_SPLIT) continue;
        uint64_t d = ts[i].dur_us;
        for (int k = 0; k < TOPN; k++) {
            if (d > top_d[k]) {
                for (int j = TOPN - 1; j > k; j--) { top_i[j] = top_i[j - 1]; top_d[j] = top_d[j - 1]; }
                top_i[k] = i; top_d[k] = d;
                break;
            }
        }
    }
    for (int k = 0; k < TOPN; k++)
        ex_log("top%d op%u code%u dur=%lluus eng=%u",
               k + 1, top_i[k], (unsigned)w->ops[top_i[k]].opcode,
               (unsigned long long)top_d[k], ts[top_i[k]].engine);

    wt_exec_shutdown();
    free(ts); free(w); free(b);
    return ex_summary();
}
