/*
 * 39_probe — GEHTP M3c: qnn2layer probe(103 节点浮点图)上板 vs HTP golden
 * =====================================================================
 * 输入(host 准备并 push 到 $D/):
 *   blob.wtop      wtop_emit 产物(slot0 标 EXT_IN, 71 ops 22 slots)
 *   tokens_0..7.raw  int32 [1,32] 128B(qnn2layer calib 同源)
 * 动作: 每个 tokens 跑 wt_exec_run_io(Level 1 注入), 输出 logits
 *   [32,256] f16 → out_N.f16.raw, host 与 q_htp_native 对拍。
 * 输出: ex_log 行含 [PASS]/[FAIL], build_examples.sh 汇总解析。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hvxhmx_v23.h"
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"

#define D "/data/local/tmp/hvxhmx23/g39"
#define N_TOKENS 32u
#define LOGITS (32u * 256u)  /* logits [1,32,256] f16 */

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
    uint32_t out_temp = 6u;   /* argv[1] 覆盖(manifest output_temp, host 传入) */
    if (argc > 1) out_temp = (uint32_t)strtoul(argv[1], NULL, 0);
    ex_open_result("39_probe");
    ex_log("out_temp=%u", out_temp);
    size_t blen = 0;
    uint8_t* blob = read_file(D "/blob.wtop", &blen);
    if (!blob) { ex_log("[FAIL] read blob"); return ex_summary() || 1; }
    /* wt_blob ~4.5MB: 设备线程栈 256KB, 必须堆分配 */
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(blob, blen, w) != WT_OK) {
        ex_log("[FAIL] wt_parse"); free(blob); free(w); return ex_summary() || 1;
    }
    ex_log("probe: ops=%u slots=%u", w->n_ops, w->n_slots);

    int bad = 0;
    /* 首轮逐 op 推进(崩溃取证: 无缓冲日志, 死点 = 最后一行之后) */
    {
        char err[128] = {0};
        int rc = 0;
        for (uint32_t i = 0; i < w->n_ops; i++) {
            ex_log("diag: op %u", i);
            rc = wt_exec_run_range(w, i, 1, NULL, NULL, err, sizeof(err));
            if (rc) { ex_log("[FAIL] diag op %u rc=%d %s", i, rc, err); bad = 1; }
        }
        wt_exec_shutdown();
        ex_log("diag: op-by-op done rc=%d", rc);
    }
    for (int i = 0; i < 8; i++) {
        char p[128];
        snprintf(p, sizeof(p), D "/tokens_%d.raw", i);
        size_t tlen = 0;
        uint8_t* tok = read_file(p, &tlen);
        if (!tok || tlen != N_TOKENS * 4u) {
            ex_log("[FAIL] tokens_%d read (len=%zu)", i, tlen);
            bad = 1; free(tok); continue;
        }
        uint8_t* out = malloc(LOGITS * 2u);
        char err[128] = {0};
        int rc = wt_exec_run_io(w, tok, out, out_temp, NULL, NULL, err, sizeof(err));
        wt_exec_shutdown();
        if (rc) { ex_log("[FAIL] run_io %d rc=%d %s", i, rc, err); bad = 1; }
        else {
            snprintf(p, sizeof(p), D "/out_%d.f16.raw", i);
            FILE* f = fopen(p, "wb");
            if (f) { fwrite(out, 1, LOGITS * 2u, f); fclose(f); }
            else bad = 1;
            ex_log("%s probe run %d (out_%d.f16.raw)", f ? "[PASS]" : "[FAIL]", i, i);
        }
        free(out); free(tok);
    }
    ex_log(bad ? "[FAIL] 39_probe overall" : "[PASS] 39_probe overall");
    free(blob); free(w);
    return ex_summary() || bad;
}
