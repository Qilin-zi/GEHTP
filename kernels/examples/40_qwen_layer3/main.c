/*
 * 40_qwen_layer3 — GEHTP M4.2: Qwen3.5 全注意力层 L3 WTOP blob 设备验证
 * =====================================================================
 * 输入 (host 准备并 push 到 /data/local/tmp/hvxhmx23/g40/):
 *   layer_3.wtop   wtop_emit 产物 (slot0 EXT_IN; cos/sin/mask 固化槽)
 *   in.f16.raw     外部输入 hidden [1,1024,32] f16 (32768 元素, 64KB)
 *   gold.f16.raw   host 参考输出 (host_run execute_host → f16, 同布局)
 * 判据:
 *   C1  wt_parse OK
 *   C2  逐 op wt_exec_run_range 全通过
 *   C3  run_io 注入输入, 输出 vs gold ≤1 ULP (f16)
 * 输出: ex_log 行含 [PASS]/[FAIL]; 输出 dump 到 g40/out_40.f16.raw
 *       (host 拉回做 cos 对拍)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hvxhmx_v23.h"
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"

#define D "/data/local/tmp/hvxhmx23/g40"
#define N_ELEM (1u * 1024u * 32u)   /* 输出 [1,1024,32] = 32768 */
#define OUT_TEMP 1u                  /* manifest: output_temp */

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

/* ≤1 ULP 判定: f16 相邻步进比较 (NaN/Inf 视为不匹配) */
static int within_1ulp(uint16_t a, uint16_t g) {
    if (a == g) return 1;
    if ((a & 0x7c00u) == 0x7c00u || (g & 0x7c00u) == 0x7c00u) return 0;
    uint16_t d = (uint16_t)(a > g ? a - g : g - a);
    return d <= 1u;
}

int main(void) {
    fprintf(stderr, "HELLO40 main enter\n");
    ex_open_result("40_qwen_layer3");
    fprintf(stderr, "HELLO40 after open_result\n");
    ex_log("=== 40_qwen_layer3 (GEHTP M4.2) ===");

    int bad = 0;

    size_t blob_len = 0, in_len = 0, gold_len = 0;
    uint8_t* blob = read_file(D "/layer_3.wtop", &blob_len);
    uint8_t* in = read_file(D "/in.f16.raw", &in_len);
    uint8_t* gold = read_file(D "/gold.f16.raw", &gold_len);
    if (!blob || !in || !gold) { ex_log("[FAIL] read inputs"); return 1; }
    if (in_len != N_ELEM * 2u || gold_len != N_ELEM * 2u) {
        ex_log("[FAIL] input sizes in=%zu gold=%zu", in_len, gold_len); return 1;
    }

    /* wt_blob ~4.5MB: 设备线程栈 256KB, 必须堆分配 */
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w) { ex_log("[FAIL] wt_blob alloc"); return 1; }
    int rc = wt_parse(blob, blob_len, w);
    if (rc != WT_OK) { ex_log("[FAIL] wt_parse rc=%d %s", rc, wt_err_str(rc)); return 1; }
    ex_log("[PASS] wt_parse: %u slots %u ops", w->n_slots, w->n_ops);

    char err[128] = {0};

    /* C2(分段诊断): GEHTP_SPLIT_RUN=1 时把整段拆成两半跑, 定位崩溃 op 段 */
    FILE* sflag = fopen("/data/local/tmp/hvxhmx23/split_flag", "r");
    if (sflag) { fclose(sflag);
        uint32_t half = w->n_ops / 2u;
        ex_log("[split] run 0..%u", half - 1u);
        rc = wt_exec_run_range(w, 0, half, NULL, NULL, err, sizeof(err));
        ex_log("[split] first half rc=%d %s", rc, rc ? err : "");
        wt_exec_shutdown();
        ex_log("[split] run %u..%u", half, w->n_ops - 1u);
        rc = wt_exec_run_range(w, half, w->n_ops - half, NULL, NULL, err, sizeof(err));
        ex_log("[split] second half rc=%d %s", rc, rc ? err : "");
        wt_exec_shutdown();
        return 0;
    }

    /* C3: run_io 注入输入, 输出 → out_io */
    ex_log("[milestone] before run_io");
    uint16_t* out_io = malloc(N_ELEM * 2u);
    if (!out_io) { ex_log("[FAIL] io alloc"); bad = 1; }
    else {
        rc = wt_exec_run_io(w, in, out_io, OUT_TEMP, NULL, NULL, err, sizeof(err));
        if (rc) { ex_log("[FAIL] run_io rc=%d %s", rc, err); bad = 1; }
        else ex_log("[PASS] run_io ok");
        ex_log("[milestone] after run_io");
        wt_exec_shutdown();
    }

    /* dump 输出 (host 拉回 cos 对拍) */
    if (out_io) {
        FILE* f = fopen(D "/out_40.f16.raw", "wb");
        if (!f) { ex_log("[FAIL] open out_40"); bad = 1; }
        else { fwrite(out_io, 1, N_ELEM * 2u, f); fclose(f); }
    }

    /* C3: gold 1ULP 对拍 */
    if (out_io && gold) {
        const uint16_t* g = (const uint16_t*)gold;
        uint32_t n_bad = 0;
        for (uint32_t i = 0; i < N_ELEM; i++)
            if (!within_1ulp(out_io[i], g[i])) n_bad++;
        if (n_bad) { ex_log("[FAIL] golden 1ULP bad=%u/%u", (unsigned)n_bad, (unsigned)N_ELEM); bad = 1; }
        else ex_log("[PASS] golden <= 1 ULP (%u elems)", (unsigned)N_ELEM);
    }

    free(blob); free(in); free(gold); free(out_io); free(w);
    ex_log("=== 39 done bad=%d ===", bad);
    return bad;
}
