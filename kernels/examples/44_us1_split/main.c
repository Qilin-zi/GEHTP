/*
 * 44_us1_split — 002 US1 设备要害门 (T017): chunk-HMX vs 整算-HMX 逐字节对拍
 * =====================================================================
 * 输入 (host pack_matmul_split 生成并 push 到 $D/):
 *   blob_fc_whole.wtop   1 × OP_MATMUL_F16      (M=32,K=1024,N=2048,flags=2 FC)
 *   blob_fc_chunk.wtop   2 × OP_MATMUL_F16_SPLIT (N_full=2048, n0=0/1024)
 * 判据 (plan.md US1 验证门 3):
 *   ① 双侧 matmul 实际引擎 = WT_ENG_HMX (排除双侧静默回落标量的假阳性)
 *   ② temp0 整算版 vs 分块版 逐字节一致 (0 ULP, memcmp)
 *   ③ 分块版两个 chunk 都 HMX (续 chunk 引擎沿用, 禁混合路径 FR-013④)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include "dc_parts.h"   /* dc_clean_ddr (blob DMA bypass 源) */

#define D "/data/local/tmp/hvxhmx23/g44"
#define MM 32u
#define NN 2048u
#define OUT_BYTES (MM * NN * 2u)

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

/* 跑一个 blob, 回填 temp0 快照 + 各 op 引擎 (engine[i] = WT_ENG_*)。返回 0 成功 */
static int run_blob(const uint8_t* blob, size_t blen, uint8_t* snap,
                    uint32_t* engine, uint32_t n_ops) {
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(blob, blen, w) != WT_OK) { ex_log("wt_parse FAIL"); free(w); return 1; }
    struct wt_op_ts* ts = calloc(n_ops, sizeof(*ts));
    wt_exec_set_ts(ts, n_ops);
    uint32_t em = 0;
    char err[128];
    int rc = wt_exec_run(w, &em, NULL, err, sizeof(err));
    if (rc) {
        ex_log("wt_exec_run FAIL rc=%d: %s", rc, err);
        wt_exec_shutdown(); free(ts); free(w); return 1;
    }
    uint8_t* t0 = wt_exec_temp(0);
    if (!t0 || wt_exec_temp_bytes(0) != OUT_BYTES) {
        ex_log("temp0 bad (t0=%p bytes=%lu)", t0, (unsigned long)wt_exec_temp_bytes(0));
        wt_exec_shutdown(); free(ts); free(w); return 1;
    }
    memcpy(snap, t0, OUT_BYTES);
    for (uint32_t i = 0; i < n_ops; i++) engine[i] = ts[i].engine;
    wt_exec_shutdown();
    free(ts); free(w);
    return 0;
}

int main(void) {
    ex_open_result("44_us1_split");
    size_t bw = 0, bc = 0;
    uint8_t* blob_w = read_file(D "/blob_fc_whole.wtop", &bw);
    uint8_t* blob_c = read_file(D "/blob_fc_chunk.wtop", &bc);
    if (!blob_w || !blob_c) { ex_log("[FAIL] read blobs"); free(blob_w); free(blob_c); return ex_summary() || 1; }
    dc_clean_ddr(blob_w, (uint32_t)bw);
    dc_clean_ddr(blob_c, (uint32_t)bc);

    uint8_t* whole = malloc(OUT_BYTES);
    uint8_t* chunk = malloc(OUT_BYTES);
    uint32_t eng_w[1] = {0xFF}, eng_c[2] = {0xFF, 0xFF};

    int rw = run_blob(blob_w, bw, whole, eng_w, 1);
    int rc = run_blob(blob_c, bc, chunk, eng_c, 2);
    if (rw || rc) { ex_log("[FAIL] run_blob"); free(whole); free(chunk); free(blob_w); free(blob_c); return ex_summary() || 1; }

    ex_log("engine whole=%u (expect %u=HMX) chunk=[%u,%u]",
           eng_w[0], (unsigned)WT_ENG_HMX, eng_c[0], eng_c[1]);

    /* ① HMX 门: 双侧必须真 HMX */
    int h1 = (eng_w[0] == WT_ENG_HMX) ? 0 : 1;
    ex_check("whole_engine_hmx", h1, 0);
    int h2 = (eng_c[0] == WT_ENG_HMX && eng_c[1] == WT_ENG_HMX) ? 0 : 1;
    ex_check("chunk_both_engine_hmx", h2, 0);

    /* ② 逐字节对拍 */
    int bad = memcmp(whole, chunk, OUT_BYTES) != 0;
    if (bad) {
        size_t nd = 0, first = (size_t)-1;
        for (size_t i = 0; i < OUT_BYTES / 2; i++) {
            if (((uint16_t*)whole)[i] != ((uint16_t*)chunk)[i]) {
                nd++;
                if (first == (size_t)-1) first = i;
            }
        }
        ex_log("byte-diff=%zu/%zu first_f16_idx=%zu", nd, OUT_BYTES / 2, first);
    }
    ex_check("chunk_vs_whole_byteexact", bad ? 1 : 0, 0);

    ex_check("suite_complete", 0, 0);
    free(whole); free(chunk); free(blob_w); free(blob_c);
    return ex_summary();
}
