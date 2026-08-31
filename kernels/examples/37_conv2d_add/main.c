/*
 * 37_conv2d_add — GEHTP conv2d+add WTOP blob 设备验证 (M2 判据, 阶段 9)
 * =====================================================================
 * 输入 (host conv_add_pipeline.sh 准备并 push 到 $D/g37/):
 *   blob.wtop       wtop_emit 产物 (slot0 标 EXT_IN, 3 slots 5 ops)
 *   blob_spill.wtop 1KB VTCM 预算变体 (含 OP_SPILL/OP_FILL 记录)
 *   in0/in1/in2.f16.raw  外部输入轮换 (NCHW 32x32x32 f16, 64KB)
 *   gold0/gold1/gold2.f16.raw  对应 fp16 纪律金标 (NCHW f16)
 * 判据:
 *   C1  wt_parse 两 blob OK
 *   C2  整步 wt_exec_run vs 逐 op wt_exec_run_range 输出 byte-exact
 *   C3  输出 vs golden ≤ 1 ULP (f16)
 *   C4  Level 1 输入注入: 三组输入轮换 (wt_exec_run_io), 每组独立对拍
 *   C5  溢出变体 blob_spill 同样 C2/C3 (spill/fill 搬运正确性)
 * 输出: ex_log 行含 [PASS]/[FAIL], build_examples.sh 汇总解析
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

#define D "/data/local/tmp/hvxhmx23/g37"
#define N_ELEM (32u * 32u * 32u)
#define OUT_TEMP 4u   /* manifest: output_temp (末段 Transpose 输出) */

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

/* mode 0: 全判据(整步/逐段/注入/金标); mode 1: 仅注入+金标(输入轮换轮次) */
static int run_one(const char* blob_path, const uint8_t* in, const uint8_t* gold,
                   const char* tag, int mode) {
    size_t blob_len = 0;
    uint8_t* blob = read_file(blob_path, &blob_len);
    if (!blob) { ex_log("[FAIL] %s: read blob", tag); return 1; }
    struct wt_blob w;
    if (wt_parse(blob, blob_len, &w) != WT_OK) { ex_log("[FAIL] %s: wt_parse", tag); free(blob); return 1; }

    uint16_t* out_fused = malloc(N_ELEM * 2u);
    uint16_t* out_split = malloc(N_ELEM * 2u);
    if (!out_fused || !out_split) { ex_log("[FAIL] %s: alloc", tag); free(blob); return 1; }

    char err[128] = {0};
    int bad = 0;

    /* 整步 (baked slot0 数据跑 wt_exec_run; slot0 已标 EXT_IN 但 wt_exec_run
     * 不看 ext 指针 → 用 blob 内固化数据) */
    int rc = wt_exec_run(&w, NULL, NULL, err, sizeof(err));
    if (rc) { ex_log("[FAIL] %s: fused run rc=%d %s", tag, rc, err); bad = 1; }
    else if (wt_exec_temp(OUT_TEMP)) memcpy(out_fused, wt_exec_temp(OUT_TEMP), N_ELEM * 2u);
    else { ex_log("[FAIL] %s: no out temp", tag); bad = 1; }
    wt_exec_shutdown();

    /* 逐 op (wt_exec_run_range 逐段) */
    int ok = 1;
    for (uint32_t i = 0; i < w.n_ops && ok; i++) {
        rc = wt_exec_run_range(&w, i, 1, NULL, NULL, err, sizeof(err));
        if (rc) { ex_log("[FAIL] %s: split op%u rc=%d %s", tag, i, rc, err); ok = 0; }
    }
    if (ok) {
        if (wt_exec_temp(OUT_TEMP)) memcpy(out_split, wt_exec_temp(OUT_TEMP), N_ELEM * 2u);
        else ok = 0;
    }
    wt_exec_shutdown();
    if (!ok) bad = 1;

    /* Level 1: 外部输入注入 (run_io; 输出 temp → out_ptr) */
    uint16_t* out_io = malloc(N_ELEM * 2u);
    if (!out_io) { ex_log("[FAIL] %s: io alloc", tag); bad = 1; }
    else {
        rc = wt_exec_run_io(&w, in, out_io, OUT_TEMP, NULL, NULL, err, sizeof(err));
        if (rc) { ex_log("[FAIL] %s: run_io rc=%d %s", tag, rc, err); bad = 1; }
        wt_exec_shutdown();
    }

    if (mode == 1) {  /* io-only 轮次: fused/split 不参与(gold 与 baked 输入不同源) */
        const uint16_t* g2 = (const uint16_t*)gold;
        uint32_t n_bad2 = 0;
        for (uint32_t i = 0; i < N_ELEM; i++)
            if (!within_1ulp(out_io[i], g2[i])) n_bad2++;
        if (n_bad2) { ex_log("[FAIL] %s: io golden 1ULP bad=%u", tag, (unsigned)n_bad2); bad = 1; }
        else ex_log("[PASS] %s: io golden <= 1 ULP", tag);
        free(out_fused); free(out_split); free(out_io); free(blob);
        return bad;
    }

    /* C2: fused vs split byte-exact */
    if (memcmp(out_fused, out_split, N_ELEM * 2u) != 0)
        { ex_log("[FAIL] %s: fused != split", tag); bad = 1; }
    else
        ex_log("[PASS] %s: fused == split byte-exact", tag);

    /* C3/C4: 输出对拍 golden (1 ULP) */
    if (gold) {
        const uint16_t* g = (const uint16_t*)gold;
        uint32_t n_bad = 0;
        for (uint32_t i = 0; i < N_ELEM; i++)
            if (!within_1ulp(out_fused[i], g[i])) n_bad++;
        if (n_bad) { ex_log("[FAIL] %s: golden 1ULP bad=%u", tag, (unsigned)n_bad); bad = 1; }
        else ex_log("[PASS] %s: golden <= 1 ULP (%u elems)", tag, (unsigned)N_ELEM);

        /* 注入输入与 baked 输入同源 → 输出应 byte-exact */
        if (memcmp(out_fused, out_io, N_ELEM * 2u) != 0)
            { ex_log("[FAIL] %s: run_io out != fused out", tag); bad = 1; }
        else
            ex_log("[PASS] %s: run_io(ext in) == fused byte-exact", tag);
    }

    free(out_fused); free(out_split); free(out_io); free(blob);
    return bad;
}

int main(void) {
    ex_open_result("37_conv2d_add");
    fprintf(stderr, "GEHTP37: result opened gfp=%d\n", 1);
    ex_log("=== 37_conv2d_add (GEHTP M2) ===");

    size_t in_len[3] = {0, 0, 0}, gold_len[3] = {0, 0, 0};
    uint8_t* ins[3] = {0}, *golds[3] = {0};
    int have_io = 1;
    for (int i = 0; i < 3; i++) {
        char p[160];
        snprintf(p, sizeof(p), D "/in%d.f16.raw", i);
        ins[i] = read_file(p, &in_len[i]);
        snprintf(p, sizeof(p), D "/gold%d.f16.raw", i);
        golds[i] = read_file(p, &gold_len[i]);
        if (!ins[i] || !golds[i] || in_len[i] != N_ELEM * 2u || gold_len[i] != N_ELEM * 2u)
            have_io = 0;
    }

    int bad = 0;
    /* C1: 主 blob + 溢出变体 */
    bad |= run_one(D "/blob.wtop", ins[0], golds[0], "main", 0);
    /* C4: 输入轮换 (Level 1 三组, io-only) */
    if (have_io)
        for (int i = 0; i < 3; i++)
            bad |= run_one(D "/blob.wtop", ins[i], golds[i], "io_round", 1);
    /* C5: 溢出变体 */
    bad |= run_one(D "/blob_spill.wtop", ins[0], golds[0], "spill", 0);

    for (int i = 0; i < 3; i++) { free(ins[i]); free(golds[i]); }
    ex_log(bad ? "[FAIL] 37_conv2d_add overall" : "[PASS] 37_conv2d_add overall");
    return ex_summary() || bad;
}
