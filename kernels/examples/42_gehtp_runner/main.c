/*
 * 42_gehtp_runner — 通用 WTOP 上板 runner ("gehtp run" 的设备侧)
 * =====================================================================
 * 与例 37/40 的区别: 无任何模型硬编码 (尺寸/temp id/路径全部运行时读取)。
 * 输入: $GDIR/job.txt (host 侧 gehtp run 推送), 行格式 "key value":
 *   blob    设备端 .wtop 绝对路径
 *   input   f16 原始输入文件 (slot0 EXT_IN 注入)
 *   output  输出写出路径
 *   out_temp 输出 temp id (来自 blob 配套 manifest.json 的 output_temp)
 * 流程: wt_parse → wt_exec_run_io(out_ptr=NULL) → wt_exec_temp_bytes 定长
 *       → fwrite 输出。判据: rc==0 且输出非空 → [PASS]。
 * wt_blob ≈4.5MB 必须堆分配 (ribbon 栈 256KB, M2 死因教训)。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hvxhmx_v23.h"
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"

#define JOB_PATH "/data/local/tmp/hvxhmx23/gehtp/job.txt"

static uint8_t* read_file(const char* p, size_t* out_len) {
    FILE* f = fopen(p, "rb");
    if (!f) return NULL;
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

static int job_get(const char* job, const char* key, char* out, size_t n) {
    size_t klen = strlen(key);
    const char* p = job;
    while (*p) {
        const char* eol = strchr(p, '\n');
        size_t line_len = eol ? (size_t)(eol - p) : strlen(p);
        if (line_len > klen + 1 && memcmp(p, key, klen) == 0 && p[klen] == ' ') {
            size_t vlen = line_len - klen - 1;
            if (vlen >= n) vlen = n - 1;
            memcpy(out, p + klen + 1, vlen);
            out[vlen] = 0;
            return 0;
        }
        p = eol ? eol + 1 : p + line_len;
    }
    return -1;
}

int main(void) {
    ex_open_result("42_gehtp_runner");
    ex_log("=== 42_gehtp_runner (generic WTOP runner) ===");
    int bad = 0;

    size_t job_len = 0;
    uint8_t* job = read_file(JOB_PATH, &job_len);
    if (!job) { ex_log("[FAIL] read %s", JOB_PATH); return ex_summary() || 1; }

    char blob_p[256] = {0}, in_p[256] = {0}, out_p[256] = {0}, temp_s[32] = {0};
    if (job_get((char*)job, "blob", blob_p, sizeof(blob_p)) ||
        job_get((char*)job, "input", in_p, sizeof(in_p)) ||
        job_get((char*)job, "output", out_p, sizeof(out_p)) ||
        job_get((char*)job, "out_temp", temp_s, sizeof(temp_s))) {
        ex_log("[FAIL] job.txt 缺键 (blob/input/output/out_temp)");
        free(job);
        return ex_summary() || 1;
    }
    free(job);
    uint32_t out_temp = (uint32_t)strtoul(temp_s, NULL, 10);

    size_t blob_len = 0, in_len = 0;
    uint8_t* blob = read_file(blob_p, &blob_len);
    uint8_t* in = read_file(in_p, &in_len);
    if (!blob || !in) { ex_log("[FAIL] read blob/input"); free(blob); free(in); return ex_summary() || 1; }

    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w) { ex_log("[FAIL] wt_blob alloc"); free(blob); free(in); return ex_summary() || 1; }
    if (wt_parse(blob, blob_len, w) != WT_OK) {
        ex_log("[FAIL] wt_parse"); free(blob); free(in); free(w);
        return ex_summary() || 1;
    }

    char err[128] = {0};
    int rc = wt_exec_run_io(w, in, NULL, out_temp, NULL, NULL, err, sizeof(err));
    if (rc) {
        ex_log("[FAIL] wt_exec_run_io rc=%d %s", rc, err);
        wt_exec_shutdown();
        free(blob); free(in); free(w);
        return ex_summary() || 1;
    }

    uint32_t obytes = wt_exec_temp_last_bytes(out_temp);
    const uint8_t* optr = wt_exec_temp(out_temp);
    if (!obytes || !optr) {
        ex_log("[FAIL] out temp %u empty", (unsigned)out_temp);
        wt_exec_shutdown();
        free(blob); free(in); free(w);
        return ex_summary() || 1;
    }
    FILE* of = fopen(out_p, "wb");
    if (!of || fwrite(optr, 1, obytes, of) != obytes) {
        ex_log("[FAIL] write %s", out_p);
        if (of) fclose(of);
        wt_exec_shutdown();
        free(blob); free(in); free(w);
        return ex_summary() || 1;
    }
    fclose(of);
    ex_log("[PASS] %u ops, out temp %u = %u bytes -> %s",
           (unsigned)w->n_ops, (unsigned)out_temp, (unsigned)obytes, out_p);

    wt_exec_shutdown();
    free(blob); free(in); free(w);
    ex_log(bad ? "[FAIL] 42_gehtp_runner overall" : "[PASS] 42_gehtp_runner overall");
    return ex_summary() || bad;
}
