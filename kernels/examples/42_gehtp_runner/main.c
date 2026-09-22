/*
 * 42_gehtp_runner — 通用 WTOP 上板 runner ("gehtp run" 的设备侧)
 * =====================================================================
 * 与例 37/40 的区别: 无任何模型硬编码 (尺寸/temp id/路径全部运行时读取)。
 * 输入: $GDIR/job.txt (host 侧 gehtp run 推送), 行格式 "key value":
 *   blob    设备端 .wtop 绝对路径
 *   input   f16 原始输入文件 (slot0 EXT_IN 注入)
 *   output  输出写出路径
 *   out_temp 输出 temp id (来自 blob 配套 manifest.json 的 output_temp)
 *   trace   (PROF W-P2, 可选) =1 开 per-op optrace 落盘行 (默认关)
 * 流程: wt_parse → wt_exec_run_io(out_ptr=NULL) → wt_exec_temp_bytes 定长
 *       → fwrite 输出。判据: rc==0 且输出非空 → [PASS]。
 * PROF W-P2: 分段 run 计时 ([run] seg_us 行) + op_ts.bin 逐 op 时间戳对
 *       (host gehtp_prof.py 的 join 输入)。run_id 省略: 每次上板为独立进程。
 * wt_blob ≈4.5MB 必须堆分配 (ribbon 栈 256KB, M2 死因教训)。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include "hvxhmx_v23.h"
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include <fcntl.h>
#include <unistd.h>

/* PROF W-P2: 共享板防撞车通道。标准构建读 $GDIR/job.txt; 变体构建经
 *   -DGEHTP_JOB_PATH='"/.../job_prof.txt"' + -o gehtp_runner_prof.so
 * 得到独立二进制+独立 job 文件, 与在途会话零共享状态 (W33 身份制同律)。 */
#ifndef GEHTP_JOB_PATH
#define GEHTP_JOB_PATH "/data/local/tmp/hvxhmx23/gehtp/job.txt"
#endif
#define JOB_PATH GEHTP_JOB_PATH
/* PROF W-P2: 逐 op 时间戳对产物 (host gehtp run 拉回, gehtp_prof.py 消费) */
#define OP_TS_PATH "/data/local/tmp/hrt/gehtp/op_ts.bin"

/* op_ts.bin 线格式 (全小端): magic 'WTS1' u32 | ver u32 | n_ops u32 | pad u32
 * | exec_wall_us u64 | n_ops × {start_us u32, dur_us u32} (相对 exec 入口) */
static int write_op_ts(const char* path, const struct wt_op_ts* ts, uint32_t n,
                       uint64_t wall_us) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) return -1;
    uint8_t hdr[24];
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'W'; hdr[1] = 'T'; hdr[2] = 'S'; hdr[3] = '1';
    uint32_t v = 1; memcpy(hdr + 4, &v, 4);
    memcpy(hdr + 8, &n, 4);
    memcpy(hdr + 16, &wall_us, 8);
    int ok = 0;
    if (write(fd, hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) ok = -1;
    size_t body = (size_t)n * sizeof(*ts);
    if (!ok && write(fd, ts, body) != (ssize_t)body) ok = -1;
    close(fd);
    return ok;
}

static uint8_t* read_file(const char* p, size_t* out_len) {
    /* POSIX read: qurt stdio fread 对 >某量文件死循环, open/read 无此问题 */
    int fd = open(p, O_RDONLY);
    if (fd < 0) return NULL;
    off_t sz = lseek(fd, 0, SEEK_END);
    if (sz <= 0) { close(fd); return NULL; }
    lseek(fd, 0, SEEK_SET);
    uint8_t* buf = malloc((size_t)sz);
    if (!buf) { close(fd); return NULL; }
    size_t got = 0;
    while (got < (size_t)sz) {
        size_t want = (size_t)sz - got;
        if (want > (16u << 20)) want = 16u << 20;  /* qurt read 单次超大有上限, 限 16MB */
        ssize_t n = read(fd, buf + got, want);
        if (n <= 0) break;
        got += (size_t)n;
    }
    close(fd);
    if (got != (size_t)sz) { free(buf); return NULL; }
    *out_len = (size_t)sz;
    return buf;
}

/* 大权重区: host 侧(64 位)给精确字节数; qurt off_t=32 位, stat/ftell 均不可用 */
static uint8_t* read_file_size(const char* p, size_t sz) {
    /* POSIX read, 大小 host 给 (qurt off_t=32 位, 2.5GB 权重不能 lseek/ftell) */
    if (sz == 0) return NULL;
    uint8_t* buf = malloc(sz);
    if (!buf) { ex_log("[rfs] malloc(%zu) FAIL", sz); return NULL; }
    ex_log("[rfs] malloc(%zu) ok", sz);
    int fd = open(p, O_RDONLY);
    if (fd < 0) { ex_log("[rfs] open %s FAIL", p); free(buf); return NULL; }
    ex_log("[rfs] open ok, reading...");
    size_t got = 0;
    while (got < sz) {
        size_t want = sz - got;
        if (want > (16u << 20)) want = 16u << 20;  /* qurt read 单次超大有上限, 限 16MB */
        ssize_t n = read(fd, buf + got, want);
        if (n <= 0) break;
        got += (size_t)n;
        if ((got & ((256u << 20) - 1)) < (16u << 20)) ex_log("[rfs] read %zu/%zu", got, sz);
    }
    close(fd);
    ex_log("[rfs] loop done got=%zu sz=%zu", got, sz);
    if (got != sz) { free(buf); return NULL; }
    ex_log("[rfs] complete %zu", sz);
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
    ex_log("M0 after ex_open");
    ex_log("=== 42_gehtp_runner (generic WTOP runner) ===");
    ex_log("M1 before job read");
    int bad = 0;

    size_t job_len = 0;
    uint8_t* job = read_file(JOB_PATH, &job_len);
    if (!job) { ex_log("[FAIL] read %s", JOB_PATH); return ex_summary() || 1; }
    ex_log("M2 job read %zu", job_len);

    static char blob_p[256], in_p[256], out_p[256], temp_s[32];
    memset(blob_p, 0, sizeof(blob_p)); memset(in_p, 0, sizeof(in_p));
    memset(out_p, 0, sizeof(out_p)); memset(temp_s, 0, sizeof(temp_s));
    if (job_get((char*)job, "blob", blob_p, sizeof(blob_p)) ||
        job_get((char*)job, "input", in_p, sizeof(in_p)) ||
        job_get((char*)job, "output", out_p, sizeof(out_p)) ||
        job_get((char*)job, "out_temp", temp_s, sizeof(temp_s))) {
        ex_log("[FAIL] job.txt 缺键 (blob/input/output/out_temp)");
        free(job);
        return ex_summary() || 1;
    }
    ex_log("M3 job keys ok %s %s", blob_p, out_p);
    static char wgt_size_s[32];
    memset(wgt_size_s, 0, sizeof(wgt_size_s));
    int have_wgt_size = (job_get((char*)job, "weights_size", wgt_size_s, sizeof(wgt_size_s)) == 0);
    /* PROF W-P2: trace 键决定 per-op 落盘行 (缺省=0 净测量; =1 取证)。
     * 引擎默认开(兼容存量 runner), 故此处必须显式两态下发。 */
    {
        static char trace_s[8];
        memset(trace_s, 0, sizeof(trace_s));
        int tr = (job_get((char*)job, "trace", trace_s, sizeof(trace_s)) == 0 && atoi(trace_s)) ? 1 : 0;
        wt_exec_set_trace(tr);
        ex_log("[prof] per-op optrace %s", tr ? "ON" : "OFF");
    }
    free(job);
    uint32_t out_temp = (uint32_t)strtoul(temp_s, NULL, 10);
    ex_log("M4 wgt_size=%s", have_wgt_size ? wgt_size_s : "(none)");

    int64_t t_read0 = HAP_perf_get_time_us();  /* PROF W-P2: seg=read */
    size_t blob_len = 0, in_len = 0;
    uint8_t* blob = read_file(blob_p, &blob_len);
    uint8_t* in = read_file(in_p, &in_len);
    if (!blob || !in) { ex_log("[FAIL] read blob/input"); free(blob); free(in); return ex_summary() || 1; }
    ex_log("M5 blob %zu in %zu", blob_len, in_len);

    int64_t t_parse0 = HAP_perf_get_time_us();  /* PROF W-P2: seg=parse */
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w) { ex_log("[FAIL] wt_blob alloc"); free(blob); free(in); return ex_summary() || 1; }
    ex_log("M6 before wt_parse");
    if (wt_parse(blob, blob_len, w) != WT_OK) {
        ex_log("[FAIL] wt_parse"); free(blob); free(in); free(w);
        return ex_summary() || 1;
    }
    int64_t t_parse1 = HAP_perf_get_time_us();
    ex_log("M7 wt_parse ok");

    /* 路线B: 加载外部权重区 (model.weights.bin 与 .wtop 同名前缀) */
    static char wgt_p[256];
    memset(wgt_p, 0, sizeof(wgt_p));
    {
        const char* dot = strrchr(blob_p, '.');
        if (dot && strcmp(dot, ".wtop") == 0) {
            size_t base_len = (size_t)(dot - blob_p);
            if (base_len + sizeof(".weights.bin") <= sizeof(wgt_p)) {
                memcpy(wgt_p, blob_p, base_len);
                memcpy(wgt_p + base_len, ".weights.bin", sizeof(".weights.bin"));
            }
        } else {
            snprintf(wgt_p, sizeof(wgt_p), "%s.weights.bin", blob_p);
        }
    }
    uint8_t* wgt = NULL;
    size_t wgt_len = 0;
    ex_log("M8 before weights load");
    int64_t t_wgt0 = HAP_perf_get_time_us();  /* PROF W-P2: seg=weights */
    if (wgt_p[0]) {
        if (have_wgt_size) {
            wgt_len = (size_t)strtoul(wgt_size_s, NULL, 10);
            wgt = read_file_size(wgt_p, wgt_len);
            if (!wgt) ex_log("[ext-weights] read_file_size(%zu) failed (malloc/io?)", wgt_len);
        } else {
            wgt = read_file(wgt_p, &wgt_len);
        }
        if (wgt) {
            wt_exec_set_ext_weights(wgt);
            ex_log("M9 ext weights set");
            ex_log("[ext-weights] %s loaded (%zu bytes)", wgt_p, wgt_len);
        } else {
            ex_log("[ext-weights] no %s (fallback to inline weights)", wgt_p);
        }
    }

    static char err[128];
    memset(err, 0, sizeof(err));
    /* 逐 op 耗时表: wt_exec_run_io 填入 op_us[i](µs), 由 wt_exec_run_range
     * 内 HAP_perf_get_time_us 打点。测完按模板聚合打印 op_timing 行。 */
    int64_t* op_us = NULL;
    if (w->n_ops) op_us = (int64_t*)malloc(sizeof(int64_t) * w->n_ops);
    if (!op_us) ex_log("[timing] op_us alloc failed (%u ops), 走无计时路径", (unsigned)w->n_ops);
    /* PROF W-P2: 逐 op 时间戳对 (start/dur, join gehtp_prof.py 用) */
    struct wt_op_ts* op_ts = NULL;
    if (w->n_ops) op_ts = (struct wt_op_ts*)malloc(sizeof(*op_ts) * w->n_ops);
    if (op_ts) wt_exec_set_ts(op_ts, w->n_ops);
    else ex_log("[prof] op_ts alloc failed (%u ops), 无 ts 产物", (unsigned)w->n_ops);
    int64_t t0 = HAP_perf_get_time_us();
    ex_log("M10 before wt_exec_run_io");
    int rc = wt_exec_run_io(w, in, NULL, out_temp, NULL, op_us, err, sizeof(err));
    int64_t total_us = HAP_perf_get_time_us() - t0;
    if (rc) {
        ex_log("[FAIL] wt_exec_run_io rc=%d %s", rc, err);
        wt_exec_shutdown();
        free(blob); free(in); free(w); free(op_us); free(op_ts);
        return ex_summary() || 1;
    }

    ex_log("M11 exec done rc=%d total_us=%lld", rc, (long long)total_us);
    if (op_us) {
        /* per-opcode 聚合: 总耗时/调用数/max 单项 (找热点算子) */
        enum { MAX_AGG = 64 };
        int64_t agg_us[MAX_AGG]; uint32_t agg_n[MAX_AGG];
        int64_t agg_max[MAX_AGG];
        memset(agg_us, 0, sizeof(agg_us));
        memset(agg_n, 0, sizeof(agg_n));
        memset(agg_max, 0, sizeof(agg_max));
        int64_t sum = 0, top1 = 0; int i_top1 = -1;
        for (uint32_t i = 0; i < w->n_ops; i++) {
            uint32_t c = w->ops[i].opcode;
            if (c < MAX_AGG) {
                agg_us[c] += op_us[i]; agg_n[c]++;
                if (op_us[i] > agg_max[c]) agg_max[c] = op_us[i];
            }
            sum += op_us[i];
            if (op_us[i] > top1) { top1 = op_us[i]; i_top1 = (int)i; }
        }
        ex_log("[timing] ops=%u sum_op_us=%lld wall_us=%lld ops_per_sec=%.1f",
               (unsigned)w->n_ops, (long long)sum, (long long)total_us,
               total_us > 0 ? (double)w->n_ops * 1e6 / (double)total_us : 0.0);
        for (uint32_t c = 0; c < MAX_AGG; c++) {
            if (!agg_n[c]) continue;
            ex_log("op_timing code=%u n=%u total_us=%lld avg_us=%.1f max_us=%lld",
                   c, agg_n[c], (long long)agg_us[c],
                   (double)agg_us[c] / (double)agg_n[c], (long long)agg_max[c]);
        }
        if (i_top1 >= 0) {
            ex_log("[timing] slowest op#%d code=%u us=%lld", i_top1,
                   w->ops[i_top1].opcode, (long long)top1);
        }
    }
    /* PROF W-P2: op_ts.bin 落盘 (rc==0 才写; 失败 run 的 ts 无意义不产) */
    if (op_ts) {
        if (write_op_ts(OP_TS_PATH, op_ts, w->n_ops, (uint64_t)total_us) == 0)
            ex_log("[prof] op_ts.bin %u ops", (unsigned)w->n_ops);
        else
            ex_log("[prof] op_ts.bin write FAIL");
    }
    int64_t t_write0 = HAP_perf_get_time_us();  /* PROF W-P2: seg=write */
    uint32_t obytes = wt_exec_temp_last_bytes(out_temp);
    const uint8_t* optr = wt_exec_temp(out_temp);
    if (!obytes || !optr) {
        ex_log("[FAIL] out temp %u empty", (unsigned)out_temp);
        wt_exec_shutdown();
        free(blob); free(in); free(w); free(op_us); free(op_ts);
        return ex_summary() || 1;
    }
    int ofd = open(out_p, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (ofd < 0) { ex_log("[FAIL] open %s", out_p); wt_exec_shutdown(); free(blob); free(in); free(w); free(op_us); free(op_ts); return ex_summary() || 1; }
    size_t wgot = 0;
    while (wgot < obytes) {
        ssize_t n = write(ofd, optr + wgot, obytes - wgot);
        if (n <= 0) break;
        wgot += (size_t)n;
    }
    close(ofd);
    if (wgot != obytes) {
        ex_log("[FAIL] short write %s", out_p);
        wt_exec_shutdown();
        free(blob); free(in); free(w); free(op_us); free(op_ts);
        return ex_summary() || 1;
    }
    int64_t t_write1 = HAP_perf_get_time_us();
    ex_log("M12 wrote %u bytes", (unsigned)obytes);
    /* PROF W-P2: run 级结构化摘要 (分段拼轴: 读入/解析/权重/执行/写回) */
    ex_log("[run] seg_us read=%lld parse=%lld wgt=%lld exec=%lld write=%lld wall=%lld",
           (long long)(t_parse0 - t_read0), (long long)(t_parse1 - t_parse0),
           (long long)(t0 - t_wgt0), (long long)total_us,
           (long long)(t_write1 - t_write0),
           (long long)(t_write1 - t_read0));
    ex_log("[PASS] %u ops, out temp %u = %u bytes -> %s",
           (unsigned)w->n_ops, (unsigned)out_temp, (unsigned)obytes, out_p);

    wt_exec_shutdown();
    free(blob); free(in); free(w); free(wgt); free(op_ts);
    ex_log(bad ? "[FAIL] 42_gehtp_runner overall" : "[PASS] 42_gehtp_runner overall");
    return ex_summary() || bad;
}
