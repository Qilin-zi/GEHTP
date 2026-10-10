/*
 * 42_gehtp_persistent — 持久化 WTOP 推理 runner (设备常驻文件队列版)
 * =====================================================================
 * 架构修正 (004 T019, 2026-10-09): 原设计是 Unix socket server, 但 qurt
 * (DSP 侧) 无 socket 支持 (posix 层仅 fcntl/mqueue/pthread/semaphore/
 * signal, 无 sys/socket.h), 且 run_main_on_hexagon 生命周期 = 一次
 * FastRPC 调用内 main() 存活 —— socket 服务在 DSP 目标上编译都过不了。
 * 持久化语义改为落在 main() 的请求循环上:
 *   - blob 解析 + (可选)外部权重加载一次, KV/SSM 状态 g_exec 进程级常驻
 *     (T015-T018 契约: wt_exec_load_states init-once, run_io 可重入)
 *   - 轮询请求文件 req → 执行 → 写响应文件 resp, host 侧
 *     (scripts/persistent_runtime.py) 经 adb 文件往返驱动
 *
 * 协议帧 (与原 socket 协议同形, 仅传输层换文件):
 *   请求文件 (host push 后由本 runner unlink 认领):
 *     4B BE hdr_len + JSON 头 + 二进制输入体 (op=run 时 in_bytes 字节)
 *     JSON: {"op": "run|reset|ping|shutdown", "handle_id": "..",
 *            "want_out": 0|1, "position": p, "kv_position": k}
 *   响应文件 (runner 写, host 读后 unlink):
 *     4B BE len + JSON {"rc":0,"token":n,"steps":u,"step_us":u,...}
 *     want_out=1 时另写 out 文件 (f16 logits, 设备端路径)
 *
 * 配置 job.txt ("key value" 行, 同 42_gehtp_runner):
 *   blob <设备端 .wtop 绝对路径>      out_temp <输出 temp id>
 *   in_bytes <run 请求输入体字节数>   req/resp/out <各文件绝对路径>
 *   idle_exit_s <无请求自动退出秒数, 0=不退出>
 *   weights_size <可选, 外部权重字节数> (0.8B decode blob 内联 2GB 不用)
 *
 * 用法 (host 侧 scripts/persistent_runtime.py 封装):
 *   adb shell "cd $DEVROOT && ADSP_LIBRARY_PATH=$DEVROOT \
 *     CDSP_LIBRARY_PATH=$DEVROOT nohup ./run_main_on_hexagon 3 \
 *     gehtp_persistent.so > <persist_dir>/server.log 2>&1 &"
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <HAP_perf.h>
#include "hvxhmx_v23.h"
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include "qurt_timer.h"

/* 共享板防撞车: 默认独立于 $GDIR/job.txt 的 persist 目录 (跨会话隔离) */
#ifndef GEHTP_JOB_PATH
#define GEHTP_JOB_PATH "/data/local/tmp/hvxhmx23/gehtp_persist/job.txt"
#endif
#define JOB_PATH GEHTP_JOB_PATH

#define POLL_SLEEP_US    2000        /* 请求文件轮询间隔 */
#define REQ_STABLE_US    5000000     /* push 中途文件大小稳定等待上限 */

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
        if (want > (16u << 20)) want = 16u << 20;  /* qurt read 单次上限 */
        ssize_t n = read(fd, buf + got, want);
        if (n <= 0) break;
        got += (size_t)n;
    }
    close(fd);
    if (got != (size_t)sz) { free(buf); return NULL; }
    *out_len = (size_t)sz;
    return buf;
}

static uint8_t* read_file_size(const char* p, size_t sz) {
    if (sz == 0) return NULL;
    uint8_t* buf = malloc(sz);
    if (!buf) { ex_log("[rfs] malloc(%zu) FAIL", sz); return NULL; }
    int fd = open(p, O_RDONLY);
    if (fd < 0) { ex_log("[rfs] open %s FAIL", p); free(buf); return NULL; }
    size_t got = 0;
    while (got < sz) {
        size_t want = sz - got;
        if (want > (16u << 20)) want = 16u << 20;
        ssize_t n = read(fd, buf + got, want);
        if (n <= 0) break;
        got += (size_t)n;
    }
    close(fd);
    if (got != sz) { free(buf); return NULL; }
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

/* ---- 极简 JSON 取值 (host 侧 json.dumps 定形, 键序无关) ---- */
static int json_get_str(char* hdr, const char* key, const char** v, size_t* vn) {
    char pat[48];
    int pl = snprintf(pat, sizeof(pat), "\"%s\"", key);
    if (pl <= 0 || (size_t)pl >= sizeof(pat)) return -1;
    char* p = strstr(hdr, pat);
    if (!p) return -1;
    char* c = strchr(p + pl, ':');
    if (!c) return -1;
    char* s = strchr(c, '"');
    if (!s) return -1;
    char* e = strchr(s + 1, '"');
    if (!e) return -1;
    *v = s + 1;
    *vn = (size_t)(e - s - 1);
    return 0;
}

static int json_get_int(char* hdr, const char* key, long* out) {
    char pat[48];
    int pl = snprintf(pat, sizeof(pat), "\"%s\"", key);
    if (pl <= 0 || (size_t)pl >= sizeof(pat)) return -1;
    char* p = strstr(hdr, pat);
    if (!p) return -1;
    char* c = strchr(p + pl, ':');
    if (!c) return -1;
    *out = strtol(c + 1, NULL, 10);
    return 0;
}

static int str_is(const char* v, size_t vn, const char* lit) {
    size_t ln = strlen(lit);
    return vn == ln && memcmp(v, lit, ln) == 0;
}

/* JSON 字符串消毒 (err 文本可能带引号/控制字符) */
static void jesc(char* dst, size_t n, const char* src) {
    size_t o = 0;
    for (const char* p = src; *p && o + 1 < n; p++)
        dst[o++] = (*p >= 0x20 && *p != '"' && *p != '\\') ? *p : '_';
    dst[o] = 0;
}

/* f16 logits 面上取 argmax (并列取最低 idx, 与 numpy argmax 同语义) */
static int32_t argmax_f16(const uint8_t* p, uint32_t bytes) {
    const __fp16* a = (const __fp16*)p;
    uint32_t n = bytes / 2u;
    if (!n) return -1;
    int32_t bi = 0;
    float bv = (float)a[0];
    for (uint32_t i = 1; i < n; i++) {
        float v = (float)a[i];
        if (v > bv) { bv = v; bi = (int32_t)i; }
    }
    return bi;
}

static int write_resp(const char* path, const char* json) {
    size_t n = strlen(json);
    uint8_t hdr[4] = { (uint8_t)(n >> 24), (uint8_t)(n >> 16),
                       (uint8_t)(n >> 8), (uint8_t)n };
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) return -1;
    int ok = 0;
    if (write(fd, hdr, 4) != 4) ok = -1;
    if (!ok && write(fd, json, n) != (ssize_t)n) ok = -1;
    close(fd);
    return ok;
}

static int dump_out_file(const char* path, uint32_t out_temp) {
    uint32_t obytes = wt_exec_temp_last_bytes(out_temp);
    const uint8_t* optr = wt_exec_temp(out_temp);
    if (!obytes || !optr) return -1;
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) return -1;
    size_t got = 0;
    while (got < obytes) {
        ssize_t n = write(fd, optr + got, obytes - got);
        if (n <= 0) break;
        got += (size_t)n;
    }
    close(fd);
    return got == obytes ? 0 : -1;
}

/* 认领一个完整请求: 文件存在且大小 == 期望值才读, 读全后 unlink。
 * 返回 0=拿到请求(req/rlen 出参); 1=暂无(轮询继续); -1=坏请求(已写错误响应) */
static int take_request(const char* req_p, const char* resp_p, uint32_t in_bytes,
                        uint8_t** req, size_t* rlen) {
    int fd = open(req_p, O_RDONLY);
    if (fd < 0) return 1;
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    if (sz < 4) { close(fd); return 1; }
    uint8_t hb[4];
    if (read(fd, hb, 4) != 4) { close(fd); return 1; }
    uint32_t hdr_len = ((uint32_t)hb[0] << 24) | ((uint32_t)hb[1] << 16) |
                       ((uint32_t)hb[2] << 8) | (uint32_t)hb[3];
    close(fd);
    if (hdr_len == 0 || hdr_len > (1u << 20)) {
        unlink(req_p);
        write_resp(resp_p, "{\"rc\":-1,\"err\":\"bad_hdr_len\"}");
        return -1;
    }
    /* 等 header 到齐后按 op 定期望总长 (run 带输入体, 其余裸) */
    uint32_t want = 4u + hdr_len;
    int64_t t0 = HAP_perf_get_time_us();
    char* hdr = NULL;
    for (;;) {
        fd = open(req_p, O_RDONLY);
        if (fd < 0) return 1;   /* host 撤回了请求 */
        sz = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        if (sz >= (off_t)want) {
            hdr = malloc(want + 1);
            /* 跳过 4B 长度前缀再读: 前缀以 00 开头, 混入会让 strstr 见空串 */
            lseek(fd, 4, SEEK_SET);
            if (hdr && read(fd, hdr, hdr_len) == (ssize_t)hdr_len) {
                hdr[want] = 0;
                close(fd);
                break;
            }
            free(hdr);
            close(fd);
            unlink(req_p);
            write_resp(resp_p, "{\"rc\":-1,\"err\":\"hdr_read_fail\"}");
            return -1;
        }
        close(fd);
        if (HAP_perf_get_time_us() - t0 > REQ_STABLE_US) {
            unlink(req_p);
            write_resp(resp_p, "{\"rc\":-1,\"err\":\"req_timeout\"}");
            return -1;
        }
        qurt_timer_sleep(POLL_SLEEP_US);
    }
    const char* opv; size_t opn;
    if (json_get_str(hdr, "op", &opv, &opn) == 0 && str_is(opv, opn, "run"))
        want += in_bytes;
    free(hdr);
    /* 等文件长到期望值 (adb push 增量写) */
    t0 = HAP_perf_get_time_us();
    for (;;) {
        fd = open(req_p, O_RDONLY);
        if (fd < 0) return 1;
        sz = lseek(fd, 0, SEEK_END);
        if (sz == (off_t)want) {
            lseek(fd, 0, SEEK_SET);
            uint8_t* buf = malloc(want);
            if (!buf) { close(fd); unlink(req_p);
                        write_resp(resp_p, "{\"rc\":-1,\"err\":\"oom\"}"); return -1; }
            size_t got = 0;
            while (got < want) {
                ssize_t n = read(fd, buf + got, want - got);
                if (n <= 0) break;
                got += (size_t)n;
            }
            close(fd);
            unlink(req_p);      /* 认领: host 见文件消失即知已受理 */
            if (got != want) {
                free(buf);
                write_resp(resp_p, "{\"rc\":-1,\"err\":\"body_read_fail\"}");
                return -1;
            }
            *req = buf;
            *rlen = want;
            return 0;
        }
        if (sz > (off_t)want) { /* 超长 = 帧错 */
            close(fd);
            unlink(req_p);
            write_resp(resp_p, "{\"rc\":-1,\"err\":\"req_oversize\"}");
            return -1;
        }
        close(fd);
        if (HAP_perf_get_time_us() - t0 > REQ_STABLE_US) {
            unlink(req_p);
            write_resp(resp_p, "{\"rc\":-1,\"err\":\"body_timeout\"}");
            return -1;
        }
        qurt_timer_sleep(POLL_SLEEP_US);
    }
}

/* 返回 1 = 收到 shutdown, 主循环退出 */
static int handle_request(struct wt_blob* w, uint8_t* req, size_t rlen,
                          uint32_t out_temp, const char* resp_p, const char* out_p) {
    uint32_t hdr_len = ((uint32_t)req[0] << 24) | ((uint32_t)req[1] << 16) |
                       ((uint32_t)req[2] << 8) | (uint32_t)req[3];
    char* hdr = malloc(hdr_len + 1);
    if (!hdr) return 0;
    memcpy(hdr, req + 4, hdr_len);
    hdr[hdr_len] = 0;

    const char* opv; size_t opn;
    int have_op = json_get_str(hdr, "op", &opv, &opn);
    long want_out = 0;
    json_get_int(hdr, "want_out", &want_out);

    int stop = 0;
    if (have_op == 0 && str_is(opv, opn, "ping")) {
        write_resp(resp_p, "{\"rc\":0,\"pong\":1}");
    } else if (have_op == 0 && str_is(opv, opn, "reset")) {
        wt_exec_state_reset();
        ex_log("[reset] state cleared");
        write_resp(resp_p, "{\"rc\":0,\"reset\":1}");
    } else if (have_op == 0 && str_is(opv, opn, "shutdown")) {
        write_resp(resp_p, "{\"rc\":0,\"bye\":1}");
        stop = 1;
    } else if (have_op == 0 && str_is(opv, opn, "run")) {
        static uint32_t g_steps = 0;
        size_t body_off = 4u + hdr_len;
        if (rlen < body_off) {
            write_resp(resp_p, "{\"rc\":-1,\"err\":\"short_body\"}");
            free(hdr);
            return 0;
        }
        static char err[128];
        memset(err, 0, sizeof(err));
        /* 输入体对齐拷贝: req+body_off 的对齐随 JSON 头长度漂移, 而引擎侧
         * HVX 消费要求对齐 (runner 喂 malloc 缓冲=8B 对齐从不出事; 实锤:
         * 短头("dbg", off%8=4)能跑, 长头 gate/matrix 帧全数死在 run_io 内) */
        static uint8_t in_body[4096] __attribute__((aligned(128)));
        if (rlen - body_off > sizeof(in_body)) {
            write_resp(resp_p, "{\"rc\":-1,\"err\":\"body_too_big\"}");
            free(hdr);
            return 0;
        }
        memcpy(in_body, req + body_off, rlen - body_off);
        int64_t t0 = HAP_perf_get_time_us();
        ex_log("[run] exec begin (want_out=%d)", (int)want_out);
        int rc = wt_exec_run_io(w, in_body, NULL, out_temp,
                                NULL, NULL, err, sizeof(err));
        int64_t us = HAP_perf_get_time_us() - t0;
        ex_log("[run] exec done rc=%d us=%lld", rc, (long long)us);
        char resp[256];
        if (rc != 0) {
            char esc[96];
            jesc(esc, sizeof(esc), err);
            snprintf(resp, sizeof(resp), "{\"rc\":%d,\"err\":\"%s\"}", rc, esc);
            ex_log("[run] FAIL rc=%d err=%s", rc, err);
        } else {
            int32_t tok = argmax_f16(wt_exec_temp(out_temp),
                                     wt_exec_temp_last_bytes(out_temp));
            ex_log("[run] argmax tok=%d", (int)tok);
            if (want_out) {
                if (dump_out_file(out_p, out_temp) != 0)
                    ex_log("[warn] dump out failed (non-fatal)");
                else
                    ex_log("[run] dumped out");
            }
            g_steps++;
            snprintf(resp, sizeof(resp),
                     "{\"rc\":0,\"token\":%d,\"steps\":%u,\"step_us\":%lld}",
                     (int)tok, (unsigned)g_steps, (long long)us);
        }
        write_resp(resp_p, resp);
    } else {
        write_resp(resp_p, "{\"rc\":-1,\"err\":\"bad_op\"}");
    }
    free(hdr);
    return stop;
}

int main(void) {
    ex_open_result("42_gehtp_persistent");
    ex_log("=== 42_gehtp_persistent (file-queue persistent runner) ===");

    size_t job_len = 0;
    uint8_t* job = read_file(JOB_PATH, &job_len);
    if (!job) { ex_log("[FAIL] read %s", JOB_PATH); return ex_summary() || 1; }

    char blob_p[256], req_p[256], resp_p[256], out_p[256], temp_s[32], inb_s[32];
    char idle_s[32], wgt_size_s[32], trace_s[32];
    memset(blob_p, 0, sizeof(blob_p)); memset(req_p, 0, sizeof(req_p));
    memset(resp_p, 0, sizeof(resp_p)); memset(out_p, 0, sizeof(out_p));
    memset(temp_s, 0, sizeof(temp_s)); memset(inb_s, 0, sizeof(inb_s));
    memset(idle_s, 0, sizeof(idle_s)); memset(wgt_size_s, 0, sizeof(wgt_size_s));
    memset(trace_s, 0, sizeof(trace_s));
    if (job_get((char*)job, "blob", blob_p, sizeof(blob_p)) ||
        job_get((char*)job, "req", req_p, sizeof(req_p)) ||
        job_get((char*)job, "resp", resp_p, sizeof(resp_p)) ||
        job_get((char*)job, "out", out_p, sizeof(out_p)) ||
        job_get((char*)job, "out_temp", temp_s, sizeof(temp_s)) ||
        job_get((char*)job, "in_bytes", inb_s, sizeof(inb_s))) {
        ex_log("[FAIL] job.txt 缺键 (blob/req/resp/out/out_temp/in_bytes)");
        free(job);
        return ex_summary() || 1;
    }
    uint32_t out_temp = (uint32_t)strtoul(temp_s, NULL, 10);
    uint32_t in_bytes = (uint32_t)strtoul(inb_s, NULL, 10);
    long idle_exit_s = job_get((char*)job, "idle_exit_s", idle_s, sizeof(idle_s)) == 0
                       ? strtol(idle_s, NULL, 10) : 600;
    // optrace 默认关: 服务态逐 op fopen+append 一步 16MB+ 文件 IO, 步时 2.5×
    int trace_on = job_get((char*)job, "trace", trace_s, sizeof(trace_s)) == 0
                   ? (int)strtol(trace_s, NULL, 10) : 0;
    int have_wgt = job_get((char*)job, "weights_size", wgt_size_s, sizeof(wgt_size_s)) == 0;
    free(job);
    ex_log("[cfg] blob=%s out_temp=%u in_bytes=%u idle=%lds%s",
           blob_p, (unsigned)out_temp, (unsigned)in_bytes, idle_exit_s,
           have_wgt ? " (ext-weights)" : "");

    unlink(req_p);
    unlink(resp_p);

    ex_log("M5 reading blob...");
    size_t blob_len = 0;
    uint8_t* blob = read_file(blob_p, &blob_len);
    if (!blob) { ex_log("[FAIL] read blob"); return ex_summary() || 1; }
    ex_log("M6 blob %zu bytes", blob_len);

    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w) { ex_log("[FAIL] wt_blob alloc"); free(blob); return ex_summary() || 1; }
    if (wt_parse(blob, blob_len, w) != WT_OK) {
        ex_log("[FAIL] wt_parse");
        free(blob); free(w);
        return ex_summary() || 1;
    }
    ex_log("M7 wt_parse ok (slots=%u ops=%u)", (unsigned)w->n_slots, (unsigned)w->n_ops);
    wt_exec_set_trace(trace_on);

    if (have_wgt) {
        size_t wgt_len = (size_t)strtoul(wgt_size_s, NULL, 10);
        static char wgt_p[256];
        const char* dot = strrchr(blob_p, '.');
        if (dot && strcmp(dot, ".wtop") == 0) {
            size_t base_len = (size_t)(dot - blob_p);
            if (base_len + sizeof(".weights.bin") <= sizeof(wgt_p)) {
                memcpy(wgt_p, blob_p, base_len);
                memcpy(wgt_p + base_len, ".weights.bin", sizeof(".weights.bin"));
            } else wgt_p[0] = 0;
        } else {
            snprintf(wgt_p, sizeof(wgt_p), "%s.weights.bin", blob_p);
        }
        if (wgt_p[0]) {
            ex_log("M8 loading ext weights (%zu B)...", wgt_len);
            uint8_t* wgt = read_file_size(wgt_p, wgt_len);
            if (wgt) {
                wt_exec_set_ext_weights(wgt);
                ex_log("M9 ext weights resident (%s)", wgt_p);
            } else {
                ex_log("[FAIL] ext weights read");
                free(blob); free(w);
                return ex_summary() || 1;
            }
        }
    }

    ex_log("[ready] serving req=%s", req_p);
    int64_t t_last = HAP_perf_get_time_us();
    int stop = 0, served = 0;
    while (!stop) {
        if (idle_exit_s > 0 &&
            HAP_perf_get_time_us() - t_last > idle_exit_s * 1000000) {
            ex_log("[idle] %lds no request, exit", idle_exit_s);
            break;
        }
        uint8_t* req = NULL;
        size_t rlen = 0;
        int r = take_request(req_p, resp_p, in_bytes, &req, &rlen);
        if (r == 0) {
            t_last = HAP_perf_get_time_us();
            served++;
            stop = handle_request(w, req, rlen, out_temp, resp_p, out_p);
            free(req);
        } else if (r < 0) {
            t_last = HAP_perf_get_time_us();
            served++;
        } else {
            qurt_timer_sleep(POLL_SLEEP_US);
        }
    }

    wt_exec_shutdown();
    free(blob);
    free(w);
    ex_log("[PASS] server stopped, served %d requests", served);
    return ex_summary();
}
