/*
 * gehtp_proxy.c — 设备 AP 侧常驻代理: TCP/HTTP ↔ DSP 文件队列 (req.bin/resp.bin)
 * =====================================================================
 * 动机: host 每步 spawn adb (~45ms 地板); 本代理在板上常驻, host 走
 * adb forward 后的 TCP 长连接 (或局域网直连), 单步传输 ~5-10ms。
 * DSP runner 协议零改动。
 *
 * 两个监听 (同时开):
 *   --bridge-port 9820  v1 桥: 帧协议 = [4B total][4B jlen][json][body]
 *       request 帧的 payload 即 req.bin 文件原样 (4B jlen + json + body);
 *       response 为 resp.bin 文件原样 (4B jlen + json)。total = payload 字节数。
 *       与 scripts/persistent_runtime.py 文件队列同语义 (op 全透传)。
 *   --http-port   9821  v2 设备侧 HTTP (token 级; 分词留 host):
 *       GET  /health
 *       POST /v1/sessions  {persist_dir, blob, out_temp, in_bytes, kv_max,
 *                           face_b64, so_name?, devroot?, handle_id?, idle_exit_s?}
 *                           → 拉起/复用 runner, 存 face 表 (positions×in_bytes-4)
 *       POST /v1/decode    {prompt_ids:[...], max_new, stream?, reset?}
 *                           → stream:1 SSE data:{"token":..}; 否则 JSON tokens[]
 *       POST /v1/shutdown  → runner 优雅退出 + 会话清空
 *   单会话 (runner 单租户, 队列全局互斥); 会话队列目录切换会同步影响桥监听。
 *
 * 构建: build_ap.sh (NDK r25c aarch64, -static)
 * 部署: push <DEVROOT>/gehtp_proxy; setsid nohup ./gehtp_proxy --dir <queue> &
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define BRIDGE_PORT_D 9820
#define HTTP_PORT_D   9821
#define XFER_TIMEOUT_S 120
#define READY_TIMEOUT_S 300
#define MAX_BODY (8u << 20)
#define MAX_PROMPT 1024
#define MAX_NEW_CAP 1024

static char g_dir[256] = "/data/local/tmp/hvxhmx23/gehtp_persist";
static pthread_mutex_t g_mtx = PTHREAD_MUTEX_INITIALIZER;

static void lg(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stdout, "[proxy] ");
    vfprintf(stdout, fmt, ap);
    fprintf(stdout, "\n");
    fflush(stdout);
    va_end(ap);
}

static void set_nodelay(int fd);

/* ---------- io helpers ---------- */
static int write_full(int fd, const void* p, size_t n) {
    const uint8_t* b = p;
    while (n) {
        ssize_t w = send(fd, b, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        b += w; n -= (size_t)w;
    }
    return 0;
}
static int read_full(int fd, void* p, size_t n) {
    uint8_t* b = p;
    while (n) {
        ssize_t r = recv(fd, b, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        b += r; n -= (size_t)r;
    }
    return 0;
}
static int recv_u32(int fd, uint32_t* v) {
    uint8_t b[4];
    if (read_full(fd, b, 4)) return -1;
    *v = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
    return 0;
}
static int send_u32(int fd, uint32_t v) {
    uint8_t b[4] = { (uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v };
    return write_full(fd, b, 4);
}

/* ---------- 文件队列事务: req(4B jlen+json+body) → resp(4B jlen+json) ---------- */
/* 调用方持 g_mtx。0=ok, -1=io 错, -2=超时 */
static int queue_xfer(const uint8_t* req, size_t rlen, uint8_t** out, size_t* olen,
                      int timeout_s) {
    char req_p[320], resp_p[320], tmp_p[320];
    snprintf(req_p, sizeof(req_p), "%s/req.bin", g_dir);
    snprintf(resp_p, sizeof(resp_p), "%s/resp.bin", g_dir);
    snprintf(tmp_p, sizeof(tmp_p), "%s/req.tmp", g_dir);
    unlink(resp_p);

    int fd = open(tmp_p, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) return -1;
    size_t off = 0;
    while (off < rlen) {
        ssize_t w = write(fd, req + off, rlen - off);
        if (w <= 0) { close(fd); unlink(tmp_p); return -1; }
        off += (size_t)w;
    }
    close(fd);
    if (rename(tmp_p, req_p) != 0) { unlink(tmp_p); return -1; }

    struct timespec tw0, tw1;
    clock_gettime(CLOCK_MONOTONIC, &tw0);
    time_t t0 = time(NULL);
    for (;;) {
        if (time(NULL) - t0 > timeout_s) { lg("xfer TIMEOUT"); return -2; }
        fd = open(resp_p, O_RDONLY);
        if (fd >= 0) {
            off_t sz = lseek(fd, 0, SEEK_END);
            if (sz >= 4) {
                lseek(fd, 0, SEEK_SET);
                uint8_t hb[4];
                if (read(fd, hb, 4) == 4) {
                    uint32_t jl = ((uint32_t)hb[0] << 24) | ((uint32_t)hb[1] << 16) |
                                  ((uint32_t)hb[2] << 8) | hb[3];
                    if (sz >= (off_t)(4 + jl) && jl > 0 && jl < (1u << 20)) {
                        uint8_t* buf = malloc(4 + jl);
                        if (buf) {
                            lseek(fd, 0, SEEK_SET);
                            if (read(fd, buf, 4 + jl) == (ssize_t)(4 + jl)) {
                                close(fd);
                                unlink(resp_p);
                                clock_gettime(CLOCK_MONOTONIC, &tw1);
                                lg("xfer %ldms (payload %zuB)",
                                   (long)((tw1.tv_sec - tw0.tv_sec) * 1000 +
                                          (tw1.tv_nsec - tw0.tv_nsec) / 1000000),
                                   rlen);
                                *out = buf;
                                *olen = 4 + jl;
                                return 0;
                            }
                            free(buf);
                        }
                    }
                }
            }
            close(fd);
        }
        usleep(2000);
    }
}

/* ---------- 最小 json 取值 (自产 json 无嵌套/转义, 够用) ---------- */
static const char* jfind(const char* js, const char* key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(js, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ' || *p == '\n' || *p == '\t') p++;
    return p;
}
static long long jget_i(const char* js, const char* key, long long dflt) {
    const char* p = jfind(js, key);
    if (!p || *p != ':') return dflt;
    return strtoll(p + 1, NULL, 10);
}
static char* jget_s(const char* js, const char* key) {
    const char* p = jfind(js, key);
    if (!p || *p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\n' || *p == '\t') p++;
    if (*p != '"') return NULL;
    const char* e = strchr(p + 1, '"');
    if (!e) return NULL;
    size_t n = (size_t)(e - p - 1);
    char* s = malloc(n + 1);
    memcpy(s, p + 1, n);
    s[n] = 0;
    return s;
}
static uint8_t* jget_b64(const char* js, const char* key, size_t* out_n) {
    char* s = jget_s(js, key);
    if (!s) return NULL;
    size_t n = strlen(s);
    uint8_t* out = malloc((n / 4) * 3 + 3);
    size_t o = 0;
    int q[4], qi = 0, npad = 0;
    for (size_t i = 0; i < n; i++) {
        int v;
        char c = s[i];
        if (c >= 'A' && c <= 'Z') v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+') v = 62;
        else if (c == '/') v = 63;
        else if (c == '=') { v = 0; npad++; }
        else continue;
        q[qi++] = v;
        if (qi == 4) {
            out[o++] = (uint8_t)((q[0] << 2) | (q[1] >> 4));
            if (npad < 2) out[o++] = (uint8_t)((q[1] << 4) | (q[2] >> 2));
            if (npad < 1) out[o++] = (uint8_t)((q[2] << 6) | q[3]);
            qi = 0;
            npad = 0;
        }
    }
    free(s);
    *out_n = o;
    return out;
}

/* ---------- 会话 (单租户) ---------- */
typedef struct {
    int used;
    char so_name[128];
    char handle_id[128];
    uint32_t out_temp, in_bytes;
    int kv_max;
    uint8_t* face;        /* positions × face_bytes */
    size_t face_bytes;    /* in_bytes - 4 */
    int face_positions;
    int dirty;            /* 已 decode 过 → 下次先 reset */
} session_t;
static session_t g_sess;

static void build_req(uint8_t** out, size_t* olen, const char* json, const uint8_t* body,
                      size_t blen) {
    size_t jl = strlen(json);
    uint8_t* buf = malloc(4 + jl + blen);
    buf[0] = (uint8_t)(jl >> 24); buf[1] = (uint8_t)(jl >> 16);
    buf[2] = (uint8_t)(jl >> 8); buf[3] = (uint8_t)jl;
    memcpy(buf + 4, json, jl);
    if (blen) memcpy(buf + 4 + jl, body, blen);
    *out = buf;
    *olen = 4 + jl + blen;
}
static int op_bare(const char* json, uint8_t** out, size_t* olen, int timeout_s) {
    uint8_t* req = NULL; size_t rl = 0;
    build_req(&req, &rl, json, NULL, 0);
    int r = queue_xfer(req, rl, out, olen, timeout_s);
    free(req);
    return r;
}
static int resp_field_i(const uint8_t* resp, size_t n, const char* key, long long* val) {
    if (n < 4) return -1;
    size_t jl = ((size_t)resp[0] << 24) | ((size_t)resp[1] << 16) |
                ((size_t)resp[2] << 8) | resp[3];
    if (jl > n - 4 || jl > 65535) return -1;
    char tmp[65536 + 1];
    memcpy(tmp, resp + 4, jl);
    tmp[jl] = 0;
    const char* p = jfind(tmp, key);
    if (!p || *p != ':') return -1;
    *val = strtoll(p + 1, NULL, 10);
    return 0;
}
static int resp_ok(const uint8_t* resp, size_t n) {
    long long rc = 1;
    if (resp_field_i(resp, n, "rc", &rc) != 0) return 0;
    return rc == 0;
}

static int runner_alive(void) {
    if (!g_sess.so_name[0]) return 0;
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "pgrep -f '[%c]%s' >/dev/null 2>&1",
             g_sess.so_name[0], g_sess.so_name + 1);
    return system(cmd) == 0;
}

static int boot_runner(const char* persist_dir, const char* devroot, const char* so_name,
                       const char* blob, uint32_t out_temp, uint32_t in_bytes,
                       long idle_s) {
    char job_p[300], log_p[300];
    snprintf(job_p, sizeof(job_p), "%s/job.txt", persist_dir);
    snprintf(log_p, sizeof(log_p), "%s/server.log", persist_dir);
    FILE* f = fopen(job_p, "w");
    if (!f) return -1;
    fprintf(f, "blob %s\nout_temp %u\nin_bytes %u\n", blob, out_temp, in_bytes);
    fprintf(f, "req %s/req.bin\nresp %s/resp.bin\nout %s/out.f16.raw\n",
            persist_dir, persist_dir, persist_dir);
    fprintf(f, "idle_exit_s %ld\n", idle_s);
    fclose(f);

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "cd %s && ADSP_LIBRARY_PATH=%s CDSP_LIBRARY_PATH=%s "
             "setsid nohup ./run_main_on_hexagon 3 %s > %s 2>&1 < /dev/null &",
             devroot, devroot, devroot, so_name, log_p);
    if (system(cmd) != 0) return -1;

    /* 等 ping 就绪 (blob 读入 + wt_parse; 内联 2GB blob 需数分钟) */
    time_t t0 = time(NULL);
    while (time(NULL) - t0 < READY_TIMEOUT_S) {
        uint8_t* resp = NULL; size_t rn = 0;
        if (op_bare("{\"op\":\"ping\"}", &resp, &rn, 10) == 0) {
            int ok = resp_ok(resp, rn);
            free(resp);
            if (ok) return 0;
        }
        usleep(200000);
    }
    return -2;
}

static void shutdown_runner(void) {
    uint8_t* resp = NULL; size_t rn = 0;
    if (op_bare("{\"op\":\"shutdown\"}", &resp, &rn, 30) == 0) free(resp);
    time_t t0 = time(NULL);
    while (time(NULL) - t0 < 10 && runner_alive()) usleep(200000);
}

/* 单步: token @position → argmax token; <0=err (-2=timeout)。调用方持 g_mtx */
static int step_run(int token, int position, int timeout_s) {
    size_t fb = g_sess.face_bytes;
    uint8_t* body = malloc(4 + fb);
    uint32_t t = (uint32_t)token;
    body[0] = (uint8_t)t; body[1] = (uint8_t)(t >> 8);
    body[2] = (uint8_t)(t >> 16); body[3] = (uint8_t)(t >> 24);
    memcpy(body + 4, g_sess.face + (size_t)position * fb, fb);

    char json[288];
    snprintf(json, sizeof(json),
             "{\"op\":\"run\",\"handle_id\":\"%s\",\"position\":%d,"
             "\"kv_position\":%d,\"want_out\":0}",
             g_sess.handle_id, position, position);
    uint8_t* req = NULL; size_t rl = 0;
    build_req(&req, &rl, json, body, 4 + fb);
    free(body);
    uint8_t* resp = NULL; size_t rn = 0;
    int r = queue_xfer(req, rl, &resp, &rn, timeout_s);
    free(req);
    if (r != 0) return r;
    long long tok = -1;
    if (resp_field_i(resp, rn, "token", &tok) != 0) tok = -1;
    free(resp);
    return (int)tok;
}

/* ---------- v1 桥线程: 帧透传 ---------- */
static void* bridge_main(void* arg) {
    int lsn = (int)(intptr_t)arg;
    lg("bridge listening");
    for (;;) {
        int c = accept(lsn, NULL, NULL);
        if (c < 0) continue;
        set_nodelay(c);
        lg("bridge conn");
        for (;;) {
            uint32_t total;
            if (recv_u32(c, &total)) break;
            if (total < 4 || total > MAX_BODY) break;
            uint8_t* payload = malloc(total);
            if (!payload) break;
            if (read_full(c, payload, total)) { free(payload); break; }
            uint8_t* resp = NULL; size_t rn = 0;
            pthread_mutex_lock(&g_mtx);
            int r = queue_xfer(payload, total, &resp, &rn, XFER_TIMEOUT_S);
            pthread_mutex_unlock(&g_mtx);
            free(payload);
            if (r == 0) {
                uint8_t* frame = malloc(4 + rn);
                frame[0] = (uint8_t)(rn >> 24); frame[1] = (uint8_t)(rn >> 16);
                frame[2] = (uint8_t)(rn >> 8); frame[3] = (uint8_t)rn;
                memcpy(frame + 4, resp, rn);
                int ok = write_full(c, frame, 4 + rn) == 0;
                free(frame); free(resp);
                if (ok) continue;
                break;
            }
            free(resp);
            char err[80];
            int n = snprintf(err, sizeof(err), "{\"rc\":-1,\"err\":\"proxy_%s\"}",
                             r == -2 ? "timeout" : "queue_fail");
            uint8_t frame[4 + 80];
            size_t fl = 4 + (size_t)n;
            uint32_t jl = (uint32_t)n;
            frame[0] = (uint8_t)(jl >> 24); frame[1] = (uint8_t)(jl >> 16);
            frame[2] = (uint8_t)(jl >> 8); frame[3] = (uint8_t)jl;
            memcpy(frame + 4, err, (size_t)n);
            send_u32(c, (uint32_t)fl);
            write_full(c, frame, fl);
            break;
        }
        close(c);
        lg("bridge conn closed");
    }
    return NULL;
}

/* ---------- v2 HTTP ---------- */
static void http_resp(int c, int code, const char* ctype, const char* body) {
    char hdr[256];
    const char* st = code == 200 ? "OK" : code == 404 ? "Not Found"
                    : code == 409 ? "Conflict" : code == 400 ? "Bad Request"
                    : code == 500 ? "Server Error" : "Error";
    int hn = snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 %d %s\r\nContent-Type: %s\r\n"
                      "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                      code, st, ctype, strlen(body));
    write_full(c, hdr, (size_t)hn);
    write_full(c, body, strlen(body));
}
static void sse_start(int c) {
    static const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                           "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    write_full(c, h, strlen(h));
}
static void sse_event(int c, const char* json) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf), "data: %s\n\n", json);
    write_full(c, buf, (size_t)n);
}

static void handle_sessions(int c, uint8_t* body) {
    char* persist_dir = jget_s((char*)body, "persist_dir");
    char* devroot = jget_s((char*)body, "devroot");
    char* so_name = jget_s((char*)body, "so_name");
    char* blob = jget_s((char*)body, "blob");
    char* handle = jget_s((char*)body, "handle_id");
    size_t face_n = 0;
    uint8_t* face = jget_b64((char*)body, "face_b64", &face_n);
    uint32_t out_temp = (uint32_t)jget_i((char*)body, "out_temp", 0);
    uint32_t in_bytes = (uint32_t)jget_i((char*)body, "in_bytes", 0);
    long kv_max = jget_i((char*)body, "kv_max", 0);
    long idle = jget_i((char*)body, "idle_exit_s", 1800);

    char resp[256];
    resp[0] = 0;
    if (!persist_dir || !blob || !face || in_bytes < 5) {
        http_resp(c, 400, "application/json",
                  "{\"error\":\"need persist_dir,blob,face_b64,in_bytes\"}");
        goto out;
    }
    if (!devroot) devroot = strdup("/data/local/tmp/hvxhmx23");
    if (!so_name) so_name = strdup("gehtp_persistent.so");
    if (!handle) handle = strdup("persist_http");

    pthread_mutex_lock(&g_mtx);
    if (g_sess.used && strcmp(g_dir, persist_dir) != 0) {
        /* 换队列目录 = 换模型: 必须先 shutdown (单租户) */
        http_resp(c, 409, "application/json",
                  "{\"error\":\"other model session active; POST /v1/shutdown first\"}");
        pthread_mutex_unlock(&g_mtx);
        goto out;
    }
    snprintf(g_sess.so_name, sizeof(g_sess.so_name), "%s", so_name);
    snprintf(g_dir, sizeof(g_dir), "%s", persist_dir);
    int need_boot = 0;
    uint8_t* pr = NULL; size_t pn = 0;
    if (op_bare("{\"op\":\"ping\"}", &pr, &pn, 5) != 0 || !resp_ok(pr, pn))
        need_boot = 1;
    free(pr);
    if (need_boot) {
        lg("booting runner (%s via %s)", so_name, devroot);
        int r = boot_runner(persist_dir, devroot, so_name, blob, out_temp,
                            in_bytes, idle);
        if (r != 0) {
            pthread_mutex_unlock(&g_mtx);
            http_resp(c, 500, "application/json",
                      "{\"error\":\"runner boot failed or not ready in 300s\"}");
            goto out;
        }
    }
    free(g_sess.face);
    g_sess.face = face; face = NULL;
    g_sess.face_bytes = in_bytes - 4;
    g_sess.face_positions = (int)(face_n / g_sess.face_bytes);
    g_sess.out_temp = out_temp;
    g_sess.in_bytes = in_bytes;
    g_sess.kv_max = (int)kv_max;
    snprintf(g_sess.handle_id, sizeof(g_sess.handle_id), "%s", handle);
    g_sess.used = 1;
    g_sess.dirty = 0;
    int positions = g_sess.face_positions;
    pthread_mutex_unlock(&g_mtx);

    snprintf(resp, sizeof(resp),
             "{\"handle_id\":\"%s\",\"booted_runner\":%s,\"max_positions\":%d}",
             handle, need_boot ? "true" : "false", positions);
    http_resp(c, 200, "application/json", resp);
out:
    free(persist_dir); free(devroot); free(so_name); free(blob); free(handle);
    free(face);
}

static void handle_decode(int c, uint8_t* body) {
    long max_new = jget_i((char*)body, "max_new", 16);
    long stream = jget_i((char*)body, "stream", 0);
    long reset = jget_i((char*)body, "reset", 1);
    long prompt[MAX_PROMPT];
    int np = 0;
    const char* p = jfind((char*)body, "prompt_ids");
    if (p && *p == ':' && (p = strchr(p + 1, '['))) {
        char* end;
        while (*p && *p != ']' && np < MAX_PROMPT) {
            long v = strtol(p + 1, &end, 10);
            if (end == p + 1) { p = strchr(p + 1, ','); if (!p) break; continue; }
            prompt[np++] = v;
            p = end;
        }
    }
    pthread_mutex_lock(&g_mtx);
    if (!g_sess.used || np == 0) {
        pthread_mutex_unlock(&g_mtx);
        http_resp(c, 400, "application/json",
                  "{\"error\":\"no session or empty prompt_ids\"}");
        return;
    }
    if (max_new < 1) max_new = 1;
    if (max_new > MAX_NEW_CAP) max_new = MAX_NEW_CAP;
    if (np + max_new > g_sess.face_positions) {
        pthread_mutex_unlock(&g_mtx);
        http_resp(c, 400, "application/json",
                  "{\"error\":\"np+max_new exceeds face table positions\"}");
        return;
    }
    if (g_sess.dirty && reset) {
        char json[192];
        snprintf(json, sizeof(json), "{\"op\":\"reset\",\"handle_id\":\"%s\"}",
                 g_sess.handle_id);
        uint8_t* rr = NULL; size_t rn = 0;
        op_bare(json, &rr, &rn, 30);
        free(rr);
    }
    if (stream) sse_start(c);

    int pos = 0, tok = -1, failed = 0;
    static int gen[MAX_NEW_CAP];       /* 单线程 decode (持锁), static 安全 */
    int ng = 0;
    for (int i = 0; i < np && !failed; i++, pos++) {
        tok = step_run((int)prompt[i], pos, XFER_TIMEOUT_S);
        if (tok < 0) failed = 1;
    }
    for (int i = 0; i < max_new && !failed; i++, pos++) {
        if (stream) {
            char ev[96];
            snprintf(ev, sizeof(ev), "{\"token\":%d,\"position\":%d}", tok, pos - 1);
            sse_event(c, ev);
        }
        gen[ng++] = tok;
        int nt = step_run(tok, pos, XFER_TIMEOUT_S);
        if (nt < 0) { failed = 1; break; }
        tok = nt;
    }
    g_sess.dirty = 1;
    pthread_mutex_unlock(&g_mtx);

    if (failed) {
        const char* e = "{\"error\":\"runner step failed or timeout\"}";
        if (stream) sse_event(c, e);
        else http_resp(c, 500, "application/json", e);
    } else if (stream) {
        char ev[64];
        snprintf(ev, sizeof(ev), "{\"done\":true,\"n\":%d}", ng);
        sse_event(c, ev);
    } else {
        char b[8192];
        int off = snprintf(b, sizeof(b), "{\"tokens\":[");
        for (int i = 0; i < ng; i++)
            off += snprintf(b + off, sizeof(b) - (size_t)off, "%s%d",
                            i ? "," : "", gen[i]);
        off += snprintf(b + off, sizeof(b) - (size_t)off, "],\"n\":%d}", ng);
        http_resp(c, 200, "application/json", b);
    }
}

static void handle_http(int c, char* hdr, uint8_t* body) {
    char method[8] = {0}, path[128] = {0};
    sscanf(hdr, "%7s %127s", method, path);

    if (!strcmp(method, "GET") && !strcmp(path, "/health")) {
        pthread_mutex_lock(&g_mtx);
        int alive = g_sess.used && runner_alive();
        int used = g_sess.used;
        pthread_mutex_unlock(&g_mtx);
        char b[96];
        snprintf(b, sizeof(b), "{\"ok\":true,\"session\":%s,\"runner_alive\":%s}",
                 used ? "true" : "false", alive ? "true" : "false");
        http_resp(c, 200, "application/json", b);
    } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/sessions")) {
        handle_sessions(c, body);
    } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/decode")) {
        handle_decode(c, body);
    } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/shutdown")) {
        pthread_mutex_lock(&g_mtx);
        if (g_sess.used) {
            shutdown_runner();
            free(g_sess.face);
            g_sess.face = NULL;
            g_sess.used = 0;
        }
        pthread_mutex_unlock(&g_mtx);
        http_resp(c, 200, "application/json", "{\"ok\":true}");
    } else {
        http_resp(c, 404, "application/json", "{\"error\":\"not found\"}");
    }
}

/* 大小写不敏感子串查找 (替代 strcasestr, bionic 无声明/静态链接易截断指针) */
static char* find_ci(const char* hay, const char* needle) {
    size_t n = strlen(needle);
    for (; *hay; hay++) {
        size_t i = 0;
        while (i < n) {
            char a = hay[i], b = needle[i];
            if (a >= 'A' && a <= 'Z') a += 32;
            if (b >= 'A' && b <= 'Z') b += 32;
            if (!a || a != b) break;
            i++;
        }
        if (i == n) return (char*)hay;
    }
    return NULL;
}

/* 头逐字节读到 \r\n\r\n (≤16KB), Content-Length 定 body */
static void* http_main(void* arg) {
    int lsn = (int)(intptr_t)arg;
    lg("http listening");
    for (;;) {
        int c = accept(lsn, NULL, NULL);
        if (c < 0) continue;
        set_nodelay(c);
        static char hdr[16384 + 1];
        size_t hn = 0;
        int head_ok = 0;
        while (hn < 16384) {
            ssize_t r = recv(c, hdr + hn, 1, 0);
            if (r <= 0) break;
            hn += (size_t)r;
            if (hn >= 4 && !memcmp(hdr + hn - 4, "\r\n\r\n", 4)) { head_ok = 1; break; }
        }
        hdr[hn] = 0;
        if (!head_ok) { close(c); continue; }
        size_t blen = 0;
        char* cl = find_ci(hdr, "content-length:");
        if (cl) blen = (size_t)strtoll(cl + 15, NULL, 10);
        if (blen > MAX_BODY) { close(c); continue; }
        uint8_t* body = malloc(blen + 1);
        if (!body) { close(c); continue; }
        if (blen && read_full(c, body, blen)) { free(body); close(c); continue; }
        body[blen] = 0;
        handle_http(c, hdr, body);
        free(body);
        close(c);
    }
    return NULL;
}

static void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static int listen_on(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(s, (struct sockaddr*)&a, sizeof(a)) != 0 || listen(s, 4) != 0) {
        lg("listen %d failed: %s", port, strerror(errno));
        exit(1);
    }
    return s;
}

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    int bport = BRIDGE_PORT_D, hport = HTTP_PORT_D;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dir") && i + 1 < argc)
            snprintf(g_dir, sizeof(g_dir), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--bridge-port") && i + 1 < argc)
            bport = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--http-port") && i + 1 < argc)
            hport = atoi(argv[++i]);
    }
    lg("gehtp_proxy dir=%s bridge=%d http=%d", g_dir, bport, hport);

    pthread_t tb, th;
    pthread_create(&tb, NULL, bridge_main, (void*)(intptr_t)listen_on(bport));
    pthread_create(&th, NULL, http_main, (void*)(intptr_t)listen_on(hport));
    pthread_join(tb, NULL);
    pthread_join(th, NULL);
    return 0;
}
