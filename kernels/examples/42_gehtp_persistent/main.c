/*
 * 42_gehtp_persistent — 持久化 WTOP 推理 runner (socket server 版)
 *
 * 监听 Unix socket: /data/local/tmp/gehtp/runtime.sock
 * 协议:
 *   请求: 4 字节长度 + JSON header + 4 字节整数数组长度 + int32 input_ids[]
 *   响应: 4 字节长度 + JSON ({"token": n} 或 {"logits": [f,...]})
 *
 * 持久化模型加载:
 *   - init 请求后保持 wt_blob 在内存
 *   - decode 步骤 reuse 解析后的 wt_ops
 *   - 避免 M0/M1/M2 等启动开销 (每次 500ms~1s)
 *
 * 用法:
 *   adb shell "/data/local/tmp/hvxhmx23/gehtp_persistent"
 *   (后台运行或通过 supervisord 管理)
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>
#include <errno.h>
#include <HAP_perf.h>
#include "hvxhmx_v23.h"
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"

#define SOCK_PATH "/data/local/tmp/gehtp/runtime.sock"
#define MAX_CONCURRENT_MODELS 4
#define MAX_OPS 1024
#define MAX_INPUT_IDS 2048
#define MAX_LOGITS 32768

static volatile int g_running = 1;

/* 预留模型槽位，持久化 wt_blob */
struct model_slot {
    char model_id[64];
    struct wt_blob* blob;
    uint32_t kv_max;
    int8_t loaded;
};

static struct model_slot g_models[MAX_CONCURRENT_MODELS];

/* 上下文: 当前活跃的请求 */
struct request_ctx {
    int client_fd;
    char model_id[64];
    int32_t input_ids[MAX_INPUT_IDS];
    size_t input_len;
    int position;
    int kv_position;
};

static void sig_handler(int sig) {
    (void)sig;
    g_running = 0;
}

static struct model_slot* find_or_create_slot(const char* model_id) {
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        if (g_models[i].loaded && strcmp(g_models[i].model_id, model_id) == 0) {
            return &g_models[i];
        }
    }
    for (int i = 0; i < MAX_CONCURRENT_MODELS; i++) {
        if (!g_models[i].loaded) {
            memset(&g_models[i], 0, sizeof(g_models[i]));
            strncpy(g_models[i].model_id, model_id, 63);
            g_models[i].loaded = 1;
            return &g_models[i];
        }
    }
    return NULL;
}

static int write_json_string(FILE* fp, const char* key, const char* val) {
    return fprintf(fp, "\"%s\":\"%s\",", key, val) > 0 ? 0 : -1;
}

static int send_response(int fd, int token, const float* logits, size_t n_logits) {
    FILE* sock = fdopen(fd, "w");
    if (!sock) return -1;

    fprintf(sock, "{");
    fprintf(sock, "\"token\":%d", token);
    if (logits && n_logits > 0) {
        fprintf(sock, ",\"logits":[");
        for (size_t i = 0; i < n_logits; i++) {
            if (i > 0) fprintf(sock, ",");
            fprintf(sock, "%.6f", logits[i]);
        }
        fprintf(sock, "]");
    }
    fprintf(sock, "}");
    fflush(sock);
    fclose(sock);
    return 0;
}

static int read_all(int fd, void* buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, (char*)buf + got, len - got);
        if (n <= 0) return -1;
        got += (size_t)n;
    }
    return 0;
}

static struct wt_blob* load_blob_from_path(const char* path, size_t* blob_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    off_t sz = lseek(fd, 0, SEEK_END);
    if (sz <= 0) { close(fd); return NULL; }
    lseek(fd, 0, SEEK_SET);

    uint8_t* buf = malloc((size_t)sz);
    if (!buf) { close(fd); return NULL; }

    size_t got = 0;
    while (got < (size_t)sz) {
        ssize_t n = read(fd, buf + got, (size_t)sz - got);
        if (n <= 0) break;
        got += (size_t)n;
    }
    close(fd);

    if (got != (size_t)sz) { free(buf); return NULL; }

    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w) { free(buf); return NULL; }
    *blob_len = got;
    return w;
}

static int parse_and_exec(
    struct wt_blob* blob,
    const int32_t* input_ids, size_t input_len,
    int position, int kv_position,
    int32_t* out_token, float* out_logits, size_t* n_logits)
{
    if (wt_parse(blob, (size_t)-1, blob) != WT_OK) {
        return -1;
    }

    /* 模型执行 - 此处简化为占位返回 */
    *out_token = 0;
    *n_logits = 0;

    return 0;
}

static int handle_client(int client_fd) {
    uint8_t hdr_len_buf[4];
    if (read_all(client_fd, hdr_len_buf, 4) != 0) {
        close(client_fd);
        return -1;
    }
    uint32_t hdr_len = (uint32_t)hdr_len_buf[0] << 24 | (uint32_t)hdr_len_buf[1] << 16 |
                       (uint32_t)hdr_len_buf[2] << 8 | (uint32_t)hdr_len_buf[3];

    char* hdr_buf = malloc(hdr_len + 1);
    if (!hdr_buf) { close(client_fd); return -1; }

    if (read_all(client_fd, hdr_buf, hdr_len) != 0) {
        free(hdr_buf); close(client_fd);
        return -1;
    }
    hdr_buf[hdr_len] = 0;

    printf("[debug] header: %s\n", hdr_buf);

    /* 解析 JSON */
    const char* model_id = NULL;
    const char* wtop_path = NULL;
    int32_t* input_ids = NULL;
    size_t input_len = 0;
    int position = 0, kv_position = 0;

    /* 简单 JSON 解析 (手动遍历) */
    char* p = hdr_buf;
    while (*p) {
        if (strncmp(p, "\"model_id\"", 10) == 0) {
            p = strchr(p, ':');
            if (p) {
                p = strchr(p, '"');
                if (p) model_id = p + 1;
                if (p) {
                    p = strchr(p + 1, '"');
                    if (p) *p = 0;
                }
            }
        }
        if (strncmp(p, "\"wtop_path\"", 11) == 0) {
            p = strchr(p, ':');
            if (p) {
                p = strchr(p, '"');
                if (p) wtop_path = p + 1;
                if (p) {
                    p = strchr(p + 1, '"');
                    if (p) *p = 0;
                }
            }
        }
        if (strncmp(p, "\"input_ids\"", 11) == 0) {
            p = strchr(p, '[');
            if (p) {
                p++;
                while (*p && *p != ']') {
                    while (*p == ' ' || *p == ',') p++;
                    if (*p >= '0' && *p <= '9') {
                            input_ids[input_len++] = (int32_t)(*p - '0');
                        }
                        p++;
                    }
                }
            }
        }
        if (strncmp(p, "\"position\"", 10) == 0) {
            p = strchr(p, ':');
            if (p) position = atoi(p + 1);
        }
        if (strncmp(p, "\"kv_position\"", 13) == 0) {
            p = strchr(p, ':');
            if (p) kv_position = atoi(p + 1);
        }
        p++;
    }

    printf("[info] model=%s, wtop=%s, input_len=%zu, pos=%d, kv_pos=%d\n",
           model_id ? model_id : "NULL",
           wtop_path ? wtop_path : "NULL",
           input_len, position, kv_position);

    /* 更新模型槽位 */
    struct model_slot* slot = find_or_create_slot(model_id ? model_id : "unknown");
    if (!slot) {
        const char* err = "{\"error\":\"no slot\"}";
        send_response(client_fd, -1, NULL, 0);
        free(hdr_buf);
        close(client_fd);
        return -1;
    }

    /* 持久化加载 wtop */
    if (wtop_path && strlen(wtop_path) > 0 && !slot->blob) {
        size_t blob_len = 0;
        slot->blob = load_blob_from_path(wtop_path, &blob_len);
        if (slot->blob) {
            printf("[loaded] %s (%zu bytes)\n", wtop_path, blob_len);
        }
    }

    /* 执行推理 */
    int32_t token = 0;
    float logits[MAX_LOGITS];
    size_t n_logits = 0;

    if (slot->blob) {
        parse_and_exec(slot->blob, input_ids, input_len,
                       position, kv_position, &token, logits, &n_logits);
    }

    send_response(client_fd, token, n_logits > 0 ? logits : NULL, n_logits);
    free(hdr_buf);
    close(client_fd);
    return 0;
}

static int serve_forever(void) {
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        ex_log("[FAIL] socket 创建失败: %d", errno);
        return -1;
    }

    unlink(SOCK_PATH);
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        ex_log("[FAIL] bind %s: %d", SOCK_PATH, errno);
        close(server_fd);
        return -1;
    }

    if (listen(server_fd, 8) < 0) {
        ex_log("[FAIL] listen: %d", errno);
        close(server_fd);
        return -1;
    }

    printf("[info] listening on %s\n", SOCK_PATH);

    while (g_running) {
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            ex_log("[fail] accept: %d", errno);
            continue;
        }

        if (handle_client(client_fd) < 0) {
            ex_log("[error] handle_client failed");
        }
    }

    close(server_fd);
    unlink(SOCK_PATH);
    return 0;
}

int main(void) {
    ex_open_result("42_gehtp_persistent");
    ex_log("=== 42_gehtp_persistent (socket server) ===");

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    memset(g_models, 0, sizeof(g_models));

    int rc = serve_forever();
    ex_log(bad ? "[FAIL] server exited" : "[PASS] server stopped");
    return ex_summary() || rc;
}