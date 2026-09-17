/* harness.c — V2.3 U11 对拍框架 (ex_* + wt_sha256 流式组合封装) */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <HAP_perf.h>

#include "harness.h"
/* ex_* 由调用方进程提供 (example_util.c 编入 test_XX.so, 运行时解析 —
 * 与 qurt/HAP 同款跨 .so 符号约定) */
void ex_log(const char* fmt, ...);
void ex_check(const char* label, int err, int tol);
int  ex_summary(void);
#include "wt_sha256.h"

static struct {
    int            active;
    wt_sha256_ctx  ctx;
    uint8_t        sha[32];
    char           sha_hex[65];
    uint32_t       bytes;
} g_h;

void harn_begin(void) { memset(&g_h, 0, sizeof(g_h)); }

void harn_emit(const void* buf, uint32_t bytes) {
    if (!g_h.active || !buf || !bytes) return;
    wt_sha256_update(&g_h.ctx, buf, bytes);
    g_h.bytes += bytes;
}

static void sha_finish(void) {
    wt_sha256_final(&g_h.ctx, g_h.sha);
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        g_h.sha_hex[i * 2]     = hx[g_h.sha[i] >> 4];
        g_h.sha_hex[i * 2 + 1] = hx[g_h.sha[i] & 15];
    }
    g_h.sha_hex[64] = 0;
}

int harn_case(const char* name, harn_fn fn, void* ud) {
    g_h.active = 1;
    wt_sha256_init(&g_h.ctx);
    g_h.bytes = 0;
    int64_t t0 = HAP_perf_get_time_us();
    int rc = fn(ud);
    int64_t us = HAP_perf_get_time_us() - t0;
    sha_finish();
    ex_log("[CASE] %s us=%lld sha=%s bytes=%u", name, (long long)us,
           g_h.sha_hex, g_h.bytes);
    ex_check(name, rc, 0);
    g_h.active = 0;
    return rc;
}

void harn_expect(const char* label, int err, int tol) {
    ex_check(label, err, tol);
}

void harn_note(const char* fmt, ...) {
    char line[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    ex_log("  %s", line);
}

const char* harn_last_sha(void) { return g_h.sha_hex; }
uint32_t harn_last_bytes(void) { return g_h.bytes; }

int harn_summary(void) { return ex_summary(); }
