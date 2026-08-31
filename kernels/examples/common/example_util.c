/*
 * example_util.c — 见 example_util.h
 */
#include "example_util.h"
#include <stdarg.h>
#include <stdlib.h>

static FILE *g_fp = NULL;
static int g_npass = 0, g_nfail = 0;

void ex_open_result(const char *name)
{
    char path[256];
    snprintf(path, sizeof(path), "/data/local/tmp/hvxhmx23/%s.txt", name);
    g_fp = fopen(path, "w");
    if (!g_fp) g_fp = stderr;
    g_npass = g_nfail = 0;
    ex_log("=== %s ===", name);
}

void ex_close_result(void)
{
    if (g_fp && g_fp != stderr) fclose(g_fp);
    g_fp = NULL;
}

void ex_log(const char *fmt, ...)
{
    if (!g_fp) return;
    va_list ap; va_start(ap, fmt);
    vfprintf(g_fp, fmt, ap);
    fputc('\n', g_fp);
    va_end(ap);
}

void ex_check(const char *label, int err, int tol)
{
    if (err <= tol) { ex_log("[PASS] %-36s err=%d tol=%d", label, err, tol); g_npass++; }
    else            { ex_log("[FAIL] %-36s err=%d tol=%d", label, err, tol); g_nfail++; }
}

int ex_summary(void)
{
    ex_log("--- summary: %d pass, %d fail ---", g_npass, g_nfail);
    int rc = (g_nfail != 0);
    ex_close_result();
    return rc;
}

/* ---- buffer 填充 (LCG 伪随机, 固定种子可复现) ---- */
static unsigned int lcg(unsigned int *s)
{
    *s = (*s * 1103515245u + 12345u) & 0x7fffffffu;
    return *s;
}

void ex_fill_u8(uint8_t *p, int n, int seed, int mod)
{
    unsigned int s = (unsigned int)seed;
    for (int i = 0; i < n; i++) p[i] = (uint8_t)(lcg(&s) % (mod > 0 ? mod : 256));
}
void ex_fill_i8(int8_t *p, int n, int seed, int mod)
{
    unsigned int s = (unsigned int)seed;
    for (int i = 0; i < n; i++) {
        int v = (int)(lcg(&s) % (mod > 0 ? mod : 7)) - (mod > 0 ? mod/2 : 3);
        p[i] = (int8_t)v;
    }
}
void ex_fill_u16(uint16_t *p, int n, int seed, int mod)
{
    unsigned int s = (unsigned int)seed;
    for (int i = 0; i < n; i++) p[i] = (uint16_t)(lcg(&s) % (mod > 0 ? mod : 0x10000));
}
void ex_fill_i16(int16_t *p, int n, int seed, int mod)
{
    unsigned int s = (unsigned int)seed;
    for (int i = 0; i < n; i++) {
        int v = (int)(lcg(&s) % (mod > 0 ? mod : 7)) - (mod > 0 ? mod/2 : 3);
        p[i] = (int16_t)v;
    }
}
void ex_fill_i32(int32_t *p, int n, int seed, int mod, int off)
{
    unsigned int s = (unsigned int)seed;
    for (int i = 0; i < n; i++)
        p[i] = (int32_t)(lcg(&s) % (mod > 0 ? mod : 200)) - off;
}
void ex_fill_f16(__fp16 *p, int n, int seed, float scale)
{
    unsigned int s = (unsigned int)seed;
    for (int i = 0; i < n; i++) {
        int v = (int)(lcg(&s) % 100) - 50;
        p[i] = (__fp16)(v * scale);
    }
}
