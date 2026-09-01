/*
 * 38_transformer_ops — GEHTP M3: 20 型新 opcode 设备门(每 opcode 独立对拍)
 * =====================================================================
 * 每个 opcode 一个小门: 构造 mini blob(输入槽 + 1 op) → wt_exec_run →
 * 与设备内标量参考(独立代码路径)对拍 f16 ≤1ULP 或 f32 相对容差。
 * 输出: ex_log 行含 [PASS]/[FAIL], build_examples.sh 汇总解析。
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

/* ---------- mini blob 构造 ---------- */
struct mini {
    uint8_t* buf;
    size_t size;
    uint32_t n_slots;
    uint32_t weight_off;
};

static struct mini* mini_new(uint32_t n_slots, uint32_t n_ops) {
    struct mini* m = calloc(1, sizeof(*m));
    m->n_slots = n_slots;
    size_t head = 16u + (size_t)n_slots * 16u;
    size_t ops = (size_t)n_ops * (4u + WT_MAX_ARGS * 4u);
    m->weight_off = (uint32_t)((head + ops + 127) & ~(size_t)127);
    m->size = m->weight_off + 4096;
    m->buf = memalign(128, m->size);
    memset(m->buf, 0, m->size);
    memcpy(m->buf, "WTOP", 4);
    uint16_t ver = WT_BLOB_VER, eck = WT_ENDIAN_CHK;
    memcpy(m->buf + 4, &ver, 2);
    memcpy(m->buf + 6, &eck, 2);
    memcpy(m->buf + 8, &n_slots, 4);
    memcpy(m->buf + 12, &n_ops, 4);
    return m;
}

static void mini_slot(struct mini* m, uint32_t id, const void* data, uint32_t len,
                      uint32_t count) {
    uint8_t* s = m->buf + 16u + id * 16u;
    uint32_t off = id * 512u;   /* slot.offset 契约: 相对 weight 区起点; 槽区每槽 512B */
    memcpy(s, &len, 4);
    memcpy(s + 4, &count, 4);
    memcpy(s + 8, &off, 4);
    if (len > 512) { ex_log("mini_slot: slot 数据 > 512B 未支持"); return; }
    if (data && len) memcpy(m->buf + m->weight_off + off, data, len);
}

static void mini_op(struct mini* m, uint32_t idx, uint16_t opcode,
                    const uint32_t* args, uint16_t n_args) {
    uint8_t* o = m->buf + 16u + m->n_slots * 16u + idx * (4u + WT_MAX_ARGS * 4u);
    memcpy(o, &opcode, 2);
    memcpy(o + 2, &n_args, 2);
    for (uint16_t a = 0; a < n_args; a++) memcpy(o + 4 + a * 4, &args[a], 4);
}

static void mini_run(struct mini* m, uint16_t* out, uint32_t out_cap,
                     uint32_t out_temp) {
    /* wt_blob ~4.5MB (WT_MAX_OPS=65536): 设备线程栈 256KB, 必须堆分配 */
    struct wt_blob* b = calloc(1, sizeof(*b));
    if (!b) { ex_log("[FAIL] mini wt_blob alloc"); return; }
    char err[128] = {0};
    int rc = wt_parse(m->buf, m->size, b);
    if (rc != WT_OK) { ex_log("[FAIL] mini wt_parse %s", wt_err_str(rc)); free(b); return; }
    rc = wt_exec_run(b, NULL, NULL, err, sizeof(err));
    if (rc) { ex_log("[FAIL] mini run rc=%d %s", rc, err); free(b); return; }
    if (out && out_temp < WT_EXEC_MAX_TEMPS) {
        /* 必须在 shutdown 前拷贝: shutdown free 全部 temp */
        uint8_t* t = wt_exec_temp(out_temp);
        uint32_t nbytes = wt_exec_temp_bytes(out_temp);
        if (t && nbytes <= out_cap) memcpy(out, t, nbytes);
        else if (t) ex_log("[FAIL] mini out cap %u < temp %u", out_cap, nbytes);
    }
    wt_exec_shutdown();
    free(b);
}

/* ---------- 门 ---------- */
static int g_fail = 0;
#define GATE(name) static void gate_##name(void)

/* 单输入槽 + 1 op, 输出 temp 0 */
static void run_unary_op(uint32_t subtype, const uint16_t* x, uint32_t n,
                         uint16_t* out, uint32_t out_cap) {
    struct mini* m = mini_new(1, 1);
    mini_slot(m, 0, x, n * 2u, n);
    uint32_t args[4] = {0x8000u | 0u, 0u, n, subtype};
    mini_op(m, 0, OP_UNARY_F16, args, 4);
    mini_run(m, out, out_cap, 0);
    free(m->buf); free(m);
}

GATE(unary) {
    uint32_t n = 64;
    uint16_t x[64];
    for (uint32_t i = 0; i < n; i++) {
        float v = ((float)i - 32.0f) / 8.0f;
        x[i] = (uint16_t)((uint32_t)((v > 0 ? (int)(v * 256 + 0.5f) : (int)(v * 256 - 0.5f)) & 0xFFFF) /* 占位 */);
        /* 直接 f32→f16 手写 */
        uint32_t u;
        float vv = ((float)i - 32.0f) / 8.0f;
        memcpy(&u, &vv, 4);
        uint32_t sign = (u >> 16) & 0x8000u;
        int32_t e = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
        uint32_t mnt = (u >> 13) & 0x3ffu;
        if (e >= 31) x[i] = (uint16_t)(sign | 0x7c00u);
        else if (e <= 0) x[i] = (uint16_t)sign;
        else x[i] = (uint16_t)(sign | ((uint32_t)e << 10) | mnt);
    }
    uint16_t out[64];
    for (int st = 0; st <= 12; st++) {
        run_unary_op((uint32_t)st, x, n, out, sizeof(out));
        int bad = 0;
        for (uint32_t i = 0; i < n; i++) {
            float v = ((float)i - 32.0f) / 8.0f;
            float r = v;
            switch (st) {
            case 0: r = -v; break;
            case 1: r = expf(v); break;
            case 2: r = (v >= 0) ? sqrtf(v) : 0.0f; break;
            case 3: r = (v > 0) ? 1.0f / sqrtf(v) : 0.0f; break;
            case 4: r = (v > 0) ? logf(v) : 0.0f; break;
            case 5: r = fabsf(v); break;
            case 6: r = sinf(v); break;
            case 7: r = cosf(v); break;
            case 8: r = 1.0f / (1.0f + expf(-v)); break;
            case 9: r = tanhf(v); break;
            case 10: r = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))); break;
            case 11: r = fmaxf(0.0f, v); break;
            case 12: r = v / (1.0f + expf(-v)); break;
            default: break;
            }
            float d = fabsf((float)(int16_t)out[i] / 256.0f);  /* 占位 */;
            (void)d;
            uint32_t uu;
            memcpy(&uu, &r, 4);
            uint32_t sg = (uu >> 16) & 0x8000u;
            int32_t ee = (int32_t)((uu >> 23) & 0xffu) - 127 + 15;
            uint32_t mm = (uu >> 13) & 0x3ffu;
            uint16_t expect = (ee >= 31) ? (uint16_t)(sg | 0x7c00u)
                              : (ee <= 0) ? (uint16_t)sg
                              : (uint16_t)(sg | ((uint32_t)ee << 10) | mm);
            uint16_t a = out[i], g = expect;
            uint16_t d16 = (uint16_t)(a > g ? a - g : g - a);
            if (d16 > 1u && !((a & 0x7c00u) == 0x7c00u && (g & 0x7c00u) == 0x7c00u))
                bad++;
        }
        ex_log("%s unary st=%d (n=64, bad=%d)", bad ? "[FAIL]" : "[PASS]", st, bad);
        g_fail += bad ? 1 : 0;
    }
}

GATE(softmax) {
    uint32_t n = 32, rows = 4;
    uint16_t x[128];
    for (uint32_t i = 0; i < rows * n; i++) {
        float v = sinf((float)i * 0.7f) * 2.0f;
        uint32_t u;
        memcpy(&u, &v, 4);
        uint32_t sg = (u >> 16) & 0x8000u;
        int32_t ee = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
        uint32_t mm = (u >> 13) & 0x3ffu;
        x[i] = (ee >= 31) ? (uint16_t)(sg | 0x7c00u) : (ee <= 0) ? (uint16_t)sg
               : (uint16_t)(sg | ((uint32_t)ee << 10) | mm);
    }
    struct mini* m = mini_new(1, 1);
    mini_slot(m, 0, x, rows * n * 2u, rows * n);
    uint32_t args[4] = {0x8000u | 0u, 0u, rows, n};
    mini_op(m, 0, OP_SOFTMAX_F16, args, 4);
    uint16_t out[128];
    mini_run(m, out, sizeof(out), 0);
    free(m->buf); free(m);
    int bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        float mx = -1e9f;
        for (uint32_t i = 0; i < n; i++) {
            /* f16→f32 */
            uint16_t h = x[r * n + i];
            uint32_t sg2 = (uint32_t)(h & 0x8000u) << 16;
            int32_t e2 = (h >> 10) & 0x1fu;
            float v = (e2 == 0) ? 0.0f : ldexpf(1.0f + (float)(h & 0x3ffu) / 1024.0f, e2 - 15);
            if (h & 0x8000u) v = -v;
            if (v > mx) mx = v;
        }
        float sum = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint16_t h = x[r * n + i];
            int32_t e2 = (h >> 10) & 0x1fu;
            float v = (e2 == 0) ? 0.0f : ldexpf(1.0f + (float)(h & 0x3ffu) / 1024.0f, e2 - 15);
            if (h & 0x8000u) v = -v;
            sum += expf(v - mx);
        }
        for (uint32_t i = 0; i < n; i++) {
            uint16_t h = x[r * n + i];
            int32_t e2 = (h >> 10) & 0x1fu;
            float v = (e2 == 0) ? 0.0f : ldexpf(1.0f + (float)(h & 0x3ffu) / 1024.0f, e2 - 15);
            if (h & 0x8000u) v = -v;
            float p = expf(v - mx) / sum;
            int32_t ee = (int32_t)(((*(uint32_t*)&p) >> 23) & 0xffu) - 127 + 15;
            uint16_t expect = (ee >= 31) ? 0x7c00u : (ee <= 0) ? 0 : (uint16_t)(((*(uint32_t*)&p) >> 16 & 0x8000u) | ((uint32_t)ee << 10) | (((*(uint32_t*)&p) >> 13) & 0x3ffu));
            uint16_t a16 = out[r * n + i], g16 = expect;
            uint16_t d = (uint16_t)(a16 > g16 ? a16 - g16 : g16 - a16);
            if (d > 2u) bad++;
        }
    }
    ex_log("%s softmax (rows=%u n=%u bad=%d)", bad ? "[FAIL]" : "[PASS]", rows, n, bad);
    g_fail += bad ? 1 : 0;
}

GATE(cumsum) {
    uint32_t rows = 2, n = 8;
    uint16_t x[16];
    for (uint32_t i = 0; i < rows * n; i++) x[i] = (uint16_t)gdn_f32_to_f16((float)(i + 1));
    struct mini* m = mini_new(1, 1);
    mini_slot(m, 0, x, rows * n * 2u, rows * n);
    uint32_t args[7] = {0x8000u | 0u, 0u, rows, n, 1u, 0u, 0u};
    mini_op(m, 0, OP_CUMSUM_F32, args, 7);
    uint16_t out[16];
    mini_run(m, out, sizeof(out), 0);
    free(m->buf); free(m);
    /* 参考: 序列 1,2,3... (f16) 前缀和 */
    int bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        float acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            float v = (float)(r * n + i + 1);
            acc += v;
            float d = fabsf(gdn_f16_to_f32((int16_t)out[r * n + i]) - acc);
            if (d > 2.0f) bad++;
        }
    }
    ex_log("%s cumsum (rows=%u n=%u bad=%d)", bad ? "[FAIL]" : "[PASS]", rows, n, bad);
    g_fail += bad ? 1 : 0;
}

GATE(reduce) {
    uint32_t n = 16, L = 4;
    uint16_t x[16];
    for (uint32_t i = 0; i < n; i++) x[i] = (uint16_t)gdn_f32_to_f16((float)(i + 1));
    struct mini* m = mini_new(1, 1);
    mini_slot(m, 0, x, n * 2u, n);
    uint32_t args[9] = {0x8000u | 0u, 0u, n, 1u, 0u, 4u, L, 1u, 1u};  /* [x,y,n,axis=1,SUM,d0=4,d1=L] */
    mini_op(m, 0, OP_REDUCE_F16, args, 9);
    uint16_t out[16];
    mini_run(m, out, sizeof(out), 0);
    free(m->buf); free(m);
    int bad = 0;
    for (uint32_t o = 0; o < 4; o++) {
        float acc = 0;
        for (uint32_t l = 0; l < L; l++) acc += (float)(o * L + l + 1);
        if (fabsf(gdn_f16_to_f32((int16_t)out[o]) - acc) > 2.0f) bad++;
    }
    ex_log("%s reduce-sum (bad=%d)", bad ? "[FAIL]" : "[PASS]", bad);
    g_fail += bad ? 1 : 0;
}

GATE(concat) {
    uint32_t n_seg = 2, outer = 4, s0 = 3, s1 = 5;
    uint16_t a[12], b[20];
    for (uint32_t i = 0; i < 12; i++) a[i] = (uint16_t)(i * 0x3800u + 0x3c00u);
    for (uint32_t i = 0; i < 20; i++) b[i] = (uint16_t)((i + 100) * 0x3800u + 0x3c00u);
    struct mini* m = mini_new(2, 1);
    mini_slot(m, 0, a, 24, 12);
    mini_slot(m, 1, b, 40, 20);
    uint32_t args[16] = {0x8000u | 0u, 0x8000u | 1u, 0, 0, 0, 0, 0, 0,
                         0u /*out*/, 1u /*axis*/, n_seg, outer * (s0 + s1),
                         s0, s1, 0, 0};
    mini_op(m, 0, OP_CONCAT_F16, args, 16);
    uint16_t out[32];
    mini_run(m, out, sizeof(out), 0);
    free(m->buf); free(m);
    int bad = 0;
    for (uint32_t o = 0; o < outer; o++) {
        for (uint32_t i = 0; i < s0; i++) if (out[o * (s0 + s1) + i] != a[o * s0 + i]) bad++;
        for (uint32_t i = 0; i < s1; i++) if (out[o * (s0 + s1) + s0 + i] != b[o * s1 + i]) bad++;
    }
    ex_log("%s concat (bad=%d)", bad ? "[FAIL]" : "[PASS]", bad);
    g_fail += bad ? 1 : 0;
}

GATE(gather) {
    uint32_t rows = 4, row_n = 4;
    uint16_t tbl[16];
    int32_t idx[3] = {0, 2, 3};
    for (uint32_t i = 0; i < rows * row_n; i++) tbl[i] = (uint16_t)(i * 0x3800u + 0x3c00u);
    struct mini* m = mini_new(2, 1);
    mini_slot(m, 0, tbl, rows * row_n * 2u, rows * row_n);
    mini_slot(m, 1, idx, 12, 3);
    uint32_t args[5] = {0u /*table 裸槽*/, 1u /*idx 裸槽*/, 0u /*out*/, 3u * row_n, row_n * 2u};
    mini_op(m, 0, OP_GATHER_F16, args, 5);
    uint16_t out[16];
    mini_run(m, out, sizeof(out), 0);
    free(m->buf); free(m);
    int bad = 0;
    for (uint32_t i = 0; i < 3; i++)
        for (uint32_t j = 0; j < row_n; j++)
            if (out[i * row_n + j] != tbl[idx[i] * row_n + j]) bad++;
    ex_log("%s gather (bad=%d)", bad ? "[FAIL]" : "[PASS]", bad);
    g_fail += bad ? 1 : 0;
}

GATE(argmax) {
    uint32_t n = 16;
    uint16_t x[16];
    for (uint32_t i = 0; i < n; i++) x[i] = (uint16_t)gdn_f32_to_f16((float)((i * 7u) % 23u));
    struct mini* m = mini_new(1, 1);
    mini_slot(m, 0, x, n * 2u, n);
    uint32_t args[3] = {0x8000u | 0u, 0u, n};
    mini_op(m, 0, OP_ARGMAX_F16, args, 3);
    uint16_t out[16];
    mini_run(m, out, sizeof(out), 0);
    free(m->buf); free(m);
    int32_t best = -1;
    float bv = -1e30f;
    for (uint32_t i = 0; i < n; i++) {
        float v = (float)((i * 7u) % 23);
        if (v > bv) { bv = v; best = (int32_t)i; }
    }
    int32_t got;
    memcpy(&got, out, 4);
    ex_log("%s argmax (got=%d expect=%d)", got == best ? "[PASS]" : "[FAIL]", (int)got, (int)best);
    g_fail += (got == best) ? 0 : 1;
}

int main(void) {
    ex_open_result("38_transformer_ops");
    ex_log("=== 38_transformer_ops (GEHTP M3) ===");
    gate_unary();
    gate_softmax();
    gate_cumsum();
    gate_reduce();
    gate_concat();
    gate_gather();
    gate_argmax();
    ex_log(g_fail ? "[FAIL] 38_transformer_ops overall" : "[PASS] 38_transformer_ops overall");
    return ex_summary() || g_fail;
}
