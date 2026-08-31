/* oplist_exec.c — 设备端 op 执行器 (switch 分发, 无图; host 已按依赖序排好)
 *
 * 约定 (对齐 PLAN.md §3; 供给面按字节尺寸认领 — t10 供给栈约定):
 *   MATMUL 的 bias / act_table / out_table 不进 op args, 执行器在 slot 表里按
 *   尺寸匹配: len==(N/32)*512 → bias; len==8*(K/32)*4 的前两个 → atbl/otbl
 *   (t10 两张表字节相同, 按出现顺序区分)。
 *   PIN: 引擎建立后把 slot 面搬入 VTCM (幂等); 引擎未建立时仅记账 (首个
 *   MATMUL 自行 staging, 正确性不依赖 PIN)。
 * temp 缓冲: id 0..7; MATMUL 写 temp[out] = i16 crouton (M,N) 面;
 *   RMSNORM 从 temp 读面、去 crouton、行 rmsnorm、写 f16 输出面。
 *   RMSNORM 输入语义: x = (float)(i16 - 32768), eps=1e-6, 双精度累加。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <HAP_perf.h>
#include <qurt.h>

#include "dc_parts.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include "wtcache.h"

#define MAX_TEMPS 8

struct wt_exec {
    struct wtcache_ctx* wc;
    struct dc_arena arena;
    struct dc_w4 e;
    int engine_ready;
    uint32_t pinned_count;
    struct wt_exec_stats st;
    uint8_t* temps[MAX_TEMPS];
    uint32_t temp_bytes[MAX_TEMPS];
};

static struct wt_exec g_exec;

uint8_t* wt_exec_temp(uint32_t id) {
    return (id < MAX_TEMPS) ? g_exec.temps[id] : NULL;
}

uint32_t wt_exec_temp_bytes(uint32_t id) {
    return (id < MAX_TEMPS) ? g_exec.temp_bytes[id] : 0;
}

static void cpu_to_vtcm(uint8_t* dst, const uint8_t* src, uint32_t bytes) {
    memcpy(dst, src, bytes);
    qurt_mem_cache_clean((qurt_addr_t)dst, bytes,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

static uint8_t* temp_get(uint32_t id, uint32_t bytes) {
    if (id >= MAX_TEMPS) return NULL;
    if (!g_exec.temps[id]) {
        g_exec.temps[id] = memalign(128, bytes);
        g_exec.temp_bytes[id] = bytes;
    }
    return g_exec.temps[id];
}

/* f16 ↔ f32 (IEEE 754 binary16, 与 vendor host 实现同算法) */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            while (!(m & 0x400u)) { m <<= 1; e++; }
            bits = sign | ((uint32_t)(127 - 15 + 1 - e - 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> (shift + 1);
        uint32_t rest = man & ((1u << shift) - 1u);
        if ((man >> shift) & 1u) half += (rest || (half & 1u)) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man >> 12) & 1u) half += ((man & 0xfffu) || (half & 1u)) ? 1u : 0u;
    return (uint16_t)half;
}

/* crouton16_row4 面 (M,cols) → row-major i16 (布局还原, 值不变) */
static void inv_crouton(const uint16_t* surf, int16_t* dst, uint32_t M, uint32_t cols) {
    uint32_t n_m32 = M / 32, n_kt = cols / 32, out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++)
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    const uint16_t* p = surf + out;
                    for (uint32_t c = kt * 32; c < kt * 32 + 32; c++) {
                        dst[(size_t)row0 * cols + c] = (int16_t)p[0];
                        dst[(size_t)(row0 + 1) * cols + c] = (int16_t)p[1];
                        p += 2;
                    }
                    out += 64;
                }
}

static int exec_matmul(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t act_s = op->args[0], w_s = op->args[1], out_t = op->args[2];
    uint32_t M = op->args[3], K = op->args[4], N = op->args[5];
    uint32_t act_b = M * K * 2u, wt_b = K * N / 2u, out_b = M * N * 2u;
    uint32_t bias_b = (N / 32u) * 512u, tbl_b = 8u * (K / 32u) * 4u;
    static dc_mutex_t mu;
    static int mu_ready;
    if (!mu_ready) { dc_mutex_init(&mu); mu_ready = 1; }

    const uint8_t *bias = NULL, *atbl = NULL, *otbl = NULL;
    for (uint32_t i = 0; i < b->n_slots; i++) {
        const uint8_t* p = b->weight_base + b->slots[i].offset;
        uint32_t L = b->slots[i].len;
        if (!bias && L == bias_b) bias = p;
        else if (!atbl && L == tbl_b) atbl = p;
        else if (!otbl && L == tbl_b) otbl = p;
    }
    if (!bias || !atbl || !otbl) {
        snprintf(err, errn, "supply slots missing (bias/atbl/otbl)");
        return -1;
    }
    if (b->slots[act_s].len != act_b || b->slots[w_s].len != wt_b) {
        snprintf(err, errn, "act/w slot size mismatch");
        return -1;
    }
    if (!g_exec.engine_ready) {
        int rc = wtcache_open(&g_exec.wc, 4096);
        if (rc != WTC_OK) { snprintf(err, errn, "wtcache 0x%X", rc); return -1; }
        void* vb = NULL; uint32_t vs = 0; void* pb = NULL; uint32_t pc = 0;
        wtcache_layout(g_exec.wc, &vb, &vs, &pb, &pc);
        uint32_t off = (pc + 2047u) & ~2047u;
        dc_arena_init(&g_exec.arena, (uint8_t*)vb + off, vs - off);
        if (dc_w4_carve(&g_exec.e, &g_exec.arena, M, K, N, atbl, otbl)) {
            snprintf(err, errn, "carve m%u k%u n%u", (unsigned)M, (unsigned)K, (unsigned)N);
            return -1;
        }
        /* wtcache_open 末尾 memset(VTCM,0) 留 dirty 零行, 驱逐会覆盖 HMX 直写的 e.out
         * (dualdomain run3 同根因); FLUSH 全 VTCM 一次清干净 */
        qurt_mem_cache_clean((qurt_addr_t)vb, vs, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        g_exec.engine_ready = 1;
    }
    uint8_t* out_ddr = temp_get(out_t, out_b);
    if (!out_ddr) { snprintf(err, errn, "temp %u alloc", (unsigned)out_t); return -1; }

    cpu_to_vtcm(g_exec.e.wt, b->weight_base + b->slots[w_s].offset, wt_b);
    cpu_to_vtcm(g_exec.e.bias, bias, bias_b);

    struct dc_dma d_act, d_out;
    dc_dma_init(&d_act, (uint8_t*)b->weight_base + b->slots[act_s].offset,
                g_exec.e.act, act_b, &mu);
    dc_dma_init(&d_out, g_exec.e.out, out_ddr, out_b, &mu);
    int bad = dc_dma_once(&d_act) || dc_w4_invoke(&g_exec.e) || dc_dma_once(&d_out);
    dc_dma_destroy(&d_act);
    dc_dma_destroy(&d_out);
    if (bad) { snprintf(err, errn, "dma/invoke"); return -1; }
    /* dst_bypass=0 写落内存; CPU 后续读 (rmsnorm/dump) 前丢弃驻留旧行 */
    qurt_mem_cache_clean((qurt_addr_t)out_ddr, out_b,
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    return 0;
}

void wt_exec_shutdown(void) {
    memset(&g_exec.st, 0, sizeof(g_exec.st));
    g_exec.pinned_count = 0;
    if (g_exec.engine_ready) {
        wtcache_close(g_exec.wc);
        g_exec.wc = NULL;
        g_exec.engine_ready = 0;
    }
    for (uint32_t i = 0; i < MAX_TEMPS; i++) {
        free(g_exec.temps[i]);
        g_exec.temps[i] = NULL;
        g_exec.temp_bytes[i] = 0;
    }
}

static int exec_rmsnorm(const struct wt_blob* b, const struct wt_op* op,
                        uint32_t m, char* err, size_t errn) {
    uint32_t x_t = op->args[0], w_s = op->args[1], y_t = op->args[2], n = op->args[3];
    if (!g_exec.temps[x_t]) { snprintf(err, errn, "rmsnorm src temp %u empty", (unsigned)x_t); return -1; }
    if (b->slots[w_s].len != n * 2u) {
        snprintf(err, errn, "rms w slot len mismatch");
        return -1;
    }
    if (!m || m % 32u || n % 32u) { snprintf(err, errn, "rms shape m%u n%u", (unsigned)m, (unsigned)n); return -1; }

    int16_t* rows = memalign(128, (size_t)m * n * 2u);
    float* w = malloc((size_t)n * 4u);
    float* xf = malloc((size_t)n * 4u);
    uint16_t* y = temp_get(y_t, (size_t)m * n * 2u);
    if (!rows || !w || !xf || !y) { snprintf(err, errn, "rms alloc"); return -1; }

    const uint16_t* w16 = (const uint16_t*)(b->weight_base + b->slots[w_s].offset);
    inv_crouton((const uint16_t*)g_exec.temps[x_t], rows, m, n);
    for (uint32_t i = 0; i < n; i++) w[i] = f16_to_f32(w16[i]);
    for (uint32_t r = 0; r < m; r++) {
        const int16_t* row = rows + (size_t)r * n;
        double acc = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            xf[i] = (float)((int32_t)(uint16_t)row[i] - 32768);  /* inv_crouton 存 u16, 勿符号扩展 */
            acc += (double)xf[i] * (double)xf[i];
        }
        float rms = (float)sqrt(acc / (double)n + 1e-6);
        for (uint32_t i = 0; i < n; i++)
            y[(size_t)r * n + i] = f32_to_f16(xf[i] / rms * w[i]);
    }
    free(rows); free(w); free(xf);
    return 0;
}

/* U16: 元素 silu, f16 面 → f16 面 (x/(1+e^-x), f32 内算) */
static int exec_silu(const struct wt_op* op, char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1], n_elem = op->args[2];
    if (!g_exec.temps[x_t]) { snprintf(err, errn, "silu src temp %u empty", (unsigned)x_t); return -1; }
    if (g_exec.temp_bytes[x_t] != n_elem * 2u) {
        snprintf(err, errn, "silu n_elem mismatch (%u vs face %u)",
                 (unsigned)n_elem, (unsigned)(g_exec.temp_bytes[x_t] / 2u));
        return -1;
    }
    uint16_t* y = (uint16_t*)temp_get(y_t, n_elem * 2u);
    if (!y) { snprintf(err, errn, "silu temp %u alloc", (unsigned)y_t); return -1; }
    const uint16_t* x = (const uint16_t*)g_exec.temps[x_t];
    for (uint32_t i = 0; i < n_elem; i++) {
        float f = f16_to_f32(x[i]);
        y[i] = f32_to_f16(f / (1.0f + expf(-f)));
    }
    return 0;
}

/* 执行 ops[first, first+count)。返回 0=全过; >0 = 失败的 op 序号 (blob 内 1 基)。
 * op_us[i] = 本段第 i 个 op 微秒 (可 NULL)。 */
int wt_exec_run_range(const struct wt_blob* b, uint32_t first, uint32_t count,
                      uint32_t* engine_m, int64_t* op_us, char* err, size_t errn) {
    uint32_t dummy_m = 0;
    if (!engine_m) engine_m = &dummy_m;
    *engine_m = g_exec.engine_ready ? g_exec.e.m : 0;
    if (first + count > b->n_ops) { snprintf(err, errn, "range oob"); return -1; }
    for (uint32_t ii = 0; ii < count; ii++) {
        uint32_t i = first + ii;
        const struct wt_op* op = &b->ops[i];
        int64_t t0 = HAP_perf_get_time_us();
        int rc = 0;
        switch (op->opcode) {
        case OP_NOP:
            g_exec.st.nop++;
            break;
        case OP_PIN: {
            uint32_t s = op->args[0];
            if (s >= b->n_slots) { snprintf(err, errn, "pin slot oob"); rc = -1; break; }
            g_exec.st.pin++;
            if (!g_exec.engine_ready) { g_exec.st.pin_skipped++; break; }
            const uint8_t* p = b->weight_base + b->slots[s].offset;
            uint32_t L = b->slots[s].len;
            if (L == g_exec.e.k * g_exec.e.n / 2u)
                cpu_to_vtcm(g_exec.e.wt, p, L);
            else if (L == (g_exec.e.n / 32u) * 512u)
                cpu_to_vtcm(g_exec.e.bias, p, L);
            g_exec.pinned_count++;
            break;
        }
        case OP_MATMUL_W4A16:
            g_exec.st.matmul++;
            rc = exec_matmul(b, op, err, errn);
            *engine_m = g_exec.engine_ready ? g_exec.e.m : 0;
            break;
        case OP_RMSNORM_F16:
            g_exec.st.rmsnorm++;
            rc = exec_rmsnorm(b, op, *engine_m, err, errn);
            break;
        case OP_SILU_F16:
            g_exec.st.silu++;
            rc = exec_silu(op, err, errn);
            break;
        default:
            snprintf(err, errn, "opcode %u unhandled", (unsigned)op->opcode);
            rc = -1;
        }
        g_exec.st.ops++;
        if (op_us) op_us[ii] = HAP_perf_get_time_us() - t0;
        if (rc) return (int)i + 1;
    }
    return 0;
}

void wt_exec_get_stats(struct wt_exec_stats* st) {
    if (st) *st = g_exec.st;
}

/* 执行全部 op。返回 0=全过; >0 = 失败的 op 序号 (1 基); engine_m 回传 MATMUL 的 M。
 * op_us[i] = 第 i 个 op 微秒。 */
int wt_exec_run(const struct wt_blob* b, uint32_t* engine_m,
                int64_t* op_us, char* err, size_t errn) {
    memset(&g_exec.st, 0, sizeof(g_exec.st));
    return wt_exec_run_range(b, 0, b->n_ops, engine_m, op_us, err, errn);
}
