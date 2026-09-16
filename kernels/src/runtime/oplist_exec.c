/* oplist_exec.c — 设备端 op 执行器 (函数指针表分派, 无图; host 已按依赖序排好)
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

#define MAX_TEMPS 256

struct wt_exec {
    struct wtcache_ctx* wc;
    struct dc_arena arena;
    struct dc_w4 e;
    int engine_ready;   /* wtcache 已开(懒初始化/exec_matmul 共用) */
    int matmul_carved;  /* exec_matmul 的 arena/carve 已完成(0=未 carve) */
    uint32_t pinned_count;
    struct wt_exec_stats st;
    uint8_t* temps[MAX_TEMPS];
    uint32_t temp_bytes[MAX_TEMPS];
    /* temp 单块池(第7步阶段一前置): 一次 memalign + bump, 替代 per-temp
     * malloc(每 run 少 ~256 次 malloc/free)。静态偏移表路径就绪后
     * (wt_exec_pool_init), bump 只做兜底, 地址全部来自编译期表。 */
    uint8_t* pool;
    uint32_t pool_cap;
    uint32_t pool_used;
    const uint32_t* static_offsets; /* 编译期 temp→池内偏移表 (TEMPOFF 槽; NULL=运行时 bump) */
    uint32_t static_off_arr[MAX_TEMPS]; /* 表本体(哨兵 0xFFFFFFFF = 表外) */
    /* 第7步阶段二: VTCM 驻留池(0x4000|temp 编码)。vtcm_off_arr[temp] =
     * VTCM 偏移(哨兵 0xFFFFFFFF = 无驻留); vtcm_pool = wtcache 的 VTCM 基址。 */
    uint8_t* vtcm_pool;
    uint32_t vtcm_pool_size;
    uint32_t vtcm_off_arr[MAX_TEMPS];
};

static struct wt_exec g_exec;
static uint32_t g_last_bytes[MAX_TEMPS];  /* 每 temp 最后写入的字节数
    (temp 复用只扩容不缩容, temp_bytes 是分配大小; run_io 输出回传需
    用最后写入大小, 否则历史大值 memcpy 越界 = PD 死, M4.2 实锤) */
static FILE* g_rtrace = NULL;  /* 统一 trace 句柄(同路径双 FILE* 在 DSP farf
                                   下句柄冲突崩溃, M4.2 实锤) */
static void rtrace(const char* msg, int v) {
    if (!g_rtrace) g_rtrace = fopen("/data/local/tmp/hvxhmx23/optrace.txt", "a");
    if (g_rtrace) { fprintf(g_rtrace, "[run_io] %s %d\n", msg, v); fflush(g_rtrace); }
}

/* Level 1 输入注入: 外部输入缓冲 (run_io 设置) */
static const uint8_t* g_ext_in = NULL;

/* slot 数据指针: addr==EXT_IN 的 slot 走外部缓冲 */
static const uint8_t* slot_ptr(const struct wt_blob* b, uint32_t s) {
    if (s >= b->n_slots) return NULL;
    /* EXT_IN 槽: 有注入缓冲用注入, 否则回退 blob 内固化数据(整步/逐段校验) */
    if (b->slots[s].addr == WT_SLOT_EXT_IN && g_ext_in) return g_ext_in;
    return b->weight_base + b->slots[s].offset;
}

/* temp/slot 引用解码: 0x8000|slot_id → slot; 0x4000|temp_id → VTCM 驻留;
 * 否则 temp */
static const uint8_t* ref_ptr(const struct wt_blob* b, uint32_t arg) {
    if (arg & 0x8000u) return slot_ptr(b, arg & 0x7FFFu);
    if (arg & 0x4000u) {
        uint32_t t = arg & 0x3FFFu;
        if (t < MAX_TEMPS && g_exec.vtcm_pool && g_exec.vtcm_off_arr[t] != 0xFFFFFFFFu)
            return g_exec.vtcm_pool + g_exec.vtcm_off_arr[t];
        return NULL;
    }
    return (arg < MAX_TEMPS) ? g_exec.temps[arg] : NULL;
}

/* 引用元素数(f16 元素): slot → count; temp → temp_bytes/2; VTCM 未跟踪=0 */
static uint32_t ref_elem_count(const struct wt_blob* b, uint32_t arg) {
    if (arg & 0x8000u) { uint32_t s = arg & 0x7FFFu; return (s < b->n_slots) ? b->slots[s].count : 0; }
    if (arg & 0x4000u) return 0;
    return (arg < MAX_TEMPS) ? g_exec.temp_bytes[arg] / 2u : 0;
}

uint8_t* wt_exec_temp(uint32_t id) {
    return (id < MAX_TEMPS) ? g_exec.temps[id] : NULL;
}

uint32_t wt_exec_temp_bytes(uint32_t id) {
    return (id < MAX_TEMPS) ? g_exec.temp_bytes[id] : 0;
}

uint32_t wt_exec_temp_last_bytes(uint32_t id) {
    return (id < MAX_TEMPS) ? g_last_bytes[id] : 0;
}

static void cpu_to_vtcm(uint8_t* dst, const uint8_t* src, uint32_t bytes) {
    memcpy(dst, src, bytes);
    qurt_mem_cache_clean((qurt_addr_t)dst, bytes,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

static uint8_t* temp_get(uint32_t id, uint32_t bytes) {
    if (id >= MAX_TEMPS) return NULL;
    /* 第7步阶段二: VTCM 驻留 temp(0x4000|temp 引用的同款判定) ——
     * 驻留张量直接放 VTCM 基址+编译期偏移, 不进 DDR 池。 */
    if (g_exec.vtcm_off_arr[id] != 0xFFFFFFFFu) {
        uint32_t off = g_exec.vtcm_off_arr[id];
        if (off + bytes > g_exec.vtcm_pool_size) return NULL;
        if (!g_exec.vtcm_pool) {
            /* 懒初始化: 首个 VTCM 驻留消费时开 wtcache 拿 VTCM 基址 */
            if (!g_exec.engine_ready) {
                if (wtcache_open(&g_exec.wc, 4096) != WTC_OK) return NULL;
                g_exec.engine_ready = 1;
            }
            void* vb = NULL; uint32_t vs = 0, pc = 0; void* pb = NULL;
            wtcache_layout(g_exec.wc, &vb, &vs, &pb, &pc);
            if (!vb || vs < g_exec.vtcm_pool_size) return NULL;
            /* 驻留池从 VTCM 尾部倒划 —— exec_matmul 的 dc_arena_init 每次
             * 从 (pc+2047)&~2047 重零起划 carve, 若驻留池从同一起点正向
             * bump 会被 carve 踩。倒划到 VTCM 尾部, 且不超过 VTCM 总容量:
             * 容量 = vs - carve_end, 驻留池放 [vs - vtcm_pool_size, vs)。
             * 注意: 懒初始化只拿 VTCM 基址, 不做 arena/carve —— matmul 的
             * carve 由 exec_matmul 自己的初始化完成(engine_ready 已置 1
             * 时 exec_matmul 跳过初始化, 用悬空 arena → binary ref fail) */
            uint32_t carve_end = (pc + 2047u) & ~2047u;
            if (carve_end + g_exec.vtcm_pool_size > vs) return NULL;
            g_exec.vtcm_pool = (uint8_t*)vb + (vs - g_exec.vtcm_pool_size);
        }
        g_exec.temps[id] = g_exec.vtcm_pool + off;
        g_exec.temp_bytes[id] = bytes;
        g_last_bytes[id] = bytes;
        /* VTCM 驻留写入: CPU/HVX 写进 VTCM 后 flush, 否则后续 op 读到的
         * 可能是 cache 里残留的旧 VTCM 内容(上板实测数值不一致根因) */
        qurt_mem_cache_clean((qurt_addr_t)g_exec.temps[id], bytes,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        return g_exec.temps[id];
    }
    /* 路径 1: 编译期静态偏移表(第7步阶段一; TEMPOFF 槽提供池+表)。
     * 偏移正确性由编译期重叠检查器保证(host 工具), 这里只做硬边界检查:
     * 静态区 = [0, pool_used), 表外哨兵 0xFFFFFFFF 回落路径 2。 */
    if (g_exec.static_offsets && g_exec.pool) {
        uint32_t off = g_exec.static_offsets[id];
        if (off != 0xFFFFFFFFu) {
            if (off + bytes > g_exec.pool_used) return NULL;
            g_exec.temps[id] = g_exec.pool + off;
            g_exec.temp_bytes[id] = bytes;
            g_last_bytes[id] = bytes;
            return g_exec.temps[id];
        }
        /* 表外 temp(广播物化/多输出等): 回落池尾预留区 bump */
    }
    /* 路径 2: bump。同 id 扩容只增不缩 —— 新 bump 区, 旧区不回收
     * (与旧 per-temp 的"仅保留最新"语义一致; M3c probe 实锤)。
     * 静态模式下: 静态区 = [0, cap), 表外 bump 区 = [cap, cap+reserve);
     * bump 从 pool_used 起(已被 TEMPOFF 槽初始化设为 cap, 即 bump 区起点)。
     * 静态模式禁扩容(池含编译期偏移, 搬坏静态区)。 */
    if (!g_exec.temps[id] || g_exec.temp_bytes[id] < bytes) {
        uint32_t aligned = (bytes + 127u) & ~127u;
        if (g_exec.pool_used + aligned > g_exec.pool_cap) {
            if (g_exec.static_offsets) return NULL;  /* 静态模式: 预留区耗尽 */
            uint32_t ncap = g_exec.pool_cap ? g_exec.pool_cap : (8u << 20);
            while (ncap < g_exec.pool_used + aligned) ncap *= 2;
            uint8_t* np = memalign(128, ncap);
            if (!np) return NULL;
            if (g_exec.pool) memcpy(np, g_exec.pool, g_exec.pool_used);
            free(g_exec.pool);
            g_exec.pool = np;
            g_exec.pool_cap = ncap;
        }
        g_exec.temps[id] = g_exec.pool + g_exec.pool_used;
        g_exec.pool_used += aligned;
        g_exec.temp_bytes[id] = bytes;
    }
    g_last_bytes[id] = bytes;
    return g_exec.temps[id];
}

/* 第7步阶段一插槽: TEMPOFF 槽解析层提供静态池与编译期偏移表。
 * static_cap = 表内静态区大小(表外 temp 在 [static_cap, total_cap) bump)。
 * 传入后 temp_get 走路径 1(表外哨兵回落路径 2); 传 NULL/NULL 恢复纯 bump。 */
void wt_exec_pool_init(uint8_t* base, uint32_t total_cap, uint32_t static_cap,
                       const uint32_t* offsets) {
    if (base) {
        g_exec.pool = base;
        g_exec.pool_cap = total_cap;
        g_exec.pool_used = static_cap ? static_cap : total_cap;
    }
    g_exec.static_offsets = offsets;
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
    if (!g_exec.matmul_carved) {
        int rc = 0;
        if (!g_exec.engine_ready) {
            rc = wtcache_open(&g_exec.wc, 4096);
            if (rc != WTC_OK) { snprintf(err, errn, "wtcache 0x%X", rc); return -1; }
        }
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
        g_exec.matmul_carved = 1;
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
        g_exec.matmul_carved = 0;
    }
    /* 单块池: 一次释放(旧 per-temp free ×256 全删) */
    free(g_exec.pool);
    g_exec.pool = NULL;
    g_exec.pool_cap = 0;
    g_exec.pool_used = 0;
    g_exec.static_offsets = NULL;
    /* VTCM 驻留: wtcache_close 后基址悬空, 必须清零 —— 否则下一个 run
     * 用残基址读悬空 VTCM = 挂死(diag/8-token 间反复 shutdown/re-run 实锤) */
    g_exec.vtcm_pool = NULL;
    g_exec.vtcm_pool_size = 0;
    for (uint32_t i = 0; i < MAX_TEMPS; i++) {
        g_exec.static_off_arr[i] = 0xFFFFFFFFu;
        g_exec.vtcm_off_arr[i] = 0xFFFFFFFFu;
        g_exec.temps[i] = NULL;
        g_exec.temp_bytes[i] = 0;
    }
    memset(g_last_bytes, 0, sizeof(g_last_bytes));
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

/* GEHTP 阶段9: im2col (f16 NHWC act → cols [th*tw, kh*kw*C] f16, 窗口 gather
 * + pad 零填充)。args: [act_ref, out_t, H, W, C, kh, kw, ph, pw, sh, sw, y0, x0, th, tw]
 * 全局坐标: 输出 (oy,ox), 窗口位置 gy=oy*sh+k-ph (pad 区 → 0)。 */
static int exec_im2col(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t H = op->args[2], W = op->args[3], C = op->args[4];
    uint32_t kh = op->args[5], kw = op->args[6];
    uint32_t ph = op->args[7], pw = op->args[8];
    uint32_t sh = op->args[9], sw = op->args[10];
    uint32_t y0 = op->args[11], x0 = op->args[12];
    uint32_t th = op->args[13], tw = op->args[14];
    const uint8_t* act = ref_ptr(b, op->args[0]);
    if (!act) { snprintf(err, errn, "im2col act ref empty"); return -1; }
    uint32_t K = kh * kw * C;
    uint16_t* cols = (uint16_t*)temp_get(op->args[1], (size_t)th * tw * K * 2u);
    if (!cols) { snprintf(err, errn, "im2col temp alloc"); return -1; }
    const uint16_t* a = (const uint16_t*)act;
    for (uint32_t oy = 0; oy < th; oy++)
        for (uint32_t ox = 0; ox < tw; ox++) {
            uint16_t* row = cols + ((size_t)oy * tw + ox) * K;
            for (uint32_t k = 0; k < kh; k++)
                for (uint32_t l = 0; l < kw; l++) {
                    long gy = (long)(y0 + oy) * sh + (long)k - ph;
                    long gx = (long)(x0 + ox) * sw + (long)l - pw;
                    for (uint32_t c = 0; c < C; c++) {
                        uint16_t v = 0;
                        if (gy >= 0 && gy < (long)H && gx >= 0 && gx < (long)W)
                            v = a[((size_t)gy * W + gx) * C + c];
                        row[(k * kw + l) * C + c] = v;
                    }
                }
        }
    return 0;
}

/* GEHTP 阶段9: conv2d 标量 GEMM (f32 累加, f16 存储 —— fp16 纪律与
 * host/ORT 金标同算法)。args: [cols_t, w_s, bias_s, out_t, M, K, N,
 * out_y0, out_x0, out_H, out_W, co0, co_n]。
 * 输出 tile 直接写入全图 out temp (NHWC [out_H, out_W, co])。 */
static int exec_conv2d(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t M = op->args[4], K = op->args[5], N = op->args[6];
    uint32_t oy0 = op->args[7], ox0 = op->args[8];
    uint32_t out_H = op->args[9], out_W = op->args[10];
    uint32_t co0 = op->args[11], co_n = op->args[12];
    const uint8_t* cols = ref_ptr(b, op->args[0]);
    const uint8_t* wp = slot_ptr(b, op->args[1]);
    const uint8_t* bp = slot_ptr(b, op->args[2]);
    if (!cols || !wp || !bp) { snprintf(err, errn, "conv2d ref empty"); return -1; }
    if (b->slots[op->args[1]].len != K * N * 2u ||
        b->slots[op->args[2]].len != N * 2u) {
        snprintf(err, errn, "conv2d w/b slot size mismatch");
        return -1;
    }
    uint16_t* out = (uint16_t*)temp_get(op->args[3], (size_t)out_H * out_W * N * 2u);
    if (!out) { snprintf(err, errn, "conv2d temp alloc"); return -1; }
    const uint16_t* A = (const uint16_t*)cols;
    const uint16_t* Wm = (const uint16_t*)wp;
    const uint16_t* Bm = (const uint16_t*)bp;
    for (uint32_t r = 0; r < M; r++) {
        for (uint32_t c = co0; c < co0 + co_n; c++) {
            float acc = f16_to_f32(Bm[c]);
            for (uint32_t k = 0; k < K; k++)
                acc += f16_to_f32(A[(size_t)r * K + k]) * f16_to_f32(Wm[(size_t)k * N + c]);
            uint32_t oy = oy0 + r / out_W, ox = ox0 + r % out_W;
            out[((size_t)oy * out_W + ox) * N + c] = f32_to_f16(acc);
        }
    }
    return 0;
}

/* GEHTP 阶段9: 纯 f16 加 (f32 累加, f16 存储; 无 ReLU)。args: [a_ref, b_ref, out_t, n] */
static int exec_add(const struct wt_blob* b, const struct wt_op* op,
                    char* err, size_t errn) {
    uint32_t n = op->args[3];
    const uint8_t* a = ref_ptr(b, op->args[0]);
    const uint8_t* b2 = ref_ptr(b, op->args[1]);
    if (!a || !b2) { snprintf(err, errn, "add ref empty"); return -1; }
    uint16_t* y = (uint16_t*)temp_get(op->args[2], (size_t)n * 2u);
    if (!y) { snprintf(err, errn, "add temp alloc"); return -1; }
    const uint16_t* A = (const uint16_t*)a;
    const uint16_t* B2 = (const uint16_t*)b2;
    for (uint32_t i = 0; i < n; i++)
        y[i] = f32_to_f16(f16_to_f32(A[i]) + f16_to_f32(B2[i]));
    return 0;
}

/* GEHTP 阶段9: spill/fill (DDR temp ↔ 池, 标量拷贝; DMA 快路径 M3)。
 * spill: [src_t, pool_s, off, n_elem]  fill: [pool_s, off, dst_t, n_elem] */
static int exec_spill(const struct wt_blob* b, const struct wt_op* op,
                      char* err, size_t errn) {
    uint32_t n = op->args[3];
    /* src 位置支持 0x8000|slot 编码(输入张量经 slot 引用) */
    const uint8_t* src = ref_ptr(b, op->args[0]);
    if (!src) { snprintf(err, errn, "spill src empty"); return -1; }
    const uint8_t* pool = slot_ptr(b, op->args[1]);
    if (!pool || b->slots[op->args[1]].len < op->args[2] + n * 2u) {
        snprintf(err, errn, "spill pool oob"); return -1;
    }
    memcpy((uint8_t*)pool + op->args[2], src, n * 2u);
    return 0;
}
static int exec_fill(const struct wt_blob* b, const struct wt_op* op,
                     char* err, size_t errn) {
    uint32_t n = op->args[3];
    const uint8_t* pool = slot_ptr(b, op->args[0]);
    if (!pool || b->slots[op->args[0]].len < op->args[1] + n * 2u) {
        snprintf(err, errn, "fill pool oob"); return -1;
    }
    /* dst 位置支持 0x8000|slot 编码 */
    uint8_t* dst = NULL;
    if (op->args[2] & 0x8000u) {
        dst = (uint8_t*)slot_ptr(b, op->args[2] & 0x7FFFu);
    } else {
        dst = temp_get(op->args[2], n * 2u);
    }
    if (!dst) { snprintf(err, errn, "fill dst empty"); return -1; }
    memcpy(dst, pool + op->args[1], n * 2u);
    return 0;
}

/* GEHTP 阶段9: f16 4-D 转置 (perm 每轴 1 字节, N=1 契约)。
 * args: [src_ref, out_t, H, W, C, perm_u32] */
static int exec_transpose(const struct wt_blob* b, const struct wt_op* op,
                          char* err, size_t errn) {
    uint32_t H = op->args[2], W = op->args[3], C = op->args[4];
    uint32_t perm = op->args[5];
    const uint8_t* src = ref_ptr(b, op->args[0]);
    if (!src) { snprintf(err, errn, "transpose src ref empty"); return -1; }
    uint16_t* dst = (uint16_t*)temp_get(op->args[1], (size_t)H * W * C * 2u);
    if (!dst) { snprintf(err, errn, "transpose temp alloc"); return -1; }
    const uint16_t* s16 = (const uint16_t*)src;
    uint32_t p[4] = {(uint8_t)perm, (uint8_t)(perm >> 8),
                     (uint8_t)(perm >> 16), (uint8_t)(perm >> 24)};
    uint32_t dims[4] = {1, H, W, C};
    uint32_t strides_in[4] = {H * W * C, W * C, C, 1};
    for (uint32_t o0 = 0; o0 < dims[p[0]]; o0++)
        for (uint32_t o1 = 0; o1 < dims[p[1]]; o1++)
            for (uint32_t o2 = 0; o2 < dims[p[2]]; o2++)
                for (uint32_t o3 = 0; o3 < dims[p[3]]; o3++) {
                    uint32_t oc[4] = {o0, o1, o2, o3};
                    uint32_t ic[4] = {0, 0, 0, 0};
                    for (int ax = 0; ax < 4; ax++) ic[p[ax]] = oc[ax];
                    size_t src_i = (size_t)ic[0] * strides_in[0] + ic[1] * strides_in[1] +
                                   ic[2] * strides_in[2] + ic[3] * strides_in[3];
                    dst[((size_t)o0 * dims[p[1]] + o1) * dims[p[2]] * dims[p[3]] +
                        o2 * dims[p[3]] + o3] = s16[src_i];
                }
    return 0;
}

/* ---- M3: 20 型新 opcode 执行(f16 面, f32 累加纪律) ---- */

static int exec_unary(const struct wt_blob* b, const struct wt_op* op,
                      char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1], n = op->args[2];
    uint32_t subtype = op->args[3];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    uint16_t* y = (uint16_t*)temp_get(y_t, n * 2u);
    if (!x || !y) { snprintf(err, errn, "unary ref fail"); return -1; }
    for (uint32_t i = 0; i < n; i++) {
        float v = f16_to_f32(x[i]);
        float r = v;
        switch (subtype) {
        case 0:  r = -v; break;
        case 1:  r = expf(v); break;
        case 2:  r = (v >= 0) ? sqrtf(v) : 0.0f; break;
        case 3:  r = (v > 0) ? 1.0f / sqrtf(v) : 0.0f; break;
        case 4:  r = (v > 0) ? logf(v) : 0.0f; break;
        case 5:  r = fabsf(v); break;
        case 6:  r = sinf(v); break;
        case 7:  r = cosf(v); break;
        case 8:  r = 1.0f / (1.0f + expf(-v)); break;
        case 9:  r = tanhf(v); break;
        case 10: r = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))); break;
        case 11: r = fmaxf(0.0f, v); break;
        case 12: r = v / (1.0f + expf(-v)); break;
        default: break;
        }
        y[i] = f32_to_f16(r);
    }
    return 0;
}

static int exec_binary(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t a_t = op->args[0], b_t = op->args[1], y_t = op->args[2];
    uint32_t n = op->args[3], subtype = op->args[4];
    if (subtype == 8) {
        /* SELECT 三元: 编码 [c,a,b,out,8] 5 参数形态(无 n): args[0]=c(cond),
           args[1]=a(真值), args[2]=b(假值), args[3]=out。n=真值元素数;
           cond/假值允许标量(count==1)广播。M4.2b 首撞: 通用解码会把
           假值 slot 当 y_t → temp_get NULL → binary ref fail。 */
        const uint16_t* c = (const uint16_t*)ref_ptr(b, a_t);
        const uint16_t* tv = (const uint16_t*)ref_ptr(b, b_t);
        const uint16_t* fv = (const uint16_t*)ref_ptr(b, y_t);
        uint32_t cn = ref_elem_count(b, a_t), an = ref_elem_count(b, b_t);
        uint32_t fn = ref_elem_count(b, y_t);
        uint16_t* y = (uint16_t*)temp_get(op->args[3], an * 2u);
        if (!c || !tv || !fv || !y || an == 0) { snprintf(err, errn, "select ref fail"); return -1; }
        for (uint32_t i = 0; i < an; i++) {
            float cv = f16_to_f32(c[cn <= 1u ? 0 : i]);
            float r = (cv != 0.0f) ? f16_to_f32(tv[i]) : f16_to_f32(fv[fn <= 1u ? 0 : i]);
            y[i] = f32_to_f16(r);
        }
        return 0;
    }
    const uint16_t* a = (const uint16_t*)ref_ptr(b, a_t);
    const uint16_t* bb = (const uint16_t*)ref_ptr(b, b_t);
    uint16_t* y = (uint16_t*)temp_get(y_t, n * 2u);
    if (!a || !bb || !y) {
        rtrace("bin null", (int)(a == NULL) | ((int)(bb == NULL) << 1) | ((int)(y == NULL) << 2));
        snprintf(err, errn, "binary ref fail");
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        float x0 = f16_to_f32(a[i]), x1 = f16_to_f32(bb[i]);
        float r = x0;
        switch (subtype) {
        case 0: r = x0 + x1; break;
        case 1: r = x0 - x1; break;
        case 2: r = x0 * x1; break;
        case 3: r = (x1 != 0) ? x0 / x1 : 0.0f; break;
        default: break;
        }
        y[i] = f32_to_f16(r);
    }
    return 0;
}

static int exec_softmax(const struct wt_blob* b, const struct wt_op* op,
                        char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1];
    uint32_t rows = op->args[2], n = op->args[3];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    uint16_t* y = (uint16_t*)temp_get(y_t, (size_t)rows * n * 2u);
    if (!x || !y) { snprintf(err, errn, "softmax ref fail"); return -1; }
    for (uint32_t r = 0; r < rows; r++) {
        const uint16_t* xr = x + (size_t)r * n;
        uint16_t* yr = y + (size_t)r * n;
        float mx = f16_to_f32(xr[0]);
        for (uint32_t i = 1; i < n; i++) mx = fmaxf(mx, f16_to_f32(xr[i]));
        float sum = 0.0f;
        for (uint32_t i = 0; i < n; i++) { float e = expf(f16_to_f32(xr[i]) - mx); yr[i] = f32_to_f16(e); sum += e; }
        for (uint32_t i = 0; i < n; i++) yr[i] = f32_to_f16(f16_to_f32(yr[i]) / sum);
    }
    return 0;
}

static int exec_concat(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t out_t = op->args[8];
    uint32_t n_seg = op->args[10], n_elems = op->args[11];
    uint32_t sizes[4] = {op->args[12], op->args[13], op->args[14], op->args[15]};
    uint32_t total_axis = 0;
    for (uint32_t i = 0; i < n_seg && i < 4; i++) total_axis += sizes[i];
    if (total_axis == 0 || n_elems % total_axis != 0) { snprintf(err, errn, "concat shape"); return -1; }
    uint32_t outer = n_elems / total_axis;
    uint16_t* y = (uint16_t*)temp_get(out_t, n_elems * 2u);
    if (!y) { snprintf(err, errn, "concat out fail"); return -1; }
    for (uint32_t k = 0; k < n_seg && k < 4; k++) {
        const uint16_t* xk = (const uint16_t*)ref_ptr(b, op->args[k]);
        if (!xk) { snprintf(err, errn, "concat in fail"); return -1; }
        /* last-dim concat: 输出按 outer 行交错, 段 k 覆盖轴区间
         * [start_k, start_k+size_k) —— 段数据为 (outer, size_k) C 序 */
        uint32_t start_k = 0;
        for (uint32_t j = 0; j < k; j++) start_k += sizes[j];
        for (uint32_t o = 0; o < outer; o++) {
            memcpy(y + (size_t)o * total_axis + start_k,
                   xk + (size_t)o * sizes[k], sizes[k] * 2u);
        }
    }
    return 0;
}

static int exec_split(const struct wt_blob* b, const struct wt_op* op,
                      char* err, size_t errn) {
    uint32_t x_t = op->args[0];
    uint32_t n_seg = op->args[10];
    uint32_t sizes[4] = {op->args[11], op->args[12], op->args[13], op->args[14]};
    uint32_t split_index = op->args[15];  /* 副本 op 取第 split_index 段写 out0 */
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    if (!x) { snprintf(err, errn, "split in fail"); return -1; }
    uint32_t total_axis = 0;
    for (uint32_t i = 0; i < n_seg && i < 4; i++) total_axis += sizes[i];
    if (total_axis == 0) { snprintf(err, errn, "split shape"); return -1; }
    /* 输入总元素: 由 temp 侧字节数推 outer = x_total/total_axis。
       x 的总元素 = 各段和; 段大小比例已知, 但总长未知 → 由 emit 端保证
       x 的 temp 分配了 Σ(outer×sizes) 元素; 这里从 g_exec.temp_bytes 推 */
    uint32_t x_elems = 0;
    if ((x_t & 0x8000u) == 0 && x_t < MAX_TEMPS) x_elems = g_exec.temp_bytes[x_t] / 2u;
    if (x_elems == 0) { snprintf(err, errn, "split x size"); return -1; }
    uint32_t outer = x_elems / total_axis;
    /* 副本语义: 本 op 只写自己的段(第 split_index 段 → out0 槽);
     * 其余段 temp 不分配(emit 已按副本只建 out0)。
     * last-dim split = concat 之逆: 段 k = 每 outer 行内 [off_k, off_k+seg)
     * 通道 strided 段, 非连续块(probe q/k/v 实锤) */
    uint32_t off = 0;
    for (uint32_t k = 0; k < n_seg && k < 4; k++) {
        if (k == split_index) {
            uint16_t* yk = (uint16_t*)temp_get(op->args[1], (size_t)outer * sizes[k] * 2u);
            if (!yk) { snprintf(err, errn, "split out fail"); return -1; }
            for (uint32_t o = 0; o < outer; o++)
                memcpy(yk + (size_t)o * sizes[k],
                       x + (size_t)o * total_axis + off, sizes[k] * 2u);
        }
        off += sizes[k];
    }
    return 0;
}

static int exec_reduce(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1], n = op->args[2];
    uint32_t axis = op->args[3], subtype = op->args[4];
    uint32_t d[4] = {op->args[5], op->args[6], op->args[7], op->args[8]};
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    if (!x) { snprintf(err, errn, "reduce in fail"); return -1; }
    uint32_t L = d[axis];
    if (L == 0) L = 1;
    uint32_t outer = n / L;
    uint16_t* y = (uint16_t*)temp_get(y_t, outer * 2u);
    if (!y) { snprintf(err, errn, "reduce out fail"); return -1; }
    for (uint32_t o = 0; o < outer; o++) {
        float acc = 0.0f;
        for (uint32_t l = 0; l < L; l++)
            acc += f16_to_f32(x[(size_t)o * L + l]);
        if (subtype == 1) acc /= (float)L;
        y[o] = f32_to_f16(acc);
    }
    return 0;
}

static int exec_cumsum(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1];
    uint32_t rows = op->args[2], n = op->args[3];
    uint32_t exclusive = op->args[5], reverse = op->args[6];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    uint16_t* y = (uint16_t*)temp_get(y_t, (size_t)rows * n * 2u);
    if (!x || !y) { snprintf(err, errn, "cumsum ref fail"); return -1; }
    for (uint32_t r = 0; r < rows; r++) {
        float acc = 0.0f;
        if (reverse) {
            for (int32_t i = (int32_t)n - 1; i >= 0; i--) {
                uint32_t idx = (size_t)r * n + (uint32_t)i;
                float v = f16_to_f32(x[idx]);
                y[idx] = f32_to_f16(exclusive ? acc : acc + v);
                acc += v;
            }
        } else {
            for (uint32_t i = 0; i < n; i++) {
                uint32_t idx = (size_t)r * n + i;
                float v = f16_to_f32(x[idx]);
                y[idx] = f32_to_f16(exclusive ? acc : acc + v);
                acc += v;
            }
        }
    }
    return 0;
}

static int exec_conv1d_ssm(const struct wt_blob* b, const struct wt_op* op,
                           char* err, size_t errn) {
    uint32_t x_t = op->args[0], w_s = op->args[1], y_t = op->args[2];
    uint32_t seq = op->args[3], C = op->args[4], k = op->args[5];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    const uint16_t* w = (const uint16_t*)slot_ptr(b, w_s);
    uint16_t* y = (uint16_t*)temp_get(y_t, (size_t)seq * C * 2u);
    if (!x || !w || !y) { snprintf(err, errn, "ssm ref fail"); return -1; }
    for (uint32_t t = 0; t < seq; t++)
        for (uint32_t c = 0; c < C; c++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < k && (int32_t)(t - j) >= 0; j++)
                acc += f16_to_f32(x[(size_t)(t - j) * C + c]) *
                       f16_to_f32(w[(size_t)j * C + c]);
            float sv = acc / (1.0f + expf(-acc));
            y[(size_t)t * C + c] = f32_to_f16(sv);
        }
    return 0;
}

static int exec_gather(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    /* 槽引用兼容两种形态(emit 对输入槽用 0x8000|slot 编码, 权重槽裸 id;
     * wt_parse 校验已按位解码, 执行侧同步) */
    uint32_t tbl_s = op->args[0] & 0x7FFFu, idx_s = op->args[1] & 0x7FFFu;
    uint32_t out_t = op->args[2];
    uint32_t n = op->args[3], row_bytes = op->args[4];
    const uint8_t* tbl = slot_ptr(b, tbl_s);
    const int32_t* idx = (const int32_t*)slot_ptr(b, idx_s);
    uint16_t* y = (uint16_t*)temp_get(out_t, n * 2u);
    if (!tbl || !idx || !y) { snprintf(err, errn, "gather ref fail"); return -1; }
    uint32_t row_n = row_bytes / 2;
    if (row_n == 0) { snprintf(err, errn, "gather row_bytes"); return -1; }
    for (uint32_t i = 0; i < n / row_n; i++) {
        int32_t r = idx[i];
        if (r < 0) r = 0;
        memcpy(y + (size_t)i * row_n, tbl + (size_t)r * row_bytes, row_bytes);
    }
    return 0;
}

/* f16×f16 GEMM (float 图; f32 累加 f16 存储)。M3c 正确性版 —— 性能版
 * 走 HMX (M7)。flags bit0 = a 转置(存 [K,M]), bit1 = w 转置(存 [N,K]),
 * bit2 = batched BMM(attention q·kᵀ/probs·v; 批数在高 16 位) */
static int exec_matmul_f16(const struct wt_blob* b, const struct wt_op* op,
                           char* err, size_t errn) {
    uint32_t a_t = op->args[0], w_s = op->args[1], out_t = op->args[2];
    uint32_t M = op->args[3], K = op->args[4], N = op->args[5];
    uint32_t flags = op->args[6];
    int t0 = (int)(flags & 1u), t1 = (int)(flags & 2u);
    int batched = (int)(flags & 4u);
    uint32_t Bn = flags >> 16;
    if (Bn == 0) Bn = 1;
    const uint16_t* a = (const uint16_t*)ref_ptr(b, a_t);
    /* w 可为 slot(权重)或 temp(运行时 B, 如 attention q·kᵀ) */
    const uint16_t* w = (const uint16_t*)ref_ptr(b, op->args[1]);
    uint16_t* y = (uint16_t*)temp_get(out_t, (size_t)M * N * Bn * 2u);
    if (!a || !w || !y) { snprintf(err, errn, "matmul_f16 ref fail"); return -1; }
    if (batched) {
        for (uint32_t bb = 0; bb < Bn; bb++)
            for (uint32_t m = 0; m < M; m++)
                for (uint32_t n = 0; n < N; n++) {
                    float acc = 0.0f;
                    for (uint32_t k = 0; k < K; k++) {
                        float av = f16_to_f32(a[((size_t)bb * M + m) * K + k]);
                        float wv = f16_to_f32(w[t1 ? ((size_t)bb * N + n) * K + k
                                                  : ((size_t)bb * K + k) * N + n]);
                        acc += av * wv;
                    }
                    y[((size_t)bb * M + m) * N + n] = f32_to_f16(acc);
                }
        return 0;
    }
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                float av = f16_to_f32(a[t0 ? (size_t)k * M + m : (size_t)m * K + k]);
                float wv = f16_to_f32(w[t1 ? (size_t)n * K + k : (size_t)k * N + n]);
                acc += av * wv;
            }
            y[(size_t)m * N + n] = f32_to_f16(acc);
        }
    return 0;
}

/* 通用 RMSNorm: 纯 f16 面直读(无 crouton)。y = x/rms(x) * gamma + bias。
 * eps=1e-6(Qwen3.5 缺省); 行宽 = gamma 槽长/2; m = n/行宽。 */
static int exec_rmsnorm2(const struct wt_blob* b, const struct wt_op* op,
                         char* err, size_t errn) {
    uint32_t x_t = op->args[0];
    uint32_t w_s = op->args[1] & 0x7FFFu, b_s = op->args[2] & 0x7FFFu;
    uint32_t y_t = op->args[3], n = op->args[4];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    const uint16_t* wv = (const uint16_t*)slot_ptr(b, w_s);
    const uint16_t* bv = (const uint16_t*)slot_ptr(b, b_s);
    uint16_t* y = (uint16_t*)temp_get(y_t, (size_t)n * 2u);
    if (!x || !wv || !y) { snprintf(err, errn, "rmsnorm2 ref fail"); return -1; }
    uint32_t rw = b->slots[w_s].len / 2u;
    if (rw == 0 || n % rw != 0) { snprintf(err, errn, "rmsnorm2 shape n%u rw%u", n, rw); return -1; }
    uint32_t m = n / rw;
    for (uint32_t r = 0; r < m; r++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < rw; i++) {
            float v = f16_to_f32(x[(size_t)r * rw + i]);
            acc += (double)v * (double)v;
        }
        float rms = (float)sqrt(acc / (double)rw + 1e-6);
        for (uint32_t i = 0; i < rw; i++) {
            float v = f16_to_f32(x[(size_t)r * rw + i]) / rms;
            v = v * f16_to_f32(wv[i]) + (bv ? f16_to_f32(bv[i]) : 0.0f);
            y[(size_t)r * rw + i] = f32_to_f16(v);
        }
    }
    return 0;
}

/* numpy 广播: 右对齐逐轴, in==1 的轴扩到 out(缺失轴视 1)。
 * in_d/out_d 为 QNN 4D 填充 dims; 压尾 1 得有效 rank 后解坐标。 */
static int exec_broadcast(const struct wt_blob* b, const struct wt_op* op,
                          char* err, size_t errn) {
    uint32_t b_t = op->args[0], y_t = op->args[1], n = op->args[2], b_elems = op->args[3];
    uint32_t in_d[4] = {op->args[4], op->args[5], op->args[6], op->args[7]};
    uint32_t out_d[4] = {op->args[8], op->args[9], op->args[10], op->args[11]};
    const uint16_t* x = (const uint16_t*)ref_ptr(b, b_t);
    uint16_t* y = (uint16_t*)temp_get(y_t, (size_t)n * 2u);
    if (!x || !y || b_elems == 0) { snprintf(err, errn, "broadcast ref fail"); return -1; }
    uint32_t r = 4;
    while (r > 0 && out_d[r - 1] == 1) r--;
    /* in 不去尾 1(末维 1 广播到 out 大维必须保留; host in_bc oid 1231 同款) */
    uint32_t ir = 4;
    if (r == 0) r = 1;
    if (ir > r) ir = r;  /* 输入 rank 不得高于输出(右对齐前导 1 已折叠) */
    uint32_t istr[4] = {1, 1, 1, 1};
    for (int i = (int)ir - 2; i >= 0; i--) istr[i] = istr[i + 1] * in_d[i + 1];
    uint32_t ostr[4] = {1, 1, 1, 1};
    for (int i = (int)r - 2; i >= 0; i--) ostr[i] = ostr[i + 1] * out_d[i + 1];
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rem = i, in_lin = 0;
        uint32_t lead = r - ir;  /* 输入右对齐: 前 lead 轴广播为 0 */
        for (uint32_t d = 0; d < r; d++) {
            uint32_t c = rem / ostr[d];
            rem %= ostr[d];
            if (d < lead) continue;
            uint32_t idim = in_d[d - lead];
            if (idim > 1) in_lin += (c % idim) * istr[d - lead];
        }
        y[i] = x[in_lin];
    }
    return 0;
}

/* 通用 N-D C 序转置: dims/perm 全参数化(rank 2/3/4) */
static int exec_transpose_gen(const struct wt_blob* b, const struct wt_op* op,
                              char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1], rk = op->args[2];
    uint32_t dims[4] = {op->args[3], op->args[4], op->args[5], op->args[6]};
    uint32_t perm = op->args[7];
    const uint16_t* s16 = (const uint16_t*)ref_ptr(b, x_t);
    if (rk < 2 || rk > 4) { snprintf(err, errn, "transpose_gen rank %u", rk); return -1; }
    size_t n = 1;
    for (uint32_t i = 0; i < rk; i++) n *= dims[i];
    uint16_t* dst = (uint16_t*)temp_get(y_t, (uint32_t)n * 2u);
    if (!s16 || !dst) { snprintf(err, errn, "transpose_gen ref fail"); return -1; }
    uint32_t p[4] = {(uint8_t)perm, (uint8_t)(perm >> 8),
                     (uint8_t)(perm >> 16), (uint8_t)(perm >> 24)};
    /* C 序输入 strides(rank 维有效, 其余 1 占位) */
    uint32_t si[4], so[4];
    si[rk - 1] = 1; so[rk - 1] = 1;
    for (int i = (int)rk - 2; i >= 0; i--) {
        si[i] = si[i + 1] * dims[i + 1];
        uint32_t pd = p[i + 1] >= rk ? 1u : dims[p[i + 1]];  /* 输出轴 i 的跨度 =
            其后续轴输出尺寸之积 */
        so[i] = so[i + 1] * pd;
    }
    uint32_t out_dims[4];
    for (uint32_t i = 0; i < rk; i++) out_dims[i] = dims[p[i]];
    uint32_t idx[4] = {0, 0, 0, 0};
    for (size_t t = 0; t < n; t++) {
        /* 输入线性坐标 → 输入多维坐标(与 perm 对应)→ 输出坐标 */
        uint32_t rem = (uint32_t)t;
        uint32_t ic[4] = {0, 0, 0, 0};
        for (uint32_t i = 0; i < rk; i++) { ic[i] = rem / si[i]; rem %= si[i]; }
        uint32_t oc[4] = {0, 0, 0, 0};
        for (uint32_t ax = 0; ax < rk; ax++) oc[ax] = ic[p[ax]];
        uint32_t ot = 0;
        for (uint32_t ax = 0; ax < rk; ax++) ot += oc[ax] * so[ax];
        dst[ot] = s16[t];
        (void)idx; (void)out_dims;
    }
    return 0;
}

static int exec_slice(const struct wt_blob* b, const struct wt_op* op,
                      char* err, size_t errn) {
    /* rank≤3 通用切片: 输出线性 → 输出坐标 → 输入坐标 = b[ax]+c*s[ax]
     * → 输入线性(d 为输入 dims, C 序)。probe RoPE 半切为末维切片形态。 */
    uint32_t x_t = op->args[0], y_t = op->args[1], n_out = op->args[2];
    uint32_t rk = op->args[3];
    uint32_t b0 = op->args[4], b1 = op->args[5], b2 = op->args[6];
    uint32_t e0 = op->args[7], e1 = op->args[8], e2 = op->args[9];
    uint32_t s0 = op->args[10], s1 = op->args[11], s2 = op->args[12];
    uint32_t d0 = op->args[13], d1 = op->args[14], d2 = op->args[15];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    uint16_t* y = (uint16_t*)temp_get(y_t, n_out * 2u);
    if (!x || !y) { snprintf(err, errn, "slice ref fail"); return -1; }
    if (rk < 1 || rk > 3) { snprintf(err, errn, "slice rank %u", rk); return -1; }
    uint32_t bb[3] = {b0, b1, b2}, ee[3] = {e0, e1, e2}, ss[3] = {s0, s1, s2};
    uint32_t dd[3] = {d0, d1, d2};
    uint32_t od[3] = {1, 1, 1};
    for (uint32_t ax = 0; ax < rk; ax++) {
        if (ss[ax] == 0) ss[ax] = 1;
        uint32_t span = (ee[ax] > bb[ax]) ? ee[ax] - bb[ax] : 0;
        od[ax] = (span + ss[ax] - 1) / ss[ax];
    }
    uint32_t istr[3], ostr[3];
    istr[rk - 1] = 1; ostr[rk - 1] = 1;
    for (int i = (int)rk - 2; i >= 0; i--) {
        istr[i] = istr[i + 1] * dd[i + 1];
        ostr[i] = ostr[i + 1] * od[i + 1];
    }
    for (uint32_t i = 0; i < n_out; i++) {
        uint32_t rem = i, in_lin = 0;
        for (uint32_t ax = 0; ax < rk; ax++) {
            uint32_t c = rem / ostr[ax];
            rem %= ostr[ax];
            in_lin += (bb[ax] + c * ss[ax]) * istr[ax];
        }
        y[i] = x[in_lin];
    }
    return 0;
}

static int exec_argmax(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1], n = op->args[2];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    int32_t* y = (int32_t*)temp_get(y_t, 4u);
    if (!x || !y) { snprintf(err, errn, "argmax ref fail"); return -1; }
    int32_t best = 0;
    float bv = -1e30f;
    for (uint32_t i = 0; i < n; i++) {
        float v = f16_to_f32(x[i]);
        if (v > bv) { bv = v; best = (int32_t)i; }
    }
    y[0] = best;
    return 0;
}

/* 执行 ops[first, first+count)。返回 0=全过; >0 = 失败的 op 序号 (blob 内 1 基)。
 * op_us[i] = 本段第 i 个 op 微秒 (可 NULL)。 */
/* 第7步阶段一: TEMPOFF 槽解析(run_range 共用) —— 从 blob 槽表扫 TEMPOFF
 * 槽, 建静态偏移表 + VTCM 偏移表 + 池。init-once: 同一 blob 的 diag 逐 op
 * 重复调用 run_range 只解析一次(槽数据在 blob 内, 跨 run 不变); shutdown
 * 清 static_offsets 后下一个 run 重新解析。 */
static int wt_exec_load_tempoff(const struct wt_blob* b, char* err, size_t errn) {
    if (g_exec.static_offsets) return 0;  /* init-once */
    for (uint32_t i = 0; i < MAX_TEMPS; i++) {
        g_exec.static_off_arr[i] = 0xFFFFFFFFu;
        g_exec.vtcm_off_arr[i] = 0xFFFFFFFFu;
    }
    for (uint32_t s = 0; s < b->n_slots; s++) {
        if (b->slots[s].addr != WT_SLOT_TEMPOFF) continue;
        const uint8_t* p = b->weight_base + b->slots[s].offset;
        uint32_t len = b->slots[s].len;
        if (len < 12u) { snprintf(err, errn, "tempoff slot short"); return -1; }
        uint32_t cap, reserve, n;
        memcpy(&cap, p, 4); memcpy(&reserve, p + 4, 4); memcpy(&n, p + 8, 4);
        if (len < 12u + n * 12u) { snprintf(err, errn, "tempoff table short"); return -1; }
        if (cap == 0 || cap > 0x80000000u) { snprintf(err, errn, "tempoff cap bad"); return -1; }
        uint32_t vtcm_cap = (reserve >> 16) * 1024u;
        reserve = (reserve & 0xFFFFu) * 1024u;
        uint8_t* pool = memalign(128, (size_t)cap + reserve);
        if (!pool) { snprintf(err, errn, "tempoff pool alloc"); return -1; }
        for (uint32_t i = 0; i < n; i++) {
            const uint8_t* e = p + 12 + i * 12;
            uint32_t tid, off, sz;
            memcpy(&tid, e, 4); memcpy(&off, e + 4, 4); memcpy(&sz, e + 8, 4);
            uint32_t tid_clean = tid & 0x3FFFu;
            if (tid_clean >= MAX_TEMPS) {
                free(pool);
                snprintf(err, errn, "tempoff tid bad %u", tid_clean);
                return -1;
            }
            if (tid & WT_REF_VTCM_FLAG) {
                if (vtcm_cap == 0 || off + sz > vtcm_cap) {
                    free(pool);
                    snprintf(err, errn, "tempoff vtcm entry bad (tid %u off %u sz %u vtcm_cap %u)",
                             tid_clean, off, sz, vtcm_cap);
                    return -1;
                }
                g_exec.vtcm_off_arr[tid_clean] = off;
            } else {
                if (off + sz > cap) {
                    free(pool);
                    snprintf(err, errn, "tempoff entry bad (tid %u off %u sz %u cap %u)",
                             tid_clean, off, sz, cap);
                    return -1;
                }
                g_exec.static_off_arr[tid_clean] = off;
            }
        }
        wt_exec_pool_init(pool, cap + reserve, cap, g_exec.static_off_arr);
        /* VTCM 驻留池: wtcache 布局(引擎初始化时补 VTCM 基址) */
        g_exec.vtcm_pool_size = vtcm_cap;
        g_exec.vtcm_pool = NULL;  /* 懒初始化: 首个 VTCM 消费 op 时开 wtcache */
        break;
    }
    return 0;
}

/* ---- 算子执行注册表(算子三件套·设备侧) ----
 * 每 opcode 一个 xop_* 入口(签名统一 wt_op_exec_fn), 注册进 g_op_exec_table[]。
 * 加新算子 = 写一个 exec_* 实现 + 一个 xop_* 薄壳 + 表加一项,
 * 不动 run_range 主循环。语义与旧 switch 逐字等价(统计计数/engine_m
 * 回传时机/NOP/PIN 内联体/KV 占位)。 */
struct wt_op_ctx {
    const struct wt_blob* b;
    const struct wt_op* op;
    uint32_t* engine_m;
    char* err;
    size_t errn;
};
typedef int (*wt_op_exec_fn)(const struct wt_op_ctx* cx);

static int xop_nop(const struct wt_op_ctx* cx) {
    (void)cx;
    g_exec.st.nop++;
    return 0;
}

static int xop_pin(const struct wt_op_ctx* cx) {
    const struct wt_blob* b = cx->b;
    const struct wt_op* op = cx->op;
    uint32_t s = op->args[0];
    if (s >= b->n_slots) { snprintf(cx->err, cx->errn, "pin slot oob"); return -1; }
    g_exec.st.pin++;
    if (!g_exec.engine_ready) { g_exec.st.pin_skipped++; return 0; }
    const uint8_t* p = b->weight_base + b->slots[s].offset;
    uint32_t L = b->slots[s].len;
    if (L == g_exec.e.k * g_exec.e.n / 2u)
        cpu_to_vtcm(g_exec.e.wt, p, L);
    else if (L == (g_exec.e.n / 32u) * 512u)
        cpu_to_vtcm(g_exec.e.bias, p, L);
    g_exec.pinned_count++;
    return 0;
}

static int xop_matmul_w4a16(const struct wt_op_ctx* cx) {
    g_exec.st.matmul++;
    int rc = exec_matmul(cx->b, cx->op, cx->err, cx->errn);
    *cx->engine_m = g_exec.engine_ready ? g_exec.e.m : 0;
    return rc;
}

static int xop_rmsnorm(const struct wt_op_ctx* cx) {
    g_exec.st.rmsnorm++;
    return exec_rmsnorm(cx->b, cx->op, *cx->engine_m, cx->err, cx->errn);
}

static int xop_silu(const struct wt_op_ctx* cx) {
    g_exec.st.silu++;
    return exec_silu(cx->op, cx->err, cx->errn);
}

static int xop_im2col(const struct wt_op_ctx* cx) {
    g_exec.st.im2col++;
    return exec_im2col(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_conv2d(const struct wt_op_ctx* cx) {
    g_exec.st.conv2d++;
    return exec_conv2d(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_add(const struct wt_op_ctx* cx) {
    g_exec.st.add++;
    return exec_add(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_spill(const struct wt_op_ctx* cx) {
    g_exec.st.spill++;
    return exec_spill(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_fill(const struct wt_op_ctx* cx) {
    g_exec.st.fill++;
    return exec_fill(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_transpose(const struct wt_op_ctx* cx) {
    g_exec.st.transpose++;
    return exec_transpose(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_unary(const struct wt_op_ctx* cx) {
    return exec_unary(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_binary(const struct wt_op_ctx* cx) {
    return exec_binary(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_softmax(const struct wt_op_ctx* cx) {
    return exec_softmax(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_concat(const struct wt_op_ctx* cx) {
    return exec_concat(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_slice(const struct wt_op_ctx* cx) {
    return exec_slice(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_split(const struct wt_op_ctx* cx) {
    return exec_split(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_reduce(const struct wt_op_ctx* cx) {
    return exec_reduce(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_cumsum(const struct wt_op_ctx* cx) {
    return exec_cumsum(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_conv1d_ssm(const struct wt_op_ctx* cx) {
    return exec_conv1d_ssm(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_gather(const struct wt_op_ctx* cx) {
    return exec_gather(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_argmax(const struct wt_op_ctx* cx) {
    return exec_argmax(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_kv_unimpl(const struct wt_op_ctx* cx) {
    (void)cx;
    snprintf(cx->err, cx->errn, "kv op 未实现 (M5)");
    return -1;
}

static int xop_matmul_f16(const struct wt_op_ctx* cx) {
    return exec_matmul_f16(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_rmsnorm2(const struct wt_op_ctx* cx) {
    return exec_rmsnorm2(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_broadcast(const struct wt_op_ctx* cx) {
    return exec_broadcast(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_transpose_gen(const struct wt_op_ctx* cx) {
    return exec_transpose_gen(cx->b, cx->op, cx->err, cx->errn);
}

static const wt_op_exec_fn g_op_exec_table[] = {
    [OP_NOP] = xop_nop,
    [OP_PIN] = xop_pin,
    [OP_MATMUL_W4A16] = xop_matmul_w4a16,
    [OP_RMSNORM_F16] = xop_rmsnorm,
    [OP_SILU_F16] = xop_silu,
    [OP_IM2COL] = xop_im2col,
    [OP_CONV2D_F16] = xop_conv2d,
    [OP_ADD_F16] = xop_add,
    [OP_SPILL] = xop_spill,
    [OP_FILL] = xop_fill,
    [OP_TRANSPOSE_F16] = xop_transpose,
    [OP_UNARY_F16] = xop_unary,
    [OP_BINARY_F16] = xop_binary,
    [OP_SOFTMAX_F16] = xop_softmax,
    [OP_CONCAT_F16] = xop_concat,
    [OP_STRIDED_SLICE_F16] = xop_slice,
    [OP_SPLIT_F16] = xop_split,
    [OP_REDUCE_F16] = xop_reduce,
    [OP_CUMSUM_F32] = xop_cumsum,
    [OP_CONV1D_SSM_F16] = xop_conv1d_ssm,
    [OP_GATHER_F16] = xop_gather,
    [OP_ARGMAX_F16] = xop_argmax,
    [OP_KV_APPEND_F16] = xop_kv_unimpl,
    [OP_KV_GATHER_F16] = xop_kv_unimpl,
    [OP_MATMUL_F16] = xop_matmul_f16,
    [OP_RMSNORM2_F16] = xop_rmsnorm2,
    [OP_BROADCAST_F16] = xop_broadcast,
    [OP_TRANSPOSE_GEN_F16] = xop_transpose_gen,
};

int wt_exec_run_range(const struct wt_blob* b, uint32_t first, uint32_t count,
                      uint32_t* engine_m, int64_t* op_us, char* err, size_t errn) {
    uint32_t dummy_m = 0;
    if (!engine_m) engine_m = &dummy_m;
    *engine_m = g_exec.engine_ready ? g_exec.e.m : 0;
    if (first + count > b->n_ops) { snprintf(err, errn, "range oob"); return -1; }
    if (wt_exec_load_tempoff(b, err, errn) != 0) return -1;
    for (uint32_t ii = 0; ii < count; ii++) {
        uint32_t i = first + ii;
        const struct wt_op* op = &b->ops[i];
        int64_t t0 = HAP_perf_get_time_us();
        int rc;
        if (op->opcode >= sizeof(g_op_exec_table) / sizeof(g_op_exec_table[0]) ||
            !g_op_exec_table[op->opcode]) {
            snprintf(err, errn, "opcode %u unhandled", (unsigned)op->opcode);
            rc = -1;
        } else {
            const struct wt_op_ctx cx = {b, op, engine_m, err, errn};
            rc = g_op_exec_table[op->opcode](&cx);
        }
        g_exec.st.ops++;
        if (op_us) op_us[ii] = HAP_perf_get_time_us() - t0;
        {
            /* 统一 trace 句柄铁律(M4.2 实锤: 同路径双 FILE* 在 DSP farf
             * 下句柄冲突挂死) —— 与 rtrace 共用 g_rtrace, 禁止自建 gf。 */
            if (!g_rtrace) g_rtrace = fopen("/data/local/tmp/hvxhmx23/optrace.txt", "a");
            if (g_rtrace) {
                fprintf(g_rtrace, "op%u code=%u rc=%d us=%lld\n",
                        (unsigned)ii, (unsigned)op->opcode, rc,
                        op_us ? (long long)op_us[ii] : -1);
                fflush(g_rtrace);
            }
        }
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

/* GEHTP 阶段9 (Level 1): 外部输入注入 + 输出回传 */
int wt_exec_run_io(const struct wt_blob* b, const void* in_ptr, void* out_ptr,
                   uint32_t out_temp,
                   uint32_t* engine_m, int64_t* op_us, char* err, size_t errn) {
    rtrace("enter", 0);
    g_ext_in = (const uint8_t*)in_ptr;
    int rc = wt_exec_run(b, engine_m, op_us, err, errn);
    rtrace("exec rc", rc);
    if (rc == 0 && out_ptr && out_temp < MAX_TEMPS && g_exec.temps[out_temp]) {
        uint32_t ob = g_last_bytes[out_temp];
        rtrace("memcpy ob", (int)ob);
        memcpy(out_ptr, g_exec.temps[out_temp], ob);
    }
    g_ext_in = NULL;
    rtrace("exit rc", rc);
    return rc;
}
