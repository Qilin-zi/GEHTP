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
#ifndef __hexagon__
#include <malloc.h>  /* memalign 原型: 缺则 64 位 host 按 int 隐式声明截断指针 → SIGSEGV */
#endif
#include <fcntl.h>
#include <unistd.h>

#include <HAP_perf.h>
#include <qurt.h>

#include "dc_parts.h"
#include "fence.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include "wtcache.h"
#include "hvxhmx_v2_binary.h"
#include "hvxhmx_v2_unary.h"
#include "hvxhmx_v2_softmax.h"
#include "hvxhmx_v2_norm.h"
#include "hvxhmx_runtime.h"
#include "hvxhmx_v2_matmul.h"
#include "hmx_crouton.h"

#define MAX_TEMPS 4096  /* 0.8B 全模型 temp id 至 1403+ (输出 temp 1401); 256 实锤死 op2616, 与 WT_EXEC_MAX_TEMPS 同步 */

struct wt_exec {
    struct wtcache_ctx* wc;
    struct dc_arena arena;
    struct dc_w4 e;
    int engine_ready;   /* wtcache 已开(懒初始化/exec_matmul 共用) */
    int matmul_carved;  /* exec_matmul 的 arena/carve 已完成(0=未 carve) */
    uint32_t carve_op;  /* 当前 carve 对应的 op 序号(0xFFFFFFFF=未 carve);
                          * 003 P2: carve footer 逐 op 定址, 换 op 必须换 carve */
    uint32_t carve_live_end;  /* 当前 carve 时的活跃驻留上沿(池内, 2048 对齐) */
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
/* PROF W-P2: per-op trace 行开关。默认开(=旧行为, 存量 runner/诊断流零回归;
 * 新 runner 按 job.txt trace 键显式 set_trace(0/1) 拿净测量/取证两态) */
static int g_trace_on = 1;
void wt_exec_set_trace(int on) { g_trace_on = on ? 1 : 0; }
/* 005 US3b (T019): bias 默认走 HMX (SC-005 已过: cos=1.0, 三判据 0 违例,
 * 2026-10-08, 46 号例)。wt_exec_force_bias_scalar 仅供测试例造标量参照,
 * 生产恒 0; qurt 无 env, 走显式 API。 */
static int g_force_bias_scalar = 0;
void wt_exec_force_bias_scalar(int on) { g_force_bias_scalar = on ? 1 : 0; }
/* PROF W-P2: 逐 op 时间戳对缓冲 (注册式, 不注册零开销) */
static struct wt_op_ts* g_ts_buf = NULL;
static uint32_t g_ts_cap = 0;
void wt_exec_set_ts(struct wt_op_ts* buf, uint32_t cap) {
    g_ts_buf = buf; g_ts_cap = buf ? cap : 0;
}
/* PROF W-P3: 本 op 实际执行引擎 (WT_ENG_*)。各 exec_* 内部设置, run_range 读回 */
static uint32_t g_last_engine = WT_ENG_SCALAR;
/* 003 编译器驱动调度: 本 op 编译期指派引擎 (来自 blob v3 footer, 0/1/2)。
 * run_range 每 op 分发前读; ver<3 (无 footer) 时恒 scalar (= 走 legacy 自决)。 */
static uint32_t g_planned_engine = WT_ENG_SCALAR;
/* 003 T011: 本 op 注解的回落策略 (与 g_planned_engine 同源自 footer u32)。
 * ver<3 恒 {1, scalar} = 允许回落标量 (legacy 自决语义不变)。 */
static uint32_t g_planned_fb_allowed = 1u;
static uint32_t g_planned_fb_engine = WT_ENG_SCALAR;
/* 003 T018: 本 op 偏差标志 (0=按计划 1=允许回落 2=已中止)。
 * 仅 v3 分发路径会置位 (engine_fallback_or_die); ver<3 无计划语义恒 0。
 * 中止态由 run_range 填充点按 rc!=0 统一置 2, 各 die 分支无需重复设。 */
static uint32_t g_op_deviation = 0;
static uint32_t g_current_op = 0xFFFFFFFFu;
static uint32_t g_current_op_global = 0xFFFFFFFFu;
static uint32_t g_current_opcode = 0u;
static void rtrace(const char* msg, int v) {
    if (!g_rtrace) g_rtrace = fopen("/data/local/tmp/hrt/gehtp/optrace.txt", "a");
    if (g_rtrace) { fprintf(g_rtrace, "[run_io] %s %d\n", msg, v); fflush(g_rtrace); }
}

static const char* engine_name(uint32_t engine) {
    switch (engine) {
    case WT_ENG_SCALAR: return "SCALAR";
    case WT_ENG_HVX:    return "HVX";
    case WT_ENG_HMX:    return "HMX";
    default:            return "UNKNOWN";
    }
}

static void log_scalar_fallback(uint32_t planned, const char* reason) {
    /* 常开 (不受 g_trace_on 门控): 回落=偏差事件, 净测量 run 也要留痕,
     * 否则 "为什么全标量" 无从定位 (0930 t2l 排查实锤)。低频, 无性能污染。 */
    if (!g_rtrace) g_rtrace = fopen("/data/local/tmp/hrt/gehtp/optrace.txt", "a");
    if (g_rtrace) {
        fprintf(g_rtrace,
                "engine_fallback op=%u code=%u planned=%s actual=%s reason=%s\n",
                (unsigned)g_current_op, (unsigned)g_current_opcode,
                engine_name(planned), engine_name(g_planned_fb_engine),
                reason && reason[0] ? reason : "unspecified");
        fflush(g_rtrace);
    }
}

/* 003 T011 (FR-004/FR-008 骨架) / T018: 计划引擎不可用时的回落策略裁决。
 * 允许回落 → 置偏差标志 1 并返回 0, 调用方按 fallback_engine 执行 (运行期
 * 唯一可执行回落实现 = 标量; fb_engine 非 scalar = 发射/运行期契约漂移,
 * 不可执行 → 按 die 处理, 绝不静默换成别的单元)。
 * 禁止回落 / 回落不可执行 → 记录拒绝并返回 -1 (调用方立即 return -1,
 * run_range 中止, 填充点记 deviation=2, 错误回传 runner)。绝不静默替换。 */
static int engine_fallback_or_die(uint32_t planned, const char* reason,
                                  char* err, size_t errn) {
    if (g_planned_fb_allowed && g_planned_fb_engine == WT_ENG_SCALAR) {
        g_op_deviation = 1;
        log_scalar_fallback(planned, reason);
        return 0;
    }
    log_scalar_fallback(planned, reason);
    /* reason 可能是 err 自身 (调用点用 HMX 写入的失败原因当输入); 先拷贝再写, 避免别名。 */
    char why[192];
    snprintf(why, sizeof why, "%s", reason && reason[0] ? reason : "unspecified");
    snprintf(err, errn, "op planned engine %u unavailable and fallback %s: %s",
             (unsigned)planned,
             g_planned_fb_allowed ? "target unexecutable" : "disallowed",
             why);
    return -1;
}

/* 逐 op 张量 dump 钩子 (精度排查): spec = "opidx:temp,opidx:temp,..." —
 * 执行到 op idx (run_range 内局部序号, 与 optrace 行号一致) 且 rc==0 后,
 * 把 temp 按 g_last_bytes 落盘 hook_<idx>_<temp>.f16.raw。
 * 下发: host 未 set 时回退 GEHTP_HOOK env (旧 hostsim 行为保持);
 * 设备由 runner 读 job.txt "hook" 键经 wt_exec_set_hook 下发 (qurt 无 env)。
 * 落盘: 设备 /data/local/tmp/hrt/gehtp/, host /tmp/。
 * 写用 POSIX open/write 16MB 分块 (qurt stdio 大件死循环教训, 同 runner 输出写法)。 */
static const char* g_hook_spec = NULL;
void wt_exec_set_hook(const char* spec) { g_hook_spec = spec; }
static void exec_hook_dump(uint32_t ii) {
    static const char* env_hook = NULL;
    static int env_tried = 0;
    const char* hook = g_hook_spec;
    if (!hook) {
        if (!env_tried) { env_hook = getenv("GEHTP_HOOK"); env_tried = 1; }
        hook = env_hook;
    }
    if (!hook || !hook[0]) return;
    const char* tok = hook;
    while (*tok) {
        const char* comma = strchr(tok, ',');
        size_t len = comma ? (size_t)(comma - tok) : strlen(tok);
        const char* col = NULL;
        for (size_t i = 0; i < len; i++) if (tok[i] == ':') col = tok + i;
        if (col) {
            char b1[16], b2[16];
            size_t l1 = (size_t)(col - tok);
            if (l1 > 15) l1 = 15;
            memcpy(b1, tok, l1); b1[l1] = 0;
            size_t l2 = len - l1 - 1;
            if (l2 > 15) l2 = 15;
            memcpy(b2, col + 1, l2); b2[l2] = 0;
            uint32_t hi = (uint32_t)strtoul(b1, NULL, 0);
            uint32_t ht = (uint32_t)strtoul(b2, NULL, 0);
            if (hi == ii && ht < MAX_TEMPS && g_exec.temps[ht]) {
                char fn[128];
#ifdef __hexagon__
                snprintf(fn, sizeof fn, "/data/local/tmp/hrt/gehtp/hook_%u_%u.f16.raw", hi, ht);
#else
                snprintf(fn, sizeof fn, "/tmp/hook_%u_%u.f16.raw", hi, ht);
#endif
                int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0666);
                if (fd >= 0) {
                    const uint8_t* p = (const uint8_t*)g_exec.temps[ht];
                    uint32_t left = g_last_bytes[ht], off = 0;
                    while (left) {
                        uint32_t want = left > (16u << 20) ? (16u << 20) : left;
                        ssize_t n = write(fd, p + off, want);
                        if (n <= 0) break;
                        off += (uint32_t)n; left -= (uint32_t)n;
                    }
                    close(fd);
                }
            }
        }
        if (!comma) break;
        tok = comma + 1;
    }
}

/* Level 1 输入注入: 外部输入缓冲 (run_io 设置) */
static const uint8_t* g_ext_in = NULL;
/* 路线B: 外部权重区基址 (run_io 的 wgt_ptr 设置; 权重 slot 的 offset 相对此) */
static const uint8_t* g_ext_wgt = NULL;

/* slot 数据指针: addr==EXT_IN 的 slot 走外部输入缓冲;
 * addr==EXT_WGT 的 slot 走外部权重区(基址+offset) */
static const uint8_t* slot_ptr(const struct wt_blob* b, uint32_t s) {
    if (s >= b->n_slots) return NULL;
    /* EXT_IN 槽: 有注入缓冲用注入, 否则回退 blob 内固化数据(整步/逐段校验) */
    if (b->slots[s].addr == WT_SLOT_EXT_IN && g_ext_in) return g_ext_in;
    /* EXT_WGT 槽: 外部权重区基址 + offset(描述符式, blob 不固化权重) */
    if (b->slots[s].addr == WT_SLOT_EXT_WGT) {
        if (!g_ext_wgt) return NULL;
        return g_ext_wgt + b->slots[s].offset;
    }
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

/* VTCM 驻留池懒初始化: 从 VTCM 尾部倒划, 避开 MATMUL carve arena。 */
static int vtcm_pool_ensure(void) {
    if (g_exec.vtcm_pool) return 0;
    if (!g_exec.engine_ready) {
        /* 反向互斥回收 (同 exec_matmul): 前置 f16/conv HMX 快路径
         * hmx_runtime_setup 持全量 VTCM → 二次 acquire 必失败, 先让出。
         * attach 模式下 base 借自 wtcache 时 engine_ready 必已置位,
         * 不会走到本分支, teardown 清指针不伤 wtcache 的 ctx。 */
        if (hmx_runtime_get_vtcm_base()) hmx_runtime_teardown();
        if (wtcache_open(&g_exec.wc, 4096) != WTC_OK) return -1;
        g_exec.engine_ready = 1;
    }
    void* vb = NULL; uint32_t vs = 0, pc = 0; void* pb = NULL;
    wtcache_layout(g_exec.wc, &vb, &vs, &pb, &pc);
    if (!vb || vs < g_exec.vtcm_pool_size) return -1;
    uint32_t carve_end = (pc + 2047u) & ~2047u;
    if (carve_end + g_exec.vtcm_pool_size > vs) return -1;
    g_exec.vtcm_pool = (uint8_t*)vb + (vs - g_exec.vtcm_pool_size);
    return 0;
}

static int ptr_in_vtcm(const uint8_t* p) {
    /* 差值式判定: VTCM 映射在 32 位地址空间顶端 (实测 [0xff800000,4GB)),
     * pool+size 可恰为 2^32 → 端点相加回绕成 0, 区间上界比较恒假 (w4dma
     * 上板实锤)。p-pool 对 p>=pool 在回绕下仍正确 (pool_size 远小于 2^32)。 */
    return g_exec.vtcm_pool && p >= g_exec.vtcm_pool &&
           (uintptr_t)(p - g_exec.vtcm_pool) < g_exec.vtcm_pool_size;
}

static uint8_t* temp_get(uint32_t id, uint32_t bytes) {
    if (id >= MAX_TEMPS) return NULL;
    /* 第7步阶段二: VTCM 驻留 temp(0x4000|temp 引用的同款判定) ——
     * 驻留张量直接放 VTCM 基址+编译期偏移, 不进 DDR 池。 */
    if (g_exec.vtcm_off_arr[id] != 0xFFFFFFFFu) {
        uint32_t off = g_exec.vtcm_off_arr[id];
        if (off + bytes > g_exec.vtcm_pool_size) return NULL;
        if (!g_exec.vtcm_pool) {
            /* 懒初始化: 首个 VTCM 驻留消费时开 wtcache 拿 VTCM 基址
             * (含 hmx 全量持有时的先让出, 见 vtcm_pool_ensure) */
            if (vtcm_pool_ensure() != 0) return NULL;
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
            uint32_t ncap = g_exec.pool_cap ? g_exec.pool_cap : (64u << 20);
            while (ncap < g_exec.pool_used + aligned) ncap *= 2;
            uint8_t* np = memalign(128, ncap);
            if (!np) return NULL;
            /* 泄漏式扩容: 旧池不 free。调用者在 temp_get 前取走的旧池
             * 指针(如 exec_slice 的 x=ref_ptr 在 y=temp_get 前)在旧池
             * free 后悬空 —— host glibc 立即 unmap 直接 SIGSEGV, DSP
             * free 后内存被后续分配复用则数值污染(上板 cos=0.986 偏差
             * 根因候选)。旧池留用, 新分配全部从新池 bump; 旧代池泄漏
             * 至进程退出(OS 回收), DSP 3GB 预算下可接受。 */
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

/* ---- HVX f32 划痕(策略 B 桥): 泄漏式单块池, 128B 对齐, 与 temp_get 同范式。
 * f32_scratch(n, nbuf, &stride): 领 nbuf 块、每块 n 个 f32(按 32 上取整对齐)的
 * 连续划痕, 返回首块指针, 后续块偏移 = stride。memalign 失败返回 NULL。
 * 旧池不 free(与 temp_get 泄漏式扩容同因: DSP free 后内存复用会污染旧指针)。 */
static float* g_f32_pool = NULL;
static uint32_t g_f32_pool_cap = 0;   /* f32 元素数(块内已按 32 对齐) */

static float* f32_scratch(uint32_t n, uint32_t nbuf, uint32_t* stride_out) {
    uint32_t stride = (n + 31u) & ~31u;
    uint32_t need = stride * nbuf;
    if (need > g_f32_pool_cap) {
        float* np = (float*)memalign(128, (size_t)need * 4u);
        if (!np) { if (stride_out) *stride_out = 0; return NULL; }
        g_f32_pool = np;
        g_f32_pool_cap = need;
    }
    if (stride_out) *stride_out = stride;
    return g_f32_pool;
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
        /* subnormal RNE (与 gdn_ref.c gdn_f32_to_f16 逐字同款): 尾数右移 shift 位。
         * 旧版误写 man >> (shift + 1), 把所有 f16 subnormal (<2^-14) 输出砍半 (0.5×),
         * 0.8B GDN 门控态(rms~1e-4) 上产生非均匀损坏, 是 logits cos 0.988 的根因。 */
        uint32_t sub = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        sub += (rem > (1u << (shift - 1u)) || (rem == (1u << (shift - 1u)) && (sub & 1u))) ? 1u : 0u;
        return (uint16_t)(sign | sub);
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

/* dc_w4_carve 需要的 arena 字节数(与 dc_parts.c 的分配序列逐项对齐:
 * act/out/wt/bias 2048B, atbl/otbl/mask/extra 128B)。用于运行期在 carve
 * 前核容量 —— 绝不能让 dc_arena_alloc 撞 OOM(设备侧 FARF 直接 abort)。 */
static uint32_t w4_carve_need(uint32_t M, uint32_t K, uint32_t N) {
    uint32_t n = 0;
    n += (M * K * 2u + 2047u) & ~2047u;
    n += (M * N * 2u + 2047u) & ~2047u;
    n += (K * N / 2u + 2047u) & ~2047u;
    n += (((N / 32u) * 512u) + 2047u) & ~2047u;
    n += (8u * (K / 32u) * 4u + 127u) & ~127u;
    n += (8u * (N / 32u) * 4u + 127u) & ~127u;
    n += 128u + 128u;
    return n;
}

/* 003 P2: carve footer 的运行期消费(替掉"首个 matmul 抢满全量 VTCM")。
 * 语义: 逐 op 重新 carve(形状变了 e.wt/e.out 尺寸就错), 且 arena 上界按
 * 当前活跃 carve 记录下压 —— 记录 = {op_index, vtcm_offset(池内), size,
 * residency_start, residency_end}; 活跃 = residency_start <= op < end。
 * 驻留 temp 的绝对地址 = 池基 + off, 池基 = VTCM 尾倒划, 所以活跃区最高
 * 端 live_top(池内) 对应绝对上界 vs - pool_size - live_top。 */
static uint32_t carve_live_top(const struct wt_blob* b, uint32_t op_index) {
    uint32_t top = 0;
    for (uint32_t i = 0; i < b->n_carve; i++) {
        const uint8_t* r = b->carve_base + (size_t)i * WT_CARVE_REC_SIZE;
        uint32_t st = wt_rd_u32(r + 12u), en = wt_rd_u32(r + 16u);
        if (op_index < st || op_index >= en) continue;
        uint32_t e1 = wt_rd_u32(r + 4u) + wt_rd_u32(r + 8u);
        if (e1 > top) top = e1;
    }
    return top;
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

    /* 003 T008: W4A16 仅 HMX carve 一路 (dc_w4_carve/invoke 即 HMX, 无标量 W4A16 路径)。
     * ver=3 且编译期未指派 HMX → 无论回落策略均硬报错 (回落标量不可得, 绝不静默)。 */
    if (b->annot_base && g_planned_engine != WT_ENG_HMX) {
        snprintf(err, errn, "W4A16 planned engine %u (no scalar path)",
                 (unsigned)g_planned_engine);
        return -1;
    }

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
    if (!(act_s & WT_REF_VTCM_FLAG) && b->slots[act_s].len != act_b) {
        snprintf(err, errn, "act slot size mismatch");
        return -1;
    }
    if (b->slots[w_s].len != wt_b) {
        snprintf(err, errn, "w slot size mismatch");
        return -1;
    }
    /* 003 T013/T014: 逐 op carve。carve footer 给本 op 活跃的 VTCM 驻留区间,
     * arena 只在驻留区之上(池内 live_top 起)领取, 绝不下压到活跃驻留上;
     * 形状或 live 上界变了才重划(指针面只依赖 基址+容量+M/K/N), 避免逐 op
     * 的 dc_w4_carve + 全 VTCM flush。 */
    uint32_t live_top = (g_exec.vtcm_pool_size && b->carve_base)
                            ? carve_live_top(b, g_current_op_global) : 0u;
    uint32_t live_end = (live_top + 2047u) & ~2047u;
    int shape_same = g_exec.matmul_carved &&
                     g_exec.e.m == M && g_exec.e.k == K && g_exec.e.n == N;
    if (!shape_same || live_end != g_exec.carve_live_end) {
        int rc = 0;
        if (!g_exec.engine_ready) {
            /* VTCM 互斥回收: 前置 MATMUL_F16/CONV2D 的 HMX 快路径 hmx_runtime_setup
             * 持全量 VTCM 且从不释放 → 同进程二次 acquire 返 0 (WTC_ERR_VTCM_ACQUIRE)。
             * 先 teardown 让出再独占。teardown 对 attach 模式只清指针 (ctx 归
             * wtcache), 但 engine_ready=0 + base 非 NULL 只能是自有 ctx, 安全。
             * 反向 (wtcache 先开, f16 后到) 走 attach 共享 scratch 区, 不冲突。 */
            if (hmx_runtime_get_vtcm_base()) hmx_runtime_teardown();
            rc = wtcache_open(&g_exec.wc, 4096);
            if (rc != WTC_OK) { snprintf(err, errn, "wtcache 0x%X", rc); return -1; }
        }
        void* vb = NULL; uint32_t vs = 0; void* pb = NULL; uint32_t pc = 0;
        wtcache_layout(g_exec.wc, &vb, &vs, &pb, &pc);
        uint32_t off = (pc + 2047u) & ~2047u;
        if (!vb || off >= vs) {
            snprintf(err, errn, "vtcm no room (off %u vs %lu)",
                     (unsigned)off, (unsigned long)vs);
            return -1;
        }
        /* 驻留池未起则先起(首个 VTCM 驻留消费点在 temp_get, 这里提前要基址) */
        if (g_exec.vtcm_pool_size && vtcm_pool_ensure() != 0) {
            snprintf(err, errn, "vtcm resident pool ensure");
            return -1;
        }
        /* arena 上界 = VTCM 尾倒划驻留池的顶端, 再按本 op 活跃 carve 上沿放回。
         * 活跃驻留 = carve footer 里 residency_start <= op < end 的记录, 占池内
         * [0, live_top); 池内 [live_top, pool_size) 此刻是死区, arena 可以借用。
         * 无 carve footer (ver<3) → live_top=0 → 上界 = vs - pool_size, 与旧
         * 布局逐字节同界。live_end 超 pool = 编译期/运行期布局漂移, 硬报错。 */
        uint32_t arena_hi = vs;
        if (g_exec.vtcm_pool_size) {
            if (live_end > g_exec.vtcm_pool_size) {
                snprintf(err, errn, "carve live %u > pool %u at op %u",
                         (unsigned)live_end, (unsigned)g_exec.vtcm_pool_size,
                         (unsigned)g_current_op_global);
                return -1;
            }
            arena_hi = vs - (g_exec.vtcm_pool_size - live_end);
        }
        uint32_t arena_cap = (arena_hi > off) ? (arena_hi - off) : 0u;
        /* dc_arena_alloc 撞 OOM 是设备侧 FARF abort, 必须运行期先核容量 */
        uint32_t need = w4_carve_need(M, K, N);
        if (arena_cap < need) {
            snprintf(err, errn, "carve cap %u < need %u (m%u k%u n%u live %u pool %lu)",
                     (unsigned)arena_cap, (unsigned)need,
                     (unsigned)M, (unsigned)K, (unsigned)N,
                     (unsigned)live_end, (unsigned long)g_exec.vtcm_pool_size);
            return -1;
        }
        dc_arena_init(&g_exec.arena, (uint8_t*)vb + off, arena_cap);
        if (dc_w4_carve(&g_exec.e, &g_exec.arena, M, K, N, atbl, otbl)) {
            snprintf(err, errn, "carve m%u k%u n%u", (unsigned)M, (unsigned)K, (unsigned)N);
            return -1;
        }
        /* wtcache_open 末尾 memset(VTCM,0) 留 dirty 零行, 驱逐会覆盖 HMX 直写的 e.out
         * (dualdomain run3 同根因); FLUSH 全 VTCM 一次清干净 */
        qurt_mem_cache_clean((qurt_addr_t)vb, vs, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        g_exec.engine_ready = 1;
        g_exec.matmul_carved = 1;
        g_exec.carve_op = g_current_op_global;
        g_exec.carve_live_end = live_end;
    }
    uint8_t* out_ddr = temp_get(out_t, out_b);
    if (!out_ddr) { snprintf(err, errn, "temp %u alloc", (unsigned)out_t); return -1; }

    cpu_to_vtcm(g_exec.e.wt, b->weight_base + b->slots[w_s].offset, wt_b);
    cpu_to_vtcm(g_exec.e.bias, bias, bias_b);

    if (act_s & WT_REF_VTCM_FLAG) {
        uint8_t* act_vtcm = (uint8_t*)ref_ptr(b, act_s);
        if (!act_vtcm || !ptr_in_vtcm(act_vtcm) || ((uintptr_t)act_vtcm & 2047u)) {
            snprintf(err, errn, "dma-prestaged act bad p=%p pool=%p sz=%lu al=%lu",
                     (void*)act_vtcm, (void*)g_exec.vtcm_pool,
                     (unsigned long)g_exec.vtcm_pool_size,
                     (unsigned long)((uintptr_t)act_vtcm & 2047u));
            return -1;
        }
        uint8_t* saved_act = g_exec.e.act;
        g_exec.e.act = act_vtcm;
        int bad = dc_w4_invoke(&g_exec.e);
        g_exec.e.act = saved_act;
        if (bad) { snprintf(err, errn, "invoke"); return -1; }
        /* invoke 只写 VTCM arena 的 e.out, 消费者在 DDR —— 与下方旧路径同款
         * dc_dma 回拷 + 丢弃驻留旧行 (w4dma 上板实锤: 漏拷 = out 全零) */
        struct dc_dma d_out;
        dc_dma_init(&d_out, g_exec.e.out, out_ddr, out_b, &mu);
        bad = dc_dma_once(&d_out);
        dc_dma_destroy(&d_out);
        if (bad) { snprintf(err, errn, "dma out"); return -1; }
        qurt_mem_cache_clean((qurt_addr_t)out_ddr, out_b,
                             QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        g_last_engine = WT_ENG_HMX;  /* 003 T017: VTCM 预置路径补漏记 */
        return 0;
    }

    struct dc_dma d_act, d_out;
    dc_dma_init(&d_act, (uint8_t*)b->weight_base + b->slots[act_s].offset,
                g_exec.e.act, act_b, &mu);
    dc_dma_init(&d_out, g_exec.e.out, out_ddr, out_b, &mu);
    int bad = dc_dma_once(&d_act) || dc_w4_invoke(&g_exec.e) || dc_dma_once(&d_out);
    dc_dma_destroy(&d_act);
    dc_dma_destroy(&d_out);
    if (bad) { snprintf(err, errn, "dma/invoke"); return -1; }
    g_last_engine = WT_ENG_HMX;  /* W-P3: W4A16 走 HMX 引擎 (成功后记, 避免失败误标) */
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
     * 用残基址读悬空 VTCM = 挂死(diag/8-token 间反复 shutdown/re-run 实锤)。
     * 003 T014: attach 的 hmx base 借自同一 wtcache VTCM, 同样悬空。 */
    if (hmx_runtime_is_attached()) hmx_runtime_teardown();
    g_exec.vtcm_pool = NULL;
    g_exec.vtcm_pool_size = 0;
    g_exec.carve_op = 0xFFFFFFFFu;
    g_exec.carve_live_end = 0u;
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

#if defined(__HVX__) || defined(__hexagon__)
/* GEHTP conv2d HMX 快路径: cols[M,K] @ Wm[:,co0:co0+co_n] + Bm[co0:co0+co_n]
 * 复用 exec_matmul_f16_hmx 的 crouton pack + HMX GEMM 序列, 三点差异:
 *   1) 权重取列子块 (co0 偏移, co_n 列); 2) bias 塞进 scale 区 out_bias 字段
 *      (cvt 一步 f16(acc+bias), 与标量 acc=bias 起累同语义, 免二次舍入);
 *   3) 输出 scatter 回 NHWC 全图 (oy0/ox0 tile 偏移)。
 * 失败返回 -1 回落标量 (未 setup / 超 VTCM / wtcache 已占同一物理 VTCM)。 */
static int exec_conv2d_hmx(const uint16_t* cols, const uint16_t* wm,
                           const uint16_t* bm, uint16_t* out,
                           uint32_t M, uint32_t K, uint32_t N,
                           uint32_t oy0, uint32_t ox0, uint32_t out_W,
                           uint32_t co0, uint32_t co_n,
                           char* err, size_t errn) {
    /* 003 T014 attach: wtcache 持 VTCM (W4A16 驻留池先开) 时不再整体放弃,
     * 把 g_vtcm_base 挂到共享 scratch 区 [pinned_end, vs-pool_size) ——
     * 二次 HAP acquire 同进程必失败, 旧行为=静默回落标量 (t2l 16 个 f16
     * matmul 全标量的根因)。scratch 会覆盖 W4A16 arena → 强制重 carve。 */
    if (hmx_runtime_get_vtcm_base() == NULL) {
        if (g_exec.engine_ready) {
            void* vb = NULL; uint32_t vs = 0, pc = 0; void* pb = NULL;
            wtcache_layout(g_exec.wc, &vb, &vs, &pb, &pc);
            uint32_t off = (pc + 2047u) & ~2047u;
            uint32_t hi = (vs > g_exec.vtcm_pool_size) ? (vs - g_exec.vtcm_pool_size) : 0u;
            uint32_t cap = (vb && hi > off) ? (hi - off) : 0u;
            if (cap == 0u) {
                snprintf(err, errn, "conv2d attach no scratch (vs=%lu pool=%lu)",
                         (unsigned long)vs, (unsigned long)g_exec.vtcm_pool_size);
                return -1;  /* 共享区放不下 → 回落标量 */
            }
            rtrace("hmx attach cap", (int)cap);
            hmx_runtime_attach((uint8_t*)vb + off, cap);
            g_exec.matmul_carved = 0;
        } else if (hmx_runtime_setup(2u * 1024u * 1024u) != 0) {
            snprintf(err, errn, "conv2d hmx setup fail"); return -1;
        }
    }
    char* vtcm = (char*)hmx_runtime_get_vtcm_base();
    unsigned int vtcm_sz = hmx_runtime_get_vtcm_size();
    if (!vtcm || vtcm_sz == 0) return -1;

    uint32_t n_row = M / 32u, n_col = co_n / 32u, n_dot = K / 32u;
    size_t a_bytes = (size_t)M * (size_t)K * 2u;
    size_t w_bytes = (size_t)co_n * (size_t)K * 2u;
    size_t o_bytes = (size_t)M * (size_t)co_n * 2u;
    size_t off_a = 0;
    size_t off_w = (a_bytes + 2047u) & ~(size_t)2047u;
    size_t off_o = (off_w + w_bytes + 2047u) & ~(size_t)2047u;
    size_t off_s = (off_o + o_bytes + 2047u) & ~(size_t)2047u;
    size_t need = off_s + 2048u;
    if (need > (size_t)vtcm_sz) { snprintf(err, errn, "conv2d hmx vtcm overflow"); return -1; }

    uint16_t* act_cr = (uint16_t*)(vtcm + off_a);
    uint16_t* wgt_cr = (uint16_t*)(vtcm + off_w);
    uint16_t* out_cr = (uint16_t*)(vtcm + off_o);
    uint16_t* sca    = (uint16_t*)(vtcm + off_s);

    /* pack act: cols[M,K] (row-major, 无转置) → crouton[(r*n_dot+kt)*1024] */
    for (uint32_t r = 0; r < n_row; ++r)
        for (uint32_t kt = 0; kt < n_dot; ++kt) {
            uint16_t* tile = act_cr + ((size_t)r * n_dot + kt) * 1024u;
            for (uint32_t m = 0; m < 32u; ++m)
                for (uint32_t k = 0; k < 32u; ++k)
                    tile[hmx_crouton_pos(m, k)] =
                        cols[(size_t)(r * 32u + m) * K + (kt * 32u + k)];
        }

    /* pack wgt 列子块: Wm[K,N] 取列 [co0, co0+co_n) → crouton[(c*n_dot+kt)*1024]
     * (row=ic=k, col=oc=n; 全局列号 = co0 + c*32 + n) */
    for (uint32_t c = 0; c < n_col; ++c)
        for (uint32_t kt = 0; kt < n_dot; ++kt) {
            uint16_t* tile = wgt_cr + ((size_t)c * n_dot + kt) * 1024u;
            for (uint32_t k = 0; k < 32u; ++k)
                for (uint32_t n = 0; n < 32u; ++n)
                    tile[hmx_crouton_pos(k, n)] =
                        wm[(size_t)(kt * 32u + k) * N + (co0 + c * 32u + n)];
        }

    /* scale 区: 每列 tile 装 scale=1.0 (偶位) + out_bias=bias (奇位), 高字全 0。
     * mxmem2 32 通道 ↔ 32 输出列; cvt 一步 f16(acc+bias)。 */
    memset(sca, 0, 2048u);
    for (uint32_t c = 0; c < n_col; ++c) {
        for (uint32_t n = 0; n < 32u; ++n) {
            sca[2u * n]      = 0x3C00u;                 /* scale = 1.0 */
            sca[2u * n + 1u] = bm[co0 + c * 32u + n];   /* out_bias = bias */
        }
        for (uint32_t r = 0; r < n_row; ++r) {
            uint16_t* out_tile = out_cr + ((size_t)r * n_col + c) * 1024u;
            hmx_enable_execution();
            hmx_unit_acquire();
            asm volatile("mxclracc.hf" ::: "memory");
            asm volatile("bias = mxmem2(%0)" :: "r"(sca) : "memory");
            for (uint32_t kt = 0; kt < n_dot; ++kt) {
                const uint16_t* act_tile = act_cr + ((size_t)r * n_dot + kt) * 1024u;
                const uint16_t* wgt_tile = wgt_cr + ((size_t)c * n_dot + kt) * 1024u;
                asm volatile(
                    "{ activation.hf = mxmem(%0, %1):deep\n"
                    "  weight.hf = mxmem(%2, %3) }\n"
                    :: "r"(act_tile), "r"(0x7FFu),
                       "r"(wgt_tile), "r"(0x7FFu) : "memory");
            }
            memset(out_tile, 0, 2048u);
            asm volatile(
                "cvt.hf = acc(%0)\n"
                "mxmem(%1, %2) = cvt\n"
                :: "r"(2), "r"(out_tile), "r"(0) : "memory");
            hmx_unit_release();
            hmx_disable_execution();
        }
    }

    /* unpack + scatter: crouton[(r*n_col+c)*1024] → NHWC out[oy][ox][co] */
    for (uint32_t r = 0; r < n_row; ++r)
        for (uint32_t c = 0; c < n_col; ++c) {
            uint16_t* tile = out_cr + ((size_t)r * n_col + c) * 1024u;
            for (uint32_t m = 0; m < 32u; ++m)
                for (uint32_t n = 0; n < 32u; ++n) {
                    uint32_t rr = r * 32u + m;
                    uint32_t oy = oy0 + rr / out_W, ox = ox0 + rr % out_W;
                    out[((size_t)oy * out_W + ox) * N + (co0 + c * 32u + n)] =
                        tile[hmx_crouton_pos(m, n)];
                }
        }
    return 0;
}
#endif /* __HVX__ / __hexagon__ */

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
#if defined(__HVX__) || defined(__hexagon__)
    /* HMX 快路径: M/K/co_n 均 32 倍数 → crouton + HMX GEMM (bias 入 scale 区) */
    /* 003 T009: ver=3 按 g_planned_engine 分发 (HMX→快路径 / scalar→标量);
     * ver<=2 保留运行时 shape 门自决。conv 无 HVX 路径。 */
    int want_hmx;
    if (b->annot_base) {
        if (g_planned_engine == WT_ENG_HMX) {
            want_hmx = 1;
        } else if (g_planned_engine == WT_ENG_HVX) {
            /* 计划 HVX 但 conv 无 HVX 路径 → 按回落策略裁决 */
            if (engine_fallback_or_die(g_planned_engine, "conv has no HVX path", err, errn) != 0) return -1;
            want_hmx = 0;
        } else {
            want_hmx = 0;  /* planned scalar */
        }
    } else {
        want_hmx = ((M % 32u) == 0u && (K % 32u) == 0u && (co_n % 32u) == 0u);
    }
    if (want_hmx) {
        if (exec_conv2d_hmx(A, Wm, Bm, out, M, K, N, oy0, ox0, out_W,
                            co0, co_n, err, errn) == 0) {
            g_last_engine = WT_ENG_HMX;  /* 003 T017: conv HMX 补漏记 */
            return 0;
        }
        /* 失败(未 setup/超 VTCM/wtcache 已占)回落标量 */
        if (b->annot_base && engine_fallback_or_die(g_planned_engine, "conv2d_hmx fail (setup/vtcm/wtcache)",
                                                    err, errn) != 0)
            return -1;   /* 003 T009/T011: 计划 HMX 失败且禁止回落 → 硬报错 */
    }
#endif
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

/* GEHTP 阶段9: 纯 f16 加 (f32 累加, f16 存储; 无 ReLU)。args: [a_ref, b_ref, out_t, n]
 * HVX 接线(M8): GEHTP_HVX_ADD=1 时走向量化 hvhx_v2_add_f16, 数学与标量
 * f32 中间路径位级等价(f16↔f32 vmpy 保真 + f32 加 + RNE 舍入); 默认关(标量),
 * n < 64 或任一指针异常时回落标量。对齐由 hvhx_v2_add_f16 内部前导/主体/尾处理。 */
static int g_hvx_add = -1;  /* -1=未探测, 0=关, 1=开 */
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
    if (g_hvx_add < 0) {
        const char* e = getenv("GEHTP_HVX_ADD");
        g_hvx_add = (e && e[0] == '0') ? 0 : 1;  /* 默认开, 显式 0 才关 */
    }
    /* 003 T010: ver=3 按 g_planned_engine 分发; ver<=2 保留 g_hvx_add 自决 */
    int want_hvx = g_hvx_add && n >= 64u;
    if (b->annot_base) want_hvx = (g_planned_engine == WT_ENG_HVX) && n >= 64u;
    if (want_hvx) {
        hvhx_v2_add_f16(y, A, B2, n);
        g_last_engine = WT_ENG_HVX;  /* W-P3: elemwise 走 HVX 快路径 */
        return 0;
    }
    if (b->annot_base && g_planned_engine != WT_ENG_SCALAR &&
        engine_fallback_or_die(g_planned_engine,
                g_planned_engine == WT_ENG_HMX ? "add has no HMX path" : "n<64 (add HVX gate)",
                err, errn) != 0)
        return -1;   /* 计划 HVX/HMX 但不可用 (n<64/无 HMX) → 按回落策略 */
    for (uint32_t i = 0; i < n; i++)
        y[i] = f32_to_f16(f16_to_f32(A[i]) + f16_to_f32(B2[i]));
    return 0;
}

/* GEHTP 阶段9: spill/fill (DDR temp ↔ 池)。
 * spill: [src_ref, pool_s, off, n_elem]  fill: [pool_s, off, dst_ref, n_elem]
 * 快路径: 两端均 DDR 走 dc_dma_once(硬件 DMA); 任一端 VTCM 驻留回退标量 memcpy
 *         —— VTCM↔DDR 描述符 + fence 收编为后续项(GEHTP_MEMPLAN_GAP_CLOSURE P1)。 */
static int dma_copy_ddr(uint8_t* dst, const uint8_t* src, uint32_t bytes) {
    if (bytes == 0) return 0;
    static dc_mutex_t mu;
    static int mu_ready;
    if (!mu_ready) { dc_mutex_init(&mu); mu_ready = 1; }
    dc_clean_ddr(src, bytes);          /* 铁律①: 写者写 DDR → DMA bypass 读, 先写回 */
    struct dc_dma d;
    if (dc_dma_init(&d, (uint8_t*)src, dst, bytes, &mu) != 0) return -1;
    int rc = dc_dma_once(&d);
    dc_dma_destroy(&d);
    qurt_mem_cache_clean((qurt_addr_t)dst, bytes,   /* 铁律③: DMA 写 → CPU 读, 先 INVALIDATE */
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    return rc ? -1 : 0;
}
static int exec_spill(const struct wt_blob* b, const struct wt_op* op,
                      char* err, size_t errn) {
    uint32_t n = op->args[3], bytes = n * 2u;
    /* src 位置支持 0x8000|slot 编码(输入张量经 slot 引用) */
    const uint8_t* src = ref_ptr(b, op->args[0]);
    if (!src) { snprintf(err, errn, "spill src empty"); return -1; }
    const uint8_t* pool = slot_ptr(b, op->args[1]);
    if (!pool || b->slots[op->args[1]].len < op->args[2] + bytes) {
        snprintf(err, errn, "spill pool oob"); return -1;
    }
    uint8_t* dst = (uint8_t*)pool + op->args[2];
    if (!ptr_in_vtcm(src) && (bytes & 7u) == 0) return dma_copy_ddr(dst, src, bytes);   /* DDR→DDR 快路径(8B 粒度门) */
    memcpy(dst, src, bytes);   /* src VTCM 驻留: 回退标量(VTCM→DDR 后续) */
    return 0;
}
static int exec_fill(const struct wt_blob* b, const struct wt_op* op,
                     char* err, size_t errn) {
    uint32_t n = op->args[3], bytes = n * 2u;
    const uint8_t* pool = slot_ptr(b, op->args[0]);
    if (!pool || b->slots[op->args[0]].len < op->args[1] + bytes) {
        snprintf(err, errn, "fill pool oob"); return -1;
    }
    const uint8_t* src = pool + op->args[1];
    /* dst 位置支持 0x8000|slot 编码 */
    uint8_t* dst = NULL;
    if (op->args[2] & 0x8000u) {
        dst = (uint8_t*)slot_ptr(b, op->args[2] & 0x7FFFu);
    } else {
        dst = temp_get(op->args[2], bytes);
    }
    if (!dst) { snprintf(err, errn, "fill dst empty"); return -1; }
    if (!ptr_in_vtcm(dst) && (bytes & 7u) == 0) return dma_copy_ddr(dst, src, bytes);   /* DDR→DDR 快路径(8B 粒度门) */
    memcpy(dst, src, bytes);   /* dst VTCM 驻留: 回退标量(DDR→VTCM 后续) */
    return 0;
}

/* P5 arity-8 DMA runlist: [src_ref,dst_ref,bytes,src_off,dst_off,flags,fence_src,fence_dst]. */
static uint8_t* dma_ref_resolve(const struct wt_blob* b, uint32_t ref,
                                uint32_t off, uint32_t bytes, int is_dst,
                                int* mem) {
    uint8_t* p = NULL;
    if ((ref & 0xC000u) == 0xC000u) {
        if (!g_exec.matmul_carved) return NULL;
        uint32_t cap = 0;
        switch (ref & 0xFFu) {
        case WT_ENG_ACT:  p = g_exec.e.act;  cap = g_exec.e.m * g_exec.e.k * 2u; break;
        case WT_ENG_OUT:  p = g_exec.e.out;  cap = g_exec.e.m * g_exec.e.n * 2u; break;
        case WT_ENG_WT:   p = g_exec.e.wt;   cap = g_exec.e.k * g_exec.e.n / 2u; break;
        case WT_ENG_BIAS: p = g_exec.e.bias; cap = (g_exec.e.n / 32u) * 512u; break;
        default: return NULL;
        }
        if (!p || (uint64_t)off + bytes > cap) return NULL;
        p += off;
        *mem = FM_VTCM;
        return p;
    }
    if (ref & 0x8000u) {
        uint32_t s = ref & 0x7FFFu;
        const uint8_t* sp = slot_ptr(b, s);
        if (!sp || s >= b->n_slots) return NULL;
        if ((uint64_t)off + bytes > b->slots[s].len) return NULL;
        p = (uint8_t*)sp + off;
    } else if (ref & 0x4000u) {
        uint32_t t = ref & 0x3FFFu;
        if (t >= MAX_TEMPS || g_exec.vtcm_off_arr[t] == 0xFFFFFFFFu) return NULL;
        if (!g_exec.vtcm_pool && vtcm_pool_ensure() != 0) return NULL;
        uint32_t base = g_exec.vtcm_off_arr[t];
        if ((uint64_t)base + off + bytes > g_exec.vtcm_pool_size) return NULL;
        p = g_exec.vtcm_pool + base + off;
        *mem = FM_VTCM;
        return p;
    } else {
        uint32_t t = ref;
        if (t >= MAX_TEMPS) return NULL;
        if (is_dst) {
            p = temp_get(t, off + bytes);
        } else {
            if (!g_exec.temps[t] || (uint64_t)off + bytes > g_exec.temp_bytes[t])
                return NULL;
            p = g_exec.temps[t];
        }
        if (!p) return NULL;
        p += off;
    }
    *mem = ptr_in_vtcm(p) ? FM_VTCM : FM_DDR;
    return p;
}

static int exec_dma(const struct wt_blob* b, const struct wt_op* op,
                    char* err, size_t errn) {
    uint32_t bytes = op->args[2];
    uint32_t src_off = op->args[3], dst_off = op->args[4];
    int src_bypass = (int)(op->args[5] & 1u), dst_bypass = (int)((op->args[5] >> 1) & 1u);
    uint32_t f_src = op->args[6], f_dst = op->args[7];
    if (bytes == 0) return 0;
    if ((f_src > 3u && f_src != WT_DMA_FENCE_NONE) ||
        (f_dst > 3u && f_dst != WT_DMA_FENCE_NONE)) {
        snprintf(err, errn, "dma fence arg bad"); return -1;
    }
    int src_mem = FM_DDR, dst_mem = FM_DDR;
    uint8_t* src = dma_ref_resolve(b, op->args[0], src_off, bytes, 0, &src_mem);
    uint8_t* dst = dma_ref_resolve(b, op->args[1], dst_off, bytes, 1, &dst_mem);
    if (!src || !dst) { snprintf(err, errn, "dma ref fail"); return -1; }

    int src_vtcm = (src_mem == FM_VTCM), dst_vtcm = (dst_mem == FM_VTCM);
    int scalar = (src_vtcm && dst_vtcm) || ((bytes & 7u) != 0);
    int fc_peer = scalar ? FC_CPU : FC_DMA;

    if (f_src != WT_DMA_FENCE_NONE &&
        fence_handoff(src, bytes, (int)f_src, fc_peer, src_mem) != FENCE_OK) {
        snprintf(err, errn, "dma fence src combo"); return -1;
    }
    if (!scalar) {
        static dc_mutex_t mu;
        static int mu_ready;
        if (!mu_ready) { dc_mutex_init(&mu); mu_ready = 1; }
        struct dc_dma d;
        if (dc_dma_init(&d, src, dst, bytes, &mu) != 0) {
            snprintf(err, errn, "dma desc alloc"); return -1;
        }
        int rc = dc_dma_once_ex(&d, src_bypass, dst_bypass);
        dc_dma_destroy(&d);
        if (rc) { snprintf(err, errn, "dma 0x%X", rc); return -1; }
    } else {
        memcpy(dst, src, bytes);
    }
    if (f_dst != WT_DMA_FENCE_NONE &&
        fence_handoff(dst, bytes, fc_peer, (int)f_dst, dst_mem) != FENCE_OK) {
        snprintf(err, errn, "dma fence dst combo"); return -1;
    }
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
    if (g_rtrace) { fprintf(g_rtrace, "unary x=%p y=%p n=%u sub=%u\n", (void*)x, (void*)y, (unsigned)n, (unsigned)subtype); fflush(g_rtrace); }
    /* HVX 快路径(策略 B): 有 f32 内核的 subtype, n >= 32 时走
     * f16→f32(向量 cvt)→内核(就地)→f32→f16。exp/sigmoid/tanh/silu 定义域
     * 无约束; sqrt/rsqrt/log 标量带负输入钳 0 守卫, 暂留标量避免 NaN 回归。 */
    /* 003 T010: ver=3 按 g_planned_engine 分发; ver<=2 保留运行时 fn!=NULL 自决 */
    int want_hvx = (n >= 32u);
    if (b->annot_base) want_hvx = (g_planned_engine == WT_ENG_HVX) && (n >= 32u);
    if (want_hvx) {
        void (*fn)(float*, const float*, uint32_t) = NULL;
        switch (subtype) {
        case 1:  fn = hvhx_v2_exp_f32;     break;
        case 8:  fn = hvhx_v2_sigmoid_f32; break;
        case 9:  fn = hvhx_v2_tanh_f32;    break;
        case 12: fn = hvhx_v2_silu_f32;    break;
        case 14: /* COPY: 位级直通(整面 memcpy 比标量 f16→f32→f16 保真且快) */
            break;
        default: break;
        }
        int hvx_ok = 0;
        if (fn) {
            float* sc = f32_scratch(n, 1, NULL);
            if (sc) {
                hvhx_v2_cvt_f16_to_f32(sc, x, n);
                fn(sc, sc, n);
                hvhx_v2_cvt_f32_to_f16(y, sc, n);
                hvx_ok = 1;
            }
        } else if (subtype == 14) {
            memcpy(y, x, (size_t)n * 2u);
            hvx_ok = 1;
        }
        if (hvx_ok) {
            g_last_engine = WT_ENG_HVX;  /* 003 T017: unary HVX 补漏记 */
            return 0;
        }
        /* HVX 不可用 (无该 subtype 内核 / scratch 不足); HMX 记无路径 (elemwise 无 HMX) */
        if (b->annot_base && engine_fallback_or_die(
                g_planned_engine,
                g_planned_engine == WT_ENG_HMX ? "unary has no HMX path" :
                (fn ? "HVX scratch unavailable" : "unary subtype has no HVX kernel"),
                err, errn) != 0)
            return -1;
        /* 回落裁决已通过 (允许回落): 无重复裁决, 直接落标量 */
    } else if (b->annot_base && g_planned_engine != WT_ENG_SCALAR) {
        /* 计划非标量但未满足 HVX 形状门 (或计划 HMX 无路径) → 按回落策略裁决。 */
        if (engine_fallback_or_die(g_planned_engine,
                g_planned_engine == WT_ENG_HMX ? "unary has no HMX path" :
                "unary shape gate (n<32)", err, errn) != 0)
            return -1;
    }
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
        case 13: /* SOFTPLUS: 稳定化 log(1+e^x), 与 host ops.cpp Neuron case 7 逐字同款 */
            r = (v > 30.0f) ? v : (v < -30.0f ? expf(v) : log1pf(expf(v))); break;
        case 14: r = v; break;  /* COPY */
        default:
            /* P1 消除静默兜底: 未知 subtype 不得静默直通(此前 0xFFFFFFFF
             * 拷贝 = 0.8B 全 -inf 根因之一)。与 emit 侧硬错误同口径。 */
            snprintf(err, errn, "unary unknown subtype %u", (unsigned)subtype);
            return -1;
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
        /* 输出元素数 = 三者广播最大值; true 为标量(an=1)时不得只写 1 元素
         * 余下读旧 temp 垃圾 (GDN 门控链全 nan 根因之一)。numpy 广播语义:
         * 各源按标量/平铺/全量三态取址 (i%cn 兜底 1<cn<n 部分广播越界读;
         * emit 通常已把 [1,1,64,64]→16 头物化为全尺寸 temp, 此处为防御性兜底)。 */
        uint32_t n = cn > an ? cn : an;
        n = fn > n ? fn : n;
        uint16_t* y = (uint16_t*)temp_get(op->args[3], n * 2u);
        if (!c || !tv || !fv || !y || n == 0) { snprintf(err, errn, "select ref fail"); return -1; }
        for (uint32_t i = 0; i < n; i++) {
            float cv = f16_to_f32(c[cn <= 1u ? 0 : (cn < n ? i % cn : i)]);
            float r = (cv != 0.0f)
                ? f16_to_f32(tv[an <= 1u ? 0 : (an < n ? i % an : i)])
                : f16_to_f32(fv[fn <= 1u ? 0 : (fn < n ? i % fn : i)]);
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
    /* HVX 快路径(策略 B): add/sub/mul 位级保真, div < 1 ULP。n >= 32 时
     * f16→f32(向量 cvt)→f32 内核→f32→f16。 */
    /* 003 T010: ver=3 按 g_planned_engine 分发; ver<=2 保留运行时 fn!=NULL 自决 */
    int want_hvx = (n >= 32u);
    if (b->annot_base) want_hvx = (g_planned_engine == WT_ENG_HVX) && (n >= 32u);
    if (want_hvx) {
        void (*fn)(float*, const float*, const float*, uint32_t) = NULL;
        switch (subtype) {
        case 0: fn = hvhx_v2_add_f32; break;
        case 1: fn = hvhx_v2_sub_f32; break;
        case 2: fn = hvhx_v2_mul_f32; break;
        case 3: fn = hvhx_v2_div_f32; break;
        default: break;
        }
        int hvx_ok = 0;
        if (fn) {
            uint32_t st = 0;
            float* sc = f32_scratch(n, 3, &st);
            if (sc) {
                float* A = sc;
                float* B = sc + st;
                float* C = sc + 2 * st;
                hvhx_v2_cvt_f16_to_f32(A, a, n);
                hvhx_v2_cvt_f16_to_f32(B, bb, n);
                fn(C, A, B, n);
                hvhx_v2_cvt_f32_to_f16(y, C, n);
                hvx_ok = 1;
            }
        }
        if (hvx_ok) {
            g_last_engine = WT_ENG_HVX;  /* 003 T017: binary HVX 补漏记 */
            return 0;
        }
        /* HVX 不可用 (无该 subtype 内核 / scratch 不足); HMX 记无路径 (elemwise 无 HMX) */
        if (b->annot_base && engine_fallback_or_die(
                g_planned_engine,
                g_planned_engine == WT_ENG_HMX ? "binary has no HMX path" :
                (fn ? "HVX scratch unavailable" : "binary subtype has no HVX kernel"),
                err, errn) != 0)
            return -1;
        /* 回落裁决已通过 (允许回落): 无重复裁决, 直接落标量 */
    } else if (b->annot_base && g_planned_engine != WT_ENG_SCALAR) {
        /* 003 T010/T011: 计划非标量但 want_hvx==false (HMX 无路径 / HVX 形状门不满足)
         * → 按回落策略裁决, 禁止回落则硬报错, 绝不静默落标量。 */
        if (engine_fallback_or_die(g_planned_engine,
                g_planned_engine == WT_ENG_HMX ? "binary has no HMX path" :
                "binary shape gate (n<32)", err, errn) != 0)
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
        case 4: r = (x0 == x1) ? 1.0f : 0.0f; break;  /* EQ: bool 语义, 与 host ops.cpp oper==3 同款 */
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
    /* HVX 快路径(策略 B): 每行 n%32==0 时走 fused softmax(HVX), 否则回落标量。
     * 整面 f16→f32 一次 cvt → 逐行 softmax_f32 → f32→f16。 */
    /* 003 T010: ver=3 按 g_planned_engine 分发; ver<=2 保留运行时自决 */
    int want_hvx = (rows >= 1u && (n % 32u) == 0u && (size_t)rows * n >= 32u);
    if (b->annot_base)
        want_hvx = (g_planned_engine == WT_ENG_HVX) &&
                   (rows >= 1u && (n % 32u) == 0u && (size_t)rows * n >= 32u);
    if (want_hvx) {
        uint32_t st = 0;
        float* sc = f32_scratch((uint32_t)((size_t)rows * n), 2, &st);
        if (sc) {
            float* pad = sc + st;
            hvhx_v2_cvt_f16_to_f32(sc, x, (uint32_t)((size_t)rows * n));
            for (uint32_t r = 0; r < rows; r++)
                hvhx_v2_softmax_f32(sc + (size_t)r * n, sc + (size_t)r * n, pad, n);
            hvhx_v2_cvt_f32_to_f16(y, sc, (uint32_t)((size_t)rows * n));
            g_last_engine = WT_ENG_HVX;  /* 003 T017: softmax HVX 补漏记 */
            return 0;
        }
        /* scratch 不足 → HVX 不可用 */
        if (b->annot_base && engine_fallback_or_die(g_planned_engine, "softmax HVX scratch unavailable",
                                                    err, errn) != 0)
            return -1;
        /* 回落裁决已通过 (允许回落): 无重复裁决, 直接落标量 */
    } else if (b->annot_base && g_planned_engine != WT_ENG_SCALAR) {
        /* 计划非标量但 want_hvx==false (HMX 无路径 / 形状门不满足) → 按回落策略裁决。 */
        if (engine_fallback_or_die(g_planned_engine,
                g_planned_engine == WT_ENG_HMX ? "softmax has no HMX path" :
                "softmax shape gate (n%32!=0 or rows*n<32)",
                err, errn) != 0)
            return -1;
    }
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
    /* 哨兵/越界一律硬错误: 旧码 d[0xFFFFFFFF] 野读栈垃圾 (0.8B 全图 1170
     * 个 Reduce 曾因此退化成恒等, l2norm 全灭 → 下游 matmul inf) */
    if (axis >= 4 || subtype > 3) {
        snprintf(err, errn, "reduce axis/subtype bad (axis=%u sub=%u)", axis, subtype);
        return -1;
    }
    /* 泛化单轴: y[o*inner+i] = reduce_l x[(o*L+l)*inner+i]  (末轴时 inner=1
     * 退化为原 contiguous 快路径; 中间轴单点归约也正确) */
    uint32_t L = d[axis];
    if (L == 0) L = 1;
    uint32_t inner = 1;
    for (uint32_t a = axis + 1; a < 4; a++) inner *= d[a];
    if ((uint64_t)L * inner == 0 || n % (L * inner) != 0) {
        snprintf(err, errn, "reduce geom (n=%u L=%u inner=%u)", n, L, inner);
        return -1;
    }
    uint32_t outer = n / (L * inner);
    uint16_t* y = (uint16_t*)temp_get(y_t, (size_t)outer * inner * 2u);
    if (!y) { snprintf(err, errn, "reduce out fail"); return -1; }
    for (uint32_t o = 0; o < outer; o++) {
        for (uint32_t i = 0; i < inner; i++) {
            const uint16_t* base = x + ((size_t)o * L) * inner + i;
            float acc = (subtype == 2) ? -3.0e38f : (subtype == 3) ? 3.0e38f : 0.0f;
            for (uint32_t l = 0; l < L; l++) {
                float v = f16_to_f32(base[(size_t)l * inner]);
                if (subtype == 2)      acc = v > acc ? v : acc;
                else if (subtype == 3) acc = v < acc ? v : acc;
                else                   acc += v;
            }
            if (subtype == 1) acc /= (float)L;
            y[(size_t)o * inner + i] = f32_to_f16(acc);
        }
    }
    return 0;
}

static int exec_pad(const struct wt_blob* b, const struct wt_op* op,
                    char* err, size_t errn) {
    uint32_t x_t = op->args[0], y_t = op->args[1];
    uint32_t outer = op->args[2], in_inner = op->args[3], out_inner = op->args[4];
    uint32_t pad_before = op->args[5];
    uint16_t pv = (uint16_t)op->args[6];
    const uint16_t* x = (const uint16_t*)ref_ptr(b, x_t);
    uint16_t* y = (uint16_t*)temp_get(y_t, (size_t)outer * out_inner * 2u);
    if (!x || !y) { snprintf(err, errn, "pad ref fail"); return -1; }
    if (pad_before + in_inner > out_inner) { snprintf(err, errn, "pad geom"); return -1; }
    for (uint32_t i = 0; i < outer; i++) {
        uint16_t* dst = y + (size_t)i * out_inner;
        for (uint32_t j = 0; j < pad_before; j++) dst[j] = pv;
        memcpy(dst + pad_before, x + (size_t)i * in_inner, (size_t)in_inner * 2u);
        for (uint32_t j = pad_before + in_inner; j < out_inner; j++) dst[j] = pv;
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
    /* 窗序 (host 图语义 lstsq 实锤): y[t] = Σ_j w[j]·x[t-(k-1)+j] (纯卷积,
     * 无激活 — 图有独立 Sigmoid/Mul 节点做 SiLU 分解, 这里再套 silu 会双倍
     * 套用 → cos(dev, silu(host))=0.9999995 实锤), 内存 tap j=0 配最老样本;
     * x 有效行 = seq-k+1 (QNN pad flush: 首 k-1 行隐式零历史, 末 k-1 行隐式
     * 零冲刷), 越界按零。旧码 w[j] 配 x[t-j] (tap 反序) 且无尾窗界 — 0.8B
     * GDN conv 全层错。 */
    uint32_t in_rows = (seq >= k) ? (seq - k + 1) : 0;
    for (uint32_t t = 0; t < seq; t++)
        for (uint32_t c = 0; c < C; c++) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < k; j++) {
                int32_t src = (int32_t)t - (int32_t)(k - 1) + (int32_t)j;
                if (src < 0 || src >= (int32_t)in_rows) continue;
                acc += f16_to_f32(x[(size_t)src * C + c]) *
                       f16_to_f32(w[(size_t)j * C + c]);
            }
            y[(size_t)t * C + c] = f32_to_f16(acc);
        }
    return 0;
}

static int exec_gather(const struct wt_blob* b, const struct wt_op* op,
                       char* err, size_t errn) {
    /* host ops.cpp Gather 逐字移植: out 坐标按 od(=td 把 axis 换成 xd)分解,
     * idx 块取 r=idx[il] 查 tbl[axis], 余维直通; 越界 r 出零。
     * args: [tbl,idx,out,n, packed(axis|xrank<<8|irank<<16), td0..4, xd0..4]
     * tbl/idx 双态: const → 0x8000|slot, 计算 op 输出 → 裸 temp id
     * (GDN gather 的 tbl 是 ScatterNd/CumulativeSum 的 state/门控 temp) */
    uint32_t out_t = op->args[2];
    uint32_t n = op->args[3], packed = op->args[4];
    uint32_t axis = packed & 0xFF, xrank = (packed >> 8) & 0xFF, irank = (packed >> 16) & 0xFF;
    uint32_t td[5] = {op->args[5], op->args[6], op->args[7], op->args[8], op->args[9]};
    uint32_t xd[5] = {op->args[10], op->args[11], op->args[12], op->args[13], op->args[14]};
    const uint16_t* tbl = (const uint16_t*)ref_ptr(b, op->args[0]);
    const int32_t* idx = (const int32_t*)ref_ptr(b, op->args[1]);
    uint16_t* y = (uint16_t*)temp_get(out_t, n * 2u);
    if (!tbl || !idx || !y) { snprintf(err, errn, "gather ref fail"); return -1; }
    if (axis >= irank || irank > 5 || xrank > 5 || xrank == 0) {
        snprintf(err, errn, "gather axis/rank bad (a=%u x=%u i=%u)", axis, xrank, irank);
        return -1;
    }
    /* od = td 把 axis 维换成 xd (out 紧凑形); 三套 stride 预计算 */
    uint32_t od5[5] = {1, 1, 1, 1, 1};
    for (uint32_t d = 0; d < axis && d < 5; d++) od5[d] = td[d];
    for (uint32_t k = 0; k < xrank && axis + k < 5; k++) od5[axis + k] = xd[k];
    for (uint32_t d = axis + 1; d < irank && d < 5; d++) {
        uint32_t ocd = d + xrank - 1;
        if (ocd < 5) od5[ocd] = td[d];
    }
    size_t os[5] = {1, 1, 1, 1, 1}, ts[5] = {1, 1, 1, 1, 1}, xs[5] = {1, 1, 1, 1, 1};
    for (int d = 3; d >= 0; d--) {
        os[d] = os[d + 1] * od5[d + 1];
        ts[d] = ts[d + 1] * td[d + 1];
        xs[d] = xs[d + 1] * xd[d + 1];
    }
    for (uint32_t i = 0; i < n; i++) {
        size_t rem = i;
        uint32_t oc[5];
        for (int d = 0; d < 5; d++) { oc[d] = (uint32_t)(rem / os[d]); rem %= os[d]; }
        size_t il = 0;
        for (uint32_t k = 0; k < xrank && k < 5; k++)
            il += (oc[axis + k] % xd[k]) * xs[k];
        int32_t r = idx[il];
        if (r < 0) r += (int32_t)td[axis];
        if (r < 0 || r >= (int32_t)td[axis]) { y[i] = 0; continue; }
        size_t src = 0;
        for (uint32_t d = 0; d < axis && d < 5; d++) src += (size_t)oc[d] * ts[d];
        src += (size_t)r * ts[axis];
        for (uint32_t d = axis + 1; d < irank && d < 5; d++) {
            uint32_t ocd = d + xrank - 1;
            src += (size_t)(ocd < 5 ? oc[ocd] : 0) * ts[d];
        }
        y[i] = tbl[src];
    }
    return 0;
}

/* A3② ScatterNd 真语义 (opcode 28): out = data 拷贝后, 按 n_idx 组 K 维
 * 坐标把 updates 块写入。恒等拷贝=数值死刑 (docs/A3_COPY_SEM_AUDIT.md 判决:
 * L0 cos 0.102)。与 host execute ScatterNd (ops.cpp) 同口径:
 * K=indices 末维, n_idx=前导维积, block=updates/n_idx; 坐标越界跳过。 */
static int exec_scatter_nd(const struct wt_blob* b, const struct wt_op* op,
                           char* err, size_t errn) {
    const uint16_t* data = (const uint16_t*)ref_ptr(b, op->args[0]);
    const int32_t* idxs  = (const int32_t*)ref_ptr(b, op->args[1]);
    const uint16_t* upd  = (const uint16_t*)ref_ptr(b, op->args[2]);
    const uint32_t out_t = op->args[3];
    const uint32_t n     = op->args[4];
    const uint32_t rank  = op->args[5];
    const uint32_t K     = op->args[11];
    const uint32_t n_idx = op->args[12];
    const uint32_t block = op->args[13];
    uint16_t* y = (uint16_t*)temp_get(out_t, (size_t)n * 2u);
    if (!data || !idxs || !upd || !y) { snprintf(err, errn, "scatter_nd ref fail"); return -1; }
    if (rank < 1 || rank > 5 || K < 1 || K > rank) { snprintf(err, errn, "scatter_nd rank/K"); return -1; }
    uint32_t od[5];
    for (uint32_t i = 0; i < 5; i++) od[i] = (i < rank) ? op->args[6 + i] : 1u;
    size_t ostr[5];
    ostr[4] = 1;
    for (int d = 3; d >= 0; d--) ostr[d] = ostr[d + 1] * od[d + 1];
    memcpy(y, data, (size_t)n * 2u);
    for (uint32_t e = 0; e < n_idx; e++) {
        size_t base = 0;
        int bad = 0;
        for (uint32_t k = 0; k < K; k++) {
            int32_t c = idxs[(size_t)e * K + k];
            if (c < 0 || c >= (int32_t)od[k]) { bad = 1; break; }
            base += (size_t)c * ostr[k];
        }
        if (bad || base + block > n) continue;
        memcpy(y + base, upd + (size_t)e * block, (size_t)block * 2u);
    }
    return 0;
}

/* f16×f16 GEMM (float 图; f32 累加 f16 存储)。M3c 正确性版 —— 性能版
 * 走 HMX (M7)。flags bit0 = a 转置(存 [K,M]), bit1 = w 转置(存 [N,K]),
 * bit2 = batched BMM(attention q·kᵀ/probs·v; 批数在高 16 位) */
/* HMX GEMM 快路径 (策略 B 同款): M/K/N 均 32 倍数 → crouton repack + HMX 点积。
 * act/wgt/out/scales 放 hmx_runtime VTCM (mxmem 只能访问 VTCM), 2KB 对齐。
 * 失败(未 setup/超 VTCM/wtcache 已占同一物理 VTCM)返回 -1, 回落标量。
 * 仅 DSP (__HVX__/__hexagon__) 编译; host 无 HMX, 走标量。 */
#if defined(__HVX__) || defined(__hexagon__)
static int exec_matmul_f16_hmx(const uint16_t* a, const uint16_t* w, uint16_t* y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t n0, uint32_t N_full,
                               int t0, int t1, char* err, size_t errn,
                               const uint16_t* bias) {
    /* 003 T014 attach: 同 exec_conv2d_hmx —— wtcache 持 VTCM 时挂共享 scratch
     * 区, 不再整体放弃 (旧行为=静默回落标量)。scratch 覆盖 W4A16 arena →
     * matmul_carved 清零, 下一个 W4A16 op 重 carve。 */
    if (hmx_runtime_get_vtcm_base() == NULL) {
        if (g_exec.engine_ready) {
            void* vb = NULL; uint32_t vs = 0, pc = 0; void* pb = NULL;
            wtcache_layout(g_exec.wc, &vb, &vs, &pb, &pc);
            uint32_t off = (pc + 2047u) & ~2047u;
            uint32_t hi = (vs > g_exec.vtcm_pool_size) ? (vs - g_exec.vtcm_pool_size) : 0u;
            uint32_t cap = (vb && hi > off) ? (hi - off) : 0u;
            if (cap == 0u) {
                snprintf(err, errn, "matmul_f16 attach no scratch (vs=%lu pool=%lu)",
                         (unsigned long)vs, (unsigned long)g_exec.vtcm_pool_size);
                return -1;
            }
            rtrace("hmx attach cap", (int)cap);
            hmx_runtime_attach((uint8_t*)vb + off, cap);
            g_exec.matmul_carved = 0;
        } else if (hmx_runtime_setup(2u * 1024u * 1024u) != 0) {
            snprintf(err, errn, "matmul_f16 hmx setup fail"); return -1;
        }
    }
    char* vtcm = (char*)hmx_runtime_get_vtcm_base();
    unsigned int vtcm_sz = hmx_runtime_get_vtcm_size();
    if (!vtcm || vtcm_sz == 0) return -1;

    uint32_t n_row = M / 32u, n_col = N / 32u, n_dot = K / 32u;
    size_t a_bytes = (size_t)M * (size_t)K * 2u;
    size_t w_bytes = (size_t)N * (size_t)K * 2u;
    size_t o_bytes = (size_t)M * (size_t)N * 2u;
    size_t off_a = 0;
    size_t off_w = (a_bytes + 2047u) & ~(size_t)2047u;
    size_t off_o = (off_w + w_bytes + 2047u) & ~(size_t)2047u;
    size_t off_s = (off_o + o_bytes + 2047u) & ~(size_t)2047u;
    size_t need = off_s + 2048u;
    if (need > (size_t)vtcm_sz) { snprintf(err, errn, "matmul_f16 hmx vtcm overflow"); return -1; }

    uint16_t* act_cr = (uint16_t*)(vtcm + off_a);
    uint16_t* wgt_cr = (uint16_t*)(vtcm + off_w);
    uint16_t* out_cr = (uint16_t*)(vtcm + off_o);
    uint16_t* sca    = (uint16_t*)(vtcm + off_s);

    /* identity scales: mxmem2 读 256B = 32 通道 x 64bit, 分两向量(先 32 通道
     * 低 32bit = [scale fp16, out_bias fp16] 交叠, 再高 32bit = shape/input-bias/尾数)。
     * scale=1.0 (0x3C00) 在偶位 uint16[2c] (c in 0..31), bias=0 奇位, 高 32bit 全 0。
     * 注: 旧 [64 scale][64 bias] 或偶位填 64 项都会踩高字(shape[42:40]/input bias[63:43])。 */
    memset(sca, 0, 2048u);
    for (uint32_t i = 0; i < 32u; ++i) sca[2u * i] = 0x3C00u;

    /* pack act: A[M,K] (或 t0 转置) → crouton[(r*n_dot+kt)*1024] */
    for (uint32_t r = 0; r < n_row; ++r)
        for (uint32_t kt = 0; kt < n_dot; ++kt) {
            uint16_t* tile = act_cr + ((size_t)r * n_dot + kt) * 1024u;
            for (uint32_t m = 0; m < 32u; ++m)
                for (uint32_t k = 0; k < 32u; ++k) {
                    size_t src = t0 ? ((size_t)(kt * 32u + k) * M + (r * 32u + m))
                                    : ((size_t)(r * 32u + m) * K + (kt * 32u + k));
                    tile[hmx_crouton_pos(m, k)] = a[src];
                }
        }

    /* pack wgt: B[K,N] (或 t1 转置) → crouton[(c*n_dot+kt)*1024] (row=ic=k, col=oc=n)。
     * 002 US1: 全局列号 = n0 + c*32 + n, 行主 [K,N_full] 布局 stride=N_full
     * (n0=0/N_full=N 时与整 op 逐字节同公式) */
    for (uint32_t c = 0; c < n_col; ++c)
        for (uint32_t kt = 0; kt < n_dot; ++kt) {
            uint16_t* tile = wgt_cr + ((size_t)c * n_dot + kt) * 1024u;
            for (uint32_t k = 0; k < 32u; ++k)
                for (uint32_t n = 0; n < 32u; ++n) {
                    size_t src = t1 ? ((size_t)(n0 + c * 32u + n) * K + (kt * 32u + k))
                                    : ((size_t)(kt * 32u + k) * N_full + (n0 + c * 32u + n));
                    tile[hmx_crouton_pos(k, n)] = w[src];
                }
        }

    /* HMX GEMM: 复刻 hmx_convf16.c convf16_hmx_multi 已验证路径
     * (crouton pair-interleave + mxmem:deep 累加 + cvt.hf=acc(2) 写回)。
     * 勿用 hvhx_v2_hmx_gemm_dot_fp16 — 那是 ggml core_dot 移植路径, 布局
     * (row-major pair) 与写回 (:after.hf) 都不同, 未上板验证。 */
    for (uint32_t r = 0; r < n_row; ++r) {
        for (uint32_t c = 0; c < n_col; ++c) {
            uint16_t* out_tile = out_cr + ((size_t)r * n_col + c) * 1024u;
            /* 005 US3b (T018): bias 走 scale 区奇位 out_bias —— cvt 一步
             * f16(acc·1.0+bias), 与 exec_conv2d_hmx 同技法 (该路已上板验证);
             * 标量路径同口径 bias[全局列号] 单次舍入。bias==NULL 时保持
             * 恒等页, 已验证 GEMM 路径零扰动。 */
            if (bias)
                for (uint32_t n = 0; n < 32u; ++n)
                    sca[2u * n + 1u] = bias[n0 + c * 32u + n];
            hmx_enable_execution();
            hmx_unit_acquire();
            asm volatile("mxclracc.hf" ::: "memory");
            asm volatile("bias = mxmem2(%0)" :: "r"(sca) : "memory");
            for (uint32_t kt = 0; kt < n_dot; ++kt) {
                const uint16_t* act_tile = act_cr + ((size_t)r * n_dot + kt) * 1024u;
                const uint16_t* wgt_tile = wgt_cr + ((size_t)c * n_dot + kt) * 1024u;
                asm volatile(
                    "{ activation.hf = mxmem(%0, %1):deep\n"
                    "  weight.hf = mxmem(%2, %3) }\n"
                    :: "r"(act_tile), "r"(0x7FFu),
                       "r"(wgt_tile), "r"(0x7FFu) : "memory");
            }
            memset(out_tile, 0, 2048u);
            asm volatile(
                "cvt.hf = acc(%0)\n"
                "mxmem(%1, %2) = cvt\n"
                :: "r"(2), "r"(out_tile), "r"(0) : "memory");
            hmx_unit_release();
            hmx_disable_execution();
        }
    }

    /* unpack out: crouton[(r*n_col+c)*1024] → row-major y[M,N_full] 列子区间 [n0, n0+N) */
    for (uint32_t r = 0; r < n_row; ++r)
        for (uint32_t c = 0; c < n_col; ++c) {
            uint16_t* tile = out_cr + ((size_t)r * n_col + c) * 1024u;
            for (uint32_t m = 0; m < 32u; ++m)
                for (uint32_t n = 0; n < 32u; ++n)
                    y[(size_t)(r * 32u + m) * N_full + (n0 + c * 32u + n)] = tile[hmx_crouton_pos(m, n)];
        }
    g_last_engine = WT_ENG_HMX;  /* W-P3: f16 GEMM 走 HMX crouton 路径 */
    return 0;
}
#endif /* __HVX__ / __hexagon__ */

static int exec_matmul_f16_impl(const struct wt_blob* b, const struct wt_op* op,
                                char* err, size_t errn, const uint16_t* bias) {
    uint32_t a_t = op->args[0], w_s = op->args[1], out_t = op->args[2];
    uint32_t M = op->args[3], K = op->args[4];
    uint32_t N_full, n0, N, flags;
    if (op->opcode == OP_MATMUL_F16_SPLIT) {
        /* 002 US1: [a_ref,w_s,out_t,M,K,N_full,n0,N_chunk,flags]
         * 本 chunk 只算列区间 [n0, n0+N), 写共享 out temp 的列子区间 */
        N_full = op->args[5]; n0 = op->args[6]; N = op->args[7]; flags = op->args[8];
    } else {
        N_full = N = op->args[5]; n0 = 0; flags = op->args[6];
    }
    int t0 = (int)(flags & 1u), t1 = (int)(flags & 2u);
    int batched = (int)(flags & 4u);
    uint32_t Bn = flags >> 16;
    if (Bn == 0) Bn = 1;
    /* FC bias (opcode 36) 仅非 batched 整 op (emit 侧保证; 手构图违禁硬错) */
    if (bias && batched) { snprintf(err, errn, "matmul_f16_bias 不支持 batched"); return -1; }
    const uint16_t* a = (const uint16_t*)ref_ptr(b, a_t);
    /* w 可为 slot(权重)或 temp(运行时 B, 如 attention q·kᵀ) */
    const uint16_t* w = (const uint16_t*)ref_ptr(b, op->args[1]);
    uint16_t* y = (uint16_t*)temp_get(out_t, (size_t)M * N_full * Bn * 2u);
    if (!a || !w || !y) { snprintf(err, errn, "matmul_f16 ref fail"); return -1; }
#if defined(__HVX__) || defined(__hexagon__)
    /* HMX 快路径: M/K/N 均 32 倍数 → crouton + HMX GEMM。
     * batched attention (q·kᵀ, Bn 头) 同走 HMX: 逐 batch 切片调 exec_matmul_f16_hmx,
     * 批维在 A/W/Y 最外层连续 (与标量 batched 路径同布局, 见下方标量分支)。
     * SPLIT 回落全有或全无 (FR-013 ④): 同逻辑 matmul 的 chunk 间共享引擎
     * 选择, 首 chunk 决定, 续 chunk (三引用一致) 沿用; 混合路径破逐字节
     * 恒等, 违禁即硬错误。chunk 间必有别的 op 插入时 key 不匹配, 重新决定。 */
    static struct { uint32_t a_t, w_s, out_t; int engine; } g_split_prev = {0, 0, 0, 0};
    /* 003 T007: ver=3 按编译期 g_planned_engine 分发; ver<=2 保留运行时 shape 门。
     * matmul 家族仅 HMX/标量两路 (无 HVX 路径)。 */
    int want_hmx;
    if (b->annot_base) {
        if (g_planned_engine == WT_ENG_HMX) {
            want_hmx = 1;
        } else if (g_planned_engine == WT_ENG_HVX) {
            /* 计划 HVX 但 matmul 无 HVX 路径 → 按回落策略裁决 */
            if (engine_fallback_or_die(g_planned_engine, "matmul has no HVX path", err, errn) != 0) return -1;
            want_hmx = 0;
        } else {
            want_hmx = 0;  /* planned scalar */
        }
    } else {
        want_hmx = ((M % 32u) == 0u && (K % 32u) == 0u && (N % 32u) == 0u);
    }
    /* 005 US3b (T019): bias 的 HMX 路径 SC-005 已过, 生产默认走 HMX;
     * g_force_bias_scalar 仅测试例 (46) 造标量参照用。 */
    if (bias && g_force_bias_scalar) want_hmx = 0;
    if (want_hmx) {
        int engine = 0;
        if (op->opcode == OP_MATMUL_F16_SPLIT &&
            g_split_prev.out_t == out_t && g_split_prev.a_t == a_t && g_split_prev.w_s == w_s)
            engine = g_split_prev.engine;  /* 续 chunk: 沿用已定引擎 */
        if (engine == 0) {
            if (batched) {
                /* 全或全无 (与 SPLIT 同纪律): 任一 batch 失败 → 整体回落标量。
                 * W 批跨步 = K*N_full (t1 不改跨步, 见标量 batched 分支下标)。
                 * batched 恒非 SPLIT (SPLIT 仅超界整 op), 不碰 g_split_prev。 */
                engine = 1;
                for (uint32_t bb = 0; bb < Bn; ++bb) {
                    const uint16_t* ab = a + (size_t)bb * M * K;
                    const uint16_t* wb = w + (size_t)bb * K * N_full;
                    uint16_t* yb = y + (size_t)bb * M * N_full;
                    if (exec_matmul_f16_hmx(ab, wb, yb, M, K, N, 0, N_full, t0, t1,
                                            err, errn, NULL) != 0) {
                        engine = -1;
                        break;
                    }
                }
            } else {
                engine = (exec_matmul_f16_hmx(a, w, y, M, K, N, n0, N_full, t0, t1, err, errn,
                                              bias) == 0)
                             ? 1 : -1;
            }
        } else if (engine == 1) {
            if (exec_matmul_f16_hmx(a, w, y, M, K, N, n0, N_full, t0, t1, err, errn,
                                     bias) != 0) {
                snprintf(err, errn, "matmul_f16 split 混合路径 (FR-013 ④ 违禁): 前 chunk HMX 本 chunk 失败");
                return -1;
            }
        }
        if (op->opcode == OP_MATMUL_F16_SPLIT) {
            g_split_prev.a_t = a_t; g_split_prev.w_s = w_s;
            g_split_prev.out_t = out_t; g_split_prev.engine = engine;
        }
        if (engine == 1) return 0;
        /* 003 T007/T011: HMX 失败 → ver=3 按回落策略裁决 (禁止回落硬报错), ver<=2 静默回落。
         * 失败原因已在 err 内 (exec_matmul_f16_hmx 写入), 此处只补回退裁决日志。 */
        if (b->annot_base && engine_fallback_or_die(g_planned_engine,
                err && err[0] ? err : "matmul_f16 HMX invoke failed", err, errn) != 0)
            return -1;
        /* engine == -1 (或非 SPLIT 失败): 回落标量 */
        if (g_planned_engine == WT_ENG_HMX) {
            g_last_engine = WT_ENG_SCALAR;  /* 003 T017: matmul 回落 HMX 时补漏记 */
        } else if (g_planned_engine == WT_ENG_HVX) {
            g_last_engine = WT_ENG_SCALAR;  /* 003 T017: matmul 回落 HVX 时补漏记 */
        }
    }
#endif
    /* 标量路径: 列号 = n0 + n, 行主 stride = N_full (n0=0/N_full=N 即旧公式) */
    if (batched) {
        for (uint32_t bb = 0; bb < Bn; bb++)
            for (uint32_t m = 0; m < M; m++)
                for (uint32_t n = 0; n < N; n++) {
                    float acc = 0.0f;
                    for (uint32_t k = 0; k < K; k++) {
                        float av = f16_to_f32(a[((size_t)bb * M + m) * K + k]);
                        float wv = f16_to_f32(w[t1 ? ((size_t)bb * N_full + n0 + n) * K + k
                                                  : ((size_t)bb * K + k) * N_full + (n0 + n)]);
                        acc += av * wv;
                    }
                    y[((size_t)bb * M + m) * N_full + (n0 + n)] = f32_to_f16(acc);
                }
        return 0;
    }
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                float av = f16_to_f32(a[t0 ? (size_t)k * M + m : (size_t)m * K + k]);
                float wv = f16_to_f32(w[t1 ? (size_t)(n0 + n) * K + k : (size_t)k * N_full + (n0 + n)]);
                acc += av * wv;
            }
            /* FC bias: f32 累加器按列加 (单次舍入, 与 host ops.cpp 同口径) */
            if (bias) acc += f16_to_f32(bias[n0 + n]);
            y[(size_t)m * N_full + (n0 + n)] = f32_to_f16(acc);
        }
    return 0;
}

static int exec_matmul_f16(const struct wt_blob* b, const struct wt_op* op,
                           char* err, size_t errn) {
    return exec_matmul_f16_impl(b, op, err, errn, NULL);
}

/* OP_MATMUL_F16_BIAS (opcode 36, FC bias 折叠): args[0..6] 同 OP_MATMUL_F16,
 * args[7] = 0x8000|bias 槽 (N 元素 f16, 按列加)。bias 槽长校验防野读。 */
static int exec_matmul_f16_bias(const struct wt_blob* b, const struct wt_op* op,
                                char* err, size_t errn) {
    uint32_t bs = op->args[7] & 0x7FFFu;
    if (bs >= b->n_slots || b->slots[bs].len < (size_t)op->args[5] * 2u) {
        snprintf(err, errn, "matmul_f16_bias bias slot bad (s=%u n=%u)",
                 (unsigned)bs, (unsigned)op->args[5]);
        return -1;
    }
    const uint16_t* bias = (const uint16_t*)slot_ptr(b, bs);
    if (!bias) { snprintf(err, errn, "matmul_f16_bias bias ref fail"); return -1; }
    return exec_matmul_f16_impl(b, op, err, errn, bias);
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
    /* HVX 快路径(策略 B): 无 bias 时走 hvhx_v2_rms_norm_mul_f32_rows
     * (逐行 rmsnorm×weight, HVX/标量自适应)。有 bias 时回落标量
     * (HVX 无 rmsnorm+weight+bias 融合内核)。rw%32==0 才触发 HVX 主体。
     *
     * [2026-09-28 精度根因] 该 HVX 路径用 Qf32 定点累加 + 快速 rsqrt
     * (hvx_vec_rsqrt_f32 / Q6_Vqf32_vmpy), 在微小行(0.8B GDN 门控态 rms~1e-4,
     * x² 低于 Qf32 分辨率 2^-31 被截断成 0)上产生 0.5× 量级偏差:
     * 实测 op911(RMSNorm) 512 行中 49 行 cos<0.99, 最差 0.887; 标量 double 路径
     * 同数据 0 行<0.99(最差 0.9967)。故默认关(标量), 仅 GEHTP_HVX_RMSNORM=1
     * 显式开启(待定点 kernel 修好后再默认开)。 */
    static int g_hvx_rmsnorm = -1;
    if (g_hvx_rmsnorm < 0) {
        const char* e = getenv("GEHTP_HVX_RMSNORM");
        g_hvx_rmsnorm = (e && e[0] == '1') ? 1 : 0;  /* 默认关, 显式 1 才开 */
    }
    /* 003 T010: ver=3 按 g_planned_engine 分发 (编译器为准, 覆盖 env); ver<=2 保留 env 自决 */
    int want_hvx = g_hvx_rmsnorm && !bv && m >= 1u && (rw % 32u) == 0u;
    if (b->annot_base)
        want_hvx = (g_planned_engine == WT_ENG_HVX) && !bv && m >= 1u && (rw % 32u) == 0u;
    if (want_hvx) {
        uint32_t st = 0;
        float* sc = f32_scratch(n, 2, &st);
        if (sc) {
            float* A = sc;          /* [n] 输入/输出(就地) */
            float* W = sc + st;     /* [rw] 权重 */
            hvhx_v2_cvt_f16_to_f32(A, x, n);
            hvhx_v2_cvt_f16_to_f32(W, wv, rw);
            hvhx_v2_rms_norm_mul_f32_rows(A, W, A, m, rw, 1e-6f);
            hvhx_v2_cvt_f32_to_f16(y, A, n);
            g_last_engine = WT_ENG_HVX;  /* 003 T017: rmsnorm2 HVX 补漏记 */
            return 0;
        }
        if (b->annot_base && engine_fallback_or_die(g_planned_engine,
                "rmsnorm2 HVX scratch unavailable", err, errn) != 0)
            return -1;
        /* 回落裁决已通过 (允许回落): 无重复裁决, 直接落标量 */
    } else if (b->annot_base && g_planned_engine != WT_ENG_SCALAR) {
        /* 003 T010/T011: 计划非标量但 want_hvx==false (HMX 无路径 / 带 bias /
         * rw%32!=0) → 按回落策略裁决, 禁止回落则硬报错, 绝不静默落标量。 */
        if (engine_fallback_or_die(g_planned_engine,
                g_planned_engine == WT_ENG_HMX ? "rmsnorm2 has no HMX path" :
                (bv ? "rmsnorm2 has bias; HVX path unavailable" :
                      "rmsnorm2 shape gate (rw%32!=0)"),
                err, errn) != 0)
            return -1;
    }
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
    /* 003 T018 补漏 (t2l op_ts v3 机检实锤: emit 对 Eltwise_Broadcast 按
     * n%32 计划 HVX, 但 broadcast 无 HVX/HMX 内核 —— 此前静默标量 =
     * planned!=actual 且 dev=0 的契约违例)。按回落策略裁决。 */
    if (b->annot_base && g_planned_engine != WT_ENG_SCALAR &&
        engine_fallback_or_die(g_planned_engine, "broadcast has no HVX/HMX path",
                               err, errn) != 0)
        return -1;
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
    const uint8_t* tp = NULL;  /* 首个 TEMPOFF 槽载荷(carve 对拍用) */
    uint32_t tn = 0, tvtcm_cap = 0;
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
        tp = p; tn = n; tvtcm_cap = vtcm_cap;
        break;
    }
    /* 003 T013: carve footer ↔ TEMPOFF 双源对拍(重数匹配)。emit 侧已有同款
     * 检查, 这里防 blob 被拼接/改写/异版工具产出后的静默布局漂移 —— 漂移时
     * arena 上界按错误的 live 集算, HMX 直写会踩活跃驻留 temp。 */
    if (b->carve_base && b->n_carve) {
        if (!tp || tvtcm_cap == 0) {
            snprintf(err, errn, "carve footer but no vtcm tempoff pool");
            return -1;
        }
        /* 双向存在性(集合语义): temp id 回收复用同段 VTCM 会有重复对, 合法。 */
        for (uint32_t i = 0; i < tn; i++) {
            const uint8_t* e = tp + 12 + i * 12;
            uint32_t tid, off, sz;
            memcpy(&tid, e, 4); memcpy(&off, e + 4, 4); memcpy(&sz, e + 8, 4);
            if (!(tid & WT_REF_VTCM_FLAG)) continue;
            uint32_t j;
            for (j = 0; j < b->n_carve; j++) {
                const uint8_t* r = b->carve_base + (size_t)j * WT_CARVE_REC_SIZE;
                if (wt_rd_u32(r + 4u) == off && wt_rd_u32(r + 8u) == sz) break;
            }
            if (j == b->n_carve) {
                snprintf(err, errn, "carve/tempoff drift: vtcm tid %u (off %u sz %u) unmatched",
                         (unsigned)(tid & 0x3FFFu), (unsigned)off, (unsigned)sz);
                return -1;
            }
        }
        for (uint32_t j = 0; j < b->n_carve; j++) {
            const uint8_t* r = b->carve_base + (size_t)j * WT_CARVE_REC_SIZE;
            uint32_t opi = wt_rd_u32(r);
            uint32_t co = wt_rd_u32(r + 4u), cs = wt_rd_u32(r + 8u);
            uint32_t st = wt_rd_u32(r + 12u), en = wt_rd_u32(r + 16u);
            if (co + cs > tvtcm_cap) {
                snprintf(err, errn, "carve rec bad: op %u off %u sz %u cap %u",
                         (unsigned)opi, (unsigned)co, (unsigned)cs, (unsigned)tvtcm_cap);
                return -1;
            }
            if (st > opi || opi >= en) {
                snprintf(err, errn, "carve rec window bad: op %u not in [%u,%u)",
                         (unsigned)opi, (unsigned)st, (unsigned)en);
                return -1;
            }
            uint32_t i;
            for (i = 0; i < tn; i++) {
                const uint8_t* e = tp + 12 + i * 12;
                uint32_t tid2, off2, sz2;
                memcpy(&tid2, e, 4); memcpy(&off2, e + 4, 4); memcpy(&sz2, e + 8, 4);
                if ((tid2 & WT_REF_VTCM_FLAG) && off2 == co && sz2 == cs) break;
            }
            if (i == tn) {
                snprintf(err, errn, "carve/tempoff drift: op %u (off %u sz %u) phantom",
                         (unsigned)opi, (unsigned)co, (unsigned)cs);
                return -1;
            }
        }
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

static int xop_dma(const struct wt_op_ctx* cx) {
    return exec_dma(cx->b, cx->op, cx->err, cx->errn);
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

static int xop_pad(const struct wt_op_ctx* cx) {
    return exec_pad(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_gather(const struct wt_op_ctx* cx) {
    return exec_gather(cx->b, cx->op, cx->err, cx->errn);
}

static int xop_scatter_nd(const struct wt_op_ctx* cx) {
    return exec_scatter_nd(cx->b, cx->op, cx->err, cx->errn);
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
static int xop_matmul_f16_bias(const struct wt_op_ctx* cx) {
    return exec_matmul_f16_bias(cx->b, cx->op, cx->err, cx->errn);
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
    [OP_DMA] = xop_dma,
    [OP_PAD_F16] = xop_pad,
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
    [OP_SCATTER_ND_F16] = xop_scatter_nd,
    [OP_ARGMAX_F16] = xop_argmax,
    [OP_KV_APPEND_F16] = xop_kv_unimpl,
    [OP_KV_GATHER_F16] = xop_kv_unimpl,
    [OP_MATMUL_F16] = xop_matmul_f16,
    [OP_MATMUL_F16_SPLIT] = xop_matmul_f16,
    [OP_MATMUL_F16_BIAS] = xop_matmul_f16_bias,
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
    const int64_t t_run0 = HAP_perf_get_time_us();  /* PROF W-P2: ts 时间轴零点 */
    for (uint32_t ii = 0; ii < count; ii++) {
        uint32_t i = first + ii;
        const struct wt_op* op = &b->ops[i];
        int64_t t0 = HAP_perf_get_time_us();
        dc_dma_reset();                 /* W-P3: 每 op 前清 UserDMA 累加器 */
        g_last_engine = WT_ENG_SCALAR;  /* W-P3: 默认标量, exec_* 内部覆盖 */
        g_op_deviation = 0;             /* 003 T018: 偏差标志逐 op 复位 */
        g_current_op = ii;              /* 回退日志上下文: 段内 op 序号 (与 optrace 行一致) */
        g_current_op_global = i;        /* 003 P2: carve 记录键 = 全局 op 序号 */
        g_current_opcode = op->opcode;
        if (b->annot_base) {
            uint32_t an = wt_rd_u32(b->annot_base + (size_t)i * 4u);
            g_planned_engine = wt_annot_engine(an);
            g_planned_fb_allowed = wt_annot_fb_allowed(an);
            g_planned_fb_engine = wt_annot_fb_engine(an);
        } else {
            g_planned_engine = WT_ENG_SCALAR;
            g_planned_fb_allowed = 1u;   /* legacy: 恒允许回落 (旧自决语义) */
            g_planned_fb_engine = WT_ENG_SCALAR;
        }
        int rc;
        if (op->opcode >= sizeof(g_op_exec_table) / sizeof(g_op_exec_table[0]) ||
            !g_op_exec_table[op->opcode]) {
            snprintf(err, errn, "opcode %u unhandled", (unsigned)op->opcode);
            rc = -1;
        } else {
            /* PROF W-P2: per-op trace 行受 g_trace_on 门控 (默认关=零落盘污染) */
            if (g_trace_on) {
                if (!g_rtrace) g_rtrace = fopen("/data/local/tmp/hrt/gehtp/optrace.txt", "a");
                if (g_rtrace) {
                    fprintf(g_rtrace, "pre%u code=%u a=%u,%u,%u,%u pool=%u\n",
                            (unsigned)ii, (unsigned)op->opcode,
                            (unsigned)op->args[0], (unsigned)op->args[1],
                            (unsigned)op->args[2], (unsigned)op->args[3],
                            (unsigned)g_exec.pool_used);
                    fflush(g_rtrace);
                }
            }
            const struct wt_op_ctx cx = {b, op, engine_m, err, errn};
            rc = g_op_exec_table[op->opcode](&cx);
            if (rc == 0) exec_hook_dump(ii);
            /* 003 T020 兜底: 已接线算子之外若有静默偏航 (rc==0 但
             * planned!=actual 且未裁决), 机检兜住 —— 允许回落且实跑即
             * fallback(scalar) 的补记 dev=1 (诚实报告既成事实); 其余按
             * 契约中止。broadcast 未接线曾在此类现形。 */
            if (b->annot_base && rc == 0 &&
                g_planned_engine != g_last_engine && g_op_deviation == 0) {
                if (g_planned_fb_allowed && g_planned_fb_engine == WT_ENG_SCALAR &&
                    g_last_engine == WT_ENG_SCALAR) {
                    g_op_deviation = 1;
                    log_scalar_fallback(g_planned_engine, "unwired op ran scalar (post-hoc)");
                } else {
                    snprintf(err, errn,
                             "op %u unadjudicated engine deviation (planned %u actual %u)",
                             (unsigned)i, (unsigned)g_planned_engine,
                             (unsigned)g_last_engine);
                    rc = -1;
                }
            }
        }
        g_exec.st.ops++;
        {
            const int64_t t1 = HAP_perf_get_time_us();
            if (op_us) op_us[ii] = t1 - t0;
            /* PROF W-P2: 打点对, 索引 = 全局 op 序号 (run_range 分段安全) */
            if (g_ts_buf && i < g_ts_cap) {
                int64_t dma_us = 0, dma_bytes = 0;
                dc_dma_get(&dma_us, &dma_bytes);   /* W-P3: 本 op UserDMA 量 */
                g_ts_buf[i].start_us  = (uint32_t)(t0 - t_run0);
                g_ts_buf[i].dur_us    = (uint32_t)(t1 - t0);
                g_ts_buf[i].dma_us    = (uint32_t)dma_us;
                g_ts_buf[i].dma_bytes = (uint32_t)dma_bytes;
                g_ts_buf[i].engine    = g_last_engine;
                /* 003 T019: 计划引擎 + 偏差。rc!=0 = 本 op 中止 (die 路径
                 * 或执行错误) → 统一记 2; 成功 op 记裁决点置位的 0/1。
                 * 中止 op 的 engine = 默认 SCALAR (exec 未及覆盖)。 */
                g_ts_buf[i].planned_engine = g_planned_engine;
                g_ts_buf[i].deviation = rc ? 2u : g_op_deviation;
            }
        }
        if (g_trace_on) {
            /* 统一 trace 句柄铁律(M4.2 实锤: 同路径双 FILE* 在 DSP farf
             * 下句柄冲突挂死) —— 与 rtrace 共用 g_rtrace, 禁止自建 gf。 */
            if (!g_rtrace) g_rtrace = fopen("/data/local/tmp/hrt/gehtp/optrace.txt", "a");
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
/* 路线B: 外部权重区注入(在 run/run_io 之前调一次; NULL=禁用 EXT_WGT 槽)。
 * 与 run_io 解耦(权重区跨多次 run 常驻, 输入每次注入) —— 签名不动,
 * 现有 5 个 runner 调用零影响。 */
void wt_exec_set_ext_weights(const void* wgt_ptr) {
    g_ext_wgt = (const uint8_t*)wgt_ptr;
}

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
