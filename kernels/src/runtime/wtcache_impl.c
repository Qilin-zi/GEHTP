/* wtcache_impl.c — 模块 1-C 权重 cache/pin DSP 侧实现
 *
 * 三段可复用机制抄自 /disk1/V81Dev/vtcm-dma-bench/vtcm_dma_bench.c (52f67807 实测):
 *   - power_on_hvx_hmx: HVX/HMX DCVS MAX + power_up (逐字抄)
 *   - VTCM acquire: query_VTCM → attr_init → set_vtcm_param → acquire → get_vtcm_ptr → hmx_lock
 *   - UserDMA: dma_desc_init/submit/wait_for_idle + dma_desc_is_done (环用)
 *   - cache 一致性: qurt_mem_cache_clean FLUSH/INVALIDATE
 *
 * VTCM 布局 (16MB, 从 base 起 128B 对齐顺序切):
 *   [pin_pool 12MB] → [ring_in 4×tile] → [ring_out 4×tile] → [temp 余量]
 * pin 区用 bump 分配器 (pin_weight 顺序领槽, 不回收); ring 区固定深度。
 * temp 区给 T1/T6 的干扰模拟分配 (故意与 pin/ring 隔离, 证明不踩)。
 *
 * DMA 在途模型: 单 UserDMA 引擎, desc 菊花链 + g_last_desc 追加。
 * depth=4 在途窗口: prime 提交 4 个 desc, ring_next 每消费 1 个就追加 1 个,
 * 全程不 dma_wait_for_idle (只 dma_desc_is_done 轮询当前消费槽)。
 * HMX compute 与 DMA 是独立硬件单元 → 真重叠 (T5 的物理基础)。
 */
#include "wtcache.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <HAP_farf.h>
#include <HAP_compute_res.h>
#include <HAP_power.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "dma_utils.h"

/* dma_utils.c 的文件级全局 (非 static, 链接可见). dma_desc_submit 在 dmpoll()==RUN
 * 路径会 dmlink(g_last_desc, new) — 若 g_last_desc 指向已 free 的 desc 即野指针写,
 * 污染后续 DMA (T4c all-mode 失败根因). 本文件每次 free 一个 desc 前清掉它. */
extern void* g_last_desc;
static inline void wtc_unlink_last(void* d) { if (d && g_last_desc == d) g_last_desc = NULL; }

#define WTC_VTCM_ALIGN     128u
#define WTC_DEFAULT_PIN_MB 12u

/* 诊断: move_back submit 计数 + 最后一次 submit 的 slot/dst (T4 调试用) */
volatile int g_wtc_mb_count = 0;
volatile void* g_wtc_mb_last_dst = 0;
volatile int g_wtc_mb_last_slot = -1;

/* round v up to next multiple of a (power of 2) */
static inline uint32_t round_up(uint32_t v, uint32_t a) {
    return (v + a - 1u) & ~(a - 1u);
}
static inline int is_pow2(uint32_t v) { return v && !(v & (v - 1)); }

/* ============================================================
 * power on HVX/HMX (逐字抄 vtcm_dma_bench.c:power_on_hvx_hmx)
 * ============================================================ */
static int power_on_hvx_hmx(void) {
    HAP_power_request_t req;
    int ctx = 0;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&ctx, &req) != 0) return -1;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = 1;
    req.dcvs_v3.dcvs_enable = 1;
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params = 1;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params = 1;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = 1;
    req.dcvs_v3.sleep_disable = 1;
    if (HAP_power_set((void *)&ctx, &req) != 0) return -2;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&ctx, &req) != 0) return -3;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&ctx, &req) != 0) return -4;
    return 0;
}

/* ============================================================
 * 会话上下文
 * ============================================================ */
struct wtcache_ctx {
    uint8_t  *vtcm_base;     /* 0xFF000000 */
    uint32_t  vtcm_size;     /* 16MB */
    uint32_t  compute_ctx;   /* HAP_compute_res_acquire 返回 */
    /* 布局 (字节偏移 from vtcm_base) */
    uint32_t  pin_cap;       /* pin 区容量 */
    uint32_t  pin_off;       /* pin 区已用 (bump) */
    uint32_t  ring_in_off;   /* ring prefetch 区起点 */
    uint32_t  ring_out_off;  /* ring move_back 区起点 */
    uint32_t  temp_off;      /* temp 区起点 */
    uint32_t  temp_used;     /* temp 已用 (bump, 可回收) */
};

int wtcache_open(struct wtcache_ctx** out, uint32_t pin_cap_bytes) {
    if (!out) return WTC_ERR_RANGE;
    struct wtcache_ctx* c = (struct wtcache_ctx*)calloc(1, sizeof(*c));
    if (!c) return WTC_ERR_OOM;

    int pwr = power_on_hvx_hmx();
    if (pwr != 0) { free(c); FARF(ALWAYS, "wtcache: power FAILED %d", pwr); return WTC_ERR_POWER; }

    unsigned int total = 0, avail = 0;
    compute_res_vtcm_page_t tp, ap;
    if (HAP_compute_res_query_VTCM(0, &total, &tp, &avail, &ap) != 0 || total == 0) {
        total = 16u * 1024u * 1024u;
    }
    c->vtcm_size = total;

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    /* T10: 与 htpw4a16_v81 (ch02 proven) 完全一致的 v1 路径。
     * v2(…,0,0) 在真 deep kernel 上 fault (T10-c 阶段 2 崩溃根因);
     * 不设 cache_mode/serialize — hmx_lock 会失败 (ch02 note)。 */
    HAP_compute_res_attr_set_vtcm_param(&attr, total, 1);
    if (compute_resource_attr_set_hmx_param) {
        compute_resource_attr_set_hmx_param(&attr, 1);
    }

    unsigned int ctx = HAP_compute_res_acquire(&attr, 100000);
    if (ctx == 0) { free(c); FARF(ALWAYS,"wtcache: acquire=0"); return WTC_ERR_VTCM_ACQUIRE; }
    c->compute_ctx = ctx;

    uint8_t* base = (uint8_t*)HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!base) compute_resource_attr_get_vtcm_ptr_v2(&attr, (void**)&base, NULL);
    if (!base) { HAP_compute_res_release(ctx); free(c); FARF(ALWAYS,"wtcache: vtcm NULL"); return WTC_ERR_VTCM_PTR; }
    c->vtcm_base = base;

    /* hmx_lock (ch02 proven, 非 lock2/SHARED — htpw4a16_v81 同款) */
    if (HAP_compute_res_hmx_lock(ctx) != 0) {
        HAP_compute_res_release(ctx); free(c); FARF(ALWAYS,"wtcache: hmx_lock FAIL"); return WTC_ERR_HMX_LOCK;
    }

    memset(c->vtcm_base, 0, c->vtcm_size);
    /* V2.2 修复 (dualdomain run3 / wt_repack W4 同族根因): memset 是 CPU write-back
     * 写, 留下 dirty 零行; 之后 HMX 直写 VTCM 的面 (e.out) 会被这些行逐出时用 0
     * 覆盖。open 末尾对全 VTCM 一次 FLUSH 把零行落定, 后续任何 carve 面都干净。 */
    qurt_mem_cache_clean((qurt_addr_t)c->vtcm_base, c->vtcm_size,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    /* 布局: pin → ring_in → ring_out → temp. ring 区在 ring_init 时再切. */
    c->pin_cap = pin_cap_bytes ? pin_cap_bytes : (WTC_DEFAULT_PIN_MB << 20);
    c->pin_cap = round_up(c->pin_cap, WTC_VTCM_ALIGN);
    if (c->pin_cap > c->vtcm_size) c->pin_cap = c->vtcm_size;
    c->pin_off = 0;
    c->ring_in_off = c->pin_cap;
    c->ring_out_off = c->pin_cap;  /* ring_init 后更新 */
    c->temp_off = c->pin_cap;
    c->temp_used = 0;

    FARF(ALWAYS, "wtcache_open: vtcm=%p size=%u pin_cap=%u (0x%X)",
         c->vtcm_base, c->vtcm_size, c->pin_cap, c->pin_cap);
    *out = c;
    return WTC_OK;
}

int wtcache_hmx_lock(struct wtcache_ctx* ctx) {
    return HAP_compute_res_hmx_lock(ctx->compute_ctx) == 0 ? 0 : -1;
}
int wtcache_hmx_unlock(struct wtcache_ctx* ctx) {
    return HAP_compute_res_hmx_unlock(ctx->compute_ctx) == 0 ? 0 : -1;
}

int wtcache_close(struct wtcache_ctx* c) {
    if (!c) return WTC_ERR_RANGE;
    if (c->compute_ctx) {
        HAP_compute_res_hmx_unlock(c->compute_ctx);
        HAP_compute_res_release(c->compute_ctx);
    }
    free(c);
    return WTC_OK;
}

void wtcache_layout(const struct wtcache_ctx* c,
                    void** vtcm_base, uint32_t* vtcm_size,
                    void** pin_base, uint32_t* pin_cap) {
    if (!c) return;
    if (vtcm_base) *vtcm_base = c->vtcm_base;
    if (vtcm_size) *vtcm_size = c->vtcm_size;
    if (pin_base)  *pin_base  = c->vtcm_base;
    if (pin_cap)   *pin_cap   = c->pin_cap;
}

/* ============================================================
 * DMA 原语: 一次 DDR↔VTCM 1D 搬运 (阻塞, 用于 pin)
 * dir: 0=DDR→VTCM, 1=VTCM→DDR
 * ============================================================ */
int dma_xfer_blocking(const void* ddr, void* vtcm, uint32_t bytes, int dir) {
    /* desc 存储从栈上 16B 对齐拿 (DMA_DESC_SIZE_1D=16) */
    void* desc = NULL;
    if (posix_memalign(&desc, 16, DMA_DESC_SIZE_1D) != 0) return WTC_ERR_OOM;
    memset(desc, 0, DMA_DESC_SIZE_1D);

    dma_desc_1d_params_t p;
    memset(&p, 0, sizeof(p));
    if (dir == 0) { /* DDR → VTCM */
        /* CPU 写 DDR → DMA bypass 读 → DMA 写 VTCM (一致, VTCM uncached 对 HMX)
         * 读前 FLUSH_INVALIDATE src; VTCM 是 DMA 直写不需 invalidate (但 HMX cache 要) */
        qurt_mem_cache_clean((qurt_addr_t)ddr, bytes, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
        p.src_address = (uint32_t)(uintptr_t)ddr;
        p.dst_address = (uint32_t)(uintptr_t)vtcm;
        p.src_bypass  = 1;
        p.dst_bypass  = 0;
    } else {        /* VTCM → DDR */
        /* CPU 写 VTCM 是 write-back — dirty 行可能未落 VTCM 内存. FLUSH 写回后
         * DMA src_bypass=1 直读内存, 不走 cache. (B 调研 + 文献§2.9 一致组合;
         * src_bypass=0 过 cache 在 src 地址被 CPU 二次写后偶发读到零, T4c i=3). */
        qurt_mem_cache_clean((qurt_addr_t)vtcm, bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        /* submit 前清 dst 的 stale dirty 行 (writeback+丢弃): 防止 DMA bypass 写
         * DDR 物理页后, CPU dcache 里该地址的旧脏行被自动 evict 覆盖 DMA 写
         * (T4c 尾部128偶发错根因). dst_bypass=1 DMA 不经 cache, 必须手动清. */
        qurt_mem_cache_clean((qurt_addr_t)ddr, bytes, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
        p.src_address = (uint32_t)(uintptr_t)vtcm;
        p.dst_address = (uint32_t)(uintptr_t)ddr;
        p.src_bypass  = 1;
        p.dst_bypass  = 1;
    }
    p.length = bytes;
    p.order  = 1;

    if (dma_desc_init(desc, &p, DMA_DESC_TYPE_1D) != DMA_SUCCESS) { free(desc); return WTC_ERR_DMA; }
    /* submit 前 wait → dmpoll 必 IDLE → dmstart, 绕开 RUN/dmlink (同 submit_one). */
    if (dma_wait_for_idle() != DMA_SUCCESS) { free(desc); return WTC_ERR_DMA; }
    void* batch = desc;
    if (dma_desc_submit(&batch, 1) != DMA_SUCCESS) { free(desc); return WTC_ERR_DMA; }
    /* dmwait (wait_for_idle) 在引擎忙时阻塞直到 idle, 排空写 pipeline (dst_bypass=1
     * 写进 DDR controller buffer). 必须在 desc_is_done 之前调 — 否则 dstate=COMPLETE
     * 先返回 (引擎接受完 desc 但写未落 RAM), dmwait 见已 idle 立即返回不排空.
     * (对齐 vtcm_dma_bench way2: 仅 wait_for_idle 即 PASS). */
    if (dma_wait_for_idle() != DMA_SUCCESS) { free(desc); return WTC_ERR_DMA; }
    while (dma_desc_is_done(desc) == DMA_INCOMPLETE) { }

    if (dir == 1) {
        qurt_mem_cache_clean((qurt_addr_t)ddr, bytes, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    wtc_unlink_last(desc);
    free(desc);
    return WTC_OK;
}

/* 公开: 排空引擎 + 清 g_last_desc, 供 bench 在每个 test 开头隔离 DMA 会话状态. */
void wtcache_dma_fence(void) {
    dma_wait_for_idle();
    g_last_desc = NULL;
}

/* ============================================================
 * 1-C-α: pin
 * ============================================================ */
int wtcache_pin_weight(struct wtcache_ctx* c,
                       const void* ddr_src, size_t bytes, int align,
                       void** vtcm_out) {
    if (!c || !ddr_src || !vtcm_out || bytes == 0) return WTC_ERR_RANGE;
    uint32_t a = (align > 0 && align <= (int)WTC_VTCM_ALIGN && is_pow2((uint32_t)align))
                 ? (uint32_t)align : WTC_VTCM_ALIGN;
    /* bump: 从 pin_off 起, 先对齐到 a */
    uint32_t start = round_up(c->pin_off, a);
    if ((uint64_t)start + bytes > c->pin_cap) { FARF(ALWAYS,"pin OOM %u+%zu > %u", start, bytes, c->pin_cap); return WTC_ERR_OOM; }
    uint8_t* dst = c->vtcm_base + start;
    int rc = dma_xfer_blocking(ddr_src, dst, (uint32_t)bytes, 0);
    if (rc != WTC_OK) return rc;
    c->pin_off = start + (uint32_t)bytes;
    *vtcm_out = dst;
    FARF(ALWAYS, "pin_weight: %zuB at vtcm+%u → %p (pin_off now %u)",
         bytes, start, dst, c->pin_off);
    return WTC_OK;
}

int wtcache_pin_verify(struct wtcache_ctx* c, const void* ddr_src,
                       const void* vtcm, size_t bytes, uint32_t* first_bad_off) {
    if (!c || !ddr_src || !vtcm) return WTC_ERR_RANGE;
    /* vtcm 是 VTCM 地址, DMA 已 INVALIDATE 过; 但 CPU 读前再 INVALIDATE 保险 */
    qurt_mem_cache_clean((qurt_addr_t)vtcm, bytes, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    const uint8_t* s = (const uint8_t*)ddr_src;
    const uint8_t* d = (const uint8_t*)vtcm;
    /* ddr_src 可能 cache 里是旧值 — FLUSH_INVALIDATE 确保 CPU 读到 DMA 前写的真值.
     * 但 ddr_src 在 pin 时已 FLUSH_INVALIDATE 过, 这里只读不写, 不用再 clean. */
    for (size_t i = 0; i < bytes; i++) {
        if (s[i] != d[i]) {
            if (first_bad_off) *first_bad_off = (uint32_t)i;
            FARF(ALWAYS, "pin_verify MISMATCH @%zu: ddr=%02x vtcm=%02x",
                 i, s[i], d[i]);
            return WTC_ERR_VERIFY;
        }
    }
    return WTC_OK;
}

/* ============================================================
 * 1-C-β: ring (prefetch/move_back 环)
 *
 * 槽模型: depth_prefetch 个 in 槽 + depth_moveback 个 out 槽, 各等大 tile_bytes.
 * 在途窗口: prefetch desc 菊花链, prime 提交 min(n, depth) 个, 每消费 1 个追加 1 个.
 * ring_next 不 dma_wait_for_idle — 只 dma_desc_is_done 轮询当前 in 槽.
 * ============================================================ */
struct wtcache_ring {
    struct wtcache_ctx* ctx;
    uint8_t* in_slots;    /* depth_prefetch 个, 各 tile_bytes */
    uint8_t* out_slots;   /* depth_moveback 个 */
    uint32_t tile_bytes;
    int      depth_in;
    int      depth_out;

    /* desc 池: 每槽一个 prefetch desc + 每输出槽一个 moveback desc (复用) */
    void**   in_descs;    /* [depth_in] */
    void**   out_descs;   /* [depth_out] */

    /* 0=阻塞 submit (submit_one, 每 submit 排空引擎; T1-T4 正确性模式, 可与 move_back 共存)
     * 1=非阻塞菊花链 (submit_chained, depth≥2 在途; T5 重叠模式, 禁 move_back).
     * 切换前必须 fence. overlap 模式下 in_descs 预分配且会话期不 free. */
    int      overlap_mode;

    /* 环游标 */
    int      in_submit;   /* 下个要 submit 的 ddr tile 序号 (host 供给) */
    int      in_consume;  /* 下个要消费 (is_done 查) 的 in 槽 idx */
    int      n_inflight;  /* 当前在途 prefetch 数 */

    /* ddr tile 供给队列 (host 在 prime/next 里推) */
    const void** ddr_sources;  /* 动态数组 */
    int      ddr_cap;
    int      ddr_count;   /* 已推入总数 */
    int      ddr_next;    /* 下个要 prefetch 的 idx */

    /* 待 move_back: 上一次 ring_next 返回的 out_slot 及其 DDR 目标.
     * 调用方在拿到 vout 后写入结果, 下一次 ring_next (或 drain) 时才 submit
     * move_back — 保证 HMX/CPU 写已在 VTCM 落盘后再搬. -1=无 pending. */
    int      pending_mb_slot;
    void*    pending_mb_target;

    uint32_t peak_vtcm;
};

/* 提交单个 desc. submit 前强制 wait_for_idle → dmpoll 必 IDLE → dma_desc_submit
 * 走 dmstart, 绝不走 RUN/dmlink 路径, 彻底绕开 g_last_desc 悬空风险.
 * 引擎在两批 desc 交接瞬态会短暂 RUN, 若 submit 撞此瞬态且 g_last_desc 悬空即野指针写.
 * 用于 pin / move_back / 正确性模式 ring (T1-T4). 代价: 每次 submit 排空引擎, 零重叠. */
static int submit_one(void* desc) {
    if (dma_wait_for_idle() != DMA_SUCCESS) return WTC_ERR_DMA;
    void* batch = desc;
    return (dma_desc_submit(&batch, 1) == DMA_SUCCESS) ? WTC_OK : WTC_ERR_DMA;
}

/* 非阻塞菊花链提交 (T5 重叠用). 不 wait_for_idle, 直接 dma_desc_submit:
 *   引擎 IDLE → dmstart (不用 g_last_desc, 安全);
 *   引擎 RUN → dmlink(g_last_desc, new) (追加到运行中的 desc 链尾).
 * 安全前提: g_last_desc 必须指向一个 live desc. overlap 模式保证这点 —
 * ring 预分配 depth_in 个 desc, 整个会话期不 free; 会话开头 fence (引擎 IDLE +
 * g_last_desc=NULL) 保证首个 submit 走 dmstart 设 g_last_desc=live; 之后绝不置
 * NULL. 会话内禁止 move_back (move_back 的 dma_xfer_blocking 会 free desc +
 * 可能清 g_last_desc). 这就是 T5 重叠的物理基础 — HMX compute 在 DSP core, DMA
 * 在 DMA 引擎, 独立硬件单元真并行. */
static int submit_chained(void* desc) {
    void* batch = desc;
    return (dma_desc_submit(&batch, 1) == DMA_SUCCESS) ? WTC_OK : WTC_ERR_DMA;
}

int wtcache_ring_init(struct wtcache_ctx* c, struct wtcache_ring** out_r,
                      size_t tile_bytes, int depth_prefetch, int depth_moveback) {
    if (!c || !out_r || tile_bytes == 0) return WTC_ERR_RANGE;
    if (tile_bytes % WTC_VTCM_ALIGN != 0) return WTC_ERR_RANGE;
    if (depth_prefetch < 1) depth_prefetch = 4;
    if (depth_moveback < 1) depth_moveback = 4;

    /* 切 VTCM: ring_in → ring_out → temp 调整 */
    uint32_t in_bytes  = depth_prefetch  * (uint32_t)tile_bytes;
    uint32_t out_bytes = depth_moveback * (uint32_t)tile_bytes;
    in_bytes  = round_up(in_bytes,  WTC_VTCM_ALIGN);
    out_bytes = round_up(out_bytes, WTC_VTCM_ALIGN);
    uint32_t in_start  = round_up(c->ring_in_off, WTC_VTCM_ALIGN);
    uint32_t out_start = round_up(in_start + in_bytes, WTC_VTCM_ALIGN);
    uint32_t temp_start = round_up(out_start + out_bytes, WTC_VTCM_ALIGN);
    if (temp_start > c->vtcm_size) { FARF(ALWAYS,"ring OOM %u > %u", temp_start, c->vtcm_size); return WTC_ERR_OOM; }

    struct wtcache_ring* r = (struct wtcache_ring*)calloc(1, sizeof(*r));
    if (!r) return WTC_ERR_OOM;
    r->ctx = c;
    r->in_slots  = c->vtcm_base + in_start;
    r->out_slots = c->vtcm_base + out_start;
    r->tile_bytes = (uint32_t)tile_bytes;
    r->depth_in = depth_prefetch;
    r->depth_out = depth_moveback;
    r->overlap_mode = 0;  /* 默认阻塞模式 (T1-T4 正确性) */
    r->in_descs  = (void**)calloc(depth_prefetch, sizeof(void*));
    r->out_descs = (void**)calloc(depth_moveback, sizeof(void*));
    r->ddr_cap = depth_prefetch + 16;
    r->ddr_sources = (const void**)calloc(r->ddr_cap, sizeof(void*));
    r->pending_mb_slot   = -1;
    r->pending_mb_target = NULL;
    if (!r->in_descs || !r->out_descs || !r->ddr_sources) {
        free(r->in_descs); free(r->out_descs); free(r->ddr_sources);
        free(r);
        return WTC_ERR_OOM;
    }
    /* 预分配 prefetch desc (overlap 模式需整会话持有; 阻塞模式复用同一池).
     * 首次 submit 前 dma_desc_init 在 ring_try_submit_prefetch 里做 (需 src/dst). */
    for (int i = 0; i < depth_prefetch; i++) {
        if (posix_memalign(&r->in_descs[i], 16, DMA_DESC_SIZE_1D) != 0) {
            for (int j = 0; j < i; j++) free(r->in_descs[j]);
            free(r->in_descs); free(r->out_descs); free(r->ddr_sources);
            free(r);
            return WTC_ERR_OOM;
        }
        memset(r->in_descs[i], 0, DMA_DESC_SIZE_1D);
    }

    c->ring_in_off  = in_start;
    c->ring_out_off = out_start;
    c->temp_off     = temp_start;
    c->temp_used    = 0;
    r->peak_vtcm    = temp_start;

    FARF(ALWAYS, "ring_init: tile=%u in[%d]@+%u out[%d]@+%u temp@+%u",
         r->tile_bytes, depth_prefetch, in_start, depth_moveback, out_start, temp_start);
    *out_r = r;
    return WTC_OK;
}

/* 内部: 推一个 ddr tile 进 prefetch 队列 */
static int ring_push_ddr(struct wtcache_ring* r, const void* ddr) {
    if (r->ddr_count >= r->ddr_cap) {
        int nc = r->ddr_cap * 2;
        const void** na = (const void**)realloc(r->ddr_sources, nc * sizeof(void*));
        if (!na) return WTC_ERR_OOM;
        r->ddr_sources = na; r->ddr_cap = nc;
    }
    r->ddr_sources[r->ddr_count++] = ddr;
    return WTC_OK;
}

/* 内部: 若有未 submit 的 ddr 且在途 < depth, 提交一个 prefetch.
 * desc 复用 in_descs[slot] (预分配, re-init 不 free):
 *   - 阻塞模式: submit_one (每 submit 排空引擎 → dmstart; 与 move_back 安全共存)
 *   - overlap 模式: submit_chained (非阻塞菊花链 → dmstart/dmlink; 禁 move_back) */
static int ring_try_submit_prefetch(struct wtcache_ring* r) {
    while (r->n_inflight < r->depth_in && r->ddr_next < r->ddr_count) {
        int slot = (r->in_consume + r->n_inflight) % r->depth_in;
        uint8_t* dst = r->in_slots + (uint32_t)slot * r->tile_bytes;
        const void* src = r->ddr_sources[r->ddr_next++];
        /* DDR src FLUSH_INVALIDATE (DMA src_bypass=1 直读内存) */
        qurt_mem_cache_clean((qurt_addr_t)src, r->tile_bytes,
                             QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
        /* re-init 预分配 desc (memset 0 清旧 next 指针, 防 dmlink 续旧链) */
        void* desc = r->in_descs[slot];
        memset(desc, 0, DMA_DESC_SIZE_1D);
        dma_desc_1d_params_t p;
        memset(&p, 0, sizeof(p));
        p.src_address = (uint32_t)(uintptr_t)src;
        p.dst_address = (uint32_t)(uintptr_t)dst;
        p.length = r->tile_bytes;
        p.order = 1;
        p.src_bypass = 1;
        p.dst_bypass = 0;
        if (dma_desc_init(desc, &p, DMA_DESC_TYPE_1D) != DMA_SUCCESS) return WTC_ERR_DMA;

        int rc = r->overlap_mode ? submit_chained(desc) : submit_one(desc);
        if (rc != WTC_OK) return rc;
        r->n_inflight++;
        r->peak_vtcm = (r->peak_vtcm > (uint32_t)(dst - r->ctx->vtcm_base + r->tile_bytes))
                       ? r->peak_vtcm : (uint32_t)(dst - r->ctx->vtcm_base + r->tile_bytes);
    }
    return WTC_OK;
}

int wtcache_ring_prime(struct wtcache_ring* r, const void* ddr_tiles[], int n) {
    if (!r || n < 0) return WTC_ERR_RANGE;
    for (int i = 0; i < n; i++) {
        int rc = ring_push_ddr(r, ddr_tiles[i]);
        if (rc != WTC_OK) return rc;
    }
    int rc = ring_try_submit_prefetch(r);
    if (rc != WTC_OK) return rc;
    /* prime 等 min(n, depth) 个就绪 (不追求重叠) */
    return WTC_OK;
}

/* 切换 ring 的 prefetch 提交模式.
 * enable=1 (overlap): 非阻塞菊花链 submit, depth≥2 在途, HMX compute 可藏在 DMA 后.
 *   切入前强制 fence (清 g_last_desc + 排空引擎), 保证首个 submit 走 dmstart.
 *   会话内禁止 move_back (overlap 模式不调用 ring_submit_moveback).
 * enable=0 (blocking): 每 submit 排空引擎, 与 move_back 安全共存 (T1-T4 默认). */
void wtcache_ring_set_overlap(struct wtcache_ring* r, int enable) {
    if (!r) return;
    wtcache_dma_fence();
    r->overlap_mode = enable ? 1 : 0;
}

/* 内部: 提交一个 move_back (VTCM out 槽 → DDR target).
 * 复用 dma_xfer_blocking (dir=1): 它已含 submit 前 wait (绕开 g_last_desc/dmlink)、
 * FLUSH(VTCM src)+src_bypass=1、submit 前 FLUSH_INVALIDATE(DDR dst) 防 evict 覆盖、
 * submit 后 wait + desc_is_done + dst INVALIDATE. 全阻塞 — move_back 串行化不影响
 * T5 重叠 (重叠来自 prefetch 藏在 HMX compute 后). */
static int ring_submit_moveback(struct wtcache_ring* r, int out_slot_idx, void* ddr_target) {
    uint8_t* src = r->out_slots + (uint32_t)out_slot_idx * r->tile_bytes;
    if (r->out_descs[out_slot_idx]) { wtc_unlink_last(r->out_descs[out_slot_idx]); free(r->out_descs[out_slot_idx]); r->out_descs[out_slot_idx] = NULL; }
    g_wtc_mb_count++;
    g_wtc_mb_last_dst = ddr_target;
    g_wtc_mb_last_slot = out_slot_idx;
    return dma_xfer_blocking(ddr_target, src, r->tile_bytes, 1);
}

int wtcache_ring_next(struct wtcache_ring* r,
                      const void* ddr_next, void* ddr_out_target,
                      int invalidate,
                      void** cur_vtcm_in, void** cur_vtcm_out) {
    if (!r) return WTC_ERR_RANGE;
    /* 1. 推入新 ddr (若有) */
    if (ddr_next) {
        int rc = ring_push_ddr(r, ddr_next);
        if (rc != WTC_OK) return rc;
    }
    /* 2. 尽量填满在途窗口 */
    int rc = ring_try_submit_prefetch(r);
    if (rc != WTC_OK) return rc;

    /* 3. 等当前消费槽就绪 (轮询, 不全局 wait) */
    if (r->n_inflight == 0) return WTC_ERR_NOT_READY;
    int slot = r->in_consume;
    void* desc = r->in_descs[slot];
    if (!desc) return WTC_ERR_NOT_READY;
    /* 自旋等这一个 desc 完成. 期间 CPU 空转 — HMX compute 在调用方做, 这里只保障数据就绪.
     * 真重叠靠调用方在拿到 cur_vtcm_in 后立刻在 *上一轮* 数据上算 (见 T5 harness). */
    while (dma_desc_is_done(desc) == DMA_INCOMPLETE) { /* spin */ }

    /* 4. 输入就绪: INVALIDATE (T3 一致性开关) */
    uint8_t* in_data = r->in_slots + (uint32_t)slot * r->tile_bytes;
    if (invalidate) {
        qurt_mem_cache_clean((qurt_addr_t)in_data, r->tile_bytes,
                             QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    /* 5. 配对输出槽 (round-robin) */
    int out_slot = r->in_consume % r->depth_out;
    uint8_t* out_data = r->out_slots + (uint32_t)out_slot * r->tile_bytes;

    /* 5b. 该 out_slot 上一轮的 move_back 可能还在读它 (depth_out 小时 round-robin
     * 回来得快). 等它 done 再让调用方写, 避免搬走半新半旧的数据. */
    if (r->out_descs[out_slot]) {
        while (dma_desc_is_done(r->out_descs[out_slot]) == DMA_INCOMPLETE) { }
    }

    /* 6. 先 submit 上轮 pending move_back (调用方已写完上轮 out_slot, 数据在 VTCM).
     * 本轮的 (out_slot, ddr_out_target) 存为 pending, 下轮 (或 drain) 再 submit —
     * 这样 move_back 的数据一定是调用方上一轮写入的结果, 不是空槽. */
    if (r->pending_mb_slot >= 0 && r->pending_mb_target) {
        rc = ring_submit_moveback(r, r->pending_mb_slot, r->pending_mb_target);
        if (rc != WTC_OK) return rc;
    }
    r->pending_mb_slot   = ddr_out_target ? out_slot : -1;
    r->pending_mb_target = ddr_out_target;

    /* 7. 游标推进 */
    r->in_consume = (r->in_consume + 1) % r->depth_in;
    r->n_inflight--;

    if (cur_vtcm_in)  *cur_vtcm_in  = in_data;
    if (cur_vtcm_out) *cur_vtcm_out = out_data;
    return WTC_OK;
}

int wtcache_ring_drain(struct wtcache_ring* r) {
    if (!r) return WTC_ERR_RANGE;
    /* 排空所有在途 prefetch (不再 submit 新的) */
    while (r->n_inflight > 0) {
        int slot = r->in_consume;
        void* desc = r->in_descs[slot];
        if (desc) {
            while (dma_desc_is_done(desc) == DMA_INCOMPLETE) { }
        }
        r->in_consume = (r->in_consume + 1) % r->depth_in;
        r->n_inflight--;
    }
    /* submit 最后一个 pending move_back (上轮 out_slot, 调用方已写完) */
    if (r->pending_mb_slot >= 0 && r->pending_mb_target) {
        int rc = ring_submit_moveback(r, r->pending_mb_slot, r->pending_mb_target);
        if (rc != WTC_OK) return rc;
        r->pending_mb_slot   = -1;
        r->pending_mb_target = NULL;
    }
    /* dmwait (wait_for_idle) 在引擎仍处理最后几个 move_back 时阻塞, 排空写 pipeline.
     * 不先轮询 desc_is_done — 否则抢先返回 (dstate=COMPLETE 时写未落 RAM), dmwait
     * 见 idle 立即返回不排空 (T4b tail-race 根因). */
    dma_wait_for_idle();
    return WTC_OK;
}

void wtcache_ring_stats(const struct wtcache_ring* r,
                        int* inflight_prefetch, int* inflight_moveback,
                        uint32_t* peak_vtcm_bytes) {
    if (!r) return;
    if (inflight_prefetch) *inflight_prefetch = r->n_inflight;
    if (inflight_moveback) *inflight_moveback = 0;  /* move_back 不单独追踪在途 (共享引擎) */
    if (peak_vtcm_bytes)   *peak_vtcm_bytes = r->peak_vtcm;
}

/* ring 释放 (在 wtcache_close 前可显式调, 省略则 ctx free 时泄漏 desc — bench 单次跑无妨) */
void wtcache_ring_destroy(struct wtcache_ring* r) {
    if (!r) return;
    for (int i = 0; i < r->depth_in; i++)  if (r->in_descs[i])  free(r->in_descs[i]);
    for (int i = 0; i < r->depth_out; i++) if (r->out_descs[i]) free(r->out_descs[i]);
    free(r->in_descs); free(r->out_descs);
    free(r->ddr_sources);
    free(r);
}
