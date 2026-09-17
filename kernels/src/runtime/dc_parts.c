/* dc_parts.c — 部件层实现 */
#include "dc_parts.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <HAP_farf.h>
#include <HAP_power.h>
#include <qurt.h>
#include <hexagon_types.h>

#include "dma_utils.h"
#include "wtcache.h"

/* w4a16_driver_dc.c */
int w4a16_invoke(const uint8_t* vtcm_act, const uint8_t* vtcm_weight,
                 const uint8_t* vtcm_bias, uint8_t* vtcm_out,
                 uint8_t* act_table_rw, uint8_t* out_table_rw,
                 uint8_t* mask_rw, uint8_t* extra_rw,
                 uint32_t m, uint32_t k, uint32_t n);

/* dma_utils.c 文件级全局 (非 static) */
extern void* g_last_desc;

/* ================= VTCM arena ================= */
void dc_arena_init(struct dc_arena* a, uint8_t* base, uint32_t size) {
    a->base = base; a->size = size; a->off = 0;
}
static uint32_t dc_round_up(uint32_t v, uint32_t al) {
    return (v + al - 1u) & ~(al - 1u);
}
uint8_t* dc_arena_alloc(struct dc_arena* a, uint32_t bytes, uint32_t align) {
    uint32_t start = dc_round_up(a->off, align);
    if ((uint64_t)start + bytes > a->size) {
        FARF(ALWAYS, "dc_arena OOM: need %u@%u have %u", bytes, start, a->size);
        return NULL;
    }
    a->off = start + bytes;
    return a->base + start;
}

uint8_t* dc_read_file(const char* path, uint32_t* bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) { FARF(ALWAYS, "dc: open FAIL %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void* p = NULL;
    if (posix_memalign(&p, 128, (size_t)sz) != 0) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return NULL; }
    fclose(f);
    *bytes = (uint32_t)sz;
    return (uint8_t*)p;
}

/* ================= HVX 负载 =================
 * 4 条独立 vadd 依赖链 × 32 向量 (4KB VTCM) — HVX 加法 pipe bound,
 * 输入驻 VTCM 小区, 无 DDR 流量 → 测的是算力不是带宽 (plan P1 注意项)。
 */
uint32_t dc_hvx_load(uint8_t* scratch4k, uint32_t iters) {
    const HVX_Vector* v = (const HVX_Vector*)scratch4k;   /* 32 向量 */
    HVX_Vector a0 = v[0], a1 = v[8], a2 = v[16], a3 = v[24];
    for (uint32_t it = 0; it < iters; ++it) {
        for (int i = 0; i < 32; i += 4) {
            a0 = Q6_Vw_vadd_VwVw(a0, v[i]);
            a1 = Q6_Vw_vadd_VwVw(a1, v[i + 1]);
            a2 = Q6_Vw_vadd_VwVw(a2, v[i + 2]);
            a3 = Q6_Vw_vadd_VwVw(a3, v[i + 3]);
        }
    }
    HVX_Vector s01 = Q6_Vw_vadd_VwVw(a0, a1);
    HVX_Vector s23 = Q6_Vw_vadd_VwVw(a2, a3);
    HVX_Vector s = Q6_Vw_vadd_VwVw(s01, s23);
    uint32_t fold[32] __attribute__((aligned(128)));
    *(HVX_Vector*)fold = s;
    uint32_t acc = 0;
    for (int i = 0; i < 32; i++) acc ^= fold[i];
    return acc;
}

/* ================= 确定性整数运算 ================= */
static void dc_norm_i16_scalar(const int16_t* restrict x, int16_t* restrict y,
                               uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        int32_t t = ((int32_t)x[i] >> 2) + 64;
        t = t > 32767 ? 32767 : t;
        t = t < -32768 ? -32768 : t;
        y[i] = (int16_t)t;
    }
}
void dc_norm_i16(const int16_t* restrict x, int16_t* restrict y, uint32_t n) {
    /* (x>>2)+64 ∈ [-8128,8255] 恒不越 int16 界 → vadd.h 无需饱和即 bit-exact;
     * 标量版分支/三目均无法触发自动向量化 (实测 ~1GB/s), 显式 HVX。 */
    if (((uintptr_t)x | (uintptr_t)y | (n << 1)) & 127u) {
        dc_norm_i16_scalar(x, y, n);
        return;
    }
    HVX_Vector c64 = Q6_Vh_vsplat_R(64);
    const HVX_Vector* xv = (const HVX_Vector*)x;
    HVX_Vector* yv = (HVX_Vector*)y;
    for (uint32_t i = 0; i < (n >> 6); ++i)
        yv[i] = Q6_Vh_vadd_VhVh(Q6_Vh_vasr_VhR(xv[i], 2), c64);
}

uint64_t dc_dot_u64(const int16_t* restrict a, const int16_t* restrict b,
                    uint32_t n) {
    uint64_t acc = 0;
    for (uint32_t i = 0; i < n; ++i)
        acc += (int64_t)a[i] * (int64_t)b[i];
    return acc;
}

/* ================= DMA 流 ================= */
int dc_dma_init(struct dc_dma* d, uint8_t* src, uint8_t* dst, uint32_t bytes,
                dc_mutex_t* mu) {
    memset(d, 0, sizeof(*d));
    if (posix_memalign(&d->desc, 16, DMA_DESC_SIZE_1D) != 0) return 0xD200;
    memset(d->desc, 0, DMA_DESC_SIZE_1D);
    d->src = src; d->dst = dst; d->bytes = bytes; d->mu = mu;
    return 0;
}

void dc_dma_destroy(struct dc_dma* d) {
    if (d->desc) { free(d->desc); d->desc = NULL; }
}

void dc_dma_fence(void) {
    dma_wait_for_idle();
    g_last_desc = NULL;
}

void dc_clean_ddr(const void* p, uint32_t bytes) {
    qurt_mem_cache_clean((qurt_addr_t)p, bytes,
                         QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
}

void dc_dma_clean_src(struct dc_dma* d) { dc_clean_ddr(d->src, d->bytes); }

int dc_dma_once(struct dc_dma* d) {
    dma_desc_1d_params_t p;
    memset(&p, 0, sizeof(p));
    /* src 已由 dc_dma_clean_src 一次性清过; bypass 1/0 契约同 1-C */
    p.src_address = (uint32_t)(uintptr_t)d->src;
    p.dst_address = (uint32_t)(uintptr_t)d->dst;
    p.src_bypass = 1;
    p.dst_bypass = 0;
    p.length = d->bytes;
    p.order = 1;
    if (dma_desc_init(d->desc, &p, DMA_DESC_TYPE_1D) != DMA_SUCCESS) return 0xD201;

    /* R-D1: mutex 内 poll-IDLE → submit 只走 dmstart 分支 (不 dmlink) */
    dc_mutex_lock(d->mu);
    while (dma_wait_for_idle() != DMA_SUCCESS) { }
    int rc = dma_desc_submit(&d->desc, 1);
    dc_mutex_unlock(d->mu);
    if (rc != DMA_SUCCESS) return 0xD202;

    /* 完成: 先 dmwait 排空写 pipeline, 再 poll 自己 desc dstate (1-C 顺序教训) */
    while (dma_wait_for_idle() != DMA_SUCCESS) { }
    while (dma_desc_is_done(d->desc) == DMA_INCOMPLETE) { }
    return 0;
}

uint64_t dc_dma_checksum(const struct dc_dma* d) {
    /* dst_bypass=0 → DMA 写经 dcache, CPU 直读一致; INVALIDATE 防自身 stale */
    qurt_mem_cache_clean((qurt_addr_t)d->dst, d->bytes,
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    const uint64_t* p = (const uint64_t*)d->dst;
    uint64_t s = 0;
    for (uint32_t i = 0; i < d->bytes / 8; ++i) s += p[i];
    return s;
}

/* ================= W4A16 引擎 ================= */
int dc_w4_carve(struct dc_w4* e, struct dc_arena* a, uint32_t m, uint32_t k,
                uint32_t n, const uint8_t* atbl_ddr, const uint8_t* otbl_ddr) {
    memset(e, 0, sizeof(*e));
    e->m = m; e->k = k; e->n = n;
    e->act  = dc_arena_alloc(a, m * k * 2, 2048);
    e->out  = dc_arena_alloc(a, m * n * 2, 2048);
    e->wt   = dc_arena_alloc(a, k * n / 2, 2048);
    e->bias = dc_arena_alloc(a, (n / 32) * 512, 2048);
    e->atbl = dc_arena_alloc(a, 8 * (k / 32) * 4, 128);
    e->otbl = dc_arena_alloc(a, 8 * (n / 32) * 4, 128);
    e->mask = dc_arena_alloc(a, 32, 128);
    e->extra = dc_arena_alloc(a, 16, 128);
    e->atbl_ddr = atbl_ddr;
    e->otbl_ddr = otbl_ddr;
    if (!e->act || !e->out || !e->wt || !e->bias || !e->atbl || !e->otbl ||
        !e->mask || !e->extra) return 0xD300;
    return 0;
}

int dc_w4_invoke(struct dc_w4* e) {
    /* 表每次回填 host 原始 offset (w4a16_invoke 会重写成绝对指针, 破坏性) */
    uint32_t ab = 8 * (e->k / 32) * 4, ob = 8 * (e->n / 32) * 4;
    memcpy(e->atbl, e->atbl_ddr, ab);
    memcpy(e->otbl, e->otbl_ddr, ob);
    return w4a16_invoke(e->act, e->wt, e->bias, e->out,
                        e->atbl, e->otbl, e->mask, e->extra,
                        e->m, e->k, e->n);
}

void dc_w4_read_out(const struct dc_w4* e, void* recv) {
    qurt_mem_cache_clean((qurt_addr_t)e->out, e->m * e->n * 2,
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    memcpy(recv, e->out, e->m * e->n * 2);
}
