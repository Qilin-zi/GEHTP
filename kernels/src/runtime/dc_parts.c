/* dc_parts.c — 部件层实现 */
#include "dc_parts.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <HAP_farf.h>
#include <HAP_power.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <hexagon_types.h>

#include "dma_utils.h"
#include "wtcache.h"
#include "w4a16_quant.h"

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
    d->src_bypass = 1; d->dst_bypass = 0;  /* 默认同 1-C 硬编码契约 */
    return 0;
}

void dc_dma_destroy(struct dc_dma* d) {
    if (d->desc) { free(d->desc); d->desc = NULL; }
}

void dc_dma_fence(void) {
    dma_wait_for_idle();
    g_last_desc = NULL;
}

/* PROF W-P3: UserDMA 累加器 (dc_dma_once 内累计, 每 op reset/get) */
static int64_t g_dma_us = 0;
static int64_t g_dma_bytes = 0;
void dc_dma_reset(void) { g_dma_us = 0; g_dma_bytes = 0; }
void dc_dma_get(int64_t* us, int64_t* bytes) {
    if (us) *us = g_dma_us;
    if (bytes) *bytes = g_dma_bytes;
}

void dc_clean_ddr(const void* p, uint32_t bytes) {
    qurt_mem_cache_clean((qurt_addr_t)p, bytes,
                         QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);
}

void dc_dma_clean_src(struct dc_dma* d) { dc_clean_ddr(d->src, d->bytes); }

int dc_dma_once(struct dc_dma* d) {
    const int64_t t_dma0 = HAP_perf_get_time_us();
    dma_desc_1d_params_t p;
    memset(&p, 0, sizeof(p));
    /* src 已由 dc_dma_clean_src 一次性清过; bypass 来自 d->src_bypass/dst_bypass
     * (dc_dma_init 默认 1/0 同 1-C 契约; OP_DMA 透传覆盖) */
    p.src_address = (uint32_t)(uintptr_t)d->src;
    p.dst_address = (uint32_t)(uintptr_t)d->dst;
    p.src_bypass = d->src_bypass;
    p.dst_bypass = d->dst_bypass;
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
    g_dma_us += HAP_perf_get_time_us() - t_dma0;
    g_dma_bytes += d->bytes;
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
static int act_is_bound(const struct dc_w4* e, const uint8_t* act_src) {
    return e->act_valid && e->act_ddr == act_src;
}

int dc_w4_run_prep(struct dc_w4* e, const uint8_t* act_src, uint32_t m,
                      uint32_t k, float wq_rms, float f_fixed) {
    if (!e || !act_src) return 0xD400;
    if (m % 32 || k % 32) return 0xD401;
    uint32_t m_pad = (m + 255u) & ~255u;
    if (m_pad != e->m) return 0xD402;  /* carve 必须按 pad 后 M */
    float as = w4a16_act_scale((const uint16_t*)act_src, m * k);
    float f = f_fixed > 0.0f ? f_fixed
              : w4a16_pow2ceil(4.0f * sqrtf((float)k) *
                               w4a16_act_rms_norm((const uint16_t*)act_src, m * k, as) *
                               wq_rms / 7.0f);
    e->act_scale = as * f;  /* dequant 用 a_scale, f 精确抵消 */
    /* 量化+crouton 融合 (零行 pad = 32768) */
    uint16_t* surf = (uint16_t*)e->act;
    uint32_t n_kt = k / 32, n_m32 = m_pad / 32;
    uint32_t out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++) {
            uint32_t k_base = kt * 32;
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    uint32_t row1 = row0 + 1;
                    for (uint32_t c = 0; c < 32; c++) {
                        surf[out++] = row0 < m
                            ? w4a16_quant_f16(((const uint16_t*)act_src)[(size_t)row0 * k + k_base + c], e->act_scale)
                            : 32768u;
                        surf[out++] = row1 < m
                            ? w4a16_quant_f16(((const uint16_t*)act_src)[(size_t)row1 * k + k_base + c], e->act_scale)
                            : 32768u;
                    }
                }
        }
    qurt_mem_cache_clean((qurt_addr_t)e->act, m_pad * k * 2,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    e->act_ddr = act_src;
    e->act_valid = 1;
    return 0;
}

int dc_w4_run_invoke(struct dc_w4* e, const uint8_t* scale, uint32_t n,
                        uint8_t* out_ddr, uint32_t out_row_bytes,
                        uint32_t m_out, float wq_rms, float f_fixed) {
    if (!e || !scale || !out_ddr) return 0xD400;
    if (!e->act_valid) return 0xD405;  /* 须先 prep */
    if (m_out % 32 || n % 32) return 0xD401;
    /* refill 尺寸按本次 invoke 的 n (dc_w4_invoke 用 e->n; carve 按分块宽
     * n_eff 时 e->n > nc — refill 4KB 越界读 2KB 表, 板挂/数据错实锤) */
    {
        uint32_t en = e->n;
        e->n = n;
        int irc = dc_w4_invoke(e);
        e->n = en;
        if (irc) return 0xD403;
    }
    /* HMX 直写出面, INVALIDATE 后 CPU 读; crouton 序直读反量化 (行≥m_out 丢弃) */
    qurt_mem_cache_clean((qurt_addr_t)e->out, e->m * n * 2,
                         QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    w4a16_dequant_crouton((const uint16_t*)e->out, e->m, n, m_out, e->act_scale,
                             (const uint16_t*)scale, (uint16_t*)out_ddr,
                             out_row_bytes);
    return 0;
}

void dc_w4_run_fini(struct dc_w4* e) {
    if (e) e->act_valid = 0;
}

/* dc_w4_run — f16 DDR act → (量化 a16 域 + M→256 零行 pad + crouton 面) →
 * kernel → 出面反量化 f16 DDR。exec_matmul 走本入口; 例程/dd_worker 的
 * dc_w4_invoke 旧契约 (act 预置 u16 面) 不变。
 * 铁律: a16 域 q=0 是 real=-1.0 — pad 零行填 32768 (零值点)。
 * 出面 = A_s·S[n]·(q-32768)/32767 (S=权重列 scale 槽, 已含 /7)。
 * lm_head 多块复用: act 源相同 → prep 一次, 各块 invoke 复用。 */
int dc_w4_run(struct dc_w4* e, const uint8_t* act_ddr, uint8_t* out_ddr,
              uint32_t m, uint32_t k, uint32_t n, const uint8_t* scale_ddr,
              uint32_t out_row_bytes, float wq_rms, float f_fixed) {
    if (!e || !act_ddr || !out_ddr || !scale_ddr) return 0xD400;
    if (m % 32 || k % 32 || n % 32) return 0xD401;
    uint32_t m_pad = (m + 255u) & ~255u;
    if (m_pad != e->m) return 0xD402;  /* carve 必须按 pad 后 M */
    if (act_is_bound(e, act_ddr)) {
        /* 复用已 prep 的 act 面 + a_scale */
        return dc_w4_run_invoke(e, scale_ddr, n, out_ddr,
                                 out_row_bytes, m, wq_rms, f_fixed);
    }
    /* 首次: 完整 prep */
    int rc = dc_w4_run_prep(e, act_ddr, m, k, wq_rms, f_fixed);
    if (rc) return rc;
    return dc_w4_run_invoke(e, scale_ddr, n, out_ddr,
                             out_row_bytes, m, wq_rms, f_fixed);
}
