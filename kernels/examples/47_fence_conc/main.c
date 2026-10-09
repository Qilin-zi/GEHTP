/*
 * 47_fence_conc — 008 T011: HMX ∥ dc_dma 并发 fence 设备验证 (US2 硬前置闸)
 * =====================================================================
 * 23_fence 验证的是串行边界 (一写一读先后交接); 本例验证 US2 双缓冲
 * 流水的真并发面: 单线程发起, HMX matmul (W4A16 引擎) 执行期间 DMA
 * desc 链 (DDR→VTCM 预取 + VTCM→DDR move_back) 同时在飞, 硬件引擎并发。
 *
 * 每轮 r (ROUNDS=1000, 图案 p=r&7, 双缓冲槽 s=r&1):
 *   1. 提交 6-desc 链: act(p+1) DDR→staging[s^1] (128KB) + 权重流
 *      wgt_src→wgt_stage (512KB=4×128KB, 纯负载) + moveback outb[s^1]→DDR
 *      sink[p-1] (128KB) — 链与 HMX invoke 同窗执行
 *   2. fence DMA→HMX @VTCM = INVALIDATE (staging[s] 供给面)
 *   3. dc_w4_invoke (HMX busy)
 *   4. fence HMX→DMA @VTCM = FLUSH (outb[s] moveback src 契约)
 *   5. poll 链完成 (2s 看护) → fence DMA→CPU @DDR = INVALIDATE →
 *      sink[p-1] 与串行参考逐字节比对
 *
 * 门 (23_fence 同款 ex_* 范式):
 *   G1 本例走的三条 fence 路径与决策表全等
 *   G2 串行参考: 8 图案各 1 轮 (阻塞 DMA in→INVALIDATE→invoke→FLUSH→
 *      DMA out→INVALIDATE→逐字节自洽), 图案敏感性必异
 *   G3 并发 1000 轮全逐字节 == 串行参考; 含 epilogue 末轮 + 权重流校验
 *   G4 看护: 任一 poll 超 2s → FAIL (invoke 本身是阻塞库调用不设看护,
 *      见 evidence 说明)
 *   G5 并发观测: overlap 均值 ≤ 0.9 × (hmx_only + dma_only) — max-vs-sum
 *      的严格并发证明 (阻塞路径的 submit/drain 固定开销不进分母)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "dma_utils.h"
#include "example_util.h"

#define A       "/data/local/tmp/hvxhmx23/assets/s256"
#define M       256u
#define ACT_B   (M * M * 2u)   /* 131072 */
#define OUT_B   (M * M * 2u)
#define NPAT    8u
#define ROUNDS  1000
#define WGT_N   4u             /* 权重流 = 4 desc × ACT_B (dc_dma 板上实测粒度) */
#define WGT_B   (WGT_N * ACT_B)
#define CH_N    (2 + (int)WGT_N)  /* act 预取 + wgt 流 + moveback */

#define POLL_TIMEOUT_US 2000000ll

static __attribute__((aligned(128))) uint8_t act_src[NPAT][ACT_B];
static __attribute__((aligned(128))) uint8_t sink[NPAT][OUT_B];
static __attribute__((aligned(128))) uint8_t refx[NPAT][OUT_B];
static __attribute__((aligned(128))) uint8_t wgt_src[WGT_B];

static uint32_t lcg = 20261009u;
static uint32_t lcg_n(void) { lcg = lcg * 1664525u + 1013904223u; return lcg; }

static dc_mutex_t mu;

/* ---- 异步 desc 链 (镜像 dc_parts.c R-D1 纪律: mutex 内 poll-IDLE 再
 * submit → 只走 dmstart 分支; desc 持久不 free; 完成先 dmwait 排空再
 * poll 自有 desc dstate) ---- */
static int desc_set(void* d, const void* src, void* dst, uint32_t bytes)
{
    dma_desc_1d_params_t p;
    memset(&p, 0, sizeof(p));
    p.src_address = (uint32_t)(uintptr_t)src;
    p.dst_address = (uint32_t)(uintptr_t)dst;
    p.length = bytes;
    p.order = 1;
    p.src_bypass = 1;
    p.dst_bypass = 0;
    return dma_desc_init(d, &p, DMA_DESC_TYPE_1D) == DMA_SUCCESS ? 0 : -1;
}

static int async_submit(void** batch, int n)
{
    dc_mutex_lock(&mu);
    while (dma_wait_for_idle() != DMA_SUCCESS) { }
    int rc = dma_desc_submit(batch, n);
    dc_mutex_unlock(&mu);
    return rc == DMA_SUCCESS ? 0 : -1;
}

static int async_poll(void** batch, int n, int64_t* took_us)
{
    int64_t t0 = HAP_perf_get_time_us();
    for (;;) {
        if (dma_wait_for_idle() == DMA_SUCCESS) {
            int done = 1;
            for (int i = 0; i < n; i++)
                if (dma_desc_is_done(batch[i]) == DMA_INCOMPLETE) { done = 0; break; }
            if (done) { *took_us = HAP_perf_get_time_us() - t0; return 0; }
        }
        if (HAP_perf_get_time_us() - t0 > POLL_TIMEOUT_US)
            { *took_us = HAP_perf_get_time_us() - t0; return -1; }
    }
}

static int64_t tm_us(void) { return HAP_perf_get_time_us(); }

int main(void)
{
    ex_open_result("47_fence_conc");

    /* ---- G1 本例三条 fence 路径与决策表全等 ---- */
    {
        int bad = 0;
        bad += fence_op_for(FC_DMA, FC_HMX, FM_VTCM) != FO_INVALIDATE;
        bad += fence_op_for(FC_HMX, FC_DMA, FM_VTCM) != FO_FLUSH;
        bad += fence_op_for(FC_DMA, FC_CPU, FM_DDR) != FO_INVALIDATE;
        ex_check("fence_conc_table_paths", bad, 0);
    }

    /* ---- 会话 (23_fence 同款布局) ---- */
    struct wtcache_ctx* wc = NULL;
    if (wtcache_open(&wc, 4096) != WTC_OK) {
        ex_log("!! wtcache_open"); ex_summary(); return 1;
    }
    void* vb = NULL; uint32_t vs = 0; void* pb = NULL; uint32_t pc = 0;
    wtcache_layout(wc, &vb, &vs, &pb, &pc);
    uint32_t off = (pc + 2047u) & ~2047u;
    struct dc_arena ar;
    dc_arena_init(&ar, (uint8_t*)vb + off, vs - off);
    uint8_t* staging[2] = { dc_arena_alloc(&ar, ACT_B, 2048),
                            dc_arena_alloc(&ar, ACT_B, 2048) };
    uint8_t* outb[2]    = { dc_arena_alloc(&ar, OUT_B, 2048),
                            dc_arena_alloc(&ar, OUT_B, 2048) };
    uint8_t* wgt_stage  = dc_arena_alloc(&ar, WGT_B, 2048);
    dc_mutex_init(&mu);
    if (!staging[0] || !staging[1] || !outb[0] || !outb[1] || !wgt_stage) {
        ex_log("!! arena %.1fKB", (vs - off) / 1024.0);
        ex_check("fence_conc_arena", 1, 0);
        wtcache_close(wc); return ex_summary();
    }

    /* ---- 资产 + dc_w4 (一次 carve, 表一次 FLUSH; act/out 面换槽) ---- */
    uint32_t b = 0;
    uint8_t* wt   = dc_read_file(A "/packed_weight.raw", &b);
    uint8_t* bias = dc_read_file(A "/folded_bias.raw", &b);
    uint8_t* atbl = dc_read_file(A "/act_table.raw", &b);
    uint8_t* otbl = dc_read_file(A "/out_table.raw", &b);
    uint8_t* act  = dc_read_file(A "/act_surface.raw", &b);
    struct dc_w4 e;
    int bad = 0;
    if (!wt || !bias || !atbl || !otbl || !act || b != ACT_B ||
        dc_w4_carve(&e, &ar, M, M, M, atbl, otbl)) {
        ex_log("!! assets/carve (act_b=%u)", b);
        ex_check("fence_conc_setup", 1, 0);
        wtcache_close(wc); return ex_summary();
    }
    memcpy(e.wt, wt, M * M / 2u);
    memcpy(e.bias, bias, (M / 32u) * 512u);
    bad += fence_handoff(e.wt, M * M / 2u, FC_CPU, FC_HMX, FM_VTCM);       /* FLUSH */
    bad += fence_handoff(e.bias, (M / 32u) * 512u, FC_CPU, FC_HMX, FM_VTCM);
    ex_check("fence_conc_setup", bad, 0);

    /* DDR 图案: act_surface 旋转 p*16KB 得 8 个互异真实 f16 图案;
     * 权重流纯 LCG (只作 DMA 负载, 内容自洽即可)。CPU 写过 → 一次 clean。 */
    for (uint32_t p = 0; p < NPAT; p++)
        for (uint32_t i = 0; i < ACT_B; i++)
            act_src[p][i] = act[(i + p * (ACT_B / NPAT)) % ACT_B];
    for (uint32_t i = 0; i < WGT_B; i++) wgt_src[i] = (uint8_t)lcg_n();
    dc_clean_ddr(act_src, sizeof(act_src));
    dc_clean_ddr(sink, sizeof(sink));
    dc_clean_ddr(wgt_src, WGT_B);

    /* ---- G2 串行参考 (阻塞 dc_dma 路径, 与并发共用指针几何) ---- */
    int64_t ser_us = 0;
    {
        struct dc_dma d;
        bad = 0;
        for (uint32_t p = 0; p < NPAT; p++) {
            int64_t t0 = tm_us();
            if (dc_dma_init(&d, act_src[p], staging[0], ACT_B, &mu)) { bad++; break; }
            if (dc_dma_once(&d)) bad++;
            dc_dma_destroy(&d);
            bad += fence_handoff(staging[0], ACT_B, FC_DMA, FC_HMX, FM_VTCM);
            e.act = staging[0]; e.out = outb[0];
            if (dc_w4_invoke(&e)) { bad++; break; }
            bad += fence_handoff(outb[0], OUT_B, FC_HMX, FC_DMA, FM_VTCM);
            if (dc_dma_init(&d, outb[0], sink[p], OUT_B, &mu)) { bad++; break; }
            if (dc_dma_once(&d)) bad++;
            dc_dma_destroy(&d);
            bad += fence_handoff(sink[p], OUT_B, FC_DMA, FC_CPU, FM_DDR);
            dc_w4_read_out(&e, refx[p]);  /* HMX 写绕 cache: 先 INVALIDATE */
            if (memcmp(sink[p], refx[p], OUT_B) != 0) bad++;
            ser_us += tm_us() - t0;
        }
        ex_check("fence_conc_serial_ref_8pat", bad, 0);
        ex_check("fence_conc_pattern_sensitivity",
                 memcmp(refx[0], refx[1], OUT_B) == 0, 0);
    }

    /* ---- G3/G4/G5 并发流水 ---- */
    int64_t ovl_us = 0, max_poll = 0, hmx_only = 0, dma_only = 0;
    int mism = 0, timedout = 0, worst = -1;
    {
        void* db[CH_N];
        void* d_act = NULL; void* d_mb = NULL;
        void* d_w[WGT_N];
        if (posix_memalign(&d_act, 16, DMA_DESC_SIZE_1D) ||
            posix_memalign(&d_mb, 16, DMA_DESC_SIZE_1D)) d_act = NULL;
        for (uint32_t i = 0; i < WGT_N; i++)
            if (posix_memalign(&d_w[i], 16, DMA_DESC_SIZE_1D)) d_w[i] = NULL;

        struct dc_dma d;
        bad = (d_act && d_mb) ? 0 : 1;
        /* prologue: 图案 0 阻塞入 staging[0] */
        bad += dc_dma_init(&d, act_src[0], staging[0], ACT_B, &mu);
        bad += dc_dma_once(&d);
        dc_dma_destroy(&d);

        /* G5 标定: hmx-only / dma-only(同 6-desc 链)分离计时。串行参考的
         * 阻塞 dc_dma_once 带 ~700µs/次 submit/drain 固定开销, 进分母会把
         * "隐藏搬运"证明变成 blocking-overhead 证明 — 等价和取干净分量。 */
        {
            int64_t t0 = tm_us();
            for (uint32_t i = 0; i < NPAT && !bad; i++) {
                bad += fence_handoff(staging[0], ACT_B, FC_DMA, FC_HMX, FM_VTCM);
                e.act = staging[0]; e.out = outb[0];
                if (dc_w4_invoke(&e)) { bad++; break; }
            }
            hmx_only = tm_us() - t0;
            bad += fence_handoff(outb[0], OUT_B, FC_HMX, FC_DMA, FM_VTCM);
            if (!bad) {
                int64_t t1 = tm_us();
                for (uint32_t i = 0; i < NPAT && !timedout; i++) {
                    void* cb[CH_N];
                    int n = 0;
                    bad += desc_set(d_act, act_src[1], staging[1], ACT_B);
                    cb[n++] = d_act;
                    for (uint32_t w = 0; w < WGT_N; w++) {
                        bad += desc_set(d_w[w], wgt_src + w * ACT_B,
                                        wgt_stage + w * ACT_B, ACT_B);
                        cb[n++] = d_w[w];
                    }
                    bad += desc_set(d_mb, outb[0], sink[0], OUT_B);
                    cb[n++] = d_mb;
                    if (bad || async_submit(cb, n)) { bad++; break; }
                    int64_t took = 0;
                    if (async_poll(cb, n, &took)) { timedout++; break; }
                    if (took > max_poll) max_poll = took;
                }
                dma_only = tm_us() - t1;
            }
        }

        for (int r = 0; r < ROUNDS && !bad && !timedout; r++) {
            uint32_t s = (uint32_t)r & 1u;
            int n = 0;
            bad += desc_set(d_act, act_src[((uint32_t)r + 1u) & 7u],
                            staging[((uint32_t)r + 1u) & 1u], ACT_B);
            db[n++] = d_act;
            for (uint32_t i = 0; i < WGT_N; i++) {
                bad += desc_set(d_w[i], wgt_src + i * ACT_B,
                                wgt_stage + i * ACT_B, ACT_B);
                db[n++] = d_w[i];
            }
            if (r >= 1) {
                bad += desc_set(d_mb, outb[s ^ 1u], sink[((uint32_t)r - 1u) & 7u], OUT_B);
                db[n++] = d_mb;
            }
            if (bad) break;

            int64_t t0 = tm_us();
            if (async_submit(db, n)) { bad++; break; }
            bad += fence_handoff(staging[s], ACT_B, FC_DMA, FC_HMX, FM_VTCM);
            e.act = staging[s]; e.out = outb[s];
            if (dc_w4_invoke(&e)) { bad++; break; }
            bad += fence_handoff(outb[s], OUT_B, FC_HMX, FC_DMA, FM_VTCM);
            int64_t took = 0;
            if (async_poll(db, n, &took)) { timedout++; break; }
            if (took > max_poll) max_poll = took;
            ovl_us += tm_us() - t0;

            if (r >= 1) {  /* moveback 已落: 逐字节对串行参考 */
                uint32_t pp = ((uint32_t)r - 1u) & 7u;
                bad += fence_handoff(sink[pp], OUT_B, FC_DMA, FC_CPU, FM_DDR);
                if (memcmp(sink[pp], refx[pp], OUT_B) != 0) { mism++; worst = r; }
            }
        }

        /* epilogue: 末轮 HMX 输出 moveback + 权重流内容校验 */
        if (!bad && !timedout) {
            struct dc_dma dm;
            bad += dc_dma_init(&dm, outb[((uint32_t)(ROUNDS - 1)) & 1u],
                               sink[((uint32_t)(ROUNDS - 1)) & 7u], OUT_B, &mu);
            bad += dc_dma_once(&dm);
            dc_dma_destroy(&dm);
            uint32_t pl = ((uint32_t)(ROUNDS - 1)) & 7u;
            bad += fence_handoff(sink[pl], OUT_B, FC_DMA, FC_CPU, FM_DDR);
            if (memcmp(sink[pl], refx[pl], OUT_B) != 0) { mism++; worst = ROUNDS; }
            bad += fence_handoff(wgt_stage, WGT_B, FC_DMA, FC_CPU, FM_VTCM);
            if (memcmp(wgt_stage, wgt_src, WGT_B) != 0) mism++;
        }
        free(d_act); free(d_mb);
        for (uint32_t i = 0; i < WGT_N; i++) free(d_w[i]);
    }

    ex_check("fence_conc_1000rounds_bitexact", bad + mism, 0);
    ex_check("fence_conc_watchdog_2s", timedout, 0);
    if (worst >= 0) ex_log("  first mismatch at round %d", worst);

    int64_t ser_avg = ser_us / (int64_t)NPAT;
    int64_t ovl_avg = ovl_us / ROUNDS;
    int64_t hmx_avg = hmx_only / (int64_t)NPAT;
    int64_t dma_avg = dma_only / (int64_t)NPAT;
    ex_log("  serial(blocking dc_dma_once)=%lldus/round — 含 ~2x700us submit/drain 固定开销",
           (long long)ser_avg);
    ex_log("  hmx_only=%lldus dma_only=%lldus equiv_sum=%lldus | overlap=%lldus/round max_poll=%lldus",
           (long long)hmx_avg, (long long)dma_avg, (long long)(hmx_avg + dma_avg),
           (long long)ovl_avg, (long long)max_poll);
    ex_check("fence_conc_overlap_vs_sum",
             ovl_avg <= ((hmx_avg + dma_avg) * 9) / 10 ? 0 : 1, 0);

    free(wt); free(bias); free(atbl); free(otbl); free(act);
    wtcache_close(wc);
    return ex_summary();
}
