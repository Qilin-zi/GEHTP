/*
 * 22_dualcore_threads — U3 dcthread 单元 设备验证 (双线程正确性)
 * =====================================================================
 * 覆盖 dc_spawn/dc_join, qurt barrier/mutex, VTCM 旗标握手,
 * 共享 DMA 锁双流, 双 dc_w4 引擎并发, dc_hvx_load/dc_norm_i16/dc_dot_u64
 * 跨线程确定性。判据全部 byte-exact / 等值 (并发不改数值):
 *   1) 并发 out == 串行 out (两引擎各自 byte-exact)
 *   2) hvx/norm/dot 两线程结果 == 主线程参考值 (确定性)
 *   3) 旗标握手完成 (f: 0→1→2)
 * hmx_lock 交接 (模块 C P3 结论): 主线程 spawn 前 unlock, worker 引擎段
 * batch lock, join 后主线程 re-lock — invoke HMX 的线程必须持有 hmx_lock。
 * 加速比只报告不设门 (4C 结论: 单 DMA 引擎 + HMX 锁, 线程并发不缩放)。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v22.h"
#include "example_util.h"

#define A "/data/local/tmp/hvxhmx23/assets/s256"
#define M 256u
#define K 256u
#define N 256u
#define NORM_N 256

static void cpu_to_vtcm(uint8_t* d, const uint8_t* s, uint32_t n) {
    memcpy(d, s, n);
    qurt_mem_cache_clean((qurt_addr_t)d, n, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
}

struct warg {
    int id;
    struct dc_w4* e;
    struct dc_dma dma;
    uint8_t* out;
    uint32_t act_bytes;
    dc_barrier_t* bstart;
    volatile uint32_t* go;
    volatile uint32_t* flag;
    uint8_t* hvx_scratch;
    const int16_t *nx, *na, *nb;
    int64_t inner_us;
    uint32_t hvx_sum;
    uint64_t dot;
    int16_t norm_out[NORM_N];
    uint32_t hs_done;
    struct wtcache_ctx* wc;                 /* P3: hmx_lock 持有线程属性 */
};

static void worker(void* p) {
    struct warg* w = (struct warg*)p;

    /* 线程内确定性元素运算 (与主线程同输入 → 必须同结果) */
    w->hvx_sum = dc_hvx_load(w->hvx_scratch, 1024);
    dc_norm_i16(w->nx, w->norm_out, NORM_N);
    w->dot = dc_dot_u64(w->na, w->nb, NORM_N);

    dc_barrier_wait(w->bstart);
    dc_flag_wait(w->go, 1);                 /* 主线程 t0 之后才放行 */

    int64_t t0 = HAP_perf_get_time_us();
    /* 模块 C P3 结论: invoke HMX 的线程必须持有 hmx_lock
     * (主线程已 unlock 交接; batch lock — 引擎段一次加锁) */
    if (wtcache_hmx_lock(w->wc) != 0) {
        w->hs_done = 0xEE01;
        w->inner_us = 0;
        goto shake;
    }
    dc_dma_once(&w->dma);                   /* 共享 submit 锁 */
    int bad = dc_w4_invoke(w->e);
    dc_w4_read_out(w->e, w->out);
    wtcache_hmx_unlock(w->wc);
    w->inner_us = HAP_perf_get_time_us() - t0;
    w->hs_done = bad ? 0xFFFF : w->inner_us > 0 ? 1 : 0;

    /* 旗标握手: id0 发 1 等 2; id1 等 1 发 2 (逻辑序无死锁) */
shake:
    if (w->id == 0) {
        dc_flag_set(w->flag, 1);
        dc_flag_wait(w->flag, 2);
    } else {
        dc_flag_wait(w->flag, 1);
        dc_flag_set(w->flag, 2);
    }
    w->hs_done = (w->hs_done == 1) ? 2 : w->hs_done;
}

int main(void) {
    ex_open_result("22_dualcore_threads");
    struct wtcache_ctx* wc = NULL;
    uint32_t bw = 0, bb = 0, ba = 0, bo = 0, b0 = 0, b1 = 0;
    uint8_t *s0 = NULL, *s1 = NULL, *c0 = NULL, *c1 = NULL;
    int16_t *nx = NULL, *na = NULL, *nb = NULL, *nr0 = NULL;
    uint64_t main_dot = 0;
    uint32_t main_hvx = 0;

    uint8_t* wt  = dc_read_file(A "/packed_weight.raw", &bw);
    uint8_t* bis = dc_read_file(A "/folded_bias.raw", &bb);
    uint8_t* at  = dc_read_file(A "/act_table.raw", &ba);
    uint8_t* ot  = dc_read_file(A "/out_table.raw", &bo);
    uint8_t* av0 = dc_read_file(A "/act_variants/v0.raw", &b0);
    uint8_t* av1 = dc_read_file(A "/act_variants/v1.raw", &b1);
    if (!wt || !bis || !at || !ot || !av0 || !av1 ||
        bw != K * N / 2 || b0 != M * K * 2 || b1 != M * K * 2) {
        ex_log("assets missing/mismatch");
        goto out;
    }
    dc_clean_ddr(wt, bw); dc_clean_ddr(bis, bb);
    dc_clean_ddr(av0, b0); dc_clean_ddr(av1, b1);   /* 铁律①: DMA bypass 源 */

    if (wtcache_open(&wc, 4096) != WTC_OK) { ex_log("wtcache_open FAIL"); goto out; }
    {
        void* vb; uint32_t vs; void* pb; uint32_t pc;
        wtcache_layout(wc, &vb, &vs, &pb, &pc);
        struct dc_arena ar;
        dc_arena_init(&ar, (uint8_t*)vb + ((pc + 2047u) & ~2047u),
                      vs - ((pc + 2047u) & ~2047u));

        struct dc_w4 e0, e1;
        if (dc_w4_carve(&e0, &ar, M, K, N, at, ot) ||
            dc_w4_carve(&e1, &ar, M, K, N, at, ot)) { ex_log("carve FAIL"); goto out; }
        cpu_to_vtcm(e0.wt, wt, bw);  cpu_to_vtcm(e0.bias, bis, bb);
        cpu_to_vtcm(e1.wt, wt, bw);  cpu_to_vtcm(e1.bias, bis, bb);

        volatile uint32_t* go   = (volatile uint32_t*)dc_arena_alloc(&ar, 4, 128);
        volatile uint32_t* flag = (volatile uint32_t*)dc_arena_alloc(&ar, 4, 128);
        if (!go || !flag) { ex_log("flag alloc FAIL"); goto out; }
        *go = 0; *flag = 0;

        nx = malloc(NORM_N * 2); na = malloc(NORM_N * 2); nb = malloc(NORM_N * 2);
        nr0 = malloc(NORM_N * 2);
        s0 = memalign(128, M * N * 2); s1 = memalign(128, M * N * 2);
        c0 = memalign(128, M * N * 2); c1 = memalign(128, M * N * 2);
        if (!nx || !na || !nb || !nr0 || !s0 || !s1 || !c0 || !c1) {
            ex_log("alloc FAIL"); goto out;
        }
        ex_fill_i16(nx, NORM_N, 7001, 0);
        ex_fill_i16(na, NORM_N, 7002, 0);
        ex_fill_i16(nb, NORM_N, 7003, 0);
        dc_norm_i16(nx, nr0, NORM_N);              /* 主线程参考值 */
        main_dot = dc_dot_u64(na, nb, NORM_N);
        uint8_t* hmain = dc_arena_alloc(&ar, 4096, 128);
        if (!hmain) { ex_log("hvx scratch FAIL"); goto out; }
        main_hvx = dc_hvx_load(hmain, 1024);

        dc_mutex_t mu;
        dc_mutex_init(&mu);
        dc_barrier_t bstart;
        dc_barrier_init(&bstart, 2);

        /* ---- 串行参考: 同引擎同输入顺序跑 ---- */
        struct dc_dma d0, d1;
        dc_dma_init(&d0, av0, e0.act, b0, &mu);
        dc_dma_init(&d1, av1, e1.act, b1, &mu);
        int64_t ts0 = HAP_perf_get_time_us();
        if (dc_dma_once(&d0) || dc_w4_invoke(&e0)) { ex_log("serial e0 FAIL"); goto out; }
        dc_w4_read_out(&e0, s0);
        if (dc_dma_once(&d1) || dc_w4_invoke(&e1)) { ex_log("serial e1 FAIL"); goto out; }
        dc_w4_read_out(&e1, s1);
        int64_t serial_us = HAP_perf_get_time_us() - ts0;

        /* ---- 并发: 双线程各自引擎/输入, 共享 DMA 锁 ---- */
        struct warg wa[2] = {
            { 0, &e0, d0, c0, b0, &bstart, go, flag, NULL, nx, na, nb, 0, 0, 0, {0}, 0 },
            { 1, &e1, d1, c1, b1, &bstart, go, flag, NULL, nx, na, nb, 0, 0, 0, {0}, 0 },
        };
        wa[0].hvx_scratch = dc_arena_alloc(&ar, 4096, 128);
        wa[1].hvx_scratch = dc_arena_alloc(&ar, 4096, 128);
        if (!wa[0].hvx_scratch || !wa[1].hvx_scratch) { ex_log("scratch FAIL"); goto out; }
        wa[0].wc = wc; wa[1].wc = wc;

        dc_thread_t t0, t1;
        int hmx_handoff = wtcache_hmx_unlock(wc);  /* P3 假设B: 主线程交锁给 worker */
        if (dc_spawn(&t0, "w22a", worker, &wa[0], 64 * 1024) ||
            dc_spawn(&t1, "w22b", worker, &wa[1], 64 * 1024)) {
            ex_log("dc_spawn FAIL (handoff=%d)", hmx_handoff);
            wtcache_hmx_lock(wc);
            goto out;
        }
        int64_t tc0 = HAP_perf_get_time_us();
        dc_flag_set(go, 1);                       /* 放行, 计时窗口打开 */
        dc_join(&t0);
        dc_join(&t1);
        int64_t conc_us = HAP_perf_get_time_us() - tc0;
        wtcache_hmx_lock(wc);                      /* 主线程取回 (后续 close 前还原) */

        /* ---- 判定: 并发数值 == 串行数值, 确定性运算 == 主线程 ---- */
        ex_check("concurrent_e0_byteexact", memcmp(c0, s0, M * N * 2) == 0 ? 0 : 1, 0);
        ex_check("concurrent_e1_byteexact", memcmp(c1, s1, M * N * 2) == 0 ? 0 : 1, 0);
        ex_check("hvx_deterministic",
                 wa[0].hvx_sum == main_hvx && wa[1].hvx_sum == main_hvx ? 0 : 1, 0);
        ex_check("norm_deterministic",
                 memcmp(wa[0].norm_out, nr0, NORM_N * 2) == 0 &&
                 memcmp(wa[1].norm_out, nr0, NORM_N * 2) == 0 ? 0 : 1, 0);
        ex_check("dot_deterministic",
                 wa[0].dot == main_dot && wa[1].dot == main_dot ? 0 : 1, 0);
        ex_check("flag_handshake_done",
                 wa[0].hs_done == 2 && wa[1].hs_done == 2 ? 0 : 1, 0);

        ex_log("serial %lld us | concurrent %lld us | ratio %.3f "
               "(HMX 锁+单 DMA 引擎, 预期 ~1, 只报告不设门)",
               (long long)serial_us, (long long)conc_us,
               conc_us ? (double)serial_us / (double)conc_us : 0.0);
        ex_log("inner_us a=%lld b=%lld", (long long)wa[0].inner_us,
               (long long)wa[1].inner_us);

        dc_dma_destroy(&d0);
        dc_dma_destroy(&d1);
    }

out:
    free(nx); free(na); free(nb); free(nr0);
    free(s0); free(s1); free(c0); free(c1);
    if (wc) wtcache_close(wc);
    return ex_summary();
}
