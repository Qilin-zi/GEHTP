/*
 * 43_hwinfo — V81 硬件事实探针 (GEHTP 编译器/执行器设计锚点, 2026-09-17)
 * =====================================================================
 * 为编译器 cost model / 执行器调度设计钉死 V81 (52f67807) 真实硬件参数。
 * 纯查询 + 锁语义, 不跑 mxmem; 期望全 PASS.
 *
 *  A1 qurt_hvx_get_units()/get_mode()   HVX 单元数与位宽 (HVX 线程天花板)
 *  A2 HAP_compute_res_query_VTCM()      VTCM 总量/可用/页尺寸 (驻留池上限)
 *  A3 PCYCLE/µs 实测钟频 (qtimer µs 窗 + busy spin, DCVS PERF 角, 7 发取中位)
 *  A4 C15:14 直读 vs qurt_get_core_pcycles() (同一计数器互证 → QNN 口径可对齐)
 *  B1 HMX 锁语义: 主持 NON_SHARED → worker 抢锁期望立即 EFAILED (进程内排他);
 *     主改持 SHARED → worker SHARED 期望即得 (SHARED=多线程使能, 单元仍唯一)
 */
#include "hvxhmx.h"
#include "example_util.h"
#include "dc_threads.h"

#include <HAP_compute_res.h>
#include <HAP_perf.h>
#include "qurt.h"

static unsigned long long rd_c1514(void) {
    unsigned long long c;
    asm volatile("%0 = C15:14" : "=r"(c));
    return c;
}

static int cmp_ull(const void *a, const void *b) {
    unsigned long long x = *(const unsigned long long *)a;
    unsigned long long y = *(const unsigned long long *)b;
    return (x > y) - (x < y);
}

/* ---- B1 worker: 分相位抢 HMX 锁 ---- */
static volatile int g_phase;          /* 0=待主持锁 1=主持NON_SHARED 2=worker抢完
                                         3=主改持SHARED 4=worker第二轮完 */
static volatile int w_rc_ns, w_rc_sh, w_rc_sh2;
static volatile long long w_wait_us;
static unsigned int g_ctx;

static void lock_worker(void *arg) {
    (void)arg;
    while (g_phase != 1) qurt_sleep(1000);
    long long t0 = hmx_perf_now_us();
    w_rc_ns = HAP_compute_res_hmx_lock2(g_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED);
    w_rc_sh = HAP_compute_res_hmx_lock2(g_ctx, HAP_COMPUTE_RES_HMX_SHARED);
    w_wait_us = hmx_perf_now_us() - t0;
    if (w_rc_sh == 0) HAP_compute_res_hmx_unlock2(g_ctx, HAP_COMPUTE_RES_HMX_SHARED);
    if (w_rc_ns == 0) HAP_compute_res_hmx_unlock2(g_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED);
    g_phase = 2;
    while (g_phase != 3) qurt_sleep(1000);
    w_rc_sh2 = HAP_compute_res_hmx_lock2(g_ctx, HAP_COMPUTE_RES_HMX_SHARED);
    if (w_rc_sh2 == 0) HAP_compute_res_hmx_unlock2(g_ctx, HAP_COMPUTE_RES_HMX_SHARED);
    g_phase = 4;
}

int main(void)
{
    ex_open_result("43_hwinfo");

    /* ---- A1 HVX 单元/位宽 ---- */
    int units = qurt_hvx_get_units();
    int mode  = qurt_hvx_get_mode();
    int n128 = (units >> 8) & 0xFF, n64 = units & 0xFF;
    ex_log("A1 hvx_units=0x%x -> %d x 128B + %d x 64B; mode=%s",
           units, n128, n64, mode == 1 ? "128B" : mode == 0 ? "64B" : "N/A");
    ex_check("A1 hvx_units nonzero", units == 0, 0);

    /* ---- A2 VTCM 总量 (acquire 前, 免被自身占用污染 avail) ---- */
    unsigned int vtcm_total = 0, vtcm_avail = 0;
    compute_res_vtcm_page_t tp, ap;
    int rc = HAP_compute_res_query_VTCM(0, &vtcm_total, &tp, &vtcm_avail, &ap);
    ex_log("A2 VTCM total=%u B (%u MiB) avail=%u B total_pages: block=%u B n=%u",
           vtcm_total, vtcm_total >> 20, vtcm_avail, (unsigned)tp.block_size, (unsigned)tp.page_list_len);
    ex_check("A2 query_VTCM rc", rc, 0);
    ex_check("A2 VTCM total >= 8MiB", vtcm_total < (8u << 20) ? 1 : 0, 0);

    /* ---- A4 计数器互证: C15:14 直读 == qurt_get_core_pcycles ---- */
    {
        unsigned long long a = qurt_get_core_pcycles();
        unsigned long long b = rd_c1514();
        long long d = (long long)(b - a);
        if (d < 0) d = -d;
        ex_log("A4 qurt_pcycles=%llu C15:14=%llu |delta|=%lld", a, b, d);
        ex_check("A4 same counter (|delta| < 10000)", d >= 10000 ? 1 : 0, 0);
    }

    /* ---- 上电 (DCVS PERF 角) + VTCM/HMX ctx ---- */
    rc = hmx_runtime_setup(0);
    if (rc != 0) {
        ex_log("[FAIL] hmx_runtime_setup rc=%d", rc);
        return ex_summary() || 1;
    }

    /* ---- A3 PCYCLE/µs 实测 (7 发 5ms busy spin, 取中位) ---- */
    {
        unsigned long long rates[7];
        for (int i = 0; i < 7; i++) {
            long long t0 = hmx_perf_now_us();
            unsigned long long c0 = qurt_get_core_pcycles();
            volatile unsigned x = 0;
            while (hmx_perf_now_us() - t0 < 5000) x += 1;   /* ~5ms busy */
            (void)x;
            unsigned long long c1 = qurt_get_core_pcycles();
            long long t1 = hmx_perf_now_us();
            long long dt = t1 - t0;
            rates[i] = dt > 0 ? (c1 - c0) / (unsigned long long)dt : 0;  /* cyc/us */
        }
        qsort(rates, 7, sizeof(rates[0]), cmp_ull);
        ex_log("A3 PCYCLE/us median=%llu min=%llu max=%llu  (=> core ~%llu MHz)",
               rates[3], rates[0], rates[6], rates[3]);
        ex_check("A3 clock sane (500..3000 cyc/us)",
                 (rates[3] < 500 || rates[3] > 3000) ? 1 : 0, 0);
    }

    /* ---- B1 HMX 锁语义 ---- */
    g_ctx = hmx_runtime_get_ctx_id();
    g_phase = 0;
    dc_thread_t t;
    if (dc_spawn(&t, "w43", lock_worker, NULL, 16 * 1024) != 0) {
        ex_log("[FAIL] dc_spawn");
    } else {
        int rc_hold = HAP_compute_res_hmx_lock2(g_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED);
        g_phase = 1;
        long long w0 = hmx_perf_now_us();
        while (g_phase != 2 && hmx_perf_now_us() - w0 < 3000000) qurt_sleep(1000);
        ex_log("B1a hold NON_SHARED rc=%d; worker抢: ns_rc=%d sh_rc=%d wait=%lld us",
               rc_hold, w_rc_ns, w_rc_sh, w_wait_us);
        ex_check("B1a main hold rc", rc_hold, 0);
        /* 期望 worker 两抢皆立即拒 (EFAILED), 非阻塞 */
        ex_check("B1b NON_SHARED excludes worker (both rc != 0)",
                 (w_rc_ns == 0 || w_rc_sh == 0) ? 1 : 0, 0);
        ex_check("B1c rejection immediate (<100ms)", w_wait_us >= 100000 ? 1 : 0, 0);
        HAP_compute_res_hmx_unlock2(g_ctx, HAP_COMPUTE_RES_HMX_NON_SHARED);

        int rc_sh = HAP_compute_res_hmx_lock2(g_ctx, HAP_COMPUTE_RES_HMX_SHARED);
        g_phase = 3;
        w0 = hmx_perf_now_us();
        while (g_phase != 4 && hmx_perf_now_us() - w0 < 3000000) qurt_sleep(1000);
        ex_log("B1d hold SHARED rc=%d; worker SHARED rc=%d", rc_sh, w_rc_sh2);
        ex_check("B1d main SHARED rc", rc_sh, 0);
        ex_check("B1e SHARED enables worker thread", w_rc_sh2 == 0 ? 0 : 1, 0);
        HAP_compute_res_hmx_unlock2(g_ctx, HAP_COMPUTE_RES_HMX_SHARED);
        dc_join(&t);
    }

    hmx_runtime_teardown();
    return ex_summary();
}
