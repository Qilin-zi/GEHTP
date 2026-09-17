/*
 * hmx_common.c — HMX 共享运行时 (V81, 真 HAP API)
 * Module: runtime
 * Duty:   hmx_runtime_setup/teardown + VTCM 申请 + power_on + perf 计时
 * Note:   所有 HMX kernel 前必须 hmx_runtime_setup(). 详见 docs/api_runtime.md.
 */
#include "hmx_common.h"

#include <string.h>
#include <stdint.h>

#if defined(__hexagon__) || defined(__HVX__)
#include <HAP_farf.h>
#include <HAP_power.h>
#include <HAP_compute_res.h>
#include <HAP_perf.h>
#else
/* host build: FARF 退化成 printf, 无真 HAP */
#include <stdio.h>
#define FARF(level, ...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#endif

/* -------------------------------------------------------------
 *  跨 NSP 全局上下文
 * ------------------------------------------------------------- */
static unsigned int  g_vtcm_ctx_id = 0;
static void         *g_vtcm_base   = NULL;
static unsigned int  g_vtcm_size   = 0;
static int           g_power_on    = 0;

unsigned int hmx_runtime_get_ctx_id(void) { return g_vtcm_ctx_id; }
void        *hmx_runtime_get_vtcm_base(void) { return g_vtcm_base; }
unsigned int hmx_runtime_get_vtcm_size(void) { return g_vtcm_size; }

/* ============================================================
 *  Power: DCVS_v3 PERFORMANCE + HVX/HMX power_up
 *  照搬 vtcm-dma-bench.c power_on_hvx_hmx (设备已验证).
 *  HMX 指令前必须 power_up HMX, 否则非法指令异常.
 * ============================================================ */
#if defined(__hexagon__) || defined(__HVX__)
static int power_on_hvx_hmx(void)
{
    static int power_ctx = 0;
    HAP_power_request_t req;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) {
        FARF(ALWAYS, "[HMX power] apptype FAIL");
        return -1;
    }

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
    if (HAP_power_set((void *)&power_ctx, &req) != 0) {
        FARF(ALWAYS, "[HMX power] DCVS_v3 FAIL");
        return -2;
    }

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) {
        FARF(ALWAYS, "[HMX power] HVX power_up FAIL");
        return -3;
    }

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) {
        FARF(ALWAYS, "[HMX power] HMX power_up FAIL");
        return -4;
    }

    g_power_on = 1;
    FARF(ALWAYS, "[HMX power] DCVS PERF + HVX + HMX power_up OK");
    return 0;
}
#else
static int power_on_hvx_hmx(void) { g_power_on = 1; return 0; }
#endif

int hmx_power_on(void)
{
    if (g_power_on) return 0;
    return power_on_hvx_hmx();
}

/* ============================================================
 *  VTCM + HMX 一次性申请 (照搬 vtcm-dma-bench Way1 + htpacc vtcm_mgr.cc)
 *  顺序: power_on → query_VTCM → attr_init → set_vtcm_param_v2
 *        → set_hmx_param → acquire → get_vtcm_ptr → memset(0)
 * ============================================================ */
int hmx_runtime_setup(unsigned int vtcm_size)
{
    if (g_vtcm_base) return 0;  /* 幂等 */

    if (hmx_power_on() != 0) {
        FARF(ALWAYS, "[HMX setup] power_on failed");
        return -1;
    }

#if defined(__hexagon__) || defined(__HVX__)
    unsigned int total_size = 0, avail_size = 0;
    compute_res_vtcm_page_t total_pages, avail_pages;

    int err = HAP_compute_res_query_VTCM(0, &total_size, &total_pages,
                                         &avail_size, &avail_pages);
    if (err) {
        FARF(ALWAYS, "[HMX setup] query_VTCM FAIL 0x%x", err);
        return -2;
    }
    FARF(ALWAYS, "[HMX setup] VTCM total=%u KiB avail=%u KiB",
         total_size / 1024, avail_size / 1024);

    /* 申请全量 VTCM (vtcm_size 仅作下限参考; 实际拿 total) */
    (void)vtcm_size;
    unsigned int req_size = total_size;

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);

    if (compute_resource_attr_set_vtcm_param_v2) {
        compute_resource_attr_set_vtcm_param_v2(&attr, req_size, 0, 0);
    } else {
        HAP_compute_res_attr_set_vtcm_param(&attr, req_size, 1);
    }
    if (compute_resource_attr_set_hmx_param) {
        compute_resource_attr_set_hmx_param(&attr, 1);
    }

    unsigned int ctx = HAP_compute_res_acquire(&attr, 100000);
    if (ctx == 0) {
        FARF(ALWAYS, "[HMX setup] acquire FAIL (returned 0)");
        return -3;
    }
    g_vtcm_ctx_id = ctx;

    void *vtcm = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!vtcm) {
        compute_resource_attr_get_vtcm_ptr_v2(&attr, &vtcm, NULL);
    }
    if (!vtcm) {
        FARF(ALWAYS, "[HMX setup] get_vtcm_ptr NULL (ctx=%u)", ctx);
        HAP_compute_res_release(ctx);
        g_vtcm_ctx_id = 0;
        return -4;
    }

    g_vtcm_base = vtcm;
    g_vtcm_size = total_size;

    /* memset 防 CX_FAULT (RPCMEM_FD §7.4, htpacc vtcm_mgr.cc:68) */
    memset(g_vtcm_base, 0, total_size);

    FARF(ALWAYS, "[HMX setup] vtcm_base=%p size=%u 2KB_aligned=%s",
         g_vtcm_base, total_size,
         (((uintptr_t)g_vtcm_base & 0x7FF) == 0) ? "YES" : "NO");
    return 0;
#else
    (void)vtcm_size;
    /* host build: 伪造 VTCM (堆分配) 供标量路径 */
    static char fake_vtcm[2 * 1024 * 1024] __attribute__((aligned(2048)));
    g_vtcm_base = fake_vtcm;
    g_vtcm_size = sizeof(fake_vtcm);
    memset(g_vtcm_base, 0, g_vtcm_size);
    return 0;
#endif
}

void hmx_runtime_teardown(void)
{
#if defined(__hexagon__) || defined(__HVX__)
    if (g_vtcm_ctx_id) {
        HAP_compute_res_release(g_vtcm_ctx_id);
    }
#endif
    g_vtcm_ctx_id = 0;
    g_vtcm_base   = NULL;
    g_vtcm_size   = 0;
}

/* ============================================================
 *  HMX 执行权限: 本线程 timeshare HMX unit
 *  (照搬 htpacc hmx_mgr.c hmx_manager_enable/disable_execution)
 * ============================================================ */
void hmx_enable_execution(void)
{
#if defined(__hexagon__) || defined(__HVX__)
    if (g_vtcm_ctx_id) {
        HAP_compute_res_hmx_lock2(g_vtcm_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
    }
#endif
}

void hmx_disable_execution(void)
{
#if defined(__hexagon__) || defined(__HVX__)
    if (g_vtcm_ctx_id) {
        HAP_compute_res_hmx_unlock2(g_vtcm_ctx_id, HAP_COMPUTE_RES_HMX_SHARED);
    }
#endif
}

/* ============================================================
 *  HMX unit 自旋锁 (照搬 htpacc hmx_mgr.c hmx_unit_acquire/release)
 *  memw_locked 原子, 比 __sync_* 在 hexagon-clang 上更可靠.
 * ============================================================ */
static volatile int g_hmx_spinlock = 0;

#if defined(__hexagon__) || defined(__HVX__)
void hmx_unit_acquire(void)
{
    int *lock_ptr = (int *)&g_hmx_spinlock;
    asm volatile(
        "1:  r0 = memw_locked(%0)     \n"
        "    p0 = cmp.eq(r0, #0)      \n"
        "    if (!p0) jump 2f         \n"
        "    memw_locked(%0, p0) = %1 \n"
        "    if (p0) jump 3f          \n"
        "2:  pause(#8)                \n"
        "    jump 1b                  \n"
        "3:"
        : "+r"(lock_ptr)
        : "r"(1)
        : "p0", "r0", "memory");
}

void hmx_unit_release(void)
{
    __sync_synchronize();
    *(volatile int *)&g_hmx_spinlock = 0;
}
#else
void hmx_unit_acquire(void) {}
void hmx_unit_release(void) {}
#endif

/* ============================================================
 *  计时 (HAP_perf qtimer, 照搬 vtcm-dma-bench now_us)
 * ============================================================ */
long long hmx_perf_now_us(void)
{
#if defined(__hexagon__) || defined(__HVX__)
    return (long long)HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
#else
    return 0;
#endif
}
