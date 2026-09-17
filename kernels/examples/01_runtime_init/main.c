/*
 * 01_runtime_init — HMX runtime 生命周期最小示例
 * =====================================================================
 * 教学: hmx_runtime_setup / get_vtcm_base / get_vtcm_size / teardown.
 * 不调任何 kernel, 只演示 runtime 如何正确初始化和释放 — 这是用任何 HMX 算子前的第一步.
 *
 * 期望结果: PASS (setup 返回 0, VTCM 基地址非 NULL, 大小 == 申请值).
 */
#include "hvxhmx.h"
#include "example_util.h"

#define VTCM_SIZE (2u * 1024u * 1024u)

int main(void)
{
    ex_open_result("01_runtime_init");

    int rc = hmx_runtime_setup(VTCM_SIZE);
    if (rc != 0) {
        ex_log("setup FAIL rc=%d  (CDSP/fastrpc 未就绪? host 侧重连)", rc);
        ex_check("hmx_runtime_setup", rc, 0);
        return ex_summary();
    }
    ex_check("hmx_runtime_setup", rc, 0);

    void *base = hmx_runtime_get_vtcm_base();
    unsigned int sz = hmx_runtime_get_vtcm_size();
    ex_log("VTCM base=%p size=%u (0x%x) ctx_id=%u",
           base, sz, sz, hmx_runtime_get_ctx_id());

    /* VTCM 基地址必须非 NULL, 大小必须 >= 申请值 (设备常给整块, 例如本机固定 16MB) */
    int err = (base == NULL);
    ex_check("VTCM base non-NULL", err, 0);
    err = (sz < VTCM_SIZE);
    ex_check("VTCM size >= request", err, 0);

    /* 计时器冒烟 (应返回递增的 us 值) */
    long long t0 = hmx_perf_now_us();
    long long t1 = hmx_perf_now_us();
    err = (t1 < t0);
    ex_check("hmx_perf_now_us monotonic", err, 0);

    hmx_runtime_teardown();
    ex_log("teardown done");
    return ex_summary();
}
