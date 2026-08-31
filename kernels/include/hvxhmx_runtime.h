/*
 * hvxhmx_runtime.h — hvxhmx 库的运行时 (runtime) 公共 API
 * =====================================================================
 * 调用任何 HMX kernel 前, 必须先 hmx_runtime_setup() 一次 (内含 power_on +
 * VTCM 申请 + memset 防 CX_FAULT; 幂等). HVX kernel 建议也先 setup (顺带
 * 给 HVX 上电). 程序退出前调 hmx_runtime_teardown() 释放.
 *
 * 本头只含"调用方需要"的运行时 API; HMX intrinsic 宏封装 (HMX_LOAD_TILES_*
 * 等) 是内部实现, 见 hmx_common.h (不对外).
 */
#ifndef HVXHMX_RUNTIME_H
#define HVXHMX_RUNTIME_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  Tile / VTCM 常量 (调用方用来给 buffer 取尺寸/对齐)
 * ============================================================ */
#define HMX_TILE_DIM      32          /* HMX 单 tile 32×32                          */
#define HMX_FP16_TILE_SZ  2048        /* fp16 tile 32×32×2B = 2KB                   */
#define HMX_U8_TILE_SZ    1024        /* u8   tile 32×32×1B = 1KB                   */
#define HMX_VTCM_ALIGN    2048        /* HMX 要求 VTCM 2KB 对齐                      */

/* ============================================================
 *  执行控制 / 电源
 * ============================================================ */

/**
 * @brief 上电 HMX (及 HVX): DCVS_v3 PERFORMANCE + HVX/HMX power_up.
 * @return 0 成功, 非 0 失败码 (HAP error).
 * @note    通常无需直接调 — hmx_runtime_setup() 内部会调. 仅在手动管理电源时使用.
 * @pre     CDSP/fastrpc 已就绪 (见 docs/data_layout.md).
 */
extern int  hmx_power_on(void);

/** @brief 使能本线程的 HMX 执行 (HMX_compute_res_attr enable). */
extern void hmx_enable_execution(void);

/** @brief 关闭本线程的 HMX 执行. */
extern void hmx_disable_execution(void);

/** @brief 独占获取 HMX 单元 (acquire). 与 hmx_unit_release 配对. */
extern void hmx_unit_acquire(void);

/** @brief 释放 HMX 单元. */
extern void hmx_unit_release(void);

/* ============================================================
 *  VTCM runtime (核心生命周期 API)
 * ============================================================ */

/**
 * @brief 初始化 HMX 运行时: power_on + 申请指定大小的 VTCM + memset 清零.
 * @param[in] vtcm_size  申请的 VTCM 字节数 (必须 2KB 对齐; 建议 2*1024*1024).
 * @return    0 成功, 非 0 失败.
 * @pre       CDSP/fastrpc 就绪; 本函数幂等 (重复调用安全).
 * @note      任何 HMX kernel (convf16/convbbb/...) 调用前必须先 setup.
 *            memset 是为了规避 CDSP CX_FAULT (VTCM 残留脏数据).
 */
extern int  hmx_runtime_setup(unsigned int vtcm_size);

/** @brief 释放运行时资源 (VTCM 归还 + HMX 下电). 与 hmx_runtime_setup 配对. */
extern void hmx_runtime_teardown(void);

/** @return 当前 runtime 持有的 VTCM 基地址 (setup 后有效; 未 setup 返回 NULL). */
extern void        *hmx_runtime_get_vtcm_base(void);

/** @return 当前 runtime 持有的 VTCM 字节数. */
extern unsigned int hmx_runtime_get_vtcm_size(void);

/** @return 当前 HMX context id. */
extern unsigned int hmx_runtime_get_ctx_id(void);

/* ============================================================
 *  计时 (性能测量用)
 * ============================================================ */

/** @return 当前 qtimer 计数换算的微秒数 (HAP_perf). 用于 kernel 计时. */
extern long long hmx_perf_now_us(void);

/* ============================================================
 *  便利内联 (调用方做尺寸/对齐校验)
 * ============================================================ */

/** @return K 是否为 HMX_TILE_DIM (32) 的倍数 (HMX 要求 K 对齐). */
static inline int hmx_k_aligned(uint32_t K) {
    return (K % HMX_TILE_DIM) == 0;
}

/** @return 链式加载 n_tiles 个 tile 时的 dW 限值 = n_tiles*tile_size - 1. */
static inline unsigned int hmx_dW_limit(uint32_t n_tiles, size_t tile_size) {
    return (unsigned int) (n_tiles * tile_size - 1);
}

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_RUNTIME_H */
