/*
 * hmx_common.h — HMX kernel 内部实现头 (不对外)
 * =====================================================================
 * 库内部使用: HMX C intrinsic 宏封装 + 内部 tile 常量 + 溢出保护.
 * 调用方请 #include "hvxhmx.h" (或 hvxhmx_runtime.h), 不要直接 include 本头.
 *
 * 公共 runtime API (hmx_runtime_setup/power_on/...) 已抽到 hvxhmx_runtime.h,
 * 本头通过 include 它获得这些声明 (保持所有 hmx/*.c 的预处理结果不变).
 *
 * 真实 V81 HMX intrinsic (来自 target/hexagon/include/hmx_hexagon_protos.h):
 *  - Q6_mxclracc / Q6_mxclracc_hf              — 清累加器 (int32 / fp32)
 *  - Q6_activation_hf_mxmem_RR                 — 加载 fp16 activation tile
 *  - Q6_weight_hf_mxmem_RR                     — 加载 fp16 weight tile
 *  - Q6_activation_ub_mxmem_RR[_cm]            — 加载 u8 activation ([:cm 压缩])
 *  - Q6_weight_b_mxmem_RR                      — 加载 int8 weight tile
 *  - Q6_mxmem_AR_before_cm_sat_ub              — cvt:sat.ub 写回
 *  - Q6_mxmem_AR_after_hf                      — cvt.hf 写回
 *  - Q6_bias_mxmem2_A                          — bias = mxmem2(addr)
 *
 * 数值精度:
 *  - int8 GEMM: 累加器 37-bit 整数, 输出 cvt + sat 到 u8
 *  - fp16  GEMM: 累加器 37-bit 浮点 (fp32 mantissa), 输出 cvt.hf 截断
 *  - 边界: K 必须是 32 倍数 (crouton 边界对齐)
 */
#ifndef HVXHMX_HMX_COMMON_H
#define HVXHMX_HMX_COMMON_H

#include "hvxhmx_types.h"
#include "hvxhmx_runtime.h"   /* 公共 runtime API + tile/VTCM 公共常量 + 内联 helper */

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#endif

/* ============================================================
 *  内部 tile 尺寸常量 (公共部分在 hvxhmx_runtime.h)
 * ============================================================ */
#define HMX_INT8_TILE_SZ  1024        /* int8 weight tile, no :cm (32×32×1B) */
#define HMX_FP16_CR_CM    2048        /* :cm u8 crouton 2KB                  */

/* HMX 一次性可链式加载的 tile 数上限: 32 (16-bit dW 字段) */
#define HMX_MAX_CHAIN     32

/* ============================================================
 *  HMX C intrinsic 宏封装 (基于 V81 真实签名, 全部走 Q6_* C intrinsic)
 *  设备上由 hexagon-clang 把 Q6_* 编译为真实 HMX 指令.
 * ============================================================ */

/* 清累加器 (FP16 / INT8 双模式) */
#define HMX_CLRACC_FP16()   do { Q6_mxclracc_hf(); } while (0)
/* 注: Q6_mxclracc 清 int32 累加器, INT8 GEMM 用同一指令 (HMX 累加器
 *  模式由最近的 tile load 决定) */
#define HMX_CLRACC_INT8()   do { Q6_mxclracc(); } while (0)

/* 加载 FP16 tile pair (act + wgt) — 1d/linear 1x32 spatial */
#define HMX_LOAD_TILES_FP16(act_addr, wgt_addr, dW_limit, spatial_mask)        \
    do {                                                                       \
        unsigned int _dW   = (unsigned int)(dW_limit);                         \
        Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)(act_addr), _dW);    \
        Q6_weight_hf_mxmem_RR    ((unsigned int)(uintptr_t)(wgt_addr), _dW);    \
    } while (0)

/* 加载 INT8 tile pair (u8 act + i8 wgt), 无 :cm */
#define HMX_LOAD_TILES_INT8(act_addr, wgt_addr, dW_limit, spatial_mask)        \
    do {                                                                       \
        unsigned int _dW   = (unsigned int)(dW_limit);                         \
        Q6_activation_ub_mxmem_RR((unsigned int)(uintptr_t)(act_addr), _dW);   \
        Q6_weight_b_mxmem_RR    ((unsigned int)(uintptr_t)(wgt_addr), _dW);    \
    } while (0)

/* 加载 INT8 tile pair, :cm 压缩 (V81 INT8 常用) */
#define HMX_LOAD_TILES_INT8_CM(act_addr, wgt_addr, dW_limit)                   \
    do {                                                                       \
        unsigned int _dW   = (unsigned int)(dW_limit);                         \
        Q6_activation_ub_mxmem_RR_cm((unsigned int)(uintptr_t)(act_addr), _dW);\
        Q6_weight_b_mxmem_RR        ((unsigned int)(uintptr_t)(wgt_addr), _dW);\
    } while (0)

/* 写回 FP16 累加器 */
#define HMX_CONSUME_ACC_FP16(out_addr, spatial_mask)                            \
    do {                                                                       \
        unsigned int _sp = (unsigned int)(spatial_mask);                       \
        Q6_mxmem_AR_after_hf((void *)(out_addr), _sp);                          \
    } while (0)

/* 写回 INT8 累加器 (sat, u8, :cm) */
#define HMX_CONSUME_ACC_INT8_CM(out_addr)                                       \
    do {                                                                       \
        Q6_mxmem_AR_before_cm_sat_ub((void *)(out_addr), (unsigned int)0);     \
    } while (0)

/* 写回 INT8 累加器 (sat, u8) */
#define HMX_CONSUME_ACC_INT8(out_addr)                                          \
    do {                                                                       \
        Q6_mxmem_AR_before_cm_sat_ub((void *)(out_addr), (unsigned int)0);     \
    } while (0)

/* ============================================================
 *  Bias 寄存器: 真实 V81 HMX 用 Q6_bias_mxmem2_A
 *  1 次装 1 个 fp16 (int8 bias 用相同指令, 但解译为 int32)
 * ============================================================ */
#define HMX_SET_BIAS_FP16(bias_vtcm_addr)                                       \
    do {                                                                       \
        Q6_bias_mxmem2_A((void *)(bias_vtcm_addr));                            \
    } while (0)

/* ============================================================
 *  累加器溢出保护 (内部断言用)
 * ============================================================ */

/* 累加器 37-bit 上限检查: int32 acc 不会溢出, 警告 */
#define HMX_ACC_OVERFLOW_INT32(acc)                                            \
    ((acc) > (1LL << 36) || (acc) < -(1LL << 36))

#endif /* HVXHMX_HMX_COMMON_H */
