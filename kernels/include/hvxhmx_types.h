/*
 * hvxhmx_types.h — V81 HVX/HMX 公共类型与精度工具
 * =====================================================================
 * 适用于 Qualcomm Hexagon V81 (SM8750 / SA8797 / SA7255 gen5_gvm)
 *  - 128B HVX 向量
 *  - 32x32 HMX systolic matrix unit
 *  - VTCM 16MB 显式管理
 *
 * 设计原则
 *   1. host / DSP 双路径兼容:
 *      - DSP 编译 (__HVX__ define): 编译器内置 HVX_Vector / Q6_* 类型,
 *        通过 <hexagon_types.h> 暴露.
 *      - host 编译 (gcc/clang ARM64): 标量 stub, 标量 fallback 路径可用.
 *   2. 全部数值以 Q-定点 (Q7/Q15/Q31) 或 IEEE-754 fp16 表示, 不混用
 *   3. 边界保护: 显式 saturate / clip / div-by-zero guard
 *   4. 命名: 类型以 hvhx_ 前缀, 宏以 HVHX_ 前缀, 简短、可读
 */
#ifndef HVXHMX_TYPES_H
#define HVXHMX_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#if defined(__HVX__) || defined(__hexagon__)
  /* ============================================================
   *  DSP 路径: 编译器内置 HVX_Vector / HVX_VectorPair / HVX_VectorPred
   * ============================================================ */
  #include <hexagon_types.h>
  #include <hvx_hexagon_protos.h>
  #include <hmx_hexagon_protos.h>
  typedef HVX_Vector        hvhx_vec_t;
  typedef HVX_VectorPair    hvhx_vecp_t;
  typedef HVX_VectorPred    hvhx_pred_t;
#else
  /* ============================================================
   *  host 路径: stub 类型, 仅供标量 fallback 路径
   * ============================================================ */
  typedef long   hvhx_vec_t   __attribute__((__vector_size__(128)));
  typedef long   hvhx_vecp_t  __attribute__((__vector_size__(256)));
  typedef long   hvhx_pred_t  __attribute__((__vector_size__(4)));
  /* 兼容名, 真实 HVX 编译时无 */
  typedef hvhx_vec_t   HVX_Vector;
  typedef hvhx_vecp_t  HVX_VectorPair;
  typedef hvhx_pred_t  HVX_VectorPred;
#endif

/* ============================================================
 *  1. 基础向量/谓词别名
 * ============================================================ */

/* 每个 HVX 向量元素个数 (128B 路径) */
#define HVHX_VEC_BYTES         128
#define HVHX_VEC_ELEM_U8       (HVHX_VEC_BYTES)        /* 128 元素/向量 */
#define HVHX_VEC_ELEM_U16      (HVHX_VEC_BYTES / 2)    /*  64 元素/向量 */
#define HVHX_VEC_ELEM_U32      (HVHX_VEC_BYTES / 4)    /*  32 元素/向量 */

/* ============================================================
 *  2. Q-定点格式 (用于 HardSwish / PReLU / scale 缩放)
 *     与 libQnnHtpV81 一致: act/scale 通常 Q15, bias Q31
 * ============================================================ */
#define HVHX_Q7_MAX     127
#define HVHX_Q7_MIN     (-128)
#define HVHX_Q15_MAX    32767
#define HVHX_Q15_MIN    (-32768)
#define HVHX_Q31_MAX    2147483647
#define HVHX_Q31_MIN    (-2147483647 - 1)

/* 通用饱和 / 截断 (编译期) */
#define HVHX_SAT_U8(v)   ((uint8_t)  ((v) > UINT8_MAX  ? UINT8_MAX  : ((v) < 0 ? 0 : (v))))
#define HVHX_SAT_I8(v)   ((int8_t)   ((v) > INT8_MAX   ? INT8_MAX   : ((v) < INT8_MIN ? INT8_MIN : (v))))
#define HVHX_SAT_U16(v)  ((uint16_t) ((v) > UINT16_MAX ? UINT16_MAX : ((v) < 0 ? 0 : (v))))
#define HVHX_SAT_I16(v)  ((int16_t)  ((v) > INT16_MAX  ? INT16_MAX  : ((v) < INT16_MIN ? INT16_MIN : (v))))
#define HVHX_SAT_I32(v)  ((int32_t)  ((v) > INT32_MAX  ? INT32_MAX  : ((v) < INT32_MIN ? INT32_MIN : (v))))

/* i32 整除四舍五入 (匹配 libQnnHtpV81 qf32 倒数的 round-to-nearest 行为) */
static inline int32_t hvhx_round_div_i32(int32_t a, int32_t b) {
    int32_t q = a / b;
    int32_t r = a % b;
    int64_t twice = (int64_t)r * 2;
    int64_t abs_b = (b < 0) ? -(int64_t)b : (int64_t)b;
    if (twice < 0) twice = -twice;
    if (twice > abs_b) q += ((a > 0) == (b > 0)) ? 1 : -1;
    return q;
}

/* 整除的 0-divisor 守卫 — 防止硬件 fault */
#define HVHX_GUARD_DIV_U8(a, b)  ((b) == 0 ? UINT8_MAX  : (uint8_t) ((a) / (b)))
#define HVHX_GUARD_DIV_U16(a, b) ((b) == 0 ? UINT16_MAX : (uint16_t)((a) / (b)))
#define HVHX_GUARD_DIV_I32(a, b) \
    ((b) == 0 ? ((a) < 0 ? INT32_MIN : INT32_MAX) : hvhx_round_div_i32((a), (b)))

/* ============================================================
 *  3. VTCM 内存管理 (16MB, 显式 alloc, DMA-friendly)
 * ============================================================ */
#define HVHX_VTCM_ALIGN       2048
#define HVHX_VTCM_TILE_FP16   2048
#define HVHX_VTCM_TILE_U8     1024

/* VTCM 区域 (bump-allocator 返回的句柄) */
typedef struct {
    void   *base;
    size_t  size;
} hvhx_vtcm_region_t;

/* HMX 上下文 */
typedef struct {
    hvhx_vtcm_region_t vtcm;
    uint32_t           enabled;
} hvhx_hmx_ctx_t;

/* ============================================================
 *  4. 工具: 谓词 = 0 (全 false), 谓词 = all-true
 *     host 路径用 stub, DSP 路径由 q6_intrinsics.h 提供 Q6_Q_vsetq_R
 * ============================================================ */
static inline hvhx_pred_t hvhx_pred_false(void) {
    hvhx_pred_t p = (hvhx_pred_t) {0};
    return p;
}

static inline hvhx_pred_t hvhx_pred_true(void) {
#if defined(__HVX__) || defined(__hexagon__)
    hvhx_vec_t zero = Q6_V_vzero();
    return Q6_Q_vcmp_eq_VwVw(zero, zero);
#else
    hvhx_pred_t p;
    memset(&p, 0xff, sizeof(p));
    return p;
#endif
}

#endif /* HVXHMX_TYPES_H */
