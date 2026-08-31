/*
 * hvx_reduction.h — V81 HVX 沿 D 维 reduction
 * =====================================================================
 *  - argminmax: 返回 (val, idx) of (min, max) along D axis
 *  - top1:      argmax of D (D ≤ 32)
 *  - reducesum: Σ a[i] along D
 *
 * 输入布局:
 *  - crouton_b/h: HxW x D, 元素按 [h, w, d] 连续 (D 通常 32)
 *  - flat_h:      一维 u16 序列
 *  - short_b:     短 (≤8) 元素 batch, byte 类型
 *  - dLE32:       D ≤ 32, 沿 D 维 reduction
 */
#ifndef HVXHMX_HVX_REDUCTION_H
#define HVXHMX_HVX_REDUCTION_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------
 *  ArgMin/Max 输出结构
 * ------------------------------------------------------------- */
typedef struct {
    int32_t min_val;   /* 最小值 (u8 扩到 i32) */
    int32_t max_val;   /* 最大值              */
    uint32_t min_idx;  /* 最小索引            */
    uint32_t max_idx;  /* 最大索引            */
} hvhx_argminmax_t;

/* 沿 D 维找 min/max 索引 (crouton_b: D=32, byte) */
void hvhx_argminmax_depth_crouton_b(const uint8_t *in, uint32_t hw,
                                    uint32_t d, hvhx_argminmax_t *out);

/* 沿 D 维找 min/max 索引 (crouton_h: D=32, half) */
void hvhx_argminmax_depth_crouton_h(const uint16_t *in, uint32_t hw,
                                    uint32_t d, hvhx_argminmax_t *out);

/* 沿 D 维找 min/max 索引 (dLE32: D≤32) */
void hvhx_argminmax_depth_dLE32_crouton_b(const uint8_t *in, uint32_t hw,
                                          uint32_t d, hvhx_argminmax_t *out);

/* 沿 D 维找 min/max 索引 (flat half) */
void hvhx_argminmax_depth_flat_h(const uint16_t *in, uint32_t hw,
                                 uint32_t d, hvhx_argminmax_t *out);

/* 沿 D 维找 min/max 索引 (short batch) */
void hvhx_argminmax_depth_short_b(const uint8_t *in, uint32_t hw,
                                  uint32_t d, hvhx_argminmax_t *out);

/* -------------------------------------------------------------
 *  Find max + index (返回单一 max)
 * ------------------------------------------------------------- */
void hvhx_find_max_and_index_in_depth_b(const uint8_t *in, uint32_t hw,
                                        uint32_t d, uint32_t *idx,
                                        uint8_t *max_val);

void hvhx_find_max_and_index_in_depth_h(const uint16_t *in, uint32_t hw,
                                        uint32_t d, uint32_t *idx,
                                        uint16_t *max_val);

/* -------------------------------------------------------------
 *  Top1 (D ≤ 32)
 * ------------------------------------------------------------- */
typedef struct {
    int32_t val;     /* max value (扩到 i32)  */
    uint32_t idx;    /* argmax index          */
} hvhx_top1_t;

void hvhx_top1_qu8_dLE32_cr2flt(const uint8_t *in, uint32_t hw,
                                uint32_t d, hvhx_top1_t *out);

/* -------------------------------------------------------------
 *  ReduceSum 沿 D 维
 *   out[i] = Σ_{d} a[i * step + d]
 *   case N: step = 1 / 2 / 4 / 6 / 8
 * ------------------------------------------------------------- */
void hvhx_reducesum_depth_u8(const uint8_t *in, uint32_t hw,
                             uint32_t d, uint32_t *sum_u32);

void hvhx_reduce_sum_u8_case_1(const uint8_t *in, uint32_t hw,
                               uint8_t *out);
void hvhx_reduce_sum_u8_case_2(const uint8_t *in, uint32_t hw,
                               uint8_t *out);
void hvhx_reduce_sum_u8_case_4(const uint8_t *in, uint32_t hw,
                               uint8_t *out);
void hvhx_reduce_sum_u8_case_6(const uint8_t *in, uint32_t hw,
                               uint8_t *out);
void hvhx_reduce_sum_u8_case_8(const uint8_t *in, uint32_t hw,
                               uint8_t *out);

void hvhx_reduce_sum_u16_case_1(const uint16_t *in, uint32_t hw,
                                uint16_t *out);
void hvhx_reduce_sum_u16_case_2(const uint16_t *in, uint32_t hw,
                                uint16_t *out);
void hvhx_reduce_sum_u16_case_4(const uint16_t *in, uint32_t hw,
                                uint16_t *out);
void hvhx_reduce_sum_u16_case_6(const uint16_t *in, uint32_t hw,
                                uint16_t *out);
void hvhx_reduce_sum_u16_case_8(const uint16_t *in, uint32_t hw,
                                uint16_t *out);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_HVX_REDUCTION_H */
