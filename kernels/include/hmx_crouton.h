/*
 * hmx_crouton.h — HMX crouton 内存布局 pack/unpack 助手
 * =====================================================================
 * HMX crouton (2KB = 32×32 fp16) 统一 pair-interleave 布局 (52f67807 实测):
 *
 *  pos(row, col) = (row/2)*64 + 2*col + (row&1)
 *
 *  每个 HVX 向量 (128B = 64 fp16) 存一对相邻 row, element-interleaved:
 *    向量 k: [row(2k)col0, row(2k+1)col0, row(2k)col1, row(2k+1)col1, ...]
 *  (PRM "每个向量存储两个 spatial" + htpacc vshuff(v1,v0,-2) 交织)
 *
 *  激活 crouton (spatial × ic): row=spatial, col=ic
 *  权重 crouton (ic × oc):      row=ic,     col=oc
 *  输出 crouton (spatial × oc):  row=spatial, col=oc
 *
 *  本族只处理单 crouton (32×32). 多 crouton (大 M/N/K) 留 Phase 3.
 */
#ifndef HVXHMX_HMX_CROUTON_H
#define HVXHMX_HMX_CROUTON_H

#include "hvxhmx_types.h"

#define HMX_CR_DIM   32
#define HMX_CR_ELMS  1024   /* 32×32 */
#define HMX_CR_FP16_SZ 2048 /* bytes */

/* 统一 pair-interleave 位置公式 (row, col → crouton fp16 offset 0..1023) */
static inline unsigned hmx_crouton_pos(unsigned row, unsigned col)
{
    return (row / 2u) * 64u + 2u * col + (row & 1u);
}

/* 激活: row-major [32][32] → crouton (pair-interleave by spatial) */
void hmx_pack_act_fp16(__fp16 *dst_crouton, const __fp16 *src_rowmajor);

/* 权重: row-major src[ic][oc] → crouton (pair-interleave by ic) */
void hmx_pack_wgt_fp16(__fp16 *dst_crouton, const __fp16 *src_ic_oc);

/* 输出: crouton → row-major [32][32] */
void hmx_unpack_out_fp16(__fp16 *dst_rowmajor, const __fp16 *src_crouton);

/* int8 路径: 同 pair-interleave 布局. Phase 1 convbbb 用. */
void hmx_pack_act_u8(uint8_t *dst_crouton, const uint8_t *src_rowmajor);
void hmx_pack_wgt_i8(int8_t *dst_crouton, const int8_t *src_ic_oc);
void hmx_unpack_out_u8(uint8_t *dst_rowmajor, const uint8_t *src_crouton);

/* 反向查表: 给定 crouton fp16 offset (0..1023), 算 (row, col) — 供调试.
 * 是 hmx_crouton_pos 的逆. */
static inline void hmx_crouton_offset_to_row_col(unsigned flat, unsigned *row, unsigned *col) {
    *row = (flat / 64u) * 2u + (flat & 1u);
    *col = (flat % 64u) / 2u;
}

#endif /* HVXHMX_HMX_CROUTON_H */
