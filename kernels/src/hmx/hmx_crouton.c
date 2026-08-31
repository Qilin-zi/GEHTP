/*
 * hmx_crouton.c — HMX crouton 内存布局 pack/unpack
 * Module: hmx-lowlevel
 * Layout: pair-interleave — pos(row,col) = (row/2)*64 + 2*col + (row&1)
 *         每个 128B 向量 (64 fp16) 存一对相邻 row, element-interleaved.
 * Note:   激活/权重/输出三者同布局. 详见 docs/data_layout.md.
 */
#include "hmx_crouton.h"
#include <string.h>

/* 统一 pair-interleave 位置: 给定 (row, col) 算 crouton fp16 offset (0..1023) */
static inline unsigned hmx_pair_interleave_pos(unsigned row, unsigned col)
{
    return (row / 2u) * 64u + 2u * col + (row & 1u);
}

/* ---- fp16 ---- */

void hmx_pack_act_fp16(__fp16 *dst_crouton, const __fp16 *src_rowmajor)
{
    for (unsigned s = 0; s < HMX_CR_DIM; ++s)
        for (unsigned i = 0; i < HMX_CR_DIM; ++i)
            dst_crouton[hmx_pair_interleave_pos(s, i)] = src_rowmajor[s * HMX_CR_DIM + i];
}

void hmx_pack_wgt_fp16(__fp16 *dst_crouton, const __fp16 *src_ic_oc)
{
    for (unsigned ic = 0; ic < HMX_CR_DIM; ++ic)
        for (unsigned oc = 0; oc < HMX_CR_DIM; ++oc)
            dst_crouton[hmx_pair_interleave_pos(ic, oc)] = src_ic_oc[ic * HMX_CR_DIM + oc];
}

void hmx_unpack_out_fp16(__fp16 *dst_rowmajor, const __fp16 *src_crouton)
{
    for (unsigned s = 0; s < HMX_CR_DIM; ++s)
        for (unsigned o = 0; o < HMX_CR_DIM; ++o)
            dst_rowmajor[s * HMX_CR_DIM + o] = src_crouton[hmx_pair_interleave_pos(s, o)];
}

/* ---- int8 (假设同 pair-interleave 布局, 待 convbbb 对照 disasm 确认) ---- */

void hmx_pack_act_u8(uint8_t *dst_crouton, const uint8_t *src_rowmajor)
{
    for (unsigned s = 0; s < HMX_CR_DIM; ++s)
        for (unsigned i = 0; i < HMX_CR_DIM; ++i)
            dst_crouton[hmx_pair_interleave_pos(s, i)] = src_rowmajor[s * HMX_CR_DIM + i];
}

void hmx_pack_wgt_i8(int8_t *dst_crouton, const int8_t *src_ic_oc)
{
    for (unsigned ic = 0; ic < HMX_CR_DIM; ++ic)
        for (unsigned oc = 0; oc < HMX_CR_DIM; ++oc)
            dst_crouton[hmx_pair_interleave_pos(ic, oc)] = src_ic_oc[ic * HMX_CR_DIM + oc];
}

void hmx_unpack_out_u8(uint8_t *dst_rowmajor, const uint8_t *src_crouton)
{
    for (unsigned s = 0; s < HMX_CR_DIM; ++s)
        for (unsigned o = 0; o < HMX_CR_DIM; ++o)
            dst_rowmajor[s * HMX_CR_DIM + o] = src_crouton[hmx_pair_interleave_pos(s, o)];
}

