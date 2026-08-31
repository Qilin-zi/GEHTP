/* -------------------------------------------------------------------------
*
* Copyright (c) 2019-2020 Qualcomm Technologies, Inc. All Rights Reserved.
* Qualcomm Technologies Proprietary and Confidential.
*
*//*!
* \file    dma_desc.h
* \brief   DMA descriptor definitions
*
*//*----------------------------------------------------------------------*/

#ifndef __DMA_DESC_H
#define __DMA_DESC_H

/****************************************************************************/
/* defines                                                                  */
/****************************************************************************/

#define DESC_NEXT_MASK               0xFFFFFFFF
#define DESC_NEXT_SHIFT              0

#define DESC_DSTATE_MASK             0x80000000
#define DESC_DSTATE_SHIFT            31
#define DESC_DSTATE_INCOMPLETE       0
#define DESC_DSTATE_COMPLETE         1

#define DESC_ORDER_MASK              0x40000000
#define DESC_ORDER_SHIFT             30
#define DESC_ORDER_NOORDER           0
#define DESC_ORDER_ORDER             1

#define DESC_BYPASSSRC_MASK          0x20000000
#define DESC_BYPASSSRC_SHIFT         29
#define DESC_BYPASSDST_MASK          0x10000000
#define DESC_BYPASSDST_SHIFT         28
#define DESC_BYPASS_OFF              0
#define DESC_BYPASS_ON               1

#define DESC_DESCTYPE_MASK           0x03000000
#define DESC_DESCTYPE_SHIFT          24
#define DESC_DESCTYPE_1D             0
#define DESC_DESCTYPE_2D             1

#define DESC_LENGTH_MASK             0x00FFFFFF
#define DESC_LENGTH_SHIFT            0
#define DESC_SRC_MASK                0xFFFFFFFF
#define DESC_SRC_SHIFT               0
#define DESC_DST_MASK                0xFFFFFFFF
#define DESC_DST_SHIFT               0

#define DESC_CACHEALLOC_MASK         0x03000000
#define DESC_CACHEALLOC_SHIFT        24
#define DESC_CACHEALLOC_NONE         0
#define DESC_CACHEALLOC_WRITEONLY    1
#define DESC_CACHEALLOC_READONLY     2
#define DESC_CACHEALLOC_READWRITE    3

#define DESC_ROIWIDTH_MASK           0x0000FFFF
#define DESC_ROIWIDTH_SHIFT          0
#define DESC_ROIHEIGHT_MASK          0xFFFF0000
#define DESC_ROIHEIGHT_SHIFT         16

#define DESC_SRCSTRIDE_MASK          0x0000FFFF
#define DESC_SRCSTRIDE_SHIFT         0
#define DESC_DSTSTRIDE_MASK          0xFFFF0000
#define DESC_DSTSTRIDE_SHIFT         16

#define DESC_SRCWIDTHOFFSET_MASK     0x0000FFFF
#define DESC_SRCWIDTHOFFSET_SHIFT    0
#define DESC_DSTWIDTHOFFSET_MASK     0xFFFF0000
#define DESC_DSTWIDTHOFFSET_SHIFT    16

/****************************************************************************/
/* types                                                                    */
/****************************************************************************/

/**************************/
/* 1D (linear) descriptor */
/**************************/
typedef struct _dma_desc_1d_t
{
    uint32_t next;
    uint32_t dstate_order_bypass_desctype_length;
    uint32_t src;
    uint32_t dst;
} dma_desc_1d_t;

/***********************/
/* 2D (box) descriptor */
/***********************/
typedef struct _dma_desc_2d_t
{
    uint32_t next;
    uint32_t dstate_order_bypass_desctype_length;
    uint32_t src;
    uint32_t dst;
    uint32_t allocation;
    uint32_t roiheight_roiwidth;
    uint32_t dststride_srcstride;
    uint32_t dstwidthoffset_srcwidthoffset;
} dma_desc_2d_t;

/****************************************************************************/
/*  prototypes                                                              */
/****************************************************************************/

static inline uint32_t dma_desc_get_next(void *d)
{
    return(((((dma_desc_1d_t *) d)->next) & DESC_NEXT_MASK) >> DESC_NEXT_SHIFT);
}

static inline void dma_desc_set_next(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->next) &= ~DESC_NEXT_MASK;
    (((dma_desc_1d_t *) d)->next) |= ((v << DESC_NEXT_SHIFT) & DESC_NEXT_MASK);
}

static inline uint32_t dma_desc_get_dstate(void *d)
{
    return(((((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) & DESC_DSTATE_MASK) >> DESC_DSTATE_SHIFT);
}

static inline void dma_desc_set_dstate(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_DSTATE_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_DSTATE_SHIFT) & DESC_DSTATE_MASK);
}

static inline uint32_t dma_desc_get_order(void *d)
{
    return(((((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) & DESC_ORDER_MASK) >> DESC_ORDER_SHIFT);
}

static inline void dma_desc_set_order(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_ORDER_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_ORDER_SHIFT) & DESC_ORDER_MASK);
}

static inline uint32_t dma_desc_get_bypasssrc(void *d)
{
    return(((((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) & DESC_BYPASSSRC_MASK) >> DESC_BYPASSSRC_SHIFT);
}

static inline void dma_desc_set_bypasssrc(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_BYPASSSRC_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_BYPASSSRC_SHIFT) & DESC_BYPASSSRC_MASK);
}

static inline uint32_t dma_desc_get_bypassdst(void *d)
{
    return(((((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) & DESC_BYPASSDST_MASK) >> DESC_BYPASSDST_SHIFT);
}

static inline void dma_desc_set_bypassdst(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_BYPASSDST_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_BYPASSDST_SHIFT) & DESC_BYPASSDST_MASK);
}

static inline uint32_t dma_desc_get_desctype(void *d)
{
    return(((((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) & DESC_DESCTYPE_MASK) >> DESC_DESCTYPE_SHIFT);
}

static inline void dma_desc_set_desctype(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_DESCTYPE_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_DESCTYPE_SHIFT) & DESC_DESCTYPE_MASK);
}

static inline uint32_t dma_desc_get_length(void *d)
{
    return(((((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) & DESC_LENGTH_MASK) >> DESC_LENGTH_SHIFT);
}

static inline void dma_desc_set_length(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) &= ~DESC_LENGTH_MASK;
    (((dma_desc_1d_t *) d)->dstate_order_bypass_desctype_length) |= ((v << DESC_LENGTH_SHIFT) & DESC_LENGTH_MASK);
}

static inline uint32_t dma_desc_get_src(void *d)
{
    return(((((dma_desc_1d_t *) d)->src) & DESC_SRC_MASK) >> DESC_SRC_SHIFT);
}

static inline void dma_desc_set_src(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->src) &= ~DESC_SRC_MASK;
    (((dma_desc_1d_t *) d)->src) |= ((v << DESC_SRC_SHIFT) & DESC_SRC_MASK);
}

static inline uint32_t dma_desc_get_dst(void *d)
{
    return(((((dma_desc_1d_t *) d)->dst) & DESC_DST_MASK) >> DESC_DST_SHIFT);
}

static inline void dma_desc_set_dst(void *d, uint32_t v)
{
    (((dma_desc_1d_t *) d)->dst) &= ~DESC_DST_MASK;
    (((dma_desc_1d_t *) d)->dst) |= ((v << DESC_DST_SHIFT) & DESC_DST_MASK);
}

static inline uint32_t dma_desc_get_roiwidth(void *d)
{
    return(((((dma_desc_2d_t *) d)->roiheight_roiwidth) & DESC_ROIWIDTH_MASK) >> DESC_ROIWIDTH_SHIFT);
}

static inline void dma_desc_set_roiwidth(void *d, uint32_t v)
{
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) &= ~DESC_ROIWIDTH_MASK;
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) |= ((v << DESC_ROIWIDTH_SHIFT) & DESC_ROIWIDTH_MASK);
}

static inline uint32_t dma_desc_get_roiheight(void *d)
{
    return(((((dma_desc_2d_t *) d)->roiheight_roiwidth) & DESC_ROIHEIGHT_MASK) >> DESC_ROIHEIGHT_SHIFT);
}

static inline void dma_desc_set_roiheight(void *d, uint32_t v)
{
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) &= ~DESC_ROIHEIGHT_MASK;
    (((dma_desc_2d_t *) d)->roiheight_roiwidth) |= ((v << DESC_ROIHEIGHT_SHIFT) & DESC_ROIHEIGHT_MASK);
}

static inline uint32_t dma_desc_get_srcstride(void *d)
{
    return(((((dma_desc_2d_t *) d)->dststride_srcstride) & DESC_SRCSTRIDE_MASK) >> DESC_SRCSTRIDE_SHIFT);
}

static inline void dma_desc_set_srcstride(void *d, uint32_t v)
{
    (((dma_desc_2d_t *) d)->dststride_srcstride) &= ~DESC_SRCSTRIDE_MASK;
    (((dma_desc_2d_t *) d)->dststride_srcstride) |= ((v << DESC_SRCSTRIDE_SHIFT) & DESC_SRCSTRIDE_MASK);
}

static inline uint32_t dma_desc_get_dststride(void *d)
{
    return(((((dma_desc_2d_t *) d)->dststride_srcstride) & DESC_DSTSTRIDE_MASK) >> DESC_DSTSTRIDE_SHIFT);
}

static inline void dma_desc_set_dststride(void *d, uint32_t v)
{
    (((dma_desc_2d_t *) d)->dststride_srcstride) &= ~DESC_DSTSTRIDE_MASK;
    (((dma_desc_2d_t *) d)->dststride_srcstride) |= ((v << DESC_DSTSTRIDE_SHIFT) & DESC_DSTSTRIDE_MASK);
}

static inline uint32_t dma_desc_get_srcwidthoffset(void *d)
{
    return(((((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) & DESC_SRCWIDTHOFFSET_MASK) >> DESC_SRCWIDTHOFFSET_SHIFT);
}

static inline void dma_desc_set_srcwidthoffset(void *d, uint32_t v)
{
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) &= ~DESC_SRCWIDTHOFFSET_MASK;
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) |= ((v << DESC_SRCWIDTHOFFSET_SHIFT) & DESC_SRCWIDTHOFFSET_MASK);
}

static inline uint32_t dma_desc_get_dstwidthoffset(void *d)
{
    return(((((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) & DESC_DSTWIDTHOFFSET_MASK) >> DESC_DSTWIDTHOFFSET_SHIFT);
}

static inline void dma_desc_set_dstwidthoffset(void *d, uint32_t v)
{
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) &= ~DESC_DSTWIDTHOFFSET_MASK;
    (((dma_desc_2d_t *) d)->dstwidthoffset_srcwidthoffset) |= ((v << DESC_DSTWIDTHOFFSET_SHIFT) & DESC_DSTWIDTHOFFSET_MASK);
}

static inline uint32_t dma_desc_get_cachealloc(void *d)
{
    return(((((dma_desc_2d_t *) d)->allocation) & DESC_CACHEALLOC_MASK) >> DESC_CACHEALLOC_SHIFT);
}

static inline void dma_desc_set_cachealloc(void *d, uint32_t v)
{
    (((dma_desc_2d_t *) d)->allocation) &= ~DESC_CACHEALLOC_MASK;
    (((dma_desc_2d_t *) d)->allocation) |= ((v << DESC_CACHEALLOC_SHIFT) & DESC_CACHEALLOC_MASK);
}

#endif /* __DMA_DESC_H */
