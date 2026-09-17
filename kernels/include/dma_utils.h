/* -------------------------------------------------------------------------
*
* Copyright (c) 2019-2020 Qualcomm Technologies, Inc. All Rights Reserved.
* Qualcomm Technologies Proprietary and Confidential.
*
*//*!
* \file    dma_utils.h
* \brief   Collection of DMA utility functions
*
*//*----------------------------------------------------------------------*/

#ifndef __DMA_UTILS_H
#define __DMA_UTILS_H

/****************************************************************************/
/* defines                                                                  */
/****************************************************************************/

#define DMA_COMPLETE        1
#define DMA_INCOMPLETE      0

#define DMA_SUCCESS         0
#define DMA_FAIL            -1

#define DMA_DESC_TYPE_1D    0
#define DMA_DESC_TYPE_2D    1

#define DMA_DESC_SIZE_1D    16
#define DMA_DESC_SIZE_2D    32

/****************************************************************************/
/* types                                                                    */
/****************************************************************************/

/*****************************/
/* DMA descriptor parameters */
/*****************************/

typedef struct _dma_desc_1d_params_t
{
    uint32_t src_address;
    uint32_t dst_address;
    uint32_t length;
    uint32_t order;
    uint32_t src_bypass;
    uint32_t dst_bypass;
} dma_desc_1d_params_t;

typedef struct _dma_desc_2d_params_t
{
    uint32_t src_address;
    uint32_t dst_address;
    uint32_t order;
    uint32_t src_bypass;
    uint32_t dst_bypass;
    uint32_t cache_allocation;
    uint32_t height;
    uint32_t width;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t src_width_offset;
    uint32_t dst_width_offset;
} dma_desc_2d_params_t;

/****************************************************************************/
/*  prototypes                                                              */
/****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Function:  dma_wait_for_idle
 * --------------------
 * Blocking wait for DMA to complete executing
 * Params: none
 * Returns: DMA_FAIL on error, else DMA_SUCCESS
 */
int dma_wait_for_idle(void);

/*
 * Function:  dma_is_done
 * --------------------
 * Non-blocking check for DMA to complete executing
 * Params: none
 * Returns: DMA_FAIL on error
 *          DMA_COMPLETE if DMA is idle and last desc is executed, else
 *          DMA_INCOMPLETE
 */
int dma_is_done(void);

/*
 * Function:  dma_dump_desc
 * --------------------
 * Dump descriptor parameters
 * Params: desc - descriptor to dump information for
 * Returns: none
 */
void dma_dump_desc(void * desc);

/*
 * Function:  dma_desc_init
 * --------------------
 * Intialize descriptor with descriptor parameters
 * Params: desc - descriptor to initialize
 *         desc_params - descriptor parameters for intializing
 *                       (of type dma_desc_1d_params_t or dma_desc_2d_params_t)
 *         desc_type - type of descriptor (DMA_DESC_TYPE_1D, DMA_DESC_TYPE_2D)
 * Returns: DMA_FAIL on error, else DMA_SUCCESS
 */
int dma_desc_init(void * desc, void * desc_params, int desc_type);

/*
 * Function:  dma_desc_submit
 * --------------------
 * Daisy chain the descriptor batch and submit for DMA execution
 * Params: desc_batch - array of descriptors for submission
 *         batch_len - numbers of descriptors for submission
 * Returns: DMA_FAIL on error, else DMA_SUCCESS
 */
int dma_desc_submit(void ** desc_batch, int batch_len);

/*
 * Function:  dma_desc_is_done
 * --------------------
 * Non-blocking check to see if descriptor has been executed
 * Params: desc - descriptor to check completion
 * Returns: DMA_FAIL on error
            DMA_COMPLETE if descriptor has been executed, else
            DMA_INCOMPLETE
 */
int dma_desc_is_done(void * desc);

#ifdef __cplusplus
}
#endif

#endif /* __DMA_UTILS_H */
