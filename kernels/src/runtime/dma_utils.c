/* -------------------------------------------------------------------------
*
* Copyright (c) 2019-2022 Qualcomm Technologies, Inc. All Rights Reserved.
* Qualcomm Technologies Proprietary and Confidential.
*
*//*!
* \file    dma_utils.c
* \brief   Collection of DMA utility functions
*
*//*----------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#define FARF_ERROR    1
#define FARF_HIGH     1
#include "HAP_farf.h"

#include "dma_utils.h"
#include "dma_regs.h"
#include "dma_desc.h"
#include "dma_cmds.h"

/*****************************************************************************/
/* globals                                                                   */
/*****************************************************************************/

void *g_last_desc = NULL;

/*****************************************************************************/
/* functions                                                                 */
/*****************************************************************************/

int dma_wait_for_idle(void)
{
    uint32_t dm0_status = dmwait() & DM0_STATUS_MASK;

    if (dm0_status == DM0_STATUS_IDLE)
        return DMA_SUCCESS;
    else
        return DMA_FAIL;
}

void dma_dump_desc(void *desc)
{
    uint32_t type;

    if (desc == NULL) {
        FARF(ERROR, "Invalid descriptor");
        return;
    }

    type = dma_desc_get_desctype(desc);

    FARF(HIGH, "DMA desc dump = 0x%lx\n", (uint32_t) desc);

    if (type == DESC_DESCTYPE_1D || type == DESC_DESCTYPE_2D) {
        FARF(HIGH, "   next             =0x%lx\n", dma_desc_get_next(desc));
        FARF(HIGH, "   length           =0x%lx\n", dma_desc_get_length(desc));
        FARF(HIGH, "   desctype         =0x%lx\n", dma_desc_get_desctype(desc));
        FARF(HIGH, "   bypass_src       =0x%lx\n", dma_desc_get_bypasssrc(desc));
        FARF(HIGH, "   bypass_dst       =0x%lx\n", dma_desc_get_bypassdst(desc));
        FARF(HIGH, "   order            =0x%lx\n", dma_desc_get_order(desc));
        FARF(HIGH, "   dstate           =0x%lx\n", dma_desc_get_dstate(desc));
        FARF(HIGH, "   src              =0x%lx\n", dma_desc_get_src(desc));
        FARF(HIGH, "   dst              =0x%lx\n", dma_desc_get_dst(desc));

        if (type == DESC_DESCTYPE_2D) {
            FARF(HIGH, "   roi_width        =0x%lx\n", dma_desc_get_roiwidth(desc));
            FARF(HIGH, "   roi_height       =0x%lx\n", dma_desc_get_roiheight(desc));
            FARF(HIGH, "   src_stride       =0x%lx\n", dma_desc_get_srcstride(desc));
            FARF(HIGH, "   dst_stride       =0x%lx\n", dma_desc_get_dststride(desc));
            FARF(HIGH, "   src_width_offset =0x%lx\n", dma_desc_get_srcwidthoffset(desc));
            FARF(HIGH, "   dst_width_offset =0x%lx\n", dma_desc_get_dstwidthoffset(desc));
            FARF(HIGH, "   cache_allocation =0x%lx\n", dma_desc_get_cachealloc(desc));
        }
    }
    else
        FARF(ERROR, "   Invalid descriptor type\n");
}

int dma_desc_init(void * desc, void * desc_params, int desc_type)
{
    int result = DMA_SUCCESS;

    if (desc == NULL) {
        FARF(ERROR, "Invalid descriptor");
        result = DMA_FAIL;
        goto out;
    }

    if (desc_params == NULL) {
        FARF(ERROR, "Invalid descriptor params");
        result = DMA_FAIL;
        goto out;
    }

    dma_desc_set_next(desc, NULL);
    dma_desc_set_dstate(desc, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_desctype(desc, desc_type);

    if (desc_type == DESC_DESCTYPE_1D) {
        dma_desc_set_src(desc, ((dma_desc_1d_params_t *) desc_params)->src_address);
        dma_desc_set_dst(desc, ((dma_desc_1d_params_t *) desc_params)->dst_address);
        dma_desc_set_order(desc, ((dma_desc_1d_params_t *) desc_params)->order);
        dma_desc_set_bypasssrc(desc, ((dma_desc_1d_params_t *) desc_params)->src_bypass);
        dma_desc_set_bypassdst(desc, ((dma_desc_1d_params_t *) desc_params)->dst_bypass);
        dma_desc_set_length(desc, ((dma_desc_1d_params_t *) desc_params)->length);
    }
    else if (desc_type == DESC_DESCTYPE_2D) {
        dma_desc_set_src(desc, ((dma_desc_2d_params_t *) desc_params)->src_address);
        dma_desc_set_dst(desc, ((dma_desc_2d_params_t *) desc_params)->dst_address);
        dma_desc_set_order(desc, ((dma_desc_2d_params_t *) desc_params)->order);
        dma_desc_set_bypasssrc(desc, ((dma_desc_2d_params_t *) desc_params)->src_bypass);
        dma_desc_set_bypassdst(desc, ((dma_desc_2d_params_t *) desc_params)->dst_bypass);
        dma_desc_set_cachealloc(desc, ((dma_desc_2d_params_t *) desc_params)->cache_allocation);
        dma_desc_set_roiwidth(desc, ((dma_desc_2d_params_t *) desc_params)->width);
        dma_desc_set_roiheight(desc, ((dma_desc_2d_params_t *) desc_params)->height);
        dma_desc_set_srcstride(desc, ((dma_desc_2d_params_t *) desc_params)->src_stride);
        dma_desc_set_dststride(desc, ((dma_desc_2d_params_t *) desc_params)->dst_stride);
        dma_desc_set_srcwidthoffset(desc, ((dma_desc_2d_params_t *) desc_params)->src_width_offset);
        dma_desc_set_dstwidthoffset(desc, ((dma_desc_2d_params_t *) desc_params)->dst_width_offset);
    }
    else {
        FARF(ERROR, "Invalid descriptor type");
        result = DMA_FAIL;
    }

 out:
    return result;
}

int dma_desc_submit(void ** desc_batch, int batch_len)
{
    int      i;
    int      result     = DMA_SUCCESS;
    uint32_t dm0_status = 0;

    if (desc_batch == NULL) {
        result = DMA_FAIL;
        FARF(ERROR, "Invalid descriptor batch\n");
        goto out;
    }

    if (batch_len < 1) {
        result = DMA_FAIL;
        FARF(ERROR, "Batch length cannot be less than 1\n");
        goto out;
    }

    for (i = 0; i < batch_len - 1; i++) {
        dma_desc_set_next(desc_batch[i], (uint32_t) desc_batch[i + 1]);
    }

    dma_desc_set_next(desc_batch[batch_len - 1], NULL);
    dm0_status = dmpoll() & DM0_STATUS_MASK;

    if (dm0_status == DM0_STATUS_IDLE) {
        dmstart(desc_batch[0]);
    }
    else if (dm0_status == DM0_STATUS_RUN) {
        if (g_last_desc == NULL) {
            FARF(ERROR, "Last descriptor not found for linking. Submission failed\n");
            result = DMA_FAIL;
            goto out;
        }
        else {
            dmlink(g_last_desc, desc_batch[0]);
        }
    }
    else {
        FARF(ERROR, "DMA not idle or running. Submission failed\n");
        result = DMA_FAIL;
        goto out;
    }

    dmpoll();

    g_last_desc = (void *) desc_batch[batch_len - 1];

 out:
    return result;
}

int dma_is_done(void)
{
    int      result     = DMA_INCOMPLETE;
    uint32_t dm0_status = dmpoll() & DM0_STATUS_MASK;

    if (dm0_status == DM0_STATUS_IDLE) {
        if (g_last_desc == NULL)
            result = DMA_COMPLETE;
        else if (dma_desc_get_dstate(g_last_desc) == DESC_DSTATE_COMPLETE)
            result = DMA_COMPLETE;
    }

    return result;
}

int dma_desc_is_done(void * desc)
{
    if (desc == NULL) {
        FARF(ERROR, "Invalid descriptor");
        return DMA_FAIL;
    }

    if (dma_desc_get_dstate(desc) == DESC_DSTATE_COMPLETE)
        return DMA_COMPLETE;
    else {
        dmpoll();
        return DMA_INCOMPLETE;
    }
}
