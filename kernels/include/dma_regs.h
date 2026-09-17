/* -------------------------------------------------------------------------
*
* Copyright (c) 2019-2020 Qualcomm Technologies, Inc. All Rights Reserved.
* Qualcomm Technologies Proprietary and Confidential.
*
*//*!
* \file    dma_regs.h
* \brief   DMA control register definitions
*
*//*----------------------------------------------------------------------*/

#ifndef __DMA_REGS_H
#define __DMA_REGS_H

/****************************************************************************/
/* defines                                                                  */
/****************************************************************************/

#define DM0_STATUS_MASK        0x00000003
#define DM0_STATUS_SHIFT       0
#define DM0_STATUS_IDLE        0
#define DM0_STATUS_RUN         1
#define DM0_STATUS_ERROR       2

#define DM0_DESC_ADDR_MASK     0xFFFFFFF0
#define DM0_DESC_ADDR_SHIFT    4

#endif /* __DMA_REGS_H */
