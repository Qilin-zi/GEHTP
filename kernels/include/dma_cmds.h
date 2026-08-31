/* -------------------------------------------------------------------------
*
* Copyright (c) 2019-2022 Qualcomm Technologies, Inc. All Rights Reserved.
* Qualcomm Technologies Proprietary and Confidential.
*
*//*!
* \file    dma_commands.h
* \brief   DMA command definitions
*
*//*----------------------------------------------------------------------*/

#ifndef __DMA_COMMANDS_H
#define __DMA_COMMANDS_H

/****************************************************************************/
/* inlines                                                                  */
/****************************************************************************/

static inline void dmstart(void * next)
{
    asm volatile (" release(%0):at" : : "r"(next));
    asm volatile (" dmstart(%0)" : : "r" (next));
}

static inline void dmlink(void * cur, void * next)
{
    asm volatile (" release(%0):at" : : "r"(next));
    asm volatile (" dmlink(%0, %1)" : : "r" (cur), "r" (next));
}

static inline unsigned int dmpoll(void)
{
    unsigned int ret = 0;
    asm volatile (" %0 = dmpoll" : "=r"(ret): : "memory");
    return ret;
}

static inline unsigned int dmwait(void)
{
    unsigned int ret = 0;
    asm volatile (" %0 = dmwait" : "=r"(ret) : : "memory");
    return ret;
}

#endif /* __DMA_COMMANDS_H */
