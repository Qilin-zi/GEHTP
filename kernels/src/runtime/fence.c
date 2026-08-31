/* fence.c — V2.3 U9 方向对偶 cache fence (决策表实现)
 *
 * 决策表来源: V2.2 docs/api_v22_overview.md 四铁律 + 4C (P4 跨线程一致 /
 * move_back src FLUSH) + T10 (HMX 面供给 INVALIDATE) 设备实测。
 * 唯一职责: 三元组 → 正确的 qurt_mem_cache_clean 调用 (或 no-op)。
 */
#include <qurt.h>

#include "fence.h"

int fence_op_for(int writer, int reader, int mem) {
    /* HVX 写者与 CPU 同为 dcache 代理 (HVX store 走 dcache) → 同 CPU 语义 */
    if (writer == FC_HVX) writer = FC_CPU;
    /* HMX 只访 VTCM (mxmem 物理限制; DDR 直访 fault) */
    if ((writer == FC_HMX || reader == FC_HMX) && mem == FM_DDR) return FO_INVALID;

    if (writer == FC_CPU) {
        if (reader == FC_DMA)
            return (mem == FM_DDR) ? FO_FLUSH_INVALIDATE : FO_FLUSH;
        if (reader == FC_HMX)  return FO_FLUSH;      /* HMX ⇒ VTCM */
        return FO_NONE;                              /* HVX/CPU 天然一致 */
    }
    if (writer == FC_DMA) {
        if (reader == FC_CPU || reader == FC_HVX || reader == FC_HMX)
            return FO_INVALIDATE;
        if (reader == FC_DMA) return FO_FLUSH;       /* 引擎间 src 保持可见 */
        return FO_INVALID;
    }
    if (writer == FC_HMX) {                          /* HMX ⇒ VTCM */
        if (reader == FC_CPU || reader == FC_HVX) return FO_INVALIDATE;
        if (reader == FC_DMA) return FO_FLUSH;
        return FO_INVALID;
    }
    return FO_INVALID;
}

int fence_handoff(void* ptr, uint32_t bytes, int writer, int reader, int mem) {
    int op = fence_op_for(writer, reader, mem);
    if (op == FO_INVALID) return FENCE_ERR_COMBO;
    if (op == FO_NONE || bytes == 0) return FENCE_OK;
    unsigned flag;
    switch (op) {
    case FO_FLUSH:          flag = QURT_MEM_CACHE_FLUSH;          break;
    case FO_INVALIDATE:     flag = QURT_MEM_CACHE_INVALIDATE;     break;
    case FO_FLUSH_INVALIDATE: flag = QURT_MEM_CACHE_FLUSH_INVALIDATE; break;
    default: return FENCE_OK;
    }
    qurt_mem_cache_clean((qurt_addr_t)ptr, bytes, flag, QURT_MEM_DCACHE);
    return FENCE_OK;
}
