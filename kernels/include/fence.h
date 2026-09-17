/* fence.h — V2.3 方向对偶 cache fence (U9, 源: dc_sync/dc_parts 升格)
 *
 * 把 V81 cache 协议铁律收敛成一张 (写者, 读者, 存储域) 三元组查询表:
 * 调用方只声明数据流向, fence 负责展开成正确的 qurt cache 操作 (或 no-op)。
 * 单一调用点约定: 写者写完之后、读者读之前, 在边界调一次。
 *
 * 语义表 (V2.2 四铁律 + 4C/T10/wtcache 实测整合):
 *   写者    读者    域      展开                        依据
 *   CPU     DMA     DDR     FLUSH_INVALIDATE(clean)     铁律① CPU写DDR→DMA bypass
 *   CPU     DMA     VTCM    FLUSH                       VTCM→DMA (dc_dma_once src 契约)
 *   CPU     HMX     VTCM    FLUSH                       cpu_to_vtcm 同式
 *   CPU     HVX     任意    no-op                       同线程 dcache 一致
 *   CPU     CPU     VTCM    no-op                       P4 实测跨线程天然一致
 *   CPU     CPU     DDR     no-op                       同簇 dcache 一致
 *   DMA     CPU     任意    INVALIDATE                  铁律③ DMA写→CPU读
 *   DMA     HVX     任意    INVALIDATE                  HVX load 走 dcache
 *   DMA     HMX     VTCM    INVALIDATE                  供给面语义 (T10)
 *   HMX     CPU     VTCM    INVALIDATE                  dc_w4_read_out 同式
 *   HMX     DMA     VTCM    FLUSH                       move_back src 契约 (T5)
 *   HMX     *       DDR     拒绝 (HMX 只访 VTCM)
 *   DMA     *       DMA     FLUSH (src 侧; dst 由引擎 bypass 写)
 */
#ifndef HVXHMX_V23_FENCE_H
#define HVXHMX_V23_FENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum fence_chan { FC_CPU = 0, FC_DMA = 1, FC_HMX = 2, FC_HVX = 3 };
enum fence_mem  { FM_DDR = 0, FM_VTCM = 1 };

#define FENCE_OK          0
#define FENCE_ERR_COMBO  -1   /* 该 (写,读,域) 组合在 V81 不存在/不合法 */

/* 展开方向对偶的 cache 操作。bytes=0 合法 (no-op)。
 * 返回 FENCE_OK 或 FENCE_ERR_COMBO。 */
int fence_handoff(void* ptr, uint32_t bytes, int writer, int reader, int mem);

/* 查表不打操作 — 供测试/静态断言决策表用 */
int fence_op_for(int writer, int reader, int mem);   /* 返回下述 FO_* */

enum fence_op {
    FO_NONE = 0,        /* 天然一致, 不需要操作 */
    FO_FLUSH,           /* qurt FLUSH (writeback) */
    FO_INVALIDATE,      /* qurt INVALIDATE (丢弃驻留行) */
    FO_FLUSH_INVALIDATE,/* writeback + discard (CPU写DDR给bypass读) */
    FO_INVALID,         /* 非法组合 */
};

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_FENCE_H */
