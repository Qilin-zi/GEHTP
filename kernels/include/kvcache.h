/* kvcache.h — U15 KV 槽缓存管理 (Genie qualla scatter/NativeKV 思路, 单层缩尺)
 *
 * 模型: nslots 个固定槽 (slot_bytes, 128 倍数), 环形复用 —
 *   posmap[slot] = 绝对位置 (KVC_POS_NONE = 空)
 *   append(pos):  槽 = pos % nslots, 写 K/V 面 (重写隔离: 只动本槽)
 *   lookup(pos):  posIdsIdx 模式 — 槽 = pos % nslots, 命中 ⇔ posmap[slot]==pos
 *                 (回绕后旧 pos 自动 miss — 位置动态语义)
 *   read(slot):   拷出 K/V 面
 *
 * 消费者 fence (依赖 U9): set_consumer 后每次写面对指定 (consumer,mem) 执行
 * fence_handoff — DMA bypass 读 DDR 的铁律① 由管理器承担, 调用方不再自查。
 */
#ifndef HVXHMX_V23_KVCACHE_H
#define HVXHMX_V23_KVCACHE_H

#include <stdint.h>

#define KVC_POS_NONE 0xFFFFFFFFu

struct kvc {
    uint8_t*  base;        /* K 面 | V 面: [slot][slot_bytes/2] 两半 */
    uint32_t  slot_bytes;  /* K+V 合计, 128 倍数 */
    uint32_t  nslots;
    uint32_t* posmap;      /* 调用方给 nslots 个 uint32 */
    int       consumer;    /* FC_CPU(默认)/FC_DMA/FC_HMX/FC_HVX */
    int       mem;         /* FM_DDR / FM_VTCM */
    uint32_t  n_append, n_evict;
};

int  kvc_init(struct kvc* c, void* base, uint32_t slot_bytes, uint32_t nslots,
              uint32_t* posmap);
/* 写 pos 的 K/V (k/v 各 slot_bytes/2 字节); 返回槽号, -1 参数错 */
int  kvc_append(struct kvc* c, const void* k, const void* v, uint32_t pos);
/* 显式槽重写 (target verify 重写 draft 槽); pos 更新 posmap */
int  kvc_scatter(struct kvc* c, uint32_t slot, const void* k, const void* v,
                 uint32_t pos);
int  kvc_lookup(const struct kvc* c, uint32_t pos);   /* 槽号, -1 miss */
int  kvc_read(const struct kvc* c, uint32_t slot, void* k, void* v);
void kvc_set_consumer(struct kvc* c, int consumer, int mem);
uint8_t* kvc_k_face(struct kvc* c, uint32_t slot);    /* 面直访 (调试/对拍) */
uint8_t* kvc_v_face(struct kvc* c, uint32_t slot);
uint32_t kvc_pos(const struct kvc* c, uint32_t slot);

#endif
