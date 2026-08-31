# api_v23_kvcache — U15 KV 槽缓存管理 (append/scatter/posmap)

源: `src/runtime/kvcache.c` · 头: `include/kvcache.h` · 例: `29_kvcache` (8 门 PASS)

## API

```c
#define KVC_POS_NONE 0xFFFFFFFF
struct kvc { base, slot_bytes, nslots, posmap, consumer, mem, n_append, n_evict; };

int  kvc_init(struct kvc* c, void* base, uint32_t slot_bytes,
              uint32_t nslots, uint32_t* posmap);   /* slot≥256 且 128 倍数, base 128 对齐 */
void kvc_set_consumer(struct kvc* c, int consumer, int mem);
int  kvc_append(struct kvc* c, const void* k, const void* v, uint32_t pos);
      /* slot = pos % nslots (回绕 evict), 返回槽号 */
int  kvc_scatter(struct kvc* c, uint32_t slot, const void* k, const void* v, uint32_t pos);
      /* verify 校正: 直接写指定槽 (非法槽拒绝) */
int  kvc_lookup(const struct kvc* c, uint32_t pos);  /* posmap[pos%n]==pos ? slot : -1 */
int  kvc_read(const struct kvc* c, uint32_t slot, void* k, void* v);
uint8_t* kvc_k_face / kvc_v_face(struct kvc* c, uint32_t slot);
uint32_t kvc_pos(const struct kvc* c, uint32_t slot);
```

- 槽 = K 面 + V 面各半 (`slot_bytes/2`); 写入内置 U9 fence
  (`FC_CPU → consumer × mem`, 典型 `FC_DMA,FM_DDR` = 铁律① FLUSH_INVALIDATE)。
- `n_evict` 精确计数 (旧 pos 被覆盖)。

## 位置语义 (DFlash posIdsIdx 对齐)

append 天然实现**位置动态**: 3N append 后, 旧 pos (<2N) 全 miss、新 pos (≥2N) 全
hit — 门 `wrap_posids_miss_old_hit_new` + `evict_count_exact` (evict=N)。

## verify 重写语义 (draft→target 校正)

draft 以 pos p 写入槽 `p%n`; target 校正时 `kvc_scatter(c, p%n, k', v', p)` —
**必须同槽且 pos 同余**, 否则 posmap 不变量破坏 (lookup 永远 miss)。
门 `verify_rewrite_posids_semantics`: 校正后 read=校正值 + lookup(pos) hit。

## 设备门 (29_kvcache, 2026-08-16, nslots=64, slot=256B)

init 非 128 倍数拒绝 / 64 槽 append+查读+posmap 逐字节 / 回绕 miss-hit-evict /
scatter 邻槽金丝雀隔离 / 槽 128B 对齐 / **DMA bypass 真读回** (dc_dma_once, 逐字节) /
verify 重写。8/8。

## 坑 (本轮实测)

- 例程读回缓冲必须在 G1 前分配 (NULL 传 kvc_read 被参数守卫挡回, 但随后 memcmp(NULL)
  直接 fault DSP — "Failed to call main" 无输出)。
- 测试用 pos 必须与槽同余 (`777 % 64 = 9 ≠ 5` → lookup 必 miss; 用 `773 = 64·12+5`)。
