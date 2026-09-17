# api_v23_arena — U10 双池对齐 arena (bump + free + 合并)

源: `src/runtime/arena.c` · 头: `include/arena.h` · 例: `24_arena` (4 门 PASS)

## 与 dc_arena 的关系

`dc_arena` (dc_parts.h) 是单池 bump、无 free — 引擎面 carve 一次成型时够用。
U10 补齐动态生命周期: **DDR/VTCM 双池单句柄**、任意 2^n 对齐 (128B HVX / 2KB HMX
一等公民)、free + 前后向合并 (boundary tag)、度量接口。

## API

```c
#define ARENA_ALIGN_HVX 128u
#define ARENA_ALIGN_HMX 2048u
enum { ARENA_DDR, ARENA_VTCM };

int    arena_init(struct arena* a, void* ddr, uint32_t ddr_bytes,
                  void* vtcm, uint32_t vtcm_bytes);   /* 池可 NULL (单池) */
void*  arena_alloc(struct arena* a, uint32_t bytes, uint32_t align, int loc);
void   arena_free(struct arena* a, void* p);
void   arena_reset(struct arena* a, int loc);         /* 语义 = 全 free */
uint32_t arena_used(const struct arena* a, int loc);
uint32_t arena_largest_free(const struct arena* a, int loc);
```

## 实现

边界标签分配器: 每区域 16B 头 `tag{size,free,req,data_off}` + 8B 尾 `ftr{size,free}`,
区域 16B 倍数; `data = align_up(区域头+16, align)`, 垫片记在 `data_off` —
**对齐由绝对地址保证**, 与池基址无关。free 时查前后邻居 tag 合并。

VTCM 池基址由调用方给 (典型: `wtcache_layout` temp 区 carve) — arena 不 acquire
任何硬件资源, 纯内存逻辑。

## 设备门 (24_arena, 2026-08-16)

- `arena_align_1000cycles`: 1000 轮随机 size/align (16..4096 / 双池) 指针恒对齐 + 无泄漏
- `arena_no_leak_full_coalesce`: 全部释放后 largest_free == cap (完全合并)
- `arena_gdnsm_w4_coexist_100iters`: gdn_sm (~1.1MB) 与 W4A16 双池 100 轮交替共存,
  frag 度量 `lf_vtcm=16578624/16769024` (≈98.9% 可用)
- `arena_frag_coalesce_bound`: 碎片上界门

## 坑

- 池基址需 16B 对齐 (memalign 128 / VTCM carve 天然满足)。
- VTCM 池容量受 wtcache temp 区限制 — gdn_tree 例的池 = `2×面 + 512KB` 余量才够。
