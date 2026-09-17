/* arena.h — V2.3 双池对齐 arena (U10; dc_arena bump 版的演进)
 *
 * dc_arena (dc_parts.h) 是单池 bump、无 free; 本单元补齐:
 *   - DDR/VTCM 双池, 单一 arena 句柄
 *   - 128B (HVX aligned load) / 2KB (HMX mxmem) 对齐一级公民
 *   - free + 前后向相邻合并 (boundary tag), 抗碎片
 *   - 度量: arena_used / arena_largest_free
 *
 * 实现: 边界标签分配器 — 每区域头 tag(16B: size/free/req/data_off) +
 * 尾 ftr(8B: size/free), 区域 16B 倍数; data = align_up(区域头+16, align),
 * 对齐垫片在头部之后, 任意请求对齐均由绝对地址保证。
 * VTCM 池基址由调用方给 (wtcache temp 区 carve — arena 不 acquire 任何
 * 硬件资源, 纯内存逻辑)。
 */
#ifndef HVXHMX_V23_ARENA_H
#define HVXHMX_V23_ARENA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ARENA_ALIGN_HVX 128u
#define ARENA_ALIGN_HMX 2048u

enum arena_loc { ARENA_DDR = 0, ARENA_VTCM = 1 };

struct arena {
    uint8_t*  base[2];
    uint32_t  cap[2];
    uint32_t  used[2];   /* 已分配请求字节 (16 对齐后) */
};

#define ARENA_OK        0
#define ARENA_ERR_PARAM -1

/* ddr/vtcm 可为 NULL (单池使用)。基址需 16B 对齐 (memalign 128/VTCM 天然满足)。 */
int   arena_init(struct arena* a, void* ddr, uint32_t ddr_bytes,
                 void* vtcm, uint32_t vtcm_bytes);
/* align=0 → 128; 必须为 2 的幂且 16..4096。返回对齐指针或 NULL(OOM)。 */
void* arena_alloc(struct arena* a, uint32_t bytes, uint32_t align, int loc);
void  arena_free(struct arena* a, void* p);
void  arena_reset(struct arena* a, int loc);
uint32_t arena_used(const struct arena* a, int loc);
uint32_t arena_largest_free(const struct arena* a, int loc);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_ARENA_H */
