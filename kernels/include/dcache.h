/* dcache.h — U21 版本化派生格式缓存 (GENERIC_B §5)
 *
 * key = (buf_id, ver(buf), fmt); 纯函数 derived = convert(source, fmt)。
 * seqlock 形态 (写者不阻塞, 读者重试):
 *   读: v0=ver → 查 key → 命中则复核 ver 未变; 未命中则 拷贝源快照 →
 *       转换 (只读快照, 不读原地址) → 复核 ver==v0 → 原子发布 (AX5)。
 *       期间版本被写者推进 → 丢弃结果返回 -1 (调用方重试)。
 * 写: bt_mark_*_write 原子推进版本 → 旧 key 永远查不到 (条目尸体由
 *     替换策略回收); dc_invalidate 提供显式双保险。
 *
 * 版本回绕安全 (F-B5): key 比较用全宽 64 位 ver, 另按 bt 的窄位宽
 * 钩子统计 "掩码撞车但全宽不等" 事件 (wrap_misses) —— 碰撞时必然未命中。
 */
#ifndef HVXHMX_V23_DCACHE_H
#define HVXHMX_V23_DCACHE_H

#include <stdint.h>
#include "btrack.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*dc_convert_fn)(const uint8_t *src, uint64_t src_size,
                             uint8_t *dst, uint64_t dst_cap,
                             uint64_t *dst_size_out, void *user);

typedef struct {
    uint32_t fmt_id;                  /* 使用方编号 (非 0) */
    uint64_t dst_cap_per_src_byte;    /* 输出容量系数 ×src_size 上取整 */
    dc_convert_fn convert;
    void *user;
} dc_format;

typedef struct {
    uint64_t hits, misses, retries, evictions;
    uint64_t convert_ns_total, hit_ns_total;  /* 时钟源可选, 缺省 0 */
    uint64_t bytes_saved;                     /* 命中省下的转换字节数 */
    uint64_t wrap_misses;                     /* F-B5: 掩码撞车保守未命中 */
} dc_stats;

typedef struct dcache_ctx dcache_ctx;

dcache_ctx *dc_create(btrack_ctx *bt,
                      uint32_t max_entries,
                      uint32_t max_total_bytes,
                      int replacement);       /* 0=LRU 1=直接映射 */

int dc_register_format(dcache_ctx *dc, const dc_format *fmt);
void dc_destroy(dcache_ctx *dc);

/* 返回: 1=命中 (*out 指向缓存字节); 0=未命中已转换入缓存;
 *       -1=一致性失败 (转换期间源被写), 调用方必须重试。 */
int dc_get_or_convert(dcache_ctx *dc,
                      uint32_t buf_id,
                      const uint8_t *src, uint64_t src_size,
                      uint32_t fmt_id,
                      const uint8_t **out, uint64_t *out_size);

int dcache_invalidate(dcache_ctx *dc, uint32_t buf_id);  /* 显式失效 (双保险) */

void dc_get_stats(const dcache_ctx *dc, dc_stats *out);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_DCACHE_H */
