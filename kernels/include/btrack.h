/* btrack.h — U21 写跟踪位图库 (GENERIC_B §3, 唯一事实来源)
 *
 * 三条写路径 (cpu/dma/peer) 全部挂钩; 位图按 blk_bytes 分块;
 * 缓冲级版本号 ver(buf) 供 B3 派生格式缓存作 key。
 *
 * 快照语义 (§3.3, 安全性地基): bt_snapshot_dirty 返回
 * "过去某时刻之后、当前时刻之前"的超集或等集 —— 允许把已清位重报脏
 * (安全, 只是慢), 绝不允许漏报 (致命, AX1 破坏)。
 *
 * DMA 铁律 (§3.4): 先标记后提交。bt_mark_dma_write 在描述符提交前
 * 保守置脏; bt_dma_complete 在引擎 done 后关闭记账。complete 丢失
 * (F-B4) 只造成脏位保守保留 + 挂起 token 统计, 正确性无损。
 *
 * 并发模式:
 *   BT_MODE_LOCK   全局自旋锁 (快照=stop-the-world, 最简单)
 *   BT_MODE_SHARD  64 分片锁 (快照=全分片加锁的 stop-the-world)
 *   BT_MODE_ATOMIC 无锁 test-and-set 位 + 原子版本; 快照=两遍读 OR
 *                  (位集单调性 ⇒ 并集 ⊇ 快照开始时刻的脏集, §6.1 F-B6)
 */
#ifndef HVXHMX_V23_BTRACK_H
#define HVXHMX_V23_BTRACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BT_MODE_LOCK   0
#define BT_MODE_SHARD  1
#define BT_MODE_ATOMIC 2

#define BT_MAX_BUFFERS    256u
#define BT_MAX_DMA_TOKENS 64u

typedef struct btrack_ctx btrack_ctx;

typedef struct bt_snapshot {
    uint64_t *words;      /* 位图字数组 (调用方用 bt_snapshot_free 释放) */
    uint32_t n_words;
} bt_snapshot;

btrack_ctx *bt_create(uint64_t mem_bytes, uint32_t blk_bytes,
                      int concurrency_mode);
void bt_destroy(btrack_ctx *bt);

/* ---- 缓冲登记 (粗粒度索引: 版本号按缓冲维护) ---- */
int bt_register_buffer(btrack_ctx *bt, uint64_t base, uint64_t size,
                       uint32_t *buf_id_out);
int bt_unregister_buffer(btrack_ctx *bt, uint32_t buf_id);

/* ---- 写钩子 (三条路径都必须挂, §1.4 失败模式) ---- */
void bt_mark_cpu_write (btrack_ctx *bt, uint64_t addr, uint64_t size);
int  bt_mark_dma_write (btrack_ctx *bt, uint64_t addr, uint64_t size,
                        uint64_t *token_out);
int  bt_dma_complete   (btrack_ctx *bt, uint64_t token);
void bt_mark_peer_write(btrack_ctx *bt, uint64_t addr, uint64_t size);

/* ---- 查询 ---- */
uint64_t bt_version(const btrack_ctx *bt, uint32_t buf_id);
int  bt_snapshot_dirty(btrack_ctx *bt, bt_snapshot *snap_out);
void bt_snapshot_free(bt_snapshot *snap);

/* ---- 清除 (仅允许对已快照过的位清除) ---- */
int  bt_clear(btrack_ctx *bt, const bt_snapshot *snap,
              uint64_t addr, uint64_t size);
int  bt_clear_all_flushed(btrack_ctx *bt, const bt_snapshot *snap);

/* ---- 合并 (外部脏源并入, 如硬件脏页位) ---- */
int  bt_merge(btrack_ctx *bt, const bt_snapshot *external);

/* ---- V2.3 扩展: 完整性置疑 (canary/审计) + redteam 钩子 ---- */
void bt_flag_suspect(btrack_ctx *bt);        /* 置疑 → B2 永久回退 FULL */
int  bt_is_suspect(const btrack_ctx *bt);
int  bt_dma_pending(const btrack_ctx *bt);   /* 挂起 token 数 (F-B4 审计) */
uint64_t bt_dirty_bytes(const btrack_ctx *bt);
uint32_t bt_ver_bits(const btrack_ctx *bt);
uint32_t bt_n_blocks(const btrack_ctx *bt);   /* bflush 区间构建用 */
uint32_t bt_blk_bytes(const btrack_ctx *bt);  /* bflush 成本/地址换算用 */
void bt_set_ver_bits(btrack_ctx *bt, uint32_t bits);  /* F-B5 回绕注入 */
int  bt_debug_bump_version(btrack_ctx *bt, uint32_t buf_id, uint32_t n);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_BTRACK_H */
