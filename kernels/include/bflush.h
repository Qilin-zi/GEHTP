/* bflush.h — U21 定向 flush 决策器 (GENERIC_B §4)
 *
 * 边界入口 bf_boundary: 快照 → 合并 (DECIDE_GRAN) → 成本估算 →
 *   C_directed·(1+MARGIN) < F_ALL → 定向计划 (逐区间 flush_range_inval)
 *   否则                            → FULL
 * 完整性置疑 (canary 触发) / 快照失败 → 强制 FULL (AX3 保守回退)。
 * 决策器只做性能选择不做正确性选择: 定向集 ⊇ dirty 快照 ⊇ 真脏集。
 *
 * 成本模型 (§4.1): C_directed = T_START·k + Σ cost(r_i);
 *   cost(r) = (r 字节数 / BLK_BYTES) × cost_per_blk_ns (线性表退化形式)。
 */
#ifndef HVXHMX_V23_BFLUSH_H
#define HVXHMX_V23_BFLUSH_H

#include <stdint.h>
#include "btrack.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 退化原因分类 (§4.4) */
#define BF_REASON_OK          0
#define BF_REASON_SUSPECT     1   /* 完整性置疑 → 永久 FULL */
#define BF_REASON_SNAP_FAIL   2
#define BF_REASON_DIRTY_LARGE 3   /* 脏量真大, 定向不划算 */

typedef struct {
    uint64_t t_start_ns;    /* 预留 (回填) */
    uint64_t n_ranges;
    int       used_full;
    uint64_t est_cost, act_cost;   /* ns, 模型口径 */
    uint64_t dirty_bytes;
    int       reason;
} bflush_report;

typedef struct {
    uint64_t n_boundaries, n_full, n_directed;
    uint64_t saved_ns;              /* Σ(F_ALL − act) 定向边界累计 */
    uint64_t n_by_reason[4];
} bflush_stats;

typedef struct bflush_ctx bflush_ctx;

bflush_ctx *bf_create(btrack_ctx *bt,
                      void (*flush_full)(void),
                      void (*flush_range_inval)(uint64_t, uint64_t),
                      uint64_t f_all_cost,     /* ns */
                      uint64_t t_start_cost);  /* ns */

int  bf_boundary(bflush_ctx *bf, bflush_report *rep);
void bf_destroy(bflush_ctx *bf);
void bf_recalibrate(bflush_ctx *bf, uint64_t new_f_all);

/* 参数 (附录 A 默认: gran=4KiB, cost/64B=20ns, margin=10%) */
void bf_set_decide_gran(bflush_ctx *bf, uint32_t gran_bytes);
void bf_set_blk_cost   (bflush_ctx *bf, uint64_t ns_per_blk); /* F-B9 扰动 */
void bf_set_margin_pct (bflush_ctx *bf, uint32_t pct);

void bf_get_stats(const bflush_ctx *bf, bflush_stats *out);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_BFLUSH_H */
