/* bflush.c — U21 定向 flush 决策器实现 (GENERIC_B §4)
 *
 * bf_boundary 流水: 快照 → DECIDE_GRAN 向上取整合并相邻粒 → 区间集 R →
 *   est = T_START·|R| + Σ bytes(r)/BLK × cost_per_blk;
 *   est·(1+MARGIN) < F_ALL → 定向执行; 否则 FULL;
 *   置疑/快照失败 → 强制 FULL (AX3)。执行后清已 flush 位。
 */
#include "bflush.h"

#include <stdlib.h>
#include <string.h>

#define BF_MAX_RANGES 8192u

struct bflush_ctx {
    btrack_ctx *bt;
    void (*flush_full)(void);
    void (*flush_range_inval)(uint64_t, uint64_t);
    uint64_t f_all, t_start;
    uint32_t decide_gran;
    uint64_t cost_per_blk;       /* ns / blk_bytes */
    uint32_t margin_pct;
    bflush_stats st;
};

bflush_ctx *bf_create(btrack_ctx *bt,
                      void (*flush_full)(void),
                      void (*flush_range_inval)(uint64_t, uint64_t),
                      uint64_t f_all_cost, uint64_t t_start_cost) {
    if (!bt || !flush_full || !flush_range_inval) return NULL;
    bflush_ctx *bf = calloc(1, sizeof *bf);
    if (!bf) return NULL;
    bf->bt = bt;
    bf->flush_full = flush_full;
    bf->flush_range_inval = flush_range_inval;
    bf->f_all = f_all_cost;
    bf->t_start = t_start_cost;
    bf->decide_gran = 4096u;
    bf->cost_per_blk = 20u;      /* 附录 A: 0.02 µs / 64B */
    bf->margin_pct = 10u;
    return bf;
}

void bf_destroy(bflush_ctx *bf) { free(bf); }

void bf_recalibrate(bflush_ctx *bf, uint64_t new_f_all) {
    if (bf) bf->f_all = new_f_all;
}

void bf_set_decide_gran(bflush_ctx *bf, uint32_t gran_bytes) {
    if (bf && gran_bytes) bf->decide_gran = gran_bytes;
}

void bf_set_blk_cost(bflush_ctx *bf, uint64_t ns_per_blk) {
    if (bf) bf->cost_per_blk = ns_per_blk;
}

void bf_set_margin_pct(bflush_ctx *bf, uint32_t pct) {
    if (bf) bf->margin_pct = pct;
}

void bf_get_stats(const bflush_ctx *bf, bflush_stats *out) {
    if (out && bf) *out = bf->st;
}

/* 快照 → 合并粒区间集。返回区间数, -1 = 超容量 (调用方转 FULL)。 */
static int build_ranges(const btrack_ctx *bt, const bt_snapshot *snap,
                        uint32_t gran, uint32_t blk_bytes,
                        uint64_t *ra, uint64_t *rl, uint32_t cap) {
    uint32_t n = 0;
    uint64_t run_start = 0;
    int in_run = 0;
    for (uint32_t w = 0; w < snap->n_words; w++) {
        uint64_t bits = snap->words[w];
        for (uint32_t bi = 0; bits; bi++, bits >>= 1) {
            if (!(bits & 1u)) continue;
            uint64_t blk = ((uint64_t)w << 6) + bi;
            if (blk >= bt_n_blocks(bt)) goto done;
            uint64_t g = ((blk * blk_bytes) / gran) * gran;  /* 向下取整粒 */
            if (in_run && g == run_start)
                continue;                        /* 同粒度内: 已覆盖 */
            if (in_run && n > 0 && g == run_start + rl[n - 1u]) {
                rl[n - 1u] += gran;              /* 相邻粒合并 */
                run_start = g;
                continue;
            }
            if (n >= cap) return -1;
            ra[n] = g; rl[n] = gran; n++;
            in_run = 1;
            run_start = g;
        }
    }
done:
    return (int)n;
}

int bf_boundary(bflush_ctx *bf, bflush_report *rep) {
    if (!bf || !rep) return -1;
    memset(rep, 0, sizeof *rep);
    bf->st.n_boundaries++;

    if (bt_is_suspect(bf->bt)) {                 /* V6: 置疑永久回退 */
        rep->used_full = 1;
        rep->reason = BF_REASON_SUSPECT;
        rep->act_cost = bf->f_all;
        bf->flush_full();
        bf->st.n_full++;
        bf->st.n_by_reason[BF_REASON_SUSPECT]++;
        return 0;
    }

    uint32_t blk_bytes = bt_blk_bytes(bf->bt);  /* 不许硬编码: 跟 bt 创建块宽 */
    if (!blk_bytes) blk_bytes = 64u;
    bt_snapshot snap;
    if (bt_snapshot_dirty(bf->bt, &snap) != 0) {
        rep->used_full = 1;
        rep->reason = BF_REASON_SNAP_FAIL;
        rep->act_cost = bf->f_all;
        bf->flush_full();
        bf->st.n_full++;
        bf->st.n_by_reason[BF_REASON_SNAP_FAIL]++;
        return 0;
    }

    uint64_t *ra = malloc(BF_MAX_RANGES * sizeof(uint64_t));
    uint64_t *rl = malloc(BF_MAX_RANGES * sizeof(uint64_t));
    int n = -1;
    if (ra && rl)
        n = build_ranges(bf->bt, &snap, bf->decide_gran, blk_bytes,
                         ra, rl, BF_MAX_RANGES);
    if (n < 0) {                                 /* 碎片超容量: 保守 FULL */
        rep->used_full = 1;
        rep->reason = BF_REASON_DIRTY_LARGE;
        rep->act_cost = bf->f_all;
        bf->flush_full();
        bf->st.n_full++;
        bf->st.n_by_reason[BF_REASON_DIRTY_LARGE]++;
    } else {
        uint64_t dirty = 0, est = 0;
        for (int i = 0; i < n; i++) {
            dirty += rl[i];
            est += bf->t_start
                 + (rl[i] / blk_bytes) * bf->cost_per_blk;
        }
        rep->n_ranges = (uint64_t)n;
        rep->dirty_bytes = dirty;
        rep->est_cost = est;
        if (est == 0 || est * (100ull + bf->margin_pct) / 100ull < bf->f_all) {
            for (int i = 0; i < n; i++)
                bf->flush_range_inval(ra[i], rl[i]);
            rep->act_cost = est;
            bf->st.n_directed++;
            if (bf->f_all > est) bf->st.saved_ns += bf->f_all - est;
        } else {
            rep->used_full = 1;
            rep->reason = BF_REASON_DIRTY_LARGE;
            rep->act_cost = bf->f_all;
            bf->flush_full();
            bf->st.n_full++;
            bf->st.n_by_reason[BF_REASON_DIRTY_LARGE]++;
        }
    }
    bt_clear_all_flushed(bf->bt, &snap);         /* AX1 收尾 */
    bt_snapshot_free(&snap);
    free(ra); free(rl);
    return 0;
}
