/*
 * 35_btrack — U21 写跟踪 + 定向 flush + 派生格式缓存 设备验证
 *            (GENERIC_B 问题书)
 * =====================================================================
 * 门 (问题书 §7 验收 + §8.3 金丝雀 + §6.1 故障目录):
 *   G1  单写单边界      定向计划 = 该区间 (1 粒度区间, 非 FULL)
 *   G2  写同块 100 次   脏位仍 1, 版本 +100, 只 flush 一个粒度
 *   G3  DMA 三态        先标后写 AX1 成立; F-B4 回调丢失 → 保守脏位保留
 *                      + 挂起 token 审计可见
 *   G4  转换期间写      dc_get_or_convert 返回 -1, 重试后新版本命中
 *   G5  版本回绕        VER_BITS=8 掩码撞车 → 保守未命中 (F-B5), 零错命中
 *   G6  空边界          零 flush 调用, 成本 ≈ 快照开销
 *   G7  sparse_13b      13 边界定向, 总成本 ≤ 全量方案 30% (V2)
 *   G8  dense_13b       脏 90% → 正确选 FULL (V2 反向)
 *   G9  reuse3/noreuse  复用 3× 命中 ≥50%; 无复用 ≤5% (V3)
 *   G10 F-B1 影子审计   绕钩子写被检出 → 置疑 → 永久 FULL (V6)
 *   G11 F-B2 标后未清   脏位保留 → 重复 flush, 正确性无损
 *   G12 F-B6 双写者     交织 mark + 快照超集性质 (ATOMIC/SHARD 双模式)
 *   G13 F-B7 缓存不扰B2 开/关 dcache 的 dirty 快照逐位一致
 *   G14 F-B8/F-B9       逐出竞态命中完整性 + 成本 ±50% 扰动正确性
 *   G15 统计对账        bf/dc 计数器与事件回放一致 (V7)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define SPACE_BYTES (8u * 1024u * 1024u)
#define BUF_BYTES   (256u * 1024u)
#define N_BUFS      32u
#define BLK         64u
#define GRAN        4096u
#define F_ALL_NS    50000ull
#define T_START_NS  1000ull

static uint8_t space[SPACE_BYTES];     /* 被跟踪地址空间 */
static uint8_t refcp[SPACE_BYTES];     /* 影子审计参考副本 (eager) */

/* flush 回调日志 (AX1 覆盖校验用) */
static struct {
    uint64_t a[512]; uint64_t l[512];
    int n, full;
} flog;
static void f_full(void) { flog.full++; }
static void f_range(uint64_t a, uint64_t s) {
    if (flog.n < 512) { flog.a[flog.n] = a; flog.l[flog.n] = s; }
    flog.n++;
}
static void flog_reset(void) { flog.n = 0; flog.full = 0; }
static int flog_covers(uint64_t addr, uint64_t size) {
    for (uint64_t off = 0; off < size; off += BLK) {
        uint64_t b = addr + off;
        int hit = 0;
        for (int i = 0; i < flog.n && i < 512; i++)
            if (b >= flog.a[i] && b + BLK <= flog.a[i] + flog.l[i]) { hit = 1; break; }
        if (!hit) return 0;
    }
    return 1;
}
static int snap_has(const bt_snapshot *s, uint64_t addr, uint64_t size) {
    for (uint64_t off = 0; off < size; off += BLK) {
        uint64_t blk = (addr + off) / BLK;
        if (!((s->words[blk >> 6] >> (blk & 63u)) & 1ull)) return 0;
    }
    return 1;
}

static uint32_t reg_buf(btrack_ctx *bt, uint32_t i) {
    uint32_t id = 0xffffffffu;
    bt_register_buffer(bt, (uint64_t)i * BUF_BYTES, BUF_BYTES, &id);
    return id;
}

/* ---- 转换函数: 纯函数 + 可选"转换期间写"注入 (G4) ---- */
struct cvt_ctx { btrack_ctx *bt; uint64_t watch; int mid_write; };
static struct cvt_ctx g_cvt;

static int cvt_xor(const uint8_t *src, uint64_t n, uint8_t *dst,
                   uint64_t cap, uint64_t *outn, void *user) {
    struct cvt_ctx *c = user;
    if (c && c->mid_write) {                 /* 写者在转换进行时推进版本 */
        bt_mark_cpu_write(c->bt, c->watch, BLK);
        c->mid_write = 0;
    }
    if (cap < n) return -1;
    for (uint64_t i = 0; i < n; i++)
        dst[i] = (uint8_t)(src[i] ^ 0xA5u ^ (uint8_t)(i & 15u));
    *outn = n;
    return 0;
}
static int cvt_rev(const uint8_t *src, uint64_t n, uint8_t *dst,
                   uint64_t cap, uint64_t *outn, void *user) {
    (void)user;
    if (cap < n) return -1;
    for (uint64_t i = 0; i < n; i++) dst[i] = src[n - 1u - i];
    *outn = n;
    return 0;
}
static void expect_xor(const uint8_t *src, uint64_t n, uint8_t *dst) {
    for (uint64_t i = 0; i < n; i++)
        dst[i] = (uint8_t)(src[i] ^ 0xA5u ^ (uint8_t)(i & 15u));
}

int main(void)
{
    ex_open_result("35_btrack");

    /* ---- G1 单写单边界 ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        reg_buf(bt, 0);
        flog_reset();
        bt_mark_cpu_write(bt, 0u * BUF_BYTES + GRAN, BLK);
        bflush_report rep;
        bf_boundary(bf, &rep);
        ex_check("G1 定向非 FULL", rep.used_full != 0, 0);
        ex_check("G1 单区间", (int)rep.n_ranges != 1, 0);
        ex_check("G1 区间=写粒", !(flog.n == 1
                 && flog.a[0] == 0u * BUF_BYTES + GRAN
                 && flog.l[0] == GRAN), 0);
        ex_check("G1 AX1 覆盖", !flog_covers(0u * BUF_BYTES + GRAN, BLK), 0);
        ex_check("G1 脏已清", bt_dirty_bytes(bt) != 0, 0);
        bf_destroy(bf); bt_destroy(bt);
    }

    /* ---- G2 写同块 100 次 ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        uint32_t b0 = reg_buf(bt, 0);
        for (int i = 0; i < 100; i++)
            bt_mark_cpu_write(bt, 0u * BUF_BYTES, BLK);
        ex_check("G2 脏位仍 1 (64B)", bt_dirty_bytes(bt) != BLK, 0);
        ex_check("G2 版本 +100", bt_version(bt, b0) != 100, 0);
        bt_destroy(bt);
    }

    /* ---- G3 DMA 三态: 先标后写 / 回调丢失 ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        reg_buf(bt, 1);
        uint64_t tok = 0;
        flog_reset();
        ex_check("G3 mark_dma rc",
                 bt_mark_dma_write(bt, 1u * BUF_BYTES, 2u * GRAN, &tok) != 0, 0);
        /* (引擎搬运后) done 回调 */
        ex_check("G3 complete rc", bt_dma_complete(bt, tok) != 0, 0);
        bflush_report rep;
        bf_boundary(bf, &rep);
        ex_check("G3 先标后写 AX1", !flog_covers(1u * BUF_BYTES, 2u * GRAN), 0);

        /* F-B4: 回调丢失 → 位保守保留, token 挂起可审计 */
        uint64_t tok2 = 0;
        bt_mark_dma_write(bt, 1u * BUF_BYTES + 8u * GRAN, GRAN, &tok2);
        flog_reset();
        bf_boundary(bf, &rep);
        ex_check("G3 F-B4 丢失仍覆盖", !flog_covers(1u * BUF_BYTES + 8u * GRAN,
                                                     GRAN), 0);
        ex_check("G3 挂起 token=1", bt_dma_pending(bt) != 1, 0);
        bt_dma_complete(bt, tok2);
        ex_check("G3 补 complete 归零", bt_dma_pending(bt) != 0, 0);
        bf_destroy(bf); bt_destroy(bt);
    }

    /* ---- G4 转换期间写 → -1 → 重试命中 ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        dcache_ctx *dc = dc_create(bt, 16, 4u << 20, 0);
        uint32_t b0 = reg_buf(bt, 2);
        uint8_t *src = space + 2u * BUF_BYTES;
        ex_fill_u8(src, 16u * 1024u, 7, 251);
        g_cvt.bt = bt; g_cvt.watch = 2u * BUF_BYTES; g_cvt.mid_write = 1;
        dc_format f1 = { 1, 1, cvt_xor, &g_cvt };
        dc_register_format(dc, &f1);
        const uint8_t *out = NULL; uint64_t osz = 0;
        int rc1 = dc_get_or_convert(dc, b0, src, 16u * 1024u, 1, &out, &osz);
        ex_check("G4 期间写返回 -1", rc1 != -1, 0);
        int rc2 = dc_get_or_convert(dc, b0, src, 16u * 1024u, 1, &out, &osz);
        ex_check("G4 重试转换 rc=0", rc2 != 0, 0);
        uint8_t exp[16u * 1024u];
        expect_xor(src, 16u * 1024u, exp);
        ex_check("G4 内容正确", memcmp(out, exp, osz) != 0 || osz != 16u*1024u, 0);
        int rc3 = dc_get_or_convert(dc, b0, src, 16u * 1024u, 1, &out, &osz);
        ex_check("G4 二次命中 rc=1", rc3 != 1, 0);
        ex_check("G4 命中内容正确", memcmp(out, exp, osz) != 0, 0);
        dc_stats st; dc_get_stats(dc, &st);
        ex_check("G4 retries≥1", !(st.retries >= 1), 0);
        dc_destroy(dc); bt_destroy(bt);
    }

    /* ---- G5 版本回绕 (F-B5): 掩码撞车必须未命中 ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        dcache_ctx *dc = dc_create(bt, 16, 4u << 20, 0);
        uint32_t b0 = reg_buf(bt, 3);
        uint8_t *src = space + 3u * BUF_BYTES;
        ex_fill_u8(src, 8u * 1024u, 11, 251);          /* ver=0 时的源 */
        g_cvt.bt = bt; g_cvt.watch = 0; g_cvt.mid_write = 0;
        dc_format f1 = { 1, 1, cvt_xor, &g_cvt };
        dc_register_format(dc, &f1);
        const uint8_t *out = NULL; uint64_t osz = 0;
        int rc1 = dc_get_or_convert(dc, b0, src, 8u * 1024u, 1, &out, &osz);
        ex_check("G5 v0 入缓存 rc=0", rc1 != 0, 0);
        ex_fill_u8(src, 8u * 1024u, 12, 251);          /* 新源内容 */
        bt_set_ver_bits(bt, 8);
        bt_debug_bump_version(bt, b0, 256);            /* ver: 0 → 256 (掩码=0) */
        uint64_t v = bt_version(bt, b0);
        ex_check("G5 ver=256", v != 256, 0);
        int rc2 = dc_get_or_convert(dc, b0, src, 8u * 1024u, 1, &out, &osz);
        ex_check("G5 撞车未命中 rc=0 (非 1)", rc2 == 1, 0);
        uint8_t exp[8u * 1024u];
        expect_xor(src, 8u * 1024u, exp);
        ex_check("G5 零错命中 (内容=新源)", memcmp(out, exp, osz) != 0, 0);
        dc_stats st; dc_get_stats(dc, &st);
        ex_check("G5 wrap_misses≥1", !(st.wrap_misses >= 1), 0);
        dc_destroy(dc); bt_destroy(bt);
    }

    /* ---- G6 空边界 ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        bflush_report rep;
        flog_reset();
        bf_boundary(bf, &rep);
        ex_check("G6 非 FULL", rep.used_full != 0, 0);
        ex_check("G6 零区间零调用", flog.n != 0 || flog.full != 0, 0);
        ex_check("G6 成本≈0", rep.act_cost != 0, 0);
        bf_destroy(bf); bt_destroy(bt);
    }

    /* ---- G7 sparse_13b: 13 边界定向, ≤30% 全量成本 (V2) ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_ATOMIC);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        for (uint32_t i = 0; i < N_BUFS; i++) reg_buf(bt, i);
        uint64_t total_act = 0;
        int all_directed = 1, covered = 1;
        for (int b = 0; b < 13; b++) {
            flog_reset();
            uint64_t wbase[4];
            for (int k = 0; k < 4; k++) {
                uint32_t buf = (uint32_t)(b * 4 + k) % N_BUFS;
                wbase[k] = (uint64_t)buf * BUF_BYTES;
                if (k % 2 == 0) bt_mark_cpu_write(bt, wbase[k], 2u * GRAN);
                else if (k == 1) {
                    uint64_t t; bt_mark_dma_write(bt, wbase[k], 2u * GRAN, &t);
                    bt_dma_complete(bt, t);
                } else bt_mark_peer_write(bt, wbase[k], 2u * GRAN);
            }
            bflush_report rep;
            bf_boundary(bf, &rep);
            if (rep.used_full) all_directed = 0;
            total_act += rep.act_cost;
            for (int k = 0; k < 4; k++)
                if (!flog_covers(wbase[k], 2u * GRAN)) covered = 0;
        }
        uint64_t full_scheme = 13ull * F_ALL_NS;
        ex_log("  sparse_13b: directed=%llu ns vs full=%llu ns (%.1f%%)",
               (unsigned long long)total_act, (unsigned long long)full_scheme,
               100.0 * (double)total_act / (double)full_scheme);
        ex_check("G7 全部定向", all_directed != 1, 0);
        ex_check("G7 成本 ≤30% 全量", !(total_act <= full_scheme / 100ull * 30ull), 0);
        ex_check("G7 AX1 覆盖 (13×4 区间)", covered != 1, 0);
        bf_destroy(bf); bt_destroy(bt);
    }

    /* ---- G8 dense_13b: 脏 90% → 必须选 FULL ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        int all_full = 1;
        for (int b = 0; b < 13; b++) {
            bt_mark_cpu_write(bt, 0, SPACE_BYTES / 100u * 90u);
            flog_reset();
            bflush_report rep;
            bf_boundary(bf, &rep);
            if (!rep.used_full || rep.reason != BF_REASON_DIRTY_LARGE) all_full = 0;
        }
        ex_check("G8 脏 90% 全选 FULL", all_full != 1, 0);
        bf_destroy(bf); bt_destroy(bt);
    }

    /* ---- G9 reuse3 / noreuse 命中率 (V3) ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        dcache_ctx *dc = dc_create(bt, 64, 16u << 20, 0);
        g_cvt.bt = bt; g_cvt.watch = 0; g_cvt.mid_write = 0;
        dc_format f1 = { 1, 1, cvt_xor, &g_cvt };
        dc_format f2 = { 2, 1, cvt_rev, NULL };
        dc_register_format(dc, &f1);
        dc_register_format(dc, &f2);
        for (uint32_t i = 0; i < 8; i++) reg_buf(bt, 4u + i);
        for (uint32_t i = 0; i < 8; i++) reg_buf(bt, 12u + i);
        /* reuse3: 每缓冲 3 消费者 × 2 格式 → 4 命中 / 6 请求 */
        const uint8_t *out = NULL; uint64_t osz = 0;
        int bad = 0;
        for (uint32_t i = 0; i < 8; i++) {
            uint8_t *src = space + (4u + i) * BUF_BYTES;
            ex_fill_u8(src, 32u * 1024u, (int)(20 + i), 251);
            for (int c = 0; c < 3; c++) {
                int r1 = dc_get_or_convert(dc, 4u + i, src, 32u * 1024u, 1,
                                           &out, &osz);
                int r2 = dc_get_or_convert(dc, 4u + i, src, 32u * 1024u, 2,
                                           &out, &osz);
                if (c > 0 && r1 != 1) bad++;
                if (c > 0 && r2 != 1) bad++;
                if (c == 0 && (r1 != 0 || r2 != 0)) bad++;
            }
        }
        dc_stats st; dc_get_stats(dc, &st);
        double hr = (double)st.hits / (double)(st.hits + st.misses);
        ex_log("  reuse3: hits=%llu misses=%llu hr=%.1f%%",
               (unsigned long long)st.hits, (unsigned long long)st.misses,
               100.0 * hr);
        ex_check("G9 reuse3 复用门全按预期", bad != 0, 0);
        ex_check("G9 reuse3 命中 ≥50%", !(hr >= 0.5), 0);
        /* noreuse: 每缓冲单消费 → 零命中 */
        for (uint32_t i = 0; i < 8; i++) {
            uint8_t *src = space + (12u + i) * BUF_BYTES;
            ex_fill_u8(src, 32u * 1024u, (int)(40 + i), 251);
            dc_get_or_convert(dc, 12u + i, src, 32u * 1024u, 1, &out, &osz);
        }
        dc_stats st2; dc_get_stats(dc, &st2);
        double hr2 = (double)(st2.hits - st.hits)
                   / (double)(st2.hits - st.hits + st2.misses - st.misses);
        ex_log("  noreuse: hr=%.1f%%", 100.0 * hr2);
        ex_check("G9 noreuse 命中 ≤5%", !(hr2 <= 0.05), 0);
        dc_destroy(dc); bt_destroy(bt);
    }

    /* ---- G10 F-B1 影子审计 + 置疑永久 FULL (V6) ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        reg_buf(bt, 30);
        uint64_t wa = 30u * BUF_BYTES;
        memcpy(refcp, space, SPACE_BYTES);       /* 影子基线重同步 */
        /* 合法写: 钩子 + 影子同步 */
        ex_fill_u8(space + wa, GRAN, 60, 251);
        memcpy(refcp + wa, space + wa, GRAN);
        bt_mark_cpu_write(bt, wa, GRAN);
        bflush_report rep;
        bf_boundary(bf, &rep);
        ex_check("G10 审计前影子一致", memcmp(space, refcp, SPACE_BYTES) != 0, 0);
        /* F-B1: 绕钩子写 (只写本体, 不打钩子, 不同步影子) */
        ex_fill_u8(space + wa + 4u * GRAN, BLK, 61, 251);
        int detected = memcmp(space, refcp, SPACE_BYTES) != 0;
        ex_check("G10 F-B1 被影子审计检出", detected != 1, 0);
        bt_flag_suspect(bt);
        flog_reset();
        bf_boundary(bf, &rep);
        ex_check("G10 置疑后 FULL", rep.used_full != 1, 0);
        ex_check("G10 原因=SUSPECT", rep.reason != BF_REASON_SUSPECT, 0);
        memcpy(refcp, space, SPACE_BYTES);          /* 观察者重同步 */
        flog_reset();
        bf_boundary(bf, &rep);
        ex_check("G10 置疑永久 (二次仍 FULL)", rep.used_full != 1, 0);
        ex_check("G10 重同步后零损坏", memcmp(space, refcp, SPACE_BYTES) != 0, 0);
        bf_destroy(bf); bt_destroy(bt);
    }

    /* ---- G11 F-B2 标后未清: 脏位保留 → 重复 flush ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        reg_buf(bt, 6);
        bt_mark_cpu_write(bt, 6u * BUF_BYTES, GRAN);
        bt_snapshot snap;
        bt_snapshot_dirty(bt, &snap);
        flog_reset();
        f_range(6u * BUF_BYTES, GRAN);              /* 手动 flush */
        /* 故障: 丢弃 bt_clear (标后未清) */
        ex_check("G11 脏位保留", bt_dirty_bytes(bt) == 0, 0);
        bflush_report rep;
        bf_boundary(bf, &rep);
        ex_check("G11 下一边界重复 flush", !(rep.used_full == 0 && rep.n_ranges >= 1), 0);
        ex_check("G11 收尾脏清零", bt_dirty_bytes(bt) != 0, 0);
        bt_snapshot_free(&snap);
        bf_destroy(bf); bt_destroy(bt);
    }

    /* ---- G12 F-B6 双写者交织: 快照超集 (ATOMIC + SHARD) ---- */
    {
        const int modes[] = { BT_MODE_ATOMIC, BT_MODE_SHARD };
        int ok = 1;
        for (int m = 0; m < 2; m++) {
            btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, modes[m]);
            uint64_t ra = 8u * BUF_BYTES, rb = 9u * BUF_BYTES;
            bt_mark_cpu_write(bt, ra, 4u * GRAN);          /* 写者 A */
            bt_snapshot s1;
            bt_snapshot_dirty(bt, &s1);
            if (!snap_has(&s1, ra, 4u * GRAN)) ok = 0;     /* ⊇ A */
            bt_mark_peer_write(bt, rb, 4u * GRAN);         /* 写者 B */
            bt_snapshot s2;
            bt_snapshot_dirty(bt, &s2);
            if (!snap_has(&s2, ra, 4u * GRAN)) ok = 0;     /* ⊇ A∪B */
            if (!snap_has(&s2, rb, 4u * GRAN)) ok = 0;
            for (uint32_t w = 0; w < s1.n_words; w++)      /* ⊇ s1 (单调) */
                if ((s2.words[w] & s1.words[w]) != s1.words[w]) ok = 0;
            bt_snapshot_free(&s1); bt_snapshot_free(&s2);
            bt_destroy(bt);
        }
        ex_check("G12 快照超集 (ATOMIC+SHARD)", ok != 1, 0);
    }

    /* ---- G12b 稀疏位图负例: 未写邻块必须判无(优先级 bug 回归门) ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        uint64_t hi = 10u * BUF_BYTES + BLK;
        bt_mark_cpu_write(bt, hi, BLK);      /* 只写第二块, 首块保持干净 */
        bt_snapshot s;
        bt_snapshot_dirty(bt, &s);
        ex_check("G12b 稀疏正例: 已写块在位", !snap_has(&s, hi, BLK), 0);
        ex_check("G12b 稀疏负例: 未写邻块判无",
                 snap_has(&s, 10u * BUF_BYTES, BLK) != 0, 0);
        bt_snapshot_free(&s);
        bt_destroy(bt);
    }

    /* ---- G13 F-B7: 开/关 dcache 的 dirty 快照逐位一致 ---- */
    {
        btrack_ctx *bt1 = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        btrack_ctx *bt2 = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        dcache_ctx *dc = dc_create(bt1, 8, 2u << 20, 0);
        g_cvt.bt = bt1; g_cvt.watch = 0; g_cvt.mid_write = 0;
        dc_format f1 = { 1, 1, cvt_xor, &g_cvt };
        dc_register_format(dc, &f1);
        uint32_t id; bt_register_buffer(bt1, 20u * BUF_BYTES, BUF_BYTES, &id);
        bt_register_buffer(bt2, 20u * BUF_BYTES, BUF_BYTES, &id);
        const uint8_t *out = NULL; uint64_t osz = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t a = (20u + (uint64_t)i % 3u) * BUF_BYTES + (uint64_t)i * GRAN;
            bt_mark_cpu_write(bt1, a, GRAN);
            bt_mark_cpu_write(bt2, a, GRAN);
            dc_get_or_convert(dc, id, space + 20u * BUF_BYTES, 8u * 1024u, 1,
                              &out, &osz);            /* 转换不产生脏块 */
        }
        bt_snapshot s1, s2;
        bt_snapshot_dirty(bt1, &s1);
        bt_snapshot_dirty(bt2, &s2);
        int same = s1.n_words == s2.n_words
                && memcmp(s1.words, s2.words, (size_t)s1.n_words * 8u) == 0;
        ex_check("G13 缓存不扰 B2 (逐位一致)", same != 1, 0);
        bt_snapshot_free(&s1); bt_snapshot_free(&s2);
        dc_destroy(dc); bt_destroy(bt1); bt_destroy(bt2);
    }

    /* ---- G14 F-B8 逐出完整性 + F-B9 成本扰动 ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        dcache_ctx *dc = dc_create(bt, 4, 2u << 20, 0);   /* 4 槽强制逐出 */
        g_cvt.bt = bt; g_cvt.watch = 0; g_cvt.mid_write = 0;
        dc_format f1 = { 1, 1, cvt_xor, &g_cvt };
        dc_register_format(dc, &f1);
        const uint8_t *out = NULL; uint64_t osz = 0;
        uint8_t exp[16u * 1024u];
        int bad = 0;
        for (uint32_t i = 0; i < 8; i++) reg_buf(bt, 16u + i);
        for (uint32_t i = 0; i < 8; i++) {                /* 8 入 4 槽 */
            uint8_t *src = space + (16u + i) * BUF_BYTES;
            ex_fill_u8(src, 16u * 1024u, (int)(80 + i), 251);
            expect_xor(src, 16u * 1024u, exp);
            int rc = dc_get_or_convert(dc, 16u + i, src, 16u * 1024u, 1,
                                       &out, &osz);
            if (rc != 0 || memcmp(out, exp, osz)) bad++;
        }
        dc_stats st; dc_get_stats(dc, &st);
        ex_check("G14 逐出发生 (≥4)", !(st.evictions >= 4), 0);
        /* F-B8: 幸存条目 (最新 4 个) 命中必须完整 */
        {
            uint8_t *src = space + 23u * BUF_BYTES;       /* 最新插入者 */
            expect_xor(src, 16u * 1024u, exp);
            int rc = dc_get_or_convert(dc, 23u, src, 16u * 1024u, 1, &out, &osz);
            if (rc != 1 || memcmp(out, exp, osz)) bad++;
        }
        for (uint32_t i = 0; i < 8; i++) {
            uint8_t *src = space + (16u + i) * BUF_BYTES;
            expect_xor(src, 16u * 1024u, exp);
            int rc = dc_get_or_convert(dc, 16u + i, src, 16u * 1024u, 1,
                                       &out, &osz);
            if (rc == 1 && memcmp(out, exp, osz)) bad++;  /* 命中半成品 = FAIL */
            if (rc == 0 && memcmp(out, exp, osz)) bad++;
        }
        ex_check("G14 命中/未命中内容完整", bad != 0, 0);

        /* F-B9: cost 表 +50% 扰动 → 只影响选择, AX1 不破 */
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        bf_set_blk_cost(bf, 30u);                          /* ±50% 扰动 */
        uint64_t wa = 24u * BUF_BYTES;
        bt_mark_cpu_write(bt, wa, 2u * GRAN);
        flog_reset();
        bflush_report rep;
        bf_boundary(bf, &rep);
        int ax1 = rep.used_full == 1 ? flog.full >= 1
                                     : flog_covers(wa, 2u * GRAN);
        ex_check("G14 F-B9 扰动后 AX1 仍立", ax1 != 1, 0);
        flog_reset();
        bf_boundary(bf, &rep);
        ex_check("G14 扰动后脏已清 (空边界)", flog.n != 0 || flog.full != 0, 0);
        bf_destroy(bf); dc_destroy(dc); bt_destroy(bt);
    }

    /* ---- G15 统计对账 (V7) ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, BLK, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        dcache_ctx *dc = dc_create(bt, 8, 2u << 20, 0);
        g_cvt.bt = bt; g_cvt.watch = 0; g_cvt.mid_write = 0;
        dc_format f1 = { 1, 1, cvt_xor, &g_cvt };
        dc_register_format(dc, &f1);
        uint32_t id; bt_register_buffer(bt, 28u * BUF_BYTES, BUF_BYTES, &id);
        const uint8_t *out = NULL; uint64_t osz = 0;
        int n_convert = 0, n_hit = 0, n_bound = 0;
        bflush_report rep;
        bf_boundary(bf, &rep); n_bound++;                       /* 空 */
        ex_fill_u8(space + 28u * BUF_BYTES, 8u * 1024u, 90, 251);
        bt_mark_cpu_write(bt, 28u * BUF_BYTES, 8u * 1024u);
        bf_boundary(bf, &rep); n_bound++;                       /* 定向 */
        for (int i = 0; i < 3; i++) {
            int rc = dc_get_or_convert(dc, id, space + 28u * BUF_BYTES,
                                       8u * 1024u, 1, &out, &osz);
            n_convert++;
            if (rc == 1) n_hit++;
        }
        bt_mark_cpu_write(bt, 28u * BUF_BYTES + 64u * GRAN, GRAN);
        bf_boundary(bf, &rep); n_bound++;                       /* 定向 */
        bflush_stats bs; bf_get_stats(bf, &bs);
        dc_stats ds; dc_get_stats(dc, &ds);
        ex_check("G15 边界数对账", bs.n_boundaries != (uint64_t)n_bound, 0);
        ex_check("G15 全+定=边界", bs.n_full + bs.n_directed != bs.n_boundaries, 0);
        ex_check("G15 转换数对账", ds.hits + ds.misses != (uint64_t)n_convert, 0);
        ex_check("G15 命中数对账", ds.hits != (uint64_t)n_hit, 0);
        ex_check("G15 省额为正", !(bs.saved_ns > 0), 0);
        bf_destroy(bf); dc_destroy(dc); bt_destroy(bt);
    }

    /* ---- G16 非 64 块宽: bflush 不得硬编码 blk_bytes ---- */
    {
        btrack_ctx *bt = bt_create(SPACE_BYTES, 128u, BT_MODE_LOCK);
        bflush_ctx *bf = bf_create(bt, f_full, f_range, F_ALL_NS, T_START_NS);
        flog_reset();
        bt_mark_cpu_write(bt, 8192u, 128u);
        bflush_report rep;
        bf_boundary(bf, &rep);
        ex_check("G16 块宽128 定向非 FULL", rep.used_full != 0, 0);
        ex_check("G16 单区间", (int)rep.n_ranges != 1, 0);
        ex_check("G16 区间地址=gran对齐8192",
                 !(flog.n == 1 && flog.a[0] == 8192u && flog.l[0] == GRAN), 0);
        ex_check("G16 AX1 覆盖", !flog_covers(8192u, 128u), 0);
        ex_check("G16 脏已清", bt_dirty_bytes(bt) != 0, 0);
        btrack_ctx *bt2 = bt_create(SPACE_BYTES, 128u, BT_MODE_LOCK);
        bt_mark_cpu_write(bt2, 8192u + 64u, 64u);
        ex_check("G16 半块按整块计脏", bt_dirty_bytes(bt2) != 128u, 0);
        bf_destroy(bf);
        bt_destroy(bt);
        bt_destroy(bt2);
    }

    return ex_summary();
}
