/*
 * 33_bledger — U19 Buffer Ledger 数据流溯源审计 设备验证
 * =====================================================================
 * 依据 docs/P1-uninitialized-read-dataflow.md 测试模板 (§12):
 *   T1 冷启动消费      断链消费即时报错 (3.1/3.2); 写入收敛修复后通过
 *   T2 行号约定矩阵    行错位 verify=NEVER + E2 rank 反查定位到错位行 (3.3)
 *   T3 canary 巡检     canary 后读 → CANARY + 数据=0xAA 模式 + 全行深 rank (3.1)
 *   T4 归还后禁读      release 后消费 → RELEASED + 归还前 canary 位样 (3.4)
 *   T5 新路径覆盖      新路径消费必须先有写入 → NEVER (3.2)
 *   T6 量化对账        qtag 写读不一致 → QTAG; 单一配置源后通过 (3.5)
 *   T7 双写检测        未读行被异 writer 覆写计数; 读后覆写合法 (3.6)
 *   T8 端到端听诊      模拟 P1 §5 draft 断链: d1 垃圾/d2 正确 → 每轮 break +
 *                      接受率崩塌 (hist 全 1); 修复引擎 breaks=0 + 回归锚
 *                      argmax(首树提议)==argmax(prefill 末行) + hist 全 3
 *   L1 timeline        文本时序图 (P1 §4.4) 非空且含统计
 *
 * 模拟管线: logits buffer [rows][VOC] f32; 写入者 W_PREFILL/W_TREE/W_DECODE/W_LATE;
 * 生成分布 = 确定性 hash 噪声 + 大峰 (argmax/rank 均确定, f32 同序可判)。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define VOC   128u        /* 词表 (f32/行) */
#define NROWS 8u          /* logits 行数 = 深度槽 */
#define NR    8u          /* T8 轮数 */

#define W_PREFILL 1u
#define W_TREE    2u
#define W_DECODE  3u
#define W_LATE    4u
#define Q_F32     1u
#define Q_UF16A   2u
#define Q_UF16B   3u

static float hnorm(uint32_t v, uint32_t seed) {   /* 确定性 [0,1) hash */
    uint32_t x = v * 2654435761u ^ (seed * 97u * 2654435761u);
    x ^= x >> 13; x *= 1274126177u; x ^= x >> 16;
    return (float)(x >> 8) / 16777216.0f;
}

static void gen_dist(float* row, uint32_t peak) { /* 噪声 + 大峰, argmax=peak */
    for (uint32_t v = 0; v < VOC; v++) row[v] = 0.001f * hnorm(v, peak);
    row[0] = -0.5f;                /* token 0 = 固定低频槽: rank 判别确定性化 */
    row[peak % VOC] += 10.0f;
}

static uint32_t argmax_row(const float* row) {
    uint32_t best = 0;
    for (uint32_t v = 1; v < VOC; v++) if (row[v] > row[best]) best = v;
    return best;
}

static uint32_t rank_of(const float* row, uint32_t tok) {  /* 0 = 头名 */
    uint32_t r = 0;
    for (uint32_t v = 0; v < VOC; v++) if (row[v] > row[tok]) r++;
    return r;
}

static int bits_are_canary(const float* row, uint32_t n) {  /* 0xAA 位样逐字 */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t w; memcpy(&w, row + i, 4);
        if (w != 0xAAAAAAAAu) return 0;
    }
    return 1;
}

int main(void) {
    ex_open_result("33_bledger");

    /* ---- G1: 模块契约 (参数域/状态机/统计) ---- */
    {
        struct bledger bl; float mem[NROWS * VOC];
        memset(mem, 0, sizeof mem);
        bl_init(&bl);
        int bad = (bl_register(&bl, 0, NULL, sizeof mem, VOC * 4u, "x") != BL_ERR_PARAM);
        bad += (bl_register(&bl, 0, mem, sizeof mem, 0, "x") != BL_ERR_PARAM);
        bad += (bl_register(&bl, 0, mem, sizeof mem + 4, VOC * 4u, "x") != BL_ERR_PARAM);
        bad += (bl_register(&bl, BL_MAX_BUFS, mem, sizeof mem, VOC * 4u, "x") != BL_ERR_RANGE);
        bad += (bl_register(&bl, 0, mem, sizeof mem, VOC * 4u, "lg") != BL_OK);
        bad += (bl_register(&bl, 0, mem, sizeof mem, VOC * 4u, "lg") != BL_ERR_PARAM); /* dup */
        bad += (bl_write(&bl, 0, BL_ANY_WRITER, 0, 1, Q_F32) != BL_ERR_PARAM);
        bad += (bl_write(&bl, 0, W_PREFILL, NROWS, 1, Q_F32) != BL_ERR_RANGE);
        bad += (bl_write(&bl, 0, W_PREFILL, 0, 0, Q_F32) != BL_ERR_RANGE);
        bad += (bl_write(&bl, 3, W_PREFILL, 0, 1, Q_F32) != BL_ERR_NOSUCH);
        bad += (bl_verify(&bl, 0, NROWS) != BL_ERR_RANGE);
        bad += (bl_verify(&bl, 0, 0) != BL_ERR_NEVER);              /* 从未写 → 断链 */
        bad += (bl_write(&bl, 0, W_PREFILL, 0, 3, Q_F32) != BL_OK);
        bad += (bl_expect(&bl, 0, 0, BL_ANY_WRITER, BL_NO_QTAG) != BL_OK);
        bad += (bl_verify(&bl, 0, 0) != BL_OK);                     /* 通配 → OK */
        bad += (bl_write(&bl, 0, W_TREE, 3, 1, Q_UF16A) != BL_OK);
        bl_expect(&bl, 0, 3, W_PREFILL, BL_NO_QTAG);
        bad += (bl_verify(&bl, 0, 3) != BL_ERR_WRITER);             /* 期望 writer 不符 */
        bl_expect(&bl, 0, 3, W_TREE, Q_UF16B);
        bad += (bl_verify(&bl, 0, 3) != BL_ERR_QTAG);               /* qtag 不一致 */
        bl_expect(&bl, 0, 3, W_TREE, Q_UF16A);
        bad += (bl_verify(&bl, 0, 3) != BL_OK);                     /* 对账通过 */
        bad += (bl.buf[0].n_writes != 2) + (bl.buf[0].n_reads != 2);
        bad += (bl.buf[0].n_breaks != 3) + (bl.n_double_write != 0);
        bad += (bl_seq(&bl) != 2);
        ex_check("bl_contract_table", bad, 0);
        bl_close(&bl);
    }

    /* ---- G2 (T1): 冷启动断链即时报错; 写入收敛修复 ---- */
    {
        struct bledger bl; float lg[NROWS * VOC] = {0};
        bl_init(&bl);
        bl_register(&bl, 0, lg, sizeof lg, VOC * 4u, "lg");
        /* 缺陷管线: prefill 只写末行 (行约定 A) */
        bl_write(&bl, 0, W_PREFILL, NROWS - 1, 1, Q_F32);
        bl_expect(&bl, 0, 0, BL_ANY_WRITER, BL_NO_QTAG);
        int rc = bl_verify(&bl, 0, 0);              /* 建树采样读行 0 → 断链 */
        /* 修复: 写入收敛到统一出口 (prefill 刷新全行) */
        bl_write(&bl, 0, W_PREFILL, 0, NROWS, Q_F32);
        int rc2 = bl_verify(&bl, 0, 0);
        int bad = (rc != BL_ERR_NEVER) + (rc2 != BL_OK);
        bad += (bl.buf[0].n_breaks != 1);
        ex_log("  T1 缺陷 verify=%d (NEVER=%d), 收敛后 verify=%d",
               rc, BL_ERR_NEVER, rc2);
        ex_check("t1_coldstart_immediate_error", bad, 0);
        bl_close(&bl);
    }

    /* ---- G3 (T2/E2): 行错位 — 值是真实行头名但不在预期位置 ---- */
    {
        struct bledger bl; float lg[NROWS * VOC] = {0};
        bl_init(&bl);
        bl_register(&bl, 0, lg, sizeof lg, VOC * 4u, "lg");
        uint32_t tok3 = 77;                          /* 生产者只写 row3 (末行) */
        gen_dist(lg + 3 * VOC, tok3);
        bl_write(&bl, 0, W_PREFILL, 3, 1, Q_F32);
        /* 消费者按 "行0" 约定采样, 别名使其读到 row3 的内存 (H1a) */
        uint32_t consumed = argmax_row(lg + 3 * VOC);
        bl_expect(&bl, 0, 0, BL_ANY_WRITER, BL_NO_QTAG);
        int rc = bl_verify(&bl, 0, 0);               /* 台账: row0 从未写 */
        uint32_t loc = 0xFFFFFFFFu, rank_loc = 0xFFFFFFFFu;
        for (uint32_t w = 0; w < NROWS; w++)
            if (bl.buf[0].rows[w].seq != 0) {        /* E2: 写过的行反查 rank */
                uint32_t r = rank_of(lg + w * VOC, consumed);
                if (r < rank_loc) { rank_loc = r; loc = w; }
            }
        int bad = (rc != BL_ERR_NEVER) + (consumed != tok3);
        bad += (loc != 3) + (rank_loc != 0);         /* 恰在 row3 rank 0 → H1a 坐实 */
        ex_log("  T2 断链 verify=%d, 消费=%u, rank0 行=%u → 行错位", rc, consumed, loc);
        ex_check("t2_row_misalign_rank_localize", bad, 0);
        bl_close(&bl);
    }

    /* ---- G4 (T3/E3): canary — 未初始化读变已知模式 + 全行深 rank ---- */
    {
        struct bledger bl; float lg[NROWS * VOC];
        memset(lg, 0, sizeof lg);
        bl_init(&bl);
        bl_register(&bl, 0, lg, sizeof lg, VOC * 4u, "lg");
        bl_canary(&bl, 0);
        for (uint32_t w = 1; w < NROWS; w++) {       /* 树评估只写 row1..7 */
            gen_dist(lg + w * VOC, 8 + w * 11);
            bl_write(&bl, 0, W_TREE, w, 1, Q_F32);
        }
        uint32_t consumed = argmax_row(lg);          /* 读 row0 = canary */
        bl_expect(&bl, 0, 0, BL_ANY_WRITER, BL_NO_QTAG);
        int rc = bl_verify(&bl, 0, 0);
        uint32_t min_rank = 0xFFFFFFFFu;
        for (uint32_t w = 1; w < NROWS; w++) {
            uint32_t r = rank_of(lg + w * VOC, consumed);
            if (r < min_rank) min_rank = r;
        }
        int bad = (rc != BL_ERR_CANARY) + !bits_are_canary(lg, VOC);
        bad += (min_rank <= 8);                      /* 无任何行 rank≈0 → 非 H1a */
        ex_log("  T3 verify=%d (CANARY=%d), 0xAA 位样=%d, 消费=%u 最浅 rank=%u",
               rc, BL_ERR_CANARY, bits_are_canary(lg, VOC), consumed, min_rank);
        ex_check("t3_canary_fingerprint_h2", bad, 0);
        bl_close(&bl);
    }

    /* ---- G5 (T4): 归还后禁读 ---- */
    {
        struct bledger bl; float lg[NROWS * VOC] = {0};
        bl_init(&bl);
        bl_register(&bl, 0, lg, sizeof lg, VOC * 4u, "lg");
        bl_write(&bl, 0, W_DECODE, 0, 1, Q_F32);
        bl_expect(&bl, 0, 0, BL_ANY_WRITER, BL_NO_QTAG);
        int rc0 = bl_verify(&bl, 0, 0);
        bl_release(&bl, 0);                          /* pool 归还 (自动 canary) */
        int rc = bl_verify(&bl, 0, 0);
        int bad = (rc0 != BL_OK) + (rc != BL_ERR_RELEASED);
        bad += !bits_are_canary(lg, VOC);
        ex_check("t4_release_forbidden_read", bad, 0);
        bl_close(&bl);
    }

    /* ---- G6 (T7): 双写检测 (未读行被异 writer 覆写) ---- */
    {
        struct bledger bl; float lg[NROWS * VOC] = {0};
        bl_init(&bl);
        bl_register(&bl, 0, lg, sizeof lg, VOC * 4u, "lg");
        bl_write(&bl, 0, W_PREFILL, 2, 1, Q_F32);    /* 写, 未读 */
        bl_write(&bl, 0, W_TREE, 2, 1, Q_F32);       /* 异 writer 覆写未读行 → 计数 */
        bl_expect(&bl, 0, 2, W_TREE, BL_NO_QTAG);
        int rc = bl_verify(&bl, 0, 2);               /* 消费 */
        bl_write(&bl, 0, W_TREE, 2, 1, Q_F32);       /* 读后刷新 = 合法 */
        int rc2 = bl_verify(&bl, 0, 2);              /* 再消费 (刷新被读到) */
        bl_write(&bl, 0, W_PREFILL, 2, 1, Q_F32);    /* 读后换写 = 合法 */
        bl_write(&bl, 0, W_DECODE, 4, 1, Q_F32);     /* 同 writer 覆写未读行 */
        bl_write(&bl, 0, W_DECODE, 4, 1, Q_F32);     /* → 不计 */
        int bad = (rc != BL_OK) + (rc2 != BL_OK) + (bl.n_double_write != 1);
        ex_check("t7_double_write_detected", bad, 0);
        bl_close(&bl);
    }

    /* ---- G7 (T5/T6): 新路径覆盖 + 量化对账 ---- */
    {
        struct bledger bl; float lg[NROWS * VOC] = {0};
        bl_init(&bl);
        bl_register(&bl, 0, lg, sizeof lg, VOC * 4u, "lg");
        bl_write(&bl, 0, W_LATE, 1, 1, Q_UF16A);     /* 新路径写入者用了旧配置 */
        bl_expect(&bl, 0, 1, W_LATE, Q_UF16B);       /* 消费点期望新量化标签 */
        int rcq = bl_verify(&bl, 0, 1);              /* → QTAG 拦截 (3.5) */
        bl_write(&bl, 0, W_LATE, 1, 1, Q_UF16B);     /* 修复: 单一配置源 */
        int rcq2 = bl_verify(&bl, 0, 1);
        bl_expect(&bl, 0, 2, W_LATE, BL_NO_QTAG);
        int rcn = bl_verify(&bl, 0, 2);              /* 新路径消费未写行 → NEVER */
        int bad = (rcq != BL_ERR_QTAG) + (rcq2 != BL_OK) + (rcn != BL_ERR_NEVER);
        ex_check("t5t6_newpath_qtag_guard", bad, 0);
        bl_close(&bl);
    }

    /* ---- G8 (T8): 端到端听诊 — P1 §5 draft 断链复刻 + 回归锚 ---- */
    {
        struct bledger bld, blf;                     /* 缺陷 / 修复 双引擎 */
        float *ld = calloc(NROWS * VOC, 4), *lf = calloc(NROWS * VOC, 4);
        float expct[VOC];
        uint32_t lcg = 20260817u;
        bl_init(&bld); bl_init(&blf);
        bl_register(&bld, 0, ld, NROWS * VOC * 4u, VOC * 4u, "draft");
        bl_register(&blf, 0, lf, NROWS * VOC * 4u, VOC * 4u, "draft");
        uint32_t acc_d = 0, acc_f = 0, anchor_d = 0, anchor_f = 0, rank_d = 0, d2ok_d = 0;
        for (uint32_t r = 0; r < NR; r++) {
            lcg = lcg * 1664525u + 1013904223u; uint32_t t1 = 8 + (lcg >> 24) % 100;
            lcg = lcg * 1664525u + 1013904223u; uint32_t t2 = 8 + (lcg >> 24) % 100;
            /* 修复引擎: row3 (末行=d1 源) 每轮都写 (写入收敛); row0 (树评估=d2 源) */
            gen_dist(lf + 3 * VOC, t1);
            bl_write(&blf, 0, r ? W_DECODE : W_PREFILL, 3, 1, Q_F32);
            gen_dist(lf + 0 * VOC, t2);
            bl_write(&blf, 0, W_TREE, 0, 1, Q_F32);
            /* 缺陷引擎: prefill/decode 路径写 row3 的调用被跳过 (3.2), 仅树评估写 */
            gen_dist(ld + 0 * VOC, t2);
            bl_write(&bld, 0, W_TREE, 0, 1, Q_F32);

            uint32_t d1f = argmax_row(lf + 3 * VOC), d2f = argmax_row(lf + 0);
            uint32_t d1d = argmax_row(ld + 3 * VOC), d2d = argmax_row(ld + 0);
            bl_expect(&blf, 0, 3, BL_ANY_WRITER, BL_NO_QTAG);
            bl_expect(&blf, 0, 0, W_TREE, BL_NO_QTAG);
            bl_expect(&bld, 0, 3, BL_ANY_WRITER, BL_NO_QTAG);
            bl_expect(&bld, 0, 0, W_TREE, BL_NO_QTAG);
            (void)bl_verify(&blf, 0, 3); (void)bl_verify(&blf, 0, 0);
            (void)bl_verify(&bld, 0, 3); (void)bl_verify(&bld, 0, 0);
            acc_f += 1 + (d1f == t1) + (d2f == t2);  /* root + 命中 drafts */
            acc_d += 1;                              /* d1 垃圾 → 链断, root only */
            d2ok_d += (d2d == t2);                   /* 部分正确: d2 本身是对的 */
            if (r == 0) {
                gen_dist(expct, t1);                 /* prefill 应写分布 (锚基准) */
                anchor_f = (d1f == argmax_row(expct));
                anchor_d = (d1d == argmax_row(expct));
                rank_d = rank_of(expct, d1d);        /* 深 rank */
            }
        }
        int bad_d2 = (d2ok_d != NR);                 /* d2 全对 = 断链只在 row3 路径 */
        int bad = (bld.buf[0].n_breaks != NR) + (blf.buf[0].n_breaks != 0);
        bad += (acc_d != NR) + (acc_f != 3 * NR);    /* 接受率: 崩塌 vs 满额 */
        bad += !anchor_f + anchor_d;                 /* 回归锚: 修复成立/缺陷不成立 */
        bad += (rank_d <= 100) + bad_d2;             /* d1 深rank; d2 部分正确 */
        ex_log("  T8 接受和 缺陷=%u/%u (hist 全1), 修复=%u/%u (hist 全3); "
               "d1 rank=%u/%u; 锚 缺陷=%u 修复=%u; d2 对 %u/%u 轮",
               acc_d, NR, acc_f, 3 * NR, rank_d, VOC, anchor_d, anchor_f, d2ok_d, NR);
        ex_check("t8_e2e_stethoscope_anchor", bad, 0);

        /* ---- G9 (L1): timeline 文本时序图 ---- */
        char tl[640];
        uint32_t n = bl_timeline(&blf, 0, tl, sizeof tl);
        int bad9 = (n == 0) || !strstr(tl, "draft") || !strstr(tl, "writes=16");
        bad9 += (bl_timeline(&bld, 0, tl, sizeof tl) == 0) || !strstr(tl, "breaks=8");
        ex_log("  L1 timeline (修复引擎, %u chars):\n%s", n, tl);
        ex_check("l1_timeline_text", bad9, 0);
        bl_close(&bld); bl_close(&blf); free(ld); free(lf);
    }
    return ex_summary();
}
