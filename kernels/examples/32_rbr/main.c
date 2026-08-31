/*
 * 32_rbr — U18 recurrent 状态部分接受回退 设备验证
 * =====================================================================
 * 依据 docs/P0-recurrent-state-rollback.md 验证金字塔:
 *   L1 模块契约  T1 快照往返 / T2 skip 一次性+CLEAR/COPY / T3 无快照与
 *                陈旧拒绝 (陷阱2 防御) / 注册冻结
 *   L2 轮级断言  T5 缺陷指纹 (等式1: replay_in == phantom_out, 用缺陷引擎
 *                证明可检出) / T6 restore 生效 (等式2: replay_in ==
 *                snapshot_src 逐字节) / T7 KV 行覆写幂等
 *   L3 端到端    16 轮混合接受 (含 j=0 强拒绝与全接受流) 后, 引擎终态 +
 *                KV 行 + 两套 n_past 账本 == 无投机基线 (f32 bitwise)
 *
 * 模拟引擎: 2 group × (conv_state[4] + recurrent_state 2×8×8), F =
 * ref_conv_step + ref_delta_token (f32 确定性, 同序运算 ⇒ bitwise 可判)。
 * 缺陷引擎 = setup 无条件 out→in copy (P0 §3.2 copyLinearAttentionStates)。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define NG   2u          /* cache group 数 */
#define CW   4u          /* conv 窗口 W */
#define DH   8u          /* 每 head dim */
#define NH   2u          /* heads/group */
#define REC  (NH * DH * DH)   /* 128 f32/group */
#define ROWF (NG * DH)        /* 每 KV 行 y 元素 */
#define NT   7               /* 树 token 数 = k+1 (k=6) */
#define NROUNDS 16

struct tok { float x; float k[NG][DH], v[NG][DH], q[NG][DH]; float beta, g; };

struct simeng {
    struct rbr* rb;
    int    naive;                       /* 1 = 缺陷引擎 (无条件 copy) */
    float  conv_in[NG][CW], conv_out[NG][CW];
    float  rec_in[NG][REC], rec_out[NG][REC];
    float  conv_w[NG][CW],  rec_w[NG][REC];
    conv_state_t cst[NG];
    rec_state_t  rst[NG];
    float  rows[NROUNDS * NT][ROWF];    /* KV 行模型 (写时定格) */
    uint32_t n_past;                    /* 引擎行账本 */
    uint32_t ledger_mismatch;           /* 两账本对账失败计数 (陷阱1) */
    float  w[CW];                       /* conv 权重 (d_inner=1) */
};

static struct simeng* eng_new(struct rbr* rb, int naive, uint32_t* lcg) {
    struct simeng* e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->rb = rb; e->naive = naive;
    for (uint32_t g = 0; g < NG; g++) {
        e->cst[g].d_inner = 1; e->cst[g].d_conv = CW; e->cst[g].win = e->conv_w[g];
        e->rst[g].h = NH;      e->rst[g].d = DH;     e->rst[g].s = e->rec_w[g];
    }
    for (uint32_t i = 0; i < CW; i++) e->w[i] = gdn_lcg_norm(lcg) * 0.3f;
    if (rb) {
        rbr_init(rb);
        for (uint32_t g = 0; g < NG; g++) {
            rbr_register(rb, g, e->conv_in[g], CW * 4u);
            rbr_register(rb, g, e->rec_in[g], REC * 4u);
        }
    }
    return e;
}

/* 一次 process: setup 三分支 → 前向 (只读 in, 写 out + KV 行) → 记账 */
static void eng_process(struct simeng* e, const struct tok* ts, uint32_t n) {
    int m;
    if (e->naive) {
        m = e->n_past ? RBR_COPY : RBR_CLEAR;      /* 缺陷: 无条件 copy */
    } else {
        if (e->n_past != rbr_n_past(e->rb)) e->ledger_mismatch++;
        m = rbr_setup_hook(e->rb, e->n_past);
    }
    if (m == RBR_CLEAR) {
        memset(e->conv_in, 0, sizeof(e->conv_in));
        memset(e->rec_in, 0, sizeof(e->rec_in));
    } else if (m == RBR_COPY) {
        memcpy(e->conv_in, e->conv_out, sizeof(e->conv_in));
        memcpy(e->rec_in, e->rec_out, sizeof(e->rec_in));
    }                                    /* RBR_SKIP: 保留 in (INV-3) */
    for (uint32_t g = 0; g < NG; g++) {
        memcpy(e->conv_w[g], e->conv_in[g], CW * 4u);
        memcpy(e->rec_w[g], e->rec_in[g], REC * 4u);
    }
    float yc, y[DH];
    for (uint32_t i = 0; i < n; i++) {
        float* row = e->rows[e->n_past + i];
        for (uint32_t g = 0; g < NG; g++) {
            ref_conv_step(&e->cst[g], e->w, &ts[i].x, &yc);
            ref_delta_token(&e->rst[g], ts[i].k[g], ts[i].v[g], ts[i].q[g],
                            ts[i].beta, ts[i].g, y);
            memcpy(row + g * DH, y, DH * 4u);
        }
    }
    for (uint32_t g = 0; g < NG; g++) {
        memcpy(e->conv_out[g], e->conv_w[g], CW * 4u);
        memcpy(e->rec_out[g], e->rec_w[g], REC * 4u);
    }
    if (e->rb) rbr_note_process(e->rb, n);
    e->n_past += n;
}

static void gen_tree(struct tok* t, uint32_t* lcg) {
    for (uint32_t i = 0; i < NT; i++) {
        t[i].x = gdn_lcg_norm(lcg) * 0.5f;
        for (uint32_t g = 0; g < NG; g++)
            for (uint32_t n = 0; n < DH; n++) {
                t[i].k[g][n] = gdn_lcg_norm(lcg) * 0.5f;
                t[i].v[g][n] = gdn_lcg_norm(lcg) * 0.5f;
                t[i].q[g][n] = gdn_lcg_norm(lcg) * 0.5f;
            }
        *lcg = *lcg * 1664525u + 1013904223u;
        t[i].beta = 0.4f + 0.4f * (float)(*lcg >> 30) / 2.0f;
        t[i].g = gdn_lcg_norm(lcg) * 0.3f;
    }
}

int main(void) {
    ex_open_result("32_rbr");
    uint32_t lcg = 20260816u;
    const int js[NROUNDS] = {0, 3, 6, 1, 5, 2, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6};

    /* ---- L1: 模块契约 (T1/T2/T3) ---- */
    {
        struct rbr r; float a[4] = {1, 2, 3, 4}, b[8] = {5, 6, 7, 8, 9, 10, 11, 12};
        rbr_init(&r);
        int g1 = (rbr_restore(&r) == RBR_ERR_NOSNAP) ? 0 : 1;   /* T3 无快照 */
        rbr_register(&r, 0, a, sizeof a);
        rbr_register(&r, 0, b, sizeof b);
        rbr_snapshot(&r);
        a[0] = 99.f; b[7] = 99.f;                               /* T1 污染后恢复 */
        int rt = rbr_restore(&r);
        g1 += (rt != RBR_OK) + (memcmp(a, (float[]){1, 2, 3, 4}, 16) != 0);
        g1 += !rbr_shadow_equals(&r);
        ex_check("l1_snapshot_roundtrip", g1, 0);

        int m1 = rbr_setup_hook(&r, 5);                         /* T2 skip 一次性 */
        int m2 = rbr_setup_hook(&r, 5);
        int m3 = rbr_setup_hook(&r, 0);
        int m4 = rbr_setup_hook(&r, 5);                         /* CLEAR 须作废 skip */
        int g2 = (m1 != RBR_SKIP) + (m2 != RBR_COPY) + (m3 != RBR_CLEAR) + (m4 != RBR_COPY);
        g2 += (r.n_skip_used != 1);
        ex_check("l1_skip_onshot_then_copy_clear", g2, 0);

        rbr_note_process(&r, 1);                                /* T3 世代陈旧 */
        int g3 = (rbr_restore(&r) != RBR_ERR_STALE);
        g3 += (rbr_register(&r, 1, a, sizeof a) != RBR_ERR_FROZEN);
        g3 += (rbr_rewind(&r, 0) != RBR_ERR_PARAM) + (rbr_rewind(&r, 2) != RBR_ERR_PARAM);
        ex_check("l1_stale_frozen_badrewind_rejected", g3, 0);
        rbr_close(&r);
    }

    /* ---- L2: 缺陷指纹 (T5) — 缺陷引擎在重放点喂入幻影状态, 必须可检出 ---- */
    {
        struct rbr rn, rb0;
        struct simeng *en = eng_new(&rn, 1, &lcg), *eb = eng_new(&rb0, 0, &lcg);
        struct tok t[NT], w[NT];
        gen_tree(w, &lcg); gen_tree(t, &lcg);
        eng_process(en, w, 1);                     /* 预热: 使回放落在 n_past>0 */
        eng_process(eb, w, 1);                     /* (真引擎回放发生在对话中段) */
        eng_process(en, t, NT);                    /* 幻影前移 */
        rbr_snapshot(&rn);                          /* INV-1: process 之后 */
        float ph_c[NG][CW], ph_r[NG][REC];
        memcpy(ph_c, en->conv_out, sizeof(ph_c));
        memcpy(ph_r, en->rec_out, sizeof(ph_r));
        eng_process(eb, t, 1);                     /* 基线: 只消费 root */

        rbr_restore(&rn);
        rbr_rewind(&rn, 1);
        en->n_past = rbr_n_past(&rn);               /* updateKV 回拨 (INV-2) */
        eng_process(en, t, 1);                      /* 缺陷 setup = COPY ⇒ in←幻影 */

        int fp = (memcmp(en->conv_in, ph_c, sizeof(ph_c)) == 0 &&
                  memcmp(en->rec_in, ph_r, sizeof(ph_r)) == 0);   /* 等式1 */
        int dv = (memcmp(en->rec_out, eb->rec_out, sizeof(en->rec_out)) != 0);
        ex_log("  T5 指纹成立=%d (等式1), 缺陷偏离基线=%d", fp, dv);
        ex_check("defect_fingerprint_detected", !(fp && dv), 0);
        rbr_close(&rn); rbr_close(&rb0); free(en); free(eb);
    }

    /* ---- L2+L3: 16 轮混合接受, 修复引擎 vs 无投机基线 ---- */
    {
        struct rbr rf, rbs;
        struct simeng *ef = eng_new(&rf, 0, &lcg), *eb = eng_new(&rbs, 0, &lcg);
        int probe_round = 0;                        /* 首个部分接受轮做 L2 探针 */
        float snap_c[NG][CW], snap_r[NG][REC];
        uint32_t tree_row0[ROWF * NT], tree_row_bytes = 0;
        int eq2 = 0, idem = 0;

        for (int r = 0; r < NROUNDS; r++) {
            struct tok t[NT]; gen_tree(t, &lcg);
            int j = js[r];                          /* 接受 tree[0..j] */
            eng_process(ef, t, NT);                 /* 树评估 (幻影全消费) */
            rbr_snapshot(&rf);                      /* INV-1 */

            if (r == probe_round) memcpy(snap_c, ef->conv_in, sizeof(snap_c)),
                                  memcpy(snap_r, ef->rec_in, sizeof(snap_r));

            if (j < NT - 1) {                       /* 部分接受 → 回退三段式 */
                if (r == probe_round) {             /* T7: 树评估行先存档 */
                    uint32_t base = rbr_n_past(&rf) - NT;
                    tree_row_bytes = (uint32_t)(j + 1) * ROWF * 4u;
                    memcpy(tree_row0, ef->rows[base], tree_row_bytes);
                }
                rbr_restore(&rf);                   /* a. 恢复 */
                uint32_t base = rbr_n_past(&rf) - NT;
                rbr_rewind(&rf, (uint32_t)j + 1);   /* b. 回拨 (INV-2) */
                ef->n_past = rbr_n_past(&rf);
                eng_process(ef, t, (uint32_t)j + 1);/* c. 重放 (SKIP 保留 in) */
                if (r == probe_round) {
                    eq2 = rbr_shadow_equals(&rf) &&
                          memcmp(ef->conv_in, snap_c, sizeof(snap_c)) == 0 &&
                          memcmp(ef->rec_in, snap_r, sizeof(snap_r)) == 0;
                    idem = memcmp(tree_row0, ef->rows[base], tree_row_bytes) == 0;
                }
            }
            eng_process(eb, t, (uint32_t)j + 1);    /* 基线: 只见接受链 */
        }
        ex_check("restore_effective_equation2", !eq2, 0);       /* T6 */
        ex_check("replay_kv_overwrite_idempotent", !idem, 0);   /* T7 */

        /* T8: 终态 + KV 行 + 账本 全部 bitwise 一致 */
        uint32_t total = eb->n_past;
        int g8 = memcmp(ef->conv_out, eb->conv_out, sizeof(ef->conv_out)) != 0;
        g8 += memcmp(ef->rec_out, eb->rec_out, sizeof(ef->rec_out)) != 0;
        g8 += memcmp(ef->rows, eb->rows, (size_t)total * ROWF * 4u) != 0;
        g8 += (ef->n_past != total) + (rbr_n_past(&rf) != total);
        g8 += ef->ledger_mismatch + eb->ledger_mismatch;
        ex_log("  T8 16 轮 (6 部分接受/10 全接受) 终态/KV/%u 行/账本 bitwise: %d err",
               total, g8);
        ex_check("e2e_rounds_bitexact_vs_baseline", g8, 0);

        /* T4/T10: 计数器精确 + 全接受流零干预 */
        uint32_t want_restore = 0, want_skip = 0, start = 0;
        for (int r = 0; r < NROUNDS; r++) {
            if (js[r] < NT - 1) {
                want_restore++;
                want_skip += (start > 0);   /* round-0 重放 n_past==0 → CLEAR, skip 不消耗 */
            }
            start += (uint32_t)js[r] + 1;
        }
        int g9 = (rf.n_snapshot != NROUNDS);
        g9 += (rf.n_restore != want_restore) + (rf.n_rewind != want_restore);
        g9 += (rf.n_skip_used != want_skip);
        ex_log("  T4/T10 snap=%u restore=%u rewind=%u skip=%u (期望 %u/%u/%u)",
               rf.n_snapshot, rf.n_restore, rf.n_rewind, rf.n_skip_used,
               NROUNDS, want_restore, want_skip);
        ex_check("counters_exact_fullaccept_zero_touch", g9, 0);

        rbr_close(&rf); rbr_close(&rbs); free(ef); free(eb);
    }
    return ex_summary();
}
