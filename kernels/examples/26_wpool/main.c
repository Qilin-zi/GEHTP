/*
 * 26_wpool — U12 常驻工人池 设备验证
 * =====================================================================
 * 覆盖 wpool.h 全 API + hmx 门交接 (依赖 23 fence: job 内 CPU→HMX VTCM
 * handoff 走 fence API)。判据:
 *   1) G1 随机到达: 24 个混合 job (norm/dot/hvxload) 池执行结果
 *      与主线程串行参考逐值一致 (并发不改数值)
 *   2) G2 hmx 门: 2 引擎 job (交接 unlock→job lock→relock) 输出与
 *      主线程串行 invoke byte-exact
 *   3) G3 spawn-per-op vs 常驻池: 16 job norm, 池路径必须快于逐 job
 *      spawn+join (op82 结论: spawn 开销 90x+)
 *   4) G4 压力: 5 轮 × 12 job, done/executed 计数 == 总投递, 每轮数值全对
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define A "/data/local/tmp/hvxhmx23/assets/s256"
#define M 256u
#define NJ 24
#define NRM 256

struct j26 {
    int kind;                       /* 0 norm 1 dot 2 hvxload */
    const int16_t *x, *a, *b;
    const uint8_t* scratch;
    int16_t out[NRM];
    uint64_t dot;
    uint32_t hsum;
    int16_t exp_out[NRM];
    uint64_t exp_dot;
    uint32_t exp_hsum;
    /* hmx 门 job */
    struct dc_w4* e;
    const uint8_t* src; uint32_t sb;
    uint8_t* dst;
    int bad;
    struct wtcache_ctx* wc;
};

static void j_run(void* p) {
    struct j26* j = (struct j26*)p;
    if (j->kind == 3) {                          /* hmx 引擎 job */
        memcpy(j->e->act, j->src, j->sb);
        fence_handoff(j->e->act, j->sb, FC_CPU, FC_HMX, FM_VTCM);
        if (wtcache_hmx_lock(j->wc) == 0) {
            j->bad = dc_w4_invoke(j->e);
            wtcache_hmx_unlock(j->wc);
            if (!j->bad) dc_w4_read_out(j->e, j->dst);
        } else j->bad = 0xEE01;
        return;
    }
    if (j->kind == 2)      j->hsum = dc_hvx_load((uint8_t*)j->scratch, 1024);
    else if (j->kind == 1) j->dot  = dc_dot_u64(j->a, j->b, NRM);
    else                   dc_norm_i16(j->x, j->out, NRM);
}

static void spawn_run(void* p) { j_run(p); }

int main(void) {
    ex_open_result("26_wpool");
    struct wtcache_ctx* wc = NULL;
    uint32_t lcg = 20260826u;
    uint32_t bw = 0, bb = 0, ba = 0, bo = 0, b0 = 0, b1 = 0;
    struct wpool pool;

    uint8_t* wt  = dc_read_file(A "/packed_weight.raw", &bw);
    uint8_t* bis = dc_read_file(A "/folded_bias.raw", &bb);
    uint8_t* at  = dc_read_file(A "/act_table.raw", &ba);
    uint8_t* ot  = dc_read_file(A "/out_table.raw", &bo);
    uint8_t* av0 = dc_read_file(A "/act_variants/v0.raw", &b0);
    uint8_t* av1 = dc_read_file(A "/act_variants/v1.raw", &b1);
    if (!wt || !bis || !at || !ot || !av0 || !av1 || b0 != M * M * 2 || b1 != M * M * 2) {
        ex_log("assets missing/mismatch"); goto out;
    }
    dc_clean_ddr(wt, bw); dc_clean_ddr(bis, bb);
    dc_clean_ddr(av0, b0); dc_clean_ddr(av1, b1);

    if (wtcache_open(&wc, 4096) != WTC_OK) { ex_log("wtcache_open FAIL"); goto out; }
    {
        void* vb; uint32_t vs; void* pb; uint32_t pc;
        wtcache_layout(wc, &vb, &vs, &pb, &pc);
        struct dc_arena ar;
        dc_arena_init(&ar, (uint8_t*)vb + ((pc + 2047u) & ~2047u),
                      vs - ((pc + 2047u) & ~2047u));
        struct dc_w4 e0, e1;
        if (dc_w4_carve(&e0, &ar, M, M, M, at, ot) ||
            dc_w4_carve(&e1, &ar, M, M, M, at, ot)) { ex_log("carve FAIL"); goto out; }
        memcpy(e0.wt, wt, bw); memcpy(e0.bias, bis, bb);
        memcpy(e1.wt, wt, bw); memcpy(e1.bias, bis, bb);
        dc_clean_ddr(e0.wt, bw); dc_clean_ddr(e0.bias, bb);
        dc_clean_ddr(e1.wt, bw); dc_clean_ddr(e1.bias, bb);

        /* G1 输入 (VTCM 128B 对齐, 同 ex22) */
        int16_t* nx = (int16_t*)dc_arena_alloc(&ar, NRM * 2, 128);
        int16_t* na = (int16_t*)dc_arena_alloc(&ar, NRM * 2, 128);
        int16_t* nb = (int16_t*)dc_arena_alloc(&ar, NRM * 2, 128);
        uint8_t* sc = dc_arena_alloc(&ar, 4096, 128);
        if (!nx || !na || !nb || !sc) { ex_log("scratch FAIL"); goto out; }
        for (int i = 0; i < NRM; i++) {
            nx[i] = (int16_t)(lcg = lcg * 1664525u + 1013904223u);
            na[i] = (int16_t)(lcg = lcg * 1664525u + 1013904223u);
            nb[i] = (int16_t)(lcg = lcg * 1664525u + 1013904223u);
        }
        for (int i = 0; i < 4096; i++) sc[i] = (uint8_t)(lcg >> 24);

        struct j26 jobs[NJ];
        memset(jobs, 0, sizeof(jobs));
        for (int i = 0; i < NJ; i++) {
            struct j26* j = &jobs[i];
            j->kind = (int)((lcg = lcg * 1664525u + 1013904223u) % 3);
            j->x = nx; j->a = na; j->b = nb; j->scratch = sc;
            /* 主线程串行参考 */
            j_run(j);                       /* 直接调 → 写 j->out/dot/hsum */
            memcpy(j->exp_out, j->out, sizeof(j->exp_out));
            j->exp_dot = j->dot; j->exp_hsum = j->hsum;
            j->out[0] = 0x7BEE; j->dot = 0; j->hsum = 0;   /* 抹掉再让池跑 */
        }

        if (wpool_open(&pool, 2, 96 * 1024)) { ex_log("wpool_open FAIL"); goto out; }
        for (int i = 0; i < NJ; i++)
            if (wpool_submit(&pool, j_run, &jobs[i])) { ex_log("queue full"); goto poolout; }
        wpool_wait_all(&pool, NJ);

        int g1 = 0;
        for (int i = 0; i < NJ; i++) {
            struct j26* j = &jobs[i];
            if (j->kind == 0 && memcmp(j->out, j->exp_out, sizeof(j->exp_out))) g1++;
            if (j->kind == 1 && j->dot != j->exp_dot) g1++;
            if (j->kind == 2 && j->hsum != j->exp_hsum) g1++;
        }
        ex_check("randarr_24jobs_value_exact", g1, 0);
        ex_log("  G1 24 job (norm/dot/hvxload 混合) 随机到达: %d mismatch", g1);

        /* G2 hmx 门: 串行参考 (主线程持锁) */
        uint8_t* r0 = memalign(128, M * M * 2);
        uint8_t* r1 = memalign(128, M * M * 2);
        uint8_t* p0 = memalign(128, M * M * 2);
        uint8_t* p1 = memalign(128, M * M * 2);
        struct j26 hj[2];
        memset(hj, 0, sizeof(hj));
        hj[0].kind = 3; hj[0].e = &e0; hj[0].src = av0; hj[0].sb = b0;
        hj[0].dst = r0; hj[0].wc = wc;
        hj[1].kind = 3; hj[1].e = &e1; hj[1].src = av1; hj[1].sb = b1;
        hj[1].dst = r1; hj[1].wc = wc;
        /* 主线程串行参考: wtcache_open 已持 hmx_lock, 不可再 lock */
        {
            struct j26* j = &hj[0];
            memcpy(j->e->act, j->src, j->sb);
            fence_handoff(j->e->act, j->sb, FC_CPU, FC_HMX, FM_VTCM);
            j->bad = dc_w4_invoke(j->e);
            if (!j->bad) dc_w4_read_out(j->e, j->dst);
            j = &hj[1];
            memcpy(j->e->act, j->src, j->sb);
            fence_handoff(j->e->act, j->sb, FC_CPU, FC_HMX, FM_VTCM);
            j->bad = dc_w4_invoke(j->e);
            if (!j->bad) dc_w4_read_out(j->e, j->dst);
        }
        int ser_bad = hj[0].bad || hj[1].bad;
        hj[0].dst = p0; hj[0].bad = 0xEE02;
        hj[1].dst = p1; hj[1].bad = 0xEE02;

        wtcache_hmx_unlock(wc);                /* P3 交接 */
        wpool_submit(&pool, j_run, &hj[0]);
        wpool_submit(&pool, j_run, &hj[1]);
        wpool_wait_all(&pool, NJ + 2);
        wpool_close(&pool);
        wtcache_hmx_lock(wc);                  /* 取回 */
        int g2 = ser_bad || hj[0].bad || hj[1].bad ||
                 memcmp(r0, p0, M * M * 2) || memcmp(r1, p1, M * M * 2);
        ex_check("hmx_gate_pool_vs_serial_byteexact", g2 ? 1 : 0, 0);
        ex_log("  G2 hmx 门交接 (unlock→job lock→relock): %s",
               g2 ? "MISMATCH" : "byte-exact");

        /* G3 spawn-per-op vs 常驻池 (16 × norm) */
        struct j26 tj[16];
        memset(tj, 0, sizeof(tj));
        for (int i = 0; i < 16; i++) {
            tj[i].kind = 0; tj[i].x = nx;
            j_run(&tj[i]);                                  /* 主线程串行参考 */
            memcpy(tj[i].exp_out, tj[i].out, sizeof(tj[i].exp_out));
            memset(tj[i].out, 0, sizeof(tj[i].out));
        }
        uint64_t ts, tp;
        {
            int64_t t0 = HAP_perf_get_time_us();
            for (int i = 0; i < 16; i++) {
                dc_thread_t t;
                if (dc_spawn(&t, "sp26", spawn_run, &tj[i], 64 * 1024)) { ex_log("spawn FAIL"); goto free3; }
                dc_join(&t);
            }
            ts = (uint64_t)(HAP_perf_get_time_us() - t0);
        }
        {
            if (wpool_open(&pool, 2, 96 * 1024)) { ex_log("wpool_open(2) FAIL"); goto free3; }
            int64_t t0 = HAP_perf_get_time_us();
            for (int i = 0; i < 16; i++) wpool_submit(&pool, j_run, &tj[i]);
            wpool_wait_all(&pool, 16);
            tp = (uint64_t)(HAP_perf_get_time_us() - t0);
        }
        int g3v = 0;
        for (int i = 0; i < 16; i++)
            if (memcmp(tj[i].out, tj[i].exp_out, sizeof(tj[i].exp_out))) g3v++;
        ex_check("spawn16_value_exact", g3v, 0);
        ex_check("pool_faster_than_spawn", tp < ts ? 0 : 1, 0);
        ex_log("  G3 spawn-per-op %llu us vs pool %llu us (ratio %.1fx)",
               (unsigned long long)ts, (unsigned long long)tp,
               tp ? (double)ts / (double)tp : 0.0);

        /* G4 压力: 5 轮 × 12 job */
        int g4 = 0;
        for (int r = 0; r < 5; r++) {
            struct j26 sj[12];
            memset(sj, 0, sizeof(sj));
            for (int i = 0; i < 12; i++) {
                struct j26* j = &sj[i];
                j->kind = (int)((lcg = lcg * 1664525u + 1013904223u) % 3);
                j->x = nx; j->a = na; j->b = nb; j->scratch = sc;
                j_run(j);
                memcpy(j->exp_out, j->out, sizeof(j->exp_out));
                j->exp_dot = j->dot; j->exp_hsum = j->hsum;
                j->out[0] = 0x7BEE; j->dot = 0; j->hsum = 0;
                wpool_submit(&pool, j_run, j);
            }
            wpool_wait_all(&pool, 16 + (uint32_t)(r + 1) * 12);
            for (int i = 0; i < 12; i++) {
                struct j26* j = &sj[i];
                if (j->kind == 0 && memcmp(j->out, j->exp_out, sizeof(j->exp_out))) g4++;
                if (j->kind == 1 && j->dot != j->exp_dot) g4++;
                if (j->kind == 2 && j->hsum != j->exp_hsum) g4++;
            }
        }
        uint32_t total = 16 + 5 * 12;   /* G3 重开池后计数从 0 起 */
        if (pool.done != total || pool.executed != total) g4++;
        ex_check("stress_5rounds_counters_exact", g4, 0);
        ex_log("  G4 5 轮压力: done=%u executed=%u total=%u", pool.done, pool.executed, total);
        wpool_close(&pool);

free3:
        free(r0); free(r1); free(p0); free(p1);
    }
out:
    if (wc) wtcache_close(wc);
    return ex_summary();
poolout:
    wpool_close(&pool);
    if (wc) wtcache_close(wc);
    return ex_summary();
}
