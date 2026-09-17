/*
 * 34_dmaring — U20 DMA 预取环 设备验证 (GENERIC_A 问题书)
 * =====================================================================
 * 问题: P_serial 窗口引擎空转 / P_bulk 起步数据未就位 → 设计预取环策略,
 * 硬约束 = 引擎门铃纪律 Law1-8 (违反=域崩溃)。
 *
 * 门 (问题书 §8 验收 + §9.2 金丝雀):
 *   G1 单组单槽        enqueue 1 组 → need READY, 1 门铃, 数据 bit-exact
 *   G2 环回绕 ×100     N=4 槽 100 组顺序消费, 零 FATAL, 槽复用数据校验,
 *                      不变量 I1/I2/I5 每步零违例
 *   G3 满 P_serial 预填 serial 相位 (深度 boost) 后 bulk 零停顿
 *   G4 F2 门铃丢失     g0 消费后 arm F2 → g1 dmstart 被吞成孤儿 → 超时
 *                      re-kick 救回 READY; G4c F4 假 done 数据未落被检出
 *   G5 Law 断言自证    (a) 合法序列零 FATAL; (b) 人为 RETIRE 窗 dmstart
 *                      → 断言层必须报 law6 (证明 harness 有效)
 *   G6 轨迹 T_eff      serial_heavy + burst_after_starve 两条内置轨迹,
 *                      oracle 上界 (预知轨迹的满登记策略) 对比 ≥0.95
 *   G7 参数扫描        W_RETIRE×10 / BW 40-70 / N∈{2,4,8} → 零 FATAL 零死锁
 *   G8 门铃经济性      stats: 每组门铃 ≤2, 均值 ≤1.5 (V7)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define N_SLOTS    4u
#define SLOT_KB    512u
#define SLOT_BYTES (SLOT_KB * 1024u)
#define GROUP_BYTES SLOT_BYTES       /* 1 desc/group: 拆分逻辑可证的边界 */
#define DDR_GROUPS 16u               /* 源区 8MB (设备 .so BSS 上限内) */

static uint8_t ddr[DDR_GROUPS * GROUP_BYTES];       /* DDR 源 (pattern) */
static uint8_t slots[N_SLOTS * SLOT_BYTES];         /* staging */

static void fill_group(uint32_t g) {
    ex_fill_u8(ddr + (size_t)(g % DDR_GROUPS) * GROUP_BYTES, GROUP_BYTES,
               (int)((g % DDR_GROUPS) + 1u), 251);
}
static uint64_t src_of(uint32_t g) {
    return (uint64_t)(uintptr_t)(ddr + (size_t)(g % DDR_GROUPS) * GROUP_BYTES);
}
static int check_group(uint32_t g, uint64_t slot) {
    return memcmp((const void *)(uintptr_t)slot,
                  ddr + (size_t)(g % DDR_GROUPS) * GROUP_BYTES,
                  GROUP_BYTES) == 0;
}

/* 轨迹驱动器: 一串 (serial_us, n_bulk_groups, bulk_us) 段; 返回总停顿 µs。
 * policy: 每段开头生产+登记该段组 (serial 窗口预取); oracle: t=0 全登记。 */
static double run_trace(unsigned n_segs, const uint32_t *serial_us,
                        const uint32_t *n_groups, const uint32_t *bulk_us,
                        double *t_eff_out, uint64_t *bytes_out, int oracle) {
    engine_adapter eng;
    sim_adapter_init(sim_default_params(), &eng);
    ring_policy *r = ring_create(&eng, N_SLOTS, RING_MAX_DESCS_SLOT,
                                 SLOT_BYTES, slots);
    uint64_t bytes = 0;
    uint32_t g = 0;
    double t0 = sim_now();
    if (oracle) {
        unsigned tot = 0;
        for (unsigned s = 0; s < n_segs; s++) tot += n_groups[s];
        for (unsigned k = 0; k < tot; k++)
            ring_enqueue(r, src_of(k), GROUP_BYTES, 1, GROUP_BYTES,
                         GROUP_BYTES);
    }
    for (unsigned s = 0; s < n_segs; s++) {
        if (!oracle)
            for (uint32_t k = 0; k < n_groups[s]; k++)
                ring_enqueue(r, src_of(g + k), GROUP_BYTES, 1, GROUP_BYTES,
                             GROUP_BYTES);
        ring_on_serial_phase(r, serial_us[s]);   /* 空窗通知 (深度提升) */
        sim_advance((double)serial_us[s]);       /* P_serial: 引擎自由跑 */
        for (uint32_t k = 0; k < n_groups[s]; k++) {
            uint64_t slot = 0;
            ring_need_result rc = ring_need(r, g, 60000, &slot);
            if (rc != RING_READY) {
                ex_log("  !! trace need g=%u rc=%d", g, (int)rc);
                ring_destroy(r); return -1.0;
            }
            if (!check_group(g, slot)) {
                ex_log("  !! trace data g=%u", g);
                ring_destroy(r); return -1.0;
            }
            ring_release(r, g);
            bytes += GROUP_BYTES;
            g++;
        }
        sim_advance((double)bulk_us[s]);         /* P_bulk 计算 */
    }
    double wall = sim_now() - t0;
    sim_report rep; sim_adapter_report(&rep);
    ring_stats st; ring_get_stats(r, &st);
    int law = rep.law, inv = ring_check_invariants(r);
    ring_destroy(r);
    if (law) ex_log("  !! trace FATAL law=%d", law);
    if (inv) ex_log("  !! trace INV=%d", inv);
    *t_eff_out = wall > 0 ? (double)bytes / wall : 0.0;  /* B/µs */
    *bytes_out = bytes;
    return st.stall_us_total;
}

int main(void)
{
    ex_open_result("34_dmaring");

    /* ---- G1 单组单槽 ---- */
    {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        ring_policy *r = ring_create(&eng, N_SLOTS, RING_MAX_DESCS_SLOT,
                                     SLOT_BYTES, slots);
        fill_group(0);
        int rc = ring_enqueue(r, src_of(0), GROUP_BYTES, 1, GROUP_BYTES,
                              GROUP_BYTES);
        uint64_t slot = 0;
        ring_need_result nr = ring_need(r, 0, 60000, &slot);
        ex_check("G1 need READY", nr != RING_READY, 0);
        ex_check("G1 enqueue rc", rc != RING_OK, 0);
        ex_check("G1 数据 bit-exact", !check_group(0, slot), 0);
        ring_stats st; ring_get_stats(r, &st);
        ex_check("G1 门铃=1", (int)st.n_doorbells != 1, 0);
        ring_destroy(r);
        sim_report rep; sim_adapter_report(&rep);
        ex_check("G1 零 FATAL", rep.law != 0, 0);
    }

    /* ---- G2 环回绕 ×100 (不变量每步) ---- */
    {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        ring_policy *r = ring_create(&eng, N_SLOTS, RING_MAX_DESCS_SLOT,
                                     SLOT_BYTES, slots);
        int bad = 0, inv = 0;
        for (uint32_t g = 0; g < 100; g++) {
            fill_group(g);
            ring_enqueue(r, src_of(g), GROUP_BYTES, 1, GROUP_BYTES,
                         GROUP_BYTES);
            uint64_t slot = 0;
            if (ring_need(r, g, 60000, &slot) != RING_READY) { bad++; continue; }
            if (!check_group(g, slot)) bad++;
            if (ring_check_invariants(r)) inv++;
            ring_release(r, g);
            if (ring_check_invariants(r)) inv++;
        }
        ex_check("G2 100 组零错", bad != 0, 0);
        ex_check("G2 不变量 I1/I2/I5 零违例", inv != 0, 0);
        sim_report rep; sim_adapter_report(&rep);
        ex_check("G2 零 FATAL", rep.law != 0, 0);
        ring_destroy(r);
    }

    /* ---- G3 满 P_serial 预填 → bulk 零停顿 ---- */
    {
        uint32_t ser[] = { 3000, 3000, 3000 }, ng[] = { 4, 4, 4 },
                 blk[] = { 300, 300, 300 };
        double te; uint64_t by;
        double stall = run_trace(3, ser, ng, blk, &te, &by, 0);
        /* 4 组×512KiB@55GB/s ≈ 38µs ≪ 3000µs serial → 理论零停顿 */
        ex_check("G3 bulk 零停顿 (stall<1µs)", !(stall >= 0 && stall < 1.0), 0);
        ex_check("G3 T_eff>0", !(te > 0), 0);
    }

    /* ---- G4 F2 门铃丢失 → 孤儿 → re-kick 救回 ---- */
    {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        ring_policy *r = ring_create(&eng, N_SLOTS, RING_MAX_DESCS_SLOT,
                                     SLOT_BYTES, slots);
        fill_group(0);
        ring_enqueue(r, src_of(0), GROUP_BYTES, 1, GROUP_BYTES, GROUP_BYTES);
        uint64_t slot = 0;
        ring_need(r, 0, 60000, &slot);
        ex_check("G4 组0 数据 READY", !check_group(0, slot), 0);
        ring_release(r, 0);
        sim_advance(5.0);                            /* 跨 retire 窗 → 真 IDLE */
        sim_adapter_fault(SIM_FAULT_F2_DOORBELL_LOST);
        fill_group(1);
        ring_enqueue(r, src_of(1), GROUP_BYTES, 1, GROUP_BYTES, GROUP_BYTES);
        /* enqueue 的 dmstart 被吞: 描述符已 SUBMIT 但引擎未醒 → 孤儿 */
        ring_need_result nr = ring_need(r, 1, 80, &slot);
        ex_check("G4 孤儿 re-kick 后 READY", nr != RING_READY, 0);
        ex_check("G4 救回数据 bit-exact", !check_group(1, slot), 0);
        ring_stats st; ring_get_stats(r, &st);
        ex_check("G4 re-kick 计数≥1", !(st.n_rekicks >= 1), 0);
        ex_check("G4 无降级", st.n_degrades != 0, 0);
        sim_report rep; sim_adapter_report(&rep);
        ex_check("G4 零 FATAL", rep.law != 0, 0);
        ring_destroy(r);
    }

    /* ---- G4c F4 假 done: 数据未落被消费者校验检出 ---- */
    {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        sim_adapter_fault(SIM_FAULT_F4_EARLY_DONE);
        ring_policy *r = ring_create(&eng, N_SLOTS, RING_MAX_DESCS_SLOT,
                                     SLOT_BYTES, slots);
        memset(slots, 0, SLOT_BYTES);                 /* 已知陈旧内容 */
        fill_group(5);
        ring_enqueue(r, src_of(5), GROUP_BYTES, 1, GROUP_BYTES, GROUP_BYTES);
        uint64_t slot = 0;
        ring_need_result nr = ring_need(r, 0, 60000, &slot);
        ex_check("G4c F4 假 done 仍报 READY", nr != RING_READY, 0);
        ex_check("G4c 数据未落被 memcmp 检出", check_group(5, slot), 0);
        ring_destroy(r);
    }

    /* ---- G5 Law 断言自证: 合法零 FATAL + RETIRE 窗 dmstart 必被抓 ---- */
    {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        ring_policy *r = ring_create(&eng, N_SLOTS, RING_MAX_DESCS_SLOT,
                                     SLOT_BYTES, slots);
        fill_group(0);
        ring_enqueue(r, src_of(0), GROUP_BYTES, 1, GROUP_BYTES, GROUP_BYTES);
        uint64_t slot = 0;
        ring_need(r, 0, 60000, &slot);
        ring_release(r, 0);
        sim_report rep; sim_adapter_report(&rep);
        ex_check("G5a canonical submit 零 FATAL", rep.law != 0, 0);
        ring_destroy(r);

        /* F1 类: 链完成停在 RETIRE 窗内, dmpoll 说谎 IDLE, 此时 dmstart
         * → 断言层必须报 law6 (RETIRE 窗内门铃=损坏) */
        sim_adapter_init(sim_default_params(), &eng);
        eng_desc *pool = eng.pool_alloc(1, 1);
        pool[0].next = NULL; pool[0].done = 0;
        pool[0].src = src_of(0);
        pool[0].dst = (uint64_t)(uintptr_t)slots;
        pool[0].bytes = 4096;
        eng.dmstart(pool);                        /* 合法起步 (IDLE) */
        sim_advance(0.5 + 4096.0 / (55.0 * 1000.0) + 0.3);  /* 落 RETIRE 窗 */
        uint32_t lie = eng.dmpoll();
        ex_check("G5b dmpoll 在 RETIRE 谎报 IDLE", lie != ENG_STATUS_IDLE, 0);
        int rc = eng.dmstart(pool);               /* 违规门铃 */
        sim_report rep2; sim_adapter_report(&rep2);
        ex_check("G5c F1 注入被断言捕获 (law6)",
                 !(rep2.law == 6 && rc == -1), 0);
        eng.pool_free(pool);
    }

    /* ---- G6 轨迹 T_eff vs oracle 上界 ---- */
    {
        /* serial_heavy: 8 段 (serial + 4 组 bulk), 组 512KiB */
        uint32_t sh_ser[8] = { 820, 310, 700, 250, 900, 400, 600, 350 };
        uint32_t sh_ng [8] = { 4, 4, 4, 4, 4, 4, 4, 4 };
        uint32_t sh_blk[8] = { 200, 200, 200, 200, 200, 200, 200, 200 };
        /* burst_after_starve: 5ms 饥饿后 13 组连发 (F9) */
        uint32_t bs_ser[2] = { 5000, 400 }, bs_ng[2] = { 13, 8 },
                 bs_blk[2] = { 600, 300 };
        double te_p, te_o; uint64_t by;
        run_trace(8, sh_ser, sh_ng, sh_blk, &te_o, &by, 1);   /* oracle 先 */
        double st_p = run_trace(8, sh_ser, sh_ng, sh_blk, &te_p, &by, 0);
        ex_log("  serial_heavy: T_eff(policy)=%.1f oracle=%.1f stall=%.1fµs",
               te_p, te_o, st_p);
        ex_check("G6a serial_heavy T_eff ≥ 0.95×oracle",
                 !(te_p >= 0.95 * te_o), 0);
        run_trace(2, bs_ser, bs_ng, bs_blk, &te_o, &by, 1);
        st_p = run_trace(2, bs_ser, bs_ng, bs_blk, &te_p, &by, 0);
        ex_log("  burst_after_starve: T_eff(policy)=%.1f oracle=%.1f stall=%.1fµs",
               te_p, te_o, st_p);
        ex_check("G6b burst_after_starve T_eff ≥ 0.95×oracle",
                 !(te_p >= 0.95 * te_o), 0);
    }

    /* ---- G7 参数扫描 (F10: W_RETIRE×10 等) ---- */
    {
        int fatal = 0;
        const double bws[] = { 40, 55, 70 };
        const double rets[] = { 2.0, 20.0 };
        const unsigned nsl[] = { 2, 4, 8 };
        static uint8_t slots8[8 * SLOT_BYTES];
        for (unsigned ib = 0; ib < 3; ib++)
        for (unsigned ir = 0; ir < 2; ir++)
        for (unsigned in = 0; in < 3; in++) {
            sim_params p = *sim_default_params();
            p.bw_eng = bws[ib];
            p.w_retire = rets[ir];
            engine_adapter eng;
            sim_adapter_init(&p, &eng);
            ring_policy *r = ring_create(&eng, nsl[in], RING_MAX_DESCS_SLOT,
                                         SLOT_BYTES, slots8);
            for (uint32_t g = 0; g < 24; g++) {
                fill_group(g);
                ring_enqueue(r, src_of(g), GROUP_BYTES, 1, GROUP_BYTES,
                             GROUP_BYTES);
                uint64_t slot = 0;
                if (ring_need(r, g, 60000, &slot) != RING_READY) { fatal++; break; }
                ring_release(r, g);
            }
            sim_report rep; sim_adapter_report(&rep);
            if (rep.law) fatal++;
            ring_destroy(r);
        }
        ex_check("G7 18 组参数扫描零 FATAL 零失败", fatal != 0, 0);
    }

    /* ---- G8 门铃经济性 ---- */
    {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        ring_policy *r = ring_create(&eng, N_SLOTS, RING_MAX_DESCS_SLOT,
                                     SLOT_BYTES, slots);
        for (uint32_t g = 0; g < 64; g++) {
            fill_group(g);
            ring_enqueue(r, src_of(g), GROUP_BYTES, 1, GROUP_BYTES,
                         GROUP_BYTES);
            uint64_t slot = 0;
            ring_need(r, g, 60000, &slot);
            ring_release(r, g);
        }
        ring_stats st; ring_get_stats(r, &st);
        double per = (double)st.n_doorbells / (double)st.n_groups;
        ex_log("  门铃/组 = %.3f (%llu/%llu)", per,
               (unsigned long long)st.n_doorbells,
               (unsigned long long)st.n_groups);
        ex_check("G8 门铃/组 ≤ 2", !(per <= 2.0), 0);
        ex_check("G8 门铃/组均值 ≤ 1.5", !(per <= 1.5), 0);
        ring_destroy(r);
    }

    /* ---- G9 law2: parked tail dmlink 只执行一条即再 park ---- */
    {
        engine_adapter eng;
        sim_adapter_init(sim_default_params(), &eng);
        eng_desc *pool = eng.pool_alloc(1, 4);
        fill_group(0); fill_group(1); fill_group(2);
        pool[0].next = NULL; pool[0].done = 0;
        pool[0].src = src_of(0);
        pool[0].dst = (uint64_t)(uintptr_t)slots;
        pool[0].bytes = 4096;
        eng.dmstart(&pool[0]);                 /* law1 全链 walk → park 在 d0 */
        sim_advance(100.0);                    /* 完成 + RETIRE 关闭 → 真 IDLE */
        ex_check("G9 d0 done", pool[0].done != 1, 0);

        pool[1].next = &pool[2]; pool[1].done = 0;   /* d1→d2 预链 */
        pool[1].src = src_of(1);
        pool[1].dst = (uint64_t)(uintptr_t)(slots + SLOT_BYTES);
        pool[1].bytes = 4096;
        pool[2].next = NULL; pool[2].done = 0;
        pool[2].src = src_of(2);
        pool[2].dst = (uint64_t)(uintptr_t)(slots + 2 * SLOT_BYTES);
        pool[2].bytes = 4096;
        eng.dmlink(&pool[0], &pool[1]);        /* law2: 只执行 d1 */
        sim_advance(100.0);
        ex_check("G9 law2 d1 执行", pool[1].done != 1, 0);
        ex_check("G9 law2 d2 未执行 (单条即 park)", pool[2].done != 0, 0);

        eng.dmlink(&pool[1], &pool[2]);        /* 再唤醒: 执行 d2 */
        sim_advance(100.0);
        ex_check("G9 law2 d2 二次唤醒执行", pool[2].done != 1, 0);

        sim_report rep; sim_adapter_report(&rep);
        ex_check("G9 零 FATAL", rep.law != 0, 0);
        eng.pool_free(pool);
    }

    return ex_summary();
}
