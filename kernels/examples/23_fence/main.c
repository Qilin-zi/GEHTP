/*
 * 23_fence — U9 方向对偶 cache fence 设备验证
 * =====================================================================
 * 判据:
 *   1) 决策表: 32 个 (写,读,域) 组合 fence_op_for 与内置期望表全等,
 *      HMX×DDR 恒拒 (FO_INVALID)
 *   2) CPU→DMA (DDR): 200 轮变长 LCG 模式, fence 边界下 DDR→VTCM 逐字节一致
 *   3) CPU→HMX→CPU (VTCM): W4A16 引擎 20 轮同输入 bit-exact + 异输入输出必异
 *   4) CPU→HVX (VTCM): memset 后 HVX 校验和模式敏感 + 往返复现 (可见性)
 *   5) CPU→DMA (VTCM, move_back 反向): VTCM→DDR 逐字节一致 (src FLUSH 契约)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define A  "/data/local/tmp/hvxhmx23/assets/s256"
#define M 256u
#define RB (4u * 1024u)

static uint32_t lcg = 20260823u;
static uint32_t lcg_n(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return lcg;
}

int main(void) {
    ex_open_result("23_fence");

    /* ---- G1 决策表全等 ---- */
    {
        /* 期望表 [writer][reader][mem] — 与 fence.h 语义表逐格对应 */
        /* [writer][reader][mem]; writer=HVX 与 CPU 同行 (dcache 代理) */
        static const int exp[4][4][2] = {
            /* CPU (及 HVX) */
            { {FO_NONE, FO_NONE}, {FO_FLUSH_INVALIDATE, FO_FLUSH},
              {FO_INVALID, FO_FLUSH}, {FO_NONE, FO_NONE} },
            /* DMA */
            { {FO_INVALIDATE, FO_INVALIDATE}, {FO_FLUSH, FO_FLUSH},
              {FO_INVALID, FO_INVALIDATE}, {FO_INVALIDATE, FO_INVALIDATE} },
            /* HMX */
            { {FO_INVALID, FO_INVALIDATE}, {FO_INVALID, FO_FLUSH},
              {FO_INVALID, FO_INVALID}, {FO_INVALID, FO_INVALIDATE} },
            /* HVX = CPU 行 */
            { {FO_NONE, FO_NONE}, {FO_FLUSH_INVALIDATE, FO_FLUSH},
              {FO_INVALID, FO_FLUSH}, {FO_NONE, FO_NONE} },
        };
        int bad = 0;
        for (int w = 0; w < 4; w++)
            for (int r = 0; r < 4; r++)
                for (int m = 0; m < 2; m++) {
                    int got = fence_op_for(w, r, m);
                    if (got != exp[w][r][m]) {
                        ex_log("  table w%d r%d m%d: got %d exp %d", w, r, m, got, exp[w][r][m]);
                        bad++;
                    }
                }
        ex_check("fence_decision_table_32", bad, 0);

        /* 非法组合运行时拒绝 */
        int rej = 0;
        uint8_t dummy[128];
        rej += (fence_handoff(dummy, 128, FC_HMX, FC_CPU, FM_DDR) != FENCE_ERR_COMBO);
        rej += (fence_handoff(dummy, 128, FC_CPU, FC_HMX, FM_DDR) != FENCE_ERR_COMBO);
        rej += (fence_handoff(dummy, 128, FC_DMA, FC_DMA, FM_DDR) == FENCE_ERR_COMBO); /* 合法: FLUSH */
        ex_check("fence_invalid_combo_rejected", rej, 0);
    }

    /* ---- 会话 (VTCM 引擎 + arena) ---- */
    struct wtcache_ctx* wc = NULL;
    int rc = wtcache_open(&wc, 4096);
    if (rc != WTC_OK) { ex_log("!! wtcache_open 0x%X", rc); ex_summary(); return 1; }
    void* vb = NULL; uint32_t vs = 0; void* pb = NULL; uint32_t pc = 0;
    wtcache_layout(wc, &vb, &vs, &pb, &pc);
    uint32_t off = (pc + 2047u) & ~2047u;
    struct dc_arena ar;
    dc_arena_init(&ar, (uint8_t*)vb + off, vs - off);
    uint8_t* vt_slot = dc_arena_alloc(&ar, RB, 2048);
    uint8_t* vt_scr  = dc_arena_alloc(&ar, RB, 128);
    dc_mutex_t mu; dc_mutex_init(&mu);

    /* ---- G2 CPU→DMA (DDR): 200 轮变长 ---- */
    {
        uint8_t* ddr = memalign(128, RB);
        uint8_t* shadow = malloc(RB);
        int bad = 0;
        for (int it = 0; it < 200; it++) {
            uint32_t bytes = 128u * (1u + lcg_n() % 32u);   /* 128B..4KB, 128 倍数 */
            for (uint32_t i = 0; i < bytes; i++) ddr[i] = (uint8_t)lcg_n();
            memcpy(shadow, ddr, bytes);
            if (fence_handoff(ddr, bytes, FC_CPU, FC_DMA, FM_DDR)) { bad++; continue; }
            struct dc_dma d;
            if (dc_dma_init(&d, ddr, vt_slot, bytes, &mu) == 0) {
                if (dc_dma_once(&d)) bad++;
                dc_dma_destroy(&d);
            } else bad++;
            fence_handoff(vt_slot, bytes, FC_DMA, FC_CPU, FM_VTCM);
            if (memcmp(vt_slot, shadow, bytes) != 0) bad++;
        }
        ex_check("fence_cpu_dma_ddr_200iters", bad, 0);
        free(ddr); free(shadow);
    }

    /* ---- G3 CPU→HMX→CPU (VTCM): W4A16 20 轮 ---- */
    {
        uint32_t b;
        uint8_t* wt = dc_read_file(A "/packed_weight.raw", &b);
        uint8_t* bias = dc_read_file(A "/folded_bias.raw", &b);
        uint8_t* atbl = dc_read_file(A "/act_table.raw", &b);
        uint8_t* otbl = dc_read_file(A "/out_table.raw", &b);
        uint8_t* act = dc_read_file(A "/act_surface.raw", &b);
        if (!wt || !bias || !atbl || !otbl || !act) {
            ex_log("!! assets missing"); ex_check("fence_cpu_hmx_setup", 1, 0);
        } else {
            struct dc_w4 e;
            if (dc_w4_carve(&e, &ar, M, M, M, atbl, otbl)) {
                ex_check("fence_cpu_hmx_carve", 1, 0);
            } else {
                memcpy(e.wt, wt, M * M / 2u);
                memcpy(e.bias, bias, (M / 32u) * 512u);
                qurt_mem_cache_clean((qurt_addr_t)e.wt, M * M / 2u,
                                     QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
                qurt_mem_cache_clean((qurt_addr_t)e.bias, (M / 32u) * 512u,
                                     QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
                uint32_t act_b = M * M * 2u, out_b = M * M * 2u;
                uint8_t* ref = malloc(out_b);
                uint8_t* cur = malloc(out_b);
                int bad = 0, ref_set = 0;
                for (int it = 0; it < 20; it++) {
                    /* 同输入; it==7 用旋转输入验证敏感性 (不参与一致性计数) */
                    const uint8_t* src = (it == 7) ? act + 256 : act;
                    memcpy(e.act, src, act_b);
                    fence_handoff(e.act, act_b, FC_CPU, FC_HMX, FM_VTCM);
                    if (dc_w4_invoke(&e)) { bad++; break; }
                    fence_handoff(e.out, out_b, FC_HMX, FC_CPU, FM_VTCM);
                    memcpy(cur, e.out, out_b);
                    if (!ref_set) { memcpy(ref, cur, out_b); ref_set = 1; }
                    else if (it != 7 && memcmp(cur, ref, out_b) != 0) bad++;
                }
                /* 敏感轮必须与 ref 不同 (fence 无副作用 + 引擎真在算) */
                int sens_differs = 0;
                {
                    memcpy(e.act, act + 256, act_b);
                    fence_handoff(e.act, act_b, FC_CPU, FC_HMX, FM_VTCM);
                    dc_w4_invoke(&e);
                    fence_handoff(e.out, out_b, FC_HMX, FC_CPU, FM_VTCM);
                    sens_differs = memcmp(e.out, ref, out_b) != 0;
                }
                ex_check("fence_cpu_hmx_vtcm_20iters", bad + (ref_set ? 0 : 1), 0);
                ex_check("fence_hmx_input_sensitivity", sens_differs ? 0 : 1, 0);
                free(ref); free(cur);
            }
        }
        free(wt); free(bias); free(atbl); free(otbl); free(act);
    }

    /* ---- G4 CPU→HVX (VTCM): 校验和可见性 ---- */
    {
        uint32_t csA1, csA2, csB;
        /* 均匀图案 XOR 折叠恒 0 (校验和天然碰撞) → 必须非均匀图案 */
        for (uint32_t i = 0; i < RB; i++) vt_scr[i] = (uint8_t)(i * 7 + 1);
        fence_handoff(vt_scr, RB, FC_CPU, FC_HVX, FM_VTCM);
        csA1 = dc_hvx_load(vt_scr, 1);
        for (uint32_t i = 0; i < RB; i++) vt_scr[i] = (uint8_t)(i * 13 + 5);
        fence_handoff(vt_scr, RB, FC_CPU, FC_HVX, FM_VTCM);
        csB = dc_hvx_load(vt_scr, 1);
        for (uint32_t i = 0; i < RB; i++) vt_scr[i] = (uint8_t)(i * 7 + 1);
        fence_handoff(vt_scr, RB, FC_CPU, FC_HVX, FM_VTCM);
        csA2 = dc_hvx_load(vt_scr, 1);
        int ok = (csA1 == csA2) && (csA1 != csB);
        ex_check("fence_cpu_hvx_vtcm_visible", ok ? 0 : 1, 0);
    }

    /* ---- G5 CPU→DMA (VTCM, 反向 move_back) ---- */
    {
        uint8_t* ddr_out = memalign(128, RB);
        uint8_t* shadow = malloc(RB);
        int bad = 0;
        for (int it = 0; it < 100; it++) {
            uint32_t bytes = 128u * (1u + lcg_n() % 32u);
            for (uint32_t i = 0; i < bytes; i++) vt_slot[i] = (uint8_t)lcg_n();
            memcpy(shadow, vt_slot, bytes);
            fence_handoff(vt_slot, bytes, FC_CPU, FC_DMA, FM_VTCM);  /* src FLUSH */
            struct dc_dma d;
            if (dc_dma_init(&d, vt_slot, ddr_out, bytes, &mu) == 0) {
                if (dc_dma_once(&d)) bad++;
                dc_dma_destroy(&d);
            } else bad++;
            fence_handoff(ddr_out, bytes, FC_DMA, FC_CPU, FM_DDR);
            if (memcmp(ddr_out, shadow, bytes) != 0) bad++;
        }
        ex_check("fence_cpu_dma_vtcm_moveback_100iters", bad, 0);
        free(ddr_out); free(shadow);
    }

    wtcache_close(wc);
    return ex_summary();
}
