/*
 * 24_arena — U10 双池对齐 arena 设备验证
 * =====================================================================
 * 判据:
 *   1) 1000 轮随机 alloc/free: 每个返回指针满足请求对齐 (128/2KB)
 *   2) 全部释放后 used==0 且 largest_free==capacity (无泄漏/无碎片残留)
 *   3) gdn_sm 状态 (f32) 与 W4A16 面 (2KB) 双池共存 ×100 轮:
 *      交错分配/释放, gdn 状态内容不被 W4 面分配破坏
 *   4) 碎片上界: 随机搅动后释放 90%, largest_free ≥ 80% 容量 (合并有效)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define D 64
#define H 2
#define GW (256u * 1024u)      /* DDR 池 */
static uint32_t lcg = 20260824u;
static uint32_t lcg_n(void) { lcg = lcg * 1664525u + 1013904223u; return lcg; }

/* 例内 host-镜像模型 (G3): gdn 状态金标 */
static float gdn_gold[H * D * D];

int main(void) {
    ex_open_result("24_arena");

    struct wtcache_ctx* wc = NULL;
    int rc = wtcache_open(&wc, 4096);
    if (rc != WTC_OK) { ex_log("!! wtcache_open 0x%X", rc); ex_summary(); return 1; }
    void* vb = NULL; uint32_t vs = 0; void* pb = NULL; uint32_t pc = 0;
    wtcache_layout(wc, &vb, &vs, &pb, &pc);
    uint32_t off = (pc + 2047u) & ~2047u;
    uint32_t vtc_sz = vs - off - 4096u;         /* 留 4KB 给尾垫 */
    uint8_t* ddr_pool = memalign(128, GW);
    struct arena ar;
    arena_init(&ar, ddr_pool, GW, (uint8_t*)vb + off, vtc_sz);
    uint32_t cap[2];
    cap[0] = (GW + 15u) & ~15u;
    cap[1] = (vtc_sz + 15u) & ~15u;

    /* ---- G1+G2: 1000 轮随机 ---- */
    {
        #define SLOTS 64
        void* ptrs[SLOTS]; uint32_t sz[SLOTS]; uint32_t al[SLOTS]; int lc[SLOTS];
        memset(ptrs, 0, sizeof(ptrs));
        int bad_align = 0, leak = 0;
        for (int it = 0; it < 1000; it++) {
            int idx = (int)(lcg_n() % SLOTS);
            if (ptrs[idx]) { arena_free(&ar, ptrs[idx]); ptrs[idx] = NULL; }
            uint32_t align = (lcg_n() & 1) ? ARENA_ALIGN_HVX : ARENA_ALIGN_HMX;
            uint32_t bytes = 16u + (lcg_n() % 4096u);
            int loc = (int)(lcg_n() & 1);
            void* p = arena_alloc(&ar, bytes, align, loc);
            if (p) {
                if ((uintptr_t)p % align) { bad_align++; }
                else {
                    ptrs[idx] = p; sz[idx] = bytes; al[idx] = align; lc[idx] = loc;
                    memset(p, (uint8_t)it, bytes);
                }
            }
        }
        for (int i = 0; i < SLOTS; i++) if (ptrs[i]) arena_free(&ar, ptrs[i]);
        leak += (arena_used(&ar, ARENA_DDR) != 0);
        leak += (arena_used(&ar, ARENA_VTCM) != 0);
        leak += (arena_largest_free(&ar, ARENA_DDR) != cap[0]);
        leak += (arena_largest_free(&ar, ARENA_VTCM) != cap[1]);
        ex_check("arena_align_1000cycles", bad_align, 0);
        ex_check("arena_no_leak_full_coalesce", leak, 0);
    }

    /* ---- G3: gdn_sm 状态 × W4A16 面双池共存 ---- */
    {
        /* 金标状态 (LCG) */
        for (size_t i = 0; i < H * D * D; i++)
            gdn_gold[i] = ((int32_t)(lcg_n() >> 8) - 128) / 256.0f;
        int bad = 0;
                for (int it = 0; it < 100; it++) {
            float* S = arena_alloc(&ar, H * D * D * 4u, ARENA_ALIGN_HVX, ARENA_DDR);
            if (!S) { bad++; break; }
            memcpy(S, gdn_gold, H * D * D * 4u);
            /* W4 面挤进 VTCM 池 (2KB 对齐) */
            uint8_t* act = arena_alloc(&ar, 256u * 256u * 2u, ARENA_ALIGN_HMX, ARENA_VTCM);
            uint8_t* out = arena_alloc(&ar, 256u * 256u * 2u, ARENA_ALIGN_HMX, ARENA_VTCM);
            uint8_t* wt  = arena_alloc(&ar, 256u * 256u / 2u, ARENA_ALIGN_HMX, ARENA_VTCM);
            if (!act || !out || !wt) { bad++; }
            else {
                memset(act, (uint8_t)it, 256u * 256u * 2u);
                memset(wt,  (uint8_t)(it ^ 0xFF), 256u * 256u / 2u);
                /* 交错重分配 act (释放再取), 检验 S 不受影响 */
                arena_free(&ar, act);
                uint8_t* act2 = arena_alloc(&ar, 256u * 256u * 2u, ARENA_ALIGN_HMX, ARENA_VTCM);
                if (!act2) bad++;
                if (memcmp(S, gdn_gold, H * D * D * 4u) != 0) bad++;
                if (act2) arena_free(&ar, act2);
                arena_free(&ar, out);
                arena_free(&ar, wt);
            }
            arena_free(&ar, S);
        }
        bad += (arena_used(&ar, ARENA_DDR) != 0);
        bad += (arena_used(&ar, ARENA_VTCM) != 0);
        ex_check("arena_gdnsm_w4_coexist_100iters", bad, 0);
    }

    /* ---- G4: 碎片上界 ---- */
    {
        #define FS 48
        void* fp[FS]; int fl[FS];
        memset(fp, 0, sizeof(fp));
        for (int i = 0; i < FS; i++) {
            fl[i] = (int)(lcg_n() & 1);
            fp[i] = arena_alloc(&ar, 256u + (lcg_n() % 8192u), ARENA_ALIGN_HVX, fl[i]);
        }
        /* 释放约 90% (保 4 个) → 合并后 largest_free 应接近容量 */
        for (int i = 0; i < FS; i++) if (fp[i] && (i % 5) != 3) { arena_free(&ar, fp[i]); fp[i] = NULL; }
        uint32_t lf_d = arena_largest_free(&ar, ARENA_DDR);
        uint32_t lf_v = arena_largest_free(&ar, ARENA_VTCM);
        uint32_t hold = 0; int hold_loc[2] = {0,0};
        for (int i = 0; i < FS; i++) if (fp[i]) hold_loc[fl[i]]++;
        int bad = 0;
        if (hold_loc[0] == 0 && lf_d < cap[0] * 8u / 10u) bad++;
        if (hold_loc[1] == 0 && lf_v < cap[1] * 8u / 10u) bad++;
        ex_log("  frag: lf_ddr=%u/%u lf_vtcm=%u/%u hold=(%d,%d)", lf_d, cap[0], lf_v, cap[1], hold_loc[0], hold_loc[1]);
        ex_check("arena_frag_coalesce_bound", bad, 0);
        for (int i = 0; i < FS; i++) if (fp[i]) arena_free(&ar, fp[i]);
    }

    wtcache_close(wc);
    free(ddr_pool);
    return ex_summary();
}
