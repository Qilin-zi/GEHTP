/*
 * 29_kvcache — U15 KV 槽缓存管理 设备验证
 * =====================================================================
 * 判据 (nslots=64, slot 256B = K 128B + V 128B, host 影子模型逐项对拍):
 *   1) append 64 pos: 逐槽 K/V 字节 + posmap 与 host 模型一致, lookup 全 hit
 *   2) 回绕位置动态 (posIdsIdx): 总写 3N, 旧 pos (<2N) miss, 新 pos hit,
 *      evict 计数精确
 *   3) scatter 重写隔离: 只动本槽字节 + 本槽 posmap, 邻槽金丝雀完好;
 *      非法槽号拒绝
 *   4) 槽 128B 对齐恒成立 (K 面 + V 面)
 *   5) DMA bypass 读回 (铁律① 真实验证): append 内置 fence 后
 *      dc_dma_once 读槽 K 面 → 与影子逐字节一致
 *   6) verify 重写语义: scatter 同槽 (target 校正) 后 read = 校正值,
 *      lookup(pos) 仍 hit
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define NS 64u
#define SB 256u          /* K 128 + V 128 */
#define TOTAL 192u       /* 3N append */

static uint8_t* ksrc;    /* [TOTAL][128] 影子 K */
static uint8_t* vsrc;

static void gen(uint32_t lcg_state, uint32_t from, uint32_t to) {
    uint32_t l = lcg_state;
    for (uint32_t p = from; p < to; p++)
        for (int i = 0; i < 128; i++) {
            ksrc[p * 128 + i] = (uint8_t)(gdn_lcg_next(&l) >> 24);
            vsrc[p * 128 + i] = (uint8_t)(gdn_lcg_next(&l) >> 16);
        }
}

int main(void) {
    ex_open_result("29_kvcache");
    struct wtcache_ctx* wc = NULL;
    uint8_t* rk = NULL, * rv = NULL, * snap = NULL;
    ksrc = malloc(TOTAL * 128); vsrc = malloc(TOTAL * 128);
    uint8_t* base = memalign(128, NS * SB);
    uint32_t* posmap_sh = malloc(NS * 4);
    rk = memalign(128, 128); rv = memalign(128, 128);
    struct kvc c;
    if (!ksrc || !vsrc || !base || !posmap_sh || !rk || !rv) {
        ex_log("alloc FAIL"); goto out;
    }
    for (uint32_t i = 0; i < NS; i++) posmap_sh[i] = KVC_POS_NONE;
    int inj = (kvc_init(&c, base, SB - 8, NS, posmap_sh) == 0);   /* 非 128 倍数必须拒 */
    ex_check("init_rejects_nonmult128", inj, 0);
    if (inj || kvc_init(&c, base, SB, NS, posmap_sh)) { ex_log("kvc_init FAIL"); goto out; }
    kvc_set_consumer(&c, FC_DMA, FM_DDR);       /* append 内置 fence 走铁律① */
    gen(20260830u, 0, TOTAL);

    /* G1 append 64 + 影子逐槽对拍 */
    int g1 = 0;
    for (uint32_t p = 0; p < NS; p++) {
        int s = kvc_append(&c, ksrc + p * 128, vsrc + p * 128, p);
        posmap_sh[p % NS] = p;
        if (s != (int)(p % NS)) g1++;
    }
    for (uint32_t p = 0; p < NS; p++) {
        int s = kvc_lookup(&c, p);
        if (s < 0) { g1++; continue; }
        kvc_read(&c, (uint32_t)s, rk, rv);
        if (memcmp(rk, ksrc + p * 128, 128) || memcmp(rv, vsrc + p * 128, 128)) g1++;
        if (kvc_pos(&c, (uint32_t)s) != posmap_sh[s]) g1++;
    }
    ex_check("append_64_readback_bytewise", g1, 0);
    ex_log("  G1 64 槽 append/查读/posmap: %d mismatch", g1);

    /* G2 回绕位置动态 */
    int g2 = 0;
    for (uint32_t p = NS; p < TOTAL; p++) {
        kvc_append(&c, ksrc + p * 128, vsrc + p * 128, p);
        posmap_sh[p % NS] = p;
    }
    for (uint32_t p = 0; p < TOTAL - NS; p++)
        if (kvc_lookup(&c, p) >= 0) g2++;               /* 旧 pos 必须 miss */
    for (uint32_t p = TOTAL - NS; p < TOTAL; p++) {
        int s = kvc_lookup(&c, p);
        if (s < 0) { g2++; continue; }
        kvc_read(&c, (uint32_t)s, rk, rv);
        if (memcmp(rk, ksrc + p * 128, 128) || memcmp(rv, vsrc + p * 128, 128)) g2++;
    }
    if (c.n_append != TOTAL) g2++;
    ex_check("wrap_posids_miss_old_hit_new", g2, 0);
    ex_log("  G2 回绕 (3N append): miss/hit/内容 %d err, evict=%u (期望 %u)",
           g2, c.n_evict, TOTAL - NS);
    ex_check("evict_count_exact", c.n_evict == TOTAL - NS ? 0 : 1, 0);

    /* G3 scatter 重写隔离 */
    int g3 = 0;
    snap = malloc(NS * SB);
    memcpy(snap, base, NS * SB);
    uint32_t snapmap[NS];
    memcpy(snapmap, c.posmap, NS * 4);
    const uint32_t slot5 = 5;
    kvc_scatter(&c, slot5, ksrc + 0, vsrc + 0, 773);     /* 重写槽 5 */
    for (uint32_t s = 0; s < NS; s++) {
        if (s == slot5) continue;
        if (memcmp(base + s * SB, snap + s * SB, SB)) g3++;
        if (c.posmap[s] != snapmap[s]) g3++;
    }
    kvc_read(&c, slot5, rk, rv);
    if (memcmp(rk, ksrc, 128) || memcmp(rv, vsrc, 128)) g3++;
    if (kvc_lookup(&c, 773) != (int)slot5) g3++;
    if (kvc_scatter(&c, NS, ksrc, vsrc, 1) != -1) g3++;  /* 非法槽拒绝 */
    ex_check("scatter_rewrite_isolation", g3, 0);

    /* G4 槽对齐 */
    int g4 = 0;
    for (uint32_t s = 0; s < NS; s++)
        if ((((uintptr_t)kvc_k_face(&c, s)) | ((uintptr_t)kvc_v_face(&c, s))) & 127u) g4++;
    ex_check("slot_alignment_128", g4, 0);

    /* G5 DMA bypass 读回 (铁律①): 需要 VTCM dst → wtcache */
    if (wtcache_open(&wc, 4096) != WTC_OK) { ex_log("wtcache_open FAIL"); goto out; }
    {
        void* vb; uint32_t vs; void* pb; uint32_t pc;
        wtcache_layout(wc, &vb, &vs, &pb, &pc);
        struct dc_arena ar;
        dc_arena_init(&ar, (uint8_t*)vb + ((pc + 2047u) & ~2047u),
                      vs - ((pc + 2047u) & ~2047u));
        uint8_t* dst = dc_arena_alloc(&ar, 128, 2048);
        if (!dst) { ex_log("dst alloc FAIL"); goto out; }
        const uint32_t slot7 = 7;
        uint32_t p7 = posmap_sh[slot7];
        dc_mutex_t mu;
        dc_mutex_init(&mu);
        struct dc_dma d;
        if (dc_dma_init(&d, kvc_k_face(&c, slot7), dst, 128, &mu)) {
            ex_log("dc_dma_init FAIL"); goto out;
        }
        int g5 = dc_dma_once(&d) || memcmp(dst, ksrc + p7 * 128, 128);
        ex_check("dma_roundtrip_bypass_read", g5, 0);
        ex_log("  G5 DMA bypass 读槽 %u (pos %u): %s", slot7, p7,
               g5 ? "STALE/FAIL" : "逐字节一致");
        dc_dma_destroy(&d);

        /* G6 verify 重写语义: draft 填 pos 900 → target scatter 校正同槽 */
        int g6 = 0;
        uint32_t sp = 900;
        int s9 = kvc_append(&c, ksrc + 128, vsrc + 128, sp);
        if (s9 != (int)(sp % NS)) g6++;
        kvc_scatter(&c, (uint32_t)s9, ksrc + 256, vsrc + 256, sp);
        kvc_read(&c, (uint32_t)s9, rk, rv);
        if (memcmp(rk, ksrc + 256, 128) || memcmp(rv, vsrc + 256, 128)) g6++;
        if (kvc_lookup(&c, sp) != s9) g6++;
        ex_check("verify_rewrite_posids_semantics", g6, 0);
    }
out:
    if (rk) free(rk); if (rv) free(rv); if (snap) free(snap);
    if (wc) wtcache_close(wc);
    return ex_summary();
}
