/* gtest_v23.c — V2.3 host 模拟 (纯 C 单元: fence/arena/kvcache/pxbridge/gdn_tree/oplist)
 * 设备实跑纪律: 上机前 host 全绿。gcc -o gtest_v23 ... && ./gtest_v23
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "fence.h"
#include "arena.h"
#include "kvcache.h"
#include "pxbridge.h"
#include "gdn_tree.h"
#include "gdn_sm.h"
#include "oplist_parse.h"
#include "rbr.h"
#include "bledger.h"

static int npass = 0, nfail = 0;
static void ck(const char* name, int err, int tol) {
    if (err <= tol) { printf("[PASS] %s err=%d\n", name, err); npass++; }
    else { printf("[FAIL] %s err=%d\n", name, err); nfail++; }
}
static double cos_sim(const float* a, const float* b, size_t n) {
    double p = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        p += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}

static void t_fence(void) {
    int bad = 0;
    struct { int w, r, m, exp; } T[] = {
        {FC_CPU,FC_DMA,FM_DDR,FO_FLUSH_INVALIDATE}, {FC_CPU,FC_DMA,FM_VTCM,FO_FLUSH},
        {FC_CPU,FC_HMX,FM_VTCM,FO_FLUSH}, {FC_CPU,FC_CPU,FM_DDR,FO_NONE},
        {FC_CPU,FC_HVX,FM_DDR,FO_NONE}, {FC_DMA,FC_CPU,FM_DDR,FO_INVALIDATE},
        {FC_DMA,FC_HVX,FM_DDR,FO_INVALIDATE}, {FC_DMA,FC_HMX,FM_VTCM,FO_INVALIDATE},
        {FC_DMA,FC_DMA,FM_VTCM,FO_FLUSH}, {FC_HMX,FC_CPU,FM_VTCM,FO_INVALIDATE},
        {FC_HMX,FC_HVX,FM_VTCM,FO_INVALIDATE}, {FC_HMX,FC_DMA,FM_VTCM,FO_FLUSH},
        {FC_HVX,FC_CPU,FM_DDR,FO_NONE}, {FC_HVX,FC_DMA,FM_DDR,FO_FLUSH_INVALIDATE},
        {FC_HMX,FC_CPU,FM_DDR,FO_INVALID}, {FC_DMA,FC_DMA,FM_DDR,FO_FLUSH},
    };
    for (unsigned i = 0; i < sizeof(T)/sizeof(T[0]); i++)
        if (fence_op_for(T[i].w, T[i].r, T[i].m) != T[i].exp) bad++;
    ck("host_fence_table", bad, 0);
}

static void t_arena(void) {
    enum { CAP = 1u << 20 };
    uint8_t* ddr = aligned_alloc(128, CAP);
    struct arena a;
    arena_init(&a, ddr, CAP, NULL, 0);
    uint32_t lcg = 7;
    void* slots[64]; uint32_t sz[64], al[64]; int used[64];
    int bad = 0;
    memset(used, 0, sizeof(used));
    for (int r = 0; r < 1000; r++) {
        int i = (lcg = lcg * 1664525u + 1013904223u) % 64;
        if (used[i]) { arena_free(&a, slots[i]); used[i] = 0; }
        else {
            sz[i] = 64 + ((lcg = lcg * 1664525u + 1013904223u) % 4096);
            al[i] = 128u << ((lcg >> 17) % 4);
            slots[i] = arena_alloc(&a, sz[i], al[i], ARENA_DDR);
            if (!slots[i]) { bad++; break; }
            if (((uintptr_t)slots[i] & (al[i] - 1))) bad++;
            used[i] = 1;
        }
    }
    for (int i = 0; i < 64; i++) if (used[i]) arena_free(&a, slots[i]);
    if (arena_used(&a, ARENA_DDR) != 0) bad++;
    if (arena_largest_free(&a, ARENA_DDR) != CAP) bad++;
    ck("host_arena_1000round_align_zero", bad, 0);
    free(ddr);
}

static void t_kvcache(void) {
    enum { NS = 16, SB = 256, TOT = 48 };
    uint8_t* base = aligned_alloc(128, NS * SB);
    uint32_t* pm = malloc(NS * 4);
    struct kvc c;
    kvc_init(&c, base, SB, NS, pm);
    uint8_t* ks = malloc(TOT * 128), *vs = malloc(TOT * 128);
    for (int i = 0; i < TOT * 128; i++) { ks[i] = i * 7; vs[i] = i * 13; }
    int bad = 0;
    for (uint32_t p = 0; p < TOT; p++) {
        int s = kvc_append(&c, ks + p * 128, vs + p * 128, p);
        if (s != (int)(p % NS)) bad++;
    }
    for (uint32_t p = TOT - NS; p < TOT; p++) {
        int s = kvc_lookup(&c, p);
        if (s < 0) { bad++; continue; }
        uint8_t rk[128], rv[128];
        kvc_read(&c, (uint32_t)s, rk, rv);
        if (memcmp(rk, ks + p * 128, 128) || memcmp(rv, vs + p * 128, 128)) bad++;
    }
    for (uint32_t p = 0; p < TOT - NS; p++)
        if (kvc_lookup(&c, p) >= 0) bad++;
    ck("host_kvcache_wrap_shadow", bad, 0);
    /* scatter 隔离 */
    uint8_t* snap = malloc(NS * SB); memcpy(snap, base, NS * SB);
    kvc_scatter(&c, 3, ks, vs, 999);
    bad = memcmp(base + 3 * SB, snap + 3 * SB, SB) == 0;
    for (uint32_t s = 0; s < NS; s++) if (s != 3 && memcmp(base + s * SB, snap + s * SB, SB)) bad++;
    ck("host_kvcache_scatter_isolation", bad, 0);
    if (kvc_scatter(&c, NS, ks, vs, 1) != -1) { ck("host_kvcache_invalid_slot", 1, 0); }
    else ck("host_kvcache_invalid_slot", 0, 0);
    free(base); free(pm); free(ks); free(vs); free(snap);
}

static void t_pxbridge(void) {
    int bad = 0;
    const float sc[3] = { 1e-3f, 1e-2f, 0.1f };
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < 65000; i += 7) {
            float x = i * sc[s];
            int16_t q = pxb_f32_to_i16(x, sc[s]);
            float d = pxb_i16_to_f32(q, sc[s]);
            if (fabsf(d - x) > sc[s] * 0.5f + sc[s] * 1e-6f) bad++;
            /* f16 桥: 差被 f16 半 ULP 吸收 (round 边界差 1 码合法) */
            float x16 = gdn_f16_to_f32(gdn_f32_to_f16(x));
            float d16 = pxb_i16_to_f32(pxb_f16_to_i16(gdn_f32_to_f16(x), sc[s]), sc[s]);
            if (fabsf(d16 - x16) > sc[s] * 0.5f + fabsf(x16) * 1.3e-3f + 1e-12f) bad++;
            if (pxb_f32_to_i16(-x, sc[s]) != (int16_t)0x8000) bad++;  /* 负钳零码 */
        }
    }
    if (pxb_f32_to_i16(0, 1e-2f) != (int16_t)0x8000) bad++;
    if (pxb_i16_to_f32((int16_t)0x8000, 1e-2f) != 0) bad++;
    if (pxb_f32_to_i16(1e9f, 0.1f) != (int16_t)0x7FFF) bad++;
    if (pxb_i16_to_f32((int16_t)0x7FFF, 0.1f) != 65535.0f * 0.1f) bad++;
    ck("host_pxbridge_envelope_identity", bad, 0);
}

static void t_gdn_tree(void) {
    const int Ts[3] = { 8, 16, 32 };
    const int D = 64;
    double worst_o = 1, worst_k = 1, worst_os = 1, worst_ks = 1;
    int detbad = 0, rejectbad = 0;
    for (int ti = 0; ti < 3; ti++) {
        int t = Ts[ti];
        int parent[32], dep[32];
        float S0[64 * 64], kf[32 * 64], vf[32 * 64], qf[32 * 64], bf[32], gf[32];
        int16_t k16[32*64], v16[32*64], q16[32*64], be16[32], g16[32], y16a[32*64], y16b[32*64];
        float *Sser = malloc((size_t)t * D * D * 4), *Scls = malloc((size_t)t * D * D * 4);
        float *ys = malloc((size_t)t * D * 4), *yc = malloc((size_t)t * D * 4), *yk = malloc((size_t)t * D * 4);
        for (int seed = 0; seed < 5; seed++) {
            uint32_t lcg = 20260901u + (uint32_t)(seed * 31 + ti);
            parent[0] = -1; dep[0] = 0;
            for (int i = 1; i < t; i++) {
                double u = (double)(gdn_lcg_next(&lcg) >> 8) / 16777216.0;
                parent[i] = (u < 0.35) ? i - 1 : (int)(gdn_lcg_next(&lcg) % (uint32_t)i);
            }
            for (int z = 0; z < t * D; z++) {
                kf[z] = gdn_lcg_norm(&lcg) * 0.5f; k16[z] = gdn_f32_to_f16(kf[z]);
                vf[z] = gdn_lcg_norm(&lcg) * 0.8f; v16[z] = gdn_f32_to_f16(vf[z]);
                qf[z] = gdn_lcg_norm(&lcg) * 0.5f; q16[z] = gdn_f32_to_f16(qf[z]);
            }
            for (int i = 0; i < t; i++) {
                bf[i] = fabsf(gdn_lcg_norm(&lcg)) + 0.2f; be16[i] = gdn_f32_to_f16(bf[i]);
                gf[i] = -fabsf(gdn_lcg_norm(&lcg)) * 0.5f; g16[i] = gdn_f32_to_f16(gf[i]);
            }
            for (int z = 0; z < D * D; z++) S0[z] = gdn_lcg_norm(&lcg) * 0.3f;
            ref_delta_tree(S0, D, t, parent, kf, vf, qf, bf, gf, ys, Sser);
            ref_tree_closed(S0, D, t, parent, kf, vf, qf, bf, gf, yc, Scls);
            double cy = cos_sim(ys, yc, (size_t)t * D), csm = 1;
            for (int i = 0; i < t; i++) {
                double c2 = cos_sim(Sser + (size_t)i*D*D, Scls + (size_t)i*D*D, (size_t)D*D);
                if (c2 < csm) csm = c2;
            }
            if (cy < worst_o) worst_o = cy;
            if (csm < worst_os) worst_os = csm;
            if (gdn_tree_serial_f16(S0, Sser, D, t, parent, k16, v16, q16, be16, g16, y16a)) rejectbad++;
            if (gdn_tree_serial_f16(S0, Sser, D, t, parent, k16, v16, q16, be16, g16, y16b) ||
                memcmp(y16a, y16b, (size_t)t * D * 2)) detbad++;
            for (int z = 0; z < t * D; z++) yk[z] = gdn_f16_to_f32(y16a[z]);
            double ck2 = cos_sim(yk, yc, (size_t)t * D), ckm = 1;
            for (int i = 0; i < t; i++) {
                double c2 = cos_sim(Sser + (size_t)i*D*D, Scls + (size_t)i*D*D, (size_t)D*D);
                if (c2 < ckm) ckm = c2;
            }
            if (ck2 < worst_k) worst_k = ck2;
            if (ckm < worst_ks) worst_ks = ckm;
        }
        free(Sser); free(Scls); free(ys); free(yc); free(yk);
    }
    printf("  oracle y=%.7f st=%.7f | kernel y=%.6f st=%.6f\n",
           worst_o, worst_os, worst_k, worst_ks);
    ck("host_tree_closed_vs_serial", worst_o < 0.99999 || worst_os < 0.99999, 0);
    ck("host_tree_kernel_vs_closed", worst_k < 0.999 || worst_ks < 0.999, 0);
    ck("host_tree_kernel_determinism", detbad, 0);
    int par2[4] = { 0, 0, 1, 2 }, par3[4] = { -1, 2, 1, 2 };
    float st[4 * 64 * 64], S0[64 * 64];
    int16_t k16[256], v16[256], q16[256], b16[4], g16[4], y16[256];
    int r1 = gdn_tree_serial_f16(S0, st, 64, 4, par2, k16, v16, q16, b16, g16, y16);
    int r2 = gdn_tree_serial_f16(S0, st, 64, 4, par3, k16, v16, q16, b16, g16, y16);
    ck("host_tree_topology_reject", (r1 == -1 && r2 == -1) ? 0 : 1 + rejectbad, 0);
}

static void t_oplist_parse(void) {
    /* 合成最小 blob: 1 slot, 3 ops (NOP/PIN/SILU) */
    uint8_t buf[512];
    memset(buf, 0, sizeof(buf));
    buf[0]='W'; buf[1]='T'; buf[2]='O'; buf[3]='P';
    buf[4]=1; buf[5]=0;              /* ver 1 */
    buf[6]=0x34; buf[7]=0x12;        /* endian */
    buf[8]=1; buf[9]=0; buf[10]=0; buf[11]=0;    /* n_slots */
    buf[12]=3; buf[13]=0; buf[14]=0; buf[15]=0;  /* n_ops */
    /* slot: len 128, cnt 1, off 0, addr 0 */
    uint32_t slot0[4] = { 128, 1, 0, 0 };
    memcpy(buf + 16, slot0, 16);
    uint32_t p = 32;
    /* NOP */
    buf[p]=0; buf[p+1]=0; buf[p+2]=0; buf[p+3]=0; p += 4;
    /* PIN [0] */
    buf[p]=3; buf[p+1]=0; buf[p+2]=1; buf[p+3]=0;
    memset(buf+p+4,0,4); p += 8;
    /* SILU [1, 2, 64] */
    buf[p]=4; buf[p+1]=0; buf[p+2]=3; buf[p+3]=0;
    uint32_t args[3] = { 1, 2, 64 };
    memcpy(buf + p + 4, args, 12); p += 16;
    /* weight 区 (align 128) */
    uint32_t woff = (p + 127) & ~127u;
    memset(buf + woff, 0xAB, 128);
    struct wt_blob b;
    int rc = wt_parse(buf, woff + 128, &b);
    ck("host_oplist_silu_parse", rc != WT_OK, 0);
    /* SILU arity 错 → WT_ERR_ARITY */
    buf[p-16+2] = 2; buf[p-16+3] = 0;
    rc = wt_parse(buf, woff + 128, &b);
    ck("host_oplist_silu_arity_reject", rc != WT_ERR_ARITY, 0);
}

static void t_rbr(void) {
    struct rbr r; float a[4] = {1,2,3,4}, b[8] = {5,6,7,8,9,10,11,12};
    rbr_init(&r);
    ck("host_rbr_restore_nosnap", rbr_restore(&r) != RBR_ERR_NOSNAP, 0);
    rbr_register(&r, 0, a, sizeof a);
    rbr_register(&r, 0, b, sizeof b);
    rbr_snapshot(&r);
    a[0] = 99.f; b[7] = 99.f;
    int bad = (rbr_restore(&r) != RBR_OK);
    bad += (memcmp(a, (float[]){1,2,3,4}, 16) != 0) + (b[7] != 12.f);
    ck("host_rbr_snapshot_roundtrip", bad, 0);

    enum rbr_setup_mode m1 = rbr_setup_hook(&r, 5);
    enum rbr_setup_mode m2 = rbr_setup_hook(&r, 5);
    enum rbr_setup_mode m3 = rbr_setup_hook(&r, 0);
    enum rbr_setup_mode m4 = rbr_setup_hook(&r, 5);   /* CLEAR 须作废挂起 skip */
    bad = (m1 != RBR_SKIP) + (m2 != RBR_COPY) + (m3 != RBR_CLEAR) + (m4 != RBR_COPY);
    bad += (r.n_skip_used != 1);
    ck("host_rbr_skip_onshot_copy_clear", bad, 0);

    rbr_note_process(&r, 7);
    bad = (rbr_restore(&r) != RBR_ERR_STALE);
    bad += (rbr_register(&r, 1, a, sizeof a) != RBR_ERR_FROZEN);
    bad += (rbr_rewind(&r, 0) != RBR_ERR_PARAM) + (rbr_rewind(&r, 8) != RBR_ERR_PARAM);
    bad += (rbr_n_past(&r) != 7);
    ck("host_rbr_stale_frozen_badrewind", bad, 0);

    bad = (rbr_rewind(&r, 3) != RBR_OK) + (rbr_n_past(&r) != 0);
    rbr_note_process(&r, 3);
    bad += (rbr_n_past(&r) != 3);
    rbr_snapshot(&r); rbr_restore(&r);
    bad += !rbr_shadow_equals(&r);
    ck("host_rbr_rewind_arith_shadow_eq", bad, 0);
    rbr_close(&r);
}

static void t_bledger(void) {
    struct bledger bl; float mem[8 * 128];
    memset(mem, 0, sizeof mem);
    bl_init(&bl);
    int bad = (bl_register(&bl, 0, mem, sizeof mem, 0, "x") != BL_ERR_PARAM);
    bad += (bl_register(&bl, 0, mem, sizeof mem + 4, 512, "x") != BL_ERR_PARAM);
    bad += (bl_register(&bl, 0, mem, sizeof mem, 512, "lg") != BL_OK);
    bad += (bl_verify(&bl, 0, 0) != BL_ERR_NEVER);          /* 从未写 → 断链 */
    bad += (bl_write(&bl, 0, 1u, 0, 2, 1u) != BL_OK);
    bl_expect(&bl, 0, 0, 2u, 0);
    bad += (bl_verify(&bl, 0, 0) != BL_ERR_WRITER);         /* 期望 writer 不符 */
    bl_expect(&bl, 0, 0, 2u, 9u);
    bad += (bl_write(&bl, 0, 2u, 0, 1, 9u) != BL_OK);       /* 异 writer 覆未读行 */
    bad += (bl.n_double_write != 1);
    bad += (bl_verify(&bl, 0, 0) != BL_OK);
    bl_release(&bl, 0);
    bad += (bl_verify(&bl, 0, 0) != BL_ERR_RELEASED);       /* 归还后禁读 */
    bl_canary(&bl, 0);
    bad += (bl_verify(&bl, 0, 0) != BL_ERR_CANARY);         /* canary 命中 */
    bad += (bl.buf[0].n_breaks != 4);
    char tl[256];
    bad += (bl_timeline(&bl, 0, tl, sizeof tl) == 0) || !strstr(tl, "lg");
    ck("host_bledger_contract", bad, 0);
    bl_close(&bl);
}

int main(void) {
    t_fence(); t_arena(); t_kvcache(); t_rbr(); t_bledger(); t_pxbridge(); t_gdn_tree(); t_oplist_parse();
    printf("--- host sim: %d pass, %d fail ---\n", npass, nfail);
    return nfail != 0;
}
