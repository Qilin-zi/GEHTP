/*
 * 16_wtcache_pin — U1 wtcache 单元 设备验证 (正确性 + 带宽)
 * =====================================================================
 * 覆盖: pin 权重 bit-exact / 多 pin 槽独立 / ring 活动下 pin 区不被破坏
 *       (T6 式) / move_back 数据完好 / DDR→VTCM 带宽。
 * 单元源: wtcache_pin_v81 (T1-T9 设备闭合), API 见 docs/api_v22_wtcache.md。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <HAP_perf.h>
#include <qurt.h>
#include "hvxhmx_v22.h"
#include "example_util.h"

#define WT_B  (128u * 1024u)
#define TILE_B (16u * 1024u)
#define NWT 3

static void fill_pat(uint8_t* p, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; p[i] = (uint8_t)(s >> 24); }
}

int main(void) {
    ex_open_result("16_wtcache_pin");
    struct wtcache_ctx* wc = NULL;
    int ok_all = 1;

    if (wtcache_open(&wc, 4u << 20) != WTC_OK) {
        ex_log("wtcache_open FAIL"); return ex_summary();
    }
    void* vb = NULL; uint32_t vs = 0; void* pb = NULL; uint32_t pc = 0;
    wtcache_layout(wc, &vb, &vs, &pb, &pc);
    ex_log("vtcm=%p size=%lu pin_cap=%lu", vb, (unsigned long)vs, (unsigned long)pc);

    /* ---- 1) 三块权重 pin + bit-exact 自检 ---- */
    uint8_t* src[NWT]; void* vt[NWT]; uint32_t sz[NWT] = { WT_B, WT_B / 2, WT_B * 3 / 4 };
    for (int i = 0; i < NWT; i++) {
        src[i] = memalign(128, sz[i]);
        fill_pat(src[i], sz[i], 0xA000 + i);
        int rc = wtcache_pin_weight(wc, src[i], sz[i], 0, &vt[i]);
        uint32_t bad = 0xFFFFFFFF;
        int vrc = (rc == WTC_OK) ? wtcache_pin_verify(wc, src[i], vt[i], sz[i], &bad) : rc;
        ex_check("pin_bitexact", vrc == WTC_OK ? 0 : 1, 0);
        ex_log("  pin%d %luB -> %p verify=%d", i, (unsigned long)sz[i], vt[i], vrc);
        ok_all &= (vrc == WTC_OK);
    }
    ex_check("pin_slots_distinct",
             (vt[0] == vt[1] || vt[1] == vt[2] || vt[0] == vt[2]) ? 1 : 0, 0);
    ex_check("pin_ptr_stable_session",
             ((uint8_t*)vt[0] >= (uint8_t*)pb &&
              (uint8_t*)vt[0] < (uint8_t*)pb + pc) ? 0 : 1, 0);

    /* ---- 2) pin 带宽: 1MB 一次 DMA 进 VTCM ---- */
    {
        uint8_t* big = memalign(128, 1u << 20);
        fill_pat(big, 1u << 20, 0xBEEF);
        int64_t t0 = HAP_perf_get_time_us();
        void* bvt = NULL;
        int rc = wtcache_pin_weight(wc, big, 1u << 20, 0, &bvt);
        int64_t dt = HAP_perf_get_time_us() - t0;
        uint32_t bad = 0;
        int vrc = (rc == WTC_OK) ? wtcache_pin_verify(wc, big, bvt, 1u << 20, &bad) : rc;
        ex_check("pin_1MB_verify", vrc == WTC_OK ? 0 : 1, 0);
        ex_log("  pin 1MB in %lld us -> %.2f GB/s", (long long)dt,
               dt > 0 ? 1048576.0 / dt / 1000.0 : 0.0);   /* B/us → GB/s */
        free(big);
    }

    /* ---- 3) ring 4+4: prime/walk/move_back, pin 区不被破坏 (T6 式) ---- */
    struct wtcache_ring* r = NULL;
    if (wtcache_ring_init(wc, &r, TILE_B, 4, 4) == WTC_OK) {
        uint8_t* tiles[8]; uint8_t* back[8];
        for (int i = 0; i < 8; i++) {
            tiles[i] = memalign(128, TILE_B); back[i] = memalign(128, TILE_B);
            fill_pat(tiles[i], TILE_B, 0xC000 + i);
            memset(back[i], 0, TILE_B);
            dc_clean_ddr(tiles[i], TILE_B);        /* 铁律①: DMA bypass 要读 */
        }
        wtcache_ring_prime(r, (const void**)tiles, 4);
        int pf_ok = 1, mb_ok = 1;
        for (int i = 0; i < 8; i++) {          /* walk: 第 i 步消费 + 预取 i+4 */
            void *vin, *vout;
            const void* next = (i + 4 < 8) ? tiles[i + 4] : NULL;
            /* T4 契约: 当轮给本轮 out 槽的 DDR 目标 (ring 内部 pending,
             * 下轮 ring_next 或 drain 才 submit — 保证调用方已写完) */
            if (wtcache_ring_next(r, next, back[i], 1, &vin, &vout)) { mb_ok = 0; break; }
            if (memcmp(vin, tiles[i], TILE_B)) pf_ok = 0;   /* 预取内容正确 */
            /* 模拟 HMX 写输出: CPU 写 VTCM → DMA 回读, 先 FLUSH (铁律) */
            memcpy(vout, tiles[i], TILE_B);
            qurt_mem_cache_clean((qurt_addr_t)vout, TILE_B,
                                 QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        }
        wtcache_ring_drain(r);
        for (int i = 0; i < 8 && mb_ok; i++) { /* 8 块全回搬 (drain 提交最后一笔); 铁律③ */
            qurt_mem_cache_clean((qurt_addr_t)back[i], TILE_B,
                                 QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            if (memcmp(back[i], tiles[i], TILE_B)) mb_ok = 0;
        }
        ex_check("ring_prefetch_bitexact", pf_ok ? 0 : 1, 0);
        ex_check("ring_moveback_bitexact", mb_ok ? 0 : 1, 0);
        int ifl = 0, imb = 0; uint32_t peak = 0;
        wtcache_ring_stats(r, &ifl, &imb, &peak);
        ex_log("  ring stats inflight=%d,%d peak_vtcm=%lu", ifl, imb, (unsigned long)peak);

        /* ring 活动后 pin 区仍然完好 */
        int pins_ok = 1;
        for (int i = 0; i < NWT; i++) {
            uint32_t bad = 0;
            if (wtcache_pin_verify(wc, src[i], vt[i], sz[i], &bad) != WTC_OK) pins_ok = 0;
        }
        ex_check("pin_survives_ring", pins_ok ? 0 : 1, 0);
        for (int i = 0; i < 8; i++) { free(tiles[i]); free(back[i]); }
    } else {
        ex_log("ring_init FAIL");
    }

    wtcache_close(wc);                          /* 铁律④ */
    (void)ok_all;
    return ex_summary();
}
