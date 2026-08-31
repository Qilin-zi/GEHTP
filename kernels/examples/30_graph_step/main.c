/*
 * 30_graph_step — U16 整步 oplist vs 逐算子下发 设备验证
 * =====================================================================
 * 在设备上从 s256 资产直接合成 blob v1 (op 序:
 *   NOP / PIN(wt) / MATMUL / PIN(wt) / RMSNORM / SILU), 判据:
 *   1) wt_parse 接受合成 blob (含新 OP_SILU_F16=4)
 *   2) fused (wt_exec_run 一次) 与 split (逐 op run_range) 三张 temp 面
 *      逐字节恒等
 *   3) SILU vs 标量 oracle (f16 半 ULP 包络)
 *   4) 统计精确: ops/nop/matmul/rmsnorm/silu/pin 各就位
 *   5) pin-skip 语义: 引擎未建立时 PIN 只记账 (skipped), 建立后真 staged
 *      — fused 首跑 skipped=1, split 二跑 (引擎已建) skipped=0
 *   6) fused vs split 总耗时报告 (含 per-op 表)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <HAP_perf.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define A "/data/local/tmp/hvxhmx23/assets/s256"
#define M 256u
#define NE (M * M)

static uint8_t* cur;
static void wr16(uint16_t v) { cur[0] = (uint8_t)v; cur[1] = (uint8_t)(v >> 8); cur += 2; }
static void wr32(uint32_t v) {
    cur[0] = (uint8_t)v; cur[1] = (uint8_t)(v >> 8);
    cur[2] = (uint8_t)(v >> 16); cur[3] = (uint8_t)(v >> 24); cur += 4;
}

int main(void) {
    ex_open_result("30_graph_step");
    uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
    uint8_t* blob = NULL;
    uint8_t* act = dc_read_file(A "/act_surface.raw", &b0);
    uint8_t* wt  = dc_read_file(A "/packed_weight.raw", &b1);
    uint8_t* bis = dc_read_file(A "/folded_bias.raw", &b2);
    uint8_t* at  = dc_read_file(A "/act_table.raw", &b3);
    uint8_t* ot  = dc_read_file(A "/out_table.raw", &b4);
    uint8_t rwbuf[512];                                  /* rms_w 就地合成 (n=256) */
    {
        uint32_t l = 20260831u;
        for (int i = 0; i < 256; i++) {
            float w = 1.0f + gdn_lcg_norm(&l) * 0.05f;
            int16_t h = gdn_f32_to_f16(w);
            rwbuf[i * 2] = (uint8_t)(h & 0xFF);
            rwbuf[i * 2 + 1] = (uint8_t)((uint16_t)h >> 8);
        }
    }
    uint8_t* rw = rwbuf;
    if (!act || !wt || !bis || !at || !ot || !rw ||
        b0 != M * M * 2 || b1 != M * M / 2 || b2 != (M / 32) * 512 ||
        b3 != 8 * (M / 32) * 4 || b4 != 8 * (M / 32) * 4) {
        ex_log("assets missing/mismatch (rw=%p b5=%u)", rw, b5);
        goto out;
    }

    /* ---- 合成 blob: 6 slots + 6 ops ---- */
    uint32_t slen[6] = { b0, b1, b2, b3, b4, b5 ? b5 : 512 };
    const uint8_t* sdata[6] = { act, wt, bis, at, ot, rw };
    uint32_t wtot = 0;
    for (int i = 0; i < 6; i++) wtot += slen[i];
    uint32_t optab = 4 + 8 + 28 + 8 + 20 + 16;          /* NOP PIN MM PIN RMS SILU */
    uint32_t woff = (16u + 6u * 16u + optab + 127u) & ~127u;
    blob = memalign(128, woff + wtot);
    if (!blob) { ex_log("blob alloc FAIL"); goto out; }
    cur = blob;
    wr32(0x504F5457); wr16(1); wr16(0x1234); wr32(6); wr32(6);
    uint32_t off = 0;
    for (int i = 0; i < 6; i++) { wr32(slen[i]); wr32(1); wr32(off); wr32(0); off += slen[i]; }
    /* ops */
    wr16(0); wr16(0);                                   /* NOP */
    wr16(3); wr16(1); wr32(1);                          /* PIN(wt) */
    wr16(1); wr16(6); wr32(0); wr32(1); wr32(0); wr32(M); wr32(M); wr32(M);
    wr16(3); wr16(1); wr32(1);                          /* PIN(wt) 再来 */
    wr16(2); wr16(4); wr32(0); wr32(5); wr32(1); wr32(M);  /* RMSNORM */
    wr16(4); wr16(3); wr32(1); wr32(2); wr32(NE);       /* SILU */
    while ((uint32_t)(cur - blob) < woff) *cur++ = 0;
    off = 0;
    for (int i = 0; i < 6; i++) { memcpy(blob + woff + off, sdata[i], slen[i]); off += slen[i]; }
    dc_clean_ddr(blob, woff + wtot);                    /* 铁律①: DMA bypass 读 */

    struct wt_blob b;
    int prc = wt_parse(blob, woff + wtot, &b);
    ex_check("blob_parse_ok", prc != WT_OK, 0);
    if (prc != WT_OK) { ex_log("parse: %s", wt_err_str(prc)); goto out; }

    /* ---- fused ---- */
    uint32_t em = 0; int64_t opus[8]; char err[128];
    uint8_t* snap0 = malloc(NE * 2); uint8_t* snap1 = malloc(NE * 2);
    uint8_t* snap2 = malloc(NE * 2);
    int rc = wt_exec_run(&b, &em, opus, err, sizeof(err));
    ex_check("fused_run_ok", rc, 0);
    if (rc) { ex_log("fused fail @op%d: %s", rc, err); goto shut; }
    memcpy(snap0, wt_exec_temp(0), NE * 2);
    memcpy(snap1, wt_exec_temp(1), NE * 2);
    memcpy(snap2, wt_exec_temp(2), NE * 2);

    struct wt_exec_stats st;
    wt_exec_get_stats(&st);
    int stbad = (st.ops != 6) + (st.nop != 1) + (st.matmul != 1) +
                (st.rmsnorm != 1) + (st.silu != 1) + (st.pin != 2);
    ex_check("stats_fused_exact", stbad, 0);
    ex_check("pin_skip_engine_not_ready", st.pin_skipped != 1 ? 1 : 0, 0);
    ex_log("  fused: ops=%u mm=%u rms=%u silu=%u pin=%u skipped=%u",
           st.ops, st.matmul, st.rmsnorm, st.silu, st.pin, st.pin_skipped);

    /* ---- split: 逐 op ---- */
    struct wt_exec_stats st0;
    wt_exec_get_stats(&st0);
    int64_t t0 = HAP_perf_get_time_us();
    int src = 0;
    for (uint32_t i = 0; i < 6 && !src; i++) {
        int64_t ou[1]; uint32_t em1 = 0;
        src = wt_exec_run_range(&b, i, 1, &em1, ou, err, sizeof(err));
        if (src) ex_log("split fail @op%u: %s", i, err);
    }
    int64_t split_us = HAP_perf_get_time_us() - t0;
    ex_check("split_run_ok", src, 0);
    if (!src) {
        int idn = memcmp(snap0, wt_exec_temp(0), NE * 2) ||
                  memcmp(snap1, wt_exec_temp(1), NE * 2) ||
                  memcmp(snap2, wt_exec_temp(2), NE * 2);
        ex_check("split_vs_fused_temps_byteexact", idn, 0);
    }
    wt_exec_get_stats(&st);
    ex_check("pin_skip_engine_ready_zero", (st.pin_skipped - st0.pin_skipped) != 0 ? 1 : 0, 0);
    ex_log("  split 总 %lld us (fused op 累计 %lld us)",
           (long long)split_us, (long long)(opus[0]+opus[1]+opus[2]+opus[3]+opus[4]+opus[5]));

    /* ---- SILU oracle (f16 半 ULP 包络) ---- */
    {
        const uint16_t* x = (const uint16_t*)snap1;
        const uint16_t* y = (const uint16_t*)snap2;
        int bad = 0;
        for (uint32_t i = 0; i < NE; i++) {
            float f = gdn_f16_to_f32((int16_t)x[i]);
            float e = f / (1.0f + expf(-f));
            float r = gdn_f16_to_f32((int16_t)y[i]);
            float ulp = fabsf(e) * 1.3e-3f + 1e-7f;
            if (fabsf(r - e) > ulp) bad++;
        }
        ex_check("silu_vs_oracle_halfulp", bad, 0);
    }
    free(snap0); free(snap1); free(snap2);
shut:
    wt_exec_shutdown();
out:
    if (blob) free(blob);
    return ex_summary();
}
