/*
 * 46_hmx_bias — 005 US3b 设备要害门 (T019): HMX+bias vs 标量+bias 对拍
 * =====================================================================
 * 输入 (host pack_hmx_bias.py 生成并 push 到 $D/):
 *   blob_hmx_bias.wtop   1 × OP_MATMUL_F16_BIAS (M=32,K=1024,N=2048 全 32 对齐)
 * 双跑同 blob:
 *   run1 wt_exec_force_bias_scalar(1) → 标量参照 (y_ref, engine 必须 SCALAR)
 *   run2 默认开关 → 生产默认引擎 (y_hmx, engine 必须 HMX —— T019 撤门后
 *        bias+32 对齐的默认就是 HMX, 本 run 即在验证生产路径)
 * 判据 (SC-005):
 *   ① 双侧引擎各自到位 (排除双跑同引擎的假阳性)
 *   ② cos(y_ref, y_hmx) ≥ 0.9999 (f32 口径)
 *   ③ 逐元素三判据之一成立: (f16 同号位距 ≤ 2) 或 (|d| ≤ 2·ulp_f16(ref))
 *      或 (|d| ≤ 2^-14 取消噪声地板)。HMX 硬件求和序 ≠ 标量序: 近零输出
 *      (Σ|a·w|~13 对消到 |y|<1e-4) 的位距按次正规数步长 2^-24 计成数百
 *      "ULP" 而绝对差 <3e-5, 属求和序噪声; 真 bias 错列的签名是绝对差
 *      ~|Δbias|≥0.1 且聚在单个 32 列 tile —— 两者均不满足, 已上板排除。
 *      地板 2^-14 = 6.1e-5, 取 2·K·eps_f32·max|a·w| ≈ 1.2e-5 (K=1024) 的
 *      5 倍裕量。异号仅在同判据豁免外才计失败; NaN 恒失败。
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "example_util.h"
#include "oplist_parse.h"
#include "oplist_exec.h"
#include "dc_parts.h"

#define D "/data/local/tmp/hvxhmx23/g46"
#define MM 32u
#define NN 2048u
#define OUT_BYTES (MM * NN * 2u)

static uint8_t* read_file(const char* p, size_t* out_len) {
    FILE* f = fopen(p, "rb");
    if (!f) { ex_log("open %s FAIL", p); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t* buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

static int run_blob(const uint8_t* blob, size_t blen, uint8_t* snap, uint32_t* engine) {
    struct wt_blob* w = calloc(1, sizeof(*w));
    if (!w || wt_parse(blob, blen, w) != WT_OK) { ex_log("wt_parse FAIL"); free(w); return 1; }
    struct wt_op_ts* ts = calloc(1, sizeof(*ts));
    wt_exec_set_ts(ts, 1);
    char err[128];
    int rc = wt_exec_run(w, engine, NULL, err, sizeof(err));
    if (rc) { ex_log("wt_exec_run FAIL rc=%d: %s", rc, err); wt_exec_shutdown(); free(ts); free(w); return 1; }
    uint8_t* t0 = wt_exec_temp(0);
    if (!t0 || wt_exec_temp_bytes(0) != OUT_BYTES) {
        ex_log("temp0 bad (t0=%p bytes=%lu)", t0, (unsigned long)wt_exec_temp_bytes(0));
        wt_exec_shutdown(); free(ts); free(w); return 1;
    }
    memcpy(snap, t0, OUT_BYTES);
    *engine = ts[0].engine;
    wt_exec_shutdown();
    free(ts); free(w);
    return 0;
}

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else { int e = -1; while (!(mant & 0x400)) { mant <<= 1; e--; } mant &= 0x3FF;
               u = sign | ((uint32_t)(114 + e) << 23) | (mant << 13); }
    } else if (exp == 31) u = sign | 0x7F800000u | (mant << 13);
    else u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; memcpy(&f, &u, 4); return f;
}

int main(void) {
    ex_open_result("46_hmx_bias");
    size_t bl = 0;
    uint8_t* blob = read_file(D "/blob_hmx_bias.wtop", &bl);
    if (!blob) { ex_log("[FAIL] read blob"); return ex_summary() || 1; }
    dc_clean_ddr(blob, (uint32_t)bl);

    uint8_t* y_ref = malloc(OUT_BYTES);
    uint8_t* y_hmx = malloc(OUT_BYTES);
    uint32_t eng_ref = 0xFF, eng_hmx = 0xFF;

    wt_exec_force_bias_scalar(1);   /* run1: 标量参照 (测试钩子, 非生产路径) */
    if (run_blob(blob, bl, y_ref, &eng_ref)) { wt_exec_force_bias_scalar(0); free(blob); free(y_ref); free(y_hmx); return ex_summary() || 1; }
    wt_exec_force_bias_scalar(0);
    int r2 = run_blob(blob, bl, y_hmx, &eng_hmx);
    if (r2) { free(blob); free(y_ref); free(y_hmx); return ex_summary() || 1; }

    ex_log("engine ref=%u (expect %u=SCALAR) hmx=%u (expect %u=HMX)",
           eng_ref, (unsigned)WT_ENG_SCALAR, eng_hmx, (unsigned)WT_ENG_HMX);
    ex_check("engine_scalar_ref", eng_ref != WT_ENG_SCALAR, 0);
    ex_check("engine_hmx_run", eng_hmx != WT_ENG_HMX, 0);

    /* SC-005 ③: 三判据 + 诊断 (top by 位距 / top by 绝对差) */
    double dot = 0, nr = 0, nh = 0;
    uint32_t max_ulp = 0, samesign = 0, xsign = 0, nans = 0, fail3 = 0;
    struct Off { uint32_t ulp; size_t idx; uint16_t a, b; } top[4] = {{0,0,0,0}}, topa[4] = {{0,0,0,0}};
    uint32_t ulp_hist[5] = {0,0,0,0,0};   /* [0]=1ulp [1]=2 [2]=3-4 [3]=5-16 [4]=>16 */
    double max_absd = 0;
    const double FLOOR = 6.103515625e-05;  /* 2^-14 取消噪声地板 */
    for (size_t i = 0; i < OUT_BYTES / 2u; i++) {
        uint16_t a = ((const uint16_t*)y_ref)[i], b = ((const uint16_t*)y_hmx)[i];
        float fa = f16_bits_to_f32(a), fb = f16_bits_to_f32(b);
        if (isnan(fa) || isnan(fb)) { nans++; continue; }
        dot += (double)fa * fb; nr += (double)fa * fa; nh += (double)fb * fb;
        double ad = fabs((double)fa - (double)fb);
        if (ad > max_absd) max_absd = ad;
        /* ulp_f16(ref): 指数位 e; e==0 (零/次正规) 步长 2^-24, 否则 2^(e-25) */
        uint32_t e = ((uint32_t)a >> 10) & 0x1Fu;
        double ulp = e ? ldexp(1.0, (int)e - 25) : 5.9604644775390625e-08;
        int flip = ((a & 0x8000u) != (b & 0x8000u));
        if (a == b) { samesign++; continue; }
        if (flip) xsign++;
        else {
            uint32_t d = a > b ? a - b : b - a;
            if (d > max_ulp) max_ulp = d;
            if (d <= 4u) ulp_hist[d <= 2u ? d - 1u : 2u]++;
            else if (d <= 16u) ulp_hist[3]++;
            else ulp_hist[4]++;
            uint32_t slot = 3;
            for (uint32_t k = 0; k < 4; k++) if (top[k].ulp < d) { slot = k; break; }
            if (slot < 4) {
                for (uint32_t k = 3; k > slot; k--) top[k] = top[k-1];
                top[slot] = (struct Off){d, i, a, b};
            }
        }
        /* 三判据: 位距≤2 / 相对 2ulp / 取消地板 (异号元素无位距, 只剩后两条) */
        int ok3 = (!flip && ((a > b ? a - b : b - a) <= 2u)) || (ad <= 2.0 * ulp) || (ad <= FLOOR);
        if (!ok3) fail3++;
        uint32_t slot = 3;
        for (uint32_t k = 0; k < 4; k++) if (topa[k].ulp < (uint32_t)(ad * 1e6)) { slot = k; break; }
        if (slot < 4) {
            for (uint32_t k = 3; k > slot; k--) topa[k] = topa[k-1];
            topa[slot] = (struct Off){(uint32_t)(ad * 1e6), i, a, b};
        }
    }
    ex_log("ulp_hist: 1=%u 2=%u 3-4=%u 5-16=%u >16=%u max_absd=%.6f",
           ulp_hist[0], ulp_hist[1], ulp_hist[2], ulp_hist[3], ulp_hist[4], max_absd);
    for (int k = 0; k < 4; k++)
        if (top[k].ulp) ex_log("offU[%d] ulp=%u idx=%zu (m=%zu n=%zu) ref=%+.6f hmx=%+.6f",
                               k, top[k].ulp, top[k].idx, top[k].idx / NN, top[k].idx % NN,
                               f16_bits_to_f32(top[k].a), f16_bits_to_f32(top[k].b));
    for (int k = 0; k < 4; k++)
        if (topa[k].ulp) ex_log("offA[%d] absd=%u.%03ue-3 idx=%zu (m=%zu n=%zu) ref=%+.6f hmx=%+.6f",
                                k, topa[k].ulp / 1000, topa[k].ulp % 1000, topa[k].idx,
                                topa[k].idx / NN, topa[k].idx % NN,
                                f16_bits_to_f32(topa[k].a), f16_bits_to_f32(topa[k].b));
    double cos = (nr > 0 && nh > 0) ? dot / (sqrt(nr) * sqrt(nh)) : 0.0;
    ex_log("cos=%.6f max_ulp=%u bit_eq=%u/%u xsign=%u nan=%u fail3=%u",
           cos, max_ulp, samesign, (unsigned)(OUT_BYTES / 2u), xsign, nans, fail3);
    ex_check("cos_ge_0.9999", cos < 0.9999, 0);
    ex_check("ulp_bound_3way", fail3, 0);
    ex_check("no_nan", nans, 0);

    ex_check("suite_complete", 0, 0);
    free(blob); free(y_ref); free(y_hmx);
    return ex_summary();
}
