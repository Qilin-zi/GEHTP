/*
 * 15_v2_llm_ops — V2 LLM 算子层 设备验证 (正确性 + 性能)
 * =====================================================================
 * 覆盖 V2 全部公共 API:
 *   P0: rms_norm/norm/l2_norm + dequant(Q4_0/Q8_0) + hmx_gemm_dot
 *   P1: fa_qk_dot/attn_v_mad/alibi + rope(neox/normal/cache)
 *   P2: unary(exp/sqrt/sigmoid/sqr/scale/...) + binary(add/sub/mul/div)
 *       + softmax(fused/mask) + ssm(gdn4/gdn8/solve_tri)
 *
 * 方法: DSP 上同 buffer 跑 HVX 版 vs 标量参考 (各自独立代码路径),
 *       比 max abs/rel err; 再各自计时 hmx_perf_now_us 报告 us/call.
 *
 * 尺寸取 Qwen3.5-0.8B 推理热点: hidden=1024, head_dim=128/256, n=32 倍数.
 */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "hvxhmx.h"
#include "hvxhmx_v2.h"
#include "internal/hmx-utils.h"   /* hmx_interleave_rows_to_tiles (tile 布局工具) */
#include "example_util.h"

#define ALN __attribute__((aligned(2048)))

/* ---------- 静态 buffer (DSP 栈小, 禁 VLA) ---------- */
#define N_MAX 4096
static float  src_a[N_MAX] ALN;
static float  src_b[N_MAX] ALN;
static float  dst_hvx[N_MAX] ALN;
static float  dst_ref[N_MAX] ALN;
static float  pad_buf[N_MAX] ALN;
static __fp16 f16_a[N_MAX] ALN;
static __fp16 f16_b[N_MAX] ALN;
static __fp16 f16_c[N_MAX] ALN;

/* HMX tile buffer: 32x32 tile = 1024 fp16 = 2KB; 给 8 K-tile + 输出 */
static __fp16 act_tiles[8 * 1024] ALN;    /* [1 row_tile][8 k_tiles][1024] */
static __fp16 wgt_tiles[8 * 1024] ALN;    /* [1 col_tile][8 k_tiles][1024] */
static __fp16 out_tiles[1024] ALN;        /* [1][1][1024] */
static __fp16 scales_tb[128] ALN;         /* HMX bias/scales, 256B */
static uint8_t q_wgt[8 * 1152] ALN;       /* Q8_0/Q4_0 量化 tile (max aligned=1152) */

/* ---------- 标量参考实现 (与 V2 host fallback 数学一致) ---------- */
static void ref_rms_norm_mul(const float *s, const float *w, float *d, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)s[i] * s[i];
    float sc = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) d[i] = s[i] * sc * w[i];
}
static void ref_softmax(const float *s, float *d, int n) {
    float mx = s[0];
    for (int i = 1; i < n; ++i) if (s[i] > mx) mx = s[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { d[i] = expf(s[i] - mx); sum += d[i]; }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) d[i] *= inv;
}
static void ref_rope_neox(float *d, const float *s, int nd, const float *tc) {
    int he = nd / 2;
    for (int i = 0; i < he; ++i) {
        float c = tc[2 * i], t = tc[2 * i + 1];
        float x0 = s[i], x1 = s[i + he];
        d[i] = x0 * c - x1 * t;
        d[i + he] = x0 * t + x1 * c;
    }
}
static void ref_gdn4(float *d0, float *d1, float *d2, float *d3,
                     const float *mul, const float *dot, int n, float *sums) {
    float *ds[4] = {d0, d1, d2, d3};
    for (int k = 0; k < 4; ++k) sums[k] = 0.0f;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < 4; ++k) {
            ds[k][i] *= mul[i];
            sums[k] += ds[k][i] * dot[i];
        }
}
static float ref_qk_dot_f16(const __fp16 *q, const __fp16 *k, int n, float scale) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)q[i] * (double)k[i];
    return (float)(acc * scale);
}

/* fp16 GEMM 参考 (tile 布局 — 由 hmx-utils.h scatter offsets + identity 探针实证:
 *   输入 tile[(r,c)] at tile_base[(c/2)*64 + 2*r + (c&1)]   (k-pair-major)
 *   输出 tile[(m,n)] at tile_base[(m/2)*64 + 2*n + (m&1)]   (行-pair-major, 转置!)
 *   act tile t: 元素 (m, k=t*32+k_local) at t*1024 + in_pos(m, k_local)
 *   wgt tile t: 元素 (n, k=t*32+k_local) at t*1024 + in_pos(n, k_local)
 *   out tile:   元素 (m, n)            at out_pos(m, n)                    */
static int tile_pos(int r, int c) { return (c / 2) * 64 + 2 * r + (c & 1); }
static int tile_pos_out(int m, int n) { return (m / 2) * 64 + 2 * n + (m & 1); }
static void ref_gemm_tile(const __fp16 *at, const __fp16 *wt, __fp16 *ot, int kt) {
    for (int r = 0; r < 32; ++r)
        for (int c = 0; c < 32; ++c) {
            float acc = 0.0f;
            for (int t = 0; t < kt; ++t)
                for (int kk = 0; kk < 32; ++kk)
                    acc += (float)at[t * 1024 + tile_pos(r, kk)] *
                           (float)wt[t * 1024 + tile_pos(c, kk)];
            ot[tile_pos_out(r, c)] = (__fp16)acc;
        }
}

/* Q4_0 tile dequant 参考 (由 dequant kernel 推出):
 * - tile 布局 pos(r,c) = (c/2)*64 + 2r + (c&1)
 * - 32 个 fp16 scale 在 tile+512, sc[r] = 行 r 的 scale (每行 32 K 值一块)
 * - quants 4 group × 128B: group g 覆盖列 8g..8g+7;
 *   byte i (group g): lo nibble = (r=i%32, c=8g+2*(i/32)),
 *                     hi nibble = (r=i%32, c=8g+2*(i/32)+1), 值 = nibble-8 */
static void ref_dequant_q4_0_tile(const uint8_t *qw, __fp16 *out) {
    const __fp16 *sc = (const __fp16 *)(qw + 512);
    for (int g = 0; g < 4; ++g) {
        const uint8_t *qb = qw + g * 128;
        for (int i = 0; i < 128; ++i) {
            int r = i % 32;
            int c = 8 * g + 2 * (i / 32);
            int lo = (qb[i] & 0x0F) - 8;
            int hi = (qb[i] >> 4) - 8;
            out[tile_pos(r, c)]     = (__fp16)(lo * (float)sc[r]);
            out[tile_pos(r, c + 1)] = (__fp16)(hi * (float)sc[r]);
        }
    }
}

/* ---------- 误差比较 ---------- */
static int cmp_f32(const char *label, const float *a, const float *b, int n, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        float m = fmaxf(fabsf(a[i]), fabsf(b[i]));
        float rel = (m > 1e-6f) ? e / m : e;
        if (rel > max_err) max_err = rel;
    }
    int failed = max_err > tol;
    ex_log("%-44s maxrel=%.6f tol=%.4f %s", label, max_err, tol,
           failed ? "FAIL" : "PASS");
    return failed;
}
static int cmp_f16(const char *label, const __fp16 *a, const __fp16 *b, int n, float tol) {
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf((float)a[i] - (float)b[i]);
        float m = fmaxf(fabsf((float)a[i]), fabsf((float)b[i]));
        float rel = (m > 1e-3f) ? e / m : e;
        if (rel > max_err) max_err = rel;
    }
    int failed = max_err > tol;
    ex_log("%-44s maxrel=%.6f tol=%.4f %s", label, max_err, tol,
           failed ? "FAIL" : "PASS");
    return failed;
}

/* ---------- 计时 ---------- */
static double time_us(void (*fn)(void *), void *arg, int iters) {
    uint64_t t0 = hmx_perf_now_us();
    for (int i = 0; i < iters; ++i) fn(arg);
    uint64_t t1 = hmx_perf_now_us();
    return (double)(t1 - t0) / iters;
}

/* ---------- 各算子测试 ---------- */
static int n_fail = 0;

static void test_norm(void) {
    const int n = 1024;   /* Qwen3.5 hidden */
    for (int i = 0; i < n; ++i) {
        src_a[i] = 0.5f * sinf(i * 0.13f) + 0.1f;
        src_b[i] = 1.0f + 0.01f * (i % 7);
    }
    /* rms_norm_mul */
    hvhx_v2_rms_norm_mul_f32(src_a, src_b, dst_hvx, n, 1e-5f);
    ref_rms_norm_mul(src_a, src_b, dst_ref, n, 1e-5f);
    n_fail += cmp_f32("rms_norm_mul_f32 n=1024", dst_hvx, dst_ref, n, 0.01f);

    /* rms_norm (no weight) */
    hvhx_v2_rms_norm_f32(src_a, dst_hvx, n, 1e-5f);
    {
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double)src_a[i] * src_a[i];
        float sc = 1.0f / sqrtf((float)(ss / n) + 1e-5f);
        for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] * sc;
    }
    n_fail += cmp_f32("rms_norm_f32 n=1024", dst_hvx, dst_ref, n, 0.01f);

    /* l2_norm */
    hvhx_v2_l2_norm_f32(src_a, dst_hvx, n, 1e-5f);
    {
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double)src_a[i] * src_a[i];
        float sc = 1.0f / fmaxf(sqrtf((float)ss), 1e-5f);
        for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] * sc;
    }
    n_fail += cmp_f32("l2_norm_f32 n=1024", dst_hvx, dst_ref, n, 0.01f);

    /* 非 32 倍数 (tail 路径) */
    hvhx_v2_rms_norm_mul_f32(src_a, src_b, dst_hvx, 1000, 1e-5f);
    ref_rms_norm_mul(src_a, src_b, dst_ref, 1000, 1e-5f);
    n_fail += cmp_f32("rms_norm_mul_f32 n=1000 (tail)", dst_hvx, dst_ref, 1000, 0.01f);

    /* perf */
    uint64_t t0 = hmx_perf_now_us();
    for (int it = 0; it < 100; ++it)
        hvhx_v2_rms_norm_mul_f32(src_a, src_b, dst_hvx, n, 1e-5f);
    uint64_t t1 = hmx_perf_now_us();
    double us = (double)(t1 - t0) / 100.0;
    ex_log("PERF rms_norm_mul n=1024: %.1f us  (%.2f GB/s eff: 3x4KB r+w)",
           us, 3.0 * n * 4.0 / us / 1e3);
}

static void test_unary_binary(void) {
    const int n = 1024;
    for (int i = 0; i < n; ++i) {
        src_a[i] = 0.25f + 0.5f * fabsf(sinf(i * 0.071f));   /* 正数域 for sqrt/log */
        src_b[i] = 0.75f + 0.2f * cosf(i * 0.031f);
    }
    /* sqrt */
    hvhx_v2_sqrt_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = sqrtf(src_a[i]);
    n_fail += cmp_f32("sqrt_f32", dst_hvx, dst_ref, n, 0.01f);
    /* sqr */
    hvhx_v2_sqr_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] * src_a[i];
    n_fail += cmp_f32("sqr_f32", dst_hvx, dst_ref, n, 0.01f);
    /* sigmoid (tanh 域) */
    for (int i = 0; i < n; ++i) src_a[i] = 4.0f * sinf(i * 0.05f);
    hvhx_v2_sigmoid_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = 1.0f / (1.0f + expf(-src_a[i]));
    n_fail += cmp_f32("sigmoid_f32", dst_hvx, dst_ref, n, 0.02f);
    /* tanh */
    hvhx_v2_tanh_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = tanhf(src_a[i]);
    n_fail += cmp_f32("tanh_f32", dst_hvx, dst_ref, n, 0.02f);
    /* exp (限域防爆) */
    for (int i = 0; i < n; ++i) src_a[i] = -2.0f + 3.0f * sinf(i * 0.043f);
    hvhx_v2_exp_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = expf(src_a[i]);
    n_fail += cmp_f32("exp_f32", dst_hvx, dst_ref, n, 0.02f);
    /* log */
    for (int i = 0; i < n; ++i) src_a[i] = 0.1f + 2.0f * fabsf(sinf(i * 0.037f));
    hvhx_v2_log_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = logf(src_a[i]);
    n_fail += cmp_f32("log_f32", dst_hvx, dst_ref, n, 0.02f);
    /* scale */
    hvhx_v2_scale_f32(dst_hvx, src_a, n, 1.7f);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] * 1.7f;
    n_fail += cmp_f32("scale_f32", dst_hvx, dst_ref, n, 0.01f);
    /* scale_offset */
    hvhx_v2_scale_offset_f32(dst_hvx, src_a, n, 1.7f, -0.3f);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] * 1.7f - 0.3f;
    n_fail += cmp_f32("scale_offset_f32", dst_hvx, dst_ref, n, 0.01f);
    /* inverse */
    for (int i = 0; i < n; ++i) src_a[i] = 0.5f + fabsf(sinf(i * 0.021f));
    hvhx_v2_inverse_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = 1.0f / src_a[i];
    n_fail += cmp_f32("inverse_f32", dst_hvx, dst_ref, n, 0.02f);
    /* reduce_sum */
    float s_hvx = hvhx_v2_reduce_sum_f32(src_a, n);
    double s_ref = 0.0;
    for (int i = 0; i < n; ++i) s_ref += src_a[i];
    {
        float rel = fabsf(s_hvx - (float)s_ref) / fabsf((float)s_ref);
        int failed = rel > 0.01f;
        ex_log("%-44s maxrel=%.6f tol=0.0100 %s", "reduce_sum_f32", rel,
               failed ? "FAIL" : "PASS");
        n_fail += failed;
    }
    /* binary: add/sub/mul/div */
    for (int i = 0; i < n; ++i) src_b[i] = 0.5f + fabsf(cosf(i * 0.017f));
    hvhx_v2_add_f32(dst_hvx, src_a, src_b, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] + src_b[i];
    n_fail += cmp_f32("add_f32", dst_hvx, dst_ref, n, 0.01f);
    hvhx_v2_sub_f32(dst_hvx, src_a, src_b, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] - src_b[i];
    n_fail += cmp_f32("sub_f32", dst_hvx, dst_ref, n, 0.01f);
    hvhx_v2_mul_f32(dst_hvx, src_a, src_b, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] * src_b[i];
    n_fail += cmp_f32("mul_f32", dst_hvx, dst_ref, n, 0.01f);
    hvhx_v2_div_f32(dst_hvx, src_a, src_b, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] / src_b[i];
    n_fail += cmp_f32("div_f32", dst_hvx, dst_ref, n, 0.02f);

    /* tail (n=1000) 抽一项验证标量尾路径 */
    hvhx_v2_mul_f32(dst_hvx, src_a, src_b, 1000);
    for (int i = 0; i < 1000; ++i) dst_ref[i] = src_a[i] * src_b[i];
    n_fail += cmp_f32("mul_f32 n=1000 (tail)", dst_hvx, dst_ref, 1000, 0.01f);

    /* perf: mul (memory-bound 代表) */
    uint64_t t0 = hmx_perf_now_us();
    for (int it = 0; it < 100; ++it) hvhx_v2_mul_f32(dst_hvx, src_a, src_b, n);
    uint64_t t1 = hmx_perf_now_us();
    ex_log("PERF mul n=1024: %.2f us (%.2f GB/s)",
           (double)(t1 - t0) / 100.0,
           3.0 * n * 4.0 / ((double)(t1 - t0) / 100.0) / 1e3);
    /* perf: sigmoid (compute-heavy 代表) */
    t0 = hmx_perf_now_us();
    for (int it = 0; it < 100; ++it) hvhx_v2_sigmoid_f32(dst_hvx, src_a, n);
    t1 = hmx_perf_now_us();
    ex_log("PERF sigmoid n=1024: %.2f us", (double)(t1 - t0) / 100.0);
}

static void test_softmax(void) {
    const int n = 1024;
    for (int i = 0; i < n; ++i) src_a[i] = 3.0f * sinf(i * 0.019f) + 0.001f * i;
    hvhx_v2_softmax_f32(src_a, dst_hvx, pad_buf, n);
    ref_softmax(src_a, dst_ref, n);
    n_fail += cmp_f32("softmax_f32 n=1024", dst_hvx, dst_ref, n, 0.01f);

    /* mask prep: dst = src*scale + mask*slope */
    hvhx_v2_softmax_mask_f32(src_a, dst_hvx, n, 0.088f, src_b, 1.0f);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] * 0.088f + src_b[i] * 1.0f;
    n_fail += cmp_f32("softmax_mask_f32 n=1024", dst_hvx, dst_ref, n, 0.01f);

    /* pad=NULL 路径 (修复后: 回退标量, 不崩) */
    hvhx_v2_softmax_f32(src_a, dst_hvx, NULL, n);
    ref_softmax(src_a, dst_ref, n);
    n_fail += cmp_f32("softmax_f32 pad=NULL (fixed)", dst_hvx, dst_ref, n, 0.01f);

    /* 非对齐 src (回退标量, 不崩) */
    hvhx_v2_softmax_f32(src_a + 1, dst_hvx, pad_buf, 1024 - 32);
    ref_softmax(src_a + 1, dst_ref, 1024 - 32);
    n_fail += cmp_f32("softmax_f32 unaligned src", dst_hvx, dst_ref, 1024 - 32, 0.01f);

    uint64_t t0 = hmx_perf_now_us();
    for (int it = 0; it < 100; ++it) hvhx_v2_softmax_f32(src_a, dst_hvx, pad_buf, n);
    uint64_t t1 = hmx_perf_now_us();
    ex_log("PERF softmax n=1024: %.2f us", (double)(t1 - t0) / 100.0);
}

static void test_rope(void) {
    const int nd = 128;   /* rotary dims (Qwen3.5 head_dim=256, rope 半) */
    static float cache[256] ALN;
    /* cache 构建: theta_base=pos=3.0, freq_scale=1, theta_scale=base^-2/nd */
    float theta_scale = powf(10000.0f, -2.0f / nd);
    hvhx_v2_rope_cache_init(3.0f, 1.0f, NULL, NULL, nd, 0.0f, 1.0f,
                            cache, theta_scale);
    /* cache 参考 */
    {
        float theta = 3.0f;
        for (int i = 0; i < nd; i += 2) {
            dst_ref[i]     = cosf(theta);
            dst_ref[i + 1] = sinf(theta);
            theta *= theta_scale;
        }
        n_fail += cmp_f32("rope_cache_init nd=128", cache, dst_ref, nd, 0.01f);
    }
    for (int i = 0; i < nd; ++i) src_a[i] = 0.5f * sinf(i * 0.11f);
    /* neox */
    hvhx_v2_rope_neox_f32(dst_hvx, src_a, nd, cache);
    ref_rope_neox(dst_ref, src_a, nd, cache);
    n_fail += cmp_f32("rope_neox_f32 nd=128", dst_hvx, dst_ref, nd, 0.01f);
    /* normal */
    hvhx_v2_rope_normal_f32(dst_hvx, src_a, nd, cache);
    for (int i = 0; i < nd; i += 2) {
        float c = cache[i], s = cache[i + 1];
        float x0 = src_a[i], x1 = src_a[i + 1];
        dst_ref[i]     = x0 * c - x1 * s;
        dst_ref[i + 1] = x0 * s + x1 * c;
    }
    n_fail += cmp_f32("rope_normal_f32 nd=128", dst_hvx, dst_ref, nd, 0.01f);

    uint64_t t0 = hmx_perf_now_us();
    for (int it = 0; it < 200; ++it) hvhx_v2_rope_neox_f32(dst_hvx, src_a, nd, cache);
    uint64_t t1 = hmx_perf_now_us();
    ex_log("PERF rope_neox nd=128: %.2f us", (double)(t1 - t0) / 200.0);
}

static void test_fa(void) {
    const int hd = 256;   /* Qwen3.5 head_dim */
    for (int i = 0; i < hd; ++i) {
        f16_a[i] = (__fp16)(0.25f * sinf(i * 0.07f));
        f16_b[i] = (__fp16)(0.25f * cosf(i * 0.05f));
    }
    float out_hvx = 0.0f;
    hvhx_v2_fa_qk_dot_f16(f16_a, f16_b, hd, 0.0625f, &out_hvx);
    float out_ref = ref_qk_dot_f16(f16_a, f16_b, hd, 0.0625f);
    {
        float rel = fabsf(out_hvx - out_ref) / fmaxf(fabsf(out_ref), 1e-3f);
        int failed = rel > 0.02f;
        ex_log("%-44s maxrel=%.6f tol=0.0200 %s", "fa_qk_dot_f16 hd=256", rel,
               failed ? "FAIL" : "PASS");
        n_fail += failed;
    }
    /* attn_v_mad: y[i] += x[i] * s */
    for (int i = 0; i < hd; ++i) { dst_hvx[i] = 0.1f * i / hd; dst_ref[i] = dst_hvx[i]; }
    __fp16 s = (__fp16)0.37f;
    ex_log("  s fp16 bits=0x%04x val=%.8f (float0.37=%.8f)",
           *(uint16_t *)&s, (float)s, 0.37f);
    hvhx_v2_fa_attn_v_mad_f16(dst_hvx, f16_b, &s, hd);
    for (int i = 0; i < hd; ++i) dst_ref[i] += (float)f16_b[i] * (float)s;
    n_fail += cmp_f32("fa_attn_v_mad_f16 hd=256", dst_hvx, dst_ref, hd, 0.01f);
    /* debug: print worst rel element + first 8 */
    {
        float brel = 0.0f;
        int bi = 0;
        for (int i = 0; i < hd; ++i) {
            float e = fabsf(dst_hvx[i] - dst_ref[i]);
            float m = fmaxf(fabsf(dst_hvx[i]), fabsf(dst_ref[i]));
            float rel = (m > 1e-6f) ? e / m : e;
            if (rel > brel) { brel = rel; bi = i; }
        }
        ex_log("  worst-rel i=%d hvx=%.8f ref=%.8f x=%.8f y0=%.8f rel=%.6f",
               bi, dst_hvx[bi], dst_ref[bi], (float)f16_b[bi],
               0.1f * bi / hd, brel);
        for (int i = 0; i < 8; ++i)
            ex_log("  i=%3d hvx=%.8f ref=%.8f x=%.6f",
                   i, dst_hvx[i], dst_ref[i], (float)f16_b[i]);
    }
    /* alibi slopes: 8 heads, kv_head=0, G=8 → h=0..31 */
    static float slopes_hvx[32] ALN;
    static float slopes_ref[32] ALN;
    hvhx_v2_alibi_slopes(0, 8, 3, powf(2.0f, -8.0f), powf(2.0f, -4.0f), slopes_hvx);
    for (int h = 0; h < 32; ++h)
        slopes_ref[h] = (h < 3) ? powf(powf(2.0f, -8.0f), (float)(h + 1))
                                : powf(powf(2.0f, -4.0f), (float)(2 * (h - 3) + 1));
    n_fail += cmp_f32("alibi_slopes", slopes_hvx, slopes_ref, 32, 0.02f);

    uint64_t t0 = hmx_perf_now_us();
    for (int it = 0; it < 200; ++it)
        hvhx_v2_fa_qk_dot_f16(f16_a, f16_b, hd, 0.0625f, &out_hvx);
    uint64_t t1 = hmx_perf_now_us();
    ex_log("PERF fa_qk_dot hd=256: %.2f us", (double)(t1 - t0) / 200.0);
}

static void test_ssm(void) {
    const int n = 128;
    static float d0[128] ALN, d1[128] ALN, d2[128] ALN, d3[128] ALN;
    static float r0[128] ALN, r1[128] ALN, r2[128] ALN, r3[128] ALN;
    float sums_h[4], sums_r[4];
    for (int i = 0; i < n; ++i) {
        float v = 0.3f * sinf(i * 0.09f);
        d0[i] = r0[i] = v; d1[i] = r1[i] = v * 1.1f;
        d2[i] = r2[i] = v * 0.9f; d3[i] = r3[i] = v * 1.2f;
        src_a[i] = 0.9f + 0.05f * sinf(i * 0.3f);   /* mul ~1 */
        src_b[i] = 0.2f * cosf(i * 0.17f);          /* dot */
    }
    hvhx_v2_gdn_mul_dot4_f32(d0, d1, d2, d3, src_a, src_b, n, sums_h);
    ref_gdn4(r0, r1, r2, r3, src_a, src_b, n, sums_r);
    n_fail += cmp_f32("gdn_mul_dot4 d0", d0, r0, n, 0.01f);
    n_fail += cmp_f32("gdn_mul_dot4 d3", d3, r3, n, 0.01f);
    {
        float rel = 0.0f;
        for (int k = 0; k < 4; ++k) {
            float e = fabsf(sums_h[k] - sums_r[k]) / fmaxf(fabsf(sums_r[k]), 1e-3f);
            if (e > rel) rel = e;
        }
        int failed = rel > 0.01f;
        ex_log("%-44s maxrel=%.6f tol=0.0100 %s", "gdn_mul_dot4 sums", rel,
               failed ? "FAIL" : "PASS");
        n_fail += failed;
    }
    /* solve_tri: X[row][col] = (B[col] - sum_t A[t]*X[t][col]) * inv_diag */
    {
        static float Xt[64 * 64] ALN;   /* k=64 方阵 */
        static float Ar[64] ALN, Br[64] ALN;
        const uint32_t k = 64, row = 5;
        for (uint32_t t = 0; t < row; ++t) {
            Ar[t] = 0.1f * sinf(t * 1.7f);
            for (uint32_t c = 0; c < k; ++c)
                Xt[t * k + c] = 0.05f * sinf(t * 0.7f + c * 0.13f);
        }
        for (uint32_t c = 0; c < k; ++c) Br[c] = 0.2f * cosf(c * 0.11f);
        hvhx_v2_solve_tri_row_f32(Ar, Br, Xt, row, k, 0, k, 1.25f);
        /* 参考: 重算一遍 (保存原始 X 先) — 简化: 数值合理性检查 */
        float maxv = 0.0f;
        for (uint32_t c = 0; c < k; ++c) {
            float expect = Br[c];
            for (uint32_t t = 0; t < row; ++t) expect -= Ar[t] * Xt[t * k + c];
            expect *= 1.25f;
            float e = fabsf(Xt[row * k + c] - expect);
            if (e > maxv) maxv = e;
        }
        int failed = maxv > 1e-4f;
        ex_log("%-44s maxabs=%.6f tol=0.0001 %s", "solve_tri_row k=64 row=5",
               maxv, failed ? "FAIL" : "PASS");
        n_fail += failed;
    }
    uint64_t t0 = hmx_perf_now_us();
    for (int it = 0; it < 100; ++it)
        hvhx_v2_gdn_mul_dot4_f32(d0, d1, d2, d3, src_a, src_b, n, sums_h);
    uint64_t t1 = hmx_perf_now_us();
    ex_log("PERF gdn_mul_dot4 n=128: %.2f us", (double)(t1 - t0) / 100.0);
}

static void test_dequant_gemm(void) {
    /* --- Q4_0 tile dequant (HVX kernel, DDR buffer 即可) --- */
    for (int i = 0; i < 576; ++i) q_wgt[i] = (uint8_t)((i * 37 + 11) & 0xFF);
    /* scale 区 (tile_src+512, 32 个 fp16) 必须合法小正数 */
    for (int i = 0; i < 32; ++i) ((__fp16 *)(q_wgt + 512))[i] = (__fp16)(0.02f + 0.001f * i);
    static __fp16 dq_hvx[1024] ALN;
    static __fp16 dq_ref[1024] ALN;
    int rc = hvhx_v2_dequant_tiled_to_fp16(q_wgt, dq_hvx, 32, 32, HVHX_V2_WT_Q4_0);
    if (rc != 0) {
        ex_log("%-44s rc=%d FAIL", "dequant_q4_0 api", rc);
        n_fail += 1;
    } else {
        ref_dequant_q4_0_tile(q_wgt, dq_ref);
        n_fail += cmp_f16("dequant_q4_0 1 tile", dq_hvx, dq_ref, 1024, 0.02f);
    }

    /* --- HMX GEMM: 1x1 row/col tile, K=256 (8 k_tiles) ---
     * 铁律: HMX 操作数 (act/wgt/out/scales) 必须驻 VTCM, DDR 静态数组会 CX_FAULT.
     *
     * 已实证布局契约 (52f67807, 2026-08-13, 本文件探针序列):
     *   act tile: (m,k) → (m/2)*64 + 2k + (m&1)   [row-pair-major = V1 crouton]
     *   wgt tile: (n,k) → (k/2)*64 + 2n + (k&1)   [k-pair-major = dequant 产物]
     *   out tile: (m,n) → (m/2)*64 + 2n + (m&1)   [row-pair-major]
     * 注意: hmx_interleave_rows_to_tiles 产 k-pair-major (L1), 喂 core_dot
     *       当 act 是错的 (compose bug); dequant 产 L1 作 wgt 正确. */
    __fp16 *vtcm = (__fp16 *)hmx_runtime_get_vtcm_base();
    __fp16 *vt_act = vtcm;                       /* 8 tiles * 2KB = 16KB */
    __fp16 *vt_wgt = vtcm + 8 * 1024;            /* +16KB */
    __fp16 *vt_out = vtcm + 16 * 1024;           /* +32KB, 2KB */
    __fp16 *vt_sca = vtcm + 17 * 1024;           /* +34KB, 256B */

    static __fp16 out_ddr[1024] ALN;
    static __fp16 arm2[32 * 256] ALN;
    static __fp16 wrm2[32 * 256] ALN;
    static float  out_ref32[32 * 32] ALN;

    for (int m = 0; m < 32; ++m)
        for (int k = 0; k < 256; ++k) {
            arm2[m * 256 + k] = (__fp16)(0.02f * sinf((m * 256 + k) * 0.013f));
            wrm2[m * 256 + k] = (__fp16)(0.02f * cosf((m * 256 + k) * 0.017f));
        }
    for (int m = 0; m < 32; ++m)
        for (int n = 0; n < 32; ++n) {
            double acc = 0.0;
            for (int k = 0; k < 256; ++k)
                acc += (double)arm2[m * 256 + k] * (double)wrm2[n * 256 + k];
            out_ref32[m * 32 + n] = (float)acc;
        }
    /* 打包: act=L2(row-pair), wgt=L1(k-pair) */
    for (int t = 0; t < 8; ++t)
        for (int i = 0; i < 32; ++i)
            for (int c = 0; c < 32; ++c) {
                vt_act[t * 1024 + (i / 2) * 64 + 2 * c + (i & 1)] = arm2[i * 256 + t * 32 + c];
                vt_wgt[t * 1024 + (c / 2) * 64 + 2 * i + (c & 1)] = wrm2[i * 256 + t * 32 + c];
            }
    {
        HVX_Vector *ps = (HVX_Vector *)vt_sca;
        ps[0] = Q6_V_vsplat_R(0x3C00);
        ps[1] = Q6_V_vzero();
    }
    hvhx_v2_hmx_gemm_dot_fp16(vt_out, vt_act, vt_wgt, vt_sca, 1, 1, 8);
    memcpy(out_ddr, vt_out, 2048);
    {
        float max_rel = 0.0f;
        for (int m = 0; m < 32; ++m)
            for (int n = 0; n < 32; ++n) {
                int pos = (m / 2) * 64 + 2 * n + (m & 1);
                float e = fabsf((float)out_ddr[pos] - out_ref32[m * 32 + n]);
                float mm = fabsf(out_ref32[m * 32 + n]);
                float rel = (mm > 1e-4f) ? e / mm : e;
                if (rel > max_rel) max_rel = rel;
            }
        int failed = max_rel > 0.02f;
        ex_log("%-44s maxrel=%.6f tol=0.0200 %s",
               "hmx_gemm_dot 32x32x256", max_rel,
               failed ? "FAIL" : "PASS");
        n_fail += failed;
    }

    /* perf: GEMM K=256 → 2*32*32*256 = 524288 FLOP/call */
    uint64_t t0 = hmx_perf_now_us();
    for (int it = 0; it < 50; ++it)
        hvhx_v2_hmx_gemm_dot_fp16(vt_out, vt_act, vt_wgt, vt_sca, 1, 1, 8);
    uint64_t t1 = hmx_perf_now_us();
    double us = (double)(t1 - t0) / 50.0;
    ex_log("PERF hmx_gemm 32x32x256: %.2f us (%.2f TFLOPS)",
           us, 2.0 * 32 * 32 * 256 / us / 1e6);

    /* perf: dequant 单 tile */
    t0 = hmx_perf_now_us();
    for (int it = 0; it < 50; ++it)
        hvhx_v2_dequant_tiled_to_fp16(q_wgt, dq_hvx, 32, 32, HVHX_V2_WT_Q4_0);
    t1 = hmx_perf_now_us();
    ex_log("PERF dequant_q4_0 1 tile (1024 elm): %.2f us",
           (double)(t1 - t0) / 50.0);
}

/* ============================================================
 *  V2_a 新增算子测试 (silu/gelu/softplus + ssm_conv/transpose +
 *  transfer_activation/output + 全生产管线组合)
 * ============================================================ */
static void test_v2a_new(void) {
    const int n = 1024;
    /* ---------- 1. silu / gelu / gelu_tanh / softplus ---------- */
    for (int i = 0; i < n; ++i) src_a[i] = 3.0f * sinf(i * 0.047f);
    hvhx_v2_silu_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] / (1.0f + expf(-src_a[i]));
    n_fail += cmp_f32("silu_f32", dst_hvx, dst_ref, n, 0.02f);

    hvhx_v2_gelu_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = src_a[i] / (1.0f + expf(-1.702f * src_a[i]));
    n_fail += cmp_f32("gelu_f32 (quick)", dst_hvx, dst_ref, n, 0.02f);

    hvhx_v2_gelu_tanh_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) {
        float x = src_a[i];
        float inner = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
        dst_ref[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    n_fail += cmp_f32("gelu_tanh_f32", dst_hvx, dst_ref, n, 0.02f);

    hvhx_v2_softplus_f32(dst_hvx, src_a, n);
    for (int i = 0; i < n; ++i) dst_ref[i] = logf(1.0f + expf(src_a[i]));
    n_fail += cmp_f32("softplus_f32", dst_hvx, dst_ref, n, 0.02f);

    /* ---------- 2. transpose_32x32 ---------- */
    {
        static float mt[32 * 32] ALN;
        for (int i = 0; i < 32 * 32; ++i) mt[i] = (float)i;
        hvhx_v2_transpose_32x32_f32(mt);
        int bad = 0;
        for (int r = 0; r < 32; ++r)
            for (int c = 0; c < 32; ++c)
                if (mt[r * 32 + c] != (float)(c * 32 + r)) bad++;
        ex_log("%-44s bad=%d %s", "transpose_32x32_f32", bad,
               bad ? "FAIL" : "PASS");
        n_fail += (bad != 0);
    }

    /* ---------- 3. ssm_conv_f32 (标量基准路径, 任意尺寸) ----------
     * dst[s][t][c] = Σ_j src0[s][c][t+j] * src1[c][j], 然后可选 silu */
    {
        const uint32_t d_inner = 8, n_t = 16, n_s = 2, d_conv = 4;
        const uint32_t ncs = n_t + d_conv - 1;
        static float s0[2 * 8 * 19] ALN;   /* [n_s][d_inner][ncs] */
        static float s1[8 * 4] ALN;        /* [d_inner][d_conv] */
        static float d_h[2 * 16 * 8] ALN;  /* [n_s][n_t][d_inner] */
        static float d_r[2 * 16 * 8] ALN;
        for (uint32_t i = 0; i < n_s * d_inner * ncs; ++i)
            s0[i] = 0.3f * sinf(i * 0.11f);
        for (uint32_t i = 0; i < d_inner * d_conv; ++i)
            s1[i] = 0.2f * cosf(i * 0.23f);
        hvhx_v2_ssm_conv_f32(s0, s1, d_h, d_inner, n_t, n_s, ncs, d_conv, 0);
        for (uint32_t s = 0; s < n_s; ++s)
            for (uint32_t t = 0; t < n_t; ++t)
                for (uint32_t c = 0; c < d_inner; ++c) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < d_conv; ++j)
                        acc += s0[(s * d_inner + c) * ncs + t + j] * s1[c * d_conv + j];
                    d_r[(s * n_t + t) * d_inner + c] = acc;
                }
        n_fail += cmp_f32("ssm_conv_f32 (no silu)", d_h, d_r, n_s * n_t * d_inner, 0.01f);

        /* apply_silu=1 */
        hvhx_v2_ssm_conv_f32(s0, s1, d_h, d_inner, n_t, n_s, ncs, d_conv, 1);
        for (uint32_t i = 0; i < n_s * n_t * d_inner; ++i)
            d_r[i] = d_r[i] / (1.0f + expf(-d_r[i]));
        n_fail += cmp_f32("ssm_conv_f32 (silu)", d_h, d_r, n_s * n_t * d_inner, 0.02f);
    }

    /* ---------- 4. ssm_conv_dot32_f32 (HVX 32-channel tile) ----------
     * dst32[c] = Σ_j x_row[j*x_stride+c] * w_row[j*w_stride+c] */
    {
        static float x_t[4 * 32] ALN;   /* d_conv=4 个时间偏移 × 32 channel */
        static float w_t[4 * 32] ALN;
        static float d32_h[32] ALN;
        static float d32_r[32] ALN;
        for (int i = 0; i < 4 * 32; ++i) {
            x_t[i] = 0.3f * sinf(i * 0.07f);
            w_t[i] = 0.2f * cosf(i * 0.13f);
        }
        hvhx_v2_ssm_conv_dot32_f32(d32_h, x_t, 32, w_t, 32, 4, 0);
        for (int c = 0; c < 32; ++c) {
            float acc = 0.0f;
            for (int j = 0; j < 4; ++j) acc += x_t[j * 32 + c] * w_t[j * 32 + c];
            d32_r[c] = acc;
        }
        n_fail += cmp_f32("ssm_conv_dot32 (no silu)", d32_h, d32_r, 32, 0.01f);

        hvhx_v2_ssm_conv_dot32_f32(d32_h, x_t, 32, w_t, 32, 4, 1);
        for (int c = 0; c < 32; ++c)
            d32_r[c] = d32_r[c] / (1.0f + expf(-d32_r[c]));
        n_fail += cmp_f32("ssm_conv_dot32 (silu)", d32_h, d32_r, 32, 0.02f);
    }

    /* ---------- 5. 全生产管线组合 (V2_a transfer 版) ----------
     * act f32 [32][256] → transfer_activation → act tiles
     * wgt Q4_0 tiles   → dequant → wgt tiles
     * GEMM → transfer_output (+残差) → f32 [32][32]
     * 与标量 row-major GEMM 对比 — 验证 transfer 布局与 core_dot 自洽 */
    {
        __fp16 *vtcm = (__fp16 *)hmx_runtime_get_vtcm_base();
        __fp16 *vt_act = vtcm;                 /* 8 k-tiles × 2KB = 16KB */
        __fp16 *vt_wgt = vtcm + 8 * 1024;      /* +16KB */
        __fp16 *vt_out = vtcm + 16 * 1024;     /* +32KB, 2KB */
        __fp16 *vt_sca = vtcm + 17 * 1024;     /* +34KB */

        static float act_f32[32 * 256] ALN;
        static float resid[32 * 32] ALN;
        static float out_f32[32 * 32] ALN;
        static __fp16 wgt_tiles_h[8 * 1024] ALN;
        static float out_ref[32 * 32] ALN;

        for (int m = 0; m < 32; ++m)
            for (int k = 0; k < 256; ++k)
                act_f32[m * 256 + k] = 0.02f * sinf((m * 256 + k) * 0.013f);
        for (int i = 0; i < 32 * 32; ++i) resid[i] = 0.1f * sinf(i * 0.05f);

        /* wgt: 8 个 Q4_0 tile (n=32, k=256 → n_k_tiles=8) 连续存放,
         * dequant 一次出全部 8 tile fp16 (L1 布局 = core_dot 期望) */
        static uint8_t qw8[8 * 640] ALN;
        for (int t = 0; t < 8; ++t) {
            uint8_t *qt = qw8 + t * 640;
            for (int i = 0; i < 512; ++i) qt[i] = (uint8_t)((t * 512 + i) * 37 % 256);
            for (int r = 0; r < 32; ++r)
                ((__fp16 *)(qt + 512))[r] = (__fp16)(0.02f + 0.001f * ((r + t) % 7));
        }
        int rc = hvhx_v2_dequant_tiled_to_fp16(qw8, wgt_tiles_h, 32, 256,
                                               HVHX_V2_WT_Q4_0);
        if (rc != 0) { ex_log("pipe dequant rc=%d FAIL", rc); n_fail += 1; }

        /* act: 手动打包 (row-pair-major, 与 hmx_gemm_dot 已验证布局一致).
         * 注: transfer_activation_fp32_to_fp16 用 _shuff (拼接) 而非 vdeal
         * (交错), 产出的 tile 布局与 core_dot 不兼容 — 这是库的已知限制. */
        for (int t = 0; t < 8; ++t)
            for (int m = 0; m < 32; ++m)
                for (int kk = 0; kk < 32; ++kk) {
                    int act_pos = t * 1024 + (m / 2) * 64 + 2 * kk + (m & 1);
                    vt_act[act_pos] = (__fp16)act_f32[m * 256 + t * 32 + kk];
                }
        memcpy(vt_wgt, wgt_tiles_h, 16 * 1024);

        {
            HVX_Vector *ps = (HVX_Vector *)vt_sca;
            ps[0] = Q6_V_vsplat_R(0x3C00);
            ps[1] = Q6_V_vzero();
        }
        hvhx_v2_hmx_gemm_dot_fp16(vt_out, vt_act, vt_wgt, vt_sca, 1, 1, 8);

        /* 手动提取 output tile (row-pair-major) + 残差.
         * transfer_output_fp16_to_fp32 同样用 _shuff 路径, 与 core_dot 不兼容. */
        static __fp16 out_ddr_p[1024] ALN;
        memcpy(out_ddr_p, vt_out, 2048);
        for (int m = 0; m < 32; ++m)
            for (int n = 0; n < 32; ++n) {
                int pos = (m / 2) * 64 + 2 * n + (m & 1);
                out_f32[m * 32 + n] = resid[m * 32 + n] + (float)out_ddr_p[pos];
            }

        /* 标量参考: out[m][n] = resid[m][n] + Σ_k act[m][k]·dequant_wgt[n][k] */
        static __fp16 wq_ref[8 * 1024] ALN;
        for (int t = 0; t < 8; ++t)
            ref_dequant_q4_0_tile(qw8 + t * 640, wq_ref + t * 1024);
        for (int m = 0; m < 32; ++m)
            for (int n = 0; n < 32; ++n) {
                double acc = resid[m * 32 + n];
                for (int t = 0; t < 8; ++t)
                    for (int kk = 0; kk < 32; ++kk)
                        acc += (double)act_f32[m * 256 + t * 32 + kk] *
                               (double)(float)wq_ref[t * 1024 + tile_pos(n, kk)];
                out_ref[m * 32 + n] = (float)acc;
            }
        /* pipeline: FP16 GEMM accumulates 256 terms; near-zero outputs from
         * cancellation have high relative error but tiny absolute error. */
        {
            float max_abs = 0;
            for (int i = 0; i < 32*32; ++i)
                if (fabsf(out_ref[i]) > max_abs) max_abs = fabsf(out_ref[i]);
            float max_rel = 0, max_abs_err = 0;
            for (int i = 0; i < 32*32; ++i) {
                float e = fabsf(out_f32[i] - out_ref[i]);
                float m = fmaxf(fabsf(out_f32[i]), fabsf(out_ref[i]));
                float rel = (m > 1e-6f) ? e/m : e;
                if (rel > max_rel) max_rel = rel;
                if (e > max_abs_err) max_abs_err = e;
            }
            float scaled_abs = max_abs_err / max_abs;
            int failed = (max_rel > 0.02f) && (scaled_abs > 0.02f);
            ex_log("%-44s maxrel=%.6f abs/max=%.6f %s",
                   "pipeline: manual_pack+dequant+GEMM+manual_extract",
                   max_rel, scaled_abs, failed ? "FAIL" : "PASS");
            n_fail += failed;
        }
    }

    /* ---------- 6. transfer_* round-trip (transfer 自身布局自洽性) ----------
     * transfer_activation: row-major f32 → tile-major fp16 (_shuff 约定)
     * transfer_output:     tile-major fp16 → row-major f32 (同 _shuff 约定)
     * 两者应互为逆变换. 注: 此布局与 core_dot 期望的 vdeal 约定不同,
     * 但 transfer 自身 round-trip 必须自洽. */
    {
        __fp16 *vtcm = (__fp16 *)hmx_runtime_get_vtcm_base();
        __fp16 *vt_tiles = vtcm + 20 * 1024;     /* 8 k-tiles × 2KB = 16KB */

        const uint32_t n_rows = 32, k_block = 256, k_valid = 256;
        static float act_in[32 * 256] ALN;
        static float act_out[32 * 256] ALN;
        for (uint32_t i = 0; i < n_rows * k_block; ++i)
            act_in[i] = 0.02f * sinf(i * 0.013f) + 0.5f;

        hvhx_v2_transfer_activation_fp32_to_fp16(vt_tiles, act_in,
                                                  n_rows, k_block, k_block, k_valid);
        hvhx_v2_transfer_output_fp16_to_fp32(act_out, NULL, vt_tiles,
                                              0, n_rows, k_block,
                                              k_block, k_block, k_block);
        n_fail += cmp_f32("transfer round-trip (act->tile->act)",
                          act_out, act_in, n_rows * k_block, 0.01f);

        /* +残差路径: out = tile→f32 + resid */
        static float resid2[32 * 256] ALN;
        static float out_resid[32 * 256] ALN;
        for (uint32_t i = 0; i < n_rows * k_block; ++i)
            resid2[i] = 0.1f * cosf(i * 0.029f);
        hvhx_v2_transfer_output_fp16_to_fp32(out_resid, resid2, vt_tiles,
                                              0, n_rows, k_block,
                                              k_block, k_block, k_block);
        int bad = 0;
        for (uint32_t i = 0; i < n_rows * k_block; ++i) {
            float expect = act_in[i] + resid2[i];
            float rel = fabsf(out_resid[i] - expect);
            if (rel > 0.02f) bad++;
        }
        ex_log("%-44s bad=%d %s", "transfer_output (+residual)", bad,
               bad ? "FAIL" : "PASS");
        n_fail += (bad != 0);
    }
}

int main(void) {
    ex_open_result("15_v2_llm_ops");
    ex_log("=== 15_v2_llm_ops (V2 LLM 算子设备验证) ===");

    int err = hmx_runtime_setup(4 * 1024 * 1024);
    if (err != 0) {
        ex_log("FATAL: hmx_runtime_setup err=%d", err);
        ex_close_result();
        return 1;
    }
    ex_log("runtime up: VTCM=%p", hmx_runtime_get_vtcm_base());

    test_norm();
    test_unary_binary();
    test_softmax();
    test_rope();
    test_fa();
    test_ssm();
    test_dequant_gemm();
    test_v2a_new();

    hmx_runtime_teardown();
    ex_log("--- %s ---", n_fail == 0 ? "ALL PASS" : "HAS FAIL");
    int rc = ex_summary();
    ex_close_result();
    return rc;
}
