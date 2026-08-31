/*
 * 14_hmx_peak_gemm — 裸 HMX K-loop 达峰值 fp16 GEMM (正确性 + TFLOPS)
 * =====================================================================
 * 教学: 公开 hmx_convf16 是 correctness-first 的参考实现, 每 tile 重新
 *       gather+pack, 在大 GEMM 上只有 ~0.5 GFLOPS (HMX 被 CPU 搬数据饿死).
 *       要逼近 HMX 硬件峰值 (~12-20 TFLOPS), 必须预打包 act/wgt 进 VTCM
 *       crouton 布局, 然后用裸 asm K-loop 背靠背发 MAC, 单次 clracc/cvt.
 *
 *       本例做 M=32, N=32, K=NK*32 的深 GEMM:
 *         1. 一次性把 NK 个 act/wgt 32×32 slice 打包进 VTCM crouton
 *         2. 裸 K-loop: clracc → bias → NK 条 activation.hf/weight.hf (共享 acc) → cvt
 *         3. 与标量 golden 逐元素对比 (容差 1 ULP, fp16 cvt 截断)
 *         4. 计时裸 K-loop + 对照公开 wrapper, 打印 TFLOPS
 *
 *       这是 docs/api_lowlevel.md "高级路径" 的可跑样例. 一般应用用 hmx_convf16
 *       即可 (数值相同); 只有大 GEMM 吞吐敏感时才走本例的低级模式.
 *
 * 数学: out[m,n] = bias[n] + Σ_{kk=0..NK-1} Σ_{k=0..31} act[m, kk*32+k] * wgt[kk*32+k, n]
 *       (M=N=32, K=NK*32; HMX acc 在 fp32 累加, 末了 cvt.hf 截断到 fp16)
 */
#include "hvxhmx.h"
#include "hmx_fields.h"    /* 低级路径: HMX_ALIGN_2KB 字段 (高级用户单独 include, 见 hvxhmx.h 注释) */
#include "example_util.h"

#ifndef NK
#define NK 8                 /* K-loop 背靠背 MAC 的 tile 数; K = NK*32. 调大→更逼近峰值 */
#endif
#define M  32
#define N  32
#define K  (NK * 32)
#define NITER_RAW   2000     /* 裸 K-loop 计时迭代 (快, 多跑几次稳定) */
#define NITER_WRAP  20       /* wrapper 计时迭代 (慢, 少跑) */

/* flat GEMM 数据 (DDR) */
static __fp16 act   [M * K] __attribute__((aligned(128)));
static __fp16 wgt   [K * N] __attribute__((aligned(128)));
static __fp16 bias  [N]     __attribute__((aligned(128)));
static __fp16 out   [M * N] __attribute__((aligned(128)));
static __fp16 gold  [M * N] __attribute__((aligned(128)));

/* 每片 32×32 slice 的行主序中转 (pack helper 要求连续 32×32 源) */
static __fp16 act_tile[HMX_CR_ELMS] __attribute__((aligned(128)));
static __fp16 wgt_tile[HMX_CR_ELMS] __attribute__((aligned(128)));
static __fp16 out_rm  [HMX_CR_ELMS] __attribute__((aligned(128)));

/* VTCM crouton 指针 (runtime 后填充) */
static __fp16 *act_cr[NK];
static __fp16 *wgt_cr[NK];
static __fp16 *out_cr;
static __fp16 *sca;

/* 裸 K-loop: clracc + bias + NK 条背靠背 MAC + cvt. acc 跨 NK tile 持续累加. */
static inline void raw_kloop_compute(void)
{
    asm volatile("mxclracc.hf" ::: "memory");
    asm volatile("bias = mxmem2(%0)" :: "r"(sca) : "memory");
    for (int kk = 0; kk < NK; kk++) {
        asm volatile(
            "{ activation.hf = mxmem(%0, %1):deep\n"
            "  weight.hf = mxmem(%2, %3) }\n"
            :: "r"(act_cr[kk]), "r"(HMX_ALIGN_2KB),
               "r"(wgt_cr[kk]), "r"(HMX_ALIGN_2KB) : "memory");
    }
    asm volatile(
        "cvt.hf = acc(%0)\n"
        "mxmem(%1, %2) = cvt\n"
        :: "r"(2), "r"(out_cr), "r"(0) : "memory");
}

/* 一次性把 NK 个 act/wgt slice 打包进 VTCM crouton (计时外, 只做一次) */
static void pack_all_into_vtcm(char *vtcm)
{
    for (int kk = 0; kk < NK; kk++) {
        act_cr[kk] = (__fp16 *)(vtcm + (0      + kk) * HMX_CR_FP16_SZ);
        wgt_cr[kk] = (__fp16 *)(vtcm + (NK     + kk) * HMX_CR_FP16_SZ);
        /* gather act 32×32: act[m, kk*32 + j] */
        for (unsigned m = 0; m < M; m++)
            for (unsigned j = 0; j < HMX_CR_DIM; j++)
                act_tile[m * HMX_CR_DIM + j] = act[m * K + kk * HMX_CR_DIM + j];
        /* gather wgt 32×32: wgt[kk*32 + j, n] */
        for (unsigned j = 0; j < HMX_CR_DIM; j++)
            for (unsigned n = 0; n < N; n++)
                wgt_tile[j * HMX_CR_DIM + n] = wgt[(kk * HMX_CR_DIM + j) * N + n];
        hmx_pack_act_fp16(act_cr[kk], act_tile);
        hmx_pack_wgt_fp16(wgt_cr[kk], wgt_tile);
    }
    out_cr = (__fp16 *)(vtcm + (2 * NK)     * HMX_CR_FP16_SZ);
    sca    = (__fp16 *)(vtcm + (2 * NK + 1) * HMX_CR_FP16_SZ);
    /* identity scales: scale=1.0 (0x3C00) 广播 + output_bias=0 → cvt 产出纯 Σ act*wgt */
    HVX_Vector *ps = (HVX_Vector *)sca;
    ps[0] = Q6_V_vsplat_R(0x3C00);
    ps[1] = Q6_V_vzero();
}

int main(void)
{
    ex_open_result("14_hmx_peak_gemm");

    if (hmx_runtime_setup(2 * 1024 * 1024) != 0) {
        ex_log("FATAL: runtime setup FAIL");
        ex_check("hmx_runtime_setup", 1, 0);
        return ex_summary();
    }
    char *vtcm = (char *) hmx_runtime_get_vtcm_base();
    if (!vtcm) { ex_log("FATAL: no VTCM"); ex_check("VTCM base non-NULL", 1, 0); return ex_summary(); }

    /* 数据: act/wgt 小幅值, 避开 ±1.0 边界 + 保证 NK tile 累加不溢出 fp16 */
    ex_fill_f16(act,  M * K, 7,  0.02f);
    ex_fill_f16(wgt,  K * N, 9,  0.02f);
    ex_fill_f16(bias, N,     11, 0.5f);

    pack_all_into_vtcm(vtcm);

    /* ---- 1. 裸 K-loop 正确性 (vs 标量 golden) ---- */
    hmx_enable_execution();
    hmx_unit_acquire();
    raw_kloop_compute();
    hmx_unit_release();
    hmx_disable_execution();

    hmx_unpack_out_fp16(out_rm, out_cr);
    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++)
            out[m * N + n] = (__fp16)((float)out_rm[m * HMX_CR_DIM + n] + (float)bias[n]);

    for (unsigned m = 0; m < M; m++)
        for (unsigned n = 0; n < N; n++) {
            float a = (float)bias[n];
            for (unsigned k = 0; k < (unsigned)K; k++)
                a += (float)act[m * K + k] * (float)wgt[k * N + n];
            gold[m * N + n] = (__fp16)a;
        }
    int maxerr = 0;
    for (unsigned i = 0; i < M * N; i++) {
        float d = (float)out[i] - (float)gold[i];
        if (d < 0) d = -d;
        int ud = (int)(d + 0.5f);
        if (ud > maxerr) maxerr = ud;
    }
    ex_check("raw K-loop vs golden (fp16)", maxerr, 1);

    /* ---- 2. 裸 K-loop 峰值计时 (acquire 提外, 纯 MAC+clracc+cvt) ---- */
    double flop = 2.0 * (double)M * (double)K * (double)N;
    hmx_enable_execution();
    hmx_unit_acquire();
    uint64_t t0 = hmx_perf_now_us();
    for (int i = 0; i < NITER_RAW; i++) raw_kloop_compute();
    uint64_t t1 = hmx_perf_now_us();
    hmx_unit_release();
    hmx_disable_execution();
    double us_raw = (double)(t1 - t0) / NITER_RAW;
    double tf_raw = flop / (us_raw * 1e-6) / 1e12;
    ex_log("raw K-loop  M=%d N=%d K=%d : %.2f us/call  %.2f TFLOPS", M, N, K, us_raw, tf_raw);

    /* ---- 3. 对照: 公开 wrapper 同尺寸 (展示差距) ---- */
    hmx_convf16(act, wgt, bias, out, M, K, N);   /* warmup */
    t0 = hmx_perf_now_us();
    for (int i = 0; i < NITER_WRAP; i++) hmx_convf16(act, wgt, bias, out, M, K, N);
    t1 = hmx_perf_now_us();
    double us_wrap = (double)(t1 - t0) / NITER_WRAP;
    double tf_wrap = flop / (us_wrap * 1e-6) / 1e12;
    ex_log("hmx_convf16 M=%d N=%d K=%d : %.1f us/call  %.4f TFLOPS", M, N, K, us_wrap, tf_wrap);
    ex_log("speedup raw/wrapper = %.0fx  (wrapper 受每 tile gather+pack 限制)", us_wrap / us_raw);

    hmx_runtime_teardown();
    return ex_summary();
}
