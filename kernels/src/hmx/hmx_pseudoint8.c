/*
 * hmx_pseudoint8.c — HMX fp16 伪 int8 GEMM (u8 × i8 → u8 via fp16 MAC)
 * Module: hmx-lowlevel
 * Math:   out[m,n] = sat_u8( Σ_k act_u8[m,k] * wgt_i8[k,n] )  (经 fp16 MAC)
 * Note:   scale=1.0 时 byte-exact: u8/i8 整数 < 2048 在 fp16 精确, 37-bit acc
 *         整数积之和 < 2^27 精确, sat_u8 两端一致. 详见 docs/api_lowlevel.md.
 */
#include "hmx_kernels.h"
#include "hmx_common.h"

/* fp16 crouton 直接入参 (调用方负责 int8→fp16 转换 + crouton 布局).
 * act/wgt/scales/out 全 2KB 对齐 VTCM fp16 crouton.
 * 数学: out[m][n] = scale[n] * Σ_k act[m][k] * wgt[k][n]  (fp16) */
void hmx_pseudoint8_32x32x32_core(
    const __fp16 * __restrict__ act_fp16,
    const __fp16 * __restrict__ wgt_fp16,
    const __fp16 * __restrict__ scales,
    __fp16       * __restrict__ out_fp16)
{
    /* 直接复用 Phase 0 验证过的 fp16 GEMM core (deep + cvt(2), 20.4 TFLOPS 配置). */
    hmx_phase0_gemm_fp16_core(act_fp16, wgt_fp16, scales, out_fp16);
}
