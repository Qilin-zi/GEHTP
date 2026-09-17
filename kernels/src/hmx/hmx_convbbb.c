/*
 * hmx_convbbb.c — INT8 GEMM 卷积族 (u8 × u8 → u8), 9 个几何变体
 * Module: hmx-gemm-int8
 * Math:   out[m,n] = sat_u8( bias[n] + Σ_k act_u8[m,k] * wgt_u8[k,n] )
 * Note:   本设备 int8 HMX 走 HVX int8 GEMM (见 docs/data_layout.md), exact.
 *         K 须 32 倍数; stride/dilate/1x1 等几何在 caller im2col 层.
 */
#include "hmx_kernels.h"
#include "hmx_common.h"
#include "hvx_int8gemm.h"

#include <string.h>

/* ============================================================
 *  公开 API: 9 个 convbbb 变体
 *  --------------------------------------------------------
 *  V81 gen5_gvm FUSA int8 HMX 是 silent NOP (见记忆
 *  v81-gen5gvm-fusa-int8-hmx-broken). 本族改走已验证的 HVX
 *  int8 GEMM (uint8×int8→uint8 sat, int16 累加器).
 *
 *  32×32×32 单 tile 走 HVX; 其他尺寸退标量.
 * ============================================================ */

/* 通用: 处理 1x1 / 3x3 等普通卷积 */
void hmx_convbbb(const uint8_t * __restrict__ act,
                 const uint8_t * __restrict__ wgt,
                 const int32_t  * __restrict__ bias,
                 uint8_t        * __restrict__ out,
                 uint32_t M, uint32_t K, uint32_t N)
{
    if (M % HMX_TILE_DIM == 0 && K % HMX_TILE_DIM == 0 &&
        N % HMX_TILE_DIM == 0 && N <= 1024) {
        static int16_t bias16[1024] __attribute__((aligned(128)));
        for (uint32_t i = 0; i < N; ++i)
            bias16[i] = (int16_t)bias[i];
        hvx_int8gemm_bias_multi(act, (const int8_t *)wgt, bias16, out, M, K, N);
        return;
    }

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int32_t acc = (int32_t) bias[n];
            for (uint32_t k = 0; k < K; ++k) {
                acc += (int32_t) act[m * K + k] *
                       (int32_t) wgt[k * N + n];
            }
            out[m * N + n] = HVHX_SAT_U8(acc);
        }
    }
}

void hmx_convbbb1x1_stride1(const uint8_t *act, const uint8_t *wgt,
                            const int32_t *bias, uint8_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

void hmx_convbbb1x1_stride1_unaligned(const uint8_t *act,
                                      const uint8_t *wgt,
                                      const int32_t *bias,
                                      uint8_t *out,
                                      uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

void hmx_convbbb1xN_stride2(const uint8_t *act, const uint8_t *wgt,
                            const int32_t *bias, uint8_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

void hmx_convbbbNx1_stride2(const uint8_t *act, const uint8_t *wgt,
                            const int32_t *bias, uint8_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

void hmx_convbbb_stride1(const uint8_t *act, const uint8_t *wgt,
                         const int32_t *bias, uint8_t *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

void hmx_convbbb_stride1_aligned(const uint8_t *act, const uint8_t *wgt,
                                 const int32_t *bias, uint8_t *out,
                                 uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

void hmx_convbbb_stride2(const uint8_t *act, const uint8_t *wgt,
                         const int32_t *bias, uint8_t *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

void hmx_convbbb_dilate_stride1(const uint8_t *act, const uint8_t *wgt,
                                const int32_t *bias, uint8_t *out,
                                uint32_t M, uint32_t K, uint32_t N)
{
    hmx_convbbb(act, wgt, bias, out, M, K, N);
}

/* ============================================================
 *  Phase 1 验证 kernel: 单 crouton int8 GEMM (32×32×32)
 *  ----------------------------------------------------------
 *  V81 HMX int8 路径 (SDK 19.0.07 实有 intrinsic, 非不存在的 .b):
 *    Q6_mxclracc               — 清累加器 (int8 用 PLAIN, 非 .hf; 生产 disasm 确认)
 *    Q6_bias_mxmem2_A          — 装 256B bias region (per-ch 64-bit: Scale fp16 等)
 *    Q6_activation_ub_mxmem_RR — UNSIGNED byte 激活 (dense blk_sm, act_rt=0x3FC)
 *    Q6_weight_b_mxmem_RR      — SIGNED byte 权重 (wgt_rt=0x7FF)
 *    Q6_mxmem_AR_after_sat_ub  — UNSIGNED sat [0,255] 写回
 *
 *  数学: out[m,n] = sat_u8( (acc + input_bias) * Scale + output_bias )
 *        acc = Σ_k act_u8[m,k] * wgt_i8[k,n]   (非对称量化: 无符号激活 × 有符号权重)
 *  Scale=1.0 (0x3C00), biases=0 → out = sat_u8(acc)
 *
 *  注: SDK 无 Q6_activation_b / Q6_mxmem_AR_after_b (recon kernel 8,9 源码用的
 *      是不存在 builtin, 在本 SDK 编译报 "Unmatched weight instruction" ICE).
 *      生产 .so 用 activation.ub:cm (压缩模式), 这里先走 dense ub 最简路径.
 * ============================================================ */
#define HMX_INT8_ACT_RT_BASIC  0x3FCu   /* SM=0b11100, channel_stop=31, dY=0 */
#define HMX_INT8_WGT_RT_2KB    0x7FFu   /* dW=2048-1 (fp16 format: 16 vec) */
#define HMX_INT8_WGT_RT_1KB    0x3FFu   /* dW=1024-1 (int8 format: 8 vec, 生产用) */

void hmx_phase1_gemm_int8_core(const uint8_t *act_vtcm,
                                const int8_t  *wgt_vtcm,
                                const void    *bias_vtcm,
                                uint8_t       *out_vtcm)
{
    hmx_enable_execution();
    hmx_unit_acquire();

    Q6_mxclracc();
    Q6_bias_mxmem2_A((void *)bias_vtcm);
    Q6_activation_ub_mxmem_RR((unsigned int)(uintptr_t)act_vtcm, HMX_INT8_ACT_RT_BASIC);
    Q6_weight_b_mxmem_RR((unsigned int)(uintptr_t)wgt_vtcm, HMX_INT8_WGT_RT_1KB);
    Q6_mxmem_AR_after_sat_ub((void *)out_vtcm, 0u);

    hmx_unit_release();
    hmx_disable_execution();
}

/* ============================================================
 *  Phase 1 诊断 core: 多模式 int8 路径 (用于布局/模式探测)
 *  mode 0: dense ub + b + after:sat.ub, act_rt=0x3FC  (blk_sm)
 *  mode 1: ub:cm + b + after:cm:sat.ub, act_rt=0x3FC  (blk_dm 压缩)
 *  mode 2: ub:deep + b + after:sat.ub, act_rt=0x7FF   (dm deep, 类 fp16 验证路径)
 *  mode 3: ub:deep + b + after:sat.ub, act_rt=0x3FC   (dm deep, recon act_rt)
 * ============================================================ */
void hmx_gemm_int8_crouton_ex(const uint8_t *act_vtcm,
                                const int8_t  *wgt_vtcm,
                                const void    *bias_vtcm,
                                uint8_t       *out_vtcm,
                                unsigned mode)
{
    hmx_enable_execution();
    hmx_unit_acquire();

    Q6_mxclracc();
    Q6_bias_mxmem2_A((void *)bias_vtcm);

    switch (mode) {
    case 0:
        Q6_activation_ub_mxmem_RR((unsigned int)(uintptr_t)act_vtcm, 0x3FCu);
        Q6_weight_b_mxmem_RR((unsigned int)(uintptr_t)wgt_vtcm, HMX_INT8_WGT_RT_2KB);
        Q6_mxmem_AR_after_sat_ub((void *)out_vtcm, 0u);
        break;
    case 1:
        Q6_activation_ub_mxmem_RR_cm((unsigned int)(uintptr_t)act_vtcm, 0x3FCu);
        Q6_weight_b_mxmem_RR((unsigned int)(uintptr_t)wgt_vtcm, HMX_INT8_WGT_RT_2KB);
        Q6_mxmem_AR_after_cm_sat_ub((void *)out_vtcm, 0u);
        break;
    case 2:
        Q6_activation_ub_mxmem_RR_deep((unsigned int)(uintptr_t)act_vtcm, 0x7FFu);
        Q6_weight_b_mxmem_RR((unsigned int)(uintptr_t)wgt_vtcm, HMX_INT8_WGT_RT_2KB);
        Q6_mxmem_AR_after_sat_ub((void *)out_vtcm, 0u);
        break;
    case 3:
        Q6_activation_ub_mxmem_RR_deep((unsigned int)(uintptr_t)act_vtcm, 0x3FCu);
        Q6_weight_b_mxmem_RR((unsigned int)(uintptr_t)wgt_vtcm, HMX_INT8_WGT_RT_2KB);
        Q6_mxmem_AR_after_sat_ub((void *)out_vtcm, 0u);
        break;
    default:
        break;
    }

    hmx_unit_release();
    hmx_disable_execution();
}
