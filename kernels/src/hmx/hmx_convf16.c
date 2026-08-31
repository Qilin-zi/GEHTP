/*
 * hmx_convf16.c — FP16 GEMM 卷积 (fp16 × fp16 → fp16), 11 个几何变体
 * Module: hmx-gemm-fp16
 * Math:   out[m,n] = bias[n] + Σ_k act[m,k] * wgt[k,n]  (fp32 acc → cvt.hf)
 * Note:   真 HMX fp16 systolic, 容差 ≤1 (fabsf). 几何变体在单 tile 层等价.
 *         详见 docs/api_hmx_gemm.md.
 */
#include "hmx_kernels.h"
#include "hmx_common.h"
#include "hmx_crouton.h"
#include "hmx_fields.h"
#include <string.h>

/* 标量 fp16 fallback (任意 M/K/N) */
static void convf16_scalar(const __fp16 *act, const __fp16 *wgt,
                           const __fp16 *bias, __fp16 *out,
                           uint32_t M, uint32_t K, uint32_t N)
{
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float acc = (float) bias[n];
            for (uint32_t k = 0; k < K; ++k) {
                acc += (float) act[m * K + k] *
                       (float) wgt[k * N + n];
            }
            out[m * N + n] = (__fp16) acc;
        }
    }
}

/* identity scales 区域: Scale=1.0 (0x3C00) 广播 + output_bias=0.
 * HMX 读此区域后 cvt 产出纯 Σ act*wgt (Phase0 per-element 验证). */
static void prep_scales_identity(__fp16 *scales)
{
#if defined(__hexagon__) || defined(__HVX__)
    HVX_Vector v_scale = Q6_V_vsplat_R(0x3c00);
    HVX_Vector *pv = (HVX_Vector *)scales;
    *pv++ = v_scale;
    *pv   = Q6_V_vzero();
#else
    memset(scales, 0, 256);
    for (int i = 0; i < 64; ++i) scales[i] = (__fp16)1.0f;
#endif
}

/* 32×32×32 tile 走 HMX (recon 验证路径). 返回 0=OK, -1=需 scalar fallback.
 * act/wgt/out 是连续 32×32 row-major; bias 是 32 元素 per-channel. */
static int convf16_hmx_tile(const __fp16 *act, const __fp16 *wgt,
                            const __fp16 *bias, __fp16 *out)
{
    if (hmx_runtime_get_vtcm_base() == NULL) {
        if (hmx_runtime_setup(2 * 1024 * 1024) != 0) return -1;
    }
    char *vtcm = (char *) hmx_runtime_get_vtcm_base();
    if (!vtcm) return -1;

    /* VTCM 布局 (全 2KB 对齐): act_cr@0, wgt_cr@2K, out_cr@4K, scales@6K */
    __fp16 *act_cr = (__fp16 *)(vtcm + 0x0000);
    __fp16 *wgt_cr = (__fp16 *)(vtcm + 0x0800);
    __fp16 *out_cr = (__fp16 *)(vtcm + 0x1000);
    __fp16 *sca    = (__fp16 *)(vtcm + 0x1800);

    hmx_pack_act_fp16(act_cr, act);
    hmx_pack_wgt_fp16(wgt_cr, wgt);
    prep_scales_identity(sca);
    memset(out_cr, 0, HMX_CR_FP16_SZ);

    hmx_phase0_gemm_fp16_core(act_cr, wgt_cr, sca, out_cr);

    __fp16 out_rm[HMX_CR_ELMS] __attribute__((aligned(128)));
    hmx_unpack_out_fp16(out_rm, out_cr);

    for (unsigned m = 0; m < HMX_CR_DIM; ++m)
        for (unsigned n = 0; n < HMX_CR_DIM; ++n)
            out[m * HMX_CR_DIM + n] = (__fp16)((float)out_rm[m * HMX_CR_DIM + n]
                                               + (float)bias[n]);
    return 0;
}

/* Phase 3: 多 tile M/N/K GEMM. M/K/N 必须是 32 倍数.
 * 对每个 (mt, nt) 输出 tile, 沿 K 维跨 kt 累加 (HMX acc 跨 activation+weight
 * pair 持续累加, 只 clracc 一次). 返回 0=OK, -1=scalar fallback. */
static int convf16_hmx_multi(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    if (hmx_runtime_get_vtcm_base() == NULL) {
        if (hmx_runtime_setup(2 * 1024 * 1024) != 0) return -1;
    }
    char *vtcm = (char *) hmx_runtime_get_vtcm_base();
    if (!vtcm) return -1;

    __fp16 *act_cr = (__fp16 *)(vtcm + 0x0000);
    __fp16 *wgt_cr = (__fp16 *)(vtcm + 0x0800);
    __fp16 *out_cr = (__fp16 *)(vtcm + 0x1000);
    __fp16 *sca    = (__fp16 *)(vtcm + 0x1800);
    prep_scales_identity(sca);

    __fp16 act_tile[HMX_CR_ELMS] __attribute__((aligned(128)));
    __fp16 wgt_tile[HMX_CR_ELMS] __attribute__((aligned(128)));
    __fp16 out_rm[HMX_CR_ELMS]   __attribute__((aligned(128)));

    const uint32_t n_mt = M / HMX_TILE_DIM;
    const uint32_t n_nt = N / HMX_TILE_DIM;
    const uint32_t n_kt = K / HMX_TILE_DIM;

    for (uint32_t mt = 0; mt < n_mt; ++mt) {
        for (uint32_t nt = 0; nt < n_nt; ++nt) {
            hmx_enable_execution();
            hmx_unit_acquire();
            asm volatile("mxclracc.hf" ::: "memory");
            asm volatile("bias = mxmem2(%0)" :: "r"(sca) : "memory");

            for (uint32_t kt = 0; kt < n_kt; ++kt) {
                /* gather act 32×32: act[(mt*32+i)*K + kt*32 + j] */
                for (unsigned i = 0; i < HMX_CR_DIM; ++i)
                    for (unsigned j = 0; j < HMX_CR_DIM; ++j)
                        act_tile[i * HMX_CR_DIM + j] =
                            act[(mt * HMX_CR_DIM + i) * K + kt * HMX_CR_DIM + j];
                /* gather wgt 32×32: wgt[(kt*32+j)*N + nt*32 + l] */
                for (unsigned j = 0; j < HMX_CR_DIM; ++j)
                    for (unsigned l = 0; l < HMX_CR_DIM; ++l)
                        wgt_tile[j * HMX_CR_DIM + l] =
                            wgt[(kt * HMX_CR_DIM + j) * N + nt * HMX_CR_DIM + l];

                hmx_pack_act_fp16(act_cr, act_tile);
                hmx_pack_wgt_fp16(wgt_cr, wgt_tile);

                asm volatile(
                    "{ activation.hf = mxmem(%0, %1):deep\n"
                    "  weight.hf = mxmem(%2, %3) }\n"
                    :: "r"(act_cr), "r"(HMX_ALIGN_2KB),
                       "r"(wgt_cr), "r"(HMX_ALIGN_2KB) : "memory");
            }

            memset(out_cr, 0, HMX_CR_FP16_SZ);
            asm volatile(
                "cvt.hf = acc(%0)\n"
                "mxmem(%1, %2) = cvt\n"
                :: "r"(2), "r"(out_cr), "r"(0) : "memory");
            hmx_unit_release();
            hmx_disable_execution();

            hmx_unpack_out_fp16(out_rm, out_cr);
            for (unsigned i = 0; i < HMX_CR_DIM; ++i)
                for (unsigned l = 0; l < HMX_CR_DIM; ++l)
                    out[(mt * HMX_CR_DIM + i) * N + nt * HMX_CR_DIM + l] =
                        (__fp16)((float)out_rm[i * HMX_CR_DIM + l]
                                 + (float)bias[nt * HMX_CR_DIM + l]);
        }
    }
    return 0;
}

/* HMX FP16 GEMM 主体: 32 倍数走 HMX tile/multi, 否则标量 */
static void convf16_hmx(const __fp16 *act, const __fp16 *wgt,
                        const __fp16 *bias, __fp16 *out,
                        uint32_t M, uint32_t K, uint32_t N)
{
    if (M % HMX_TILE_DIM == 0 && K % HMX_TILE_DIM == 0 &&
        N % HMX_TILE_DIM == 0) {
        if (M == HMX_TILE_DIM && K == HMX_TILE_DIM && N == HMX_TILE_DIM) {
            if (convf16_hmx_tile(act, wgt, bias, out) == 0) return;
        } else {
            if (convf16_hmx_multi(act, wgt, bias, out, M, K, N) == 0) return;
        }
    }
    convf16_scalar(act, wgt, bias, out, M, K, N);
}

/* 11 个 API 变体: 单 crouton 层面几何等价 (32×32×32 GEMM).
 * 多 spatial tile 几何差异 (dY/dC/SP_MASK) 留 Phase 3. */
void hmx_convf16(const __fp16 *act, const __fp16 *wgt,
                 const __fp16 *bias, __fp16 *out,
                 uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_1x1_stride1(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_1x1_stride1_unaligned(const __fp16 *act, const __fp16 *wgt,
                                       const __fp16 *bias, __fp16 *out,
                                       uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_1xN_stride2(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_Nx1_stride2(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_stride1(const __fp16 *act, const __fp16 *wgt,
                         const __fp16 *bias, __fp16 *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_stride1_aligned(const __fp16 *act, const __fp16 *wgt,
                                 const __fp16 *bias, __fp16 *out,
                                 uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_stride2(const __fp16 *act, const __fp16 *wgt,
                         const __fp16 *bias, __fp16 *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_NxN_stride1(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_5x5_stride1(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

void hmx_convf16_dilate_stride1(const __fp16 *act, const __fp16 *wgt,
                                const __fp16 *bias, __fp16 *out,
                                uint32_t M, uint32_t K, uint32_t N)
{
    convf16_hmx(act, wgt, bias, out, M, K, N);
}

/* ============================================================
 *  Phase 0 验证 kernel: 单 crouton fp16 GEMM (32×32×32)
 *  VTCM 指针入参 (2KB 对齐 crouton). identity scale → 纯 MAC.
 *  设备 per-element PASS (20.4 TFLOPS, 5 mode).
 * ============================================================ */
void hmx_phase0_gemm_fp16_core(const __fp16 *act_vtcm,
                               const __fp16 *wgt_vtcm,
                               const __fp16 *scales_vtcm,
                               __fp16       *out_vtcm)
{
    hmx_enable_execution();
    hmx_unit_acquire();

    asm volatile("mxclracc.hf" ::: "memory");
    asm volatile("bias = mxmem2(%0)" :: "r"(scales_vtcm) : "memory");

    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "  weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act_vtcm), "r"(HMX_ALIGN_2KB),
           "r"(wgt_vtcm), "r"(HMX_ALIGN_2KB)
        : "memory");

    asm volatile(
        "cvt.hf = acc(%0)\n"
        "mxmem(%1, %2) = cvt\n"
        :: "r"(2), "r"(out_vtcm), "r"(0)
        : "memory");

    hmx_unit_release();
    hmx_disable_execution();
}

/* ============================================================
 *  Phase 1 诊断 core: 可配置 act_rt / deep / writeback 模式
 *  mode 0: deep, act_rt=0x7FF, cvt(2)+mxmem=cvt  (Phase 0 原)
 *  mode 1: non-deep, act_rt=0x7FF, cvt(2)+mxmem=cvt
 *  mode 2: non-deep, act_rt=0x3FC, mxmem_AR_after_hf (recon)
 *  mode 3: non-deep, act_rt=0x77C, mxmem_AR_after_hf (PRM basic)
 *  mode 4: deep, act_rt=0x7FF, mxmem_AR_after_hf
 * ============================================================ */
void hmx_gemm_fp16_crouton_ex(const __fp16 *act_vtcm,
                               const __fp16 *wgt_vtcm,
                               const __fp16 *scales_vtcm,
                               __fp16       *out_vtcm,
                               unsigned mode)
{
    hmx_enable_execution();
    hmx_unit_acquire();

    Q6_mxclracc_hf();
    Q6_bias_mxmem2_A((void *)scales_vtcm);

    switch (mode) {
    case 0:
        asm volatile(
            "{ activation.hf = mxmem(%0, %1):deep\n"
            "  weight.hf = mxmem(%2, %3) }\n"
            :: "r"(act_vtcm), "r"(HMX_ACT_RT_FULL),
               "r"(wgt_vtcm), "r"(HMX_WGT_RT_2KB) : "memory");
        asm volatile(
            "cvt.hf = acc(%0)\n"
            "mxmem(%1, %2) = cvt\n"
            :: "r"(2), "r"(out_vtcm), "r"(0) : "memory");
        break;
    case 1:
        asm volatile(
            "{ activation.hf = mxmem(%0, %1)\n"
            "  weight.hf = mxmem(%2, %3) }\n"
            :: "r"(act_vtcm), "r"(HMX_ACT_RT_FULL),
               "r"(wgt_vtcm), "r"(HMX_WGT_RT_2KB) : "memory");
        asm volatile(
            "cvt.hf = acc(%0)\n"
            "mxmem(%1, %2) = cvt\n"
            :: "r"(2), "r"(out_vtcm), "r"(0) : "memory");
        break;
    case 2:
        Q6_activation_hf_mxmem_RR((uintptr_t)act_vtcm, HMX_ACT_RT_BASIC);
        Q6_weight_hf_mxmem_RR((uintptr_t)wgt_vtcm, HMX_WGT_RT_2KB);
        Q6_mxmem_AR_after_hf((void *)out_vtcm, 0u);
        break;
    case 3:
        Q6_activation_hf_mxmem_RR((uintptr_t)act_vtcm, 0x77Cu);
        Q6_weight_hf_mxmem_RR((uintptr_t)wgt_vtcm, HMX_WGT_RT_2KB);
        Q6_mxmem_AR_after_hf((void *)out_vtcm, 0u);
        break;
    case 4:
        Q6_activation_hf_mxmem_RR_deep((uintptr_t)act_vtcm, HMX_ACT_RT_FULL);
        Q6_weight_hf_mxmem_RR((uintptr_t)wgt_vtcm, HMX_WGT_RT_2KB);
        Q6_mxmem_AR_after_hf((void *)out_vtcm, 0u);
        break;
    default:
        break;
    }

    hmx_unit_release();
    hmx_disable_execution();
}
