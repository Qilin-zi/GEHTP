/*
 * hmx_kernels.h — V81 HMX kernel 汇总入口
 * =====================================================================
 * HMX (Hexagon Matrix eXtension) 提供 32x32 systolic GEMM, 适用于
 * 卷积和全连接层. 本目录仅包含 V81 目标 (v81 generic), 不含 v73/v75/v79/v85
 * 兼容变体.
 *
 * 数学形式 (所有卷积):
 *   output[h, w, oc] = bias[oc] + Σ_{kh, kw, ic} input[h*S+kh*D, w*S+kw*D, ic]
 *                                          * weight[kh, kw, ic, oc]
 *
 * VTCM 约束:
 *   - HMX crouton 2KB 对齐 (32 行 × 32 列 × 2B)
 *   - 累加器 37-bit, 一次 GEMM tile = 32×32×K (K 是 K 维长度)
 */
#ifndef HVXHMX_HMX_KERNELS_H
#define HVXHMX_HMX_KERNELS_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  Conv2D b × b → b (u8 输入, u8 权重, u8 输出)
 * ============================================================ */
void hmx_convbbb(const uint8_t * __restrict__ act,
                 const uint8_t * __restrict__ wgt,
                 const int32_t  * __restrict__ bias,
                 uint8_t        * __restrict__ out,
                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbb1x1_stride1(const uint8_t *act, const uint8_t *wgt,
                            const int32_t *bias, uint8_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbb1x1_stride1_unaligned(const uint8_t *act, const uint8_t *wgt,
                                      const int32_t *bias, uint8_t *out,
                                      uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbb1xN_stride2(const uint8_t *act, const uint8_t *wgt,
                            const int32_t *bias, uint8_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbbNx1_stride2(const uint8_t *act, const uint8_t *wgt,
                            const int32_t *bias, uint8_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbb_stride1(const uint8_t *act, const uint8_t *wgt,
                         const int32_t *bias, uint8_t *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbb_stride1_aligned(const uint8_t *act, const uint8_t *wgt,
                                 const int32_t *bias, uint8_t *out,
                                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbb_stride2(const uint8_t *act, const uint8_t *wgt,
                         const int32_t *bias, uint8_t *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convbbb_dilate_stride1(const uint8_t *act, const uint8_t *wgt,
                                const int32_t *bias, uint8_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D b × b → h (u8 输入, u8 权重, u16 输出)
 * ============================================================ */
void hmx_convbbh(const uint8_t * __restrict__ act,
                 const uint8_t * __restrict__ wgt,
                 const int32_t  * __restrict__ bias,
                 uint16_t       * __restrict__ out,
                 uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D b × c → b (u8 输入, i16 权重, u8 输出)
 * ============================================================ */
void hmx_convbcb(const uint8_t * __restrict__ act,
                 const int16_t  * __restrict__ wgt,
                 const int32_t  * __restrict__ bias,
                 uint8_t        * __restrict__ out,
                 uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D b × n → b (u8 输入, i16 权重, u8 输出) — 4 个变体
 * ============================================================ */
void hmx_convbnb(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint8_t *out,
                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convbnb_1x1_stride1(const uint8_t *act, const int16_t *wgt,
                             const int32_t *bias, uint8_t *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convbnb_1x1_stride1_sparsity(const uint8_t *act,
                                      const int16_t *wgt,
                                      const uint8_t *sparsity,
                                      const int32_t *bias, uint8_t *out,
                                      uint32_t M, uint32_t K, uint32_t N);

void hmx_convbnb_1x1_stride1_unaligned(const uint8_t *act, const int16_t *wgt,
                                       const int32_t *bias, uint8_t *out,
                                       uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D f × f → f (fp16 输入, fp16 权重, fp16 输出) — 11 个
 * ============================================================ */
void hmx_convf16(const __fp16 *act, const __fp16 *wgt,
                 const __fp16 *bias, __fp16 *out,
                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_1x1_stride1(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_stride1(const __fp16 *act, const __fp16 *wgt,
                         const __fp16 *bias, __fp16 *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_stride2(const __fp16 *act, const __fp16 *wgt,
                         const __fp16 *bias, __fp16 *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_NxN_stride1(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_5x5_stride1(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_dilate_stride1(const __fp16 *act, const __fp16 *wgt,
                                const __fp16 *bias, __fp16 *out,
                                uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_stride1_aligned(const __fp16 *act, const __fp16 *wgt,
                                 const __fp16 *bias, __fp16 *out,
                                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_1x1_stride1_unaligned(const __fp16 *act, const __fp16 *wgt,
                                       const __fp16 *bias, __fp16 *out,
                                       uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_1xN_stride2(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convf16_Nx1_stride2(const __fp16 *act, const __fp16 *wgt,
                             const __fp16 *bias, __fp16 *out,
                             uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D h × b → h (u8 act, i8 wgt, u16 out) — .so 反汇编 ground truth
 *  activation.ub + weight.b → mxmem:after:sat.uh = acc:2x1
 *  :2x1 = u16 输出格式标识 (非空间扩展, uh_writeback 探测确认).
 *  本设备 int8 HMX + .uh 写回均坏 → 走 HVX int8 GEMM.
 * ============================================================ */
void hmx_convhbh(const uint8_t *act, const int8_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh1x1_stride1(const uint8_t *act, const int8_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh1x1deep_stride1(const uint8_t *act, const int8_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh1x1_stride1_unaligned(const uint8_t *act, const int8_t *wgt,
                                      const int32_t *bias, uint16_t *out,
                                      uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh1xN_stride2(const uint8_t *act, const int8_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbhNx1_stride2(const uint8_t *act, const int8_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh_NxN_stride1(const uint8_t *act, const int8_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh_5x5_stride1(const uint8_t *act, const int8_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh_stride1(const uint8_t *act, const int8_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh_stride1_aligned(const uint8_t *act, const int8_t *wgt,
                                 const int32_t *bias, uint16_t *out,
                                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh_stride2(const uint8_t *act, const int8_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convhbh_dilate_stride1(const uint8_t *act, const int8_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D h × n → h (u8 act, i16 wgt, u16 out)
 *  activation.ub + weight.n → mxmem:after:sat.uh = acc:2x1
 * ============================================================ */
void hmx_convhnh(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh1x1_stride1(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh_NxN_stride1(const uint8_t *act, const int16_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh1x1deep_stride1(const uint8_t *act, const int16_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh1x1_stride1_unaligned(const uint8_t *act, const int16_t *wgt,
                                      const int32_t *bias, uint16_t *out,
                                      uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh1xN_stride2(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnhNx1_stride2(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh_5x5_stride1(const uint8_t *act, const int16_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh_stride1(const uint8_t *act, const int16_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh_stride1_aligned(const uint8_t *act, const int16_t *wgt,
                                 const int32_t *bias, uint16_t *out,
                                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh_stride2(const uint8_t *act, const int16_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N);

void hmx_convhnh_dilate_stride1(const uint8_t *act, const int16_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D h × c → h (u8 act, i16 wgt, u16 out) — :2x2 写回格式
 *  activation.ub + weight.c → cvt.uh = acc:2x2
 *  与 convhnh 数学等价 (u8×i16→u16), 仅格式差异.
 * ============================================================ */
void hmx_convhch(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convhch1x1_stride1(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhch1x1deep_stride1(const uint8_t *act, const int16_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

void hmx_convhch1xN_stride2(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhchNx1_stride2(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhch_5x5_stride1(const uint8_t *act, const int16_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Conv2D h × h → h (u8 act, i8 wgt, u16 out) — :2x2 写回格式
 *  activation.ub + weight.b → cvt.uh = acc:2x2
 *  与 convhbh 数学等价 (u8×i8→u16), 仅格式差异.
 * ============================================================ */
void hmx_convhhh(const uint8_t *act, const int8_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);

void hmx_convhhh1x1_stride1(const uint8_t *act, const int8_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N);

void hmx_convhhh1x1deep_stride1(const uint8_t *act, const int8_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

void hmx_convhhh_NxN_stride1(const uint8_t *act, const int8_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N);

void hmx_convhhh_dilate_stride1(const uint8_t *act, const int8_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N);

/* ============================================================
 *  Depthwise Conv (深度可分离)
 * ============================================================ */
void hmx_dwconvf16(const __fp16 *act, const __fp16 *wgt,
                   const __fp16 *bias, __fp16 *out,
                   uint32_t H, uint32_t W, uint32_t C);

void hmx_dwconvf16_dilate_stride1(const __fp16 *act, const __fp16 *wgt,
                                  const __fp16 *bias, __fp16 *out,
                                  uint32_t H, uint32_t W, uint32_t C);

void hmx_dwconvbbb(const uint8_t *act, const uint8_t *wgt,
                   const int32_t *bias, uint8_t *out,
                   uint32_t H, uint32_t W, uint32_t C);

/* ============================================================
 *  Element-wise Add (残差连接)
 *    out = a + b + bias; out = max(0, out)
 * ============================================================ */
void hmx_add(const __fp16 *a, const __fp16 *b,
             const __fp16 *bias, __fp16 *out,
             uint32_t M, uint32_t N);

/* ============================================================
 *  Phase 0 验证 kernel: 单 crouton fp16 GEMM (VTCM 指针入参)
 *  证明 hmx_common.c runtime 修好后真 HMX 指令能跑.
 * ============================================================ */
void hmx_phase0_gemm_fp16_core(const __fp16 *act_vtcm,
                               const __fp16 *wgt_vtcm,
                               const __fp16 *scales_vtcm,
                               __fp16       *out_vtcm);

/* Phase 1 诊断: 可配置 mode (0=deep/cvt, 1=nondeep/cvt, 2=recon, 3=PRM, 4=deep/after) */
void hmx_gemm_fp16_crouton_ex(const __fp16 *act_vtcm,
                               const __fp16 *wgt_vtcm,
                               const __fp16 *scales_vtcm,
                               __fp16       *out_vtcm,
                               unsigned mode);

/* Phase 1 验证: 单 crouton int8 GEMM (32×32×32) — VTCM 指针入参
 * int8 路径 (SDK 实有 intrinsic): activation.ub + weight.b + mxmem:after:sat.ub
 * 数学: out[m,n] = sat_u8( Σ_k act_u8[m,k]*wgt_i8[k,n] ) (Scale=1.0,bias=0).
 * bias: 256B region, per-channel 64-bit (Scale=0x3C00 rest=0 → cvt=acc). */
void hmx_phase1_gemm_int8_core(const uint8_t *act_vtcm,
                                const int8_t  *wgt_vtcm,
                                const void    *bias_vtcm,
                                uint8_t       *out_vtcm);

/* Phase 1 诊断: mode 0=dense ub/sat, 1=ub:cm/cm:sat, 2=dense ub/no-sat, 3=dense ub/0x7FF */
/* Phase 1 HMX fp16 伪 int8 GEMM: 复用 fp16 MAC 路径做 int8 GEMM (sat 后 exact).
 * act/wgt/scales/out 全 fp16 crouton (2KB 对齐 VTCM). scale=1.0 (0x3C00).
 * 数学: out = scale * Σ act*wgt  (fp16); 调用方 int8→fp16 + fp16→sat_u8. */
void hmx_pseudoint8_32x32x32_core(const __fp16 * __restrict__ act_fp16,
                                  const __fp16 * __restrict__ wgt_fp16,
                                  const __fp16 * __restrict__ scales,
                                  __fp16       * __restrict__ out_fp16);

void hmx_gemm_int8_crouton_ex(const uint8_t *act_vtcm,
                                const int8_t  *wgt_vtcm,
                                const void    *bias_vtcm,
                                uint8_t       *out_vtcm,
                                unsigned mode);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_HMX_KERNELS_H */
