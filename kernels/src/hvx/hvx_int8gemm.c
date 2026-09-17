/*
 * hvx_int8gemm.c — HVX int8 GEMM core (u8 × i8 → u8), 纯 HVX
 * Module: hvx-lowlevel
 * Math:   out[m,n] = sat_u8( Σ_k act_u8[m,k] * wgt_i8[k,n] )
 * Note:   wgt 须 [32][128] padded 布局 (stride=128B). int16 acc, K=32 时
 *         要求 max|Σ act*wgt| < 32768. 本文件必须 -O1 编译 (软件流水线器
 *         stale-vector bug, 见 build_libs.sh). 详见 docs/api_lowlevel.md.
 */
#include <string.h>
#include "hvx_int8gemm.h"
#include "hvxhmx_types.h"

#if defined(__hexagon__) || defined(__HVX__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#define DIM 32
#define WGT_STRIDE 128  /* HVX_Vector 宽度, 每行 pad 到 128B */

/* dense [32][32] wgt → padded [32][128] wgt (32B data + 96B zero per row) */
void hvx_int8gemm_pack_wgt(const int8_t * __restrict__ wgt_dense,
                           int8_t        * __restrict__ wgt_padded)
{
    memset(wgt_padded, 0, DIM * WGT_STRIDE);
    for (int k = 0; k < DIM; k++)
        memcpy(wgt_padded + k * WGT_STRIDE, wgt_dense + k * DIM, DIM);
}

void hvx_int8gemm_32x32x32_core(
    const uint8_t * __restrict__ act_v,   /* [32][32] row-major, 1KB        */
    const int8_t  * __restrict__ wgt_v,   /* [32][128] padded, 4KB, 128B 对齐 */
    uint8_t       * __restrict__ out_v)   /* [32][32] row-major, 1KB        */
{
#if defined(__hexagon__) || defined(__HVX__)
    for (int m = 0; m < DIM; m++) {
        const HVX_Vector *wvp = (const HVX_Vector *)wgt_v;
        /* 显式 prologue (k=0): load wgt[0], 首次 MAC. 避免编译器缺 prologue 的 stale-vector. */
        HVX_Vector va0 = Q6_Vb_vsplat_R((signed char)act_v[m * DIM]);
        HVX_Vector vw  = *wvp++;
        HVX_VectorPair acc = Q6_Wh_vmpy_VubVb(va0, vw);
        for (int k = 1; k < DIM; k++) {
            HVX_Vector va = Q6_Vb_vsplat_R((signed char)act_v[m * DIM + k]);
            vw = *wvp++;
            acc = Q6_Wh_vmpyacc_WhVubVb(acc, va, vw);
        }
        HVX_Vector outvec = Q6_Vub_vasr_VhVhR_sat(
            Q6_V_hi_W(acc), Q6_V_lo_W(acc), 0);
        union { HVX_Vector v; uint8_t b[128]; } u;
        u.v = outvec;
        memcpy(out_v + m * DIM, u.b, DIM);
    }
#else
    for (int m = 0; m < DIM; m++)
        for (int n = 0; n < DIM; n++) {
            int32_t s = 0;
            for (int k = 0; k < DIM; k++)
                s += (int32_t)act_v[m*DIM+k] * (int32_t)wgt_v[k*WGT_STRIDE+n];
            out_v[m*DIM+n] = HVHX_SAT_U8(s);
        }
#endif
}

/* ============================================================
 *  bias-aware 32×32×32 GEMM core
 *  out[m][n] = sat_u8( bias[n] + Σ_k act_u8[m][k] * wgt_i8[k][n] )
 *
 *  bias 以 int16 加到累加器 lane (vasr 之前), lane 映射:
 *    acc pair lo halfword j = product for wgt byte 2j (偶 column)
 *    acc pair hi halfword j = product for wgt byte 2j+1 (奇 column)
 *  故 bias_lo[j]=bias[2j], bias_hi[j]=bias[2j+1] (j=0..15, 其余 lane 填 0).
 * ============================================================ */
void hvx_int8gemm_bias_32x32x32_core(
    const uint8_t * __restrict__ act_v,
    const int8_t  * __restrict__ wgt_v,
    const int16_t * __restrict__ bias_v,    /* [32] */
    uint8_t       * __restrict__ out_v)
{
#if defined(__hexagon__) || defined(__HVX__)
    static int16_t bias_lo_arr[64] __attribute__((aligned(128)));
    static int16_t bias_hi_arr[64] __attribute__((aligned(128)));
    for (int j = 0; j < 64; j++) { bias_lo_arr[j] = 0; bias_hi_arr[j] = 0; }
    for (int j = 0; j < 16; j++) {
        bias_lo_arr[j] = bias_v[2 * j];
        bias_hi_arr[j] = bias_v[2 * j + 1];
    }
    HVX_Vector bias_lo = *(const HVX_Vector *)bias_lo_arr;
    HVX_Vector bias_hi = *(const HVX_Vector *)bias_hi_arr;

    for (int m = 0; m < DIM; m++) {
        const HVX_Vector *wvp = (const HVX_Vector *)wgt_v;
        HVX_Vector va0 = Q6_Vb_vsplat_R((signed char)act_v[m * DIM]);
        HVX_Vector vw  = *wvp++;
        HVX_VectorPair acc = Q6_Wh_vmpy_VubVb(va0, vw);
        for (int k = 1; k < DIM; k++) {
            HVX_Vector va = Q6_Vb_vsplat_R((signed char)act_v[m * DIM + k]);
            vw = *wvp++;
            acc = Q6_Wh_vmpyacc_WhVubVb(acc, va, vw);
        }
        HVX_Vector lo = Q6_V_lo_W(acc);
        HVX_Vector hi = Q6_V_hi_W(acc);
        lo = Q6_Vh_vadd_VhVh_sat(lo, bias_lo);
        hi = Q6_Vh_vadd_VhVh_sat(hi, bias_hi);
        HVX_Vector outvec = Q6_Vub_vasr_VhVhR_sat(hi, lo, 0);
        union { HVX_Vector v; uint8_t b[128]; } u;
        u.v = outvec;
        memcpy(out_v + m * DIM, u.b, DIM);
    }
#else
    for (int m = 0; m < DIM; m++)
        for (int n = 0; n < DIM; n++) {
            int32_t s = bias_v[n];
            for (int k = 0; k < DIM; k++)
                s += (int32_t)act_v[m*DIM+k] * (int32_t)wgt_v[k*WGT_STRIDE+n];
            out_v[m*DIM+n] = HVHX_SAT_U8(s);
        }
#endif
}

/* ============================================================
 *  bias-aware 32×32×32 GEMM core, uint16 写回 (convbbh 用)
 *  同 MAC 循环; bias 加到 acc lane 后, 负值钳 0, 直接按 uint16 写出.
 *  输出 lane 映射: out[2j]=lo[j], out[2j+1]=hi[j] (j=0..15).
 * ============================================================ */
void hvx_int8gemm_bias_u16_32x32x32_core(
    const uint8_t  * __restrict__ act_v,
    const int8_t   * __restrict__ wgt_v,
    const int16_t  * __restrict__ bias_v,
    uint16_t       * __restrict__ out_v)
{
#if defined(__hexagon__) || defined(__HVX__)
    static int16_t bias_lo_arr[64] __attribute__((aligned(128)));
    static int16_t bias_hi_arr[64] __attribute__((aligned(128)));
    for (int j = 0; j < 64; j++) { bias_lo_arr[j] = 0; bias_hi_arr[j] = 0; }
    for (int j = 0; j < 16; j++) {
        bias_lo_arr[j] = bias_v[2 * j];
        bias_hi_arr[j] = bias_v[2 * j + 1];
    }
    HVX_Vector bias_lo = *(const HVX_Vector *)bias_lo_arr;
    HVX_Vector bias_hi = *(const HVX_Vector *)bias_hi_arr;
    HVX_Vector vzero = Q6_V_vzero();

    for (int m = 0; m < DIM; m++) {
        const HVX_Vector *wvp = (const HVX_Vector *)wgt_v;
        HVX_Vector va0 = Q6_Vb_vsplat_R((signed char)act_v[m * DIM]);
        HVX_Vector vw  = *wvp++;
        HVX_VectorPair acc = Q6_Wh_vmpy_VubVb(va0, vw);
        for (int k = 1; k < DIM; k++) {
            HVX_Vector va = Q6_Vb_vsplat_R((signed char)act_v[m * DIM + k]);
            vw = *wvp++;
            acc = Q6_Wh_vmpyacc_WhVubVb(acc, va, vw);
        }
        HVX_Vector lo = Q6_V_lo_W(acc);
        HVX_Vector hi = Q6_V_hi_W(acc);
        lo = Q6_Vh_vadd_VhVh_sat(lo, bias_lo);
        hi = Q6_Vh_vadd_VhVh_sat(hi, bias_hi);
        lo = Q6_Vh_vmax_VhVh(lo, vzero);
        hi = Q6_Vh_vmax_VhVh(hi, vzero);
        union { HVX_Vector v; int16_t h[64]; } ul, uh;
        ul.v = lo; uh.v = hi;
        uint16_t *row = out_v + m * DIM;
        for (int j = 0; j < 16; j++) {
            row[2 * j]     = (uint16_t)ul.h[j];
            row[2 * j + 1] = (uint16_t)uh.h[j];
        }
    }
#else
    for (int m = 0; m < DIM; m++)
        for (int n = 0; n < DIM; n++) {
            int32_t s = bias_v[n];
            for (int k = 0; k < DIM; k++)
                s += (int32_t)act_v[m*DIM+k] * (int32_t)wgt_v[k*WGT_STRIDE+n];
            out_v[m*DIM+n] = HVHX_SAT_U16(s);
        }
#endif
}

/* ============================================================
 *  对外 API: host dense [32][32] 指针入参, 内部 pack → padded → core
 *  M=K=N=32 固定 (单 tile). 不满足边界退标量.
 * ============================================================ */
void hvx_int8gemm(const uint8_t *act, const int8_t *wgt,
                  uint8_t *out, uint32_t M, uint32_t K, uint32_t N)
{
    if (M != DIM || N != DIM || K != DIM) {
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t n = 0; n < N; n++) {
                int32_t s = 0;
                for (uint32_t k = 0; k < K; k++)
                    s += (int32_t)act[m*K+k] * (int32_t)wgt[k*N+n];
                out[m*N+n] = HVHX_SAT_U8(s);
            }
        return;
    }
    static int8_t wgt_padded[DIM * WGT_STRIDE] __attribute__((aligned(128)));
    hvx_int8gemm_pack_wgt(wgt, wgt_padded);
    hvx_int8gemm_32x32x32_core(act, wgt_padded, out);
}

#define HVX_I8G_MAX_K 256

/* 多 tile M/N/K int8 GEMM (uint8 写回).
 * 每个 nt tile: 打包 wgt[0..K-1][nt*32..nt*32+31] → [K][128] padded (一次),
 *   再对全部 M 行做 K 维 MAC 累加 (int16 acc), bias+vasr 写 32 列. */
void hvx_int8gemm_bias_multi(const uint8_t * __restrict__ act,
                             const int8_t  * __restrict__ wgt,
                             const int16_t * __restrict__ bias,
                             uint8_t       * __restrict__ out,
                             uint32_t M, uint32_t K, uint32_t N)
{
#if defined(__hexagon__) || defined(__HVX__)
    if (M % DIM == 0 && K % DIM == 0 && N % DIM == 0 && K <= HVX_I8G_MAX_K) {
        static int8_t  wgt_padded[HVX_I8G_MAX_K * WGT_STRIDE] __attribute__((aligned(128)));
        static int16_t bias_lo_arr[64] __attribute__((aligned(128)));
        static int16_t bias_hi_arr[64] __attribute__((aligned(128)));

        const uint32_t n_nt = N / DIM;

        for (uint32_t nt = 0; nt < n_nt; ++nt) {
            memset(wgt_padded, 0, K * WGT_STRIDE);
            for (uint32_t k = 0; k < K; ++k)
                memcpy(wgt_padded + k * WGT_STRIDE,
                       &wgt[k * N + nt * DIM], DIM);

            for (int j = 0; j < 64; j++) { bias_lo_arr[j] = 0; bias_hi_arr[j] = 0; }
            for (int j = 0; j < 16; j++) {
                bias_lo_arr[j] = bias[nt * DIM + 2 * j];
                bias_hi_arr[j] = bias[nt * DIM + 2 * j + 1];
            }
            HVX_Vector bias_lo = *(const HVX_Vector *)bias_lo_arr;
            HVX_Vector bias_hi = *(const HVX_Vector *)bias_hi_arr;

            for (uint32_t m = 0; m < M; ++m) {
                const HVX_Vector *wvp = (const HVX_Vector *)wgt_padded;
                HVX_Vector va0 = Q6_Vb_vsplat_R((signed char)act[m * K]);
                HVX_Vector vw  = *wvp++;
                HVX_VectorPair acc = Q6_Wh_vmpy_VubVb(va0, vw);
                for (uint32_t k = 1; k < K; ++k) {
                    HVX_Vector va = Q6_Vb_vsplat_R((signed char)act[m * K + k]);
                    vw = *wvp++;
                    acc = Q6_Wh_vmpyacc_WhVubVb(acc, va, vw);
                }
                HVX_Vector lo = Q6_V_lo_W(acc);
                HVX_Vector hi = Q6_V_hi_W(acc);
                lo = Q6_Vh_vadd_VhVh_sat(lo, bias_lo);
                hi = Q6_Vh_vadd_VhVh_sat(hi, bias_hi);
                HVX_Vector outvec = Q6_Vub_vasr_VhVhR_sat(hi, lo, 0);
                union { HVX_Vector v; uint8_t b[128]; } u;
                u.v = outvec;
                memcpy(out + m * N + nt * DIM, u.b, DIM);
            }
        }
        return;
    }
#endif
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            int32_t s = bias[n];
            for (uint32_t k = 0; k < K; ++k)
                s += (int32_t)act[m * K + k] * (int32_t)wgt[k * N + n];
            out[m * N + n] = HVHX_SAT_U8(s);
        }
}

/* 多 tile M/N/K int8 GEMM (uint16 写回). 同上结构, 负值钳 0 后按 u16 写. */
void hvx_int8gemm_bias_u16_multi(const uint8_t  * __restrict__ act,
                                 const int8_t   * __restrict__ wgt,
                                 const int16_t  * __restrict__ bias,
                                 uint16_t       * __restrict__ out,
                                 uint32_t M, uint32_t K, uint32_t N)
{
#if defined(__hexagon__) || defined(__HVX__)
    if (M % DIM == 0 && K % DIM == 0 && N % DIM == 0 && K <= HVX_I8G_MAX_K) {
        static int8_t  wgt_padded[HVX_I8G_MAX_K * WGT_STRIDE] __attribute__((aligned(128)));
        static int16_t bias_lo_arr[64] __attribute__((aligned(128)));
        static int16_t bias_hi_arr[64] __attribute__((aligned(128)));
        HVX_Vector vzero = Q6_V_vzero();

        const uint32_t n_nt = N / DIM;

        for (uint32_t nt = 0; nt < n_nt; ++nt) {
            memset(wgt_padded, 0, K * WGT_STRIDE);
            for (uint32_t k = 0; k < K; ++k)
                memcpy(wgt_padded + k * WGT_STRIDE,
                       &wgt[k * N + nt * DIM], DIM);

            for (int j = 0; j < 64; j++) { bias_lo_arr[j] = 0; bias_hi_arr[j] = 0; }
            for (int j = 0; j < 16; j++) {
                bias_lo_arr[j] = bias[nt * DIM + 2 * j];
                bias_hi_arr[j] = bias[nt * DIM + 2 * j + 1];
            }
            HVX_Vector bias_lo = *(const HVX_Vector *)bias_lo_arr;
            HVX_Vector bias_hi = *(const HVX_Vector *)bias_hi_arr;

            for (uint32_t m = 0; m < M; ++m) {
                const HVX_Vector *wvp = (const HVX_Vector *)wgt_padded;
                HVX_Vector va0 = Q6_Vb_vsplat_R((signed char)act[m * K]);
                HVX_Vector vw  = *wvp++;
                HVX_VectorPair acc = Q6_Wh_vmpy_VubVb(va0, vw);
                for (uint32_t k = 1; k < K; ++k) {
                    HVX_Vector va = Q6_Vb_vsplat_R((signed char)act[m * K + k]);
                    vw = *wvp++;
                    acc = Q6_Wh_vmpyacc_WhVubVb(acc, va, vw);
                }
                HVX_Vector lo = Q6_V_lo_W(acc);
                HVX_Vector hi = Q6_V_hi_W(acc);
                lo = Q6_Vh_vadd_VhVh_sat(lo, bias_lo);
                hi = Q6_Vh_vadd_VhVh_sat(hi, bias_hi);
                lo = Q6_Vh_vmax_VhVh(lo, vzero);
                hi = Q6_Vh_vmax_VhVh(hi, vzero);
                union { HVX_Vector v; int16_t h[64]; } ul, uh;
                ul.v = lo; uh.v = hi;
                uint16_t *row = out + m * N + nt * DIM;
                for (int j = 0; j < 16; j++) {
                    row[2 * j]     = (uint16_t)ul.h[j];
                    row[2 * j + 1] = (uint16_t)uh.h[j];
                }
            }
        }
        return;
    }
#endif
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            int32_t s = bias[n];
            for (uint32_t k = 0; k < K; ++k)
                s += (int32_t)act[m * K + k] * (int32_t)wgt[k * N + n];
            out[m * N + n] = HVHX_SAT_U16(s);
        }
}

/* 诊断版 HVX core: dump acc pair raw int16 (padded wgt 入参) */
void hvx_int8gemm_diag(const uint8_t *act_v, const int8_t *wgt_v_padded, int16_t *dump)
{
#if defined(__hexagon__) || defined(__HVX__)
    int m = 0;
    HVX_VectorPair acc = Q6_W_vzero();
    const HVX_Vector *wvp = (const HVX_Vector *)wgt_v_padded;
    for (int k = 0; k < 32; k++) {
        uint8_t a = act_v[m*32+k];
        HVX_Vector va = Q6_Vb_vsplat_R((signed char)a);
        HVX_Vector vw = *wvp++;
        acc = Q6_Wh_vmpyacc_WhVubVb(acc, va, vw);
    }
    HVX_Vector lo = Q6_V_lo_W(acc);
    HVX_Vector hi = Q6_V_hi_W(acc);
    *(HVX_Vector *)(dump)     = lo;
    *(HVX_Vector *)(dump+64)  = hi;
#endif
}
