/*
 * hmx_convhnh.c — INT8 GEMM (u8 × i16 → u16), 3 个变体
 * Module: hmx-gemm-int8
 * Math:   out[m,n] = sat_u16( bias[n] + Σ_k act_u8[m,k] * wgt_i16[k,n] )
 * Note:   i16 权重先 fit-check → i8 截断 (值域 [-128,127] 无损); 不 fit 退标量.
 *         走 HVX int8 GEMM, exact.
 */
#include "hmx_kernels.h"
#include "hmx_common.h"
#include "hvx_int8gemm.h"
#include <string.h>

static void convhnh_hvx(const uint8_t *act, const int16_t *wgt,
                        const int32_t *bias, uint16_t *out,
                        uint32_t M, uint32_t K, uint32_t N)
{
    if (M % HMX_TILE_DIM == 0 && K % HMX_TILE_DIM == 0 &&
        N % HMX_TILE_DIM == 0 && N <= 1024 && K * N <= 64 * 1024) {
        int fits = 1;
        for (uint32_t i = 0; i < K * N; i++) {
            int16_t v = wgt[i];
            if (v < -128 || v > 127) { fits = 0; break; }
        }
        if (fits) {
            static int8_t  wgt_i8[64 * 1024] __attribute__((aligned(128)));
            static int16_t bias16[1024]      __attribute__((aligned(128)));
            for (uint32_t i = 0; i < K * N; i++)
                wgt_i8[i] = (int8_t)wgt[i];
            for (uint32_t i = 0; i < N; i++)
                bias16[i] = (int16_t)bias[i];
            hvx_int8gemm_bias_u16_multi(act, wgt_i8, bias16, out, M, K, N);
            return;
        }
    }

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int32_t acc = bias[n];
            for (uint32_t k = 0; k < K; ++k)
                acc += (int32_t)act[m * K + k] * (int32_t)wgt[k * N + n];
            out[m * N + n] = acc < 0 ? 0 : (acc > 65535 ? 65535 : (uint16_t)acc);
        }
    }
}

void hmx_convhnh(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh1x1_stride1(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh_NxN_stride1(const uint8_t *act, const int16_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh1x1deep_stride1(const uint8_t *act, const int16_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh1x1_stride1_unaligned(const uint8_t *act, const int16_t *wgt,
                                      const int32_t *bias, uint16_t *out,
                                      uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh1xN_stride2(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnhNx1_stride2(const uint8_t *act, const int16_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh_5x5_stride1(const uint8_t *act, const int16_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh_stride1(const uint8_t *act, const int16_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh_stride1_aligned(const uint8_t *act, const int16_t *wgt,
                                 const int32_t *bias, uint16_t *out,
                                 uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh_stride2(const uint8_t *act, const int16_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhnh_dilate_stride1(const uint8_t *act, const int16_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N)
{
    convhnh_hvx(act, wgt, bias, out, M, K, N);
}
