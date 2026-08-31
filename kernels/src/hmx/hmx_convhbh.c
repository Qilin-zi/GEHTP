/*
 * hmx_convhbh.c — INT8 GEMM (u8 × i8 → u16), 3 个变体
 * Module: hmx-gemm-int8
 * Math:   out[m,n] = sat_u16( bias[n] + Σ_k act_u8[m,k] * wgt_i8[k,n] )
 * Note:   走 HVX int8 GEMM (本设备 int8 HMX + .uh 写回均坏, 见 docs/data_layout.md),
 *         per-element exact. 几何/stride 差异在 caller 层.
 */
#include "hmx_kernels.h"
#include "hmx_common.h"
#include "hvx_int8gemm.h"
#include <string.h>

static void convhbh_hvx(const uint8_t *act, const int8_t *wgt,
                        const int32_t *bias, uint16_t *out,
                        uint32_t M, uint32_t K, uint32_t N)
{
    if (M % HMX_TILE_DIM == 0 && K % HMX_TILE_DIM == 0 &&
        N % HMX_TILE_DIM == 0 && N <= 1024 && K <= 256) {
        static int16_t bias16[1024] __attribute__((aligned(128)));
        for (uint32_t i = 0; i < N; i++)
            bias16[i] = (int16_t)bias[i];
        hvx_int8gemm_bias_u16_multi(act, wgt, bias16, out, M, K, N);
        return;
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

void hmx_convhbh(const uint8_t *act, const int8_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh1x1_stride1(const uint8_t *act, const int8_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh1x1deep_stride1(const uint8_t *act, const int8_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh1x1_stride1_unaligned(const uint8_t *act, const int8_t *wgt,
                                      const int32_t *bias, uint16_t *out,
                                      uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh1xN_stride2(const uint8_t *act, const int8_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbhNx1_stride2(const uint8_t *act, const int8_t *wgt,
                            const int32_t *bias, uint16_t *out,
                            uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh_NxN_stride1(const uint8_t *act, const int8_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh_5x5_stride1(const uint8_t *act, const int8_t *wgt,
                             const int32_t *bias, uint16_t *out,
                             uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh_stride1(const uint8_t *act, const int8_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh_stride1_aligned(const uint8_t *act, const int8_t *wgt,
                                 const int32_t *bias, uint16_t *out,
                                 uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh_stride2(const uint8_t *act, const int8_t *wgt,
                         const int32_t *bias, uint16_t *out,
                         uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}

void hmx_convhbh_dilate_stride1(const uint8_t *act, const int8_t *wgt,
                                const int32_t *bias, uint16_t *out,
                                uint32_t M, uint32_t K, uint32_t N)
{
    convhbh_hvx(act, wgt, bias, out, M, K, N);
}
