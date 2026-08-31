/*
 * hmx_convbbh.c — INT8 GEMM (u8 × u8 → u16)
 * Module: hmx-gemm-int8
 * Math:   out[m,n] = sat_u16( bias[n] + Σ_k act_u8[m,k] * wgt_u8[k,n] )
 * Note:   与 convbbb 同族, 仅输出饱和到 u16 (用于 concat/高 bit 输出层).
 *         走 HVX int8 GEMM, exact.
 */
#include "hmx_kernels.h"
#include "hmx_common.h"
#include "hvx_int8gemm.h"

#include <string.h>

void hmx_convbbh(const uint8_t * __restrict__ act,
                 const uint8_t * __restrict__ wgt,
                 const int32_t  * __restrict__ bias,
                 uint16_t       * __restrict__ out,
                 uint32_t M, uint32_t K, uint32_t N)
{
    if (M % HMX_TILE_DIM == 0 && K % HMX_TILE_DIM == 0 &&
        N % HMX_TILE_DIM == 0 && N <= 1024) {
        static int16_t bias16[1024] __attribute__((aligned(128)));
        for (uint32_t i = 0; i < N; ++i)
            bias16[i] = (int16_t)bias[i];
        hvx_int8gemm_bias_u16_multi(act, (const int8_t *)wgt, bias16, out, M, K, N);
        return;
    }

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int32_t acc = (int32_t) bias[n];
            for (uint32_t k = 0; k < K; ++k) {
                acc += (int32_t) act[m * K + k] *
                       (int32_t) wgt[k * N + n];
            }
            out[m * N + n] = HVHX_SAT_U16(acc);
        }
    }
}
