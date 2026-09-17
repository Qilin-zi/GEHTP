/*
 * 03_convbbb_int8 — u8×u8→u8 GEMM (HVX int8 路径) + golden
 * =====================================================================
 * 教学: int8 族在本设备走 HVX (HMX int8 silent NOP). hmx_convbbb 做 u8×u8,
 *       int32 累加 + sat u8. exact (err=0).
 *
 * 数学: out[m,n] = sat_u8( bias[n] + Σ_k act[m,k]*wgt[k,n] )
 */
#include "hvxhmx.h"
#include "example_util.h"

#define M 32
#define K 32
#define N 32

static uint8_t act [M*K] __attribute__((aligned(128)));
static uint8_t wgt [K*N] __attribute__((aligned(128)));
static int32_t bias[N]  __attribute__((aligned(128)));
static uint8_t out [M*N] __attribute__((aligned(128)));
static uint8_t gold[M*N] __attribute__((aligned(128)));

int main(void)
{
    ex_open_result("03_convbbb_int8");

    if (hmx_runtime_setup(2*1024*1024) != 0) {
        ex_log("FATAL: runtime setup FAIL");
        return ex_summary();
    }

    ex_fill_u8 (act, M*K, 7, 8);       /* act ∈ [0,7] */
    ex_fill_u8 (wgt, K*N, 9, 8);       /* wgt ∈ [0,7] */
    ex_fill_i32(bias, N, 11, 200, 100);/* bias ∈ [-100,100] */

    hmx_convbbb(act, wgt, bias, out, M, K, N);

    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            int32_t a = bias[n];
            for (uint32_t k = 0; k < K; k++) a += (int32_t)act[m*K+k] * (int32_t)wgt[k*N+n];
            gold[m*N+n] = a < 0 ? 0 : (a > 255 ? 255 : (uint8_t)a);
        }

    int maxerr = 0;
    for (uint32_t i = 0; i < M*N; i++) {
        int d = (int)out[i] - (int)gold[i];
        if (d < 0) d = -d;
        if (d > maxerr) maxerr = d;
    }
    ex_check("hmx_convbbb 32x32x32 u8xu8->u8", maxerr, 0);  /* exact */

    hmx_runtime_teardown();
    return ex_summary();
}
