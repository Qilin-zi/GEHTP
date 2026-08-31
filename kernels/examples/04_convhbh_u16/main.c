/*
 * 04_convhbh_u16 — u8×i8→u16 GEMM (HVX int8) + golden
 * =====================================================================
 * 教学: hmx_convhbh: u8 act × i8 wgt → u16 (宽动态, 不 sat). HVX int8 GEMM.
 *       也示范 hmx_convhhh (同数学, :2x2 格式). exact.
 *
 * 数学: out[m,n] = sat_u16( bias[n] + Σ_k act[m,k]*wgt_i8[k,n] )
 */
#include "hvxhmx.h"
#include "example_util.h"

#define M 32
#define K 32
#define N 32

static uint8_t  act [M*K] __attribute__((aligned(128)));
static int8_t   wgt [K*N] __attribute__((aligned(128)));
static int32_t  bias[N]  __attribute__((aligned(128)));
static uint16_t out [M*N] __attribute__((aligned(128)));
static uint16_t gold[M*N] __attribute__((aligned(128)));

static int run(const char *label, int which)
{
    ex_fill_u8(act, M*K, 7, 4);
    ex_fill_i8(wgt, K*N, 9, 6);        /* i8 wgt ∈ ~[-3,2] */
    ex_fill_i32(bias, N, 11, 200, 100);

    if (which == 0) hmx_convhbh(act, wgt, bias, out, M, K, N);
    else            hmx_convhhh(act, wgt, bias, out, M, K, N);

    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            int32_t a = bias[n];
            for (uint32_t k = 0; k < K; k++) a += (int32_t)act[m*K+k] * (int32_t)wgt[k*N+n];
            gold[m*N+n] = a < 0 ? 0 : (a > 65535 ? 65535 : (uint16_t)a);
        }
    int maxerr = 0;
    for (uint32_t i = 0; i < M*N; i++) {
        int d = (int)out[i] - (int)gold[i];
        if (d < 0) d = -d;
        if (d > maxerr) maxerr = d;
    }
    ex_check(label, maxerr, 0);
    return maxerr;
}

int main(void)
{
    ex_open_result("04_convhbh_u16");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("FATAL: setup FAIL"); return ex_summary(); }

    run("hmx_convhbh u8xi8->u16", 0);
    run("hmx_convhhh u8xi8->u16", 1);

    hmx_runtime_teardown();
    return ex_summary();
}
