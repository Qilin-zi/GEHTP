/*
 * 02_convf16_gemm — fp16 GEMM (真 HMX) + 标量 golden 对比
 * =====================================================================
 * 教学: 用 hmx_convf16 做 32×32×32 fp16 矩阵乘, 与标量嵌套循环 golden 逐元素对比.
 *       容差 1 ULP (fp16 cvt.hf 截断).
 *
 * 数学: out[m,n] = bias[n] + Σ_k act[m,k]*wgt[k,n]
 */
#include "hvxhmx.h"
#include "example_util.h"

#define M 32
#define K 32
#define N 32

static __fp16 act [M*K] __attribute__((aligned(128)));
static __fp16 wgt [K*N] __attribute__((aligned(128)));
static __fp16 bias[N]   __attribute__((aligned(128)));
static __fp16 out [M*N] __attribute__((aligned(128)));
static __fp16 gold[M*N] __attribute__((aligned(128)));

static int run_one(uint32_t m, uint32_t k, uint32_t n, const char *label)
{
    ex_fill_f16(act,  m*k, 7,  0.01f);   /* act ∈ ~[-0.5,0.5] 避开 ±1.0 边界 */
    ex_fill_f16(wgt,  k*n, 9,  0.01f);
    ex_fill_f16(bias, n,   11, 0.01f);

    hmx_convf16(act, wgt, bias, out, m, k, n);

    /* 标量 golden */
    for (uint32_t mm = 0; mm < m; mm++)
        for (uint32_t nn = 0; nn < n; nn++) {
            float a = (float)bias[nn];
            for (uint32_t kk = 0; kk < k; kk++)
                a += (float)act[mm*k+kk] * (float)wgt[kk*n+nn];
            gold[mm*n+nn] = (__fp16)a;
        }

    /* 对比: raw fabsf(out-gold), tol=1 (匹配 test_all_hmx 口径, fp16 cvt.hf 截断) */
    int maxerr = 0;
    for (uint32_t i = 0; i < m*n; i++) {
        float d = (float)out[i] - (float)gold[i];
        if (d < 0) d = -d;
        int ud = (int)(d + 0.5f);
        if (ud > maxerr) maxerr = ud;
    }
    ex_check(label, maxerr, 1);   /* fp16 容差 1 */
    return maxerr;
}

int main(void)
{
    ex_open_result("02_convf16_gemm");

    if (hmx_runtime_setup(2*1024*1024) != 0) {
        ex_log("FATAL: runtime setup FAIL");
        ex_check("hmx_runtime_setup", 1, 0);
        return ex_summary();
    }

    run_one(M, K, N, "hmx_convf16 32x32x32");

    hmx_runtime_teardown();
    return ex_summary();
}
