/*
 * 12_multitile_gemm — 大尺寸 fp16 GEMM (M/N/K > 32 多 tile) + golden
 * =====================================================================
 * 教学: hmx_convf16 支持 M/N/K > 32. 库内部循环 32×32 tile; K 维用 activation+weight
 *       pair 累加. 容差 1 ULP. 测 4 种维度组合, 都 K 对齐 32.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define MAXDIM 64
static __fp16 act [MAXDIM*MAXDIM] __attribute__((aligned(128)));
static __fp16 wgt [MAXDIM*MAXDIM] __attribute__((aligned(128)));
static __fp16 bias[MAXDIM]        __attribute__((aligned(128)));
static __fp16 out [MAXDIM*MAXDIM] __attribute__((aligned(128)));
static __fp16 gold[MAXDIM*MAXDIM] __attribute__((aligned(128)));

static int run(uint32_t M, uint32_t K, uint32_t N, const char *label)
{
    ex_fill_f16(act,  M*K, 7,  0.01f);
    ex_fill_f16(wgt,  K*N, 9,  0.01f);
    ex_fill_f16(bias, N,   11, 0.01f);
    hmx_convf16(act, wgt, bias, out, M, K, N);
    for (uint32_t m=0;m<M;m++) for (uint32_t n=0;n<N;n++){
        float a=(float)bias[n];
        for (uint32_t k=0;k<K;k++) a += (float)act[m*K+k]*(float)wgt[k*N+n];
        gold[m*N+n]=(__fp16)a;
    }
    /* 容差口径与 test_all_hmx 一致: raw fabsf(out-gold) ≤ 1 (fp16 systolic 截断) */
    int e=0;
    for (uint32_t i=0;i<M*N;i++){float d=(float)out[i]-(float)gold[i]; if(d<0)d=-d; int ud=(int)(d+0.5f); if(ud>e)e=ud;}
    ex_check(label, e, 1);
    return e;
}

int main(void)
{
    ex_open_result("12_multitile_gemm");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("FATAL: setup FAIL"); return ex_summary(); }

    run(64, 32, 32, "hmx_convf16 64x32x32 (M>32)");
    run(32, 32, 64, "hmx_convf16 32x32x64 (N>32)");
    run(32, 64, 32, "hmx_convf16 32x64x32 (K>32)");
    run(64, 64, 64, "hmx_convf16 64x64x64 (all>32)");

    hmx_runtime_teardown();
    return ex_summary();
}
