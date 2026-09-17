/*
 * 07_add — elementwise add (残差) fp16 + golden
 * =====================================================================
 * 教学: hmx_add: out = max(0, a + b + bias) (含 ReLU). fp16, 容差 1 ULP.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define M 32
#define N 32

static __fp16 a[M*N]   __attribute__((aligned(128)));
static __fp16 b[M*N]   __attribute__((aligned(128)));
static __fp16 bias[N]  __attribute__((aligned(128)));
static __fp16 out[M*N] __attribute__((aligned(128)));
static __fp16 gold[M*N] __attribute__((aligned(128)));

int main(void)
{
    ex_open_result("07_add");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("FATAL: setup FAIL"); return ex_summary(); }

    ex_fill_f16(a,    M*N, 7, 0.01f);
    ex_fill_f16(b,    M*N, 9, 0.01f);
    ex_fill_f16(bias, N,   11, 0.01f);

    hmx_add(a, b, bias, out, M, N);

    for (uint32_t m=0;m<M;m++) for (uint32_t n=0;n<N;n++){
        float v = (float)a[m*N+n] + (float)b[m*N+n] + (float)bias[n];
        if (v < 0) v = 0;
        gold[m*N+n] = (__fp16)v;
    }
    int e=0;
    for (uint32_t i=0;i<M*N;i++){float d=(float)out[i]-(float)gold[i]; if(d<0)d=-d; int ud=(int)(d*1024+0.5); if(ud>e)e=ud;}
    ex_check("hmx_add fp16 (relu)", e, 1);

    hmx_runtime_teardown();
    return ex_summary();
}
