/*
 * 13_compat_dlsym — v73/v75/v79 兼容层 dlsym + 功能验证
 * =====================================================================
 * 教学: dlopen libhvxhmx.so, dlsym 老版本符号 (hmx_v73_ 系列 / v75_ / v79_),
 *       调用并与标量 golden 对比. wrapper 内部尾调 v81 族函数, 行为由构造保证.
 *
 * 这也是 test_compat_sym.c 的简化教学版.
 */
#include "hvxhmx.h"
#include "example_util.h"
#include <dlfcn.h>

#define M 32
#define K 32
#define N 32

static uint8_t  act[M*K]  __attribute__((aligned(128)));
static uint8_t  wgt[K*N]  __attribute__((aligned(128)));
static int32_t  bias[N]   __attribute__((aligned(128)));
static uint8_t  out[M*N]  __attribute__((aligned(128)));
static uint8_t  gold[M*N] __attribute__((aligned(128)));

typedef void (*fn_u8u8_u8)(const uint8_t*, const uint8_t*, const int32_t*,
                           uint8_t*, uint32_t, uint32_t, uint32_t);

static void golden_u8u8_u8(void)
{
    for (uint32_t m=0;m<M;m++) for (uint32_t n=0;n<N;n++){
        int32_t a=bias[n];
        for (uint32_t k=0;k<K;k++) a += (int32_t)act[m*K+k]*(int32_t)wgt[k*N+n];
        gold[m*N+n] = a<0?0:(a>255?255:(uint8_t)a);
    }
}
static int cmp(void){int e=0;for(uint32_t i=0;i<M*N;i++){int d=abs((int)out[i]-(int)gold[i]);if(d>e)e=d;}return e;}

int main(void)
{
    ex_open_result("13_compat_dlsym");

    void *h = dlopen("libhvxhmx_v2.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { ex_log("FATAL: dlopen FAIL: %s", dlerror()); ex_check("dlopen libhvxhmx_v2.so", 1, 0); return ex_summary(); }
    ex_check("dlopen libhvxhmx_v2.so", 0, 0);

    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("FATAL: setup FAIL"); return ex_summary(); }

    ex_fill_u8 (act, M*K, 7, 8);
    ex_fill_u8 (wgt, K*N, 9, 8);
    ex_fill_i32(bias, N, 11, 200, 100);

    /* 全部用 convbbb (u8×u8→u8) 族符号, 签名一致. 其它族的精度见 compat_layer.md. */
    const char *syms[] = {
        "hmx_v73_convbbb1x1_stride1",
        "hmx_v73_convbbb_stride2",
        "hmx_v73_convbbb1x1deep_stride1",
        "hmx_v73_convbbb_dilate_stride1",
        "hmx_convbbb1x1_stride1",        /* v81 新增几何变体 */
        "hmx_convbbbNx1_stride2",
    };
    int nresolved = 0;
    for (int i = 0; i < (int)(sizeof(syms)/sizeof(syms[0])); i++) {
        fn_u8u8_u8 f = (fn_u8u8_u8)dlsym(h, syms[i]);
        if (!f) { ex_log("[MISS] %s", syms[i]); continue; }
        nresolved++;
        memset(out, 0, sizeof(out));
        f(act, wgt, bias, out, M, K, N);
        golden_u8u8_u8();
        ex_check(syms[i], cmp(), 0);
    }
    ex_log("resolved %d/%d symbols", nresolved, (int)(sizeof(syms)/sizeof(syms[0])));

    hmx_runtime_teardown();
    dlclose(h);
    return ex_summary();
}
