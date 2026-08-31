/*
 * 05_i16_weight_convs — i16 权重族 (convbcb/bnb/hch/hnh) + golden
 * =====================================================================
 * 教学: 4 个 i16 权重族都是 u8 act × i16 wgt, 区别在输出精度/格式:
 *   - convbcb : → u8  (sat)
 *   - convbnb : → u8  (数学同 bcb; 基本型无 sparsity)
 *   - convhch : → u16 (:2x2 格式)
 *   - convhnh : → u16 (:2x1 格式, 数学同 hch)
 * 全部 exact. HVX int8 GEMM 内部把 i16 拆半处理.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define M 32
#define K 32
#define N 32

static uint8_t  act  [M*K] __attribute__((aligned(128)));
static int16_t  wgt  [K*N] __attribute__((aligned(128)));
static int32_t  bias [N]  __attribute__((aligned(128)));
static uint8_t  out_u8 [M*N] __attribute__((aligned(128)));
static uint16_t out_u16[M*N] __attribute__((aligned(128)));
static uint8_t  gold_u8 [M*N] __attribute__((aligned(128)));
static uint16_t gold_u16[M*N] __attribute__((aligned(128)));

static void golden_u8(void)
{
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            int32_t a = bias[n];
            for (uint32_t k = 0; k < K; k++) a += (int32_t)act[m*K+k] * (int32_t)wgt[k*N+n];
            gold_u8[m*N+n] = a<0?0:(a>255?255:(uint8_t)a);
        }
}
static void golden_u16(void)
{
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            int32_t a = bias[n];
            for (uint32_t k = 0; k < K; k++) a += (int32_t)act[m*K+k] * (int32_t)wgt[k*N+n];
            gold_u16[m*N+n] = a<0?0:(a>65535?65535:(uint16_t)a);
        }
}
static int cmp_u8(void){ int e=0; for(uint32_t i=0;i<M*N;i++){int d=abs((int)out_u8[i]-(int)gold_u8[i]); if(d>e)e=d;} return e; }
static int cmp_u16(void){ int e=0; for(uint32_t i=0;i<M*N;i++){int d=abs((int)out_u16[i]-(int)gold_u16[i]); if(d>e)e=d;} return e; }

int main(void)
{
    ex_open_result("05_i16_weight_convs");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("FATAL: setup FAIL"); return ex_summary(); }

    ex_fill_u8 (act,  M*K, 7, 4);
    ex_fill_i16(wgt,  K*N, 9, 6);
    ex_fill_i32(bias, N,   11, 200, 100);

    hmx_convbcb(act, wgt, bias, out_u8, M, K, N);
    golden_u8(); ex_check("hmx_convbcb  u8xi16->u8",  cmp_u8(),  0);

    hmx_convbnb(act, wgt, bias, out_u8, M, K, N);
    golden_u8(); ex_check("hmx_convbnb  u8xi16->u8",  cmp_u8(),  0);

    hmx_convhch(act, wgt, bias, out_u16, M, K, N);
    golden_u16(); ex_check("hmx_convhch  u8xi16->u16", cmp_u16(), 0);

    hmx_convhnh(act, wgt, bias, out_u16, M, K, N);
    golden_u16(); ex_check("hmx_convhnh  u8xi16->u16", cmp_u16(), 0);

    hmx_runtime_teardown();
    return ex_summary();
}
