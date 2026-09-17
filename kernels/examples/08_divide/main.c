/*
 * 08_divide — HVX 整除 (u8/u16/i32 + floor) + golden
 * =====================================================================
 * 教学: 5 个 HVX 除法函数. 全 exact.
 *   - hvhx_divide_u8/u16: 倒数法
 *   - hvhx_divide_flat_i32: round-to-nearest (不是截断!)
 *   - hvhx_floor_divide_u8/u16: floor (无符号=截断)
 * 除零饱和: u8→0xFF, u16→0xFFFF, i32→±INT32_MAX.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define N 1024

static uint8_t  a_u8[N]  __attribute__((aligned(128)));
static uint8_t  b_u8[N]  __attribute__((aligned(128)));
static uint8_t  o_u8[N]  __attribute__((aligned(128)));
static uint8_t  g_u8[N]  __attribute__((aligned(128)));
static uint16_t a_u16[N] __attribute__((aligned(128)));
static uint16_t b_u16[N] __attribute__((aligned(128)));
static uint16_t o_u16[N] __attribute__((aligned(128)));
static uint16_t g_u16[N] __attribute__((aligned(128)));
static int32_t  a_i32[N] __attribute__((aligned(128)));
static int32_t  b_i32[N] __attribute__((aligned(128)));
static int32_t  o_i32[N] __attribute__((aligned(128)));
static int32_t  g_i32[N] __attribute__((aligned(128)));

static int cmp_u8(void){int e=0;for(int i=0;i<N;i++){int d=abs((int)o_u8[i]-(int)g_u8[i]);if(d>e)e=d;}return e;}
static int cmp_u16(void){int e=0;for(int i=0;i<N;i++){int d=abs((int)o_u16[i]-(int)g_u16[i]);if(d>e)e=d;}return e;}
static int cmp_i32(void){int e=0;for(int i=0;i<N;i++){int64_t d=(int64_t)o_i32[i]-(int64_t)g_i32[i];if(d<0)d=-d;if(d>e)e=(int)d;}return e;}

int main(void)
{
    ex_open_result("08_divide");
    /* HVX 族也先 setup (顺带 HVX 上电) */
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("WARN: setup FAIL (HVX 仍可跑)"); }

    /* u8: b ∈ [1,15] 避免大量除零; 少量 b=0 测饱和 */
    ex_fill_u8(a_u8, N, 7, 200);
    ex_fill_u8(b_u8, N, 9, 15);
    for (int i=0;i<8;i++) b_u8[i]=0;   /* 几个除零 */
    hvhx_divide_u8(a_u8, b_u8, o_u8, N);
    for (int i=0;i<N;i++) g_u8[i] = (b_u8[i]==0)?0xFF:(uint8_t)(a_u8[i]/b_u8[i]);
    ex_check("hvhx_divide_u8", cmp_u8(), 0);

    hvhx_floor_divide_u8(a_u8, b_u8, o_u8, N);
    for (int i=0;i<N;i++) g_u8[i] = (b_u8[i]==0)?0xFF:(uint8_t)(a_u8[i]/b_u8[i]);
    ex_check("hvhx_floor_divide_u8", cmp_u8(), 0);

    /* u16 */
    ex_fill_u16(a_u16, N, 21, 0);
    ex_fill_u16(b_u16, N, 23, 1000);
    for (int i=0;i<8;i++) b_u16[i]=0;
    hvhx_divide_u16(a_u16, b_u16, o_u16, N);
    for (int i=0;i<N;i++) g_u16[i] = (b_u16[i]==0)?0xFFFF:(uint16_t)(a_u16[i]/b_u16[i]);
    ex_check("hvhx_divide_u16", cmp_u16(), 0);

    hvhx_floor_divide_u16(a_u16, b_u16, o_u16, N);
    for (int i=0;i<N;i++) g_u16[i] = (b_u16[i]==0)?0xFFFF:(uint16_t)(a_u16[i]/b_u16[i]);
    ex_check("hvhx_floor_divide_u16", cmp_u16(), 0);

    /* i32 round-to-nearest */
    ex_fill_i32(a_i32, N, 31, 4000, 2000);
    ex_fill_i32(b_i32, N, 33, 500, 0);
    for (int i=0;i<N;i++) if(b_i32[i]==0) b_i32[i]=1;
    for (int i=0;i<8;i++) b_i32[i]=0;
    hvhx_divide_flat_i32(a_i32, b_i32, o_i32, N);
    for (int i=0;i<N;i++){
        if (b_i32[i]==0) g_i32[i] = (a_i32[i]>=0)?INT32_MAX:INT32_MIN;
        else {
            /* round-to-nearest (banker 不考虑, 简单四舍五入) */
            int64_t a=(int64_t)a_i32[i]*2 + (b_i32[i]>0?1:-1)*(b_i32[i]>0?b_i32[i]:-b_i32[i]);
            int64_t q = a / (2*(int64_t)b_i32[i]);
            g_i32[i] = (int32_t)q;
        }
    }
    ex_check("hvhx_divide_flat_i32 (round)", cmp_i32(), 1);

    hmx_runtime_teardown();
    return ex_summary();
}
