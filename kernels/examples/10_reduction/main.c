/*
 * 10_reduction — HVX 归约 (argminmax / find_max / top1 / reducesum) + golden
 * =====================================================================
 * 教学: 沿 depth 维归约. 输入 flat [hw][d], 输出每行一个结果. 全 exact.
 *
 * 注意: 函数名含 "crouton" 指内部处理方式, 输入仍是 flat [hw][d] 行主序.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define HW 8
#define D  32

static uint8_t  in_u8 [HW*D] __attribute__((aligned(128)));
static uint16_t in_u16[HW*D] __attribute__((aligned(128)));
static hvhx_argminmax_t am[HW] __attribute__((aligned(128)));
static hvhx_top1_t      t1[HW] __attribute__((aligned(128)));
static uint32_t sumbuf[HW] __attribute__((aligned(128)));
static uint8_t  mv_b[HW]  __attribute__((aligned(128)));
static uint16_t mv_h[HW]  __attribute__((aligned(128)));
static uint32_t idx[HW]   __attribute__((aligned(128)));

int main(void)
{
    ex_open_result("10_reduction");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("WARN: setup FAIL"); }

    /* argminmax u8 crouton: per-row min/max + idx */
    ex_fill_u8(in_u8, HW*D, 42, 0);
    hvhx_argminmax_depth_crouton_b(in_u8, HW, D, am);
    int e=0;
    for (int hw=0;hw<HW;hw++){
        int mn=256,mxv=-1; uint32_t mi=0,mai=0;
        for (int d=0;d<D;d++){int v=in_u8[hw*D+d]; if(v<mn){mn=v;mi=d;} if(v>mxv){mxv=v;mai=d;}}
        int t; t=abs((int)am[hw].min_val-mn); if(t>e)e=t;
        t=abs((int)am[hw].max_val-mxv); if(t>e)e=t;
        t=abs((int)am[hw].min_idx-(int)mi); if(t>e)e=t;
        t=abs((int)am[hw].max_idx-(int)mai); if(t>e)e=t;
    }
    ex_check("hvhx_argminmax_depth_crouton_b", e, 0);

    /* argminmax u16 flat */
    ex_fill_u16(in_u16, HW*D, 77, 0);
    hvhx_argminmax_depth_flat_h(in_u16, HW, D, am);
    e=0;
    for (int hw=0;hw<HW;hw++){
        int mn=65536,mxv=-1; uint32_t mi=0,mai=0;
        for (int d=0;d<D;d++){int v=in_u16[hw*D+d]; if(v<mn){mn=v;mi=d;} if(v>mxv){mxv=v;mai=d;}}
        int t; t=abs((int)am[hw].min_val-mn); if(t>e)e=t;
        t=abs((int)am[hw].max_val-mxv); if(t>e)e=t;
        t=abs((int)am[hw].min_idx-(int)mi); if(t>e)e=t;
        t=abs((int)am[hw].max_idx-(int)mai); if(t>e)e=t;
    }
    ex_check("hvhx_argminmax_depth_flat_h", e, 0);

    /* find_max u8 */
    ex_fill_u8(in_u8, HW*D, 88, 0);
    hvhx_find_max_and_index_in_depth_b(in_u8, HW, D, idx, mv_b);
    e=0;
    for (int hw=0;hw<HW;hw++){
        int mxv=-1; uint32_t mai=0;
        for (int d=0;d<D;d++){int v=in_u8[hw*D+d]; if(v>mxv){mxv=v;mai=d;}}
        int t; t=abs((int)mv_b[hw]-mxv); if(t>e)e=t;
        t=abs((int)idx[hw]-(int)mai); if(t>e)e=t;
    }
    ex_check("hvhx_find_max_in_depth_b", e, 0);

    /* top1 u8 (per-row) */
    ex_fill_u8(in_u8, HW*D, 99, 0);
    hvhx_top1_qu8_dLE32_cr2flt(in_u8, HW, D, t1);
    e=0;
    for (int hw=0;hw<HW;hw++){
        int mxv=-1; uint32_t mai=0;
        for (int d=0;d<D;d++){int v=in_u8[hw*D+d]; if(v>mxv){mxv=v;mai=d;}}
        int t; t=abs((int)t1[hw].val-mxv); if(t>e)e=t;
        t=abs((int)t1[hw].idx-(int)mai); if(t>e)e=t;
    }
    ex_check("hvhx_top1_qu8_dLE32_cr2flt", e, 0);

    /* reducesum u8 */
    ex_fill_u8(in_u8, HW*D, 123, 0);
    hvhx_reducesum_depth_u8(in_u8, HW, D, sumbuf);
    e=0;
    for (int hw=0;hw<HW;hw++){uint32_t s=0; for(int d=0;d<D;d++) s+=in_u8[hw*D+d]; int t=abs((int)sumbuf[hw]-(int)s); if(t>e)e=t;}
    ex_check("hvhx_reducesum_depth_u8", e, 0);

    hmx_runtime_teardown();
    return ex_summary();
}
