/*
 * 06_dwconv — 深度卷积 (fp16 + u8) + golden
 * =====================================================================
 * 教学: hmx_dwconvf16 / hmx_dwconvbbb: 3×3 深度卷积, 每通道独立空间卷积.
 *   数据布局: act[H][W][C] 行主序 (无 padding), wgt[C][9] 通道主序.
 *   卷积: out[h,w,c] = bias[c] + Σ_{kh,kw∈[0,3)} act[hh,ww,c] * wgt[c,kh,kw]
 *   其中 hh=h+kh, ww=w+kw; 越界样本跳过 (clamped, 不补零).
 *   fp16 容差: raw fabsf ≤ 1 (匹配 test_all_hmx 口径); u8 exact.
 */
#include "hvxhmx.h"
#include "example_util.h"

#define H 5
#define W 5
#define C 9

static __fp16 a_f [H*W*C] __attribute__((aligned(128)));
static __fp16 w_f [C*9]   __attribute__((aligned(128)));
static __fp16 b_f [C]     __attribute__((aligned(128)));
static __fp16 o_f [H*W*C] __attribute__((aligned(128)));
static __fp16 g_f [H*W*C] __attribute__((aligned(128)));

static uint8_t a_u [H*W*C] __attribute__((aligned(128)));
static uint8_t w_u [C*9]   __attribute__((aligned(128)));
static int32_t b_i [C]     __attribute__((aligned(128)));
static uint8_t o_u [H*W*C] __attribute__((aligned(128)));
static uint8_t g_u [H*W*C] __attribute__((aligned(128)));

int main(void)
{
    ex_open_result("06_dwconv");
    if (hmx_runtime_setup(2*1024*1024) != 0) { ex_log("FATAL: setup FAIL"); return ex_summary(); }

    /* fp16 depthwise: act[H*W*C], wgt[C*9] (channel-major), bias[C] */
    ex_fill_f16(a_f, H*W*C, 7,  0.01f);
    ex_fill_f16(w_f, C*9,   9,  0.01f);
    ex_fill_f16(b_f, C,     11, 0.01f);
    hmx_dwconvf16(a_f, w_f, b_f, o_f, H, W, C);
    for (uint32_t h=0;h<H;h++) for (uint32_t w=0;w<W;w++) for (uint32_t c=0;c<C;c++) {
        float acc = (float)b_f[c];
        for (int kh=0;kh<3;kh++) for (int kw=0;kw<3;kw++) {
            int hh=(int)h+kh, ww=(int)w+kw;
            if (hh>=0 && hh<(int)H && ww>=0 && ww<(int)W)
                acc += (float)a_f[hh*W*C+ww*C+c] * (float)w_f[c*9+kh*3+kw];
        }
        g_f[h*W*C+w*C+c] = (__fp16)acc;
    }
    int e=0;
    for (uint32_t i=0;i<H*W*C;i++){float d=(float)o_f[i]-(float)g_f[i]; if(d<0)d=-d; if(d>e)e=(int)(d+0.5f);}
    ex_check("hmx_dwconvf16 3x3 fp16", e, 1);

    /* u8 depthwise */
    ex_fill_u8 (a_u, H*W*C, 13, 256);
    ex_fill_u8 (w_u, C*9,   15, 256);
    ex_fill_i32(b_i, C,     17, 100, 50);
    hmx_dwconvbbb(a_u, w_u, b_i, o_u, H, W, C);
    for (uint32_t h=0;h<H;h++) for (uint32_t w=0;w<W;w++) for (uint32_t c=0;c<C;c++) {
        int32_t acc = b_i[c];
        for (int kh=0;kh<3;kh++) for (int kw=0;kw<3;kw++) {
            int hh=(int)h+kh, ww=(int)w+kw;
            if (hh>=0 && hh<(int)H && ww>=0 && ww<(int)W)
                acc += (int32_t)a_u[hh*W*C+ww*C+c] * (int32_t)w_u[c*9+kh*3+kw];
        }
        g_u[h*W*C+w*C+c] = acc<0?0:(acc>255?255:(uint8_t)acc);
    }
    e=0; for (uint32_t i=0;i<H*W*C;i++){int d=abs((int)o_u[i]-(int)g_u[i]); if(d>e)e=d;}
    ex_check("hmx_dwconvbbb 3x3 u8", e, 0);

    hmx_runtime_teardown();
    return ex_summary();
}
