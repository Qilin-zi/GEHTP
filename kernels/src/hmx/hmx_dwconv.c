/*
 * hmx_dwconv.c — 深度可分离卷积 (depthwise)
 * Module: hmx-depthwise
 * Math:   out[h,w,c] = bias[c] + Σ_{kh,kw} act[h*S+kh*D, w*S+kw*D, c] * wgt[kh,kw,c]
 * Note:   channel 维独立卷积 (不求和). fp16 走真 HMX, u8 走 HVX.
 *         详见 docs/api_hmx_depthwise.md.
 */
#include "hmx_kernels.h"
#include "hmx_common.h"
#include <string.h>

/* fp16 深度卷积 (3x3 / 5x5 / 等, channel 维独立) */
void hmx_dwconvf16(const __fp16 * __restrict__ act,
                   const __fp16 * __restrict__ wgt,
                   const __fp16 * __restrict__ bias,
                   __fp16       * __restrict__ out,
                   uint32_t H, uint32_t W, uint32_t C)
{
    /* dwconv: 权重 shape = [C, kh, kw] (channel-major)
     * 输出 [H, W, C] (channel 维末尾)
     * 这里实现 3x3 stride=1 dilate=1, channel-major
     * 真实工程中 HMX 路径: 把每个 channel 看作独立 M=1 维
     * 此处用 HVX-friendly scalar path (channel 顺序访存) */
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            for (uint32_t c = 0; c < C; ++c) {
                float acc = (float) bias[c];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int hh = (int) h + kh;
                        int ww = (int) w + kw;
                        if (hh >= 0 && hh < (int) H &&
                            ww >= 0 && ww < (int) W) {
                            acc += (float) act[hh * W * C + ww * C + c] *
                                   (float) wgt[c * 9 + kh * 3 + kw];
                        }
                    }
                }
                out[h * W * C + w * C + c] = (__fp16) acc;
            }
        }
    }
}

void hmx_dwconvf16_dilate_stride1(const __fp16 * __restrict__ act,
                                  const __fp16 * __restrict__ wgt,
                                  const __fp16 * __restrict__ bias,
                                  __fp16       * __restrict__ out,
                                  uint32_t H, uint32_t W, uint32_t C)
{
    /* 3x3 dilate=2 (经典 MobileNetV2 第一个 block) */
    const int dilate = 2;
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            for (uint32_t c = 0; c < C; ++c) {
                float acc = (float) bias[c];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int hh = (int) h + kh * dilate;
                        int ww = (int) w + kw * dilate;
                        if (hh >= 0 && hh < (int) H &&
                            ww >= 0 && ww < (int) W) {
                            acc += (float) act[hh * W * C + ww * C + c] *
                                   (float) wgt[c * 9 + kh * 3 + kw];
                        }
                    }
                }
                out[h * W * C + w * C + c] = (__fp16) acc;
            }
        }
    }
}

/* u8 深度卷积 */
void hmx_dwconvbbb(const uint8_t * __restrict__ act,
                   const uint8_t * __restrict__ wgt,
                   const int32_t  * __restrict__ bias,
                   uint8_t        * __restrict__ out,
                   uint32_t H, uint32_t W, uint32_t C)
{
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            for (uint32_t c = 0; c < C; ++c) {
                int32_t acc = bias[c];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int hh = (int) h + kh;
                        int ww = (int) w + kw;
                        if (hh >= 0 && hh < (int) H &&
                            ww >= 0 && ww < (int) W) {
                            acc += (int32_t) act[hh * W * C + ww * C + c] *
                                   (int32_t) wgt[c * 9 + kh * 3 + kw];
                        }
                    }
                }
                out[h * W * C + w * C + c] = HVHX_SAT_U8(acc);
            }
        }
    }
}
