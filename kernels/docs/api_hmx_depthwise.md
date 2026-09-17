# HMX Depthwise 卷积 API 参考

声明: [`include/hmx_kernels.h`](../include/hmx_kernels.h) (Depthwise 段)

深度可分离卷积的"深度"腿: 每个通道独立做 3×3 空间卷积, 不跨通道混合.

```
out[h, w, c] = bias[c] + Σ_{kh=0..2, kw=0..2} act[h*S+kh*D, w*S+kw*D, c] * wgt[kh, kw, c]
```

## hmx_dwconvf16 (fp16)

```c
void hmx_dwconvf16(const __fp16 *act, const __fp16 *wgt,
                   const __fp16 *bias, __fp16 *out,
                   uint32_t H, uint32_t W, uint32_t C);
```
3×3 fp16 深度卷积, stride 1.

| 参数 | 含义 |
|------|------|
| `act` | [H+2][W+2][C] (含 padding=1) |
| `wgt` | [3][3][C] |
| `bias` | [C] |
| `out` | [H][W][C] |
| `H,W,C` | 输出高/宽/通道 |

## hmx_dwconvf16_dilate_stride1

```c
void hmx_dwconvf16_dilate_stride1(const __fp16 *act, const __fp16 *wgt,
                                  const __fp16 *bias, __fp16 *out,
                                  uint32_t H, uint32_t W, uint32_t C);
```
空洞 (dilation) 变体, stride 1.

## hmx_dwconvbbb (u8)

```c
void hmx_dwconvbbb(const uint8_t *act, const uint8_t *wgt,
                   const int32_t *bias, uint8_t *out,
                   uint32_t H, uint32_t W, uint32_t C);
```
3×3 u8 深度卷积, int32 bias, sat u8 输出.

> depthwise 走标量/H VX 路径 (HMX 矩阵单元不适合 per-channel 空间卷积). H/W/C 小时
> 性能不如 GEMM 族, 但深度可分离网络必需.

example: [examples/06_dwconv](../examples/06_dwconv/main.c)
