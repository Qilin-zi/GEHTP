# 低级 Core API 参考 (高级用户)

声明: [`include/hmx_kernels.h`](../include/hmx_kernels.h) (诊断段), [`include/hvx_int8gemm.h`](../include/hvx_int8gemm.h), [`include/hmx_crouton.h`](../include/hmx_crouton.h)

这些是库内部用的低级 core / 诊断函数, 暴露给需要精细控制 (手工 VTCM 打包、自定义
tile 循环、精度诊断) 的高级用户. **普通用户用高层 API (hmx_convf16 等), 不需要这些.**

## HVX int8 GEMM core

所有 int8 族内部走这些. 直接用可获得最大控制.

### hvx_int8gemm_pack_wgt
```c
void hvx_int8gemm_pack_wgt(const int8_t *wgt, int8_t *packed, uint32_t K, uint32_t N);
```
把行主序 [K][N] i8 权重打包成 HVX 友好的 [32][128] padded 布局 (32B data + 96B zero/row).
 padded 布局规避 hexagon-clang 19.0.07 软件流水线 stale-vector bug.

### hvx_int8gemm_32x32x32_core
```c
void hvx_int8gemm_32x32x32_core(const uint8_t *act, const int8_t *wgt,
                                int32_t *acc_out);
```
单 tile 32×32×32 int8 GEMM, 输出 int32 累加器 (不 sat). `Q6_Wh_vmpyacc_WhVubVb` (u8×i8→i16).

### hvx_int8gemm_bias_32x32x32_core
```c
void hvx_int8gemm_bias_32x32x32_core(const uint8_t *act, const int8_t *wgt,
                                     const int32_t *bias, uint8_t *out);
```
+ int32 bias + sat u8 输出.

### hvx_int8gemm_bias_u16_32x32x32_core
```c
void hvx_int8gemm_bias_u16_32x32x32_core(const uint8_t *act, const int8_t *wgt,
                                         const int32_t *bias, uint16_t *out);
```
+ bias, u16 输出 (不 sat).

### hvx_int8gemm / hvx_int8gemm_bias_multi
```c
void hvx_int8gemm           (const uint8_t *act, const int8_t *wgt, int32_t *out,
                             uint32_t M, uint32_t K, uint32_t N);
void hvx_int8gemm_bias_multi(const uint8_t *act, const int8_t *wgt,
                             const int32_t *bias, uint8_t *out,
                             uint32_t M, uint32_t K, uint32_t N);
```
多 tile 版 (M/N/K > 32).

### hvx_int8gemm_diag
```c
void hvx_int8gemm_diag(const uint8_t *act, const int8_t *wgt, ...);
```
诊断变体 (详签名见头). 不建议生产用.

> ⚠️ `hvx_int8gemm.c` 必须用 **`-O1`** 编译 (软件流水线 stale-vector bug, `-O2` 生成
> 错误代码). build_libs.sh 已处理.

## Crouton 打包 / 解包 helper

[`include/hmx_crouton.h`](../include/hmx_crouton.h). 把行主序数据打包进 HMX pair-interleave
crouton 布局, 或反向解包.

```c
/* 位置计算: pos(row,col) = (row/2)*64 + 2*col + (row&1) */
static inline uint32_t hmx_crouton_pos(uint32_t row, uint32_t col);

/* fp16 crouton (2KB) */
void hmx_pack_act_fp16 (const __fp16 *row_major, __fp16 *crouton);
void hmx_pack_wgt_fp16 (const __fp16 *row_major, __fp16 *crouton);
void hmx_unpack_out_fp16(const __fp16 *crouton, __fp16 *row_major);

/* u8 crouton (1KB) */
void hmx_pack_act_u8  (const uint8_t *row_major, uint8_t *crouton);
void hmx_pack_wgt_i8  (const int8_t  *row_major, int8_t  *crouton);
void hmx_unpack_out_u8(const uint8_t *crouton, uint8_t *row_major);
```

## HMX 诊断 core (实验性, 非稳定)

这些是逆向期用于探测 HMX 行为的 core, 保留供精度诊断. 数学正确但 API 可能变.

```c
void hmx_phase0_gemm_fp16_core(const __fp16 *act_vtcm, const __fp16 *wgt_vtcm,
                               const __fp16 *scales_vtcm, __fp16 *out_vtcm);   /* 单 crouton fp16 */

void hmx_gemm_fp16_crouton_ex (const __fp16 *act, const __fp16 *wgt,
                               const __fp16 *scales, __fp16 *out, unsigned mode);  /* mode 0..4 */

void hmx_phase1_gemm_int8_core(const uint8_t *act_vtcm, const int8_t *wgt_vtcm,
                               const void *bias_vtcm, uint8_t *out_vtcm);     /* 单 crouton int8 */

void hmx_pseudoint8_32x32x32_core(const __fp16 *act_fp16, const __fp16 *wgt_fp16,
                                  const __fp16 *scales, __fp16 *out_fp16);   /* fp16 伪 int8 */

void hmx_gemm_int8_crouton_ex(const uint8_t *act_vtcm, const int8_t *wgt_vtcm,
                              const void *bias_vtcm, uint8_t *out_vtcm, unsigned mode);
```

> `hmx_phase1_gemm_int8_core` / int8 诊断 core 在本设备 int8 HMX silent NOP, 仅用于
> 编码验证 (指令 byte-match 生产 .so), **不产出正确数值**. 数值正确性用 `hvx_int8gemm_*`.

## HMX 字段布局 (极高级)

[`include/hmx_fields.h`](../include/hmx_fields.h) — HMX tile load 指令的字段 bit 布局
(dW / spatial / scale / deep 等). 仅当手工构造 HMX 指令编码时需要.
