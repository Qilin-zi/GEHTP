# HMX GEMM / 卷积 API 参考

声明: [`include/hmx_kernels.h`](../include/hmx_kernels.h)

9 个族, ~60 个函数. 全部数学形式:

```
out[m, n] = bias[n] + Σ_{k=0..K-1} act[m, k] * wgt[k, n]      (然后按族 sat)
```

族间区别: **数据类型** (act/wgt/out 精度) + **输出饱和**. 几何变体 (stride/dilate/...)
区别: **地址步进** (caller 的 im2col 层).

> 本设备 int8 HMX silent NOP → 所有 int8 族内部走 HVX int8 GEMM, 数学等价, exact.

## 通用参数约定

| 参数 | 含义 | 约束 |
|------|------|------|
| `act` | 激活, 行主序 [M][K] | 128B 对齐 |
| `wgt` | 权重, 行主序 [K][N] | 128B 对齐 |
| `bias` | per-channel 偏置 [N] | 长度 = N |
| `out` | 输出 [M][N] | 128B 对齐 |
| `M` | act 行 = 输出行 | 建议 32 倍数 |
| `K` | 收缩维 | **必须 32 倍数** |
| `N` | wgt 列 = 输出列 | 建议 32 倍数 |

---

## convf16 族 (fp16 × fp16 → fp16) — 真 HMX

```c
void hmx_convf16(const __fp16 *act, const __fp16 *wgt,
                 const __fp16 *bias, __fp16 *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
基本型 fp16 GEMM. 走真 HMX fp16 硬件 (20.4 TFLOPS). 容差 ≤1 ULP.

**变体 (11 个)** — 数学相同, 几何/对齐差异:
| 函数 | 用途 |
|------|------|
| `hmx_convf16` | 基本型 (任意 stride 由 caller im2col) |
| `hmx_convf16_1x1_stride1` | 1×1 卷积 = 纯矩阵乘, 最快 |
| `hmx_convf16_stride1` | stride 1 通用 |
| `hmx_convf16_stride2` | stride 2 下采样 |
| `hmx_convf16_NxN_stride1` | N×N 空间卷积 |
| `hmx_convf16_5x5_stride1` | 5×5 |
| `hmx_convf16_dilate_stride1` | 空洞卷积 |
| `hmx_convf16_stride1_aligned` | act/wgt 128B 对齐 (更快) |
| `hmx_convf16_1x1_stride1_unaligned` | 非 128B 对齐 |
| `hmx_convf16_1xN_stride2` | 可分离横腿 |
| `hmx_convf16_Nx1_stride2` | 可分离竖腿 |

**示例** (完整见 [examples/02_convf16_gemm](../examples/02_convf16_gemm/main.c)):
```c
hmx_runtime_setup(2*1024*1024);
hmx_convf16(act, wgt, bias, out, 32, 32, 32);   /* fp16 GEMM 32×32×32 */
```

---

## convbbb 族 (u8 × u8 → u8)

```c
void hmx_convbbb(const uint8_t *act, const uint8_t *wgt,
                 const int32_t *bias, uint8_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
u8 act × u8 wgt, int32 累加, sat 到 u8. HVX int8 GEMM 路径. exact (err=0).

**变体 (9 个)**:
| 函数 | 用途 |
|------|------|
| `hmx_convbbb` | 基本型 |
| `hmx_convbbb1x1_stride1` | 1×1 (注意: 维度前缀无下划线) |
| `hmx_convbbb1x1_stride1_unaligned` | 非对齐 |
| `hmx_convbbb1xN_stride2` | 横腿 |
| `hmx_convbbbNx1_stride2` | 竖腿 |
| `hmx_convbbb_stride1` / `_aligned` / `_stride2` | stride 系列 |
| `hmx_convbbb_dilate_stride1` | 空洞 |

> **命名约定**: 维度前缀 (`1x1`/`Nx1`) **无下划线** (`convbbb1x1`); 单词前缀
> (`stride`/`dilate`/`5x5`) **有下划线** (`convbbb_stride1`). convf16 族则全有下划线.

---

## convbbh (u8 × u8 → u16)

```c
void hmx_convbbh(const uint8_t *act, const uint8_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
u8×u8, 输出 u16 (宽动态范围, 不 sat). 用于需要后续高精度处理的中间层.

---

## convbcb (u8 × i16 → u8)

```c
void hmx_convbcb(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint8_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
u8 act × **i16** wgt → sat u8. i16 权重支持负值.

---

## convbnb 族 (u8 × i16 → u8) — 4 变体

```c
void hmx_convbnb(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint8_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
与 convbcb 数学等价 (u8×i16→u8), 但支持 **sparsity** (结构化稀疏).

| 函数 | 特性 |
|------|------|
| `hmx_convbnb` | 基本型 |
| `hmx_convbnb_1x1_stride1` | 1×1 |
| `hmx_convbnb_1x1_stride1_sparsity` | 带 sparsity mask (多一个 `const uint8_t *sparsity` 参数) |
| `hmx_convbnb_1x1_stride1_unaligned` | 非对齐 |

sparsity 变体签名 (注意 8 参数):
```c
void hmx_convbnb_1x1_stride1_sparsity(const uint8_t *act,
                                      const int16_t *wgt,
                                      const uint8_t *sparsity,  /* 非 zero 通道 mask */
                                      const int32_t *bias, uint8_t *out,
                                      uint32_t M, uint32_t K, uint32_t N);
```

---

## convhbh 族 (u8 × i8 → u16) — 12 变体

```c
void hmx_convhbh(const uint8_t *act, const int8_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
u8 act × **i8** wgt → u16. HVX int8 GEMM (`Q6_Wh_vmpyacc_WhVubVb`: u8×i8→i16 acc).

**变体 (12 个)**: `hmx_convhbh`, `1x1_stride1`, `1x1deep_stride1`, `1x1_stride1_unaligned`,
`1xN_stride2`, `Nx1_stride2`, `_NxN_stride1`, `_5x5_stride1`, `_stride1`, `_stride1_aligned`,
`_stride2`, `_dilate_stride1`.

> `:2x1` = u16 输出格式标识 (非空间扩展). `deep` = K>32 深 crouton.

---

## convhhh 族 (u8 × i8 → u16) — 5 变体, :2x2 格式

```c
void hmx_convhhh(const uint8_t *act, const int8_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
数学与 convhbh 等价 (u8×i8→u16), 仅 HMX 写回格式差异 (`:2x2` vs `:2x1`). 库内走 HVX,
对外行为相同.

**变体**: `hmx_convhhh`, `1x1_stride1`, `1x1deep_stride1`, `_NxN_stride1`, `_dilate_stride1`.

---

## convhnh 族 (u8 × i16 → u16) — 11 变体

```c
void hmx_convhnh(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
u8 act × i16 wgt → u16. 与 convhch 数学等价, 格式差异.

**变体 (11 个)**: `hmx_convhnh`, `1x1_stride1`, `_NxN_stride1`, `1x1deep_stride1`,
`1x1_stride1_unaligned`, `1xN_stride2`, `Nx1_stride2`, `_5x5_stride1`, `_stride1`,
`_stride1_aligned`, `_stride2`, `_dilate_stride1`.

---

## convhch 族 (u8 × i16 → u16) — 6 变体, :2x2 格式

```c
void hmx_convhch(const uint8_t *act, const int16_t *wgt,
                 const int32_t *bias, uint16_t *out,
                 uint32_t M, uint32_t K, uint32_t N);
```
数学与 convhnh 等价 (u8×i16→u16), `:2x2` 写回格式.

**变体 (6 个)**: `hmx_convhch`, `1x1_stride1`, `1x1deep_stride1`, `1xN_stride2`,
`Nx1_stride2`, `_5x5_stride1`.

---

## 多 tile (M/N/K > 32)

所有族都支持 M/N/K > 32. 库内部:
- M/N 维: 外层循环 32×32 tile
- K 维: 用 activation+weight pair 循环累加 (不需 dC deep 字段), identity scales +
  标量 bias 后加

验证: convf16 多 tile M/N/K = 64×32×32 / 32×32×64 / 32×64×32 / 64×64×64 全 err=0.
example: [examples/12_multitile_gemm](../examples/12_multitile_gemm/main.c).

## 选型速查

| 你的场景 | 推荐 |
|----------|------|
| fp16 推理 | `hmx_convf16` (真 HMX) |
| 1×1 fp16 | `hmx_convf16_1x1_stride1` (最快) |
| u8 量化推理 | `hmx_convbbb` (u8 out) 或 `hmx_convbbh` (u16 out, 宽动态) |
| 带负权重的 u8 | `hmx_convbcb`/`convbnb` (i16 wgt) |
| 高精度中间层 | `convhbh`/`convhnh` (u16 out) |
| 结构化稀疏 | `hmx_convbnb_1x1_stride1_sparsity` |
| 不确定 | 基本型 (无后缀) |
