# api_v23_pxbridge — U13 f32↔f16↔INT16 精度桥 (unipolar 契约)

源: `src/hvx/pxbridge.c` · 头: `include/pxbridge.h` · 例: `27_pxbridge` (20 门 PASS)

## unipolar 契约 (V81 HMX INT16 的真实语义)

V81 HMX INT16 = **ufixed16, offset 恒 -32768** (AIMET 自适应 offset 会 0xc26 —
见 memory v81-int16-symmetric-activation-requirement)。因此桥是**非对称单极**的:

```c
/* code = clamp(round(x/scale), 0, 65535) - 32768 */
int16_t  pxb_f32_to_i16 (float x, float scale);
float    pxb_i16_to_f32 (int16_t q, float scale);   /* = (q+32768)*scale */
uint16_t pxb_f32_to_f16 (float x);
float    gdn_f16_to_f32 / gdn_f32_to_f16;           /* RNE, 库内同源 */
/* 向量版 + 桥: pxb_i16_to_f16_v (code 侧 f16 位型) */
```

- 零 ⇔ 码 `0x8000`; **负输入钳到零码** (不是镜像!) — 契约行为, 非 bug。
- 上界码 `0x7FFF` = 65535·scale。

## 误差包络 (设备实测)

| 变换 | 包络 |
|------|------|
| f32→i16→f32 | `sc·0.5 + sc·5e-3` (半量子 + 商舍入; ulp(65535)≈0.002 量子, 1e-3 slack 不够) |
| f32→f16→f32 | `|r|·1.2e-3 + 6.5e-8` (次正规区绝对底 = f16 subnormal 半量子) |
| i16→f16→f32 vs 直接 decode | `sc·0.5 + sc·1e-3 + |via|·1.2e-3 + 6.5e-8` |
| code 空间步长 | 容差门 `|Δd−sc| ≤ |d1|·1.3e-3 + 1e-7` (f32 乘法舍入, 精确等式过严) |

## 设备门 (27_pxbridge, 2026-08-16, sc ∈ {1e-3,1e-2,0.1,1} ×20 门)

`f16_roundtrip_ulp_envelope` / `i16_halfstep_envelope_zero_exact` / 零码↔零值 /
`negative_clamps_zero_code` / `f16_i16_combo_envelope` / `f16_bridge_envelope` /
`code_space_linear_exact` / `clamp_extremes` / `batch_vs_scalar_identical`。

## 坑 (本轮实测)

- 生成器必须 **u∈[0,1)** unipolar — 双极生成器负半全超包络 (err≈49893)。
- f16 位型按 code 值 (最大 65504) 占满 f16 动态范围, 桥恒等门用包络不用恒等。
