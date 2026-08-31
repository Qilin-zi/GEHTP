# U4 w4a16 — W4A16 HMX GEMM 引擎 功能手册

声明: `dc_w4_*` 于 [`include/dc_parts.h`](../include/dc_parts.h) ·
kernel: `src/v22/w4a16_driver_dc.c` + `w4a16_v81deep_conv1x1_kernel.inc` (t10 生产路径,
与 htpw4a16_v81 同源) · host 打包: `host/pack_oplist.cc` + `host/weight_layout_pack.py`
· 测试: examples/17, 18, 21

## 数学

```
out[m,n] (u16 crouton 面) = round( (Σ_k (act[m,k]-32768) · W[k,n]) / 7 ) + bias_fold[n] + 32768
```
- 激活 u16 对称 (offset 32768), 权重 4-bit packed (w4), 计算 16-bit — HMX mxmem。
- 引擎面: act (M×K×2) / out (M×N×2) / wt (K×N/2) / bias (N/32×512) /
  atbl/otbl (8×(K/32)×4), 全部 **2KB 对齐** carve。

## 形状约束 (硬)

| 约束 | 值 | 来源 |
|------|----|------|
| M | **256 倍数** | m_t=8 tile 硬约束 (M=32/128 FAIL 实测); 小 M → U8 pad-256 |
| K, N | 32 倍数 | tile 宽 |
| 面对齐 | 2KB | HMX mxmem (T10) |

## 数值契约 (设备闭合)

| 形状 | 判据 | 结果 |
|------|------|------|
| 256³ | vs QNN gold | **bit-exact 65536/65536** |
| M=256 K=2560 N=2560 | vs 标量 oracle `Y_gold.raw` | cos 0.99999988, **max 37 LSB** (tol 40) |
| 复跑/跨引擎重开 | byte-exact | PASS (example 17/21 门) |

oracle 公式 (`assets/s2560/Y_gold.json`):
`Y[n,m]=clip(round_div7(Σ_k (A[m,k]-32768)·W[k,n])+32768)`, (N,M) row-major u16。

## API (U2 封装)

```c
dc_w4_carve(&e, &ar, m, k, n, atbl_ddr, otbl_ddr);   /* 一次性, HMX 面 2KB 对齐 */
dc_w4_invoke(&e);                                    /* 每次调用: 表回填→FLUSH→kernel */
dc_w4_read_out(&e, recv);                            /* INVALIDATE + memcpy 出 DDR */
```

## 性能 (设备闭合基线)

| 项 | 值 |
|----|----|
| K2560 N2560 invoke | **1047 µs ≈ 3.2 TFLOPS** (摊薄) |
| op 全路径 (含 3.2MB 权重 restage + DMA) | ~32 ms (U6 op_us[3]) |
| 小 M 成本 | 与 M 无关 (tile-walk bound) → 见 U8 |

## host 侧配套

- `host/pack_oplist.cc` — blob 组包 (slot 表 + q8_0 crouton vendored 打包)。
- `host/weight_layout_pack.py` — 权重布局打包 (packed_weight/folded_bias/表)。
- `host/inv_crouton16.py` — crouton16_row4 面编解码 (host 侧验证)。
- 资产: `assets/s256/` (256³ 自检集 + 位恒等金标 `Y_gold_2563.raw` — W4A16 256³
  闭合 oracle, (N,M) 线性, byte-exact 门。`Y_ref_v0.raw` 是 int8 全精度标量金标,
  与 w4 引擎仅 ~5.7% 逐位相同, **不是引擎参考, 不作门**),
  `assets/s2560/` (K2560 oracle 集), `assets/smallm/` (U8 pad 面)。

## 已知行为

- invoke 破坏性重写表面 (表回填); 复用引擎先重传 act/wt/bias。
- out 是 crouton16_row4 布局 — CPU 消费用 `dc_w4_read_out` 出 DDR 后按
  `inv_crouton` 还原 (example 21 `minv_crouton` 为镜像)。
