# api_v23_gemmdispatch — U17 MatMul 三路由决策 + 执行体

源: `src/runtime/gemm_dispatch.c` · 头: `include/gemm_dispatch.h`
· 例: `31_gemm_dispatch` (8 门 PASS)

## 决策表 (纯函数, host 可对拍)

```c
enum { GR_W4A16, GR_SMALLM, GR_DENSE_F16 };
int gemm_route_for(uint32_t m, uint32_t k, uint32_t n);
```

| M | 路由 | 依据 |
|---|------|------|
| `M ≥ 256 && M%256==0` | **GR_W4A16** | HMX W4 引擎; 单 invoke 仅 M=256, M>256 拆块拼接 |
| `M < 32` | **GR_SMALLM** | pad-256 复用 W4 引擎取前 M 行 (例 18: 与 M=256 同价) |
| 其余 | **GR_DENSE_F16** | f16 便携 kernel, f32 累加 |

边界: 1 SMALLM / 32 DENSE / 128 DENSE / 256, 512 W4A16。

## 执行体

```c
void gemm_f16_dense(a, w, c, m, k, n);          /* f16 位型, f32 累加 (gdn_sm 同源软转) */
int  gemm_smallm_pad256(struct dc_w4* e, const int16_t* act_lin, uint32_t m,
                        int16_t* out_lin);      /* e: carve(M=256,K,N), wt/bias 已 stage */
int  gemm_w4a16_m256(struct dc_w4* e, const int16_t* act_lin, uint32_t m,
                     int16_t* out_lin);         /* m 为 256 倍数, 逐块 invoke+解码拼接 */
void gemm_crouton_encode/decode(lin, surf, rows, cols);  /* rows%32==0, cols%32==0 */
```

## M=512 拆块依据 (重要)

W4 kernel descriptor 是 **ABI 常量段** (`n_tiles_pow2=32` → M-loop 8×32 行,
`m_total_minus_step=8`; w4a16_driver_dc.c 注释 "改了等于重验 2-B")。单 invoke
只有 M=256 是闭合形状; **512 面直发单 invoke 高位行全错** (实测 512/512 行不一致,
row0 即错)。`gemm_w4a16_m256` 在 dispatch 层按 256 块拆: encode→flush→invoke→
read_out→decode 逐块, 输出线性拼接。

## 设备门 (31_gemm_dispatch, 2026-08-16)

- `route_table_exact`: 600×3 扫描 + 5 边界 0 mismatch
- `dense_f16_vs_scalar_oracle`: cos≥0.9999 + ULP 包络
- `w4a16_256_gold_byteexact`: 256³ vs Y_gold_2563 **65536/65536**
- `smallm_*`: 行 0 == 全引擎行 0 == 金标行 0; M=1 med 1510us vs M=256 1029us (ratio 1.47)
- `crouton512_roundtrip_byteexact`: 512 行编解码往返恒等 (分离缓冲)
- `w4a16_512_split_gold`: 拆块后上半==A 金标 / 下半==B 金标 (byte-exact)

## 坑 (本轮实测, 两条硬教训)

1. **crouton decode 不能原地**: 读流 (phase 外层) 与写行 (g 外层) 顺序不同, 原地在
   高位行读到未解码残片 — 256 面实测 32767/65536 败。必须分离缓冲。
2. **热路径禁每次 memalign/free 128KB**: 设备实测引入 ~38ms 分配抖动 (smallm
   M=1 从 1.5ms 劣化到 39ms)。库内 scratch 用进程级单例 (`scratch_ensure`,
   引擎路径调用方串行化, 共享安全)。
