# U8 smallm — 小-M GEMV pad-256 路线 功能手册

U4 的**用法单元** (无独立 kernel): decode 场景 M=1/16 时, 把激活面 pad 到 256 行再喂
W4A16 引擎, 利用 "成本与 M 无关" 的 tile-walk 特性。

测试: [examples/18_smallm_gemv](../examples/18_smallm_gemv/main.c) ·
资产生成: `host/gen_smallm_assets.py` · 源: htpw4a16_v81 MODULE A 结论

## 原理 (MODULE A A1 设备结论)

| 事实 | 数值 |
|------|------|
| M=1 pad-256 invoke 成本 | == M=256 (tile-walk bound, 同引擎面只换 act) |
| pad 行内容 | **raw=32768** (对称激活的中性点: x=0) |
| pad 行输出 | 与激活数据无关, 恒定 (可对拍) |
| K2560 N2560 | ~1047 µs/步, 与 M 无关 |

即 decode GEMV 不需要专门 kernel: 一步 pad-256 引擎调用即得,
多余行丢弃。`m1 ≤ mf + mf/16` 是 example 18 的成本门。

## 资产约定 (`assets/smallm/`)

| 文件 | 内容 |
|------|------|
| `act_p1_v0.raw` / `act_p1_v1.raw` | 256×K 面: 行0 = s256 v0/v1 解码行0, 行1..255 = 32768 |
| `act_p16_v0.raw` | 行0..15 = s256 v0 解码行0..15, 行16..255 = 32768 |

面为 crouton16_row4 打包布局, 由 `host/gen_smallm_assets.py` 从 `assets/s256` 生成
(round-trip 自检)。注意: pad 行在打包面中的字节位置由布局函数决定 —
`gen_smallm_assets.py` 是唯一生成器, 不要手写。

## 用法 (example 18 摘)

```c
/* 与 M=256 完全相同的 carve/invoke/read_out, 只换 act 面 */
cpu_to_vtcm(e.act, act_p1_v0, M*K*2);
dc_w4_invoke(&e);
dc_w4_read_out(&e, out);
/* out 是 crouton16_row4 面 — 先解码成线性 (M,N) (例程内 minv_crouton),
 * 解码后的行0 即 GEMV 结果; p1/p16 的行 0..N-1 分别等于 full 的行 0..N-1 */
```

## 判据 (example 18, 解码后全 byte-exact)

1. `p1_row0_eq_full_row0` — M=1 pad 行0 == full-256 行0
2. `p16_rows0_15_eq_full` — M=16 pad 行0..15 == full 行0..15
3. `pad_rows_invariant` — v0/v1 两个 p1 面: pad 行 (1..255) byte-equal, 行0 不同
4. `row0_tracks_input` — 行0 随输入变化
5. `cost_M_invariant` — 50 次中位 invoke: M=1 ≤ M=256 × (1+1/16)

## 何时不用

M 可凑成 256 倍数时直接 U4; 只有 M<256 的零星 GEMV 才 pad。K/N 仍须 32 倍数。
