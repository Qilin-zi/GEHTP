# U5 gdnsm — GDN 递归状态机 kernel 族 功能手册

声明: [`include/gdn_sm.h`](../include/gdn_sm.h) · 源: `src/v22/gdn_ref.c` (标量 oracle)
+ `src/v22/gdn_kern.c` (HVX kernel) (源模块 gdn_sm_v81 G1-G9 闭合) ·
测试: [examples/19_gdn_sm](../examples/19_gdn_sm/main.c)

## 功能

Gated DeltaNet 三件套, 状态一律**外置 f32** (调用方持有), kernel 纯函数步进:

| 族 | kernel | oracle |
|----|--------|--------|
| B1 conv1d (d_conv 因果卷积 + SiLU) | `conv1d_step_f16` / `conv1d_block_f16` | `ref_conv_step` / `ref_conv_block` |
| B2 delta-rule chunk 更新 (32 head 批) | `delta_chunk_f16` | `ref_delta_token` |
| B3 solve-tri `T=(I+L)^{-1}` | `solve_tri_f16` | `ref_solve_tri` |

数据: f16 存 `int16_t`, kernel 内部 f32 计算。判定真值 = oracle (host numpy 对拍已到
1e-14)。

## 数值契约 (chunk 形式, 每 head)

```
G_j = Σ_{t<=j} g_t                       (inclusive cumsum)
L[j,i] = exp(G_j-G_i)·(k_j·k_i)·β_j       (严格下三角)
b_j    = β_j·(v_j - exp(G_j)·S0·k_j)
u      = (I+L)^{-1} b                     ← B3
y_t    = exp(G_t)·(S0·q_t) + Σ_{i<=t} exp(G_t-G_i)·(q_t·k_i)·u_i
S'     = exp(G_C)·S0 + Σ_i exp(G_C-G_i)·u_i·k_i^T
```
per-token oracle (等价): `S *= exp(g); yk=S·k; δ=β(v-yk); S += δ⊗k; y=(S·q)/√D`。

## API 细节

```c
conv_state_t st = { d_inner, d_conv, win_f32 };          /* win: (d_conv-1)×d_inner */
conv1d_step_f16 (&st, w_f16, x_f16, y_f16);
conv1d_block_f16(&st, w_f16, x_f16, y_f16, m);           /* m token 一次 */

rec_state_t rs = { nheads, d, s_f32 };                   /* s: heads×d×d */
delta_chunk_f16(&rs, k16, v16, q16, beta16, g16, y16, c, ntok);
/* 缓冲 [h][ntok][dim]; 传 base+t0*dim 只加 token 偏移, head 步幅内部算
 * c = 本 chunk 实际 token 数 (≤ GDN_C_MAX=64, 尾块可不对齐) */

solve_tri_f16(L_f16, T_f16, c);                          /* L 严格下三角 C×C */

gdn_f32_to_f16 / gdn_f16_to_f32;                         /* f16 工具 */
gdn_lcg_next / gdn_lcg_norm;                             /* 可复现输入 (host python 同 LCG) */
```

## 拆分一致性

`delta_chunk_f16(c=16)` == `c=8 + c=8` (输出与终态) — chunk 大小是自由参数
(example 19 有门 `chunk_split_16_8_8`)。

## 判据 (example 19)

| 门 | 判据 |
|----|------|
| f16_roundtrip_idempotent | encode(decode(h))==h + 半 ULP 界 (20000 随机, 覆盖次正规) |
| conv_step_cos / delta_100tok_cos / delta_state_cos | ≥0.9999 |
| solve_tri_cos | ≥0.99995 |
| conv_block_state_bytes / conv_state_bitexact_guarded | 状态 byte/bit-exact (64KB 守卫区) |

## 坑 (MODULE B 实测)

- f16 次正规: `e=-1` 起步的 normalize 循环 — 曾用 `e=0` 出错 (G2)。
- `inv_crouton` 存 u16 于 `int16_t`, **读时必须 `(uint16_t)` cast 防符号扩展**。
- 状态外置意味着多线程/多域共享状态要自己加锁 (每 head 独立可无锁分片)。
