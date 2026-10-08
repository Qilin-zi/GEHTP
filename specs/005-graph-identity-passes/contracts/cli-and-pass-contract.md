# Contract: CLI 旗标与 pass 行为契约

本特性对外暴露面 = 三个 CLI 旗标语义 + stats 输出格式 + op_data 标记约定。消费方：编译器用户（hnnx_compile/gehtp compile）、发射器（wtop_emit）、设备执行器（oplist_exec）。

## CLI 契约（镜像既有 `--no-pass`/`--enable-pass` 机制，零新旗标语法）

```bash
hnnx_compile ... [--no-pass <name>] [--enable-pass <name>] [--opt-stats]
# name ∈ { view_fold, eltwise_const_fold, fc_bias_fold, ... 既有 pass 名 }
```

| 契约项 | 语义 |
|---|---|
| 默认态 | view_fold / eltwise_const_fold / fc_bias_fold 默认**开**（与 fc_bias_fold 现状一致） |
| `--no-pass view_fold` | 本 pass 完全不跑（run_phase 空转 0 开销）——逃生与 A/B 对照用 |
| `--opt-stats` | 每 pass 打印 `matched/rewritten`，格式 `<name>=<m>/<r>`（stats_line 既有格式，test_opt_passes 子串断言依赖它，不得改格式） |

**不变式**: `--no-pass` 组合下编译必须成功且产物可上板（对照门 T006/T012 依赖此性质）。

## stats 输出契约

```text
[opt-stats] ... view_fold=<matched>/<rewritten> eltwise_const_fold=<m>/<r> fc_bias_fold=<m>/<r> before=<n> after=<n'>
```

- matched = 谓词命中次数（含因门拒绝前的命中），rewritten = 实际改写次数。
- 幂等性可从外部观测：同图二跑，rewritten 必须为 0。

## op_data 标记契约（P3，复用 e4d1dbb）

| 标记 | 写方 | 读方 | 语义 |
|---|---|---|---|
| `has_bias` (numeric=1) | pass_fc_bias（FC **与** MatMul 链头） | extract_fc_extra / extract_matmul_extra、op_matmul.cpp 发射门、host execute | inputs[2] 是 dense bias；发射 OP_MATMUL_F16_BIAS(36) |

**门约束（US3b 后现状）**: 32 对齐形状（M%32==0 ∧ K%32==0 ∧ N%32==0）的 MatMul 带 has_bias 标记发射 36 号**合法**——pass 侧 HMX 门与发射侧硬报错均已随 US3b（69893e4）撤销，单测 [15b] 改正例。仍保留的排除: **batched**（flags&4）不折（设备硬错防御）；SPLIT N 分块维持 `!has_bias` 条件（SPLIT+bias 的 HMX 通道已接线但 SC-005 未覆盖，超 VTCM 极端形保持整 op）。

## 设备契约（3b 已落地，2026-10-08）

OP_MATMUL_F16_BIAS(36) 的 HMX 路径 bias 应用 = **scale 区奇位 out_bias 技法**: 每 (r,c) tile 循环内 `sca[2n]=0x3C00(1.0), sca[2n+1]=bias[col]` → `mxmem2(sca)` → cvt 一步 `f16(acc·1.0+bias)`（单次舍入，与 exec_conv2d_hmx 同源；较原设想的"unpack 前逐列加"零额外 pass）。SC-005 已过（46 号例 6/6: 引擎双到位 / cos=1.000000 / 逐元素三判据 0 违例 / NaN 0；"≤2 ULP"任务书口径细化为三判据——f16 次正规区对消输出的位距膨胀，详见 US3b 证据）。运行时 `if (bias) want_hmx = 0` 已撤销，生产默认 bias+32 对齐走 HMX；`wt_exec_force_bias_scalar(int)` 为测试专用钩子（默认 0=生产 HMX；qurt 无 env，显式 API）。
