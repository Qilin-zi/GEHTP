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

**硬约束**: 32 对齐形状（M%32==0 ∧ K%32==0 ∧ N%32==0）的 MatMul **不得**带标记发射 36 号（3b 前会被强制标量路径）——pass 侧 HMX 门保证，发射侧遇到违禁组合必须硬报错（对齐既有 "W4A16+bias" 硬错风格）。

## 设备契约（仅 3b，P3b 前不生效）

OP_MATMUL_F16_BIAS(36) 的 HMX 路径：bias 在 VTCM 内逐列加（f32 累加器），与标量路径数值等价（cos ≥ 0.9999、逐列 ≤ 2 ULP f16）。3b 验证通过前，设备端 `if (bias) want_hmx = 0` 保持不变。
