# T018 — US3 融合 pass 准入普查 (2026-10-10, host 编译器)

## 结论: NO-GO — 8 条 legacy 融合规则在 0.8B 全图匹配 0, 无准入可谈

8 条 `fusion_*` 规则的 matcher 用 **op 类型名短名** (`optimization_passes.cpp`
kLegacyFusionRules), 但 0.8B 的 QNN IR 用 **长名**。逐条对照如下:

| rule (producer→consumer) | pass 名 | producer 命中? | consumer 命中? | hits |
|---|---|---|---|---|
| Conv→Relu | fusion_conv_relu | ✗ (IR 是 DepthWiseConv2d) | ✗ (IR 是 ElementWiseNeuron) | 0 |
| Conv→Clamp | fusion_conv_clamp | ✗ | ✗ | 0 |
| MatMul→Add | fusion_matmul_add | ✓ (MatMul 在 IR) | ✗ (IR 是 Eltwise_Binary) | 0 |
| MatMul→Gelu | fusion_matmul_gelu | ✓ | ✗ | 0 |
| MatMul→Relu | fusion_matmul_relu | ✓ | ✗ | 0 |
| Add→Relu | fusion_add_relu | ✗ | ✗ | 0 |
| Add→Sigmoid | fusion_add_sigmoid | ✗ | ✗ | 0 |
| Dense→Add | fusion_dense_add | ✗ (IR 是 FullyConnected) | ✗ | 0 |

0.8B 图 20 个 distinct op 类型名 (net.json `graph` 实测):
`Cast Concat CumulativeSum DepthWiseConv2d ElementWiseNeuron Eltwise_Binary
Eltwise_Ternary Eltwise_Unary FullyConnected Gather MatMul Pad Reduce Reshape
RmsNorm ScatterNd Softmax Split StridedSlice Transpose`

无一条 rule 的 (producer, consumer) 短名对同时出现 → **8/8 匹配 0**。
即 `--enable-pass fusion_*` 全开跑 0.8B 编译, matched/rewritten 全 0,
`.bin` 与全关逐字节相同 (无回归也无收益, 与 [[gehtp-pass-registry-refactor]]
已证结论一致)。

## 为何不可"修短名再放行"

即使把规则名对齐真实 IR (Conv2d→DepthWiseConv2d 等), 激活融合的池子也
不成立: [[gehtp-006-activation-fusion-census]] 实测激活池 224 op / 1.02%,
上界 1.46% < 3% 门; 终态 top3 = MATMUL 29.3% / BROADCAST 24.8% / SLICE 9.7%。
融合不触及这三大热点, op 数 14188→≤9000 参照门靠融合**无法达成**。

## FusionAdmissionRow 归档

| rule | hits | synth_bitexact | fullgraph_cos_delta | top1_not_worse | default_on |
|---|---|---|---|---|---|
| fusion_conv_relu | 0 | N/A | N/A (0 命中无图变) | N/A | NO |
| fusion_conv_clamp | 0 | N/A | N/A | N/A | NO |
| fusion_matmul_add | 0 | N/A | N/A | N/A | NO |
| fusion_matmul_gelu | 0 | N/A | N/A | N/A | NO |
| fusion_matmul_relu | 0 | N/A | N/A | N/A | NO |
| fusion_add_relu | 0 | N/A | N/A | N/A | NO |
| fusion_add_sigmoid | 0 | N/A | N/A | N/A | NO |
| fusion_dense_add | 0 | N/A | N/A | N/A | NO |

`default_on=NO` 全表: 无规则过准入门 → US3 无默认集放行 (FR-005/FR-007 不触发)。

## 下游影响

- **T019 (合成图位级门) / T020 (全图数值门) / T021 (翻默认) / T023 (P1 复跑)**:
  因 0 命中 → 无图变 → 全图 cos Δ=0、P1 broadcast 回落仍 0 天然成立, 无
  "默认集放行"可验; 视为**条件跳过** (NO-GO 阻塞, 非失败)。
- **T022 (新规则提案)**: 条件触发 (op 数 >9000), 但 006 普查已证激活融合
  上界 1.46% < 3% 门, 新激活融合规则亦无法达 ≤9000; 见 006 归档。

## 复现

```
python3 -c "…enumerate net.json graph node 'type'…"   # 20 distinct 类型名
# 或全量编译: build_host/hnnx_compile --net-json … --opt-stats
#             --enable-pass fusion_conv_relu … (8 条) → matched/rewritten 全 0
```

资产: 类型名枚举脚本内联; net.json =
`/disk1/gehtp/.scratch_a3/qwen35_08b_v2.wtop.work/qwen35_08b_v2_net.json`。
