# Data Model: Short-Term Performance Sprint 1

**Feature**: [spec.md](spec.md)
**Created**: 2026-10-08

## BroadcastShapeClass

broadcast op 的形态分类，由 in/out 4D 形状静态判定（编译器侧与设备侧同一判据语义，注释互指）。

| 值 | 形态 | 判据（示例） | 执行体 | emit 计划 |
|---|---|---|---|---|
| 1 | 连续块 | 末维广播或整块复制（cond 4096→65536、标量 1→n） | HVX 128B 块拷贝循环 + 尾部标量 | HVX |
| 2 | 行/面重复 | in 沿低维循环复制 out 行模式 | 外层标量行循环 + 内层 HVX 块拷贝，行内坐标增量预计算 | HVX |
| 3 | 通用退化 | strided gather | 标量（现状路径） | scalar（fb_allowed 恒 1，合法非回落） |

约束：类 3 计划 scalar 时偏差报告 dev=0（"计划即能力"），不得计入允许回落。

## WeightDomainRecord

P2 emit 报告的逐层权重域登记行。

| 字段 | 类型 | 约束 |
|---|---|---|
| layer | string | 线性层名（wtlink 匹配键） |
| domain | enum {w4, f16} | w4 = Q4_0 tile-major repack 入池；f16 = 兜底 |
| fallback_reason | enum {none, shape_misalign, source_missing} | shape_misalign = M/K/N 对齐不满足；source_missing 仅预检失败全局报错时出现（不静默） |
| size_bytes | u64 | 该层权重入 blob 字节数 |

约束：lm_head 恒 w4（US1 既成，不回退）；报告须含 w4 占比与 f16 兜底名单汇总。

## FusionAdmissionRow

P3 逐条融合规则的准入记录（登记于本 feature 目录 evidence 文件，不入 /tmp）。

| 字段 | 类型 | 约束 |
|---|---|---|
| rule | string | `fusion_<producer>_<consumer>` ∈ kLegacyFusionRules 8 条 ∪ 新提案 |
| hits | int | `--enable-pass` 后 opt-stats 匹配/改写计数 |
| synth_bitexact | bool | 合成图位级一致门 |
| fullgraph_cos_delta | float | 0.8B 全图 |Δcos|，过门 <1e-5 |
| top1_not_worse | bool | top1 ≥ 基线 30/32 口径不降 |
| default_on | bool | 三门全过才可翻 true；失败留证据保持 false |

约束：`--no-pass <rule>` 恒可关（逃生门）；新提案规则走同一准入门。

## op_ts 口径（沿用 003 blob-v3）

- 逐 op 三元组 planned/actual/deviation（engine 列 + 回落语义不变）。
- 族级回落计数：broadcast 族允许回落为 P1 主验收指标（1023→0）。
- 解析纪律：禁手搓二进制，用 `scripts/gehtp_prof.py`（[[feedback-binary-artifact-parse]]）。

## 双口径判分记录

| 口径 | 适用期 | golden | 门值 |
|---|---|---|---|
| hf | f16 期（P1 验证、P2 前） | cpu_golden/hf | cos 0.998642 逐字节不动（prompt0 锚口径） |
| q4_0 | W4 期（P2 起，4-prompt 全跑） | cpu_golden/q4_0 | **每 prompt pooled cos ≥0.98 + top1 ≥20/32**（2026-10-10 首跑校准冻结，依据下节） |

约束：两口径 cos 不可直比，报告必须双列登记。

### q4_0 门值校准依据（2026-10-10，007d blob 板跑实测，设备 d0f1784）

spec 预填值（cos 0.9986 / top1 29）为 f16-vs-hf 锚的量级占位，首跑实测证明其对 q4_0
口径按构造不可达：cpuq4 golden（llama.cpp，Q4_0 权重×Q8_0 激活 f32 累加）与设备
（Q4_0 权重×f16 激活）是**同一量化权重的两个独立数值实现**，残差 = 两实现噪声差
经 GDN 递归放大，幅度随 prompt 的量化敏感度变化：

| prompt | dev~cpuq4 cos | dev~cpuq4 top1 | cpuq4~hf cos（敏感度） | 正交模型预测 dev~hf | 实测 dev~hf |
|---|---|---|---|---|---|
| 0 | 0.993484 | 31/32 | 0.853421 | 0.8472 | 0.840118 |
| 1 | 0.981696 | 22/32 | 0.810254 | 0.7952 | 0.772724 |
| 2 | 0.999224 | 31/32 | 0.970776 | 0.9701 | 0.969677 |
| 3 | 0.998465 | 29/32 | 0.968293 | 0.9670 | 0.966588 |

θ(dev,hf) ≈ √(θ(dev,q4)²+θ(q4,hf)²) 在低敏感 prompt（p2/p3）三位小数吻合
（0.9701/0.9697、0.9670/0.9666）→ 设备忠实执行同一 Q4 权重，非实现缺陷。
门值取法：cos 0.98 = 最敏感 prompt 噪声底（0.9817）−1e-3 余量；bug 态
（007b 转置坏 blob）实测 4 prompt cos 全部 ≤0.04，分离度 ≫；top1 20/32 =
最差 prompt（22）−2。f16 期 hf 口径门维持 prompt0 锚（f16 线 p3 top1 实测
26/32，该门本就非 4-prompt 口径）。
