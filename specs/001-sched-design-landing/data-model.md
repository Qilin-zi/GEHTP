# Phase 1 Data Model: 调度设计原则落地

> 2026-09-17。实体均为编译期/报告期记录，无设备侧格式改动。

## E1 访问类型分类（AccessClass）

| 字段 | 类型 | 说明 |
|---|---|---|
| class | enum {VECTOR, HMX, DMA, SCALAR, UNKNOWN} | 操作数被访问的方式 |
| scope | (opcode, operand_idx) | 静态表的键 |

规则（作用于 alloc 请求 = 张量）：
- 任一消费者呈 SCALAR 或 UNKNOWN → 张量 **DDR-only**
- 全部消费者 ∈ {VECTOR, HMX, DMA} → 可驻留 VTCM（仍受预算约束）
- 无消费者（图输出/死代码）→ 保守 DDR

## E2 内存规划条目（MemPlan entry 扩展）

既有（TAG_MEM_PLAN）：`{op_id u32, out_idx u32, offset u32, size u32, flags u32}`（flags bit0=in_vtcm）。
**本特性不改该记录**。新增的只是 host 侧可审计日志行（不进 blob）：

| 字段 | 说明 |
|---|---|
| op_id, out_idx | 定位 |
| access_verdict | "vtcm-ok" / "ddr-scalar" / "ddr-unknown" / "ddr-output" |
| consumer_evidence | 消费者 (op_id, opcode, operand_idx, class) 列表 |

产出位置：编译期 `--audit` 日志（stderr/文件），不进产物。

## E3 manifest 可选字段（dompath_estimate）

`*.wtop.manifest.json` 新增可选键（旧消费方不认识则忽略，向后兼容）：

```json
{
  "dompath_estimate": {
    "cycles": 123456,
    "source": "table",
    "per_op_table_version": 1,
    "critical_chain_len": 37
  }
}
```

| 字段 | 约束 |
|---|---|
| cycles | u64；初版为估计表求和 |
| source | "table"（内置估计表）或 "measured"（43_hwinfo/optrace 校准后） |
| per_op_table_version | 单调递增；表内容变化必须 bump |
| critical_chain_len | 关键链 op 数（可观测性） |

不变式（SC-005）：`dompath_estimate.cycles ≤ 同图实测 wall`。违反即估计表 bug。

## E4 单元指派审计记录（unit-assign audit）

编译期 `--audit` 每算子一行：

| 字段 | 说明 |
|---|---|
| op_id / opcode | 定位 |
| unit | "HMX" / "HVX"（= 当前实际派发，不改行为） |
| rule_match | 规则表判定（"hmx-large" / "hvx-narrow" / "no-rule"） |
| warn | 窄输出→HMX 时 "narrow-on-hmx"（警告级，不 fail） |

规则表（初版，阈值参数化 `--unit-rule-config`）：
- K 维 ≥ 512 的 MatMul/Conv 类 → 建议 HMX
- 输出最窄维 ≤ 128 → 建议 HVX（窄输出 readback 主导）
- 其余 → no-rule（保持现状派发）

## E5 性能报告 schema（lint 对象）

报告文件（Markdown 或 txt）中每个性能数字块必须满足：

| 必需标注 | 合法值 |
|---|---|
| category | `真算`(HMX) / `装料`(HVX) / `卸料`(DMA) 之一 |
| metric | `num_dominant_path` / `cycles_used` / `graph-wall` / `retire-interval` 之一 |
| shape | 自由文本（如 `64x64x64`、`H=32 C=256`），非空 |
| scenario | `single` / `sequence` / `graph` 之一 |
| sample | 采样法：`reps2-N median` 或 `ACAC paired` 或显式标注 `single-rep (structure-only)` |

lint 规则详见 [contracts/perf-report-lint.md](contracts/perf-report-lint.md)。
词汇定义单一事实源：`docs/GEHTP_SCHED_HWFACTS.md` §4（lint 只引用不复制）。

## 状态流转

本特性实体均为**静态记录**，无运行期状态机。唯一"流转"是 `dompath_estimate.source`：
`table` →（43_hwinfo 回填 + optrace 校准）→ `measured`，由后续数据演进触发，格式不变。
