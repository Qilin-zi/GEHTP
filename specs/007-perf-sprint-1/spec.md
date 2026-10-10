# Feature Specification: Short-Term Performance Sprint 1（broadcast HVX / W4A16 全图 / 融合准入）

**Created**: 2026-10-08
**Status**: Draft
**Input**: 承接 006 目录原名冲刺线的立项研究（2026-10-08 10:39 版 plan/research，撞号裁决后改号 007；006 目录现为激活融合线容器）。目标：0.8B prefill **35.31s → ≤20s**，003 契约全程不破。

## Background *(依据与盘子)*

当前基线锚（库 a42720a5→d33e31b3 系列）：003 v4 = wall **35.31s** / cos 0.998642 / top1 30/32 / **14188 op** / 允许回落 **1023 次**。三个正交方向的盘子均已代码实查：

| 方向 | 现状实查 | 预期收益 |
|---|---|---|
| ① broadcast 执行体 | `exec_broadcast`（oplist_exec.c:2080）标量**每元素除法/取模重算坐标**，1622 op × 4.8ms = **7.84s**；emit 按 n%32 计划 HVX 但无 HVX 内核 → 1023 次允许回落（计划与能力不一致） | **-7.3s** 量级 |
| ② W4A16 全图 | `--gguf/--match` 通道完整（wtop_emit.cpp:263+ 读 TSV 匹配表，命中权重 Q4_0 tile-major repack → HMX W4A16）但**从未在 0.8B 启用**（v4 全 f16 是当年 GGUF 源缺失残留，非能力缺口）；match 表由 `scripts/gguf_wtlink.py` 生成，非外部资产；lm_head 已设备验证 51.7× | 权重 1.6GB→**~0.5GB** + W4 HMX 吞吐 |
| ③ 融合规则准入 | `kLegacyFusionRules` 8 条已注册为 `fusion_*` pass 默认关，`--enable-pass` 逐条可控，机制零开发 | per-op 开销（~1.4s 记录 + 逐 op fence）压减，op 数 14188→≤9000 为参照门 |

约束：003 契约不破（偏差报告不变量 / carve footer / ver1 legacy 逐字节）；精度锚不回归（f16 期 0.998642，W4 期切 q4_0 golden 口径判分）。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - broadcast HVX 执行体 + 计划对齐 (Priority: P1)

0.8B 图里 SELECT/binary 广播输入物化产生 1622 个 broadcast op，运行时全部标量每元素坐标除法。本故事补 HVX 执行体并让 emit 计划与内核能力一致：按 broadcast 结构分派三种形态（连续块循环 / 行面重复 / 通用退化），全部消除 per-element 除法；偏差报告里 broadcast 族回落归零。

**Why this priority**: 单方向 -7.3s 是三块最大且风险最低（无格式变动、无精度重锚）。

**Independent Test**: v17/prefill 在新库下跑通：broadcast 族允许回落 1023→0、族耗时 ≤1s、全链 cos 0.998642 逐字节不动（纯索引重排无浮点语义变化——连续块拷贝不改字节）。

**Acceptance Scenarios**:

1. **Given** 末维广播或整块复制形（cond 4096→65536、标量 1→n 类），**When** 执行，**Then** 走 HVX 128B 块拷贝循环（坐标志外按块推进），尾部标量，计划 HVX 实跑 HVX（dev=0）。
2. **Given** 行/面重复形（in 沿低维循环复制 out 行模式），**When** 执行，**Then** 外层标量行循环 + 内层 HVX 块拷贝，行内坐标增量预计算。
3. **Given** 通用退化形（strided gather），**When** emit 计划，**Then** 计划 scalar（合法非回落，偏差报告不出该项）——"计划即能力"。
4. **Given** 编译器与设备两侧，**When** 形态判定，**Then** 以同一形状判定函数为准（两处注释互指，防漂移）。

---

### User Story 2 - W4A16 权重域推广全部线性层 (Priority: P2)

`--gguf` Q4_0 直读通道从"仅 lm_head"推广到全图线性层：wtlink 全量匹配 → emit 逐层权重域报告 + 显式 f16 兜底名单 → 板跑切 q4_0 golden 判分。

**Why this priority**: 收益叠加内存与吞吐双线，但跨 wtlink/emit/引擎/判分四层且精度重锚，复杂度最高，排在 P1 落地后。

**Independent Test**: `cpu_golden/q4_0` 双门（judge_logits.py 现成）：cos ≥ 0.9986 同量级 + top1 ≥ 29/32（门值以首跑实测校准后冻结写入 data-model）；emit 报告 w4 占比与兜底名单落盘。

**Acceptance Scenarios**:

1. **Given** /disk2 GGUF 源在位（编译期存在性预检兜底），**When** `gguf_wtlink.py` 生成全量 match 表并仅追加 `--gguf/--match` 编译，**Then** 命中权重 Q4_0 入池、`const_data_size` 折半、判定走 `OP_MATMUL_W4A16` HMX。
2. **Given** M/K/N 对齐约束不满足的 FC，**When** emit，**Then** 显式 f16 兜底并列名（显式降级名单，非静默）。
3. **Given** W4 期产物，**When** 判分，**Then** q4_0 golden 口径出报告（与 f16 期 hf 口径双列登记，防误读——不同参考不可直比）。
4. **Given** lm_head（已走 W4），**When** 本故事落地，**Then** 不回退（51.7× 保持）。

---

### User Story 3 - 融合 pass 准入制放行 (Priority: P3)

8 条 legacy 融合规则（Conv+Relu/Clamp、MatMul+Add/Gelu/Relu、Add+Relu/Sigmoid、Dense+Add）先统计后放行：逐条 `--enable-pass` 跑命中数 → 合成图位级一致 + 0.8B 全图值差门 → 过门进默认集。命中不足 ≤9000 参照门时按 op_ts 热点评估补规则（GDN 门控链候选，走同一准入门）——**条件项非承诺项**。

**Why this priority**: 机制现成（零开发），工作量为逐条数值门与统计；放行顺序放在 P1 之后（P1 消 broadcast 回落，P3 的热点统计才干净）。

**Independent Test**: 每条规则独立三图验证（合成位级 / 全图 cos Δ<1e-5 + top1 不降 / `--no-pass` 可关）；过门规则 `p.enabled` 翻 true 后 ctest 不回归。

**Acceptance Scenarios**:

1. **Given** 规则 `fusion_matmul_add`（预期主力：残差/偏置链），**When** 启用编译，**Then** opt-stats 报告命中数且数值门通过才入默认集。
2. **Given** 任一规则数值门失败，**When** 处置，**Then** 保持默认关并在准入表登记失败证据，不放行。
3. **Given** 8 条全开后 op 数仍 >9000，**When** 评估，**Then** 按 op_ts 热点链提案 1-2 条新规则走同一准入门，不阻塞 P1/P2 收口。

### Edge Cases

- **P1↔P3 交互**：融合吸收 broadcast（作为融合核输入物化）→ P1 验证须在 P3 放行后复跑。
- **P2 判分口径切换**：f16→q4_0 golden 的 cos 不可直接比较（不同参考），报告双列登记。
- **P2 源缺失**：/disk2 清理史教训——编译期 GGUF 源存在性预检，缺失即显式失败而非静默全 f16。
- **P2 形状边界**：W4A16 HMX 的 M/K/N 对齐约束（沿 lm_head 分块条件）不满足的 FC 必须 emit f16 兜底并列名。
- **P3 首版归因**：全默认开会让数值风险一次性引爆不可归因——逐条准入是硬纪律。
- **P1 类 3 语义**：通用退化形计划 scalar = 合法（非回落、dev=0），不得为凑"零回落"指标硬塞 HVX gather。
- **共享树/板**：oplist_exec.c 与 003 T007/004 reconcile 同文件不同段，改前 `git status`；板资产 `/data/local/tmp/ggml08b_003/` 属 ggml 对比线勿混。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 设备端 MUST 提供 broadcast 三形态执行体（连续块循环 / 行面重复 / 通用退化标量兜底），形态 1/2 消除 per-element 坐标除法；形态判据与编译器侧同一函数语义（注释互指）。
- **FR-002**: emit `assign_engine` 对 broadcast MUST 按形态分派计划（类 1/2 → HVX；类 3 → scalar，fb_allowed 沿用恒 1）；落地后偏差报告中 broadcast 族允许回落 MUST 为 0。
- **FR-003**: W4 供给链 MUST 启用：`gguf_wtlink.py` 全量匹配表 + 编译期 GGUF 源存在性预检；emit 报告 MUST 输出逐层权重域（w4/f16）与显式 f16 兜底名单（显式降级，非静默）。
- **FR-004**: W4 精度门 MUST 切 q4_0 golden 口径：cos ≥ 0.9986 同量级 + top1 ≥ 29/32，门值首跑校准后冻结；f16 期中间态（P1 单独验证）仍用 hf 口径 0.9986 锚。
- **FR-005**: 融合规则 MUST 走准入制：先 `--enable-pass` 统计命中，再合成图位级一致 + 0.8B 全图 cos Δ<1e-5 + top1 不降，过门才进默认集；`--no-pass fusion_x` 逃生门恒可用。
- **FR-006**: 003 契约 MUST 不破：偏差报告不变量（planned/actual/deviation 语义）、carve footer 对拍、ver1 legacy 逐字节、blob-v3 五场景验收全绿。
- **FR-007**: 所有改动 MUST 幂等可关：P3 规则默认集变更仅 `enabled` 翻转（stats_line 计数保留）；P1/P2 无 CLI 语义变化（P2 为编译配方显式追加）。
- **FR-008**: 共享文件改动 MUST 遵守跨会话协调协议（git status 前置 / 精确暂存 / 板上私有名与互斥）。

### Key Entities

- **BroadcastShapeClass**: 形态 1 连续块 / 2 行面重复 / 3 通用退化，由 in/out 4D 形状静态判定；决定执行体与计划引擎。
- **WeightDomainRecord**: 层名 × 权重域（w4/f16）× 兜底原因（形状约束/源缺失）。
- **FusionAdmissionRow**: 规则名 × 命中数 × 合成图位级 × 全图 cos Δ × top1 × 默认集状态。
- **op_ts 口径**: planned/actual/deviation 逐 op 三元组 + 族级回落计数（P1 验收主指标）。
- **双口径判分**: hf（f16 期）与 q4_0（W4 期）golden 门值冻结记录。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: broadcast 族允许回落 1023 → **0**；broadcast 族耗时 7.84s → **≤1s**。
- **SC-002**: 全链 wall：P1 单独 35.31s → **≤28s**；三方向叠加 → **≤20s**（冲刺总门）。
- **SC-003**: W4 启用后权重域内存 1.6GB → **≤0.6GB**；f16 兜底名单显式；lm_head 51.7× 不回退。
- **SC-004**: q4_0 golden 门 cos ≥ 0.9986 同量级 + top1 ≥ 29/32（首跑校准冻结）。
- **SC-005**: 每条放行融合规则：合成图位级一致 + 全图 cos Δ<1e-5 + top1 不降；未过门规则默认关且留证据。
- **SC-006**: 003 blob-v3 五场景验收 + ctest 全绿不回归。
- **SC-007**: f16 期中间态（P1 单独验证）cos 0.998642 逐字节不动。

## Assumptions

- 本 spec 承接 2026-10-08 10:39 版原 006 冲刺立项研究（plan/research 复原入本目录，出处与裁决记录见跨会话协调表）；006 目录名 perf-sprint-1 现为激活融合线容器，名不副实留待两会话协调改名。
- /disk2/aiDAPTIV/Qwen35-0.8B.gguf 可用（存在性预检兜底）；`cpu_golden/{hf,q4_0,text}` 双 golden 现成，无需新建 host W4 参考链（host 链无 W4 语义，历史判据即此分工）。
- 补融合规则为条件项（命中不足时才评估）；GDN 主体链（Slice/Gather/broadcast 物化）不在 8 条规则集内属已知。
- 不做：HVX gather 通用内核（形态 3 罕见、复杂度不成比例）、自研 PTQ 量化、host 链 W4 语义。
- 设备 d0f1784（指纹 d33e31b3）可用；共享树/板纪律沿 003/004 既定。
