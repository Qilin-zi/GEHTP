---
description: "Task list for feature implementation"
---

# Tasks: Short-Term Performance Sprint 1（broadcast HVX / W4A16 全图 / 融合准入）

**Input**: Design documents from `/specs/007-perf-sprint-1/`（worktree `/disk1/gehtp-wt/007-perf-sprint-1`，分支 `007-perf-sprint-1` off `7deb63b`）

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/contracts.md, quickstart.md

**Organization**: 按用户故事分组（US1 broadcast HVX / US2 W4A16 全图 / US3 融合准入），各自独立可验收；MVP = US1。

**Tests**: 验证即任务内的"门"（evidence 文件 + 板跑判分），不单列 TDD 测试任务；契约测试除外（T009，plan 明确要求）。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- 路径均为 repo 相对路径；设备操作带 007_ 私有名纪律（d0f1784：getenforce 预检、pgrep run_main 互斥、推前 df）

---

## Phase 1: Setup（worktree 构建引导）

**Purpose**: 让 007 worktree 具备编译/上板能力并锚定基线证据

- [x] T001 容器 `gehtp-toolchain:24.04` 内 cmake configure+build `/disk1/gehtp-wt/007-perf-sprint-1/compiler` → `build_linux`（kernels 先 source env.sh，参照 [[gehtp-build-env]]）
- [x] T002 [P] `kernels/` 基线全量构建：`cd kernels && ./build_libs.sh all`（验证工具链可用；本任务不推板，仅本地产物自洽）
- [x] T003 基线锚采集：现有 003 v4 产物跑 `scripts/gehtp_prof.py` 取 broadcast 族回落 1023 / 族耗时 7.84s / wall 35.31s / cos 0.998642，登记 `specs/007-perf-sprint-1/evidence/baseline.md`（判据入库不入 /tmp）

**Checkpoint**: 构建可用、基线数字落盘

---

## Phase 2: Foundational（阻塞全部故事的前置分析）

**Purpose**: 形态分布定量，决定 US1 HVX 覆盖面与 US3 热点排序

- [x] T004 [P] broadcast 形态普查：对 0.8B 图全部 1622 个 broadcast op 按 data-model `BroadcastShapeClass`（形态1 连续块/形态2 行面重复/形态3 strided gather）分类统计，产出分布表与各形态 wall 占比 → `specs/007-perf-sprint-1/evidence/broadcast_census.md`（host 工具或脚本分析图 dump，只读不改码）

**Checkpoint**: 形态 1/2 覆盖率已知（-7.3s 估计的实证）；US1/US2/US3 可开工

---

## Phase 3: User Story 1 - broadcast HVX 执行体 + 计划对齐 (Priority: P1) 🎯 MVP

**Goal**: 消灭 1023 次允许回落与 per-element 坐标除法，broadcast 族 7.84s→≤1s，wall ≤28s，cos 逐字节不动

**Independent Test**: v17/prefill 板跑——回落 1023→0、族耗时 ≤1s、judge hf 口径 cos 0.998642 逐字节不变（纯索引重排）

### Implementation for User Story 1

- [x] T005 [P] [US1] 设备侧形态判定 + 三形态分派骨架：`kernels/src/runtime/oplist_exec.c` exec_broadcast 段（:2080 一带）——判定语义 = data-model BroadcastShapeClass 表，注释与 emit 侧互指（FR-001，防两侧漂移）；形态 3 暂走现状标量路径
- [x] T006 [US1] 形态 1 执行体：连续块 HVX 128B 块拷贝循环（v60，参照 `kernels/src/runtime/hvx_binary.c` 范式：HVX 循环 + 尾部标量），坐标志外仅按块推进，零 per-element 除法（depends T005）
- [x] T007 [US1] 形态 2 执行体：外层标量行循环 + 内层 HVX 块拷贝，行内坐标增量预计算（depends T006，同文件同段顺序改）
- [x] T008 [P] [US1] 编译器侧形态判定 + `assign_engine` 分派：类 1/2→HVX、类 3→scalar（fb_allowed 恒 1），`compiler/tools/wtop_emit.cpp` + `compiler/tools/wtop_ops/wtop_ops.hpp`；与 T005 同一判据语义、注释互指（FR-002）
- [x] T009 [US1] 契约测试：`compiler/tests/test_wtop_emit.cpp` 增三类形态的 planned engine 断言（类 3 计划 scalar = 合法非回落 dev=0）（depends T008）
- [x] T010 [US1] 全量验证构建 + 板跑：compiler ctest 47/47 → `build_libs.sh all`（多文件勿用 one，防陈旧 arity）→ d0f1784 推 `007_` 私有名资产（推前 df）跑 v17/prefill（depends T007, T009）
- [x] T011 [US1] P1 门验收 → `specs/007-perf-sprint-1/evidence/p1_gate.md`：回落 1023→0、broadcast 族耗时 ≤1s（SC-001）、wall ≤28s（SC-002 阶段门）、cos 0.998642 逐字节不动（SC-007）、类 3 不计入回落（偏差报告不变量）（depends T010）

**Checkpoint**: MVP 达成，可独立交付

---

## Phase 4: User Story 2 - W4A16 权重域推广全部线性层 (Priority: P2)

**Goal**: 权重 1.6GB→≤0.6GB + W4 HMX 吞吐；判分切 q4_0 口径双列登记；lm_head 51.7× 不回退

**Independent Test**: emit 权重域报告（w4 占比 + f16 兜底名单）+ `cpu_golden/q4_0` 双门 cos ≥0.9986 同量级 + top1 ≥29/32（首跑校准冻结）

### Implementation for User Story 2

- [ ] T012 [P] [US2] 全量匹配表 + 源预检：`scripts/gguf_wtlink.py` 生成 0.8B 全量 match.tsv（`--gguf /disk2/aiDAPTIV/Qwen35-0.8B.gguf --out` repo 内路径），并加编译期 GGUF 源存在性预检——缺失显式失败，不静默全 f16（FR-003；/disk2 清理史教训）
- [ ] T013 [US2] emit 权重域报告：`compiler/tools/wtop_emit.cpp` 输出逐层 WeightDomainRecord（字段 verbatim：layer / domain∈{w4,f16} / fallback_reason∈{none,shape_misalign,source_missing} / size_bytes）+ w4 占比汇总 + f16 兜底名单（注意与 T008 同文件，先 git status 确认 US1 已提交）
- [ ] T014 [US2] 形状边界兜底：M/K/N 对齐（沿 US1 lm_head 分块条件）不满足的 FC 显式 f16 并列名，`compiler/tools/wtop_ops/op_matmul.cpp`（depends T012, T013）
- [ ] T015 [US2] 编译 0.8B：v4 配方仅追加 `--gguf/--match` → `const_data_size` 折半核验 + 报告落盘 `evidence/w4_emit_report.md`（depends T014）
- [ ] T016 [US2] 板跑 + q4_0 判分重锚：judge_logits.py 切 `cpu_golden/q4_0`，双门 cos ≥0.9986 同量级 + top1 ≥29/32，首跑校准后**冻结门值回填 data-model.md 双口径表**；与 hf 口径双列登记（FR-004，两口径 cos 不可直比）（depends T015）
- [ ] T017 [US2] P2 门验收 → `evidence/p2_gate.md`：权重 ≤0.6GB（SC-003）、lm_head 51.7× 不回退、f16 兜底名单显式（depends T016）

**Checkpoint**: P1+P2 双独立交付在位

---

## Phase 5: User Story 3 - 融合 pass 准入制放行 (Priority: P3)

**Goal**: 8 条 `fusion_*` 规则先统计后放行，过门进默认集；op 数 14188→≤9000 为参照门；补规则为条件项

**Independent Test**: 每规则三门（合成图位级 / 全图 cos Δ<1e-5 + top1 不降 / `--no-pass` 可关）+ 准入表落盘

### Implementation for User Story 3

- [ ] T018 [US3] 逐条普查：8 条规则各 `--enable-pass fusion_<p>_<c> --opt-stats` 跑 0.8B 编译，命中数登记 `evidence/fusion_admission.md`（FusionAdmissionRow 字段 verbatim：rule / hits / synth_bitexact / fullgraph_cos_delta / top1_not_worse / default_on）（**依赖 T011**：P1 消回落后热点统计才干净）
- [ ] T019 [P] [US3] 合成图位级门 harness：每规则合成正例图，全开 vs 全关 host 输出位级比对（md5），`compiler/tests/` 或 host 工具
- [ ] T020 [US3] 全图数值门：过位级门的规则跑 0.8B 全图，|Δcos|<1e-5 + top1 不降（hf 口径，f16 期）（depends T018, T019）
- [ ] T021 [US3] 过门规则翻默认：`compiler/src/opt/pass_fusion.cpp` 注册处 `enabled` 翻 true（仅此位，stats_line 计数保留），`--no-pass fusion_x` 逃生门验证 + ctest 47/47 不回归（FR-005/FR-007）（depends T020）
- [ ] T022 [US3] 条件项：8 条全开后 op 数仍 >9000 → 按 `evidence/broadcast_census.md` + op_ts 热点提案 1-2 条新规则（GDN 门控链候选），走同一准入门；不阻塞 P1/P2 收口（depends T021，条件触发）
- [ ] T023 [US3] P1 复跑：P3 放行后重验 broadcast 族回落仍为 0（融合吸收 broadcast 物化的交互，spec Edge Cases）（depends T021）

**Checkpoint**: 三故事全部独立在位

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: 冲刺总门 + 003 契约回归 + 协调收尾

- [ ] T024 冲刺总门：三方向叠加全链 wall ≤20s（SC-002 总门）+ op 数对照 ≤9000 参照 → `evidence/sprint_gate.md`（depends T011, T017, T023）
- [ ] T025 [P] 003 契约回归：blob-v3 五场景验收（devA/devB 回落取证、carve footer、ver1 legacy 逐字节）+ ctest 47/47 终局（SC-006，FR-006）
- [ ] T026 quickstart.md 全流程验证跑一遍（含三条逃生门：emit 回旧判据 / 去 --gguf--match / --no-pass）
- [ ] T027 [P] 协调收尾：跨会话协调表登记（worktree/分支/设备库指纹变化如重推）/ 结论写回 memory；006 目录改名协调（specs/006-perf-sprint-1 名不副实，与归档的 006 会话或用户确认）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1**: 无依赖，T001/T002 可并行
- **Phase 2**: 依赖 T001（host 工具）；T004 与 T003 可并行
- **US1**: T005 ∥ T008（设备/编译器两侧不同文件）→ T006 → T007 → T010；T008 → T009 → T010；T010 → T011
- **US2**: T012 ∥ T013（脚本/emit 不同文件；T013 与 T008 同文件 wtop_emit.cpp——**必须等 US1 提交后再动**）→ T014 → T015 → T016 → T017
- **US3**: T018 **阻塞于 T011**（热点统计要 P1 后的干净基线）；T019 与 T018 并行 → T020 → T021 → T022（条件）/ T023
- **Phase 6**: T024 阻塞于 T011+T017+T023；T025 随时可跑；T026 阻塞于 T024；T027 最后

### User Story Dependencies

- **US1 (P1)**: 无跨故事依赖 — MVP
- **US2 (P2)**: 代码面与 US1 独立，仅 wtop_emit.cpp 文件冲突需顺序（US1 先）；判分口径切换自成体系
- **US3 (P3)**: 统计起点依赖 US1 收口（T018←T011）；放行后须回验 US1（T023）

### Parallel Opportunities

- T001 ∥ T002；T003 ∥ T004
- US1 内 T005（设备侧）∥ T008（编译器侧）
- US2 内 T012 ∥ T013（T008 完成后）
- US3 内 T018 ∥ T019
- T025 ∥ T024

---

## Parallel Example: User Story 1

```bash
# 两侧同时开工（不同 repo 子树）：
Task: "T005 设备侧形态判定+分派骨架 (kernels/src/runtime/oplist_exec.c)"
Task: "T008 编译器侧形态判定+assign_engine 分派 (compiler/tools/wtop_emit.cpp)"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 构建 + 基线 → Phase 2 形态普查
2. US1 全链（T005-T011）→ 板跑四门（回落 0 / 族 ≤1s / wall ≤28s / cos 位级）
3. **STOP and VALIDATE**：MVP 独立交付，-7.3s 到手

### Incremental Delivery

1. US1 → MVP 交付（最大单点 24.8% wall，006 census 已实证）
2. US2 → q4_0 重锚交付（判分口径切换只在本期）
3. US3 → 默认集翻转 + 条件补规则
4. Phase 6 总门 ≤20s + 003 契约回归 + 协调收尾

---

## Notes

- 共享纪律（FR-008）：`oplist_exec.c` exec_broadcast 段与 003 T007/004 reconcile 段同文件不同段，动手前 `git status`；板 `/data/local/tmp/ggml08b_003/` 属 ggml 对比线勿混
- 设备库重编用 `all`（`one` 有 .signed.so 跳过重建 → 陈旧 arity 陷阱）；库指纹变化须登记协调表
- 证据全部入 `specs/007-perf-sprint-1/evidence/`，不入 /tmp
- 每 task 或逻辑组一 commit（worktree 内）；回合主线（merge 回 task/prof-wp）时须与 003 T007/004 reconcile 段对表
