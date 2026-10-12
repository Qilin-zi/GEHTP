---
description: "Task list for 静态内存规划收口 (010-static-mem-plan-closure)"
---

# Tasks: 静态内存规划收口（编译器决策全下发 → 执行器纯查表）

**Input**: Design documents from `/specs/010-static-mem-plan-closure/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/blob-v3-extensions.md, quickstart.md

**Tests**: 本特性不采用 TDD；每收口点的验收是「byte-exact 回归 + 黄金对拍」，已在各 phase 的验证任务中固化（见 quickstart.md）。

**Organization**: 任务按 user story 分组。注意 build 顺序与 spec 的 US 编号不同：US4（pin_cap）是 US1（restage）的前置，故排在 US1 之前。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无依赖）
- **[Story]**: 所属 user story（US1-US6）
- 描述含精确文件路径与行号锚点（来自 research 已核实代码位点）

---

## Phase 1: Setup（基线锚定）

**Purpose**: 在动任何代码前锁死回归基线，保证「不回退」可判定

- [ ] T001 跑 `build_examples.sh` 确认 35 例 + 261 门 + W4A16 256³ 65536/65536 全绿，记录 cos 0.998642 / wall 6.31s 基线锚（对照 specs/010-static-mem-plan-closure/spec.md 的 SC-005）

---

## Phase 2: Foundational（共享 blob 注解 plumbing）

**Purpose**: 所有 story 共用的 op_annot 打包/解包 helper

- [ ] T002 [P] 在 kernels/include/oplist_parse.h 新增 `wt_annot_set_reuse(u32*,int)` / `wt_annot_reuse(u32)` 打包/解包 helper（对称 `wt_annot_pack`，操作 bit5，`oplist_parse.h:54-58`），供 US1/US5 共用

**Checkpoint**: 共享注解 helper 就绪，各 story 可并行展开

---

## Phase 3: User Story 4 - pin_cap 编译期计算与下发 (Priority: P1) —— US1 的前置

**Goal**: 编译期真算 pin_cap 并写进 blob，执行器读出替代硬编码 4096

**Independent Test**: 两个不同权重常驻需求的模型编译出不同 pin_cap 进 blob，执行器读 blob 值；关池口径 byte-exact（quickstart 验证 B）

### Implementation

- [ ] T003 [US4] 在 compiler/include/hnnx/ir/graph_prepare.hpp 的 `MemPlan` 结构（`graph_prepare.cpp:704-710`）新增 `pin_cap: u64` 字段
- [ ] T004 [US4] 在 compiler/src/ir/graph_prepare.cpp 的 `vtcm_auto_scan`（`:670`）真算 pin_cap（替换固定 align2048 余量假设），公式 `pool = 16MB − scratch_max − pin_cap`，语义对齐 `wtcache_impl.c:108`（字节、2048 对齐、0 → 默认 12MB）
- [ ] T005 [US4] 在 compiler/tools/wtop_emit.cpp 的 TEMPOFF 槽头写 `pin_cap`（`wtop_emit.cpp:668-697`，头部 `[cap][reserve][pin_cap][n]`，pin_cap 单位字节 2048 对齐）
- [ ] T006 [US4] 在 kernels/src/runtime/oplist_exec.c 的 TEMPOFF 解析（`:2550` 起）读 pin_cap，替代两处 `wtcache_open(wc, 4096)`（`:315, :590`）；`pin_cap == 0` 回退 4096（关池口径 byte-exact）
- [ ] T007 [US4] 验证 B：关池口径 pin_cap=0 回退 4096 且 byte-exact；非零口径日志显示 pin_cap 读自 blob（见 quickstart.md 验证 B）

**Checkpoint**: pin_cap 可配置化完成，US1 可开始

---

## Phase 4: User Story 1 - 权重 pin/restage 时序静态化 (Priority: P1) 🎯 MVP

**Goal**: 编译期决定权重 pin 常驻 + restage 时序，显式下发序列，执行器按序执行、删除无条件 restage 与 PIN/MATMUL 重复搬

**Independent Test**: 「同权重连续多个 matmul」图 pin 一次、零重复搬运，`wt_w3_report` restage 次数 == 编译期规划值（quickstart 验证 D）

### Implementation

- [ ] T008 [US1] 在 compiler/src/ir/graph_prepare.cpp 新建权重常驻规划 pass（复用 `compute_lifetimes`，输出**权重 slot** 的 pin/restage 序列；权重是 const，greedy/cp_solver 均 skip，故为全新规划面，见 research R1）
- [ ] T009 [US1] 在 compiler/tools/wtop_emit.cpp 发射显式 `OP_PIN`(3) + `OP_DMA`(33) 序列（超预算权重换入换出，复用既有 opcode）
- [ ] T010 [US1] 在 kernels/src/runtime/oplist_exec.c 的 `xop_pin`（`:3078-3091`）从「记账 + engine_ready 时又搬一次」改为真正 pin 一次，消除 PIN/MATMUL 各搬一次的重复搬运（D2）
- [ ] T011 [US1] 在 kernels/src/runtime/oplist_exec.c 的 `exec_matmul`（`:657-658`）改「该权重已 pin 且驻留区间覆盖当前 op → 跳过 `cpu_to_vtcm`」，直接引用常驻区
- [ ] T012 [US1] 在 kernels/src/runtime/oplist_exec.c 的 `wt_w3_report` 增加 restage 计数（供 SC-003 与编译期规划值对账）
- [ ] T013 [US1] 验证 D：已常驻权重零重复搬运、restage 次数 == 规划值；超预算图换入换出严格按序列（见 quickstart.md 验证 D）

**Checkpoint**: 权重搬运消除，核心性能价值落地

---

## Phase 5: User Story 2 - W4A16 辅助面身份显式标注 (Priority: P1)

**Goal**: 编译期标注 bias/atbl/otbl 三槽身份，执行器删按 len 猜，顺带修 K≠N 误判

**Independent Test**: K≠N 合成图不再误报 supply slots missing；K=N 256³ bit-exact（quickstart 验证 A）

### Implementation

- [ ] T014 [US2] 在 kernels/include/oplist_parse.h 将 `WT_ARITY_MATMUL` 6 → 9
- [ ] T015 [US2] 在 compiler/tools/wtop_emit.cpp 的 `OP_MATMUL_W4A16` 发射追加 `bias_slot/atbl_slot/otbl_slot` 三个 slot id（arity 6→9，见 contracts 契约 A）
- [ ] T016 [US2] 在 kernels/src/runtime/oplist_exec.c 的 `exec_matmul`（`:562-571`）删按 `len` 反推逻辑，改读 args 三槽 id（`bias = weight_base + slots[bias_slot].offset` 等）；顺带消除 `tbl_b` 用 K 同时匹配 atbl+otbl 的 K≠N 误判
- [ ] T017 [US2] 验证 A：K≠N 合成图不误报 + K=N W4A16 256³ 65536/65536 bit-exact（见 quickstart.md 验证 A）

**Checkpoint**: 正确性 bug 修复，US2 独立完成

---

## Phase 6: User Story 3 - 权重 crouton pack 编译期化 (Priority: P1)

**Goal**: 编译期产出 crouton 布局权重槽，执行器删运行时 pack，Q4_0 内存守恒

**Independent Test**: 执行器零运行时 crouton 重排、输出 byte-exact；Q4_0 权重槽仍 640B/tile 未反量化（quickstart 验证 E）

### Implementation

- [ ] T018 [P] [US3] 在 compiler/tools/weight_layout_pack_f16.cc 新增 f16 权重 crouton 打包（复用 kernels/include/hmx_crouton.h 的 `hmx_crouton_pos`，与运行期参考 `oplist_exec.c:2010-2019` 逐位一致）
- [ ] T019 [P] [US3] 在 compiler/tools/weight_layout_pack_q4.cc 新增 Q4_0 量化态 crouton 打包（640B/tile 重排、**禁反量化成 f16**，与运行期参考 `oplist_exec.c:1982-2008` 的 dequant 输入对齐）
- [ ] T020 [US3] 在 compiler/tools/wtop_emit.cpp 权重槽写布局标签 `FP16_CROUTON | Q4_0_CROUTON`（见 contracts 契约 E、data-model 实体 5）
- [ ] T021 [US3] 在 kernels/src/runtime/oplist_exec.c 的 `exec_matmul_f16_hmx`（`:1970-2020`）删运行时 pack 循环，改直接 mxmem 加载预打包槽（Q4_0 保留 LUT dequant、仅删 permutation）
- [ ] T022 [US3] 在 kernels/src/runtime/oplist_exec.c 的 `exec_conv2d_hmx`（`:878-887`）删运行时 pack 循环，改直接加载预打包槽
- [ ] T023 [US3] 验证 E：零运行时 crouton 重排 + byte-exact + Q4_0 槽仍 640B/tile；重测 pack 耗时按基线归零（对照 T017 LUT 剩余量，非旧账 20.6s，见 quickstart.md 验证 E）

**Checkpoint**: 运行时重排消除，US3 独立完成

---

## Phase 7: User Story 5 - carve 复用决策编译期预判 (Priority: P2)

**Goal**: 编译期预判连续同形 op 可复用 carve，执行器删 shape_same 运行时判断

**Independent Test**: 连续同形 matmul 复用决策来自编译期标注，日志零 shape_same（quickstart 验证 C）

### Implementation

- [ ] T024 [US5] 在 compiler/tools/wtop_emit.cpp 编译期判定「本 op 与上一 matmul M/K/N 相同且 carve footer live_top 不变」，置 `op_annot bit5`（见 contracts 契约 C）
- [ ] T025 [US5] 在 kernels/src/runtime/oplist_exec.c 的 `exec_matmul`（`:589-591`）删 `shape_same` 运行时判断，改读 bit5（`wt_annot_reuse`），bit5=1 跳过 re-carve 分支
- [ ] T026 [US5] 验证 C：可复用 op 跳过 re-carve、carve 复用结果与旧 shape_same 路径一致（见 quickstart.md 验证 C）

**Checkpoint**: 残留运行时判断清完，US5 完成（可延后）

---

## Phase 8: User Story 6 - qwen3-0.8b 端到端验收 (Priority: P1)

**Goal**: 0.8B 全链端到端编译上板，精度与性能双门不回退

**Independent Test**: cos ≥ 0.998642 且 wall ≤ 6.31s + 全量回归全绿（quickstart 验证 E2E）

### Implementation

- [ ] T027 [US6] 0.8B 全链编译 + 上板端到端推理，双门 cos ≥ 0.998642、wall ≤ 6.31s（对照 T001 基线锚）
- [ ] T028 [US6] 全量回归 `build_examples.sh` 35 例 + 261 门 + W4A16 256³ 全绿（SC-002）

**Checkpoint**: 整条收口线最终闸门通过

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: 收尾与跨 story 一致性

- [ ] T029 [P] 更新 docs/GEHTP_TOOLS.md 与相关文档，记录 pin_cap/crouton 布局标签/op_annot bit5 等新契约
- [ ] T030 终检执行日志零 `fallback`、零按尺寸猜、零超规划 restage、零运行时 crouton 重排（SC-001）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，立即开始（基线不绿则后续「不回退」无法判定）
- **Foundational (Phase 2)**: 依赖 Setup；T002 供 US1/US5 共用
- **US4 (Phase 3)**: 依赖 Foundational —— **是 US1 的前置**（pin_cap 决定常驻预算）
- **US1 (Phase 4)**: 依赖 US4 —— 核心 MVP
- **US2 (Phase 5) / US3 (Phase 6) / US5 (Phase 7)**: 均独立，Foundational 后即可并行
- **US6 (Phase 8)**: 依赖 US1-US5 全收口
- **Polish (Phase 9)**: 依赖全部

### User Story Dependencies

- **US4 (pin_cap)**: 无跨 story 依赖，先做
- **US1 (restage)**: 依赖 US4（pin_cap 决定哪些权重放得进常驻区）
- **US2 (辅助面)**: 独立，含 K≠N bug 修复
- **US3 (crouton)**: 独立（f16 路径与 Q4_0 路径内部顺序：f16 先、Q4_0 后）
- **US5 (carve)**: 独立，P2 可延后
- **US6 (E2E)**: 依赖全部收口

### Within Each User Story

- 编译侧（emit/规划）与执行侧（oplist_exec.c）须同 commit 同步（contracts 各契约的「emit/exec 同步」要求）
- 每个验证任务（T007/T013/T017/T023/T026）前须先与旧行为逐字节对齐（byte-exact gate）

### Parallel Opportunities

- T018（f16 pack）与 T019（Q4_0 pack）分属不同文件，可 [P] 并行
- US2、US3、US5 三个 story 在 Foundational 后互不依赖，可由不同开发者并行
- US4→US1 是串行链（性能核心），与 US2/US3/US5 可并行推进

---

## Implementation Strategy

### MVP First（US4 + US1）

1. Phase 1 Setup + Phase 2 Foundational
2. Phase 3 US4（pin_cap 可配置化）—— 小、是前置
3. Phase 4 US1（权重 pin/restage 静态化）—— **核心性能价值**
4. **STOP & VALIDATE**：验证 D（restage 次数对账）+ 双门不回退
5. 这是最小可交付：消除 decode 的 DDR 权重重复搬运

### Incremental Delivery

1. Setup + Foundational → 基线就绪
2. US4 → pin_cap 可配置 → 验证
3. US1 → 权重搬运消除（MVP!）→ 验证
4. US2（bug 修）→ US3（crouton）→ US5（carve）逐个独立验证
5. US6 E2E 全量回归收口

### Parallel Team Strategy

- 开发者 A：US4 → US1（性能核心，串行）
- 开发者 B：US2（辅助面 bug，独立）
- 开发者 C：US3（crouton，独立）
- US5（carve）可最后串行或延后

---

## Notes

- [P] = 不同文件、无依赖；[Story] 映射到 spec 的 user story 便于追溯
- 每个 story 独立可测、可交付增量价值
- byte-exact 是硬 gate：删运行时判断/重排前，emit 侧必须先与旧行为逐字节对齐
- Q4_0 内存守恒是硬约束（US3）：pack 量化态，禁反量化成 f16 存储
- 每完成一个 story 停在 checkpoint 独立验证，再进下一个
- 避免：模糊任务、同文件冲突、破坏 story 独立性的跨 story 依赖
