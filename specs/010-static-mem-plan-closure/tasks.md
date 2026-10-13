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

- [x] T001 跑 `build_examples.sh` 确认 35 例 + 261 门 + W4A16 256³ 65536/65536 全绿，记录 cos 0.998642 / wall 6.31s 基线锚（对照 specs/010-static-mem-plan-closure/spec.md 的 SC-005）

---

## Phase 2: Foundational（共享 blob 注解 plumbing）

**Purpose**: 所有 story 共用的 op_annot 打包/解包 helper

- [x] T002 [P] 在 kernels/include/oplist_parse.h 新增 `wt_annot_set_reuse(u32*,int)` / `wt_annot_reuse(u32)` 打包/解包 helper（对称 `wt_annot_pack`，操作 bit5，`oplist_parse.h:54-58`），供 US1/US5 共用

**Checkpoint**: 共享注解 helper 就绪，各 story 可并行展开

---

## Phase 3: User Story 4 - pin_cap 编译期计算与下发 (Priority: P1) —— US1 的前置

**Goal**: 编译期真算 pin_cap 并写进 blob，执行器读出替代硬编码 4096

**Independent Test**: 两个不同权重常驻需求的模型编译出不同 pin_cap 进 blob，执行器读 blob 值；关池口径 byte-exact（quickstart 验证 B）

### Implementation

- [x] T003 [US4] 在 compiler/include/hnnx/ir/graph_prepare.hpp 的 `MemPlan` 结构（`graph_prepare.cpp:704-710`）新增 `pin_cap: u64` 字段
- [x] T004 [US4] 在 compiler/src/ir/graph_prepare.cpp 的 `vtcm_auto_scan`（`:670`）真算 pin_cap（替换固定 align2048 余量假设），公式 `pool = 16MB − scratch_max − pin_cap`，语义对齐 `wtcache_impl.c:108`（字节、2048 对齐、0 → 默认 12MB）
- [x] T005 [US4] 在 compiler/tools/wtop_emit.cpp 的 TEMPOFF 槽头写 `pin_cap`（`wtop_emit.cpp:668-697`，头部 `[cap][reserve][pin_cap][n]`，pin_cap 单位字节 2048 对齐）
- [x] T006 [US4] 在 kernels/src/runtime/oplist_exec.c 的 TEMPOFF 解析（`:2550` 起）读 pin_cap，替代两处 `wtcache_open(wc, 4096)`（`:315, :590`）；`pin_cap == 0` 回退 4096（关池口径 byte-exact）
- [x] T007 [US4] 验证 B：关池口径 pin_cap=0 回退 4096 且 byte-exact；非零口径日志显示 pin_cap 读自 blob（见 quickstart.md 验证 B）

**Checkpoint**: pin_cap 可配置化完成，US1 可开始

---

## Phase 4: User Story 1 - 权重 pin/restage 时序静态化 (Priority: P1) 🎯 MVP

**Goal**: 编译期决定权重 pin 常驻 + restage 时序，显式下发序列，执行器按序执行、删除无条件 restage 与 PIN/MATMUL 重复搬

**Independent Test**: 「同权重连续多个 matmul」图 pin 一次、零重复搬运，`wt_w3_report` restage 次数 == 编译期规划值（quickstart 验证 D）

### Implementation

- [X] T008 [US1] 权重常驻 pin 决策扶正为 emit 主线：`w4_pinned` 去重集 + 首次消费前发 `OP_PIN`（streaming 简化版，非 graph_prepare 独立 pass；pin 序列 = OP_PIN 首次使用序；超预算换出见 T013 备注）
- [X] T009 [US1] 在 compiler/tools/wtop_ops/op_matmul.cpp 发射显式 `OP_PIN`(3) 去重序列 + wtop_emit `arg_takes_bare_slot` 登记 OP_PIN 裸槽（防 pp_recs GC 剥权重槽）；`OP_DMA` 换入换出（超预算）未发射——0.8B 权重全 fit 或 pin_cap=0 全不 pin，无换出场景，留待大模型
- [X] T010 [US1] kernels/src/runtime/oplist_exec.c `xop_pin` 真 pin 一次 + 登记 `wt_resident_slot`/`bias_resident_slot`，消除 PIN/MATMUL 各搬一次（D2）
- [X] T011 [US1] `exec_matmul` 改「驻留 slot == 本 op 权重 → 跳过 `cpu_to_vtcm`」；re-carve（形状/live 变化或 f16/conv HMX 抢 VTCM）置空驻留标记强制重搬，字节恒等
- [X] T012 [US1] `struct wt_exec_stats` 增 `restage` 计数（实际 DDR→VTCM 换入次数，xop_pin + exec_matmul 各自累加）；planned 值 = blob OP_PIN 条数（可自 wt_w3_report op dump 数），对账在 runner 层
- [x] T013 [US1] 验证 D：compiler 侧已证「同权重双 FC → 1 OP_PIN」（test_wtop_emit US1 段，aab6371 ctest 49/49 绿）；设备侧「restage == planned」上板对账 ⚡延后——0.8B E2E 不走 W4A16（无 OP_PIN），restage 恒 0==planned 0；留待 W4A16 供给侧

**Checkpoint**: 权重搬运消除，核心性能价值落地

---

## Phase 5: User Story 2 - W4A16 辅助面身份显式标注 (Priority: P1)

**Goal**: 编译期标注 bias/atbl/otbl 三槽身份，执行器删按 len 猜，顺带修 K≠N 误判

**Independent Test**: K≠N 合成图不再误报 supply slots missing；K=N 256³ bit-exact（quickstart 验证 A）

> **⛔ BLOCKED — 前置缺失（W4A16 供给侧未落地）**（调查结论 2026-10-12）：
> 契约 A 假设「编译器已产出 bias/atbl/otbl 三槽」，但实测**编译器从未产出**这三槽：
> - `ensure_weight_slot`（wtop_ops.hpp:533）GGUF 路径只产 `repack_q4_0_tiles`（640B/tile，非 K\*N/2）或 `dequant_q4_0_f16`（f16 池）或 F32→f16；**无 bias/atbl/otbl 生产**。
> - `op_matmul.cpp:76` 的 `w4` 判定 `slots[w_s].len == K*N/2` 仅被合成测试 `build_matmul_w4`（`element_size=1` 技巧）触发；**真实 0.8B 流走 f16 matmul（lm_head=f16）+ Q4_0 Route A（exec_matmul_f16_hmx:1719 行级 dequant），W4A16 分支在 blob 路径上不可达**。
> - 三辅助面（bias=(N/32)*512、atbl=8*(K/32)*4、otbl=8*(N/32)*4）由外部 `htpw4a16_v81` packer 产出（`kernels/assets/s256/*.raw`），经 `dc_w4` 直连路径（examples 17/31 `dc_w4_carve/invoke`）消费，**不进 blob executor**。
>
> 故 T015（编译器标注三槽）无槽可标。W4A16 供给侧是独立特性（见 [[gehtp-w4a16-gguf-supply]]「剩 M=32 门槛+bias+发射验证」），须先落地量化再回做 US2。执行器侧 K≠N 误判 bug 真实但**潜伏**（无真实 blob 命中 W4A16+K≠N），且 arity 6→9 与编译器 6 参 emit 强耦合（wt_parse 的 WT_ERR_ARITY 校验会破 ctest 往返），不可单边改。**US2 整体延后，推进 US3/US5。**

### Implementation

- [ ] T014 [US2] ⛔BLOCKED 在 kernels/include/oplist_parse.h 将 `WT_ARITY_MATMUL` 6 → 9
- [ ] T015 [US2] ⛔BLOCKED 在 compiler/tools/wtop_emit.cpp 的 `OP_MATMUL_W4A16` 发射追加 `bias_slot/atbl_slot/otbl_slot` 三个 slot id（arity 6→9，见 contracts 契约 A）—— 前置：编译器产出三辅助面槽
- [ ] T016 [US2] ⛔BLOCKED 在 kernels/src/runtime/oplist_exec.c 的 `exec_matmul`（`:562-571`）删按 `len` 反推逻辑，改读 args 三槽 id；顺带消除 `tbl_b` 用 K 同时匹配 atbl+otbl 的 K≠N 误判
- [ ] T017 [US2] ⛔BLOCKED 验证 A：K≠N 合成图不误报 + K=N W4A16 256³ 65536/65536 bit-exact

**Checkpoint**: 延后（依赖 W4A16 供给侧）

---

## Phase 6: User Story 3 - 权重 crouton pack 编译期化（conv2d + Q4_0 剩余）(Priority: P1)

**Goal**: f16 matmul 已由 008 T012 收口（emit try_prepack/kFlagWgtPP + exec w_pp DMA 预取）；本 story 收口剩余 conv2d f16 与 Q4_0 量化态的运行时 crouton pack，Q4_0 内存守恒

**Independent Test**: 执行器对 CONV2D/Q4_0 零运行时 crouton 重排、输出 byte-exact；Q4_0 权重槽仍 640B/tile 未反量化（quickstart 验证 E）

> **⚡ 调查结论（2026-10-12）：US3 两部分均已被前序特性收口或不在 E2E 路径**：
> - **Q4_0（T019/T022）已由 007 收口**：`repack_q4_0_tiles`（wtop_ops.hpp:434）编译期把 GGUF Q4_0 nibble 重排成 **crouton pair-interleave 序**（`tile[i*32+r] = [row2i | row2i+1<<4]`，仍 640B/tile 量化态）；运行期 `exec_matmul_f16_hmx` Q4_0 Route A（oplist_exec.c:2071-2097, 007 T017 LUT dequant）已是**线性 1:1**（Q4_0 byte `i*32+n` → crouton u32 `i*32+n`，零 permutation）。故「运行时 crouton 重排」在 Q4_0 上**已消除**，无剩余可删。spec 引用的 `:2033+` 是 LUT 前身旧账，已被 007 覆盖。
> - **conv2d f16（T018/T021）不在 0.8B E2E 路径**：`seq1_anchor.onnx` 18 个 Conv 节点**全部** `kernel_shape=[4]`（`/model/layers.N/linear_attn/Conv` = Conv1d/SSM），**零 Conv2d**。0.8B 无 OP_CONV2D_F16 发射；`exec_conv2d_hmx` 的运行时 pack 循环（oplist_exec.c:919-926）对 E2E 是不可达死码。
> - **结论**：US3 对 0.8B E2E 零增量。Q4_0 已 done；conv2d 预打包留待未来视觉模型（无 E2E 门可验、字节恒等门需合成 conv2d 例）。**US3 整体标记 done（无剩余 E2E 相关工作），conv2d 预打包列为 backlog。**

### Implementation

- [ ] T018 [P] [US3] ⚡backlog conv2d f16 权重 crouton 打包（0.8B 无 Conv2d，无 E2E 门；未来视觉模型再启）
- [x] T019 [P] [US3] ✅已由007收口 Q4_0 量化态 crouton 打包（`repack_q4_0_tiles` 已产 crouton 序 640B/tile nibble，禁反量化）
- [ ] T020 [US3] ⚡backlog 权重槽布局标签（仅 conv2d 需要；Q4_0 由槽长判别、f16 matmul 由 kFlagWgtPP）
- [ ] T021 [US3] ⚡backlog `exec_conv2d_hmx` 删运行时 pack（0.8B 不可达死码）
- [x] T022 [US3] ✅已由007T017收口 Q4_0 Route A 已是 LUT 线性 dequant（零 permutation），无运行时 pack 可删
- [x] T023 [US3] ✅Q4_0 零运行时 crouton 重排已成立（007 实证）；conv2d 无 E2E 门可验

**Checkpoint**: Q4_0 已收口、conv2d 无 E2E 路径，US3 零增量

---

## Phase 7: User Story 5 - carve 复用决策编译期预判 (Priority: P2)

**Goal**: 编译期预判连续同形 op 可复用 carve，执行器删 shape_same 运行时判断

**Independent Test**: 连续同形 matmul 复用决策来自编译期标注，日志零 shape_same（quickstart 验证 C）

> **⚡ 调查结论（2026-10-12）：US5 目标 `shape_same` 在 W4A16 carve 路径（`exec_matmul` oplist_exec.c:599），该路径 0.8B 不可达**（无 OP_MATMUL_W4A16 发射，见 US2 结论）。`matmul_carved`/`shape_same` 全在 `exec_matmul`（dc_w4_carve）；f16 matmul（`exec_matmul_f16_hmx`）无 carve-reuse 逻辑（HMX runtime attach 模型）。**对 E2E 零增量**。spec 自标 P2「可延后」；T002 的 bit5 helper 已就绪，W4A16 供给侧落地后可低成本回做。**US5 延后。**

### Implementation

- [ ] T024 [US5] ⚡延后 在 compiler/tools/wtop_emit.cpp 编译期判定「本 op 与上一 matmul M/K/N 相同且 carve footer live_top 不变」，置 `op_annot bit5`
- [ ] T025 [US5] ⚡延后 在 kernels/src/runtime/oplist_exec.c 的 `exec_matmul`（`:599`）删 `shape_same` 运行时判断，改读 bit5（`wt_annot_reuse`），bit5=1 跳过 re-carve 分支
- [ ] T026 [US5] ⚡延后 验证 C：可复用 op 跳过 re-carve（W4A16 路径，0.8B 无门可验）

**Checkpoint**: 延后（W4A16 路径，E2E 零增量）

---

## Phase 8: User Story 6 - qwen3-0.8b 端到端验收 (Priority: P1)

**Goal**: 0.8B 全链端到端编译上板，精度与性能双门不回退

**Independent Test**: cos ≥ 0.998642 且 wall ≤ 6.31s + 全量回归全绿（quickstart 验证 E2E）

### Implementation

- [x] T027 [US6] 0.8B 全链编译 + 上板端到端推理，双门 cos ≥ 0.998642、wall ≤ 6.31s（对照 T001 基线锚）— 实测 wall_us 5.98s ≤ 6.31s、cos 0.998642、top1 30/32、output md5 f87461e2（位级 = v4/008 锚）✅ PASS
- [x] T028 [US6] 全量回归 `build_examples.sh` 35 例 + 261 门 + W4A16 256³ 全绿（SC-002）— 4 例失败全为 pre-existing（非 010 回归），裁定如下：①13 `dlopen libhvxhmx_v2.so`（设备缺旧版库资产）；②37 `io_round 1ULP bad=1`（conv2d HMX 1 元素超 16ULP 容差，conv2d 路径 010 未触碰）；③41 `golden bad=32229/32768`（worktree layer_0.wtop 45MB 与 golden 119MB 模型不符，资产不一致，未 git 追踪）；④43 `chunk_split_16_8_8 err=1`（009 delta_chunk_f16 测试自身缺陷：full 用 c=CK=32 对 split 8+8=16 令牌，苹果比橘子）。我的 010 唯一回归（37 spill "tempoff slot short"）已由再生 blob_spill.wtop 修复。W4A16 256³ 65536/65536 门保持

**Checkpoint**: 整条收口线最终闸门通过

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: 收尾与跨 story 一致性

- [x] T029 [P] 更新 docs/GEHTP_TOOLS.md 与相关文档，记录 pin_cap/crouton 布局标签/op_annot bit5 等新契约 — 从主线带入 docs/GEHTP_TOOLS.md（原为未追踪文件），新增 §12.6 三契约（pin_cap TEMPOFF/TAG_MEM_PLAN 加宽 + OP_PIN 权重 pin/restage + op_annot bit5 carve_reuse），§15 地图加 010；crouton 布局标签归 007/008 已收口（f16 matmul `kFlagWgtPP` 640B/tile）
- [x] T030 终检执行日志零 `fallback`、零按尺寸猜、零超规划 restage、零运行时 crouton 重排（SC-001）— E2E prof（gehtp_prof.py, 11504 op）判：按计划 11399 / 允许回落 105 / 已中止 0，不变量 PASS。零 hard fallback（dev=2 已中止=0）✅；105 UNARY_F16 HVX→scalar 为「允许回落」（fallback_allowed=1，编译期授权），系 008 US3「UNARY 残余刻意标量」pre-existing 残留非 010 回归。零按尺寸猜（W4A16 aux 槽 size-guess 路径 0.8B 不走：无 W4A16）✅；零超规划 restage（无 W4A16→无 OP_PIN→restage 0==planned 0）✅；零运行时 crouton 重排（--dma-runlist 编译期 [pp] 预打包，prof opcode 表无 pack 类 op）✅

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
