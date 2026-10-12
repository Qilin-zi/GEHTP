# Feature Specification: 静态内存规划收口（编译器决策全下发 → 执行器纯查表）

**Feature Branch**: `010-static-mem-plan-closure`

**Created**: 2026-10-10

**Last Amended**: 2026-10-12（并入 `static_mem_plan_roadmap.md` 的 crouton-pack 范围 + 重排优先级 + 代码核实后的架构修正）

**Status**: Draft

**Input**: 静态内存规划收口：编译器内存决策全部下发进 wtop blob（W4A16 辅助面身份显式标注、pin_cap 编译期计算下发、carve 复用编译期预判、权重 pin/restage 时序静态化、权重 crouton pack 编译期化），执行器退化为纯查表机械执行，删除硬编码/按尺寸猜/运行时判断/运行时重排；qwen3-0.8b 端到端编译上板验证精度性能不回退

## Background *(依据与盘子)*

- **链已基本打通，剩 5 个运行时决策点**：编译器（host，逆向 libHtpPrepare）已完成 ST-Cut 调度、FancyAllocator、cp_solver 的 pin/remat/comp 决策、block table、tiling、指令选择；执行器（自研 skel `oplist_exec.c`）经 wtop blob（v3）消费。blob 已具备 engine 注解 footer、TEMPOFF 静态 temp 表、carve footer、OP_DMA/SPILL/FILL、STATE 槽，执行器已消费大部分。剩余「编译器算好却没下发、执行器还在自己猜/硬编码/运行时重排」的收口点。
- **五个收口点（已对代码核实，含 10-12 roadmap 新增的 E）**：
  - **A** W4A16 辅助面身份：`oplist_exec.c:562-571` 按 len 反推 bias/atbl/otbl；`tbl_b=8*(K/32)*4` 同时匹配 atbl+otbl，otbl 实长 `8*(N/32)*4` → K≠N 必误判；atbl/otbl 同长身份靠槽顺序。
  - **B** pin_cap：执行器两处写死 4096（`oplist_exec.c:325,600`）；编译期 `MemPlan` 无 pin_cap 字段，只在 `vtcm_auto_scan` 当固定余量假设（`graph_prepare.cpp:670`）。**非「只差序列化」——须先补算**。
  - **C** carve 复用：`oplist_exec.c:589-591` 运行时 `shape_same` 判 M/K/N 决定重 carve；carve footer 已是 5×u32=20B 满（op_index/vtcm_offset/size_bytes/residency_start/residency_end）。
  - **D** 权重 restage：`oplist_exec.c:657-658` 每 MATMUL 无条件 `cpu_to_vtcm` 搬 wt+bias；`xop_pin`（`oplist_exec.c:3078-3091`）仅记账、engine_ready 时又搬一次 → **D2 PIN 与 MATMUL 各搬一次权重**。
  - **E** 权重 crouton pack 运行时发生：**f16 matmul 已由 008 T012 收口**（emit `try_prepack`/`kFlagWgtPP`，exec `w_pp` DMA 线性预取，主路径零标量散写 pack）；剩余运行时 pack 在 `exec_conv2d_hmx`（`oplist_exec.c:888-895`，f16 conv）与 Q4_0 Route A（`oplist_exec.c:2033+`，运行时反量化+pack）。这是输入无关的纯重排，本应编译期一次性做掉。
- **证据基线**：cos 锚 0.998642 / wall 6.31s（008 终态锚，lib `libhvxhmx_008t21.so` 签名 `e885485f`、commit `de5574a`；wall 口径 = `[timing] wall_us`，**非** `[run] seg wall` 11.44s——后者含 blob 冷读 4.47s 一次性 I/O）；byte-exact 回归门 = 35 例 + 261 门 + W4A16 256³ 65536/65536 bit-exact。
- **源文档**：`static_mem_plan_gaps.md`（收口动作对照表与已确认结论）、`static_mem_plan_roadmap.md`（2026-10-12 行动计划，新增 E 并重排优先级）。
- **量化口径警示**：roadmap 阶段 1 引用的「权重 pack 20.6s/全图」是**标量 pack 旧账**（`oplist_exec.c:1983` 自述「标量 pack 9.72s/20.6s → LUT 法」）；T017 已用 LUT 法优化过，LUT 后剩余 pack 耗时未重测。阶段 1 真实收益须以重测为准，非 20.6s → 0。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 权重 pin/restage 时序静态化 (Priority: P1)

编译期决定哪些权重 pin 常驻、何时 restage，显式下发执行序列，执行器按序执行，删除每 op 无条件 restage 与 PIN/MATMUL 重复搬运。

**Why this priority**: 全特性的核心价值（pin 后复用、消除重复权重搬运），命中 decode 的 DDR I/O 瓶颈（decode 每一步重跑同批权重，restage 是放大项）。但权重常驻时序是全新规划面（greedy 与 cp_solver 都 skip const 权重），需在默认路径新建该规划。

**Independent Test**: 一个含「同权重连续多个 matmul」的图，编译期下发 pin+复用序列，执行器对已常驻权重零重复搬运（restage 次数 == 编译期规划值）。

**Acceptance Scenarios**:

1. **Given** 同权重被多个连续 matmul 消费，**When** 编译期 pin 决策下发，**Then** 执行器对已常驻权重不重复搬运，`wt_w3_report` 中 restage 次数 == 编译期规划值。
2. **Given** 权重需换入换出（超常驻预算），**When** 执行，**Then** 换入换出严格按编译期下发序列，无超出规划的 restage。
3. **Given** PIN 与首个消费 MATMUL 的同一权重，**When** 执行，**Then** 权重只搬一次（PIN 记账不再与 MATMUL 各搬一次）。

---

### User Story 2 - W4A16 辅助面身份显式标注 (Priority: P1)

编译期把 W4A16 的 bias/atbl/otbl 三个辅助面的 slot 身份显式写进 blob，执行器按标注取槽，删除按 len 反推；顺带消除 K≠N 误判。

**Why this priority**: 唯一同时是「正确性 bug」+「独立最小切片」的收口点；K≠N 会误报 supply slots missing，K=N 时 atbl/otbl 同长、身份靠槽顺序兜底。

**Independent Test**: 编译一个含 W4A16 matmul 且 K≠N 的合成 blob，执行器按标注取三槽，不触发按尺寸猜，K≠N 用例不再误报；K=N 用例位级同旧行为。

**Acceptance Scenarios**:

1. **Given** 含 W4A16 matmul 的图（K≠N），**When** 编译并上板执行，**Then** 三辅助面按编译期标注身份取槽，执行日志零按尺寸猜，不再误报 supply slots missing。
2. **Given** 存量 W4A16 256³ 回归（K=N），**When** 收口后重跑，**Then** 65536/65536 bit-exact 门保持，输出逐字节同旧行为。

---

### User Story 3 - 权重 crouton pack 编译期化（conv2d + Q4_0 剩余）(Priority: P1)

f16 matmul 的 crouton 预打包已由 008 T012 收口（emit `try_prepack`/`kFlagWgtPP`，exec `w_pp` DMA 预取）。本 story 收口剩余两条运行时 pack 路径：conv2d f16 与 Q4_0 量化态。

**Why this priority**: 权重 crouton 重排是输入无关的纯重排，运行时每跑一次白做一次。f16 matmul 半壁已被 008 做掉，剩余 conv2d + Q4_0 是同一重排问题的收尾；收益上界由重测决定（标量 pack 20.6s 已被 T017 LUT 优化，剩余量待重测）。

**Independent Test**: 编译期产出 conv2d f16 + Q4_0 crouton 布局的 blob，执行器对 CONV2D/Q4_0 matmul 零运行时 crouton 重排，数值 byte-exact 不破，Q4_0 槽仍 640B/tile 量化态。

**Acceptance Scenarios**:

1. **Given** 含 f16 conv2d 权重的图，**When** 编译并上板执行，**Then** `exec_conv2d_hmx` 不再做运行时 crouton 重排（直接读编译期打包槽），输出逐字节同旧行为。
2. **Given** 含 Q4_0 权重 matmul 的图，**When** 编译并上板执行，**Then** Q4_0 Route A 不再运行时反量化+pack（改 LUT dequant 直接读量化态打包槽），槽仍 640B/tile，输出逐字节同旧行为。

---

### User Story 4 - pin_cap 编译期计算与下发 (Priority: P1)

编译期真正算出 pin_cap（现 `MemPlan` 无字段），序列化进 blob，执行器读出替代两处硬编码 4096。

**Why this priority**: D（权重 pin 复用）的前置；今天执行器固定 4096，编译期对权重常驻区的规划无法下发。

**Independent Test**: 用不同权重常驻需求编译两个 blob，编译期产出的 pin_cap 随需求变化并写入 blob，执行器读 blob 值替代 4096。

**Acceptance Scenarios**:

1. **Given** 编译期规划了权重常驻区，**When** 生成 blob，**Then** blob 携带编译期 pin_cap，执行器读该值（而非硬编码 4096），日志无 4096 硬编码路径。
2. **Given** 无权重常驻规划（关池口径），**When** 执行，**Then** 执行器行为与旧硬编码口径逐字节一致（byte-exact 门保持）。

---

### User Story 5 - carve 复用决策编译期预判 (Priority: P2)

编译期预判哪些连续 op 可复用 carve，写进 carve footer/注解，执行器删除 `shape_same` 运行时判形状。

**Why this priority**: 残留的运行时判断里价值最低（shape_same 只是重 carve 跳过优化）；carve footer 20B 已满，需扩 24B 或新注解槽。零性能收益、有 byte-exact 回归风险，可延后。

**Independent Test**: 连续同形 matmul 序列的复用决策由编译期给出，执行器不再用 shape_same 判形状即可跳过重 carve。

**Acceptance Scenarios**:

1. **Given** 连续同形 matmul，**When** 执行，**Then** 复用决策来自编译期标注，执行日志零运行时形状判断，carve 复用结果与旧 shape_same 路径一致。

---

### User Story 6 - qwen3-0.8b 端到端验收 (Priority: P1)

0.8B 全链端到端编译 + 上板验证，精度与 wall-clock 性能均不回退。

**Why this priority**: 整条收口线的最终闸门；五个收口点的正确性都归结为「全链不回归」。

**Independent Test**: 0.8B 全链编译上板，双门（cos / wall）对照当前库锚不回退。

**Acceptance Scenarios**:

1. **Given** 0.8B 模型全链编译，**When** 上板执行端到端推理，**Then** cos ≥ 0.998642、top1 达标，wall-clock 不回退于 6.31s 库锚。
2. **Given** 全链回归集，**When** 收口后跑全量回归，**Then** 35 例 + 261 门 + W4A16 256³ bit-exact 全绿。

---

### Edge Cases

- **K≠N 的 W4A16 matmul**：辅助面 atbl/otbl 长度不同，按尺寸猜必误判 → 收口 A 后由显式标注覆盖。
- **无 carve footer（ver≤2）旧 blob**：执行器须退回旧布局语义（live_top=0），收口后仍兼容存量 ver≤2 blob。
- **权重超常驻预算**：换入换出路径须按编译期序列执行，越界显式降级而非静默 restage。
- **pin_cap 未规划（关池口径）**：执行器须回退到与旧硬编码逐字节一致的行为。
- **Q4_0 权重 pack 编译期化后槽契约**：编译期打包的 Q4_0 权重槽须与执行器现消费契约（`repack_q4_0_tiles` 640B/tile、dequant LUT 零损契约）逐位对齐，避免编译期/运行期布局漂移。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 编译器 MUST 决定权重 pin 常驻与 restage 时序，并显式下发执行序列（pin 决策从 cp_solver 的 env 旁路扶正为默认 emit 主线输出）。
- **FR-002**: 执行器 MUST 按编译期下发的 pin/restage 序列执行，删除每 op 无条件 `cpu_to_vtcm` 搬权重与 PIN/MATMUL 重复搬运。
- **FR-003**: 编译器 MUST 为每个 W4A16 matmul 在 blob 中显式标注 bias/atbl/otbl 三个辅助面的 slot 身份。
- **FR-004**: 执行器 MUST 按标注取辅助面，删除按 len 反推（`oplist_exec.c:562-571`）；K≠N 不再误判。
- **FR-005**: 编译器 MUST 在打包阶段直接产出 conv2d f16 与 Q4_0 权重的 crouton 布局（f16 matmul 已由 008 T012 的 `kFlagWgtPP` 收口），执行器删除剩余运行时 crouton 重排（`exec_conv2d_hmx` pack 循环 + Q4_0 Route A 反量化 pack）。
- **FR-006**: 编译器 MUST 真正算出 pin_cap（`MemPlan` 新增字段）并序列化进 blob。
- **FR-007**: 执行器 MUST 从 blob 读出 pin_cap 替代两处硬编码 4096（`oplist_exec.c:325,600`）。
- **FR-008**: 编译器 MUST 预判连续 op 的 carve 复用并写入 blob，执行器删除 `shape_same` 运行时判断。
- **FR-009**: 每项收口后 MUST 保持 byte-exact 回归（35 例 + 261 门 + W4A16 256³ 65536/65536）。
- **FR-010**: 执行日志 MUST 零 fallback、零按尺寸猜、零超出编译期规划的 restage、零运行时 crouton 重排。
- **FR-011**: `wt_w3_report` MUST 可统计 restage 次数并可与编译期规划值对账。

### Key Entities *(include if feature involves data)*

- **wtop blob**：编译器产物，承载全部内存决策（辅助面标注、pin_cap、carve 复用、pin/restage 序列、预打包 crouton 权重槽）的序列化载体，v3 格式。
- **辅助面标注**：每个 W4A16 matmul 的 bias/atbl/otbl 三槽身份（slot id）。
- **pin_cap**：权重常驻区容量上限，编译期算出、执行器消费。
- **carve footer**：逐 op 的 VTCM carve 区间与复用标志（现 5×u32=20B，扩展复用标志需扩 24B 或新注解槽）。
- **pin/restage 序列**：编译期下发的权重常驻/换入换出时序（OP_PIN/OP_DMA 等）。
- **预打包 crouton 权重槽**：编译期产出的 fp16/Q4_0 权重 crouton 布局，执行器直接 mxmem 加载。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 执行日志中零运行时决策——零 fallback、零按尺寸猜、零超出编译期规划的 restage、零运行时 crouton 重排。
- **SC-002**: 全量回归 byte-exact——35 例 + 261 门全绿，W4A16 256³ 65536/65536 bit-exact 门保持。
- **SC-003**: `wt_w3_report` 中 restage 次数 == 编译期规划值（可作为门）。
- **SC-004**: 剩余运行时 crouton pack（conv2d f16 + Q4_0）耗时按重测基线归零（上界 ≤ 重测后的 LUT 剩余 pack 量，非旧账 20.6s）。
- **SC-005**: qwen3-0.8b 端到端编译上板，精度 cos ≥ 0.998642 且 wall-clock ≤ 6.31s（`[timing] wall_us` 口径，对照 008 终态锚不回退）。

## Assumptions

- **基线锚**：当前库锚 = cos 0.998642 / 6.31s（跨会话协调表口径），「不回退」= 不劣于此锚。
- **D 的架构决策（已修正）**：cp_solver 的对家是**贪心 FancyAllocator**（两者都产出 `vtcm_allocations_`，同 `AllocResult` 契约），**不是 mem_plan**（mem_plan 是 DDR 侧偏移布局，另一维度）。关键澄清：**两者都 `skip const`（权重）张量**（`graph_prepare.cpp:4776-4782`）——cp_solver 的 `pin_slot` 是解校验记账、`cp_plans_` 只覆盖 temp 张量 spill/fill，**均不产「权重 pin/restage 时序」**。故权重常驻时序是**全新规划面**，既不在 greedy 也不在 cp_solver 现有产出里。本特性的决策不是「cp_solver vs mem_plan 二选一」，而是「在默认（greedy/mem_plan 家族）路径新建权重常驻 + restage 规划，复用现有 lifetime 分析，而非复活 env 门控的 CP oracle」。
- **精度不因布局决策漂移**：收口点 A–E 全是内存布局/搬运决策（offset、spill/fill、remat、pin、pack），不改变数值；cos 应 bit-identical。若 cos 漂移属 bug 而非预期偏差。
- **E 的量化口径**：20.6s 是标量 pack 旧账，T017 LUT 后剩余量未重测；阶段 1 收益以重测基线为准，不承诺 20.6s → 0。
- **carve 复用标志载体**：默认扩 carve record 至 24B 或另起注解槽，具体取舍在 plan 阶段定。
- **存量兼容**：ver≤2 旧 blob（无 carve footer / 无辅助面标注）执行器须保持旧语义，回归门覆盖。
