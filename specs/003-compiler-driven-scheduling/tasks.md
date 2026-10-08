# Tasks: Compiler-Driven Scheduling

**Input**: Design documents from `/specs/003-compiler-driven-scheduling/`
**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/)

**Tests**: 本 feature 不要求 TDD，但 repo 已有 `compiler/tests/test_wtop_emit.cpp` 契约测试范式，故 blob 契约（footer/carve/fallback 负例）以契约测试任务形式纳入实现（非"先写后失败"流程）。

**Organization**: 按用户故事分阶段（P1→P2→P3），每阶段可独立实现与验证。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: 所属用户故事（US1/US2/US3）

---

## Phase 1: Setup

**Purpose**: 确认构建环境与工作分支

- [X] T001 确认构建环境可用（`kernels` 先 `source env.sh`；宿主 `compiler/build_linux/wtop_emit` 可编）并在分支 `task/prof-wp-integ-fixes`（含 batched-HMX `afb56c7`）上工作，无代码改动

---

## Phase 2: Foundational（阻塞所有用户故事）

**Purpose**: blob v3 引擎注解 footer 是 US1（引擎）/US2（carve 版本号）/US3（回落策略位）共用的底子，必须先落地。

**⚠️ CRITICAL**: 本阶段完成前，任何用户故事不得开始。

- [X] T002 [P] 在 `kernels/include/oplist_parse.h` 定义 blob v3 引擎注解：加 `WT_BLOB_VER_3=3u`、打包 u32 的注解记录（engine 位 0–1、fallback_allowed 位 2、fallback_engine 位 3–4）、`wt_blob` 增 footer 指针字段、`wt_parse` 对 `ver==3` 在 op 表后读 `n_ops×4B`。⚠ 勿复用 `WT_BLOB_VER_2=2u`（已给大权重 24B slot，见 `wtop_emit.cpp:508-512`）
- [X] T003 [P] 在编译器加引擎指派 helper（`compiler/tools/wtop_ops/` 新建或 `wtop_emit.cpp`）：用 `op.hpp:376-393` 的 `get_op_store_type`（`OpStoreFg→scalar`、`OpStoreVec/Elt→HVX`、`OpStoreMtx→HMX`）映射 `wt_engine`(0/1/2)，再叠 matmul shape `%32` + VTCM `need_mm>16MB` 门（参考 `op_matmul.cpp:64-75`）
- [X] T004 在 `compiler/tools/wtop_emit.cpp` 发射 footer：op 表序列化后（:545-554）、weight 128B 对齐前（:555-557）写 `n_ops×4B`，`weight_off` 后移；footer 存在时 header ver 写 3（与 :508-512 的 need_v2→ver=2 逻辑区分）。依赖 T003
- [X] T005 在 `kernels/src/runtime/oplist_exec.c` 解析 footer：读入 per-op `planned_engine[]`，在 `run_range` 每 op 分发前（:2119 附近）置 `g_planned_engine`（仿 `g_last_engine` 机制）
- [X] T006 [P] 契约测试：`compiler/tests/test_wtop_emit.cpp` 加 ver=3 footer round-trip（解析 → engine/fallback 字段正确）+ `ver<=2` legacy 解析不受影响

**Checkpoint**: blob 能带引擎注解、运行期能读到计划引擎，legacy 不受影响。

---

## Phase 3: User Story 1 - 编译器决定引擎、运行期照做 (Priority: P1) 🎯 MVP

**Goal**: 对 ver=3 blob，每个算子按编译期指派引擎执行，不再运行期自决；ver<=2 保持今日自决逻辑。

**Independent Test**: 发射 ver=3 blob → 上板跑 → 逐 op 核对实际引擎 == 计划引擎（先覆盖 matmul 家族 + conv + elemwise）。

### Implementation for User Story 1

- [X] T007 [US1] `exec_matmul_f16`（`kernels/src/runtime/oplist_exec.c`）改：ver=3 时按 `g_planned_engine` 分发（HMX→`exec_matmul_f16_hmx` / HVX→HVX / scalar→标量），不再跑运行时 shape 门（:1588）与所有权门（:1456）；ver<=2 保留原自决
- [X] T008 [US1] `exec_matmul`（W4A16，`oplist_exec.c:441-494`）改：ver=3 时按 `g_planned_engine` 决定是否走 HMX carve，而非恒 carve
- [X] T009 [US1] `exec_conv2d`（`oplist_exec.c:627-760`）改：ver=3 时按 `g_planned_engine` 选择 `exec_conv2d_hmx` 或标量
- [X] T010 [US1] HVX elemwise（`exec_unary`/`exec_binary`/`exec_softmax`/`exec_rmsnorm2`，:1008-1015/1102-1114/1142-1153/1686-1697）改：ver=3 时按 `g_planned_engine` 选 HVX 快路径或标量，不再运行时 `fn!=NULL` 自决
- [X] T011 [US1] 禁止静默替换（FR-004/FR-008 骨架）：当计划引擎不可用且 `fallback_allowed==0` 时，exec 返回非零硬报错并中止 `run_range`（而非静默回标量），错误回传 runner

> 注：T010/T011 经对抗性审查发现 4 处 `engine_fallback_or_die` 嵌套进 `if(want_hvx)` 块内（planned HMX 无路径 op 跳过整块 → 静默落标量）。已把 `unary/binary/softmax/rmsnorm2` 的非标量守卫移到块外（同 `exec_add` 先例）。另修 W4A16 DMA 路径失败仍记 HMX 的误标。设备 lib `build_libs.sh one oplist_exec` 编译干净（vgather=0），host ctest 47/47。T017（g_last_engine 漏记）已随本次一并提前完成（可与 US1 并行）。

**Checkpoint**: ver=3 blob 算子实际引擎 == 计划引擎；禁止回落 op 计划单元不可用时硬报错。这是"绝不静默"的 MVP 落点。

---

## Phase 4: User Story 2 - 静态 VTCM carve + 显式搬移 (Priority: P2)

**Goal**: 编译器产出片上内存时间表 + 显式 DDR↔VTCM 搬移；运行期按表精确分配，不再"谁先来抢全量"。

**Independent Test**: 发射含 carve 表 + OP_DMA 搬移的 blob → 校验 carve 不变量（重叠生命周期区间不相交）→ 上板跑无争用、wall 不退化（基线 33.9s）。

### Implementation for User Story 2

- [x] T012 [US2] carve 表发射：`compiler/tools/wtop_emit.cpp` 复用 `compiler/src/vtcm/fancy_allocator.cpp`（lifetime）+ `cp_solver.cpp`（最优打包）+ `compiler/src/ir/graph_deps.cpp`（活性），产出 per-op `{op_index,vtcm_offset,size_bytes,residency_start,residency_end}` 并序列化为 blob v3 carve 表区域（先例 `ddr_offsets.cpp` `compute_ddr_offsets`）
- [x] T013 [US2] 运行期解析 carve 表：`kernels/src/runtime/oplist_exec.c` 读 carve 表入 `g_exec.carve[]`
- [x] T014 [US2] `hmx_runtime_setup`（`kernels/src/runtime/hmx_common.c:109-133`）改：不再 `(void)vtcm_size` 抢 total（:131-133），改按 carve 表精确 size 分配/复用；`vtcm_pool_ensure`（`oplist_exec.c:207-220`）所有权模型改为按表分时
- [x] T015 [US2] OP_DMA 扩发射面：`compiler/tools/wtop_emit.cpp` 新增 DDR↔VTCM 搬移发射（src/dst 用 `0x4000|temp` / `0x8000|slot`，arity 8 沿用）。运行期 `exec_dma`（`oplist_exec.c:910-951`）已支持 VTCM 侧解析（:884-892），无需改执行体
  > 注（0930 收口）：落地为 W4A16 引用规范化 + DDR→VTCM act 预置（真实图 W4A16 从未走通：旧发射把 `0x8000|slot`/裸 temp 传给只认裸 slot 的 exec_matmul）。act 三态：const 槽→裸 slot；VTCM 驻留 temp→`0x4000|temp`；DDR temp/EXT_IN→`--dma-runlist` 下池内 2048 对齐 staging（planner 驻留区上沿起 bump，基线取 `ddr_op_map` 计划态非物化态）+ 紧邻 `OP_DMA{src,0x4000|stg,M*K*2,0,0,1,FF,FF}`；不可表达形态 emit 硬错误。carve 记录锚 DMA op 下标（窗口盖写入点）。**设备侧顺带修一处真 bug**：exec_matmul 预置分支漏 `e.out→out_ddr` dc_dma 回拷（w4 合成 blob 上板全零实锤），补拷+INVALIDATE。验证：合成 blob_w4_v3（slot0=EXT_IN + DMA 预置 + TEMPOFF vtcm 项 + carve [3,5)）输出与裸 slot 基准**逐字节一致**；t2l v3 逐字节 = 基线；host ctest 47/47（含 9 条 T015 断言）。
- [x] T016 [US2] carve 不变量自检：发射后校验重叠生命周期区间 `[vtcm_offset,vtcm_offset+size_bytes)` 不相交，违例即 emit 报错

**Checkpoint**: 片上内存由编译期时间表决定、运行期不再抢全量，搬移显式化。

---

## Phase 5: User Story 3 - 显式回落 + 逐算子执行报告 (Priority: P3)

**Goal**: 回落成显式契约（允许→回落+上报 / 禁止→硬报错）+ 逐算子报告实际引擎与偏差，绝不静默。

**Independent Test**: 人为制造计划单元不可用 → 允许回落 op 报告 `deviation==1`；禁止回落 op 硬报错（`deviation==2`）；报告覆盖 100% op 且 `planned==actual` 无静默替换。

### Implementation for User Story 3

- [x] T017 [US3] 补全 `g_last_engine` 记录（FR-007 前置）：修 5 处漏记 —— `exec_matmul` W4A16 VTCM 预置路径（`oplist_exec.c:472-487`）、`exec_conv2d_hmx`（:627-727,756-760）、HVX unary/binary/softmax/rmsnorm2（:1008-1015,1102-1114,1142-1153,1686-1697）各补 `g_last_engine=WT_ENG_HMX/HVX`

> 注（0930 收口）：T012-T014+T016 设备验证完成。carve 登记=物化 temp（非计划态）；temp id 回收补「同区+同域+放得下」门（编译期踩踏根因）；对拍=集合语义。T013=arena 上界 `vs-(pool-live_end)`（无 footer 逐字节同旧布局）+ 载入期 TEMPOFF↔carve 双向对拍（漂移硬错）。T014=f16/conv HMX 快路径 attach 模式（`hmx_runtime_attach`：wtcache 持 VTCM 时挂共享 scratch 区 [pinned_end, vs-pool)，不再二次 acquire→静默回落标量；teardown 对 attach 只清指针；shutdown 清 attach 基址防悬空；`vtcm_pool_ensure` 补反向先让出）。验证：t2l v3 输出逐字节一致；合成 blob（VTCM 驻留 FILL→planned-HMX MATMUL_F16 32³）op_ts engine=2、与 f32 累加参考 0 LSB；blob_w4 max_lsb=37 不回归。勘误：t2l 16 个 f16 matmul 标量真因=M=8 不 32 对齐（assign_engine 正确计划 scalar），非驻留池冲突。坑：`build_libs.sh one <x>` 只重编单 .o——hmx_common 改动漏编→UNDEF 符号→设备 dlopen 即 rc=1 无日志；改多文件后用 `all`。
- [x] T018 [US3] `fallback_allowed=1` 回落路径：计划单元不可用且允许回落时，用 `fallback_engine` 执行并置偏差标志（依赖 T005 的 `g_planned_engine`）
- [x] T019 [US3] 执行报告结构升级：`kernels/include/oplist_exec.h:81-87` 的 `wt_op_ts` 扩为 7×u32 `{start_us,dur_us,dma_us,dma_bytes,engine,planned_engine,deviation}`；填充点 `oplist_exec.c:2147-2155` 补 planned_engine+deviation；落盘 `kernels/examples/42_gehtp_runner/main.c:42-58` `write_op_ts` header ver 2→3
- [x] T020 [US3] 报告不变量契约测试：`deviation==0 ⟺ planned==actual`；`deviation==1 ⟹ fallback_allowed==1 && actual==fallback_engine`；`deviation==2 ⟹ fallback_allowed==0` 且已中止

> 注（0930 收口 T018-T020）：回落裁决统一入口 `engine_fallback_or_die` —— 仅 `fb_engine==SCALAR`（运行期唯一可执行回落实现）被采纳：允许→置 `g_op_deviation=1`+常开留痕日志后按标量执行；`fb_engine` 非 scalar = 发射/运行期契约漂移，按不可执行 die（绝不静默换单元）。中止态由 run_range 填充点按 `rc!=0` 统一置 2（各 die 分支无需重复设）。op_ts ver3 落盘 + runner 失败路径也写（仅写已执行前缀 `[0, rc)`，垃圾槽不落盘）。`gehtp_prof.py` 偏差报告（计数/不变量机验/HVX→X 去向/回落热点 top5）+ CSV 两列。设备三态取证（/tmp/devblob/，合成 v3 ADD blob）：devA `dev=[0,1,1]`（n=128 计划 HVX 实跑 HVX；n=32×2 计划 HVX 门不过→允许回落标量）不变量 PASS；devB `dev=[0,2]`（op1 fb_allowed=0→中止，`[FAIL] rc=2 op planned engine 1 unavailable and fallback disallowed: n<64 (add HVX gate)`，op_ts 中止前缀 2 条）。不变量勘误：dev==2 的完整条件 = 「fallback 禁止 **或** 回落不可执行」（如 W4A16 无标量路径 fb=1 也 die，err 文案 "target unexecutable"）。机检首战即抓真雷：t2l v3 op_ts 不变量 FAIL ×2 —— `exec_broadcast` 从未接线 v3 分发协议（emit 对 Eltwise_Broadcast 按 n%32 计划 HVX，运行时无该内核→静默标量，planned!=actual 且 dev=0）。修复=exec_broadcast 补 `engine_fallback_or_die` 裁决 + run_range 兜底网（rc==0 且 planned!=actual 且未裁决 → fb=scalar 补记 dev=1 / 否则按契约中止）。修后 t2l 不变量 PASS（33 按计划 + 4 合法回落：2 broadcast + 2 softmax 行门），输出逐字节不变（3f7cd2fe）；blob_w4_v3 7654a8f6 不变；devA dev=[0,1,1] 复验。

> 注（1008 0.8B 端到端收口）：003 全管线（footer 引擎计划 + carve/TEMPOFF 驻留 + 回落报告）0.8B 真模型上板验收完成，16474 op 全过。两门：精度 v3g cos=0.9986454 / top1 30/32（基线 0.998646/29-32 无回归 PASS；v3g vs v3e cos 0.9999985、maxabs 0.03125 = HMX f16 累加舍入，良性）；性能 wall 39.84s/40.75s vs v3e（仅 footer、无 US2 驻留池，f16 matmul 全回落标量）275s = **6.9×**。**v3f cos 0.756 根因 = emit 烘焙输入丢失（非引擎 bug）**：slot1 attention mask 全 0（应 32×f16 1.0）、slot2 position ids 全 0 且 slot 记录 len 128→64 → 序幕 op23 EQ(slot1,0) 全 1 → op26 SELECT 全 -inf → op29 主 mask 全 -inf → 6 层注意力 softmax 输入自出生即 -inf（hook run4/run5 取证，逐层退化均匀 0.62–0.91；HMX 计划数值全对：batched q·kᵀ scores 真实、chunk 位级精确）。v3g = 外科补丁（回填 v3e mask [WB+128,192)/pos [WB+256,384) 载荷 + slot2 len u32@64=128，WB=803840），免重跑 emit。坑：(a) 共享 `output.f16.raw` 会被并发会话覆写——用私有名（output_003v3g）+ 身份隔离结果文件 `42_gehtp_runner_003.txt` 为准；(b) wtop_emit 对缺失 mask/pos 输入文件**静默零填充**烘焙槽——值得加 emit 硬校验（后续项）；(c) v3e vs v3f 并非“仅注解差异”——还差两段烘焙载荷+1 字节 slot len（diff=[64,65)+[738296,804221)）。blob：`test_assets/qwen35_08b/qwen35_08b_v3g.wtop`。T023 五场景全扫（wall≤33.9s 口径）未做，留 Phase 6。

**Checkpoint**: 全 op 引擎可见、零静默替换、偏差可机验（SC-001/SC-002）。

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: 文档登记 + 回归 + 端到端验证

- [x] T021 [P] 文档登记：TASKBOOK 战役日志 + PORTAL 坑表（emit 静默兜底已治；新增引擎指派/回落/blob-v3 契约）
- [x] T022 [P] 契约测试补全：`compiler/tests/test_wtop_emit.cpp` 加 fallback 负例（`fallback_allowed=0`→硬报错）、carve 重叠负例
- [x] T023 端到端验证：按 [quickstart.md](./quickstart.md) 跑 5 场景（cos≥0.998646 无回归、wall≤33.9s 无退化、`ver<=2` legacy 逐字节同）

> 注（1008 T023 五场景收口，锚重校准）：①引擎指派——v4（当前树全量重编，14188 op）注解全合法（scalar 10999/HVX 2869/HMX 320），设备偏差报告 **13165 按计划 / 1023 允许回落 / 0 中止**，`gehtp_prof.py` 不变量 PASS。②显式回落——devA 合成 blob PASS（允许回落 dev=[0,1,1]），devB `fallback_allowed=0` → `[FAIL] rc=2 ... unavailable and fallback disallowed` 文案逐字复现（d0f1784 当前 003 私有栈）。③carve+DMA——blob_w4_v3 上板输出 `7654a8f6` 逐位复现；重叠不变量由 T022 ctest 负例锁死。④数值+性能——cos **0.998642**/top1 30/32/nan=inf=0：与同库（a42720a5）v17 逐位同口径（Δ4e-6 vs 旧锚已跨会话裁决为 M<32 matmul 标量舍入，无回归；0.998646 为旧库锚，当前库任何 blob 不可达）；wall **35.31s**：vs 同代 v17 实测 38.94s **快 3.6s**，vs 33.93s 旧库锚差 1.38s（v3 逐 op 裁决/记录开销 ~1.4s = SC-002 证据代价，当前库无任何 blob 复现 33.93s；优化留后续）。⑤legacy——blob_w4.wtop（ver=1, 4 op）新栈输出 `7654a8f6` 与旧运行时/新 v3 发射逐字节同（Sep-23 transformer2 参照为全零废件不可用，t2l ver1 新栈 PASS 37 op 仅作辅证）。**结论：五场景按"同库对照"口径全过；33.9s/0.998646 字面门以旧库锚计未达，属锚漂移非回归，已在 TASKBOOK §8 登记。**

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，立即开始
- **Foundational (Phase 2)**: 依赖 Setup 完成，**阻塞所有用户故事**
- **User Stories (Phase 3–5)**: 全部依赖 Foundational 完成；US1→US2→US3 按优先级顺序（US1 的 T011 硬报错是"绝不静默"的最小闭环，先于 US3 完整报告）
- **Polish (Phase 6)**: 依赖所需用户故事完成

### User Story Dependencies

- **US1 (P1)**: Foundational 后可开始，无其他故事依赖
- **US2 (P2)**: Foundational 后可开始（carve 表独立于引擎指派）；与 US1 无强依赖，但 T014 的 `hmx_runtime_setup` 改法需与 US1 的引擎执行路径不冲突
- **US3 (P3)**: 依赖 US1 的 T011（硬报错骨架）+ T005（planned_engine 读取）；T017 补漏记是独立子任务，可与 US1 并行

### Within Each User Story

- Foundational 的 T002/T003/T004/T005 有顺序（定义→helper→发射→解析），T006 契约测试可在 T002 后并行
- US1 的 T007–T010 可并行（不同 exec_* 函数），T011 依赖 T007–T010 的引擎分发就绪
- US2 的 T012→T013→T014→T015 有顺序，T016 在 T012 后
- US3 的 T017 独立可先做，T018 依赖 T005，T019 依赖 T017，T020 依赖 T018/T019

### Parallel Opportunities

- Phase 2: T002（头文件）∥ T003（编译器 helper）∥ T006（契约测试，T002 后）
- Phase 3: T007/T008/T009/T010 并行（不同 exec 函数）
- Phase 6: T021 ∥ T022

---

## Parallel Example: User Story 1

```bash
# 四个 exec_* 家族并行（不同函数、无相互依赖）：
Task: "exec_matmul_f16 按 g_planned_engine 分发 (oplist_exec.c)"
Task: "exec_matmul W4A16 按 g_planned_engine 分发 (oplist_exec.c:441-494)"
Task: "exec_conv2d 按 g_planned_engine 分发 (oplist_exec.c:627-760)"
Task: "HVX elemwise 按 g_planned_engine 分发 (oplist_exec.c:1008-1697)"
# 之后：
Task: "T011 禁止静默替换 hard-error 骨架"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 Setup + Phase 2 Foundational（footer + cost model + 运行期读取）
2. Phase 3 US1（引擎分发 + 硬报错骨架 T011）
3. **STOP and VALIDATE**: 发射 ver=3 blob，逐 op 核对实际引擎==计划引擎、禁止回落 op 硬报错
4. 关键：US1 已含 T011（`fallback_allowed=0`→硬报错），故 MVP 就**没有"编译器决定了但运行期还能静默换"的中间态**

### Incremental Delivery

1. Setup + Foundational → 底子就绪（legacy 不受影响，可随时回归）
2. + US1 → 引擎指派 + 无静默替换（MVP，本会话核心痛点）
3. + US2 → 片上内存静态规划 + 显式搬移
4. + US3 → 完整执行报告 + 允许回落路径
5. 每阶段加完都能独立验证、不破坏前序

### 落地纪律（承接本会话既有结论）

- 改 matmul HMX 门时先逐 op 拆 `op_ts.bin` 的 engine×batched 字段，勿信 prof top-20（幸存者偏差）
- 设备侧改动需 `kernels/build_libs.sh one <lib>` 重编 + `gehtp setup` 推签名 lib + 复跑判分（cos/top1/wall 三门）
- 回 `task/prof-wp` 时用 `git merge --ff-only task/prof-wp-integ-fixes`（batched-HMX `afb56c7` 未混入其它提交）

---

## Notes

- [P] = 不同文件、无依赖，可并行
- [Story] 标签映射任务到用户故事，便于追溯
- 每阶段结束可在 Checkpoint 独立验证
- 避免：模糊任务、同文件冲突、跨故事依赖破坏独立性
- blob ver 与 op_ts.bin ver 是两套独立版本号（勿混）

### 1008 全新管线 E2E 复现（免补丁收口）

- 用户令：删全部中间产物、以当前树工具从头到尾编译上板 qwen3.5-0.8B，精度性能不回退，禁补丁硬编码。
- 全链路：hnnx_compile（--no-pass fc_bias_fold --opt-stats，13min，21139 节点）→ wtop_emit 三输入（ids i32 / mask **f16** / pos i32）→ v4.wtop（slots=1942 ops=14188，2.04GB）。
- 烘焙槽独立解析验证 SLOTS OK：slot1=32×f16 全 1.0（0x3C00）、slot2=i32 0..31、slot0 EXT_IN——与 v3g 实证好布局逐字节同构，纯 emit 产出零补丁。mask 必须喂 f16 文件：i32 的 1 按设备 f16 读出是 +0.0×2 → EQ(0,0)=1 → mask 全 -inf（v3f 同族）。
- ops 14188 vs v3g 16474：-2286 = pass 注册制默认关的融合边界（opt-stats: view_fold 2286/2286 仍跑，eltwise_const_fold 54），执行序变化 → f16 舍入路径差异。
- 设备 d0f1784 独占窗口（等 005 v17 空档）：[PASS] 14188 ops，wall **35.31s**（v3g 39.84s，-11.4%）；judge p0 cos **0.998642** / top1 **30/32**（v3g 0.9986454/30/32，Δcos=-3.4e-6 噪声带内）。三门不回退，通过。
- 产物：`.scratch_003_rebuild/`（tagged.bin/wtop_emit log/verify 脚本/上板脚本/prof_v4），`test_assets/qwen35_08b/qwen35_08b_v4.wtop`(+manifest) 与 output_003v4.f16.raw。
- p1-3 判分低分属无效对照（设备仅喂 p0 ids，v3g 同口径），非回归证据。
