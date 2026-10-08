---
description: "Task list for 005-graph-identity-passes implementation"
---

# Tasks: 005-graph-identity-passes

**Input**: Design documents from `/specs/005-graph-identity-passes/`（spec.md / plan.md / research.md / data-model.md / contracts/cli-and-pass-contract.md / quickstart.md）

**Organization**: 按 User Story 分相（US1 view_fold P1 / US2 eltwise_const_fold P2 / US3 MatMul bias P3 分 3a+3b），每相独立可测可交付。测试任务按 spec SC-003 要求包含（正例/反例/幂等三图起步）。

**Format**: `- [ ] [ID] [P?] [Story?] Description`
- **[P]** = 不同文件且无未完成依赖，可并行
- **[Story]** = US1/US2/US3（Setup/Foundational/Polish 相无 Story 标签）

**硬纪律**（全任务有效，来自 plan.md Constitution Check）：
1. 恒等性：P1 位级 / P2 ≤1 ULP f16 / P3 与标量同口径
2. 幂等：二跑零改写
3. 逃生：`--no-pass view_fold|eltwise_const_fold|fc_bias_fold` 必须可关且编译成功
4. 协调：改 graph_prepare.cpp / op_matmul.cpp / oplist_exec.c 前查 `git status`（003 会话在途）；提交精确暂存（e4d1dbb 先例）
5. 设备：推 blob 前必查 `adb -s d0f1784 shell 'df -h /data/local/tmp'`（磁盘常满）

---

## Phase 1: Setup（共享骨架）

**Purpose**: 两个新 pass 文件落地并注册进 PassManager，构建绿、stats 可见、逃生门可用——后续所有 Story 在此之上开发。

- [x] T001 [P] 创建 `compiler/include/hnnx/opt/pass_view_fold.hpp` + `compiler/src/opt/pass_view_fold.cpp` 骨架：GraphPass{name="view_fold", phase=PHASE_1, enabled=true, matcher 恒 false}，加入 `compiler/CMakeLists.txt` 源列表
- [x] T002 [P] 创建 `compiler/include/hnnx/opt/pass_eltwise_const.hpp` + `compiler/src/opt/pass_eltwise_const.cpp` 骨架：GraphPass{name="eltwise_const_fold", phase=PHASE_1, enabled=true, matcher 恒 false}，加入 `compiler/CMakeLists.txt`
- [x] T003 在 `compiler/src/ir/graph_prepare.cpp` 的 `register_builtin_passes()` 注册两个新 pass（共享文件——动手前 `git status` 确认 003 会话状态）

**Checkpoint**: 容器构建成功；`ctest` 47/47；`--opt-stats` 出现 `view_fold=0/0 eltwise_const_fold=0/0`（stats_line 格式 `<name>=<m>/<r>` 不得改）；`--no-pass view_fold` 可接受。

---

## Phase 2: Foundational（阻塞性前置）

**Purpose**: US1/US2 的目标校准与共享代数件——必须先于所有 Story 完成。

- [x] T004 [P] 普查工具（prepare 后视角）：`compiler/tools/host_run.cpp` 加 `--dump-views` 调试旗标（或等价一次性 dump），统计 prepare 后图里 identity 视图数 / SS∘SS 对 / SS←Reshape 对 / Transpose∘Transpose 对 / Eltwise_Binary 按 opcode 的 const 操作数链数；产出报告入 `.scratch_a3/qwen35_08b_v2.wtop.work/`，修正 SC-001/盘子口径（net.json 口径仅上界，见 research.md R5）
- [x] T005 [P] slice 合成代数：在 `compiler/src/opt/pass_view_fold.cpp` 实现纯函数 `compose_slice(a, b) → SliceSpec ∪ {不可合}`——输入 = 四标量掩码（begin_mask/end_mask/new_axes_mask/shrink_axes int 位掩码）+ tensor_param const "ranges"（rank×3 i32，经 `find_param_const` + `read_param_u32` 读取，与 extract_strided_slice_extra 同源）；掩码语义完整消化（data-model.md SliceSpec 字段表；解析后无 0x80000000 哨兵残留）；**先读 oplist_exec.c slice 执行段锚定 ranges 三元组布局**（research.md R2 风险项）。配套单测入 `compiler/tests/test_opt_passes.cpp`：全 1 步长链 / 带 mask 链 / shrink 链 / 负步长链 / 跨 axis 不相交链，每例断言期望输出形状与索引

**Checkpoint**: 普查报告定盘；compose_slice 单测绿——US1 规则可直接消费。

---

## Phase 3: User Story 1 - view_fold 视图链消解 (Priority: P1) 🎯 MVP

**Goal**: StridedSlice/Reshape/Transpose 链合并或吸收，0.8B 消解 ≥1000 视图 op，设备 wall 降 ≥4%（基线 39.5s/16438 ops）。
**Independent Test**: 0.8B v2 重编译后 host_run 对拍折叠前后输出**位级一致**（视图变换纯索引重排，md5 必须相同）+ `--opt-stats view_fold=m/r` + ctest 全绿。

### Tests for User Story 1

- [x] T006 [US1] 在 `compiler/tests/test_opt_passes.cpp` 增 [13] view_fold 段（先写、先败）：SS∘SS 合并正例×2 / 中间张量旁路反例（单消费者门）/ 幂等二跑零改写；断言含 stats_line 子串

### Implementation for User Story 1

- [x] T007 [US1] SS∘SS 合并 + identity 删除规则（`compiler/src/opt/pass_view_fold.cpp`）：相邻 StridedSlice 经 T005 代数合成单 op；合成结果全区间恒等 → 改线删 op；单消费者门（中间切片被旁路不合）
- [x] T008 [US1] Reshape 吸收规则（同文件）：仅当 Reshape 是**纯 1 维增删**（元素数不变且非 1 维的维序不变）且单消费者 → SS 直接索引底层张量，Reshape 标 dead；跨维重排不吸收（research.md R3 判据）
- [x] T009 [P] [US1] Transpose∘Transpose 恒等对（同文件独立规则）：两置换复合为恒等 → 双删；非恒等不动（无通用合成发射路径）
- [x] T010 [US1] P1 集成门：0.8B v2 重编译 → host_run 对拍位级（quickstart.md §3）→ 上板 wall+op 数对比（quickstart.md §4，先查设备磁盘）→ ctest；SC-001/SC-002/SC-003 记录入 `.work/prof/`

**Checkpoint**: US1 独立可交付（MVP）——视图链消解上板见效。

---

## Phase 4: User Story 2 - eltwise_const_fold 同型合并 (Priority: P2)

**Goal**: `Mul(c1)∘Mul(c2)→Mul(c1·c2)`、`Add(c1)∘Add(c2)→Add(c1+c2)`、`Div(c)→Mul(1/c)`，0.8B ~155 链 + 36 div。
**Independent Test**: `--no-pass view_fold` 关闭 P1 单独验证：host 对拍 max|d| ≤ 1 ULP f16 + 幂等二跑 + ctest。

### Tests for User Story 2

- [x] T011 [US2] 在 `compiler/tests/test_opt_passes.cpp` 增 [14] 段（先写、先败）：同型合并正例 / Div→Mul 正例 / **除数为 0 或 denormal 反例（不触发，保留原 op）** / 旁路反例 / 幂等

### Implementation for User Story 2

- [x] T012 [US2] 合并 + Div→Mul 实现（`compiler/src/opt/pass_eltwise_const.cpp`）：同型 EB(c1)∘EB(c2) → 单 EB（新 const f64 运算后 cast 回原 dtype，经 append_node 入池、旧双 const 交 fixpoint DCE 回收——与 fc_bias bias 并入同构，data-model.md ConstScalar）；opcode 按数字枚举判同（0=ADD/13=MULTIPLY 已证，同型只信枚举值相等，2/18 于 T004 普查时 `scalar_str` 交叉验证一次）；Div→Mul 仅当除数 const 且非零非 denormal，1/c 用 f32 先算再入池；单消费者门
- [x] T013 [US2] P2 集成门：0.8B 编译 + host 对拍（≤1 ULP f16）+ 上板（quickstart.md §3/§4）+ ctest；SC-002/SC-003
  - 结果(2026-09-30): host 位级 n_diff=0; **设备开 Div→Mul 致行 28-31 NaN** → 拆独立 pass `eltwise_div_to_mul` 默认关(证据 .work/prof/US1-US2-evidence-20260930.md); 同型合并真图 54/0(无可合并对)

**Checkpoint**: US1+US2 均独立可用。

---

## Phase 5: User Story 3a - MatMul+bias 折叠推广·纯图级 (Priority: P3)

**Goal**: fc_bias_fold 匹配器接受 MatMul 链头 + 发射门放行，折叠 0-5 个非对齐形（正确性优先，性能归 3b）。
**Independent Test**: `--no-pass view_fold --no-pass eltwise_const_fold` 单独验证：ctest 全绿 + 0.8B `--opt-stats fc_bias_fold` 折叠数报告 + 上板无回退（SC-004）。

### Tests for User Story 3a

- [x] T014 [US3] 在 `compiler/tests/test_opt_passes.cpp` 增 [15] 段（先写、先败）：非对齐 MatMul+bias 正例 / 32 对齐反例（M%32==0 ∧ K%32==0 ∧ N%32==0 不折）/ batched 反例（flags&4 排除）

### Implementation for User Story 3a

- [x] T015 [US3] 匹配器扩展（`compiler/src/opt/pass_fc_bias.cpp`）：链头接受 `"MatMul"`（在既有 FullyConnected 旁）；沿用 HMX 形状门；排除 batched（设备硬错 `matmul_f16_bias 不支持 batched`）；op_data 写同一 `has_bias=1` 标记（contracts 表：FC 与 MatMul 链头同标记）
- [x] T016 [US3] 发射门放行（`compiler/tools/wtop_ops/op_matmul.cpp:40`）：`nm == "FullyConnected"` → `nm == "FullyConnected" || nm == "MatMul"`，has_bias 核实逻辑不变；**同时加硬报错**：32 对齐 MatMul 带 has_bias 标记发射 36 号 → 编译期 error（契约"违禁组合必须硬报错"，对齐既有 W4A16+bias 硬错风格）。**共享文件**：动手前 `git status` 查 003 在途
- [x] T017 [US3] P3a 集成门：0.8B 编译 + `--opt-stats` 折叠数（预期 0-5，注意力系 32 对齐形被门挡是**刻意**防倒退）+ 上板无回退 + ctest；SC-004；合成小 blob（非对齐 MatMul+bias）host_run 对拍手算

**Checkpoint**: US3a 安全合入——不撤 HMX 门、零设备语义改动。

---

## Phase 6: User Story 3b - HMX bias 上板（设备窗口，与 003 协调）

**Goal**: HMX 路径 GEMM 后 VTCM 内逐列加 bias，数值验证后撤 `want_hmx=0`，解锁 42 候选全折。
**Independent Test**: 合成 blob（32 对齐 + bias）上板，HMX vs 标量路径 cos ≥ 0.9999 且逐列 max|d| ≤ 2 ULP f16（SC-005）。

### Implementation for User Story 3b

- [ ] T018 [US3] 内核实现（`kernels/src/runtime/oplist_exec.c`）：`exec_matmul_f16_hmx` 加 bias 参数——GEMM 出 crouton 后、unpack 写回前，VTCM 内 f32 累加器逐列加（与标量路径同口径）；**协调**：该文件 003 会话在途（T014/T015 落地空窗再动，或走私有名隔离）
- [ ] T019 [US3] 合成 blob 上板验证：pack_oplist 造 32 对齐形 + bias 的 matmul（HMX 路径）对比标量路径输出；SC-005 达标后才撤 `if (bias) want_hmx = 0`（oplist_exec.c:1857）
- [ ] T020 [US3] P3b 集成门：0.8B 撤 MatMul 折叠的 HMX 门重编译 → 42 候选全折 → 上板 wall 对比（bias 上 HMX 后注意力 matmul 不再回落标量）→ SC-001 口径复测 + ctest

**Checkpoint**: 全 Story 完成，US3 收益兑现。

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T021 [P] 文档收尾：更新 `specs/005-graph-identity-passes/` 各文档状态与实测数字；跨会话协调表（memory/gehtp-cross-session-coordination）登记本线改动面
- [ ] T022 按 `specs/005-graph-identity-passes/quickstart.md` 五门全跑一遍（ctest → opt-stats → host 恒等对拍 → 设备门 → 逃生门），记录最终证据链
- [ ] T023 三 pass 叠加终验 + 提交：0.8B 全开编译对拍；`git status` 精确暂存（不吸收 003 在途改动，e4d1dbb 先例）；确认"前 16 行全零"共享设备问题时按 SC-006 口径记录（host 对拍 + op 数/wall 为准）

---

## Dependencies & Execution Order

### Phase Dependencies
- **Phase 1 Setup** → **Phase 2 Foundational**（普查校准 + 代数件）→ **Phase 3-6**：US1 → US2 → US3a → US3b 顺序交付（也可 US1∥US2∥US3a 并行——不同 pass 文件无冲突；US3b 硬依赖 US3a + 设备/003 窗口）
- **Phase 7** 依赖全部预期 Story 完成

### User Story Dependencies
- **US1 (P1)**: 依赖 Phase 2（T004 口径 + T005 代数）；最大盘子（~4s），MVP 主体
- **US2 (P2)**: 仅依赖 Phase 1 骨架 + T004 普查口径；可与 US1 并行
- **US3 (P3)**: 3a 仅依赖 Phase 1；3b 依赖 3a 全部 + 003 会话 oplist_exec.c 空窗
- US2/US3 与 US1 无相互依赖，各自 `--no-pass` 可独立关闭验证

### Within Each User Story
测试先写先败 → 规则实现 → 集成门（0.8B 上板）→ 下一 Story

### Parallel Opportunities
- T001 ∥ T002（不同文件）；T004 ∥ T005（普查工具 vs 代数）
- US1/US2/US3a 三 Story 相可并行（pass_view_fold.cpp / pass_eltwise_const.cpp / pass_fc_bias.cpp 三文件独立）
- T009（Transpose 规则）∥ T007-T008 同文件内先后，不并行

---

## Parallel Example: User Story 1 + US2 + US3a

```text
# Foundational 完成后三线并行（不同文件）:
Task: "T007-T009 view_fold 规则" → compiler/src/opt/pass_view_fold.cpp
Task: "T012 eltwise 合并"        → compiler/src/opt/pass_eltwise_const.cpp
Task: "T015 fc_bias MatMul 匹配" → compiler/src/opt/pass_fc_bias.cpp
# 共享文件收口点（不可并行，改前 git status）: T003/T016 graph_prepare.cpp / op_matmul.cpp
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)
1. Phase 1 骨架 + Phase 2 普查/代数
2. Phase 3 US1（view_fold——~4s 盘子，占全部收益大头）
3. **STOP 验证**: host 位级 + 上板 wall 降 ≥4% → 即可交付

### Incremental Delivery
1. US1 交付（MVP）→ 2. US2（+0.3-0.5s）→ 3. US3a（安全合入，折叠数可为 0）→ 4. US3b（设备窗口，解锁 42 候选）

### 风险窗口
- US3b 的 oplist_exec.c 与 003 会话冲突：安排在其 T014/T015 落地后空窗
- 设备磁盘常满 + job.txt 并发覆盖：推前 `df`、用私有 job 名（跨会话协调表纪律）

---

## Notes
- 每任务完成即 ctest 回归；改共享文件的任务（T003/T016/T018）提交前必查 `git status` 精确暂存
- 任务完成判据以 quickstart.md 对应门为准，证据（md5/cos/wall/op 数）落 `.work/prof/`
