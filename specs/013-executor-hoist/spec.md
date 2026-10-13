# Feature Specification: 执行器上移 — 编译期接管运行时规划（B→A→C→D）

**Feature Branch**: `013-executor-hoist`

**Created**: 2026-10-13

**Status**: Draft

**Input**: User description: "执行器上移四段线(B→A→C→D)：B=W4A16 供给侧三槽；A=VTCM 静态布局一统；C=carve 静态布局表；D=形态 flag 化。终局双门=0.8B 精度不回退 + 性能实测优化。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - W4A16 供给侧直连（B，Priority: P1）

编译器在编译期把 W4A16 matmul 的 bias/atbl/otbl 三张供给表作为独立槽直接发射进 op 参数（arity 6→9），并落地从未存在的 requant 生成器（Q4_0 GGUF → 打包权重 + 两张 scale 表 + folded_bias）。执行器从 op 参数直读三槽，删掉现在每个 W4A16 matmul 对全槽表 O(n_slots) 的长度匹配扫描。此后 W4A16 路径 E2E 可达（此前编译器从不产三槽，运行时靠长度匹配兜底）。

**Why this priority**: 供给侧缺失是 010 US2 遗留的已知 blocker；B 是 A/C 的前置（无 w4 线则 A/C 的受益无法在真实负载上验证）；且是本特性性能主秀（W4A16 全图激活）的供给侧钥匙。

**Independent Test**: 合成单 op 微 blob（003 T014 法）对拍 P5 时代 dc_w4 已知答案；随后 0.8B 全图 W4A16 编译上板跑通双门（cos/top1/exec）。可独立交付：即使 A/C/D 全部不做，B 单独即交付 W4A16 E2E 可达 + 扫描删除。

**Acceptance Scenarios**:

1. **Given** 一份 Q4_0 GGUF 权重输入，**When** 编译 W4A16 线，**Then** 发射的 blob 含 packed 权重槽 + atbl/otbl（i*M*8 布局，K≠N 分列）+ folded_bias 三类供给槽，且 OP_MATMUL_W4A16 的第 7-9 参数直接携带三槽 id。
2. **Given** 一个 9-arity W4A16 op，**When** 执行器运行它，**Then** 三供给表指针由 op 参数直读获得，全程无对槽表的线性扫描。
3. **Given** 一个旧 6-arity v3 blob，**When** 同一执行器加载它，**Then** 解析与执行行为与现状逐字节一致（legacy 扫描路径保留至 D 段后删除）。
4. **Given** 0.8B 全图 W4A16 编译产物（151 个可 W4A16 线性层 + 36 个 N=16 回退层混合），**When** 上板运行，**Then** cos ≥0.98、top1 ≥30/32（q4 冻结门口径），且跑通即证明 E2E 可达。

---

### User Story 2 - VTCM 静态布局一统（A，Priority: P1）

编译器在编译期产一张覆盖全图的 VTCM 静态布局表（pin 区 / carve arena / 驻留池三区，驻留按消费引擎分域：HMX 消费→VTCM、HVX/scalar→DDR），并过 BudgetReport 三方预算门。执行器装载期开一次、静态分区；HMX 运行时 setup 从"持有全量 VTCM 从不释放"改为持有界区域；删掉 wtcache/carve/驻留池三方的 teardown/re-acquire 探戈；全 VTCM 16MB cache flush 从"每次 re-carve"降为装载期一次。

**Why this priority**: VTCM 归属是 002 起就存在的架构真空（编译期不规划、运行时抢全量），是双域/tile-major 的硬前置；删除 re-carve 全量 flush 对 w4 线与 gdn 线（块 carve）均有直接墙钟收益。

**Independent Test**: op_ts 实证 re-carve flush 次数从 N 次降为 1 次；三线（默认/gdn/w4）位级或 cos 门对拍 B 末态。可独立交付：布局表 + 有界区域改造单独成立。

**Acceptance Scenarios**:

1. **Given** 编译期产出的 VTCM 布局表，**When** 执行器装载 blob，**Then** VTCM 一次性按表分区，后续运行无运行时布局决策。
2. **Given** 任意 op 序列（含 HMX 与 HVX 交替），**When** 执行全程，**Then** 无 hmx_runtime_setup 与 wtcache/驻留池之间的 teardown/re-acquire 互斥回收发生。
3. **Given** 多次形状变化的 re-carve 事件，**When** op_ts 取证，**Then** 全 VTCM flush 仅装载期出现一次。
4. **Given** 布局表，**When** 编译期预算检查，**Then** resident_pool + hmx_arena ≤ VTCM 预算，超限在编译期硬报错而非运行时。

---

### User Story 3 - carve 静态布局表（C，Priority: P2）

编译器把每个 W4A16 matmul 的 carve 布局结果（e.wt/bias/act/out 在 VTCM 内的偏移 + per-op arena_hi）序列化进 blob，同 shape 布局去重共享。执行器热路径退化为"基址 + 编译期偏移"的指针切换；装载期一次重算 dc_w4_carve 对拍 emit 表，漂移在装载期硬报错（契约哨兵不删，只从逐 op 热路径搬家到装载期）。

**Why this priority**: 003 已定谳 carve 编译期闭式可算；但 C 依赖 A 的静态布局基座，且单独收益小于 A，故排后。

**Independent Test**: w4 线位级对拍 A 末态；matmul 族逐 op 时长 op_ts 对比量化。

**Acceptance Scenarios**:

1. **Given** 同 shape 的多个 matmul，**When** blob 发射，**Then** 它们共享同一 layout id（去重）。
2. **Given** 一个 W4A16 matmul op，**When** 执行，**Then** carve 指针由基址+emit 偏移直接得出，热路径无 carve_live_top 扫描、无 dc_arena_init/dc_w4_carve 计算。
3. **Given** emit 表与运行时重算结果不一致的 blob（手工拼接/异版工具产出），**When** 装载，**Then** 装载期硬报错拒绝执行，而非运行期静默错地址。

---

### User Story 4 - 形态 flag 化（D，Priority: P2）

broadcast 的形态/shape-class、GEMV 的引擎选择、Q4_0 槽判别从运行时逐 op 重推导改为编译期写入 op flag/注解。运行时在注解路径直接使用编译期结论；双侧判据漂移检测从逐 op 挪到装载期一次校验。

**Why this priority**: 纯卫生项，单点收益最小；且与 012 改同一 exec_matmul_f16 分发段，**硬前置 = 012 落地**。

**Independent Test**: f16 线位级对拍；seq1/decode 线（GEMV 主场，17.70s 锚）非回归。

**Acceptance Scenarios**:

1. **Given** 一个已注解的 broadcast op，**When** 执行，**Then** 形态直接取自 op flag，运行时不再调用 shape-class 计算。
2. **Given** M<32 的 matmul，**When** 计划引擎为 HVX，**Then** 直接走 GEMV，无 M<32 运行时前置判断。
3. **Given** emit 侧与执行器侧形态判据漂移的 blob，**When** 装载，**Then** 装载期一次性校验拦截。

---

### User Story 5 - 零回归纪律与终局双门（Priority: P1）

每一阶段（B/A/C/D）完成后，未触及的路径必须位级不变（f16 默认线输出 md5 f87461e2、gdn 线 cos ≥0.999930）；触及的路径必须过本段精度门。终局按用户口径验收：0.8B 精度不回退 + 性能实测优化（op_ts 逐段取证）。

**Why this priority**: 提速以结果损坏为代价毫无意义；"位级或锚值门"是本仓库既定验收语言。

**Independent Test**: 每阶段设备双门脚本 + 终局汇总表（三线 cos/top1/exec + 各段 op_ts 前后对照）。

**Acceptance Scenarios**:

1. **Given** 任一阶段收口，**When** 跑 f16 默认线全链，**Then** 输出与基线逐位一致（md5 f87461e2）；gdn 线 cos ≥0.999930 / top1 32/32。
2. **Given** 终局，**When** 汇总三线，**Then** 默认/gdn 线 exec ≤2.77s 不回退，w4 线 exec 较 25.7s 锚至少降 1.5×，且各段收益均有 op_ts 证据。

### Edge Cases

- 旧 6-arity v3 blob 与新库共存：parse 按 n_args 分派（9 直读 / 6 legacy 扫描），板上共享库同时服务两类 blob。
- w4 全图 blob 尺寸逼近 CDSP 堆上限（实测 2.04-2.89GB 之间）：编译期对 blob 尺寸硬门，超限报错。
- cos-min 边界假 FAIL：严格 >= 对锚值边界判假，回归门一律设在锚值之下（如 0.999930 门用 0.9999）。
- 012 并发同文件（oplist_exec.c）：worktree 隔离开发；D 段开工前 012 必须已落地；回合 merge 时 reconcile。
- W4A16 线中 36 个 N=16 线性层不可 W4A16（007 供给侧普查）：这些层回退 f16，混合图必须可编译可运行。
- VTCM 布局超预算（如 lm_head chunk 类大块）：编译期预算门硬报错并给出诊断，而非运行时失败。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 编译器 MUST 从 Q4_0 GGUF 权重生成 W4A16 三类供给槽（打包权重、atbl/otbl scale 表（i*M*8 布局，K≠N 分列）、folded_bias），布局与已验证的发射 4 表契约一致。
- **FR-002**: OP_MATMUL_W4A16 的 op 参数 MUST 从 6 扩到 9（第 7-9 位直接携带 bias/atbl/otbl 槽 id），arity 在 011 单点注册表登记。
- **FR-003**: 解析器 MUST 同时接受 6-arity 与 9-arity 两种 W4A16 op（按 n_args 分派），旧 v3 blob 的解析与执行行为逐字节不变。
- **FR-004**: 对 9-arity blob，执行器 MUST 从 op 参数直读三供给槽，热路径不得出现 O(n_slots) 槽表扫描。
- **FR-005**: 编译器 MUST 产单张全图 VTCM 静态布局表（pin 区 / carve arena / 驻留池），驻留按消费引擎分域（HMX→VTCM，HVX/scalar→DDR），并过 BudgetReport 三方预算门（超限编译期硬报错）。
- **FR-006**: HMX 运行时初始化 MUST 持有界 VTCM 区域而非全量，执行全程不得出现 hmx_runtime_setup 与 wtcache/驻留池之间的 teardown/re-acquire 互斥回收。
- **FR-007**: 执行器 MUST 将全 VTCM cache flush 收敛为装载布局时一次；re-carve 事件不再触发全量 flush。
- **FR-008**: 编译器 MUST 序列化 per-op carve 布局（e.wt/bias/act/out 的 VTCM 内偏移 + arena_hi），同 shape 去重共享 layout id。
- **FR-009**: 执行器热路径对 carve MUST 仅做指针切换；装载期 MUST 一次重算对拍 emit 表，漂移装载期硬报错。
- **FR-010**: broadcast 形态/shape-class、GEMV 引擎选择、Q4_0 槽判别 MUST 编译期写入 op flag/注解；注解路径运行时不得逐 op 重推导；判据漂移检测挪装载期（此段硬前置 = 012 落地）。
- **FR-011**: 每阶段收口 MUST 过零回归门：f16 默认线输出位级（md5 f87461e2）；gdn 线 cos ≥0.999930 / top1 32/32；w4 线本段精度门不回退。
- **FR-012**: 0.8B 全图 W4A16 线 MUST E2E 可达（可编译、可上板运行、出双门数值），精度门 cos ≥0.98 / top1 ≥30/32（q4 冻结门口径）。

### Key Entities

- **W4A16 供给槽组**: 打包权重槽 + atbl/otbl scale 表槽 + folded_bias 槽；由 requant 生成器产出，被 9-arity op 参数直接引用。
- **VTCM 静态布局表**: 编译期产出的全图 VTCM 分区描述（三区偏移与尺寸 + 分域归属），装载期一次性生效。
- **carve 布局记录**: 单个 matmul 形状的 VTCM 内偏移组 + per-op arena_hi；同 shape 共享。
- **三线基线锚**: 默认/gdn 线（cos 0.999930 / 32/32 / exec 2.77s）、f16 非融合线（md5 f87461e2 / 5.99s）、007 q4 线（cos 0.993484 / 25.7s）——全程回归对拍基准。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 默认/gdn 线精度不回退：cos ≥0.999930、top1 32/32，exec ≤2.77s。
- **SC-002**: w4 线性能实测优化：exec 较 25.7s 锚至少降 1.5×（≤17.1s），冲刺 f16 锚量级（5.99s）；cos ≥0.98、top1 ≥30/32。
- **SC-003**: op_ts 取证三项：re-carve 全 VTCM flush 次数 N→1；W4A16 matmul 族逐 op 时长前后对照；各阶段收益逐段可量化。
- **SC-004**: 执行器终态（9-arity blob）：热路径零运行时规划——无槽表扫描、无 carve 计算、无形态重推导、无 VTCM 归属仲裁。

## Assumptions

- D 段开工前 012 已落地（同文件同段 reconcile）；B/A/C 可与 012 并行（worktree 隔离）。
- 设备 d0f1784 + 私有名纪律（lib/runner/blob/结果名私有，共享 op_ts.bin 拉取后按 size 自证）。
- 0.8B 正典旗标全程使用（含 --dma-runlist、--no-pass fc_bias_fold 等，见 E2E 正典旗标记忆）。
- host 侧 ctest 基线 49/49；板侧 build_examples 337 pass 为环境健康门。
- w4 线精度门沿用 q4 冻结门（0.98/20→本特性改用 cos 0.98 + exec 门分离），不追 f16 线位级。
- feature.json 在本特性链期间指向 013，结束后还原 012。
