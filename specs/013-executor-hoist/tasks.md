---
description: "Task list for 013-executor-hoist"
---

# Tasks: 013-executor-hoist

**Input**: Design documents from `/specs/013-executor-hoist/`

**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Organization**: P 前置 → US-B(供给侧) → US-A(VTCM 布局) → US-C(carve 序列化) → US-D(形态 flag, 硬前置 012) → Polish。US5(零回归纪律)不单列段——落为各段收口门任务(T012/T017/T020/T025/T028)。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- 门值引用 contracts/ 原文；每段收口 = 位级/双门 + op_ts 取证

---

## Phase 1: Setup（前置）

**Purpose**: worktree 隔离 + 三线基线取证 + 环境健康门

- [ ] T001 建 worktree `/disk1/gehtp-wt/013-executor-hoist` 分支 `013-executor-hoist` off `origin/task/prof-wp`(734986b)；确认主树 012 未提交 WIP(oplist_exec.c)不被触碰——改前查 [[gehtp-cross-session-coordination]]
- [ ] T002 三线基线复测并存档 `specs/013-executor-hoist/baseline/`：默认/gdn 线(cos 0.999930/top1 32/32/exec ~2.77s)、f16 非融合线(md5 f87461e2/exec 5.99s)、007 q4 线(cos 0.993484/exec 25.7s)；op_ts + 输出 md5 全留档
- [ ] T003 环境健康门：host ctest 49/49 + build_examples 337 pass + check_device_golden 可跑（任一红先修环境；0.8B 正典旗标清单确认含 --dma-runlist/--no-pass fc_bias_fold）

**Checkpoint**: 基线锚在手，后续每段对拍有据

---

## Phase 2: US-B — W4A16 供给侧直连（Priority: P1）🎯 MVP

**Goal**: 编译器产三槽 + 9-arity 直读，W4A16 E2E 可达，删槽表扫描

**Independent Test**: 合成单 op blob 对拍 P5 已知答案 + 0.8B w4 全图双门

- [ ] T004 [P] [US1] 新建 `compiler/tools/requant_w4a16.cc`：Q4_0 GGUF(经 gguf_wtlink 读取层) → packed_w(K*N/2) + atbl(8*(K/32)*4, i*M*8) + otbl(8*(N/32)*4, K≠N 分列) + folded_bias((N/32)*512) 四槽；小矩阵对拍单测先行（契约=contracts/contract-w4a16-op.md 供给侧节）
- [ ] T005 [P] [US1] `kernels/include/op_spec.inc`：OP_MATMUL_W4A16 `arity=9` + 新增可选 `legacy_arity` 列(默认 0，本 opcode=6)——单点注册表纪律不回潮（R1）
- [ ] T006 [US1] `compiler/tools/wtop_ops/op_matmul.cpp` + `wtop_emit.cpp`：W4A16 op 构造 6→9 参，args[6..8]=bias/atbl/otbl 槽引用(0x8000|s)；emit 供给侧接 requant 产物（依赖 T004,T005）
- [ ] T007 [US1] `kernels/src/runtime/oplist_parse.c`：arity 校验放宽为 `n_args==arity || (legacy_arity && n_args==legacy_arity)`；不 bump WT_BLOB_VER（R2）
- [ ] T008 [P] [US1] `compiler/tools/test_wtop_emit.cpp`：§9-arity 序列化→反序列化往返(全槽引用无损) + legacy 6 参新 parse 逐字节等价断言
- [ ] T009 [US1] `kernels/src/runtime/oplist_exec.c` exec_matmul：`n_args==9` 直读三槽（槽长校验保留=emit 契约哨兵），**删 O(n_slots) 扫描**（仅 9-arity 路径；6-arity legacy 路径逐字节不动）（依赖 T006,T007）
- [ ] T010 [US1] 合成 blob 设备门（003 T014 法）：单 W4A16 op 微 blob(9-arity) 上板 d0f1784，输出 == P5 dc_w4 已知答案位级；同源 6-arity 版双跑对照一致；私有名纪律
- [ ] T011 [US1] 0.8B w4 全图编译 + 设备双门：151 W4A16 层 + 36 N=16 回退层混合图；门 = cos ≥0.98 / top1 ≥30/32 / **exec ≤17.1s（≥1.5× vs 25.7s 锚，本段即断言）**；编译期 blob 尺寸 < CDSP 堆下限 2.04GB 硬门
- [ ] T012 [US1] B 段零回归门（US5）：f16 默认线输出 md5 == f87461e2（位级）；gdn 线 cos ≥0.9999(锚 0.999930，cos-min 门设锚下防假 FAIL) / top1 32/32；host ctest 全绿；B 段收益进 results

**Checkpoint**: MVP 交付——W4A16 E2E 可达 + 扫描已删，A/C/D 全不做也独立成立

---

## Phase 3: US-A — VTCM 静态布局一统（Priority: P1）

**Goal**: 单张编译期布局表，运行时零布局决策，探戈与全量 flush 消失

**Independent Test**: op_ts 实证全 VTCM flush=1 + 三线位级/双门对拍 B 末态

**依赖**: US-B（w4 线是 A 的受益验证负载）

- [ ] T013 [P] [US2] `compiler/tools/wtop_emit.cpp`：TEMPOFF 槽扩展布局段（R4 载体）——`[magic][ver][region_pin][region_arena][region_resident]` 各 {off,size}，三区互斥；驻留按消费引擎分域（HMX→VTCM / HVX,scalar→DDR，008 GATE3 铁律）；BudgetReport 门 `resident_pool+hmx_arena≤16MB` 编译期硬报错（契约=contracts/contract-vtcm-layout.md）
- [ ] T014 [P] [US2] `kernels/src/runtime/hmx_common.c`：hmx_runtime_setup 增 (base,size) 区域参数，0/NULL=全量兼容旧义（R5）；17_w4a16_gemm 等旧调用点零改动编译通过
- [ ] T015 [US2] `kernels/src/runtime/wtcache_impl.c`：wtcache 布局由 emit 布局表驱动（attach 共享 scratch 分区语义，003 T014 底座）
- [ ] T016 [US2] `kernels/src/runtime/oplist_exec.c`：装载期按表一次分区（此后零运行时布局决策）；删 hmx↔wtcache/驻留池 teardown/re-acquire 探戈；flush 收敛=装载后全 VTCM 一次 + re-carve 仅 arena 区段 flush + 驻留写 flush 保留（R6 正确性边界）（依赖 T013,T014,T015）
- [ ] T017 [US2] A 段门（US5）：三线对拍 B 末态——f16 线 md5 f87461e2 / gdn 线 cos ≥0.9999 / w4 线位级；op_ts 实证全 VTCM flush 次数=1、探戈零发生（trace 断言）；预算门在编译期拦截一次超限构造（负例测试）；A 段收益进 results

**Checkpoint**: VTCM 归属真空收口，双域/tile-major 解锁

---

## Phase 4: US-C — carve 静态布局表（Priority: P2）

**Goal**: carve 结果编译期序列化，热路径指针切换，哨兵挪装载期

**Independent Test**: w4 线位级 vs A 末态 + matmul 族 op_ts 对照

**依赖**: US-A（布局基座）

- [ ] T018 [US3] `compiler/tools/wtop_emit.cpp`：carve footer 扩展序列化——每布局 {layout_id,M,K,N,n0,off_wt,off_bias,off_act,off_out,arena_hi}，去重键=(M,K,N,n0)，op 注解引用 layout_id；旧字段保留（超集扩展，003 双源对拍哨兵继续生效）（契约=contracts/contract-carve-layout.md）
- [ ] T019 [US3] `kernels/src/runtime/oplist_exec.c`：装载期逐 layout 重算 dc_w4_carve/w4_carve_need 对拍 emit 偏移（漂移→拒绝装载）；热路径=四指针直赋+arena_hi 查 emit 值，**删 carve_live_top 扫描与热路径 dc_arena_init/dc_w4_carve**（依赖 T018）
- [ ] T020 [US3] C 段门（US5）：w4 线位级 vs A 末态；matmul 族逐 op 时长前后对照量化；手工破坏 emit 偏移一字段的 blob → 装载期硬报错（哨兵有效性）；C 段收益进 results

**Checkpoint**: 执行器 carve 面只剩指针切换

---

## Phase 5: US-D — 形态 flag 化（Priority: P2，硬前置 012 落地）

**Goal**: broadcast 形态/GEMV 引擎/Q4 槽判别编译期入 flag，运行时零重推导

**Independent Test**: f16 线位级 + seq1 17.70s 锚非回归

- [ ] T021 [US4] 前置检查：012 已落地主线（exec_matmul_f16_impl 分发段稳定）；否则本段挂起——B/A/C 收口不受影响
- [ ] T022 [P] [US4] `compiler/tools/wtop_ops/op_broadcast.cpp` + `wtop_emit.cpp`：broadcast form ∈ {1 整块,2 行循环,3 strided→计划标量} 入 op flags（判据函数=编译器侧同款，007 双侧互指契约）
- [ ] T023 [P] [US4] `wtop_emit.cpp` matmul 注解/flags：GEMV 走 planned_engine 编码（M<32 编译期已知）+ Q4_0 槽判别位（替代 len==640*tiles 探针，对齐 w_pp bit3 先例）
- [ ] T024 [US4] `kernels/src/runtime/oplist_exec.c`：注解路径直用 emit 形态（broadcast 删 shape-class 运行时重算、matmul 删 M<32 前置判与 Q4 探针）；双侧判据漂移检测挪装载期一次校验（哨兵不删只搬家）（依赖 T022,T023）
- [ ] T025 [US4] D 段门（US5）：f16 默认线 md5 f87461e2 位级；seq1/decode 线（GEMV 主场，17.70s 锚）非回归；broadcast 族 op_ts 不回退；D 段收益进 results

**Checkpoint**: 执行器热路径零运行时规划达成（SC-004）

---

## Phase 6: Polish & Cross-Cutting

- [ ] T026 删 legacy 6-arity 扫描路径（exec_matmul 内 n_args==6 分支）——先全量确认板上无私有 6-arity blob 依赖，删除后 ctest + 三线重跑
- [ ] T027 回合主线 merge：merge/013-into-mainline → origin/task/prof-wp；reconcile 面=012 的 HMX 会话外提段(exec_matmul_f16_hmx)与 013 B/A/C 段；merge 后 ctest 49/49 + 三线终验
- [ ] T028 终局双门汇总 `specs/013-executor-hoist/results.md`（US5/用户验收口径）：精度不回退（默认/gdn cos 0.999930/32/32；w4 ≥0.98/≥30/32）+ 性能实测优化（w4 线 exec ≤17.1s=≥1.5× vs 25.7s 锚，stretch 5.99s 仅记录不设门；默认线 ≤2.77s；各段 op_ts 证据表）
- [ ] T029 收口卫生：feature.json 还原 012（备份 /tmp/feature_json_012_backup.json）；更新跨会话协调表 memory；docs 侧如需补 W4A16 9-arity 说明挂 kernels/docs/

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup(P1)**: 无依赖，立即可开
- **US-B(P2=Phase 2)**: 依赖 Setup；**MVP 段**——可与 012 并行（B 主改 compiler 侧+exec_matmul(w4) 段，与 012 的 exec_matmul_f16_hmx 段不撞）
- **US-A(Phase 3)**: 依赖 US-B（w4 线=受益验证负载）
- **US-C(Phase 4)**: 依赖 US-A（布局基座）
- **US-D(Phase 5)**: **硬前置 012 落地**（同段 reconcile）；对 B/A/C 无依赖
- **Polish(Phase 6)**: 依赖各段收口；T026 额外依赖 D

### User Story Dependencies

- US1(B) → US2(A) → US3(C) 严格串行（受益链）
- US4(D) 独立于 B/A/C，只等 012
- US5(零回归) = 横切，落各段门任务

### Parallel Opportunities

- T004(requant) ∥ T005(op_spec) ∥ T008(test_wtop_emit) ∥ T013(emit 布局段) ∥ T14(hmx) —— 不同文件可并行
- T022 ∥ T023（D 段两 emit 面）
- 012 与本特性 B/A/C 并行（worktree 隔离）

---

## Implementation Strategy

### MVP First (US-B Only)

1. Setup 三任务 → 2. US-B 全段(T004-T012) → 3. **STOP 停验证**：合成 blob 位级 + 0.8B w4 双门 + f16 线零回归 → 交付（W4A16 E2E 可达即独立价值）

### Incremental Delivery

B(MVP) → A → C → D → Polish；每段收口即独立可验收增量，中途停止不破坏已交付段。

---

## Notes

- [P] = 不同文件、无未完成依赖
- 每段收口门数值引用 contracts/ 原文，勿凭记忆改门（cos-min 假 FAIL：门设锚值之下）
- 板上资产全私有名（lib/runner/blob/结果名）；共享 op_ts.bin 拉取后按 size 自证
- 提交节奏：每任务或逻辑组一 commit；段收口门不过禁进下段
