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

- [x] T001 建 worktree `/disk1/gehtp-wt/013-executor-hoist` 分支 `013-executor-hoist`——实际基 **a57a4cd**（012 已收口 push，fetch 后 rebase d804e03；原计划 734986b 已过时）；主树无未提交 tracked WIP（012 已提交）
- [x] T002 三线基线复测并存档 `baseline/baseline-20261013.md`：gdn/pp 线**新鲜复跑**（wall 2751.8ms/md5 9d1c0877 位级同 012 锚/不变量 PASS，op_ts 28312B 留档）；f16 线文档锚（板无 blob，T012 复跑）；q4 线 25.7s 文档锚（R9 冻结分母）
- [x] T003 环境健康门：**ctest 48/49 预存红**（test_wtop_emit 5 子断言 = 65c4f2e 010 merge 引入，aab6371 复验 0 红——修复归 B 段 T006/T008）；examples 321/9 预存（012 同树当日复现）；gdn 线复跑位级命中 = 板路健康 ✓

**Checkpoint**: 基线锚在手，后续每段对拍有据

---

## Phase 2: US-B — W4A16 供给侧直连（Priority: P1）🎯 MVP

**Goal**: 编译器产三槽 + 9-arity 直读，W4A16 E2E 可达，删槽表扫描

**Independent Test**: 合成单 op blob(M=256)对拍 P5 已知答案 + 三线零回归(f16 位级 / gdn 双门)

- [x] T004 [P] [US1] 新建 `compiler/tools/requant_w4a16.cc`：Q4_0 GGUF(经 gguf_wtlink 读取层) → packed_w(K*N/2) + atbl(8*(K/32)*4, i*M*8) + otbl(8*(N/32)*4, K≠N 分列) + folded_bias((N/32)*512) 四槽；小矩阵对拍单测先行（契约=contracts/contract-w4a16-op.md 供给侧节）——**降级 2026-10-13**：atbl/otbl 闭式已落地+对拍全绿(ctest 50/50)；packed_w/folded_bias 因 oracle 已失(DFlashSpecCppRunner/htpw4a16_v81)降级为 requant_w4a16.h 尾注文档化 gap, T010 复用板上 oracle .raw
- [x] T005 [P] [US1] `kernels/include/op_spec.inc`：OP_MATMUL_W4A16 `arity=9` + 新增可选 `legacy_arity` 列(默认 0，本 opcode=6)——单点注册表纪律不回潮（R1）
- [x] T006 [US1] `compiler/tools/wtop_ops/op_matmul.cpp` + `wtop_emit.cpp`：W4A16 op 构造 6→9 参，args[6..8]=bias/atbl/otbl 槽引用(0x8000|s)；emit 供给侧接 requant 产物（依赖 T004,T005）
- [x] T007 [US1] `kernels/src/runtime/oplist_parse.c`：arity 校验放宽为 `n_args==arity || (legacy_arity && n_args==legacy_arity)`；不 bump WT_BLOB_VER（R2）
- [x] T008 [P] [US1] `compiler/tools/test_wtop_emit.cpp`：§9-arity 序列化→反序列化往返(全槽引用无损) + legacy 6 参新 parse 逐字节等价断言
- [x] T009 [US1] `kernels/src/runtime/oplist_exec.c` exec_matmul：`n_args==9` 直读三槽（槽长校验保留=emit 契约哨兵），**删 O(n_slots) 扫描**（仅 9-arity 路径；6-arity legacy 路径逐字节不动）（依赖 T006,T007）
- [x] T010 [US1] 合成 blob 设备门（003 T014 法）：单 W4A16 op 微 blob(9-arity) 上板 d0f1784，输出 == P5 dc_w4 已知答案位级；同源 6-arity 版双跑对照一致；私有名纪律——**实测 2026-10-13**：mk_w4_v9.py 把 blob_w4.wtop(v1 6-arity) 仅改 matmul op3 6→9 参(追加 args[6..8]=bias/atbl/otbl 裸 slot 2/3/4) → blob_w4_v9.wtop；私有 lib(013)+runner(013)+job(013) 上板，9-arity 与 6-arity 双跑输出 md5 均=7654a8f6 == dev_w4.f16.raw 逐字节一致(T009 直读 == legacy 扫描, 且 T005/T007 未破 legacy)
- [x] T011 [US1] W4A16 emit M 门槛（重定范围 2026-10-13）：emit 侧仅 `M%256==0` 才发 W4A16(opcode 1=OP_MATMUL_W4A16)，否则回落 f16（对齐 M<32→GEMV / 32≤M<256→dense_f16 路由）——避免真实 emit 产出 M=32/M=1 的 0xA002 失败 W4A16。原生 W4A16 的 0.8B E2E 收益断言显式移出 013 范围（M%256 NO-GO 裁决），B 段不设 w4 全图性能门
- [x] T012 [US1] B 段零回归门（US5）：f16 默认线输出 md5 == f87461e2（位级）；gdn 线 cos ≥0.9999(锚 0.999930，cos-min 门设锚下防假 FAIL) / top1 32/32；host ctest 全绿；B 段收益进 results——**实测 2026-10-13**：复用板上现存 blob（f16 `qwen35_08b_010.wtop` md5 3f857926 / gdn `qwen35_08b_fused_pp.wtop` md5 a7935fdf）+ 私有 lib_013/runner_013；f16 out_temp=2796（非 m4/v4 的 1620，010 blob 经更多 pass n_ops=11504）→ 输出 md5 **f87461e2** 位级、exec 5.98s；gdn out_temp=96 → cos **0.999930**/top1 **32/32**/md5 9d1c0877、exec 2.76s；host ctest **50/50**。零回归机理=B 段改动仅触 opcode 1，f16/gdn 无 W4A16 为死代码。证据 evidence/t012-*

**Checkpoint**: MVP 交付——W4A16 供给侧机制(4 表生成 + 9-arity + 扫描已删 + M 门槛)可达，合成 M=256 blob 验证；原生 W4A16 0.8B E2E 移出范围(M%256 NO-GO)。A/C/D 全不做也独立成立

---

## Phase 3: US-A — VTCM 静态布局一统（Priority: P1）

**Goal**: 单张编译期布局表，运行时零布局决策，探戈与全量 flush 消失

**Independent Test**: op_ts 实证全 VTCM flush=1 + f16 位级/gdn 双门对拍 B 末态

**依赖**: US-B（供给侧机制：扫描已删 + M 门槛已装，A 在其上做 VTCM 布局）

- [x] T013 [P] [US2] `compiler/tools/wtop_emit.cpp`：TEMPOFF 槽扩展布局段（R4 载体）——`[magic][ver][region_pin][region_arena][region_resident]` 各 {off,size}，三区互斥；驻留按消费引擎分域（HMX→VTCM / HVX,scalar→DDR，008 GATE3 铁律）；BudgetReport 门 `resident_pool+hmx_arena≤16MB` 编译期硬报错（契约=contracts/contract-vtcm-layout.md）——**实测 2026-10-13**：emit 在 TEMPOFF temp 表后追加 32B 布局段（`magic 'VTCL' 0x5654434C` + `ver 1` + pin/arena/resident 三区各 {off,size}），旧运行时 `len<16+n*12` 检查是 `<` 非 `==` 故尾随段向后兼容；test_vtcm_auto 三形态全绿（a_auto pin=0/arena=0/resid 尾倒 16MB；b_auto 同；c_auto pin=16384/arena_off=align2048），ctest 50/50
- [x] T014 [P] [US2] `kernels/src/runtime/hmx_common.c`：hmx_runtime_setup 增 (base,size) 区域参数，0/NULL=全量兼容旧义（R5）；17_w4a16_gemm 等旧调用点零改动编译通过——**实测 2026-10-13**：改签名会毁 ~20 个 `hmx_runtime_setup(2*1024*1024)` 旧调用点（base=(void*)2097152=UB），故新增独立 `hmx_runtime_setup_region(void* base, size)`：base=0→委托全量 setup，base!=0→attach 共享 scratch（同 hmx_runtime_attach 语义）；旧签名不动、旧调用点零扰动
- [ ] T015 [US2] `kernels/src/runtime/wtcache_impl.c`：wtcache 布局由 emit 布局表驱动（attach 共享 scratch 分区语义，003 T014 底座）
- [ ] T016 [US2] `kernels/src/runtime/oplist_exec.c`：装载期按表一次分区（此后零运行时布局决策）；删 hmx↔wtcache/驻留池 teardown/re-acquire 探戈；flush 收敛=装载后全 VTCM 一次 + re-carve 仅 arena 区段 flush + 驻留写 flush 保留（R6 正确性边界）（依赖 T013,T014,T015）
- [ ] T017 [US2] A 段门（US5）：对拍 B 末态——f16 线 md5 f87461e2 / gdn 线 cos ≥0.9999（原生 w4 线已移出范围 M%256 NO-GO）；op_ts 实证全 VTCM flush 次数=1、探戈零发生（trace 断言）；预算门在编译期拦截一次超限构造（负例测试）；A 段收益进 results

**Checkpoint**: VTCM 归属真空收口，双域/tile-major 解锁

---

## Phase 4: US-C — carve 静态布局表（Priority: P2）

**Goal**: carve 结果编译期序列化，热路径指针切换，哨兵挪装载期

**Independent Test**: matmul 族 op_ts 对照 + f16/gdn 位级 vs A 末态（W4A16 carve 机制走合成 M=256 blob）

**依赖**: US-A（布局基座）

- [ ] T018 [US3] `compiler/tools/wtop_emit.cpp`：carve footer 扩展序列化——每布局 {layout_id,M,K,N,n0,off_wt,off_bias,off_act,off_out,arena_hi}，去重键=(M,K,N,n0)，op 注解引用 layout_id；旧字段保留（超集扩展，003 双源对拍哨兵继续生效）（契约=contracts/contract-carve-layout.md）
- [ ] T019 [US3] `kernels/src/runtime/oplist_exec.c`：装载期逐 layout 重算 dc_w4_carve/w4_carve_need 对拍 emit 偏移（漂移→拒绝装载）；热路径=四指针直赋+arena_hi 查 emit 值，**删 carve_live_top 扫描与热路径 dc_arena_init/dc_w4_carve**（依赖 T018）
- [ ] T020 [US3] C 段门（US5）：f16/gdn 位级 vs A 末态（W4A16 carve 机制走合成 M=256 blob）；matmul 族逐 op 时长前后对照量化；手工破坏 emit 偏移一字段的 blob → 装载期硬报错（哨兵有效性）；C 段收益进 results

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
- [ ] T028 终局双门汇总 `specs/013-executor-hoist/results.md`（US5/用户验收口径）：精度不回退（默认/gdn cos 0.999930/32/32）+ 性能实测优化（默认线 ≤2.77s；B 段=供给侧机制无性能门；各段 op_ts 证据表；原生 W4A16 M%256 NO-GO 裁决归档）
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

1. Setup 三任务 → 2. US-B 全段(T004-T012) → 3. **STOP 停验证**：合成 M=256 blob 位级 + f16 线零回归 + gdn 双门 → 交付（W4A16 供给侧机制可达即独立价值）

### Incremental Delivery

B(MVP) → A → C → D → Polish；每段收口即独立可验收增量，中途停止不破坏已交付段。

---

## Notes

- [P] = 不同文件、无未完成依赖
- 每段收口门数值引用 contracts/ 原文，勿凭记忆改门（cos-min 假 FAIL：门设锚值之下）
- 板上资产全私有名（lib/runner/blob/结果名）；共享 op_ts.bin 拉取后按 size 自证
- 提交节奏：每任务或逻辑组一 commit；段收口门不过禁进下段
