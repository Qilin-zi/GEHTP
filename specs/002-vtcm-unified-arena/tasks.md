---
description: "Task list for VTCM 统一内存规划（N 分块 + 单 acquire 四区 arena + supertile 接线）"
---

# Tasks: VTCM 统一内存规划

**Input**: Design documents from `/specs/002-vtcm-unified-arena/`

**Prerequisites**: plan.md (required), spec.md (required)

**Tests**: 各 story 含验证门（host 契约测试 + golden 逐字节 + 设备上板），按 plan.md「验证门」节执行。

**Organization**: 按 user story 分组；US1 独立可交付（MVP），US2 门控 US3。

**阶段验收铁律（SC-008）**：每个 Phase 的 Checkpoint 都含 **0.8B 上板双门——精度（全模型 cos 不劣于基线，即 cos≈0.988039 / top1 26/32 无回归）+ 性能（wall 记录对照基线 275s / 上一阶段值）**，双过才算该阶段完成。

> ⚠️ **基线精度现实（2026-09-28 实测）**：本树 0.8B 现状基线 cos=**0.988039** / top1=**26/32**，**未达**判据脚本的黄金门（cos≥0.9999 且 top1 32/32）。这是 0.8B 线**既有的精度缺口**（根因未定位，见 [[gehtp-08b-select-gather-fix]] / [[gehtp-08b-accuracy-bisect]]），与本特性（VTCM 规划 + N 分块，纯性能项）正交。**本特性精度门的语义 = 不回归基线**（cos 保持 ~0.988、top1 保持 26/32），而非把 0.988 抬到 0.9999——后者是独立任务。若后续 N 分块/HMX 化意外改变精度（升或降），必须停下来查数值纪律（FR-013），不得静默放行。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无依赖）
- **[Story]**: US1=matmul N 分块 / US2=统一 arena / US3=supertile 接线
- 设备任务前置惯例：先 `adb shell 'ps -A | grep hexagon'` 查共享设备冲突；推签名版 lib（[[gehtp-p5-device-gate]] / [[gehtp-build-env]]）

## Path Conventions

- kernels: `kernels/src/runtime/`, `kernels/include/`
- compiler: `compiler/src/`, `compiler/tools/`, `compiler/include/`, `compiler/tests/`
- Spec artifacts: `specs/002-vtcm-unified-arena/`

---

## Phase 1: Setup（基线与去险设施）

**Purpose**: 锁定基线数据 + 搭好后续所有验证门要用的取证链

- [x] T001 抓取基线：0.8B 现状上板一轮（v5 blob + 签名 lib，同输入），落 `.scratch_002_baseline/`（勿入 git）——**这是所有阶段双门的对照锚**。实测：**wall=275009297us(≈275s)**、ops=16443、输出 15892480B sha256=`9421a855dd8a4404`、**cos=0.988039 / top1=26/32 / max|d|=3.7727**；code=24(MATMUL_F16) n=326 合计 254.7s=92.6% wall，最慢 op#16442(lm_head N=248320) 223.9s。取证链：`.scratch_002_baseline/prof/qwen35_08b_v5.{device.txt,op_ts.bin}`
- [ ] T002 [P] host 回归基线：`source env.sh` 后重编 test_wtop_emit（勿用残留容器二进制），确认全过；110 golden conv_add blob sha 记录（98cbad5c… 现状值）
- [ ] T003 [P] ~~确认 opcode 空位~~ 已核实：OP_PAD_F16=34 为枚举尾（oplist_parse.h:156），`OP_MATMUL_F16_SPLIT = 35`；WT_MAX_ARGS=16 ≥ 9 无结构障碍（oplist_parse.h:58）

**Checkpoint**: 基线数据在手，后续每个阶段结束都能对照

---

## Phase 2: User Story 1 - matmul 编译期 N 分块（Priority: P1）🎯 MVP

**Goal**: 超 VTCM 界的 f16 matmul 编译期 N 分块，全部走 HMX，数值逐字节恒等（五纪律见 plan.md US1）

**Independent Test**: 设备合成 blob **chunk-HMX vs 整算-HMX 逐字节对拍**（host SHA 门两侧都是标量，测不到要害）；0.8B 双门

### 契约测试（先写先挂）

- [ ] T010 [US1] test_wtop_emit 新增 SPLIT 契约段：发射端产出 arity 9 / opcode 35 / n0 递增覆盖 [0,N)；wt_parse 解析往返一致（compiler/tests/test_wtop_emit.cpp）——先跑确认 FAIL
- [ ] T011 [P] [US1] test_wtop_emit 新增旧兼容段：arity 7 的 OP_MATMUL_F16 blob 解析+执行行为不变

### 实现

- [ ] T012 [US1] kernels/include/oplist_parse.h：新增 `OP_MATMUL_F16_SPLIT = 35` + `#define WT_ARITY_MATMUL_F16_SPLIT 9` + 线格式注释 `[a_ref,w_s,out_t,M,K,N_full,n0,N_chunk,flags]`
- [ ] T013 [US1] kernels/src/runtime/oplist_parse.c：`arity_of` 与 `arg_is_slot`（idx 0,1 为引用位）各加 SPLIT case（:36/:69 附近）
- [ ] T014 [US1] compiler/tools/wtop_ops/op_matmul.cpp：HMX-eligible（非 batched + M/K/N 全 32 对齐 + f16 权重）且 `need(M,K,N) > 预算` 时，发射 ceil(N/N_chunk) 个 SPLIT op；act/wgt/out 三引用共享，每 chunk 只差 n0/N_chunk 两标量；预算口径 US2 前 = 运行时 VTCM total（与现状 need>vtcm_sz 同口径）；N_chunk 取 32 倍数（W4A16 未来若分块需 64 倍数，本特性不做）
- [ ] T015 [US1] kernels/src/runtime/oplist_exec.c exec_matmul_f16（:1536 起）：分派 SPLIT——① wgt pack 循环（:1481-1491）列下标加 n0（[K,N] 布局 stride=N_full；FC [N,K] 布局行基址加 n0×K）② unpack scatter（:1523-1530）改 `y[(r*32+m)*N_full + n0 + c*32+n]` ③ 标量回落（:1558 起）同加 n0 列偏。**五纪律对照**（plan.md US1）：每 tile clracc 保留、每 tile 重装 bias 页、k 升序不动、**回落全有或全无**（任一 chunk 失败 → 整 op 全部标量，禁止混合）
- [ ] T016 [US1] host 合成验证：N4096/K1024 + 边界（N_chunk=N、N=32、M=32）逐字节 vs 标量参考

### 设备验证（含 SC-008 双门）

- [ ] T017 [US1] **要害门**：合成 blob 同一 matmul 的整算版与 chunk 版双侧强制 HMX，设备输出逐字节对拍（plan.md US1 验证门 3）
- [ ] T018 [US1] 0.8B 重编译 + 上板：optrace 确认 6 个 N4096 op = WT_ENG_HMX、无 `matmul_f16 hmx vtcm overflow`；**双门：cos ≥ 基线判据 + wall 对照 T001**
- [ ] T019 [US1] round-trip 自检：`hnnx_compile → wtop_emit → wt_parse` 无错；110 golden 逐字节不变

**Checkpoint**: US1 独立交付——超界回落消除，旧 blob 零回归，**0.8B 精度+性能双门过**。此时可 commit。

---

## Phase 3: User Story 2 - VTCM 单 acquire 四区共存（Priority: P1）

**Goal**: 一次 acquire + 编译期四区照抄；f16 HMX 与 W4A16/驻留池共存，互斥回落消除

**Independent Test**: 0.8B 全模型上板 op70 后 f16 matmul 全 WT_ENG_HMX + 双门；micro-test 100× 循环无数值漂移

### 去险（先于实现，设备）

- [ ] T020 [US2] micro-test **回归确认**双 acquire fail 行为（双源已证：V81HexSim 记账模型 + GEHTP 设备史 WTC_ERR_VTCM_ACQUIRE，非 alias——见 plan.md 风险表）+ 验证 attach 模式不再触发二次 acquire
- [ ] T021 [US2] micro-test 100× open/close/setup/teardown + 逐次 cos：电源投票表耗尽、悬空指针、memset 污染三个面
- [ ] T022 [US2] micro-test T10 回归（`set_vtcm_param_v2(...,0,0)` 深核，wtcache_impl.c:127-128）+ **锁行为实验**（exclusive hmx_lock 持有时 f16 no-op 化的 HMX 指令正常执行，C2 设备面）+ **fence 负测试**（写 VTCM 后不 flush 直接 HMX 读，复现旧坑——sim 对此零覆盖，只能设备验）

### 契约测试（先写先挂）

- [ ] T023 [US2] test_wtop_emit 新增 TAG_MEM_PLAN v2 段：hnnx_compile 产出 ver=2 + 四区字段；wtop_emit 消费后 TEMPOFF 槽带扩展段；wt_parse 往返一致——先跑确认 FAIL
- [ ] T024 [P] [US2] v1 兼容段：ver=1 bin → wtop_emit → blob 与现状逐字节一致

### 编译期实现

- [ ] T025 [US2] compiler/src/ir/graph_prepare.cpp：四区尺寸求解——**插入点：do_prepare2_late step 6（:584-606）内、compute_ddr_offsets 调用之前**，先扫 op list：pin=PIN op 实际需求；carve=max over W4A16 ops 单 op tile 需求；**scratch=不动点 `max over HMX-eligible ops of need(M,K,N_c=1024)`**（N_floor=1024 保底每 chunk ≥1024 列，循环依赖的破解点，plan.md US2 §0）；resident=0（US3 回填）；四区偏移 2048B 对齐（对齐红线：vmem≥128B/HMX≥2048B）；总和 >8MB 时按 resident→scratch 顺序收缩 + stderr warn
- [ ] T026 [US2] graph_prepare.cpp:1591-1620 + :2375-2413：TAG_MEM_PLAN 序列化升 v2（插入 4×u32）+ 反序列化按 ver 分派（v1 旧布局行为不变）
- [ ] T027 [US2] compiler/tools/wtop_emit.cpp:465-492：TEMPOFF 槽尾部追加 4×u32 四区尺寸段（旧 blob 由 len 判定无扩展）

### 运行时实现

- [ ] T028 [US2] kernels/src/runtime/oplist_exec.c：新增 `wt_exec_arena_ensure()`——run_range 开头检测 TEMPOFF 扩展段，存在则一次 acquire（VTCM total + set_hmx_param + **exclusive hmx_lock** + memset + flush），登记 `g_exec.arena_base` + 四区基址/尺寸
- [ ] T029 [US2] oplist_exec.c:1771-1825 wt_exec_load_tempoff：解析扩展段（len 判定）；无扩展 → 现状懒路径
- [ ] T030 [US2] kernels/src/runtime/wtcache_impl.c：attach 模式——arena 已存在时跳过 query/acquire/memset/lock（:113-157 条件化），只登记基址；加 power_on static-once 守卫（:57 局部 ctx 改 static，FR-008a）
- [ ] T031 [US2] kernels/src/runtime/hmx_common.c：attach 模式——arena 已存在时 `g_vtcm_base=arena_base+scratch_off`、跳过 acquire（:111 幂等检查前接 hook）；memset 后补 `qurt_mem_cache_clean(FLUSH)`（:169，FR-008b，照抄 wtcache_impl.c:156-157）；**attach 模式下 teardown 只清 g_vtcm_base、不 release ctx**（U3）
- [ ] T032 [US2] **锁统一（C2）**：arena 模式下 `hmx_enable_execution`/`hmx_disable_execution`（hmx_common.c:202-218 的每-tile lock2 SHARED 对）退化为 no-op，`g_vtcm_ctx_id = arena ctx`；非 arena 模式维持现状逐 tile 加锁
- [ ] T033 [US2] oplist_exec.c 互斥 guard 改写：exec_matmul_f16_hmx（:1435）/ exec_conv2d_hmx（:622）——arena 模式改 `need > scratch_size → -1`；非 arena（旧 blob）保持现状 guard
- [ ] T034 [US2] oplist_exec.c:444 附近：W4A16 dc_arena_init 范围从 `vs-off` 收窄为 carve_size 硬界（防生长撞驻留尾区）
- [ ] T035 [US2] wt_exec_shutdown（:493-519）：arena 模式补 `arena_base=NULL` 清悬空（对照 vtcm_pool 清空的既有注释语义）

### 设备验证（含 SC-008 双门）

- [ ] T036 [US2] 0.8B 重编译（v2 路径）+ 上板：op70 后 f16 matmul 零回落（optrace 全 WT_ENG_HMX）；**双门：cos ≥ 基线判据 + wall 对照 T001/T018（预期减半量级，SC-004）**
- [ ] T037 [US2] 回归：110 golden 逐字节不变（v1 路径未动，SC-005）；旧 blob（无扩展段）在新 lib 上行为同现状

**Checkpoint**: 互斥回落消除 + US1 叠加 → 86.9% wall 大头回收，**0.8B 精度+性能双门过**。此时可 commit。

---

## Phase 4: User Story 3 - supertile 账本接线（Priority: P2）

**Goal**: supertile-interior temp 编译期驻留 VTCM 尾区，省 DDR round-trip

**Independent Test**: elementwise 链图编译出 0x4000 驻留 temp，设备 temp_get 返回 VTCM 指针；0.8B 双门（cos 不变 + wall 对照 US2 后数值）；110 golden 逐字节不变

### 契约测试（先写先挂）

- [ ] T040 [US3] test_wtop_emit 新增驻留段：含 supertile 的合成图 → TAG_MEM_PLAN entries 带 in_vtcm → blob TEMPOFF 带 0x4000 temp → wt_parse 往返一致——先跑确认 FAIL

### 实现

- [ ] T041 [US3] **成本模型修数（FR-014，先于一切接线）**：compiler/src/vtcm/supertile.cpp:81 的 48GB/s 硬编码改分档实测值——spill 读回 37.5 / 写出 56.0 / DMA 59.9+860cyc 启动（V81HexSim 器件标定）；`ddr_saved` 按流类型分档计价；常数集中到画像节式单点（仿 DD-09 纪律）
- [ ] T042 [US3] compiler/include/hnnx/vtcm/supertile.hpp + src/vtcm/supertile.cpp：`SuperTileSolver` 加 `tiles()` accessor；graph_prepare.cpp:4784-4785 create_supertiles 末尾拷入 `GraphPrepare::supertiles_`（G1 开闸，纯机械无行为变化）
- [ ] T043 [US3] supertile DP 预算修正（G2 账本）：dp_group_graph.cpp:327-346 / supertile.cpp:61-75 的预算入口先扣 scratch+pin+carve 再跑 can_merge
- [ ] T044 [US3] compiler/src/vtcm/ddr_offsets.cpp:83-107：两级分配前注入——supertile-interior temp（output 被同组消费且非图输出）预分配驻留区固定偏移、entries 打 in_vtcm=true；预算不足整组回退 DDR + stderr warn（不静默）
- [ ] T045 [US3] 预算统一（G6）：hnnx_compile `plan_vtcm_budget_` 默认逻辑——含可驻留 supertile 时自动设为 resident+scratch 需求；supertile DP 的 vtcm_size_ 硬编码 3MB（graph_prepare.cpp:3887-3896）改为 8MB−scratch−pin−carve 口径
- [ ] T046 [US3] compiler/tools/wtop_ops/wtop_ops.hpp:156-158：fresh_temp silent-drop 命中 ddr_temp_vtcm 候选时 stderr 告警（FR-012）
- [ ] T047 [US3] resident_size 回填：TAG_MEM_PLAN v2 的 resident_size = Σ 驻留 temp（128B 对齐）（T025 留的 0 在此接通）

### 设备验证（含 SC-008 双门）

- [ ] T048 [US3] elementwise 链图上板：rtrace/optrace 取证驻留 temp 的 temp_get 返回 VTCM 指针；**0.8B 双门：cos 不变 + wall 对照 T036**；**回归 AS4 场景：带 0x4000 但无 W4A16 op 的 blob 在 arena 模式照常工作**
- [ ] T049 [US3] 预算不足回退路径：构造驻留需求超界的合成图，确认 warn + DDR 回退而非错数据
- [ ] T050 [US3] 110 golden 逐字节不变（无 supertile 图零回归）

**Checkpoint**: 三 story 全绿，**0.8B 精度+性能双门过**。此时可 commit + push。

---

## Phase 5: Polish & 收尾

- [ ] T051 [P] 更新 docs/GEHTP_PARALLEL_TASKBOOK.md 战役日志（本特性三阶段结果 + 各阶段 wall 数据）
- [ ] T052 [P] 更新记忆文件：[[gehtp-vtcm-ownership-gap]] 标记已修复、[[gehtp-vtcm-coexistence-plan]] 标记被本方案取代、[[gehtp-v81hexsim-asset]] 补实施回链
- [ ] T053 全量回归：test_wtop_emit + 110 golden + 0.8B 上板三连，确认 SC-001~008 全绿
- [ ] T054 commit + push origin task/prof-wp（SSH 直连，[[gehtp-github-access]]）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，立即开始
- **US1 (Phase 2)**: 依赖 Phase 1 基线；**与 US2/US3 无代码依赖，可独立交付**
- **US2 (Phase 3)**: 依赖 Phase 1；建议 US1 完成后开始（US1 的分块预算在 US2 后切换到 scratch_size，顺序做避免返工）
- **US3 (Phase 4)**: **硬依赖 US2**（驻留池只有在无人 memset 全量时才存活）；T041 成本修数是 T043/T044 的前置（错误带宽会让 DP 做错决策）；T047 依赖 T025（v2 格式）+ T044（驻留需求）
- **Polish (Phase 5)**: 依赖全部

### Within Each Story

- 契约测试先写先挂（T010/T011、T023/T024、T040）
- 编译期（emit/parse 格式）先于运行时（exec）
- host 验证过才上板；设备任务先查共享冲突
- **每阶段收口必过 0.8B 双门（精度 cos + 性能 wall）**

### Parallel Opportunities

- T002 / T003 可并行
- T011 与 T010 可并行（不同测试段）
- T024 与 T023 可并行
- T020 / T021 / T022 三个 micro-test 编写可并行（同一设备执行需串行）
- T051 / T052 可并行

## Implementation Strategy

### MVP First（US1 独立交付）

1. Phase 1 基线 → Phase 2 US1 完成 → **停下来过双门**（T017/T018）
2. 单独 commit：超界回落消除，风险最低的一步先落袋

### 增量交付

1. US1 → commit（MVP）
2. US2 → commit（大头收益：互斥消除；micro-test 先行）
3. US3 → commit（驻留收益；T041 修数先行）
4. Polish → push

### 风险集中点

- T020-T022 是全套件的风险阀门：锁行为与 fence 负测试若暴露问题，回退到 [[gehtp-vtcm-coexistence-plan]] 的时间复用方案（plan.md 已注明），US3 随之取消
- T033 的 guard 改写是行为分歧点：arena/非 arena 两路径必须都回归
- **scale 页 ob 字段位置未验证**（sim=128+4o vs GEHTP conv=4n+2 互斥）：本特性不受影响（恒等页 ob=0），但任何后续 ob 融合 bias 工作必须先补器件探针

---

## Notes

- 每完成一个 Checkpoint commit 一次；设备验证任务遵循 [[gehtp-p5-device-gate]] 惯例（签名 lib、查共享、备份 job.txt）
- kernels 改动 host 验证：先 `source env.sh` 再 build_libs.sh；compiler 单 target host 重编再跑，勿信残留容器二进制（[[gehtp-build-env]]）
- 数值门一律逐字节/cos 判据，不接受「看起来对」
- V81HexSimNew 仿真器（105 服务器，本机副本 /disk1/scratch_v81sim/）可作成本常数与数值语义的对照源；fence/DMA 竞态它不能背书（[[gehtp-v81hexsim-asset]]）
