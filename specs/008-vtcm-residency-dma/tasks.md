---
description: "Task list: 全图 VTCM 驻留 + 显式 DMA 流水（008 中期性能线）"
---

# Tasks: 全图 VTCM 驻留 + 显式 DMA 流水（008）

**Input**: [spec.md](spec.md) / [plan.md](plan.md)（必读）/ [research.md](research.md)（D1-D8 代码实查，机制现状矩阵）/ [data-model.md](data-model.md) / [contracts/contracts.md](contracts/contracts.md)（C1-C5）/ [quickstart.md](quickstart.md)（4 个验证场景）

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: 契约测试任务按 house 惯例纳入（test_wtop_emit.cpp 范式，非 TDD 流程）

**Organization**: 按 user story 分阶段独立可测；US1 = MVP。**库纪律**：改共享文件前 `git status`（协调表重叠段：oplist_exec.c↔004、wtop_emit.cpp↔007、graph_prepare.cpp↔003 区、op_matmul.cpp↔007/009）；板上 d0f1784 私有名+互斥；判据/evidence 入库不入 /tmp。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: US1-US4（Setup/Foundational/Polish 无标）

---

## Phase 1: Setup（共享基础设施）

**Purpose**: 隔离工作树 + 锚与设备登记

- [x] T001 建 worktree `/disk1/gehtp-wt/008-vtcm-residency-dma`（分支 `008-vtcm-residency-dma` off task/prof-wp tip；照 004/007 先例；开工前主树 `git status` + 读跨会话协调表 008/009 行）
- [x] T002 [P] 锚与设备登记 → `specs/008-vtcm-residency-dma/evidence/anchors.md`：d0f1784 getenforce Permissive 预检、pgrep run_main 互斥、当前设备库指纹、基线 blob（007 交卷锚；未交卷则 v4 35.31s 口径 + 回退预案标注）

---

## Phase 2: Foundational（预算基线冻结，无代码改动）

**Purpose**: 把 spec 最大风险（工作集预算）变成冻结数字；阻塞 US1 阈值校准

- [x] T003 现编译器重发 v4 同图（发射配方照抄 `.scratch_003_rebuild/`（run_v4_board.sh / wtop_emit_v4.log 参照），只采集不改造），stderr 抓 `[ddr] TAG_MEM_PLAN: cap= vtcm_cap= n_temps= n_spills=` 与 `[ddr] TEMPOFF slot: cap= vtcm_cap= n_temps= (vtcm=)` 两行 + `emit_ddr_offset_table` TSV → 冻结入 `evidence/budget-baseline.md`。**若两行仍不出现 = has_mem_plan 旁路根因未消**，把诊断（序列化缺环/pass 未跑）写成 T005 的直接输入
- [x] T004 [P] census `[vtcm]` 1107 op 语义重导（`scripts/gehtp_prof.py` + manifest：确认 = `0x4000|temp` 引用旗标层标注而非池驻留事实），结论并入 budget-baseline.md 防后续误读

**Checkpoint**: 盘子口径冻结（resident 可用额度 / n_temps / spill 规模量级）；US1 判据可校准

---

## Phase 3: User Story 1 - 全图激活 VTCM 驻留 (Priority: P1) 🎯 MVP

**Goal**: 启用+扩面库内「第 7 步」静态 temp 规划，预算三方单源化，coverage ≥80%，f16 期位级不动
**Independent Test**: quickstart 场景 2 —— emit 出 residency-report、新 blob 板跑输出与基线逐字节 md5 一致、coverage ≥80%、预算门负例硬报错

### Implementation for User Story 1

- [x] T005 [US1] mem_plan 激活：`compiler/src/ir/graph_prepare.cpp` 诊断并修复 0.8B 图 `has_mem_plan()==false`（research D1：v4 emit 日志 88 行无 [ddr] 行 = 整条旁路）——重发后 stderr 出现两行 [ddr]
- [x] T006 [US1] 预算单源化：`compiler/tools/wtop_emit.cpp` 加 `--vtcm-budget N`（默认 16777216）透传 GraphPrepare，**弃** `get_vtcm_tile_size()` 陈旧源（graph_prepare.cpp:476-477，budget×0.75）；C2 惰性门 = 不加旗标时编译/emit md5 与现库同图一致
- [x] T007 [US1] 驻留标记扩面：`compiler/src/vtcm/ddr_offsets.cpp` 非 matmul temp（eltwise/broadcast/reshape 产物）进 lifetime 打包 + `in_vtcm` 判据（大小/生命期阈值可配；未驻留项进 spill 名单含 reason——显式降级非静默）。**T010 校准修正**：`in_vtcm` 判据必须按消费引擎分域（HMX 消费→VTCM / HVX·scalar 消费→DDR），无差别驻留实测净负 +11.5s（见 `evidence/budget-baseline.md` GATE3 表）— **落地实证：coverage 0.8395（SC-001 ≥80% PASS，非 matmul temp 已入 lifetime 打包）**
- [x] T008 [US1] 三方全局预算门 + `--residency-report PATH`：`compiler/tools/wtop_emit.cpp` 校验 `resident+staging×2+carve_peak ≤ budget`（超限硬报错，错误文本含三项数值）；落盘 BudgetReport（resident/staging/carve/slack 四行）+ spill 名单 + coverage（temp 字节加权）。**TEMPOFF 槽保持唯一真相源**（沿 :694-716 carve 一致性校验，禁第二套表）
- [x] T009 [P] [US1] 契约测试 `compiler/tests/test_wtop_emit.cpp`：预算门负例（`--vtcm-budget 1048576` → 硬报错退出）/ TEMPOFF 追加字段 round-trip / 惰性门（无新旗标产物 md5 同）
- [x] T010 [US1] 板跑三门（quickstart 场景 2）：GATE1 位级 PASS（md5 `f87461e2…` == v4 锚，003/007 双 lib）／ GATE2 PASS（coverage 0.8395）／ GATE3 校准完成（07lib A/B：无驻留 24.24s vs 全图驻留 37.65s，ΣΔ 闭合；**全图无差别驻留净负，HMX +2% / HVX 广播 43×**）——GateRecord 冻结值见 `evidence/budget-baseline.md`（机理候选留 US2 V81 迹重放裁决）

**Checkpoint**: SC-001 + SC-004(f16 列) + FR-001/002/003 达成；US1 独立成立

---

## Phase 4: User Story 2 - tile 级 DMA 双缓冲 + 权重预取 (Priority: P2)

**Goal**: 搬运藏进计算（HMX 族 wall ≤ 锚 50%），dma_us>0，失败显式回落
**Independent Test**: quickstart 场景 3 —— fence micro-test 证据 + op_ts 双时间轴 + 位级门 + 回落负例偏差不变量 PASS

### Implementation for User Story 2

- [x] T011 [US2] **fence 并发 micro-test（硬前置闸）**：设备端 HMX matmul 与 `dc_dma` 搬运并发 ≥100 轮、双向 fence（fence.c 决策表：DMA→HMX@VTCM=INVALIDATE / HMX→DMA=FLUSH），输出逐字节比对 + 超时看护 → `evidence/fence-microtest.txt`（含库指纹）。**未过门 T012-T015 一律不上板**（V81 不背书此竞态）
  **PASS（1009, d0f1784, libhvxhmx_007 md5 5a023fbb）**：`kernels/examples/47_fence_conc` 1000 轮 7/7 门×4 复跑全绿——逐字节==串行参考、看护 max_poll 16µs、overlap 31µs ≤ 0.9×(hmx 28+dma 14) 真并发实证（DMA 全隐藏，链 ~55GB/s）。**附带定谳：阻塞 `dc_dma_once` 每调用 ~700µs 固定开销 → T012/T013 发射器必须用 desc 链一次 submit+晚 poll 范式**（见 evidence 判读 4）
- [x] T012 [US2] `--dma-runlist` 扩全图：`compiler/tools/wtop_emit.cpp` + `compiler/tools/wtop_ops/op_matmul.cpp` 主力 HMX 族（in_proj/out_proj/lm_head chunk）权重/激活按执行序预取发射；staging 槽在 T008 预算表内分配（不得与驻留激活争额）。**探针修正（1009，`evidence/vtcm-access-probe.txt`）**：DMA 非瓶颈（55GB/s），pack_wgt 标量散写=66.4% wall → 机制改 **f16 权重编译期预打包 crouton 进 blob（W4A16 预置发射同款）+ 运行期 DMA DDR→VTCM 线性预取**（7MB≈130µs 藏进前一 op 计算）；unpack 标量散读 5.59ms/op 并档 HVX 化（007 P1 范式，G7 已证 HVX 线性读 VTCM=快侧 9.4×）
- [x] T013 [US2] 双缓冲轮换：DmaStreamDesc `buf_slot 0/1` + 发射序「tile i+1 搬运发射 → tile i invoke → completion poll」；任一环失败回落串行 + deviation 记录（003「允许回落+上报」契约，绝不静默）
  **数据裁决 NO-GO（1010）**：T012 板跑实测 Σdma_us=22.53ms = **0.17% wall**（完美双缓冲的收益上界）；MATMUL dma/cpt 0.6%、SPLIT 1.7%（T014 重叠率列）；「~700µs/调用固定开销」在真图不成立（22.53ms 已含全部阻塞开销，325 matmul 均 69µs）。回落+上报义务由 T015 搬运面回落兑现。证据 `evidence/t013-dmapipe-datadriven-nogo.md`（006 census 同款先例）
- [x] T014 [P] [US2] op_ts 双时间轴报告：`kernels/examples/42_gehtp_runner/main.c`（dma_us 落盘已有 ver3）+ `scripts/gehtp_prof.py` 加 HMX 族重叠率列（dma 在途 ÷ compute dur）
  **PASS（1010）**：prof per-opcode 聚合表加 Σdma_us + dma/cpt 列，DMA 实测行加「占wall=T013 可隐藏上界」；ver3 旧档解析回归 PASS。顺带 ver=4（+stg_fb，T015 三腿不变量）。证据 `evidence/t014-dual-timeline-overlap.txt`
- [x] T015 [US2] 板跑（quickstart 场景 3）：dma_us>0 ／ SC-002 HMX 族 dur ≤ 50% 锚 ／ 位级 md5 同 ／ 构造 staging 失败 → deviation=1 + 串行继续（偏差报告不变量 PASS）
  **PASS（1010, d0f1784, libhvxhmx_008t15 signed md5 8fcf91b9）**：双跑四门全绿——A(fault=0) Σdma 22.74ms/0 搬运面回落/不变量 PASS/md5 f87461e2/wall 13.04s；B(fault=1) [fault] mask=1 + [stg] 搬运面回落 182 次 dev=1/md5 同/不变量 PASS。op_ts 记录升 ver=4（+stg_fb 三腿不变量 `dev==0 ⟺ planned==actual ∧ stg_fb==0`）。证据 `evidence/t015-staging-fallback-boardrun.txt`（含三元 init 反转 bug 教训）

**Checkpoint**: SC-002 + FR-004/005 达成

---

## Phase 5: User Story 3 - 残余搬运消化 (Priority: P3)

**Goal**: STRIDED_SLICE/UNARY(reshape)/TRANSPOSE_GEN 三族合计 ≤2s，census 驱动留证据
**Independent Test**: US1 落地后重跑 census → digest；三族门 + 位级（消除类）/计划=实际（HVX 类）

### Implementation for User Story 3

- [x] T016 [US3] US1 落地后重跑 census（`scripts/gehtp_prof.py --csv`）→ `evidence/census-digest.md`（三族新盘子；spec 表内 7-8s 投影作废，以实测为准）
  **PASS（1010）**：wall 13.04s 实测，三族合计 **7.246s = 55.6%**（STRIDED_SLICE 3.06s/UNARY 2.90s/TRANSPOSE 1.28s）。有效带宽 ~20MB/s = 标量非线性爬行（峰值 0.03%）。UNARY 92.3% 是 Reshape（2.594s 可折叠）；TRANSPOSE 恒改字节序全归 HVX。≤2s 门取决于 STRIDED_SLICE stride 分解（T017 第一步补 contiguous 计数）。证据 `evidence/census-digest.md`
- [x] T017 [US3] 驻留面零拷贝折叠：纯 reshape/slice（不改字节序）→ emit 期引用消除（opt-stats 计数，op 消失；位级一致）
  **已收口（1010, commit b630974/422ab5a, 板跑 md5 f87461e2 PASS）**：Reshape 引用消除零拷贝折叠 + slice stride 分解 census。exec wall 13.04s → 10.40s（UNARY 92.3% Reshape 折叠后 2.594s 归零）。T018 起盘三族 = SLICE 3.05s / TRANSPOSE 1.28s / UNARY 0.34s。
- [x] T018 [US3] 剩余物理搬运 HVX 化：`kernels/src/runtime/oplist_exec.c`（`hvx_copy_f16_uu`/分段 memcpy，007 P1 broadcast 三形态范式：块拷贝循环+尾部标量；emit assign_engine 计划=实际，dev=0）
  **STRIDED_SLICE 已收口（1010, 板跑 md5 f87461e2 PASS）**：`exec_slice` 最内轴 stride==1 → 逐行 `hvhx_v2_copy_f16` 块拷贝（odometer 免 div/mod）；emit 侧 `op_strided_slice` 双分支对 `ns[nrk-1]==1` 置 HVX 注解（孪生判据）。板跑 code=15 **3.05s → 0.082s（37×）**，三族合计 **1.69s ≤2s 门已达**（TRANSPOSE 1.28s + UNARY 0.33s 待 T018b/c 继续）。exec wall 10.40 → 7.42s。不变量 PASS（按计划 11399/允许回落 105=UNARY 预置虚报/中止 0，无 slice 回落）。
  **TRANSPOSE 已收口（1010, commit 425a04d/eaa9a72, 板跑 md5 f87461e2 PASS）**：`exec_transpose_gen` 最内轴固定（p[rk-1]==rk-1，96/175 op）→ 逐块 `hvhx_v2_copy_f16`（pinv 映回输入坐标，odometer 免 div/mod）；emit 侧 `op_transpose` 对 `np[nrk-1]==nrk-1` 置 HVX 注解（孪生）。最内轴真置换（64↔128 36op）走通用标量回落 odometer 化（输出线性连续写+跨步读，免 per-element 4 div+4 mod）。板跑 code=27 **1.28s → 0.143s（9×）**。三族合计 0.56s。exec wall 7.42 → 6.31s。UNARY 0.33s 残留（SQRT 36/SOFTPLUS 18 为 NaN 守卫/公式差异刻意标量，NEG 30 可 HVX 但收益 ~0.04s 忽略）记 T018c 残留。
- [x] T019 [US3] 板跑：三族合计 ≤2s + digest 归档；不达标则 digest 指认残余根因（不硬凑门，FR-006）
  **PASS（1010, 板跑 md5 f87461e2）**：三族合计 **0.560s ≤ 2s**（STRIDED_SLICE 0.082s + TRANSPOSE 0.143s + UNARY 0.335s）。exec wall **6.31s ≤ 8s**（007 锚 24.24s → 6.31s，3.84×）。不变量 rc=0（无未裁决引擎偏差）。digest `evidence/t018-three-family-digest.md`。

**Checkpoint**: SC-005 前半达成

---

## Phase 6: User Story 4 - per-op 开销批量化 (Priority: P4)

**Goal**: 逐 op 裁决+记录 ~1.4s → ≤0.5s，幂等可关
**Independent Test**: `--op-ts-sample` 跑通且报告口径标注；默认模式 op_ts 逐条与现库一致

### Implementation for User Story 4

- [x] T020 [P] [US4] op_ts 抽样/批写：`kernels/examples/42_gehtp_runner/main.c` 加 `--op-ts-sample K`（默认关=全量逐条）；抽样模式报告头显式口径标注（防 census 类对拍误读）
  **PASS（1010, d0f1784, 板跑 md5 f87461e2）**：job.txt 键 `op_ts_sample K`（K≥2 生效，0/缺省=全量）；`wt_exec_set_ts_sample(K)` 下发；op_ts.bin **pad 字段复用承载 K**（0=全量，非破坏）+ 设备日志 `[prof] op_ts full/sample=K`；`scripts/gehtp_prof.py` 读 pad=K stderr 口径警告。全量模式 pad=0/11504 全非零；sample=8 pad=8/1438 非零（11504/8 精确）。
- [x] T021 [US4] run_range 裁决批化：`kernels/src/runtime/oplist_exec.c` ver=3 静态计划下消逐 op 引擎重裁决（planned 已定，deviation 语义与偏差不变量不动——003 契约）
  **PASS（1010, 取证收口）**：实查 `wt_annot_engine/fb_allowed/fb_engine` 全 `static inline`（oplist_parse.h:291-293），计划引擎从 footer 一次 `wt_rd_u32` + 掩码加载，零重裁决——**「逐 op 引擎重裁决」003 已实质消除**，T021 无需再批化（未强行加冗余）。本轮仅给 ts 填充加抽样门 `(g_ts_sample==0 || i%g_ts_sample==0)`，裁决/偏差不变量仍逐 op 结算。
- [x] T022 [US4] 板跑：开销 ≤0.5s（抽样口径）+ 幂等门（默认模式行为/位级与现库一致）
  **PASS（1010, d0f1784, libhvxhmx_008t21, 板跑 md5 f87461e2 双跑）**：A 默认 full exec 6.308s / B sample=8 exec 6.310s / t20 对照 6.308s——抽样 Δ=+2ms（噪声），**per-op 裁决+记录开销 ~0，远低于 0.5s 门**；`sum_op_us`↔`wall_us` 缝隙仅 16-18ms。~1.4s 是 v4 时代陈旧估计（运行时自决已被 003 静态计划消掉）。幂等门 = 默认模式位级同现库。证据 `evidence/t022-us4-optsample-gate.md`。

**Checkpoint**: SC-005 后半达成

---

## Phase 7: Polish & 收口

- [x] T023 终局板跑：SC-003 总门 ≤8s（007 锚口径；007 未交卷按 v4 35.31s 回退等比重校并登记）+ 双口径判分登记（`scripts/judge_logits.py` f16/W4 双列，沿 007 冻结值）— **exec 6.31s ≤8s PASS（3.84× 于 007 锚 24.24s）；f16 位级 md5 f87461e2 恒等 + cos 0.998642/top1 30/32 不回归；W4 列沿 007 冻结门 0.98/20 未消费（Phase 1 纯 f16）**。证据 `evidence/t023-final-board-gate.md`
- [x] T024 [P] 003 blob-v3 五场景 + ctest 全绿复验（devA/devB/w4v3/t2l/ver1-legacy；证据入库不入 /tmp）— **五场景逐位复现（devA 2c08b292/devB rc=2 die/w4v3+w4v1 7654a8f6/t2l 7940320b，偏差不变量 PASS）+ ctest 48/48（重编含 US3 emit）**。证据 `evidence/t024-regression-003-five-scenarios.md`
- [x] T025 [P] quickstart.md 四场景全复跑 + evidence 归档收口（specs/008 目录内）— **四场景→证据映射 + 终态六门全绿（exec 6.31s / coverage 0.8395 / md5 f87461e2 / W4 沿 007 / per-op ~0 / 三族 0.56s）**。证据 `evidence/t025-quickstart-four-scenarios-closeout.md`
- [x] T026 跨会话协调表 + memory 回写：BudgetReport 口径向 009 GDN 融合线交接（窗口=009 US3 前）、004 reconcile 面终态、库指纹变更登记 — **协调表 008 行尾更新 + 新建 memory `gehtp-008-phase1-complete.md` + MEMORY.md 索引登记**

---

## Dependencies & Execution Order

### Phase 依赖
- Setup(T001-T002) → Foundational(T003-T004) → **US1(T005-T010, MVP)** → US2(T011-T015) / US3(T016-T019) → US4(T020-T022) → Polish(T023-T026)
- **T011 是 T012-T015 的硬前置闸**（fence 未实证不上流水）
- T016 依赖 T010（census 基线 = US1 落地后）

### Story 依赖
- **US1**: Foundational 后即可开工，与 007 W4 供给无依赖（f16 位级门最干净）——007 延期不阻塞
- **US2**: 依赖 US1 的预算表与 staging 槽语义（T008/T012）+ T011 过门
- **US3**: 依赖 US1（驻留面折叠前提）；与 US2 可并行起步（不同文件段：emit 折叠 vs dma 发射，**同文件 wtop_emit.cpp/oplist_exec.c 不同段，并行时逐日 git status**）
- **US4**: 独立，排最后（收益垫底）；与 US2/US3 可并行

### 并行示例
```
Foundational 后:
  A: T005 → T006 → T007 → T008 → T009(并行) → T010   (US1 主线)
  B: T011 (fence micro-test, 独立文件, 与 US1 并行)
US1 收口后:
  A: T012 → T013 → T015    (US2)
  B: T016 → T017 → T019    (US3)
```

## Implementation Strategy

### MVP First（US1 only）
1. T001-T004 基线冻结（**T003 是全链最早可跑的纯测量，无代码改动**）
2. T005-T010 US1 落地 → **STOP 验证**：位级 + coverage ≥80% + 预算门负例（此时已可交付「已实现→已兑现」第一跳）

### Incremental
US1 → +US2（SC-002）→ +US3（三族 ≤2s）→ +US4（≤0.5s）→ T023 总门。每步位级/契约门全绿才进下一步；门值首跑校准冻结后只收紧不放松。

## Notes
- 改 `oplist_exec.c`（temp 物化/run_range 段）与 004 状态表段同文件——每次开工前 `git status`；004 回合主线时 reconcile
- census `out_b` 列含逻辑展开字节（Σ1.2TB/max34GB 已证伪）——**任何预算讨论只用 T003 冻结口径**
- 板上判分 llmserv env；op_ts 解析纪律：byte24 起、gehtp_prof.py --strict（二进制判读纪律）
  **PASS（1010, d0f1784, libhvxhmx_008t12 signed 81f034d3）**：四门全绿——位级 md5==v4 锚 / Σdma 22.53ms 1.4GB@66.7GB/s / 不变量 PASS(中止0, matmul 零回落) / MATMUL 18.1% wall≤50%；exec 24.24→13.38s（−44.8%）。两翻车教训入 `evidence/t012-prepack-boardrun.txt`：GC 裸槽 id 白名单六位点（漏标=活槽剥空 rc=38；泛标=temp 撞号 2.89GB 爆设备堆）
