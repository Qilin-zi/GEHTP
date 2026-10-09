# Implementation Plan: 全图 VTCM 驻留 + 显式 DMA 流水（中期性能线）

**Created**: 2026-10-09
**Branch**: `008-vtcm-residency-dma`（实施时照 004/007 先例进独立 worktree——共享工作树建/切分支波及全会话）
**Feature**: [spec.md](spec.md)

## What This Plan Delivers

四层性能兑现，目标 0.8B prefill **007 锚(≤20s) → ≤8s**：

1. **US1 激活全图 VTCM 驻留**：启用并扩面库里已存在的「第 7 步」静态 temp 规划（TEMPOFF 双表 + vtcm_off_arr 池驻留，v4 从未走此路径——research D1 实锤），预算三方单源化 + 全局门，coverage ≥80%。
2. **US2 tile 级 DMA 双缓冲 + 权重按执行序预取**：`--dma-runlist` 从溢出场景扩到全图，staging 槽 ×2 轮换；fence 并发 micro-test 硬前置。
3. **US3 残余搬运消化**：US1 后重跑 census，驻留面 reshape/slice 折叠消除 + 剩余 HVX 化，三族 ≤2s。
4. **US4 per-op 开销批量化**：op_ts 批写/抽样 + 静态计划下裁决批化，≤0.5s。

全程 003 契约不破（TEMPOFF 唯一真相源 / carve 一致性 / ver 逐字节 / 五场景）；f16 期位级门最干净（纯寻址变化）。

## Technical Context

- **机制现状矩阵**（research.md 汇总表，全部代码实查 2026-10-09）：FancyAllocator lifetime+spill / TEMPOFF 双表+一致性校验 / 运行时池驻留+写后 flush / OP_DMA 执行体+fence / `--dma-runlist` **全现成**；缺口 = 0.8B 图 `has_mem_plan()` 旁路、驻留扩面、三方全局预算门、双缓冲发射、fence 实证、预算基线冻结。
- **预算口径**：VTCM=16MB(16777216)；危险陈旧源 `get_vtcm_tile_size()`（graph_prepare.cpp:476-477，budget×0.75）收编为 `--vtcm-budget` 显式单源。census `out_b` 列含逻辑展开字节（Σ1.2TB/max34GB 已证伪）禁止作物理预算。
- **fence**：决策表（fence.c，V2.2 四铁律）覆盖 DMA↔HMX@VTCM=INVALIDATE 主路径；V81 不背书并发竞态 → micro-test 是 US2 唯一实证关。
- **构建/设备**：C++17 容器 `gehtp-toolchain:24.04`；C 库 `build_libs.sh`（多文件用 all）；设备 d0f1784（Permissive 预检 / pgrep 互斥 / 私有名）；判分 llmserv env + judge_logits.py 双口径（f16 位级 / W4 期沿 007 q4_0 冻结值）。
- **锚**：主口径 = 007 收口锚；US1 与 W4 供给无依赖可先行（v4 35.31s 口径回退预案）；库指纹纪律沿 004。
- **协调面（改前 git status）**：`oplist_exec.c` temp 物化段（↔004 状态表段）、`wtop_emit.cpp`（↔007 P2 供给段）、`graph_prepare.cpp`（003 WIP 区）、`op_matmul.cpp`（↔007/009）；009 GDN 融合线消费本线 BudgetReport 口径（窗口=009 US3 前）。
- **Source files**：`compiler/src/vtcm/ddr_offsets.cpp`、`fancy_allocator.cpp`、`compiler/src/ir/graph_prepare.cpp`（mem_plan 生成/预算）、`compiler/tools/wtop_emit.cpp`（TEMPOFF/预算门/report/CLI）、`compiler/tools/wtop_ops/op_matmul.cpp`（0x4000 refs/OP_DMA 发射）、`kernels/src/runtime/oplist_exec.c`（temp_get/exec_dma/run_range 裁决）、`kernels/src/runtime/fence.c`（micro-test 消费方）、`kernels/examples/42_gehtp_runner/main.c`（op_ts 落盘）、`scripts/gehtp_prof.py`（coverage/双时间轴报告）、`compiler/tests/test_wtop_emit.cpp`（契约测试）。
- **Complexity**：US1 中（策略+校验，机制现成）；US2 高（并发时序+fence 实证）；US3 中；US4 低。

## Constitution Check

Constitution 为空模板（沿 007 裁定）：以 C1-C5 契约 + 库纪律替代——blob 追加不破坏、opcode 不新增（复用 OP_DMA）、判据入库不入 /tmp、共享文件 git status 前置、门值首跑校准冻结。设计后复评：无违例（预算门/spill 名单满足「绝不静默」；TEMPOFF 单真相源防双源漂移）。

## Project Structure

`specs/008-vtcm-residency-dma/`（本目录）— spec / research(D1-D8 代码实查) / data-model / contracts(C1-C5) / quickstart / 本 plan。

## Implementation Phases

### Phase 0: 预算基线冻结（R 系，US1 前置，无代码改动）

- T001 现编译器重发 v4 同图（003 配方照抄），采集 `[ddr]` 两行 + `emit_ddr_offset_table` TSV → `evidence/budget-baseline.md` 冻结（盘子口径：vtcm_cap/n_temps/n_spills/vtcm 项数）
- T002 census `[vtcm]` 1107 op 语义重导（gehtp_prof/manifest 层面确认 = 0x4000 引用旗标非池驻留），并入基线文档防误读
- 验收：基线文档冻结；若重发即产出 TAG_MEM_PLAN（说明 v4 旁路仅因旧输入），US1 盘子直接量化

### Phase 1: US1 驻留扩面 + 预算单源化（MVP）

- T101 `graph_prepare.cpp` mem_plan 生成路径对 0.8B 图激活（has_mem_plan 恒真路径诊断+修复）；预算源收编 `--vtcm-budget`（默认 16777216），弃 get_vtcm_tile_size 陈旧源
- T102 驻留标记扩面：非 matmul temp（eltwise/broadcast/reshape 产物）进 lifetime 打包 + `in_vtcm` 判据（大小/生命期阈值可配，report 显式名单）
- T103 emit 全局三方预算门（resident+staging×2+carve_peak ≤ budget，超限硬报错含各项数值）+ `--residency-report`（BudgetReport+spill+coverage）
- T104 契约测试（test_wtop_emit.cpp）：预算门负例 / TEMPOFF 追加字段 round-trip / 惰性门（无新旗标 md5 同现库）
- T105 板跑三门：位级（vs 基线 blob md5）/ coverage ≥80% / wall 首锚记录
- 验收：SC-001 + SC-004(f16 列) + FR-001/002/003

### Phase 2: US2 DMA 双缓冲 + 预取（fence 前置）

- T201 fence 并发 micro-test（HMX matmul × dc_dma 并发 ≥100 轮双向 fence，设备端，证据入 evidence）——**未过门本 phase 其余任务不上板**
- T202 `--dma-runlist` 扩全图：权重/激活按执行序预取发射（从 spill 场景到主力 HMX 族），预算内 staging 槽分配
- T203 双缓冲轮换：DmaStreamDesc buf_slot 0/1 + 发射序（tile i+1 搬运 → tile i invoke → completion poll），失败回落串行 + deviation（003 契约）
- T204 op_ts 双时间轴：dma_us 与 dur 并存可判重叠；gehtp_prof 加 HMX 族重叠率列
- T205 板跑：dma_us>0 / SC-002（HMX 族 ≤50% 锚）/ 位级门 / 回落负例
- 验收：SC-002 + FR-004/005

### Phase 3: US3 残余搬运消化（census 驱动）

- T301 US1 落地后重跑 census → digest（三族新盘子）
- T302 驻留面零拷贝折叠（reshape/slice → 引用消除，emit 期，位级）
- T303 剩余物理搬运 HVX 化（007 P1 三形态范式复用，计划=实际引擎）
- T304 板跑：三族合计 ≤2s + census digest 落盘（不硬凑门，残余根因指认）
- 验收：SC-005 前半 + FR-006

### Phase 4: US4 per-op 开销批量化

- T401 op_ts 批写/抽样模式（runner `--op-ts-sample`，默认关=现行为逐条一致）
- T402 run_range 裁决批化（ver=3 静态计划下消逐 op 重裁决；deviation 语义不变）
- T403 板跑：开销 ≤0.5s（抽样口径标注）+ 幂等门
- 验收：SC-005 后半 + FR-007

### Phase 5: 收口

- T501 终局板跑：SC-003 总门（≤8s 或回退锚等比重校）+ 双口径判分登记
- T502 003 五场景 + ctest 全绿复验；evidence 归档（判据入库不入 /tmp）
- T503 协调表/memory 回写（含 009 线 BudgetReport 口径交接）
- 验收：SC-003/006 + FR-008/009

## Phase 依赖与并行

T0 → T1 → (T2 ∥ T3 起步) → T4 → T5；T201 是 T202-T205 的硬前置闸；T301 依赖 T105。US1 可在 007 收口前独立推进（f16 位级门），Phase 2 性能门以 007 锚计。
