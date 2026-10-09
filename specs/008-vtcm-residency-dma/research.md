# Research: 全图 VTCM 驻留 + 显式 DMA 流水

**Created**: 2026-10-09 | **Status**: frozen（代码实查 2026-10-09，库 = task/prof-wp @ d33e31b3 系）

## D1 — 「第 7 步」静态 temp 规划已在库里，v4 从未走此路径（本线最重要发现）

**Decision**: US1 不新建机制——激活存量件（TEMPOFF 双表 + vtcm_off_arr + carve footer），补的是「启用 + 扩面 + 预算统一」。

**Evidence**（2026-10-09 实查）:
- 编译器侧 `compute_ddr_offsets`（`compiler/src/vtcm/ddr_offsets.cpp`）本来就是 **VTCM 生命期分配器的 DDR 降级复用**：`FancyAllocator::allocate_with_lifetime`（reverse-life-end 排序 + 同生命期分组 + 事件扫描复用），签名已带 `vtcm_budget` + `spilled_out` + `vtcm_cap_out`——**spill 出口原生存在**。
- emit 侧（`compiler/tools/wtop_emit.cpp:117-131`）：`if (gp.has_mem_plan())` 为门，消费 `mp.entries`（含 **`in_vtcm` per-entry 旗标**）/`mp.spills`，stderr 报 `[ddr] TAG_MEM_PLAN: cap= vtcm_cap= n_temps= n_spills=`；TEMPOFF 槽（:578-581）再报 `[ddr] TEMPOFF slot: cap= vtcm_cap= n_temps= (vtcm=)`。
- 运行时侧（`kernels/src/runtime/oplist_exec.c:308-323`）：`temp_get` 首路径即 VTCM 驻留（`vtcm_off_arr[id]` 命中 → `vtcm_pool + off` 直写直读 + 写后 `qurt_mem_cache_clean` FLUSH，注释注明上板实测数值不一致的根因修复）；池布局在 wtcache 区顶端（:292），与 carve 区（`carve_end`）显式互斥。
- **v4 未启用的实锤**：`.scratch_003_rebuild/wtop_emit_v4.log`（88 行，stderr 在内）**无任何 `[ddr]` 行** → v4 输入图 `has_mem_plan()==false`，整条路径静默旁路。census 里 `[vtcm]` 1107 op 是 op 名/引用旗标层面的既有标注（0x4000|temp 引用，op_matmul.cpp:82 先例），非池驻留事实。

**Alternatives considered**: 新建独立驻留表槽（否决——与 TEMPOFF 双源真相冲突，:694-716 的 carve/TEMPOFF 一致性注释明确警告"第二套真相"会造踩踏/全 -inf 难查）。

## D2 — 工作集预算冻结法（spec 最大风险的关法）

**Decision**: 用**现编译器对 0.8B 图重发**采集 `[ddr] TAG_MEM_PLAN` + `[ddr] TEMPOFF` 两行 stderr（cap / vtcm_cap / n_temps / n_spills / vtcm= 项数），作为 BudgetReport 冻结基线；辅以 `emit_ddr_offset_table` TSV（kernels/host/check_offsets 裁判，ddr_offsets.cpp 注释）做逐 temp 对拍。

**Rationale**: census CSV `out_b` 列含逻辑展开字节（Σ=1.2TB / max 34GB，非物理量，已实测证伪）不可作预算；编译器自己算的 cap 是唯一权威物理口径。ggml decode 参照 3.23MB/241 槽/spill=0（cooked_meta.json）给量级信心。

**How**: R1 任务 = 现库重发 v4 同图（003 的发射配方照抄，编译输入在 0.8B 资产线），不追求一次到位——首跑数字只冻结「盘子」，扩面后数字迭代但口径不变。

## D3 — 预算源治理（危险陈旧源收编）

**Decision**: 三方预算（驻留 Σ + 权重 staging ×双缓冲 + carve 峰值）必须单源化，显式 `--vtcm-budget`（默认 16MB=16777216 口径），emit 期全局校验 `三方和 ≤ 预算` 超限硬报错。

**Evidence**: `graph_prepare.cpp:476-477` `vtcm_size_ = get_vtcm_tile_size()`（=budget×0.75，反汇编重建的陈旧常数，跨会话协调表已标"危险陈旧源"）；现状校验只有 carve↔TEMPOFF 双源一致性（wtop_emit.cpp:694-716）与 `vtcm_used > ddr_vtcm_cap` 单项（:688），**缺全局三方和门**。

## D4 — fence 决策表已覆盖双缓冲主路径，micro-test 是实证关不是开发关

**Evidence**: `kernels/src/runtime/fence.c`（50 行）`fence_op_for` 决策表：DMA→HMX @VTCM = `FO_INVALIDATE`（供给面失效）、CPU→DMA @VTCM = `FO_FLUSH`、HMX 只访 VTCM（HMX×DDR 组合恒 FO_INVALID，物理限制非缺陷）。`exec_dma`（oplist_exec.c:1116-1164）已消费 fence args（f_src/f_dst）+ `dc_dma_once_ex` 真引擎。

**Decision**: US2 前置任务 = 设备端 micro-test（HMX matmul 与 dc_dma 搬运并发循环 ≥100 次，双向 fence，输出逐字节比对 + 超时看护），过门才放流水发射。表源是 V2.2 文档四铁律 + 部分设备实测（文件头注释），V81HexSim 已知不能背书 fence/DMA 竞态。

## D5 — US2 通道现成度：`--dma-runlist` 已发真 OP_DMA，缺的是全图化 + 轮换

**Evidence**: `wtop_emit.cpp:853/874` `--dma-runlist`（"溢出搬移发射真 OP_DMA(非 SPILL/FILL)"，003 T015）；`op_matmul.cpp:90-102` DDR 源权重/激活 → OP_DMA 预置 staging（"DDR 源需 --dma-runlist + VTCM 池余量"）；发射注释（wtop_emit.cpp:466）"DMA 与 matmul 相邻发射" = **当前串行相邻，无 tile 级重叠**。

**Decision**: 双缓冲 = staging 槽 ×2 轮换（`DmaStreamDesc.buf_slot 0/1`）+ 发射序改为「tile i+1 搬运发射 → tile i invoke → completion poll」；沿用 003「允许回落+上报」契约，任一环失败回串行并记 deviation。具体发射序形态（runlist 重排 vs 就地双发）plan T2xx 细化，本阶段只定槽语义。

## D6 — 残余搬运盘子（US3）

census/prof 双口径（v4，31.78s）：STRIDED_SLICE 3.06s(2373) + UNARY 3.05s(2909，其中 2685 是 Reshape 拷贝) + TRANSPOSE_GEN 1.29s(175)。**US1 落地后须重跑 census**：驻留面上的 reshape/slice 大概率折叠为引用消除（零拷贝寻址），剩余物理搬运走 HVX（007 P1 broadcast 三形态范式复用）。spec 表内 7-8s 是立项投影，择优以重跑后 digest 为准。

## D7 — per-op 开销（US4）

v4 逐 op 裁决 + op_ts 记录 ≈1.4s（007 spec 口径）。ver=3 计划已静态，`run_range` 每 op 的引擎重裁决冗余可批化；`write_op_ts`（runner main.c）逐 op 落盘可批写/抽样（`--op-ts-sample`，默认关保持全量）。门 ≤0.5s。

## D8 — 锚与依赖处置

007 收口锚（≤20s）为主口径；**US1 与 W4 供给无依赖**（v4 纯 f16 图即可验证驻留——位级对拍判据反而最干净），可先行。007 延期时锚回退 v4 35.31s 实测重校，比例门纪律不放松。协调面：`oplist_exec.c` temp 物化段（004 状态表/KV 段同文件）、`graph_prepare.cpp`（003 WIP 区）、`wtop_emit.cpp`（007 P2 供给段）——全部改前 `git status`。

## 汇总：机制现状矩阵（本线开工基线）

| 件 | 状态 | 位置 |
|---|---|---|
| FancyAllocator lifetime 打包 + spill 出口 | ✅ 现成 | ddr_offsets.cpp / fancy_allocator.cpp |
| TEMPOFF 双表（DDR 偏移 + VTCM 驻留项）+ carve 一致性校验 | ✅ 现成 | wtop_emit.cpp:550-716 |
| 运行时 vtcm_off_arr 池驻留 + 写后 flush | ✅ 现成 | oplist_exec.c:308-323 |
| OP_DMA 执行体（VTCM 侧 + fence + dc_dma 引擎） | ✅ 现成 | oplist_exec.c:1116-1164 |
| `--dma-runlist` 搬移显式化发射 | ✅ 现成（仅溢出场景） | wtop_emit.cpp:853, op_matmul.cpp:90-102 |
| 0.8B 图 mem_plan 启用 | ❌ v4 未走 | wtop_emit.cpp:117 门 |
| 全图驻留扩面（非 matmul temp） | ❌ | 待做 |
| 三方全局预算门 + 预算源单源化 | ❌ | 待做（D3） |
| tile 级双缓冲轮换发射 | ❌ | 待做（D5） |
| fence 并发 micro-test 实证 | ❌ | 待做（D4） |
| v4 重发预算冻结基线 | ❌ | 待做（D2） |
