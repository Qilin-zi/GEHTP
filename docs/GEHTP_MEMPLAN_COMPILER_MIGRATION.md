# GEHTP 内存规划收编编译器迁移计划（wtop_emit → hnnx_compile）

> 生成日期 2026-09-16。决策来源：09-16 用户拍板——**编译器产出的规划为唯一真相源**，
> wtop_emit 退化为打包器。代码基线：`29af37d`（算子三件套·host 发射侧重构后，
> wtop_emit.cpp 557 行 + tools/wtop_ops/ 16 handler）。
> 本文所有锚点均已逐条对代码核实（非文档转述）。

---

## 0. 迁移原则（先读）

1. **行为保持**：第一阶段不改算法只搬位置——规划决策与现状逐字节一致，全部存量门零回归。
   算法升级（cp_solver 默认化、跨组复用、真 DMA runlist）是独立的后续阶段（§3 M4），不混入。
2. **设备侧零改动**：WTOP blob 格式（16B 头/slot 表/op 记录/TEMPOFF 槽/权重区）一字节不变；
   `wt_parse`/`oplist_exec`/runner 全部不动。迁移只改 host 侧「规划在哪算、从哪读」。
3. **打包留 emit，规划收编译器**：权重打包（GGUF/Q4_0 repack、f32→f16 RNE、按 const id 去重、
   128B 对齐）、`--input-f16` 固化、EXT_IN 标记、op lowering（wtop_ops/ 16 handler）、
   blob 布局、manifest——这些是打包职责，不动。
4. **与在途工作解耦**：4B 的 `Transpose_perm` 缺口是 op lowering 问题（wtop_ops/），与本计划无关；
   VL 线正在改 graph_prepare.cpp/ops.cpp/host_run.cpp，动工前按多会话协调规则
   （查 mtime + ListAgents + SendMessage 排序）错峰。
5. **每阶段独立 commit、独立可验收、可回退**；设备侧因 blob 不变而无需回退。

---

## 1. 现状审计：规划在两处各做一遍，设备只用 emit 那遍

### 1.1 决策项对账（实证）

| 决策项 | 编译器侧（hnnx_compile） | emit 侧（wtop_emit） | 设备实际用 |
|---|---|---|---|
| 执行顺序 | ST-Cut 产 `plan_order_`（do_prepare2 Step8，`graph_prepare.cpp:4041`）→ TAG_PLAN_ORDER(0x504C) | **Kahn 重排定稿**（`wtop_emit.cpp:278-306`，因 plan_order 非保证拓扑） | **emit 定稿序** |
| DDR temp 偏移 | `compute_ddr_offsets`（`vtcm/ddr_offsets.cpp:23`，GraphPrepare 成员，两级池分配）已实现但**仅被 emit 调用** | 调用之 → `em.ddr_op_map`（`wtop_emit.cpp:113-142`）→ TEMPOFF 槽（`:403-435`） | **emit 产 TEMPOFF** |
| VTCM 驻留集合+偏移 | 同上（`in_vtcm` 标记，`ddr_offsets.cpp:83-107`） | 同上 → TEMPOFF reserve 高 16 位 + temp_id\|0x4000 | **emit 产 TEMPOFF** |
| spill 集合+插入点 | ① FancyAllocator 贪心整组 spill → TAG_SPILL_FILL_OP(0x5346) 记录（`graph_prepare.cpp:1454-1460`）；② `compute_ddr_offsets` 的 `finalize_spills`（`ddr_offsets.cpp:126-147`） | **只用 ②**：`ddr_spill_map` → 消费前插 OP_FILL（`:347-357`）/生产后插 OP_SPILL（`:365-377`）；① 只用于量池尺寸（`:389-401`，不发射） | **emit 插桩** |
| 池几何 | — | TEMPOFF `[cap][reserve 低16b=bump KB\|高16b=VTCM KB]`；spill 池槽（`:197-206`） | **emit 定** |
| temp 编号 | —（编译器无 temp 概念） | `Emitter::fresh_temp` 活性复用分配（`wtop_ops.hpp:136-163`） | emit 分配 |

### 1.2 编译器规划产出如何被丢弃（三条实证）

1. `serialize_opdef`（`graph_prepare.cpp:1971`）的 TAG_OP_RECORD 字段 = name/grouping/op_id/flags/
   inputs/OutputDef/const 偏移/extra——**不含任何 VTCM/DDR 偏移**。FancyAllocator 在 do_prepare2
   里算的全套结果没有任何序列化出口。
2. TAG_SPILL_FILL_OP 记录（编译器贪心 spill 决策）在 `wtop_emit.cpp:389-401` 的循环体是
   `continue`——只取 `ddr_offset+size` 最大值建池槽，**决策本体不发射**。
3. `plan_order_` 被 emit 注释明确定性「调度序, 非保证拓扑」（`wtop_emit.cpp:279-281`），
   由 emit 的 Kahn 重排定稿——编译器连「顺序」这个最基础的规划项都不是定稿方。

### 1.3 要收编的 emit 层规划资产（清单=迁移工作项）

| # | 资产 | 位置 | 收编去向 |
|---|---|---|---|
| A | `compute_ddr_offsets` 调用 + `ddr_op_map`/`ddr_op_vtcm`/`ddr_spill_map` 填充 | `wtop_emit.cpp:113-142` | 编译器 do_prepare2_late 内调用（本就是 GraphPrepare 成员函数，零算法改动），结果序列化进 tagged.bin；emit 改为从 bin 读 |
| B | Kahn 拓扑定稿 | `wtop_emit.cpp:282-306` | 同上，do_prepare2_late 内对 `plan_order_` 定稿；emit 的 Kahn 降级为校验 |
| C | TEMPOFF 表数据（cap/reserve/entries/vtcm 标记） | `wtop_emit.cpp:403-435` | 数据来自编译器 plan；emit 只保留「查表写槽」的打包动作 |
| D | SPILL/FILL 插桩位置决策（哪个 op 前/后插） | `wtop_emit.cpp:347-377` | 决策 = spill 集合（编译器产）；插桩动作本身是发射，留 emit |
| E | `bump_reserve` 策略（`max(4MB, cap/4)`） | `wtop_emit.cpp:410` | 收进 plan（reserve 字段由编译器算好） |
| F | spill 池尺寸 | `wtop_emit.cpp:197-206` | 尺寸 = plan 的 spill_total；建槽动作留 emit（必须在输入槽之后） |
| G | `fresh_temp` 静态偏移登记 + 广播 size 守卫 | `wtop_ops.hpp:141-162` | **不动**（查表逻辑，表源换而已） |

### 1.4 官方 .so 参照

官方架构即本计划目标：`fa::RuntimeAllocator` 在 Graph 内定稿（`make_allocator @0xf4e3b0`）、
spill/fill 记录由编译器写入 runlist、设备纯执行。本计划与 `GEHTP_BACKPORT_PLAN.md` §1 同向——
但注意 BACKPORT_PLAN 的 RuntimeAllocator 层级恢复属**算法升级**（跨组复用语义），归本计划 M4，
不是收编前置。

---

## 2. 目标架构

### 2.1 分工（收编后）

```
hnnx_compile（规划定稿方）                    wtop_emit（打包器）
─────────────────────────────              ─────────────────────────────
loader → do_prepare1 → do_prepare2           deserialize tagged.bin
  → do_prepare2_late 【新语义】:               → 读 TAG_MEM_PLAN → em.ddr_* 表
      Kahn 定稿 plan_order_                  → 权重打包/输入固化（不动）
      → compute_ddr_offsets(budget)          → op lowering（wtop_ops/，不动）
      → mem_plan_ 成员                        → 按定稿序发射 + 按 plan 插桩
  → serialize:                                → TEMPOFF/slot 表照抄 plan
      既有记录 + TAG_MEM_PLAN【新】            → blob + manifest（不动）
```

### 2.2 tagged.bin 格式扩展（一个新 tag，已查空）

`TAG_MEM_PLAN = 0x4D50`（'MP'；tag 表 `serializer.hpp:108-131` 无撞车）。记录体：

```
[version u32 = 1]
[ddr_cap u32]            // = 现 em.ddr_static_cap
[vtcm_cap u32]           // = 现 em.ddr_vtcm_cap
[bump_reserve u32]       // = 现 max(4MB, cap/4)（资产 E，编译器算好）
[n_entries u32]
  n × {op_id u32, out_idx u32, offset u32, size u32, flags u32}   // 20B；flags bit0=in_vtcm
[n_spills u32]
  n × {op_id u32, spill_off u32, size u32, pad u32}               // 16B
```

设计要点：
- **键 = (op_id, out_idx) 而非 temp_id**。temp 编号是 emit 层 fresh_temp 的概念；
  现状 `fresh_temp`（`wtop_ops.hpp:147-162`）本就以 op_id 查 `ddr_op_map` 再登记 temp，
  表源换成 TAG_MEM_PLAN 后该函数零改动。
- spill 池尺寸 = `max(spill_off+size)` 128B 对齐，emit 从 n_spills 自算，无需显式字段。
- `plan_order_` 定稿后仍走既有 TAG_PLAN_ORDER（语义升级为保证拓扑），不加新 tag。
- 无 TAG_MEM_PLAN 的旧 bin：emit 回退现路径（自算），保留一个版本周期（§4 开放问题 3）。

### 2.3 预算参数通路（推荐方案 A，见 §4）

`hnnx_compile` 新增 `--ddr-budget/--vtcm-budget` 两旗标 → 传入 `prepare()` →
do_prepare2_late 用。`scripts/gehtp` 第 3 步加传两旗标；第 4 步旗标在 M2 期保留作
shadow 校验，M3 删除。（历史注脚：旧 `conv_add_pipeline.sh` 本就把预算传给 hnnx_compile，
后来才挪到 emit——本次是拨乱反正。）

---

## 3. 分阶段

### M1 拓扑定稿上提（顺序唯一真相源）【已落地】

- 状态：已提交。host 门全绿——conv_add/spill 变体 blob 与基线**逐字节全同**
  （tagged.bin 按预期变化：TAG_PLAN_ORDER 改载定稿序）；ctest 44/44
  （test_scheduler_integration 断言 2 改 M1 语义：同集合+拓扑性）。
  设备 quickstart 门因设备多窗口互斥待补跑（blob 与已知绿基线逐字节同，风险为零）。

- do_prepare2_late（`graph_prepare.cpp:481`，现近空壳）内对 `plan_order_` 做 Kahn 定稿
  （代码从 `wtop_emit.cpp:282-306` 平移，含 orphan 回退与确定性 sort）。
- serialize 写定稿序；`wtop_emit.cpp:278-306` 的 Kahn 改校验（重算不一致 → stderr warn，
  仍用 bin 序）。
- **门**：tagged round-trip 测试全绿；conv_add byte-exact ×3；quickstart `SAMPLE ALL GREEN`。

### M2 静态规划收编（核心阶段）

- do_prepare2_late：Kahn 后调 `compute_ddr_offsets`（预算来自新旗标）→ 填 `mem_plan_` 成员。
- serialize：`do_serialize` 在 TAG_PLAN_ORDER 后写 TAG_MEM_PLAN；deserialize 对称读。
- `wtop_emit.cpp:113-142`：若 bin 含 TAG_MEM_PLAN → 直接填 `em.ddr_op_map/ddr_op_vtcm/
  ddr_spill_map/ddr_static_cap/ddr_vtcm_cap`（跳过自算）；emit 旗标保留，与 plan 不一致时
  stderr warn（shadow 期）。
- TEMPOFF 生成（`:403-435`）数据源切换；reserve 直接用 plan 值。
- **门**：conv_add byte-exact ×3（32768/32768）；spill 变体（--ddr-budget 4096）byte-exact；
  阶段一/二位级一致复跑（65536/65536，top1 256/256）；L3 81-op 对拍 |d|≤0.001；
  host_run --bin 回归。

### M3 旧链路退役

- 删 `wtop_emit.cpp:389-401` 的 `spill_fill_recs()` 量池段（spill 池尺寸已有真源）。
- 编译器贪心 spill → TAG_SPILL_FILL_OP 重定语义为「成本模型观察记录」（或禁用发射）；
  文档同步。
- 删 emit 的 shadow 自算路径与两预算旗标；gehtp 脚本第 4 步不再传预算。
- 文档更新：GEHTP_PORTAL.md（§1 五步表 + --ddr-budget 描述）、README.md 路线段、
  本计划状态标注。
- **门**：全量回归 + 文档审读。

### M4 算法升级（独立排期，不在本计划验收范围）

- cp_solver 默认化（现 env HNNX_VTCM_ALLOCATOR=cp* 门控）；跨组复用（BACKPORT_PLAN §1
  RuntimeAllocator 语义）；真 DMA runlist 算子（第7步阶段三方向，收编 matmul 内嵌的
  cpu_to_vtcm/dc_dma_once 为显式 runlist 条目）。
- 收编完成后这些升级才有正确的落点（编译器），这正是本计划的远期收益。

---

## 4. 开放问题（动工前拍板）

1. **预算参数通路**：方案 A = hnnx_compile 加旗标（推荐，显式、可进 manifest 溯源）；
   方案 B = 环境变量（与 HNNX_* 既有风格一致但隐式）。
2. **TAG_MEM_PLAN 记录粒度**：单 tag 双 vector（本文推荐）vs 拆两个 tag。
3. **旧 bin 兼容期**：无 plan 的 tagged.bin 回退 emit 自算——保留一个版本周期还是立即报错。
4. **多会话错峰**：VL 线在改 graph_prepare.cpp/ops.cpp/host_run.cpp；M1 触碰
   graph_prepare.cpp（do_prepare2_late + serialize），动工前必须 ListAgents 排序。

---

## 5. 验收门总表（每阶段必过，缺一不可）

| 门 | 命令/位置 | 期望 |
|---|---|---|
| tagged round-trip | compiler tests | serialize→deserialize→re-serialize 恒等 |
| conv_add byte-exact ×3 | 对拍基线 | 32768/32768 × 3 |
| spill 变体 | --ddr-budget 4096 | byte-exact 32768/32768 |
| quickstart | `scripts/gehtp_quickstart.sh` | `=== SAMPLE ALL GREEN ===` |
| 阶段一/二 位级 | 第7步设备门 | 65536/65536，top1 256/256 |
| L3 81-op | 例40/41 线 | 对拍 81/81，端到端 \|d\|≤0.001 |
| host_run --bin | 回归 | 与设备逐 op 对拍不退化 |

## 6. 风险与回退

- **格式变更波及 deserialize 消费方**（host_run --bin、wtop_emit、round-trip 测试）→
  同 PR 同步改；round-trip 测试守门。
- **shadow 期决策不一致**（编译器 plan vs emit 重算）→ M2 保留 emit 重算+warn 一个版本；
  warn 即停线排查，不放行。
- **多会话冲突**（VL 线）→ 协调规则；必要时 M1 等 VL 线提交窗口。
- **回退**：每阶段独立 commit；blob 格式不变 → 设备侧零回退成本；host 侧 revert 即恢复。

## 7. 锚点索引

| 项 | 位置 |
|---|---|
| compute_ddr_offsets（两级池分配本体） | compiler/src/vtcm/ddr_offsets.cpp:23-124 |
| finalize_spills | compiler/src/vtcm/ddr_offsets.cpp:126-147 |
| do_prepare2_late（收编落点） | compiler/src/ir/graph_prepare.cpp:481 |
| TAG_OP_RECORD（无偏移实证） | compiler/src/ir/graph_prepare.cpp:1971 |
| TAG_SPILL_FILL_OP 写侧 | compiler/src/ir/graph_prepare.cpp:1454-1460 |
| tag 表（0x4D50 空） | compiler/include/hnnx/serialize/serializer.hpp:108-131 |
| emit 规划五资产 | compiler/tools/wtop_emit.cpp:113-142 / 278-306 / 347-377 / 389-435 |
| fresh_temp 静态登记+广播守卫 | compiler/tools/wtop_ops/wtop_ops.hpp:136-163 |
| 设备 TEMPOFF 消费（不动） | kernels/src/runtime/oplist_exec.c:1031-1085 |
| 设备 ref_ptr 0x4000 VTCM 解码（不动） | kernels/src/runtime/oplist_exec.c:77-86 |
