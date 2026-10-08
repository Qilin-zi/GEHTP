# Research: Compiler-Driven Scheduling

Phase 0 输出。每个决策以「Decision / Rationale / Alternatives considered」给出。文件:行引用相对仓库根。

## R1 — 引擎指派如何进 blob（FR-001/002/003）

**Decision**: blob 增加一个**逐算子引擎注解表**，作为 op 表之后的 footer 区域；`wt_blob.ver` bump 到 **3** 以区分有无该区域（`ver==2` 已被大权重 24B slot 记录占用，见下文）。每个 op 一条定长记录，打包进一个 `u32`：

| 位 | 字段 | 含义 |
|----|------|------|
| 0–1 | engine | 0=scalar, 1=HVX, 2=HMX, 3=reserved（对应 `wt_engine`） |
| 2 | fallback_allowed | 0=禁止回落（FR-008），1=允许回落 |
| 3–4 | fallback_engine | 允许回落时回落到哪个单元 |
| 5–31 | 保留 | 置 0 |

**Rationale**: ① 引擎概念必须**跨所有 opcode 统一**（FR-001 覆盖 matmul/conv/unary/binary/softmax/rmsnorm2/reduce/scatter/gather 等 30+ opcode），而逐 opcode 复用 `args[]` 的 flag 位只对 matmul 可行。② 作为独立 footer 区域不碰 `wt_op.args[]` 的 arity 契约（`WT_MAX_ARGS` 不变、`WT_ERR_ARITY` 校验不变），对旧解析器零影响。③ `ver` bump 天然实现 FR-009 向后兼容：`ver<=2`（无 footer）→ 运行期维持今日的自决逻辑（legacy 路径）；`ver==3` → 运行期读 footer 照做。⚠ **`WT_BLOB_VER_2=2u` 已存在**，是给大权重（`weight_area>0xFFFFFF00`）切 24B slot 记录用的（`wtop_emit.cpp:508-512` 的 `need_v2`），**引擎注解必须用 ver=3，勿复用 ver=2**。④ 现有 matmul `flags` 的 bit0/1=t0/t1、bit2=batched、bit16–31=Bn（`op_matmul.cpp:43-56`）已占，bit3–4 虽空，但混用会制造"有的 op 看 flag、有的看 footer"的不对称，拒绝。

**Alternatives considered**:
- *matmul-family 复用 flags bit3–4 + 其余走 footer*：零额外空间，但两套编码、非均匀，弃。
- *给每个 opcode 塞进 args[] 尾部*：破坏 arity 契约与现有 opcode 语义，且要改 30+ 处 parse/exec，弃。
- *侧信道 slot*（如 `WT_SLOT_TILING` 式专门槽）：槽语义已被表/驻留占满，且与"逐 op"粒度不符，弃。

## R2 — 静态 VTCM carve 时间表的结构与算法（FR-005/006）

**Decision**: 复用现有 lifetime 模型与求解器，产出**逐 op 的 VTCM carve 表**（同样走 v2 footer 或独立区域），运行期读表按精确大小分配，取代 `hmx_runtime_setup` 的"抢全量"。

- **活性/生命周期**：`compiler/src/ir/graph_deps.cpp` + `graph_prepare.cpp` 已算 op 依赖与张量活性，是 carve 的输入。
- **分配器**：`compiler/src/vtcm/fancy_allocator.cpp`（lifetime-based）提供基础分配；`compiler/src/vtcm/cp_solver.cpp`（provably-optimal）在争用严重时给出最优打包。`compute_ddr_offsets`（`ddr_offsets.cpp`）是"编译期布局决定、运行期照做"的既有先例。
- **carve 表记录**：`{ op_index, vtcm_offset, size_bytes, residency:[first_op, last_op] }`，生命周期重叠的 op 不共享重叠区间（对应 spec Edge Case）。
- **运行期改造**：`hmx_runtime_setup`（`kernels/src/runtime/hmx_common.c:109`）当前 `(void)` 掉请求 size、恒申请 total（:132-133）。改为**按 carve 表精确 size 分配**；`vtcm_pool_ensure`/`vtcm_pool` 的所有权模型从"单一全量池"改为"按表分时复用"。

**Rationale**: 不必新造求解器——GEHTP 已有可证明最优的 carve 先例，缺的只是"把结果发进 blob 并让运行期服从"。HMX 的 crouton 缓冲尺寸由 M/K/N 决定（`op_matmul.cpp:64-72` 的 `need_mm` 已能算），成本模型/发射器可直接复用该公式生成 per-op 尺寸。

**Alternatives considered**:
- *运行期 LRU/惰性 evict*：仍是运行时仲裁，违背 FR-005，弃。
- *全局唯一静态偏移（无生命周期复用）*：简单但 16MB 会被 0.8B 的多 op 峰值撑爆（lm_head 单 op K=1024 N=248320 即需超界，已由 SPLIT 分块），弃。
- *让 f16 HMX 与 W4A16 wtcache 从同一 arena carve 子块共享*（`gehtp-vtcm-ownership-gap` 里提过的正解）：与"逐 op 静态 carve"是**互补**的同一方向，纳入 P2 的 arena 统一实现，而非替代。

## R3 — 显式 DDR↔VTCM 搬移 op（FR-006）

**Decision**: 扩展既有 `OP_DMA`（opcode 33, arity 8，`kernels/include/oplist_parse.h:149/197`）的**发射面**，从"仅 DDR→DDR spill/fill"扩到"含 DDR↔VTCM"。运行期执行器已具备 fence 纪律（`dc_dma_once_ex` + CPU→DMA→HMX 同步）与 `0xC000` 引擎面引用编码（`WT_REF_ENG_FLAG`），仅需让 src/dst 的 VTCM 侧引用（`0x4000`）在 exec 路径被正确解析。

**Rationale**: OP_DMA 就是现成的通用搬移 op，不该新造 opcode。编译器当前只在 `wtop_emit.cpp:417/444/447` 发 DDR→DDR；扩到 VTCM 侧是"把已有能力接到 carve 表上"，改动集中在发射器与 `oplist_exec.c` 的 OP_DMA 执行分支。

**Alternatives considered**:
- *新 opcode（如 OP_VTCM_MOVE）*：重复实现 fence/引用语义，弃。
- *隐式搬移（运行时按引用自动搬）*：又退回运行期自决，违背 FR-006，弃。

## R4 — 成本模型（FR-001 的"按形状+VTCM 预算+单元争用"）

**Decision**: 把今日运行期的 shape/VTCM gate **上移到编译期**，并复用 IR 已有的引擎标记。

- 输入：op 形状（matmul 的 M/K/N）、VTCM 预算（16MB）、单元争用（同生命周期内其他 op 是否也要 HMX）。
- 计算：`compiler/include/hnnx/ir/mid_level_ir.hpp:981` 的 `HwWrapper` + `select_wrapper(op_type, dtype, soc_type)` 给出粗粒度 wrapper；`op.hpp:69-84` 的 `OpStoreType`（`RESOURCE_HVX=0x4`/`RESOURCE_HMX=0x8`）+ `get_op_store_type()` 给出 store 类型。映射：`OpStoreMtx→HMX`、`OpStoreVec/Elt→HVX`、`OpStoreFg→scalar`，再叠加 shape 对齐（`%32`）与 VTCM 预算门（`need_mm > 16MB` 则 SPLIT 或回落）。
- 产出：per-op `engine`（写 R1 的 footer）+ 回落策略（写 R5）。

**Rationale**: 引擎概念在 IR 里**已经存在**，只是 emit 时被丢弃（`get_op_store_type` 结果没进 blob）。本 feature 的"成本模型"实质是把现有的 `select_wrapper`/`get_op_store_type`/运行期 gate 串成一条 emit 路径，而非新发明。

**Alternatives considered**:
- *运行时动态成本模型*：即现状，违背 FR-001，弃。
- *纯 shape 表驱动（无争用）*：忽略"单元争用"输入，会在两 op 抢 HMX 时退回非确定性，弃——必须显式建模争用（carve 表已含生命周期，可反推争用）。

## R5 — 显式回落契约 + 执行报告（FR-004/007/008/010）

**Decision**: 回落策略进 R1 的 footer（`fallback_allowed` + `fallback_engine`）。运行期偏离路径二选一，绝不静默：

- `fallback_allowed=1`：用 `fallback_engine` 执行，**记录偏差**进执行报告。
- `fallback_allowed=0`：返回非零、中止执行（硬报错）。

**执行报告**：补齐 `g_last_engine`（`kernels/src/runtime/oplist_exec.c:83`）——当前只在 :494（W4A16 HMX）与 :796（elemwise HVX）两处记录，conv/unary/binary/softmax/rmsnorm2 等 HMX/HVX 路径**漏记成 scalar**。改为每个 op 执行结束时统一记录 `actual_engine`，与 footer 的 `engine` 比对产出 `deviation` 标志。报告载体扩展现有 `op_ts.bin`（ver=2, 20B/op 含 engine u32）为 ver=3：加 `planned_engine` + `deviation` 字段。

**Rationale**: FR-007 的"逐算子报告实际单元+偏差"只有先修好漏记才有意义；footer 里的计划值 + 运行期实际值 + 偏离标志三者齐备，才能让"绝不静默"可验证（spec SC-001/002）。

**Alternatives considered**:
- *回落策略硬编码在运行期*：无法让编译器对数值敏感算子标"禁止回落"（FR-008），弃。
- *偏差只打印不落盘*：不可机验（SC-002 要求"100% 可见且可验证"），弃。

## 待实现阶段切分（与 spec Assumptions 对齐）

- **P1**：R1 + R4 + R5 的 footer 与"照做/报错"骨架（先让编译器决定、运行期服从 + 禁止回落硬报错）。
- **P2**：R2 + R3（carve 表 + OP_DMA 扩到 VTCM 侧）。
- **P3**：R5 的完整执行报告（g_last_engine 补全 + op_ts.bin v3 + 偏差验证）。

> 注意：P3 的"偏差上报通道"是 P1"绝不静默"承诺的强制力来源，实施时与 P1 一起落地，避免出现"编译器决定了但运行期仍能悄悄换"的中间态（见 spec 完成报告的提醒）。
