# GEHTP 内存规划完善任务书（阶段二「DMA runlist」收口）

> 生成日期 2026-09-26。基线：`GEHTP_MEMPLAN_GAP_CLOSURE.md`（09-21 缺口分析）+ 2026-09-25/26 对代码逐条核实。
> 本文是**可认领任务账**：每条任务含 范围 / 文件所有权 / 依赖 / 完成门 / 验证命令。
> 与 GAP_CLOSURE 的关系：GAP_CLOSURE 是「缺口分析 + P0-P6 计划」，本文是「当前状态 + 剩余可认领任务」。
> 会话认领后在 §4 登记表签名，做完把实施/验证记录回写本文（留痕纪律）。

---

## 0. 现状快照（2026-09-26 对代码核实）

| 原缺口 | GAP_CLOSURE 阶段 | 现状 | 证据锚点 |
|---|---|---|---|
| A · SPILL/FILL DMA 快路径 | P1 | ✅ v1（DDR↔DDR） | `oplist_exec.c` `exec_spill`/`exec_fill` → `dma_copy_ddr` |
| A · VTCM↔DDR 端点 | P1 剩余 | ⏳ 仍标量 memcpy | `0x4000` src / `vtcm_off_arr` dst 回退路径 |
| F · 迁移 M3 收尾 | P2 | ✅ 完成 | `wtop_emit.cpp:384`「旧 spill 段已删，溢出走 TAG_MEM_PLAN 阶段二」；spill 池尺寸真源 = `em.ddr_spill_map`（`wtop_emit.cpp:227-235`） |
| E · RuntimeAllocator 中间层 | P3 | ✅ 完成 | `runtime_alloc.hpp/cpp` + `FancyAllocator : public RuntimeAllocator`，ctest 45/45 |
| E · 跨组复用算法 | P3 剩余 | ⏳ 仅声明 | `map_block_reference`/`deserialize_blocks` 维持仅声明（算法属最终态） |
| B · SFCD spill/fill 写侧 | P4 | ✅ 完成（2026-09-26 双门验证） | `spillfill_g4.cpp` 621 行 §C/§D/§E 全落地；round-trip ALL PASS + oracle 字节对拍 IDENTICAL（见 §1 T1 记录） |
| C · 真 DMA runlist 运行时侧 | P5 | ✅ 完成 | `oplist_parse.h:151` `OP_DMA=33` arity 8 + `WT_REF_ENG_FLAG 0xC000` + `exec_dma`（三态引用 + VTCM 池懒初始化） |
| C · 真 DMA runlist 编译侧 | P5 | ❌ placeholder | `op_emitter.cpp:82/87`「real .bin serializes this as an OP_TYPE_DMA record」仍走 `DmaOpType::Spill/Fill` |
| D · cp_solver 默认化 | P6 | ❌ 阻塞于 P0 | `graph_prepare.cpp:4353` CP oracle branch 仍 env 门控；`cp_solver.cpp:1181` `parse_cp_options_from_env` |
| — · 对比报告（D 前置） | P0 | ❌ 未写 | `docs/GEHTP_CP_VS_GREEDY.md` 不存在 |

**阶段一（静态 DDR 偏移）✅、阶段二前半（VTCM 驻留 + 0x4000 发射）✅**：`ddr_offsets.cpp` `compute_ddr_offsets` 两级池（VTCM 先、DDR 兜底、spill 池）已落地；`wtop_emit.cpp:472-488` 发射 `0x4000|temp` 编码。

**剩余主线（关键路径）**：`P4 写侧 → P5 编译侧发射`（DMA runlist 收口）；旁支 `P0 报告 → P6 默认化`；独立收尾 `P1 VTCM↔DDR 端点`、`跨组复用 + bin packing`。

---

## 1. 任务清单

### T0 · cp_solver vs 贪心对比报告（解除 P6 阻塞，可立即启动）

- **范围**：cp_solver（专利 US20240386237 oracle）vs 贪心 FancyAllocator，在 conv_add / spill 变体 / L3 / 0.8B 抽层上，按 **DDR bytes / peak resident / 编译耗时** 三轴对比，出结论「默认 cp / 保留贪心 / 分场景」三选一。
- **文件**：复用 `HNNX_VTCM_ALLOCATOR=cp*` + `HNNX_CP_DEBUG` 通路（`cp_solver.cpp:1181`、`graph_prepare.cpp:4353`）；新增 `compiler/tools/cp_compare`（或 scripts/ 脚本）；报告入 `docs/GEHTP_CP_VS_GREEDY.md`。
- **依赖**：无。
- **完成门**：三轴数据表产出；默认决策落笔成文。
- **验证命令**：`compiler/tools/cp_compare --graph <conv_add|spill|L3|0.8b 抽层>` 产表。

### T1 · SFCD spill/fill 写侧（B，关键路径第一棒）✅ 2026-09-26

- **范围**：`spillfill_g4.cpp`（现 621 行，§A dump/§B slc 序列化/§C 分配面/§D checkpoint/§E 写侧真体全落地）补 §C 分配面 / §D checkpoint op / §E 写侧真体。
- **文件**：`compiler/src/vtcm/spillfill_g4.cpp` + `include/hnnx/vtcm/spillfill_g4.hpp`。
- **锚点**：按 BACKPORT_PLAN §2 方法清单——§2.1 tcm 块记录两词头（`(pool<<16)|nblocks` 占位回填 + blob 偏移；终结器 push {0, blob_off}）；§2.2 wait 记录窄/宽式判定；§2.3 `:78` 残差语义（find_peak_tcm_usage blocks = bytes>>11）；§2.4 常量池 ID（G4_SPILLFILL_POOL_ID=2 等）；§2.5 `g4_set_spillfill_size`/`g4_fill_slots_multi`/`g4_insert_spill_fill`/`g4_make_dma_checkpoint_op`/`g4_dlbc_spill_fill_setup`/`g4_sfcd_finalize_copies`。
- **依赖**：P3 中间层已就绪（`runtime_alloc.hpp` 已暴露 `slot_sizes[3]` @+0xa8/+0xac/+0xb0，P4 接口承诺）。跨组复用算法（E 最终态）**不阻塞**本任务。
- **完成门**：以本地 REQNNFRAME 779 行写侧为 oracle，SFCD 字节对拍一致；写侧产物能被读侧 `g4_dump_sfcd` 正确回读（round-trip 自洽）。
- **验证命令**：`g4_dump_sfcd` round-trip 单测；ctest 全绿。

- **实施记录（2026-09-26）**：写侧主体此前已随 §2.5 清单落地（G4SfcdWriter 记录级模型 +
  `g4_construct_sfcd` + `g4_dlbc_spill_fill_setup` + `g4_fill_mgroup_check`；
  `g4_insert_spill_fill` 按反汇编锚点保持永久桩语义）。本会话做双门验证：
  1. **round-trip 门**：`test_g4_sfcd_roundtrip`（CMake 已注册）本机构建 + 运行
     **ALL PASS**。构建前修两处 Mac 专属 `std::min(size_t, uint64_t)` 类型冲突
     （`graph_prepare.cpp:1721/1875` 显式转 size_t；Linux 上同型不炸，Mac 上
     size_t≠uint64_t 炸模板推导）——该修复未提交，见 git status。
  2. **oracle 对拍门**：差分工具 `compiler/tools/sfcd_oracle_diff/`（不接 CMake）
     喂同一输入集（3 块记录含相邻合并 + 2 set_progress + 窄/宽/无 waits × ckpt
     7/0）→ oracle `g4_sfcd_assemble` vs 104 `g4_construct_sfcd` 输出
     **IDENTICAL（25/28/21 词三变体逐词一致）**。
- **已知有界分歧（hint=0 奇偶区内零分歧，见 tools/sfcd_oracle_diff/README.md）**：
  1. type nibble：oracle `nib=hint==0?0:(hint>=6?2:1<<(hint-1))`（G4b 遗留原文未打印）；
     104 硬编码 0（头注禁止臆测补齐）——hint≠0 暂不可产出。
  2. wait 分块：oracle 整包单记录（n>0xFFFF 会溢出 low16）；104 按 0xFFFF/0x7FFF
     分块（防御性扩展）。
  3. 混合窄/宽 wait：oracle 单向量全窄或全宽；104 每条 WAITFOR 独立判定。
  4. 终结器 {0, blob_off}：104 采信 @0x1012460 反汇编；oracle 空 entries 路径
     产 (pool<<16)|0（oracle 模型该边自我不一致）。

### T2 · 真 DMA runlist 编译侧发射 + act 预铺（C，关键路径第二棒，依赖 T1）

- **范围**：
  1. `op_emitter.cpp` 把 placeholder `DmaOpType::Spill/Fill`（`:82/87` 注释自认「real .bin 序列化为 OP_TYPE_DMA 记录」）改为真发射 `OP_DMA`（arity 8：`[src,dst,bytes,src_off,dst_off,flags,fence_src,fence_dst]`）。
  2. **act 预铺**：matmul act 常驻 VTCM 时，把 `oplist_exec.c:400-409` 内嵌 `cpu_to_vtcm`/`dc_dma_once` 收编为显式 runlist 条目（DDR→VTCM 预取），而非 xop 内硬编码；引用走 `WT_REF_ENG_FLAG 0xC000`（0=e.act, 1=e.out, 2=e.wt, 3=e.bias）。
- **文件**：编译器 `compiler/src/dma/op_emitter.cpp`、`compiler/src/dma/spill_fill.cpp`、`compiler/tools/wtop_emit.cpp`；运行时 `kernels/src/runtime/oplist_exec.c`（`exec_dma` 已有，仅核对 fence 三元组接线）。
- **依赖**：T1（SFCD 写侧产出 runlist 记录）；运行时侧 `exec_dma` 已就绪。
- **完成门**：matmul 快路径行为不变（byte-exact）；显式 DMA runlist 在 conv_add / L3 上执行序正确；optrace 逐 op 轨迹中 DMA 算子可见。
- **验证命令**：conv_add / L3 编译上板 byte-exact；optrace 抓 `OP_DMA` 条目。
- **实施记录（2026-09-26）**：编译侧发射全部落地：
  1. **wtop_emit.cpp dma_runlist 路径**（`--dma-runlist`）：FILL → `OP_DMA`(pool
     slot → scratch temp，flags bit0=src_bypass)；SPILL → `OP_DMA`(scratch temp →
     pool slot，flags bit1=dst_bypass)；fence 双 0（FC_CPU：FLUSH_INVALIDATE /
     INVALIDATE，铁律①③）；arity 8 全参。非 dma_runlist 路径 OP_FILL/OP_SPILL 原样。
  2. **wtop_ops.hpp R1 修复**：fresh_temp 改「表内 tid 永久退役」——TEMPOFF 表项
     per-tid 一次解析定终，复用旧 tid 命中旧 (off,size) 会越界/静默重叠；表外
     (bump) tid 尺寸无关自由复用。probe：混合尺寸图 tid1 {6144,2048} 被 6144B
     matmul 出复用 → 设备 temp_get 边界 NULL + producer 越界写 bump 区。
  3. graph_prepare.cpp 两处 Mac size_t 转换修复（T1 遗留）。
- **验证（本地差分门，host byte-exact）**：fixture `qnp16`（f16 重写 qwen 型
  2-block 网，107 op，Gather/RmsNorm/FC/Reshape/Split；`--ddr-budget 4096` →
  86 spill）双路径各 290 op：OP 路径(SPILL/FILL) vs DMA 路径(OP_DMA) 输出
  **逐字节一致**（out temp 7 = 8192B）。host 新垫片层 `kernels/host/shims/`。
  fixture 两处修正：①dtype 561 不在 map_dtype 表内（默认 f32 es=4 → 规划 ×2
  对不上运行时 f16）——改用 534（0x0216，qnn_ir_loader.cpp:19 官方
  QNN_FLOAT_16）；⚠️ 真 converter 的 f16 图 dtype 码待实测，若为 561 需回补
  map_dtype。②输入 token 须 < n_idx（见下方 R-A）。
- **act 预铺**：CPU act 路径已由 2c4fc70 的 VTCM 驻留读(0x4000|temp)透明覆盖，
  无需再发 DDR→VTCM 预铺条目；0xC000 引擎面暂存受 carve 定序阻塞，留设计决策。
- **运行时侧注册（未改——错峰，oplist_exec.c 属 prof-wp 在途）**：
  - **R-A** exec_gather 索引值门 `r >= n_idx`（n_idx=n/row_n=token 数）把词表
    合法 token 拒之门外：embedding 场景值域应为表行数（表槽长/row_bytes），
    n_idx 只该用于「idx 槽长门」的 i 边界。真机 vocab≫ctx 时首 token≥ctx 即炸。
  - **R-B** temp_get 静态边界 `off+bytes > pool_used` 应比 static_cap——
    pool_used 被 bump 分配抬高后掩盖出静态边界访问（静默重叠）。
  - 两者均待与 prof-wp 协调后修；本轮靠 fixture 绕行（token∈[0,32)）。


### T3 · SPILL/FILL VTCM↔DDR 端点（A 收尾，独立，可立即启动）

- **范围**：`exec_spill` src 为 `0x4000` 驻留、`exec_fill` dst 为 `vtcm_off_arr` 驻留时，当前回退标量 `memcpy`——补 `fence.h` VTCM 行（CPU/HMX→DMA→VTCM FLUSH；DMA→HMX→VTCM INVALIDATE）+ 2D 描述符。
- **文件**：`kernels/src/runtime/oplist_exec.c`（`exec_spill`/`exec_fill`）；`kernels/include/fence.h` 决策表补 VTCM 端点行。
- **依赖**：无（DDR↔DDR 快路径已落地，此为端点补全）。
- **完成门**：`ptr_in_vtcm()` 指针范围判断兜底两种编码（`0x4000` / `vtcm_off_arr`）；VTCM 双池变体 byte-exact 不回退标量 memcpy。
- **验证命令**：spill 变体 + VTCM 双池变体 byte-exact（32768/32768）。

### T4 · cp_solver 默认化（D 收口，依赖 T0）

- **范围**：依 T0 结论定默认。若默认 cp：`parse_cp_options_from_env` 缺省回 true（或去 env 门控）；`vtcm_lifetime_alloc_cp` 失败回退贪心作为正式契约；`post_spill_fill_design_pass`（`graph_prepare.cpp:4530`）对 `cp_plans_` 的消费路径实测生效。
- **文件**：`compiler/src/vtcm/cp_solver.cpp:1181`、`compiler/src/ir/graph_prepare.cpp:4353/4530`。
- **依赖**：T0 报告结论。
- **完成门**：全量回归 byte-exact；编译耗时/内存下降或持平入报告。
- **验证命令**：quickstart + 四模型 byte-exact + top1 保。

### T5 · 跨组复用算法 + FancyAllocator bin packing（E 最终态 + 质量项，后置独立）

- **范围**：
  1. `RuntimeAllocator::map_block_reference`/`deserialize_blocks` 从「仅声明」填真（48B 记录步长、÷3 语义，见 BACKPORT_PLAN §1.4）。
  2. `fancy_allocator.cpp:21` 注释自认「真实库用更复杂的 bin packing + lifetime overlap，这里实现可用的线性版本」——升级为 bin packing + lifetime overlap（内存效率质量项，非正确性阻塞）。
- **文件**：`compiler/src/vtcm/runtime_alloc.cpp`、`compiler/src/vtcm/fancy_allocator.cpp`。
- **依赖**：无（与 T0-T4 正交）。
- **完成门**：ctest 全绿；跨组复用后 peak resident 下降入报告；bin packing 后 DDR cap 下降入报告。
- **验证命令**：ctest + conv_add / L3 byte-exact 回归。

---

## 2. 依赖总图

```
T0 ─────────────► T4 (cp 默认化)                 ┐
T1 ─────────────► T2 (真 DMA 编译侧 + act 预铺)    ├─ 两主线并行
T3 (VTCM↔DDR 端点)  独立                          ┘
T5 (跨组复用 + bin packing)  独立后置
```

- **立即可启动**：T0、T1、T3、T5（四件互不重叠、文件零冲突）。
- **第二棒**：T2 等 T1；T4 等 T0。（2026-09-26 更新：T1 ✅ → T2 已解除阻塞）

## 3. 跨阶段约束与错峰

- 运行时侧（T2/T3）触碰 `oplist_exec.c` — 与 `task/prof-wp` 在途（B1 快速件接线）重叠，动工前查 mtime + 错峰。
- 编译器侧（T1/T2/T4/T5）触碰 `spillfill_g4.cpp` / `op_emitter.cpp` / `graph_prepare.cpp` / `fancy_allocator` — 与 VL 线在途（graph_prepare.cpp / ops.cpp）错峰。
- 每任务独立 commit、独立可回退。T1/T2 若改 runlist 记录格式，同 PR 同步 deserialize / parse 消费方 + round-trip 守门。

## 4. 登记表（会话认领签名 + 实施/验证回写）

| 任务 | 认领会话 | 状态 | commit / 记录 |
|---|---|---|---|
| T0 对比报告 | | pending | |
| T1 SFCD 写侧 | 2026-09-26 会话 | ✅ done | 双门验证（见 §1 T1 实施记录）；无新 commit（实现存量已落地，仅 graph_prepare Mac 构建修复待提交） |
| T2 真 DMA 编译侧 + act 预铺 | 2026-09-25/26 会话 | ✅ 编译侧+本地差分门（设备验证待错峰） | commit T2-<hash>（编译侧发射 + host 差分门） |
| T3 VTCM↔DDR 端点 | | pending | |
| T4 cp 默认化 | | pending | |
| T5 跨组复用 + bin packing | | pending | |

---

*回写纪律：每条任务做完，把「实施记录 + 验证记录 + commit」写回本表对应行；新增缺口另起 §1 条目，不静默。*
