# GEHTP 内存规划缺口收敛计划（阶段二「DMA 算子 runlist」收口）

> 生成日期 2026-09-21。基线：内存规划现状调查（六项缺口）。所有锚点逐条对代码核实。
> 目标：README M7 阶段二「VTCM 驻留 + DMA 算子 runlist」全部落地。
> 现状：阶段一（静态 DDR 偏移）✅、阶段二前半（VTCM 驻留）✅、阶段二后半（DMA 算子 runlist）❌。

## 0. 一句话现状

功能正确性上，VTCM 分配 / 驻留 / DDR 搬运主链已通（conv_add + spill 变体 + VTCM 双池变体
均 32768/32768 byte-exact）。剩余六项缺口集中在「把 DMA 从 xop 内嵌搬成 runlist 一等公民」
+「分配器补中间层」+「cp_solver 转正」+「M3 清理」。

## 1. 缺口清单与依赖

| ID | 缺口 | 侧 | 依赖 | 风险 |
|---|---|---|---|---|
| A | SPILL/FILL DMA 快路径 | 运行时 | 无 | 低 |
| D | cp_solver 默认化 | 编译器 | 无（需对比报告） | 中 |
| F | 迁移 M3 收尾 | 编译器+emit | 无 | 低 |
| E | RuntimeAllocator 跨组复用 | 编译器 | 无（B 依赖它） | 高 |
| B | SFCD spill/fill 写侧 | 编译器 | E | 高 |
| C | 真 DMA runlist 算子 | 编译器+运行时 | A+B | 高 |

关键路径：**E → B → C**（A 并行推进）；D 独立（先出报告再定默认）；F 独立清理。
A/D/F 三件互不重叠、文件零冲突，可并行开工。

## 2. 分阶段（每阶段独立 commit + 独立验收 + 可回退）

### P0 对比报告（D 前置，可立即启动）
- 内容：cp_solver vs 贪心 FancyAllocator 在 conv_add / spill 变体 / L3 / 0.8B 抽层上，
  按 DDR bytes / peak resident / 编译耗时三轴对比。
- 落点：复用现有 `HNNX_VTCM_ALLOCATOR=cp*` + `HNNX_CP_DEBUG` 通路
  （`cp_solver.cpp:1181` `parse_cp_options_from_env`；`graph_prepare.cpp:4353` CP oracle branch）；
  新增 `compiler/tools/cp_compare` 或 scripts/ 脚本；报告入 `docs/GEHTP_CP_VS_GREEDY.md`。
- 门：三轴数据表产出；结论明确「默认 cp / 保留贪心 / 分场景」三选一。

### P1 运行时 DMA 快路径（A，独立）
- 内容：`oplist_exec.c:597-627` `exec_spill`/`exec_fill` 的标量 `memcpy` → `dc_dma_once`
  （DDR→DDR 1D 描述符）。
- 落点：`kernels/src/runtime/oplist_exec.c`；复用 `dc_parts.c` `dc_dma_init/once/destroy`
  与 `fence.h` 决策表（writer=DMA, reader=DMA, mem=DDR → FO_FLUSH 源侧，dst 引擎 bypass 写）。
- 注意：spill/fill 是 DDR↔DDR，不经过 VTCM；`exec_spill` 的 `0x8000|slot` 源 /
  `exec_fill` 的 `0x8000|slot` 目的编码语义必须保留；`dc_dma_once` 需要 mutex
  （现 matmul 用全局 `mu`，通用 spill/fill 需同锁或按线程）。
- 门：spill 变体 byte-exact 不变（32768/32768）；大 spill（≥1MB）耗时下降入报告；
  conv_add / L3 回归全绿。

- 实施记录（2026-09-21）：v1 已落地——`exec_spill`/`exec_fill` 换 `dma_copy_ddr`
  （`dc_dma_once` 1D 描述符），仅当两端均 DDR 且 `bytes%8==0` 走 DMA 快路径；
  src 先 `dc_clean_ddr`（FLUSH_INVALIDATE）、dst 后 `qurt_mem_cache_clean INVALIDATE`；
  vgather=0 + SWIV 签名通过（`./build_libs.sh one oplist_exec`）。
- **剩余项（勿忘）**：VTCM↔DDR 搬运——`exec_spill` src 为 0x4000 驻留、`exec_fill` dst 为
  `vtcm_off_arr` 驻留时，当前回退标量 `memcpy`。后续：为 VTCM 端点补 `fence.h` VTCM 行
  （CPU/HMX DMA VTCM FLUSH、DMA HMX VTCM INVALIDATE）+ 2D 描述符，收编进真 DMA runlist（P5）。
  `ptr_in_vtcm()` 指针范围判断兜底两种编码（0x4000 / vtcm_off_arr）。
- 验证记录（2026-09-21）：例37 `main` 变体 byte-exact PASS（32768/32768）——修
  `main.c OUT_TEMP 4→0`（manifest `output_temp=0` 是权威，main.c 硬编码 4 是旧值，
  temp4 从未被写 → "no out temp"）。spill 变体 op5–op8（DDR↔DDR DMA 路径）rc=0 全过，
  仅 op9 因下述 emitter bug 失败。
- **剩余项（勿忘）· emitter 侧**：spill 变体 op9 `OP_SPILL src=4` 与末段 transpose 实际
  `y_t=0`（liveness 复用 temp0）不一致 → 设备 "spill src empty"（temp4 从未被写）。
  根因在 `wtop_emit.cpp` spill 插桩用 `em.op_temp[tkey(id,0)]` 记「首次分配 temp id」，
  未随 liveness 复用（release_at/reuse）同步；manifest `output_temp` 用
  `em.src_ref(输出节点输入)` 解析（=0）是对的。修法：spill 插桩 src 改用与 manifest 同源的
  `em.src_ref`，或让 `op_temp` 在 liveness 复用后同步。属编译器侧，与运行时 P1 无关。


### P2 迁移 M3 收尾（F，独立清理）
- 内容：删 `wtop_emit.cpp:389-401` `spill_fill_recs()` 量池段（spill 池尺寸已有真源）；
  删 emit shadow 自算路径 + 两预算旗标；编译器贪心 spill → TAG_SPILL_FILL_OP 重定为
  「成本模型观察记录」（或禁用发射）。
- 落点：`compiler/tools/wtop_emit.cpp`、`compiler/src/ir/graph_prepare.cpp:1454-1460`、
  `scripts/gehtp` 第 4 步。
- 门：tagged round-trip 全绿；conv_add / spill / VTCM 变体 byte-exact 回归；旧 bin 回退
  路径到期评估结论入文档。

### P3 RuntimeAllocator 跨组复用（E，编译器重构，B 前置）
- 内容：恢复 `fa::RuntimeAllocator` 中间层，`FancyAllocator : public RuntimeAllocator`
  （现状 `fancy_allocator.hpp:62` 是独立类，`fancy_allocator.cpp:78` 注释自认
  「跨组复用由 RuntimeAllocator::make_allocator 负责，本仓只能顺序 bump」）。
- 落点：新增 `compiler/include/hnnx/vtcm/runtime_alloc.hpp` + `compiler/src/vtcm/runtime_alloc.cpp`。
  按 BACKPORT_PLAN §1：§1.2 `make_allocator` 槽位工厂（`@0xf4e3b0`）；§1.3 字段布局
  （sizeof 0xb8，tcm_pool_base/size、largest_memory_alloc_size、shared_tensors、
  shared_spillfill_base）；§1.4 `map_block_reference`/`deserialize_blocks` 记录面
  （48B 记录步长、÷3 语义）；§1.5 错误处理两型（返回错误码 vs 抛 out_of_range）。
- 注意：先做「行为保持」重构（继承 + 字段搬移后 ctest 全绿），跨组复用语义
  逐步填进 RuntimeAllocator（算法升级，见 P5 之后的最终态）。
- 门：ctest 全绿；conv_add / L3 byte-exact 回归；`make_allocator` 分派路径单测。

- 实施记录（2026-09-21，task/p3-runtime-alloc）：中间层已恢复——新增
  `compiler/include/hnnx/vtcm/runtime_alloc.hpp` + `compiler/src/vtcm/runtime_alloc.cpp`；
  `FancyAllocator : public RuntimeAllocator`，`mode_` 上移至基类（§1.3 +0x10）；
  `make_allocator`/`make_allocator_at` 槽位工厂落地（§1.2，fancy 分派 + Mode 恒 0 +
  旧者经虚析构删除）；§1.3 字段布局定稿（sizeof 0xb8，含 slot_sizes[3] @+0xa8/+0xac/+0xb0，
  P4 接口承诺，成员偏移由 test_runtime_alloc 运行时钉死）；tensor_base.hpp 旧
  fa::RuntimeAllocator stub 移除改含 runtime_alloc.hpp（ODR 收口）；
  map_block_reference/deserialize_blocks 维持仅声明（记录面算法属 P5+）；
  RuntimeAllocator 暂不继承 hnnx::Allocator（原因见 runtime_alloc.hpp 头注，M35 统一）。
  验收：ctest 45/45 全绿（44 旧 + test_runtime_alloc 53 项）；本机 GCC 9.4 构建需
  `-include /tmp/p3_gcc9_bit_cast_shim.h`（__builtin_bit_cast 为 GCC11+ 内置，
  仅构建目录注入）。跨组复用算法不在本阶段，行为与基线逐字节一致。

### P4 SFCD spill/fill 写侧（B，依赖 E）
- 内容：`spillfill_g4.cpp`（现 388 行，仅读侧）补 §C 分配面 / §D checkpoint op / §E 写侧真体。
- 落点：`compiler/src/vtcm/spillfill_g4.cpp` + `include/hnnx/vtcm/spillfill_g4.hpp`。
  按 BACKPORT_PLAN §2 方法清单：§2.1 tcm 块记录两词头（`(pool<<16)|nblocks` 占位回填 +
  blob 偏移；终结器 push {0, blob_off}）；§2.2 wait 记录窄/宽式判定；§2.3 `:78` 残差语义
  （find_peak_tcm_usage blocks = bytes>>11）；§2.4 常量池 ID（G4_SPILLFILL_POOL_ID=2 等）；
  §2.5 `g4_set_spillfill_size` / `g4_fill_slots_multi` / `g4_insert_spill_fill` /
  `g4_make_dma_checkpoint_op` / `g4_dlbc_spill_fill_setup` / `g4_sfcd_finalize_copies` 等。
- 门：以本地 REQNNFRAME 779 行写侧为 oracle，SFCD 字节对拍一致；写侧产出能被读侧
  `g4_dump_sfcd` 正确回读（round-trip 自洽）。

### P5 真 DMA runlist 算子（C，依赖 A+B）
- 内容：把 matmul 内嵌 `cpu_to_vtcm`/`dc_dma_once`（`oplist_exec.c:400-409`）收编为显式
  runlist DMA 条目，引擎按 runlist 派发 DMA 而非 xop 内硬编码。
- 落点：编译器侧 `compiler/src/dma/op_emitter.cpp:262/270`（现 placeholder
  `DmaOpType::Spill/Fill`，注释「real .bin uses OP_TYPE_DMA」）改为真发射；
  运行时侧 `oplist_exec.c` 新增 `exec_dma` 处理 + opcode 登记。
- 语义：DMA 算子携带 (src, dst, bytes, src_bypass, dst_bypass, fence 三元组)，
  引擎走 `dc_dma_once` + `fence_handoff`。
- 门：matmul 快路径行为不变（byte-exact）；显式 DMA runlist 在 conv_add / L3 上执行序正确；
  optrace 逐 op 轨迹中 DMA 算子可见。

### P6 cp_solver 默认化（D 收口，依赖 P0 报告 + E）
- 内容：依 P0 结论定默认。若默认 cp，则 `parse_cp_options_from_env` 缺省回 true
  （或去掉 env 门控），`vtcm_lifetime_alloc_cp` 失败回退贪心作为正式契约。
- 落点：`compiler/src/vtcm/cp_solver.cpp:1181` + `graph_prepare.cpp:4353`。
- 门：全量回归 byte-exact；编译耗时/内存下降或持平入报告；`post_spill_fill_design_pass`
  （`graph_prepare.cpp:4530`）对 `cp_plans_` 的消费路径实测生效。

## 3. 跨阶段约束与错峰

- 运行时侧（P1/P5）触碰 `oplist_exec.c` — 与 `task/prof-wp` 在途（B1 快速件接线）重叠，
  动工前按 §2.3 查 mtime + ListAgents 错峰。
- 编译器侧（P2/P3/P4/P6）触碰 `graph_prepare.cpp` / `fancy_allocator` / `spillfill_g4` —
  与 VL 线在途（graph_prepare.cpp / ops.cpp）错峰。
- 每阶段独立 commit、独立可回退。P1/P2/P3/P6 不改 blob 记录格式 → 设备侧零回退；
  P4/P5 若改 runlist 记录格式，同 PR 同步 deserialize / parse 消费方 + round-trip 守门。

## 4. 验收门总表

| 阶段 | 门 | 命令/位置 | 期望 |
|---|---|---|---|
| P0 | 对比报告 | cp_compare 脚本 | 三轴数据表 + 默认决策 |
| P1 | spill 变体 byte-exact | --ddr-budget 4096 | 32768/32768 不变 |
| P1 | 大 spill 耗时 | ≥1MB spill | 下降入报告 |
| P2 | tagged round-trip | compiler tests | serialize→deserialize 恒等 |
| P3 | ctest + conv_add 回归 | ctest + quickstart | 全绿 + byte-exact |
| P4 | SFCD 写读 round-trip | g4_dump_sfcd | 写侧产物回读自洽 |
| P5 | matmul 快路径不变 | conv_add / L3 | byte-exact |
| P6 | 全量回归 | quickstart + 四模型 | byte-exact + top1 保 |
