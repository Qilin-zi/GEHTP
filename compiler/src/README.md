# src/ — HTP 图编译器 C++ 源码

按真实 `libQnnHtp.so` 的内部组织分模块实现，每个子目录对应一组 `.cc` 源文件。代码内注释标注反汇编来源（`Source: <file>.cc @ 0x<addr> (<size> bytes)`）。

## 子模块

| 目录 | 对应真实源文件 | 职责 |
|------|----------------|------|
| `ir/` | `graph_prepare.cc`, `op_def.cc` | 图 IR: GraphPrepare / OpDef / 拓扑排序 / DCE / CSE / const_prop / gen_Shape / 序列化主循环 |
| `opt/` | `graph_opt_pass.cc`, `tcm_migration.cc` | 优化 pass: 6 阶段 PHASE_0..5 调度 / fixpoint / fusion 规则集 |
| `cost/` | `cost_model.cc` | 代价模型: v75/v73 init + 解析估算 |
| `vtcm/` | `vtcm_alloc.cc` | VTCM 分配器: 线性 bump 分配 / 块表 / 持久化池 |
| `tiling/` | `tiler.cc`, `simple_tiler.cc` | tiling: SimpleTiler / ConvTiler / MatMulTiler / 分发 / conform |
| `serialize/` | `serialize_oplist.cc`, `deserializer.cc` | .bin 序列化/反序列化: tagged-record 流 / 双模式 (prescan+write) |
| `dma/` | `spill_fill.cc`, `op_emitter.cc` | DMA: spill/fill 插入 / op 发射 / preload |
| `mcast/` | `grdep_mcast_optimizer.cc` | 多播优化: supercast 合并 + ILP(CLP/HiGHS) 框架 |
| `scheduler/` | `dp_sequencer.cc`, `mlh_model.cc` | DP 调度器: SVF/LVF + MLH 模型 |
| `api/` | `c_interface.cpp`, `hexagon_nn_env.cc` | C API 入口 + 编译环境 |
| `ops/` | `*.cc` (200+ op) | op 库: 154 op 注册 / 参数解析 / 指令选择 / host execute |

## 关键文件大小

- `ir/graph_prepare.cpp` 46 KB — 核心，最多反汇编注释 (134 条)
- `ops/ops.cpp` 26 KB — 137 个 op 注册 + execute 分派
- `vtcm/fancy_allocator.cpp` 10 KB — VTCM 分配
- `cost/cost_model.cpp` 10 KB

总: 23 个 `.cpp`, 186 KB。
