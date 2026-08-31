# src/ir/ — 图 IR 核心

图编译器核心,对应真实 `graph_prepare.cc` / `op_def.cc`。反汇编注释最密集 (134 条)。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `graph_prepare.cpp` | `graph_prepare.cc` @ 0xF66360 等 | GraphPrepare 全部: prepare/do_prepare1/2/late, 拓扑排序, DCE, CSE, const_prop, gen_Shape, serialize/deserialize 主循环, fusion 调用 |
| `opdef.cpp` | `op_def.cc` | string_tag_t::map_str, OpDef::hash_key/input_count, OpDef_Const 构造, fibonacci_hash |
| `op_registry.cpp` | `op_registry_prepare.cc` @ 0x10BE710 | OpRegistry::generate (从 io.opdef_ptr 取名 -> generate_by_name) |
| `tensor.cpp` | `tensor_prepare.cc` | Tensor::persistent_clone/num_elements, tensor_generator_scalar |

## 关键实现

- `do_prepare1`: const_prop -> gen_Shape(形状推断) -> order_nodes -> allocate_io_tensors
- `run_optimize_passes_single_registry`: 6 阶段 PHASE_0..5 fixpoint + tcm_migration + fusion
- `do_serialize`: config 记录 + const pool + block table + segment plan + IO + runlist ops + 0xBEEFF00D
- `deserialize`: 解析 tagged-record 流重建 GraphPrepare (含 const pool 往返)
