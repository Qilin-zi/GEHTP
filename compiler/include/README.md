# include/ — C++ 头文件

与 `src/` 子模块一一对应，全部在 `hnnx/` 命名空间下。字段偏移注释来自反汇编（`+0xNNN` 标注真实二进制布局）。

## 关键类型 (hnnx/ir/types.hpp)

- `OpDef` — 图节点 (op_id, flags, name_tag, inputs, output_def, const_data_offset, op_data)
- `Op` — 执行 op 虚基类 (cost / serialize_internal / execute)
- `InputConn` / `OutputDef` / `InputDef` — 连接与形状描述
- `DType` — Float32/Int8/Int4/FP8/MXFP4 等 13 种
- `OpDefFlags` — ENABLED/CONST/DEAD/SHAPE/DYNAMIC/SLICED

## 各头文件

| 头文件 | 内容 |
|--------|------|
| `ir/graph_prepare.hpp` | GraphPrepare 类 (prepare/optimize/serialize/deserialize + for_each_op) |
| `ir/types.hpp` | OpDef / Op / Graph / DType / fibonacci_hash |
| `ir/op_registry.hpp` | OpRegistry 单例 (register_op / generate_by_name) |
| `ir/tensor.hpp` | Tensor |
| `opt/optimization_passes.hpp` | 6 阶段阈值 / GraphOptInfo / FusionRule |
| `ops/ops.hpp` | TypicalOp + 200+ op 声明 + HwWrapper 指令选择 |
| `ops/hlx_hmx.hpp` | HLX/HMX 扩展 op (17 个) |
| `serialize/serializer.hpp` | Serializer / Deserz / Deserializer + BinFormatTag |
| `vtcm/fancy_allocator.hpp` | FancyAllocator + VtcmCacheInstance |
| `tiling/tiler.hpp` | Tiler / SimpleTiler / ConvTiler / MatMulTiler / Supertiler |
| `cost/cost_model.hpp` | CostSource |
| `mcast/mcast_optimizer.hpp` | McastOptimizer / McSend |
| `scheduler/dp_sequencer.hpp` | DPSequencer / SequencerConfig / DPOpGraph |
| `dma/spill_fill.hpp` | SpillFill / OpEmitter |
| `api/hexagon_nn_env.hpp` | HexagonNNEnv / OpIoPtrs / OpRef |

总: 18 个 `.hpp`, 75 KB。
