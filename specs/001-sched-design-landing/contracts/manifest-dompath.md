# Contract: manifest dompath_estimate 可选字段

> 落点：`compiler/tools/wtop_emit.cpp` manifest 写出段（617-646）。2026-09-17。

## 写出契约

- `*.wtop.manifest.json` 在既有键之外新增可选键 `dompath_estimate`
- 字段集与约束见 [../data-model.md](../data-model.md) E3
- 估计失败/无表覆盖时：**键整体缺省**（不写 `"cycles": 0` 这类伪值）

## 消费契约（向后兼容）

- 消费方（gehtp run、host_run、外部脚本）对缺失键必须按"无估计"处理，禁止报错
- 键存在时 `source` 必填且 ∈ {"table","measured"}；`per_op_table_version` ≥ 1
- 不变式：`cycles ≤` 同图实测 wall；校验脚本读 manifest + 实测 wall 对照

## 计算契约

- 求和沿定稿序（TAG_PLAN_ORDER 拓扑序）的关键依赖链，不是全图简单求和
- per-op 周期表初版为编译器内置静态表（粗粒度：opcode 类 × 形状档），
  表版本号 `per_op_table_version` 从 1 起；任何表内容变更 bump 版本
- 估计来源为估计表时 `source="table"`；43_hwinfo/optrace 校准后才允许 `"measured"`

## 验收

- 编译 conv_add / L3：manifest 含字段且 `source="table"`
- host_run 对拍不变；旧 blob（无字段）被 gehtp run 正常消费
- 校验：conv_add 的 `cycles` ≤ 其设备实测 wall（设备恢复后复核；挂起不阻塞）
