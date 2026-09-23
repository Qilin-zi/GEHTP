# Contract: 访问类型分类谓词（access-classify）

> 内部编译器接口契约。落点：`compiler/src/vtcm/ddr_offsets.cpp` 调用点 +
> `op_def.hpp` 旁新静态表。2026-09-17。

## 静态表（新文件，不改既有 OpDef 布局）

```cpp
// 每个 opcode 的每个操作数的访问类（消费方视角）
enum class HnnxAccessClass : uint8_t { VECTOR, HMX, DMA, SCALAR, UNKNOWN };

// 表项: (opcode, operand_idx) -> class；查不到 = UNKNOWN
// 初版覆盖 20 型 opcode 家谱（0.8B 全模型集合）
```

## 谓词（编译期）

```
bool tensor_vtcm_eligible(GraphPrepare&, uint64_t producer_op_id, uint32_t out_idx,
                          std::string* verdict, std::string* evidence);
```

- 返回 true = 可参与 VTCM 池分配；false = 强制 DDR
- verdict ∈ {"vtcm-ok", "ddr-scalar", "ddr-unknown", "ddr-output"}；evidence = 消费者明细
- 调用点：`compute_ddr_offsets` 构建 `reqs` 后、VTCM 池分配前；false 的请求直接跳过
  VTCM 尝试进 DDR 队列
- `vtcm_budget == 0`（阶段一默认）时 pass 不执行、行为逐字节不变

## 行为保持不变式

- 默认配置（无 --plan-vtcm-budget）下产物与基线逐字节一致
- 加谓词后唯一允许的变化：`--plan-vtcm-budget > 0` 时 SCALAR/UNKNOWN 张量不再带 in_vtcm

## 验收

- 单测：合成图（MatMul 激活 + Gather 索引 scratch）→ 激活 vtcm-ok、scratch ddr-scalar
- 回归：conv_add ×3、spill 变体、L3 |d|≤0.001、ctest 44/44 + 新增分类测试
