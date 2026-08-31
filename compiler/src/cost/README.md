# src/cost/ — 代价模型

对应真实 `cost_model.cc`。为 op 工厂选最低 cost 候选提供估算。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `cost_model.cpp` | `cost_model.cc` | CostSource: init_for_soc("v75"/"v73"), get_prediction_from_cost_model (查表 + 解析公式) |

## 用法

```cpp
costbased::CostSource cs;
cs.init_for_soc("v75");
float cost = cs.get_prediction_from_cost_model("Conv", op, nullptr, {});
```

test_e2e 验证:
- Conv 查表 cost = 1000
- analytical (1x222x222x32) cost ≈ 1.4e+07
