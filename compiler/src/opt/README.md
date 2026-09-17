# src/opt/ — 优化 pass

对应真实 `graph_opt_pass.cc` / `tcm_migration.cc`。6 阶段优化调度 + fusion 规则集。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `optimization_passes.cpp` | `graph_opt_pass.cc` @ 0x11C12E0 等 | GraphOptInfo/Pass/Context, attempt (hash+arity 匹配), **apply_fusion_rules** (8 条规则), run_phase_fixpoint |
| `fixpoint.cpp` | `graph_prepare.cc` Phase vfunc[6] | fixpoint 循环: DCE -> order -> CSE -> clear dirty |

## 6 阶段 (types.hpp)

```
PHASE_0=3000  PHASE_1=10190  PHASE_2=11892
PHASE_3=12492 PHASE_4=21101  PHASE_5=22000  TERM=0xFFFFFFFF
```

## fusion 规则集 (apply_fusion_rules)

```
Conv + Relu    -> ConvActivations
Conv + Clamp  -> ConvActivations
MatMul + Add  -> MatMul        (bias 融合)
MatMul + Gelu -> MatMul
MatMul + Relu -> MatMul
Add + Relu    -> Add
Add + Sigmoid -> Add
Dense + Add   -> Dense
```

匹配条件: consumer 恰好 1 输入, producer 只被该 consumer 使用; 融合后 producer 标 dead, DCE 回收。
