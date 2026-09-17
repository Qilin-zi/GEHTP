# src/scheduler/ — DP 调度器

对应真实 `dp_sequencer.cc` / `mlh_model.cc`。Data Path 时序调度。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `dp_sequencer.cpp` | `dp_sequencer.cc` | DPSequencer::sequence: 按 DAG 拓扑 + SVF/LVF 并行度排序 op, 输出执行序列 |
| `mlh_model.cpp` | `mlh_model.cc` | MLHModel: Memory Latency Hiding 模型, 估算重叠度 |

## test_e2e 验证

```
Conv(10) -> Relu(20) -> Softmax(30)
sequencer 输出: [10, 20, 30] (拓扑序)
```

SequencerConfig: svf_en=true, svf0_parallelism_cfg=2。
