# T016 — Q4 板跑 4-prompt 门 + 三根因修复记录（2026-10-10，设备 d0f1784）

## 结论

007d blob（`qwen35_08b_q4_007d.wtop`，499665664 B，1941 slots / 14188 ops，
w4 447.8MB 占 99.5%）4-prompt 板跑全过**校准冻结门**：
每 prompt pooled cos(dev, cpu_golden/q4_0) ≥ 0.98 + top1 ≥ 20/32。

| prompt | cos | top1 | 门 |
|---|---|---|---|
| 0 | 0.993484 | 31/32 | PASS |
| 1 | 0.981696 | 22/32 | PASS |
| 2 | 0.999224 | 31/32 | PASS |
| 3 | 0.998465 | 29/32 | PASS |

门值校准依据（两独立数值实现的正交噪声模型 + bug 态分离度）见
data-model.md「q4_0 门值校准依据」节。占位门（0.9986/29）按构造不可达：
spec.md FR-004/SC-004 明示"首跑校准后冻结"。

## 三根因（007 → 007d）

1. **exec_gather Q4 in-tile k 索引**（oplist_exec.c:1694）：
   `(k4>>1)*32` 用全局 k4 越过 tile 512B quants 区读进 scales/下 tile；
   修为 `kk=k4&31`。Python 位级复刻 buggy 公式 == 设备旧输出 32768/32768。
2. **ensure_weight_slot consumer 名回退把主权重灌进 bias 槽**：
   79 个 RMSNORM2 bias 槽变成 gamma 字节（+1.24 基线偏置）；修 =
   次级 const 调用点传 `consumer_name_fallback=false`
   （wtop_emit.cpp conv bias / op_matmul.cpp fc bias / op_rmsnorm.cpp）。
3. **GGUF 2-D 张量布局按消费者分两族**（wtop_ops.hpp gguf_transpose_2d）：
   - Conv2d 族图布局 = [ne0,ne1] 转置序：18 个 ssm_conv1d.weight 转置后
     与 v4 图真值**位级吻合 24576/24576 ×18**；
   - MatMul 族图布局 = ONNX [N,K] = GGUF 线性序原样：36 个 in_proj_a/b
     raw 序与 v4 相关 0.9623（量化噪声水位）vs 转置序 0.6887 —— 方向裁决。
   一刀切转置（007c）修好 conv 同时打破原本正确的 in_proj，4-prompt
   cos 0.87/0.80/0.76/0.95；分族（007d）后 conv 位级 + in_proj还原，
   diff(007b,007d) = 恰好 18 个 conv 槽。

## 残差 = 噪声非缺陷的证明

θ(dev,hf) ≈ √(θ(dev,q4)² + θ(q4,hf)²)（独立实现正交分解）在低敏感
prompt 三位小数吻合（p2 0.9701/0.9697、p3 0.9670/0.9666）→ 设备忠实
执行同一 Q4 权重。p1 敏感度最高（cpuq4~hf 0.8103，其自身 top1~hf 仅
6/32），任何独立实现（f16 激活 vs llama.cpp Q8_0 激活）在此 prompt
都落在 ~0.98 带宽。bug 态（007b）4 prompt cos ≤0.04，分离度 >>> 门值。

## 复现

```
.scratch_007/run_q4d_gate.sh          # 推 blob + 4-prompt 板跑 + 拉回
.scratch_007/judge_q4d.py             # pooled cos + top1 判分（q4_0 golden）
```

资产：`.scratch_007/qwen35_08b_q4_007d.wtop*`、`output_q4d_{0..3}.f16.raw`、
`q4d_gate.log`、`run_q4d_gate.sh`、`judge_q4d.py`。
f16 对照线（v4 blob 同 4-prompt vs hf）：0.998642/0.998613/0.999425/0.998822
（p3 top1 26/32 —— hf 口径门本为 prompt0 锚，双口径不可直比）。
