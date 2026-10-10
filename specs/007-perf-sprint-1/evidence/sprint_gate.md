# T024 — 冲刺总门 (SC-002: 三方向叠加 wall ≤20s) (2026-10-10, 设备 d0f1784)

## 结论: 门未达 — wall 27.4s (>20s), op 数 14188 (>9000), 根因 US3 融合 NO-GO

| 指标 | 003 v4 基线 | 当前 (US1+US2 Q4+LUT) | 冲刺门 | 判 |
|---|---|---|---|---|
| 全链 wall | 35.31s | **27.4s** (exec 25.7s) | ≤20s | ✗ FAIL |
| op 数 | 14188 | **14188** | ≤9000 (参照) | ✗ FAIL |
| 允许回落 | 1023 | **0** (code26 n=1622 77ms) | 0 (SC-001) | ✓ |

单 prompt (prefill 32 token) 板跑, q4d blob (476.5MB) + LUT lib (7aed109),
设备日志 `[run] seg_us read=1058138 parse=18445 wgt=24776 exec=25720851
write=27840 wall=27426032`。

## 三方向贡献分解

| 方向 | 计划 | 实测 | 判 |
|---|---|---|---|
| ① broadcast HVX (US1) | 7.84s→≤1s | code26 7.84s→0.077s (-7.7s), 回落 1023→0 | ✓ 全交付 |
| ② W4A16 (US2) | 权重 1.6GB→0.5GB + W4 HMX 吞吐 | 权重 476.5MB ✓; 但 Q4 dequant 标量化, MATMUL_F16 10.6s + lm_head 4.6s = 15.2s matmul 族 (vs f16 直接 HMX) → **wall 净 +3.2s** | ✗ 内存交付, 速度未达 |
| ③ 融合 (US3) | op 14188→≤9000 | 8 条 fusion_* 匹配 0 (NO-GO, 见 fusion_admission.md) → op 数不变 14188 | ✗ NO-GO |

## 热点分布 (当前 wall 25.7s exec 内)

| opcode | n | total_us | 占比 |
|---|---|---|---|
| 24 MATMUL_F16 (Q4 dequant+GEMM) | 325 | 10.62s | 41.3% |
| 35 MATMUL_SPLIT (lm_head) | 32 | 4.61s | 17.9% |
| 15 (GDN/SSM 递推态) | 2373 | 3.35s | 13.0% |
| 11 UNARY | 2909 | 2.93s | 11.4% |
| 12 SOFTMAX | 1655 | 1.08s | 4.2% |
| 27 (attention/conv) | 175 | 1.27s | 5.0% |

matmul 族 (code24+35) = 15.2s = 59% wall, 其中 Q4 dequant+pack 是标量
(LUT 优化后 2× 于标量, 但仍单线程 DSP 核, 非 HVX 向量化)。

## 为何 ≤20s 不可达 (根因)

1. **US3 融合 NO-GO**: 8 条 legacy 规则短名 vs IR 长名 (DepthWiseConv2d/
   Eltwise_Binary/ElementWiseNeuron/FullyConnected) 全 0 命中; 且 006 普查
   激活融合上界 1.46% < 3% 门 — 即使修名, op 数 14188→≤9000 靠融合亦不可达。
2. **US2 Q4 速度未交付**: W4A16 的价值在内存 (1.6GB→0.48GB), 但 dequant
   标量化使 matmul 族 15.2s vs f16 直接 HMX 更慢; LUT 已把标量 dequant
   20.6s→10.6s (2×), 但要追上 f16 线需 HVX 向量化 dequant (此前 probe 证伪,
   风险未清)。

≤20s 门以三方向全部交付为前提; ③ 不可交付 → 门按构造不可达。诚实结论:
**冲刺总门未达, US1 超交付 (-7.7s), US2 内存交付速度净负, US3 NO-GO**。

## 复现

```
.scratch_007/run_q4d_lut.sh     # prompt0 单跑, wall 27.4s / op 14188 / 回落 0
grep seg_us .scratch_007/q4d_lut_0.device.txt
```

资产: `.scratch_007/q4d_lut_0.device.txt` (op_timing 全表)、
`q4d_lut_0.op_ts.bin` (WTS1, 未刷新需时间戳核对)。
