# 021: decode 批处理 (M>1, 破 10GB/s 带宽墙)

2026-11-04 立项 stub (与 020 一起)。**依赖 020** ([[gehtp-020-static-dma-schedule]])。

## north-star 关系

020 交付「编译期固定的连续 DMA 流」= 权重搬运与计算解耦的载体。021 在此之上把
M 从 1→N, 让**一次权重流服务 N 条序列**, 算术强度 ×N, memory-bound 翻 compute-bound。
这是打破 decode「10GB/s 带宽天花板」的唯一结构性杠杆 ([[gehtp-decode-batching-feasibility]])。

## 问题

decode M=1 每个 matmul 是 GEMV: 读整个 W 只算 1 行, 算术强度 ≈1 FLOP/byte,
深 memory-bound (0.6B f16 118ms/步 = 读 ~1.2GB ≈ 10GB/s 已到 DRAM 顶)。
批 N 序列 = M 1→N, 同 matmul 权重读不变、MACs ×N → 强度 ≈32× (N=32)。

| 形态 | 0.6B | 0.8B |
|---|---|---|
| decode M=1 f16 (旧锚) | 8.46 tok/s | 3.90 tok/s |
| decode M=1 Q4 (T008 实测) | 21.26 tok/s | — |
| decode Q4 + 2 NSP (T008 实测) | 42.48 tok/s | — |
| prefill M=32 (=批处理 decode 上界) | 17.7 tok/s (未优化) | 65.3 tok/s |
| llama.cpp 对照 (批处理口径) | 45.2 tok/s | 45.2 tok/s |

## 三件改造 (都非内核)

1. **N 序列 KV cache**: 28 层 × 2(K+V) × 1024 × 64 × 2B ≈ 7.3MB/序列, 32 序列 = 235MB DDR。
2. **连续批调度器**: 聚合 N 条请求 → 一次 M=N forward (现 `gehtp_server.py` 每 token 单次 M=1)。
3. **批-M decode 图**: 合并 prefill M=32 形状 + decode 持久 KV cache, 复用 020 footer 流。

## 目标 (诚实)

- 吞吐: 0.6B decode 8.46 → 65-85 tok/s (对标/超 llama.cpp 45.2); 2 NSP 双序列再 ×2。
- **单序列延迟不降反升** (请求等 batch 凑齐): 换吞吐不换快, 与 llama.cpp 同口径。
- 单序列提速须投机解码 (草稿模型), 另线。

## 门

- 吞吐 ≥ 45.2 tok/s (超 llama.cpp)。
- 精度: 批处理不改变每序列独立输出的数值 (逐序列位级同单序列, cos ≥ 0.9986)。
- 2 NSP: 双域各跑一路 batch, 合计吞吐 ~2×。

## 非目标

- 单序列延迟优化 / 投机解码 (另线)。
- 0.6B prefill 优化移植 (配套但独立, 光批不移植只有 ~2×)。

## 待 020 收口后细化

020 T009 收口时把本 stub 扩成完整 speckit 六件。

## 020 收口衔接 (2026-11-04)

020 T008 上板 PASS (证据 `../020-decode-static-dma-schedule/evidence/t008-2nsp-onboard.md`):
- footer 连续 DMA 流已编译期固定 + 双域兼容 (单份天然, 零编译期改动)。
- 实测吞吐锚: Q4 单序列 21.26 tok/s → 2 NSP 双序列 42.48 tok/s (device 1.998×,
  每序列 step_us 无退化 = **无 DRAM 争用**, 推翻「DRAM 共享瓶颈」)。
- 021 批处理在 42.48 tok/s 之上再乘 M>1 算术强度杠杆: 同 matmul 权重读不变、MACs ×N,
  memory-bound 翻 compute-bound (目标破 10GB/s 墙 → 超 llama.cpp 45.2)。
- **下一步**: 本 stub 扩成完整 speckit 六件 (spec 细化 + plan + tasks), 三件改造
  (N 序列 KV cache / 连续批调度器 / 批-M decode 图) 均非内核, 复用 020 footer 流。
