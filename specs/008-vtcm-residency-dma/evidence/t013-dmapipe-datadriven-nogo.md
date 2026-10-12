# T013 双缓冲轮换 — 数据裁决 NO-GO (2026-10-10)

## 裁决

**不实施** DmaStreamDesc buf_slot 0/1 双缓冲轮换。板跑实测把该机制的可隐藏
时间上界定死在 **0.17% wall** — 投入产出不成比例, 时间应转向 US3 搬运算子
(STRIDED_SLICE 25.0% + UNARY 22.1% + TRANSPOSE_GEN 9.5% + BINARY 8.1%)。

## 数据链 (全部来自 T012 收口板跑, 见 t012-prepack-boardrun.txt + t014-dual-timeline-overlap.txt)

- 0.8B v4 图 (14188 ops), exec wall = 13.38s, Σdma_us = 22.53ms (1.4GB @ 66.7GB/s)。
  `Σdma_us / wall = 0.17%` — 即使双缓冲把全部 DMA 在途时间完美藏进计算,
  收益上界 = 22.5ms。
- per-opcode 重叠率列 (gehtp_prof.py, 008 T014): MATMUL_F16 dma/cpt = 0.6%,
  OP35 (lm_head SPLIT chunk) = 1.7%。权重搬运已不是 matmul 族的主要成本。
- dc_dma_once_ex 的计时 (t_dma0 → 累加) **覆盖整个阻塞调用** (desc init +
  poll-IDLE + submit + 完成 poll), 即 22.53ms 已含全部固定开销 — 探针时代
  "~700µs/调用固定开销" 的外推在真图上不成立 (325 次 matmul 平均 69µs/次)。
- MATMUL avg 7.47ms/op 的残留归因维持 US2 探针结论: per-op dispatch/carve
  记账 (US4 靶, T020-T022 批量化), 非 DMA。

## 机制现状与契约

- 现行 bit3 权重预取 = 阻塞 dc_dma_once (串行 by construction), 位级等价
  已被 GATE1 锚定 (md5 f87461e2)。
- FR-004 的「重叠失败 MUST 显式回落串行并记偏差」义务由 **T015 搬运面回落**
  兑现 (bit3 DMA 失败 → cpu_to_vtcm 串行 + dev=1 + stg_fb, 见
  t015-staging-fallback-boardrun.txt): 回落+上报契约成立, 只是回落方向是
  「DMA→CPU 拷贝」而非「流水→串行」。
- 若未来图形态变化 (权重驻留面扩大 / DMA 占比回升), 以本文件公式
  (Σdma_us / wall) 用 gehtp_prof.py 重估后再立项。

## 记录

- tasks.md T013 关闭为数据裁决 NO-GO (006 激活融合普查同款先例)。
