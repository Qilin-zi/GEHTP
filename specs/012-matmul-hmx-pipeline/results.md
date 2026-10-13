# Results: 012 MatMul HMX 流水线

日期: 2026-10-13 | 设备 d0f1784 | 库 libhvxhmx_012 (hoist) + wtop_emit 011 HEAD (734986b)

## 终态总表 (SC-001 达成: −79.7%, 目标 −20%)

| 指标 | 009 老 blob 基线 | ctrl (011 重 emit, 无 pp) | pp (011 重 emit + --dma-runlist) | pp vs ctrl | pp vs 老基线 |
|---|---|---|---|---|---|
| wall (op_ts) | 14120.0ms | 13575.5ms | **2755.5ms** | **−79.7%** | −80.5% |
| 输出 md5 | 02c69d93… | 9d1c0877… | 9d1c0877… | **位级一致** | 数值升级* |
| cos / top1 | 0.998639 / 29 | 0.999930 / 32/32 | 0.999930 / 32/32 | — | — |
| MATMUL_F16 avg | 44334µs | ~43700µs | **5706µs** | 7.7× | 7.8× |
| OP35 avg | 136330µs | ~136000µs | **13423µs** | 10.1× | 10.2× |
| Σdma_us | 0 | 0 | 22590 (1.4GB, 66.5GB/s) | — | — |

\* 老基线 blob 是 RoPE 修复前 emit (cos 0.9986 旧锚); 011 工具 fold=92 后 cos=0.999930 = [[gehtp-010-merge-mainline]] 正典精度。位级门在 pp↔ctrl 之间成立 (同工具同图, 唯一差 = [pp]182 槽 + flags bit3)。

## T005-T008 实验台账 (假设逐一裁决)

| 实验 | 改动 | wall | 输出 md5 | 结论 |
|---|---|---|---|---|
| baseline1/2 | 无 | 14121.5/14120.0 | 02c69d93×2 | 基线可复现 (0.01%) |
| lift1 | HMX 会话外提 (acquire/release 192→1/op) | 14116 | 02c69d93 (位级同) | **零增益** — 逐 tile 握手不是成本 |
| lift2 | ggml 链式装载 (range=2048·n_dot−1) | 25986 (+84%) | 3dc2202a | **否定** — 4 op n_dot>32 range 溢出回落标量 12.1s; 其余 177 op 44.0ms vs 44.4ms = 链式本身中性 |
| res1 | act 驻留 | (stale lib, 数据无效) | 02c69d93 | 编译从未成功 (见下), 板跑用的是旧库 |
| ctrl | 011 工具重 emit (fold=92) + hoist | 13575.5 | 9d1c0877 | 新锚 |
| **pp** | **ctrl + --dma-runlist ([pp]182)** | **2755.5** | **9d1c0877 (位级同)** | **SC-001 达成** |

## 真根因: 标量 crouton 权重 pack

- 归因证据: 每 (r,c) tile 固定 ~531µs, **与 n_dot(K) 无关** (K=64 与 K=1024 同价), 逐 op 零截距线性 (N=6144:102.4ms, N=4096:68.4ms, 斜率 531µs/列)。若为带宽/MAC 应随 K 缩放。
- 531µs/tile × 32K 元素/tile ≈ 16ns/元素 = CPU 逐元素 gather DDR 权重 + scatter 写 VTCM crouton 的标量 pack 成本。
- 老无 pp: Σdma_us=0 (prof) + 440 槽无 [pp] (emit log) 实锤 blob 侧漏旗标。
- 008 T012 已建好的 w_pp 路径 (emit 镜像预打包 + 运行期线性 DMA) 生效后 pack 成本消失, matmul 归位到走查+DMA (~5.7ms/op)。

## act 驻留: ISA 判死 (FR-004 相关裁决记录)

- Hexagon 汇编器硬规则: `packet has HMX weight without an activation` — act/wgt 装载必须**同指令包成对**。
- clang 侧: HMXActivateWeightMutation (HexagonSubtarget.cpp:382) 负责配对, wgt-only 循环触发 `Unmatched weight instruction` assert (exit 134); `#pragma clang loop pipeline/disable` 与 noinline 均无法绕开。
- 结论: "act K 链驻留跨列复用" 在 v81 HMX ISA 不存在; MX 流入 = n_row×n_col×(act+wgt) 全链为下界, 唯一合法优化 = 同包链式 (实测中性)。已在 oplist_exec.c 落注记。

## 残余热点 (pp 后新图景, 下回合输入)

| 项 | 耗时 | 占 wall | 备注 |
|---|---|---|---|
| MATMUL_F16 181 op | 1033ms | 37.5% | 已归位 (走查+DMA+unpack) |
| OP37 gdn_block 18 op | 647ms | 23.5% | avg 35.9ms, 内部 conv matmul 未 pp 化 → 下一个 pack 靶子 |
| RMSNORM2_F16 79 | 250ms | 9.1% | |
| OP36 18 | 203ms | 7.4% | 同 gdn 族 |
| DMA | 22.6ms | 0.82% | US3 异步化收益上界 |

## 产物清单

- 库: `.scratch_012/libhvxhmx_012.signed.so` (md5 同 kernels/lib/libhvxhmx_v23.signed.so, hoist-only diff)
- blob: `.scratch_012/qwen35_08b_fused_{ctrl,pp}.wtop` (+ manifest)
- 跑板: `.scratch_012/run_012_board.sh <tag> [blob]`
- op_ts/output: `.scratch_012/{ctrl,pp}.op_ts.bin`, `output_012_{ctrl,pp}.f16.raw`
