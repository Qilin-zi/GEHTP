# Runtime 接口契约：异步 DMA 与 HMX 会话

本特性引入两个内部运行时接口契约：异步 DMA（支撑 op 级权重预取）与 HMX 会话（支撑 tile 级流水）。这些是 `oplist_exec.c` 执行器与 `dc_parts.c` / `hmx_common.c` 底层之间的所有权与 fence 约定。

## 契约 A — 异步 DMA

新增 `dc_dma_submit` / `dc_dma_wait`，拆分当前阻塞的 `dc_dma_once`。

| 原语 | 语义 | 阻塞 |
|---|---|---|
| `dc_dma_init` | 分配 descriptor，设置 src/dst/bytes/bypass | 否 |
| `dc_dma_submit` | `dmstart` 触发搬运，立即返回 | 否 |
| `dc_dma_wait` | poll `dma_desc_is_done` 直到 COMPLETE | 是 |
| `dc_dma_destroy` | 释放 descriptor | 否 |

**fence 铁律**（在 submit/wait 之间保持，不得省）：
1. submit 前：源 DDR 区 FC_CPU→FC_DMA clean（`dc_dma_clean_src`）。
2. wait 后：目标 VTCM 区 DMA→HMX invalidate（`fence_handoff` FC_DMA→FC_HMX）。

**所有权**：
- 提交后、wait 完成前，目标暂存区处于「写中」，任何读方（HMX/HVX/标量）不得访问。
- wait 完成前不得对同一 descriptor 再次 submit 或 destroy。
- 异步期间源 DDR 区不得被 CPU 改写。

**预算约束**（FR-007）：双暂存区总占用不得超过可用片内预算；超界时回落串行搬运。

## 契约 B — HMX 会话

一次 acquire/release 覆盖整段 tile 走查。

| 阶段 | 原语 | 次数 |
|---|---|---|
| 获取 | `hmx_enable_execution` + `hmx_unit_acquire` | 1 次/走查 |
| 逐 tile | `mxclracc.hf` → `bias=mxmem2` → N×`mxmem:deep` → `cvt.hf=acc` → `mxmem=cvt` | 每 tile 一次 |
| 释放 | `hmx_unit_release` + `hmx_disable_execution` | 1 次/走查 |

**不变式**：
- 每个 tile 累加开始前必须 `mxclracc.hf`（累加器边界复位，FR-003）。
- 会话内不允许嵌套 acquire；异常路径必须 release（FR-008）。
- 单线程 runtime 无 HMX 竞争，但长持 HAP lock 的语义（跨 tile 保持资源/电源使能）需设备验证无副作用。

## 契约 C — 位级一致性

流水化/异步化的输出 MUST 与串行逐 tile、逐 op 基线位级一致（f16 路径位级；Q4_0 反量化路径在文档化舍入容差内）。这是所有优化落地的前置门（FR-004）。
