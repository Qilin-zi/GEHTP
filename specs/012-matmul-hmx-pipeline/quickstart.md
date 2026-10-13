# Quickstart: MatMul HMX 流水线验证

本指南列出证明特性端到端生效的可运行验证场景。接口契约见 [contracts/runtime-interfaces.md](contracts/runtime-interfaces.md)，实体与状态见 [data-model.md](data-model.md)。

## 前置

- 已编译内核并生成签名设备库（先 `source env.sh` 再 build，见 [[gehtp-build-env]]）。
- 设备 `d0f1784` 可达；已有 0.8B blob + golden 输出。
- 已有逐 op 时间 profile 工具 `scripts/gehtp_prof.py`。

## 场景 1 — R1 外提握手后位级一致（硬门）

**目的**: 证明把 acquire/release 提到 tile 循环外不改变数值。

1. 重编 `oplist_exec.c`（握手外提改动）→ 签名库 → 推设备。
2. 跑设备 golden 对拍（`check_device_golden`）。
3. **预期**: cos ≥ 0.9986、top-1 32/32、max|d| 不劣于串行基线；输出与改动前位级一致。

## 场景 2 — R1 收益（逐 op profile）

**目的**: 证明单个大 FC matmul 的 tile 握手串行被消除。

1. 用 `gehtp_prof.py` 对比改动前后 `op_ts.bin`。
2. **预期**: attn_qkv（M=32, K=1024, N=6144）逐 op 段从 ~102ms 显著收缩；matmul+FC 占 prefill 墙钟比例下降；prefill 总墙钟下降（目标 ≥20%）。

## 场景 3 — R3 异步 DMA 预取

**目的**: 证明下一个 matmul 权重搬运与当前计算重叠。

1. 启用 `dc_dma_submit`/`dc_dma_wait` 路径，跑含多 matmul 的 blob。
2. **预期**: profile 中权重搬运停顿从关键路径消失；`stg_fb=0`（无串行回退）；输出与基线位级一致。

## 场景 4 — 全量套件零回归

**目的**: 证明边界场景（单 tile / 单 matmul / 非 matmul）不回归。

1. 跑 `build_examples.sh`（基线 337 pass）。
2. 跑单 matmul、`n_col=1`、非 matmul 微 blob。
3. **预期**: 全绿；单 tile/单 matmul 路径行为与改动前一致（无额外延迟、无回退报错）。

## 场景 5 — 会话复位

**目的**: 证明流水/预取状态在 blob 边界与跨执行会话干净复位。

1. 连续跑多次（或 GEHTP_PERSISTENT 多次 step）。
2. **预期**: 无残留的「读中」暂存区被覆盖、无 stale 完成态导致 use-before-ready；每次会话首 op 正常发起。


## 场景 6 — 012 实证收口 (2026-10-13)

1. `wtop_emit ... --dma-runlist` 重 emit → emit log 须见 `[pp] T012 权重预打包 182 槽`。
2. 板跑 pp blob vs 无 pp ctrl blob: 输出 md5 位级同 (`9d1c0877…`), cos 0.999930/top1 32/32。
3. **预期**: wall 13575→2756ms (−79.7%, SC-001 达成); MATMUL_F16 avg 44.3→5.7ms, OP35 136→13.4ms。
4. 回归: examples 321 pass/9 FAIL(全预存), judge 双门 PASS。资产: `.scratch_012/qwen35_08b_fused_{ctrl,pp}.wtop`。
