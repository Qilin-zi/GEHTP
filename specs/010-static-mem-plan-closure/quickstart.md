# Quickstart: 静态内存规划收口 — 验证指南

验证五收口点各自独立可测，最后 0.8B 端到端收口。数据模型与字节契约见 [data-model.md](data-model.md) / [contracts/blob-v3-extensions.md](contracts/blob-v3-extensions.md)。

## 前置

- 编译产物基线：当前库锚 cos 0.998642 / 6.31s（跨会话协调表口径）。
- 设备环境：`llmserv` conda env + 设备上板（见 GEHTP serving 环境）；host 编译 `source env.sh`（kernels 先 source）。
- 回归基线：`build_examples.sh` 35 例 + 261 门 + W4A16 256³ 全绿。

## 验证 D — 权重 restage 静态化

1. 造「同权重连续多个 matmul」图，编译期下发 pin+复用序列。
2. 上板执行：已常驻权重零重复搬运；`wt_w3_report` restage 次数 == 编译期规划值。
3. 超预算图：换入换出严格按序列，无超规划 restage。

**验收**: restage 次数对账（spec Story 1 + SC-003）。

## 验证 A — W4A16 辅助面显式标注（K≠N 修复）

1. 造一个 K≠N 的 W4A16 合成图，编译出 blob（新 arity=9）。
2. 执行器上板跑：预期不再报 `supply slots missing`，日志零按尺寸猜。
3. 回归 K=N 的 W4A16 256³：预期 65536/65536 bit-exact，输出与旧行为逐字节一致。

**验收**: 场景 1 不误报 + 场景 2 bit-exact（spec Story 2）。

## 验证 B — pin_cap 编译期下发

1. 用两个不同权重常驻需求的模型编译，`vtcm_auto_scan` 各算出 pin_cap（`[memplan]` 日志可见）。
2. 检查 blob TEMPOFF 槽头 pin_cap 字段 == 编译期算值。
3. 上板执行：`wtcache_open` 日志显示 pin_cap 读自 blob，非 4096；关池口径（pin_cap=0）回退 4096 且 byte-exact。

**验收**: blob 携带 pin_cap + 执行器读它（spec Story 4）。

## 验证 E — 权重 crouton pack 编译期化（conv2d + Q4_0 剩余）

1. 造含 f16 conv2d 的图，编译期产出 crouton 布局权重槽（f16 matmul 已由 008 T012 收口，不在本验证）。
2. 上板执行：`exec_conv2d_hmx` 不再做运行时 crouton 重排，输出与旧行为逐字节一致。
3. Q4_0 图：确认权重槽仍 640B/tile（量化态重排，未反量化成 f16）；LUT dequant 输出与旧路径逐位一致。
4. 重测运行时权重 pack 耗时，确认按基线归零（对照 T017 LUT 后的剩余量，非旧账 20.6s）。

**验收**: 零运行时 crouton 重排 + Q4_0 内存守恒 + byte-exact（spec Story 3 + SC-004）。

## 验证 C — carve 复用编译期预判（P2，可延后）

1. 造连续同形 matmul 序列，编译后 op_annot bit5 在「可复用」op 上置位、形状切换 op 清零。
2. 上板执行：可复用 op 跳过 re-carve，日志零 `shape_same` 运行时判断。
3. 对比旧 shape_same 路径：carve 复用结果一致。

**验收**: 复用决策来自编译期标注（spec Story 5）。

## 验证 E2E — qwen3-0.8b 不回退

1. 0.8B 全链编译 + 上板端到端推理。
2. 双门：cos ≥ 0.998642、wall ≤ 6.31s。
3. 全量回归：`build_examples.sh` 35 例 + 261 门 + W4A16 256³ 全绿。

**验收**: SC-005 + SC-002。

## 一键回归

```bash
# host 编译后全量回归（byte-exact 门）
bash build_examples.sh
# 设备 golden 对拍
python scripts/check_device_golden.py
# host 数值对拍
bash scripts/run_host_ci.sh
```

> 具体命令以 tasks.md 阶段最终落地为准；本文件只给验证骨架，不含实现体。
