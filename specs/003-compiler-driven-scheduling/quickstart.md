# Quickstart: Compiler-Driven Scheduling 验证指南

端到端验证场景（证明三件套可用），不包含实现代码。实现细节见 `tasks.md`（后续 `/speckit-tasks` 生成）。

## 前置

- 构建环境：`kernels` 先 `source env.sh`（见 memory `gehtp-build-env`）。
- 判分环境：`llmserv` conda env 跑 OpenAI 服务 + pytest（memory `gehtp-serving-env`）。
- 设备：d0f1784（活板预检、`run_main_on_hexagon` 互斥、getenforce=Permissive）。

## 场景 1 — 编译器指派引擎，运行期照做（P1）

1. 重编发射器 + 执行器：`compiler/build_linux/wtop_emit` 与 `kernels/build_libs.sh one oplist_exec`。
2. 发射 0.8B v17 blob 的 v3 版本（带引擎注解表）。
3. 校验 blob：解析 `ver==3`，逐 op 读注解表 `engine` 字段非空且 ∈ {0,1,2}。
4. `gehtp run` 上板，产出执行报告。

**期望**：报告里 `deviation==0` 对全部 op 成立（`planned_engine == actual_engine`），对应 SC-001。

## 场景 2 — 显式回落不静默（P3，与 P1 同落）

1. 人为构造一个计划单元不可用的情形（如强制某 op 计划 HMX 但运行期 HMX 被占）。
2. 一个标 `fallback_allowed=1` 的 op：应回落且报告 `deviation==1`、附 `fallback_engine`。
3. 一个标 `fallback_allowed=0` 的 op：应硬报错中止（`deviation==2`），而非静默换单元。

**期望**：无任何"静默换单元"，对应 SC-002。

## 场景 3 — 静态 VTCM carve + 显式搬移（P2）

1. 发射含 VTCM carve 表 + OP_DMA 搬移 op 的 blob。
2. 校验 carve 表不变量：生命周期重叠的记录 `[offset, offset+size)` 不相交。
3. 上板跑，核对搬移按表发生、无 VTCM 争用/越界。

**期望**：端到端 wall 不退化（基线 33.9s），片上内存不再"谁先来抢全量"。

## 场景 4 — 数值 + 性能门（总门）

- 判分：`judge_logits.py`，cos ≥ 0.998646（不回归）、top1 ≥ 29/32、nan=inf=0。
- 耗时：wall ≤ 33.9s（不退化），且此前标量回退的 matmul 类（如 batched 注意力）保持走加速路径。

**期望**：对应 SC-003/SC-004。

## 场景 5 — 向后兼容（FR-009）

1. 用旧 v1 blob（如当前 v17）在新运行时上跑一遍。

**期望**：输出逐字节同旧运行时（新运行时见 `ver==1` 走 legacy 路径），对应 SC-005。

## 自动化回归

- `ctest --output-on-failure`（compiler/tests，基线 44/44）新增：
  - v2 注解表解析契约（正例 + `fallback_allowed=0` 硬报错负例）。
  - carve 表不变量（重叠 → 拒绝）。
  - 执行报告不变量（`deviation` 与 `planned/actual` 的一致性）。
