# T026 — quickstart.md 全流程 + 三条逃生门验证（2026-10-10）

## 结论

**PASS**：quickstart 五段（基线 / P1 / P2 / P3 / 契约兜底）均已在本 sprint 各门中实际跑通
（T010-T017 设备板跑 + T024 总门 + T025 ctest 48/48）；三条逃生门逐条验证可退出。

## 全流程覆盖映射

| quickstart 段 | 本 sprint 执行任务 | 结果 |
|---|---|---|
| 0 基线对照 | T003 baseline.md（v4 锚 35.31s/cos 0.998642/broadcast 回落 918） | 已采 |
| 1 P1 broadcast HVX | T010-T011 板跑（回落 918→0、族 ≤1s、wall 24.2s、cos 位级不动） | PASS |
| 2 P2 W4A16 | T012-T017（match 表 + w4report + q4_0 双门 + LUT） | PASS |
| 3 P3 融合准入 | T018-T023（8/8 匹配 0 → NO-GO） | 记证 |
| 4 契约兜底 | T025（ctest 48/48 + diff 范围审计） | PASS |

## 三条逃生门（逐条验证可退出）

### 门 1 — P1「emit 回旧判据 n%32」（代码回退，无 CLI 开关依赖）

broadcast 双侧改动全部收敛在单提交 `ab73491`（`feat(007 US1): broadcast 三形态
HVX 分派`）：emit 侧 `op_eltwise_binary/ternary.cpp` + `wtop_ops.hpp` broadcast_engine_annot、
设备侧 `hvx_binary.c` + `oplist_exec.c` exec_broadcast。`git revert ab73491` 即干净退回
n%32 计划 + 旧标量执行体，不触碰 US2/US3 或 003 契约面（`git show --stat ab73491`
= 9 文件自包含，无跨提交依赖）。

### 门 2 — P2「去 --gguf/--match 回 v4 全 f16」

- `wtop_emit` 不带 `--gguf/--match` → 产出 v4 全 f16 blob
  `.scratch_007/qwen35_08b_v4_007.wtop`（2041268992 B，2026-10-08 23:19），
  其 emit 日志 `emit_007.log` **0 条** `[gguf]/[w4report]` 标记（对照 q4 配方
  `emit_q4_007.log` 3 条标记）。
- 该 v4 blob 已上板跑通：`v4_007.device.txt` 12 条 runner M-里程碑 + `v4_007.op_ts.bin`。
- 反向验证：`--gguf` 与 `--match` 已加 FR-003 成对硬门（单给其一即报错退出，防静默全 f16）。

### 门 3 — P3「--no-pass fusion_*」

- `hnnx_compile.cpp:314` 与 `host_run.cpp:254` 均接线 `--no-pass <name>`（可重复），
  help 文案明示「关闭单个图优化 pass」。
- 端到端实测：`build_host/hnnx_compile --net-json test_models/qnn_probe/model_float_net.json
  --no-pass fusion_matmul_add --opt-stats` → 编译成功（107 op → 144 nodes / 19 steps / 11 kernels），
  `--opt-stats` 显示 `fusion_matmul_add=0/0`（0 命中，印证 US3 NO-GO）。
- 融合规则本就默认关（`enabled=false`），`--no-pass` 是纯逃生冗余；`--enable-pass fusion_*`
  亦可实验性开启（T018 已跑）。

## 判据

三条逃生门均可独立退出，无「关不掉」或「退出即破坏契约」的死锁。quickstart.md 命令
与产物路径与实测一致（`gguf_wtlink.py` / `wtop_emit` / `hnnx_compile` / `judge_logits.py`
均在 `scripts/`、`compiler/tools/`、`compiler/build_*`）。
