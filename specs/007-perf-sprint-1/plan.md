# Implementation Plan: Short-Term Performance Sprint 1（broadcast HVX / W4A16 全图 / 融合准入）

**Created**: 2026-10-08（原 006 冲刺线 10:39 版复原改号 007；006 目录现为激活融合线容器）
**Branch**: `007-perf-sprint-1`（实施前照 004 先例进独立 worktree——共享工作树建/切分支波及全会话）
**Feature**: [spec.md](spec.md)

## What This Plan Delivers

三项性能兑现，目标 0.8B prefill **35.31s→≤20s**：

1. **运行时补 HVX broadcast 执行体**并让 emit 计划与内核能力一致（消灭 1023 次允许回落，**-7.3s**）。
2. **W4A16 权重域推广全部线性层**（`--gguf` Q4_0 直读通道从未在 0.8B 上启用，权重 **1.6GB→~0.5GB** + HMX W4 吞吐）。
3. **融合 pass 准入制放行**（8 条 legacy 规则先统计命中再逐条数值门，op 数 **14188→≤9000**）。

全程 003 契约（偏差报告不变量 / carve / legacy 逐字节）不破。

## Technical Context

- **构建**：C++17 容器 `gehtp-toolchain:24.04`（共享 compiler/build_linux，ctest 只认本线目标）；C 设备库 `kernels/build_libs.sh`（改多文件用 all 勿用 one）。
- **设备**：d0f1784（getenforce Permissive 预检、pgrep run_main 互斥、003 私有名纪律）。
- **broadcast 现状**：`exec_broadcast`（kernels/src/runtime/oplist_exec.c:2080）标量逐元素除法/取模重算索引（均 4.8ms/op）；已接 v3 回落裁决；emit 侧 `Eltwise_Broadcast` 按 n%32 计划 HVX。
- **W4 现状**：emit `--gguf/--match`（compiler/tools/wtop_emit.cpp:263-282）读 TSV 匹配表+GGUF，Q4_0 tile-major repack；match 表由 `scripts/gguf_wtlink.py` 生成（`--gguf X.gguf --out match_map.json`）；US1 已设备验证 W4A16 HMX（blob_w4 逐位对拍）；lm_head 走此路（US1 51.7×）；v4 全图 f16 是当年 GGUF 源缺失残留。
- **融合现状**：`kLegacyFusionRules` 8 条经 `register_one_fusion_rule` 注册 `fusion_<p>_<c>` 默认 `enabled=false`；0.8B GDN 图预期命中集中在 MatMul+Add；GDN 链不在规则集内——命中不足 ≤9000 目标按热点补规则。
- **精度判据**：`test_models/qwen35_08b/cpu_golden/{hf,q4_0,text}` 双 golden 现成；P2 后判分切 q4_0 口径（judge_logits.py）。
- **基线锚**：v4 = wall 35.31s / cos 0.998642（f16 锚）/ top1 30/32 / 14188 op / 回落 1023。
- **Constitution**：空模板，以 FR-006 与库纪律替代（opcode 只追加/subtype 不动 ABI、设备互斥、判据入库不入 /tmp）；本计划不新增 opcode、不改 slot 布局 ✅。
- **Source files**：
  - `kernels/src/runtime/oplist_exec.c`（exec_broadcast HVX 路径 P1 核心）
  - `kernels/src/runtime/hvx_binary.c` 等既有 HVX 件（P1 参照范式：HVX 循环+尾部标量）
  - `compiler/tools/wtop_ops/wtop_ops.hpp` / `op_matmul.cpp`（P1 计划对齐 + P2 W4 命中面）
  - `compiler/tools/wtop_emit.cpp`（P2 供给 + emit 报告扩展）
  - `compiler/src/opt/pass_fusion.cpp` / `pass_manager.cpp`（P3 准入）
  - `scripts/gguf_wtlink.py`
  - `compiler/tests/test_wtop_emit.cpp`（P1 契约测试）
  - `scripts/gehtp_prof.py`（回落计数/族耗时）
- **Complexity**：P1 中；P2 高（wtlink→emit→板跑判分三段）；P3 中低。

## Project Structure

`specs/007-perf-sprint-1/`（本目录，七件套）

## Reference P2 Research (already done, see [research.md](research.md))

D1 broadcast 三形态分派；D2 W4 供给链与口径切换；D3 融合准入制。详见 research.md。

## Implementation Phases

### Phase 1: broadcast HVX 执行体 + emit 计划对齐 (P1)

- T101 设备端 broadcast 三形态执行体（`oplist_exec.c` exec_broadcast 段 + `hvx_binary.c` 参照范式）
- T102 emit `assign_engine` 按形态分派计划（类 1/2→HVX、类 3→scalar，两侧同判据注释互指）
- T103 契约测试（test_wtop_emit.cpp）+ 板跑：回落 1023→0、族耗时 ≤1s、cos 位级不动
- 验收：SC-001/SC-002（P1 单独 ≤28s）/SC-007

### Phase 2: W4A16 全图启用 (P2)

- T201 `gguf_wtlink.py` 全量 match 表 + 编译期 GGUF 源存在性预检
- T202 emit 权重域报告 + 显式 f16 兜底名单（形状约束不满足的 FC）
- T203 板跑 + q4_0 golden 判分重锚（双列登记 f16/q4_0 口径）
- 验收：SC-003/SC-004；lm_head 51.7× 不回退

### Phase 3: 融合 pass 准入制放行 (P3)

- T301 逐条 `--enable-pass fusion_*` 跑 opt-stats 统计命中
- T302 数值门：合成图位级 + 0.8B 全图 cos Δ<1e-5 + top1 不降 → 过门进默认集
- T303（条件项）命中不足 ≤9000 时按 op_ts 热点评估补规则（GDN 门控链候选）
- T304 P1 验证复跑（P1↔P3 交互：融合吸收 broadcast 物化后确认回落仍为 0）
- 验收：SC-005/SC-006；三方向叠加 wall ≤20s（SC-002 总门）

## Implementation Strategy

MVP = Phase 1（收益最大、风险最低、无精度重锚）。P2 在 P1 落地后独立推进（判分口径切换只在 P2 期）。P3 与 P1/P2 正交，放行顺序在后（P1 消回落后热点统计才干净），末步 P1 复跑收口。所有阶段共享 003 契约回归兜底（五场景 + ctest 47/47）。
