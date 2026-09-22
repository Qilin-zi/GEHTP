# Implementation Plan: GEHTP 调度设计原则落地（编译器+执行器）

**Branch**: `001-sched-design-landing` | **Date**: 2026-09-17 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-sched-design-landing/spec.md`

## Summary

把 `docs/GEHTP_SCHED_HWFACTS.md` 已定稿的调度/测量原则落成五个独立可验收的工程件：
(1) 编译器 VTCM 驻留决策前加**访问类型分类谓词**（消费方驱动，标量/data-dependent 访问
的张量禁入 VTCM）；(2) 常量去重门禁收紧为**只读常量专属** + 回归测试（防假依赖串行化）；
(3) HMX/HVX 单元指派的**规则表 + 可审计记录**（行为保持：初版不改默认派发，只记录与守门）；
(4) manifest 可选字段输出 **dominant-path 静态下界估计**（估计表驱动，blob 零改动）；
(5) 性能报告**三分类口径门禁脚本**（真算/装料/卸料分列 + 口径/形状/场景标注）。
五条用户故事对应五个独立阶段，各自独立 commit、独立验收、可独立回退。

## Technical Context

**Language/Version**: 编译器 C++17/20（CMake+Ninja，host Linux x86_64）；设备 kernel C11（Hexagon SDK 6.6.0.0，`-mv81 -mhvx -mhmx`）；脚本 bash/python3

**Primary Dependencies**: QAIRT SDK（`/4090disk2/QCtools/qairt_2.48.40.260702`）；Hexagon SDK 6.6.0.0（`/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.6.0.0`）；SWIV 签名（`/4090disk2/QCtools/swiv_build_utility.py`）；既有 `libhvxhmx_v23` 库与 WTOP 引擎（本特性不改设备侧）

**Storage**: 文件——tagged.bin（tagged-record runlist）、`.wtop` blob、`.wtop.manifest.json`；本特性唯一新增存储 = manifest 可选字段（blob 格式零改动）

**Testing**: ctest 44/44（compiler/tests）；host 对拍门（conv_add ×3 byte-exact、spill 变体、L3 81-op |d|≤0.001）；设备 quickstart（`scripts/gehtp_quickstart.sh`，设备恢复后跑）

**Target Platform**: host 编译 = Linux x86_64；执行 = V81 CDSP（52f67807，当前因虚拟化 SSR 通道故障不可用——本特性全部门禁设计为 host 先行，设备复核挂起不阻塞）

**Project Type**: 图编译器 + 设备执行引擎工具链

**Performance Goals**: 本特性是 M7 性能收敛的前置谓词与测量标尺，自身不改性能数字；防御目标 = VTCM 误驻留 0 起、窄输出误派 HMX 0 起、假依赖合并 0 起

**Constraints**: 行为保持（存量 blob 逐字节不变）；blob 格式兼容（新信息只走 manifest 可选字段/新文件，不动 tagged.bin tag 表）；每阶段独立 commit 可回退；多会话错峰（VL 线在途 `graph_prepare.cpp`/`ops.cpp`/`host_run.cpp`，动工前查 mtime + 会话排序）；硬件常量禁止硬编码他机型数字（43_hwinfo 回填前以符号/配置占位）

**Scale/Scope**: 编译器 20 型 opcode 家谱、16 个 wtop_ops handler；本特性触及 ≤6 个编译器文件 + 1 个新脚本 + 测试

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

本仓 `.specify/memory/constitution.md` 为未填充模板（无显式条款）。以仓内既定工程惯例为
事实门禁（来源：GEHTP_PORTAL §4/§6、GEHTP_MEMPLAN_COMPILER_MIGRATION §0/§5）：

| 事实门禁 | 本特性符合性 |
|---|---|
| 行为保持：存量门零回归、blob 逐字节不变 | ✅ 五个阶段均以"默认关闭/只读新增"落地；验收门即存量门 |
| 设备侧零改动优先（WTOP blob/runner 不动） | ✅ 全部改动在 host 编译器与脚本；manifest 新增为可选字段 |
| 每阶段独立 commit、可回退 | ✅ 五故事 = 五阶段，依赖序 P1→P2→P3→P4→P5 但各自可独立合入 |
| 多会话错峰 | ✅ 计划中显式标注避让文件与动工前检查 |

**结论：PASS（无违规需 justify）**。

Phase 1 后复核：data-model 全部实体为 host 侧记录（审计日志/manifest 可选字段），
contracts 三个接口均不改 blob 与设备消费方，quickstart 验收以存量门为基线——
与设计前门禁判定一致，无新增违规。**维持 PASS**。

## Project Structure

### Documentation (this feature)

```text
specs/001-sched-design-landing/
├── plan.md              # 本文件
├── research.md          # Phase 0：五个落地件的方案决策
├── data-model.md        # Phase 1：规划条目/manifest 字段/审计记录/报告 schema
├── quickstart.md        # Phase 1：验证命令与期望
├── contracts/           # Phase 1：内部接口契约（谓词签名、manifest schema、lint 规则）
│   ├── access-classify.md
│   ├── manifest-dompath.md
│   └── perf-report-lint.md
└── checklists/
    └── requirements.md  # specify 阶段质量清单（已全过）
```

### Source Code (repository root)

```text
compiler/
├── src/vtcm/ddr_offsets.cpp        # P1：compute_ddr_offsets 前的访问分类谓词接入点
├── src/ir/graph_prepare.cpp        # P1：分类 pass（消费方驱动）；⚠️ VL 线在途文件，错峰
├── include/hnnx/ir/op_def.hpp      # P1/P3：opcode→访问模式/单元规则 静态表（新表，不改既有项）
├── tools/wtop_emit.cpp             # P2：权重去重处注释门禁；P4：manifest 写 dompath_estimate
├── tools/wtop_ops/                 # P3：单元指派审计记录（不改派发行为）
└── tests/                          # 每阶段新增对应 ctest

scripts/
└── perf_report_lint.py             # P5：三分类口径门禁（新文件）

docs/
└── GEHTP_SCHED_HWFACTS.md          # 设计原则真源（已存在，本特性引用不回写）
```

**Structure Decision**: 单仓多组件（compiler host 工具链 + kernels 设备库 + scripts）。
本特性只动 compiler/ 与 scripts/；kernels/ 设备侧零改动（43_hwinfo 探针已交付，属测量资产
非本特性实现项）。

## Complexity Tracking

无 Constitution Check 违规，本表留空。

## Phase 索引

- Phase 0 研究决策：见 [research.md](research.md)（五个落地件的技术选型与替代方案）
- Phase 1 设计：见 [data-model.md](data-model.md)、[contracts/](contracts/)、[quickstart.md](quickstart.md)
- Phase 2 任务分解：由 `/speckit-tasks` 生成（不在本命令范围）
