# Implementation Plan: Graph Identity Passes

**Branch**: `005-graph-identity-passes`（工作树 task/prof-wp，多会话共享——不切分支，见 Constitution Check） | **Date**: 2026-09-30 | **Spec**: [spec.md](spec.md)

> **状态（2026-10-08 完成）**: US1+US2 `951cef7` / US3a `1e131fb` / US3b `69893e4`；kernels 侧 T018/T019 代码留主树在途（HEAD 无 opcode36 exec 基座，随 kernels 统一提交落库，见跨会话协调表）。性能门**超标兑现**: op 16438→14152（-13.9%，门 -6%）、wall 39.5s→31.49s（-20.3%，门 -4%）。Div→Mul 拆 `eltwise_div_to_mul` 默认关（设备 NaN 实证）。终局五门: `.work/prof/005-final-gates-20261008.md`。

**Input**: Feature specification from `/specs/005-graph-identity-passes/spec.md`

## Summary

在 PassManager 注册制下新增三个图级恒等 pass，对齐真库 libQnnHtp.so 同级能力：**P1 `view_fold`**（StridedSlice/Reshape/Transpose 视图链合并与吸收，盘子 ~4s/35.6s）、**P2 `eltwise_const_fold`**（同型 const 二元链合并 + Div→Mul，~0.3-0.5s）、**P3 `fc_bias_fold` 推广到 MatMul**（分 3a 纯图级 / 3b HMX-bias 设备验证两阶段）。全部沿用 e4d1dbb（fc_bias_fold）确立的纪律：数学恒等、幂等、单消费者门、`--no-pass` 逃生、`--opt-stats` 计数。

## Technical Context

**Language/Version**: C++17（compiler，宿主 gcc-9.4 可编 / 容器 gcc-13 canonical）；P3b 另有 Hexagon C（kernels）

**Primary Dependencies**: 既有 PassManager 注册制（pass_manager.cpp，a5bdc70）、scalar_params/op_data 机制（e4d1dbb）、tagged 序列化、wtop_emit 算子发射器（op_matmul.cpp）、设备 oplist_exec.c

**Storage**: N/A（编译器 pass，无持久化新增——op_data 标记沿用既有序列化字段）

**Testing**: ctest（47 既有 + test_opt_passes 新增图）；host_run 对拍（位级/ULP）；0.8B v2 上板 judge_logits.py + prof 对比

**Target Platform**: Linux x86-64 宿主编译 + Hexagon v81 HTP 设备（d0f1784）

**Project Type**: compiler pass 库（htp_core 静态库 + CLI 工具）

**Performance Goals**: 0.8B v2 设备 wall 39.5s → ≤37.9s（SC-001 ≥4%）；op 数 16438 → ≤15450（≥6%）

**Constraints**: P1/P2/3a **零设备语义改动**（恒等变换）；3b 前不撤 HMX 门；与 003 会话共享 graph_prepare.cpp/wtop_emit/oplist_exec.c（改前 `git status` 协调）；设备磁盘常满（推 blob 前 `df`）

**Scale/Scope**: 0.8B v2 图 16438 ops（SS 3507 / Reshape 3836 / EB 4313 / MatMul 139 / FC 218）；3 个新 pass + 1 处发射门扩展 + 1 处内核扩展

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` 为未填充模板（无既定门）——按项目事实约束执行：

- **门 1（恒等性）**: 三 pass 数值恒等（P1 位级、P2 ≤1 ULP、P3 与标量路径同口径）——SC-002/SC-005 强制。
- **门 2（幂等）**: 二跑零改写，单测覆盖。
- **门 3（逃生）**: `--no-pass view_fold|eltwise_const_fold|fc_bias_fold` 可关。
- **门 4（协调）**: 共享文件改前查 `git status`；提交精确暂存（e4d1dbb 先例）。
- **无违规**，无需 Complexity Tracking。

## Project Structure

### Documentation (this feature)

```text
specs/005-graph-identity-passes/
├── plan.md              # 本文件
├── research.md          # Phase 0: 操作码映射/参数源/HMX 门/基数口径
├── data-model.md        # Phase 1: SliceSpec 代数实体 + pass 数据流
├── quickstart.md        # Phase 1: 验证命令手册
├── contracts/           # Phase 1: CLI/stats/op_data 契约
└── tasks.md             # T001-T015（已生成）
```

### Source Code (repository root)

```text
compiler/
├── include/hnnx/opt/
│   ├── pass_view_fold.hpp        # P1: 视图链 pass 声明
│   ├── pass_eltwise_const.hpp    # P2: 同型合并 pass 声明
│   └── pass_fc_bias.hpp          # 既有，P3a 扩展匹配器（不加文件）
├── src/opt/
│   ├── pass_view_fold.cpp        # P1: slice 代数 + 三条规则
│   ├── pass_eltwise_const.cpp    # P2: 同型合并 + Div→Mul
│   └── pass_fc_bias.cpp          # P3a: MatMul 链头
├── src/ir/graph_prepare.cpp      # 仅注册点 +1 行/pass（共享文件，协调）
├── tests/test_opt_passes.cpp     # 新增测试图 [13][14][15]
├── tools/wtop_ops/op_matmul.cpp  # P3a: 发射门放行 MatMul（共享文件）
└── CMakeLists.txt                # +2 源文件
kernels/src/runtime/oplist_exec.c # P3b: HMX bias（003 在途，独立窗口）
```

**Structure Decision**: 完全复用 pass 注册制骨架（pass_manager/opt 目录惯例），新 pass 一个 .cpp/.hpp 对，不引入新目录或新抽象层；slice 代数作为 pass_view_fold.cpp 内部自由函数（单测经 test_opt_passes 黑盒驱动，不为它单独建库——YAGNI）。

## Complexity Tracking

> 无 Constitution Check 违规，不需要。
