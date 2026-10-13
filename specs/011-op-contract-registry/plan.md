# Implementation Plan: 算子输入契约单点注册表

**Branch**: `011-op-contract-registry` | **Date**: 2026-10-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/011-op-contract-registry/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command.

## Summary

把散落在 6 处的算子输入契约合并为**单一声明式注册表 `wt_op_spec[]`**，使 arity、参数身份、参数语义成为机器可读的**唯一事实源**。

- **P0（本 spec 主交付）**：合并 `#define WT_ARITY_*` 宏 + `arity_of()` switch + `arg_is_slot()` switch 三张表为一张表，`arity_of()` / `arg_is_slot()` 改从注册表读取，取值逐值等价。
- **P1（本 spec 交付）**：给每个参数加 `name`（语义名）+ `kind`（5 类编码），把 enum 注释里的人工语义升级为程序可校验的字段。
- **P2（明确排除）**：`exec_*` 按名取参、emit 侧（`weight_input_index` + `op_*.cpp`）改读注册表——后续单独立项。

核心约束：**byte-exact**（现有 35 例 + 261 门 + W4A16 65536/65536 回归逐位不变）。

## Technical Context

**Language/Version**: C11（kernels/runtime，设备执行器）+ C++17（compiler，宿主）。两棵树已共享同一份 `kernels/include/oplist_parse.h` 与 `kernels/src/runtime/oplist_parse.c`。

**Primary Dependencies**: 无外部依赖。改动限于内部头/源：`kernels/include/oplist_parse.h`、`kernels/src/runtime/oplist_parse.c`（`oplist_exec.c` 仅 P2 才碰）。

**Storage**: N/A —— 注册表是编译期常量数组（static const），无运行时存储。

**Testing**: 既有 byte-exact 回归（35 例 + 261 门 + W4A16 65536/65536）+ host 单测（`oplist_parse.c` 平台无关，host/设备同一份源码）+ 新增「新旧表逐值对拍」parity 单测。

**Target Platform**: Hexagon DSP（设备侧 `.signed.so`）+ Linux 宿主（编译器 + 单测）。

**Project Type**: 编译器/执行器工具链（内部重构）。

**Performance Goals**: 无（不回归即可）。arity/身份查询是每 op 解析一次 O(1)，非热点；不期望任何 wall 变化。

**Constraints**:
1. byte-exact 硬门（FR-006 / SC-001）——迁移前后编译产物逐位一致。
2. 不改 blob 线格式（wt_op/wt_slot/wt_blob 结构、wire 编码）。
3. `arity_of()` / `arg_is_slot()` 的返回值必须与迁移前**逐值等价**（含 default 分支语义）。
4. 设备侧 `.signed.so` 与宿主编译器必须读到同一份契约（已天然满足：两边同源 `oplist_parse.c/.h`）。

**Scale/Scope**: 36 个 opcode；`arity_of()` 34 个 case、`arg_is_slot()` 32 个 case、`#define WT_ARITY_*` 36 个、enum 语义注释 1 段（oplist_parse.h:133–262）。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**结果：无适用原则（空模板）**。`.specify/memory/constitution.md` 仍是未填写的模板（`[PROJECT_NAME]` / `[PRINCIPLE_*]` 全为占位符），不存在可校验的 MUST/SHOULD 条款。因此本特性的约束全部来自 spec 自身的 FR/SC（尤其 FR-006 byte-exact、FR-008 不碰 exec_* 解码）。

> 注：若后续启用 constitution，byte-exact 与「单点事实源」应升格为其 MUST 条款。本次不自行发明原则。

## Project Structure

### Documentation (this feature)

```text
specs/011-op-contract-registry/
├── plan.md              # 本文件
├── research.md          # Phase 0 输出：6 处契约散落地图 + 决策
├── data-model.md        # Phase 1 输出：wt_op_spec 数据模型
├── quickstart.md        # Phase 1 输出：parity 对拍 + byte-exact 验证
├── contracts/           # Phase 1 输出：注册表 schema 契约
│   └── op-spec-schema.md
└── tasks.md             # Phase 2 输出（/speckit-tasks，本命令不创建）
```

### Source Code (repository root)

```text
kernels/
├── include/
│   └── oplist_parse.h       # enum(133-262) + WT_ARITY_* 宏(265-300) + wt_* structs
│                            #   → 新增 wt_op_spec 表 + wt_arg_spec 结构（或独立 op_spec.inc）
└── src/runtime/
    ├── oplist_parse.c        # arity_of()(10-50) + arg_is_slot()(53-88) + wt_parse()(90+)
    │                         #   → 两函数改读 wt_op_spec[]；arity/身份宏删除或派生化
    └── oplist_exec.c         # exec_* + g_op_exec_table —— P2 才碰，本 spec 不动

compiler/
└── tools/wtop_ops/
    ├── wtop_ops.hpp          # weight_input_index()/op_registry() —— P2 才读注册表
    └── op_*.cpp              # emit 编码 handler —— P2 才碰

compiler/tests/
└── test_wtop_emit.cpp        # 已链接 oplist_parse.c，天然覆盖 parity 对拍
```

**Structure Decision**: 注册表放入 `kernels/include/`（与 `oplist_parse.h` 同目录，或作为其内含的 `.inc` 段），因为该目录已被 kernels(C) 与 compiler(C++) 两棵树**共同 include**——单点事实源不需要新建共享通道，放在现有共享头里即可。设备侧 `.signed.so` 与宿主编译器自动读到同一份表。

## Complexity Tracking

> 仅在 Constitution Check 有需申辩的违规时填写。本次无适用原则，故无违规。
