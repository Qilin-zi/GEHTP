# Implementation Plan: Compiler-Driven Scheduling

**Branch**: `003-compiler-driven-scheduling` | **Date**: 2026-09-28 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-compiler-driven-scheduling/spec.md`

## Summary

把"算子用哪个计算单元 + 数据如何进出片上内存"从**运行期临时决定**上移到**编译期规划**，并让任何运行期偏离都**显式上报而非静默**。三件套：

1. **引擎指派（P1）**：编译器按成本模型（形状 + VTCM 预算 + 单元争用）给每个算子指派计算单元，写入 blob；运行期照做，不再自决。
2. **静态 VTCM carve + 显式搬移（P2）**：编译器产出片上内存时间表 + 显式 DDR↔VTCM 搬移 op；运行期按表执行，不再"谁先来抢全量"。
3. **显式回落（P3）**：blob 逐算子记录"允许/禁止回落 + 回落单元"，运行期做不到计划时要么按记录回落并上报、要么硬报错。

技术路线（见 [research.md](./research.md)）：blob 增加**逐算子引擎注解表**（footer，bump ver），复用编译期已有的 `HwWrapper`/`OpStoreType`/`RESOURCE_HVX|HMX`（现被 emit 丢弃）与 `FancyAllocator`/`cp_solver` carve 先例，扩展既有 `OP_DMA`（arity 8，含 fence）从 DDR→DDR 到 DDR↔VTCM，补齐 `g_last_engine` 记录形成逐算子执行报告。

## Technical Context

**Language/Version**: C11（设备端运行时 `kernels/src/runtime/`）、C++17（宿主编译器 `compiler/`）、Bash（构建/推送脚本）

**Primary Dependencies**: Qualcomm Hexagon SDK / Hexagon HAL（`HAP_compute_res_acquire`、CDSP domain 3）、宿主 libc。无第三方框架。

**Storage**: 模型权重 + blob（`.wtop`）在磁盘；运行期 DDR（片外）+ VTCM（片内，总量 16MB）分时复用。

**Testing**: CTest（`compiler/tests/`，基线 44/44，含 `test_wtop_emit.cpp` 的 blob 解析契约）；设备侧 quickstart + `judge_logits.py` 数值门（cos/top1）。

**Target Platform**: Qualcomm Hexagon CDSP（设备 d0f1784）；宿主 = x86_64 Linux（emitter/编译器跑宿主，执行跑设备）。

**Project Type**: 编译器 + 运行时（推理引擎）

**Performance Goals**: 端到端 wall 不退化（基线 33.9s）；此前被静默回退标量的算子类改走加速路径。

**Constraints**: VTCM 总量 16MB；输出 cos 不回归（基线 0.998646 / top1 29/32）；blob 格式向后兼容（旧产物必须原样可跑）。

**Scale/Scope**: 0.8B 参数模型、单 blob 约 16k op；目标为模型无关的通用图编译能力，0.8B 作验证载体。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` 是**未填充的模板桩**（无真实项目原则、无 `Core Principles` 内容）。因此本 feature **无硬性 gate 可执行**，跳过违规判定。若后续补齐 constitution，需回补本节的合规检查。

## Project Structure

### Documentation (this feature)

```text
specs/003-compiler-driven-scheduling/
├── plan.md              # 本文件
├── research.md          # Phase 0 输出：五个设计决策
├── data-model.md        # Phase 1 输出：实体与字段
├── quickstart.md        # Phase 1 输出：端到端验证指南
├── contracts/           # Phase 1 输出：blob 格式 v3 + 执行报告契约
│   ├── blob-format-v3.md
│   └── execution-report.md
└── tasks.md             # Phase 2 输出（/speckit-tasks，本命令不生成）
```

### Source Code (repository root)

本 feature 横跨编译器（宿主）与运行时（设备）两侧，按现有目录落点：

```text
compiler/                          # 宿主侧（C++17）
├── tools/
│   ├── wtop_emit.cpp              # 发射器：emit 主流程（现 OP_DMA 只发 DDR→DDR）
│   └── wtop_ops/                  # op 降级：op_matmul.cpp 等（现丢弃引擎概念）
├── src/
│   ├── vtcm/                      # carve 先例：fancy_allocator.cpp / cp_solver.cpp / ddr_offsets.cpp
│   ├── dma/spill_fill.cpp         # 搬移先例
│   ├── ir/                        # graph_deps.cpp / graph_prepare.cpp（活性/依赖）
│   └── serialize/serializer.cpp   # 序列化
├── include/hnnx/ir/               # IR 头：op.hpp（OpStoreType/RESOURCE_*）/ mid_level_ir.hpp（HwWrapper）
└── tests/test_wtop_emit.cpp       # 契约测试（44 基线之一）

kernels/                           # 设备侧（C11）
├── src/runtime/
│   ├── oplist_exec.c              # 执行器：exec_matmul_f16 / g_last_engine / vtcm_pool_ensure
│   └── hmx_common.c               # hmx_runtime_setup（现忽略 size、抢全量 VTCM）
└── include/
    ├── oplist_parse.h             # blob 线格式：wt_op / wt_slot / opcode / 槽类型
    └── oplist_exec.h              # wt_engine 枚举
```

**Structure Decision**: 沿用现有编译/运行分侧的目录结构，不改动目录骨架。引擎注解表、VTCM carve 表、搬移 op 全部落在既有 `compiler/tools/wtop_emit.cpp`（发射）与 `kernels/{include/oplist_parse.h,src/runtime/oplist_exec.c}`（解析/执行）两侧；carve 算法复用 `compiler/src/vtcm/` 现有求解器。

## Complexity Tracking

> 无 Constitution 违规需要 justify（constitution 为空模板）。本表留空。
