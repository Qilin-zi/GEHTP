# Implementation Plan: 静态内存规划收口

**Branch**: `010-static-mem-plan-closure` | **Date**: 2026-10-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/010-static-mem-plan-closure/spec.md`

## Summary

编译器（host）已把内存规划做到位（ST-Cut、FancyAllocator、cp_solver、block table、tiling、指令选择），blob v3 也已承载大部分机制；本特性把最后 5 个「编译器算好却没下发、执行器还在自己猜/硬编码/运行时重排」的收口点闭合：A 显式标注 W4A16 辅助面身份、B 编译期算 pin_cap 下发、C 编译期预判 carve 复用、D 权重 pin/restage 时序静态化、E 权重 crouton pack 编译期化。执行器 `wt_exec_run` 退化为纯查表机械执行，删除硬编码/按尺寸猜/运行时判断/运行时重排；qwen3-0.8b 端到端不回退。

## Technical Context

**Language/Version**: C（设备运行时/内核，`kernels/`）；C++17（host 编译器，`compiler/`）

**Primary Dependencies**: Hexagon SDK（Q6_* intrinsic）、qurt、自研 wtop blob（v3，`oplist_parse.h`）、QNN/ggml（模型导出侧，本特性不触碰）

**Storage**: wtop blob（二进制 tagged.bin）+ 路线 B 外置权重文件；无数据库

**Testing**: `build_examples.sh`（35 例 + 261 门）、W4A16 256³ 65536/65536 bit-exact 门、设备 golden 对拍（check_device_golden）、host 数值对拍（run_host_ci）、`wt_w3_report` 统计

**Target Platform**: Hexagon v81/v73 HTP（DSP）+ host Linux 编译器

**Project Type**: 编译器 + 设备运行时（embedded/交叉编译）

**Performance Goals**: 精度 cos ≥ 0.998642 且 wall-clock ≤ 6.31s（当前库锚不回退）；权重 crouton pack 运行时耗时按重测基线归零（上界 = T017 LUT 后的剩余量，非旧账 20.6s）

**Constraints**: VTCM 16MB 口径、byte-exact 回归纪律（禁位级漂移）、零运行时决策（零 fallback/按尺寸猜/超规划 restage/运行时 crouton 重排）、blob 槽位与 opcode 向后兼容（ver≤2 旧 blob 仍可跑）、Q4_0 内存收益守恒（E 不得把量化态反量化为 f16 存储）

**Scale/Scope**: 0.8B 模型 ~6K-14K op；5 个收口点，每个独立可验收

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` 为未填充模板（全 `[PLACEHOLDER]`），无正式宪法条款，**无形式 gate**。本项目事实上的纪律（从跨会话协调表 / 验证线三件套继承）作为等效 gate：

| 等效 gate | 本特性对应 |
|-----------|-----------|
| byte-exact 回归门（35 例 + 261 门 + W4A16 256³） | FR-009 / SC-002 |
| cos 锚门 0.998642（不回归，非抬到 0.9999） | FR-009 / SC-005 |
| 数值纪律（三判据 + 次正规地板，禁位级漂移） | 收口 A/C/D/E 改动须先与旧行为逐字节对齐 |
| Q4_0 内存收益守恒（blob 4.3× 压缩不破） | E 须 pack 量化态而非反量化 f16 |

**结论**：无违反项。A/C/D/E 删运行时判断/重排后，emit 侧必须先与旧行为逐字节对齐，否则违反 byte-exact gate —— 这是本特性唯一的硬约束来源；E 额外受 Q4_0 内存守恒约束。

## Project Structure

### Documentation (this feature)

```text
specs/010-static-mem-plan-closure/
├── plan.md              # 本文件
├── research.md          # Phase 0：四个开放项的裁决（R1-R4）
├── data-model.md        # Phase 1：blob 扩展的数据模型（含 E 权重槽）
├── quickstart.md        # Phase 1：可运行的验证场景（含 E）
├── contracts/           # Phase 1：blob v3 契约扩展（含契约 E）
│   └── blob-v3-extensions.md
└── tasks.md             # Phase 2（/speckit-tasks 产出，本命令不建）
```

### Source Code (repository root)

```text
kernels/
├── include/hmx_crouton.h        # hmx_crouton_pos 公式（E 的单一真值源）
├── include/oplist_parse.h       # blob 格式常量/注解打包（A/C 契约点）
├── src/runtime/oplist_exec.c    # 执行器（A/B/C/D/E 删硬编码/按尺寸猜/运行时判断/运行时重排）
└── src/runtime/wtcache_impl.c   # wtcache_open pin_cap（B 消费点）
compiler/
├── src/ir/graph_prepare.cpp     # mem_plan（B 算 pin_cap、D 新建权重常驻规划）
├── src/vtcm/cp_solver.cpp       # CP oracle（env 门控，仅 temp 张量，不产权重 restage）
├── tools/wtop_emit.cpp          # blob 发射（A/B/C/D/E 序列化侧）
├── tools/wtop_ops/wtop_ops.hpp  # CarveEntry / op_annot 结构（C 载体）
└── tools/weight_layout_pack.*   # E 新增：权重 crouton 编译期打包（net-new）
```

**Structure Decision**: 无新目录。全部改动落在既有 `kernels/`（执行器侧）与 `compiler/`（规划侧）两棵树，blob 是两者唯一接口（`contracts/` 只文档化该接口的扩展）。E 在 `compiler/tools/` 下新增权重打包模块（`weight_layout_pack`），执行器删对应运行时 pack 循环。

## Complexity Tracking

无 Constitution 违规，无复杂度豁免需记录。
