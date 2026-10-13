# Implementation Plan: 013-executor-hoist

**Branch**: `013-executor-hoist` | **Date**: 2026-10-13 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/013-executor-hoist/spec.md`

## Summary

把执行器剩余的五类运行时规划按 B→A→C→D 四段上移编译期：B=W4A16 供给侧三槽直连（requant 生成器落地 + arity 6→9 + 删槽表扫描，激活 W4A16 E2E）；A=VTCM 静态布局一统（单张布局表 + hmx_runtime_setup 有界区域 + flush 收敛）；C=carve 布局序列化（热路径指针切换）；D=形态 flag 化（前置 012 落地）。每段过零回归门（f16 线位级 / gdn 线 cos 0.999930），终局双门 = 精度不回退 + w4 线 exec 较 25.7s 锚 ≥1.5×。

## Technical Context

**Language/Version**: C（kernels，Hexagon SDK hexagon-clang + 宿主 gcc 双目标）；C++17（compiler/tools）

**Primary Dependencies**: Hexagon SDK（HAP/QURT/HVX/HMX intrinsic、FastRPC 签名链）；011 op_spec.inc 单点注册表（kernels/include/op_spec.inc）；007 gguf_wtlink GGUF 读取层；008 BudgetReport 预算门；003 carve footer 基建

**Storage**: blob 文件格式（wtop：槽表 + op 表 + 注解/carve/GDN footer + 权重区）；无外部存储

**Testing**: host ctest（49/49 基线，test_wtop_emit/test_opt_passes）；kernels build_examples（337 pass）；check_device_golden（0.8B 双门）；合成 blob 法（003 T014 先例，P5 dc_w4 已知答案）；op_ts + scripts/gehtp_prof.py 取证

**Target Platform**: Qualcomm Hexagon HTP CDSP（板 d0f1784，VTCM 16MB 口径）+ 宿主 x86_64 Linux（编译器与 host 单测）

**Project Type**: 图编译器 + 设备执行器（library/compiler）

**Performance Goals**: 默认/gdn 线 exec ≤2.77s 不回退；w4 线 exec ≤17.1s（≥1.5×，stretch 5.99s）；re-carve 全 VTCM flush 次数 N→1

**Constraints**: cos 门（默认线 ≥0.999930、w4 线 ≥0.98）；CDSP 堆 2.04-2.89GB（blob 尺寸硬门）；VTCM 预算（resident_pool+hmx_arena≤16MB，编译期门）；D 段硬前置 012 落地；主树 oplist_exec.c 有 012 未提交 WIP（worktree 隔离开发）

**Scale/Scope**: 0.8B（~14k op，151 可 W4A16 线性层 + 36 N=16 回退）；三线基线对拍（默认/gdn、f16 非融合、007 q4）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution 为未填充模板（`.specify/memory/constitution.md` 全占位符，无生效 MUST 原则）→ 本 gate 空转通过，无违规可判。已在 spec checklist Notes 记录，建议后续回合独立填充 constitution（不在本特性 scope 内）。

## Project Structure

### Documentation (this feature)

```text
specs/013-executor-hoist/
├── plan.md              # 本文件
├── research.md          # Phase 0：九项关键裁决
├── data-model.md        # Phase 1：供给槽/op/布局表/锚
├── quickstart.md        # Phase 1：端到端验证指南
├── contracts/           # Phase 1：三个 blob 级契约
│   ├── contract-w4a16-op.md
│   ├── contract-vtcm-layout.md
│   └── contract-carve-layout.md
└── tasks.md             # /speckit-tasks 产出（本命令不建）
```

### Source Code (repository root)

```text
compiler/tools/
├── wtop_emit.cpp            # B: 9-arity 发射 + 三槽供给；A: VTCM 布局表(TEMPOFF 扩展)；C: carve 布局序列化；D: 形态 flag
├── requant_w4a16.cc         # B: 新建——Q4_0 GGUF → 打包权重+atbl/otbl+folded_bias 生成器（复用 gguf_wtlink 读取层）
├── hnnx_compile.cpp         # B/A: 供给侧接线与预算门挂点
└── wtop_ops/
    ├── op_matmul.cpp        # B: W4A16 op 构造(6→9 参)
    └── op_broadcast.cpp     # D: 形态 flag 注入

kernels/include/
├── op_spec.inc              # B: OP_MATMUL_W4A16 arity 9 + legacy_arity 6（单点注册表扩展）
└── oplist_parse.h / oplist_exec.h

kernels/src/runtime/
├── oplist_exec.c            # B: exec_matmul 直读三槽删扫描；A: 布局表装载+删探戈+flush 收敛；C: 指针切换；D: 注解路径直用形态
├── oplist_parse.c           # B: 双 arity 分派
├── hmx_common.c             # A: hmx_runtime_setup 有界区域
└── wtcache_impl.c           # A: 布局表驱动的 wtcache 布局

tests（host）
├── compiler/tools/test_wtop_emit.cpp   # B/C/D: 序列化往返 + 契约断言
└── （ctest 既有套件为回归基线）
```

**Structure Decision**: 复用既有 compiler/kernels 双树布局；新增文件仅 `requant_w4a16.cc` 一个；其余全部为既有文件扩展。契约面集中在 blob 格式（三份 contracts/）与 op_spec.inc 注册表。

## Complexity Tracking

> 无 Constitution 违规需要豁免——表留空。
