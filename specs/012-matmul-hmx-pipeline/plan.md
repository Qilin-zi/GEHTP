# Implementation Plan: MatMul HMX 流水线（tile 级计算流水 + op 级权重预取）

**Branch**: `012-matmul-hmx-pipeline` | **Date**: 2026-10-13 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/012-matmul-hmx-pipeline/spec.md`

## Summary

把 f16 matmul（opcode 24/35/36）从「逐 op 权重重物化 + 逐 tile HMX 握手」升级为流水执行：主杠杆是把 `hmx_enable_execution`/`hmx_unit_acquire`/`hmx_unit_release`/`hmx_disable_execution` 从 `(r,c)` tile 循环内提到循环外（192 次握手 → 1 次），次杠杆是新增异步 DMA（`dc_dma_submit`/`dc_dma_wait`）让下一个 matmul 的权重搬运与当前计算重叠。累加器边界由 `mxclracc.hf` 提供，输出须与串行基线位级一致。

## Technical Context

**Language/Version**: C（Hexagon v81 DSP）+ inline asm（Hexagon V81 MX/HMX 指令、Q6_mxmem_* intrinsics）

**Primary Dependencies**: Hexagon SDK（`HAP_compute_res_hmx_lock2`）、QuRT RTOS（`qurt_mem_cache_*`）、DMA descriptor API（`dmstart`/`dmwait`/`dmlink`/`dma_desc_is_done`）、HMX intrinsics

**Storage**: 权重在 DDR blob；片内暂存在 VTCM（16MB 预算）；无持久化存储

**Testing**: 设备 golden 对拍（`check_device_golden`）、逐 op profile（`gehtp_prof.py`）、host 参考（`host_run`）、`build_examples.sh`（基线 337 pass）

**Target Platform**: Qualcomm Hexagon v81 HTP/HMX DSP（设备 `d0f1784`）

**Project Type**: 编译器 + 自定义运行时（library）

**Performance Goals**: prefill 墙钟 −20%（matmul+FC 约占墙钟 88%）

**Constraints**: VTCM 16MB 暂存预算；输出位级一致（f16 噪声容差内）；单线程 runtime；`dmlink` 链式当前被禁（R-D1 只走 `dmstart`）

**Scale/Scope**: Qwen3.5-0.8B，18 层；权重 f16 ~2GB / Q4_0 476MB

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

项目 `.specify/memory/constitution.md` 为未填充模板（无实际原则条款），无强制约束。故本检查**空过**（无违规需评估）。落地纪律以 spec 的 FR 与 contracts 的 fence/位级一致性契约为准。

## Project Structure

### Documentation (this feature)

```text
specs/012-matmul-hmx-pipeline/
├── plan.md              # 本文件
├── research.md          # Phase 0: R1 握手外提 / R2 tile 流水 / R3 异步 DMA
├── data-model.md        # Phase 1: HMX 会话/暂存区/权重搬运/预取描述符/累加器边界
├── quickstart.md        # Phase 1: 五个可运行验证场景
├── contracts/           # Phase 1: runtime-interfaces.md（异步 DMA + HMX 会话 + 位级一致）
└── tasks.md             # Phase 2 (/speckit-tasks, 本命令不生成)
```

### Source Code (repository root)

```text
kernels/src/runtime/
├── oplist_exec.c        # 执行器: exec_matmul_f16_hmx 的 tile 走查(握手外提/流水)
├── hmx_common.c         # HMX 锁: hmx_enable/disable + unit_acquire/release
├── dc_parts.c           # DMA: dc_dma_once → 拆 dc_dma_submit/wait(异步)
kernels/src/hmx/
└── hmx_convf16.c        # 参考实现(逐 tile 握手, 外提的对照基线)
kernels/include/
├── dc_parts.h           # dc_dma_submit/wait 接口声明
└── dma_cmds.h/dma_desc.h# dmstart/dmwait/dmlink/dma_desc_is_done 原语(已有)
compiler/tools/wtop_ops/
└── op_matmul.cpp        # emit 侧(如需 flags/预取标记变更; 本特性主改动在 runtime)
```

**Structure Decision**: 主改动集中在 `kernels/src/runtime/oplist_exec.c`（执行器）与 `dc_parts.c`（DMA 异步化），`hmx_common.c`/`hmx_convf16.c` 只读作对照。emit 侧（`op_matmul.cpp`）本特性预期无需改动（预取由 runtime 前瞻 op 列表实现），除非需要新增 flags。

## Complexity Tracking

> 无宪法违规需 justify。
