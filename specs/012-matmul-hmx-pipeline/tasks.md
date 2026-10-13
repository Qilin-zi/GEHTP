---
description: "Task list for 012-matmul-hmx-pipeline feature implementation"
---

# Tasks: MatMul HMX 流水线（tile 级计算流水 + op 级权重预取）

**Input**: Design documents from `/specs/012-matmul-hmx-pipeline/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/runtime-interfaces.md, quickstart.md

**Tests**: 设备侧验证为本特性硬门（spec FR-004/SC-002/SC-003），验证任务内嵌在每阶段；无独立 TDD 测试文件。

**Organization**: 按 user story 分组（US1 tile 流水 / US2 数值一致 / US3 op 权重预取 / US4 零回归），来自 spec.md P1→P2 优先序。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: 所属 user story（US1/US2/US3/US4）
- 所有路径相对 `/disk1/gehtp/`

## Path Conventions

- 运行时: `kernels/src/runtime/`（oplist_exec.c, dc_parts.c, hmx_common.c）
- 头文件: `kernels/include/`（dc_parts.h）
- 参考实现: `kernels/src/hmx/hmx_convf16.c`（只读对照）
- 工具: `scripts/gehtp_prof.py`
- 设备门脚本: 按 [[gehtp-validation-line]] 的 check_device_golden / build_examples.sh

---

## Phase 1: Setup（共享基础设施）

**Purpose**: 基线锚定与改动前置

- [X] T001 记录改动前基线: 设备 d0f1784 跑 0.8B prefill blob, 保存 op_ts.bin + 墙钟 + 输出 md5 到 specs/012-matmul-hmx-pipeline/baseline/（含 attn_qkv 单 op 耗时 ~102ms 锚）; 用 scripts/gehtp_prof.py 出逐 op 表
- [X] T002 [P] 源 env.sh 后重编当前 HEAD 内核并重推签名库, 确认基线可复现（同 blob 输出位级一致）——[[gehtp-build-env]] 流程

**Checkpoint**: 基线锚定, 后续所有改动对照此基线验证

---

## Phase 2: Foundational（阻塞性前置）

**Purpose**: R1 的机制前置——共享 worktree 卫生与设备验证入口

- [X] T003 核对 /disk1/gehtp-wt/010-static-mem-plan-closure/kernels/src/runtime/oplist_exec.c 的未提交脏改动（并发会话 US2 遗留, 见 [[gehtp-010-merge-mainline]]）: 若仍在, 将本特性改动落在正确基线（origin/task/prof-wp HEAD）上, 不动该脏文件; 改前先查 [[gehtp-cross-session-coordination]] roster
- [X] T004 确认设备验证三件套可跑: check_device_golden（0.8B cos≥0.9986/top1 32/32 门）、build_examples.sh（337 pass 基线）、单 op 微 blob（mm32 matmul 1024/1024）——任一不可跑则先修环境再继续

**Checkpoint**: 可以安全改 oplist_exec.c 并有可信的设备门

---

## Phase 3: User Story 1 — tile 级计算流水化（Priority: P1）🎯 MVP

**Goal**: exec_matmul_f16_hmx 的 tile 走查从逐 tile 握手改为一次 HMX 会话跑完整个 (r,c) 面; 192 次 acquire/release → 1 次

**Independent Test**: 单 mm32 微 blob 输出位级 = 基线; attn_qkv 单 op 耗时显著收缩（目标 <20ms）

### 实现

- [X] T005 [US1] 握手外提落地并设备验证: 位级同 (lift1 md5=基线), 零性能增益 — 已保留 (192→1 次/op); 附带裁决: act 驻留被 ISA 判死 (汇编器 "packet has HMX weight without an activation" 硬规则), 见 results.md
- [X] T006 [US1] bias/sca 复核: sca 恒等页 init 在会话外一次 (oplist_exec.c memset(sca)+0x3C00 循环), bias 奇位写在 (r,c) 内逐列照旧, 外提无语义扰动; pp/ctrl 位级同佐证
- [X] T007 [US1] 设备位级门: pp blob vs ctrl blob 输出 md5 位级同 (9d1c0877), cos 0.999930/top1 32/32 双绿; 老锦标 blob (009 emit) 另立 ctrl 锚
- [X] T008 [US1] 收益测量+真根因: 531µs/tile 与 n_dot 无关 → 归因标量 wgt pack (blob 漏 --dma-runlist); 重 emit [pp]182 后 wall 13.58s→2.76s (−79.7%) 位级同, SC-001 达成; 详见 results.md

**Checkpoint**: US1 独立成立——单 matmul 握手串行消除, 输出位级一致

---

## Phase 4: User Story 2 — 数值与正确性不因流水化而变（Priority: P1）

**Goal**: 全量正确性门——流水化不引入累加器跨 tile 残留、部分和错位、bias 折入错误

**Independent Test**: build_examples.sh 337 pass 全绿 + 0.8B 双门（cos≥0.9986 / top1 32/32）

### 实现

- [X] T009 [US2] 边界用例核查: n_col=1（单 tile）、n_row×n_col 极小、M<32 GEMV、batched、SPLIT 续 chunk（g_split_prev 引擎沿用键不受握手外提影响）各跑一遍微 blob, 输出对基线
- [X] T010 [US2] 全量 build_examples.sh（DEVICE=d0f1784）: 337 pass / 0 FAIL; 出现新 FAIL 逐个归因（区分预存 vs 本改动引入）
- [X] T011 [US2] 0.8B E2E 双门: 用 [[gehtp-08b-e2e-canonical-flags]] 正典旗标（--dma-runlist 等）跑全链, cos-min 边界注意 0.9986 假 FAIL 坑（[[gehtp-009-merge-mainline]]）

**Checkpoint**: US1+US2 合并成立——流水化可安全合入

---

## Phase 5: User Story 3 — op 级权重预取（Priority: P2）

**Goal**: 当前 matmul 计算期间, 提前发起下一个 matmul 的权重 DDR→VTCM 搬运（异步 DMA + 双暂存区乒乓）

**Independent Test**: 多 matmul blob 的 profile 中权重搬运停顿从关键路径消失; stg_fb=0; 输出位级一致

### 实现

- [X] T012 [US3] kernels/src/runtime/dc_parts.c: 新增 dc_dma_submit（dmstart 触发即返）与 dc_dma_wait（poll dma_desc_is_done）; 拆分 dc_dma_once_ex 为 submit+wait 两段, dc_dma_once 保持 ≡ submit+wait 兼容（既有调用点零扰动）
- [X] T013 [US3] kernels/include/dc_parts.h: 声明 dc_dma_submit/dc_dma_wait, 注明契约 A 所有权规则（写中禁读、完成前禁再 submit/destroy、异步期间源禁 CPU 改写）
- [X] T014 [US3] kernels/src/runtime/oplist_exec.c: matmul 执行前前瞻下一 op——命中同族可预取 matmul（opcode 24/35/36 且权重为 w_pp 槽）时, 对双暂存区乒乓发起 dc_dma_submit; 当前 op 开头 dc_dma_wait 对应屏障 + FC_DMA→FC_HMX invalidate（fence_handoff, 铁律①同 w_pp 路径）
- [X] T015 [US3] 预算与回落（FR-007）: 下一权重 > 单暂存区或双区总占用超 VTCM 预算 → 不预取、维持串行; 非可预取 op → 前瞻失效干净回落; blob 边界/会话开始复位预取态（FR-008）
- [X] T016 [US3] 设备门: 多 matmul 微 blob + 0.8B 全链位级对拍; 确认无 use-before-ready（wait 屏障前无 HMX 读）、无乒乓覆盖; profile 验证搬运停顿隐藏

**Checkpoint**: US3 独立成立——搬运与计算重叠, 无正确性回归

---

## Phase 6: User Story 4 — 无重叠机会时零回归（Priority: P2）

**Goal**: 单 matmul / 单 tile / 非 matmul 边界与现状完全一致

**Independent Test**: 单 matmul、n_col=1、非 matmul 微 blob 时序与行为 ≤ 基线 + 小容差

### 实现

- [X] T017 [US4] 单 matmul（无前瞻目标）微 blob: 走一次串行搬运, 无双缓冲分配、无额外延迟; 时序对基线
- [X] T018 [US4] matmul 后紧跟非 matmul op 的 blob: 前瞻失效路径只为本 op 发起一次串行搬运, 无报错无空预取
- [X] T019 [US4] GEHTP_PERSISTENT 多 step 连跑: 每会话首 op 预取态干净复位, 无跨会话 stale 完成态

**Checkpoint**: 全 story 成立, 特性完整

---

## Phase 7: Polish & Cross-Cutting

- [X] T020 [P] specs/012-matmul-hmx-pipeline/results.md 汇总终态: 各门数值（cos/top1/pass 数/墙钟）+ attn_qkv 单 op 前后对照表
- [X] T021 [P] 更新 specs/012-matmul-hmx-pipeline/{research.md,quickstart.md} 勾落已验证结论（R1 外提实测开销、R2 是否继续的判据）
- [X] T022 R2 决策点（tile 软件流水, 本期默认 NO-GO 除非 T008 数据显示外提后装载仍占大头）: 若 NO-GO 在 research.md 记录依据; 若 GO 另立后续任务段
- [X] T023 回归总门: T001 基线 vs 终态全对比（0.8B 双门 + 337 examples + prefill 墙钟 ≥20% 降幅核验 SC-001）
- [X] T024 commit 分组提交（US1+US2 一组 / US3+US4 一组）, 按团队惯例 push 前确认

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: 立即可开始; T001/T002 可并行
- **Phase 2 Foundational**: 依赖 Phase 1; 阻塞全部 story
- **Phase 3 (US1)**: 依赖 Phase 2 —— MVP 主体
- **Phase 4 (US2)**: 依赖 Phase 3（对 US1 的产物做正确性收口）
- **Phase 5 (US3)**: 依赖 Phase 4（dc_parts.c 改动独立, 但 oplist_exec.c 接线须在 US1 落定后）
- **Phase 6 (US4)**: 依赖 Phase 5（回落路径针对预取机制）
- **Phase 7 Polish**: 依赖全部所需 story 完成

### User Story Dependencies

- US1 → US2（串行, 同文件位级门）
- US1 落定 → US3（T012/T013 可提前并行, T014 起串行）
- US3 → US4（回落语义依赖预取机制在场）

### Parallel Opportunities

- T001 ∥ T002（Phase 1）
- T005 完成后 T007/T008 部分可并行（位级门与 profile 测量不同脚本）
- T012 ∥ T013（不同文件）——可在 Phase 4 期间提前
- T020 ∥ T021（不同文件, Phase 7）

---

## Parallel Example: Phase 5 提前并行

```bash
# US1 验证期间可先行（不同文件）:
Task: "T012 dc_parts.c 新增 dc_dma_submit/dc_dma_wait"
Task: "T013 dc_parts.h 接口声明 + 契约 A 注释"
```

---

## Implementation Strategy

### MVP First（US1+US2 Only）

1. Phase 1 基线锚定 → Phase 2 前置核查
2. Phase 3 US1 握手外提 → 位级门 → 收益测量
3. Phase 4 US2 全量正确性门
4. **STOP and VALIDATE**: 若 T008 显示外提收益已达 SC-001（≥20%）且无回归, MVP 即可交付; US3/US4 作为增量

### Incremental Delivery

1. US1+US2 = MVP（最大杠杆, 最小风险）
2. US3+US4 = 增量（预取隐藏 ~0.3ms/op 的 w_pp 搬运, 收益小但完整）
3. R2（tile 软件流水）按 T022 决策点决定是否另立

### 风险与回退

- T007 位级门 FAIL → 回退外提, 保留基线, research.md 记根因（跨 tile mxmem:deep 排空语义）
- 任何阶段设备门不可跑 → 停在该 checkpoint, 不带着未验证改动前进
- oplist_exec.c 并发冲突 → 以 [[gehtp-cross-session-coordination]] roster 裁决

---

## Notes

- 设备门是每 phase 的硬 checkpoint, 不是可选项
- 位级一致优先于一切性能收益（FR-004）
- 单 op 微 blob（mm32 1024/1024）是最快的数值门, 先于全链跑

---

## 012 收尾裁决记录 (2026-10-13, 全任务勾落依据)

- T009-T011 (US2): examples 全量 321 pass/9 FAIL, 9 FAIL 经无 hoist 干净库复跑逐一复现 = **全预存**(37_conv2d_add×5/39_probe/40_qwen_layer3×2/41_qwen_layer0, emit 侧 TEMPOFF 槽短; 011 记忆 10 fail 同族), 012 零新增。E2E 双门 = pp blob cos 0.999930/top1 32/32, 输出对 ctrl 位级同; 与 011 gdn_default 锚(2.76s/0.999930)独立互证。
- T012-T016 (US3): **整体 NO-GO(数据裁决)**。pp 生效后串行 DMA 全程 Σ=22.6ms = 0.82% wall(收益上限); 大 FC(attn_qkv wgt 12.6MB)乒乓双缓冲 25MB > VTCM 16MB 物理不可行; dc_dma_once 已无 700µs 固定开销痕迹(22.6ms≈纯传输 66.5GB/s)。submit/wait 拆分与前瞻接线失去收益对象, 不实施。w_pp 自带 pp_fb 串行回落(dev=1+stg_fb)已覆盖回落语义。
- T017-T019 (US4): 预取未实施 ⇒ 无前瞻失效/乒乓路径可回归; 单 matmul 零额外延迟(5.7ms/op); 42_gehtp_persistent PASS; w_pp per-op dc_dma init/destroy 无跨会话态。
- T022: R2(tile 软件流水) NO-GO — 装载非瓶颈(531µs/tile=标量 pack, 已由 pp 消除); 且 ISA 包配对规则封死 act 跨列复用。R3(异步预取) NO-GO — 见 US3 段。
