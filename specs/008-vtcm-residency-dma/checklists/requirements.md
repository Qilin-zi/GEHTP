# Specification Quality Checklist: 全图 VTCM 驻留 + 显式 DMA 流水

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-10-09
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] Focused on user value and business needs（性能线：wall 门 + 精度门双口径，沿 003-007 房屋风格）
- [x] All mandatory sections completed（User Scenarios & Testing / Requirements / Success Criteria / Assumptions）
- [x] No implementation details —— **房屋风格例外登记**：本库 spec 传统为技术型（含文件/机制实名，见 003/007 先例），按项目惯例保留；「无实现细节」项按此口径判过

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain（关键不确定项均以 informed default + Assumptions/Dependencies 登记：门值首跑校准冻结、工作集实测冻结、fence 前置）
- [x] Requirements are testable and unambiguous（FR-001~FR-009 均有可验收判据）
- [x] Success criteria are measurable（SC-001~SC-006 量化）
- [x] All acceptance scenarios are defined（US1-US4 各 2-4 条 Given/When/Then）
- [x] Edge cases are identified（6 条：reconcile/双口径预算/编译期 OOM/census 基线漂移/fence 竞态/spill 判读）
- [x] Scope is clearly bounded（Assumptions「不做」清单：双域/GDN 融合核/f16 重排/supertile）
- [x] Dependencies and assumptions identified（007 上游 + 004 同文件 + fence 硬前置）

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows（驻留→流水→消化→开销，依赖序即优先序）
- [x] Feature meets measurable outcomes defined in Success Criteria（总门 ≤8s + 分项门）
- [x] US1 独立可测（与 W4 供给无依赖，锚回退预案在 Assumptions）

## Notes

- 最大技术风险 = 并发激活工作集预算可行性：已在 Assumptions 显式登记（census out_b 列含逻辑展开字节不可用作物理预算，research 阶段以 ddr_offsets 实测冻结），并设编译期校验兜底（FR-001）。
- US2 有唯一外部硬前置（fence.c HMX+DDR micro-test），FR-005 设为阻塞门。
- 与 007 的关系：锚依赖但 US1 可先行；spec 内已写回退预案，无需用户裁决。
- 本 checklist 生成时全项通过（1 次迭代，无 NEEDS CLARIFICATION）。
