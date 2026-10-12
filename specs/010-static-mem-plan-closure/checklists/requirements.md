# Specification Quality Checklist: 静态内存规划收口

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-10-10
**Last Updated**: 2026-10-12（并入 crouton-pack 范围 + 重排优先级 + 架构修正后复审）
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- **项目惯例例外**：本项目（GEHTP 编译器/执行器）的 spec 惯例是技术规格（对照 `specs/008`、`specs/009`），
  会引用文件路径、opcode、行号等实现细节——这些是特性域本身，不是「泄漏」。上表
  「No implementation details」「Written for non-technical stakeholders」两项按项目惯例
  视为 N/A，不计为缺陷。
- **验证结论（10-12 复审）**：无 [NEEDS CLARIFICATION] 标记。收口点由 4 扩为 5
  （A/B/C/D + 新增 E「权重 crouton pack 编译期化」），全部 FR 可测、SC 可量化
  （cos ≥ 0.998642 / wall ≤ 6.31s / 35+261 门 / 65536 bit-exact / pack 归零）。
- **本轮修正（并入用户代码分析结论）**：
  1. **D 线架构假设改写**：cp_solver 的对家是贪心 FancyAllocator（同 `vtcm_allocations_`
     契约），非 mem_plan（DDR 布局）；pin/restage 时序已存在于 cp_solver env 旁路，
     本特性是「扶正为默认 emit 主线」而非「二选一」。
  2. **量化口径警示**：20.6s pack 是标量旧账，T017 LUT 已优化，SC-004 改为按重测基线归零。
  3. **精度不漂移**：A–E 均为布局/搬运决策，cos 应 bit-identical，漂移属 bug。
- **待 /speckit-plan 落地的开放项**：pin_cap 的 blob 载体（reserve 字段已占，需新槽/扩字段）、
  carve 复用标志载体（扩 24B 或新注解槽）、E 的 crouton 权重槽契约（与 `repack_q4_0_tiles`
  逐位对齐）、D 的 cp_solver 时序扶正为 emit 主线的接线点。
