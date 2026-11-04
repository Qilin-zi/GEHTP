# Specification Quality Checklist: 020 decode 静态 DMA 调度

**Purpose**: 立项前规格完整度自检（本线为技术 spec，非业务 spec，部分项 N/A）
**Created**: 2026-11-04
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] 聚焦技术问题与方案（问题/方案/门/非目标，018 同款风格）
- [x] 全部必填段已填（问题/资产/方案/数值声明/门/非目标）
- [ ] N/A — 「非技术利益相关者」视角不适用于本内部性能基建 spec

## Requirement Completeness

- [x] 无 [NEEDS CLARIFICATION] 遗留（3 个待澄清已显式列在 spec 末，进 /speckit-clarify）
- [x] 门可测（G1 零回归 / G2 精度 cos / G3 性能 ~4% / G4 2 NSP 吞吐）
- [x] 成功标准可度量（37.9→~36.5ms，cos≥0.9999，吞吐~2×）
- [x] 边界清晰（非目标 4 项：批处理本身 / 非 matmul / prefill / 4B 40-bit）
- [x] 依赖与假设已列（前置 018；north-star=批处理前件）

## Feature Readiness

- [x] 方案落在编译期固定（footer 契约 + 执行器两层拆分）
- [x] north-star 已写进 spec（批处理 M>1 依赖，8–16×）
- [x] 诚实标注单序列收益 ~4%，不夸大

## Notes

- 019 号已被 `019-4b-decode-40bit-stream` 占用，本线顺延 **020**。
- 3 个待澄清（双缓冲粒度 / footer 覆盖面 / 2 NSP footer 生成方式）需在 /speckit-plan 前定。
