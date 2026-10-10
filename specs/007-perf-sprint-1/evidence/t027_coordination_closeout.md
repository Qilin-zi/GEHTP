# T027 — 协调收尾（2026-10-10）

## 结论：PASS（登记完成，006 改名留用户确认）

## 1. 跨会话协调表登记（memory `gehtp-cross-session-coordination`）

- 007 feature 清单行更新：**T001-T027 全勾**（末笔 `f351478`），三 US 结局 +
  最终锚 + 冲刺总门未达诚实结论。
- 设备库指纹登记：**libhvxhmx_007.so 重推（LUT 版，板上指纹 6ebf39fa，2026-10-10）**，
  007d blob `qwen35_08b_q4_007d.wtop`（476.5MB）在板 `gehtp_007/`。
- 共享基线未回退：f16 锚 wall 24.24s / cos 0.998642 位级（US1）；Q4 锚 exec 25.7s
  / 权重 476.5MB / cos 0.993484（q4_0 口径）。

## 2. 结论写回 memory

- 新建 `gehtp-007-sprint1-closeout.md`（三 US 结局 + 最终锚 + 不可重蹈判断 +
  遗留项），MEMORY.md 索引已加一行。
- 关联既有 [[gehtp-007-us1-broadcast-hvx]] [[gehtp-007-t016-q4-gate]]。

## 3. 006 目录改名协调（待用户确认，未擅自执行）

- 主树 `/disk1/gehtp/specs/006-perf-sprint-1/`（**untracked**）名不副实：其内容为
  **激活融合普查**（NO-GO 归档，`census-report.md` + spec/plan 组），而
  "perf-sprint-1" 名已被 007（broadcast HVX / W4A16 / 融合准入）实际占用。
- **建议改名** `specs/006-perf-sprint-1/` → `specs/006-activation-fusion-census/`，
  与归档内容一致，消除与 007 的撞名歧义。
- 因该目录属归档 006 会话、位于主树 untracked 区（多会话共享），按协调协议
  **不擅自改**，留待用户/006 会话确认后执行。

## 4. 冲刺线终态速查

| 指标 | 值 | 门 | 判 |
|---|---|---|---|
| f16 wall | 24.24s | ≤28s (SC-002 阶段门) | PASS |
| Q4 exec | 25.7s | — | — |
| 权重 | 476.5MB | ≤0.6GB (SC-003) | PASS |
| broadcast 回落 | 0 | 0 (SC-001) | PASS |
| 总门 wall | 27.4s | ≤20s (SC-002 总门) | FAIL（诚实，根因 US3 NO-GO + Q4 标量 dequant） |
| op 数 | 14188 | ≤9000 (参照) | FAIL |
| ctest | 48/48 | 47/47 | PASS |
