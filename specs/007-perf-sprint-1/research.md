# Research: Short-Term Performance Sprint 1（原 006 冲刺线 2026-10-08 10:39 版复原）

**Feature**: [spec.md](spec.md)
**Restored**: 2026-10-08（撞号裁决后从原 006 目录迁入；内容为当日立项研究原文）

## D1: broadcast 内核形态 — 三形态分派

**Decision**: 按形状结构分派三种执行形态，形态 1/2 走 HVX，形态 3 保留标量。

**Rationale / 现状实查**：
- per-element div 极贵：1622 op × 4.8ms = **7.84s**，纯坐标重算开销。
- **形态 1（连续块循环）**：末维广播 / 整块复制（cond 4096→65536、标量 1→n 类）→ HVX 128B 块拷贝循环（v60），尾部标量，坐标志外仅按块推进。
- **形态 2（行/面重复）**：in 沿低维循环复制 out 行模式 → 外层标量行循环 + 内层 HVX 块拷贝，行内坐标增量预计算。
- **形态 3（通用退化）**：strided gather → 保留标量 = 计划 scalar（合法非回落）。
- **计划对齐**：emit `assign_engine` 类 1/2→HVX、类 3→scalar（fb_allowed 恒 1）；两侧同一形状判定函数，两处注释互指。

**Alternatives**: HVX gather 通用内核（形态 3 罕见、复杂度不成比例，不做）。

## D2: W4 供给链 — 全量匹配 + 口径切换

**Decision**: wtlink 全量表 + 编译期源预检 + emit 显式兜底名单 + q4_0 golden 判分。

**Rationale / 关键点**：
- `gguf_wtlink.py --gguf /disk2/aiDAPTIV/Qwen35-0.8B.gguf --out <repo 内 match.tsv>` 全量表 + **编译期源存在性预检**（/disk2 清理史教训）。
- v4 配方仅追加 `--gguf/--match`（无 CLI 语义变化）。
- emit 报告逐层权重域 + f16 兜底名单（显式降级 FR-003）。
- **精度门切 q4_0 口径**：cos ≥0.9986 同量级 + top1 ≥29/32（首跑校准后冻结）；f16 期中间态仍 hf 口径。
- **形状边界**：W4A16 需 M/K/N 对齐（沿 US1 分块条件），不满足 FC 显式 f16 兜底；lm_head 不动。

**Alternatives**: 自研 PTQ 量化（不做，GGUF Q4_0 直读已有设备验证）。

## D3: 融合准入 — 先统计后放行

**Decision**: 8 条 `kLegacyFusionRules` 逐条 `--enable-pass` 跑 opt-stats 统计命中 → 数值门 → 过门进默认集。

**Rationale / 数值门**：
- 合成图位级一致 + 0.8B 全图 **cos Δ<1e-5** + top1 不降。
- 过门进默认集（`p.enabled` 翻 true），`--no-pass` 仍可关。
- 预期主力 `fusion_matmul_add`（残差/偏置链）。
- 8 条全开仍 >9000 则按 op_ts 热点评估 1-2 条新规则（**GDN 门控链候选**）同一准入门，条件项不阻塞 P1/P2。

**Alternatives**: 全默认开（数值风险一次性引爆不可归因，否）。

## 跨项风险

- **P1↔P3 交互**：融合吸收 broadcast 物化 → P1 验证须 P3 放行后复跑。
- **P2 口径切换**：f16→q4_0 判分双列登记，防误读。
- 共享板/树纪律沿 003；`/data/local/tmp/ggml08b_003/` 属 ggml 对比线勿混。
