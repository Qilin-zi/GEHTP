# Contracts: 全图 VTCM 驻留 + 显式 DMA 流水

**Created**: 2026-10-09 | 外部接口 = blob 格式 / emit CLI / 判分报告；项目内库无公网 API

## C1 — blob 契约（ver=3 追加，零破坏）

- TEMPOFF 槽（`0xFFFFFFFD`）为 temp 静态规划**唯一真相源**：DDR 偏移项 + VTCM 驻留项（in_vtcm=1）同表；carve footer 与其一致性校验（wtop_emit.cpp:694-716）语义不变。本线允许**追加**字段/区域（如预算头），禁止改既有字段位序。
- 驻留 temp 的消费引用 = `0x4000|temp_id`（WT_REF_VTCM_FLAG，既有语义）；`vtcm_off_arr` 命中即池驻留，未命中路径（TEMPOFF DDR / bump）优先级与禁扩容规则不变。
- 全局预算门：`resident + staging×2 + carve_peak ≤ vtcm_budget`（默认 16777216）违例 = **emit 期硬报错退出**，错误文本含三方各项数值（可归因，非运行时 NULL）。
- ver≤2 legacy blob：解析与执行逐字节不变（003 五场景之一，不得回归）。
- OP_DMA args 语义（arity 8：{src_ref,dst_ref,bytes,src_off,dst_off,bypass,f_src,f_dst}）不变；双缓冲仅新增发射序与 buf_slot 轮换，不新增 opcode。

## C2 — emit CLI 契约（幂等可关）

| 旗标 | 语义 | 默认 |
|---|---|---|
| `--vtcm-budget N` | 三方预算总口径（字节） | 16777216 |
| `--dma-runlist` | 溢出/预取搬移发真 OP_DMA（既有，本线扩面） | off |
| `--residency-report PATH` | BudgetReport+spill 名单+coverage 落盘 | off（stderr 两行恒开） |
| `--op-ts-sample K` | op_ts 每 K 个采样 + 口径标注（US4） | off=全量 |

不加新旗标时：编译产物与现库同图同 md5（惰性门，沿 002/007 先例）。

## C3 — 精度双门（判分报告格式沿 004/007）

- f16 期：设备输出与同库锚 blob **逐字节一致**（驻留/DMA 均纯寻址与时序变化）；HF 金标门 cos 0.998642 族 / top1 30/32 不回归。
- W4 期（007 P2 交卷后）：q4_0 golden 门沿 007 冻结值，双列登记不得混口径。
- fence micro-test（US2 前置）：HMX 计算与 dc_dma 并发 ≥100 轮循环，输出逐字节一致 + 无挂死；证据（log+设备指纹）入 evidence。

## C4 — 性能门（首跑校准冻结，沿 007 纪律）

SC-001 coverage ≥80%（temp 字节加权，residency-report 口径）；SC-002 HMX 族 wall ≤ 007 锚 50%；SC-003 全链 ≤8s；SC-005 三族搬运 ≤2s + per-op 开销 ≤0.5s；SC-006 ctest + 003 五场景全绿。门值一经首跑校准冻结，后续只许收紧。

## C5 — 跨会话文件契约

`oplist_exec.c`（temp 物化段 ↔ 004 状态表段）、`wtop_emit.cpp`（↔ 007 P2 供给段）、`graph_prepare.cpp`（003 WIP 区）、`op_matmul.cpp`（↔ 007/009）：改前 `git status`，发现他线未提交 diff 停下协调；009 GDN 融合线的块内权重驻留策略消费本线 BudgetReport 口径（对齐窗口 = 009 US3 前）。
