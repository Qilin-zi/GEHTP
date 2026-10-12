# Data Model: 全图 VTCM 驻留 + 显式 DMA 流水

**Created**: 2026-10-09 | 依据 [spec.md](spec.md) / [research.md](research.md)

## 实体

### MemPlanEntry（编译期，已存在，本线扩面）

temp/op 级静态分配项。来源 `GraphPrepare::mem_plan()`（`compute_ddr_offsets` 产物）。

| 字段 | 类型 | 语义 | 校验 |
|---|---|---|---|
| temp_id / op_id | u32 | 0.8B 全模型 temp id 至 1403+（MAX_TEMPS=4096） | < MAX_TEMPS |
| offset | u32 | 池内偏移（128B 对齐） | `offset+size ≤ cap`（同池） |
| size_bytes | u32 | 物理字节 | 128 对齐 |
| in_vtcm | bool | 驻留位（=1 → VTCM 池偏移；=0 → DDR 池偏移） | 与消费面 ref 编码一致 |
| live 窗口 | [u32,u32) | [生产 op 执行序, 最后消费者+1) | 活性重叠 ⇒ 偏移不重叠（emit 期硬校验，沿 FancyAllocator 不变式） |

**Spill 项**（`MemSpillEntry`）：`{temp_id, offset, size, reason}`——显式降级名单，非静默。

### BudgetReport（本线新增，emit stderr + evidence 落盘）

| 行 | 语义 | 门 |
|---|---|---|
| resident Σ | 驻留 temp 并发峰值（生命期打包后） | 三方和 ≤ `--vtcm-budget`（默认 16777216） |
| staging Σ×2 | 权重/激活预取双缓冲轮换槽 | 同上 |
| carve 峰值 | 003 carve footer 并发和（已有口径） | 同上 |
| slack | 预算 − 三方和 | ≥0 否则 emit 硬报错 |
| coverage | 驻留 temp 字节加权覆盖率 | ≥80%（SC-001） |

**唯一真相源** = TEMPOFF 槽（carve footer 与之一致性校验沿用 wtop_emit.cpp:694-716，禁止第三套表）。

### DmaStreamDesc（US2，发射侧结构）

| 字段 | 语义 |
|---|---|
| op_idx / tile_idx | 归属计算 op 与 tile 序 |
| src_ref / dst_ref | `0x8000|slot` / `0x4000|temp` / DDR offset（exec_dma 既约） |
| bytes / fence 对 | 沿 OP_DMA arity 8 args 语义（f_src/f_dst ∈ {NONE,FLUSH,INVALIDATE,FLUSH_INVALIDATE}） |
| buf_slot | 轮换槽 0/1（双缓冲） |

不变量：tile i 的 invoke 与 tile i+1 的搬运并发时，两 buf_slot 的区间不相交；失败 → 回落串行 + deviation 记录（003 契约）。

### CensusDigest（US3）

`{opcode, n, wall_us, 处置(消除/HVX/保留), 证据指针}`；以 US1 落地后的重跑 census 为基线（v4 口径仅立项参考）。

### GateRecord（判分登记，沿 004/007 惯例）

f16 期（位级，锚 0.998642 逐字节族）与 W4 期（q4_0 golden，007 冻结值）双列；wall 锚列注库指纹 + 设备（d0f1784）。

## 状态迁移

- temp 物化：`未物化 → 驻留池直写（vtcm_off_arr 命中）→ DDR 静态区（TEMPOFF）→ bump 兜底`（现运行时三径保留，顺序即优先级，禁扩容规则沿现状）。
- 预算校验：emit 期三方和超限 → **硬报错退出**（非运行时 NULL）；spill → 合法降级 + 名单。

## 关系

MemPlanEntry(1)──消费──▶ op ref 编码（0x4000|temp / 裸 temp）；DmaStreamDesc(1)──服务──▶ HMX tile 序；三者共享 BudgetReport 预算约束。
