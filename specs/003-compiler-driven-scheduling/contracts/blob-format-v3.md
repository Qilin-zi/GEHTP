# Contract: Blob 格式 v3（引擎注解 + VTCM carve 表）

**面向接口**：编译器（宿主）→ 运行时（设备）的 `.wtop` blob 线格式。本契约描述 v3 相对 v1/v2 的增量；基础结构见 `kernels/include/oplist_parse.h`。

## 现有版本（勿混淆）

| ver | 含义 | slot 记录 |
|-----|------|-----------|
| 1 | 基础格式 | 16B `{len u32, count u32, offset u32, addr u32}` |
| 2 | 大权重偏移（`weight_area > 0xFFFFFF00`） | 24B `{len,count,offset_lo,offset_hi,addr,reserved}` |
| **3** | **新增：含逐算子引擎注解表（+可选 carve 表）** | 随权重大小 16B 或 24B |

`WT_BLOB_VER=1u`、`WT_BLOB_VER_2=2u`（`oplist_parse.h:42,50`）；v2 由 `wtop_emit.cpp:508-512` 的 `need_v2`（`force_v2 || weight_area>v1_off_max || ext_weight_area>v1_off_max`）决定。

## 版本规则

- `ver <= 2`：无注解表、无 carve 表 → 运行期走**今日自决逻辑**（legacy，FR-009）。
- `ver == 3`：含下述引擎注解表（P1/P3），可选含 carve 表（P2）。

## 引擎注解表（v3 必含）

op 表之后、weight 区 128B 对齐之前的平行数组，`n_ops` 条，每条 4 字节（`u32` 打包）：

| 位 | 字段 | 值 |
|----|------|----|
| 0–1 | engine | 0=scalar, 1=HVX, 2=HMX, 3=reserved（对应 `wt_engine`） |
| 2 | fallback_allowed | 0/1 |
| 3–4 | fallback_engine | 0/1/2（`fallback_allowed=0` 时置 0） |
| 5–31 | 保留 | 0 |

**序列化位置**：`wtop_emit.cpp` 在 op 表写完后（`wtop_emit.cpp:545-554`）、weight 区 128B 对齐前（`:555-557`）写 `n_ops×4B`；`weight_off` 计算随之后移。运行期 `wt_parse` 读 `ver==3` 时在同位置读该表。

**运行期契约**：
- `ver==3` 时，每个算子的实际执行单元 MUST 等于 `engine`（除非走回落）。
- 计划单元不可用时：`fallback_allowed=1` → 用 `fallback_engine` 并记录偏差；`fallback_allowed=0` → 返回非零中止（硬报错）。**不得静默换单元**（FR-004）。
- `fallback_allowed=0` 用于数值敏感算子（FR-008/FR-010）。

## VTCM carve 表（v3 可选，P2 落地）

逐条（仅需片内的 op）定长记录：

| 字段 | 字节 | 说明 |
|------|------|------|
| op_index | u32 | 归属算子 |
| vtcm_offset | u32 | 片内偏移（字节） |
| size_bytes | u32 | 占用字节 |
| residency_start | u32 | 生命周期首 op 序号 |
| residency_end | u32 | 生命周期末 op 序号（含） |

**不变量**：生命周期重叠（区间 `[start,end]` 相交）的任意两记录，`[vtcm_offset, vtcm_offset+size_bytes)` 不相交。

**运行期契约**：运行期 MUST 按记录精确分配（`hmx_runtime_setup` 从"忽略 size 抢全量"改为"按 size 分配"，`hmx_common.c:131-133`），不得超出 `[offset, offset+size)` 写。

## 搬移 op（OP_DMA 扩展面）

opcode 33, arity 8（`WT_ARITY_DMA`）不变。运行期 `exec_dma`（`oplist_exec.c:910-951`）**已完整支持**：src/dst 引用解析含 `0x8000|slot`（DDR）、`0x4000|temp`（VTCM，`:884-892`）、`0xC000|surface`（引擎面，`:863-877`），fence 纪律（`fence_handoff` before/after，`:926-951`）。**本 feature 只扩发射面**：编译器当前只在 `wtop_emit.cpp:416-422/443-449` 发 DDR→DDR spill/fill，需新增 DDR↔VTCM 搬移的发射。

## 向后兼容（FR-009）

- `ver<=2` 旧 blob 在新运行时上 MUST 逐字节同输出（见 `ver<=2` 即走 legacy 路径）。
- `ver==3` 新 blob 在旧运行时上解析器 MUST 报版本错误（`WT_ERR_VER=-3`）而非误读——设备运行时随编译器 lockstep 更新。
