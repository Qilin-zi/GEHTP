# Data Model: 静态内存规划收口

本特性的数据面 = 编译器 → 执行器的 wtop blob（v3）扩展。五个收口点各对应一个（组）实体。字段布局的字节级定义见 [contracts/blob-v3-extensions.md](contracts/blob-v3-extensions.md)，本文只列实体语义、关系与校验规则。

## 实体

### 1. W4A16 辅助面标注（AuxSurfaceAnnotation）

- **归属**: 每个 `OP_MATMUL_W4A16` op 一条。
- **字段**: `bias_slot: u32`、`atbl_slot: u32`、`otbl_slot: u32`（三个 slot id）。
- **关系**: 引用 `slot` 表；三个 id 互不相同且都指向 weight 区有效槽。
- **校验规则**:
  - `atbl_slot.len == 8*(K/32)*4`，`otbl_slot.len == 8*(N/32)*4`，`bias_slot.len == (N/32)*512`（与 `w4_carve_need` 逐项对齐）。
  - 三 id 均在 `[0, n_slots)`，且非 EXT_IN/EXT_OUT 特殊槽。
- **取代**: 执行器原按 `len` 反推逻辑（`oplist_exec.c:562-571`）删除。

### 2. 内存配置（MemoryConfig，TEMPOFF 槽）

- **归属**: 全 blob 一条，挂 `WT_SLOT_TEMPOFF` 槽。
- **字段**: `cap: u32`（静态 temp 表容量）、`reserve: u32`（低 16 位 = 表外 bump 预留 KB | 高 16 位 = VTCM 池大小 KB）、`pin_cap: u32`（**新增**，权重常驻区字节）、`n: u32`、`records[n]`（每项 `{temp_id, offset, size}`）。
- **关系**: `pin_cap` 是 `vtcm_auto_scan` 算 pool 的独立输入：`pool = 16MB − scratch_max − pin_cap`。
- **校验规则**:
  - `pin_cap` 为 2048 对齐（`wtcache_open` 内 `round_up(pin_cap, WTC_VTCM_ALIGN)`）。
  - `pin_cap + pool + scratch_max ≤ 16MB`；超界编译期显式降级（关池），不静默。
  - 无驻留规划（关池口径）时 `pin_cap = 0` → 执行器回退旧硬编码 4096 逐字节一致。
- **状态**: 无运行期迁移；单次 `wt_exec_run` 内不可变。

### 3. carve 复用标志（CarveReuseFlag）

- **归属**: 每 op 一条，挂在引擎注解 footer `op_annot` 的 bit5。
- **字段**: `reuse: 1 bit`（bit5，其余 bit0-4 引擎/回落语义不变，bit6-31 保留）。
- **关系**: 与 `carve footer`（逐 op 的 `{op_index, vtcm_offset, size_bytes, residency_start, residency_end}`）协同 —— `reuse=1` 仅当「形状(M/K/N)不变 且 live_top 不变」时由编译期置位。
- **校验规则**:
  - `reuse=1` 的 op 其 M/K/N 必与上一 matmul 同；否则编译期自校验报错。
  - 首个 matmul 或形状切换的 op，`reuse=0`。
- **取代**: 执行器 `shape_same` 运行时判断（`oplist_exec.c:589-591`）删除。

### 4. 权重 pin/restage 计划（WeightRestagePlan）

- **归属**: 全图一份，编译期算出（**新建规划面**，greedy 与 cp_solver 都 skip const 权重），序列化为显式 `OP_PIN` + `OP_DMA` 序列（或等价注解）。
- **字段**: 每权重 slot 一条：`pin: bool`（是否常驻）、`restage_points: [op_index]`（需换入的点位）。
- **关系**: 依赖 `pin_cap`（实体 2）决定哪些权重放得进常驻区；序列与 op 流并行下发。
- **校验规则**:
  - 同权重被多个连续 matmul 消费 → 只 pin 一次，`restage 次数 == 编译期规划值`。
  - 超 `pin_cap` 的权重不进常驻区，换入换出按序列执行，越界显式降级。
  - `wt_w3_report` 的 restage 统计可与规划值对账。
- **取代**: 执行器每 op 无条件 `cpu_to_vtcm(e.wt/e.bias)`（`oplist_exec.c:657-658`）删除。

### 5. 预打包 crouton 权重槽（CroutonWeightSlot）

- **归属**: 每个 conv2d f16 / Q4_0 权重 slot 一条，编译期打包进 blob（替代运行时 pack）。**f16 matmul 不在本实体**（已由 008 T012 的 `kFlagWgtPP` + `w_pp` DMA 预取收口）。
- **字段**: `layout: enum {FP16_RAW, FP16_CROUTON(conv2d), Q4_0_CROUTON}`、`tile_bytes: u32`（每 32×32 tile 的字节：f16 crouton = 2048，Q4_0 crouton = 640）、`n_tiles: u32`（tile 数）。
- **关系**: 与权重 slot 数据同址；`hmx_crouton_pos`（`kernels/include/hmx_crouton.h`）是布局公式单一真值源。
- **校验规则**:
  - 编译期 pack 输出与运行期参考实现逐位一致（f16 conv 参考 `oplist_exec.c:888-895`、Q4_0 参考 `:2033+` Route A）。
  - **Q4_0 内存守恒**：pack 发生在量化态（640B/tile），不得反量化成 f16 存（否则 4× 膨胀，破坏 4.3× 内存收益）。
  - `tile_bytes` 与 `hmx_crouton_pos` 覆盖的 tile 数一致，缺 tile 编译期报错。
- **取代**: 执行器 `exec_conv2d_hmx` 的 f16 conv pack 循环 + Q4_0 Route A 反量化 pack 删除，改直接 mxmem 加载。

## 实体关系图（文本）

```
MemoryConfig ──pin_cap──> 决定 WeightRestagePlan 的常驻预算
AuxSurfaceAnnotation ──引用──> slot 表（bias/atbl/otbl 三个 id）
CarveReuseFlag ──协同──> carve footer（reuse 仅当形状+live_top 不变）
WeightRestagePlan ──依赖──> MemoryConfig.pin_cap
CroutonWeightSlot ──替换──> 运行时 pack（conv2d f16/Q4_0 权重槽的编译期重排）
```

## 状态转换

- **WeightRestagePlan**：权重在 `未驻留 → 已 pin 常驻 →（超预算时）换出 → 换入` 间迁移，迁移点全部由编译期序列决定，执行器零自主决策。
- **CroutonWeightSlot**：无运行期状态；布局在编译期一次性定型，运行期只读。
