# Feature Specification: 全图 VTCM 驻留 + 显式 DMA 流水（中期性能线）

**Created**: 2026-10-09
**Status**: Draft
**Input**: 用户裁决立项（ggml 对比分析会话：差距不在「机制缺失」而在「已实现机制未全图兑现」）。承接 007 收口锚（≤20s），目标 **≤8s**；ggml 终态参照（激活常驻 VTCM + 权重流预取 + 带宽极限态）。

## Background *(依据与盘子)*

对标差距定量（同板 d0f1784，2026-10-08/09 实测）：ggml cooked plan = **241 VTCM 槽 / 3.23MB / spill=0**（全激活常驻）+ MBUF 2GB 权重按执行序预取 + decode 带宽极限态 16.7GB/s；GEHTP v4（31.78s op_ts）= `Σdma_us=0`（零异步搬运）、激活每 op DDR 往返（三分账「搬」40.5%）、carve 只盖 W4A16（v4 纯 f16 **零命中**）。

**已实现未兑现盘点**（本 spec 的存在理由——把机制变成性能）：

| 机制 | 现状实查 | 缺口 |
|---|---|---|
| 003 carve 表 + 按表分配（T012-T014/T016） | w4v3 合成 blob 验收全绿 | 只接 W4A16 路径；f16/激活不消费 |
| OP_DMA 显式搬移（T015） | `exec_dma` VTCM 侧解析已支持，arity 8 | 发射面从未扩：无 tile 级流水、无预取 |
| fancy_allocator + cp_solver lifetime 打包 | 只服务 carve | 未服务全图激活驻留 |
| TILING 槽（WT_SLOT_TILING） | 落地 bit-identical | 纯重构零行为（chunk 决策搬家，搬运循环仍在 per-opcode exec 逐趟跑） |
| `[vtcm]` 旗标 | v4 有 1107 op 已标 | 旗标≠驻留事实（prof 实测激活仍 DDR 往返） |

**盘子**（v4 口径 31.8s；007 收口后投影）：

| 项 | v4 实测 | 007 后投影 | 本线目标 |
|---|---|---|---|
| 残余搬运族（STRIDED_SLICE 3.06s + UNARY-reshape 3.05s + TRANSPOSE 1.29s + broadcast 残余 ≤1s） | ~8.4s | **~7-8s** | ≤2s（US3） |
| 激活 DDR 往返（搬 40.5% wall 中非 broadcast 部分） | ~13s | ~5-6s | 驻留消化（US1） |
| HMX 权重搬运零重叠（lm_head chunk 132ms ≈ 3.9 GFLOPS = 0.03% 峰值，DMA 串行于计算） | 计入 matmul 族 | 计入 W4 fleet | DMA 藏进计算（US2） |
| per-op 裁决 + op_ts 记录 | ~1.4s | ~1.4s | ≤0.5s（US4） |

约束：003 契约不破（偏差报告不变量 / carve footer / ver1 legacy 逐字节 / blob-v3 五场景）；精度锚不回归（f16 期位级、W4 期沿 007 重锚的 q4_0 golden 口径）；**绝不静默**（spill/降级必须显式名单，003 FR-004 纪律延伸）。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 全图激活 VTCM 驻留表 (Priority: P1)

编译期对全图 temp 做 lifetime 分析（`graph_deps` 活性已有）+ 区间打包（`fancy_allocator`/`cp_solver` 复用），产出**驻留槽表** `{temp_id, vtcm_offset, size, live 窗口}`，与 carve 面、权重 staging 三方预算统一校验（`Σ激活驻留 + Σ并发 carve + staging ≤ 16MB`），超限 temp 进**显式 spill 名单**（落 DDR + 原因）。运行时 temp 物化改查槽表：producer 直写 VTCM 槽、consumer 直读，不再默认 DDR 实体化。纯寻址变化，f16 期逐字节不动。

**Why this priority**: 「搬」类 wall 的根因是激活不驻留；ggml 参照（3.23MB/241 槽/0 spill）证明量级可行；机制件全现成，是「已实现→已兑现」最短路径。且与 W4 供给无依赖，007 延期可先行。

**Independent Test**: emit 输出驻留覆盖率报告（temp 字节加权）+ 预算表 + spill 名单；v4 同图位级对拍（驻留不改浮点语义→逐字节同）；wall 对比（v4 口径先跑出首锚）。

**Acceptance Scenarios**:

1. **Given** 全图 temp lifetime 图，**When** emit，**Then** 槽表无重叠（沿 T016 重叠校验先例）且预算表落盘（驻留/staging/carve/slack 四行）。
2. **Given** 并发工作集超预算的 temp，**When** emit，**Then** 进 spill 名单并列原因（容量冲突/形状异常），**不**静默落 DDR。
3. **Given** 驻留 temp 的 producer/consumer op，**When** 执行，**Then** 中间不落 DDR（prof dma/bytes 覆盖可证），偏差报告不变量不破。
4. **Given** ver≤2 legacy blob，**When** 运行，**Then** 逐字节同（驻留为 ver=3+ 配方，legacy 不受影响）。

---

### User Story 2 - tile 级 DMA 双缓冲 + 权重按执行序预取 (Priority: P2)

OP_DMA 发射面扩展：HMX 大算子（in_proj/out_proj/lm_head chunk 族）的 tile i+1 权重/激活预取与 tile i 计算重叠（两 staging 槽轮换；`act+wgt+out < VTCM/2` 约束沿 crouton-major 设计 §4）；权重流按执行序提前发射（ggml MBUF 最小版）。只改搬运时序不改数值。**前置**：fence.c HMX+DDR 并发语义过 micro-test（003 移交单已知项，DMA 流水化硬前置）。

**Why this priority**: 007 W4 化后 matmul fleet 的下一堵墙是搬运零重叠（lm_head chunk 0.03% 峰值的主因之一）；依赖 US1 的槽表与预算，排 P2。

**Independent Test**: op_ts `dma_us > 0` 且 HMX 族 wall 对比（07 锚口径）；cos 位级不动；fence micro-test（HMX 计算与 DMA 并发往返一致性，100 次循环无竞态）。

**Acceptance Scenarios**:

1. **Given** 带 tile 划分的 HMX 算子，**When** 执行，**Then** tile i 计算期 tile i+1 搬运在途（op_ts 双时间轴可证），任一环失败显式回落串行并记偏差。
2. **Given** 权重执行序，**When** emit，**Then** 预取发射不早于槽表就绪窗口（不与在驻留激活争预算）。
3. **Given** fence 语义未过 micro-test，**When** P2 排期，**Then** 阻塞（前置门，不得带病上流水）。

---

### User Story 3 - 残余搬运 op 消化（census 驱动） (Priority: P3)

US1 驻留后重跑 census：部分 STRIDED_SLICE/UNARY-reshape/TRANSPOSE 变**寻址问题**（驻留面零拷贝消除，emit 期折叠引用）；剩余物理搬运 HVX 化（`hvx_copy_f16_uu`/分段 memcpy 先例，007 P1 broadcast 范式复用）。逐族决策留证据（消除/HVX/保留+原因）。

**Why this priority**: 盘子 ~7-8s 是第二大头，但收益部分被 US1 吸收（驻留后消失），须 census 后择优，排 P3。

**Independent Test**: 三族合计 wall 门 ≤2s；消除类零语义变化（位级）；HVX 类沿 007 P1 对拍范式。

**Acceptance Scenarios**:

1. **Given** 驻留槽上的纯 reshape/slice（不改字节序），**When** emit，**Then** 折叠为引用消除（op 消失，opt-stats 计数）。
2. **Given** 物理搬运不可避免的族，**When** 执行，**Then** HVX 块拷贝 + 尾部标量，计划=实际引擎。
3. **Given** 消化后仍 >2s，**When** 归因，**Then** census digest 落盘指认残余根因（不硬凑门）。

### User Story 4 - per-op 开销批量化 (Priority: P4)

op_ts 记录批写/抽样模式（默认全量保留，`--op-ts-sample` 旋钮可关）+ 逐 op 裁决/栅栏合并（ver=3 计划已静态，逐 op 重裁决冗余）。

**Why this priority**: ~1.4s 固定税，机制简单但收益垫底，排 P4。

**Independent Test**: 抽样模式 op_ts 完整性契约（抽样统计口径注明）；全量模式位级同现行为；wall 对比。

**Acceptance Scenarios**:

1. **Given** 性能采集场景，**When** 抽样开启，**Then** op 数/族耗时统计口径显式标注（防误读 006 census 类对拍）。
2. **Given** 默认模式，**When** 运行，**Then** 行为与现库一致（幂等可关）。

### Edge Cases

- **004 decode 线 reconcile**：`oplist_exec.c` temp 物化段与 004 状态表/KV 段同文件——改前 `git status`，回合时 reconcile。
- **007 部分兑现**：f16 兜底层存在时权重 staging 预算按 f16 口径（K·N·2）重算，预算表编译期自适应双口径。
- **三方预算冲突**（驻留+staging+carve > 16MB）：编译期报（OOM 提前，003 carve-compiletime 先例），不运行时 `init_vtcm_pool` 返 NULL。
- **US1↔US3 交互**：驻留改变 census 基线——US3 的盘子以 US1 落地后 census 为准，spec 表内投影值仅立项参考。
- **fence 竞态**：DMA 流水化前必须过 fence.c micro-test（V81HexSim 不能背书 fence/DMA 竞态，已知边界）。
- **spill 判读**：spill 名单 ≠ 失败（合法降级），但必须显式（与 007 P2 f16 兜底名单同纪律）。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 编译期 MUST 产出全图激活驻留槽表（lifetime 区间打包，槽区间不相交），并落盘预算表（激活驻留 Σ / 权重 staging / carve 峰值 / slack 四行）；预算超限时 MUST emit 期显式报错，禁止运行时才发现。
- **FR-002**: 超预算 temp MUST 进显式 spill 名单（含原因），运行时沿名单落 DDR；spill 是合法降级但不可静默。
- **FR-003**: 驻留 temp 的物化 MUST 查槽表直读写（producer/consumer 不经 DDR 中转）；legacy（ver≤2）blob 行为逐字节不变。
- **FR-004**: OP_DMA 发射面 MUST 支持 tile 级双缓冲（staging 槽轮换）与权重按执行序预取；重叠失败 MUST 显式回落串行并记偏差（沿 003「允许回落+上报」契约）。
- **FR-005**: DMA 重叠正确性 MUST 过 fence micro-test（HMX 计算与 DMA 并发一致性循环验证）后方可启用流水；未过门前 P2 不得上板。
- **FR-006**: 残余搬运族消化 MUST census 驱动留证据（消除/HVX/保留+原因 per 族）；消除类位级一致，HVX 类计划=实际。
- **FR-007**: op_ts 抽样模式 MUST 显式标注统计口径；默认全量模式行为与现库一致（幂等可关）。
- **FR-008**: 003 契约 MUST 不破：偏差报告不变量、carve footer 对拍、ver1 legacy 逐字节、blob-v3 五场景验收全绿。
- **FR-009**: 共享文件改动 MUST 遵守跨会话协调协议（git status 前置 / 精确暂存 / 板上私有名与互斥 / 004 reconcile 段登记）。

### Key Entities

- **ResidencySlot**: `{temp_id, vtcm_offset, size_bytes, live_start_op, live_end_op, spill, spill_reason}`。
- **BudgetReport**: emit 落盘四行预算 + spill 名单 + 驻留覆盖率（temp 字节加权）。
- **DmaStreamDesc**: `{op_idx, tile_idx, direction, src_ref, dst_ref, buf_slot}`（双缓冲轮换槽 0/1）。
- **CensusDigest**: 残余搬运族 per-opcode `{n, wall, 处置(消除/HVX/保留), 证据指针}`。
- **口径双列**: f16 期（位级）与 W4 期（q4_0 golden 门，沿 007 重锚值）判分登记。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 激活驻留覆盖率（temp 字节加权）≥ **80%**；spill 名单显式；预算表落盘。
- **SC-002**: HMX 大算子族 wall（lm_head chunk + 线性层）：DMA/计算重叠后 ≤ 007 锚的 **50%**（首跑校准冻结具体门值）。
- **SC-003**: 全链 wall：007 锚（≤20s）→ **≤8s**（总门；007 未全兑现时锚回退实测重校，比例门不放松纪律）。
- **SC-004**: 精度不回归：f16 期逐字节不动；W4 期 q4_0 golden 门（007 冻结值）不降。
- **SC-005**: 残余搬运三族合计 ≤ **2s**；per-op 开销 ≤ **0.5s**。
- **SC-006**: 003 blob-v3 五场景验收 + ctest 全绿不回归；fence micro-test 通过证据入库。

## Assumptions

- 007 已收口为本线锚与 W4 供给方；**US1 与 W4 无依赖可先行**（锚回退 v4 35.31s 口径重校门，处置预案已含）。
- 并发激活工作集可装 `16MB − staging − carve`（ggml decode 参照 3.23MB；prefill 32 估算每层 ~1-2MB 短命复用；**精确值 research 阶段以 ddr_offsets/lifetime 实测冻结**——census CSV 的 out_b 列含逻辑展开字节，不可作物理预算）。
- 设备 d0f1784 可用；V81HexSim 可用于带宽/分块数值预演但不背书 fence/DMA 竞态（其已知边界）。
- 不做（长期线另立项）：双 CDSP 域、GDN 整块融合核（对齐 GDNCHFUSE）、f16 权重 tile-major 重排（007 兜底层少量）、supertile 驻留（002 US3 休眠待嫁接）。

## Dependencies

- **上游**：007 收口（锚 + W4A16 全图供给 + broadcast HVX 消回落——US3 census 的干净基线）。
- **同文件协调**：004 decode 线（`oplist_exec.c` temp 物化/KV 状态表段）、003 已收口段（carve/footer 解析，只读复用）。
- **硬前置**：fence.c HMX+DDR 并发 micro-test（US2 唯一外部前置）。
