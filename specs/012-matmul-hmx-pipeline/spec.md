# Feature Specification: MatMul HMX 流水线（tile 级计算流水 + op 级权重预取）

**Feature Branch**: `012-matmul-hmx-pipeline`

**Created**: 2026-10-13

**Status**: Draft

**Input**: User description: "把 f16 matmul 从逐 op/逐 tile 串行升级为流水：tile 级重叠装载与 MAC、消逐 tile HMX 握手（主）；op 级预取下一个权重搬运（次）。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Tile 级计算流水化 (Priority: P1)

在 prefill（M=32）里，一个大 FC matmul 的 GEMM 走查被拆成 192 个 32×32 tile，每个 tile 都做一次完整的 HMX 上电/握手/清零累加器/读出/断电。这 192 次串行握手是单个 matmul ~100ms 的主体（真计算仅 ~0.03ms）。执行器应把整段 tile 走查作为**一条连续的 HMX 流**：一次上电、tile 间装载与累加重叠、一次断电，消除逐 tile 的串行握手。

**Why this priority**: matmul+FC 约占 prefill 墙钟 88%，而每个 matmul 的 ~100ms 里主体是逐 tile 握手串行，不是搬运也不是 MAC。这是单一最大杠杆。

**Independent Test**: 跑 0.8B prefill blob，对比改动前后逐 op 时间 profile 与墙钟；单个 attn_qkv 应从 ~102ms 显著下降，输出位级一致。

**Acceptance Scenarios**:

1. **Given** 一个 M=32 的 matmul（n_row=1、n_col=192），**When** 它执行，**Then** 整段走查只做一次 HMX acquire/release，而非 192 次。
2. **Given** tile 走查流水化，**When** tile c+1 的数据装载进行时，**Then** tile c 的累加仍在进行（装载与 MAC 重叠，MAC 单元不空转）。
3. **Given** 所有 tile 完成，**When** 累加器读出，**Then** 每个 tile 的部分和与串行逐 tile 执行逐位一致。

### User Story 2 - 数值与正确性不因流水化而变 (Priority: P1)

流水化把「逐 tile 清零累加器」改成了「连续流中的边界复位」，稍有不慎就会串列污染。流水化后的输出必须与串行逐 tile 执行逐位一致，不得引入部分和错位或 bias 折入错误。

**Why this priority**: 提速若以结果损坏为代价毫无意义，这是硬门。

**Independent Test**: 跑设备 golden 对拍与全量 op 套件，cos / top-1 与逐位输出必须与基线一致。

**Acceptance Scenarios**:

1. **Given** 连续流执行，**When** 每个 tile 的累加开始，**Then** 累加器在该 tile 边界正确复位（无跨 tile 残留）。
2. **Given** 部分和累积，**When** 最终读出，**Then** 累加顺序的改变不改变每个输出元素的值（浮点累加在文档化容差内位级一致）。
3. **Given** 输出写回，**When** bias 折入与 unpack 执行，**Then** 结果与逐 tile 版本位级一致。

### User Story 3 - op 级权重预取（次） (Priority: P2)

在 tile 流水之外，执行器还应在当前 matmul 计算时提前启动下一个 matmul 的权重搬运（DDR→片内暂存），把搬运时间藏进计算里。

**Why this priority**: 权重搬运在 w_pp（预打包 DMA）下约 0.3ms、标量散写下约 40ms，比 tile 握手小一个量级，故为次目标；但在未预打包路径仍有价值。

**Independent Test**: 跑含多 matmul 的 blob，验证当前 matmul 计算期间已发起下一 matmul 的权重搬运，且总墙钟中搬运停顿消失。

**Acceptance Scenarios**:

1. **Given** 一串连续 matmul，**When** 执行器计算第 N 个，**Then** 它已启动第 N+1 个的权重搬运。
2. **Given** 两个相邻 matmul，**When** 下一权重搬运与当前计算并行，**Then** 两者使用不同暂存区（无「用中覆盖」）。
3. **Given** 下一个 op 不是 matmul，**When** 执行器前瞻，**Then** 回落串行搬运而不报错。

### User Story 4 - 无重叠机会时零回归 (Priority: P2)

当没有下一个可预取的 matmul、或 matmul 只有单 tile 时，执行器必须与今天行为一致，不增开销、不改行为。

**Why this priority**: 防止优化在简单/边界场景里造成回归。

**Independent Test**: 跑单 matmul、单 tile、非 matmul 微 blob，验证时序与行为相对基线一致。

**Acceptance Scenarios**:

1. **Given** 一个单 matmul 无前瞻，**When** 它运行，**Then** 它做一次串行搬运（无双缓冲、无额外延迟）。
2. **Given** 一个 matmul 后紧跟非 matmul，**When** 前瞻，**Then** 只为该 matmul 发起一次串行搬运。

### Edge Cases

- 当「下一个权重」大于单个暂存区时怎么办？（回落串行或拆分搬运）
- 当双缓冲总占用超出片内预算时怎么办？（16MB 预算，lm_head 单权重约 16.76MB 已超）——不得超预算，退化单缓冲/串行。
- 当当前 op 与下一 op 权重槽被共享/别名时怎么办？（不得重复搬运或互相覆盖）
- 当前瞻到的 matmul 走 staging 回退路径时怎么办？（不得预取进随后被回退路径覆盖的缓冲区）
- 当 matmul 的 n_col=1（单 tile）或 n_row×n_col 极小，流水化收益为零时怎么办？（直接走现有逐 tile 路径，零扰动）
- 在 blob 边界（首 op / 末 op / 跨执行会话）怎么办？（流水与预取状态干净复位）

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 一个 matmul 的 GEMM tile 走查 MUST 作为一次连续的 HMX 流执行（acquire 一次、release 一次），而非每 32×32 tile 一次完整的 acquire/reset/readout/release 循环。
- **FR-002**: tile 之间的数据装载（activation/weight 进入 MAC 单元）MUST 与前一个 tile 的累加重叠（流水），使 MAC 单元不因等待装载而空转。
- **FR-003**: 流水化下每个 tile 的累加器 MUST 在该 tile 边界正确复位，部分和 MUST 正确累积，不得出现跨 tile 残留或错位。
- **FR-004**: 流水化后的输出（含 bias 折入与 unpack）MUST 与串行逐 tile 执行位级一致。
- **FR-005**: 执行器 MUST 在当前 matmul 计算尚未结束时启动下一个 matmul 的权重搬运，使搬运与计算重叠。
- **FR-006**: 权重预取 MUST 使用至少两个相互独立的暂存区，避免搬入覆盖计算正在读的权重；并用完成屏障保证权重搬运完成且可见后才允许读取它的计算开始（杜绝 use-before-ready）。
- **FR-007**: 当下一个 op 不是可预取的 matmul、或暂存区占用会超出片内预算时，执行器 MUST 回落为串行搬运。
- **FR-008**: 流水与预取状态（暂存区、搬运描述符、前瞻记录、累加器流水段）MUST 在每个执行会话开始及 blob 边界处干净复位。

### Key Entities *(include if feature involves data)*

- **HMX 流水段（HMX pipeline stage）**: 连续流中一个 tile 的装载→累加→读出阶段；相邻段的装载与累加重叠。
- **累加器边界复位（Accumulator boundary reset）**: 每个 tile 累加开始前的清零点，保证跨 tile 无残留。
- **权重搬运（Weight transfer）**: 把权重张量从片外搬进片内暂存区的动作；具有大小、源、目标暂存区与完成状态。
- **暂存区（Staging buffer）**: 计算期间持有权重的片内区域；至少两个交替（乒乓）使用。
- **预取描述符 / 前瞻记录（Prefetch descriptor / lookahead）**: 对「下一个 matmul 权重及其搬运」的记录，用于提前发起与使用前等待完成。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 0.8B 模型 prefill 墙钟显著下降（目标 ≥20%），单个大 FC matmul 的逐 tile 握手串行被消除（逐 op 时间 profile 中 matmul 段大幅收缩），以墙钟与逐 op profile 验证。
- **SC-002**: 设备输出保真度不变：cosine 相似度与 top-1 命中与串行基线一致（满足项目精度门，如 cos ≥ 0.9986、top-1 32/32）。
- **SC-003**: 全量 op 套件（设备 example）在流水化执行下保持全绿——无累加器污染、无数据竞争崩溃、无 use-before-ready 故障。
- **SC-004**: 单 matmul、单 tile、非 matmul 边界场景运行无可测回归（≤ 基线 + 小容差）。

## Assumptions

- 目标负载为 Qwen3.5-0.8B；权重占用约 2GB（f16）/ 476MB（Q4_0），片内暂存预算 16MB——因此权重预取是逐 op 级别，非整模型。
- 主目标为 prefill 的 HMX 路径（M≥32，tile 走查 + 权重 pack/DMA）；decode 的 GEMV 路径（M<32，权重流式读取）为次目标，可复用重叠原理但结构不同。
- 权重搬运在 w_pp（预打包 DMA）下约 0.3ms、标量散写下约 40ms，均远小于逐 tile 握手串行（~100ms），故 tile 流水为主、op 预取为次。
- 「双域执行」（HTP0+HTP1）是独立的并行杠杆，不在本特性范围内。
- 执行器持有完整静态 op 列表，可做逐 op 前瞻而无需运行时探测。
