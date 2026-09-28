---
description: "Feature spec: VTCM 统一内存规划（matmul 编译期 N 分块 + 单 acquire 四区 arena + supertile 账本接线）"
---

# Feature Specification: VTCM 统一内存规划

**Feature Branch**: `task/prof-wp`
**Created**: 2026-09-28
**Status**: Draft
**Input**: 用户要求「做个规划：supertile 账本接线 + matmul 编译期 N 分块」，并确认 Path U（统一 arena）为方向

## 背景与问题

GEHTP 的内存规划体系（TAG_MEM_PLAN / TEMPOFF，编译期 FancyAllocator 生命期分配）只覆盖**图张量**（有生命期的中间激活）。VTCM 大块所有权不在编译期规划内，三个消费者各自运行时惰性 `HAP_compute_res_acquire` **全量 VTCM** 并 memset 整块：

1. f16 HMX matmul/conv scratch — `hmx_runtime_setup()`（hmx_common.c，调用方传的 2MB 被 `(void)` 掉，恒拿 total）
2. W4A16 wtcache — `wtcache_open()`（wtcache_impl.c，同样全量）
3. `0x4000` 驻留 temp 池 — 从 wtcache 的 VTCM 尾部倒划（oplist_exec.c temp_get 懒初始化）

唯一协调是单向 guard：`exec_matmul_f16_hmx`/`exec_conv2d_hmx` 首行 `if (g_exec.engine_ready || g_exec.vtcm_pool) return -1`。后果（0.8B 实测 wall=370.7s）：

- **互斥损失**：~op70 首个 W4A16 打开 wtcache 后，其后 100+ 个 f16 HMX-eligible matmul 永久回落标量
- **超界损失**：6 个 N4096/K1024 matmul 的 VTCM 需求 8.31MB > 8MB，恒 `need > vtcm_sz` 回落标量
- 两类合计：132 个 HMX-eligible matmul = 322.3s = **86.9% wall**

同时编译期 supertile DP solver（src/vtcm/supertile.cpp）每轮编译都在跑，但结果锁在 `SuperTileSolver` 私有字段里（无 accessor），`GraphPrepare::supertiles_` 从未被填充，下游零消费——QNN 原版里 supertile 正是「驻留张量 + HMX scratch 记同一本账」的地方。

### 已钉死的 ground truth（探查结论，改动设计以此为准）

| # | 结论 | 证据 |
|---|------|------|
| G1 | solver 结果不可达：无 accessor；唯一写 `GraphPrepare::supertiles_` 的 `make_one_supertile` 是死代码 | compiler/src/vtcm/supertile.cpp:188-199, compiler/src/ir/graph_prepare.cpp:4834 |
| G2 | supertile `vtcm_requirement` = 纯 output 张量字节，不含 scratch/input/对齐 | supertile.cpp:236-249 |
| G3 | `compute_ddr_offsets` 的 VTCM 驻留选择 = FancyAllocator 生命期 first-fit，无 supertile 输入 | compiler/src/vtcm/ddr_offsets.cpp:83-107, fancy_allocator.cpp:97-170 |
| G4 | `0x4000` 只在 TEMPOFF 槽 temp-table 打 bit；运行时 temp_get 懒开 wtcache 尾划，无 W4A16 op 也能跑通 | wtop_emit.cpp:482-489, oplist_exec.c:234-253 |
| G5 | f16↔wtcache 单向互斥；`hmx_runtime_teardown` 在 kernels/src 零调用点 | oplist_exec.c:1433-1435 |
| G6 | `plan_vtcm_budget_` 默认 0 → 生产编译从不触发 0x4000 发射；supertile DP 用另一预算 vtcm_size_=3MB | hnnx_compile.cpp:313-314, graph_prepare.cpp:3887-3896 |
| G7 | opcode arity 硬表 `n_args != ar → WT_ERR_ARITY`；扩 matmul 参数必须新增 opcode | oplist_parse.c:122-124 |
| G8 | 4 个历史 must-fix 全未落地：wtcache power_on 非幂等（每次 open 漏 4 个 HAP_power_set 投票）、无 quiesce、hmx_common memset 无 post-flush、f16 matmul 无 N-block | wtcache_impl.c:57, hmx_common.c:169 |

## User Scenarios & Testing

### User Story 1 - 大 N matmul 不再回落标量（Priority: P1）🎯 MVP

编译 0.8B 这类含 N4096/K1024 matmul 的模型时，编译期把超 VTCM 界的 MATMUL_F16 沿 N 维切成若干 chunk op，运行时每个 chunk 都装得下 VTCM，全部走 HMX。

**Why this priority**: 独立可交付、不改所有权模型、风险最低；直接消除 6 个超界 op 的标量回落。

**Independent Test**: 用含 N4096/K1024 matmul 的 blob 编译+上板，optrace 确认这些 op 走 WT_ENG_HMX 而非 scalar；输出与标量参考 SHA 逐字节一致（N 分块不动 K 累加序 → 数值恒等）。

**Acceptance Scenarios**:

1. **Given** N4096/K1024/M 任意 32 倍数的 f16 matmul, **When** 编译期 need(M,K,N) > VTCM 预算, **Then** 发射 ceil(N/N_chunk) 个 `OP_MATMUL_F16_SPLIT` chunk op，每个 need ≤ 预算
2. **Given** chunk 化 blob, **When** 设备执行, **Then** 各 chunk 写共享 out temp 的列子区间 [n0, n0+N_chunk)，最终输出与整 matmul 标量参考逐字节一致
3. **Given** 旧 blob（原 OP_MATMUL_F16 arity 7）, **When** 新 runtime 加载, **Then** 行为与之前完全一致（向后兼容）
4. **Given** 合成 blob 同一 matmul 各编译为整算版与 chunk 版, **When** 设备上两版都强制走 HMX, **Then** 两者输出逐字节一致（chunk-HMX vs 整算-HMX——host 侧 SHA 门两侧都是标量，测不到这一层，必须设备对拍）
5. **Given** 某 chunk 无法走 HMX, **When** 执行, **Then** 整个 op 的所有 chunk 统一走标量（全有或全无，禁止混合路径）

---

### User Story 2 - VTCM 单 acquire 四区共存（Priority: P1）

run 开始（或首次使用）时 exec 层一次 `HAP_compute_res_acquire` 持全量 VTCM + HMX，按编译期下发的 `{pin_size, carve_size, scratch_size, resident_size}` 划四区；f16 HMX 从 scratch 区取工作缓冲（不再 acquire/memset），wtcache 用 pin/carve 区（不再 acquire/memset 全量），驻留 temp 待尾划区。

**Why this priority**: 消除互斥损失（86.9% wall 的大头），且是 US3 的物理前提（驻留 temp 只有在无人再 memset 全量时才可能存活）。

**Independent Test**: 0.8B 全模型上板，op70 之后的 f16 matmul optrace 显示 WT_ENG_HMX（不再回落）；**精度+性能双门**：全模型 cos 与基线一致且 wall 记录对照基线。

**Acceptance Scenarios**:

1. **Given** 同图混合 W4A16 + f16 matmul, **When** 首个 W4A16 打开 wtcache 后, **Then** 后续 f16 matmul 仍走 HMX（从 scratch 区划 chunk），不回落
2. **Given** 单 acquire 已发生, **When** f16 HMX op 到达, **Then** 不再二次 acquire、不再全量 memset
3. **Given** micro-test（100× open/close/setup/teardown 循环 + 逐次 cos）, **When** 高频切换, **Then** 无电源投票泄漏、无悬空指针、数值逐次一致
4. **Given** blob 带 0x4000 驻留 temp 但无 W4A16 op, **When** 执行, **Then** 驻留池照常工作（现状已支持，回归确认）

---

### User Story 3 - supertile-interior temp 编译期驻留（Priority: P2）

编译期把 supertile solver 的成员关系接进内存规划：supertile 内部边（生产者和消费者同在一个 supertile）对应的 temp 打 `in_vtcm=true` + 固定偏移，经 TAG_MEM_PLAN → TEMPOFF → `0x4000` 落到 VTCM 尾划池，省掉这些中间张量的 DDR round-trip。

**Why this priority**: 建立在 US2 之上（没有单 arena，驻留池会被 memset 踩掉）；收益是 elementwise 链的 DDR 流量，量级小于 US1/US2。

**Independent Test**: 编译含 elementwise 链的图，确认 TAG_MEM_PLAN entries 带 in_vtcm、blob TEMPOFF 带 0x4000；设备上 temp_get 返回 VTCM 指针；**0.8B 上板双门**（cos 不变 + wall 对照 US2 后数值）；无 supertile 的存量 golden（110 conv_add）逐字节不变。

**Acceptance Scenarios**:

1. **Given** solver 产出 supertile 成员表, **When** compute_ddr_offsets 跑, **Then** 组内中间 temp 进 entries 且 in_vtcm=true、偏移落在驻留区
2. **Given** 预算不足（Σ驻留需求 > resident_size）, **Then** 超出的组回退 DDR（不发 0x4000），不静默错
3. **Given** 无 supertile 的图（如 110 golden）, **When** 重编译, **Then** blob 逐字节不变

## Requirements

### Functional Requirements

- **FR-001**: 新增 opcode `OP_MATMUL_F16_SPLIT = 35`（arity 9: `[a_ref, w_s, out_t, M, K, N_full, n0, N_chunk, flags]`；已核实 OP_PAD_F16=34 为枚举尾、WT_MAX_ARGS=16），旧 `OP_MATMUL_F16`（arity 7）保留不动
- **FR-002**: emitter 对「M/K/N 全 32 对齐 + 非 batched + f16 + need 超预算」的 matmul 自动 N 分块；`N_chunk = max32(反解 need(M,K,N_c) ≤ scratch 预算)`。注：f16 路径 N_chunk 取 32 倍数即可；**W4A16 若未来也走编译期 N 分块，N_chunk 必须 64 的倍数**（n-tile 成对共享 chunk 流，V81HexSim 裁决备案），本特性不做 W4A16 分块
- **FR-003**: 运行时 chunk 写共享 out temp 列子区间（scatter stride=N_full，列基址 n0），标量回落路径同样支持 n0
- **FR-004**: TAG_MEM_PLAN 升级 v2，新增四区尺寸字段；v1 读取端静默忽略（现状行为）保持
- **FR-005**: exec 层单 acquire：wtcache_open / hmx_runtime_setup 改「attach 到既有 arena」模式，禁止二次 acquire 与全量 memset；**blob 无 TAG_MEM_PLAN v2 扩展段（旧 blob）时维持现状懒 acquire 路径，行为完全不变**
- **FR-006**: HMX 锁统一到一个 ctx（W4A16 走 mxmem 需要 set_hmx_param + exclusive hmx_lock，设备已证）
- **FR-007**: f16 HMX guard 语义从「互斥 return -1」改为「从 scratch 区划，超 scratch_size 才回落」
- **FR-008**: must-fix 落地：wtcache power_on static-once 守卫；hmx_common memset 后补 cache flush（照抄 wtcache_impl.c:152-157 已修同款）
- **FR-009**: supertile solver 结果暴露（accessor 或填充 GraphPrepare::supertiles_）
- **FR-010**: supertile-interior temp 预置 in_vtcm + 固定偏移注入 compute_ddr_offsets；驻留区从 VTCM 预算尾部预留
- **FR-011**: 预算统一：plan_vtcm_budget_ 由 solver 需求 + scratch_size 反推（或 hnnx_compile 默认打开），消除 G6 的双预算分裂
- **FR-012**: fresh_temp silent-drop（out_idx!=0 / size shrink）对 supertile 驻留 temp 显式告警，不静默剥除
- **FR-013**: N 分块数值纪律（V81HexSim 累加器语义裁决，逐字节恒等的成立条件）：① 每 tile `mxclracc` 不可省；② 每 tile 重装 bias 页（scratch 时间复用区会被别的 op 踩）；③ K 维升序不动；④ **回落是 op 级全有或全无**——任一 chunk 无法走 HMX 则整个 op 的所有 chunk 都走标量，禁止部分 chunk HMX + 部分标量的混合路径（混合即破逐字节恒等）
- **FR-014**: supertile DP 成本模型的 DDR 带宽改分档实测值（替换 supertile.cpp:81 的 48GB/s 硬编码）：spill 读回 37.5 GB/s、写出 56.0 GB/s、DMA 59.9 GB/s + 860cyc/次启动项（V81HexSim 器件标定，52f67807）；常数集中管理（仿其 DD-09 画像节纪律）

### Key Entities

- **TAG_MEM_PLAN v2**: `[ver=2][ddr_cap][vtcm_cap][bump_reserve][pin_size][carve_size][scratch_size][resident_size][n_entries]...`
- **VTCM 四区布局**: `[0..pin_size) wtcache pin | [pin..pin+carve) W4A16 carve | [..scratch) HMX scratch（时间复用）| [vs-resident..vs) 驻留 temp 池（尾划）`
- **OP_MATMUL_F16_SPLIT**: 编译期 N 分块的 chunk op，共享 out temp
- **supertile 成员表**: op_id 集合 → interior temp 集合 → in_vtcm 候选

## Success Criteria

- **SC-001**: 0.8B 的 6 个 N4096 matmul 全部走 HMX，无 `vtcm overflow` 回落
- **SC-002**: 0.8B op70 后 f16 matmul 零回落（optrace 全 WT_ENG_HMX）
- **SC-003**: 0.8B 全模型输出 cos 与基线一致（≥0.999 门槛沿用既有判据）
- **SC-004**: 0.8B wall 从 370.7s 显著下降（目标：132 个 HMX-eligible op 的 322.3s 大头回收，预期 wall 减半量级）
- **SC-005**: 110 golden conv_add blob 逐字节不变（无 supertile 图零回归）
- **SC-006**: test_wtop_emit 全量通过（含新增 chunk/驻留契约段）
- **SC-007**: micro-test 100× 循环无泄漏无数值漂移
- **SC-008**: **每个 User Story 的阶段验收门都含 0.8B 上板双门**：精度（全模型 cos ≥ 基线判据）+ 性能（wall 记录并对照基线 370.7s/上一阶段值）——双过才算该阶段完成

## Out of Scope（明确不做）

- HMX scratch 的跨 op DMA/compute 重叠调度（double-buffer pipeline）——是后续优化，不在本特性
- supertile cost model 与 libHtpPrepare 的逐字节保真（现有简化 DP 够用，先接线后调参；带宽常数修正按 FR-014 做）
- batched BMM（attention q·kᵀ）的 HMX 化——dispatch 条件不含 batched，维持现状
- VTCM 部分尺寸 acquire（两 ctx 各持部分 VTCM）——设备行为未验证，本设计用单 acquire 绕开
- **Path T 时间复用的 quiesce/teardown 切换**——被 attach 模式取代，不实施（G8 中 quiesce 项随之作废；参见 [[gehtp-vtcm-coexistence-plan]] 留档）
- **f16 matmul 经 scale 页 ob 字段融合 bias**——ob 真实字节位置两代码库互斥（V81HexSim=128+4o vs GEHTP conv=4n+2）且器件探针从未区分，属未验证地面；要用必须先补器件探针（本特性 matmul 走恒等页 ob=0，不受影响）
- **fence/缓存一致性的仿真器验证**——V81HexSim 的 fence 全是 no-op，fence.h 三元组表的正确性只能靠器件定向测试
