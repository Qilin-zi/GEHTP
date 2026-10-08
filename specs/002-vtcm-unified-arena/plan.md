# Implementation Plan: VTCM 统一内存规划

**Branch**: `task/prof-wp` | **Date**: 2026-09-28 | **Spec**: [spec.md](spec.md)

> ⚠️ **2026-09-29 裁决**：US2（四区 arena）退役，VTCM carve 归 003 P2（见 [tasks.md](tasks.md) 顶部）；US3（supertile 驻留）保留，待 003 P2 后嫁接；US1 已完成。本文 US2 章节仅作存档。

**Input**: Feature specification from `/specs/002-vtcm-unified-arena/spec.md`

## Summary

把 VTCM 从「三个消费者运行时各自抢全量」改为「编译期规划四区尺寸/偏移、运行时一次 acquire 后照抄」，并在此之上交付两个收益点：(US1) matmul 编译期 N 分块消除超界回落；(US3) supertile-interior temp 编译期驻留消除 DDR round-trip。US2（单 acquire arena）是 US3 的物理前提，同时消除 f16↔W4A16 互斥回落。

## Technical Context

**Language/Version**: C99（kernels，Hexagon DSP + host 双编译）/ C++17（compiler host 工具）

**Primary Dependencies**: 现有 GEHTP 工具链（`hnnx_compile` → `wtop_emit` → blob；runner `42_gehtp_runner`）；HAP compute-res API（VTCM/HMX acquire）

**Storage**: 无新增持久化；TAG_MEM_PLAN 记录格式升级 v2（tagged.bin 内）；blob TEMPOFF 槽复用现有格式

**Testing**: host `test_wtop_emit`（契约/回归）+ 110 golden conv_add 逐字节比对 + 设备上板（optrace / cos / wall）

**Target Platform**: SA8797P V81（VTCM 8MB，映射 [0xff800000, 4GB)，已实锤 4GB 回绕坑）+ host x86 模拟

**Performance Goals**: 0.8B wall 370.7s → 减半量级（回收 132 HMX-eligible matmul 的 322.3s）

**Constraints**:
- 旧 blob 兼容：v1 TAG_MEM_PLAN / arity-7 MATMUL_F16 blob 在新 runtime 行为不变
- 数值恒等：N 分块不动 K 累加序，输出逐字节一致
- vgather=0 门不破（kernels 构建约束）
- 设备 lib 推签名版；多会话共享设备 d0f1784 先查进程冲突

**Scale/Scope**: 0.8B（16297 ops）为主要验证模型；110 conv_add golden 为零回归基线

## 目标架构：单 acquire + 编译期规划四区

```
VTCM [0 ───────── pin ────── carve ────── scratch ────── resident ────── vs=8MB]
       W4A16 钉住    W4A16 arena    f16 HMX 工作区         0x4000 驻留池
       (PIN op)     (dc_w4_carve)  (crouton,时间复用)     (supertile-interior)
```

**分工原则（对齐 TEMPOFF 既有哲学）**：编译期算 `{pin_size, carve_size, scratch_size, resident_size}` + 各区偏移，经 TAG_MEM_PLAN v2 下发；运行时唯一必须做的事是**一次 acquire 拿基址**（基址运行时才知道），然后 `区基址 = vtcm_base + 编译期偏移`。三个消费者都不再各自 acquire、不再全量 memset。

### 四区尺寸语义

| 区 | 内容 | 现状 | Path U |
|---|------|------|--------|
| pin_size | W4A16 xop_pin 钉住的权重/偏置 | 恒 4KB（wtcache_open 第二参） | 按 blob PIN 需求算 |
| carve_size | W4A16 dc_arena 工作区（权重 tile 流式 carve） | 「剩余全量」无界 | 编译期封顶 |
| scratch_size | f16 HMX crouton 工作区（A+W+O+2KB 页） | 全量 8MB，超界回落 | 编译期定；同时是 US1 N 分块反解预算 |
| resident_size | 0x4000 驻留 temp 池（尾划） | TEMPOFF vtcm_cap | = Σ supertile 驻留需求（128B 对齐） |

scratch 为**时间复用**区（每 op 用完即弃、下一 op 重铺），其余三区为**空间分区**（run 期间各自独占）。

## Constitution Check

`.specify/memory/constitution.md` 为空模板，无治理约束，跳过。

## Phase 结构总览

依赖链（spec.md §0 的论证：A 与 B 在所有权层互相冲突，不能平行）：

```
US1 (matmul N 分块)     —— 独立，先做。不改所有权。
US2 (单 acquire arena)  —— 门控 US3；同时解锁「互斥」那一半损失。
US3 (supertile 驻留)    —— 依赖 US2（驻留池只有在无人 memset 全量时才存活）。
```

---

## Phase B / US1：matmul 编译期 N 分块

### 设计

**opcode 扩展（G7 决定必须新增）**：`OP_MATMUL_F16_SPLIT = 35`（已核实 OP_PAD_F16=34 为枚举尾，oplist_parse.h:156；WT_MAX_ARGS=16 ≥ 9），arity 9：

```
[a_ref, w_s, out_t, M, K, N_full, n0, N_chunk, flags]
```

- `N_full`：完整 N（out temp 的行距/scatter 依据）
- `n0`：本 chunk 列基址；`N_chunk`：本 chunk 列数（均 32 倍数）
- flags 语义与 OP_MATMUL_F16 相同（bit0 a 转置 / bit1 w 转置；batched 不进门，维持现状）

**触点**：

| 位置 | 改动 |
|------|------|
| kernels/include/oplist_parse.h | opcode 枚举 + `WT_ARITY_MATMUL_F16_SPLIT 9` + 线格式注释 |
| kernels/src/runtime/oplist_parse.c:10-45 | `arity_of` 加 case；:69 `arg_is_slot` 加 case（idx 0,1 可为 slot/temp 引用） |
| compiler/tools/wtop_ops/op_matmul.cpp | f16 非 batched、M/K/N 全 32 对齐且 need(M,K,N)>预算时：发射 `ceil(N/N_chunk)` 个 SPLIT op（**act/wgt/out 三者都共享原引用**，每 chunk 只差 n0/N_chunk 两个标量） |
| kernels/src/runtime/oplist_exec.c:1536 | `exec_matmul_f16` 分派 SPLIT：HMX 路径复用 exec_matmul_f16_hmx 主体，unpack scatter 改 `y[(r*32+m)*N_full + n0 + c*32+n]`；标量回落同样加 n0 列偏 |

**分块预算**：US2 落地前用运行时 `vtcm_sz`（hmx_runtime_setup 报告的 total，现状同口径）；US2 后切换为 TAG_MEM_PLAN v2 的 scratch_size。`N_chunk = 最大 32 倍数使 need(M,K,N_c) ≤ 预算`。

**数值恒等论证与纪律（V81HexSim 累加器语义裁决：成立，附条件）**：输出元素 y[m,n] = 「同一 fp16 乘积流按 k 升序进累加器（器件 FP37 / 建模 double 精确和）→ 单次 RNE 窄化」的确定性函数；N 切分只改变哪些 n 列在同一轮 HMX 调用里算，不改任何元素的乘积流 → chunk 与整算在任意 K 下逐位恒等。**成立依赖五条纪律**（实施时必须逐条对照）：
1. 每 tile `mxclracc.hf` 不可省（省了是整 tile 错，非 ulp 级）
2. 每 tile 重装 `bias = mxmem2`（scratch 是时间复用区，chunk 间隙页内容可能被别的 op 覆盖，跳过重装 = 读 stale）
3. K 维升序不动
4. 全 chunk 同走 HMX；**回落是 op 级全有或全无**——任一 chunk 失败则整 op 全部走标量（混合路径即破逐字节恒等）
5. W4A16 若未来也分块：N_chunk 必须 64 的倍数（n-tile 成对共享 chunk 流）；本特性只做 f16

**验证门层级警示**：host 的「chunk vs 标量 SHA」两侧都是标量路径，测不到要害——设备门必须是「chunk-HMX vs 整算-HMX」逐字节对拍（合成 blob 强制双侧 HMX）。

**为什么权重零拷贝**：`y[m, n0:n0+Nc] = A[M,K] @ W[K, n0:n0+Nc]`——w 的列区间与 out 的列区间是同一个 n0，一个参数两处用。chunk 的 w 引用仍指向整权重 slot，运行时 pack 循环按 n0/N_full 寻址（[K,N] 布局 stride=N_full；FC 的 [N,K] 布局按行连续），blob 体积零增长、权重零复制。

### 验证门

1. host：test_wtop_emit 新增 SPLIT 发射/解析契约段，全量过
2. host：N4096/K1024 合成图，chunk 输出 vs 标量参考 SHA 一致
3. **设备：合成 blob 双侧强制 HMX，chunk 版 vs 整算版逐字节对拍**（上条警示的要害门）
4. 设备：0.8B optrace 中 6 个 N4096 op 走 WT_ENG_HMX，无 `vtcm overflow`；**0.8B 双门：cos ≥ 基线判据 + wall 记录对照 370.7s 基线**
5. round-trip：`hnnx_compile → wtop_emit → wt_parse` 自检过；110 golden 逐字节不变

---

## Phase 0 / US2：VTCM 单 acquire 四区共存

### 设计

**0. 四区尺寸的编译期求解（破解 C1 循环依赖 + U1/U2）**

插入点：`do_prepare2_late` step 6（graph_prepare.cpp:584-606）内、`compute_ddr_offsets` 调用**之前**，先扫一遍 op list 求四区尺寸：

- `pin_size`：按 blob PIN op 的实际需求（现状恒 4KB 的合理化）
- `carve_size`：max over W4A16 ops 的单 op tile 需求（dc_w4_carve per-op 消耗模型的峰值；无 W4A16 则 0）
- `scratch_size`：**不动点定义** = max over HMX-eligible ops 的 `need(M, K, N_c=N_floor)`，N_floor=1024（保证每 chunk ≥1024 列的最小预算，可调参数）；某 op 连 N_floor 都装不下 → 该 op 不在 HMX 覆盖范围（维持回落，或未来 K 维分块）。US1 的 `N_chunk` 再据此反解：`最大 32 倍数使 need(M,K,N_c) ≤ scratch_size`——单向依赖，循环消除
- `resident_size`：本阶段 0（US3 回填）
- 约束校验：pin+carve+scratch+resident ≤ VTCM total（8MB），不满足时按 resident→scratch 顺序收缩并 stderr 显式 warn

**对齐红线（V81HexSim 器件语义）**：arena 内所有 tile 基址 ≥128B 对齐（vmem 128B 窗静默截断）；HMX crouton 目标 ≥2048B（2KB 窗）。四区偏移一律 2048B 对齐取整，同时满足两条。

**1. TAG_MEM_PLAN v2（FR-004）**

```
[ver=2][ddr_cap][vtcm_cap][bump_reserve]
[pin_size][carve_size][scratch_size][resident_size]   ← 新增 4×u32
[n_entries] n×20B{op_id,out_idx,offset,size,flags}     ← 同 v1
[n_spills] n×16B                                       ← 同 v1
```

- 序列化：graph_prepare.cpp:1591-1620（ver 改 2，插入 4 字段）
- 反序列化：graph_prepare.cpp:2375-2413（`ver==1` 走旧布局，`ver==2` 读新字段；旧 bin 静默 false → 现状行为不变）
- wtop_emit：读 v2 后把四区尺寸编进 TEMPOFF 槽 reserve 扩展（见下）
- TEMPOFF 槽复用：现有 reserve 字 `[hi16=vtcm KB][lo16=bump KB]` 不够放四个尺寸 → **TEMPOFF 槽尾部追加 4×u32 区尺寸段**，runtime 解析按 `len` 判断有无扩展（旧 blob len 短 → 四区默认 = 现状等价行为）

**2. exec 层单 acquire（FR-005/006）**

新增 `wt_exec_arena_ensure()`（oplist_exec.c）：

- 触发点：wt_exec_run_range 开头，当 TEMPOFF 扩展段存在（vtcm 四区尺寸非全零）时
- 动作：一次 `HAP_compute_res_acquire`（VTCM total + set_hmx_param）+ exclusive hmx_lock + 全量 memset + flush（只此一次）；记录 `g_exec.arena_base` 与四区基址/尺寸
- wtcache_open 增加 attach 模式：arena 已存在时只登记基址/尺寸，**跳过** query/acquire/memset/lock（wtcache_impl.c:113-157 的条件化）
- hmx_runtime_setup 同样：arena 已存在时 `g_vtcm_base = arena_base`（hmx_common.c:111 幂等检查之前接 hook），**跳过** acquire/memset
- 无扩展段的旧 blob：完全走现状路径（懒初始化、各自 acquire）——新旧行为由 blob 内容分野，不靠版本号硬切

**3. guard 语义改写（FR-007）+ 锁统一（FR-006 含 C2 修正）**

- `exec_matmul_f16_hmx` / `exec_conv2d_hmx` 首行（oplist_exec.c:1435 / :622）：
  - arena 模式：`need > scratch_size → -1`（其余照跑，scratch 基址 = arena_base + scratch_off）
  - 非 arena 模式（旧 blob）：保持现状 guard 不变
- exec_matmul（W4A16）dc_arena_init 范围从「vs-off 全量」收窄为 carve_size 硬界（oplist_exec.c:444 附近）
- **锁统一**：arena ensure 已持 exclusive hmx_lock（设备已证模式，wtcache_impl.c:148）；arena 模式下 `hmx_enable_execution`/`hmx_disable_execution`（hmx_common.c:202-218 的 lock2 SHARED 每-tile 对）**退化为 no-op**，且 `g_vtcm_ctx_id = arena ctx`——否则每-tile lock2 与 arena 的 exclusive 锁冲突/冗余（C2 缺口）。非 arena 模式维持现状逐 tile 加锁。
- **teardown 语义（U3）**：attach 模式下 `hmx_runtime_teardown` 只清 `g_vtcm_base=NULL`、**不 release ctx**（ctx 归 arena 所有）；非 arena 模式维持现状。

**4. must-fix 落地（FR-008，G8）**

- wtcache_impl.c:55-90 `power_on_hvx_hmx` 加 static-once 守卫（4 个 HAP_power_set 投票只发一次）
- hmx_common.c:169 memset 后补 `qurt_mem_cache_clean(FLUSH)`（照抄 wtcache_impl.c:152-157 的 V2.2 已修同款）
- wt_exec_shutdown：arena 模式下释放顺序 = 现状（wtcache_close 已含 unlock+release），加 `arena_base=NULL` 清悬空

**5. 去险 micro-test（先于一切设备改动验证）**

- ~~双 acquire 行为实验~~ **已收窄**：双源证据确认「第二个全量 acquire 返回 0 = 失败，非 alias」——V81HexSim 记账模型（runtime.cpp bump 池尽→r0=0）+ GEHTP 设备史（f16 占坑后 wtcache_open 报 WTC_ERR_VTCM_ACQUIRE，[[gehtp-vtcm-ownership-gap]]）。micro-test 改为**回归确认**这个 fail 行为 + 验证 attach 模式不再触发二次 acquire
- 100× open/close/setup/teardown 循环 + 逐次 cos（电源投票表耗尽、悬空指针）
- 锁行为实验：exclusive hmx_lock 持有时 f16 路径 no-op 化后的 HMX 指令执行正常（C2 的设备面）
- 注意 T10 遗留警告：wtcache_impl.c:127-128 `set_vtcm_param_v2(...,0,0)` 在深核下 fault 过，acquire 参数模式改动要回归
- **fence 不在 sim 覆盖内**（V81HexSim 的 fence 全 no-op），flush/invalidate 正确性只能在设备上验（写 VTCM 后不 flush 直接 HMX 读 = 复现旧坑的负测试）

### 验证门

1. micro-test 全过（无泄漏、cos 逐次一致、锁行为正常）
2. 0.8B 上板：op70 后 f16 matmul 零回落（optrace WT_ENG_HMX）；**0.8B 双门：cos ≥ 基线判据 + wall 对照基线/US1 后数值（预期减半量级，SC-004）**
3. 回归：110 golden 编译产物逐字节不变（v1 格式路径未动）
4. host：test_wtop_emit 全量过

---

## Phase A / US3：supertile 账本接线

### 设计

**0. 成本模型修数（FR-014，先于一切接线）**

supertile.cpp:81 的 `ddr_bw = 48GB/s` 硬编码两头不靠（V81HexSim 器件标定，52f67807 锚）：

| 用途 | 现值 | 改为 | 偏差 |
|------|------|------|------|
| spill 读回（vmem load 流聚合帽） | 48 | **37.5 GB/s** | 乐观 28% |
| spill 写出（vmem store 流） | 48 | **56.0 GB/s** | 悲观 17% |
| DMA refill/writeback | 48 | **59.9 GB/s + 860cyc/次启动**（链式 635/desc） | 漏固定项 |

`ddr_saved = intermediate_bytes / ddr_bw_` 按流类型分档计价；常数集中到画像节式单点（仿 V81HexSim DD-09 纪律，TSV 可覆盖）。**收益评估别忘死区**（其 pct-shift 翻车教训：预测无死区入账→系统性乐观）：GEHTP op_ts.bin 自带真死区（`start[i+1]−start[i]−dur[i]`），评估时注入。

**1. 结果暴露（G1，FR-009）**

`SuperTileSolver` 加 `const std::vector<SuperTile>& tiles() const` accessor（supertile.hpp:82-85 现只有 solve/apply）；`create_supertiles()` 末尾把 `supertile_solver_->tiles()` 拷入 `GraphPrepare::supertiles_`（graph_prepare.cpp:4784-4785 之间接线，死代码 make_one_supertile 不动）。纯机械，无行为变化。

**2. 账本修正（G2）**

驻留决策的字节账 = interior temp 的 output 字节（128B 对齐）——supertile.cpp:236-249 的现公式对驻留场景恰好正确（驻留的就是输出张量）。但 DP 的可行性判据（can_merge 的 vtcm_size 预算）要**先扣掉 scratch_size + pin/carve** 再跑（dr_group_graph.cpp:327-346 / supertile.cpp:61-75 的预算入口），否则驻留池会挤掉 HMX 工作区。

**3. 注入 compute_ddr_offsets（G3，FR-010）**

ddr_offsets.cpp:83-107 两级分配之前：

- 从 `gp.get_supertiles()` 取成员表 → 找出 interior temp（op 的 output 被同组内另一 op 消费、且不是图输出）
- 这些 temp 预分配驻留区固定偏移（尾划布局，编译期排好），entries 里 `in_vtcm=true`
- 剩余 VTCM 预算（resident_size 扣掉已驻留部分）再走 FancyAllocator first-fit（现状逻辑不动）
- 预算不足时整组回退 DDR（in_vtcm=false），stderr 显式 warn，不静默

**4. 预算统一（G6，FR-011）**

- `plan_vtcm_budget_` 默认 0 的根因修复：hnnx_compile 增加默认值逻辑——模型含可驻留 supertile 时自动设 `resident需求 + scratch需求`；或 `--plan-vtcm-budget auto`
- supertile DP 的 `vtcm_size_`（graph_prepare.cpp:3887-3896 的 min(4MB,4MB)×0.75=3MB 硬编码）改为与设备 VTCM total（8MB）减 scratch/pin/carve 后的一致口径

**5. 防静默剥除（FR-012）**

wtop_ops.hpp:156-158 fresh_temp 的 silent-drop 条件命中 `ddr_temp_vtcm` 候选 temp 时 stderr 告警（out_idx!=0 / size shrink 导致驻留标记丢失要可见）。

### 验证门

1. 编译含 elementwise 链的图：TAG_MEM_PLAN entries 带 in_vtcm；blob TEMPOFF 带 0x4000；wt_parse 自检过
2. 设备：驻留 temp 的 temp_get 返回 VTCM 指针（optrace/rtrace 取证）；**0.8B 双门：cos 不变 + wall 对照 US2 后数值**
3. 110 golden 逐字节不变（无 supertile 图零回归，SC-005）
4. 预算不足回退路径：构造超界图，确认 warn + DDR 回退而非错数据
5. 回归 US2 AS4 场景：带 0x4000 驻留 temp 但无 W4A16 op 的 blob 在 arena 模式下照常工作

---

## 风险登记

| 风险 | 等级 | 缓解 |
|------|------|------|
| ~~双 acquire 设备行为未知~~ 已解：fail 非 alias（V81HexSim 记账模型 + GEHTP 设备史 WTC_ERR_VTCM_ACQUIRE 双源一致） | ~~高~~ 低 | micro-test 回归确认即可；attach 模式仍做防御性错误处理 |
| T10 遗留：v2 acquire 参数深核 fault | 中 | micro-test 覆盖；参数模式与 wtcache 现状保持一致 |
| 统一后 wtcache exclusive lock 与 hmx_common lock2(SHARED) 模式冲突 | 中 | 统一到设备已证的 exclusive hmx_lock（wtcache_impl.c:148 模式）；arena 模式 f16 侧锁 no-op 化（C2） |
| N 分块发射改变 op 数 → 0.8B golden blob 变化 | 低 | 预期内；数值逐字节门兜底（含设备 chunk-HMX vs 整算-HMX 对拍）；golden 需重新登记 |
| **scale 页 ob 字段位置未验证**（sim=128+4o vs GEHTP conv=4n+2 互斥，器件探针从未区分） | 中 | 本特性 matmul 走恒等页 ob=0 不受影响；任何 ob 融合 bias 想法必须先补器件探针（spec Out of Scope 已列） |
| **fence 正确性仿真器零覆盖**（V81HexSim fence 全 no-op，漏 flush 对拍也绿） | 中 | fence 改动只能设备定向测试背书；勿拿 sim 对拍当 fence 证据 |
| fresh_temp silent-drop 剥除驻留标记 | 中 | FR-012 显式告警 |
| 驻留池与 dc_arena 生长相撞（arena 曾拿全量 vs-off） | 高 | US2 把 carve_size 做成硬界，物理隔离 |
| 多会话共享设备 d0f1784 撞车 | 中 | 上板前 `adb shell 'ps -A \| grep hexagon'`；job.txt 备份 .pre_ 后缀（[[gehtp-p5-device-gate]] 惯例） |

## 工作量估算

| 阶段 | 规模 | 主要文件 |
|------|------|---------|
| US1 N 分块 | ~150 行 | op_matmul.cpp, oplist_parse.{h,c}, oplist_exec.c, test_wtop_emit.cpp |
| US2 统一 arena | ~250 行 + micro-test | oplist_exec.c, wtcache_impl.c, hmx_common.c, graph_prepare.cpp, wtop_emit.cpp |
| US3 supertile 接线 | ~150 行 | supertile.hpp/cpp, ddr_offsets.cpp, graph_prepare.cpp, wtop_ops.hpp |

## 参考

- 探查 ground truth G1-G8：spec.md §背景
- 互斥/回落实测数据：记忆 [[gehtp-vtcm-ownership-gap]]（0.8B 分桶 132 ops = 86.9% wall）
- 时间复用替代方案（不取，留作 fallback）：[[gehtp-vtcm-coexistence-plan]]
- 设备坑清单：[[gehtp-p5-device-gate]]（4GB 回绕 / 签名 lib / 共享设备）
- V81HexSimNew 仿真器（105 服务器，常数/数值裁决/边界）：[[gehtp-v81hexsim-asset]]；本机副本 /disk1/scratch_v81sim/
