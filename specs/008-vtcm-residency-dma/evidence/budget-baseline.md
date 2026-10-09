# 预算基线冻结（T003/T004）

**日期**: 2026-10-09 | **运行目录**: worktree `.scratch_008/` | **编译器**: worktree 自建 @7deb63b + 008 改动（host gcc 构建）

## T003 — 基线重发（已完成, 1008 23:32-23:46, 14 分钟）

命令（v4 同图 + 预算旗标，唯一差异 = `--ddr-budget 2147483648 --plan-vtcm-budget 16777216`）：

```
hnnx_compile --net-json .scratch_a3/qwen35_08b_v2.wtop.work/qwen35_08b_v2_net.json \
  --weights-bin .../model.params.bin --format tagged \
  --ddr-budget 2147483648 --plan-vtcm-budget 16777216 --opt-stats \
  --ddr-offsets evidence/ddr_offsets_p0.tsv \
  --output .scratch_008/qwen35_08b_008p0.tagged.bin
```

产物：tagged.bin 4,077,473,680 B（v4 = 4,077,228,560，+245KB = mem_plan 载荷）；TSV `evidence/ddr_offsets_p0.tsv`（n_temps=12517, pool_cap=2147483648，5 列 = temp_id/offset/size/live_start/live_end，无 in_vtcm 列——驻留事实以 [ddr] 行为准）。

### p0 emit 冻结数字（`wtop_emit_p0.log`，2026-10-09）

```
[ddr] TAG_MEM_PLAN: cap=2147483648 vtcm_cap=16777216 n_temps=12517 n_spills=0
[ddr] TEMPOFF slot 1942: cap=2147483648 vtcm_cap=16777216 n_temps=1675 (vtcm=1674)
[vtcm] pool=16777216(reserve 16777216) arena_max=16760832 carve_peak=3082496 staging_sum=0 slack=0 coverage=0.8401
error: VTCM 三方预算门超限: resident_pool=16777216 + hmx_arena=16760832 > budget=16777216 ...
```

**盘子读数**：
- 物化 temp 1675 个，**1674 个 VTCM 驻留**（99.9% 计数面）；coverage=**0.8401**（temp 字节加权，**已过 SC-001 的 80% 门**）；n_spills=0。
- **carve_peak=3,082,496 B（2.94MB）= 驻留实际生命期峰值** —— 16MB 预算只用了 3MB 高水位，`vtcm_cap` 口径 = **预算**（ddr_offsets.cpp:35 `vtcm_cap_out = vtcm_budget`），非用量；预留头寸留给 emit 期 staging（op_matmul vtcm_stage_top 按 cap 校验）。
- **arena_max=16,760,832 B ≈ 16MB**：SPLIT 二分按 `kVTCMTotal=16MB` 定 chunk（op_matmul.cpp:136 旧口径），最大 chunk 贴满 16MB——与满额池冲突，三方门如实拦截（**门按设计触发，p0 blob 不落盘 = 预期**）。
- staging_sum=0（本图无 W4A16，符合预期）。

### p1 配方（由盘子导出）

`--plan-vtcm-budget 4194304`（4MB ≥ carve_peak 2.94MB + staging 头寸）重编 + emit 侧 SPLIT 阈值单源化为 `16MB − pool`（已实现）→ 预期 gate PASS（pool 4MB + arena ≤12MB ≤16MB）。p1 编译已启动（`.scratch_008/hnnx_compile_p1.log`）。

## p1 emit 冻结数字（2026-10-09 00:20, gate PASS）

```
[ddr] TAG_MEM_PLAN: cap=2147483648 vtcm_cap=4194304 n_temps=12517 n_spills=0
[ddr] TEMPOFF slot 1942: cap=2147483648 vtcm_cap=4194304 n_temps=1675 (vtcm=1674)
[vtcm] pool=4194304(reserve 4194304) arena_max=12570624 carve_peak=3082496 staging_sum=0 slack=12288 coverage=0.8401
[carve] 12516 entries, footprint 3082496 B (cap 4194304)
WTOP OK: slots=1943 ops=14180 bytes=2041539584 (gguf hits=0 miss=0)
```

- **三方门 PASS**：pool 4MB + arena 12,570,624 ≤ 16MB（slack 12,288）。SPLIT 阈值单源化（16MB−pool=12MB）后最大 chunk arena 12.57MB 贴 12MB 界。
- 驻留面与 p0 恒等（1674/1675, coverage 0.8401, carve_peak 2.94MB）——pool 从 16MB 降到 4MB 只影响 SPLIT 粒度，不影响驻留分配。
- carve 12516 条 = 12516 个 op 锚定生命期窗口去重后落到 1674 个 (offset,size) 池区（temp id 回收 + 生命期复用，双源一致性检查 PASS）。
- **8-op 漂移（14188→14180）已归因，非 US1 面**：18 个 2-chunk SPLIT 新增 + lm_head 32→42 chunk（阈值变化，写不相交 N 列，值恒等）+ **fc_bias fold 差**（opcode36: 0→18 / ADD −18 / BROADCAST −18 / matmul24 −36）。v4 原编译 `fc_bias_fold=0/0`（旧 checkout），当前二进制 18/18 默认开。

### p2 配方（纯驻离测试版, 2026-10-09 启动）

fold 的 HMX bias 单舍入 vs 旧 matmul+ADD 双舍入可能破坏位级 md5 门 → p2 = p1 配方 + `--no-pass fc_bias_fold`（hnnx_compile.cpp:315 已有该旗标, PassManager::disable 对注册前名也生效）。期望：opcode36=0、opt-stats `fc_bias_fold=0/0`、after≈21139、ops = v4 14188 + SPLIT 阈值变化。**p2 是 T010 板跑主产物**（SC-004 位级门量的是纯驻留，fold 差属 005/007 线口径）。

## T005 — mem_plan 激活判定（已定论：配方层修复，无代码缺口）

p0 emit 出现完整 `[ddr] TAG_MEM_PLAN` + `[ddr] TEMPOFF` 两行（v4 同图 88 行日志零 [ddr] 行）——旁路根因 = v4 编译配方未带预算旗标，**带旗标重发即激活全链**（规划→TAG_MEM_PLAN→TEMPOFF→运行时 vtcm_off_arr 池驻留）。

## T008 门与惰性门实证（2026-10-09）

- **惰性门 PASS**：新 wtop_emit 无旗标重发 v4 tagged.bin → blob md5 `9f358639` 与基线 `qwen35_08b_v4.wtop` **逐字节同**（门对合法 blob 静默）。
- **SPLIT-N 陷阱（已修）**：门初版按 args[5]=N_full 计 SPLIT arena → v4 的 32-chunk lm_head 被算成 ~500MB、惰性门拦截误杀。修 = SPLIT 按 args[7]=N_chunk 计（wtop_emit.cpp gate 块内注释）。**惰性门流程当场抓住真 bug——门本身也过门**。
- 报告先落盘再判门（超限跑的报告 = 诊断证据）。

## T004 — census `[vtcm]` 旗标语义（已定论）

**结论：旗标是句法（引用编码形态），不是语义（池驻留事实）。**

- `scripts/gehtp_prof.py:277`：`op.vtcm = any((x & 0xC000) == 0x4000 for x in a)` —— census 的 vtcm 列 = 该 op **任一参数以 `0x4000|temp`（WT_REF_VTCM_FLAG）编码**。
- v4 运行时（`oplist_exec.c:237-247` 引用解码）：`0x4000|temp` 在 `vtcm_off_arr[t]==0xFFFFFFFF`（无 TEMPOFF 表，v4 实况）时**回落解析为 `g_exec.temps[t]`（DDR 静态区/bump）**。
- 故 v4 的 1107 个 `[vtcm]` op = 发射侧选了驻留引用形态（op_matmul.cpp:82 等先例），运行时全部解析到 DDR。**p1 blob 落地后这批引用变成真池驻留**——旗标数 1107 是「引用面就绪」基数，非驻留覆盖数。

## 门控根因（T005 诊断，随 T003 定稿）

- 静态规划唯一门 = `graph_prepare.cpp:591` `if (plan_ddr_budget_ > 0)`；预算来自 `hnnx_compile --ddr-budget`（:328）+ `--plan-vtcm-budget`（:329），缺省 0 = 不规划。
- v4 的 tagged.bin 编译时无预算旗标 → 无 TAG_MEM_PLAN → `wtop_emit.cpp:117` `if (gp.has_mem_plan())` 旁路 → 无 `[ddr]` 行/无 TEMPOFF 槽（emit 日志 88 行实证）。
- scripts/gehtp 编译段（L79-89）预算旗标为可选透传——v4 配方未带。**修复 = 配方层（带旗标重编），已由 p0 实证。**

## T010 首跑失败与根因（2026-10-09 00:45, p2 blob, d0f1784 + libhvxhmx_003 md5 4059fa2a）

```
[FAIL] wt_exec_run_io rc=-1 tempoff pool alloc
```

- 运行期 TEMPOFF 解析（oplist_exec.c:2242）`memalign(128, cap + reserve)`，cap = `--ddr-budget` = **2GB** → 设备堆（DSP 预算 ~3GB，blob 已占 2GB）不可能满足。**首个真图 TEMPOFF 板跑**才暴露（此前板跑全是无 mem_plan 的 v4 路径或小 cap 合成 blob）。
- **次生缺陷（同因实锤）**：bump_reserve = max(4MB, cap/4)（graph_prepare.cpp:600）= 512MB = 524288KB，TEMPOFF reserve 低 16 位 KB 域**静默回绕成 0** —— 与 VTCM 4GB 回绕同类陷阱。
- **修复双管**：①配方 — p3 `--ddr-budget 33554432`（TSV 实测 max extent **18.10MB**/12535 planned/550MB sum → 32MB 不绑规划；bump=8MB；总 memalign 40MB DSP 安全）；②代码 — wtop_emit TEMPOFF 块 reserve 域溢出硬门（pool/bump 各 >65535KB 即 exit 3，test_wtop_emit 全绿）。
- p2 板跑 census 前置全对（ops=14216 = v4+28 纯 SPLIT 差，opcode36=0），失败点在运行期池分配，非 blob 结构。

## T010 p3 二跑失败与划痕窗根因（2026-10-09 01:3x, p3 blob @32MB DDR）

```
[FAIL] wt_exec_run_io rc=1311 broadcast ref fail
```

- op1310 (OP_BROADCAST_F16) 输出 `temp_get` NULL：**8MB bump 在 op1310 恰好耗尽**。根因 = 广播物化 temp 恒 `fresh_temp(op_id, 6/7)`（out_idx≠0 不做静态登记）→ 1622 个全走运行期泄漏式 bump，流过 103.8MB，泄漏口径需求 108.9MB —— **超 reserve 低 16 位 KB 域上限 64MB，bump 扩容路线本图数学不可行**。
- **修复（emit 侧，运行期/设备 lib 零改动，md5 仍 4059fa2a）= 静态划痕窗**（wtop_ops.hpp `fresh_temp`）：`out_idx≥6` 且图 `last_use` 查无此项（= 生命期限于本 op）的物化 temp，从 DDR 静态区**顶部向下**发静态偏移，**逐 op 复位窗深**；全图只占单 op 并发峰值。窗与规划器静态区碰撞 / 窗溢出 = 硬错误。
- p3 重 emit 冻结数字：

```
[scratch] 物化窗 peak=262144B (planner_end=15892480B cap=33554432B) entries=1622
[bump] plan=8388608 sim_peak=0 chosen=8388608        ← 表外需求仿真 0, 全部静态化
[ddr] TEMPOFF slot 1942: cap=33554432 vtcm_cap=4194304 n_temps=3253 (vtcm=1630 scratch=1622)
[vtcm] pool=4194304 arena_max=12570624 carve_peak=3082496 slack=12288 coverage=0.8395
```

## T010 三门结果（2026-10-09, p3 blob, d0f1784, 全套私有名 runner/job/output）

| 门 | 结果 | 证据 |
|---|---|---|
| GATE1 位级 | **PASS** `md5=f87461e2a5089c48413e70463f4aa7dc` == v4 锚（libhvxhmx_003 md5 4059fa2a）；**007 lib 叠加复验同 md5** | `.scratch_008/output_008.f16.raw` / `output_008x007.f16.raw` |
| GATE2 覆盖 | **PASS** coverage=0.839522（1630/1631 物化 temp VTCM 驻留，唯一 nonresident = 输出 temp 3252 planner_displaced 15.9MB） | `budget_report_p3.txt` |
| GATE3 wall 校准 | **完成**：全图无差别驻留 = 净负优化，见下表 | op_ts census ×2 |

## GATE3 校准：驻留 wall 归因（007 lib 同库 A/B）

同为 libhvxhmx_007.so（md5 5a023fbb，广播 HVX 内核）：

- **无驻留基线** = 007 会话 `qwen35_08b_v4_007.wtop`（14188 ops = v4 图恒等，fold/块化同 v4）：`wall_us=24,240,344`（24.24s）
- **p3 驻留**（1630 temp 全驻 4MB 池 + 12MB SPLIT 阈值）：`wall_us=37,651,234`（37.65s）
- **Δ = +13.41s**，per-opcode 闭合（ΣΔ=13.3s ≈ 总差）：

| opcode | n | 无驻留 | p3 驻留 | Δ | 倍率 |
|---|---|---|---|---|---|
| BROADCAST_F16 | 1622 | 0.082s (50µs/op) | 3.516s (2168µs/op) | **+3.43s** | **43×** |
| OP35 (lm_head SPLIT) | 32→78 | 4.247s | 6.176s | **+1.93s** | 重切块 |
| STRIDED_SLICE | 2373 | 3.353s | 4.952s | +1.60s | 1.48× |
| UNARY | 2909 | 2.997s | 4.563s | +1.57s | 1.52× |
| REDUCE | 1170 | 0.338s | 1.567s | +1.23s | 4.6× |
| SCATTER_ND | 1154 | 0.140s | 1.097s | +0.96s | 7.8× |
| CONV1D_SSM | 18 | 0.324s | 1.007s | +0.68s | 3.1× |
| TRANSPOSE/BINARY/GATHER/RMSNORM/其余 | — | — | — | +2.0s | 1.3-2.3× |
| MATMUL_F16 | 325→307 | 9.439s | 9.629s | +0.19s | **1.02×** |

**结论（冻结，GateRecord）**：
1. **HMX 读驻留无惩罚**（MATMUL +2%）——驻留机制本身成立（位级 + 预算门 + carve 一致性全绿）。
2. **HVX/scalar 算子读 VTCM 驻留 temp 全线 1.3-43× 变慢**（同内核同数据量：广播 HVX DDR 输入 50µs → VTCM 输入 2168µs）——**全图无差别 f16 驻留是净负（−11.5s 扣除重切块 +1.9s）**。
3. 重切块 +1.93s 是 pool 4MB 强制 SPLIT 阈值 16→12MB 的连锁（32→78 chunk，实测 ~42ms/chunk 固定开销）。
4. **机理待裁**：V81 器件书实测 VTCM ~109GB/s/线程平坦、7 拍延迟 → **非 VTCM 带宽本身**；候选 = 跨核别名（书：VTCM 按核私有互相不可见）/ bank 冲突 / HVX 内核对 VTCM 输入的访存形态。US2 用 V81HexSim 迹重放裁决（fence/DMA 竞态照旧不背书）。
5. **US2 设计输入**：驻留必须按消费引擎分域——HMX 消费 → VTCM 驻留；HVX/scalar 消费 → DDR（时间复用候选 B 口径）。

GateRecord 冻结值：f16 位级锚 `f87461e2…`（003/007 双 lib）；coverage 0.8395；wall 24.24s（无驻留基线，007 lib）/ 37.65s（US1 全图驻留，007 lib）/ 43.62s（US1 @003 lib，广播标量回落 1023 次）；库指纹 5a023fbb（007）/ 4059fa2a（003）；设备 d0f1784。

叠加跑资产：`.scratch_008/{run_008x007_board.sh, build_runner_008x007.sh, x007.device.txt, x007.op_ts.bin, output_008x007.f16.raw}`（runner NEEDED=libhvxhmx_007.so；板上 2GB blob 复用未重推）。
