# 契约：wtop blob v3 扩展（编译器 ↔ 执行器接口）

本文是 5 个收口点的**字节级契约**。改动点分「槽数据」与「op 注解」两类；emit（`wtop_emit.cpp`）与 exec（`oplist_parse.c` / `oplist_exec.c`）必须同步。所有多字节字段小端。

## 契约 A — W4A16 辅助面显式标注

**现状**: `OP_MATMUL_W4A16` arity=6（`[act_slot, w_slot, out_temp, M, K, N]`），辅助面身份执行器按 `len` 反推。

**改动**: arity 6 → 9，追加三个 slot id：

```
OP_MATMUL_W4A16: [act_slot, w_slot, out_temp, M, K, N, bias_slot, atbl_slot, otbl_slot]
```

- `WT_ARITY_MATMUL` 6 → 9（`oplist_parse.h`）。
- 执行器 `exec_matmul` 删掉按 `len` 猜（`oplist_exec.c:562-571`），改为 `bias = weight_base + slots[bias_slot].offset` 等。
- **向后兼容**: ver≤2 或旧 emit 产出的 6 参 blob 不可同时存在新 opcode；arity 校验（`wt_parse` 返回 `WT_ERR_ARITY`）天然兜底，混跑即报错，不静默。
- **修 bug**: `tbl_b` 现用 K 同时匹配 atbl+otbl（`oplist_exec.c:549`），K≠N 必误判；显式标注后消失。

## 契约 B — TEMPOFF 槽头新增 pin_cap

**现状**: TEMPOFF 槽数据 = `[cap u32][reserve u32][n u32][n × {temp_id, offset, size}]`；`reserve` = 低16 bump KB | 高16 池 KB。执行器 `wtcache_open(wc, 4096)` 两处硬编码。

**改动**: 头部插一个 u32：

```
TEMPOFF 槽数据: [cap u32][reserve u32][pin_cap u32][n u32][n × {temp_id u32, offset u32, size u32}]
```

- `pin_cap` 单位**字节**（对齐 `wtcache_impl.c:108` 语义），2048 对齐。
- emit（`wtop_emit.cpp:668-697`）写入；exec（`oplist_exec.c:2550` 起 TEMPOFF 解析）读出，替代两处 4096（`oplist_exec.c:325,600`）。
- `pin_cap == 0` → 执行器回退旧硬编码 4096（关池口径，byte-exact 门覆盖）。
- **升版**: TEMPOFF 头 3 字段 → 4 字段是格式变更，须 emit/exec 同 commit 同步；不升 blob 全局 ver（仍 v3），因 TEMPOFF 槽自带长度可自描述（或加内部 magic 标记，取舍在 tasks 阶段定）。

## 契约 C — op_annot 保留位 bit5 作 carve 复用标志

**现状**: 引擎注解 footer 每 op 一条 u32：bit0-1 engine、bit2 fallback_allowed、bit3-4 fallback_engine、bit5-31 保留（`oplist_parse.h:54-58`）。

**改动**: bit5 定义为 carve 复用标志：

```
op_annot bit5 = 1  → 本 op 复用上一 carve（跳过 dc_w4_carve + 全 VTCM flush）
op_annot bit5 = 0  → 正常逐 op carve
```

- 新增打包/解包 helper（对称 `wt_annot_pack`）：`wt_annot_set_reuse(a, 1)` / `wt_annot_reuse(a) = (a >> 5) & 1u`。
- emit 侧编译期判定：本 op 与上一 matmul 的 M/K/N 相同 且 carve footer 的 live_top 不变 → 置位。
- 执行器 `exec_matmul` 删 `shape_same`（`oplist_exec.c:589-591`），改读 bit5；bit5=1 跳过 re-carve 分支。

## 契约 D — 权重 pin/restage 序列

**现状**: 执行器每 MATMUL 无条件 `cpu_to_vtcm(e.wt)` + `cpu_to_vtcm(e.bias)`（`oplist_exec.c:657-658`）。

**改动**: 编译期产出权重常驻决策（**新建规划面**，greedy/cp_solver 都 skip const 权重），显式下发序列（复用既有 `OP_PIN`(3) 与 `OP_DMA`(33) opcode）：

```
OP_PIN  [slot]                → 把权重 slot pin 进常驻区（一次）
OP_DMA  [src, dst, bytes, ...] → 换入/换出搬移（超预算时）
OP_MATMUL_*                   → 若权重已 pin，跳过 cpu_to_vtcm，直接引用常驻区
```

- 执行器 MATMUL 执行体：读「本 op 权重是否已 pin」的编译期判定（经 op_annot 或新注解槽，载体在 tasks 定），已 pin 则跳过 `cpu_to_vtcm`。
- `wt_w3_report` 增加 restage 计数，可与编译期规划值对账（SC-003）。
- **前提**: 依赖契约 B（pin_cap）决定常驻预算；D 的时序生产者 = 默认 greedy 路径新建（见 research R1）。

## 契约 E — 权重 crouton 编译期打包（conv2d + Q4_0 剩余）

**现状**: **f16 matmul 已由 008 T012 收口**（emit `try_prepack`/`kFlagWgtPP` flags bit3，exec `w_pp` DMA 线性预取，主路径零标量散写 pack）。剩余运行时 crouton 重排（`hmx_crouton_pos`）在 `exec_conv2d_hmx`（`oplist_exec.c:888-895`，f16 conv）与 `exec_matmul_f16_hmx` Q4_0 Route A（`oplist_exec.c:2033+`，运行时反量化+pack）。

**改动**: 编译期打包剩余权重槽为 crouton 布局，执行器删运行时 pack：

```
权重槽布局标签（新注解字段）: FP16_RAW | FP16_CROUTON(conv2d) | Q4_0_CROUTON
FP16_CROUTON  tile = 2048B（32×32 f16，hmx_crouton_pos 序）
Q4_0_CROUTON  tile = 640B（量化态重排，不反量化）
f16 matmul 复用 008 既有 kFlagWgtPP（flags bit3），不新增标签
```

- **Q4_0 内存守恒**：pack 发生在量化态（640B/tile），禁止先反量化成 f16（否则 4× 膨胀，破坏 4.3× 内存收益）。
- emit 侧新增 `weight_layout_pack`（net-new，`compiler/tools/`）；复用 `hmx_crouton_pos` 作为单一真值源。
- 执行器 `exec_conv2d_hmx`（`:888-895`）删 f16 conv pack 循环、Q4_0 Route A（`:2033+`）删反量化+pack，改直接 mxmem 加载预打包槽；Q4_0 保留 LUT dequant、仅删 permutation。
- **byte-exact**：编译期 pack 输出与运行期参考实现逐位一致（黄金对拍）。

## 回归契约

- 每项收口后 `build_examples.sh` 全量 byte-exact（35 例 + 261 门 + W4A16 256³ 65536/65536）。
- ver≤2 旧 blob（无 carve footer / 无辅助面标注 / 无 pin_cap / 未打包权重）执行器保持旧语义，回归门覆盖。
- 执行日志零 `fallback`、零按尺寸猜、零超规划 restage、零运行时 crouton 重排（SC-001）。
