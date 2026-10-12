# 静态内存规划未打通的问题与目标

## 背景

编译器（host，逆向 libHtpPrepare）已经完成内存规划：ST-Cut 调度、FancyAllocator、
cp_solver 的 pin/remat/comp 决策、block table、tiling、指令选择。执行器（自研 skel，
`kernels/src/runtime/oplist_exec.c`）通过 wtop blob 消费这些规划。

但"编译器算好 → 落进 blob → 执行器机械执行"这条链未完全打通：执行器仍存在
**硬编码、按尺寸猜测、惰性判断**，与"编译器做极致静态规划、执行器只调算子搬运+计算"
的目标不符。

blob 格式（`kernels/include/oplist_parse.h`）已演进到 v3，具备大部分承载机制
（engine 注解 footer、TEMPOFF 静态 temp 表、carve footer、OP_DMA/SPILL/FILL、STATE 槽），
且执行器已消费其中大部分。剩余未打通的是以下 4 个收口点。

---

## 问题清单

### 问题 A：W4A16 辅助面身份未显式标注

- **现象**：`OP_MATMUL_W4A16` 的 bias / atbl / otbl 三个辅助面没有显式 slot id，
  执行器在 slot 表里按尺寸（len）反推哪个槽是 bias、哪个是 act_table、哪个是 out_table。
- **证据**：`kernels/src/runtime/oplist_exec.c:539-558`（`bias_b = (N/32)*512`，
  `tbl_b = 8*(K/32)*4`，逐槽匹配 `if (L == bias_b) bias = p`）。
- **影响**：执行器做运行时"猜身份"，编译器明明知道却未标注；猜错即错。
  K≠N 时必错：`tbl_b = 8*(K/32)*4`（exec_matmul:539）同时匹配 atbl 与 otbl，
  但 otbl 实际长 `8*(N/32)*4`（w4_carve_need:511），matcher 找不到 otbl →
  误报 "supply slots missing"；且 atbl/otbl 同长，身份完全靠槽顺序兜底。

### 问题 B：wtcache pin_cap 硬编码

- **现象**：执行器 `wtcache_open(wc, 4096)` 的 pin_cap 写死为 4096，编译器侧
  memplan 已算出 pin_cap 余量，但未写入 blob。
- **证据**：`kernels/src/runtime/oplist_exec.c:315, 590`（两处硬编码 4096）；
  编译器 `compiler/src/ir/graph_prepare.cpp:670` 只在 `vtcm_auto_scan` 里把
  pin_cap 当固定余量假设（align2048）来算池大小，并未真正算出/存字段。
- **影响**：编译期对权重常驻区的规划无法下发，执行器固定一个值。
- **纠错**：`MemPlan` 结构（graph_prepare.cpp:704-710）只有 ddr_cap/vtcm_cap/
  bump_reserve/entries/spills，无 pin_cap 字段。故不是「编译器已算好只差序列化」，
  而是「编译器未算、执行器写死 4096」——收口 B 需先在编译期真算出 pin_cap。

### 问题 C：carve 复用决策运行时判断

- **现象**：执行器用 `shape_same` 在运行时判断形状是否变化，从而决定是否重新
  carve；编译器完全可以在编译期预判（它知道每个 op 的形状）。
- **证据**：`kernels/src/runtime/oplist_exec.c:579`（`shape_same` 判断）。
- **影响**：残留运行时形状判断，未完全静态化。

### 问题 D：权重 restage 时序未静态化

- **现象**：每次 MATMUL op 重新 `cpu_to_vtcm` 搬运权重（restage），未按编译期
  pin 决策复用已常驻的权重。
- **证据**：`kernels/src/runtime/oplist_exec.c` MATMUL 执行体；编译器侧
  `compiler/src/vtcm/cp_solver.cpp:1096`（`pin_slot` 决策）未完整落到执行。
- **影响**：权重重复搬运，违背"pin 后复用"的静态规划；编译期 pin 决策白算。

---

## 目标

### 目标态

编译器把内存决策**全部**定死进 blob（辅助面身份、pin_cap、carve 复用、pin/restage
时序），执行器 `wt_exec_run` 退化为**纯查表 + 机械执行**，删除所有硬编码、按尺寸猜、
惰性/运行时判断。

> 前提（见「已确认」）：B 的 pin_cap 编译期尚未算出（MemPlan 无字段），须先补算；
> D 的 pin/restage 时序须先定 cp_solver 与主线 mem_plan 谁主产（两套并行规划）。

### 逐问题收口目标

| # | 目标 |
|---|---|
| A | `OP_MATMUL_W4A16` 显式携带 `bias_slot/atbl_slot/otbl_slot`，执行器删掉按 len 猜 |
| B | 编译期真算出 pin_cap 并写入 blob（TEMPOFF reserve 高 16 位是"池大小 KB"非 pin_cap，须另起槽/扩字段），执行器读出替代 4096 |
| C | 编译期预判 carve 复用（哪些连续 op 可复用），写入 carve footer / 注解，执行器删 `shape_same` |
| D | 编译期决定 pin 权重与 restage 时序，显式发射 `OP_PIN` + `OP_DMA` 序列，执行器按序列执行、删除无条件 restage |

---

## 验收标准

1. **回归不破**：每项收口后 `build_examples.sh` 全量回归保持 byte-exact
   （35 例 + 261 门；W4A16 256³ 的 65536/65536 bit-exact 门保持）。
2. **零运行时决策**：执行日志中零 `fallback`、零按尺寸猜、零超出编译期规划的 restage。
3. **可观测**：`wt_w3_report` 统计里 restage 次数 == 编译期规划值，可作为门。
4. **端到端不回退**：qwen3-0.8b 模型端到端编译 + 上板验证通过，精度（cos / top1）
   与 wall-clock 性能均不回退（对照当前库锚 cos 0.998642 / 35.31s，见跨会话协调表）。

---

## 已确认（原「待确认」，已对代码核实）

1. **pin_cap 未存进结构**：`MemPlan`（graph_prepare.cpp:704-710）无 pin_cap 字段，
   cp_solver 之外编译期根本没算 pin_cap，只有 `vtcm_auto_scan` 里的固定余量假设。
   收口 B 需从「算出 pin_cap」做起，不是只差序列化。
2. **carve footer 实际是 5×u32 = 20B**，非 `{op_index, vtcm_offset, size}` 三字段：
   `{op_index, vtcm_offset, size_bytes, residency_start, residency_end}`
   （wtop_ops.hpp:97-103 定义，wtop_emit.cpp:860-864 写入）。20B 已被 5 字段用满，
   加 reuse 标志要么扩 record 到 24B、要么另起注解槽。

---

## 收口动作对照表（对代码核实后）

| # | 现状（已核实） | 收口动作 | 依赖/风险 |
|---|---|---|---|
| A | exec_matmul 按 len 猜 bias/atbl/otbl；`tbl_b` 用 K 同时匹配 atbl+otbl，K≠N 必误判 | `OP_MATMUL_W4A16` 显式带 bias_slot/atbl_slot/otbl_slot，删按 len 猜 | arity 6→9 需 bump；emit 侧已知三槽身份，改动局部 |
| B | 执行器两处写死 pin_cap=4096；编译期未算 pin_cap（MemPlan 无字段） | 编译期真算出 pin_cap → 写 blob → 执行器读出替 4096 | reserve 字段已占，需新槽/扩字段；先定 pin_cap 语义 |
| C | carve 复用靠 shape_same(M/K/N) 运行时判 | 编译期预判连续可复用 op → 写 carve footer/注解，删 shape_same | carve 记录 20B 已满，需扩 24B 或新注解槽 |
| D | MATMUL 每 op 无条件 cpu_to_vtcm restage；cp_solver pin 决策 env 门控、未接 emit | 定架构：cp_solver 输出发射 OP_PIN+OP_DMA，或主线 mem_plan 产 pin/restage 时序；执行器删无条件 restage | **全局架构决策**：cp_solver 与 mem_plan 是两套并行规划，先定谁主产 |

**备注**：验收标准 1（byte-exact 回归）对 A/C/D 尤其硬 —— 删运行时判断后，emit 侧须先与旧行为逐字节对齐；A 的 K≠N 触发是现存潜在 bug，收口 A 顺带修。
