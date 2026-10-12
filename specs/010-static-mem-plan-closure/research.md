# Research: 静态内存规划收口 — 四个开放项裁决

四个开放项来自 spec 的 Assumptions：D 的时序生产者、B 的 pin_cap 载体、C 的 carve 复用载体、E 的 crouton pack 载体与内存守恒。均已对代码核实并裁决。

## R1 — D 的 pin/restage 时序生产者：默认（greedy）路径新建，非 CP oracle

- **Decision**: 权重 pin/restage 时序由默认（greedy/mem_plan 家族）路径新建规划产出；CP oracle（`cp_solver.cpp`）不作为主产路径。
- **Rationale**: 关键澄清（对 gap 文档与 10-12 初版 spec 纠偏）：**权重是 const 张量，greedy 与 cp_solver 都 `skip` 它**（`graph_prepare.cpp:4776-4782` 的 `od.is_const() continue`）。cp_solver 的 `pin_slot`（`cp_solver.cpp:1071-1096`）位于**解校验器**（Eq.4/5/8/9/12 复算）内、是对 temp 张量 PAGE_IN 事件的校验记账，不是规划主输出；`cp_plans_ = to_spill_fill_plans(sol, keep)`（`graph_prepare.cpp:4875`）只覆盖 **temp 张量** spill/fill。**谁都不产「权重 slot 的 restage」**——这是全新规划面。把它挂在默认路径（而非 env 门控的 CP oracle）上，才与「执行器纯查表」目标同源、且默认路径即受益。
- **Alternatives considered**: ① 复活 CP oracle 主产权重时序 —— 放弃，理由：CP 问题集只含非 const temp 张量，权重根本不在其输入里；复活成本高且与默认路径二象。② 独立新 pass 完全从零 —— 采纳：新 pass 复用现有 lifetime 分析（`compute_lifetimes` / `vtcm_lifetime_alloc` 同源），但输出是「权重 slot 的 pin + restage 序列」，不进 `vtcm_allocations_`。
- **cp_solver 对家的正确表述**: cp_solver 的对家是**贪心 FancyAllocator**（两者都产出 `vtcm_allocations_`，同 `AllocResult` 契约），**不是 mem_plan**（`MemPlan` 是 DDR 侧偏移布局，`graph_prepare.cpp:698-710`，另一维度）。

## R2 — B 的 pin_cap 载体：TEMPOFF 槽头新增 u32 字段（字节）

- **Decision**: pin_cap 序列化为 TEMPOFF 槽数据头新增一个 u32（单位字节），头部 `[cap][reserve][n]` → `[cap][reserve][pin_cap][n]`；emit 与 exec 同步改、TEMPOFF 格式升版。
- **Rationale**: ① `reserve` 字段已拆满（低 16 位 = 表外 bump 预留 KB，高 16 位 = VTCM 池大小 KB），无空位；② `MemPlan`（`graph_prepare.cpp:704-710`）现无 pin_cap 字段，须新增并在 `vtcm_auto_scan` 里**真算**（而非固定 align2048 余量假设）；③ pin_cap 语义已核实为**字节**（`wtcache_impl.c:108,160-166`：`pin_cap_bytes`，0 → 默认 12MB，是 VTCM 底部权重常驻区容量）。硬编码 4096 实为 4KB，等于把权重常驻区压到趋零 —— B 不只是「参数化 4096」，是「让权重常驻区真正可用」。
- **Alternatives considered**: ① 复用 reserve 高 16 位 —— 不可行（已被池大小占）；② 另起新槽类型 —— 可行但割裂「内存配置」单一真值源，弃；③ 从 pool 反推 —— 不可行（pin_cap 是 pool 计算的独立输入）。
- **关系**: pin_cap（VTCM 底部权重常驻区）+ pool（VTCM 顶部 temp 驻留区）+ scratch（HMX arena）须 ≤ 16MB；`vtcm_auto_scan` 现公式 `pool = 16MB − scratch_max − head` 中的 `head` 含 pin_cap 余量假设，收口后改为 `pool = 16MB − scratch_max − pin_cap(真算)`。

## R3 — C 的 carve 复用载体：op_annot 保留位 bit5（per-op，零扩容）

- **Decision**: carve 复用标志用 per-op 引擎注解 footer（`op_annot`）的保留位 bit5，不扩 carve record。
- **Rationale**: ① `op_annot` 每 op 一条 u32，现只占 bit0-4，**bit5-31 空闲**，与「每 op 一个复用决策」天然对齐；② carve footer record 已是 5×u32=20B 满，扩 24B 会改变对齐与解析，成本高；③ 复用决策本质是「本 op 是否复用上一 op 的 carve」，编译期已知（形状 + live_top 都由 op 维与 carve footer 闭式算出），一个 bit 足够。
- **Alternatives considered**: ① carve record 扩 24B 加 reuse u32 —— 可行但动对齐/解析，弃；② 另起注解槽 —— 过度设计，弃。
- **语义**: `op_annot bit5 = 1` → 本 op 复用上一 carve（跳过 `dc_w4_carve` + 全 VTCM flush）；执行器据此删掉 `shape_same` 运行时判断（`oplist_exec.c:589-591`），改为读 bit5 + 编译期 live_top。

## R4 — E 的 crouton pack 编译期化载体与内存守恒

- **Decision**: 权重 crouton 打包在编译器 `compiler/tools/` 新增 `weight_layout_pack` 模块（**net-new**），复用 `hmx_crouton_pos`（`kernels/include/hmx_crouton.h:28`）作为布局公式的单一真值源；执行器删除 `exec_matmul_f16_hmx` / `exec_conv2d_hmx` 的运行时 pack 循环，改直接加载编译期已打包槽。
- **对 roadmap 纠偏**: roadmap 阶段 1 写「复用现成的 `repack_q4_0_crouton` 逻辑 + `weight_layout_pack.py`」——经核实**这些符号不存在**（grep 全 compiler/ 无 `repack_q4_0_crouton`/`weight_layout_pack`/`pack_oplist`）。唯一存在的是 `hmx_crouton_pos` 公式（`kernels/include/hmx_crouton.h`）。故 E 的打包模块与 Q4_0 repack 逻辑都要**新写**，不能假设有现成实现可复用。
- **Q4_0 内存守恒（硬约束）**: Q4_0 权重槽现为 640B/tile（0.5B/elem 量化态 + 64 scale），运行期做「LUT 反量化 + crouton pack」（`oplist_exec.c:1982-2008`）产出 f16 crouton 面（2048B/tile）。**编译期 pack 必须发生在量化态**（把 Q4_0 nibble 重排成 crouton 序，仍 640B/tile），不得先反量化成 f16 再存（否则权重槽 4× 膨胀，破坏 Q4_0 的 4.3× 内存收益，违背「Q4_0 内存收益已闭环」）；运行期保留 LUT dequant、仅删 permutation 步骤。
- **byte-exact 对齐**: 编译期 pack 输出必须与运行期参考实现（f16 `oplist_exec.c:2010-2019`、Q4_0 `:1982-2008`）逐位一致，以运行期逻辑为参考实现做黄金对拍。
- **收益口径**: E 的运行时节省是「permutation」这步；roadmap 的「20.6s/全图」是标量 pack 旧账，T017 LUT 已优化，剩余量待重测（SC-004 按重测基线）。
- **Alternatives considered**: ① Q4_0 先反量化再存 f16 crouton —— 弃（4× 内存膨胀，破坏 Q4_0 收益）；② 保留运行期 pack、只对 f16 路径做编译期化 —— 可作阶段内切分（f16 先做、Q4_0 后做），降低风险。

## 待 /speckit-tasks 细化的实现序

1. D（新建权重常驻规划，依赖 B 的 pin_cap）→ 2. A（最独立，含 K≠N 修）→ 3. B（pin_cap 真算+载体）→ 4. E（crouton pack 编译期化，f16 先、Q4_0 后）→ 5. C（bit5 载体，P2 可延后）→ 6. E2E 验收。

依赖链：D 依赖 B；E 的 f16 路径独立、Q4_0 路径独立；A 独立；C 独立。A/E(f16)/B 可并行。
