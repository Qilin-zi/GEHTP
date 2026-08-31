# 反汇编修正计划

> 目标：把 4 个函数中所有**编造内容**替换为**实际反汇编结果**。
> 原则：每一段实现都必须有对应的反汇编指令地址作为依据，不得凭推断。
> 本文件是唯一真相源，每完成一项打 [x]。

.so 路径（本机）：`/Users/apple/Documents/ge/real_so/libHtpPrepare.so`（SDK 2.48.40.260702，x86-64 host 端 HTP 图编译器，98MB）
符号清单：`audit_verify/symbol_coverage.csv`（5758 行）；asm 转储：`audit_verify/asm/`

---

## 现状总览（2026-08-28 更新）

### 1. 管线解码进度

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 早期审计（M12–M15） | ✅ | 4 个编造重灾区函数修正（pools/tcm/plugin/spill pass） |
| P1 调度管线解码（M17–M30） | ✅ | schedule_for_alloc@0x1302900 全链逐指令贯通 |
| P1 重实现（M31） | ✅ | st_cut.hpp/.cpp ~1100 行，逐段 [0x地址] 注释 |
| 提案器指令级替换（M32） | ✅ | 0x130ab30 全函数解码，概要级替身废弃 |
| P2–P6 后续管线 | [~] | M36+M36b+M37+M38+M39 两批完成（supertiles/phys_alloc 全解；sequencing_stage 骨架+三集成点；alloc 链桥接全解+internal 主体六段+**0xE=VTCM 装箱失败→CBS 重试来源闭环**）；M38b allocate_io_tensors@0xf69940 全解；M40 变体 B 装箱 pass 循环骨架（ctx 字段图）；**M41 放置核心 0x13a4d10 全解（first-fit/best-fit 变粒度装箱+entry 字段全图）+ 策略步进 0x13a7290 全解（大件/普通分区+桶压峰值+策略梯步进）**；M36c tile 子系统（tiler.h DSL/TileShapeBase）全解+V81"25 参"核验（未证实）；**M42 tcm_migration@0x13219c0 外壳全解（三分之四规则已证实 [0x1321bd4]；3472/3459 判决+六遍 op 扫描+错误路径 1819/1824/1829）**；**M43 四策略排序助手全解（FFD 键梯：纯尺寸/阈值双档/峰值/混合）**；**M44 驱动相位全解（0x13b5e80 ctx 构造+38 槽策略梯 .rodata 表+趟执行器 0x13b1470：f0/f1=受限试放上界档位、B 跳冗余前缀从槽 24 起梯、A 从槽 0、检查点/溢出恢复）**；**M45 四放置包装全解（0x13a98f0/0x13a9660/0x13aa750/0x13a78d0 外壳：先受限后全量、重叠消解、块表 u16@+2 落盘、ctx+0x1e0 物化）**；**M46 边界物化 0x13a78d0 外壳+数据流（0x13a5500 生命周期排序+按止桶分组并区间+ctx+0x200 组记录→派生 0x1e0/0x1f8/0x210）**；**M47 spill/fill 四层全解（insert_spill_fill 永久桩+SFCD 二进制格式+slc rec_type 三值枚举+池 2 三槽/near-far 位；"六类"证伪）**；余量见 F3/M40+/M42b/G4b |

**P1 全链（已贯通）**：schedule_for_alloc（选项 CSV/预算/重试循环 1303ee0-13044e3）
→ full_schedule 择优恢复 → prep_round（五表快照 0x1307780 / aux 曲线 0x130cea0）
→ develop_schedule 0x130d3e0（评分扫描 → 提案 0x130ab30 → 执行 0x1309a60
→ 修正 0x130b9a0 → 判定 0x130d050 → 分裂 0x130b600 → 随机重启 0x130bc70；
终段 Kahn 原地重写工作序）→ 建图层（connect_nodes 0x1300cc0 替换对 /
洪泛 0x1309230 / 搬运 0x13096b0 / 迭代 0x1309810 / 重放 0x13080a0）。

### 2. 重实现验证状态

- 代码：`REQNNFRAME/REQNN/include/hnnx/scheduler/st_cut.hpp` + `src/scheduler/st_cut.cpp`
  + `tests/test_st_cut.cpp`（CMake 已挂 htp_core / test_st_cut 目标）
- 测试 **7/7 通过**：槽互指累加、洪泛+搬运、grain/aux/峰值、验证器合法/倒置序、
  replay 三 case、develop_schedule 区间守恒+排列不变、full_schedule 端到端排列不变
- `g++ -Wall -Wextra` 零警告；ASan 干净（本机无 cmake 二进制，直编为准）

### 3. 遗留「反汇编未完全理解」清单（按影响排序）

1. ~~候选表 ctx+0x240 填充方~~ ✅ M33 已解（stcut_read_nodes@0x12fc820，见 F2/M33 条目）。
2. 标记原语内部：0x1305130（=stcut_add_dependencies，段名实证）层号推进边界；
   0x1308c50 收尾段 1308f80-130922f。
3. iterate 收敛性：.so 无迭代上限，依赖上游不变式；实现加 guard_cap 防御并标注。
4. 执行器 arg4 语义（现以 rec.sub_entry 代入）。
5. [ctx+0x278] 阈值基址语义（疑似总可用内存，未证）。
6. 0x12fffb0 排序方向（实现取降序）；0x130bc70 中后段 130c519-130cb4d。
7. 0x1450f0 等效除数；[rsp+0x1530]/[rsp+0x1f0] 写入点。
8. 杂项：map A（ctx+0x228）装载期插入方在 load_replacement_plan 主体
   （0x1299ff4–0x12f0000，未转储）；初始表 flags&4 分布、CSV 逐轮重载、
   PRNG（LCG 等价）、delay_dma_again 0x12fc740 / convert_to_ids（地址留档）。

---

## 当前状态审计（编造项清单）

### #2 make_persistent_pools @ 0xf455c0 (8477B) — 最严重 ✅ 已修正 (Phase A 完成)
- 符号权威地址：`0xf455c0`（sym_master.txt；旧记 0xF453D0 为错，导致注释地址偏移 +0x200）
- 完整反汇编报告：`audit_verify/reports/M12_make_persistent_pools_disasm.md`
- 已反汇编：入口 / 主循环 / 3 次 allocate_new_pool / gather_const_blocks / hash table / dynamic_cast 全流程
- 修正结论：
  1. **persistent vs scratch 判断**：block 分类用 `persistent_block_desc.flags(+0x10)` 的
     `0x10000000`（persistent 位，0xf45a03）与 `0x40000000`（replaceable 位，0xf45a97）。
     **`__dynamic_cast`（0xf46f1a）只用于 `is_dynamic_inputs_active`（0xf46f2b），不参与 block 分类**。
     原"反汇编用 dynamic_cast 检查 tensor 类型"是误述。
  2. **3 次 allocate_new_pool 参数**（均已反汇编）：
     - #1 0xf46135 persistent：pool_id=`[desc+0x14]`(数据驱动), flags=0xf, bool=1
     - #2 0xf46a81 replaceable：pool_id=`[desc+0x14]`, flags=0xf, bool=1（后接 weight_compression + placeholder_mapping）
     - #3 0xf47365 scratch：pool_id=`0x100`(固定), size=`0x100`(固定), flags=0xf, **bool=0**（后接 memset）
  3. **128 对齐**：**本函数内不存在 128 对齐指令**。`and $0x1fffe` 是 hash 探针步长，
     `and $0xfffffff0` 是 hash 槽位计数 16B 对齐。pool size 对齐在 allocate_new_pool(0x6eb370) 内部。
     原 128 对齐已从 fancy_allocator.cpp 删除。
  4. **block 遍历顺序**：倒序（0xf45950/0xf45998），enabled flag 在数组元素 +0x9（步长 0x10）。
  5. **调用顺序**：gather_const_blocks(0xf4603f, r8d=0x60000000/0xff0000/0) →
     hash table 插入(fib_hash 0x740f1de9 + 线性探测) → 扩容(2^(bsr+3)) →
     allocate_new_pool #1/#2 → dynamic_cast → 2nd pass → allocate_new_pool #3。

### #1 allocate_tcm_blocks_internal @ 0x13b2bc0 (12155B) ✅ 已修正 (Phase B 完成)
- 符号权威地址：`0x13b2bc0`（sym_master.txt；旧记 0x13B29D0 为错）
- 真实范围 0x13b2bc0-0x13b5b3b；objdump 文件含后续相邻函数（13b5b40+ 不属于本函数）
- 完整报告：`audit_verify/reports/M13_allocate_tcm_blocks_internal_disasm.md`
- 修正结论：
  1. **free list**：函数范围内**无** force_contiguous/link_blocks/free-list 调用（link_blocks 在 0x13b83bb，属后续函数）。原"删除编造 free list"方向正确，已确认。
  2. **bump pointer 分配**：2048 对齐**确认**。0x13b4e75 `and $0xfffff800`（align_up）；0x13b3291（24B/槽累加内 align down）。
  3. **budget check**：真实位置 0x13b32b2（cmp [r12+0x48] vs total）→ 0x13b5336（error 0xd）→ 0x13b535f（qnndsp_log）。**旧注释 0x13b5123 是 hash 表扩容，不是 budget check，已删**。
  4. 旧注释 4 处编造地址（sort 0x13b3c0b→0x13b3dfb、align 0x13b4c85→0x13b4e75、imul 0x13b3208→0x13b33f8、STAT 0x13b30d5→0x13b32bb）均已修正。

### #4 run_plugin_rewrites @ 0x10d87e0 (3381B) ✅ 已修正 (Phase C 完成)
- 符号权威地址：`0x10d87e0`（sym_master.txt；旧记 0x10D85F0 为错，注释地址偏移 -0x1F0）
- 完整反汇编报告：`audit_verify/reports/M15_run_plugin_rewrites_disasm.md`
- 修正结论：
  1. **获取注册表**：`0x10d8809 call 0x10d8700`（静态局部函数，无符号），非 0x10d8510。
  2. **early/late**：`0x10d8838` lea "early"(0x567933b) / `0x10d883f` lea "late"(0x461d8ea)；
     `0x10d8846` test bl / `0x10d8848` cmovne — bl!=0(late_phase=true)→"early"。字符串内容已 rodata 确认。
  3. **DCE 与 collect_deletable 互斥**（旧注释"总是调用 collect_deletable"是编造）：
     `0x10d92f7` cmpb [this+0x6ea4] → `0x10d92ff` je：flag==0→`collect_deletable_nodes`(0x10d9313)；
     flag!=0→`remove_dead_code(false)`(0x10d9306)。
  4. supersede_op 调用点：0x10d8f4e/0x10d8f6b（主分支）、0x10d90ef/0x10d927a（$MultiTemp 分支）。

### #3 post_spill_fill_design_pass @ 0x129ed30 (768B) ✅ 已修正 (Phase D 完成)
- 符号权威地址：`0x129ed30`（sym_master.txt；旧记 0x129EB40 为错，注释地址偏移 -0x1F0）
- 完整反汇编报告：`audit_verify/reports/M14_post_spill_fill_design_pass_disasm.md`
- 修正结论：
  1. **子函数是静态局部函数，无符号**：0x12932c0/0x129eec0/0x1299cb0/0x1299cf0 全部不存在。
     真实目标：0x12934b0（主 sequencing）、0x1298d00、0x129f0b0、0x1299ea0/0x1299ee0/0x1299f20/0x1299dc0。
     sym_master 在 0x1290000–0x129f800 区间只有 post_spill_fill_design_pass 一个符号。
  2. **判断逻辑**：`0x129edbb mov 0x6120(%r15)` / `0x129edc4 jle 0x129ee5a` —— `[this+0x6120]` op count。
     已修正回 op count 检查（非"vtcm_requirement > vtcm_size"）。
  3. **spill_fill_needed_ 标志不存在**：函数内只有 `[this+0x611d]`(0x129ed4d) 与 `[this+0x611e]`(0x129ee5a)。
  4. objdump 将 0x12934b0 等误标为 `sequencing_stage@@Base+0xb2410`，但 sym_master 确认
     sequencing_stage 真实地址是 0x11e10a0 —— function-boundary overlap。

---

## 执行计划

### 阶段 A: #2 make_persistent_pools（最严重，优先）✅ 完成

**A1. 反汇编入口段 0xf455c0 - 0xf45600** [x]
- 主循环入口 0xf45950、block 倒序遍历 0xf45998、enabled flag 0xf459a2
- dynamic_cast 调用点 0xf46f1a（仅用于 is_dynamic_inputs_active，不参与 block 分类）

**A2. 反汇编中段 0xf45950 - 0xf4726a** [x]
- gather_const_blocks 0xf4603f（r8d=0x60000000/0xff0000/0）
- 3 次 allocate_new_pool 参数（见上，pool_id 数据驱动/固定、bool 1/1/0）
- apply_weight_compression 0xf46ac8、set_placeholder_mapping 0xf46bde

**A3. 反汇编尾段 0xf47346 - 0xf47374** [x]
- 对齐结论：**无 128 对齐**；`and $0x1fffe`=探针步长、`and $0xfffffff0`=槽位计数 16B 对齐

**A4. 修正实现** [x]
- 注释地址全部改为 0xf455c0 基准（原偏移 +0x200 已修）
- 删除编造的 128 对齐、pool_id=0/1 固定猜测
- 修正 dynamic_cast 误述（改为"仅 is_dynamic_inputs_active"）
- 补充 hash table（fib_hash 0x740f1de9 + 线性探测 + 扩容）机制

### 阶段 B: #1 allocate_tcm_blocks_internal ✅ 完成

**B1. 反汇编中段（主循环）** [x]
- 无 free list 调用（已确认删除编造的 free list 逻辑）
- bump pointer + 2048 对齐（0x13b4e75 / 0x13b3291）+ hash probe（imul 0x740f1de9）确认

**B2. 反汇编尾段（budget check + 统计）** [x]
- 真实 budget check：0x13b32b2/0x13b5336/0x13b535f
- 真实 STAT：0x13b32bb/0x13b32e9/0x13b32f1
- 发现旧注释把 hash 表扩容(0x13b5123)误标为 budget check

**B3. 修正实现** [x]
- 4 处编造地址修正（sort/align/imul/budget+STAT）
- free list 确认无、2048 对齐确认

### 阶段 C: #4 run_plugin_rewrites ✅ 完成

**C1. 反汇编入口段** [x]
- early/late 分支点：0x10d8838 "early"(0x567933b) / 0x10d883f "late"(0x461d8ea) /
  0x10d8846 test bl / 0x10d8848 cmovne（bl!=0→"early"）
- 注册表检查：0x10d8809 call 0x10d8700 / 0x10d880e cmp [rax+0x18] / 0x10d8813 je 0x10d934b

**C2. 反汇编中段（主循环）** [x]
- 主循环 0x10d8891（mov [r13+0x6d58]）→ 0x10d92e0
- supersede_op 调用点：0x10d8f4e/0x10d8f6b/0x10d90ef/0x10d927a

**C3. 反汇编尾段（$MultiTemp + DCE）** [x]
- DCE 互斥分支：0x10d92f7 cmp [this+0x6ea4] → 0x10d92ff je → collect_deletable(0x10d9313)
  / remove_dead_code(0x10d9306)

**C4. 修正实现** [x]
- early/late 字符串内容 + cmovne 语义已确认
- DCE 与 collect_deletable 互斥（删除"总是调用"编造）
- 全函数地址 +0x1F0（基址 0x10D85F0 → 0x10d87e0）

### 阶段 D: #3 post_spill_fill_design_pass ✅ 完成

**D1. 子函数确认为无符号静态局部函数** [x]
- 0x12932c0 不存在；真实主阶段 = 0x12934b0（静态局部函数）
- objdump 误标为 sequencing_stage@@Base+0xb2410，但 sym_master 确认 sequencing_stage 在 0x11e10a0

**D2. design pass 调用点** [x]
- 0x129ee15 call 0x129f0b0（构建 5 字符串 vector，静态局部函数）

**D3. cleanup 调用点** [x]
- 0x129ee24 call 0x1299ea0 / 0x129ee33 call 0x1299ee0 / 0x129ef30 call 0x1299f20 / 0x129effc call 0x1299dc0

**D4. 修正实现** [x]
- 判断逻辑改回 [this+0x6120] op count（0x129edbb/0x129edc4）
- 删除 spill_fill_needed_ 标志
- 全函数地址 +0x1F0（基址 0x129EB40 → 0x129ed30）

### 阶段 F: 调度/内存/spill 管线（P1）✅ 解码+重实现完成（M17–M32）

> 2026-08-24 用户指示：先列全函数清单，再聚焦「调度/内存/spill」管线逐指令反汇编重建，
> 顶层 IR 优化 pass 暂时跳过。清单见 `audit_verify/reports/symbol_coverage.csv`（5758 行）
> 与 `M16_symbol_coverage.md`。
> **2026-08-28 状态**：schedule_for_alloc 全链（M17–M30 解码 → M31 重实现 →
> M32 提案器指令级替换）完成；测试 7/7；遗留未解项见顶部「现状总览 §3」。

**F1. 全符号清单** [x]
- symbol_coverage.csv（demangled/class/addr/size/mangled 5 列，5758 行）
- M16：顶层类分布 + P0–P6 管线反汇编顺序 + 跳过清单（IR opt pass）

**F2. schedule_for_alloc @0x1302900（调度器本体）第一轮** [x]
- 报告：`audit_verify/reports/M17_schedule_for_alloc_disasm.md`
- 确证：基址 0x1302900（reference 0x1302710 错 +0x1F0）；两层主循环（①哈希键 BST 分组桶
  ②逐 op MAC 门控）；PCG/LCG 常数 0x5851f42d4c957f2d/0x14057b7ef767814f = **调度键哈希**
  （非 tie-break）；MAC 门控 0xf4240/0xb2d05e00/0x1450f0/−0x81650；$0x800=snprintf 缓冲
  （非 2KB 对齐，纠正 M05）；6 个布尔选项 flag（0x5638/0x5650/0x5668/0x5680/0x5698/0x56b0）
- ⚠️ 新发现编造：`graph_prepare.cpp:1678-1786` 把 schedule_for_alloc 的 PCG 哈希当成
  order_nodes 的「同层 tie-break 随机打散」——**待修正**（PCG 哈希实为调度键 BST 插入）。
- 待办：~40 个无符号静态局部 helper 逐个反汇编（0x1306750/0x13065f0/0x1306a20/0x130ebe0/
  0x1307200/0x1301040/0x130ea30/0x130ff40/0x1310fc0/0x13111b0/0x13112e0/0x13125e0/
  0x1312cc0/0x1312e00/0x1313030、0x855470 簇、0x868c50 簇、0x8f02c0 簇）。
- ✅ M18 已完成 4 个 helper：0x13065f0（结果构造/max）、0x1306750（7 参 wrapper）、
  0x1306a20（红黑树去重+过滤+闭包）、0x1307170（ctx 析构），并给出 ctx 结构布局。
- ✅ M19 已完成 5 个 helper：0x130cea0（前缀和）、0x130d050（直方图+阈值 bool）、
  0x130d3e0（真实 7 参调度核心）、0x130ea30（vector<pair> push_back）、0x130eb50（strtod 选项解析）。
- ✅ M20 已完成 9 个 helper（含跨文件）+ 3 处对 M17/M19 的纠正，见
  `M20_schedule_for_alloc_helpers_disasm2.md`：
  - 0x868c50 = `__tree_balance_after_insert`（**非权重回调**，纠正 M19）；权重实由
    0x130ccd0 计算（`Σ加项−Σ减项`）写入树节点 +0x28。
  - 0x1305130（过滤式索引收集）、0x130b840（vector<64B> push_back）、0x130ab30（按 u16 标签收集同组）、
    0x13125e0（introsort）。
  - 0x855470/0x855400/0x8553f0 = **异常抛出**（bad_array_new_length / length_error，非 op 调度，纠正 M17 §8）；
    0xd79830 = **Java Random LCG PRNG**（0x5DEECE66D，非 mutex 解锁，纠正 M17/M19）；
    0x8f02c0 = vector<24B> move-merge（÷3 魔数）。
- ✅ M21 已完成剩余 5 个文件内 helper，见 `M21_schedule_for_alloc_helpers_disasm3.md`：
  - 0x1307200 = 权重树查找/插入 + 前缀和累加（对 [ctx+0x158] 每索引查 node->count，未命中则插 48B 节点 count=0 并 rebalance）。
  - 0x1307780 = 一次性物化 5 个源 vector（+0x68/+0x80/+0xb0/+0xe0/+0xf8）→ 工作副本（+0x190/+0x1a8/+0x1c0/+0x1f0/+0x208），[ctx+0x220] 一次性门。
  - 0x13087f0 = 从 add/subtract 集 swap-with-last 移除 op + filter 置 0x80 + 追加 [ctx+0x178]。
  - 0x1309a60 = spill 候选筛选：memset [ctx+0x128] 清零 → 按权重阈值（[ctx+0xc8]）收集 → 置 1。
  - **0x130ebe0 = 逗号分隔整数域解析器（strtol，跳过 N 逗号返回 bool），非「6 选项 flag 检查器」——纠正 M17 §4/§8**。
  - 解开 M20 遗留：[ctx+0x128] u16 三态写入方=0x1309a60（0→memset，1→1309dcf）；权重表 24B/slot 全字段
    （+0x0 field0/组 id、+0x4 idx、+0x8 weight u64、+0x10 filter bit0x4=纳入/bit0x80=已移除）；新增
    [ctx+0xc8] 阈值数组、[ctx+0x80] 组标志字节数组、[ctx+0x98] u16 tag 数组。
  - 节点 48B 布局修正：+0x30 是 ctx「树大小」计数器（非节点域，M20 记「+0x30 extra」为误）。
- ✅ M22 已完成 M21 §8 全部 5 项，见 `M22_schedule_for_alloc_helpers_disasm4.md`：
  - **0x1309940** = load_replacement_plan 包装器（阈值 0x5f5e0ff=99999999，4 次 call 0x1300cc0，
    尾调用 0x1309810）。
  - **0x13080a0** = 6-case 记录重放 dispatch（24B 记录 op/value/type，逆序 pop，跳转表 0x55b3d04
    已提取：type0 置权重/type1 日志/type2 移出 add 集/type3 pop 计数/type4 加入 add 集/type5 日志）。
  - **0x853eb0** = `vector<u32>::push_back` 冷路径（快路径内联，7 处调用点 push 一个 u32）。
  - **[ctx+0x128] ≥2 递增写入方不存在**：全区域仅 `movw $0x1`（5 处）+ memset 清零，无 incw/addw；
    `cmpw $0x1;ja`（>1 跳过）无生产者，属死代码。ctx = 栈局部 &[rsp+0x1290]（13047f9/13049eb），
    排除外部写入方。
  - **0x1307200 返回值消费方 = qnndsp_log 调试统计**（1304801/13049f3，非 0x130d3e0）——
    纠正 M21 §1.5「待与 0x130d3e0 交叉」。
- ✅ M23 已完成 0x1300cc0 + 0x1309810，见 `M23_replacement_pair_core_and_iteration_driver_disasm5.md`：
  - **0x1300cc0** = 替换对记录核心（u32(ctx,arg2,arg3,threshold,callback)）。在 add集[arg2] 找
    field0==arg3 的槽：命中则累加权重 `min(threshold+旧,99999999)` 返回；未命中按 [ctx+0xc] 分
    快路径（复用 [ctx+0x178] 空闲槽栈 pop 2 索引）与慢路径（push_back 2 槽），两路均建
    slotA(field0=arg3)/slotB(field0=arg2) 双向互指对 + 清零 [ctx+0xc8] 两项 + 同步 [ctx+0x110]
    数组到槽数 + 回调 0x1307f60 记 {槽,值,type}。公共尾 0x1307820/0x1307960 各 2 次建双向链。
  - **0x1309810** = 调度迭代驱动：清 [ctx+0xc8] 阈值数组 → 首次 0x1309230 探测 → 循环A 反复
    0x13096b0 批量处理（累计处理数 r12 + [ctx+0x288] 计时）→ 循环B 0x1309230 探测（[ctx+0x290]
    计时），非0 回循环A，返回 r12。rdtsc 双采样净效果 = 耗时/16。
  - **命名纠正（M22）**：objdump 标签 `load_replacement_plan@@Base+0x72b0` 是回退标签，非调用
    关系——`load_replacement_plan(DataReader&) @0x12f9820`（486B）不调用 0x1300cc0，它反序列化
    进 `this->[0x72e0]`。0x1300cc0 的真实调用方是 0x1309940（4次，被 0x1309a60 spill 候选筛选
    调用 2 次），属调度/替换管线。
- ✅ M24 已完成 0x1309230 + 0x13096b0 + 0x1300cc0 下游 6 helper + [ctx+0xc] 写方闭环，
  见 `M24_flood_collect_and_weight_pull_disasm6.md`：
  - **0x1309230** = 组可达洪泛收集：清 [ctx+0x128] 三态表与 [ctx+0x110] 位图 → seed 打 tag=1 →
    逐代遍历 frontier，内层遍历 add集[成员]：目标 tag==0、[ctx+0xc8] 阈值<权重、[ctx+0x80]
    bit0x2 组约束通过则收集（打 tag=1、置位位图、计数 r8++、触达标志 |= field0==target）；
    触达 target 或 frontier 空则停，返回收集计数（130962f）。
  - **0x13096b0** = 递归容量受限权重搬运：遍历 add集[seed]，[ctx+0x110] 位图 bt 测 pending，
    可用量 = 权重表[ebx].weight − [ctx+0xc8][ebx]，clamp min(可用量,cap)（1309783 cmovge），
    递归 call 自身于 field0；返回量入账本（成员 +、idx 对槽 −，13097b3/13097cc）；单步
    <10000（0x2710）提前返回。返回累计搬运量。
  - **下游 helper**：0x1307820 = add集([ctx+0xe0]) push_back；0x1307960 = subtract集
    ([ctx+0xf8]) push_back；0x1307f60 = 24B 账本记录 push_back（1.5x 扩容，/3 魔数
    0xAAAAAAAAAAAAAAAB，返回新记录指针 end−0x18）；0x1307aa0 = 节点槽退役（清两集、
    [ctx+0x80]=0x83、push [ctx+0x160] 节点槽空闲表）；0x1301040 = 节点注册
    （[ctx+0xc]!=0 追加 [ctx+0x68]，==0 复用 pop [ctx+0x160]，flags 字节入 [ctx+0x88] 账本）；
    0x1307c90/0x1307d90 = 替换对重指向 A/B 侧（swap-with-last 移除 + 重建 add/subtract 链 +
    field0 改指）。
  - **[ctx+0xc] 闭环**：ctx 构造函数 0x12fba20（schedule_for_alloc 1302a3b 调用）12fba54
    `movl $0x40` 初始化（建表期）→ 0x1307780 快照函数 1307812 清 0（复用期）+ 130d1c2 清 0；
    全范围无其他写指令。0x1300cc0 内 orl $0x4 写的是权重表 filter(+0x10)，不回写 [ctx+0xc]。
  - **新 ctx 字段**：+0x160/0x168/0x170 节点槽空闲表 vector（0x1307aa0 push / 0x1301040 pop，
    与权重槽空闲表 +0x178/0x180 是两个表）；+0x190/0x1a8/0x1c0/0x1f0/0x208 五张阶段快照表
    （0x1307780 一次性拷贝）；+0x220 快照守卫；+0x2a0 洪泛访问计数。
- ✅ M25 原函数名恢复（st_cut.cc，18 个 stcut_* 名），见 `M25_st_cut_name_recovery.md`：
  - **方法**：二进制 strip 掉 .symtab（仅 5758 .dynsym 导出），但 qnndsp 日志宏把 __func__
    字符串嵌进 .rodata（源文件名 `st_cut.cc`@0x55b3db9 证实 CU）。计时器 0x130ea30 已反汇编
    证实 = `vector<pair<const char*,u64>>::push_back({name, rdtsc>>4})`（16B 元素，2x 扩容，
    旧块 operator delete），名字标注其**后**的 stage 调用（SCHED_CYCLES 差分数据源）。
  - **原名→地址（M26 修正后 12 个；配对规则=计时→调用距离≤0x30 且目标在 helper 簇
    0x12fc000–0x1307000）**：stcut_read_nodes=0x12fc820、stcut_collate_sibs=0x12fd0e0、
    stcut_node_hash=0x1300210、stcut_quick_early_sort=0x1300600、stcut_block_relate=0x1300a40、
    **stcut_connect_nodes=0x1300cc0（M23「load_replacement_plan 静态核心」误标更正；名字与
    M23 功能解读一致：连接节点 arg2→arg3 建替换对并设阈值）**、stcut_add_dependencies=0x1305130、
    stcut_strong_relevel=0x12fd600、stcut_arrange_sibs=0x12fffd0、stcut_clean_sibs=0x13056f0、
    stcut_delay_dma=0x13065f0、stcut_delay_dma_again=0x12fc740。
    内联/未定：stcut_relate_by_tensor、stcut_topo_presort、stcut_measure_peak
    （初版误配 0x130ebe0——实为 CSV 整数解析器，M26 更正）、stcut_full_schedule
    （SCHED_ITERS/FlowRetry 重试循环）、stcut_convert_to_ids（初版 0x13065f0@13047c3 作废）。
  - **选项解析器家族（M26）**：SCHED_OPTIONS（OT/IT/Rg/Rt/AB/AM/EO/HD/DD/TR/LT/RC/RP/DB）
    从 CSV 解析，1303ee9–1304060 条件调用（GraphPrepare+0x5638/0x5650/0x5668/0x5680/
    0x5698/0x56b0 optional 步长 0x18）：0x130ebe0=CSV 第 N 整数（','跳字段→strtol→
    cmova 取值/0）、0x130eb50=CSV 第 N 浮点（strtod，失败取默认 [rsi+0x128]→[rdi+0x128]）。
  - **叶子 helper 无原名**（无日志调用）：0x1309230/0x13096b0/0x1307820/0x1307960/0x1307f60/
    0x1307aa0/0x1301040/0x12fba20/0x1307780/0x1307c90/0x1307d90/0x1309810 等，保留工作名。
  - **可推广**：.rodata 共 908 个源文件名字符串，管线相关已定位：graph_prepare.cc@0x461dfc7、
    cost_based_scheduler.cc@0x469b688、sharing_plan.cc@0x55b3acb、fa_alloc.cc@0x461a177（P3）、
    run_order_to_alloc_info.cc@0x55aefa1、supertile.cc@0x55b44e8（P2）、insert_spillfill.cc@
    0x462ae1f（P5）、vtcm_alloc.cc@0x55bae43、slc_allocator.cc@0x55af563、tiler.cc@0x55b98a4
    （P6）。P2–P6 反汇编时按同法恢复名字。
- ✅ M26 已完成 schedule_for_alloc 内联准备段 13039e5–1303ee9 + 选项解析循环头，
  见 `M26_measure_peak_region_and_options.md`（并修正 M25 两处误配对）：
  - **超时预算**：rdtsc 起点@1303a5a；预算 = u64(this+0x6830)×double(this+0x55f8)，
    基准常量 0xB2D05E00=3e9（3GHz TSC）@1303e80；deadline → [rsp+0x68]。
  - **重试状态三件套**：节点序 u32 列表（[rsp+0x1140]）两次深拷贝（vector@rsp+0x10e0、
    @rsp+0x90，各 _Znwm+movups 64B/轮）+ 配置对象深拷贝 call 0xf6f1f0（六个 24B string
    逐字段克隆；⚠ objdump 标签 sap_reduce_bandwidth 是回退标签，真身 0xf6ed80 止于 0xf6efdc）。
  - **两道门**：this+0x55dc(i32)≤0 跳过；[rsp+0x18]<<0xb ≤ double 换算值 跳过（门 2 操作数来源未解）。
  - **SCHED_OPTIONS 装载**：optional{flag@+0, csv 指针@+0x10} 0x18 步长 ×6+（this+0x5638..0x56b0）；
    csv_int(0x130ebe0) 返回值 test $0x1 当布尔、csv_float(0x130eb50)；假/未置位回退默认
    this+0x55d4/0x55d8。
  - **未解**：measure_peak/full_schedule 源级边界（内联+块重排）；0xf6f1f0 对象身份；
    选项↔OT/IT/…逐一对应（M27）。
- ✅ M27 已完成内联 full_schedule 重试主循环 1303ee0–13044e3 + 循环尾/统计块
  1304596–1304ab9，见 `M27_full_schedule_retry_loop.md`：
  - **循环结构**：回边 13044c5→1303ee0；r14=轮次(1 起)，CSV 参数按 (r14−1) 字段重载；
    每轮 = 装参数→预算公式→0x1306750(准备,首轮拍五表快照)→0x1306a20(执行,失败
    "Bad Schedule (seed=%u)" 行 948/可选 Fatal 行 950 by this+0x55e8)→0x13065f0
    (flow,grain 数 ×2048=字节)→记 FLOWS/CYCcles 向量→择优→判定→重置。
  - **三份节点序分工闭环**：[rsp+0x10e0]=最优序(1304325 择优写入/13044cb 退出恢复)、
    [rsp+0x90]=原始序(坏调度+每轮尾重置)、[rsp+0x1140]=工作序；[rsp+0x18]=历史最优。
  - **迭代预算公式**（两遍同式 130408b/1304363）：(opt×1e6+3e9−n×530000)/
    ((n×0x1450f0)>>32+26)，n=工作表#1 元素数；TIMEOUT=[rsp+0x1530]已用量>预算
    (opt3) → 行 971 日志(%llu=opt3 原值)+停。
  - **SCHED_OPTIONS 字段映射全解**（1304642 实参重建，M26 未解项）：OT=+0x55e4/
    IT=+0x55d8/Rg=+0x55d4/Rt=+0x55dc(=最大轮数门!)/AB=+0x55e0/AM=+0x55e8(=Fatal
    开关)/EO=+0x5618/HD=+0x5619/DD=+0x561a/TR=+0x55f8(=预算乘数 double)/LT=+0x5620/
    RC=+0x5628/RP=+0x562c/DB=+0x5630；0x4628a0e=空串""宏尾参。
  - **统计块**：this+0x5634 总闸；GRAINS_CUT/FLOW/RELEVEL←[rsp+0x1510/0x1518/0x1520]
    (行 1000-1002)；1304770 差分 STAT 循环(行 694)；13047c3 0x13065f0 最终复算。
  - **未解**：[rsp+0x1530]/[rsp+0x1f0] 写入点；0x1450f0 等效除数；CSV 值是否写回
    this；convert_to_ids 收窄为完全内联(1304596 后无独立 call)；计时归属异常
    (full_schedule push@1304510 在循环后,重试时间按差分规则记 measure_peak 名下)。
- ✅ M28 已完成循环体三大件 + 新函数 0x1307170，见 `M28_loop_body_three_functions.md`：
  - **0x13065f0(stcut_delay_dma)=目标函数**：遍历工作序，权重树(48B 节点
    key@+0x20/count@+0x28)查 count，前缀和取运行最大 = 内存峰值(grain,×2048=字节)。
    重试循环最小化的就是这个值。0x1307200=同遍历 Σ 前缀和(仅调试)。
  - **0x1306750=本轮准备**：[ctx+0x158]=&工作序 vector(★关键存根)；call 0x1307780
    首轮五表快照；v30=vector<u64>(n) 经 0x130cea0 填充；call 0x130d3e0 真引擎
    (8 参,副作用写 ctx)。
  - **0x1306a20=排序合法性验证器**(st_cut.cc:1782)：倒序扫工作序+BFS 传递闭包
    (邻接=快照#4 add-set@+0x1f0；过滤快照#3@+0x1c0 flags&4、快照#2@+0x1a8 &2)；
    遇已标记 id → "invalid in this ordering: %u %llx" → 返 0 = Bad Schedule 判定源。
  - **0x1307170=ctx 五表恢复**(新)：(+0x68,+0x80,+0xb0,+0xe0,+0xf8)←(+0x190,
    +0x1a8,+0x1c0,+0x1f0,+0x208)；★更正 M27 括注——五工作表实为 +0x68/+0x80/
    +0xb0/+0xe0/+0xf8(非 +0x98/+0x110/+0x128,栈槽没错)；循环尾 1304434 即其内联。
  - **0x130d3e0=真调度引擎**(M29)：≥2878 指令、0x168 帧、调 0x130f5b0×4/
    0x130b840×3、间接 call ×53+。0x130cea0=辅助向量填充器(M29)。
  - **未解**：权重树 count 写入方；add-set 吸收方向(M23 对齐)；0x130d3e0/0x130cea0。
- ✅ M29 已完成 develop_schedule 全解，见 `M29_develop_schedule_130d3e0.md`：
  - **0x130d3e0 = develop_schedule()**(st_cut.cc:3101/3331 日志自证；函数体
    0x130d3e0-0x130e87f,~1200 指令,唯一 retq)：贪心切分 journal(64B/条,区间=
    [field28.lo32, +value3)) + 批内拓扑发射,**原地重写工作序 [ctx+0x158]**——
    M28"谁重排工作序"答案。
  - 主循环：评分=Σ(aux[k]+3·aux[k]·[aux[k]>阈值])（超阈值 3 倍惩罚面积,
    阈值=((u64)ctx+0x278)×[ctx+0x270 对象+0x128]÷2048 grain）→ 0x130ab30 提案 →
    rdtsc 夹 0x1309a60 执行(耗时累计 ctx+0x280) → 0x130b9a0 修正 → 0x130cea0
    重填被切区间 → 0x130d050/0x130b600 记录分裂 → 0x130bc70 随机重启
    (门 [ctx+0x270]+0x158≥2, rand(ctx+0x2a8)%K==0)。超时门=[ctx+0x2a0]≥r9 预算
    (★M24 遗留 [ctx+0x2a0] 读方关闭)。
  - 终段：每记录批收集 (id,[ctx+0x140][id]) + set<u32>(32B 节点同验证器)；
    0x13125e0 std::sort 比较器 0x12fffb0=(node指针,id) 字典序=节点创建序；
    多趟 Kahn(ctx+0xe0 邻接 flags&4 的 canon 不在 set 才发射) → 130e79b
    ★写回 [[ctx+0x158]].begin[range_start++]=id。
  - **0x130cea0 = aux 前缀和填充器**(全解)：aux[k]=Σ权重树 count 前缀和=内存
    累计曲线;(ctx,aux,start,len)——★M28"终点 n"更正为"长度 n"。
  - **0x130ccd0 = count 写入方**(★M28 遗留#1 关闭)：懒建树时 count=
    Σ_{rel∈ctx+0xe0[id]} ctx+0xb0[rel].field@+8(grain 权重,spill/DMA 量)。
  - 闭环：切分把工作序划成内存区间,区间内按拓扑重排,区间边界=内存重用档位；
    0x13065f0 目标函数与 0x130cea0 共用同一棵权重树(ctx+0x28/0x30/0x18)。
  - 未解(→M30)：0x130ab30/0x1309a60/0x130b600/0x130d050/0x130bc70/0x130a560
    六 helper 内部；[ctx+0x2a0] 递增指令；[ctx+0x278] 语义；field0==2 两用性。
- ✅ M30 已完成切分六件套内部，见 `M30_cut_helpers.md`：
  - **0x1309a60=迁移执行器**：0x1309940 估量≥1e8→0x13080a0 慢路径；从 pivot 换缓冲
    BFS 泛洪(ctx+0xe0 邻接；门 ctx+0x80&3==0+ctx+0x128 未标；flags&4 边另需
    表[rel.@4].@8>ctx+0xc8[rel.@4])；可达节点★改归属 ctx+0x98→新记录号+入 out；
    两遍导出"对方不属本记录"的 flags&4 边 @4 句柄。
  - **0x130b9a0=工作序稳定划分**：boundary=区间起+区间长−|out|；压实父保留前缀,
    迁移节点经临时向量轮转填尾洞(源位写 0)——与分裂器 parent.@0x18−=cut_size 自洽。
  - **0x130d050=分裂判定器**(u8)：0x1305130 打层标记→切集远端标记直方图；
    <8 桶返 0；前缀和×0x2AAAAAAA(1/6 定点)，(桶数×最大前缀)/总和<0x340000000 才切。
  - **0x130b600=记录分裂器**：0x1301040 建 kind=2/1 新槽；parent.@0x18-=cut_size；
    子记录=父区间尾段经 0x130b840 压入；链式链接(@0 后继/@8 自身)；切集每 id
    0x1300cc0×2 双侧重接线。
  - **0x130a560=★净重计算器(M29 修正)**：ctx+0x140[id]=Σ(add 集 非@4 @8)−Σ(sub
    集 非@4 @8)=节点净内存增量；⇒终段批排序第一键是净重,非"节点指针/创建序"。
  - **0x130a330=局部权重树**：0x1305130 打标前快照+0x1308c50 重打标,双标记非零
    且未冻结入树,count=(新−1)×旧。
  - **0x130bc70=随机重启**(0x130bc70-0x130ccbf)：记录链摘除+@0/@8 清零判死,拷
    add/sub 两集,关联树收 add 集;重合并循环快照对账:权重表两处减账(130c3e7/
    130c3f1)+type=4 重放记录经 0x13080a0 重加,活/快照 add 集成员计数对账
    (130c490/130c4f0)——把该记录的关系编辑回滚向快照态(撤销这刀重来)。
  - 闭环："切在哪(提案)/谁搬(执行)/搬去哪(修正)/值不值(判定)/怎么记账(分裂)"
    五问全解,调度-内存 spill 管线自 schedule_for_alloc 至工作序重写全链贯通。
  - 未解：0x130bc70 中后段 0x130c519-0x130cb4d(同构对账展开未逐条);
    0x1305130/0x1308c50 标记原语;[ctx+0x278];0x12fffb0 方向。

**✅ M24/M25 遗留收尾（本日）**：
  - **0x130ec60/65 + 0x130eca0/a5 = 树节点递归析构**(M25 修正:非 CSV 解析器变体)：
    前者 POD 节点(=ctx 权重树,节点 0x30B key@0x20/count@0x28)；后者 map<key,vector>
    节点{buf@0x28,end@0x30}。
  - **0x13073c0 = ctx 析构**(0x13073c0-0x13076de,末尾尾跳 0x130ec60 销毁权重树;
    异常孪生内联在 0x12fba20 尾部 12fc07b-12fc229)。ctx 新字段:+0x228/0x230
    map<key,vector> 树;+0x240/+0x258/+0x1d8 vector;加/减集与快照=vector<vector<u32>>。
  - **0x1307c90/0x1307d90 唯一调用方 = 0x130bc70 随机重启**(130c18a/130c797/
    130c7f8)——重指向是重启回滚接线工具(M24 遗留#1 关闭)。
  - **0x1308b10=vector<24B>::push_back、0x12fc610=vector<u64>::push_back**(0x1300cc0
    慢路径 1300e6f/1300e8c+1300eb8/1300ed3;另有第三份 0x12fc4d0 仅 0x12fba20 尾部
    用)(M24 遗留#2 关闭)。
  - **0x1308c50 = 层级标记 BFS**(标记=层级号≥2,沿 add 集 flags&4 边泛洪,subtract
    集扫描决定输出集;收尾段 1308f80-130922f 未逐条)——解释 0x130a330 的
    count=(新−1)×旧 公式(M30 未解#2 半关)。
  - **0x12fc740 = stcut_delay_dma_again 内部**(清出向量→flags&0xB==0 过滤→投影
    [ctx+0x38 向量][id]);**0x1305130=stcut_add_dependencies** 4 调用方全对上
    (1303766/1305738/130a354/130d07c)。
  - 转储:asm/M25_parser_variants_130ec50_130ed80、M25_ctx_dtor_probe_12fbfc0_12fc290、
    M25_teardown_probe_1307{380,680}、M24_slowpath_1308b10_1308d00、
    M24_slowpath_12fc4d0_12fc820、M24_1308c50_1308f80、CU_range_12f0000_1313000。

- ✅ M31 已完成 stcut 管线重实现（M17–M30 结论可执行化），新增三文件：
  `include/hnnx/scheduler/st_cut.hpp`（结构层）+ `src/scheduler/st_cut.cpp`
  （实现层，~950 行，逐段 [0x地址] 注释）+ `tests/test_st_cut.cpp`（冒烟测试），
  CMakeLists 已挂 `src/scheduler/st_cut.cpp`（htp_core）与 `test_st_cut` 目标。
  - **结构层**：StCutSlot(24B)/StCutReplayRec(24B)/StCutRec(64B,asm 实测布局)
    static_assert 锁布局；StCutConfig([ctx+0x270] 已解字段)/StCutOptions(SCHED_OPTIONS
    this+0x55d4…)/StCutContext(全部 ctx 偏移作成员,含计时 +0x280/288/290、
    +0x2a0 inner-loop、+0x2a8 RNG)。
  - **实现层分层**：M31-3 基础（快照/还原五表、grain 树懒建）→ M31-4 建图
    （connect_nodes 快/慢路径+四向建链、register_node mode 分路、replay 6-case、
    flood/weight_pull/iterate、batch_accounting 4 连接、mark/level_mark/cleanup、
    build_initial_state）→ M31-5 切分（提案/执行/修正/判定/分裂/净重/局部树/重启）
    → M31-6 调度（aux/评分峰值/验证/prep_round/develop_schedule 含终段 Kahn/
    iteration_budget 魔数式/full_schedule 十二步重试循环）。
  - **实现期 asm 复核三处更正**（写码时直接读 asm 发现，报告已同步）：
    ①分裂器子记录布局 @0x10=A/@0x14=B/@0x20=判定/@0x38=新id（M30 表格
    「@0x20=B_idx/@0x24=A_idx」为栈帧偏移误读）；②journal 链域是记录索引
    （rbx=(end−begin)>>6 先算后 push）；③journal[child].@0x30=Σ切集权重
    [130b7ec movq r15]。
  - **M29/M30 矛盾裁决**：执行器落点归属写"新记录号"（非父 id）——0x130b9a0
    以 ownership==新记录号为划分键是唯一自洽读法。
  - **实现期 bug 修复**：fix_range 划分方向（父保留前缀|迁入新记录的节点填尾,
    与子记录继承父区间尾自洽——原稿 keep/move 反了）；分裂器 push_back 后
    journal 重分配致父引用悬空→改索引访问；终段 Kahn 趟内无进展死循环
    （.so 130e810 `eb fe` 不可达自旋）→按序直发兜底；重启"树值/@3倍"槽定位
    未证→改 canon 匹配等价定位+显式标注。
  - **冒烟测试**（test_st_cut.cpp，7 组）：槽互指/累加、洪泛+搬运、grain/aux/
    峰值、验证器合法序与倒置序、replay 三 case、develop_schedule 区间守恒+
    排列不变、full_schedule 端到端排列不变。**已全过**（g++ -Wall -Wextra 零
    警告 + ASan 干净；cmake 本机无 cmake 二进制，以直编为准）。
  - **调通期第二轮 asm 更正/修复**（ASan 定位三连环）：
    ①weight_pull seed==target 返 cap 吸收（13096ca-13096d3，M24 §2 笔误）
    ②register_node 空闲栈空时 je 落回追加路（13010a4-13010ab，非无条件 pop）
    ③**fix_range 扫描全长 = span_len**（调用序实证：130dc5b 落地段先于 130b698
    父区间收缩；曾误写 span_len+|out| → 越界读 → 工作序脏值非排列）
    ④分裂器 call1 源=对端槽 canon [130b769-130b771]、两次注册取计数器连续值、
    child.id=调用方栈参 v1 [130b6ed]；⑤B 侧 flags 恒 mode|4 [1300e13] 不变式
    （hard 才给 A 侧加 4）——验证器走 A 侧前向闭包的依据。
  - **iterate 收敛病理 + 防御**（反汇编未完全理解）：洪泛在 touched 代早停致
    叶子槽永不清账 → A/B 互馈；seed==target 纯吸收致循环 A 不退出。.so 无迭代
    上限——真实输入靠上游不变式（锚≠切点、真实权重、劈半候选）收敛；实现加
    guard_cap=4×节点数+64 两级守卫并标注。
  - **BFS 跨记录逃逸 → 归属围栏门（防御性）**：执行器 BFS 闭包可走出父记录
    （替身提案器切集=全区间时实测 node 反复跨记录重迁）→ |out|>span →
    [130b698] 无符号下溢。.so 围栏机制（跨记录边改道冻结枢纽 A/B[group_flags
    &3]+账本门）精确边界未拆——实现以「ownership==rec.id 才迁移」等价保证
    |out|≤span，分裂器另加钳位双保险。
  - **未解段如实标注于代码**（不实现/概要级，均带 [0x地址]）：0x1305130 内部
    （层号边界）、0x1308c50 收尾 1308f80-130922f、0x12fffb0 排序方向（实现取降序）、
    0x130bc70 中后段 130c519-130cb4d（重指向触发条件）、[ctx+0x278] 语义、
    0x1450f0 等效除数、[rsp+0x1530]/[rsp+0x1f0] 写入点、执行器 arg4（以
    rec.sub_entry 代入）、提案器候选数（以 add 集非&4 关系数代入）、初始表
    flags&4 分布（hard 两侧对称置 4）、PRNG 0xd79830（LCG 等价）、CSV 逐轮重载
    （固定 StCutOptions 代替）、delay_dma_again 0x12fc740/convert_to_ids（地址留档）。

- ✅ M32 提案器 0x130ab30 逐指令解码完毕，概要级替身已替换（130ab30-130b500
  全函数：入口邻对选择 → 外部候选路 → 通用路 → 随机兜底 → 统一切集劈半 →
  终验登记；含全部门与魔数）。
  - **签名实测**：f(ctx, StCutRec* best, journal指针, arg4 全函数未用,
    排除集 /*键@0x1c·32B 节点·跨轮持久*/, out_before /*arg6*/, out_after
    /*栈参 0xc0*/) → u32 切点。
  - **邻对选择** [130ab55-130ab7d]：fwd = journal[best.@0]（初始记录 @0=2
    哨兵恰指向 3 号初始记录）；filter_id = journal[journal[fwd].@8].id；
    picked = {fwd, back} 中 id 较大者（正常链 = 前向后继）。
  - **外部候选路**（journal[picked].@0x20≠0）：表=ctx+0x240[journal[picked].
    @0x24]，门「表号<表数 ∧ 非空 ∧ 字节≤0xef」，过滤 ownership==filter_id，
    切点=hits[count/2]；==0 或空 → 落通用路。
  - **通用路**：局部树 0x130a330(ctx, best.@0x14, journal[best.@0].@0x10, &树)
    ——★两趟打标种子不同（arg2 第一趟 / arg3=前向邻居 sub_entry 第二趟
    [130a341→130a391]，M30 §6 笔误已更正）；扫区间跳 ctx+0x80&4，w = 树值
    ×cand_tables[节点]元素数（表字节≥0xf1 → w=0）+ jitter；w>best ∧ 不在
    排除集 → 接受。
  - **统一切集构造**：表=cand_tables[切点]，门同外部路；verdict≠0 → 切点改取
    「距两端最远且不在排除集」元素（严格递增距离扫描），verdict==0 → 中位；
    按切点劈半：前→arg6、后→arg7（= 执行器 batch_accounting 循环2/循环3 的
    v5/v6——「切集=整个区间」的 M31 替身认知作废）。
  - **终验+登记**：切点==0 或已在排除集 → 二次随机兜底（不复查）；最终插入
    排除集（32B 节点+再平衡+size++）后返回——同节点不会被二次提案。
  - **候选表 ctx+0x240 填充方**：✅ M33 已解（stcut_read_nodes@0x12fc820，
    见 F3/M33 条目）。
  - 实现侧同步：执行器签名拆两半切集；develop 持久化排除集；
    build_local_tree 补第二入口参。

**F3. 后续解码计划（M33 起）**：

- **M33 候选表填充方（ctx+0x240 的写方）** ✅ 完成（2026-08-28）
  - **填充方 = stcut_read_nodes @ 0x12fc820**（双重命名实证：函数内告警串
    "Node index %u is larger than size %zu"@0x55b3d7c + 文件名 "st_cut.cc"@0x55b3db9；
    计时器段名池 0x55b3de5 "stcut_read_nodes"，调用点 schedule_for_alloc [13030e0]）。
  - **结构链**：cand_tables 槽由 register_node 0x1301040 级联 resize 创建
    （新 id = 旧节点数+1 [13012a9]，四表 resize：u16 标记表+0x128、两张
    `vector<vector<u32>>` +0xe0/+0xf8、cand_tables+0x240（8B 槽）[13013bc-1301416]）；
    **内容由 read_nodes 填充**：遍历 map A（ctx+0x228 = `map<组键, vector<u32>>`），
    每组成员表经扁平索引（vector<pair<payload,id>> 16B 槽，槽位=节点 id，
    0x12fcdd0 reserve + 0x12fce80 push）收集→std::sort（0x130ed40，比较器
    0x12fd0d0 按 id——同基址指针比较优化）→有序 id 写回成员表
    [12fcbf0-12fcc51]，随后 `cand_tables[id] = &组成员表`（指针全组共享）
    [12fcc79]；越界成员告警跳过 [12fcac8]；组表指针收集进 ctx+0x258
    （vector<void*>，0x12fcfa0 push）[12fc916]。
  - **语义**：map A = 强关联组——strong_relevel@0x12fd600 开头即遍历每组取
    成员 u16 层级 [12fd6cc] 做组内层级归并，同组节点一起分层；提案器的
    候选池 = pivot 所在组的成员表。跨组节点后组覆盖先组（map 键升序）。
  - **st_cut.cc 14 段建图准备链全部命名**（段名池 0x55b3de5-0x55b3ee1 依序）：
    read_nodes@0x12fc820 → collate_sibs@0x12fd0e0（op 记录 0xd0 步长，两张
    关联表 @0x38/@0x50 经 op→node 映射填两张按节点索引的 `vector<vector<u32>>`）
    → relate_by_tensor（内联 1303380-13033dc：out[i]=Σweight[relA]+Σweight[relB]）
    → node_hash@0x1300210 → quick_early_sort（内联）→ block_relate@0x1300a40
    → connect_nodes（内联驱动 13036e6：组表 flags&3 跳过，逐成员 call
    0x1300cc0）→ add_dependencies@0x1305130 → strong_relevel@0x12fd600
    → arrange_sibs@0x12fffd0 → clean_sibs@0x13056f0 → topo_presort →
    delay_dma@0x13065f0 → measure_peak。
  - 实现落地：ctx 增 relation_groups（map A）+ group_list_keys（+0x258）；
    register_node 补 cand_tables 级联 resize；新增 stcut_read_nodes（排序/越界/
    指派/登记，.so 指针共享以按值拷贝等价——消费方只读）；测试组 8
    （乱序组排序、同组共享、未入组为空、越界剔除、跨组覆盖、带组 develop
    排列+区间守恒）——8/8 过，零警告，ASan 干净。
  - 遗留：map A 的装载期插入方在 load_replacement_plan 主体
    （0x1299ff4–0x12f0000，未转储，组键类型未证）——不影响调度侧行为建模。
- **M34 标记原语内部 + 重放 0x13080a0 + 记账闭环** ✅ 完成（2026-08-28）
  - **0x1308c50 &4 反向孪生逐指令全解**（与 0x1305130 骨架逐指令同构，两处
    &4 极性相反——本函数只走 flags&4 槽 = B 半边，canon=前驱）：
    memset 外部表 arg4 [1308c7a-1308c86]；种子=arg3（esi 死参数）[1308ccb]；
    level=2 [1308cdf]；循环1 就绪门扫 subtract 集 ctx+0xf8——非 &4 跳过
    [1308dc2 ★]、对端 = 配对槽 canon（wt[wt[slot].idx].canon）[1308daf-1308db8]、
    mark[对端]>1 已安置 [1308dbb]、有未安置 &4 对端 → 不就绪 [1308dc9]；就绪
    登记 B 队前沿索引 [1308ddf]；循环2 扩散扫 add 集 ctx+0xe0——非 &4 跳过
    [1308f6b ★]、child=canon [1308f72]、mark[child]!=0 跳过 [1308f79]、
    A.push+占位 1 [1308f2d/1308f3d]；层末倒序 ×2 展开 swap-remove 写正式层号
    [13090e0-130914b]；level++ [1308d0e]。
    **语义**：反向波沿 B 半边走前驱；就绪门只认 &4 对端（= 后继，经 subtract
    集 B 槽伙伴链）——非 &4 的 A 槽不构成依赖（正向原语相反）。
  - **0x13080a0 重放例程 6 型全解**（跳转表 0x55b3d04 经 PT_LOAD vaddr→文件偏移
    python 提取；`cmpq $5; ja` → 型 0-5；记录 24B {op@0, value@8, type@0x10}，
    逆序处理 = 撤销）：
    - **型0→0x130811f**：wt[op].weight = value（权重恢复）。
    - **型1→0x1308136 / 型5→0x13081fe**：纯日志（GetLogPriorityLevel≥100，
      串 0x55b42b0/0x55b4288），不动状态。
    - **型2→0x1308151**：撤销建对——四次换尾删除：扫1 A 从 add_set[B.canon]
      [1308178-1308232]、扫2 B 从 add_set[A.canon] [1308252-130829d]、扫3 B 从
      subtract_set[B.canon] [13082a0-13082ed]、扫4 A 从 subtract_set[A.canon]
      [13082f0-1308335]；权重表槽保留。
    - **型3→0x13080d4**：[ctx+0x70]-=8、[ctx+0x88]-=1（节点槽弹出）。
    - **型4→0x13081a8**：型2 的精确逆——四次压回：push1 op→add_set[B.canon]
      [13081ef]、push2 B→add_set[A.canon] [130842a]、push3 B→subtract_set[B.canon]
      [1308561]、push4 A→subtract_set[A.canon] [130868f]。
    - **槽宿主不变式**（由型2/型4 四向操作导出，与 connect 建对四向
      [1300ed8-1300f07] add(src,A)/add(dst,B)/sub(src,B)/sub(dst,A) 互证）：
      A{canon=后继, 非&4} ∈ add_set[前驱]+subtract_set[后继]；
      B{canon=前驱, &4} ∈ add_set[后继]+subtract_set[前驱]。
  - **connect 0x1300cc0 记账位点**：建对路 [1300f84-1300fa3] cb 非空推
    {op=A 槽, 0, 型2}（经 0x1307f60）；命中路 [1300fd6-1300ff9] 推
    {op=A, 旧权重, 型0} 后 [1301008-130101d] wt[A].weight =
    min(0x5F5E0FF, 旧+threshold)。
  - **执行器慢路 0x1309a60 账本生命周期**：局部账本 rsp+0x90
    （[1309a8a-1309aa1] 清零）→ 记账调用 [1309ab1] → 返回 ≥0x5F5E0FF（饱和）
    → [1309acc] replay(ctx,&L1) → [1309afd] 空候选重试（r8=r9=&空，仍记账）→
    [1309b4b] memset 标记表 → 末尾 [1309f37] replay(rsp+0x90)。**脚手架在单次
    trial 内全暂态**——跨 trial 不残留（我实现此前残留 → 死锁，根因）。
  - **提案器双种子验证** [130a330→130adc3]：esi=[rbp+0x14]（最优记录 add_entry）
    喂 0x1305130；edx=journal[best.@0].@0x10（fwd 记录 sub_entry）喂 0x1308c50。
  - **★结构性死锁结论（数学论证）**：v2（add_entry 锚）无入边 → 前向不可达；
    v1（sub_entry 锚）无出边 → 反向（孪生）不可达。两形状在 .so 代码中同样
    存在——.so 经 develop 循环记录选择 + 执行器 BFS 所有权语义规避（后者
    **反汇编未完全理解**，执行器围栏以文档化差异处理）。中点种子的正向/
    反向原语均按设计死锁（等永不可达的对端）。
  - **发散点（显式记录）**：原语加 max_levels 形参（默认 UINT32_MAX = .so 忠实
    无界）；概要级调用方（build_local_tree / cut_verdict [130d07c]）传
    node_count+2（收敛波每层 ≥1 节点 ⇒ ≤n 层，完备上界）。
  - **撤销既有错误结论**："hard→A&4" 作废——硬边不落 A 侧 &4（若落，
    0x1308c50 就绪门自锁）；&4 恒为 B 半边结构标记。
  - cleanup 0x13087f0：与型2 同构的四扫删除（非重连），esi=槽。
  - 实现落地：StCutReplayRec 账本（connect cb 第 5 参 → trial 局部账本 →
    饱和重试 → 末尾重放）；stcut_replay_apply 全 6 型；split 重连/导出循环/
    双标记原语全部按址对齐。测试 9/9 过（含中点种子死锁护栏正测），零警告，
    ASan+UBSan 干净。
- **M35 P1 收尾杂项** [~] 七项完成（2026-08-28 三批；0x130bc70 全函数闭环），余量待续
  - **✅ develop 循环记录选择 130d890-130daa8 逐指令全解**：
    - **评分公式**（Path A，out_vec 数 == 游标时）：
      `score(rec) = Σ_{k∈[start, start+span)} (aux[k] + (aux[k] > thr ? 3·aux[k] : 0))`
      [130d9e0-130da3a 2× 展开；130d9ec cmovle——超阈值计 4 倍]；rec@0x00==0 死记录
      跳过 [130d9a1]；span≤0 评分 0 [130d9b7 jbe]；更优门 = 无 best 或严格大于
      [130d970-130d98f]。回溯绑定：aux = prep_round 局部曲线（develop arg5 = r8，
      rdi=[[rsp+0x10]]）；记录区间 {span@0x18, start@0x28}（初始记录 #2={span=n,
      start=0}、#3={span=0,start=n}——rodata 模板 0x39ad0b0={link=2}、
      0x3966e50={@0x08=1} 已提取验证）。
    - **阈值 thr**（.so 整数运算）：`v=(double)(u64)[ctx+0x278] × [[ctx+0x270]+0x128]`
      → cvttsd2si [130d91f] → 负数 +0x7ff [130d924-130d92e] → sar 11 [130d94b]。
      其中 (u64→double) 是 LLVM uitofp 展开（unpcklps 拼 {x_lo,0x43300000}/
      {x_hi,0x45300000} 两段精确 double [130d8f5/130d8fc]，2^52/2^84 拼接技巧，
      常数在 0x3970340/0x3970350）。⇒ **[ctx+0x278] 的 C++ 源型应为整型**。
    - **Path B（out_vec 数 ≠ 游标）** [130da40-130da8e]：node = out_vec[cursor]；
      rid = ownership[node]（ctx+0x98 u16 表 [130da50]）；从链头 1 沿 link_fwd 步进
      [130da7e] 找 rec.id==rid [130da77]，步到 link==2 兜底 best=2 [130da82]。
      **★Path B 直接替换评分结果且跳过提案器** [130dadf jne→130db16]。
    - **游标推进**：落地段 [130df50 游标++]；out_vec 仅在 count==cursor 时
      push [rsp+0x8] 低 32 位 [130de5b-130de60]（= 提案器返回值 [130db11]；
      Path B 轮 = 轮初哨兵 0x7FFFFFFFFFFFFFFC 截断值 [130d7f9-130d814]）。
  - **✅ 执行器参数语义破译**（M30 旧注 pivot/arg4 双误订正）：
    esi = journal[best].add_entry；edx = **journal[journal[best].link_fwd].sub_entry**
    （= 执行器 seed [rsp+0x14]：BFS 种子 [1309b6c-1309b70]、导出循环扫描基
    [1309f3c]）；**ecx = develop 游标**（[130db51 ← rsp+0x18]；记账 connect 对端
    [130996e] 与 add集 扫描下标 [130997c]）。提案器 pivot 只回 develop [rsp+0x8]，
    从不进执行器。
  - **✅ 所有权围栏语义**：ctx+0x98 u16 表 = 节点→持有记录事件 id（develop 每轮
    扩容到节点数并全填记录#2 id [130d61d-130d7ab]）；执行器归属改写**无围栏、
    无条件**（越界门 [1309bf3] + 跳过种子 [1309bcb]），M30 自造
    「ownership[ebp]==rec.id」门已删——闭包范围实际由记账阈值门/flags&4 账本门/
    mark 表/group 门共同限制 [1309d30-1309ef7]。测试 9/9 证实删门后无
    underflow 回归（M34 账本修复后自造围栏本已多余）。
  - **✅ 记账核心 0x1309940 头段**：connect(add_entry→cursor, CAP, 账本) [1309974]
    + 扫 add_set[cursor] 非&4 槽、canon≠fwd_sub_entry → connect(canon→fwd_sub_entry)
    [1309990-13099d8]。
  - 测试 9/9 过，零警告，ASan+UBSan 干净。
  - **✅ 终段 journal 发射循环 130e1be-130e801 全解**（2026-08-28 第二批）：
    - **逐记录三步**：外层 64B 步长走 journal，rcx=rec.start@0x28、rdx=start+span@0x18，
      `cmpq/jae` span≤0 跳过 [130e1f8-130e205]（死记录#1 全零模板 span=0 自然落此门，
      非旧实现的 link_fwd==0 判据）；内层 r12∈[start,end)：node=working_order[r12]
      [ctx+0x158.data [130e227-130e235]]，**pair push 无条件** {u32 node, i64 净重
      [ctx+0x140][node] [130e239]}（快路径内联 130e23d-130e249 / 扩容 130e260-
      130e318，重复无害），同节点去重插入 0x20B 红黑树（key=node@0x1c）[130e322+]。
    - **比较器 0x12fffb0 破译（余量首项关闭）**：SysV 按 16B 值传参 a→(rdi,rsi)、
      b→(rdx,rcx)，`cmpq %rcx,%rsi; jl ret` 净重严格升序在前 [12fffb2-12fffb5]；
      `cmpl %edx,%edi; setb+and` 平值按节点号升序 [12fffba-12fffbf]。排序本体
      0x13125e0 = libc++ std::sort（深度限 2·log2n [130e5b7-130e5de]）。
      ⚠️ 旧实现取降序（当时标注"方向未解"）已订正为升序。
    - **多趟 Kahn 排空** [130e620-130e7ee]：趟内逐 pair——node 已离树（walk 空
      130e650/130e653）→ 跳过；在树 → 扫 add_set[node] 槽号 s [130e6c9]，
      weight_table[s].flags&4 [130e6cf]（B 半边，canon=前驱）且 canon 仍在树
      [130e6d9-130e6f5 树查] → **推迟整 pair**；否则删树节点（树删除 130e6fa-
      130e796，size-- [130e782]）并 **发射 working_order[[rsp+0x8]++]=node**
      [130e7a0-130e7b6]，输出游标每记录从 start 起步 [130e1fb]。全趟结束树非空
      → r13 重置回向量头重扫 [130e620→130e631 r13=r12]。.so **无死锁逃生阀**
      （依赖账本前驱 DAG 不变量保证每趟必有可发元素；批空树非空的 130e810 `eb fe`
      自旋为 __builtin_unreachable）。重实现保留按序直发阀并注明分歧。
    - ctx 字段对齐：+0xb0=weight_table（24B 槽，flags&4@+0x10）、+0xe0=add_set
      （每节点 vector<u32> 槽号）、+0x140=net_weight（净重，0x130a560 整体覆写）、
      +0x158=working_order。测试 9/9 保持全绿。
  - **✅ 0x1450f0 等效除数破译**：[1304393-130439e] `(n×0x1450f0)>>32 + 26`
    为纯 mulhi 无修正。0x1450f0 不是任何整除数的标准魔数
    （2^32/0x1450f0=3225.806 非整数），但 = **n/3226 的免修正上取整魔数**：
    3226×0x1450f0 − 2^32 = 258144 → 解析精确上界 n<16639，暴力验证
    n∈[0,16638] 与 n/3226 逐点相等、首个失配 n=19355（d=3225 在 3225 即破）。
    ⇒ 源码语义 `denom = 26 + n/3226`，stcut_iteration_budget 注释已更新。
  - **✅ [rsp+0x1530]/[rsp+0x1f0] 写入点破译**：
    - [rsp+0x1f0] 续行标志写点 [1303ff0]：`cont = (0x130ebe0 选项getter.rax&1)
      ? (其 rdx≠0) : (this+0x5618≠0)` [1303fd7-1303ff0 setne/cmovnel]。
    - [rsp+0x1530] 已用量 = **ctx+0x2a0**（帧内嵌 ctx 基 [rsp+0x1290]：
      0x1290+0x2a0=0x1530，无直接写点之谜即此）：ctor 清零 [12fbc0f]、
      迭代原语每次 +1 [13093f1]、develop 预算门读 [130d7d1/130d820]、
      重试循环头 `budget+used` 传 prep_round [13040fc]、循环尾
      `used ≤ budget2(opt=Rt)` 续轮 [13043b3 cmpq/jbe]。三停机条件齐：
      used>budget2 / flow×2048>cap [1304347] / cont==0 [130434f]。
      重实现 inner_loop_count 注释与 full_schedule 头注释已更新。
  - **✅ 0x130bc70 random_restart 全函数逐指令全解**（2026-08-28 第三批，
    130bc70-130cb4d 无剩余黑盒；重实现整体重写 + 新增 0x13087f0 全解实现）：
    - **① 双向跨接摘链** [130bcb3/130bcc0/130bcc8]：back.link_fwd=fwd、
      fwd.link_back=back（旧实现漏了第二方向）、rec.@0/@8 同清判死。
    - **② 副本**：B=subtract_set[rec.sub_entry]（栈 [rsp+0xf0]）、
      add_c=add_set[rec.add_entry]（[rsp+0xd0]）[130bce1-130be0b]。
    - **③ 关联树** [130be2e-130be84]：对 B 每个非&4 槽 s，
      `key = wt[wt[s].idx].canon`（**一级 idx 间接**，旧实现的 wt[s].canon 错）→
      payload=s（重复 key 覆写 [130be74]）。
    - **④⑤⑥ 预删**：aslot0=rel_tree[fwd.add_entry]（查 key=[rsp+0x48]=
      journal[fwd].@0x14 [130bf6f-130c066]，未命中插入 payload=0）≠0 → 从 B 删
      [130c0c5-130c102]；r15d=add_c 中首个 wt[slot].canon==fwd.sub_entry 的槽
      [130c086-130c0b4] ≠0 → 从 add_c 删 [130c114-130c14d]。
    - **⑦ 重合并外循环**（每 rel，非&4 [130c1ba] 且 canon≠fwd.sub_entry [130c1c4]）：
      - **内层快照对账**（snap_subtract[far] 每 srel）[130c1da-130c43c]：
        canon2=snap_wt[snap_wt[srel].idx].canon [130c2e8-130c2f0]；
        aslot=rel_tree[canon2]（find-or-insert，未命中 payload=0 [130c371-130c3ce]）；
        **双减账** wt[aslot].weight-=w 且 wt[rel].weight-=w [130c3e7/130c3f1]
        （旧实现只减一处且 aslot 定位法错误）；replay **{op=srel, 0, type=4}**
        push+apply [130c3f6-130c43c]（旧实现 op=aslot 错）。
      - **add 侧计数门（每 srel，内层体内）** [130c453-130c5b6]：
        count(add_set[far] 非&4) > count(snap_add[far] 非&4)（adcq 计的是
        **非&4** [130c490/130c4f0]，非旧注的 &4；门在 [130c50e jbe]）→
        remove_edge(ctx, aslot, null)（aslot 在 add_set[far] 找不到时先走两圈
        LOG4 诊断 [130c5d9-130c6d1]，诊断后同样落 remove_edge [130c669 je 130c559]）
        + 从 B 副本 swap-remove [130c5a0-130c5b6]。
      - **aslot0 补偿减账（每 rel）** [130c207-130c226]：
        wt[aslot0].weight -= wt[rel].weight（内循环退出后）。
      - **subtract 侧计数门（每 rel）** [130c22b-130c749]：
        count(subtract_set[far] 非&4) > count(snap_subtract[far] 非&4)
        → remove_edge(ctx, rel, null) [130c73f]；否则 repoint_B(ctx, rel,
        **fwd.add_entry**) [130c17b-130c18a]（旧注 @0x24 错，[rsp+0x48]=&@0x14）。
    - **⑧ 块2** [130c74e-130c7c9]：B 剩余非&4 s：r15d≠0 则
      wt[r15d].weight-=wt[s].weight [130c7bf-130c7c4]，repoint_A(ctx, s,
      fwd.sub_entry) [130c790-130c797]。
    - **⑨ 尾声** [130c7cb-130c80a]：aslot0≠0 且 !(wt[aslot0].flags&0x80)
      [130c7e0 testb $-0x80] → repoint_A(aslot0, fwd.sub_entry)；r15d≠0 →
      remove_edge(r15d, null)。
    - **⑩ 归属改写+区间合并** [130c80f-130c8d7]：rec 区间内
      ownership[working_order[k]]=fwd.id（u16）；fwd.span+=rec.span；rec.span=0。
    - **⑪ 块3/4** [130c8dc-130cad1]：add_set[rec.add_entry] 与
      subtract_set[rec.sub_entry] 副本中 flags&0x84==0（testb $-0x7c）的槽逐个
      remove_edge [130c9b4/130ca8b]。
    - **⑫ 收尾** [130ca97-130cad1]：retire_node(rec.add_entry)/retire_node(
      rec.sub_entry) [130caa7/130cab3]；rec.@0x24/add_entry/sub_entry 三清零
      [130cac2/130caca/130cad1]。
    - **✅ 0x13087f0 remove_edge 同步全解重写**（旧 cleanup_node 概要级误读）：
      p1=canon(slot)、pslot=idx(slot)、p2=canon(pslot) [130882a-1308837]；
      四处 swap-with-last：add_set[p2]←slot [130883f] / add_set[p1]←pslot
      [1308880] / subtract_set[p2]←pslot [13088d0] / subtract_set[p1]←slot
      [130891f]；replay≠0 追加 {slot,0,type=4} [130895d-130897f]；
      **不置 0x80、无输出参**（旧的 |=0x80/tmp.push 均删）。分裂器调用点
      [130b7b6] 传 rsp+0x30 一次性汇后即 delete 且从不读 [130b7bb-130b7ce]
      → 等价 nullptr。
    - 测试：新增 test_random_restart（develop 后重启子记录：判死/区间并入/
      双向跨接无残留指向/归属改记 fwd.id/区间守恒），10/10 全绿。
  - **✅ iterate 收敛性收口**（M35 末项）：1309810-1309930 全区间复核，仅三处
    出口测试——初探归零 [1309852]、循环 A 搬运归零 [13098c5]、循环 B 探测归零
    [1309924]；无上限比较/计数器（cmp/dec 全区仅 memset 尺寸减法 1309837 与
    计时减法 13098b5/1309917）⇒ **确认 .so 无迭代上限，终止纯靠不变式**：
    洪泛叶提前停 [13095a0] 与 seed==target 纯吸收两种不收敛形态均由上游
    pivot≠S 锚/账本 DAG 不变式排除。重实现防御上限保留并注明忠实分歧。
    **M35 全部七项闭环。**
- **M36 P2 首批：create_supertiles@0x1313ac0 + 小函数簇** ✅ 完成（2026-08-28）
  - 报告：audit_verify/reports/M36_create_supertiles_disasm.md；
    实现：src/vtcm/create_supertiles_m36.cpp + include/hnnx/vtcm/create_supertiles_m36.hpp
    （与旧 DP 重构 supertile.cpp 并存；旧文件 86 行地址注释 0x13138d0 有误，
    实为 0x1313ac0；其引用的 make_one_supertile@0x1314d50 为 M16 §5 死代码）
  - 指令级闭环 6 小函数：insert_spill_fill@0x106d7d0（永久桩返回 -1，fmt
    '%s:17::ERROR:insert_spill_fill not supported'）、autothread_size@0x10c3690
    （tbl@0x39b7650={1,8,8,32}，ceil+round_up+min）、make_SyncOp@0xdac440
    （vptr 0x5ec31f8=SyncOp）、make_dma_checkpoint_op@0xd95ac0（b!=0→Wait
    0x5ec2488 / b==0→Set 0x5ec2568）、force_contiguous_allocate_mcrecv@0xf82590
    （编排壳）、allocate_for_reschedule_grdep@0xf82970（set_tcm_pool/
    set_largest_memory_alloc_size/allocate_tcm_blocks 链，graph_prepare.cc:6141-6149）
  - create_supertiles 三阶段全解：Phase1 按 pair<resolved_id,tag> 分组
    [0x1313c58]；Phase2 单元素 erase+stable_sort [0x1313efc-0x13141a0]；
    Phase3 预算（gp[0x5fd8]<<10 / 命中 gp[0x74c0]，哈希表 gp+0x6e40）+除数
    搜索+chunk id (entry<<10)|seq（0x3ff 回绕跳 0x400 倍数）
    [0x131415e-0x13143ce]；fmt@0x55b4541='...:406::ERROR:unexpected value
    size for grouping...'（supertile.cc:406 逐字节验证）
  - op-id 间接表（GraphPrepare+0x7440）三解码函数全指令：resolve_table_index
    @0x12face0 / resolve_full_id @0x12fad60（远跳结果=(delta<<10)|(id&0x3ff)）
    / extract_field @0x12faf20（负 tag 读下一表项原值）；tag 位布局 bit31 溢出|
    bit30 chunk|3bit 类型<<27|5bit 字段<<22|22bit payload
  - grdep 侧 static（build_graph_deps 区间内，最近符号错觉）：fc5910=依赖记录
    有效切片数（0x200000→元素数或 0x40 累加；空/短 throw
    "num_internal_threads"；0x400000→尾调 self_slicing_num_slices）；
    fc5af0=per-chunk 依赖注册（count<=1 早退；resource_flags&0xc 必须 ∈{4,8}
    否则 ERROR 'grdep_main.cc:3457 does not have valid resource flags'）
  - 抽出并修正 spill_fill.hpp 三处文档错误（地址 0xd958d0→0xd95ac0；Set/Wait
    映射反了；serialize 槽位声明错误——Wait=0xd96b30/60/70，Set=0xd96c70/ca0/cb0）
  - 测试：tests/test_m36_supertile.cpp 42 项全过（ASan+普通双构建），
    既有 test_cost_model/test_context_bin 无回归
  - **M36b 补证（2026-08-28 同日，报告 v2 §5）**：
    - 黑名单写者闭环：静态初始化 thunk [0x710806-0x710865] 六次
      string_tag_t::map_str —— q::*InputSlicePad/q::*OutputSlice/q::Concat/
      q::ConvLayer.opt.activations_to_vtcm/q::SlicePad_shape_inplace/
      q::Slice_contig.tcm（字面量逐字节验证）；与记录 +0x28 tag 指针六连 cmp
      [0x1313b95-0x1313bc2]
    - 后缀串闭环：supertile.cc:138 DEBUG(4) 'Combine %zu tiles into supertile'
      @0x55b44b0 的装饰，static ~0x1313720-0x13137d1：flags&0x4→' (HVX)'
      [0x131375a]、&0x8→' (HMX)' [0x1313768]、皆无→' (error)'；另有独立非空标志
      加 ' (autothreaded)'；GetLogPriorityLevel()>=4 门 [0x131379a]
    - Phase1 资格谓词修正（反编译）：(count(gp+0x5fe8)>1 && flags&0x8) ||
      (count(gp+0x5fe0)>1 && (flags&0x110004)==4) —— 先线程数后资源位
    - 0x12fa730 尾部全解 [0x12fa880-0x12fab07]：tag→{type,value,seq} 三元组流
      （根恒 payload27+1；链循环 bit31 定 value 形态、bit30 定 next 来源：
      payload22 链续 / entries[idx] 原值 / 0 终链；远跳补 seq|=delta<<10）
    - fc5af0 尾部：三段 memset-0 [0xfc5c50/0xfc5ce0/0xfc5d00] + 逐成员调
      0xf9dda0（不变式 返回大小==i+1 [0xfc5e03-0xfc5e1e]），flags 门 &0x200/
      &0xf==5/&0x200000/&0x100400
    - 实现同步升级：CreateSupertilesConfig 改名字黑名单（默认 6 名）+hmx/hvx
      计数；SupertileCandidate 加 op_name/resource_flags/record_valid；新增
      decode_tag_triples；测试扩到 66 项全过（ASan 净）
  - **M36 遗留（反汇编未完全理解；M36b 后仍开放）**：
    1. 0x12fa5d5 远路径、0x12fb8a0（三元组追加原语）内部未读
    2. 0x12fa730 尾部结果记录 {终值, 终表项下标} [0x12faa0e-0x12faa40] 与
       倒序写槽的数组物理次序：语义未完全理解（实现按访问序，已声明差异）
    3. fc5af0 槽值精确公式（依赖 0xf9dda0）→ 归 build_graph_deps 里程碑
    4. Phase3 除数搜索的 c/b 操作数实体归属、chunk 起始 seq=0、0x7ff 哨兵
       跳过——均半证，实现中已注明
    5. insert_spill_fill 本 so 为桩，重建依据参考文档（证据来源差异已声明）
- **M37 P2：phys_alloc_in_runlist@0xf72b00（874B，带符号方法）** ✅ 完成（2026-08-28）
  - 报告：audit_verify/reports/M37_phys_alloc_in_runlist_disasm.md；
    实现：src/vtcm/phys_alloc_m37.cpp + include/hnnx/vtcm/phys_alloc_m37.hpp
  - 全指令闭环：`for op in runlist` → 三重门（this+0x6208 字节门 [0xf72b23] /
    this+0x74c8 指针后 map 的桶数!=0 [0xf72b47] / count(id) 命中）→ supertile
    成员扇出（节点 [0x38..0x40) vector<Op*>，null 成员 continue [0xf72dc1]，
    成员 vtable+0x48 返回值被丢弃 [0xf72dcf 无 test]）→ 本体 vtable+0x48
    [0xf72ce9] → 非 0 短路 + qnndsp_log('graph_prepare.cc:2173::ERROR:could not
    allocate memory for op %llx!!' @0x461d95b) [0xf72e48]
  - 桶定位内联 unordered_map find 两遍（幂桶 id&(n-) [0xf72bb5] / 非幂 id%n
    含 32 位 div 快路径 [0xf72bd6]；节点 [0]=next/[8]=hash/[0x10]=key）；
    第二遍是 unordered_map::at —— 断言串 'unordered_map::at: key not found'
    @0x398e262 实证 [0xf72e5e → 0x8c6970]；即源码 count()+at() 双查找
  - 遗留：节点 key→vector 间 node+0x18..0x37 的 0x20B 未读；vtable+0x48 槽
    方法名未证；get_extra_info(...)->[0] 字段含义半证；0x8c6970 内部未读；
    this+0x6208/0x74c8 的置位方在别处未定位
  - 测试：tests/test_m37_phys_alloc.cpp 12 项全过（ASan 净），M36/回归哨兵无回归
- **M38 P2：sequencing_stage@0x11e10a0 首批（阶段骨架 + 三大集成点）** [~] 骨架完成（2026-08-28）
  - 签名：`GraphPrepare::sequencing_stage(VtcmCacheInstance&, int&, bool, vector<tagx<1>>&, SequencingStageInfo&, bool)`；
    汇编 dump：audit_verify/asm/M37_sequencing_stage_11e10a0_11e615a.asm（命名沿用旧 M37 文件名）+
    M37_seq_statics_12934b0_129a000.asm（属 post_spill_fill_design_pass，非本函数）
  - 源文件：graph_sequencing.cc；阶段名/行号来自 168 条 .rodata 引用（`%s:行号:文本`）
  - **模式选择（prologue）**：this+0x5b04 int：0→"sfp"、1→"cbs"，写入 this+0x5560（std::string 阶段标签）
  - **阶段序列（行号:地址锚）**：:133 11e178f 调度入口（sched_threshold_ratio/tcm_reduction_index/is_retry）→
    :167/:168 11e1a2e/11e1a59 dp_seq/initial_sequencer->run → :220 11e1c8d Using stcut →
    :241/:244 11e2062 peak_tcm_pre_spill → :246 11e208c spill/fill 二次运行 →
    :280 11e1ede num_post_initial_sequencer_nodes [prepare_initial_sequencer_done] →
    :305 11e1f58 combine_fill_ops → :325 11e21e7 marking_weights → :364 11e2307 do_split_fork_join_oplist →
    :390 11e241a assign_blocks_to_spillfill_mgrps → :411-:420 move_joins_for_parallelism →
    :462 11e2754 TCM_CALC 校验 → :566-:616 CBS 使能链 → :639-:642 11e3748 CB 级搜索限制 →
    :671 hextimate → :716 selected_sequencer_intended → CB PRE0 11e4354 →
    [VTCM Allocation] 11e43d1 → :737 11e4428 rewrite_opmap_for_spillfill → :751 11e449e →
    :759/:761 分配结果 → retry 11e49ad → CB PRE2 11e4a51 [Parallelization Optimization] →
    :838-:875 11e4d0f 组播/MCID → :892 11e4fcb schedule_for_perf → :896 11e5018 conflict 约束 →
    CB POST0 11e51c8 → CB POST1 11e51fa
  - **集成点 1（P1→stcut）** [11e1cdc]：门 info+0x374（0/1）与 info+0x3c0；
    `Graph::new_id(0)`→[rsp+0x370] [11e1cb8]，`schedule_for_alloc@plt`
    (this, map@this+0x6d40, GraphDeps, out_runlist) [11e1cdc]；实体=**0x1302900(10286B)**
    ——P1 全部 st_cut 工作的宿主；随后 mark_time_point("prepare_stcut_done") [11e1cee]、
    show_runlist [11e1d1a]、dump_runlist("ST-Cut") [11e1d2e]；返回非零→错误路径 11e2ba4
  - **集成点 2（spill/fill 二次运行）**：:246 后 [11e20ad] 测 info+0x374：==0 →
    [rsp+0x268] 销毁旧 sequencer 并经 **0x122b000**（未符号静态，esi=1）重建 [11e2995]；
    两分支会合 [11e29fa] 做虚调用 `callq *0x10(%rax)`（run）→ r14d=状态 [11e2a0b]，
    runlist 从 [rsp+0x378..0x388] 搬入 [11e2a13-11e2a49]；非零→:258 错误
    [11e2a5c]，且 [rsp+0x58]==0 → 错误退出 11e2ba4，否则 :260 "can retry"
  - **集成点 3（P2→VTCM 分配）** [11e453b]：
    `link_source_destructive_operands@plt`=0xf82680 [11e44c7] → dump_runlist("Alloc") [11e4509] →
    4th bool = (!0x85f1d0(r15,0x462bd80) || this+0x59b0==0) [11e4518-11e4530] →
    **`allocate_for_reschedule_grdep@plt`(this, vtcm_cache@[rsp+0x68], runlist, bool) [11e453b]**
    → ebp：0=成功(:761)；**0xE=VTCM 超限** [11e4661 cmpl $0xe] → CBS 降阈值重试块
    （需 0x85f1d0 真且 this+0x59b0≠0）：0xf8ce00 重排 [11e4756]、:779/:784 日志、
    **二次 allocate_for_reschedule_grdep [11e488c]**、"CBS_" 前缀转储、
    mark_tp("prepare_retry_cbs_allocate_VTCM") [11e49b9]
  - **allocate_for_reschedule_grdep@0xf82970（294B，全解）**：graph_prepare.cc:6141-:6149；
    ① this+0x1d8 对象虚调用 vt+0x28，参数=vtcm_cache 的 (begin,end)（容器首两个字段）
    [:6141 set_tcm_pool] [f829ae-f829c3]；② 同对象 vt+0x30(this+0x5d08)
    [:6143 set_largest_memory_alloc_size] [f829f1-f82a09]；③ this 虚调用 vt+0x1c0() 取分配器
    [f82a37-f82a3d]；④ `fa::FancyAllocator::allocate_tcm_blocks(GraphDeps@this+0x7468,
    runlist, Options@this+0x54d0, bool)` [f82a57] → 返回 int
  - **P2 核心链**：allocate_tcm_blocks=**0x13b2a50(356B)** →
    allocate_tcm_blocks_internal=**0x13b2bc0(12155B，下一批目标)**
  - **schedule_for_perf 段**：:892 [11e4fcb] → 0x1004750（未符号静态=perf 主体）[11e4ff6] →
    mark_tp("prepare_schedule_for_perf") [11e5007] → 0x1004950（未符号静态）[11e5043] →
    :896 add_tcm_memory_conflict_constraints → show_runlist_cb/dump_runlist；
    hextimate 迹线用 0x10222f0/0x105d010（fa 区静态）[11e5115/11e512f]
  - 遗留（反汇编未完全理解）：0x85f1d0(r15,0x462bd80) 布尔谓词语义未证（r15=runlist 实参）；
    0x122b000 工厂、0xf8ce00、0x1004750/0x1004950 内部未读；multicast 段 11e4d0f-11e4fcb 未细读；
    this+0x59b0 三态(0/1/其他)完整含义半证；graph_sequencing.cc :905-:904 尾段未对齐
- **M39 P2/P6：allocate_tcm_blocks 链（桥接层全解 + 主体六段 + 0xE 闭环）** [~] 两批（2026-08-28）
  - **allocate_tcm_blocks@0x13b2a50（356B，全解）**：fa::FancyAllocator 方法；
    ① this+0xd4=flag(byte) [13b2a6c]、flag 真→this+0xd0+=1（尝试计数）[13b2a78]；
    ② opts+0xaf2==0 → 强制 {this+0xd4=1, this+0xd0=4} [13b2a8d/13b2a94]；
    ③ this+0xf8 容器 reserve（÷3 魔数 0xAAAAAAAAAAAAAAAB + 16B 元素跨度）[13b2aa5-13b2af5]；
    ④ **NSP 循环**：r15d=grdep+0x160（NSP 数）[13b2afc]；r15d==0 → 单次
    internal(..., nsp=0) [13b2b5d-13b2b7a]；否则 for(nsp=1..r15d)：
    `select_nsp@plt(this,nsp)` [13b2b25] → `allocate_tcm_blocks_internal@plt
    (this, grdep, runlist_view{begin,count=bytes>>2}, opts, nsp)` [13b2b47]，
    返回非零即中止 [13b2b4c]；
    ⑤ 收尾：this+0x100=this+0xf8 复位 [13b2b93] + 0xf4a6f0 容器清理 [13b2b9d]；
    返回首个非零码或 0
  - **allocate_tcm_blocks_internal@0x13b2bc0（12155B，骨架）**：
    dump=audit_verify/asm/M39_alloc_tcm_internal_13b2bc0.asm（2632 行）；
    **源文件=vtcm_alloc.cc**（:2425/:2429 行锚 + 文件名串 0x55bae43 实证——
    即计划 P6 的 vtcm_alloc.cc，P2 分配核心落在 P6 文件中）
  - 返回码分类（**续批修正**——首批误读 0x7FF/0x42 为返回码，实为算术常数）：
    **0**=成功（[13b4114] 等）；**0xD**=TCM 保留溢出 [13b5336]——:2425 错误
    "ERROR:Reserving %zd + %zd (pinned) = %d which is >= %zd bytes TCM" [13b5343]，
    参数：r8=rbp=X-budget(rbp-=r12+0xd8 [13b5333])、r9d=ebx、2 个栈参 [13b535b/13b535d]；
    **0xE 来自两个放置策略变体**（见下）。
    常数更正：0x7FF [13b30b1 等 3 处]=2KB 舍入加数 `(x+0x7FF)>>11`；0x42 [13b4c3d/13b50b1]
    =`66-lzcnt` 的 log2 常数（`orq $0xe`+bsr+xor $0x3f → 表容量 `1<<(msb+3)`，
    lzcnt≤5 → 抛错路径 [13b50da]），均非返回码
  - 异常分类：15 处 `__cxa_throw` 全部 `runtime_error("hash lookup failed"|"minimap::at")`
    [13b543c-13b5733]——容器断言，非业务错误
  - 值得注意的调用：`sort_blocks_by_reverse_lifetime_end@plt`（带符号，按生命周期终点逆排序）、
    `obtain_loc_of_tcm_blocks+0x220` 静态 [13b2bc0 区]、0x13b6350/0x13c2bc0/0x13b5e80 等
    fa 区静态未读
  - **开放项（0xE 之谜已解，续批）**：**0xE=VTCM 装箱失败码**，来自两个放置策略变体：
    `[rsp+0x30]`（序言 `opts+0x68 == "reserving"` 串比较 [13b2bf8-13b2c5a]）二选一——
    真→`0x13b2080(triples)` [13b40d3]、假→`0x13a66c0(triples)` [13b40ea]，返回码直通 r14d
    [13b40ef]（此前 13b40c7 先经 0x13b5e80 驱动相位）；变体 A：失败 `ebx=0xE`
    [13b2338] (:2048 "WARNING:Failure to allocate within the VTCM size of 0x%llx
    bytes!" / :2049 "Ran %d internal passes of VTCM allocation.")、成功 0 [13b2362]
    (:2070)；变体 B 同构 [13a7193=0xE :1210/:1211；13a71bd=0 :1266]；
    `r14+0x74` = 内部贪心迭代轮数（log 的 %d 参数）——**即 M38 中 sequencing_stage
    [11e4661] cmpl $0xe 触发 CBS 降阈值重试的来源**；两变体各自前段有
    `movslq r14+0x78` + `cmpl $5, [r14+0x78*8+0x80]` 门 [13a7146-13a7153]（策略表选择）
  - **主体结构（续批，[13b2bc0-13b31cc] 全解）**：
    ① 序言：opts+0x68=="reserving" 判定→[rsp+0x30]；`this+0x360`=opts+0x858（阶段标签）
    [13b2c65]；`this+0xd8 = (opts+0xe70 + 0x7FF) & ~0x7FF`（pinned 预留，2KB 向上取整）
    [13b2c79-13b2c90]；三个 48B 本地容器 + 0x13b5b40(rsp+0x160, 0x400) 初始化 [13b2ce4]
    ② 枚举主循环 [13b2d45-13b2d6e]：遍历 runlist i（view.count=[rsp+0x40]）；
    rec=grdep+0x80+(id-1)*0xD0；过滤：nsp≠0 → rec+0x18==nsp [13b2d92]；
    rec flags bit7（testb $-0x80）[13b2d99]；成员向量 rec+0x80/0x88
    ③ 成员过滤 [13b2df2-13b2e17]：mrec=grdep+0x148+(mid-1)*0x48，u16 flags@+0x20：
    bit0 清→跳过 [13b2e0d]；bit0 置+bit7 置→**容器 B**（u32 vector，[rsp+0x180..]）
    [13b2f50-13b2f71]；bit0 置+bit7 清→**容器 A**（12B 三元组 {block_id, run_pos, 0}，
    [rsp+0x1c0..]，初始 24KB/2048 元素 [13b2d01-13b2d24]）
    ④ [13b3098] 0x13b28d0(grdep?, 容器A, view.begin, view.count)——排序/整理静态
    ⑤ 三元组放置表构建 [13b30e6-13b31bd]：mrec+0x40==0 → block_id 置 0 [13b3101-13b3108]；
    0xfa2d50(rsp+0x1a0, mrec) 填区间容器 [13b3117]；mrec+0x00/0x08 = pair 向量；
    对每对 (first,second)：slot∈[first,second)：
    `this+0xf8[slot]*16 = {.., u16@+2=(this+0xe0[slot]*24 的值+0x7FF)>>11（2KB 舍入 KB）,
    u32@+4=run_pos, u32@+8=三元组[2], u32@+0xc=block_id}` [13b3180-13b31b2]
    ⑥ [13b3209] 0x13b5cb0(容器B.begin, 容器B.end, this+0xd8)——bit7 成员处理
  - **M40 变体 B 装箱循环骨架（0x13a66c0，2026-08-28）**：
    - **ctx 结构修正**：[rsp+0x1c0] 非纯 vector——0x13b5e80(内部) 是**驱动相位**，
      把分配上下文构造进去；三元组 vector 在放置表建成后即被释放 [13b31cc-13b31d7]，
      区域复用为 ctx（≥0x1E0B）。已证 ctx 字段：+0x00 三元组 vector 头（残留）、
      **+0x08 opts\***（读 +0x858 标签）、+0x18 表基址指针、+0x28 完成计数、
      **+0x30 待放项数**、**+0x34 钩子项索引**、+0x40 待放项数组（u32）、
      **+0x70 桶参数**、**+0x74 pass 计数**、**+0x78 当前策略号**、
      **+0x80[i*8] 策略梯**（==5 为终态）、+0x8c[i*8] 字节标志（序言扫描找首个非零）、
      +0x1ac/+0x1b0 桶数、+0x1b8/0x1c0/0x1c8 侵入链表头/尾/标志、+0x1d0 每 pass 清零 16B
    - **变体 B（非 reserving 默认）流程** [13a66c0-13a7216]：
      ① ctx+0x1ac=0 [13a6710]；序言扫 +0x8c[i*8] 找首个非零 → ctx+0x78=i [13a671b-13a673f]；
      若 +0x80[i*8]==5 直达报告 [13a6745]
      ② **pass 循环** [13a6770 回边]：ctx+0x74++ [13a6770]；ctx+0x1d0 清零 [13a6775]；
      侵入链表复位（splice +0x1b8/0x1c0、清 +0x1c8、逐节点析构）[13a6780-13a67d0]
      ③ 项循环（i∈[rsp+0x20]，界=ctx+0x30）[13a67dd]：项号=ctx+0x40[i] [13a6883]；
      entry=表基址[ctx+0x18 解引用]+0x50*项号，取 field@0/@4 [13a6887-13a689f]；
      构造闭包（vtable 0x6058c28，捕获 r12）[13a68cb-13a68da]；
      **0xf54470(ctx+0x1b8 链表, field0, field4, 闭包)** 收集候选 → 容器 rsp+0x70 [13a68fa]；
      0x13a3e60(容器, r12) 再收集 [13a6911]；
      **0x13a4d10(ctx, 容器, ctx+0x70, entry, 闭包) = 放置核心** [13a696c]；
      结果对 (u64: id=r12d, hi=>>0x20) 二分有序插入 entry+0x30 表
      [13a69d4-13a6a72]，撞键更新/插入 0x13a1eb0 [13a6a86/13a6adb]
      ④ 调试钩子：opts+0x858 != ""（0x4628a0e=空串，strlen=0 实证）且 i==ctx+0x34
      → 0x13a6250(ctx) [13a685e-13a6871；尾部镜像 13a712c-13a7131]
      ⑤ **收敛判定** [13a7136-13a7159]：placed(rbx) >= ctx+0x28 → 成功 0 [13a7138 jae→13a71b4]；
      否则 **0x13a7290(ctx)=策略步进**（改写 ctx+0x78/重建空闲表）[13a7141]；
      ctx+0x80[ctx+0x78*8]==5（策略梯耗尽）→ 报告路径 13a7159（:1210 WARNING、
      :1211 打印 ctx+0x74 轮数、ebx=0xE [13a7193]）；否则回 ② 下一 pass
    - 变体 A（"reserving"）[13b2080]：序言桶数学 ctx+0x1ac=ctx+0x70*30/16、
      ctx+0x1b0=前者-min(x,4)*ctx+0x70/4（x=[ctx[0]+0xd0]）[13b20b2-13b20e2]；
      主体 SIMD 化（movdqa 常数 0x4623520/30/40/50、0x398ec80、0x396ced0）——结构未读
  - 开放（反汇编未完全理解）：变体 A 主体 SIMD 细节、0x13b6350 清理；mrec(0x48) 字段
      语义半证（0x13b5e80 驱动相位/策略梯初始化 → M44 已解）
  - **M41 放置核心与策略步进全解（2026-08-28）** ✅
    （dump=audit_verify/asm/M41_place_core_13a4d10.asm，变体区补档
    audit_verify/asm/M39_variants_13a6690_13b2bc0.asm）
    - **0x13a7290 策略步进全解（1506B，13a7290-13a7871）**：
      ① 槽读取：idx=ctx+0x78；code=u32@ctx+0x80[idx*8]、paramA=u8@ctx+0x7e[idx*8]、
      paramB=u8@ctx+0x7f[idx*8]，code 溢存 [rsp+0xc] [13a72af-13a72c7、13a733f]；
      **策略槽布局修正**：0x7e/0x7f/0x80/0x8c 四个数组同用 i*8 步长
      ② 复位：表[ctx+0x18 解引用]每 0x50 步长 entry 的 +0x30 结果表复位（begin/end
      splice）+ +0x48 字节清零；主循环 0xA0 步长（一次两项），奇数尾单处理 [13a731d]
      ③ **code==0**（[13a7334 je→13a74bd]）：ctx+0x34=0、ctx+0x38=0；pending 空则直达
      epilogue [13a74d2]；否则 **SIMD iota 填充**（常数 0x4623520/30/40/50/60/70+
      0x398ec80/0x396ced0 全为 lane 偏移，非策略数据）[13a771b-13a784c] = 全量恒等序
      ④ **code==3 预计算** [13a75d8-13a7719]：ctx+0x58 桶向量（4B 元素）不足 ctx+0x20
      个 → 0xcf27b0=_M_fill_insert 补零 [13a75d8-13a760a]；循环 A：每项
      （first=entry@0、second=entry@4，first>second 跳过）对 j∈[first,second] 桶[j]+=
      entry@0xc——**桶压累积** [13a7617-13a7677]；循环 B：[first,second) 上**滚动最大值
      扫描**（rax 恒指当前最大，非局部峰）→ entry@0x14=峰值桶压 [13a7682-13a76b1]
      ⑤ **分区循环** [13a73b5-13a7484]：r13d=min(max(ctx+0x20,0)>>4,0x7D0=2000)
      [13a7378-13a7393]；r12d=(paramA==0)?(ctx+0x70)>>5:2 [13a73a0-13a73a9]；
      对每项读 f8=entry@8、f0xc=entry@0xc、f16=entry@0x10：
      **大件→rsp+0x40** 当 f8≥r13d 或 f16≥r12d 或（paramB≠0 且
      ((f8≥r13d/10 且 f16≥ctx+0x70>>8) 或 (f8≥r13d/4 且 f0xc≥ctx+0x70>>5)))；
      否则**普通件→rsp+0x20**；追加 0x111f290=push_back(vector*, &i) [13a7446/13a7460]
      ⑥ 助手分发：code 4→0x13a5f80(ctx,大件表,r13d) [13a74a8]；3→0x13a5cd0(ctx,大件表)
      [13a752d]；2→0x13a5a00(ctx,大件表,r13d) [13a753c]；其余(1)→0x13a5780(ctx,大件表)
      [13a754e]——四个助手内部**反汇编未完全理解**（排序/筛选启发式）
      ⑦ 汇合 [13a755b-13a75c9]：ctx+0x34=大件表元素数 [13a756f]；ctx+0x40 pending
      清空后 0x13ad2e0=vector::insert **先大件表后普通表** [13a7582/13a7598]
      ——pending=处理后大件++普通件（大件先放）；ctx+0x38=(ctx+0x34>0) setg [13a75a2]
      ⑧ **epilogue [13a784e]: ctx+0x78 += 1——每次调用步进一档**；调用方随即查新档
      code==5 → 0xE（与变体 B ⑤ 收敛判定闭环吻合）
    - **0x13a4d10 放置核心全解（833B，13a4d10-13a5051）**：
      签名 (ctx, container 占用区间表, limit=ctx+0x70, entry, 闭包)；闭包内表清空
      [r8+8]=[r8+0] [13a4d37-13a4d3f]；container 元素=8B {u32 start,u32 end}
      - 空表路径 [13a4fe6-13a5008]：size=entry@0xc；插 [0,size)；size>limit 返 0 否则返 1
      - **总空闲** [13a4d57-13a4d89]=Σ(start_i−end_{i-1})+max(0,limit−last_end)
      （start≥limit 即断）；free<size → 插 [last_end,last_end+size) 返 0 [13a4dc4 jb]
      - **first-fit** [13a4da0-13a4dbb]：首个 gap≥size 锚点 ebp；anchor+size≤limit →
      插 [anchor,anchor+size) 返 1 [13a4dcd jbe→13a500e→13a5011]
      - **慢路径** [13a4ddb-13a4fe1]（anchor+size>limit 但 free≥size）：container 复制到
      本地 vector [13a4dec-13a4e37]；entry+0x18/+0x20=**块 id 列表（u32 元素）**；
      外层每块：best-gap=argmax(最大间隙, 尾部) 锚点 [rsp+0x2c] [13a4e90-13a4edc]；
      块尺寸 u16=表[ctx+0x10 解引用]+id*16+2 [13a4ee4-13a4efa]（即 this+0xf8 放置表）；
      装得下→消耗循环（anchor+=块尺寸、剩余 r12d−=块尺寸、逐块 commit）
      [13a4f80-13a4fb6]；装不下→anchor 跳过全部剩余块尺寸和 [13a4f36-13a4f58]；
      每 chunk **双记账**：0xfa5be0(闭包,&start,&end) [13a4f69/13a4fca] +
      0x13a1b30(本地副本,start,end) 有序插入 [13a4fdc]
      - 返回值=!r14b [13a503c xorb]；冷块 [13a5052] setne——**块全消耗(rbx==r14)才返 1**；
      失败路径同样记账（诊断用）
      - **entry(0x50) 字段全图（M40 半证→全解）**：+0x00 first(生命周期起桶)、
      +0x04 second(止桶)、+0x08 尺寸A(比 N16 阈值)、+0x0c 尺寸B(桶压贡献/放置需求)、
      +0x10 尺寸C(比 ctx+0x70 派生阈值)、+0x14 峰值桶压(code3 写)、
      +0x18/+0x20/+0x28 块 id vector、+0x30/+0x38/+0x40 结果有序表 vector、+0x48 标志字节
      - **语义**：区间表 first-fit/best-fit 变粒度装箱；返回码供变体 B placed 计数收敛
    - **M43 四个策略排序助手全解（2026-08-31）** ✅（dump 同 M41_place_core；
      编号避让并行会话的 M42=tcm_migration）
      - 公共骨架：读大件表 → _Znwm 临时键记录数组（零填充）→ 逐项构造
      {排序键…, id} → std::__sort（深度限 2·ilog2n，16B/20B/24B 元素各一实例化：
      0x13b9f30/另/0x13bca80）→ SIMD 转置抽 id 列回写大件表 → 释放。全部取负键=降序。
      - **助手 1 = 0x13a5780（589B，策略码 1/"其余"）**：16B 记录
        {-f8, -f0xc, -second, id} [13a5840-13a5869] →
        **(尺寸A↓, 尺寸B↓, 止桶↓)** 纯 first-fit-decreasing
      - **助手 2 = 0x13a5a00（670B，策略码 2，参数 N16）**：20B 记录
        {flag=(f8<N16), k@4, k@8, -second, id}，cmov 双档 [13a5adb-13a5b0f]：
        flag=0 档（f8≥N16）@4=-f8/@8=-f0xc；flag=1 档（f8<N16）@4=-f0xc/@8=-f8 →
        **[f8≥N16: (f8↓,f0xc↓)] ++ [f8<N16: (f0xc↓,f8↓)]**，止桶↓ 决胜
      - **助手 3 = 0x13a5cd0（638B，策略码 3）**：20B 记录
        {-peak, -f8, -f0xc, -second, id} [13a5d90-13a5dc5] →
        **(峰值桶压↓, 尺寸A↓, 尺寸B↓, 止桶↓)**——与 code 3 预计算
        （桶压累积+峰值标注）衔接的峰值感知排序
      - **助手 4 = 0x13a5f80（670B，策略码 4，参数 N16）**：24B 记录
        {flag=(f8<N16), k@4, k@8, -f0xc, -second, id}，cmov 双档 [13a605b-13a6099]：
        flag=0 档 @4=-f8/@8=-peak；flag=1 档 @4=-peak/@8=-f8 →
        **[f8≥N16: (f8↓,峰值↓)] ++ [f8<N16: (峰值↓,f8↓)]**，f0xc↓、止桶↓ 决胜
      - **策略梯全语义**：同一"大件先放 + FFD"框架，四档排序键递进
        （纯尺寸 → 阈值双档 → 峰值 → 阈值×峰值），加 code 0 恒等序与 code 5 终态；
        梯上具体槽序/参数 → **M44 已解**（38 槽 .rodata 表 + f0/f1 消费语义）
    - **M44 驱动相位全解：ctx 构造 0x13b5e80 + 38 槽策略梯表 + 趟执行器 0x13b1470（2026-08-31）** ✅
      （dump：asm/M44_driver_13b5e80.asm、asm/M44_pass_exec_13b1470.asm；后者原字节亦在
      M39_variants 档 13a66c0-13b2bc0 内）
      - **ctx 构造 0x13b5e80**（rdi=ctx, rsi=FancyAllocator this, rdx=opts, rcx=入口表
        vector, r8d=count）字段写入：ctx+0x00=this（**修正** M40"陈旧三元组头"记载）、
        ctx+0x08=opts、ctx+0x10=&this+0xf8（放置块尺寸表）、ctx+0x18=入口表头
        （0x50 步长 entry 表）、ctx+0x20=桶/表槽数；**memcpy(ctx+0x7c, 0x55bb074, 0x130)**
        [13b5ed6-13b5ee2]=策略梯模板（38×8B .rodata 拷入）；侵入链表头 ctx+0x1b8/0x1c0/
        0x1c8 初始化；ctx+0x28=[this+0x48]−[this+0xd8]（TCM 预算字节）；
        ctx+0x30=(ctx+0x28>>4)×0xCCCCCCCD 低 32 位（5 的 mod 2^32 逆元；(bytes>>4)÷16=÷80）
        =entry 数 [13b600d-13b6026]；pending=0..count−1 iota（SIMD，常数 0x4623520..70
        为纯 lane 偏移）[13b6090-13b621c]；**ctx+0x70=ctx+0x28>>0xB（2KB 桶数）**
        [13b621e-13b6226]——放置上限语义钉死；ctx+0x74=0 趟计数
      - **38 槽策略梯表（.rodata 0x55bb074；8B 记录 {u8 f0, u8 f1, u8 paramA, u8 paramB,
        u32 code} 落 ctx+0x7c+8i）**：槽 0-11（f0=1,f1=0）：code1(B0/B1)、code2(B0/B1)、
        code3(A0/A1×B0/B1)、code4(A0/A1×B0/B1)；槽 12-23（f0=1,f1=1）同样 12 组合；
        槽 24（f0=1,f1=1,code1,B0/A0）；槽 25-35（f0=0）：code1(B1)/code2(B1)/
        code3(A×B)/code4(A×B)；槽 36 全零=code0；**槽 37 code=5 终态**（实证字节）
      - **f0/f1 消费语义（趟执行器内）** [13b1c1b-13b1c43]：R=ctx+0x1ac（保留区桶数，
        变体 A 序言计算，B 置 0）；大件（i<ctx+0x34 大件数）受限试放上限
        **= f0 ? R : 0（再 f1 ? ÷2）**，为 0 时改走无受限路径；普通件上限恒 R。
        **M45 修正**：该值是 0x13a4d10 的 limit 参数（受限区间上界）而非放置起点——
        保留区 [0,R)（f1 时上半 [R/2,R)）是优先落点，受限失败全量重试
      - **变体分派（两入口共享一梯）**：变体 B=0x13a66c0 默认——序言 ctx+0x1ac=0
        [13a6710] + 梯扫描 `rec[k+2].f0≠0`（基址 ctx+0x8c+8k，k 自 −1 起；
        **jne 回跳=遇 0 退出**，M40"找首个非零"系反向误载）→ ctx+0x78=24 [13a673f]，
        即跳过 R=0 下行为与 24-35 重复的前缀槽 0-23；终态判 code==5 [13a6745]；
        趟循环内联（0x13a4d10 直调 [13a696c] + 0x13a3e60/0x13a1eb0/0xf54470/0xf53a10），
        失败→0x13a7290 步进 [13a7141] 重试，终态 0xE 上报。
        变体 A=0x13b2080 reserving——R≠0 [13b20bf]，**ctx+0x78=0** [13b2156] 从槽 0
        起梯；每趟调趟执行器 0x13b1470 [13b22e8] + 0x13a7290 [13b22f4]
      - **趟执行器 0x13b1470（rdi=ctx）**：快照当前槽 f0→[rsp+6]、f1→[rsp+7]
        [13b1499-13b14a5]；复位段：清 ctx+0x1d0；摘除销毁侵入链表 ctx+0x1b8（节点
        +0x10 过 0x13b6660 后 delete）[13b14bd-13b151b]；回滚 ctx+0x230/0x248←
        ctx+0x228/0x240 [13b152d-13b1535]；清 ctx+0x268/0x2a0/0x2e0 三条 0x28 步长链表。
        主循环 i∈[0,ctx+0x30)：id=pending[i]（ctx+0x40 表）[13b1c46-13b1c4b]；
        大件 entry+0x48=1 + 0x13a98f0(ctx,id,受限上限) [13b1c9e-13b1cbc]；
        i==bigcount 边界且上限≠0 → 0x13a78d0 物化大件区间（构建 ctx+0x1e0）
        [13b1c79-13b1c85]、上限=0 → 0x13a9660 [13b1d1d-13b1d27]；普通件 →
        0x13aa750(ctx,i,受限上限,**f0 第 4 参**)
        [13b1d5b-13b1d6b]；每 128 个普通件向 ctx+0x320 追加 0x50B 检查点
        （0x13c0540/0x13c0220）[13b1cd0-13b1d3b]；项末 entry 结果末桶<<11（×2048
        桶→字节）对 ctx+0x28 预算做溢出判定 [13b1d73-13b1d96]，溢出且 opts+0x80≠0
        经 0x13ae0d0(ctx,&next_i) 恢复续跑 [13b1dac-13b1dc9]；opts+0x858≠""
        （空串字面量 @0x4628a0e）时在边界调调试钩子 0x13a6250 [13b1b8e-13b1c16]
      - **语义总结**：VTCM 内存池建立=以 2KB 桶为单位的区间装箱重试管线——
        0x13b5e80 构造（预算/桶数/梯模板）→ 变体序言定 R 与梯入口 →
        0x13a7290 排序（M41/M43）→ 0x13a4d10 放置（M41）→ 0x13b1470 趟调度/检查点/
        溢出恢复；槽 37 终态→0xE→上游 CBS 重排
    - **M45 四个放置包装全解（2026-08-31）** ✅（字节在 M39_variants 档；
      趟执行器→0x13a4d10 放置核心的桥）
      - **0x13a98f0（ctx, id, limit；13a98f0-13aa03c）大件受限放置**：
        entry=id×0x50，first/second 取出 [13a991c-13a9941]；在侵入链表 ctx+0x1b8
        注册回调节点 0xf54470（节点头带 0x6058c28 函子）[13a99cf]；0x13a3e60 由
        entry+0x18 块 id 表构造容器 [13a99ea]；**先受限试放 0x13a4d10(ctx,容器,
        limit=参数,entry,闭包)** [13a9a42]，失败→全量 limit=ctx+0x70 重试
        [13a9a99-13a9ab3]；成功后对 ctx+0x228/0x240（已放置项的有序起/止桶数组）
        二分定位 [13a9a70-13a9b34]，0x13b8800 合并闭包 [13a9b6a] 做**重叠消解**：
        冲突项经 0x13bedb0 摘链 → 挂 ctx+0x258/0x290 重排链 → 0x13a2e20 重排队 →
        再放置（limit=ctx+0x70）循环至失败 [13a9c47-13a9c4e]；收尾：闭包 {lo,hi}
        写 entry+0x30 结果表（0x13a1b30）[13a9d1c]；跨块时**块表（[[ctx+0x10]]=
        this+0xf8）u16@+2 = 块内起始偏移**逐块落盘 [13a9d40-13a9d64]；执行回调
        0xf53a10(链表,first,second,id) [13a9d7a]；first/second 有序插入
        ctx+0x228/0x240（0x13aa1b0）[13a9dfb/13a9f27]
      - **0x13a9660（ctx, id；13a9660-13a986c）大件无受限放置**：同骨架但直接
        **edx=ctx+0x70 全量上限** [13a975f]，无受限首试（f0=0 槽/上限为 0 时走此路）
      - **0x13aa750（ctx, i, limit, f0；13aa750-13ac279）普通件放置**：id=pending[i]
        [13aa77d-13aa78f]；ctx+0x38≠0（存在大件）时对 **ctx+0x1e0**（大件区间边界
        数组）做 lb(first)/ub(second_prev) 二分 [13aa855-13aa90a]，prev=
        i−(i≠bigcount)；**f0≠0 → 0x13a1b30(容器,0,ctx+0x70) 种入全新全空闲区间**
        （丢弃既有占用重建）[13ab744-13ab75e]，f0=0 → 复用现有占用容器；同样
        **先受限（edx=limit 参数）后全量（edx=ctx+0x70）** [13ab947/13ab9d0]，
        重叠消解再放置用受限 limit [13acdd2]
      - **0x13a78d0（ctx；13a78d0-13a924b）边界物化**：大件全放完后（i==bigcount）
        调用；**构建 ctx+0x1e0**（leaq 0x1e0 [13a7ae8]；0x13a5500 排序 + 大量
        _Znwm/memmove 向量构造）供普通件包装查询——即"保留区占用图"物化；
        内部 6.5K 细节反汇编未完全理解（M46 候选）
      - **0x13a4d10 参数语义修正（M41 补充）**：第 3 参 rdx=放置上限（受限试放时
        =保留区上界 R 或 R/2，全量时 =ctx+0x70 桶数），非"起点"
    - **M46 边界物化 0x13a78d0 外壳+数据流（2026-08-31）** ✅（字节在 M39_variants 档）
      - 序：复制 pending[0..bigcount)（大件 id，SIMD copy）[13a793a-13a7a9f] →
        **0x13a5500(ctx,&副本) 排序**：12B 记录 {rol64(entry 头 8B,0x20), id}=
        键 (first<<32)|second，std::__sort 升序 → **按（生命周期起,止）排序大件 id**
        [13a55af-13a55e4]
      - 主循环 [13a7b12-13a7d2f]：**按相同 second（止桶）分组**（相邻比较 [13a7b9f]）；
        组内逐成员 0x13a2e20(局部区间集@[rsp+0xf0], entry+0x30) **并置放置结果区间**；
        每组向 **ctx+0x200**（0x18B 步长记录，内嵌 vector&lt;u32&gt;）追加记录并压入
        成员 id [13a7b44-13a7b47、13a7be5-13a7c03]
      - 派生索引：ctx+0x1e0（普通件包装 lb(first)/ub(second_prev) 查询的边界数组）、
        ctx+0x1f8（对数组，0x13aa9d0 读 [−8]/[0]）、ctx+0x210 三者指针预置
        [13a7ad0-13a7aef]，后续 0x13be830/0x111f290 构造 [13a7e2c-13a7e5c]；
        三数组逐元素内容反汇编未完全理解（M46b 候选）
      - **语义**：把大件放置结果按生命周期物化为"保留区占用图"，供普通件阶段
        判定生命周期交叠与容器构造——两阶段（大件→普通件）装箱的桥
- **M38b P2：allocate_io_tensors@0xf69940 全解（2687B 无省略）** ✅（2026-08-28）
  - 命名说明：本条与上面 M38(sequencing_stage) 非同一工作；本条工件先于并行会话的
    M38 编号落盘，文件前缀沿用 M38_（report/M38_allocate_io_tensors_disasm.md、
    asm/f3/M38_allocate_io_tensors.asm），代码为 alloc_io_tensors_m38.*、测试 test_m38_alloc_io
  - 函数：GraphPrepare::allocate_io_tensors，graph_prepare.cc:6676/6696/6740 三行锚 +
    note_new_node 行号 0x1a38=6712；五段结构 A(输入节点)/B(输出节点头)/C(主循环)/
    D(新 Output 节点)/E(致命错)
  - **语义**：为图端点补建张量；Output 节点 op 列表（OpDef+0x30 vector<OpRef>）中
    OpDef+0x9 bit0 置位的 op 各生成一个 **q::QNN_Cast** 节点
    （驻留名 guard@0x6245270 [0xf69ff8-0xf6a032]），id=Graph::new_id(老id)；最后
    new 一个 **q::Output** 节点（'Output'@0x4696e72, guard@0x6245280 [0xf6a298]）携带
    （新/老）id 数组插入节点表 this+0x6d58 [0xf6a1b4]，老 Output 节点
    +0x9|=3 [0xf6a236] 或 mark_op_deletable [0xf6a24f] + collect_deletable_nodes [0xf6a257]
  - **Graph::new_id@0xd2f680（19B 全解）**：`(seq++ << 32) | (hint & 0xFFFFFFFF)`，
    计数器在 Graph+0x10 [0xd2f682-0xd2f68f]
  - **OutputDef 结构 80B(0x50) 双重证实**：改写循环步长 [0xf69f50] + 计数算术
    (表字节数>>4)×0xCCCCCCCCCCCCCCCD（5 的 mod 2^64 逆元，0x50=5×16）[0xf69e4d-0xf69e6f]
  - Cast 改写：OutputDef+4 枚举 1→7 且 +0x48 −=0x80；2→3 且 −=0x8000 [0xf69f40-0xf69f74]
    （算术已证，量化语义反汇编未完全理解）
  - make_op_node_impl 7 参签名 `(string_tag_t, y, OpRef const*, m, OutputDef const*, m, y)`
    [0xf69e7b, sym_master]；失败→return −1 [0xf69fe4]
  - this 布局新登记：+0x5340 输入节点 id / +0x5348 输出节点 id（失效均清 0）、
    +0x5350/+0x5368 vector、+0x5370 tensor-info 数组(16B 步长)、+0x45dc 配置字、
    +0x6d58 节点表 / +0x6d60 id→OpDef 表（两棵相邻树）、+0x6da0 vector
  - 遗留 8 项（报告 §8）：collect_multi_outputdef、0xd129e0、make_op_node_impl、
    改写的量化语义、0x10baa00、note_new_node/mark_op_deletable/collect_deletable_nodes 内部、
    OpDef+0x48 内嵌 80B 记录同构、新节点 +0x48..0x98 布局
  - 测试：tests/test_m38_alloc_io.cpp 34 项全过（ASan 净）；M36/M37/cost_model/context_bin 哨兵无回归
- **M36c P6：tile 子系统（tiler.h DSL / tiling::TileShapeBase）** ✅（2026-08-28）
  - 动机：V81 对拍缺口 G2（"25 参 tile 子系统"）；报告
    audit_verify/reports/M36c_tile_shape_subsystem_disasm.md
  - 函数：TileShapeBase 簇 0x137bea0–0x1391e40（get/OPTION_INT/OPTION_UINT/CONSTVAL_INT×2/
    IS_OP/PRODUCER_FOR/SAME_SHAPE/crouton·flat·weights×3 重载/minimize_tiling/
    gen_perf_Shape/debug）+ GraphOptInfo::declare_tiling_rule@0x138f8b0（160B 全解）
  - **统一名字解析模板**（六访问器同构）：`"*"`@0x57d37ee 通配→基准对象
    [0x138f95f]；名字表 8B 线性 strcmp 扫 [0x138f976-0x138f9a6]；命中→基准+0x30
    OpRef 表解引用 [0x138f9d3-0x138f9ed]；miss→`%s:888::ERROR:invalid name in
    TILE_SHAPE or TILE_SIZE specification`（tiling_registration.cc）[0x138f9af]
  - OPTION_UINT 硬件事务特例：tcm_size→nn_os_vtcm_get_hardware_size [0x138fab0]、
    tcm_size_for_tiling→GraphPrepare::get_vtcm_tile_size [0x138fad3]；其余查 ctx+0x54d0
    选项表（0x10f6590，内部遗留）
  - declare_tiling_rule：注册门 `testb $0x48,0xd(holder)` [0x138f8dc]；0x20B 记录
    `{fn=*([[holder+0x18]]+8), uint, holder, name}` 追加全局向量 0x6248428
    [0x138f8e2-0x138f914]；返回 0
  - 规则定义侧调用簇：A=0x1bcda00–0x1bd7000（conv 系）、B=0x29a4000–0x29e2000
    （ConvLayer 全变体/Batch_MatMul/Group `_G0`..`_G3`/SWIN_MHA…，51419 行 dump）；
    DSL 宏名 DEF_TILE_PROPERTIES / QHPI_TileShapeRequired（tiler.h，%s:36/148/241/263
    行锚）；两阶段 `tile shape apply`/`tile size apply`（0x567433b/0x5680436）
  - TinyVector<uint,8> 布局证实：+0 计数、+4 起元素（gen_perf_Shape [0x1391d65-0x1391d7a]，
    minimize_tiling=ctx+0x5554>0 [0x1391d27] 时全零）
  - **"25 参"核验：未证实**——参数面为三层：角色名空间（开放按 op）+ 选项键簇
    （central_tiler 13 键+周边）+ 8 维 TinyVector；无 25 元定长接口。已列入对拍核验点
  - 遗留 7 项（报告 §2）：0x10f6590/getconst_int_impl、三胞胎 OpDef 重载主体、
    DType+TV 布局构造体、0x1391de0 无名驱动、簇 B 逐规则、比较循环尾、M40+ 锚修正
  - 测试：tests/test_m36c_tile_shape.cpp 20 项全过（ASan 净）；五哨兵无回归
- **M42 P2：GraphPrepare::tcm_migration(uint,bool) @0x13219c0 外壳全解** ✅（2026-08-28）
  - 动机：对拍缺口 G3（"三分之四规则"）；报告
    audit_verify/reports/M42_tcm_migration_disasm.md；dump
    audit_verify/asm/f3/M42_tcm_migration.asm（2297 行）
  - **三分之四规则已证实**：`(get_vtcm_tile_size()/4)*3`，uint32 先除后乘
    [0x1321bc8-0x1321bda]；全量/3/4 成对存配置块(rsp+0x140)+0x28/+0x2c
  - 外壳显式比较（P4 估计 [0x13227d2]、P6 B 支 [0x1322ba0]、P6 判决 [0x1322f8b]、
    P11 复查 [0x1323d0f]）**一律用全量预算**；3/4 值仅经配置块供 helper（M42b）
  - 阶段流水 P0-P12：DCE 前处理（动态输入激活时二次 DCE）→ 预算 → TLS 门(1819)
    → scan(1824) → 六遍 op 扫描 → 错误计数边界(1829) → 转换四连 0x132a8b0/20/
    0x132bbf0/0x132c1f0 → ctx 终结化 + "prepare_tcm_migration" 计时点
  - 判决语义：需求>全量 → **3472 ERROR（fmt@0x55b5ba1，带预算+需求两实参）+计数**
    [0x1322f8b-0x1322fbd]；否则 GetLogPriorityLevel>0 → **3459 WARNING 不计数**
    [0x1322fc7-0x1322ff2]。累计口径=每张量 (size_after−size_before)（变换 0x1332210
    前后各测一次 0x132ca20）[0x1322d9f-0x1322da7]
  - 勘误两则：0x55b5ba1=3472（原误记 3459）；0x5555…序列=SWAR popcount 单比特测试
    [0x1323594-0x13235f4]，非除法魔数
  - 杂项锚："q::TableLookup" string_tag 静态初始化 [0x13227f5-0x1322838]；
    ctx(rsp+0x78) 字段图含 +0x30 vector<OpDef*>（六处迭代）与 +0x78 错误计数
  - 遗留（M42b，报告 §5 十项）：0x13256f0 scan、0x1332210 变换（3/4 值消费点）、
    0x132ca20 尺寸、0x13318c0 估计器、转换四连内部、0x133dce0 属性传播、
    pkg_flag 相邻全局 0x6247d38-0x6247d50、this+0x6b50-0x6b70 字段
  - 测试：tests/test_m42_tcm_migration.cpp 37 项全过（ASan 净）；五哨兵无回归
- **M47 P2/P5/P6：spill/fill 四层全解（G4 收口）** ✅（2026-08-31）
  - 动机：对拍缺口 G4（蓝图 §8.8 行 1899）；报告
    audit_verify/reports/G4_spillfill_disasm.md；dump
    audit_verify/asm/f3/G4_grdep_spillfill.asm（11177 行）+
    G4_slc_allocator.asm（6719 行）+G4_slc_post_pass.asm（10708 行）+
    全库 objdump /tmp/htp/all_text.asm（846MB，可 grep 调用图）
  - **四层结构**：① 桩 `GraphPrepare::insert_spill_fill`@0x106d7d0=log 后
    return -1（"insert"在 .so 里是从未实现的 API 面）；② 引擎 grdep_spillfill.cc
    （0x100b000-0x1016000，SFCD 生成/重写/检查点/DMA join）；③ 序列化
    slc_allocator.cc（rec_type 0=spillfill/1=waitfor/2=setprogress 三值枚举
    [0x1294cdb 分派]，0x40 步长记录，键表 8 键实证）+slc_graph_prepare.cc
    （post_spill_fill_design_pass@0x129ed30 驱动，门 [this+0x611d]）；④ 分配面
    （FancyAllocator 三槽 [0xf4ccf0] 每槽+64KB、池 2、shared(bit0)/far(bit4) 位
    [0xf4ce98/0xf4cec2]、far 阈值 env_MB<<20 [0xf4ceb3]、is_shared_spillfill
    [0xd8b550]、make_dma_checkpoint_op@0xd95ac0 双 vtable 0x5ec2488/0x5ec2568）
  - **SFCD 二进制格式全解**（dump 区 0x1013a90，内联容器=
    serialize_blob_epilogue@0xFDB330）：3 词头（total/checkpoint/记录数&0xFFFFFF）
    +四类记录（tcm 块 (pool<<16)|nblocks / 0x80 wait（bit16 单词·对两形态）/
    0x81 set / 非法:2095 双打印后**立即返回** [0x1013f34-0x1013f42]）；子项
    tcm_off=w&~0xF、type=w&0xF、游标 64B 步进 [0x1013c58]；尾校验
    游标−基址==[+0]+4 [0x1013e93]；nblocks==0 合法 [0x1013bcc]
  - grdep 三槽填法（调用点 [0x1010c09-0x1010cc5]）：多域=逐槽 64K 取整；
    单域=argmax（峰值 2 严格大→槽 2，否则 p1>p0?1:0），全零不分配
  - 遗留（G4b，报告 §8 七项）：0x129f0b0 设计主体、generate_internsp_spillfill、
    rewrite_op 簇、fill_mgroup_after_ops 谓词、SFCD 写侧与 type nibble 表、
    dlbc_spill_fill_setup/force_contiguous_allocate_mcrecv_blocks、:244 侧预算
  - 测试：tests/test_g4_spillfill.cpp 121 项全过（ASan 净）；六哨兵无回归
- **M40+ P2–P6 管线其余** [ ]（优先级由下方对拍缺口清单排序，源文件名已定位）
  - P2 余：allocate_tcm_blocks_internal 两变体装箱循环（M39 主体六段 + M40 变体 B 骨架 +
    **M41 放置核心/策略步进已全解**；dump=audit_verify/asm/M39_variants_13a6690_13b2bc0.asm；
    余变体 A 主体 SIMD 细节与 0x13a78d0 派生三数组逐元素内容〔M46b〕
    （外壳+数据流 → M46 已解）；
    ~~0x13b5e80 驱动相位~~〔策略梯槽序初始化+38 槽表+趟执行器 0x13b1470 → M44 全解〕、
    ~~四个放置包装 0x13a98f0/0x13aa750/0x13a9660~~〔M45 全解；0x13a78d0 外壳已解内部未读〕）、
    ~~tcm_migration@0x13219c0~~
    （M42 外壳全解，剩 M42b helper）、~~四策略排序助手~~（M43 全解）；P3 fa_alloc.cc@0x461a177；
    P5 insert_spillfill.cc@0x462ae1f（~~桩~~〔M47 全解：永久桩+SFCD/slc/分配面四层；余 G4b〕）；
    P6 slc_allocator.cc@0x55af563 / tiling_registration.cc@0x55b9f70 + tiler.h DSL
    （**锚修正**：原记 tiler.cc@0x55b98a4 有误；tile 子系统主体已由 M36c 解出，
    余簇 B 逐规则与布局构造体）（vtcm_alloc.cc 已由 M39 进入）；
    另：sharing_plan.cc@0x55b3acb、run_order_to_alloc_info.cc@0x55aefa1。

#### M40 对拍缺口清单（2026-08-28，源自 V81_GRAPH_COMPILER_ARCH_DESIGN.md 评审稿）

上游蓝图 V81GC 把下列逆向知识当作"已复原"引用，但本工程实际状态参差——
**此表即 M40+ 的排序依据**（蓝图 M-E2/M-E3 里程碑的硬依赖排最前）：

| # | 蓝图引用处 | 它以为已有的知识 | 本工程实际状态 | 对应待办 | 蓝图里程碑 |
| --- | --- | --- | --- | --- | --- |
| G1 | 8.7.2 | CBS 全量（约束调度/双种子/四快照/0xe 重试环） | M39 主体六段+0xE 闭环；**装箱循环 M40-M44 已通读**（变体 B 内联趟/变体 A 趟执行器/策略梯/放置核心/四排序助手）；余变体 A SIMD 细节+0x13a78d0 派生数组逐元素内容 | M46b | **M-E3 前置（高）** |
| G2 | 8.4.3 | SuperTile 聚类全貌 + **25 参 tile 子系统** | M36 create_supertiles 全解；**M36c tile 子系统（tiler.h DSL）全解，"25 参"核验未证实**（参数面=角色名空间+选项键簇+8 维 TV）；fc5af0 尾槽值公式仍遗留 | ~~M36c~~ 剩尾归 build_graph_deps | **M-E2（中高）** |
| G3 | 2.5 | tcm_migration "三分之四规则" | **M42 外壳全解**：`(tile_size/4)*3` 指令级证实（先除后乘）；外壳显式比较用全量预算、3/4 值入配置块+0x2c 供 helper；helper 内部（0x1332210 等）= M42b | ~~M42~~ 剩 M42b | M-E1+（中） |
| G4 | 8.8 | spill/fill 六类分类（蓝图称"2.4 复原"） | **M47 全解**：P5 insert_spill_fill@0x106d7d0=永久桩；SFCD 二进制格式（内联于 serialize_blob_epilogue@0xFDB330，dump 区 0x1013a90：4 类记录+64B 步进+尾校验）；slc rec_type 三值枚举 0/1/2 [0x1294cd7]；分配面（池 2 三槽 [0xf4ccf0]/shared·far 位 [0xf4ce98/0xf4cec2]）；**"六类"指令级不存在**——3 类记录×fill/spill 方向×near/far×3 槽；§8.8→§2.4 引用断裂；grdep 重写簇+0x129f0b0 设计主体=G4b | ~~M47~~ 剩 G4b | M-E3（中） |
| G5 | 4.7 | set_lifetimes + fa 扫描线装箱 + 常量池 near/far | 早期里程碑部分覆盖（需核对 fa::FancyAllocator 内部） | 核对 | M-A/B（低） |
| G6 | 8.2.1 | order_nodes 双链表 ping-pong/种子升序/批内 LIFO/批间 FIFO | 早期里程碑（M24 flood/collect 系）部分覆盖，需核对 | 核对 | M-B（低） |
| G7 | 4.7 | 八段 ctxbin 字节级 | M24/M27 序列化系列 + test_context_bin 哨兵在跑 | 大体已有 | — |

蓝图转述断言的**核验点**（按本工程规矩"反汇编优先于文档"，推进到相应位置时逐条核，
对不上须回告蓝图改文档）：

- [x] "0xE 重试环"（蓝图 8.7.2）——已证：11e4661 cmpl $0xe ↔ 13b2338 ebx=0xE，闭环
- [x] "do_prepare2_retry_loop 同型"降级链（蓝图 4.5）——已证：sequencing_stage retry 块同型
- [ ] "Phase D = const-prop/CSE/DCE fixpoint"（蓝图 7.1）——pass 注册表侧待核
- [ ] "per-layer avg 6 inst"（蓝图 9.4.3，QNN 生产产物指令密度）——序列化侧待核
- [ ] "QNN 解剖 ~200MB KV 区"（蓝图 8.5.3）——ctxbin 常量池侧待核
- [ ] "L3 字节确定性工程"（蓝图 4.7）——序列化侧待核
- [x] M38b 补充：quant 族（蓝图 6.3 族 11）的 Cast 插入模式——已证图出口 OpDef+0x9 bit0
      → q::QNN_Cast + OutputDef 枚举 1→7(−0x80)/2→3(−0x8000) 零点平移算术 [0xf69f40-0xf69f74]
- [x] "25 参 tile 子系统"（蓝图 8.4.3/ADR-7/Q.3）——**核验未证实**（M36c）：tile 子系统
      实体=tiler.h 按名取参 DSL；参数面为角色名空间（开放）+central_tiler 13 键簇+
      TinyVector<uint,8>；指令级无 25 元定长接口。**须回告蓝图改述或补出处**
- [x] "三分之四规则"（蓝图 2.5）——**已证实**（M42）：`shrl $2` + `leal (%rax,%rax,2)`
      = (tile_size/4)*3，uint32 先除后乘 [0x1321bd4-0x1321bda]；存配置块+0x2c。
      外壳显式比较（P4/P6/P11）一律用全量预算+0x28，3/4 值由 helper 消费（M42b）
- [ ] "Phase D = const-prop/CSE/DCE fixpoint"（蓝图 7.1）——**外壳级旁证**（M42）：
      tcm_migration 前处理逐字调用 dead_code_removal_and_cse（动态输入激活时两次）
      [0x13219fa/0x1321a1a]；pass 注册表侧仍待核
- [x] "spill 缓冲走 mempool near/far（2.4 复原的六类 fill/spill 分类）"（蓝图 8.8
      行 1899）——**半句证实半句不成立**（M47）：near/far 证实（PoolDesc+0x1e
      bit4=far，阈值 env_MB<<20 [0xf4ceb3]，谓词 can_mempool_be_far@0xf4cee0，
      注入面 set_shared_spillfill@0xd280b0）；"六类分类"指令级**不存在**——实际
      分类=3 类 SFCD 记录（tcm 块/wait/set）×fill/spill 方向（is_fill 参数
      [0x1013b5f]）×near/far 池位×3 槽 mempool；§2.4 全文是三条器件红线、无任何
      分类表，**引用断裂**；最接近"六"的是蓝图 §8.4 move 族（dma/copy/spill/fill/
      mcsend/mcrecv）——设计词汇而非 .so 指令层。**须回告蓝图改述 §8.8 并修引用**

---

### 阶段 E: 验证 ✅ 完成（本机受限）

**E1. 语法检查** [x]
- `g++ -fsyntax-only graph_prepare.cpp`：通过（0 错误，本次修改为纯注释）
- `g++ -fsyntax-only fancy_allocator.cpp`：我修改的 447-883 区域通过；
  唯一报错在 **1175 行** `std::min(vtcm_size_, VTCM_4MB_LIMIT)`（size_t vs uint64_t
  类型不匹配）—— 这是 **macOS 特有问题**（LP64 下 size_t=unsigned long、uint64_t=unsigned long long），
  Linux x86-64 目标下两者同为 unsigned long，不存在此错，且与本反汇编修正无关（预存问题，非本次引入）。

**E2/E3. 全量构建/测试** [部分受限]
- 本机无 cmake（仅 make），且目标为 x86_64-linux-clang（Linux 交叉编译），无法在本 macOS 环境完整构建运行。
- 需在 Linux 目标机（含 V81HexSim lib）上执行 `cmake .. && make` + 全部 test_*.exe 完成最终回归。

---

## 验收标准

每个修正项必须满足：
1. 实现代码中的每段逻辑都有对应的反汇编指令地址注释
2. 不得有"我推断"、"可能是"、"应该是"等措辞
3. 如果反汇编结果与原实现冲突，以反汇编为准
4. 如果某段反汇编无法理解（如高度优化的 SIMD），标注"反汇编未完全理解"而非编造
