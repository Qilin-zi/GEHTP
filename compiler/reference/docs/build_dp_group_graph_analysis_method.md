# build_dp_group_graph 切分逻辑 —— 反汇编确认方法论

> 目标:确认 `libQnnHtpPrepare.so` 中 SuperTile 前置的 "build DP group graph" 切分逻辑。
> 目标 .so: `libQnnHtp.so` (x86_64-linux-clang, QAIRT 2.48.0.260626, 103 MB)
> 工具链: Python 3.11 + pyelftools 0.33 + capstone 5.0.9(已在本机装好)
>
> ⚠️ **核心结论先说**:`build_dp_group_graph` 这个函数名在二进制里**不存在**。它是反汇编整理文档作者对 `create_supertiles` 内部"构建 DPGroupGraph"逻辑的**臆测命名**。真实逻辑分散在 `DPGroupGraph` 类的多个方法中(已通过 RTTI 字符串证实类与方法真实存在)。

---

## 1. 环境准备(已就绪)

| 项 | 状态 |
|----|------|
| .so 文件 | `C:\Users\RQILIN\Downloads\qairt-v2.48\...\lib\x86_64-linux-clang\libQnnHtp.so` (103MB) ✅ |
| Python | `C:\Users\RQILIN\AppData\Local\Programs\Python311\python.exe` 3.11.9 ✅ |
| capstone | 5.0.9 ✅ (`pip install capstone`) |
| pyelftools | 0.33 ✅ (`pip install pyelftools`) |
| objdump/ida/ghidra | 无(不需要,capstone 足够) |

---

## 2. 确认流程总览(6 步)

```
Step 1: 解析 ELF dynsym,定位导出函数地址
Step 2: 反汇编 create_supertiles,提取 call 目标(内部 stripped 函数)
Step 3: 全局字符串扫描,定位 dp_group_graph.cc 文件 + DPGroupGraph 类的 RTTI
Step 4: 扫描 .text 中引用 "dp_group_graph.cc" 字符串的指令,圈定代码区间
Step 5: 反汇编该区间,按函数边界切分,提取每个函数的日志字符串
Step 6: 从日志行号 + RTTI 方法名还原真实切分语义
```

---

## 3. 各步关键证据

### Step 1:dynsym 给出 3 个导出锚点

`.symtab` 被 strip,只剩 `.dynsym`(7690 个导出符号)。DPGroupGraph 相关**全部不在导出表**(内部静态类),但能定位到:

| 地址 | size | 符号 |
|------|------|------|
| `0x13138d0` | 5244 | `GraphPrepare::create_supertiles()` |
| `0x1314d50` | 591 | `GraphPrepare::make_one_supertile(...)` |
| `0xfac220` | 8216 | `GraphPrepare::build_graph_deps()` |

### Step 2:create_supertiles 调用 21 个 stripped 内部函数

反汇编 `0x13138d0`(5244 字节,1246 条指令)提取 21 个 `call` 目标,**全部 UNRESOLVED**(被 strip):

```
0x6ed270 x9   0x6f3480 x8   0x6ef2b0 x5   0x6ed690 x3   ...
0x13152f0 x1  0x1316bc0 x1  0x1316c00 x2  (邻近 supertile.cc)
```

`create_supertiles` 的日志字符串全部是 `supertile.cc` 行号(267/276/285/297/314/320/328/341/406),内容是 **supertile 分组判定**(duplicate offset / non-consecutive / too large / trivial / SD info),**没有 dp_group_graph 相关** —— 证明它只是**消费** DPGroupGraph 结果,不是构建者。

### Step 3:字符串扫描证实 DPGroupGraph 类真实存在(非 Ghidra 推断)

全文件扫描 `dp_group` / `DPGroupGraph` 命中关键证据:

| 字符串 va | 内容 | 证实 |
|-----------|------|------|
| `0x469ed11` | `dp_group_graph.cc` | 源文件名存在 ✅ |
| `0x55a8d05` | `dp_group_graph.json` | graphviz dump 调试输出 ✅ |
| `0x55a8e90` | `dp_group_graph_with_inputs.json` | 带 inputs 的 dump ✅ |
| `0x55a908a` | `dp_group_graph_unbroken.json` | "未切分" dump ✅ |
| `0x469f69b` | `12DPGroupGraph` (RTTI typeinfo) | 类的 RTTI ✅ |
| `0x469f6b3` | `7DPGraphI11DPGroupNodeE` | 继承 `DPGraph<DPGroupNode>` ✅ |
| `0x469f6ca` | `6DPNodeI11DPGroupNodeE` | 继承 `DPNode<DPGroupNode>` ✅ |
| `0x469fabe` | `ZN12DPGroupGraph39split_merge_ops_into_predecessor_groupsE...` | **真实方法名** ✅ |
| `0x469fd44` | `ZN12DPGroupGraph22isolated_s_dim_breakupEvE3$_4` | **真实方法名** ✅ |
| `0x1198760` | `ZN7new_mlh31BaseDPMLSelectorFeatureSuperSetC1ERK9DPOpGraphRK12DPGroupGraph...` | DP 选择器类 ✅ |

**结论**:`DPGroupGraph`、`DPOpGraph`、`DPOpNode`、`DPGroupNode`、`BaseDPMLSelectorFeatureSuperSet` 都是真实 C++ 类,有 RTTI。但 `build_dp_group_graph` 这个名字**没有任何字符串/符号匹配** —— 它是文档作者的伪命名。

### Step 4:dp_group_graph.cc 代码区间 = 0x115000–0x1167xxx

扫描 `.text` 中所有 `rip-relative lea/mov` 引用 `dp_group_graph.cc` 字符串的指令,共 ~40 处,全部落在:

```
0x11511a3 ~ 0x1167b4b   (dp_group_graph.cc 代码区间)
```

邻近还有两个具名方法的 RTTI 引用点:
- `0x11752d0` — `split_merge_ops_into_predecessor_groups` 的 lambda invoker
- `0x117bbf0` — `isolated_s_dim_breakup` 的 lambda invoker

⚠️ **坑**:这两个地址其实是 `std::function` 的 **lambda 闭包 invoker**,不是切分主体函数。反汇编后内部全是 popcount(`0x5555...`/`0x3333...`/`0x0f0f...`)+ hash 常量 `0x9ddfea08eb382d69` + 链表遍历 —— 用于在 `unordered_map<DPOpNode, DPGroupNode>` 里**查桶**。真正的方法体在 0x115000–0x1167xxx。

### Step 5:区间内 14 个真实函数(dp_group_graph.cc 主体)

按函数边界(`push rbp/r15` + `sub rsp` 开头,`ret` + nop padding 结尾)切分 0x114f00–0x117c500,找到 14 个带 `dp_group_graph.cc` 日志的函数:

| 起始地址 | size~ | 角色(从日志推断) |
|----------|-------|-------------------|
| `0x1150ec0` | 1664 | group 初始化 |
| `0x1151540` | 528 | op_id → layerId 解析(concat-tree fallback) |
| `0x1152370` | 5088 | **concat-tree group creation** + op_id based fallback |
| `0x11537b0` | 2560 | input groups layer_id |
| `0x1155b40` | 960 | stat 收集 |
| `0x11564d0` | **19856** | **★ 主体:merge lower 32 groups + branch linear order + popular groups + group breakup** |
| `0x115b260` | 2160 | stat |
| `0x115c0c0` | 9520 | **group breakup + branch linear order 设置** |
| `0x115ecb0` | 2560 | breakup propagate |
| `0x115f6b0` | 4368 | break merging / invalid linear order 处理 |
| `0x11607c0` | 3712 | "unable to break any groups" 退出路径 |
| `0x11677a0` | 1376 | **DDR break up**(行 1785/1789/1790/1796/1797/1802) |
| `0x1167e40` | 1536 | DDR break up 续 |

### Step 6:dp_group_graph.cc 完整日志行号(切分语义还原)

按行号排序的日志字符串(每个 `%s:NNN:` 对应源码一行):

```
行 232:  WARNING: Unknown output op type "%s"
行 330:  Was looking for OpID = 0x%llx (%s) but not found in concat_result_graph
行 337:  ERROR:0x%llx Expecting all ops in op_def_map to have offsets
行 346:  Was looking for OpID ... found in concat_result_graph but layerId is %zu which is not >0
行 353:  Attempted concat-tree group creation, but data is unavailable,
         falling back to op_id based groups.        ← ★ 切分策略选择
行 623:  STAT
行 631:  Forcing op_id_based_layer_id for input groups
行 824:  STAT
行 943:  STAT: num_post_merge_lower_32_groups / AFTER MERGE LOWER 32 GROUPS  ← ★ 合并低 32 组
行 998:  STAT
行 1066: Setting branch linear order inside group breakup        ← ★ branch 顺序
行 1167: STAT
行 1214: STAT
行 1225: Warning: Forced to change branch heuristic due to TCM break up
         propagate or break of merged groups, even though it was locked
行 1326: Uh-oh, was not able to break any groups. Best we can do now
         is exit group breaking and hope we fit.        ← ★ 切分失败兜底
行 1362: Warning: Branch linear order is invalid, perhaps due to
         breaking up merged groups. Skipping break mering
行 1769: STAT
行 1776: STAT
行 1785: Warning: Forced to change branch heuristic due to DDR break up ← ★ DDR 切分
行 1789/1790/1796/1797/1802: STAT (DDR break up 阶段)
行 1887: ERROR:dp_node is null
行 2254: branch_linear_order is largest_branch_first          ← ★ branch 启发式
行 2293: branch_linear_order is df_level_isinput_name
行 2307: branch_linear_order is df_io_distance_size_name
行 2453: POPULAR_GROUPS: group: %s, parent_max_spill: %8zuB,
         num_moved_groups: %5zu                           ← ★ popular groups 迁移
行 2462/2463: STAT: popular_groups_num / popular_groups_single_child_num
行 2493: average dram footprint before moving outputs: %zd
行 2526: average dram footprint before moving expanding groups: %zd
行 2649: average dram footprint: %zd
```

---

## 4. 还原后的真实切分逻辑

**`build_dp_group_graph` = DPGroupGraph 构造,真实流程(从日志行号 + RTTI 方法名还原):**

```
DPGroupGraph 构造流程 (dp_group_graph.cc, 行号顺序):

1. concat-tree group creation (行 330-353)
   ├─ 在 concat_result_graph 里查每个 OpID 的 layerId
   ├─ 若 concat-tree 数据可用 → 用 concat-tree 分组
   └─ 否则 → fallback 到 op_id based groups (行 353)
      └─ input groups 强制用 op_id_based_layer_id (行 631)

2. merge lower 32 groups (行 943)
   └─ num_post_merge_lower_32_groups / "AFTER MERGE LOWER 32 GROUPS"

3. branch linear order 选择 (行 1066, 2254-2307)
   ├─ largest_branch_first          (行 2254)
   ├─ df_level_isinput_name         (行 2293)
   └─ df_io_distance_size_name      (行 2307)
   └─ "Setting branch linear order inside group breakup" (行 1066)

4. TCM break up (行 1225, 1362)
   ├─ 若 VTCM 超预算 → 切分 merged groups
   ├─ 可能强制改变 branch heuristic (即使已锁定, 行 1225)
   └─ 若 linear order 失效 → 跳过 break merging (行 1362)

5. group breakup 主体 (0x11564d0, 19856 字节)
   ├─ break merged groups until fit
   ├─ 失败兜底: "was not able to break any groups... exit and hope we fit" (行 1326)
   └─ isolated_s_dim_breakup()  ← 真实方法名(RTTI 证实)

6. popular groups 迁移 (行 2453-2649)
   ├─ POPULAR_GROUPS: 按 parent_max_spill 迁移 group
   ├─ moving outputs (行 2493) → moving expanding groups (行 2526)
   └─ 计算 average dram footprint (行 2649)

7. DDR break up (行 1785-1802)
   ├─ 若 DDR 超预算 → 切分
   └─ 可能再次强制改变 branch heuristic (行 1785)

8. split_merge_ops_into_predecessor_groups()  ← 真实方法名(RTTI 证实)
   └─ 将 merge op 切分到前驱组 (lambda invoker @ 0x11752d0)

输出: DPGroupGraph (op 被分组为可独立分配 VTCM 的子图)
   ├─ dump 调试: dp_group_graph.json / _with_inputs.json / _unbroken.json
   └─ 供 create_supertiles (0x13138d0) 消费做 SuperTile DP 合并
```

**关键概念(已从 RTTI + 日志证实):**

| 概念 | 证据 |
|------|------|
| `DPGroupGraph` | RTTI typeinfo `12DPGroupGraph` @ 0x469f69b |
| `DPGroupNode` | RTTI `11DPGroupNode` @ 0x469fe0f |
| `DPOpGraph` / `DPOpNode` | RTTI `9DPOpGraph` / `8DPOpNode`(在方法签名里) |
| `concat-tree group creation` | 日志行 353 |
| `op_id based groups` (fallback) | 日志行 353/631 |
| `branch linear order` | 日志行 1066/2254/2293/2307 |
| `TCM break up` | 日志行 1225 |
| `DDR break up` | 日志行 1785 |
| `isolated_s_dim_breakup` | RTTI 方法名 @ 0x469fd44 |
| `split_merge_ops_into_predecessor_groups` | RTTI 方法名 @ 0x469fabe |
| `BaseDPMLSelectorFeatureSuperSet` | RTTI @ 0x1198760 |
| `seq_sf_group_breakup_cfg` / `_merge_cfg` | 配置字符串(多处引用) |

---

## 5. 复现脚本(已验证可用)

所有脚本在 `C:\Users\RQILIN\AppData\Local\Temp\opencode\`:

| 脚本 | 作用 |
|------|------|
| `disasm_create_supertiles.py` | 反汇编 create_supertiles,提取 call 目标 → `create_supertiles.disasm` |
| `find_dp_strings.py` | 全局字符串扫描 dp_group/DPGroupGraph |
| `dp_symbols.py` | dynsym + mangled 方法名提取 → `dp_symbols.txt` |
| `supertile_callgraph.py` | 反汇编内部函数,提取字符串引用 → `supertile_callgraph.txt` |
| `dpgraph_funcs.py` | 扫描 .text 引用 dp_group_graph.cc 的指令 → `dpgraph_funcs.txt` |
| `dpgraph_strings_full.py` | 全 .text 反汇编,提取区间字符串引用 → `dpgraph_strings_full.txt` |
| `dp_split_methods.py` | 反汇编具名 lambda invoker(确认是 hash 查找) |
| `dpgraph_funcs_list.py` | 按函数边界切分区间,列日志 → `dpgraph_funcs_list.txt` |

运行:
```powershell
$py='C:\Users\RQILIN\AppData\Local\Programs\Python311\python.exe'
& $py 'C:\Users\RQILIN\AppData\Local\Temp\opencode\<script>.py'
```

---

## 6. 可信度与边界

### ✅ 已指令级确认
- `create_supertiles` @ `0x13138d0` size 5244(dynsym 导出)
- `make_one_supertile` @ `0x1314d50` size 591(dynsym 导出)
- `DPGroupGraph` / `DPGroupNode` / `DPOpGraph` / `DPOpNode` 类的 RTTI 存在(字符串扫描)
- 两个真实方法名:`split_merge_ops_into_predecessor_groups` / `isolated_s_dim_breakup`(RTTI)
- dp_group_graph.cc 代码区间 0x115000–0x1167xxx(40 处字符串引用)
- 14 个函数的日志行号(232~2649)与切分语义

### ⚠️ 推断(未逐指令验证)
- 各函数到行号的精确映射(函数边界检测靠 prologue 启发式,可能误差)
- `concat-tree group creation` vs `op_id based groups` 的具体判定条件
- `branch linear order` 三种启发式的选择逻辑
- `popular groups` 迁移的优先级公式
- `TCM/DDR break up` 的具体切分点选择

### ❌ 已证伪
- **`build_dp_group_graph` 这个函数名不存在**于二进制(符号表/字符串/RTTI 全无匹配)
- `0x11752d0` / `0x117bbf0` 不是切分主体,是 lambda hash 查找辅助
- `create_supertiles` 不构建 DPGroupGraph,只消费它

---

## 7. 下一步如需更深入

要逐指令验证某函数(如 0x11564d0 主体,19856 字节):

1. 用 `dpgraph_funcs_list.py` 已圈定函数边界
2. 对该函数单独反汇编(改 `dp_split_methods.py` 的目标地址)
3. 关键模式:
   - `cmp [r15+0x6338]` → VTCM 预算字段(对照 ALGORITHM_PRINCIPLES_VERIFIED.md)
   - `call` 到 0x6ed270(x9)/0x6f3480(x8)/0x6ef2b0(x5) → 这些是 std::vector/map 操作辅助
   - `lea rcx, [rip+...]` 引用 `seq_sf_group_breakup_cfg` → 读取切分配置
4. 交叉引用:找谁调用了这些 dp_group_graph.cc 函数 —— 用 xref 扫描 `call 0x11564d0` 等

需我继续逐指令验证 0x11564d0 主体函数,告诉我即可。
