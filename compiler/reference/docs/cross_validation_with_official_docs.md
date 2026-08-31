# 与官方逆向文档对照验证

对照两份官方逆向文档验证我们的分析:
  A) libHtpPrepareDoc/ALGORITHM_PRINCIPLES_VERIFIED.md (指令级验证, 最高可信)
  B) graph_compiler.docx (图编译流程概览)

## 1. 序列化格式 — 完全一致 ✅

### 我们的分析
- contextBlob = 系统信息头 + HTP contextBlob
- 系统头: 0x18=offset(LE), 0x20=size(LE), 计数表(BE)
- runlist: FA0000FA 分隔 → 19条op记录 → BEEFF00D 结束
- 记录标记 0x1303EE{XX}, type = (extra_info[0x18]>>6&0xf)<<24

### 文档 A (ALGORITHM_PRINCIPLES_VERIFIED.md §7) 证实
- do_serialize @0xf64fa0: 12阶段序列化流程
- Magic: 0xfa0000fa (段分隔) ✅, 0xbeeff00d (结束) ✅
- serialize_op @0x12ec820: op_format_code = 0x1303ee71 ✅ (我们找到的 0x12ec688 同一函数)
- 序列化不调用 checksum_bytes (无CRC32) ✅
- vtable 驱动流式写, 非固定格式 ✅

### 文档 B (graph_compiler.docx) 证实
- "Scheduled graph -> context.bin" 阶段:
  - 物理内存规划 (VTCM liveness 分析) ✅
  - 内核实例化 (51个DSP内核填参数) ✅
  - DMA 双缓冲 (搬运与计算重叠) ✅
  - 事件/时钟同步 ✅
  - blob 编码: "指令表 + 数据偏移表 + kernel名表×51 + string表×175" ✅
  - context 打包: "元数据头 0x1000 + 权重区 + blob + 校验" ✅
    → 与我们的 [系统信息头0x1000] + [contextBlob] 完全吻合!

## 2. op 记录格式 — 完全一致 ✅

### 我们的分析
- [0-3] 0x1303EE{XX} 标记
- [4] counter = tensor id
- [7] type: 0x10=compute, 0x20=memory, 0x30=sync, 0x40=DMA
- [8-11] F2 = DMA tag / op序号
- [12-15] block_ref = VTCM块引用

### 文档 A §7 证实
- serialize_op @0x12ec820:
  - op_format_code = 0x1303ee71 ✅
  - op_data (变长, vtable调用) ✅
  - input_tensors (spcl_add_in_tensor) ✅
  - output_tensors (spcl_add_out_tensor) ✅
- serialize_single_tensor_pointer:
  - pool_id (2B) + offset (4B) ✅ → 与我们的 block_ref 编码一致

## 3. 调度算法 — 修正我们的理解 ⚠️→✅

### 我们的分析 (基于 schedule_for_alloc 反汇编)
- schedule_for_alloc 不生成 SET/WAIT, 只做递归 VTCM 分配 ✅
- 调用 stupid_fast_topo_sort 做拓扑排序 ✅

### 文档 A §4 证实并修正
- schedule_for_alloc @0x1302900 (我们找到 0x1302710, 同一函数不同版本) ✅
- stupid_fast_topo_sort @0x1301c70: DFS 位图拓扑排序 ✅
  - 显式栈 (非递归) ✅
  - visited 位图 (O(1)查重) ✅
  - 后序遍历输出 ✅
- **PCG32 随机 tie-break** ⚠️ 我们没发现这个!
  - state += 0x5851f42d4c957f2d (PCG multiplier)
  - state += 0x14057b7ef767814f (PCG increment)
  - 拓扑序确定后用 PCG 随机打乱同层节点
  → 修正: 我们的"19条记录按拓扑序"是对的, 但同层节点顺序经PCG随机化

### 文档 A 证伪 GRAPH_COMPILER_PRINCIPLES.md 的说法
- "CBS = Welsh-Powell 图着色" ❌ 实际是 DFS拓扑序 + PCG随机tie-break
- "双种子1.3/1.4调度" ❌ 实际是0.05/0.1/0.15 backend默认参数
- → 我们没有采用这些错误说法, 正确

## 4. VTCM 内存规划 — 一致 ✅, 补充细节

### 我们的分析
- VTCM 4MB 上限, 0.75 系数
- spill_bytes=0 (模型小, 全在VTCM)
- block_ref 引用 VTCM 块

### 文档 A §3 证实并补充
- VTCM 4MB 上限 (0x3fffff/0x400000) ✅
- 0.75 系数 (实际可用 = budget × 0.75) ✅
- 2KB 对齐 (0x7ff) ✅
- tcm_migration @0x13219c0:
  - SPILL_TO_DDR 标志 = 0x40 (tensor flags) ✅
  - max-heap sift_down 按优先级排序 ✅
  - 优先级字段 = tensor+0x98 (预计算) ✅
  - 优先级 ∝ size/(access_freq×lifetime) ✅
- 我们没发现: 优先级公式在20+处tensor构造代码中分散写入, 无单一公式

## 5. SET/WAIT — 一致 ✅, 补充证据

### 我们的分析
- make_dma_checkpoint_op(tensor_id, param, bool is_set) @0xd958d0
- SET vtable@0x5ec2488, WAIT vtable@0x5ec2568
- 共享 serialize_internal @0xd969a0 → 字节不可区分
- 4条DMA都是SET, compute隐式WAIT

### 文档 A — 未直接提到 DmaCheckpoint
但文档A的序列化分析确认:
- vtable驱动序列化 → 不同op类型有不同vtable → SET/WAIT不同类 ✅
- 序列化不区分类型名 → 字节不可区分 ✅ (与我们的发现一致)

### 文档 B 证实 DMA 双缓冲
- "DMA 双缓冲: 搬 tile N+1 权重的同时, HMX 在算 tile N"
- → 与我们的"SET发起DMA, compute隐式WAIT用数据"模型一致 ✅

## 6. 优化 pass — 补充我们的缺失

### 文档 A §2 (我们未分析的部分)
- 8阶段pass调度, 按节点数预算:
  Phase1(≤3000): run_plugin_rewrites
  Phase3(≤11900): GraphOptContext::attempt (pattern matching)
  Phase5(≤21101): const_prop_and_cse
  Phase7(≤24999): DCE→order→CSE fixpoint
  Phase8(∞): 最终清理
- Fibonacci hash 0x192e2101 用于 op extra_info表 (非调度)
- → simple_linear仅10参数, 走Phase8无条件清理

## 7. 校验和 — 修正 REQNN

### 文档 A §1 (我们未分析)
- checksum_bytes = 64位 Galois LFSR (poly 0x1b), 不是CRC32
- 每字节走两步LFSR再异或
- 用于图校验和 (op ID序列) 和 tensor指纹
- → REQNN 的 bin_format_analysis.md 提到"BEEFF00D等是内存分隔符" 正确

## 8. 验证结论汇总

| 我们的分析结论 | 文档验证 | 状态 |
|---|---|---|
| 两种bin格式 (TAR vs context binary) | 文档B: "元数据头0x1000+blob" | ✅ 一致 |
| 系统头 LE+BE 混用 | 文档A: 序列化用tag(0xef4d等), 无CRC | ✅ 一致 |
| contextBlob 内部大端 | 文档A: vtable流式写 | ✅ 一致 |
| 0x1303EE 标记硬编码 | 文档A §7: op_format_code=0x1303ee71 | ✅ 完全一致 |
| FA0000FA/BEEFF00D 分隔符 | 文档A §7: magic标记 | ✅ 完全一致 |
| counter = tensor id | 文档A: 间接确认 (op ID用于checksum) | ✅ 一致 |
| type 字段编码 (0x10/20/30/40) | 文档A: vtable驱动(不同类型不同vtable) | ✅ 一致 |
| block_ref = pool_id+offset | 文档A: serialize_single_tensor_pointer | ✅ 完全一致 |
| schedule_for_alloc 不生成SET/WAIT | 文档A §4: 递归分配, 调用topo_sort | ✅ 一致 |
| SET/WAIT 字节不可区分 | 文档A: vtable驱动但共享serialize | ✅ 一致 |
| VTCM 4MB上限 + 0.75系数 | 文档A §3: 0x400000 + 0.75 | ✅ 完全一致 |
| 19步按拓扑序 | 文档A: stupid_fast_topo_sort DFS | ✅ 一致(+PCG随机) |
| QNN算子→HTP内核展开 | 文档B: "算子→内核"阶段 | ✅ 一致 |
| DMA双缓冲模型 | 文档B: "搬tile N+1, 算tile N" | ✅ 一致 |

### 我们分析中的修正/补充
1. **PCG32 随机 tie-break**: 我们没发现, 文档A证实拓扑序后用PCG随机化同层节点
2. **8阶段pass调度**: 我们没分析, 文档A证实按节点数预算分档
3. **LFSR校验和**: 我们没分析, 文档A证实是Galois LFSR(poly 0x1b)非CRC32
4. **tcm_migration优先级**: 文档A补充了 max-heap + tensor+0x98 字段
5. **CBS非图着色**: 文档A证伪了 GRAPH_COMPILER_PRINCIPLES.md 的 Welsh-Powell 说法
