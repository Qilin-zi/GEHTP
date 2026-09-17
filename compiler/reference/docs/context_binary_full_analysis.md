# QNN HTP Context Binary 完整逆向分析报告

> 基于 `simple_linear` 模型 (MatMul+Add, float32, 10 参数) 的端到端逆向工程
> 工具链: QAIRT 2.48.0.260626 + libQnnHtp.so (x86_64-linux-clang)
> 日期: 2026-08-06

## 1. 模型与管线

### 原始模型 (`make_model.py`)
```
input [1,3,4] --MatMul(W[4,2])--> tmp --Add(b[2])--> output [1,3,2]
W = [[1,2],[3,4],[5,6],[7,8]]  b = [0.1, 0.2]
```
数学上: `y = x·W + b`，一个线性层。

### 编译管线
```
ONNX ─qnn-onnx-converter─> .cpp + .bin(TAR) ─model-lib-gen─> .so/.dll
                                                          └─qnn-context-binary-generator─> context binary (.bin)
                                                                                              └─板端 libQnnHtp 加载执行
```

## 2. 两种 .bin 对比

| | converter bin | context binary |
|---|---|---|
| 文件 | `simple_linear.bin` | `simple_linear_context.bin` |
| 大小 | 10240 B | 45840 B |
| 格式 | POSIX ustar TAR 归档 | 系统信息头 + HTP contextBlob |
| 字节序 | 数据 LE | 系统头 LE+BE 混用, contextBlob BE |
| 内容 | 静态张量 .raw (W.raw 32B, b.raw 8B) | 编译后的整图 (6 个 section) |
| 用途 | 链接期权重容器 (编进 .so) | 运行期序列化图 (板端直接执行) |
| 权重位置 | W@0x200, b@0x600 | W@0x9000, b@0x9100 (常量池内) |

## 3. Context Binary 布局

```
文件 45840 B (0xB310)
┌────────────────────────────────────────────────────────────┐
│ [段1] 系统信息头 [0x0000, 0x1000)  4096 B                 │
│   [0x18] contextBlob 偏移 = 0x1000  (LE u64)              │
│   [0x20] contextBlob 大小 = 0xA310  (LE u64)              │
│   [0x130] ioTensorSize = 0x4000  (BE)                     │
│   [0x177] constSize = 0x2000    (BE)                      │
│   [0x1D5] dspArch = 68 (V68)   (BE)                      │
│   [0x1E8] "simple_linear" (图名)                          │
│   [0x334] "v2.48.0.260626120635" (buildId)                │
│   [0x3C0,0x1000) 零填充 (4KB 对齐)                        │
├────────────────────────────────────────────────────────────┤
│ [段2] contextBlob [0x1000, 0xB310)  41744 B  内部大端     │
│   ┌─ 头部/记录描述符 [0x1000,0x1100)  "ROOT"/"PICKLE"标签 │
│   ├─ ioTensor 段     [0x2000,0x6000)  张量描述+图配置    │
│   ├─ opData 段       [0x75E0,0x7B00)  op记录+内核名表    │
│   ├─ const 段        [0x9000,0xB000)  W/b权重+extent表   │
│   ├─ ddrTensor 段    [0xA000,0xA400)  DDR张量元数据      │
│   └─ trailer         [0xB000,0xB310)  图名/op名回指     │
└────────────────────────────────────────────────────────┘
```

**字节序规律**: host 用的小端字段 (offset/size) + DSP 用的大端字段 (计数/尺寸表) 混用 — host 包装层与 DSP 内核的分界。

## 4. op 记录格式 (opData 段)

runlist 在 `[0x77A0, 0x7B00)`: `FA0000FA` 分隔符 → 19 条记录 → `BEEFF00D` 结束标记。

### 记录结构 (反汇编 serialize_op @0x12ec630 证实)
```
[0-3]   0x1303EE{XX}  标记 (XX=记录ID, 硬编码 0x1303ee71 起始递增)
[4]     counter       = tensor id (字节可证: 匹配 net.json id 1-10)
[7]     type          = (extra_info[0x18]>>6&0xf) << 24
                       0x10=compute, 0x20=memory, 0x30=sync, 0x40=DMA
[8-11]  F2            = DMA tag / op 序号 (make_dma_checkpoint_op 的 param)
[12-15] block_ref     = idx | (0x03<<24), 低16位=VTCM块索引
[16+]   tensor_ids[]  变长, 输入/输出张量 id 列表
```

### counter = tensor id 映射 (字节可证)
| counter | tensor | 来源 |
|---|---|---|
| 0 | graph_node | 特殊值 (图IO节点) |
| 1 | input [1,4,3] | net.json id=1 |
| 2 | perm_in [0,2,1] | id=2, Transpose perm |
| 3 | input_ncf [1,3,4] | id=3, Transpose 输出 |
| 4 | pre_reshape [3,4] | id=4, Reshape 输出 |
| 5 | W [2,4] | id=5, 权重 |
| 6 | b [2] | id=6, 偏置 |
| 7 | output_fc [3,2] | id=7, FC 输出 |
| 8 | output_ncf [1,3,2] | id=8, Reshape 输出 |
| 9 | perm_out [0,2,1] | id=9, 输出 Transpose perm |
| 10 | output [1,2,3] | id=10, 最终输出 |

## 5. HTP 内核名表 (opData @0x7638)

11 个内核 (含 tiling 后缀):
```
Shape                    形状推断 (编译期)
*InputSlice@Ff.s4*6.     输入切片: f32, HVX向量宽4, 6路并行
@DmaCheckpointSet        DMA SET: 发起DMA并记录tag
@DmaCheckpointWait       DMA WAIT: 等待DMA完成
Const                    常量加载 (const pool→VTCM)
Transpose_impl@Ff*2.fi.t 转置: f32, 2维, flat+isolated
flat_from_vtcm@ff.Ff.    VTCM内平坦化: f32→f32布局转换
MatMul_bias@ff*4         矩阵乘+偏置: f32, 向量宽4
Transpose_impl@ff*2.fi.t 转置(输出侧)
*OutputSlice@ff.s4*3.     输出切片: f32, 3路并行
@SyncOp                  同步屏障
```

### tiling 后缀语义 (从 ELF 字符串归纳)
- **数据类型**: Ff/ff=float32, Fi/fi=f32+isolated, fB/FB=f32→u8, fe/Fe=f32→bf16
- **并行度**: *2=2D转置, *3=3路并行, *4=向量宽4, *6=6路并行, s4=HVX向量宽4
- **布局**: .t=transpose, .fi=flat+isolated, .Ff=布局转换

## 6. SET/WAIT 分析 (反汇编 + tag 依赖)

### 反汇编证据
- `make_dma_checkpoint_op(tensor_id, param, bool is_set)` @0xd958d0
- is_set=true → SET vtable@0x5ec2488 (class DmaCheckpointSet)
- is_set=false → WAIT vtable@0x5ec2568 (class DmaCheckpointWait)
- **两者 serialize_internal 相同** (@0xd969a0) → context binary 字节不可区分
- 文档: SET="Records DMA tag in table", WAIT="Waits for DMA to complete"

### tag 依赖 (字节可证)
4 条 DMA 记录的 tag 与 compute 记录 F2 匹配:
- `rec5`(compute F2=0x11) ↔ `rec7,8`(DMA tag=0x11) — W/b 权重域
- `rec12`(compute F2=0x1A) ↔ `rec13`(DMA tag=0x1A) — output_ncf 域

### 推断: HTP 异步 DMA 模型
SET 发起 DMA(tag) → 后续 compute 隐式 WAIT 同 tag → 用数据
→ **4 条 DMA 都是 SET**, WAIT 由 compute 执行时隐式完成

## 7. 19 步执行表 (最终版, 带可信度)

| # | type | 推断内核 | 可信 | 操作 |
|---|---|---|---|---|
| 0 | compute | Input node setup | 中 | 图输入节点初始化(hash) |
| 1 | compute | Input node setup | 中 | 图输入节点初始化(hash) |
| 2 | compute | `*InputSlice@Ff.s4*6.` | 高 | input[1,4,3] 切片 6路并行 |
| 3 | memory | perm搬运 | 高 | perm_in[0,2,1] 加载 |
| 4 | memory | input_ncf布局转换 | 高 | 转置结果[1,3,4] 布局转换 |
| 5 | compute | Reshape/flat_from_vtcm | 中 | pre_reshape 2D化, 依赖W DMA |
| 6 | compute | FC输入准备 | 中 | pre_reshape续 |
| 7 | DMA | DmaCheckpoint **SET** | 高 | W权重DMA(tag=0x11) |
| 8 | DMA | DmaCheckpoint **SET** | 高 | b偏置DMA(tag=0x11) |
| 9 | compute | `MatMul_bias@ff*4` (x·W) | 高 | FC矩阵乘 |
| 10 | compute | `MatMul_bias@ff*4` (+b) | 高 | FC偏置加 |
| 11 | DMA | DmaCheckpoint SET | 中 | output_fc搬出(tag=0x16) |
| 12 | compute | `flat_from_vtcm@ff.Ff.` | 高 | output_fc平坦化 |
| 13 | DMA | DmaCheckpoint **SET** | 高 | output_ncf搬出(tag=0x1A) |
| 14 | compute | Output node setup | 中 | 图输出节点(hash) |
| 15 | sync | `@SyncOp` | 高 | 同步屏障 |
| 16 | memory | perm搬运 | 高 | perm_out 加载 |
| 17 | compute | `Transpose_impl@ff*2.fi.t` | 高 | 最终转置 |
| 18 | compute | `*OutputSlice@ff.s4*3.` | 高 | 输出切片→output[1,2,3] |

**可信度**: 高 13/19, 中 6/19, 低 0/19

## 8. QNN 算子 → HTP 内核展开

原始 5 个 QNN 算子被 HTP 编译器展开为 19 步:

| QNN 算子 | → HTP 内核 | 步骤 |
|---|---|---|
| (图节点) | Input node setup ×2 | 0,1 |
| input_ncf (Transpose) | InputSlice + perm搬运 + 布局转换 | 2,3,4 |
| MatMul_0_pre_reshape (Reshape) | Reshape + FC输入准备 | 5,6 |
| (权重加载) | DmaCheckpoint SET ×2 | 7,8 |
| MatMul_0 (FullyConnected) | MatMul_bias ×2 (x·W, +b) | 9,10 |
| (结果搬出) | DmaCheckpoint SET | 11 |
| MatMul_0_post_reshape (Reshape) | flat_from_vtcm + DmaCheckpoint SET | 12,13 |
| (图节点) | Output node setup | 14 |
| (同步) | SyncOp | 15 |
| MatMul_0_post_reshape_transpose (Transpose) | perm搬运 + Transpose_impl + OutputSlice | 16,17,18 |

**关键洞察**: 一个 `y=x·W+b` 线性层在 HTP 上变成 19 步 — 2步节点初始化、3步输入切片+转置、2步Reshape、2步权重DMA、2步MatMul+bias、3步结果搬出+平坦化、1步同步、3步输出转置+切片。额外开销全在布局转换(NCHW↔NHWC)和DMA同步,这是HTP DSP向量架构的固有代价。

## 9. 反汇编证据索引

| 发现 | 函数 | 地址 | 证据类型 |
|---|---|---|---|
| 0x1303EE 标记硬编码 | serialize_op | 0x12ec688 | 反汇编 |
| type 字段编码 | serialize_op | 0x12ec793 | 反汇编 (shr 6, and 0xf) |
| block_ref 编码 | serialize_op | 0x12ec8b7 | 反汇编 |
| SET/WAIT 存在 | make_dma_checkpoint_op | 0xd958d0 | 反汇编 (bool is_set) |
| SET vtable | DmaCheckpointSet | 0x5ec2488 | 反汇编 |
| WAIT vtable | DmaCheckpointWait | 0x5ec2568 | 反汇编 |
| SET/WAIT 字节不可区分 | serialize_internal | 0xd969a0 | 反汇编 (共享) |
| SET 语义 | .rodata 字符串 | 0x039B26D0 | 文档 |
| WAIT 语义 | .rodata 字符串 | 0x039B2690 | 文档 |
| SyncOp 存在 | make_SyncOp | 0xdac250 | 符号表 |
| schedule_for_alloc 职责 | schedule_for_alloc | 0x1302710 | 反汇编 (递归分配,不生成SET/WAIT) |
| counter = tensor id | - | - | 字节验证 (匹配 net.json) |
| 内核名 tiling 后缀 | .rodata | 0x039Cxxxx | ELF 字符串 |

## 10. 仍待确认 (需进一步反汇编)

1. `deserialize_graph` (0xcf28c0, ~16KB) 内部 SET/WAIT 决策逻辑
2. rec11 (tag=0x16) 是 SET 还是 WAIT (无 compute 匹配此 tag)
3. rec0,1,14 (图节点 setup) 的具体内核名
4. rec5,6 (Reshape 细分) 的精确 tiling 后缀

## 相关文件
- `re_converter_bin.py` — converter bin (TAR) RE
- `re_context_bin.py` / `re_context_bin_full.py` — context binary RE
- `re_op_final_v3.py` — 19 步执行表最终版
- `verify_17steps.py` — 17 步表验证 (证明原表有误)
- `REQNN/reference/docs/schedule_analysis.md` — schedule_for_alloc 反汇编分析
- `REQNN/reference/docs/schedule_for_alloc.disasm` — 全量反汇编 (2608 行)
- `REQNN/reference/docs/deserialize_runlist.disasm` — deserialize_runlist 反汇编
- `REQNN/reference/docs/bin_format_analysis.md` — bin 格式分析
