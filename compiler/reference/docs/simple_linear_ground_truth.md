# 真实 context binary 样本字节级分析 (simple_linear, v2.48.0.260626)

> 真实 SDK 编译 simple_linear 模型的 context binary，作为格式编码器的 ground truth。
> 源: dev133 上 `qnn-context-binary-generator --model libsimple_linear.so --backend libQnnHtp.so` 产出。

## 模型结构 (simple_linear.cpp)

```
input[1,4,3] float32
  → Transpose(perm=[0,2,1]) → input_ncf[1,3,4]
  → Reshape → MatMul_0_pre_reshape[3,4]
  → FullyConnected(W[2,4], b[2]) → output_fc[3,2]   (MatMul + bias)
  → Reshape → output_fc_ncf[1,3,2]
  → Transpose(perm=[0,2,1]) → output[1,2,3]
```

## 文件总布局 (45832 bytes = 0xB308)

```
[段1] 系统信息头 [0x0000, 0x1000)  4096 B   LE+BE 混用
  [0x18] contextBlob offset = 0x1000    (LE u64)
  [0x20] contextBlob size   = 0xA308    (LE u64, 41736 bytes)
  [0x130] ioTensorSize      = 0x00400000 (BE, 256KB? 实际是计数/尺寸表)
  [0x177] constSize         = 0x00200000 (BE)
  [0x1D5] dspArch           = 0x0000    (BE, 文档说 68=V68; 此样本为 0)
  [0x1E8] graph name        = "simple_linear"
  [0x334] buildId           = "v2.48.0.260626120635"
  [0x3C0,0x1000) 零填充 (4KB 对齐)

[段2] contextBlob [0x1000, 0xB308)  41736 B  内部大端
  ├─ 头部/记录描述符 [0x1000, ~0x1100)
  │   [0x1098] "ROOT"
  │   [0x10DC] "PICKLE/256/simple_linear"
  ├─ ioTensor 段     [~0x2000, ~0x6000)  张量描述+图配置
  ├─ 内核名表        [0x73D0, 0x7490)  11 个内核名 (见下)
  ├─ opData 段       [0x7534, ~0x7B00)
  │   [0x7534] FA0000FA 分隔符
  │   [0x75D0] 0x1303EE71 (op 记录起始标记)
  │   → 19 条 op 记录
  │   → BEEFF00D 结束标记 (在 opData 末尾)
  ├─ const 段        [~0x9000, ~0xB000)  W/b 权重 + extent 表
  ├─ ddrTensor 段    [~0xA000, ~0xA400)
  └─ trailer         [~0xB000, 0xB308)  图名/op 名回指
      [0xB0D0] "simple_linear"
      [0xB118] "MatMul_0_post_reshape_transpose"
      [0xB148] "MatMul_0_post_reshape"
      [0xB170] "MatMul_0"
      [0xB18C] "MatMul_0_pre_reshape"
      [0xB1BC] "input_ncf"
      [0xB22C] "output"
      [0xB29C] "input"
```

## 内核名表 (opData 引用的目标, 0x73D0-0x7490)

11 个内核，和文档分析完全一致：

| 偏移 | 内核名 | 推断用途 | QNN op 来源 |
|------|--------|---------|------------|
| 0x73D0 | `Shape` | 形状推断 (编译期) | (图节点) |
| 0x73D6 | `*InputSlice@Ff.s4*6.` | 输入切片: f32, HVX向量宽4, 6路并行 | Transpose(input_ncf) |
| 0x73EB | `@DmaCheckpointSet` | DMA SET: 发起DMA并记录tag | (权重加载) |
| 0x73FD | `@DmaCheckpointWait` | DMA WAIT: 等待DMA完成 | (隐式) |
| 0x7410 | `Const` | 常量加载 (const pool→VTCM) | (权重) |
| 0x7416 | `Transpose_impl@Ff*2.fi.t` | 转置: f32, 2维, flat+isolated | Transpose(input_ncf) |
| 0x742F | `flat_from_vtcm@ff.Ff.` | VTCM内平坦化: f32→f32布局转换 | Reshape |
| 0x7445 | `MatMul_bias@ff*4` | 矩阵乘+偏置: f32, 向量宽4 | FullyConnected |
| 0x7456 | `Transpose_impl@ff*2.fi.t` | 转置(输出侧) | Transpose(output) |
| 0x746F | `*OutputSlice@ff.s4*3.` | 输出切片: f32, 3路并行 | Transpose(output) |
| 0x7485 | `@SyncOp` | 同步屏障 | (同步) |

### tiling 后缀语义 (从 ELF 字符串归纳)
- **数据类型**: Ff/ff=float32, Fi/fi=f32+isolated, fB/FB=f32→u8, fe/Fe=f32→bf16
- **并行度**: *2=2D转置, *3=3路并行, *4=向量宽4, *6=6路并行, s4=HVX向量宽4
- **布局**: .t=transpose, .fi=flat+isolated, .Ff=布局转换

## op 记录格式 (opData 段, 反汇编 serialize_op @0x12ec630 证实)

```
[0-3]   0x1303EE{XX}  标记 (XX=记录ID, 硬编码 0x1303ee71 起始递增)
[4]     counter       = tensor id (字节可证: 匹配 net.json id 1-10)
[7]     type          = (extra_info[0x18]>>6&0xf) << 24
                       0x10=compute, 0x20=memory, 0x30=sync, 0x40=DMA
[8-11]  F2            = DMA tag / op 序号
[12-15] block_ref     = idx | (0x03<<24), 低16位=VTCM块索引
[16+]   tensor_ids[]  变长, 输入/输出张量 id 列表
```

### counter = tensor id 映射 (字节可证)
| counter | tensor | net.json id |
|---|---|---|
| 0 | graph_node | (特殊值, 图IO节点) |
| 1 | input [1,4,3] | 1 |
| 2 | perm_in [0,2,1] | 2 (Transpose perm) |
| 3 | input_ncf [1,3,4] | 3 (Transpose 输出) |
| 4 | pre_reshape [3,4] | 4 (Reshape 输出) |
| 5 | W [2,4] | 5 (权重) |
| 6 | b [2] | 6 (偏置) |
| 7 | output_fc [3,2] | 7 (FC 输出) |
| 8 | output_ncf [1,3,2] | 8 (Reshape 输出) |
| 9 | perm_out [0,2,1] | 9 (输出 Transpose perm) |
| 10 | output [1,2,3] | 10 (最终输出) |

## 19 步执行表 (从字节+反汇编推断)

| # | type | 内核 | 操作 |
|---|---|---|---|
| 0 | compute | Input node setup | 图输入节点初始化 |
| 1 | compute | Input node setup | 图输入节点初始化 |
| 2 | compute | `*InputSlice@Ff.s4*6.` | input[1,4,3] 切片 6路并行 |
| 3 | memory | perm搬运 | perm_in[0,2,1] 加载 |
| 4 | memory | input_ncf布局转换 | 转置结果[1,3,4] 布局转换 |
| 5 | compute | Reshape/flat_from_vtcm | pre_reshape 2D化, 依赖W DMA |
| 6 | compute | FC输入准备 | pre_reshape续 |
| 7 | DMA | `@DmaCheckpointSet` | W权重DMA(tag=0x11) |
| 8 | DMA | `@DmaCheckpointSet` | b偏置DMA(tag=0x11) |
| 9 | compute | `MatMul_bias@ff*4` (x·W) | FC矩阵乘 |
| 10 | compute | `MatMul_bias@ff*4` (+b) | FC偏置加 |
| 11 | DMA | `@DmaCheckpointSet` | output_fc搬出(tag=0x16) |
| 12 | compute | `flat_from_vtcm@ff.Ff.` | output_fc平坦化 |
| 13 | DMA | `@DmaCheckpointSet` | output_ncf搬出(tag=0x1A) |
| 14 | compute | Output node setup | 图输出节点初始化 |
| 15 | sync | `@SyncOp` | 同步屏障 |
| 16 | memory | perm搬运 | perm_out 加载 |
| 17 | compute | `Transpose_impl@ff*2.fi.t` | 最终转置 |
| 18 | compute | `*OutputSlice@ff.s4*3.` | 输出切片→output[1,2,3] |

## 对比: 我们的工程 vs 真实 SDK

| 对比项 | 真实 SDK (simple_linear_context.bin) | 我们的工程 |
|--------|-------------------------------------|-----------|
| 大小 | 45832 字节 | 2088 字节 |
| 系统头 | 4096B (ioTensorSize/constSize/dspArch/graph名/buildId) | 无 |
| contextBlob | 大端, 6 段划分 | 全 LE, 扁平 |
| op 记录 | 0x1303EE71 + tensor_id + type + block_ref + tensor_ids | 自定义 TAG_OP_RECORD |
| 内核名表 | 11 个真实 HTP 内核名 (MatMul_bias@ff*4 等) | 无, op 名直存 |
| DMA/同步 | @DmaCheckpointSet/@DmaCheckpointWait/@SyncOp + tag | 无 |
| 执行计划 | 19 步调度后的执行表 | 仅图结构 |
| 能上板 | ✅ | ❌ |

## 下一步 (阶段 A: 格式编码器)

需要实现:
1. 系统信息头编码器 (4096B, LE+BE 混用)
2. contextBlob 大端编码 (6 段)
3. op 记录格式 (0x1303EE71 + tensor_id + type + block_ref + tensor_ids)
4. 内核名表编码 (引用 Skel .so 里的内核名)
5. const 段 (权重 extent 表 + 数据块)
6. trailer (图名/op 名回指)

关键依赖:
- 阶段 B (内核选择器): 决定每个 op 用哪个内核名
- 阶段 C (调度器): 决定 19 步执行表的顺序和 DMA/SyncOp 插入
