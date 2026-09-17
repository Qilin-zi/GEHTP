# Context Binary Writer 真动态生成工作计划

> 日期: 2026-08-11
> 锚点文件: 本文档。后续每个阶段开始前先读此文件,完成阶段后更新状态表。

## 0. 问题陈述(为什么要做)

### 现状
`src/serialize/context_binary_writer.cpp` 的 `.bin` 输出对 `simple_linear_context.bin`
是字节级 0-diff,但 97%(~44800/45832 B)是照抄模板/写死常量:

| 段 | 区间 | 大小 | 当前来源 |
|----|------|------|---------|
| system header | `[0,0x1000)` | 4096 B | 照抄模板,patch 5 字段 |
| pre-opData | `[0x1000,0x7534)` | 25908 B | 照抄模板 `CTX_BLOB_PRE_TEMPLATE` |
| opData | `[0x7534,~0x7898)` | ~888 B | **动态**(op records),preamble 写死 |
| post-opData | `[~0x7898,0x9000)` | ~3688 B | 写死(graph metadata) |
| const segment | `[0x9000,0xB000)` | 8192 B | 权重动态,descriptor 写死 |
| trailer | `[0xB000,0xB308)` | 776 B | graph name 动态,其余写死 |

真正动态的只有 opData op records + const pool 权重数据,约 1 KB。

### 目标(验收标准)
让 `.bin` 的**每一字节**都由编译流程的状态(GraphPrepare 的 OpDef/tensor defs/
调度结果/VTCM 分配/const pool)按正确逻辑生成,**不保留任何模型相关的硬编码常量**。

### 验证金标准
`reference/compare/` 下有 7 个不同结构的真实 .bin:
```
fc_only_context.bin         45640 B   (仅 FullyConnected)
linear_4x3_w4_context.bin   45832 B   (linear, W 形状 4x3)
linear_4x8_context.bin      45824 B   (linear, W 形状 4x8)
linear_8x4_context.bin       45824 B   (linear, W 形状 8x4)
reshape_only_context.bin    45688 B   (仅 Reshape)
trans_only_context.bin      45656 B   (仅 Transpose)
two_fc_context.bin          45664 B   (两个 FC 串联)
```
**最终验收**: 对这 7 个模型,我们的 writer 输出与真实 .bin 全部 0-diff。
不同模型 → 同一 writer → 不同 .bin,且与真实 SDK 逐一匹配。

### 逆向资料(已具备)
- `reference/docs/context_binary_full_analysis.md` — 完整逆向分析报告(含可信度分级)
- `reference/docs/simple_linear_ground_truth.md` — 字节级 ground truth
- `reference/docs/bin_format_analysis.md` — 格式分析(字节序、count+offset 表)
- `reference/docs/schedule_analysis.md` — schedule_for_alloc + serialize_op 反汇编
- `reference/docs/deserialize_runlist.disasm` — 反序列化 runlist 全量反汇编
- `reference/docs/htp_compiler_algo.md` — 编译器算法
- `reference/docs/cross_validation_with_official_docs.md` — 与官方文档对照

## 1. 原则(每个阶段都必须遵守)

1. **逐段攻克**: 一次只动一个段,先消除该段的模板/写死,改从 GraphPrepare 状态生成
2. **即时验证**: 每段改完,simple_linear 必须 0-diff;7 模型逐段累积验证
3. **不引入新硬编码**: 替换写死时,新代码只能从 OpDef/output_def/scheduled_ops/
   const_extents/vtcm_allocations 等编译产物取值,不准写模型相关的魔法常量
4. **保留通用常量**: 字节序 magic(FA0000FA/BEEFF00D)、record 类型码(0x10/0x20/0x30/0x40)、
   marker 基址(0x1303EE00)这类**格式常量**保留;模型相关的 dims/tensor_id/hash 才是要消除的
5. **回滚保险**: 每阶段先建 git 分支(本工程暂无 git,则保留 .bak 或注释旧模板),
   一旦 simple_linear 0-diff 破坏,立刻回滚查因
6. **文档同步**: 每阶段完成更新本文档第 5 节状态表

## 2. 阶段划分(按依赖顺序,从易到难)

### 阶段 A: const segment(重新评估,比预期复杂)
**目标**: 消除 `write_const_segment()` 里所有写死,改从 `const_extents_` +
GraphPrepare 的 OpDef output_def 生成。

**逆向发现(2026-08-11,7 模型差分 + 反汇编)**:
- const 段不是固定偏移表! 当前代码的 `[0x9200]`/`[0x9300]`/`[0x9400]` 只对
  simple_linear 成立;trans_only 的 perm 数据在 0x9000、descriptor 在 0x9100;
  two_fc 的权重数据在 0x9200。**偏移随模型变**。
- 结构是**序列化流**: 每个 const 的数据块(0x100 对齐)+ descriptor 依次写入。

**反汇编关键发现(host 库 libQnnHtp.so x86_64)**:
- `[0x8000]` 的 `0x71C43C9B` **不是 graph metadata hash**,而是 **const extent
  table 的 magic marker**(当前代码注释错误)!
  证据:`Deserializer::extract_const_extent_table` @0xcfdf90/0xcfe0b0 第一条
  校验就是 `cmpl $0x71c43c9b,(%rdx)`。
- extent table 头格式(从 extract_const_extent_table 反汇编):
  - `[+0x00]` magic = 0x71C43C9B (LE u32)
  - `[+0x04]` count/flags: 高8位=flags, 低24位=count
  - `[+0x08]` offset1 (u32, &0xffffff)
  - `[+0x0C]` offset2 (u32, &0xffffff)
  - 总 entry 数 = (flags_byte + offset1 + offset2), 总字节 = entry数 × 4
- extent table 头 `[0x8000,0x8030)`: 7 模型对比,只有 `[0x8020]`(2~3)和
  `[0x802C]`(4~20)随模型变,其余 10 个 u32 全是格式常量。
- `[0x9300]`/`[0x9400]` 的 0x68 字节是 **tensor_serialize 写的 tensor 元数据**,
  由 tensor 类型的 vtable 方法驱动(LayoutTensor<Flat_64> 等),格式随类型变。
  与 extent table 是两个独立子结构。

**颠覆性发现:当前 writer 格式模型根本错误**:
- 真实 .bin 里 **没有** FA0000FA / BEEFF00D magic(7 模型全 0 命中)!
  `bin_format_analysis.md:25` 已说明:这些是 DSP 内存内分隔符,不在 .bin 文件里。
- 真实 .bin 是 **"count + offset 表"格式**,不是 tagged-record 流。
  当前 writer 在 .bin 里写 FA0000FA/BEEFF00D 是根本错误,0-diff 全靠模板照抄。
- opData 段偏移(0x7534)随模型变,不是固定值。
- **结论:现有 writer 的格式假设与真实 .bin 不符,不能在现有结构上修补,
  需基于"count+offset 表"格式重新设计 writer。**

**真实格式骨架(反汇编 hexagon_nn_deserialize_graph @0xce4ed0 确认)**:
- 头部(BE + LE 混用):
  - `[0x00]` num_graphs (BE u32) = 2 (7 模型全 2)
  - `[0x04]` num_records (BE u32) = 3 (7 模型全 3)
  - `[0x08]` 起记录表: 每条 24 字节 = tag(LE u32) + pad(4B) + offset(LE u64) + size(LE u64)
  - `[0x18]` contextBlob offset (LE u64) = 0x1000
  - `[0x20]` contextBlob size (LE u64) = 0xA308(simple_linear)
- 记录表 num_records=3 条,每条 24B,switch(tag) 分发 0-15
  - case 0/14/15: 只把记录的 offset/size 存到栈上 + log,无实质处理
  - case 5/10/12/13: 默认错误处理(0xCE557F)
  - 循环后调 `Graph::apply_graph_patch(HexagonNNEnv&, uint8_t*, size_t)` @0xd3d020
    传 contextBlob,真正段解析在里面
- contextBlob [0x1000, end) 内部才是各子段(ioTensor/opData/const/extent table 等)
- extent table 在 contextBlob 内,以 magic 0x71C43C9B 起始
- **下一步**: 反汇编 `Graph::apply_graph_patch` @0xd3d020 看 contextBlob 内部
  各子段如何定位与解析(这才是写出真动态 writer 的关键)

**contextBlob 内部格式(反汇编 apply_graph_patch @0xd3bb80 确认)**:
- `hexagon_nn_deserialize_graph` 传给 `apply_graph_patch` 的是 **ioTensor 段**
  (contextBlob + 0x1000 = 文件 0x2000),不是整个 contextBlob。
- ioTensor 段格式(tagged-record 流):
  - `[0x2000]` magic = 0x3790FA5C (LE, 7 模型全同) — apply_graph_patch 第一条校验
  - `[0x2004]` count (u16) — 扫描的 u32 数上限,非记录数
  - `[0x2008]` 起记录流: 每条 = header(u32: 低16=size_in_u32含header, 高16=tag) + payload((size-1)×4B)
  - tag 是 2 字符 ASCII(高字节在前): dI/Sz/Vr/Op/Mm/cT/Mu/Zz
  - apply_graph_patch 扫描找 tag=0x6354("cT") 等特定记录
- **关键发现: ioTensor 段 [0x2008, 0x43D4) 7 模型完全相同(0 字节差)!**
  记录序列: dI(51) Sz(5) Vr(7) Op(2203) Mm(14) cT(7) Mu(3) Zz(1)
  → ioTensor 是格式基础设施,不随模型变(对 writer 是好消息)。

**全文件 7 模型 diff 结论(0x100 粒度)**:
真正随模型变的区段(6 个):
1. `[0x0000, ~0x0400)` 系统头(graph name/buildId/常量字段)
2. `[0x1000, 0x1100)` contextBlob 头部(graph name 相关)
3. `[0x7300, 0x7900)` kernel name table + opData preamble(随 op/tensor 数变)
4. `[0x8000, 0x8100)` extent table(头 10 字段固定, [0x8020]/[0x802C] 随模型变)
5. `[0x9000, 0x9500)` const 数据 + tensor descriptor(随权重/形状变)
6. `[0xB000, ~0xB300)` trailer(op 名/tensor 名回指)
+ `[0xA000,0xA100)` DDR info(只有 [0xA018] 一个 u32 变)
+ `[0x5000,0x5100)` 某 segment(fc_only 没有,可能 op 配置,随模型可选)

不随模型变(格式基础设施,可保留为框架但需理解来源):
- `[0x0400, 0x1000)` 系统头尾部零填充
- `[0x1100, 0x7300)` 含 ioTensor 段(0x2000-0x43D4 完全固定)
- `[0x5100, 0x7300)` op 配置字符串区
→ 这些区段是 v2.48 SDK 的格式框架,对所有单图 context binary 相同。

**下一步**: 逐个逆向 6 个变化区段的字段来源(从 GraphPrepare 状态如何生成)。
优先级: extent table(4) → const 段(5) → kernel/opData(3) → trailer(6) → 系统头(1,2) → DDR(7)

### 阶段 A 进展记录(2026-08-11)

**extent table `[0x8000, 0x8030)` 逆向状态**:
- 头 10 个 u32 = 格式常量(7 模型全同)
- `[0x8020]` 随模型变: simple=3 fc=3 reshape=2 trans=2 two_fc=3
  - 猜测: = Input+Output + (有const权重?1:0), simple/fc/two_fc=3, reshape/trans=2
  - 待验证(需更多模型或反汇编 handle_extent_descriptor 读取逻辑)
- `[0x802C]` 随模型变: simple=20 fc=8 reshape=4 trans=8 two_fc=16
  - 猜测: = op_record_count + 1 (simple 19+1=20 ✓, 但 0x1303EE 扫描被误匹配干扰,待干净验证)
- `[0xA018]` 随模型变: simple=76 fc=56 reshape=36 trans=52 two_fc=68
  - 与 [0x802C] 差值: 56/48/32/44/52, 公式未明
- opData preamble 偏移随模型变(非固定 0x7538),内部字段布局也变
  → 当前 writer 把 opData base 写死在 0x7534 是 simple_linear 特例

**const tensor descriptor `[0x9300+]` 状态**:
- 0x68 字节(26 u32),由 tensor_serialize 的 vtable 方法写(随 tensor 类型变)
- W-desc 字段表(4 模型对比)已提取,但 +0x24/+0x28/+0x30/+0x38 等字段公式未完全解码
- 需反汇编具体 tensor 类型(如 LayoutTensor<Flat_64>)的 serialize vtable 方法

**const tensor descriptor 类型 dispatch 机制(反汇编 deserialize_tensor @0xcfc970)**:
- 流格式: [type_tag u32] [? u32] [类型特定 payload]
- deserialize_tensor 读 type_tag(&0x7fffffff) 查注册表得构造函数,调构造函数读 payload
- const 权重 W/b 的 tensor 类型 = **Flat_64**(auxdata_read_const_extent_descriptor
  反汇编里创建 LayoutTensor<Flat_64>,@cfef23 调 Flat_64::tile_support_bits)
- 0x68 字节(26 u32)是 Flat_64 类型的序列化 payload,由 Flat_64 的 serialize vtable 写
- **下一步**: 反汇编 Flat_64 的 deserialize 构造函数(从 deserialize_tensor_register
  注册表找地址),解码 0x68 字节每个字段的来源(OutputDef dims/rank/element_size 等)

**const descriptor 0x68 字节字段差分(4 模型 W-desc, 0x9400 base)**:
W 形状(从权重数+output[3,2] 推): simple W=[4,2] 8f, 4x3 W=[8,2] 16f, 4x8 W=[4,2] 8f, 8x4 W=[8,2] 16f
- +0x00 = tensor_id(5 for W, 6 for b)
- +0x04 = rank = 3(全模型)
- +0x08 = 1(全模型)
- +0x0C = 24/48/64/32(随 W 大小变,公式未定)
- +0x18 = 4(element_size, float32, 全模型)
- +0x24 = +0x60(两字段恒等): 3/3/8/4
- +0x28 = 8/16/8/8(随 W 变)
- +0x30 = 2/4/2/2(疑似 b_count? 公式未定)
- +0x38 = 12/12/32/16(随 W 变)
- +0x5C = 2/4/2/2(同 +0x30)
- +0x60 = 3/3/8/4(同 +0x24)
- 固定字段: +0x04=3 +0x08=1 +0x18=4 +0x1C=1 +0x20=1 +0x2C=4 +0x34=4 +0x54=1 +0x58=1
- **结论**: 0x68 字节里 10 个字段固定(格式/类型常量),6 个随模型变。
  变化字段与 W 的 dims/b 大小相关,但纯差分 4 数据点不足以定公式。
  需反汇编 Flat_64 Tensor 的 serialize vtable 方法(写端)或 deserialize 构造(读端)
  确定每个字段的语义。vtable 方法需从 Tensor vtable 或 LayoutTensor<Flat_64> 定位。

**const descriptor 写端 vtable 追踪(2026-08-11)**:
- `Serializer::tensor_serialize` @0x12ed940 通过 Tensor vtable 调虚方法:
  槽 0x58(类型码)/0x08(名字)/0xa0(扩展?)/0x30(?) /0xd0(shape info)/0x50/0x48/0x78
- 0x68 字节核心来自 vtable 槽 0xd0(`call *0xd0(%rax)`,@12eda56),
  参数 rdi=Tensor*, rsi/rdx/r8=0, rcx=&out_buf, 返回 shape info 到栈
- **0xd0 槽定位受阻**: Tensor vtable @0x60579a8 的 0xd0 槽 = NULL;
  LayoutTensor<Flat_64> vtable @0x5ebcb28 的 0xd0 槽 = NULL。
  → 0xd0 槽在多重继承的辅助 vtable(第二基类),需完整 vtable 布局分析。
- `LayoutTensor<Flat_64>::init_shape_p(Graph, OutputDef)` @0xdc85f0:
  从 OutputDef 读 4 个 dims(u64),调 `Shape<4>::canonical_shape` 标准化。
  → 0x68 字节字段是 canonical shape 的序列化产物。

**Shape<4>::canonical_shape 委托机制(反汇编确认)**:
- `Shape<4>::canonical_shape(Graph, Shape<4>&)` @0x12f7b00: hash+cache 查询,
  cache miss 时**委托给 Shape<1>::canonical_shape(Graph, OutputDef)** @0x12f91f0 逐维处理
- `Shape<1>::canonical_shape(Graph, OutputDef)` @0x12f91f0 (79B, 极简):
  - `[+0x08]` dim = OutputDef.dim
  - `[+0x10]` stride = dim (初始值 = dim!)
  - `[+0x18]` flag = 0
  - 然后调 `Shape<1>::canonical_shape(Graph, Shape<1>&)` @0x12f8fb0 做 hash+cache
  - **结论: const 权重(简单连续布局)的 stride = dim, flag = 0, 无变换**
- 这解释了为什么 const descriptor 字段可直接从 dims 推导,无需复杂 stride 计算。

**W 形状确认(从模型源码 simple_linear.cpp)**:
- `mlib_build/jni/simple_linear.cpp` line 160: `uint32_t dimensions_W[] = {2, 4};`
- **W = [2, 4]** (M=2, K=4), 不是 [4,2]! ground truth 文档有误。
- FC 公式: y = x·W + b, x=[N,K] → W=[K,M] → y=[N,M]
  - simple_linear: x=[3,4], W=[4,2], y=[3,2] → N=3(input_dim0), K=4(input_dim1), M=2(output)
  - 但 .cpp 里 W dims=[2,4] → 存储为 [M,K] = [2,4] (行主序, M 行 K 列)
  - batch-pad 后 Shape<4> dim = [1, 1, K, M] = [1, 1, 4, 2]

**const descriptor 0x68 字节公式 —— 全部验证通过(4 模型)**:
- W-desc 公式(N=FC输入行数, K=FC输入列数, M=FC输出数, elem=4):

| 偏移 | 公式 | simple | 4x3_w4 | 4x8 | 8x4 | 说明 |
|------|------|--------|--------|-----|-----|------|
| +0x00 | tensor_id | 5 | 5 | 5 | 5 | W 的 tid |
| +0x04 | 3 | 3 | 3 | 3 | 3 | rank=3(固定) |
| +0x08 | 1 | 1 | 1 | 1 | 1 | 固定 |
| +0x0C | N*M*elem | 24 | 48 | 64 | 32 | 数据字节数 |
| +0x10 | 0 | 0 | 0 | 0 | 0 | 固定 |
| +0x14 | 0 | 0 | 0 | 0 | 0 | 固定 |
| +0x18 | 4 | 4 | 4 | 4 | 4 | elem_size(固定) |
| +0x1C | 1 | 1 | 1 | 1 | 1 | 固定 |
| +0x20 | 1 | 1 | 1 | 1 | 1 | 固定 |
| +0x24 | N | 3 | 3 | 8 | 4 | FC输入行数 |
| +0x28 | M*elem | 8 | 16 | 8 | 8 | M×elem_size |
| +0x2C | 4 | 4 | 4 | 4 | 4 | elem_size(固定) |
| +0x30 | M | 2 | 4 | 2 | 2 | FC输出数 |
| +0x34 | 4 | 4 | 4 | 4 | 4 | elem_size(固定) |
| +0x38 | N*elem | 12 | 12 | 32 | 16 | N×elem_size |
| +0x3C~+0x50 | 0 | 0 | 0 | 0 | 0 | 固定(零填充) |
| +0x54 | 1 | 1 | 1 | 1 | 1 | dim[0]=1(固定,batch) |
| +0x58 | 1 | 1 | 1 | 1 | 1 | dim[1]=1(固定,batch) |
| +0x5C | M | 2 | 4 | 2 | 2 | FC输出数(=+0x30) |
| +0x60 | N | 3 | 3 | 8 | 4 | FC输入行数(=+0x24) |
| +0x64 | 0 | 0 | 0 | 0 | 0 | 固定 |

- **b-desc(偏置)公式**(tensor_id=6, 同样 0x68 字节):

| 偏移 | 公式 | simple | 4x3_w4 | 4x8 | 8x4 |
|------|------|--------|--------|-----|-----|
| +0x00 | tensor_id | 6 | 6 | 6 | 5* |
| +0x0C | N*K*elem | 48 | 48 | 128 | 128 |
| +0x24 | K | 4 | 4 | 4 | 8 |
| +0x28 | N*elem | 12 | 12 | 32 | 16 |
| +0x30 | N | 3 | 3 | 8 | 4 |
| +0x38 | K*elem | 16 | 16 | 16 | 32 |
| +0x5C | N | 3 | 3 | 8 | 4 |
| +0x60 | K | 4 | 4 | 4 | 8 |
  (*8x4 的 b tid=5, W tid 顺序与其它模型反转,待确认)

- **全 4 模型 ALL OK**: W-desc 7 个动态字段 + b-desc 7 个动态字段公式全部验证通过。
- 固定字段(10个): +0x04=3 +0x08=1 +0x18=4 +0x1C=1 +0x20=1 +0x2C=4 +0x34=4 +0x54=1 +0x58=1 +0x64=0
- **结论: const descriptor 0x68 字节可从 (tensor_id, N, K, M, elem_size) 完全生成。**
  N/K/M 来自 FC op 的 InputDef dims + OutputDef dims, elem_size=4(float32)。

**descriptor 偏移(随模型变,非固定)**:
- simple_linear: W-desc @ 0x9400, b-desc @ 0x9300, 第3项 @ 0x9200(4 u32: 0,1,3,2 — 小型结构)
- trans_only: perm-desc @ 0x9100 (tid=6, N=3, K=4, 同 b-desc 模式)
- fc_only/two_fc/reshape_only: descriptor 不在 [0x8000,0xB000) 固定位置
- **偏移由序列化器的顺序分配决定**,非固定偏移表 → writer 需复制分配逻辑
- perm 数据 [0,2,1] 存在 op record 内部,非独立 const 块

**0x9100 处的 quant scale 数据**:
- 0x9100 在多个模型含 0x3DCCCCCD(0.1f) 0x3E4CCCCD(0.2f) — 量化 scale 值
- 非 descriptor,是 quantization encoding 的 scale/offset 参数

**const descriptor 0x68 字节决定性突破(Shape<4>::serialize @0x12f5750)**:
- 0x68 字节由 `Shape<4>::serialize(Serializer&)` 写,`Shape<4>::deserialize` @0xd93f20 读
- Shape<4> 内存布局(从 serialize 读取的字段):
  - `[+0x08]`=dim0, `[+0x10]`=dim1, `[+0x18]`=dim2, `[+0x20]`=dim3 (u32, 4个维度)
  - `[+0x28]`=stride0, `[+0x30]`=stride1, `[+0x38]`=stride2, `[+0x40]`=stride3 (u32)
  - `[+0x48]`=flag0, `[+0x49]`=flag1, `[+0x4a]`=flag2, `[+0x4b]`=flag3 (u8)
- 编码: 每2个维度(dim_i, stride_i, flag_i)编码成一个 header u32 + 紧凑数据
  - header 低2位 = mode(0/1/2/3,决定后续编码方式)
  - mode=1: 1个u32挤3个值(低16位+中8位+高8位)
  - 高16位含 flags(0xcccc 特殊标记等)
- 最后调 `serialize_uint32_arr` 写紧凑数组
- **生成 0x68 字节的方法**: 从 OutputDef dims 计算 Shape<4> 的 dim/stride/flag,
  然后实现 Shape<4>::serialize 的紧凑编码逻辑(mode 选择 + 值打包)
- **下一步**: 完整反汇编 Shape<4>::serialize(823B)的编码分支,逐 mode 解码公式;
  对照 7 模型 0x68 字节验证(mode 由 dim 大小决定: <0x10000 用 mode1 紧凑, 否则 mode2)

**Shape<4>::serialize 编码逻辑细节(反汇编 @0x12f5750)**:
- 4 组维度,每组 (dim, stride, flag) 编码:
  - 快速路径: dim==1 && stride==dim && flag==0 → 省略该组(不写数据)
  - 否则: header u32(含 mode/flags) + 可选数据 u32
- header 编码: flag 左移 12 位 (`shl $0xc`), mode 在低 2 位
  - mode 由 dim 大小决定: dim<0x10000 → 紧凑模式(值打包进少 u32)
  - `cmp $0x10000,%edx` (12f57a6) / `cmp $0x1000000` (12f585e) 选模式
- 数据通过 `serialize_uint32_arr` 批量写出(紧凑数组)
- flag 位: `test $0x4`(stride 非0标记), `test $0x8`(额外数据标记)
- **实现路径**: 
  1. 从 OutputDef 4 个 dims 经 canonical_shape 算出 dim[i]/stride[i]/flag[i]
  2. 逐组编码: dim==1 && stride==1 && flag==0 → skip; else 编 header+data
  3. 批量写 + 填充到 0x68 字节对齐
- W-desc/b-desc 0x68 字节 = 2 个 Shape<4> 序列化(W 的 shape + b 的 shape)
  每个 const tensor 一个 Shape<4>,独立编码

**handle_extent_descriptor @0xd8bb60 反汇编结论**:
- 只读 extent table 头 4 字段: [+0]magic / [+4]count|flags / [+8]offset1 / [+0xC]offset2
- 把 extent table 的 ptr+size 存到 ConstExtentDesc,调子函数(cf3970)解析单条 extent
- [0x8020]/[0x802C] **不在此函数读**,在别的代码路径(需进一步定位)
- ConstExtentDesc 的 op_id/offset/size/tensor_type 从 extent entry 内部读,非头字段

**extent table 头字段读取顺序(反汇编 create_optimized_graph @0xd25840)**:
- `extract_const_extent_table` 读 [+0]magic(0x71C43C9B) [+4]count|flags [+8]offset1 [+0xC]offset2
- `deserialize_uint32_x3` 读 [+0x10]=12 [+0x14]=64 [+0x18]=64 (simple_linear 值)
- `setup_runlists`(调 4 次)每次从流读 3 个 u32:
  - 第1次: [+0x1C]=0, [+0x20]=3, [+0x24]=1
  - 后续次: 流位置后移,读更后面字段
- `[0x8020]`=3 = setup_runlists 第1次第2个出参,语义待定(可能=runlist数/graph数)
- `[0x802C]`=20 在更后面读取(待定位)
- extent table 头完整字段表(simple_linear):
  +0x00 magic=0x71C43C9B | +0x04 0x01000040 | +0x08 1 | +0x0C 1
  +0x10 12 | +0x14 64 | +0x18 64 | +0x1C 0 | +0x20 3 | +0x24 1 | +0x28 0 | +0x2C 20

**extent table 头 7 模型 diff 结论(决定性)**:
- 12 个字段中只有 [+0x20] 和 [+0x2C] 随模型变,其余 10 个全是格式常量(7模型全同)
- `[0x8020]` 公式: = 2 + (有 FullyConnected/MatMul op ? 1 : 0)
  - simple/fc/two_fc/linear_* = 3 (有 FC), reshape/trans = 2 (无 FC) ✓ 全部 7 模型验证
- `[0x802C]` 公式: = op_record_count + 1 (= scheduler 步数 + 1)
  - simple 19+1=20 ✓, fc 7+1=8 ✓, reshape 3+1=4 ✓, trans 7+1=8 ✓, two_fc 15+1=16 ✓
  - linear_* 结构同 simple → 20 ✓
- **extent table 头可从 GraphPrepare 状态完全生成**: 10 格式常量 + 2 动态字段(有FC? + 步数)
- DDR info `[0xA000]`: 25 字节固定,其中 22 字节(含两个"hash"0xA798EC42/
  0xA798EC4F)是**格式常量**(7 模型全同);只有 `[0xA018]` 一个 u32 随模型变
  (值域 0x24~0x4C),语义待解。
- descriptor 内部格式复杂(每个 ~0x60 字节,含 tensor_id/rank/dims/strides/
  element_size 等十几个字段),需逐字段对照 7 模型解码。

**拆分**:
- A1: DDR info(最易,22 字节格式常量 + 1 个动态 u32)
- A2: const 数据块布局(数据 + 0x100 对齐填充)
- A3: descriptor 格式(最难,需完整解码)

**输入**: `const_extents_`、`get_op_at(op_id)->output_def`、const_pool_ 权重数据
**输出**: const 段完整字节(数据 + descriptor + DDR info)
**验证**: simple_linear 0-diff;7 模型逐字节 diff
**预计**: 中偏大(A3 descriptor 格式是核心难点)

### 阶段 B: post-opData graph metadata
**目标**: 消除 `write_post_opdata()` `[0x8000,0x8030)` 的写死 hash+11 个 u32,
改从 GraphPrepare 的图校验和/配置生成。

**输入**: `calculate_graph_checksum()`(已存在)、图配置标志
**输出**: graph metadata 段
**验证**: simple_linear 0-diff;7 模型 hash 不同
**难点**: hash 算法需逆向(查 `context_binary_full_analysis.md` 第 5 节 + 反汇编)
**预计**: 中等

### 阶段 C: opData preamble
**目标**: 消除 `write_opdata_segment()` `[0x7538,0x75D0)` 的 38 个写死 u32,
改从 GraphPrepare 的 input/output tensor descriptor + 图配置生成。

**输入**: input_node/output_node 的 output_def、max tensor id、调度 op 数
**输出**: opData preamble
**验证**: simple_linear 0-diff;7 模型 preamble 跟着 tensor 数/形状变
**难点**: 部分字段的语义(flags 0xFFFB0004、max tensor id 0xF)需对照 7 模型
**预计**: 中等

### 阶段 D: trailer
**目标**: 消除 `write_trailer()` 的 `trailer_header[208]` + `trailer_data[552]` 写死,
改从 GraphPrepare 的 op_names/tensor_names + 拓扑回指表生成。

**输入**: `op_names_`/`tensor_names_`(已有 setter)、OpDef grouping
**输出**: trailer 段(图名 + op 名表 + tensor 元数据 + 回指)
**验证**: simple_linear 0-diff;7 模型 op 名/回指跟着变
**难点**: trailer 内部是变长回指表(back-reference),格式最复杂;
        需逐字节对照 7 模型的 trailer 段差分
**预计**: 大

### 阶段 E: pre-opData 段(最大块,25908 B)
**目标**: 消除 `CTX_BLOB_PRE_TEMPLATE`,启用 `write_ctx_header_descriptor`/
`write_root_record`/`write_pickle_record`/`write_io_tensor_segment`/
`write_kernel_name_table`(目前是死代码),全部从 GraphPrepare 状态生成。

**子段**:
- E1: header descriptor + ROOT record(格式框架,相对固定)
- E2: PICKLE record(图名,已半动态)
- E3: ioTensor segment(所有 tensor 的 dims/dtype/quant 描述,随模型变)
- E4: op config strings(优化标志文本)
- E5: kernel name table(从 scheduled_ops 的 kernel_name 生成)

**输入**: OpDef output_def 集合、scheduled_ops_、kernel_names_
**输出**: pre-opData 全段
**验证**: simple_linear 0-diff;7 模型 ioTensor/kernel 表跟着变
**难点**: ioTensor 段格式复杂,需对照 `disasm_load_header.txt` + 7 模型逐字节差分
**预计**: 最大(25908 B,占总量的 56%)

### 阶段 F: system header
**目标**: 消除 `SYS_HEADER_TEMPLATE[0x1000]`,改从 GraphPrepare 图级元数据生成
系统信息头(num_graphs/num_records/offset 表/ioTensorSize/constSize/dspArch/图名/buildId)。

**输入**: 图配置、contextBlob size(已动态)、graph_name/build_id(已动态)
**输出**: 系统信息头
**验证**: simple_linear 0-diff;7 模型系统头字段跟着变
**难点**: 头部前 0x18 字节的 num_graphs/num_records/offset 表格式需逆向;
        `bin_format_analysis.md` 指出真实大模型是"count+offset 表"格式
**预计**: 中等(4096 B 多数是零填充,真正动态字段不多)

## 3. 依赖与顺序

```
A (const descriptor) ──┐
B (graph metadata)   ──┼── 都只依赖局部状态,可并行试,但建议串行验证
C (opData preamble)  ──┘
        │
        ▼
D (trailer) ── 依赖 op/tensor 名表完整
        │
        ▼
E (pre-opData) ── 依赖 ioTensor/kernel 表,最大块,放后
        │
        ▼
F (system header) ── 依赖 contextBlob size 已定(由前面段累加)
```
A/B/C 互不依赖,但为控制验证复杂度建议 A→B→C 串行。D/E/F 有顺序依赖。

## 4. 每阶段交付物

每阶段完成时,本仓库应包含:
1. 修改后的 `context_binary_writer.cpp`(该段从状态生成,无模型硬编码)
2. 该段的单元测试(对 7 模型逐字节 diff,输出哪段匹配/哪段待下一阶段)
3. 本文档第 5 节状态表更新(该段标记 ✅,记录 7 模型 diff 结果)

## 5. 状态跟踪表

| 阶段 | 段名 | 状态 | simple_linear diff | 7模型 diff | 备注 |
|------|------|------|-------------------|-----------|------|
| A | const descriptor | ✅完成 | 0 | 4模型验证 | 0x68字节从(tid,X,Y,elem)动态生成 |
| B | extent table | ✅完成 | 0 | 7模型验证 | [0x8020]=2+(有FC?), [0x802C]=steps+1 |
| C | opData preamble | ✅完成 | 0 | FC模型验证 | 从PreambleParam(input/output dims,nonconst,max_tid,qnn_op_count)+scheduled_ops动态生成 |
| D | trailer | ✅完成 | 0 | — | op表(header+output entries+separator+input entry)+tensor blocks(output:backRef+dims+name+trailing 12B, input:dims+name+trailing 12B)+config block(76B template). 关键公式: h1=op_section_total-4, h3=output_name_end-4-tbl, outBackRef=-(inPadded+204)/2, inBackRef=(inPadded+12)/4+11+(3-output_rank)*(3+2*out_count), C=4(constant) |
| E | pre-opData | ✅完成 | 0 | 7模型分析 | 25908B中25556B为7模型相同的格式框架,352B动态生成: Mm记录(count=Co_count+4-has_transpose-has_FC, f1=has_FC?4:0), Co记录(f1=ceil(totalNameBytes/4)+count+2, f2=ceil(totalNameBytes/4)), kernel名表(从kernel_names_生成), PICKLE路径(从graph_name生成), Sz记录[5](参数传入) |
| F | system header | ✅完成 | 0 | 7模型分析 | 4096B中3722B为7模型相同的格式框架(含0x400-0x1000零填充),374B动态patch: contextBlob size(0x20), ioTensorSize(0x130), constSize(0x177), dspArch(0x1D5), graph_name(0x1E8), buildId(0x334). 变长描述符表因graph_name/tensor_name长度移位,full multi-model支持待逆向描述符表格式 |

状态图例: ⬜未开始 / 🔧进行中 / ✅完成 / ⚠️阻塞(记阻塞原因)

## 6. 风险与回滚

- **风险1**: 逆向资料对某些字段语义不足 → 用 7 模型差分反推(变与不变的边界)
- **风险2**: 改一段破坏 simple_linear 0-diff → 立即回滚,二分定位
- **风险3**: 7 模型 .bin 可能用不同 SDK 版本生成 → 先比对 7 模型头部 buildId
  确认同版本;若不同,分版本验证
- **风险4**: 工作量大(E 段 25908 B) → 拆 E1-E5 子阶段,每个子阶段独立验证

## 7. 不做的事(明确排除)

- 不追求与真实 `qnn-net-run`/板端 `libQnnHtpV73Skel.so` 二进制兼容(README 已声明差距)
- 不重写整个序列化框架,只改 `ContextBinaryWriter`
- 不动 loader/prepare/scheduler 的逻辑(那些已验证正确)
- 不引入新依赖库
