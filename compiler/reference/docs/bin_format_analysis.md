# QNN .bin Context Binary 格式分析

从 `libQnnHtpV73QemuDriver.so` (Hexagon DSP, 11.5 MB) 符号表 + 反汇编 + **真实 .bin 样本头部**提取。

## 真实 .bin 样本

```
test_minimal.serialized.bin  (742 MB, Qwen3 4B 模型)
路径: /data01/rqilin/qwen3_llm_v2/example2/host_linux/assets/artifacts/ar128-cl2048_v73/
```

## 字节序: 大端 (确认!)

真实 .bin 头部 (大端 uint32):
```
[0x00] 00 00 00 02  = num_graphs = 2
[0x04] 00 00 00 03  = num_records = 3
[0x08] 00 00 00 00  = 0
[0x0C] 00 00 00 01  = 1
[0x10] B8 03 00 00  = 0xB8030000 (偏移, 大端)
[0x18] 00 10 00 00  = 0x100000 = 1MB (块大小)
[0x20] 00 90 5E 2E  = 0x2E5E9000 (大偏移/基址)
```

## 无文件级 BEEF/FA00 magic!

搜索 4096 字节头部: **BEEFF00D / FA0000FA / FA0000FE 全部 0 命中**。
这些常量是 DSP **内存内**的分隔符,不在 .bin 文件里。
真实 .bin 是 **计数 + 偏移表** 格式,不是 tagged-record 流。

## 序列化 API (反汇编确认)

```
hexagon_nn_serialize_to_mem  (0x5ec380, 44B) — 极简: 仅 log + return -1
  → 实际序列化逻辑在 Graph::compile / compile_exec_list / compile_lists
hexagon_nn_deserialize_graph (0x5ef880, 2012B) — 真实主入口
  → switch(0-11) 跳转表, 读头部计数+偏移, 分段加载
```

## hexagon_nn_deserialize_graph 反汇编关键

```asm
5ef880: 入参 r0-r5, stack frame 216 字节
5ef8e0: r22 = add(r2, #8)           ; r2 = 数据指针, 从 r2+8 读
5ef920: r2 = memw(r22+#-8)          ; 读 switch case 值
5ef928: if (cmp.gtu(r2, #11)) ...   ; switch 0-11
5ef930: r3 = add(pc, #2461736)      ; 跳转表基址
5ef934: r2 = memw(r3 + r2<<#2)      ; 查表
5ef938: r2 = add(r2, r3)            ; 相对偏移
5ef93c: jumpr r2                    ; 跳转
5ef944: r1:0 = memd(r22+#8)         ; 读 8 字节 (偏移对)
5ef94c: r1:0 = add(r1:0, r17:16)    ; 偏移相加 (基址 + 偏移)
```

## Deserializer::load_header (0x6017c0, 12B)

```asm
6017c0: jumpr r31              ; 直接返回 (极简)
6017c4: memb(r0+#4) = #0       ; 写 0 到 this+4
6017c8: memw(r0+#0) = r2       ; 写 r2 到 this+0
```
→ load_header 只是设读位置,实际 header 解析在 deserialize_graph 主体。

## Deserializer::get_name (0x6017e0)

含类型匹配常量:
```asm
p0 = cmp.eq(r2, ##1908685979)   = 0x71A6009B  (类型 magic 1)
p0 = cmp.eq(r2, ##-339869634)   = 0xEBC0FEFE  (类型 magic 2)
```
→ 这些是类型注册表的 magic, 用于匹配 tensor 类型。

## 序列化 C API (符号表确认)

```
hexagon_nn_serialize_to_mem     0x5ec380  44B
hexagon_nn_serialize_file       (未找到符号)
hexagon_nn_serialize             (未找到符号)
hexagon_nn_multigraph_serialize  (未找到符号)
hexagon_nn_deserialize_graph     0x5ef880  2012B
gpe_serialize_to_mem            (符号存在)
gpe_serialize                    (符号存在)
gpe_validate_pickle              (符号存在)
```

## 反序列化流程 (从符号表 + 反汇编推断)

```
hexagon_nn_deserialize_graph(data, size, graph, ...)
  1. 读头部: num_graphs, num_records, 偏移表 (大端)
  2. switch(0-11) 按 record 类型分发
  3. 每条 record: 读偏移对 (8B), 加基址得到绝对地址
  4. Deserializer::load_header(offset) — 设读位置
  5. extract_const_extent_table(tag)
  6. extract_const_extent_data(offset, size, dst, ...)
  7. auxdata_class_index(tag, bool)
  8. resize_object_tables(runlist_auxdata_seg_desc)
  9. segmentjob_deserialize_ops(seg, job)
```

## Deserz 低层读取 (真实签名, 已对齐)

```
deserialize_fread(void* dst, uint32_t size, bool align4)
deserialize_str()                              → string
deserialize_uint32_x2() / x3() / x4()          → 2/3/4 个 uint32
deserialize_uint32_arr(uint32_t*, uint32_t)
deserialize_buf(uint32_t size, void* dst)
deserialize_buf_withlen(uint32_t size, void* dst)
deserialize_shared_obj_func(void** out)
apply_segment_fixups(runlist_fixup_state&)
```

## Tensor 类型 (ConcreteTensor 模板实例, 30+ 种)

```
PlainFloat, PlainFloat16, PlainBFloat16
QuantUint8/Int8/Uint16/Int16/Int32
QUint8Crouton, QUint16Crouton, QUint8Crouton4x1_TCM
SWeights, pkSWeights, shflSWeights, SWeights_TCM
F16Weights, WeightsF16, pkWeightsF16, pkWeightsB16
QFloat_TCM, QFloatCrouton_TCM
Flat_8/16/32/64, Flat5D_8/16/32
Crouton_16/32, Crouton_16_DeepAR
```
blocktable 按 tensor 类型分多种 accessor (`LayoutTensor<T>::blocktable_accessor()`)

## RuntimeAllocator (VTCM, 真实签名)

```
handle_extent_descriptor(ConstExtentDesc&, Deserializer&, far_vm_ptr_t, uint64_t)
deserialize_pools(HexagonNNEnv&, Deserializer&, far_vm_ptr_t, uint32_t, wide_iovec_t&)
deserialize_switchable_pool_info(vector<uint64_t>&)
get_pool_and_buf_info(Deserializer&, uint32_t, bool)
set_shared_spillfill(far_vm_ptr_t, uint32_t)
```

## 仍需 (可继续在 dev133 反汇编)

1. switch case 0-11 各自读取的字段 (需反汇编 0x5ef940-0x5f0000)
2. 真实 .bin 偏移表结构 (需更大样本或解析 742MB 文件)
3. Op::serialize_internal 字段级布局
4. blocktable 编码 (按 tensor 类型)
5. pickle 格式 (weight 序列化)
