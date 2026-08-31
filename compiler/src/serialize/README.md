# src/serialize/ — .bin 序列化 / 反序列化

对应真实 `serialize_oplist.cc` / `deserializer.cc` / `graph_auxdata.cc`。tagged-record 流格式。

## 文件

| 文件 | 真实来源 | 职责 |
|------|----------|------|
| `serializer.cpp` | `serialize_oplist.cc` | Serializer (双模式 prescan+write), write_tagged_record (状态机), serialize_fwrite/uint32 |
| `deserializer.cpp` | `deserializer.cc` @ 0xCFCD30 等 | Deserializer/Deserz: load_header, auxdata_deserialize_segments, handle_auxdata_deser (9 种 tag) |
| `bin_format.cpp` | `serialize_oplist.cc` / `const_extent_serialize.cc` | encode/decode_bin_tag, make_plan_for_deser_by_segments, serialize_blocktable/qp_record |

## .bin 格式

```
[uint32 encoded_tag][uint32 word_count][uint32 third][data...]
tag 编码: (tag & 0xFFFF | tag << 16) ^ 0xFFFF
分隔符: 0xFA0000FA (normal)  0xFA0000FE (aux)  0xBEEFF00D (end)
```

## 重实现自定义 tag (serializer.hpp)

```
TAG_OP_RECORD       = 0x4F50  runlist op 描述符
TAG_IO_TENSOR_DESC  = 0x494F  IO 张量描述符
TAG_GRAPH_HEADER     = 0x4748  图头 (op/IO 计数)
0xCF56               const pool 数据块
0x4254               block table
0x5347               segment plan
```

注: 这些自定义 tag 与真实 QNN `.bin` 不兼容,仅供往返验证。
