#define HNNX_SER_PRECISE 1 // 本 TU 为精确族: Op/Tensor/GraphPrepare/Graph 用全局名 (导出符号与 .so 一致)
#include "hnnx/serialize/serializer.hpp"
#include <cstring>
#include <vector>

namespace hnnx {

// .bin deserialization
// Source: deserializer.cc, deser_concurrent.cc, graph_auxdata.cc
//
// .bin 格式 (从 QemuDriver 符号表 + 错误字符串推断):
// 1. BinHeader (大端计数 + 偏移表, 无文件级 magic)
// 2. tagged-record 流: [encoded_tag][word_count][third][data...]
// 3. Separators: 0xFA0000FA (normal), 0xFA0000FE (aux), 0xBEEFF00D (end)
// 4. const extent table + data
// 5. class index (类型注册表)
// 6. segment spans + per-segment ops
// 7. pickle (weight 序列化)
//
// 读取顺序 (错误字符串):
// "Magic check failed. Expecting %lx got %lx"   (类型注册表 magic, TYPE_MAGIC_1/2)
// "Could not read number of graph of virtual addresses from serialized binary"
// "Could not read number of shared weights from serialized binary"
// "failed to read segment blob."
// "Failed to read pickle size for graph deserialization"
// "Failed to read pickle VA for graph deserialization"

// ---------------------------------------------------------------------------
// Deserializer ctor @0xcfcf20 —— Deserz 基内联同体 (full_deser=this;
// buf_limit 不写), 后接扩展区: objindex/tensorconn/word@0x108 清零,
// name_buf[4096]@0x10a 不初始化, 0x1110/0x1118/0x1120 清零,
// 0x1128..0x1138 清零 (segments 头), 双 map 空自引用, 三 vector 清零。
// ---------------------------------------------------------------------------
Deserializer::Deserializer(char const *data, size_t size, ::Graph *graph)
    : Deserz(this, data, size, graph), objindex(), tensorconn{}, tc_flag1(false), tc_flag2(false),
      name_buf(), pad_110a{}, x_1110(0), x_1118(0), x_1120(0), segs_1128{}, pad_1140{}, named_cexdescs(),
      named_weight_bufs(), pad_1190{}, op_deserialize_fn_list(), tensor_deserialize_fn_list(),
      blocktable_link_table()
{
    // @0xcfd0b8 (扩展段, maps/vector 初始化之后): buf_limit = p —— 基类段
    // 0xcfcf92 直跳 +0x78 不写; 滑动窗口语义由首次 fill_buffer 展开
    // (窗口步长 0x8000, 与 Serializer::f_2d4 一致)。
    buf_limit = data;

    // name_buf 故意不初始化 (ctor 无对应 store)
}

Deserializer::~Deserializer()
{
    // 段表 (x_1118) 为重实现侧 new[] 所得 (见 auxdata_deserialize_segments)
    delete[] reinterpret_cast<DeserSegmentSpan *>(x_1118);
    x_1118 = 0;
}

// load_header: 真实 @0x6017c0 极简 (仅设读位置); 大端头解析在
// hexagon_nn_deserialize_graph 主体。重实现侧: 解析进 bufstart 前 0x28 字节。
void Deserializer::load_header(size_t offset)
{
    (void)offset;
}

// auxdata_deserialize_segments: 段计划 @0xcf5210 (756B)。
// 段描述符 0x1d0 字节; 表落 +0x1118, 计数落 +0x1110 (旧注释对齐)。
void Deserializer::auxdata_deserialize_segments(uint32_t param)
{
    if (param <= 0x89) return;

    uint32_t segment_count = (param - 0x12) / 0x0f;
    if (segment_count > 0 && segment_count < 1000) {
        delete[] reinterpret_cast<DeserSegmentSpan *>(x_1118);
        x_1118 = reinterpret_cast<uint64_t>(new DeserSegmentSpan[segment_count]());
        x_1110 = segment_count;
    }
}

// auxdata_read_const_extent_descriptor @0xcfeE20 (628B)
void Deserializer::auxdata_read_const_extent_descriptor(uint32_t tag) { (void)tag; }

// resize_object_tables @0xcf5540 (221B)
void Deserializer::resize_object_tables(runlist_auxdata_seg_desc const &desc) { (void)desc; }

// segmentjob_deserialize_ops @0xd33620 (653B)
void Deserializer::segmentjob_deserialize_ops(uint32_t seg_idx, uint32_t job_idx)
{
    (void)seg_idx;
    (void)job_idx;
}

// handle_auxdata_deser @0xd39ae0 (253B)
void Deserializer::handle_auxdata_deser(uint32_t tag, uint32_t size)
{
    switch (tag) {
        case TAG_IO_DMA_BYPASS:
        case TAG_SPILL_FILL_INSTEAD:
        case TAG_EXTENDED_UDMA:
        case TAG_IO_TENSORS_CONFIG:
        case TAG_MULTICAST_CONFIG:
        case TAG_NUM_SEGMENTS:
        case TAG_SELF_SLICING:
            deserialize_skip_words(1);
            break;
        case TAG_EXTRA_CONFIG:
            deserialize_skip_words(0x2c / 4);
            break;
        case TAG_RUNLIST_SEGMENT_DESC:
            deserialize_skip_words(2);
            break;
        default:
            deserialize_skip_words(size);
            break;
    }
}

// auxdata_class_index: 类型注册表 (class_id → 构造函数)
void Deserializer::auxdata_class_index(uint32_t tag, bool flag)
{
    (void)tag;
    (void)flag;
}

void Deserializer::auxdata_temparr_sizes(uint32_t tag) { (void)tag; }

void Deserializer::extract_const_extent_table(uint32_t tag) { (void)tag; }

void Deserializer::extract_const_extent_data(uint64_t offset, uint32_t size, void *dst, uint64_t a, uint64_t b)
{
    (void)a;
    (void)b;
    char const *src = bufstart + offset;
    if (src + size > bufend) {
        errstr = "const extent out of range";
        return;
    }
    std::memcpy(dst, src, size);
}

uint32_t Deserializer::crate_size_according_to_segments()
{
    uint32_t total = 0;
    DeserSegmentSpan const *spans = reinterpret_cast<DeserSegmentSpan const *>(x_1118);
    for (unsigned i = 0; i < x_1110 && spans; ++i) total += uint32_t(spans[i].byte_size);
    return total;
}

void Deserializer::get_forward_span(uint32_t seg, uint32_t a, uint32_t b)
{
    (void)seg;
    (void)a;
    (void)b;
}

void Deserializer::skip_to_after_span(DeserSegmentSpan const &span)
{
    size_t end = size_t(span.byte_offset) + size_t(span.byte_size);
    if (end <= buffer_remain()) set_read_cursor(reinterpret_cast<unsigned char *>(const_cast<char *>(bufstart)) + end);
}

const char *Deserializer::get_name(uint64_t a, uint64_t b)
{
    (void)a;
    (void)b;
    return "";
}

} // namespace hnnx
