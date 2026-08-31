// test_m33_serializer.cpp — M33 布局/行为验证 (Serializer/Deserializer/Deserz)
//
// 静态: sizeof(Serializer)=0x360 / FileSerializer=0x398 / Deserz=0xd8 /
//       Deserializer=0x11e8 (头内 static_assert 的运行时复述)。
// 运行时: Deserializer 构造基类状态 (Deserializer 路径 buf_limit 不写)、
//       游标/对齐 fread、u32 x2/x3/x4、字符串记录、错误通道、
//       段表 auxdata_deserialize_segments / crate_size_according_to_segments、
//       bin tag 编解码。
//
// 注: Serializer/FileSerializer 的 ctor 运行时构造需 fa::FancyAllocator 真层级
//     (dynamic_cast 源 hnnx::Allocator 完整 + 派生关系), 属 M35 —— 此处仅静态布局。
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/serialize/deserz.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace hnnx;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg)                                                                                            \
    do {                                                                                                            \
        if (cond) {                                                                                                 \
            ++g_pass;                                                                                               \
        } else {                                                                                                    \
            ++g_fail;                                                                                               \
            std::printf("FAIL: %s (line %d)\n", msg, __LINE__);                                                     \
        }                                                                                                           \
    } while (0)

static_assert(sizeof(Deserz) == 0xd8, "Deserz 0xd8");
static_assert(sizeof(Deserializer) == 0x11e8, "Deserializer 0x11e8");
static_assert(sizeof(Serializer) == 0x360, "Serializer 0x360");
static_assert(sizeof(FileSerializer) == 0x398, "FileSerializer 0x398");
// 虚槽计数: Deserz 4 (D1/D0/fill_buffer/deserialize_fread), Serializer 25,
// FileSerializer 25 (派生面一致)。以成员指针表大小近似复核 Deserz 面:
static void test_static_layout()
{
    CHECK(sizeof(DeSerError) == 8, "DeSerError 单指针");
    CHECK(sizeof(DCrate) == 0x18, "DCrate 0x18");
    CHECK(sizeof(runlist_fixup_state) == 0x38, "runlist_fixup_state 0x38");
    CHECK(sizeof(deser_segment_span) == 16, "deser_segment_span");
}

static void test_deserializer_base_state()
{
    char buf[64];
    std::memset(buf, 0xAB, sizeof buf);
    Deserializer d(buf, sizeof buf, nullptr);

    // Deserz 基状态 (ctor @0xcfcf20 内联同体)
    CHECK(d.read_cursor() == (unsigned char *)buf, "bufp = p");
    CHECK(d.buffer_offset() == 0, "offset 0");
    CHECK(d.buffer_remain() == 64, "remain = n");
    CHECK(d.read_end() == (unsigned char *)buf, "buf_limit = p (扩展段 0xcfd0b8 补写, 滑窗起点)");
    CHECK(d.is_base_deser(), "full_deser 自引用 (+0x50 = this)");
    CHECK(d.shared_ctx() == &d, "shared_ctx = full_deser");
    CHECK(d.runtime_allocator() == nullptr, "allocator 零初始化");
    CHECK(d.scratch_ptr() == nullptr && d.scratch_end() == nullptr, "DCrate nextp/limitp 空");
    CHECK(!d.is_compressed(), "format_version = 0");
    CHECK(d.errstr == nullptr, "errstr 空");

    // buffer 起点不因构造被改动 (只写对象自身)
    CHECK(buf[0] == (char)0xAB && buf[63] == (char)0xAB, "缓冲未触");
}

static void test_fread_and_helpers()
{
    unsigned char bytes[] = {0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
                             0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00};
    Deserializer d((char const *)bytes, sizeof bytes, nullptr);

    uint32_t arr[2] = {0, 0};
    d.deserialize_uint32_arr(arr, 2);
    CHECK(arr[0] == 1 && arr[1] == 2, "fread 顺序读");
    CHECK(d.buffer_offset() == 8, "游标前移");

    // 对齐读: 当前偏移 8 已对齐 → 无 pad; 读 1 字节后偏移 9, align4 → 跳到 12
    uint8_t one = 0;
    d.deserialize_buf_withlen(1, &one);
    CHECK(one == 3 && d.buffer_offset() == 12, "align4 填充");

    // 缓冲 16B: 第 1 字 @12 读到 4; 第 2 字时 bufp==buf_limit==bufend →
    // fill_buffer 抛 length_error("deserialize underflow") (@0xcfde73, 16B 异常,
    // 日志 "deserializer.cc:462::ERROR:over-read of serialized data")
    bool threw = false;
    try {
        auto x2 = d.deserialize_uint32_x2();
        CHECK(x2.a == 4, "uint32_x2 首字");
    } catch (std::length_error const &e) {
        threw = true;
        CHECK(std::strcmp(e.what(), "deserialize underflow") == 0, "underflow 异常文案");
    }
    CHECK(threw, "读尽抛 length_error (非 errstr)");

    // 空目的地 = 跳过 (deserialize_buf 转发 fread, dst=null 走跳过分支)
    Deserializer d2((char const *)bytes, sizeof bytes, nullptr);
    d2.deserialize_buf(8, nullptr);
    CHECK(d2.buffer_offset() == 8, "dst=null 跳过");
    d2.deserialize_skip_words(2);
    CHECK(d2.buffer_offset() == 16, "skip_words n*4");
}

static void test_deserialize_str()
{
    // u32 长度前缀 + 4 对齐字节串: "abc\0" + pad
    unsigned char bytes[] = {3, 0, 0, 0, 'a', 'b', 'c', 0};
    Deserializer d((char const *)bytes, sizeof bytes, nullptr);
    std::string_view sv = d.deserialize_str();
    CHECK(sv == "abc", "长度前缀字符串");
    CHECK(d.buffer_offset() == 8, "字符串后含对齐");
}

static void test_segments()
{
    Deserializer d(nullptr, 0, nullptr);
    // segment_count = (param - 0x12) / 0x0f; param=0x12+3*0x0f=0x3F → 3
    d.auxdata_deserialize_segments(0x3F);
    CHECK(d.crate_size_according_to_segments() == 0, "3 零跨度段 → 0 字节");
    d.auxdata_deserialize_segments(0x89); // ≤0x89 → 不建表
    d.auxdata_deserialize_segments(0x12 + 0x0f * 2);
    CHECK(d.crate_size_according_to_segments() == 0, "重建 2 段仍为 0");
}

static void test_bin_tag()
{
    CHECK(encode_bin_tag(0x5647) == ((0x5647 & 0xFFFF) | (0x5647u << 16)) ^ 0xFFFF, "encode 公式");
    // 格式契约: tag 为 16 位 (编码时打包进双半字, 任一半可复原); 0xFFFFFFFF 非法
    for (uint32_t t : {0x5647u, 0xEF4Du, 0xD352u, 0x5350u, 0u, 0xFFFFu, 0x4F50u}) {
        CHECK(decode_bin_tag(encode_bin_tag(t)) == t, "tag 往返");
    }
}

int main()
{
    test_static_layout();
    test_deserializer_base_state();
    test_fread_and_helpers();
    test_deserialize_str();
    test_segments();
    test_bin_tag();

    std::printf("test_m33_serializer: %d pass, %d fail\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
