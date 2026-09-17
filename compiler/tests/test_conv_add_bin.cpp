// test_conv_add_bin: const 池验证(P0) + serialize_opdef 参数 schema(P1) + round-trip(阶段 4)
//
// 图: 同 test_conv_add_host + conv 显式 stride/pad/dilation const 输入
//     (模拟 net.json 摄取的 tensor_param 注入, 名字含关键词)。
// P0 验收:
//   prepare 后 const_pool_ 含 W/B 字节; const_extents_ 记录齐全
// P1 验收:
//   serialize 输出含 0xCF56 池记录 / 0x4345 extent 表 / 0xFA0000FA 分隔 / 0xBEEFF00D 尾
//   deserialize round-trip: 图结构一致 + conv extra_info 字节一致
//   re-serialize 两次字节全同(确定性)
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/serialize/serializer.hpp"  // encode_bin_tag

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static bool load_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz > 0 ? (size_t)sz : 0);
    if (sz > 0 && std::fread(out.data(), 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static OutputDef make_od4(uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3) {
    OutputDef od{};
    od.rank = 4;
    od.dims[0] = d0; od.dims[1] = d1; od.dims[2] = d2; od.dims[3] = d3;
    od.element_size = 4;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    return od;
}

// 与 graph_prepare.cpp 的 ConvExtraInfo 同布局(设备侧同构解析契约)
struct ConvExtraInfo {
    uint32_t sh, sw, ph_begin, ph_end, pw_begin, pw_end, dh, dw, group, kh, kw;
    uint64_t weight_src, bias_src;
};

static bool find_word(const std::vector<uint8_t>& buf, uint32_t word, size_t& pos) {
    // 逐字节扫描: 记录载荷含 4 对齐 padding 后, 目标字可能落在非对齐偏移
    for (size_t i = 0; i + 4 <= buf.size(); i++) {
        uint32_t v;
        std::memcpy(&v, buf.data() + i, 4);
        if (v == word) { pos = i; return true; }
    }
    return false;
}

static bool find_bytes(const std::vector<uint8_t>& hay, const void* needle, size_t nlen) {
    const uint8_t* p = static_cast<const uint8_t*>(needle);
    return std::search(hay.begin(), hay.end(), p, p + nlen) != hay.end();
}

int main() {
    std::printf("=== conv_add serialize round-trip test ===\n\n");

    const std::string dir = std::string(HNNX_GEHTP_TEST_DATA_DIR) + "/conv_add/";
    std::vector<uint8_t> wb, bb;
    bool have = load_file(dir + "W.f32.raw", wb) && load_file(dir + "B.f32.raw", bb);
    if (!have) {
        std::printf("SKIP: conv_add 数据缺失 (先跑 test_models/conv_add/gen_all.sh)\n");
        return 0;
    }
    // W OIHW -> [kh,kw,ci,co](QNN IR 布局)
    const float* Wraw = reinterpret_cast<const float*>(wb.data());
    std::vector<float> Wq(32 * 32 * 3 * 3);
    for (size_t co = 0; co < 32; co++)
        for (size_t ci = 0; ci < 32; ci++)
            for (size_t kh = 0; kh < 3; kh++)
                for (size_t kw = 0; kw < 3; kw++)
                    Wq[((kh * 3 + kw) * 32 + ci) * 32 + co] =
                        Wraw[((co * 32 + ci) * 3 + kh) * 3 + kw];

    register_all_ops();

    GraphPrepare gp;
    auto od_nchw = make_od4(1, 32, 32, 32);
    gp.append_node("Input", 1, nullptr, 0, &od_nchw, 1, nullptr);

    const int32_t perm_in[4] = {0, 2, 3, 1};
    const int32_t perm_out[4] = {0, 3, 1, 2};
    OutputDef perm_od{};
    perm_od.rank = 1; perm_od.dims[0] = 4; perm_od.element_size = 4;
    perm_od.dtype = static_cast<uint32_t>(DType::Int32);
    gp.append_const_node(2, perm_od, reinterpret_cast<const uint8_t*>(perm_in), sizeof(perm_in));
    gp.get_op_at(2)->name_tag = string_tag_t::map_str("X_perm");

    InputDef t1in[2] = {{1, 0}, {2, 0}};
    gp.append_node("Transpose", 3, t1in, 2, &od_nchw, 1, nullptr);

    auto w_od = make_od4(3, 3, 32, 32);
    gp.append_const_node(4, w_od,
                         reinterpret_cast<const uint8_t*>(Wq.data()), Wq.size() * sizeof(float));
    gp.get_op_at(4)->name_tag = string_tag_t::map_str("W");

    OutputDef b_od{};
    b_od.rank = 1; b_od.dims[0] = 32; b_od.element_size = 4;
    b_od.dtype = static_cast<uint32_t>(DType::Float32);
    gp.append_const_node(5, b_od, bb.data(), bb.size());
    gp.get_op_at(5)->name_tag = string_tag_t::map_str("B");

    // stride/pad/dilation const(模拟 net.json 摄取注入, 名字含关键词)
    const uint32_t stride_v[2] = {1, 1};
    const uint32_t pad_v[4] = {1, 1, 1, 1};      // [ph_begin, ph_end, pw_begin, pw_end]
    const uint32_t dil_v[2] = {1, 1};
    OutputDef p2_od{}; p2_od.rank = 1; p2_od.dims[0] = 2; p2_od.element_size = 4;
    p2_od.dtype = static_cast<uint32_t>(DType::Int32);
    gp.append_const_node(11, p2_od, reinterpret_cast<const uint8_t*>(stride_v), sizeof(stride_v));
    gp.get_op_at(11)->name_tag = string_tag_t::map_str("conv1_stride");
    OutputDef p4_od{}; p4_od.rank = 1; p4_od.dims[0] = 4; p4_od.element_size = 4;
    p4_od.dtype = static_cast<uint32_t>(DType::Int32);
    gp.append_const_node(12, p4_od, reinterpret_cast<const uint8_t*>(pad_v), sizeof(pad_v));
    gp.get_op_at(12)->name_tag = string_tag_t::map_str("conv1_pad_amount");
    gp.append_const_node(13, p2_od, reinterpret_cast<const uint8_t*>(dil_v), sizeof(dil_v));
    gp.get_op_at(13)->name_tag = string_tag_t::map_str("conv1_dilation");

    InputDef cin[6] = {{3, 0}, {4, 0}, {5, 0}, {11, 0}, {12, 0}, {13, 0}};
    gp.append_node("Conv2d", 6, cin, 6, &od_nchw, 1, nullptr);

    InputDef ain[2] = {{6, 0}, {3, 0}};
    gp.append_node("Eltwise_Binary", 7, ain, 2, &od_nchw, 1, nullptr);

    gp.append_const_node(8, perm_od, reinterpret_cast<const uint8_t*>(perm_out), sizeof(perm_out));
    gp.get_op_at(8)->name_tag = string_tag_t::map_str("Z_perm");
    InputDef t2in[2] = {{7, 0}, {8, 0}};
    gp.append_node("Transpose", 9, t2in, 2, &od_nchw, 1, nullptr);

    InputDef oin[1] = {{9, 0}};
    gp.append_node("Output", 10, oin, 1, nullptr, 0, nullptr);

    HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    GraphStatus st = gp.prepare(env);
    CHECK(st == GraphStatus::Success, "prepare");

    // ---- P0: const 池 ----
    const auto& pool = gp.const_pool();
    CHECK(find_bytes(pool, Wq.data(), Wq.size() * sizeof(float)),
          "P0: const_pool 含 W 字节 ([kh,kw,ci,co])");
    CHECK(find_bytes(pool, bb.data(), bb.size()),
          "P0: const_pool 含 B 字节");
    bool w_extent = false, b_extent = false;
    for (const auto& e : gp.const_extents()) {
        if (e.size == Wq.size() * sizeof(float)) w_extent = true;
        if (e.size == bb.size()) b_extent = true;
    }
    CHECK(w_extent && b_extent, "P0: const_extents 覆盖 W/B");

    // ---- P1: serialize ----
    std::vector<uint8_t> buf(1u << 18, 0);
    size_t out_size = 0;
    bool ok = gp.serialize(buf.data(), buf.size(), out_size);
    CHECK(ok && out_size > 0 && out_size <= buf.size(), "serialize");
    buf.resize(out_size);

    size_t pos = 0;
    CHECK(find_word(buf, encode_bin_tag(0xCF56), pos), "P1: .bin 含 0xCF56 池记录");
    CHECK(find_word(buf, encode_bin_tag(0x4345), pos), "P1: .bin 含 0x4345 const extent 表");
    CHECK(find_word(buf, 0xFA0000FA, pos), "P1: .bin 含 0xFA0000FA 分隔");
    // 0xBEEFF00D 尾分隔; 其后至 out_size 为 serializer 尾部预留区
    // (preload 区 0x108B, 当前全零 —— 实测: 分隔符 pos 40700, out_size 40968)
    bool bee = find_word(buf, 0xBEEFF00D, pos);
    bool tail_zeros = true;
    for (size_t i = pos + 4; i < buf.size(); i++) tail_zeros &= (buf[i] == 0);
    CHECK(bee && pos + 4 <= buf.size() && tail_zeros,
          "P1: .bin 尾含 0xBEEFF00D + 零预留区(preload)");
    // 每个 op 一条 TAG_OP_RECORD
    size_t op_records = 0;
    for (size_t i = 0; i + 4 <= buf.size(); i += 4) {
        uint32_t v;
        std::memcpy(&v, buf.data() + i, 4);
        if (v == encode_bin_tag(0x4F50)) op_records++;
    }
    CHECK(op_records >= 4, "P1: TAG_OP_RECORD 条数 >= 4 (Transpose/Conv2d/Add/Transpose)");

    // ---- round-trip ----
    GraphPrepare gp2;
    bool rt = gp2.deserialize(buf.data(), buf.size());
    CHECK(rt, "deserialize");
    CHECK(gp2.op_count() == gp.op_count(), "round-trip: op 数一致");

    // conv 的 extra_info 值(sh=sw=1, pad=1, kh=kw=3)
    const OpDef* conv2 = gp2.get_op_at(6);
    CHECK(conv2 != nullptr, "round-trip: Conv2d 存在");
    // extra = 60B fixed + tiling 段(阶段6: 16B hdr + 1×76B 全图 tile, 默认配置)
    bool extra_ok = false;
    if (conv2 && conv2->serialized_extra.size() == sizeof(ConvExtraInfo) + 16 + 76) {
        ConvExtraInfo e;
        std::memcpy(&e, conv2->serialized_extra.data(), sizeof(ConvExtraInfo));
        extra_ok = (e.sh == 1 && e.sw == 1 && e.ph_begin == 1 && e.ph_end == 1 &&
                    e.pw_begin == 1 && e.pw_end == 1 && e.dh == 1 && e.dw == 1 &&
                    e.group == 1 && e.kh == 3 && e.kw == 3 &&
                    e.weight_src == 4 && e.bias_src == 5);
    }
    CHECK(extra_ok, "round-trip: ConvExtraInfo 值正确 (sh=1 kh=3 pad=1 w=4 b=5)");

    // const 池逐字节一致
    CHECK(gp2.const_pool() == pool, "round-trip: const 池逐字节一致");

    // re-serialize 两次字节全同(确定性)
    std::vector<uint8_t> buf2(1u << 18, 0);
    size_t out_size2 = 0;
    ok = gp2.serialize(buf2.data(), buf2.size(), out_size2);
    CHECK(ok && out_size2 == out_size, "re-serialize: 尺寸一致");
    bool bytes_same = (out_size2 == out_size) &&
                      (std::memcmp(buf.data(), buf2.data(), out_size) == 0);
    CHECK(bytes_same, "re-serialize: 两次字节全同");

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
