// test_conv_tiling: 单算子 conv 空间分块(阶段 6)
//
// 1. 公式级: compute_conv_tiles 的 halo 推导/边界钳位/C-split
//    (3×3 s1 same / 5×5 s1 pad2 / stride-2 / C-split 各几何)
// 2. 集成: conv_add 图 tiling 2×2 → prepare → serialize → extra_info
//    tiling 段 num_tiles=4, 逐 tile 描述与 compute_conv_tiles 一致;
//    round-trip re-serialize 字节全同
// 3. 数值等价: 按 tile 描述执行(halo 切片 + 全图坐标 pad 处理)与全图
//    conv 参考 byte-exact(同累加序)
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/tiling/conv_tiling.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static OutputDef make_od4(uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3) {
    OutputDef od{};
    od.rank = 4;
    od.dims[0] = d0; od.dims[1] = d1; od.dims[2] = d2; od.dims[3] = d3;
    od.element_size = 4;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    return od;
}

struct ConvExtraInfoFixed {
    uint32_t sh, sw, ph_begin, ph_end, pw_begin, pw_end, dh, dw, group, kh, kw;
    uint64_t weight_src, bias_src;
};

static void formula_tests() {
    std::printf("[1] formula-level tile geometry\n");

    // 3×3 s1 same pad1, 32×32 → tile 16×16 = 2×2 tiles
    {
        auto ts = compute_conv_tiles(32, 32, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 16, 16, 0);
        CHECK(ts.size() == 4, "3x3 s1 same 16x16 -> 4 tiles");
        CHECK(ts[0].out_y0 == 0 && ts[0].out_h == 16 && ts[0].in_y0 == 0 && ts[0].in_h == 17,
              "tile(0,0): out[0..16), in rows [0,17) (18 行: 16+3-1 钳去 pad)");
        // 公式: 无钳位时 in_h = th + kh - 1 = 18; 上边界钳位砍掉 pad 区 1 行 → 17
        CHECK(ts[1].out_x0 == 16 && ts[1].in_x0 == 15 && ts[1].in_w == 17,
              "tile(0,1): in cols [15,32) = 17 (左 halo 1)");
        CHECK(ts[3].out_y0 == 16 && ts[3].in_y0 == 15 && ts[3].in_h == 17,
              "tile(1,1): in rows [15,32) = 17 (右下角)");
        CHECK(ts[0].co_n == 32 && ts[0].co0 == 0, "无 C-split: co_n == 32");
    }

    // 5×5 s1 pad2: 顶行 tile 的输入行 = [0, 18)(15+5-2), 中部 tile 才有完整 halo
    {
        auto ts = compute_conv_tiles(32, 32, 3, 32, 32, 8, 5, 5, 1, 1, 2, 2, 16, 32, 0);
        CHECK(ts.size() == 2, "5x5 s1 pad2 tile_h=16 -> 2 tiles");
        CHECK(ts[0].in_h == 18 && ts[0].in_y0 == 0, "top tile: in rows [0,18)");
        CHECK(ts[1].in_y0 == 14 && ts[1].in_h == 18, "bottom tile: in rows [14,32) (halo 2)");
    }

    // stride 2, 64 输入 pad0, kh3: out=31; tile 16 → 行数 = (th-1)*sh + kh
    {
        auto ts = compute_conv_tiles(64, 64, 3, 31, 31, 8, 3, 3, 2, 2, 0, 0, 16, 31, 0);
        CHECK(ts.size() == 2, "stride2 31x31 tile_h=16 -> 2 tiles");
        CHECK(ts[0].in_h == 33 && ts[1].in_h == 31,
              "tile rows: [0,33)=33 与 [32,63)=31 (公式 (th-1)*sh+kh)");
    }

    // C-split: co_per_tile=16 → 2 通道段 × 空间 tile
    {
        auto ts = compute_conv_tiles(32, 32, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 16, 16, 16);
        CHECK(ts.size() == 8, "C-split 16 + 2x2 空间 -> 8 tiles");
        CHECK(ts[0].co0 == 0 && ts[0].co_n == 16 && ts[4].co0 == 16 && ts[4].co_n == 16,
              "通道段: [0,16) 与 [16,32)");
    }

    // 全图(不切): tile_h=0 → 单 tile 整图
    {
        auto ts = compute_conv_tiles(32, 32, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 0, 0, 0);
        CHECK(ts.size() == 1 && ts[0].out_h == 32 && ts[0].in_h == 32,
              "tile_h=0 -> 单 tile 整图");
    }
}

// 按 tile 描述执行(halo 切片 + 全图坐标 pad 处理), 输出写回全图
static void conv_tiled_ref(const float* x, const float* w, const float* b,
                           float* out, const std::vector<ConvTileDesc>& tiles) {
    const size_t C = 32;
    for (const auto& t : tiles) {
        for (uint32_t oy = t.out_y0; oy < t.out_y0 + t.out_h; oy++)
            for (uint32_t ox = t.out_x0; ox < t.out_x0 + t.out_w; ox++)
                for (uint32_t co = t.co0; co < t.co0 + t.co_n; co++) {
                    float acc = b[co];
                    for (uint32_t kh = 0; kh < t.kh; kh++)
                        for (uint32_t kw = 0; kw < t.kw; kw++) {
                            // 全图坐标; 越界 = pad 0
                            long gy = (long)oy * t.sh + kh - t.ph_begin;
                            long gx = (long)ox * t.sw + kw - t.pw_begin;
                            if (gy < 0 || gy >= (long)C || gx < 0 || gx >= (long)C) continue;
                            for (uint32_t ci = 0; ci < t.ci; ci++) {
                                acc += x[(gy * C + gx) * C + ci]
                                     * w[((kh * t.kw + kw) * t.ci + ci) * t.co + co];
                            }
                        }
                    out[(oy * C + ox) * C + co] = acc;
                }
    }
}

static void conv_full_ref(const float* x, const float* w, const float* b, float* out) {
    std::vector<ConvTileDesc> one = compute_conv_tiles(
        32, 32, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 0, 0, 0);
    conv_tiled_ref(x, w, b, out, one);
}

int main() {
    std::printf("=== conv single-op tiling test ===\n\n");

    formula_tests();

    // 数值等价: 随机输入, 2×2 tile vs 全图 byte-exact
    {
        std::vector<float> x(32 * 32 * 32), w(3 * 3 * 32 * 32), b(32);
        uint32_t seed = 42;
        for (auto& v : x) { seed = seed * 1664525u + 1013904223u; v = (float)((seed >> 8) % 2001) / 1000.0f - 1.0f; }
        for (auto& v : w) { seed = seed * 1664525u + 1013904223u; v = (float)((seed >> 8) % 2001) / 1000.0f - 1.0f; }
        for (auto& v : b) { seed = seed * 1664525u + 1013904223u; v = (float)((seed >> 8) % 2001) / 1000.0f - 1.0f; }

        std::vector<float> full(32 * 32 * 32), tiled(32 * 32 * 32);
        conv_full_ref(x.data(), w.data(), b.data(), full.data());
        auto ts = compute_conv_tiles(32, 32, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 16, 16, 0);
        conv_tiled_ref(x.data(), w.data(), b.data(), tiled.data(), ts);
        CHECK(std::memcmp(full.data(), tiled.data(), full.size() * 4) == 0,
              "2x2 tiled execution == full conv (byte-exact)");
    }

    // 集成: 序列化 tiling 段 + round-trip
    {
        std::printf("[2] serialized tiling section\n");
        register_all_ops();

        GraphPrepare gp;
        auto od_nchw = make_od4(1, 32, 32, 32);
        gp.append_node("Input", 1, nullptr, 0, &od_nchw, 1, nullptr);

        const int32_t perm_in[4] = {0, 2, 3, 1}, perm_out[4] = {0, 3, 1, 2};
        OutputDef perm_od{};
        perm_od.rank = 1; perm_od.dims[0] = 4; perm_od.element_size = 4;
        perm_od.dtype = static_cast<uint32_t>(DType::Int32);
        gp.append_const_node(2, perm_od, reinterpret_cast<const uint8_t*>(perm_in), sizeof(perm_in));
        InputDef t1in[2] = {{1, 0}, {2, 0}};
        gp.append_node("Transpose", 3, t1in, 2, &od_nchw, 1, nullptr);

        auto w_od = make_od4(3, 3, 32, 32);
        std::vector<float> Wq(32 * 32 * 3 * 3, 0.5f);
        gp.append_const_node(4, w_od,
                             reinterpret_cast<const uint8_t*>(Wq.data()), Wq.size() * 4);
        OutputDef b_od{};
        b_od.rank = 1; b_od.dims[0] = 32; b_od.element_size = 4;
        b_od.dtype = static_cast<uint32_t>(DType::Float32);
        std::vector<float> Bv(32, 0.1f);
        gp.append_const_node(5, b_od, reinterpret_cast<const uint8_t*>(Bv.data()), Bv.size() * 4);

        const uint32_t stride_v[2] = {1, 1}, pad_v[4] = {1, 1, 1, 1}, dil_v[2] = {1, 1};
        OutputDef p2_od{}; p2_od.rank = 1; p2_od.dims[0] = 2; p2_od.element_size = 4;
        p2_od.dtype = static_cast<uint32_t>(DType::Int32);
        gp.append_const_node(11, p2_od, reinterpret_cast<const uint8_t*>(stride_v), 8);
        gp.get_op_at(11)->name_tag = string_tag_t::map_str("conv1_stride");
        OutputDef p4_od{}; p4_od.rank = 1; p4_od.dims[0] = 4; p4_od.element_size = 4;
        p4_od.dtype = static_cast<uint32_t>(DType::Int32);
        gp.append_const_node(12, p4_od, reinterpret_cast<const uint8_t*>(pad_v), 16);
        gp.get_op_at(12)->name_tag = string_tag_t::map_str("conv1_pad_amount");
        gp.append_const_node(13, p2_od, reinterpret_cast<const uint8_t*>(dil_v), 8);
        gp.get_op_at(13)->name_tag = string_tag_t::map_str("conv1_dilation");

        InputDef cin[6] = {{3, 0}, {4, 0}, {5, 0}, {11, 0}, {12, 0}, {13, 0}};
        gp.append_node("Conv2d", 6, cin, 6, &od_nchw, 1, nullptr);
        InputDef ain[2] = {{6, 0}, {3, 0}};
        gp.append_node("Eltwise_Binary", 7, ain, 2, &od_nchw, 1, nullptr);
        gp.append_const_node(8, perm_od, reinterpret_cast<const uint8_t*>(perm_out), 16);
        InputDef t2in[2] = {{7, 0}, {8, 0}};
        gp.append_node("Transpose", 9, t2in, 2, &od_nchw, 1, nullptr);
        InputDef oin[1] = {{9, 0}};
        gp.append_node("Output", 10, oin, 1, nullptr, 0, nullptr);

        // tiling 2×2(空间)
        TilingConfig cfg;
        cfg.conv_height_tiling = 2;
        cfg.conv_width_tiling = 2;
        gp.set_tiling_config(cfg);

        HexagonNNEnv env;
        env.set_soc_type(75);
        env.set_num_nsps(1);
        CHECK(gp.prepare(env) == GraphStatus::Success, "prepare (tiling 2x2)");

        std::vector<uint8_t> buf(1u << 18, 0);
        size_t out_size = 0;
        CHECK(gp.serialize(buf.data(), buf.size(), out_size) && out_size > 0, "serialize");
        buf.resize(out_size);

        GraphPrepare gp2;
        CHECK(gp2.deserialize(buf.data(), buf.size()), "deserialize");
        const OpDef* conv = gp2.get_op_at(6);
        CHECK(conv != nullptr, "Conv2d in round-trip graph");

        // extra 布局: 60B fixed + [4 u32 hdr][4×19 u32 descs]
        const size_t exp_size = sizeof(ConvExtraInfoFixed) + 16 + 4 * 19 * 4;
        CHECK(conv->serialized_extra.size() == exp_size, "tiling extra 尺寸 (60+16+304=380)");
        bool sec_ok = false;
        if (conv->serialized_extra.size() == exp_size) {
            uint32_t hdr[4];
            std::memcpy(hdr, conv->serialized_extra.data() + sizeof(ConvExtraInfoFixed), 16);
            sec_ok = (hdr[0] == 16 && hdr[1] == 16 && hdr[2] == 32 && hdr[3] == 4);
            std::vector<ConvTileDesc> descs(4);
            std::memcpy(descs.data(), conv->serialized_extra.data() + sizeof(ConvExtraInfoFixed) + 16,
                        4 * sizeof(ConvTileDesc));
            // 与 compute_conv_tiles 直接对比
            auto expect = compute_conv_tiles(32, 32, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 16, 16, 0);
            for (size_t i = 0; i < 4; i++)
                sec_ok &= (std::memcmp(&descs[i], &expect[i], sizeof(ConvTileDesc)) == 0);
        }
        CHECK(sec_ok, "tiling 段: hdr(16,16,32,4) + 4 tile 描述 == compute_conv_tiles");

        std::vector<uint8_t> buf2(1u << 18, 0);
        size_t out_size2 = 0;
        CHECK(gp2.serialize(buf2.data(), buf2.size(), out_size2) && out_size2 == out_size &&
              std::memcmp(buf.data(), buf2.data(), out_size) == 0,
              "re-serialize 字节全同 (tiling 段随 extra round-trip)");
    }

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
