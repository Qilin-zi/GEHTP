// test_spill_fill_serialize: spill/fill 接入序列化(阶段 7)
//
// 验收:
//   1. 默认预算(8MB): conv_add 全驻留, spill_fill_recs() 空, .bin 无 SF 记录
//   2. 强制小预算(1KB): 分配器溢出 → .bin 含 0x4453(spill 计数)与 0x5346
//      每张量 DMA 记录; ddr_offset 非占位、128B 对齐、按 op_id 升序、互异;
//      size == 张量字节数
//   3. round-trip: deserialize 恢复 spill_fill_recs() 一致; re-serialize 字节全同
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ops/ops.hpp"
#include "hnnx/serialize/serializer.hpp"  // encode_bin_tag

#include <cstdio>
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

static OutputDef make_od4(uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3) {
    OutputDef od{};
    od.rank = 4;
    od.dims[0] = d0; od.dims[1] = d1; od.dims[2] = d2; od.dims[3] = d3;
    od.element_size = 4;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    return od;
}

static void build_conv_add(GraphPrepare& gp) {
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

    InputDef cin[3] = {{3, 0}, {4, 0}, {5, 0}};
    gp.append_node("Conv2d", 6, cin, 3, &od_nchw, 1, nullptr);
    InputDef ain[2] = {{6, 0}, {3, 0}};
    gp.append_node("Eltwise_Binary", 7, ain, 2, &od_nchw, 1, nullptr);
    gp.append_const_node(8, perm_od, reinterpret_cast<const uint8_t*>(perm_out), sizeof(perm_out));
    InputDef t2in[2] = {{7, 0}, {8, 0}};
    gp.append_node("Transpose", 9, t2in, 2, &od_nchw, 1, nullptr);
    InputDef oin[1] = {{9, 0}};
    gp.append_node("Output", 10, oin, 1, nullptr, 0, nullptr);
}

static size_t count_tag(const std::vector<uint8_t>& buf, uint32_t enc) {
    size_t n = 0;
    for (size_t i = 0; i + 4 <= buf.size(); i++) {
        uint32_t v;
        std::memcpy(&v, buf.data() + i, 4);
        if (v == enc) n++;
    }
    return n;
}

// 解析 0x5346 记录: [enc][wc][?][u64 op_id][u32 block][u64 ddr][u64 size]
static std::vector<GraphPrepare::SpillFillRec> parse_sf(const std::vector<uint8_t>& buf) {
    std::vector<GraphPrepare::SpillFillRec> recs;
    uint32_t enc = encode_bin_tag(0x5346);
    for (size_t i = 0; i + 12 + 28 <= buf.size(); i++) {
        uint32_t v;
        std::memcpy(&v, buf.data() + i, 4);
        if (v != enc) continue;
        GraphPrepare::SpillFillRec r{};
        std::memcpy(&r.op_id, buf.data() + i + 12, 8);
        std::memcpy(&r.block_id, buf.data() + i + 20, 4);
        std::memcpy(&r.ddr_offset, buf.data() + i + 24, 8);
        std::memcpy(&r.size, buf.data() + i + 32, 8);
        recs.push_back(r);
    }
    return recs;
}

int main() {
    std::printf("=== spill/fill serialization test ===\n\n");
    register_all_ops();

    // 1. 默认预算: 无溢出
    {
        GraphPrepare gp;
        build_conv_add(gp);
        HexagonNNEnv env;
        env.set_soc_type(75);
        env.set_num_nsps(1);
        CHECK(gp.prepare(env) == GraphStatus::Success, "prepare (default 8MB)");
        CHECK(gp.spill_fill_recs().empty(), "默认预算: 无 spill(全驻留)");

        std::vector<uint8_t> buf(1u << 18, 0);
        size_t out_size = 0;
        CHECK(gp.serialize(buf.data(), buf.size(), out_size) && out_size > 0, "serialize");
        buf.resize(out_size);
        CHECK(count_tag(buf, encode_bin_tag(0x5346)) == 0, "默认预算: .bin 无 SF 记录");
    }

    // 2. 强制小预算: 溢出 + 记录
    {
        GraphPrepare gp;
        build_conv_add(gp);
        gp.set_vtcm_budget(1024);  // 1KB ≪ 128KB 张量
        HexagonNNEnv env;
        env.set_soc_type(75);
        env.set_num_nsps(1);
        CHECK(gp.prepare(env) == GraphStatus::Success, "prepare (1KB budget)");

        std::vector<uint8_t> buf(1u << 18, 0);
        size_t out_size = 0;
        CHECK(gp.serialize(buf.data(), buf.size(), out_size) && out_size > 0, "serialize");
        buf.resize(out_size);

        CHECK(count_tag(buf, encode_bin_tag(0x4453)) >= 1, ".bin 含 0x4453 spill 配置记录");
        auto recs = parse_sf(buf);
        std::printf("  SF 记录数: %zu\n", recs.size());
        CHECK(!recs.empty(), ".bin 含 0x5346 每张量 DMA 记录");

        // ddr_offset: 非占位、128B 对齐、升序、互异; size == 128KB(32*32*32*4)
        bool ok = true;
        uint64_t prev = 0;
        for (size_t i = 0; i < recs.size(); i++) {
            const auto& r = recs[i];
            ok &= (r.ddr_offset >= 0x1000 && (r.ddr_offset & 127) == 0);
            ok &= (r.size == 32 * 32 * 32 * 4);
            if (i > 0) ok &= (r.ddr_offset > prev && r.op_id > recs[i - 1].op_id);
            prev = r.ddr_offset;
        }
        CHECK(ok, "SF 记录: ddr_offset 非占位/128B 对齐/升序/互异, size=128KB");

        // round-trip
        GraphPrepare gp2;
        CHECK(gp2.deserialize(buf.data(), buf.size()), "deserialize");
        const auto& rr = gp2.spill_fill_recs();
        bool same = (rr.size() == recs.size());
        if (same)
            for (size_t i = 0; i < recs.size(); i++)
                same &= (rr[i].op_id == recs[i].op_id && rr[i].block_id == recs[i].block_id &&
                         rr[i].ddr_offset == recs[i].ddr_offset && rr[i].size == recs[i].size);
        CHECK(same, "round-trip: spill_fill_recs 恢复一致");

        std::vector<uint8_t> buf2(1u << 18, 0);
        size_t out_size2 = 0;
        CHECK(gp2.serialize(buf2.data(), buf2.size(), out_size2) && out_size2 == out_size &&
              std::memcmp(buf.data(), buf2.data(), out_size) == 0,
              "re-serialize 字节全同 (SF 记录随往返)");
    }

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
