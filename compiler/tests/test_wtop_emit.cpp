// test_wtop_emit: tagged .bin → WTOP blob 转换(阶段 8, M1 host 闭环验收)
//
// 流程: 程序化建 conv_add 图 → prepare → serialize(.bin)→ 调用 wtop_emit
// 二进制 → wt_parse 校验 blob → 断言 op/slot 契约 → 负例(截断/坏 magic/
// 坏 arity)返回期望错误码。
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ops/ops.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "oplist_parse.h"

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
    gp.get_op_at(2)->name_tag = string_tag_t::map_str("X_perm");
    InputDef t1in[2] = {{1, 0}, {2, 0}};
    gp.append_node("Transpose", 3, t1in, 2, &od_nchw, 1, nullptr);

    auto w_od = make_od4(3, 3, 32, 32);
    std::vector<float> Wq(32 * 32 * 3 * 3, 0.5f);
    gp.append_const_node(4, w_od,
                         reinterpret_cast<const uint8_t*>(Wq.data()), Wq.size() * 4);
    gp.get_op_at(4)->name_tag = string_tag_t::map_str("W");
    OutputDef b_od{};
    b_od.rank = 1; b_od.dims[0] = 32; b_od.element_size = 4;
    b_od.dtype = static_cast<uint32_t>(DType::Float32);
    std::vector<float> Bv(32, 0.1f);
    gp.append_const_node(5, b_od, reinterpret_cast<const uint8_t*>(Bv.data()), Bv.size() * 4);
    gp.get_op_at(5)->name_tag = string_tag_t::map_str("B");

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
    gp.append_const_node(8, perm_od, reinterpret_cast<const uint8_t*>(perm_out), sizeof(perm_out));
    gp.get_op_at(8)->name_tag = string_tag_t::map_str("Z_perm");
    InputDef t2in[2] = {{7, 0}, {8, 0}};
    gp.append_node("Transpose", 9, t2in, 2, &od_nchw, 1, nullptr);
    InputDef oin[1] = {{9, 0}};
    gp.append_node("Output", 10, oin, 1, nullptr, 0, nullptr);
}

static bool write_file(const std::string& path, const void* data, size_t len) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data, 1, len, f) == len;
    std::fclose(f);
    return ok;
}

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

int main() {
    std::printf("=== wtop_emit test (M1 host closure) ===\n\n");
    register_all_ops();

    // 1. 建图 → prepare → serialize
    GraphPrepare gp;
    build_conv_add(gp);
    HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    CHECK(gp.prepare(env) == GraphStatus::Success, "prepare");

    std::vector<uint8_t> bin(1u << 18, 0);
    size_t bin_size = 0;
    CHECK(gp.serialize(bin.data(), bin.size(), bin_size) && bin_size > 0, "serialize .bin");
    bin.resize(bin_size);

    const std::string dir = "/tmp/wtop_emit_test/";
    std::system(("mkdir -p " + dir).c_str());
    const std::string bin_path = dir + "conv_add.bin";
    const std::string in_path = dir + "X.f16.raw";
    const std::string blob_path = dir + "blob.wtop";
    const std::string man_path = dir + "manifest.json";
    CHECK(write_file(bin_path, bin.data(), bin.size()), "write .bin");

    // 2. 输入 f16(全 0.25)
    std::vector<uint8_t> in_f16(32 * 32 * 32 * 2, 0);
    {
        uint16_t v = 0x3400;  // f16 0.25
        for (size_t i = 0; i + 2 <= in_f16.size(); i += 2)
            std::memcpy(in_f16.data() + i, &v, 2);
    }
    CHECK(write_file(in_path, in_f16.data(), in_f16.size()), "write input f16");

    // 3. 调用 wtop_emit
    std::string cmd = std::string(WTOP_EMIT_PATH) + " --bin " + bin_path +
                      " --input-f16 " + in_path + " --out " + blob_path +
                      " --manifest " + man_path + " 2>/dev/null";
    int rc = std::system(cmd.c_str());
    CHECK(rc == 0, "wtop_emit exit 0");

    // 4. wt_parse 校验 + 契约断言
    std::vector<uint8_t> blob;
    CHECK(load_file(blob_path, blob), "read blob");
    wt_blob wb{};
    CHECK(wt_parse(blob.data(), blob.size(), &wb) == WT_OK, "wt_parse OK");
    CHECK(wb.n_slots == 3, "3 slots (input/W/B)");
    CHECK(wb.n_ops == 5, "5 ops (T/IM2COL/CONV/ADD/T)");
    CHECK(wb.slots[0].len == 65536 && wb.slots[0].count == 32768, "slot0 = 输入 f16 64KB");
    CHECK(wb.slots[1].len == 18432 && wb.slots[1].count == 9216, "slot1 = W f16 18432B");
    CHECK(wb.slots[2].len == 64 && wb.slots[2].count == 32, "slot2 = B f16 64B");
    {
        const uint16_t expect_op[5] = {OP_TRANSPOSE_F16, OP_IM2COL, OP_CONV2D_F16,
                                       OP_ADD_F16, OP_TRANSPOSE_F16};
        bool seq = true;
        for (uint32_t i = 0; i < 5; i++) seq &= (wb.ops[i].opcode == expect_op[i]);
        CHECK(seq, "opcode 序列 [T, IM2COL, CONV, ADD, T]");
        // TRANSPOSE 参数: [src(0x8000|slot0), out=0, H,W,C, perm]
        CHECK((wb.ops[0].args[0] & 0x8000) != 0, "首 Transpose src = 0x8000|slot0 (输入注入)");
        CHECK(wb.ops[0].args[5] == (0u | (2u << 8) | (3u << 16) | (1u << 24)),
              "首 Transpose perm = [0,2,3,1] 打包");
        // CONV: M=32*32 K=288 N=32
        CHECK(wb.ops[2].args[4] == 1024 && wb.ops[2].args[5] == 288 && wb.ops[2].args[6] == 32,
              "CONV M=1024 K=288 N=32");
        CHECK(wb.ops[2].args[1] == 1 && wb.ops[2].args[2] == 2, "CONV w_slot=1 bias_slot=2");
        // ADD: n_elem = 32768
        CHECK(wb.ops[3].args[3] == 32768, "ADD n_elem=32768");
    }

    // 5. 负例
    {
        std::vector<uint8_t> bad;
        // 截断
        bad.assign(blob.begin(), blob.begin() + 8);
        CHECK(wt_parse(bad.data(), bad.size(), &wb) == WT_ERR_SHORT, "neg: 截断 -> SHORT");
        // 坏 magic
        bad = blob;
        bad[0] = 'X';
        CHECK(wt_parse(bad.data(), bad.size(), &wb) == WT_ERR_MAGIC, "neg: 坏 magic -> MAGIC");
        // 坏 arity: 把首 op 的 n_args 改掉
        bad = blob;
        {
            wt_blob wb2{};
            wt_parse(blob.data(), blob.size(), &wb2);
            uint32_t n_slots = wb2.n_slots;
            size_t op0 = 16 + (size_t)n_slots * 16;
            uint16_t wrong = 0;
            std::memcpy(bad.data() + op0 + 2, &wrong, 2);
        }
        CHECK(wt_parse(bad.data(), bad.size(), &wb) == WT_ERR_ARITY, "neg: 坏 arity -> ARITY");
    }

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
