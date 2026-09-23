// test_wtop_emit: tagged .bin → WTOP blob 转换(阶段 8, M1 host 闭环验收)
//
// 流程: 程序化建 conv_add 图 → prepare → serialize(.bin)→ 调用 wtop_emit
// 二进制 → wt_parse 校验 blob → 断言 op/slot 契约 → 负例(截断/坏 magic/
// 坏 arity)返回期望错误码。
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ir/scalar_params.hpp"
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
    od_nchw.dtype = static_cast<uint32_t>(DType::Float16);  // 对齐真实管线: 输入 f16
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
    wt_blob* wb = new wt_blob{};
    CHECK(wt_parse(blob.data(), blob.size(), wb) == WT_OK, "wt_parse OK");
    CHECK(wb->n_slots == 3, "3 slots (input/W/B)");
    CHECK(wb->n_ops == 5, "5 ops (T/IM2COL/CONV/ADD/T)");
    CHECK(wb->slots[0].len == 65536 && wb->slots[0].count == 32768, "slot0 = 输入 f16 64KB");
    CHECK(wb->slots[1].len == 18432 && wb->slots[1].count == 9216, "slot1 = W f16 18432B");
    CHECK(wb->slots[2].len == 64 && wb->slots[2].count == 32, "slot2 = B f16 64B");
    {
        /* M3c: Transpose 统一走 OP_TRANSPOSE_GEN_F16(形状参数化; 旧 4D
         * NCHW 契约对 transformer 张量全错) */
        const uint16_t expect_op[5] = {OP_TRANSPOSE_GEN_F16, OP_IM2COL, OP_CONV2D_F16,
                                       OP_ADD_F16, OP_TRANSPOSE_GEN_F16};
        bool seq = true;
        for (uint32_t i = 0; i < 5; i++) seq &= (wb->ops[i].opcode == expect_op[i]);
        CHECK(seq, "opcode 序列 [TG, IM2COL, CONV, ADD, TG]");
        // GEN 参数: [src, out, rank, d0..3, perm]
        CHECK((wb->ops[0].args[0] & 0x8000) != 0, "首 Transpose src = 0x8000|slot0 (输入注入)");
        CHECK(wb->ops[0].args[2] == 4, "首 Transpose rank=4");
        CHECK(wb->ops[0].args[7] == (0u | (2u << 8) | (3u << 16) | (1u << 24)),
              "首 Transpose perm = [0,2,3,1] 打包");
        // CONV: M=32*32 K=288 N=32
        CHECK(wb->ops[2].args[4] == 1024 && wb->ops[2].args[5] == 288 && wb->ops[2].args[6] == 32,
              "CONV M=1024 K=288 N=32");
        CHECK(wb->ops[2].args[1] == 1 && wb->ops[2].args[2] == 2, "CONV w_slot=1 bias_slot=2");
        // ADD: n_elem = 32768
        CHECK(wb->ops[3].args[3] == 32768, "ADD n_elem=32768");
    }

    // 5. 负例
    {
        std::vector<uint8_t> bad;
        // 截断
        bad.assign(blob.begin(), blob.begin() + 8);
        CHECK(wt_parse(bad.data(), bad.size(), wb) == WT_ERR_SHORT, "neg: 截断 -> SHORT");
        // 坏 magic
        bad = blob;
        bad[0] = 'X';
        CHECK(wt_parse(bad.data(), bad.size(), wb) == WT_ERR_MAGIC, "neg: 坏 magic -> MAGIC");
        // 坏 arity: 把首 op 的 n_args 改掉
        bad = blob;
        {
            wt_blob* wb2 = new wt_blob{};
            wt_parse(blob.data(), blob.size(), wb2);
            uint32_t n_slots = wb2->n_slots;
            size_t op0 = 16 + (size_t)n_slots * 16;
            uint16_t wrong = 0;
            std::memcpy(bad.data() + op0 + 2, &wrong, 2);
        }
        CHECK(wt_parse(bad.data(), bad.size(), wb) == WT_ERR_ARITY, "neg: 坏 arity -> ARITY");
    }

    /* 6. operation 映射契约(2026-09-22 静默兜底治理):
     * EQUAL(binary op=3) → OP_BINARY_F16 sub=4;
     * SOFTPLUS(neuron op=7) → OP_UNARY_F16 sub=13;
     * 未知 operation → emit 硬错误非零(不再静默 ADD/0xFFFFFFFF 进 blob)。
     * 建图模式同 build_conv_add: Input → 单 op → Output。 */
    {
        auto scalar_op = [](double v) {
            ScalarParam p; p.name = "operation"; p.is_numeric = true; p.value_num = v;
            return pack_scalar_params(std::vector<ScalarParam>{p});
        };
        auto run_case = [&](const char* tag, const char* op_name, double oper,
                            uint16_t expect_opcode, uint32_t expect_sub, bool expect_fail) {
            GraphPrepare g;
            auto od = make_od4(1, 1, 1, 32);
            od.dtype = static_cast<uint32_t>(DType::Float16);
            g.append_node("Input", 1, nullptr, 0, &od, 1, nullptr);
            auto sp = scalar_op(oper);
            InputDef oin[1] = {{3, 0}};
            if (std::strcmp(op_name, "Eltwise_Binary") == 0) {
                auto cw = make_od4(1, 1, 1, 32);  // const f32 权重域(同 conv_add 惯例)
                std::vector<float> c0(32, 0.0f);
                g.append_const_node(2, cw, reinterpret_cast<const uint8_t*>(c0.data()), c0.size() * 4);
                g.get_op_at(2)->name_tag = string_tag_t::map_str("c0");
                InputDef ein[2] = {{1, 0}, {2, 0}};
                g.append_node(op_name, 3, ein, 2, &od, 1, sp.data(), sp.size());
            } else {
                InputDef uin[1] = {{1, 0}};
                g.append_node(op_name, 3, uin, 1, &od, 1, sp.data(), sp.size());
            }
            g.append_node("Output", 4, oin, 1, nullptr, 0, nullptr);
            std::string pre = std::string("  [") + tag + "] ";
            if (g.prepare(env) != GraphStatus::Success) {
                CHECK(false, (pre + "prepare").c_str());
                return;
            }
            std::vector<uint8_t> b2(1u << 16, 0);
            size_t b2_size = 0;
            if (!g.serialize(b2.data(), b2.size(), b2_size) || b2_size == 0) {
                CHECK(false, (pre + "serialize").c_str());
                return;
            }
            const std::string bp = dir + tag + ".bin";
            const std::string ip = dir + tag + ".f16.raw";
            const std::string op2 = dir + tag + ".wtop";
            const std::string mp = dir + tag + ".manifest.json";
            write_file(bp, b2.data(), b2_size);
            std::vector<uint8_t> in16(64, 0);
            write_file(ip, in16.data(), in16.size());
            std::string c2 = std::string(WTOP_EMIT_PATH) + " --bin " + bp +
                             " --input-f16 " + ip + " --out " + op2 +
                             " --manifest " + mp + " 2>/dev/null";
            int rc2 = std::system(c2.c_str());
            if (expect_fail) {
                CHECK(rc2 != 0, (pre + "未知 operation -> emit 硬错误(非零)").c_str());
                return;
            }
            if (rc2 != 0) {
                // 诊断: 去重定向重跑一次, 让 emit 的报错直接可见
                std::system((std::string(WTOP_EMIT_PATH) + " --bin " + bp +
                             " --input-f16 " + ip + " --out " + op2 +
                             " --manifest " + mp).c_str());
                CHECK(false, (pre + "emit exit 0").c_str());
                return;
            }
            std::vector<uint8_t> bl2;
            if (!load_file(op2, bl2)) { CHECK(false, (pre + "read blob").c_str()); return; }
            wt_blob* w2 = new wt_blob{};
            if (wt_parse(bl2.data(), bl2.size(), w2) != WT_OK) { CHECK(false, (pre + "wt_parse").c_str()); delete w2; return; }
            CHECK(w2->n_ops == 1 && w2->ops[0].opcode == expect_opcode,
                  (pre + "单 op opcode 契约").c_str());
            if (expect_opcode == OP_BINARY_F16)
                CHECK(w2->ops[0].args[4] == expect_sub, (pre + "BINARY subtype 契约").c_str());
            else
                CHECK(w2->ops[0].args[3] == expect_sub, (pre + "UNARY subtype 契约").c_str());
            delete w2;
        };
        run_case("eq", "Eltwise_Binary", 3.0, OP_BINARY_F16, 4, false);
        run_case("softplus", "ElementWiseNeuron", 7.0, OP_UNARY_F16, 13, false);
        run_case("badbin", "Eltwise_Binary", 99.0, 0, 0, true);
        run_case("badneu", "ElementWiseNeuron", 99.0, 0, 0, true);
    }

    /* 7. P5 OP_DMA 解析契约 (OP_DMA=33, arity 8, GAP_CLOSURE C):
     * 手组最小 blob (1 slot + 1 op + weight 区), 断言:
     *   正例: OP_DMA arity 8 通过; src/dst 带 0xC000|引擎面编码豁免槽界;
     *   负例: 坏 arity -> ARITY; 0x8000|越界槽 -> BAD_REF;
     *         非 DMA op 携带 0xC000 -> BAD_REF (豁免仅限 OP_DMA)。
     * 另: --dma-runlist 作用于无溢出计划的 conv_add bin, 输出必须与默认
     *     形态逐字节一致 (旗标不扰动默认发射路径)。 */
    {
        auto build_blob = [](uint16_t opcode, const std::vector<uint32_t>& args) {
            std::vector<uint8_t> b;
            auto w16 = [&b](uint16_t v) { b.push_back((uint8_t)(v & 0xff)); b.push_back((uint8_t)(v >> 8)); };
            auto w32 = [&b](uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((uint8_t)(v >> (8 * i))); };
            b.insert(b.end(), {'W', 'T', 'O', 'P'});
            w16(WT_BLOB_VER); w16(WT_ENDIAN_CHK);
            w32(1); w32(1);                       // 1 slot, 1 op
            w32(128); w32(1); w32(0); w32(0);     // slot0: len=128 off=0 addr=0
            w16(opcode); w16((uint16_t)args.size());
            for (uint32_t a : args) w32(a);
            while (b.size() % 128) b.push_back(0);
            b.resize(b.size() + 128, 0);          // weight 区
            return b;
        };
        wt_blob* w3 = new wt_blob{};
        /* 正例: OP_DMA(0x8000|slot0 → 0x4000|temp9, 128B, fence CPU→DMA→HMX) */
        std::vector<uint8_t> dma = build_blob(OP_DMA,
            {0x8000u | 0u, WT_REF_VTCM_FLAG | 9u, 128u, 0u, 0u, 1u, 0u /*FC_CPU*/, 2u /*FC_HMX*/});
        CHECK(wt_parse(dma.data(), dma.size(), w3) == WT_OK, "P5: OP_DMA arity 8 wt_parse OK");
        CHECK(w3->ops[0].opcode == 33 && w3->ops[0].n_args == 8, "P5: OP_DMA=33 arity=8 契约");
        /* 正例: src/dst 0xC000|引擎面 豁免槽界 */
        std::vector<uint8_t> eng = build_blob(OP_DMA,
            {WT_REF_ENG_FLAG | WT_ENG_OUT, WT_REF_ENG_FLAG | WT_ENG_ACT, 128u, 0u, 0u, 1u, 2u, 0u});
        CHECK(wt_parse(eng.data(), eng.size(), w3) == WT_OK, "P5: OP_DMA 0xC000 引擎面豁免");
        /* 负例: 坏 arity */
        {
            std::vector<uint8_t> bad = build_blob(OP_DMA,
                {0x8000u | 0u, WT_REF_VTCM_FLAG | 9u, 128u, 0u, 0u, 1u, 0u});
            CHECK(wt_parse(bad.data(), bad.size(), w3) == WT_ERR_ARITY, "P5 neg: OP_DMA arity 7 -> ARITY");
        }
        /* 负例: 0x8000|越界槽 */
        {
            std::vector<uint8_t> bad = build_blob(OP_DMA,
                {0x8000u | 999u, WT_REF_VTCM_FLAG | 9u, 128u, 0u, 0u, 1u, 0u, 2u});
            CHECK(wt_parse(bad.data(), bad.size(), w3) == WT_ERR_BAD_REF, "P5 neg: OP_DMA src 越界槽 -> BAD_REF");
        }
        /* 负例: 非 DMA op 携带 0xC000 (豁免仅限 OP_DMA) */
        {
            std::vector<uint8_t> bad = build_blob(OP_PIN, {WT_REF_ENG_FLAG | WT_ENG_OUT});
            CHECK(wt_parse(bad.data(), bad.size(), w3) == WT_ERR_BAD_REF,
                  "P5 neg: PIN 带 0xC000 -> BAD_REF (豁免仅 OP_DMA)");
        }
        delete w3;
        /* --dma-runlist 不扰动默认形态: conv_add (无溢出计划) 两次发射逐字节一致 */
        {
            const std::string blob_dma = dir + "blob_dmaflag.wtop";
            std::string c3 = std::string(WTOP_EMIT_PATH) + " --bin " + bin_path +
                             " --input-f16 " + in_path + " --out " + blob_dma +
                             " --dma-runlist 2>/dev/null";
            CHECK(std::system(c3.c_str()) == 0, "P5: wtop_emit --dma-runlist exit 0");
            std::vector<uint8_t> bdm;
            CHECK(load_file(blob_dma, bdm), "P5: read --dma-runlist blob");
            CHECK(bdm == blob, "P5: 无溢出计划下 --dma-runlist 输出 == 默认形态 (逐字节)");
        }
    }

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
