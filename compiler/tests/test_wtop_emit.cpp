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

/* 建 FullyConnected 图: Input[M,K](f16) @ W[N,K](const f32 池, 存 [N,K]) -> Output[M,N]。
 * 用 FC 而非 MatMul: 真实 0.8B 的超界 matmul (lm_head N248320 等) 全是 FC (flags=2
 * 非 batched), 而 prepare 会把 rank<4 张量补成 [1,1,..,..] → MatMul 被误判
 * batched(Bn=1), 分块条件 (非 batched) 永不命中。FC 的 ExtraFc 无 batched 字段,
 * 恒非 batched, 与真实发射一致。 */
static void build_matmul(GraphPrepare& gp, uint32_t M, uint32_t K, uint32_t N) {
    auto f32_od2 = [](uint64_t d0, uint64_t d1) {
        OutputDef od{};
        od.rank = 2; od.dims[0] = d0; od.dims[1] = d1;
        od.element_size = 4;
        od.dtype = static_cast<uint32_t>(DType::Float32);
        return od;
    };
    OutputDef w_od = f32_od2(N, K);   /* FC 权重存 [N,K] */
    std::vector<uint8_t> W(N * K * 4, 0);
    /* 确定性非零整数取值 (f16 精确表示), 供 T016 数值对拍; T010/T011 只查
     * opcode/形状不看值, 不受影响 */
    for (size_t i = 0; i < N * K; i++) {
        float v = (float)((int)(i % 37) - 18);
        std::memcpy(W.data() + i * 4, &v, 4);
    }
    gp.append_const_node(1, w_od, W.data(), W.size());
    gp.get_op_at(1)->name_tag = string_tag_t::map_str("W");

    OutputDef in_od{};
    in_od.rank = 2; in_od.dims[0] = M; in_od.dims[1] = K;
    in_od.element_size = 2;
    in_od.dtype = static_cast<uint32_t>(DType::Float16);
    gp.append_node("Input", 10, nullptr, 0, &in_od, 1, nullptr);

    InputDef mm_in[2] = {{10, 0}, {1, 0}};
    OutputDef mm_od{};
    mm_od.rank = 2; mm_od.dims[0] = M; mm_od.dims[1] = N;
    mm_od.element_size = 2;
    mm_od.dtype = static_cast<uint32_t>(DType::Float16);
    gp.append_node("FullyConnected", 20, mm_in, 2, &mm_od, 1, nullptr);

    InputDef oin[1] = {{20, 0}};
    gp.append_node("Output", 40, oin, 1, nullptr, 0, nullptr);
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

/* —— T016 host 合成数值验证: f16 转换 + FC 标量参考 ——
 * 直接复制 oplist_exec.c:341-385 的 f16↔f32 (IEEE 754 binary16, 与 exec
 * 标量路径同算法) 及 exec_matmul_f16 标量回落 FC(t1=1, 非 batched) 循环
 * (:1610-1619) 的 n0/N_full 版本。目的: 验证发射出的 (n0,N_chunk,N_full)
 * 分块把同一乘积流按列拆开重放后逐字节等于整算。 */
static float r_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            while (!(m & 0x400u)) { m <<= 1; e++; }
            bits = sign | ((uint32_t)(127 - 15 + 1 - e - 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
static uint16_t r_f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> (shift + 1);
        uint32_t rest = man & ((1u << shift) - 1u);
        if ((man >> shift) & 1u) half += (rest || (half & 1u)) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man >> 12) & 1u) half += ((man & 0xfffu) || (half & 1u)) ? 1u : 0u;
    return (uint16_t)half;
}
/* FC(t1=1) 标量: y[m*N_full + n0+n] = Σ_k a[m*K+k]·w[(n0+n)*K+k] (f32 累加) */
static void fc_ref(const uint16_t* a, const uint16_t* w, uint16_t* y,
                   uint32_t M, uint32_t K, uint32_t N_full, uint32_t n0, uint32_t N) {
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++)
                acc += r_f16_to_f32(a[(size_t)m * K + k]) * r_f16_to_f32(w[(size_t)(n0 + n) * K + k]);
            y[(size_t)m * N_full + (n0 + n)] = r_f32_to_f16(acc);
        }
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

    /* 8. US1 matmul 编译期 N 分块契约 (OP_MATMUL_F16_SPLIT=35, arity 9):
     * 线格式 [a_ref, w_s, out_t, M, K, N_full, n0, N_chunk, flags]。
     * 大 N 超 VTCM 界 → ceil(N/N_chunk) 个 SPLIT op, n0 递增覆盖 [0,N),
     * act/wgt/out 三引用跨 chunk 共享; 小 N 不超界 → 旧 OP_MATMUL_F16 (24) arity 7。
     * 预算口径(US1): 运行时 VTCM total = 16MB(设备 A2 query_VTCM 实锤 16777216),
     * need 公式与 exec_matmul_f16_hmx 一致 =
     * align2048(M*K*2) + align2048(N*K*2) + align2048(M*N*2) + 2048。 */
    {
        auto run_matmul_emit = [&](uint32_t M, uint32_t K, uint32_t N) {
            GraphPrepare g;
            build_matmul(g, M, K, N);
            if (g.prepare(env) != GraphStatus::Success) {
                CHECK(false, "matmul prepare");
                return (wt_blob*)nullptr;
            }
            std::vector<uint8_t> b2(1u << 27, 0);  /* 128MB: 超界 N(8192×1024) 的 f32 权重池 ~32MB */
            size_t b2_size = 0;
            if (!g.serialize(b2.data(), b2.size(), b2_size) || b2_size == 0) {
                CHECK(false, "matmul serialize");
                return (wt_blob*)nullptr;
            }
            char tag[64];
            std::snprintf(tag, sizeof(tag), "mm_%u_%u_%u", M, K, N);
            const std::string bp = dir + tag + ".bin";
            const std::string ip = dir + tag + ".f16.raw";
            const std::string op2 = dir + tag + ".wtop";
            const std::string mp = dir + tag + ".manifest.json";
            write_file(bp, b2.data(), b2_size);
            std::vector<uint8_t> in16((size_t)M * K * 2, 0);
            write_file(ip, in16.data(), in16.size());
            std::string c2 = std::string(WTOP_EMIT_PATH) + " --bin " + bp +
                             " --input-f16 " + ip + " --out " + op2 +
                             " --manifest " + mp + " 2>/dev/null";
            if (std::system(c2.c_str()) != 0) {
                std::system((std::string(WTOP_EMIT_PATH) + " --bin " + bp +
                             " --input-f16 " + ip + " --out " + op2 +
                             " --manifest " + mp).c_str());
                CHECK(false, "matmul emit exit 0");
                return (wt_blob*)nullptr;
            }
            std::vector<uint8_t> bl2;
            if (!load_file(op2, bl2)) { CHECK(false, "matmul read blob"); return (wt_blob*)nullptr; }
            wt_blob* w2 = new wt_blob{};
            if (wt_parse(bl2.data(), bl2.size(), w2) != WT_OK) {
                CHECK(false, "matmul wt_parse"); delete w2; return (wt_blob*)nullptr;
            }
            return w2;
        };

        /* T010: 大 N (8192/K1024, need≈17MB) 超 16MB 界 → SPLIT 分块 */
        {
            wt_blob* w = run_matmul_emit(32, 1024, 8192);
            CHECK(w && w->n_ops >= 2, "US1: N=8192 超界 → 分块 op 数 ≥2");
            if (w && w->n_ops >= 2) {
                bool all_split = true, arity9 = true, shared_ref = true, n0_cov = true;
                for (uint32_t i = 0; i < w->n_ops; i++) {
                    if (w->ops[i].opcode != 35) all_split = false;      /* OP_MATMUL_F16_SPLIT */
                    if (w->ops[i].n_args != 9) arity9 = false;
                }
                const uint32_t a0 = w->ops[0].args[0], s1 = w->ops[0].args[1], t2 = w->ops[0].args[2];
                for (uint32_t i = 0; i < w->n_ops; i++) {
                    if (w->ops[i].args[0] != a0 || w->ops[i].args[1] != s1 || w->ops[i].args[2] != t2)
                        shared_ref = false;
                }
                uint32_t expect_n0 = 0;
                for (uint32_t i = 0; i < w->n_ops; i++) {
                    if (w->ops[i].args[6] != expect_n0) n0_cov = false;
                    if (w->ops[i].args[3] != 32 || w->ops[i].args[4] != 1024 || w->ops[i].args[5] != 8192) n0_cov = false;
                    if (w->ops[i].args[7] == 0 || (w->ops[i].args[7] % 32) != 0) n0_cov = false;
                    expect_n0 += w->ops[i].args[7];
                }
                if (expect_n0 != 8192) n0_cov = false;
                CHECK(all_split, "US1: 全 op opcode=35 (SPLIT)");
                CHECK(arity9, "US1: 全 op arity=9");
                CHECK(shared_ref, "US1: act/wgt/out 三引用跨 chunk 共享");
                CHECK(n0_cov, "US1: n0 递增覆盖 [0,N) 且 M/K/N_full/N_chunk(32×) 一致");
            }
            delete w;
        }

        /* T011: 小 N (256/K512) 不超界 → 旧 OP_MATMUL_F16 arity 7 行为不变 */
        {
            wt_blob* w = run_matmul_emit(32, 512, 256);
            CHECK(w && w->n_ops == 1, "US1: N=256 不超界 → 单 op (旧兼容)");
            if (w && w->n_ops == 1) {
                CHECK(w->ops[0].opcode == OP_MATMUL_F16 && w->ops[0].n_args == 7,
                      "US1: 小 N 走 OP_MATMUL_F16(24) arity=7 不变");
                CHECK(w->ops[0].args[3] == 32 && w->ops[0].args[4] == 512 && w->ops[0].args[5] == 256,
                      "US1: 小 N M/K/N 参数不变");
            }
            delete w;
        }

        /* T016: host 合成数值验证 —— 发射出的分块 (n0,N_chunk,N_full) 经 FC 标量
         * 重放后逐字节等于整算标量 (五纪律之「k 升序不动」: N 只切列, 每元素
         * 乘积流不变)。输入/权重取确定性非零整数 (f16 精确), 非零保证 n0 错位
         * 会在零值下不可见。 */
        {
            auto pat_a = [](size_t i) -> uint16_t { return r_f32_to_f16((float)((int)(i % 29) - 14)); };
            auto pat_w = [](size_t i) -> uint16_t { return r_f32_to_f16((float)((int)(i % 37) - 18)); };

            /* (a) 大 N=8192/K1024 发射分块: 逐 chunk 标量 vs 整算标量 */
            {
                uint32_t M = 32, K = 1024, N_full = 8192;
                wt_blob* w = run_matmul_emit(M, K, N_full);
                CHECK(w && w->n_ops >= 2, "T016: N=8192 发射分块 op ≥2");
                if (w && w->n_ops >= 2) {
                    std::vector<uint16_t> a((size_t)M * K), wgt((size_t)N_full * K);
                    for (size_t i = 0; i < a.size(); i++) a[i] = pat_a(i);
                    for (size_t i = 0; i < wgt.size(); i++) wgt[i] = pat_w(i);
                    std::vector<uint16_t> y((size_t)M * N_full, 0);
                    for (uint32_t i = 0; i < w->n_ops; i++) {
                        fc_ref(a.data(), wgt.data(), y.data(), M, K, N_full,
                               w->ops[i].args[6], w->ops[i].args[7]);
                    }
                    std::vector<uint16_t> yref((size_t)M * N_full, 0);
                    fc_ref(a.data(), wgt.data(), yref.data(), M, K, N_full, 0, N_full);
                    CHECK(y == yref, "T016: 分块标量 vs 整算标量逐字节一致");
                }
                delete w;
            }

            /* (b) 边界形状: M=32 单行 tile / N=32 单列 tile / 部分尾 chunk / 退化单块 */
            {
                struct Case { uint32_t M, K, N_full, Nc; };
                const Case cases[] = {
                    {32, 64, 256, 32},     /* Nc=32 单列 chunk */
                    {32, 32, 32, 32},      /* 单 tile */
                    {32, 1024, 4096, 384}, /* 部分尾 chunk (4096 % 384 != 0) */
                    {32, 128, 1024, 1024}, /* 退化: Nc == N_full 单块 */
                };
                for (const auto& c : cases) {
                    std::vector<uint16_t> a((size_t)c.M * c.K), wgt((size_t)c.N_full * c.K);
                    for (size_t i = 0; i < a.size(); i++) a[i] = pat_a(i);
                    for (size_t i = 0; i < wgt.size(); i++) wgt[i] = pat_w(i);
                    std::vector<uint16_t> y((size_t)c.M * c.N_full, 0);
                    for (uint32_t n0 = 0; n0 < c.N_full; n0 += c.Nc) {
                        uint32_t nc = (c.Nc < c.N_full - n0) ? c.Nc : (c.N_full - n0);
                        fc_ref(a.data(), wgt.data(), y.data(), c.M, c.K, c.N_full, n0, nc);
                    }
                    std::vector<uint16_t> yref((size_t)c.M * c.N_full, 0);
                    fc_ref(a.data(), wgt.data(), yref.data(), c.M, c.K, c.N_full, 0, c.N_full);
                    char msg[96];
                    std::snprintf(msg, sizeof(msg), "T016: 边界 M=%u K=%u N=%u Nc=%u 逐字节",
                                  c.M, c.K, c.N_full, c.Nc);
                    CHECK(y == yref, msg);
                }
            }
        }
    }

    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
