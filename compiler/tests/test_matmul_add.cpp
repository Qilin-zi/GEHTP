// test_matmul_add: compile and run a tiny MatMul+Add network
// Network: Input [1,4] -> MatMul(W:[4,8]) -> Add(b:[8]) -> Output [1,8]
// Verifies: build graph -> prepare -> serialize -> deserialize -> execute

#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ops/ops.hpp"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace hnnx;

int main() {
    std::printf("=== MatMul+Add Tiny Model Compile & Execute ===\n\n");
    register_all_ops();

    // --- Define network dims ---
    // Input:  [1,4]  float32
    // Weight: [4,8]  float32  (MatMul: out = in @ W)
    // Bias:    [8]    float32  (Add: out = mm + b)
    // Output: [1,8]  float32
    constexpr int IN = 4, OUT = 8;
    HexagonNNEnv env;
    env.set_num_nsps(1);
    env.set_soc_type(75);

    // --- Build graph ---
    GraphPrepare gp;

    // 1. Weight const node: W [4,8] -> 4*8*4 = 128 bytes
    OutputDef w_od{};
    w_od.rank = 2;
    w_od.dtype = static_cast<uint32_t>(DType::Float32);
    w_od.dims[0] = IN;
    w_od.dims[1] = OUT;
    w_od.element_size = 4;
    std::vector<float> W(IN * OUT, 0.0f);
    for (int i = 0; i < IN * OUT; i++) W[i] = static_cast<float>(i % 7) * 0.1f;
    std::vector<uint8_t> w_bytes(IN * OUT * 4);
    std::memcpy(w_bytes.data(), W.data(), w_bytes.size());
    op_id_t w_id = gp.append_const_node(1, w_od, w_bytes.data(), w_bytes.size());

    // 2. Bias const node: b [8] -> 8*4 = 32 bytes
    OutputDef b_od{};
    b_od.rank = 1;
    b_od.dtype = static_cast<uint32_t>(DType::Float32);
    b_od.dims[0] = OUT;
    b_od.element_size = 4;
    std::vector<float> B(OUT, 0.5f);
    std::vector<uint8_t> b_bytes(OUT * 4);
    std::memcpy(b_bytes.data(), B.data(), b_bytes.size());
    op_id_t b_id = gp.append_const_node(2, b_od, b_bytes.data(), b_bytes.size());

    // 3. Input node: [1,4]
    OutputDef in_od{};
    in_od.rank = 2;
    in_od.dtype = static_cast<uint32_t>(DType::Float32);
    in_od.dims[0] = 1;
    in_od.dims[1] = IN;
    in_od.element_size = 4;
    op_id_t in_id = gp.append_node("Input", 10, nullptr, 0, &in_od, 1, nullptr);

    // 4. MatMul node: inputs = [Input(10), Weight(1)], output [1,8]
    InputDef mm_in[2];
    mm_in[0].rank = 10; mm_in[0].dtype = 0;
    mm_in[1].rank = 1;  mm_in[1].dtype = 0;
    OutputDef mm_od{};
    mm_od.rank = 2;
    mm_od.dtype = static_cast<uint32_t>(DType::Float32);
    mm_od.dims[0] = 1;
    mm_od.dims[1] = OUT;
    mm_od.element_size = 4;
    op_id_t mm_id = gp.append_node("MatMul", 20, mm_in, 2, &mm_od, 1, nullptr);

    // 5. Add node: inputs = [MatMul(20), Bias(2)], output [1,8]
    InputDef add_in[2];
    add_in[0].rank = 20; add_in[0].dtype = 0;
    add_in[1].rank = 2;  add_in[1].dtype = 0;
    OutputDef add_od = mm_od;
    op_id_t add_id = gp.append_node("Add", 30, add_in, 2, &add_od, 1, nullptr);

    // 6. Output node: input = [Add(30)]
    InputDef out_in;
    out_in.rank = 30; out_in.dtype = 0;
    op_id_t out_id = gp.append_node("Output", 40, &out_in, 1, nullptr, 0, nullptr);

    std::printf("[1] Graph built: Input(%llu) -> MatMul(%llu) -> Add(%llu) -> Output(%llu)\n",
                (unsigned long long)in_id, (unsigned long long)mm_id,
                (unsigned long long)add_id, (unsigned long long)out_id);
    std::printf("    Weight const: id=%llu (%d bytes), Bias const: id=%llu (%d bytes)\n",
                (unsigned long long)w_id, IN * OUT * 4,
                (unsigned long long)b_id, OUT * 4);

    // --- Prepare (compile) ---
    GraphStatus st = gp.prepare(env);
    std::printf("[2] Prepare: %s\n", st == GraphStatus::Success ? "OK" : "FAIL");
    if (st != GraphStatus::Success) {
        std::printf("FAILED: prepare returned %d\n", static_cast<int>(st));
        return 1;
    }

    // Inspect graph after prepare (fusion may have changed it)
    std::printf("[2b] Post-prepare ops:\n");
    {
        auto opdefs = gp.get_sorted_opdefs();
        for (auto* od : opdefs) {
            std::printf("     id=%llu name=%s ninputs=%zu\n",
                        (unsigned long long)od->op_id,
                        od->name_tag ? od->name_tag->name() : "?",
                        od->inputs.size());
        }
    }

    // --- Serialize ---
    std::vector<uint8_t> bin(64 * 1024);
    size_t bin_size = 0;
    bool ser_ok = gp.serialize(bin.data(), bin.size(), bin_size);
    std::printf("[3] Serialize: %s, .bin size = %zu bytes\n", ser_ok ? "OK" : "FAIL", bin_size);
    if (!ser_ok) { std::printf("FAILED: serialize\n"); return 1; }

    // --- Deserialize (reload) ---
    GraphPrepare gp2;
    bool des_ok = gp2.deserialize(bin.data(), bin_size);
    std::printf("[4] Deserialize: %s\n", des_ok ? "OK" : "FAIL");
    if (!des_ok) { std::printf("FAILED: deserialize\n"); return 1; }

    // Verify graph structure round-tripped
    OpDef* mm2 = gp2.get_op_at(mm_id);
    OpDef* add2 = gp2.get_op_at(add_id);
    if (!mm2 || !add2) { std::printf("FAILED: ops missing after deserialize\n"); return 1; }
    std::printf("[5] Round-trip graph: MatMul(%llu) has %zu inputs, Add(%llu) has %zu inputs\n",
                (unsigned long long)mm_id, mm2->inputs.size(),
                (unsigned long long)add_id, add2->inputs.size());
    if (mm2->inputs.size() != 2 || add2->inputs.size() != 2) {
        std::printf("FAILED: input count mismatch\n"); return 1;
    }
    // Verify const pool data round-tripped
    OpDef* w2 = gp2.get_op_at(w_id);
    OpDef* b2 = gp2.get_op_at(b_id);
    std::printf("[5a] Const: W offset=%llu size=%llu flags=0x%x is_const=%d, B offset=%llu size=%llu flags=0x%x is_const=%d\n",
                (unsigned long long)(w2 ? w2->const_data_offset : 0),
                (unsigned long long)(w2 ? w2->const_data_size : 0),
                w2 ? w2->flags : 0, w2 ? (int)w2->is_const() : -1,
                (unsigned long long)(b2 ? b2->const_data_offset : 0),
                (unsigned long long)(b2 ? b2->const_data_size : 0),
                b2 ? b2->flags : 0, b2 ? (int)b2->is_const() : -1);

    // --- Execute ---
    std::vector<float> input = {1.0f, 0.5f, -0.3f, 2.0f};
    auto result = gp2.execute_host(input);
    std::printf("[6] Execute: ok=%s, output.size=%zu\n",
                result.ok ? "true" : "false", result.output.size());
    if (!result.ok) { std::printf("FAILED: execute returned not ok\n"); return 1; }

    // --- Compute reference (MatMul + Add) by hand ---
    std::vector<float> ref(OUT, 0.0f);
    for (int j = 0; j < OUT; j++) {
        float acc = 0.0f;
        for (int i = 0; i < IN; i++) acc += input[i] * W[i * OUT + j];
        ref[j] = acc + B[j];
    }

    // --- Compare ---
    std::printf("[7] Output comparison (model vs reference):\n");
    bool pass = true;
    for (int j = 0; j < OUT; j++) {
        float diff = std::fabs(result.output[j] - ref[j]);
        bool ok = diff < 1e-4f;
        std::printf("    [%d] model=%.4f  ref=%.4f  diff=%.6f  %s\n",
                    j, result.output[j], ref[j], diff, ok ? "OK" : "MISMATCH");
        if (!ok) pass = false;
    }

    std::printf("\n=== %s ===\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}


