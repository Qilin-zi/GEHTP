// test_execute_chain: verify hexagon_nn_* full call chain
// Tests: open->append_node->prepare->execute->serialize->deserialize->execute->close

#include "hnnx/api/c_interface.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/ops/ops.hpp"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

int main() {
    std::printf("=== Execute Chain Test ===\n\n");

    register_all_ops();

    int graph_id = 0;
    int ret = hexagon_nn_open("test", 75, &graph_id);
    CHECK(ret == 0 && graph_id > 0, "hexagon_nn_open");

    // Build: Input -> Relu -> Output
    // Input: float32, shape [1, 1, 1, 4]
    hnnx::OutputDef out_def{};
    out_def.rank = 4;
    out_def.dims[0] = 1; out_def.dims[1] = 1;
    out_def.dims[2] = 1; out_def.dims[3] = 4;
    out_def.element_size = 4;
    out_def.dtype = static_cast<uint32_t>(hnnx::DType::Float32);

    ret = hnnx::hexagon_nn_append_node(graph_id, 1, "Input", nullptr, 0, &out_def, 1);
    CHECK(ret == 0, "append_node(Input)");

    // Relu: Input(id=1) -> Relu
    // InputDef encodes (src_id, out_idx) as (rank, dtype)
    hnnx::InputDef relu_in{};
    relu_in.rank = 1;  // src_id = Input node's op_id
    ret = hnnx::hexagon_nn_append_node(graph_id, 2, "Relu", &relu_in, 1, &out_def, 1);
    CHECK(ret == 0, "append_node(Relu)");

    // Output: Relu(id=2) -> Output
    hnnx::InputDef out_in{};
    out_in.rank = 2;  // src_id = Relu node's op_id
    ret = hnnx::hexagon_nn_append_node(graph_id, 3, "Output", &out_in, 1, nullptr, 0);
    CHECK(ret == 0, "append_node(Output)");

    // Prepare
    hnnx::HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    ret = hnnx::hexagon_nn_prepare(graph_id, env);
    CHECK(ret == 0, "hexagon_nn_prepare");

    // Execute: input = [-1, 2, -3, 4], expect Relu -> [0, 2, 0, 4]
    float input[] = {-1.0f, 2.0f, -3.0f, 4.0f};
    float output[4] = {};
    ret = hnnx::hexagon_nn_execute(graph_id, 1, input, 1, output);
    CHECK(ret == 0, "hexagon_nn_execute");

    bool correct = true;
    float expected[] = {0.0f, 2.0f, 0.0f, 4.0f};
    for (int i = 0; i < 4; i++) {
        if (std::fabs(output[i] - expected[i]) > 1e-5f) correct = false;
    }
    CHECK(correct, "execution result: Relu([-1,2,-3,4]) == [0,2,0,4]");
    std::printf("  Result: [%.1f, %.1f, %.1f, %.1f]\n",
                output[0], output[1], output[2], output[3]);

    // Serialize + Deserialize round-trip
    uint8_t buf[8192];
    size_t out_size = 0;
    ret = hnnx::hexagon_nn_serialize_to_mem(graph_id, buf, sizeof(buf), &out_size);
    CHECK(ret == 0 && out_size > 0, "serialize_to_mem");

    // Close and reopen to verify deserialize
    hnnx::hexagon_nn_close(graph_id);
    ret = hnnx::hexagon_nn_open("test_deser", 75, &graph_id);
    CHECK(ret == 0 && graph_id > 0, "re-open");

    ret = hnnx::hexagon_nn_deserialize(graph_id, buf, out_size);
    CHECK(ret == 0, "deserialize");

    // Execute again on the deserialized graph
    ret = hnnx::hexagon_nn_execute(graph_id, 1, input, 1, output);
    CHECK(ret == 0, "execute after deserialize");

    correct = true;
    for (int i = 0; i < 4; i++) {
        if (std::fabs(output[i] - expected[i]) > 1e-5f) correct = false;
    }
    CHECK(correct, "post-deserialize result matches");

    hnnx::hexagon_nn_close(graph_id);

    std::printf("\n=== %s ===\n", failed ? "FAILED" : "ALL PASSED");
    return failed;
}


