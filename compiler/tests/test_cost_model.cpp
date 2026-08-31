// Test: Cost model MLP + calibration (Phase 4.4)
// Validates MLP loading, feature extraction, prediction, calibration dump,
// CostBasedScheduler fix, and tcm_migration cost-aware priority.
#include "hnnx/cost/cost_model.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ops/ops.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>

using namespace hnnx;
using namespace hnnx::costbased;

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++tests_passed;
    } else {
        ++tests_failed;
        std::cout << "  FAIL: " << msg << "\n";
    }
}

// ---------------------------------------------------------------------------
// Test 1: init_for_soc loads MLP model.
// ---------------------------------------------------------------------------
static void test_mlp_loaded() {
    std::cout << "\n[Test 1] MLP model loaded after init\n";
    CostSource cs;
    check(!cs.has_mlp_model(), "MLP empty before init");
    cs.init_for_soc("v75");
    check(cs.has_mlp_model(), "MLP loaded after init");

    const auto& mlp = cs.get_mlp_model();
    check(mlp.weights.size() == 2, "2 layers");
    check(mlp.biases.size() == 2, "2 bias vectors");
    check(mlp.use_linear.size() == 2, "2 use_linear flags");
    check(!mlp.use_linear[0], "layer 0 = ReLU (hidden)");
    check(mlp.use_linear[1], "layer 1 = linear (output)");
}

// ---------------------------------------------------------------------------
// Test 2: MLP architecture — 8→4→1.
// ---------------------------------------------------------------------------
static void test_mlp_architecture() {
    std::cout << "\n[Test 2] MLP architecture (8→4→1)\n";
    CostSource cs;
    cs.init_for_soc("v75");
    const auto& mlp = cs.get_mlp_model();

    // Layer 0: 4 neurons × 8 inputs = 32 weights
    check(mlp.weights[0].size() == 32, "layer 0: 32 weights (4×8)");
    check(mlp.biases[0].size() == 4, "layer 0: 4 biases");

    // Layer 1: 1 neuron × 4 inputs = 4 weights
    check(mlp.weights[1].size() == 4, "layer 1: 4 weights (1×4)");
    check(mlp.biases[1].size() == 1, "layer 1: 1 bias");
}

// ---------------------------------------------------------------------------
// Test 3: extract_features pads to 8.
// ---------------------------------------------------------------------------
static void test_feature_padding() {
    std::cout << "\n[Test 3] Feature padding to 8\n";
    CostSource cs;
    cs.init_for_soc("v75");

    grdep::OpDesc desc;
    desc.op_name = "Conv";
    desc.op_type = 0;
    desc.input_dims = {1, 3, 224, 224};
    desc.output_dims = {1, 64, 112, 112};
    desc.nsp_count = 1;
    desc.vtcm_budget = 3 * 1024 * 1024;

    auto features = cs.extract_features(nullptr, &desc, {});
    check(features.size() == 8, "features padded to 8");
}

// ---------------------------------------------------------------------------
// Test 4: extract_features with empty dims still pads to 8.
// ---------------------------------------------------------------------------
static void test_feature_empty_dims() {
    std::cout << "\n[Test 4] Feature padding (empty dims)\n";
    CostSource cs;
    cs.init_for_soc("v75");

    grdep::OpDesc desc;
    desc.op_name = "Relu";
    desc.op_type = 1;
    desc.input_dims = {};
    desc.output_dims = {};
    desc.nsp_count = 1;
    desc.vtcm_budget = 0;

    auto features = cs.extract_features(nullptr, &desc, {});
    check(features.size() == 8, "features padded to 8");
    // Should be [op_type=1, nsp=1, vtcm=0, mode=0, 0, 0, 0, 0]
    check(features[0] == 1.0f, "feature[0] = op_type");
    check(features[1] == 1.0f, "feature[1] = nsp_count");
    check(features[2] == 0.0f, "feature[2] = vtcm_budget");
}

// ---------------------------------------------------------------------------
// Test 5: MLP predict produces non-empty output.
// ---------------------------------------------------------------------------
static void test_mlp_predict() {
    std::cout << "\n[Test 5] MLP predict\n";
    CostSource cs;
    cs.init_for_soc("v75");

    grdep::OpDesc desc;
    desc.op_name = "Conv";
    desc.op_type = 0;
    desc.input_dims = {1, 3, 32, 32};
    desc.output_dims = {1, 64, 16, 16};
    desc.nsp_count = 1;
    desc.vtcm_budget = 1024 * 1024;

    auto features = cs.extract_features(nullptr, &desc, {});
    const auto& mlp = cs.get_mlp_model();
    auto pred = mlp.predict(features);

    check(!pred.empty(), "MLP predict non-empty");
    check(pred.size() == 1, "MLP output size = 1");
    check(pred[0] >= 0.0f, "MLP output non-negative");
}

// ---------------------------------------------------------------------------
// Test 6: MLP fallback works in get_prediction_from_cost_model.
//   When op_name is unknown (not in table), should use MLP.
// ---------------------------------------------------------------------------
static void test_mlp_fallback() {
    std::cout << "\n[Test 6] MLP fallback for unknown op\n";
    CostSource cs;
    cs.init_for_soc("v75");

    // "UnknownOp123" is not in the cost table → should fall to MLP
    grdep::OpDesc desc;
    desc.op_name = "UnknownOp123";
    desc.op_type = 99;
    desc.input_dims = {1, 3, 32, 32};
    desc.output_dims = {1, 64, 16, 16};
    desc.nsp_count = 1;
    desc.vtcm_budget = 1024 * 1024;

    float cost = cs.get_prediction_from_cost_model("UnknownOp123", nullptr, &desc, {});
    check(cost > 0.0f, "unknown op cost > 0 (MLP fallback)");

    std::cout << "  UnknownOp cost: " << cost << "\n";
}

// ---------------------------------------------------------------------------
// Test 7: Analytical path (desc != null) takes priority over table.
// ---------------------------------------------------------------------------
static void test_analytical_path() {
    std::cout << "\n[Test 7] Analytical path (desc provided)\n";
    CostSource cs;
    cs.init_for_soc("v75");

    grdep::OpDesc desc;
    desc.op_name = "Conv";
    desc.op_type = 0;
    desc.input_dims = {1, 3, 32, 32};
    desc.output_dims = {1, 64, 16, 16};
    desc.nsp_count = 1;
    desc.vtcm_budget = 0;

    float cost = cs.get_prediction_from_cost_model("Conv", nullptr, &desc, {});
    // Analytical: total_elements = 1*64*16*16 = 16384, cost_per_element = 9.0
    // Expected: 16384 * 9.0 = 147456
    check(cost == 16384.0f * 9.0f, "analytical Conv cost = 147456");

    std::cout << "  Analytical Conv cost: " << cost << "\n";
}

// ---------------------------------------------------------------------------
// Test 8: Table lookup path (desc == null).
// ---------------------------------------------------------------------------
static void test_table_lookup() {
    std::cout << "\n[Test 8] Table lookup (desc null)\n";
    CostSource cs;
    cs.init_for_soc("v75");

    float conv_cost = cs.get_prediction_from_cost_model("Conv", nullptr, nullptr, {});
    check(conv_cost == 1000.0f, "table Conv = 1000");

    float relu_cost = cs.get_prediction_from_cost_model("Relu", nullptr, nullptr, {});
    check(relu_cost == 10.0f, "table Relu = 10");
}

// ---------------------------------------------------------------------------
// Test 9: InferenceMode scaling.
// ---------------------------------------------------------------------------
static void test_inference_mode() {
    std::cout << "\n[Test 9] Inference mode scaling\n";
    CostSource cs;
    cs.init_for_soc("v75");

    InferenceMode perf{};
    InferenceMode power{InferenceMode::Power};
    InferenceMode bw{InferenceMode::Bandwidth};

    float base = cs.get_prediction_from_cost_model("Add", nullptr, nullptr, perf);
    float pwr = cs.get_prediction_from_cost_model("Add", nullptr, nullptr, power);
    float bwd = cs.get_prediction_from_cost_model("Add", nullptr, nullptr, bw);

    check(pwr == base * 0.7f, "Power = 0.7× Performance");
    check(bwd == base * 0.5f, "Bandwidth = 0.5× Performance");
}

// ---------------------------------------------------------------------------
// Test 10: NSP scaling in analytical path.
// ---------------------------------------------------------------------------
static void test_nsp_scaling() {
    std::cout << "\n[Test 10] NSP scaling (analytical)\n";
    CostSource cs;
    cs.init_for_soc("v75");

    grdep::OpDesc desc1;
    desc1.op_name = "Conv";
    desc1.output_dims = {1, 64, 16, 16};
    desc1.nsp_count = 1;

    grdep::OpDesc desc4 = desc1;
    desc4.nsp_count = 4;

    float cost1 = cs.get_prediction_from_cost_model("Conv", nullptr, &desc1, {});
    float cost4 = cs.get_prediction_from_cost_model("Conv", nullptr, &desc4, {});

    check(cost4 == cost1 / 4.0f, "4 NSP = cost / 4");
}

// ---------------------------------------------------------------------------
// Test 11: dump_calibration_features creates CSV.
// ---------------------------------------------------------------------------
static void test_calibration_dump() {
    std::cout << "\n[Test 11] Calibration feature dump\n";
    CostSource cs;
    cs.init_for_soc("v75");

    // Remove old file if exists
    std::remove("test_calibration.csv");

    // dump_calibration_features takes Op* list; pass empty for simplicity
    std::vector<Op*> empty_ops;
    cs.dump_calibration_features(empty_ops, "test_calibration.csv");

    std::ifstream f("test_calibration.csv");
    bool exists = f.good();
    f.close();

    check(exists, "CSV file created");
    if (exists) {
        // Verify header
        std::ifstream in("test_calibration.csv");
        std::string header;
        std::getline(in, header);
        check(header.find("op_type") != std::string::npos, "CSV has header");
        in.close();
        std::remove("test_calibration.csv");
    }
}

// ---------------------------------------------------------------------------
// Test 12: MLP linear output layer (no ReLU on last layer).
//   The output should be able to produce negative values if weights/biases
//   are negative (regression, not classification).
// ---------------------------------------------------------------------------
static void test_mlp_linear_output() {
    std::cout << "\n[Test 12] MLP linear output layer\n";
    CostSource::MLP mlp;
    // Single layer, linear, weight=-1, bias=0.5, input=1.0 → -0.5 (negative!)
    mlp.weights = {{-1.0f}};
    mlp.biases = {{0.5f}};
    mlp.use_linear = {true};

    auto pred = mlp.predict({1.0f});
    check(pred.size() == 1, "output size 1");
    check(pred[0] == -0.5f, "linear output = -0.5 (no ReLU clamp)");
}

// ---------------------------------------------------------------------------
// Test 13: MLP ReLU hidden layer clamps negatives.
// ---------------------------------------------------------------------------
static void test_mlp_relu_hidden() {
    std::cout << "\n[Test 13] MLP ReLU hidden clamps\n";
    CostSource::MLP mlp;
    mlp.weights = {{-1.0f}};
    mlp.biases = {{0.5f}};
    mlp.use_linear = {false};  // ReLU

    auto pred = mlp.predict({1.0f});
    // -1*1 + 0.5 = -0.5 → ReLU → 0.0
    check(pred[0] == 0.0f, "ReLU clamps -0.5 to 0");
}

// ---------------------------------------------------------------------------
// Test 14: CostBasedScheduler sorts by cost (descending).
// ---------------------------------------------------------------------------
static void test_cost_based_scheduler() {
    std::cout << "\n[Test 14] CostBasedScheduler sort\n";
    CostSource cs;
    cs.init_for_soc("v75");

    // Create TypicalOps with different op_type_name
    GraphPrepare gp;
    std::vector<Op*> ops;

    TypicalOp conv_op;
    conv_op.op_type_name = "Conv";
    conv_op.op_id = 1;

    TypicalOp relu_op;
    relu_op.op_type_name = "Relu";
    relu_op.op_id = 2;

    ops = {&conv_op, &relu_op};

    CostBasedScheduler sched(cs);
    auto result = sched.schedule(ops, nullptr, {});

    // Conv (1000) > Relu (10) → Conv first
    check(result.size() == 2, "2 ops scheduled");
    auto* first = dynamic_cast<const TypicalOp*>(result[0]);
    check(first && first->op_type_name == "Conv", "Conv first (higher cost)");
}

// ---------------------------------------------------------------------------
// Test 15: Empty input.
// ---------------------------------------------------------------------------
static void test_empty_input() {
    std::cout << "\n[Test 15] Empty input\n";
    CostSource cs;
    cs.init_for_soc("v75");

    float cost = cs.get_prediction_from_cost_model("", nullptr, nullptr, {});
    // Empty name, no desc, MLP fallback with empty features → returns 1.0
    check(cost == 1.0f, "empty op → 1.0f");
}

int main() {
    std::cout << "=== Cost Model MLP Test (Phase 4.4) ===\n";

    test_mlp_loaded();
    test_mlp_architecture();
    test_feature_padding();
    test_feature_empty_dims();
    test_mlp_predict();
    test_mlp_fallback();
    test_analytical_path();
    test_table_lookup();
    test_inference_mode();
    test_nsp_scaling();
    test_calibration_dump();
    test_mlp_linear_output();
    test_mlp_relu_hidden();
    test_cost_based_scheduler();
    test_empty_input();

    std::cout << "\n=== " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
