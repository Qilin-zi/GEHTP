// test_ops: Unit tests for Phase 2 host-side op execute
// Tests Conv2D, math, activation, shape, pooling, normalization ops
#include "hnnx/ops/ops.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/ir/tensor.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace hnnx;

static constexpr float EPS = 1e-4f;
static bool near(float a, float b, float eps = EPS) { return std::fabs(a - b) < eps; }

// Helper: create OutputDef with given rank and dims
static OutputDef make_od(uint32_t rank, const std::vector<uint64_t>& dims,
                         DType dt = DType::Float32) {
    OutputDef od{};
    od.rank = rank;
    od.dtype = static_cast<uint32_t>(dt);
    for (uint32_t i = 0; i < rank && i < dims.size(); i++)
        od.dims[i] = dims[i];
    od.element_size = 4;
    return od;
}

// Helper: execute a named op
static std::vector<float> run_op(const std::string& name,
                                 const std::vector<const uint8_t*>& inputs,
                                 const OutputDef& out_def,
                                 const std::vector<OutputDef>& in_defs = {},
                                 const std::vector<uint8_t>& params = {}) {
    size_t n = 1;
    for (uint32_t i = 0; i < out_def.rank && i < 5; i++)
        n *= out_def.dims[i];
    if (n == 0) n = 1;

    TypicalOp op;
    op.op_type_name = name;
    op.params = params;
    std::vector<float> output(n, -999.0f);
    op.execute(inputs, reinterpret_cast<uint8_t*>(output.data()), out_def, in_defs);
    return output;
}

int main() {
    // ===== Test 1: Conv2D basic 3x3 =====
    {
        // Input: [1, 3, 3, 1], Weight: [3, 3, 1, 1], Output: [1, 1, 1, 1]
        // Simple 3x3 average-like kernel (all weights = 1/9)
        float inp[] = {1,2,3, 4,5,6, 7,8,9};
        float wt[9];
        for (int i = 0; i < 9; i++) wt[i] = 1.0f / 9.0f;
        auto od_in  = make_od(4, {1, 3, 3, 1});
        auto od_wt  = make_od(4, {3, 3, 1, 1});
        auto od_out = make_od(4, {1, 1, 1, 1});
        std::vector<const uint8_t*> inputs = {
            reinterpret_cast<const uint8_t*>(inp),
            reinterpret_cast<const uint8_t*>(wt),
            nullptr
        };
        auto result = run_op("Conv", inputs, od_out, {od_in, od_wt});
        // Average of 1..9 = 5.0
        assert(near(result[0], 5.0f));
        std::cout << "Conv2D 3x3 avg: " << result[0] << " (expected 5.0)\n";
    }

    // ===== Test 2: Conv2D with stride-2 =====
    {
        // Input: [1, 4, 4, 1], Weight: [2, 2, 1, 1], Output: [1, 2, 2, 1]
        // All-ones kernel, stride 2
        float inp[16];
        for (int i = 0; i < 16; i++) inp[i] = 1.0f;
        float wt[] = {1, 1, 1, 1};
        auto od_in  = make_od(4, {1, 4, 4, 1});
        auto od_wt  = make_od(4, {2, 2, 1, 1});
        auto od_out = make_od(4, {1, 2, 2, 1});
        std::vector<const uint8_t*> inputs = {
            reinterpret_cast<const uint8_t*>(inp),
            reinterpret_cast<const uint8_t*>(wt), nullptr
        };
        auto result = run_op("Conv", inputs, od_out, {od_in, od_wt});
        // Each 2x2 patch of all-ones = sum of 4 ones = 4.0
        for (int i = 0; i < 4; i++) assert(near(result[i], 4.0f));
        std::cout << "Conv2D stride-2: all outputs = 4.0\n";
    }

    // ===== Test 3: Math ops =====
    {
        float a[] = {4.0f, 9.0f, 16.0f, 25.0f};
        auto od = make_od(1, {4});
        std::vector<const uint8_t*> inputs = {reinterpret_cast<const uint8_t*>(a)};

        auto r_sqrt = run_op("Sqrt", inputs, od);
        assert(near(r_sqrt[0], 2.0f) && near(r_sqrt[1], 3.0f));
        assert(near(r_sqrt[2], 4.0f) && near(r_sqrt[3], 5.0f));

        auto r_abs = run_op("Abs", {reinterpret_cast<const uint8_t*>(
            std::vector<float>{-3.0f, 2.0f, -1.0f, 0.0f}.data())}, od);
        assert(near(r_abs[0], 3.0f) && near(r_abs[1], 2.0f));

        auto r_ceil = run_op("Ceiling", {reinterpret_cast<const uint8_t*>(
            std::vector<float>{1.2f, -0.5f, 3.0f, 2.7f}.data())}, od);
        assert(near(r_ceil[0], 2.0f) && near(r_ceil[1], 0.0f));
        assert(near(r_ceil[2], 3.0f) && near(r_ceil[3], 3.0f));

        std::cout << "Math ops: Sqrt, Abs, Ceiling verified\n";
    }

    // ===== Test 4: Power op =====
    {
        float base[] = {2.0f, 3.0f, 4.0f, 5.0f};
        float exp[]  = {3.0f, 2.0f, 0.5f, 1.0f};
        auto od = make_od(1, {4});
        std::vector<const uint8_t*> inputs = {
            reinterpret_cast<const uint8_t*>(base),
            reinterpret_cast<const uint8_t*>(exp)
        };
        auto r = run_op("Power", inputs, od);
        assert(near(r[0], 8.0f));    // 2^3
        assert(near(r[1], 9.0f));    // 3^2
        assert(near(r[2], 2.0f));    // 4^0.5
        assert(near(r[3], 5.0f));    // 5^1
        std::cout << "Power: [2^3, 3^2, 4^0.5, 5^1] verified\n";
    }

    // ===== Test 5: Activation ops =====
    {
        float a[] = {-2.0f, -1.0f, 0.0f, 1.0f};
        auto od = make_od(1, {4});
        std::vector<const uint8_t*> inputs = {reinterpret_cast<const uint8_t*>(a)};

        // Gelu: approximate values
        auto r_gelu = run_op("Gelu", inputs, od);
        assert(near(r_gelu[2], 0.0f, 0.01f));  // Gelu(0) = 0
        assert(r_gelu[3] > 0.8f);                // Gelu(1) ≈ 0.8413

        // LeakyRelu: alpha = 0.01
        auto r_lr = run_op("LeakyRelu", inputs, od);
        assert(near(r_lr[0], -0.02f));  // -2 * 0.01
        assert(near(r_lr[2], 0.0f));
        assert(near(r_lr[3], 1.0f));

        // Elu: alpha = 1.0
        auto r_elu = run_op("Elu", inputs, od);
        assert(near(r_elu[2], 0.0f));
        assert(r_elu[0] < 0.0f && r_elu[0] > -1.0f);  // e^-2 - 1 ≈ -0.865

        // Swish: x * sigmoid(x)
        auto r_sw = run_op("Swish", inputs, od);
        assert(near(r_sw[2], 0.0f, 0.01f));  // Swish(0) = 0
        assert(r_sw[3] > 0.7f);               // Swish(1) ≈ 0.7311

        // HardSigmoid
        auto r_hs = run_op("HardSigmoid", inputs, od);
        assert(near(r_hs[0], 0.1f));   // clamp(-2*0.2+0.5, 0, 1) = 0.1
        assert(near(r_hs[2], 0.5f));   // clamp(0*0.2+0.5, 0, 1) = 0.5
        assert(near(r_hs[3], 0.7f));   // clamp(1*0.2+0.5, 0, 1) = 0.7

        // Softplus: log(1+exp(x))
        auto r_sp = run_op("Softplus", inputs, od);
        assert(r_sp[2] > 0.69f && r_sp[2] < 0.70f);  // log(2) ≈ 0.6931

        std::cout << "Activations: Gelu, LeakyRelu, Elu, Swish, HardSigmoid, Softplus\n";
    }

    // ===== Test 6: Reshape (copy) =====
    {
        float a[] = {1, 2, 3, 4, 5, 6};
        auto od_in  = make_od(2, {2, 3});
        auto od_out = make_od(1, {6});
        std::vector<const uint8_t*> inputs = {reinterpret_cast<const uint8_t*>(a)};
        auto r = run_op("Reshape", inputs, od_out, {od_in});
        for (int i = 0; i < 6; i++) assert(near(r[i], (float)(i + 1)));
        std::cout << "Reshape: [2,3] -> [6] data preserved\n";
    }

    // ===== Test 7: Concat along last axis =====
    {
        float a[] = {1, 2, 3};
        float b[] = {4, 5, 6};
        auto od_a = make_od(2, {1, 3});
        auto od_b = make_od(2, {1, 3});
        auto od_out = make_od(2, {1, 6});
        std::vector<const uint8_t*> inputs = {
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b)
        };
        // axis=1 (last axis)
        int32_t axis = 1;
        std::vector<uint8_t> params(4);
        std::memcpy(params.data(), &axis, 4);
        auto r = run_op("Concat", inputs, od_out, {od_a, od_b}, params);
        assert(near(r[0], 1.0f) && near(r[1], 2.0f) && near(r[2], 3.0f));
        assert(near(r[3], 4.0f) && near(r[4], 5.0f) && near(r[5], 6.0f));
        std::cout << "Concat: [1,3]+[1,3] along axis=1 -> [1,6]\n";
    }

    // ===== Test 8: MinMax and SquaredDifference =====
    {
        float a[] = {1, 5, 3, 7};
        float b[] = {4, 2, 6, 0};
        auto od = make_od(1, {4});
        std::vector<const uint8_t*> inputs = {
            reinterpret_cast<const uint8_t*>(a),
            reinterpret_cast<const uint8_t*>(b)
        };
        auto r_min = run_op("MinMax", inputs, od);
        assert(near(r_min[0], 1.0f) && near(r_min[1], 2.0f));
        assert(near(r_min[2], 3.0f) && near(r_min[3], 0.0f));

        auto r_sq = run_op("SquaredDifference", inputs, od);
        assert(near(r_sq[0], 9.0f));   // (1-4)^2
        assert(near(r_sq[1], 9.0f));   // (5-2)^2
        assert(near(r_sq[2], 9.0f));   // (3-6)^2
        assert(near(r_sq[3], 49.0f));  // (7-0)^2
        std::cout << "MinMax + SquaredDifference verified\n";
    }

    // ===== Test 9: MaxPool =====
    {
        // Input: [1, 4, 4, 1], pool 2x2, stride 2 -> Output: [1, 2, 2, 1]
        float inp[16];
        for (int i = 0; i < 16; i++) inp[i] = (float)i;
        auto od_in  = make_od(4, {1, 4, 4, 1});
        auto od_out = make_od(4, {1, 2, 2, 1});
        std::vector<const uint8_t*> inputs = {reinterpret_cast<const uint8_t*>(inp)};
        auto r = run_op("MaxPool", inputs, od_out, {od_in});
        assert(near(r[0], 5.0f));   // max(0,1,4,5)
        assert(near(r[1], 7.0f));   // max(2,3,6,7)
        assert(near(r[2], 13.0f));  // max(8,9,12,13)
        assert(near(r[3], 15.0f));  // max(10,11,14,15)
        std::cout << "MaxPool 2x2: [5, 7, 13, 15] verified\n";
    }

    // ===== Test 10: AvgPool =====
    {
        float inp[16];
        for (int i = 0; i < 16; i++) inp[i] = (float)i;
        auto od_in  = make_od(4, {1, 4, 4, 1});
        auto od_out = make_od(4, {1, 2, 2, 1});
        std::vector<const uint8_t*> inputs = {reinterpret_cast<const uint8_t*>(inp)};
        auto r = run_op("AvgPool", inputs, od_out, {od_in});
        assert(near(r[0], 2.5f));   // avg(0,1,4,5)
        assert(near(r[1], 4.5f));   // avg(2,3,6,7)
        assert(near(r[2], 10.5f));  // avg(8,9,12,13)
        assert(near(r[3], 12.5f));  // avg(10,11,14,15)
        std::cout << "AvgPool 2x2: [2.5, 4.5, 10.5, 12.5] verified\n";
    }

    // ===== Test 11: LayerNorm =====
    {
        // Input: [1, 4] -> normalize to zero-mean/unit-var, then scale+bias
        float inp[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float scale[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float bias[] = {0.0f, 0.0f, 0.0f, 0.0f};
        auto od = make_od(2, {1, 4});
        std::vector<const uint8_t*> inputs = {
            reinterpret_cast<const uint8_t*>(inp),
            reinterpret_cast<const uint8_t*>(scale),
            reinterpret_cast<const uint8_t*>(bias)
        };
        auto r = run_op("LayerNorm", inputs, od);
        // Mean = 2.5, Var = 1.25, InvStd = 1/sqrt(1.25+1e-5) ≈ 0.8944
        // Normalized: (1-2.5)*0.8944, (2-2.5)*0.8944, ...
        float inv_std = 1.0f / std::sqrt(1.25f + 1e-5f);
        assert(near(r[0], (1.0f - 2.5f) * inv_std, 0.01f));
        assert(near(r[1], (2.0f - 2.5f) * inv_std, 0.01f));
        assert(near(r[2], (3.0f - 2.5f) * inv_std, 0.01f));
        assert(near(r[3], (4.0f - 2.5f) * inv_std, 0.01f));
        std::cout << "LayerNorm: normalized [-1.34, -0.45, 0.45, 1.34]\n";
    }

    // ===== Test 12: tensor_generator_scalar =====
    {
        OutputDef od{};
        od.rank = 0;  // scalar
        od.dtype = static_cast<uint32_t>(DType::Float32);
        od.element_size = 4;
        float val = 3.14f;
        auto* t = tensor_generator_scalar(nullptr, &od, reinterpret_cast<const uint8_t*>(&val));
        assert(t != nullptr);
        assert(t->dtype == static_cast<uint32_t>(DType::Float32));
        assert(t->num_elements() == 1);
        assert(t->get_element_size() == 4);
        delete t;
        std::cout << "tensor_generator_scalar: scalar Float32 created\n";
    }

    // ===== Test 13: Conv2D with bias =====
    {
        // Input: [1, 2, 2, 1], Weight: [1, 1, 1, 2], Bias: [2]
        // 1x1 conv with 2 output channels, bias = [10, 20]
        float inp[] = {1, 2, 3, 4};
        float wt[] = {2, 3};  // Kh=1, Kw=1, Cin=1, Cout=2
        float bias[] = {10, 20};
        auto od_in  = make_od(4, {1, 2, 2, 1});
        auto od_wt  = make_od(4, {1, 1, 1, 2});
        auto od_out = make_od(4, {1, 2, 2, 2});
        std::vector<const uint8_t*> inputs = {
            reinterpret_cast<const uint8_t*>(inp),
            reinterpret_cast<const uint8_t*>(wt),
            reinterpret_cast<const uint8_t*>(bias)
        };
        auto r = run_op("Conv", inputs, od_out, {od_in, od_wt});
        // For pixel (0,0): in=1, out_ch0 = 1*2+10=12, out_ch1 = 1*3+20=23
        assert(near(r[0], 12.0f));  // pixel 0, ch 0
        assert(near(r[1], 23.0f));  // pixel 0, ch 1
        // For pixel (0,1): in=2, out_ch0 = 2*2+10=14, out_ch1 = 2*3+20=26
        assert(near(r[2], 14.0f));
        assert(near(r[3], 26.0f));
        std::cout << "Conv2D 1x1 + bias: verified\n";
    }

    // ===== Test 14: Rsqrt and Log =====
    {
        float a[] = {1.0f, 4.0f, 16.0f, 100.0f};
        auto od = make_od(1, {4});
        std::vector<const uint8_t*> inputs = {reinterpret_cast<const uint8_t*>(a)};

        auto r_rs = run_op("Rsqrt", inputs, od);
        assert(near(r_rs[0], 1.0f));
        assert(near(r_rs[1], 0.5f));
        assert(near(r_rs[2], 0.25f));
        assert(near(r_rs[3], 0.1f));

        auto r_log = run_op("Log", inputs, od);
        assert(near(r_log[0], 0.0f));
        assert(near(r_log[1], std::log(4.0f)));
        assert(near(r_log[2], std::log(16.0f)));

        std::cout << "Rsqrt + Log verified\n";
    }

    std::cout << "\nAll Phase 2 op tests passed!\n";
    return 0;
}
