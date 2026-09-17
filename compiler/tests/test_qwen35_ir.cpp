// test_qwen35_ir: M1 编译器前端验收(0.8B QNN IR 摄取 + scalar_params 通路 + 往返)
//
// 1) 合成 mini net.json(dict schema + scalar_params + tensor_params)加载:
//    - op 计数/类型/输入输出正确
//    - scalar_params 经 op_data 打包, unpack 后 operation/axis 值正确
// 2) 全量 0.8B net.json(存在时, SKIP 守卫): 加载 + serialize→deserialize
//    →re-serialize 字节确定性(3.58GB 池, 较慢)
// 3) serialize 容量保护: 小缓冲返回所需字节(不溢出)
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/ir/scalar_params.hpp"
#include "hnnx/ops/ops.hpp"
#include <map>
#include <set>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static bool write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << content;
    return f.good();
}

// 合成 mini net.json: Input → Eltwise_Binary(op=MUL, 二元) → RmsNorm(epsilon) → Output
// (dict schema, 与 2.48 qairt-dlc-to-json 输出同构; 数值非关键, 结构+scalar 通路是目标)
static const char* MINI_NET_JSON = R"json({
  "model.cpp": "", "model.bin": "",
  "converter_command": "",
  "copyright_str": "",
  "op_types": ["Eltwise_Binary", "RmsNorm"],
  "graph": {
    "tensors": {
      "x": {"id": 1, "type": 0, "dataFormat": 0, "data_type": 562, "dims": [1, 4]},
      "w": {"id": 2, "type": 4, "dataFormat": 0, "data_type": 562, "dims": [1, 4],
            "quant_params": {"definition": 0, "encoding": 0, "scale_offset": {"scale": 0.0, "offset": 0}}},
      "mul_out": {"id": 3, "type": 3, "dataFormat": 0, "data_type": 562, "dims": [1, 4]},
      "norm_out": {"id": 4, "type": 1, "dataFormat": 0, "data_type": 562, "dims": [1, 4]}
    },
    "nodes": {
      "mul_node": {
        "package": "qti.aisw", "type": "Eltwise_Binary",
        "tensor_params": {}, "scalar_params": {"operation": {"50": 2}, "packageName": {"1544": "qti.aisw"}},
        "param_map": {"operation": 1, "packageName": 2},
        "input_names": ["x", "w"], "output_names": ["mul_out"], "macs_per_inference": "0"
      },
      "norm_node": {
        "package": "qti.aisw", "type": "RmsNorm",
        "tensor_params": {"axes": {"axes_t": {"id": 5, "type": 4, "data_type": 565, "dims": [1], "data": [1]}}},
        "scalar_params": {"epsilon": {"50": 0.001}, "packageName": {"1544": "qti.aisw"}},
        "param_map": {"epsilon": 1, "packageName": 2},
        "input_names": ["mul_out"], "output_names": ["norm_out"], "macs_per_inference": "0"
      }
    }
  }
})json";

static void test_mini_roundtrip() {
    std::printf("\n--- mini net.json: dict schema + scalar_params 通路 ---\n");
    const std::string path = "/tmp/qwen35_mini_net.json";
    CHECK(write_file(path, MINI_NET_JSON), "write mini net.json");

    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t n = loader.load_net_json(path);
    CHECK(n == 2, "load_net_json: 2 op nodes");
    CHECK(gp.get_op_at(1) != nullptr, "op mul_node 存在");

    // scalar 通路: op_data 含 operation=MUL(值 2)
    const OpDef* mul = nullptr;
    gp.for_each_op([&](const OpDef* od) {
        if (od && od->name_tag && std::string(od->name_tag->name() ? od->name_tag->name() : "") == "Eltwise_Binary")
            mul = od;
    });
    CHECK(mul != nullptr, "找到 Eltwise_Binary 节点");
    if (mul) {
        auto sp = unpack_scalar_params(mul->op_data);
        const ScalarParam* op = scalar_get(sp, "operation");
        CHECK(op != nullptr && op->is_numeric && op->as_int() == 2,
              "operation scalar == 2 (MUL) 经 op_data 通路");
        // tensor_param 轴 const 节点存在(RmsNorm axes)
        bool found_axes = false;
        gp.for_each_op([&](const OpDef* od) {
            if (od && od->is_const() && od->const_data_size >= 4) found_axes = true;
        });
        CHECK(found_axes, "tensor_param (axes) 建为 const 节点");
    }

    // serialize 容量保护: 小缓冲 → 返回所需字节
    register_all_ops();
    HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    CHECK(gp.prepare(env) == GraphStatus::Success, "prepare mini");
    {
        std::vector<uint8_t> tiny(0x40, 0);
        size_t need = 0;
        bool ok = gp.serialize(tiny.data(), tiny.size(), need);
        CHECK(!ok && need > tiny.size(), "小缓冲 serialize 返回 false + 所需字节");
        std::vector<uint8_t> big(need, 0);
        size_t got = 0;
        ok = gp.serialize(big.data(), big.size(), got);
        CHECK(ok && got <= big.size(), "足够缓冲 serialize 成功");
        // round-trip
        GraphPrepare gp2;
        CHECK(gp2.deserialize(big.data(), got), "deserialize");
        std::vector<uint8_t> re(need * 2, 0);
        size_t got2 = 0;
        CHECK(gp2.serialize(re.data(), re.size(), got2), "re-serialize");
        CHECK(got2 == got && std::memcmp(big.data(), re.data(), got) == 0,
              "re-serialize 字节确定性");
    }
}

static void test_full_08b() {
    const std::string path = "/disk2/GEHTP/test_models/qwen35_08b/conv/model_net.json";
    std::ifstream f(path);
    if (!f) { std::printf("\n--- 全量 0.8B 测试 SKIP (net.json 不存在) ---\n"); return; }
    f.close();
    std::printf("\n--- 全量 0.8B net.json 往返 (慢) ---\n");

    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t n = loader.load_net_json(path);
    std::printf("  loaded %u op nodes\n", n);
    CHECK(n > 10000, "0.8B 加载 op 节点 > 10000");

    // op 型计数与 op_inventory 一致(20 型)
    {
        // op_inventory.json 实测的 20 种 QNN op 型(loader 另有 *_post_*_shape
        // 合成辅助 const, 不计入)
        const std::set<std::string> expected = {
            "Cast", "Concat", "CumulativeSum", "DepthWiseConv2d", "ElementWiseNeuron",
            "Eltwise_Binary", "Eltwise_Ternary", "Eltwise_Unary", "FullyConnected",
            "Gather", "MatMul", "Pad", "Reduce", "Reshape", "RmsNorm",
            "ScatterNd", "Softmax", "Split", "StridedSlice", "Transpose",
            "Input", "Output"};  // 图边界节点
        std::map<std::string, int> counts;
        std::set<std::string> others;
        gp.for_each_op([&](const OpDef* od) {
            if (!od || od->is_const() || !od->name_tag || !od->name_tag->name()) return;
            std::string nm = od->name_tag->name();
            if (expected.count(nm)) counts[nm]++;
            else {
                // loader 合成的 tensor_param const: *_shape/*_axes/*_pad_amount/*_ranges
                static const char* suffixes[] = {"_shape", "_axes", "_pad_amount", "_ranges"};
                bool is_param_const = false;
                for (const char* sfx : suffixes) {
                    size_t l = std::strlen(sfx);
                    if (nm.size() >= l && nm.compare(nm.size() - l, l, sfx) == 0) is_param_const = true;
                }
                if (!is_param_const) others.insert(nm);
            }
        });
        CHECK(counts.size() == expected.size(), "op 型 20 种 + Input/Output 齐(与 op_inventory 一致)");
        if (!others.empty()) {
            std::printf("  others: ");
            for (const auto& t : others) std::printf("%s ", t.c_str());
            std::printf("\n");
        }
        CHECK(others.empty(), "无 20 型之外的非 shape-const 节点");
        for (const auto& [t, c] : counts)
            std::printf("    %-20s %d\n", t.c_str(), c);
    }

    register_all_ops();
    HexagonNNEnv env;
    env.set_soc_type(75);
    env.set_num_nsps(1);
    CHECK(gp.prepare(env) == GraphStatus::Success, "prepare 0.8B");

    {
        std::vector<uint8_t> small(1u << 20, 0);
        size_t need = 0;
        bool ok = gp.serialize(small.data(), small.size(), need);
        CHECK(!ok && need > small.size(), "1MB 缓冲返回所需字节(容量保护)");
        std::printf("  required=%zu bytes (%.2f GB)\n", need, need / 1e9);
        std::vector<uint8_t> big(need, 0);
        size_t got = 0;
        ok = gp.serialize(big.data(), big.size(), got);
        CHECK(ok, "0.8B serialize 成功");
        GraphPrepare gp2;
        CHECK(gp2.deserialize(big.data(), got), "0.8B deserialize");
        std::vector<uint8_t> re(need + 4096, 0);
        size_t got2 = 0;
        CHECK(gp2.serialize(re.data(), re.size(), got2), "0.8B re-serialize");
        CHECK(got2 == got && std::memcmp(big.data(), re.data(), got) == 0,
              "0.8B round-trip 字节确定性");
    }
}

int main() {
    std::printf("=== test_qwen35_ir (M1) ===\n");
    test_mini_roundtrip();
    test_full_08b();
    std::printf("\n%s (%d failures)\n", failed ? "FAILED" : "ALL PASS", failed);
    return failed ? 1 : 0;
}
