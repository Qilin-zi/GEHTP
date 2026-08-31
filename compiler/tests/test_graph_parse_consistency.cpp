// test_graph_parse_consistency.cpp 鈥?Verify QnnIRLoader graph parsing is correct
// by cross-checking two independent parses of the same converter output:
//   Path A: load_net_json(_net.json)   鈥?parse JSON descriptor
//   Path B: load_cpp(.cpp + .bin)       鈥?parse C source
// Both are produced by the same qnn-onnx-converter run, so they MUST agree.
//
// Verification per model:
//   1. Both paths load successfully (ops > 0)
//   2. Same node count
//   3. Same node types (by node name)
//   4. Same node input/output tensor NAMES (not hex IDs)
//   5. Same tensor count
//   6. Same tensor dims per tensor name
//   7. Same tensor data_type per tensor name
//
// This tests the PARSING (stage 2a: .cpp/.json -> in-memory graph) end-to-end.
// before_graph.json is NOT the baseline 鈥?that's HtpPrepare's post-import state
// with injected quant tensors and reassigned hex IDs, a different layer.
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/frontend/json.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const std::string& msg) {
    if (cond) { g_pass++; std::cout << "  OK: " << msg << "\n"; }
    else { g_fail++; std::cerr << "FAIL: " << msg << "\n"; }
}

int test_model(const std::string& dir, const std::string& name) {
    using namespace hnnx;
    std::cout << "\n============================================\n";
    std::cout << "  " << name << "\n";
    std::cout << "============================================\n";

    std::string netjson_path = dir + "\\" + name + "_net.json";
    std::string cpp_path = dir + "\\" + name + ".cpp";
    std::string bin_path = dir + "\\" + name + "_weights.bin";
    if (!std::ifstream(bin_path).good()) bin_path = dir + "\\" + name + ".bin";

    // Path A: load_net_json
    std::cout << "[A] load_net_json(" << name << "_net.json)...\n";
    GraphPrepare gpA;
    QnnIRLoader loaderA(gpA);
    uint32_t opsA = 0;
    try {
        opsA = loaderA.load_net_json(netjson_path);
    } catch (std::exception& e) {
        std::cerr << "  load_net_json exception: " << e.what() << "\n";
    }
    std::cout << "  opsA=" << opsA << " nodes=" << loaderA.nodes().size()
              << " tensors=" << loaderA.tensors().size() << "\n";
    check(opsA > 0, name + ": load_net_json succeeded");

    // Path B: load_cpp
    std::cout << "[B] load_cpp(.cpp + .bin)...\n";
    GraphPrepare gpB;
    QnnIRLoader loaderB(gpB);
    uint32_t opsB = loaderB.load_qnn_ir(cpp_path, bin_path);
    std::cout << "  opsB=" << opsB << " nodes=" << loaderB.nodes().size()
              << " tensors=" << loaderB.tensors().size() << "\n";
    check(opsB > 0, name + ": load_cpp succeeded");

    if (opsA == 0 || opsB == 0) return 1;

    // [1] Node count
    std::cout << "[1] Node count: A=" << loaderA.nodes().size()
              << " B=" << loaderB.nodes().size() << "\n";
    check(loaderA.nodes().size() == loaderB.nodes().size(),
          name + ": same node count");

    // [2] Node types by name (normalize Eltwise_Binary <-> ElementWiseBinary)
    auto norm_type = [](const std::string& s) -> std::string {
        std::string r;
        for (char c : s) if (c != '_') r += c;
        size_t pos;
        while ((pos = r.find("Eltwise")) != std::string::npos)
            r.replace(pos, 7, "ElementWise");
        return r;
    };
    std::map<std::string, std::string> typesA, typesB;
    for (const auto& n : loaderA.nodes()) typesA[n.name] = n.type;
    for (const auto& n : loaderB.nodes()) typesB[n.name] = n.type;
    int type_match = 0, type_mismatch = 0;
    for (const auto& [nm, ty] : typesA) {
        auto it = typesB.find(nm);
        if (it == typesB.end()) continue;
        if (norm_type(it->second) == norm_type(ty)) type_match++;
        else { type_mismatch++; std::cout << "  TYPE DIFF " << nm << ": A=" << ty << " B=" << it->second << "\n"; }
    }
    std::cout << "[2] Types: match=" << type_match << " mismatch=" << type_mismatch << "\n";
    check(type_mismatch == 0, name + ": node types match");

    // [3] Input/output tensor names per node
    int io_mismatch = 0;
    for (const auto& nA : loaderA.nodes()) {
        for (const auto& nB : loaderB.nodes()) {
            if (nA.name != nB.name) continue;
            if (nA.input_names != nB.input_names || nA.output_names != nB.output_names) {
                io_mismatch++;
                if (io_mismatch <= 5) {
                    std::cout << "  IO DIFF " << nA.name << ":\n";
                    if (nA.input_names != nB.input_names) {
                        std::cout << "    A in: "; for (auto& s:nA.input_names) std::cout << s << " "; std::cout << "\n";
                        std::cout << "    B in: "; for (auto& s:nB.input_names) std::cout << s << " "; std::cout << "\n";
                    }
                    if (nA.output_names != nB.output_names) {
                        std::cout << "    A out: "; for (auto& s:nA.output_names) std::cout << s << " "; std::cout << "\n";
                        std::cout << "    B out: "; for (auto& s:nB.output_names) std::cout << s << " "; std::cout << "\n";
                    }
                }
            }
            break;
        }
    }
    std::cout << "[3] IO mismatches: " << io_mismatch << "\n";
    check(io_mismatch == 0, name + ": input/output tensor names match");

    // [4] Tensor count
    std::cout << "[4] Tensors: A=" << loaderA.tensors().size()
              << " B=" << loaderB.tensors().size() << "\n";
    check(loaderA.tensors().size() == loaderB.tensors().size(),
          name + ": same tensor count");

    // [5] Tensor dims + data_type by name
    int dim_mismatch = 0, dt_mismatch = 0, tmiss = 0;
    for (const auto& [tname, tA] : loaderA.tensors()) {
        auto itB = loaderB.tensors().find(tname);
        if (itB == loaderB.tensors().end()) { tmiss++; continue; }
        if (tA.dims != itB->second.dims) {
            dim_mismatch++;
            if (dim_mismatch <= 5) {
                std::cout << "  DIM DIFF " << tname << ": A=";
                for (auto d:tA.dims) std::cout << d << " ";
                std::cout << "B=";
                for (auto d:itB->second.dims) std::cout << d << " ";
                std::cout << "\n";
            }
        }
        if (tA.data_type != itB->second.data_type) {
            dt_mismatch++;
            if (dt_mismatch <= 5)
                std::cout << "  DTYPE DIFF " << tname << ": A=" << tA.data_type << " B=" << itB->second.data_type << "\n";
        }
    }
    std::cout << "[5] Tensor: missing=" << tmiss << " dim_mismatch=" << dim_mismatch
              << " dtype_mismatch=" << dt_mismatch << "\n";
    check(tmiss == 0, name + ": no tensors missing");
    check(dim_mismatch == 0, name + ": tensor dims match");
    check(dt_mismatch == 0, name + ": tensor data_type match");

    return (g_fail > 0) ? 1 : 0;
}

int main() {
    using namespace hnnx;
    std::cout << "=== Graph Parse Consistency: load_net_json vs load_cpp ===\n";

    struct Model { std::string dir; std::string name; };
    std::vector<Model> models = {
        {R"(C:\Users\RQILIN\Documents\Default Project\REQNN\test_models\transformer2)", "transformer2"},
        {R"(C:\Users\RQILIN\Documents\Default Project\test_models\typical_structures\resnet50_block)", "resnet50_block"},
        {R"(C:\Users\RQILIN\Documents\Default Project\test_models\typical_structures\mobilenetv2_block)", "mobilenetv2_block"},
        {R"(C:\Users\RQILIN\Documents\Default Project\test_models\typical_structures\yolov8_block)", "yolov8_block"},
        {R"(C:\Users\RQILIN\Documents\Default Project\test_models\typical_structures\mobilebert_block)", "mobilebert_block"},
    };

    int fails = 0;
    for (const auto& m : models) {
        int before = g_fail;
        test_model(m.dir, m.name);
        if (g_fail > before) fails++;
    }

    std::cout << "\n============================================\n";
    std::cout << "  " << (models.size() - fails) << "/" << models.size()
              << " models passed, " << g_pass << " OK, " << g_fail << " FAIL\n";
    std::cout << "============================================\n";
    return fails > 0 ? 1 : 0;
}




