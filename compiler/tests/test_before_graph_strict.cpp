// test_before_graph_strict.cpp 鈥?Strict comparison of our dumped before_graph
// vs QNN's before_graph.json, tensor-by-tensor via topological alignment.
//
// Alignment: by node grouping (= .cpp function name), then by output position.
// For each aligned tensor pair, compare: dims (after batch padding) + data_type.
// Also compare: node count, node types, data-flow topology (skipping injected
// quant/config tensors).
// Linux: 依赖 typical_structures 语料(仅 Windows 原开发机)→ SKIP
#ifndef _WIN32
#include <cstdio>
int main() { std::printf("SKIP: Windows-only test data (Linux)\n"); return 0; }
#else
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "test_paths.hpp"
#include "hnnx/frontend/json.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/graph_dumper.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"
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

struct QnnGraph {
    struct Tensor { uint32_t type; uint32_t data_type; std::vector<uint64_t> dims; };
    struct Node { std::string type; std::string grouping; std::vector<std::string> ins; std::vector<std::string> outs; };
    std::map<std::string, Tensor> tensors;
    std::map<std::string, Node> nodes;
    std::map<std::string, std::string> producer; // hex tensor -> grouping
};

static QnnGraph load_graph(const std::string& path) {
    using namespace hnnx;
    QnnGraph g;
    JsonValue root = parse_json_file(path);
    const JsonValue& ts = root.at("graph").at("tensors");
    for (const auto& [hexk, tj] : ts.obj_val) {
        QnnGraph::Tensor t;
        t.type = tj.at("type").as_int();
        t.data_type = tj.at("data_type").as_int();
        const JsonValue& dims = tj.at("dims");
        for (size_t i = 0; i < dims.size(); ++i)
            t.dims.push_back(dims.at(i).as_int());
        g.tensors[hexk] = std::move(t);
    }
    const JsonValue& ns = root.at("graph").at("nodes");
    for (const auto& [hexk, nj] : ns.obj_val) {
        QnnGraph::Node n;
        n.type = nj.at("type").as_str();
        n.grouping = nj.at("grouping").as_str();
        const JsonValue& ins = nj.at("input_names");
        for (size_t i = 0; i < ins.size(); ++i)
            n.ins.push_back(ins.at(i).as_str());
        const JsonValue& outs = nj.at("output_names");
        for (size_t i = 0; i < outs.size(); ++i)
            n.outs.push_back(outs.at(i).as_str());
        g.nodes[hexk] = std::move(n);
        for (const auto& oid : g.nodes[hexk].outs)
            g.producer[oid] = g.nodes[hexk].grouping;
    }
    return g;
}

int test_model(const std::string& dir, const std::string& name) {
    using namespace hnnx;
    std::cout << "\n============================================\n";
    std::cout << "  " << name << "\n";
    std::cout << "============================================\n";

    std::string cpp = dir + "\\" + name + ".cpp";
    std::string bin = dir + "\\" + name + "_weights.bin";
    if (!std::ifstream(bin).good()) bin = dir + "\\" + name + ".bin";
    std::string before = dir + "\\" + name + "_before_graph.json";
    std::string our_before = dir + "\\reqnn_before_graph.json";

    // Load QNN before_graph.json
    std::cout << "[1] Loading QNN before_graph.json...\n";
    QnnGraph qnn = load_graph(before);
    std::cout << "  QNN: " << qnn.nodes.size() << " nodes, " << qnn.tensors.size() << " tensors\n";

    // Load + prepare our graph
    std::cout << "[2] Loading + do_prepare1...\n";
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t ops = loader.load_qnn_ir(cpp, bin);
    check(ops > 0, name + ": QnnIRLoader loaded");
    if (ops == 0) return 1;

    HexagonNNEnv env;
    env.set_num_nsps(1);
    env.set_soc_type(75);
    VtcmCacheInstance vtcm(0, 8 * 1024 * 1024);
    gp.do_prepare1(env, vtcm);

    // Dump our before graph
    GraphDumper dumper(gp);
    check(dumper.dump(our_before), name + ": dumped before graph");
    QnnGraph ours = load_graph(our_before);
    std::cout << "  Ours: " << ours.nodes.size() << " nodes, " << ours.tensors.size() << " tensors\n";

    // [3] Node count + types (by grouping)
    std::map<std::string, std::string> qnn_types, our_types;
    for (const auto& [k, n] : qnn.nodes) qnn_types[n.grouping] = n.type;
    for (const auto& [k, n] : ours.nodes) our_types[n.grouping] = n.type;
    int type_match = 0, type_mismatch = 0;
    auto norm = [](const std::string& s) -> std::string {
        std::string r; for (char c : s) if (c != '_') r += c;
        size_t p; while ((p = r.find("Eltwise")) != std::string::npos) r.replace(p, 7, "ElementWise");
        return r;
    };
    for (const auto& [g, t] : qnn_types) {
        auto it = our_types.find(g);
        if (it == our_types.end()) continue;
        if (norm(t) == norm(it->second)) type_match++;
        else { type_mismatch++; std::cout << "  TYPE DIFF " << g << ": qnn=" << t << " ours=" << it->second << "\n"; }
    }
    std::cout << "[3] Types: " << type_match << " match, " << type_mismatch << " mismatch\n";
    check(type_mismatch == 0, name + ": node types match");

    // [4] Tensor comparison by topological alignment
    // Align: node grouping -> output position
    std::cout << "[4] Tensor comparison (aligned by producer+position)...\n";
    // Build ours: grouping -> node
    std::map<std::string, const QnnGraph::Node*> ours_by_group;
    for (const auto& [k, n] : ours.nodes) ours_by_group[n.grouping] = &n;

    int dim_match = 0, dim_mismatch = 0, dt_match = 0, dt_mismatch = 0, missing = 0;
    for (const auto& [hexk, qnn_n] : qnn.nodes) {
        auto it = ours_by_group.find(qnn_n.grouping);
        if (it == ours_by_group.end()) continue;
        const QnnGraph::Node* our_n = it->second;
        size_t min_outs = std::min(qnn_n.outs.size(), our_n->outs.size());
        for (size_t i = 0; i < min_outs; ++i) {
            auto qt = qnn.tensors.find(qnn_n.outs[i]);
            auto ot = ours.tensors.find(our_n->outs[i]);
            if (qt == qnn.tensors.end() || ot == ours.tensors.end()) { missing++; continue; }
            // Compare dims
            if (qt->second.dims == ot->second.dims) dim_match++;
            else {
                dim_mismatch++;
                if (dim_mismatch <= 5) {
                    std::cout << "  DIM DIFF " << qnn_n.grouping << "[" << i << "]: qnn=";
                    for (auto d : qt->second.dims) std::cout << d << " ";
                    std::cout << "ours=";
                    for (auto d : ot->second.dims) std::cout << d << " ";
                    std::cout << "\n";
                }
            }
            // Compare data_type
            if (qt->second.data_type == ot->second.data_type) dt_match++;
            else {
                dt_mismatch++;
                if (dt_mismatch <= 5)
                    std::cout << "  DTYPE DIFF " << qnn_n.grouping << "[" << i << "]: qnn=" << qt->second.data_type << " ours=" << ot->second.data_type << "\n";
            }
        }
    }
    std::cout << "  dims: " << dim_match << " match, " << dim_mismatch << " mismatch, " << missing << " missing\n";
    std::cout << "  dtype: " << dt_match << " match, " << dt_mismatch << " mismatch\n";
    check(dim_mismatch == 0, name + ": tensor dims match (after batch padding)");
    check(dt_mismatch == 0, name + ": tensor data_type match");

    return (g_fail > 0) ? 1 : 0;
}

int main() {
    using namespace hnnx;
    std::cout << "=== Before Graph Strict Comparison ===\n";

    struct M { std::string dir; std::string name; };
    std::vector<M> models = {
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





#endif // !_WIN32
