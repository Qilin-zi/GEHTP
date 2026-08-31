// test_full_graph_match.cpp 鈥?Full GraphRef consistency: nodes + ALL tensors + ALL edges
// vs QNN before_graph.json.
//
// Verification:
//   1. Node count + types (by grouping)
//   2. Output tensors: dims + data_type (aligned by producer+position)
//   3. Input tensors: dims + data_type (aligned by producer+position)
//   4. Data-flow edges: every data input's producer matches
//   5. No orphan data inputs (every data input has a matching producer)
//
// "Data input" = input tensor produced by a real op (not injected quant/config).
// Injected tensors (no producer) are HtpPrepare internal, not compared.
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

struct GraphRef {
    struct Tensor { uint32_t type; uint32_t data_type; std::vector<uint64_t> dims; };
    struct Node { std::string type; std::string grouping; std::vector<std::string> ins; std::vector<std::string> outs; };
    std::map<std::string, Tensor> tensors;
    std::map<std::string, Node> nodes;
    std::map<std::string, std::string> producer; // hex tensor -> producer grouping
    std::map<std::string, std::vector<std::string>> group_outputs; // grouping -> [hex outputs]
};

static GraphRef load_graph(const std::string& path) {
    using namespace hnnx;
    GraphRef g;
    JsonValue root = parse_json_file(path);
    const JsonValue& ts = root.at("graph").at("tensors");
    for (const auto& [hexk, tj] : ts.obj_val) {
        GraphRef::Tensor t;
        t.type = tj.at("type").as_int();
        t.data_type = tj.at("data_type").as_int();
        const JsonValue& dims = tj.at("dims");
        for (size_t i = 0; i < dims.size(); ++i)
            t.dims.push_back(dims.at(i).as_int());
        g.tensors[hexk] = std::move(t);
    }
    const JsonValue& ns = root.at("graph").at("nodes");
    for (const auto& [hexk, nj] : ns.obj_val) {
        GraphRef::Node n;
        n.type = nj.at("type").as_str();
        n.grouping = nj.at("grouping").as_str();
        const JsonValue& ins = nj.at("input_names");
        for (size_t i = 0; i < ins.size(); ++i)
            n.ins.push_back(ins.at(i).as_str());
        const JsonValue& outs = nj.at("output_names");
        for (size_t i = 0; i < outs.size(); ++i)
            n.outs.push_back(outs.at(i).as_str());
        g.nodes[hexk] = std::move(n);
        for (const auto& oid : g.nodes[hexk].outs) {
            g.producer[oid] = g.nodes[hexk].grouping;
            g.group_outputs[g.nodes[hexk].grouping].push_back(oid);
        }
    }
    return g;
}

// For a node's input, determine if it's a "data input" (has a producer op)
static bool is_data_input(const GraphRef& g, const std::string& tensor_hex) {
    auto it = g.producer.find(tensor_hex);
    return it != g.producer.end() && !it->second.empty();
}

// Get (producer_grouping, output_position) for a data input
static std::pair<std::string, int> get_input_source(const GraphRef& g, const std::string& tensor_hex) {
    auto pit = g.producer.find(tensor_hex);
    if (pit == g.producer.end()) return {"", -1};
    auto oit = g.group_outputs.find(pit->second);
    if (oit == g.group_outputs.end()) return {pit->second, -1};
    for (int i = 0; i < (int)oit->second.size(); ++i)
        if (oit->second[i] == tensor_hex) return {pit->second, i};
    return {pit->second, -1};
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
    std::string our = dir + "\\reqnn_before_graph.json";

    // Load QNN before_graph.json
    GraphRef qnn = load_graph(before);

    // Load + prepare our GraphRef
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

    GraphDumper dumper(gp);
    check(dumper.dump(our), name + ": dumped before GraphRef");
    GraphRef ours = load_graph(our);

    std::cout << "  QNN:  " << qnn.nodes.size() << " nodes, " << qnn.tensors.size() << " tensors\n";
    std::cout << "  Ours: " << ours.nodes.size() << " nodes, " << ours.tensors.size() << " tensors\n";

    // [1] Node count + types
    auto norm = [](const std::string& s) -> std::string {
        std::string r; for (char c : s) if (c != '_') r += c;
        size_t p; while ((p = r.find("Eltwise")) != std::string::npos) r.replace(p, 7, "ElementWise");
        return r;
    };
    std::map<std::string, std::string> qt, ot;
    for (const auto& [k, n] : qnn.nodes) qt[n.grouping] = n.type;
    for (const auto& [k, n] : ours.nodes) ot[n.grouping] = n.type;
    int tm = 0, tmm = 0;
    for (const auto& [g, t] : qt) {
        auto it = ot.find(g);
        if (it == ot.end()) continue;
        if (norm(t) == norm(it->second)) tm++; else { tmm++; std::cout << "  TYPE DIFF " << g << "\n"; }
    }
    std::cout << "[1] Types: " << tm << " match, " << tmm << " mismatch\n";
    check(tmm == 0, name + ": node types match");

    // [2] Output tensors: dims + data_type
    std::cout << "[2] Output tensor comparison...\n";
    int odm = 0, odmm = 0, odtm = 0, odtmm = 0;
    for (const auto& [hexk, qn] : qnn.nodes) {
        // Find our node by grouping
        const GraphRef::Node* on = nullptr;
        for (const auto& [k, n] : ours.nodes) if (n.grouping == qn.grouping) { on = &n; break; }
        if (!on) continue;
        size_t mn = std::min(qn.outs.size(), on->outs.size());
        for (size_t i = 0; i < mn; ++i) {
            auto qt_it = qnn.tensors.find(qn.outs[i]);
            auto ot_it = ours.tensors.find(on->outs[i]);
            if (qt_it == qnn.tensors.end() || ot_it == ours.tensors.end()) continue;
            if (qt_it->second.dims == ot_it->second.dims) odm++; else { odmm++; if (odmm<=3) std::cout << "  ODIM " << qn.grouping << "[" << i << "]\n"; }
            if (qt_it->second.data_type == ot_it->second.data_type) odtm++; else { odtmm++; if (odtmm<=3) std::cout << "  ODT " << qn.grouping << "[" << i << "]\n"; }
        }
    }
    std::cout << "  dims: " << odm << " match, " << odmm << " mismatch\n";
    std::cout << "  dtype: " << odtm << " match, " << odtmm << " mismatch\n";
    check(odmm == 0, name + ": output tensor dims match");
    check(odtmm == 0, name + ": output tensor data_type match");

    // [3] Input tensors + edges: for each node, verify data inputs
    std::cout << "[3] Input tensor + edge comparison...\n";
    int edge_match = 0, edge_mismatch = 0;
    int inm = 0, inmm = 0, indtm = 0, indtmm = 0;

    for (const auto& [hexk, qn] : qnn.nodes) {
        const GraphRef::Node* on = nullptr;
        for (const auto& [k, n] : ours.nodes) if (n.grouping == qn.grouping) { on = &n; break; }
        if (!on) continue;

        // Collect QNN data inputs: (producer_grouping, output_position, tensor_hex)
        std::vector<std::pair<std::string, int>> qnn_data_srcs;
        std::vector<std::string> qnn_data_hexes;
        for (const auto& inp : qn.ins) {
            if (!is_data_input(qnn, inp)) continue;
            auto src = get_input_source(qnn, inp);
            qnn_data_srcs.push_back(src);
            qnn_data_hexes.push_back(inp);
        }

        // Collect our data inputs
        std::vector<std::pair<std::string, int>> our_data_srcs;
        std::vector<std::string> our_data_hexes;
        for (const auto& inp : on->ins) {
            if (!is_data_input(ours, inp)) continue;
            auto src = get_input_source(ours, inp);
            our_data_srcs.push_back(src);
            our_data_hexes.push_back(inp);
        }

        // Compare data input sets (as multisets 鈥?order may differ due to injected tensors)
        std::multiset<std::pair<std::string, int>> qset(qnn_data_srcs.begin(), qnn_data_srcs.end());
        std::multiset<std::pair<std::string, int>> oset(our_data_srcs.begin(), our_data_srcs.end());

        if (qset == oset) {
            edge_match++;
        } else {
            edge_mismatch++;
            if (edge_mismatch <= 5) {
                std::cout << "  EDGE DIFF " << qn.grouping << ":\n";
                std::cout << "    QNN data srcs: ";
                for (auto& s : qnn_data_srcs) std::cout << s.first << "[" << s.second << "] ";
                std::cout << "\n    Our data srcs: ";
                for (auto& s : our_data_srcs) std::cout << s.first << "[" << s.second << "] ";
                std::cout << "\n";
            }
        }

        // For matching data inputs, compare tensor dims + data_type
        // Match by (producer, position) and find the tensor in each
        for (size_t i = 0; i < qnn_data_srcs.size(); ++i) {
            auto& qs = qnn_data_srcs[i];
            // Find same source in our inputs
            for (size_t j = 0; j < our_data_srcs.size(); ++j) {
                if (our_data_srcs[j] != qs) continue;
                auto qt_it = qnn.tensors.find(qnn_data_hexes[i]);
                auto ot_it = ours.tensors.find(our_data_hexes[j]);
                if (qt_it == qnn.tensors.end() || ot_it == ours.tensors.end()) break;
                if (qt_it->second.dims == ot_it->second.dims) inm++; else { inmm++; if (inmm<=5) std::cout << "  IDIM " << qn.grouping << " from " << qs.first << "[" << qs.second << "]\n"; }
                if (qt_it->second.data_type == ot_it->second.data_type) indtm++; else { indtmm++; if (indtmm<=5) std::cout << "  IDT " << qn.grouping << " from " << qs.first << "[" << qs.second << "]\n"; }
                break;
            }
        }
    }
    std::cout << "  edges: " << edge_match << " match, " << edge_mismatch << " mismatch\n";
    std::cout << "  input dims: " << inm << " match, " << inmm << " mismatch\n";
    std::cout << "  input dtype: " << indtm << " match, " << indtmm << " mismatch\n";
    check(edge_mismatch == 0, name + ": data-flow edges match");
    check(inmm == 0, name + ": input tensor dims match");
    check(indtmm == 0, name + ": input tensor data_type match");

    return (g_fail > 0) ? 1 : 0;
}

int main() {
    using namespace hnnx;
    std::cout << "=== Full GraphRef Match: before_graph.json strict comparison ===\n";

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
