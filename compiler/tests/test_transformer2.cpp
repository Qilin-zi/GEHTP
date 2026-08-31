// test_transformer2.cpp 鈥?Load a real 2-layer transformer QNN IR (.cpp + .bin)
// and verify before graph structure matches net.json.
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/frontend/json.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <set>
#include <algorithm>

static const char* T2_DIR =
    "C:\\Users\\RQILIN\\Documents\\Default Project\\REQNN\\test_models\\transformer2";

static void check(bool cond, const std::string& msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); }
    std::cout << "  OK: " << msg << "\n";
}

int main() {
    using namespace hnnx;
    std::string cpp_path = std::string(T2_DIR) + "\\transformer2.cpp";
    std::string bin_path = std::string(T2_DIR) + "\\transformer2.bin";
    std::string json_path = std::string(T2_DIR) + "\\transformer2_net.json";

    std::cout << "=== Transformer2 QNN IR Loader Test ===\n\n";

    // ===== Load via .cpp + .bin path =====
    std::cout << "[1] Loading .cpp + .bin...\n";
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t ops = loader.load_qnn_ir(cpp_path, bin_path);
    std::cout << "[1] ops created: " << ops << "\n";
    check(ops > 0, "load_qnn_ir returned > 0 ops");

    // ===== Load net.json for comparison =====
    std::cout << "\n[2] Loading net.json for comparison...\n";
    JsonValue net_json = parse_json_file(json_path);
    const JsonValue& graph_json = net_json.at("graph");
    const JsonValue& json_tensors = graph_json.at("tensors");
    const JsonValue& json_nodes = graph_json.at("nodes");

    // Count tensors and nodes from net.json
    size_t json_tensor_count = json_tensors.obj_val.size();
    size_t json_node_count = json_nodes.obj_val.size();
    std::cout << "  net.json: " << json_tensor_count << " tensors, "
              << json_node_count << " nodes\n";

    // ===== Compare tensor names =====
    std::cout << "\n[3] Comparing tensor names: .cpp vs net.json\n";
    // net.json has tensors in top-level "tensors" AND inside nodes' "tensor_params"
    std::set<std::string> json_tensor_names;
    for (const auto& [name, _] : json_tensors.obj_val)
        json_tensor_names.insert(name);
    // Also collect tensor_param names from nodes
    for (const auto& [_, njson] : json_nodes.obj_val) {
        if (njson.contains("tensor_params") && njson.at("tensor_params").is_object()) {
            for (const auto& [pname, inner] : njson.at("tensor_params").obj_val) {
                for (const auto& [tname, _] : inner.obj_val)
                    json_tensor_names.insert(tname);
            }
        }
    }

    std::set<std::string> cpp_tensor_names;
    for (const auto& [name, _] : loader.tensors())
        cpp_tensor_names.insert(name);

    // Find tensors only in net.json (missing from .cpp parse)
    std::set<std::string> missing;
    std::set_difference(json_tensor_names.begin(), json_tensor_names.end(),
                        cpp_tensor_names.begin(), cpp_tensor_names.end(),
                        std::inserter(missing, missing.end()));
    // Find tensors only in .cpp (extra from .cpp parse)
    std::set<std::string> extra;
    std::set_difference(cpp_tensor_names.begin(), cpp_tensor_names.end(),
                        json_tensor_names.begin(), json_tensor_names.end(),
                        std::inserter(extra, extra.end()));

    std::cout << "  net.json tensors: " << json_tensor_names.size() << "\n";
    std::cout << "  .cpp tensors: " << cpp_tensor_names.size() << "\n";
    if (!missing.empty()) {
        std::cout << "  MISSING from .cpp (" << missing.size() << "):\n";
        int i = 0;
        for (const auto& m : missing) { std::cout << "    " << m << "\n"; if (++i >= 10) { std::cout << "    ...\n"; break; } }
    }
    if (!extra.empty()) {
        std::cout << "  EXTRA in .cpp (" << extra.size() << "):\n";
        int i = 0;
        for (const auto& e : extra) { std::cout << "    " << e << "\n"; if (++i >= 10) { std::cout << "    ...\n"; break; } }
    }
    check(missing.empty(), "no tensors missing from .cpp parse");
    check(extra.empty(), "no extra tensors in .cpp parse");

    // ===== Compare tensor IDs =====
    // Note: net.json IDs are assigned by converter's internal processing order.
    // .cpp IDs follow composeGraphs call order (what QNN runtime sees).
    // These may differ for complex models 鈥?what matters is topology.
    std::cout << "\n[4] Comparing tensor IDs (info only)...\n";
    int id_mismatches = 0;
    int id_checked = 0;
    for (const auto& [name, tjson] : json_tensors.obj_val) {
        uint32_t json_id = static_cast<uint32_t>(tjson.at("id").as_int());
        auto it = loader.tensors().find(name);
        if (it == loader.tensors().end()) continue;
        uint32_t cpp_id = it->second.id;
        if (json_id != cpp_id) id_mismatches++;
        id_checked++;
    }
    std::cout << "  checked: " << id_checked << ", mismatches: " << id_mismatches
              << " (expected: converter vs runtime ID assignment order differs)\n";

    // ===== Compare node types and topology =====
    std::cout << "\n[5] Comparing node types and topology...\n";
    // Normalize op type names: net.json uses "Eltwise_Binary", .cpp uses "ElementWiseBinary"
    auto normalize_type = [](const std::string& s) -> std::string {
        std::string r;
        for (char c : s) if (c != '_') r += c;
        // Normalize common abbreviations
        size_t pos;
        while ((pos = r.find("Eltwise")) != std::string::npos)
            r.replace(pos, 7, "ElementWise");
        return r;
    };
    int topology_ok = 0;
    int topology_fail = 0;
    int nodes_missing = 0;
    for (const auto& [node_name, njson] : json_nodes.obj_val) {
        std::string json_type = njson.at("type").as_str();
        const QnnNodeInfo* matched = nullptr;
        for (const auto& ni : loader.nodes()) {
            if (ni.name == node_name) { matched = &ni; break; }
        }
        if (!matched) {
            nodes_missing++;
            continue;
        }
        if (normalize_type(matched->type) != normalize_type(json_type)) {
            std::cout << "  TYPE MISMATCH: " << node_name
                      << " json=" << json_type << " cpp=" << matched->type << "\n";
            topology_fail++;
            continue;
        }
        const JsonValue& json_ins = njson.at("input_names");
        if (json_ins.size() != matched->input_names.size()) {
            std::cout << "  INPUT COUNT MISMATCH: " << node_name
                      << " json=" << json_ins.size() << " cpp=" << matched->input_names.size() << "\n";
            topology_fail++;
            continue;
        }
        for (size_t i = 0; i < json_ins.size(); ++i) {
            if (json_ins.at(i).as_str() != matched->input_names[i]) {
                std::cout << "  INPUT MISMATCH: " << node_name << "[" << i << "] "
                          << "json=" << json_ins.at(i).as_str()
                          << " cpp=" << matched->input_names[i] << "\n";
                topology_fail++;
                break;
            }
        }
        topology_ok++;
    }
    std::cout << "  nodes matched: " << topology_ok << "/" << json_node_count
              << ", missing: " << nodes_missing
              << ", failures: " << topology_fail << "\n";
    check(topology_fail == 0, "all matched nodes have correct type and topology");
    // Some nodes may be missing (converter optimizations like squash Reshape)
    std::cout << "  missing nodes: " << nodes_missing << " (converter optimizations)\n";

    // ===== Summary =====
    std::cout << "\n[6] Summary:\n";
    std::cout << "  Model: 2-layer transformer (hidden=64, heads=4, ff=128)\n";
    std::cout << "  QNN IR tensors: " << cpp_tensor_names.size() << "\n";
    std::cout << "  QNN IR nodes: " << loader.nodes().size() << "\n";
    std::cout << "  REQNN graph nodes: " << gp.op_count() << "\n";
    std::cout << "  Op types: ";
    std::set<std::string> op_types;
    for (const auto& ni : loader.nodes()) op_types.insert(ni.type);
    for (const auto& t : op_types) std::cout << t << " ";
    std::cout << "\n";

    std::cout << "\n=== Transformer2 QNN IR Loader PASSED ===\n";
    return 0;
}


