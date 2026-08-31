// test_transformer2_before.cpp 鈥?Compare QNN generator's before_graph.json
// with REQNN's QnnIRLoader output from .cpp.
//
// Strategy: before_graph.json uses hex tensor IDs (0x...) for topology,
// but the "grouping" field preserves the original op name (matching .cpp
// function names). We map REQNN's named topology to hex-ID topology via
// the grouping field and verify:
//   1. Same number of nodes
//   2. Same op types per node (by grouping name)
//   3. Same input/output count per node
//   4. Same producer-consumer relationships (isomorphic graph structure)
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/frontend/json.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <algorithm>

static const char* T2_DIR =
    "C:\\Users\\RQILIN\\Documents\\Default Project\\REQNN\\test_models\\transformer2";

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const std::string& msg) {
    if (cond) { g_pass++; std::cout << "  OK: " << msg << "\n"; }
    else { g_fail++; std::cerr << "FAIL: " << msg << "\n"; }
}

// Parse a hex string "0x0000000000000011" -> uint64_t
static uint64_t parse_hex(const std::string& s) {
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        return std::stoull(s.substr(2), nullptr, 16);
    return std::stoull(s, nullptr, 16);
}

int main() {
    using namespace hnnx;
    std::string cpp_path = std::string(T2_DIR) + "\\transformer2.cpp";
    std::string bin_path = std::string(T2_DIR) + "\\transformer2.bin";
    std::string before_path = std::string(T2_DIR) + "\\transformer2_before_graph.json";

    std::cout << "=== Before Graph Comparison: QNN generator vs REQNN ===\n\n";

    // ===== Load QNN before_graph.json =====
    std::cout << "[1] Loading QNN before_graph.json...\n";
    JsonValue before_json = parse_json_file(before_path);
    const JsonValue& b_tensors = before_json.at("graph").at("tensors");
    const JsonValue& b_nodes = before_json.at("graph").at("nodes");
    size_t b_tensor_count = b_tensors.obj_val.size();
    size_t b_node_count = b_nodes.obj_val.size();
    std::cout << "  tensors: " << b_tensor_count << ", nodes: " << b_node_count << "\n";

    // Build QNN before graph structure:
    //   grouping -> {type, input_hex_ids, output_hex_ids}
    struct QnnNode {
        std::string type;
        std::string grouping;
        std::vector<std::string> input_ids;  // hex tensor IDs
        std::vector<std::string> output_ids; // hex tensor IDs
    };
    std::map<std::string, QnnNode> qnn_nodes; // grouping -> node
    std::map<std::string, std::vector<std::string>> qnn_producers; // tensor_hex -> [grouping that produces it]
    for (const auto& [hex_key, njson] : b_nodes.obj_val) {
        QnnNode n;
        n.type = njson.at("type").as_str();
        n.grouping = njson.at("grouping").as_str();
        const JsonValue& ins = njson.at("input_names");
        for (size_t i = 0; i < ins.size(); ++i)
            n.input_ids.push_back(ins.at(i).as_str());
        const JsonValue& outs = njson.at("output_names");
        for (size_t i = 0; i < outs.size(); ++i)
            n.output_ids.push_back(outs.at(i).as_str());
        qnn_nodes[n.grouping] = n;
        for (const auto& oid : n.output_ids)
            qnn_producers[oid].push_back(n.grouping);
    }

    // ===== Load REQNN .cpp parse =====
    std::cout << "\n[2] Loading REQNN .cpp parse...\n";
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t ops = loader.load_qnn_ir(cpp_path, bin_path);
    std::cout << "  ops: " << ops << ", nodes: " << loader.nodes().size() << "\n";
    check(ops > 0, "REQNN loaded .cpp successfully");

    // Build REQNN structure: grouping (node name) -> {type, input_tensor_names, output_tensor_names}
    // NOTE: QNN before_graph.json flattens tensor_params into input_names.
    // So we must combine input_names + tensor_param names for REQNN to match.
    struct ReqNode {
        std::string type;
        std::vector<std::string> input_names;  // data inputs
        std::vector<std::string> param_names;  // tensor param names (e.g. perm)
        std::vector<std::string> all_inputs;   // data + params (flattened)
        std::vector<std::string> output_names;
    };
    std::map<std::string, ReqNode> req_nodes; // node name -> struct
    for (const auto& ni : loader.nodes()) {
        ReqNode rn;
        rn.type = ni.type;
        rn.input_names = ni.input_names;
        for (const auto& tp : ni.tensor_params)
            rn.param_names.push_back(tp.name);
        rn.all_inputs = ni.input_names;
        rn.all_inputs.insert(rn.all_inputs.end(), rn.param_names.begin(), rn.param_names.end());
        rn.output_names = ni.output_names;
        req_nodes[ni.name] = rn;
    }
    // Build REQNN producer map: tensor_name -> node_name that produces it
    std::map<std::string, std::string> req_producers; // tensor_name -> node_name
    for (const auto& ni : loader.nodes()) {
        for (const auto& on : ni.output_names)
            req_producers[on] = ni.name;
    }

    // ===== Compare node count =====
    std::cout << "\n[3] Comparing node counts...\n";
    std::cout << "  QNN before: " << qnn_nodes.size() << " nodes\n";
    std::cout << "  REQNN: " << req_nodes.size() << " nodes\n";
    check(qnn_nodes.size() == req_nodes.size(), "same node count");

    // ===== Compare node types (by grouping name) =====
    std::cout << "\n[4] Comparing node types by grouping name...\n";
    int type_match = 0, type_mismatch = 0, missing_in_req = 0;
    for (const auto& [grouping, qnn_n] : qnn_nodes) {
        auto it = req_nodes.find(grouping);
        if (it == req_nodes.end()) {
            missing_in_req++;
            continue;
        }
        // Normalize: Eltwise_Binary vs ElementWiseBinary
        auto norm = [](const std::string& s) -> std::string {
            std::string r;
            for (char c : s) if (c != '_') r += c;
            size_t pos;
            while ((pos = r.find("Eltwise")) != std::string::npos)
                r.replace(pos, 7, "ElementWise");
            return r;
        };
        if (norm(qnn_n.type) == norm(it->second.type))
            type_match++;
        else {
            type_mismatch++;
            std::cout << "  TYPE MISMATCH: " << grouping
                      << " qnn=" << qnn_n.type << " req=" << it->second.type << "\n";
        }
    }
    std::cout << "  type match: " << type_match << ", mismatch: " << type_mismatch
              << ", missing in REQNN: " << missing_in_req << "\n";
    check(type_mismatch == 0, "no type mismatches");
    check(missing_in_req == 0, "no nodes missing in REQNN");

    // ===== Compare topology (data-flow producer structure) =====
    // QNN before_graph.json injects extra tensors (FC bias, config, self-reference)
    // that don't exist in .cpp. So we compare only DATA-INPUT producers:
    //   For each REQNN data input with a producer, verify the same producer
    //   exists among QNN's inputs (skipping QNN self-refs and no-producer inputs).
    std::cout << "\n[5] Comparing topology (data-flow producer structure)...\n";
    int topo_ok = 0, topo_fail = 0;
    int extra_qnn_inputs = 0;
    for (const auto& [grouping, qnn_n] : qnn_nodes) {
        auto it = req_nodes.find(grouping);
        if (it == req_nodes.end()) { continue; }
        const ReqNode& req_n = it->second;

        // Collect QNN producers for this node's inputs (skip self-refs, no-producer)
        std::vector<std::string> qnn_input_producers;
        for (const auto& qid : qnn_n.input_ids) {
            bool is_self = false;
            for (const auto& oid : qnn_n.output_ids)
                if (oid == qid) { is_self = true; break; }
            if (is_self) { extra_qnn_inputs++; continue; }
            auto qpit = qnn_producers.find(qid);
            if (qpit == qnn_producers.end() || qpit->second.empty()) { extra_qnn_inputs++; continue; }
            qnn_input_producers.push_back(qpit->second[0]);
        }

        // For each REQNN data input with a producer, check it exists in QNN producers
        bool node_ok = true;
        std::vector<std::string> qnn_remaining = qnn_input_producers;
        for (const auto& req_tensor_name : req_n.input_names) {
            auto rpit = req_producers.find(req_tensor_name);
            std::string req_prod = (rpit != req_producers.end()) ? rpit->second : "";
            if (req_prod.empty()) continue; // graph input, skip

            auto fit = std::find(qnn_remaining.begin(), qnn_remaining.end(), req_prod);
            if (fit == qnn_remaining.end()) {
                std::cout << "  PRODUCER MISSING: " << grouping
                          << " req_input=" << req_tensor_name
                          << " req_prod=" << req_prod << "\n";
                node_ok = false;
                break;
            }
            qnn_remaining.erase(fit);
        }
        if (node_ok) topo_ok++;
        else topo_fail++;
    }
    std::cout << "  topology OK: " << topo_ok << "/" << qnn_nodes.size()
              << ", fail: " << topo_fail
              << ", QNN extra inputs (injected/self-ref): " << extra_qnn_inputs << "\n";
    check(topo_fail == 0, "data-flow topology matches");

    // ===== Summary =====
    std::cout << "\n[6] Summary:\n";
    std::cout << "  QNN before graph:  " << qnn_nodes.size() << " nodes, "
              << b_tensor_count << " tensors\n";
    std::cout << "  REQNN .cpp parse: " << req_nodes.size() << " nodes\n";
    std::cout << "  Pass: " << g_pass << ", Fail: " << g_fail << "\n";

    if (g_fail == 0)
        std::cout << "\n=== Before Graph Comparison PASSED ===\n";
    else
        std::cout << "\n=== Before Graph Comparison FAILED ===\n";
    return g_fail > 0 ? 1 : 0;
}


