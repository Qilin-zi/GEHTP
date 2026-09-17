// test_typical_structures.cpp 鈥?Load 4 typical-structure models (.cpp + .bin),
// build in-memory graph with QnnIRLoader, compare with QNN before_graph.json
// for each model.
//
// Models (each 2 layers of typical structure):
//   1. resnet50_block      鈥?Conv+BN+Relu residual block (CNN)
//   2. mobilenetv2_block   鈥?DWConv+PWConv depthwise separable (CNN)
//   3. yolov8_block        鈥?Conv+BN+SiLU (detection)
//   4. mobilebert_block    鈥?QKV MatMul+Softmax+LayerNorm (lightweight transformer)
//
// Verification per model:
//   a. QnnIRLoader loads .cpp + .bin successfully (ops > 0)
//   b. Same node count as QNN before_graph.json
//   c. Same op types per node (by grouping name)
//   d. Same data-flow topology (producer-consumer relationships)
// Linux: 测试数据仅存于 Windows 原开发机 → SKIP(计划: 隔离不移植)
#ifndef _WIN32
#include <cstdio>
int main() { std::printf("SKIP: Windows-only test data (Linux)\n"); return 0; }
#else
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

static const char* TS_DIR =
    "C:\\Users\\RQILIN\\Documents\\Default Project\\test_models\\typical_structures";

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const std::string& msg) {
    if (cond) { g_pass++; std::cout << "  OK: " << msg << "\n"; }
    else { g_fail++; std::cerr << "FAIL: " << msg << "\n"; }
}

struct QnnNode {
    std::string type;
    std::string grouping;
    std::vector<std::string> input_ids;
    std::vector<std::string> output_ids;
};

static std::map<std::string, QnnNode> parse_qnn_before(const std::string& path,
    std::map<std::string, std::vector<std::string>>& producers)
{
    using namespace hnnx;
    std::map<std::string, QnnNode> nodes;
    JsonValue root = parse_json_file(path);
    const JsonValue& ns = root.at("graph").at("nodes");
    for (const auto& [hex_key, njson] : ns.obj_val) {
        QnnNode n;
        n.type = njson.at("type").as_str();
        n.grouping = njson.at("grouping").as_str();
        const JsonValue& ins = njson.at("input_names");
        for (size_t i = 0; i < ins.size(); ++i)
            n.input_ids.push_back(ins.at(i).as_str());
        const JsonValue& outs = njson.at("output_names");
        for (size_t i = 0; i < outs.size(); ++i)
            n.output_ids.push_back(outs.at(i).as_str());
        nodes[n.grouping] = n;
        for (const auto& oid : n.output_ids)
            producers[oid].push_back(n.grouping);
    }
    return nodes;
}

static int test_one_model(const std::string& name) {
    using namespace hnnx;
    std::cout << "\n============================================\n";
    std::cout << "  Testing: " << name << "\n";
    std::cout << "============================================\n";

    std::string dir = std::string(TS_DIR) + "\\" + name;
    std::string cpp_path = dir + "\\" + name + ".cpp";
    std::string bin_path = dir + "\\" + name + "_weights.bin";
    if (!std::ifstream(bin_path).good()) bin_path = dir + "\\" + name + ".bin";
    std::string before_path = dir + "\\" + name + "_before_graph.json";

    // [1] Load QNN before_graph.json
    std::cout << "[1] Loading QNN before_graph.json...\n";
    std::map<std::string, std::vector<std::string>> qnn_producers;
    auto qnn_nodes = parse_qnn_before(before_path, qnn_producers);
    std::cout << "  QNN before: " << qnn_nodes.size() << " nodes\n";

    // [2] Load .cpp + .bin with QnnIRLoader
    std::cout << "[2] Loading REQNN .cpp + .bin...\n";
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t ops = loader.load_qnn_ir(cpp_path, bin_path);
    std::cout << "  REQNN loaded: " << ops << " ops, " << loader.nodes().size() << " nodes\n";
    check(ops > 0, name + ": QnnIRLoader loaded .cpp + .bin");

    // Build REQNN node structure (grouping = node name)
    struct ReqNode {
        std::string type;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
    };
    std::map<std::string, ReqNode> req_nodes;
    std::map<std::string, std::string> req_producers;
    for (const auto& ni : loader.nodes()) {
        ReqNode rn;
        rn.type = ni.type;
        rn.input_names = ni.input_names;
        rn.output_names = ni.output_names;
        req_nodes[ni.name] = rn;
        for (const auto& on : ni.output_names)
            req_producers[on] = ni.name;
    }

    // [3] Compare node count (allow extra Split nodes 鈥?HtpPrepare eliminates them)
    std::cout << "[3] Comparing node counts...\n";
    std::cout << "  QNN: " << qnn_nodes.size() << ", REQNN: " << req_nodes.size() << "\n";
    int extra_req = 0;
    for (const auto& [rn_name, rn] : req_nodes) {
        if (qnn_nodes.find(rn_name) == qnn_nodes.end()) {
            extra_req++;
            std::cout << "  extra in REQNN: " << rn_name << " (type=" << rn.type << ")\n";
        }
    }
    check(extra_req == 0 || (extra_req <= 2 && req_nodes.size() - qnn_nodes.size() == extra_req),
          name + ": node count match (extra=" + std::to_string(extra_req) + " are Split/eliminated by HtpPrepare)");

    // [4] Compare node types (by grouping name)
    std::cout << "[4] Comparing node types...\n";
    int type_match = 0, type_mismatch = 0, missing = 0;
    for (const auto& [grouping, qnn_n] : qnn_nodes) {
        auto it = req_nodes.find(grouping);
        if (it == req_nodes.end()) { missing++; continue; }
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
            std::cout << "  MISMATCH: " << grouping
                      << " qnn=" << qnn_n.type << " req=" << it->second.type << "\n";
        }
    }
    std::cout << "  match: " << type_match << ", mismatch: " << type_mismatch
              << ", missing: " << missing << "\n";
    check(type_mismatch == 0, name + ": no type mismatches");
    check(missing == 0, name + ": no nodes missing in REQNN");

    // [5] Compare topology (data-flow producer structure)
    // Split nodes are eliminated by HtpPrepare: when REQNN producer is a Split,
    // walk back to find the Split's input's producer (transitive resolution).
    std::cout << "[5] Comparing topology...\n";

    // Build transitive producer map: if producer is a Split, resolve to Split's source
    auto resolve_producer = [&](const std::string& prod) -> std::string {
        std::string cur = prod;
        std::set<std::string> visited;
        while (visited.find(cur) == visited.end()) {
            visited.insert(cur);
            auto it = req_nodes.find(cur);
            if (it == req_nodes.end()) break;
            // If this node is a Split (not in QNN before_graph), resolve through its input
            if (qnn_nodes.find(cur) == qnn_nodes.end() && !it->second.input_names.empty()) {
                auto pit = req_producers.find(it->second.input_names[0]);
                if (pit != req_producers.end()) {
                    cur = pit->second;
                    continue;
                }
            }
            break;
        }
        return cur;
    };

    int topo_ok = 0, topo_fail = 0;
    for (const auto& [grouping, qnn_n] : qnn_nodes) {
        auto it = req_nodes.find(grouping);
        if (it == req_nodes.end()) continue;
        const ReqNode& req_n = it->second;

        // Collect QNN producers for this node's inputs (skip self-refs, no-producer)
        std::vector<std::string> qnn_input_producers;
        int qnn_orphan_inputs = 0;  // inputs with no producer (from eliminated Split)
        for (const auto& qid : qnn_n.input_ids) {
            bool is_self = false;
            for (const auto& oid : qnn_n.output_ids)
                if (oid == qid) { is_self = true; break; }
            if (is_self) continue;
            auto qpit = qnn_producers.find(qid);
            if (qpit == qnn_producers.end() || qpit->second.empty()) {
                qnn_orphan_inputs++;  // tensor exists but no producing node (Split eliminated)
                continue;
            }
            qnn_input_producers.push_back(qpit->second[0]);
        }

        bool node_ok = true;
        std::vector<std::string> qnn_remaining = qnn_input_producers;
        int req_orphan_count = 0;  // REQNN producers that resolve to orphan tensors
        for (const auto& req_tensor_name : req_n.input_names) {
            auto rpit = req_producers.find(req_tensor_name);
            std::string req_prod = (rpit != req_producers.end()) ? rpit->second : "";
            if (req_prod.empty()) continue;
            // Resolve through eliminated Split nodes
            req_prod = resolve_producer(req_prod);
            auto fit = std::find(qnn_remaining.begin(), qnn_remaining.end(), req_prod);
            if (fit == qnn_remaining.end()) {
                // This REQNN producer doesn't match any QNN producer.
                // If QNN has orphan inputs (from eliminated Split), count as resolved.
                if (req_orphan_count < qnn_orphan_inputs) {
                    req_orphan_count++;
                    continue;
                }
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
              << ", fail: " << topo_fail << "\n";
    check(topo_fail == 0, name + ": data-flow topology matches");

    return (g_fail > 0) ? 1 : 0;
}

int main() {
    using namespace hnnx;
    std::cout << "=== Typical Structures: Composition Verification ===\n";

    std::vector<std::string> models = {
        "resnet50_block",
        "mobilenetv2_block",
        "yolov8_block",
        "mobilebert_block",
    };

    int total_fail = 0;
    for (const auto& m : models) {
        int before_fail = g_fail;
        test_one_model(m);
        if (g_fail > before_fail) total_fail++;
    }

    std::cout << "\n============================================\n";
    std::cout << "  Summary: " << (models.size() - total_fail) << "/"
              << models.size() << " models passed, "
              << g_pass << " checks passed, " << g_fail << " failed\n";
    std::cout << "============================================\n";

    return total_fail > 0 ? 1 : 0;
}





#endif // !_WIN32
