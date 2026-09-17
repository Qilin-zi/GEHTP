// test_before_after_dump.cpp 鈥?Load QNN IR .cpp, run stage2 (do_prepare1),
// dump REQNN's before graph JSON, and compare with QNN's before_graph.json.
//
// Pipeline (per graph_compiler_annotated.docx):
//   Stage 1: QNN IR (.cpp) 鈫?IR.so         [QnnIRLoader::load_cpp]
//   Stage 2: IR.so 鈫?Backend graph          [GraphPrepare::do_prepare1]
//            (鍥惧鍏?瑙ｆ瀽 鈫?褰㈢姸鎺ㄦ柇 鈫?甯搁噺鎶樺彔 鈫?绠楀瓙瑙勮寖鍖?鈫?閲忓寲褰掍竴鍖?
//   Stage 3: Backend graph 鈫?Scheduled graph [GraphPrepare::do_prepare2]
//            (绠楀瓙鈫掑唴鏍?鈫?甯冨眬浼犳挱 鈫?铻嶅悎 鈫?鏉冮噸棰勫鐞?鈫?鍒?tile)
//
// Before graph = stage 2 output (before optimization passes)
// After graph  = stage 3 output (after optimization passes, scheduled)
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "test_paths.hpp"
#include "hnnx/frontend/json.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/graph_dumper.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"
#include <iostream>
#include <string>
#include <set>
#include <map>
#include <algorithm>

static const char* T2_DIR =
    TP_T2_DIR;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const std::string& msg) {
    if (cond) { g_pass++; std::cout << "  OK: " << msg << "\n"; }
    else { g_fail++; std::cerr << "FAIL: " << msg << "\n"; }
}

// Parse QNN before_graph.json and extract:
//   - set of tensor hex keys
//   - map: grouping -> {type, input_hex_ids, output_hex_ids}
struct QnnGraphRef {
    struct Node {
        std::string type;
        std::string grouping;
        std::vector<std::string> input_ids;
        std::vector<std::string> output_ids;
    };
    struct Tensor {
        uint32_t id;
        uint32_t type;
        uint32_t data_type;
        std::vector<uint64_t> dims;
    };
    std::map<std::string, Tensor> tensors;  // hex key -> tensor
    std::map<std::string, Node> nodes;       // hex key -> node
    // grouping -> node (for lookup by op name)
    std::map<std::string, std::string> grouping_to_hex;
    // hex tensor key -> producer grouping
    std::map<std::string, std::string> producer;
};

static QnnGraphRef parse_qnn_graph(const std::string& path) {
    using namespace hnnx;
    QnnGraphRef r;
    JsonValue root = parse_json_file(path);
    const JsonValue& g = root.at("graph");
    const JsonValue& ts = g.at("tensors");
    for (const auto& [hex_key, tjson] : ts.obj_val) {
        QnnGraphRef::Tensor t;
        t.id = static_cast<uint32_t>(tjson.at("id").as_int());
        t.type = static_cast<uint32_t>(tjson.at("type").as_int());
        t.data_type = static_cast<uint32_t>(tjson.at("data_type").as_int());
        const JsonValue& dims = tjson.at("dims");
        for (size_t i = 0; i < dims.size(); ++i)
            t.dims.push_back(static_cast<uint64_t>(dims.at(i).as_int()));
        r.tensors[hex_key] = std::move(t);
    }
    const JsonValue& ns = g.at("nodes");
    for (const auto& [hex_key, njson] : ns.obj_val) {
        QnnGraphRef::Node n;
        n.type = njson.at("type").as_str();
        n.grouping = njson.at("grouping").as_str();
        const JsonValue& ins = njson.at("input_names");
        for (size_t i = 0; i < ins.size(); ++i)
            n.input_ids.push_back(ins.at(i).as_str());
        const JsonValue& outs = njson.at("output_names");
        for (size_t i = 0; i < outs.size(); ++i)
            n.output_ids.push_back(outs.at(i).as_str());
        r.nodes[hex_key] = n;
        r.grouping_to_hex[n.grouping] = hex_key;
        for (const auto& oid : n.output_ids)
            r.producer[oid] = n.grouping;
    }
    return r;
}

int main() {
    using namespace hnnx;
    std::string cpp_path = std::string(T2_DIR) + "/transformer2.cpp";
    std::string bin_path = std::string(T2_DIR) + "/transformer2.bin";
    std::string before_path = std::string(T2_DIR) + "/transformer2_before_graph.json";
    std::string reqnn_before_path = std::string(T2_DIR) + "/reqnn_before_graph.json";

    std::cout << "=== REQNN Before Graph Dump & Compare ===\n\n";

    // ===== Step 1: Load QNN IR (.cpp + .bin) =====
    std::cout << "[1] Loading QNN IR (.cpp + .bin)...\n";
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t ops = loader.load_qnn_ir(cpp_path, bin_path);
    std::cout << "  ops loaded: " << ops << "\n";
    check(ops > 0, "QNN IR loaded");

    // ===== Step 2: Dump before graph BEFORE do_prepare1 =====
    // QNN dumps before_graph.json at "graph_before_optimization" point:
    // right after graph import, before any optimization passes.
    // So we dump immediately after loading, before do_prepare1.
    std::cout << "\n[2] Dumping REQNN before graph (before do_prepare1)...\n";
    GraphDumper dumper(gp);
    bool dump_ok = dumper.dump(reqnn_before_path);
    check(dump_ok, "REQNN before graph dumped");

    // ===== Step 3: Run stage 2 (do_prepare1) =====
    std::cout << "\n[3] Running do_prepare1 (stage 2: import + shape inference + const prop)...\n";
    HexagonNNEnv env;
    env.set_num_nsps(1);
    env.set_soc_type(75);
    VtcmCacheInstance vtcm(0, 8 * 1024 * 1024);
    GraphStatus s1 = gp.do_prepare1(env, vtcm);
    std::cout << "  do_prepare1 status: " << static_cast<int>(s1) << "\n";
    check(s1 == GraphStatus::Success, "do_prepare1 succeeded");

    // ===== Step 4: Parse both graphs and compare =====
    std::cout << "\n[4] Parsing and comparing graphs...\n";
    QnnGraphRef qnn = parse_qnn_graph(before_path);
    QnnGraphRef reqnn = parse_qnn_graph(reqnn_before_path);

    // 4a: Compare node count
    std::cout << "\n[4a] Node counts:\n";
    std::cout << "  QNN before:   " << qnn.nodes.size() << " nodes\n";
    std::cout << "  REQNN before: " << reqnn.nodes.size() << " nodes\n";

    // 4b: Compare node types by grouping
    std::cout << "\n[4b] Comparing node types by grouping...\n";
    int type_match = 0, type_missing = 0, type_mismatch = 0;
    for (const auto& [qhex, qnn_n] : qnn.nodes) {
        const std::string& grouping = qnn_n.grouping;
        bool found = false;
        for (const auto& [rhex, rnode] : reqnn.nodes) {
            if (rnode.grouping == grouping) {
                found = true;
                // Normalize type names
                auto norm = [](const std::string& s) -> std::string {
                    std::string r;
                    for (char c : s) if (c != '_') r += c;
                    size_t pos;
                    while ((pos = r.find("Eltwise")) != std::string::npos)
                        r.replace(pos, 7, "ElementWise");
                    return r;
                };
                if (norm(qnn_n.type) == norm(rnode.type))
                    type_match++;
                else {
                    type_mismatch++;
                    std::cout << "  TYPE MISMATCH: " << grouping
                              << " qnn=" << qnn_n.type << " req=" << rnode.type << "\n";
                }
                break;
            }
        }
        if (!found) type_missing++;
    }
    std::cout << "  type match: " << type_match << ", mismatch: " << type_mismatch
              << ", missing in REQNN: " << type_missing << "\n";
    check(type_mismatch == 0, "no type mismatches");
    check(type_missing == 0, "no nodes missing in REQNN");

    // 4c: Compare topology (data-flow producer-consumer)
    std::cout << "\n[4c] Comparing data-flow topology...\n";
    int topo_ok = 0, topo_fail = 0;
    for (const auto& [qhex, qnn_n] : qnn.nodes) {
        const std::string& grouping = qnn_n.grouping;
        // Find matching REQNN node
        const QnnGraphRef::Node* rnode = nullptr;
        for (const auto& [rhex, rn] : reqnn.nodes) {
            if (rn.grouping == grouping) { rnode = &rn; break; }
        }
        if (!rnode) continue;

        // Collect QNN input producers (skip self-refs and no-producer)
        std::vector<std::string> qnn_input_producers;
        for (const auto& qid : qnn_n.input_ids) {
            bool is_self = false;
            for (const auto& oid : qnn_n.output_ids)
                if (oid == qid) { is_self = true; break; }
            if (is_self) continue;
            auto qpit = qnn.producer.find(qid);
            if (qpit == qnn.producer.end() || qpit->second.empty()) continue;
            qnn_input_producers.push_back(qpit->second);
        }

        // Collect REQNN input producers
        std::vector<std::string> req_input_producers;
        for (const auto& rid : rnode->input_ids) {
            auto rpit = reqnn.producer.find(rid);
            if (rpit == reqnn.producer.end() || rpit->second.empty()) continue;
            req_input_producers.push_back(rpit->second);
        }

        // Check: every REQNN producer should be in QNN producers
        bool ok = true;
        std::vector<std::string> qnn_remaining = qnn_input_producers;
        for (const auto& rp : req_input_producers) {
            auto fit = std::find(qnn_remaining.begin(), qnn_remaining.end(), rp);
            if (fit == qnn_remaining.end()) {
                std::cout << "  PRODUCER MISSING: " << grouping
                          << " req_prod=" << rp << " not in QNN\n";
                ok = false;
                break;
            }
            qnn_remaining.erase(fit);
        }
        if (ok) topo_ok++;
        else topo_fail++;
    }
    std::cout << "  topology OK: " << topo_ok << "/" << qnn.nodes.size()
              << ", fail: " << topo_fail << "\n";
    check(topo_fail == 0, "data-flow topology matches");

    // ===== Summary =====
    std::cout << "\n[5] Summary:\n";
    std::cout << "  QNN before graph:   " << qnn.nodes.size() << " nodes, "
              << qnn.tensors.size() << " tensors\n";
    std::cout << "  REQNN before graph: " << reqnn.nodes.size() << " nodes, "
              << reqnn.tensors.size() << " tensors\n";
    std::cout << "  Pass: " << g_pass << ", Fail: " << g_fail << "\n";
    std::cout << "  REQNN dump: " << reqnn_before_path << "\n";

    if (g_fail == 0)
        std::cout << "\n=== Before Graph Dump & Compare PASSED ===\n";
    else
        std::cout << "\n=== Before Graph Dump & Compare FAILED ===\n";
    return g_fail > 0 ? 1 : 0;
}


