#pragma once
#include "hnnx/ir/graph_prepare.hpp"
#include <string>

namespace hnnx {

// Dump a GraphPrepare's current state to a JSON file in the same format as
// QNN's before_graph.json / after_graph.json (from --save_backend_op_mapping).
//
// Format:
//   { "header": {...}, "model.cpp": "htp_graph",
//     "graph": { "tensors": {...}, "nodes": {...} } }
//
// tensor: { "0xHEXID": { "id": N, "type": T, "data_type": D, "dims": [...] } }
// node:   { "0xHEXID": { "type": "...", "grouping": "...",
//           "tensor_params": {...}, "scalar_params": {...},
//           "input_names": ["0x...", ...], "output_names": ["0x...", ...] } }

class GraphDumper {
public:
    explicit GraphDumper(GraphPrepare& gp);

    bool dump(const std::string& path) const;
    std::string dump_string() const;

private:
    GraphPrepare& gp_;

    std::string hex_id(uint64_t id) const;
    void dump_tensors(std::string& out) const;
    void dump_nodes(std::string& out) const;
    void dump_scalar_params(std::string& out, const OpDef& op) const;
};

} // namespace hnnx
