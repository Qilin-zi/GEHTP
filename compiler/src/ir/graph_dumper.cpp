#include "hnnx/ir/graph_dumper.hpp"
#include "hnnx/ir/types.hpp"
#include <cstdio>
#include <sstream>
#include <fstream>
#include <algorithm>

namespace hnnx {

GraphDumper::GraphDumper(GraphPrepare& gp) : gp_(gp) {}

std::string GraphDumper::hex_id(uint64_t id) const {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(id));
    return std::string(buf);
}

void GraphDumper::dump_tensors(std::string& out) const {
    out += "    \"tensors\": {\n";
    bool first = true;
    // Iterate all opdefs; each has an output tensor
    // We need to collect all tensors: input tensors, const tensors, and op output tensors
    // Use for_each_op via the public API
    // We'll collect into a vector to sort by op_id
    struct TensorEntry {
        uint64_t id;
        uint32_t type;      // 3=NATIVE, 4=STATIC
        uint32_t data_type; // QNN encoding: 562=float32, 50=uint32
        std::vector<uint64_t> dims;
    };
    std::vector<TensorEntry> entries;

    // Helper: map DType to QNN data_type (matching QNN enum values)
    auto map_dtype = [](uint32_t dt) -> uint32_t {
        switch (static_cast<DType>(dt)) {
            case DType::Float32: return 562;  // 0x0232
            case DType::Int32:   return 50;   // 0x0032
            case DType::UInt8:   return 50;   // 0x0032 (uint8 maps to 50 in before_graph)
            case DType::Float16: return 818;  // 0x0332
            default:             return dt;   // passthrough for non-DType values (e.g. 1032 quant marker)
        }
    };

    gp_.for_each_op([&](const OpDef* op) {
        if (!op) return;
        std::string nm = op->name_tag ? op->name_tag->name() : "";
        // Skip Output sink node — it has no output tensor
        if (nm == "Output") return;
        TensorEntry te;
        te.id = op->op_id;
        // type: const -> 4 (STATIC), Input -> 4 (HtpPrepare treats APP_WRITE as STATIC),
        // op output -> 3 (NATIVE)
        if (op->is_const()) te.type = 4;
        else if (nm == "Input") te.type = 4;  // HtpPrepare: APP_WRITE -> STATIC
        else te.type = 3; // NATIVE for op outputs
        te.data_type = map_dtype(op->output_def.dtype);
        for (uint32_t i = 0; i < op->output_def.rank && i < 5; ++i)
            te.dims.push_back(op->output_def.dims[i]);
        entries.push_back(std::move(te));
    });

    // Sort by id
    std::sort(entries.begin(), entries.end(),
              [](const TensorEntry& a, const TensorEntry& b) { return a.id < b.id; });

    for (const auto& te : entries) {
        if (!first) out += ",\n";
        first = false;
        out += "      \"" + hex_id(te.id) + "\": {\n";
        out += "        \"id\": " + std::to_string(te.type == 4 ? 0 : 0) + ",\n"; // id field (sequential, we use 0 placeholder)
        out += "        \"type\": " + std::to_string(te.type) + ",\n";
        out += "        \"data_type\": " + std::to_string(te.data_type) + ",\n";
        out += "        \"dims\": [";
        for (size_t i = 0; i < te.dims.size(); ++i) {
            if (i > 0) out += ", ";
            out += std::to_string(te.dims[i]);
        }
        out += "]\n";
        out += "      }";
    }
    out += "\n    }";
}

void GraphDumper::dump_scalar_params(std::string& out, const OpDef& op) const {
    std::string op_type = op.name_tag ? op.name_tag->name() : "";
    std::string op_name = op_type; // grouping = op name; for REQNN we use the name_tag
    // In QNN format, grouping is the original op name. REQNN stores it in name_tag.
    // For now, use op_type as grouping placeholder.

    out += "        \"scalar_params\": {\n";
    out += "            \"output_step_size\": {\n";
    out += "              \"562\": 1\n";
    out += "             },\n";
    out += "            \"output_zero_offset\": {\n";
    out += "              \"50\": 0\n";
    out += "             },\n";
    out += "            \"output_rank\": {\n";
    out += "              \"50\": " + std::to_string(op.output_def.rank) + "\n";
    out += "             },\n";
    out += "            \"qnn_op_type\": {\n";
    out += "              \"1544\": \"" + op_type + "\"\n";
    out += "             },\n";
    out += "            \"qnn_op_name\": {\n";
    out += "              \"1544\": \"" + op_name + "\"\n";
    out += "             },\n";
    out += "            \"mem_dram_read\": {\n";
    out += "              \"100\": 0\n";
    out += "             },\n";
    out += "            \"mem_dram_write\": {\n";
    out += "              \"100\": 0\n";
    out += "             },\n";
    out += "            \"mem_vtcm_read\": {\n";
    out += "              \"100\": 0\n";
    out += "             },\n";
    out += "            \"mem_vtcm_write\": {\n";
    out += "              \"100\": 0\n";
    out += "             }\n";
    out += "        }";
}

void GraphDumper::dump_nodes(std::string& out) const {
    out += "    \"nodes\": {\n";
    bool first = true;

    // Collect op nodes (skip Input/Output/$Const)
    struct NodeEntry {
        uint64_t id;
        std::string type;
        std::string grouping;
        std::vector<uint64_t> input_ids;
        std::vector<uint64_t> output_ids;
        const OpDef* op;
    };
    std::vector<NodeEntry> entries;

    gp_.for_each_op([&](const OpDef* op) {
        if (!op) return;
        std::string nm = op->name_tag ? op->name_tag->name() : "";
        if (nm == "Input" || nm == "Output" || nm == "$Const" || nm == "Const") return;
        if (op->is_const()) return;

        NodeEntry ne;
        ne.id = op->op_id;
        ne.type = nm;
        // grouping: use op->grouping if set (original QNN node name), else fall back to type
        ne.grouping = op->grouping.empty() ? nm : op->grouping;
        ne.op = op;
        for (const auto& conn : op->inputs)
            ne.input_ids.push_back(conn.src_id);
        ne.output_ids.push_back(op->op_id);
        entries.push_back(std::move(ne));
    });

    // Sort by id
    std::sort(entries.begin(), entries.end(),
              [](const NodeEntry& a, const NodeEntry& b) { return a.id < b.id; });

    for (const auto& ne : entries) {
        if (!first) out += ",\n";
        first = false;
        out += "      \"" + hex_id(ne.id) + "\": {\n";
        out += "        \"type\": \"" + ne.type + "\", \n";
        out += "        \"grouping\": \"" + ne.grouping + "\", \n";
        out += "        \"tensor_params\": {},\n";
        dump_scalar_params(out, *ne.op);
        out += ",\n";
        out += "        \"input_names\": [\n";
        for (size_t i = 0; i < ne.input_ids.size(); ++i) {
            out += "          \"" + hex_id(ne.input_ids[i]) + "\"";
            if (i + 1 < ne.input_ids.size()) out += ",";
            out += "\n";
        }
        out += "        ],\n";
        out += "        \"output_names\": [";
        for (size_t i = 0; i < ne.output_ids.size(); ++i) {
            if (i > 0) out += ", ";
            out += "\"" + hex_id(ne.output_ids[i]) + "\"";
        }
        out += "]\n";
        out += "      }";
    }
    out += "\n    }";
}

std::string GraphDumper::dump_string() const {
    std::string out;
    out += "{\n";
    out += " \"header\": {\n";
    out += "      \"header_version\": {\n";
    out += "          \"major\": 1,\n";
    out += "         \"minor\": 0,\n";
    out += "          \"patch\": 0\n";
    out += "      },\n";
    out += "       \"version\": {\n";
    out += "           \"major\": 1,\n";
    out += "           \"minor\": 0,\n";
    out += "           \"patch\": 0\n";
    out += "       },\n";
    out += "       \"artifact_type\": \"HTP_GRAPH\"\n";
    out += "    },\n";
    out += "   \"model.cpp\": \"htp_graph\",\n";
    out += "   \"graph\": {\n";
    dump_tensors(out);
    out += ",\n";
    dump_nodes(out);
    out += "\n   }\n";
    out += "}\n";
    return out;
}

bool GraphDumper::dump(const std::string& path) const {
    std::string content = dump_string();
    std::ofstream f(path);
    if (!f) return false;
    f << content;
    return f.good();
}

} // namespace hnnx
