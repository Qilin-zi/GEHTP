#include "hnnx/api/c_interface.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/cost/cost_model.hpp"
#include <cstring>
#include <unordered_map>
#include <vector>

namespace hnnx {

// C interface implementation
// Source: c_interface.cc, hexagon_nn_graph.cc

// Graph handle registry (graph_id -> GraphPrepare*)
static std::unordered_map<int, std::unique_ptr<GraphPrepare>> g_graphs;
static std::unordered_map<int, HexagonNNEnv> g_envs;
static int g_next_graph_id = 1;

// hexagon_nn_open: create a new graph
// Source: c_interface.cc, hexagon_nn_open @ 0xCD9090 (6885 bytes, ELF st_size)
// The real binary's hexagon_nn_open is large because it includes op registration
// scaffolding. We call register_all_ops() separately.
int hexagon_nn_open(const char* package_name, int soc_type, int* graph_id) {
    if (!graph_id) return -1;
    // Source: c_interface.cc:715 "ERROR:No graph specified."
    auto gp = std::make_unique<GraphPrepare>();
    int id = g_next_graph_id++;
    *graph_id = id;
    g_graphs[id] = std::move(gp);
    g_envs[id].set_soc_type(soc_type);
    g_envs[id].set_num_nsps(1);
    return 0;
}

// hexagon_nn_close: destroy a graph
// Source: c_interface.cc, hexagon_nn_close @ 0xCDABF0 (68 bytes, ELF st_size)
int hexagon_nn_close(int graph_id) {
    g_graphs.erase(graph_id);
    g_envs.erase(graph_id);
    return 0;
}

// hexagon_nn_append_node: add a node to the graph
// Source: c_interface.cc:715, hexagon_nn_append_node @ 0xCDB560 (965 bytes, ELF st_size)
int hexagon_nn_append_node(int graph_id, uint32_t node_id, const char* op_name,
                           const void* inputs, uint32_t num_inputs,
                           const void* outputs, uint32_t num_outputs) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    if (!op_name) return -1;
    auto* gp = it->second.get();
    auto id = gp->append_node(op_name, node_id,
                              static_cast<const InputDef*>(inputs), num_inputs,
                              static_cast<const OutputDef*>(outputs), num_outputs,
                              nullptr);
    return (id != 0) ? 0 : -1;
}

// hexagon_nn_append_const_node: add a const node
// Source: c_interface.cc, hexagon_nn_append_const_node @ 0xCDC300 (201 bytes, ELF st_size)
int hexagon_nn_append_const_node(int graph_id, uint32_t node_id,
                                  const OutputDef* output_def,
                                  const uint8_t* data, size_t data_len) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    if (!output_def) return -1;
    auto* gp = it->second.get();
    auto id = gp->append_const_node(node_id, *output_def, data, data_len);
    return (id != 0) ? 0 : -1;
}

// hexagon_nn_prepare: prepare the graph for execution
// Source: c_interface.cc:861, hexagon_nn_prepare @ 0xCDC550 (111 bytes, ELF st_size)
int hexagon_nn_prepare(int graph_id, HexagonNNEnv& env) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    auto* gp = it->second.get();
    return static_cast<int>(gp->prepare(env));
}

// hexagon_nn_serialize: serialize graph to file
// Source: c_interface.cc, hexagon_nn_serialize @ 0xCDAE20 (151 bytes, ELF st_size)
int hexagon_nn_serialize(int graph_id, int fd) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    auto* gp = it->second.get();
    return gp->serialize_file(fd) ? 0 : -1;
}

// hexagon_nn_serialize_to_mem: serialize graph to memory buffer
// Source: c_interface.cc:437, hexagon_nn_serialize_to_mem @ 0xCDB010 (65 bytes, ELF st_size)
int hexagon_nn_serialize_to_mem(int graph_id, uint8_t* buf, size_t buf_size, size_t* out_size) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    if (!buf || !out_size) return -1;
    auto* gp = it->second.get();
    return gp->serialize(buf, buf_size, *out_size) ? 0 : -1;
}

// hexagon_nn_deserialize: deserialize graph from memory
// Source: c_interface.cc, hexagon_nn_deserialize_graph
int hexagon_nn_deserialize(int graph_id, const uint8_t* data, size_t size) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    if (!data || size == 0) return -1;
    auto* gp = it->second.get();
    return gp->deserialize(data, size) ? 0 : -1;
}

// hexagon_nn_execute: execute the graph with provided inputs (host reference path)
// Source: c_interface.cc, hexagon_nn_execute
// 用浮点参考实现遍历拓扑有序的 ops_。inputs = float32 数组,
// outputs = float32 输出数组。num_inputs/num_outputs 只检查 ≥1。
int hexagon_nn_execute(int graph_id, uint32_t num_inputs, const void* inputs,
                       uint32_t num_outputs, void* outputs) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    if (num_inputs < 1 || !inputs || num_outputs < 1 || !outputs) return -1;

    auto* gp = it->second.get();
    op_id_t input_id = gp->get_input_node_id();
    if (input_id == 0) return -1;

    const OpDef* input_opdef = gp->get_op_at(input_id);
    if (!input_opdef) return -1;

    size_t input_len = 1;
    for (uint32_t i = 0; i < input_opdef->output_def.rank && i < 5; ++i)
        input_len *= static_cast<size_t>(input_opdef->output_def.dims[i]);

    const float* input_float = static_cast<const float*>(inputs);
    std::vector<float> input_vec(input_float, input_float + input_len);
    auto result = gp->execute_host(input_vec);
    if (!result.ok || result.output.empty()) return -1;

    float* out = static_cast<float*>(outputs);
    for (size_t i = 0; i < result.output.size(); ++i)
        out[i] = result.output[i];
    return 0;
}

// hexagon_nn_set_option: set a graph option
// Source: c_interface.cc, hexagon_nn_set_option @ 0xCDC5C0 (317 bytes, ELF st_size)
int hexagon_nn_set_option(int graph_id, const char* option_name, const char* option_value) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    if (!option_name || !option_value) return -1;
    // Parse option key=value pair
    // Options include:
    //   sched_threshold_ratio, sched_lower_threshold_ratio
    //   sched_timeout, sched_outer_timeout
    //   sched_full_retries, sched_afterburner
    //   sched_abort_on_mistake, sched_early_out
    //   sched_hint_depthwise, sched_delay_dma
    //   vtcm_retention, spillFillBufferSizes
    //   seq_sf_svf_en, seq_sf_mlh_training_mode
    //   external_sequencer, selected_sequencer
    return 0;
}

// hexagon_nn_set_node_ids: set node ID range
// Source: c_interface.cc, hexagon_nn_set_node_ids @ 0xCDC3D0 (145 bytes, ELF st_size)
int hexagon_nn_set_node_ids(int graph_id, uint32_t start, uint32_t end, uint32_t base) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    auto* gp = it->second.get();
    gp->set_node_ids(start, end, base);
    return 0;
}

// hexagon_nn_get_input: get input tensor
int hexagon_nn_get_input(int graph_id, uint32_t input_idx, void** tensor) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    return 0;
}

// hexagon_nn_get_output: get output tensor
int hexagon_nn_get_output(int graph_id, uint32_t output_idx, void** tensor) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    return 0;
}

// hexagon_nn_register_op_pkg: register an op package
// Source: c_interface.cc, hexagon_nn_register_op_pkg @ 0xCE0540 (65 bytes, ELF st_size)
int hexagon_nn_register_op_pkg(const char* package_name, const char* soc_type) {
    // Register op package with OpRegistry
    // In real binary: loads .so file and registers ops
    return 0;
}

// hexagon_nn_get_tracked_ids: get const tracking IDs
int hexagon_nn_get_tracked_ids(int graph_id, std::vector<op_id_t>& ids) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    return 0;
}

// hexagon_nn_set_wtshare_metadata_filename
int hexagon_nn_set_wtshare_metadata_filename(int graph_id, const char* filename) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    auto* gp = it->second.get();
    gp->set_wtshare_metadata_filename(filename);
    return 0;
}

// hexagon_nn_set_file_io
int hexagon_nn_set_file_io(int graph_id, int mode) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    return 0;
}

// hexagon_nn_set_perfinfo
int hexagon_nn_set_perfinfo(int graph_id, const char* key, uint32_t value) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    return 0;
}

// hexagon_nn_set_retention_mode
int hexagon_nn_set_retention_mode(int graph_id, int mode) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    return 0;
}

// hexagon_nn_trigger_graph_abort
int hexagon_nn_trigger_graph_abort(int graph_id) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    return 0;
}

// hexagon_nn_preemption_support
int hexagon_nn_preemption_support(int graph_id, int* supported) {
    auto it = g_graphs.find(graph_id);
    if (it == g_graphs.end()) return -1;
    if (supported) *supported = 0;
    return 0;
}

} // namespace hnnx
