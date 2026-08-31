#include "hnnx/api/hexagon_nn_env.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hnnx {

using op_id_t = uint64_t;

int hexagon_nn_open(const char* package_name, int soc_type, int* graph_id);
int hexagon_nn_close(int graph_id);
int hexagon_nn_append_node(int graph_id, uint32_t node_id, const char* op_name,
                           const void* inputs, uint32_t num_inputs,
                           const void* outputs, uint32_t num_outputs);
int hexagon_nn_append_const_node(int graph_id, uint32_t node_id,
                                  const struct OutputDef* output_def,
                                  const uint8_t* data, size_t data_len);
int hexagon_nn_prepare(int graph_id, HexagonNNEnv& env);
int hexagon_nn_serialize(int graph_id, int fd);
int hexagon_nn_serialize_to_mem(int graph_id, uint8_t* buf, size_t buf_size, size_t* out_size);
int hexagon_nn_deserialize(int graph_id, const uint8_t* data, size_t size);
int hexagon_nn_execute(int graph_id, uint32_t num_inputs, const void* inputs,
                       uint32_t num_outputs, void* outputs);
int hexagon_nn_set_option(int graph_id, const char* option_name, const char* option_value);
int hexagon_nn_set_node_ids(int graph_id, uint32_t start, uint32_t end, uint32_t base);
int hexagon_nn_get_input(int graph_id, uint32_t input_idx, void** tensor);
int hexagon_nn_get_output(int graph_id, uint32_t output_idx, void** tensor);
int hexagon_nn_register_op_pkg(const char* package_name, const char* soc_type);
int hexagon_nn_get_tracked_ids(int graph_id, std::vector<op_id_t>& ids);
int hexagon_nn_set_wtshare_metadata_filename(int graph_id, const char* filename);
int hexagon_nn_set_file_io(int graph_id, int mode);
int hexagon_nn_set_perfinfo(int graph_id, const char* key, uint32_t value);
int hexagon_nn_set_retention_mode(int graph_id, int mode);
int hexagon_nn_trigger_graph_abort(int graph_id);
int hexagon_nn_preemption_support(int graph_id, int* supported);

} // namespace hnnx
