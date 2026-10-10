#pragma once
#include "hnnx/ir/types.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/frontend/json.hpp"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace hnnx {

struct QnnTensorInfo {
    std::string name;
    uint32_t id = 0;
    uint32_t type = 0;       // 0=APP_WRITE, 1=APP_READ, 3=NATIVE, 4=STATIC
    uint32_t data_type = 0; // QNN encoding: 0x0232=float32, 0x0132=uint32
    std::vector<uint32_t> dims;
    std::vector<uint8_t> data; // inline const data (perm etc.)
    bool is_param = false;     // true if tensor_param inside a node
};

struct QnnScalarParam {
    std::string name;
    uint32_t id = 0;
    bool is_numeric = false;
    double value_num = 0.0;
    std::string value_str;  // 字符串值(数字也留原文, 保真)
};

struct QnnNodeInfo {
    std::string name;
    std::string type;          // "Transpose", "Reshape", "FullyConnected"
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<QnnTensorInfo> tensor_params;
    std::vector<QnnScalarParam> scalar_params;  // scalar_params {name: {id: value}}
};

class QnnIRLoader {
public:
    explicit QnnIRLoader(GraphPrepare& gp);

    // Load QNN IR from a net.json file path.
    // Returns number of op nodes created (0 = failure).
    uint32_t load_net_json(const std::string& path);

    // Load QNN IR from .cpp source file (authoritative IR).
    // Parses addTensor_*() and addNode_*() functions.
    // Returns number of op nodes created (0 = failure).
    uint32_t load_cpp(const std::string& cpp_path);

    // Load weight .bin file (TAR archive with W.raw, b.raw etc).
    // Returns map of var_name (W, b) -> raw bytes.
    static std::map<std::string, std::vector<uint8_t>> load_weight_bin(const std::string& bin_path);

    // Combined: load .cpp (graph structure) + .bin (weights).
    // cpp_path = path to simple_linear (no extension) or simple_linear.cpp
    // bin_path = path to simple_linear.bin
    // Returns number of op nodes created (0 = failure).
    uint32_t load_qnn_ir(const std::string& cpp_path, const std::string& bin_path);

    // Access parsed graph info (for verification)
    const std::map<std::string, QnnTensorInfo>& tensors() const { return tensors_; }
    const std::vector<QnnNodeInfo>& nodes() const { return nodes_; }
    const std::map<std::string, op_id_t>& tensor_opids() const { return tensor_opids_; }
    const std::map<std::string, std::vector<uint8_t>>& weights() const { return weights_; }
    // net.json 路径的权重注入(2.48 params.bin 适配后经 TAR 加载进来)
    void set_weights(std::map<std::string, std::vector<uint8_t>> w) { weights_ = std::move(w); }

    // --static-input name=file: 导入期把图输入物化为 const 节点(值进入管线
    // 时机从 emit 提前到导入, 缺文件/尺寸不符 = 硬错)。仅支持 f16 语义
    // (f32/f16 文件 → 池内 f32 拓宽, emit 权重路径洗回 f16, 与输入槽烘焙
    // 位级同字节)。".i32.raw" 拒绝: 设备槽需 i32 原文, const 通用路径会
    // f32→f16 洗白 — 保持 emit --input-f16 供给(其 P1 硬门守)。
    // 主输入(id 最小的图输入, emit 的 EXT_IN 槽 0)不可物化。
    bool materialize_static_input(const std::string& tensor_name, const std::string& file_path,
                                  std::string& err);

private:
    GraphPrepare& gp_;
    std::map<std::string, QnnTensorInfo> tensors_;
    std::vector<op_id_t> input_decl_json_;  // type==0 张量按 JSON 键序(声明序)
    std::vector<QnnNodeInfo> nodes_;
    std::map<std::string, op_id_t> tensor_opids_; // tensor name -> REQNN op_id
    std::map<std::string, std::vector<uint8_t>> weights_; // var_name -> bytes
    std::map<std::string, op_id_t> param_dedup_; // (dims+data) signature -> const id
    op_id_t next_param_id_ = 0x10000; // param const IDs in high range
    op_id_t quant_marker_id_ = 0;     // shared quant marker const id (0=not created)
    op_id_t output_rank_id_ = 0;      // shared output_rank const id
    op_id_t scale_id_ = 0;            // shared scale const id

    void parse_tensors(const JsonValue& graph_json);
    void parse_nodes(const JsonValue& graph_json);
    void extract_tensor_params(QnnNodeInfo& node, const JsonValue& node_json);
    void extract_scalar_params(QnnNodeInfo& node, const JsonValue& node_json);
    // scalar_params → op_data 字节 blob(append_node 的 ops_data), 消费方解析
    static std::vector<uint8_t> pack_scalar_params(const QnnNodeInfo& node);
    void build_tensor_opid_map();
    uint32_t build_graph();

    // .cpp parsing helpers
    static std::vector<std::string> extract_functions(const std::string& src, const std::string& prefix);
    static std::string extract_string_arg(const std::string& func, const std::string& field, size_t start = 0);
    static std::vector<uint32_t> extract_dims_array(const std::string& func, const std::string& var_name);
    static std::vector<std::string> extract_string_list(const std::string& func, const std::string& var_name);
    static std::vector<uint8_t> extract_inline_data(const std::string& func, const std::string& var_name);
    static std::string map_qnn_tensor_type(const std::string& type_str);
    static DType map_qnn_datatype(const std::string& dt_str);

    static DType map_dtype(uint32_t qnn_dtype);
    static void fill_output_def(OutputDef& od, const std::vector<uint32_t>& dims, DType dt);
    static std::vector<uint8_t> pack_data(const JsonValue& data_arr, uint32_t qnn_dtype);
};

} // namespace hnnx
