#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/frontend/json.hpp"
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <regex>
#include <cctype>

namespace hnnx {

QnnIRLoader::QnnIRLoader(GraphPrepare& gp) : gp_(gp) {}

DType QnnIRLoader::map_dtype(uint32_t qnn_dtype) {
    switch (qnn_dtype) {
        case 0x0232: return DType::Float32; // QNN_DATATYPE_FLOAT_32
        case 0x0332: return DType::Float16; // QNN_DATATYPE_FLOAT_16
        case 0x0432: return DType::Int32;   // QNN_DATATYPE_INT_32
        case 0x0032: return DType::Int32;   // QNN_DATATYPE_INT_32 (alt encoding)
        case 0x0132: return DType::Int32;   // QNN_DATATYPE_UINT_32 -> closest
        case 0x0832: return DType::Int8;    // QNN_DATATYPE_SINT_8 (guess)
        case 0x0732: return DType::UInt8;   // QNN_DATATYPE_UINT_8 (guess)
        default:      return DType::Float32;
    }
}

void QnnIRLoader::fill_output_def(OutputDef& od, const std::vector<uint32_t>& dims, DType dt) {
    od.rank = static_cast<uint32_t>(dims.size());
    od.dtype = static_cast<uint32_t>(dt);
    od.flags = 0;
    od.quant_params = 0;
    for (size_t i = 0; i < dims.size() && i < 5; ++i)
        od.dims[i] = dims[i];
    uint32_t es = 4;
    switch (dt) {
        case DType::Float32: case DType::Int32: case DType::Float16: es = 4; break;
        case DType::Int16: es = 2; break;
        case DType::Int8: case DType::UInt8: case DType::Bool: es = 1; break;
        default: es = 4; break;
    }
    od.element_size = es;
    od.quant_scale = 0;
    od.quant_offset = 0;
    od.extra[0] = od.extra[1] = od.extra[2] = 0;
}

std::vector<uint8_t> QnnIRLoader::pack_data(const JsonValue& data_arr, uint32_t qnn_dtype) {
    std::vector<uint8_t> packed;
    if (!data_arr.is_array()) return packed;
    size_t n = data_arr.size();
    if (qnn_dtype == 0x0132 || qnn_dtype == 0x0432) {
        // uint32 / int32: 4 bytes each
        packed.resize(n * 4);
        for (size_t i = 0; i < n; ++i) {
            uint32_t v = static_cast<uint32_t>(data_arr.at(i).as_int());
            std::memcpy(packed.data() + i * 4, &v, 4);
        }
    } else if (qnn_dtype == 0x0232) {
        // float32: 4 bytes each
        packed.resize(n * 4);
        for (size_t i = 0; i < n; ++i) {
            float v = static_cast<float>(data_arr.at(i).as_num());
            std::memcpy(packed.data() + i * 4, &v, 4);
        }
    } else {
        packed.resize(n * 4, 0);
    }
    return packed;
}

void QnnIRLoader::parse_tensors(const JsonValue& graph_json) {
    const JsonValue& tensors = graph_json.at("tensors");
    for (const auto& [name, tjson] : tensors.obj_val) {
        QnnTensorInfo ti;
        ti.name = name;
        ti.id = static_cast<uint32_t>(tjson.at("id").as_int());
        ti.type = static_cast<uint32_t>(tjson.at("type").as_int());
        ti.data_type = static_cast<uint32_t>(tjson.at("data_type").as_int());
        const JsonValue& dims_arr = tjson.at("dims");
        for (size_t i = 0; i < dims_arr.size(); ++i)
            ti.dims.push_back(static_cast<uint32_t>(dims_arr.at(i).as_int()));
        tensors_[name] = std::move(ti);
    }
}

void QnnIRLoader::extract_tensor_params(QnnNodeInfo& node, const JsonValue& node_json) {
    if (!node_json.contains("tensor_params")) return;
    const JsonValue& tp = node_json.at("tensor_params");
    for (const auto& [param_name, inner] : tp.obj_val) {
        // inner is an object: {tensor_name: {id, type, data_type, dims, data, ...}}
        for (const auto& [tname, tjson] : inner.obj_val) {
            QnnTensorInfo ti;
            ti.name = tname;
            ti.is_param = true;
            ti.id = static_cast<uint32_t>(tjson.at("id").as_int());
            ti.type = static_cast<uint32_t>(tjson.at("type").as_int());
            ti.data_type = static_cast<uint32_t>(tjson.at("data_type").as_int());
            const JsonValue& dims_arr = tjson.at("dims");
            for (size_t i = 0; i < dims_arr.size(); ++i)
                ti.dims.push_back(static_cast<uint32_t>(dims_arr.at(i).as_int()));
            if (tjson.contains("data") && tjson.at("data").is_array())
                ti.data = pack_data(tjson.at("data"), ti.data_type);
            node.tensor_params.push_back(std::move(ti));
        }
    }
}

void QnnIRLoader::parse_nodes(const JsonValue& graph_json) {
    const JsonValue& nodes_json = graph_json.at("nodes");
    for (const auto& [name, njson] : nodes_json.obj_val) {
        QnnNodeInfo ni;
        ni.name = name;
        ni.type = njson.at("type").as_str();
        const JsonValue& ins = njson.at("input_names");
        for (size_t i = 0; i < ins.size(); ++i)
            ni.input_names.push_back(ins.at(i).as_str());
        const JsonValue& outs = njson.at("output_names");
        for (size_t i = 0; i < outs.size(); ++i)
            ni.output_names.push_back(outs.at(i).as_str());
        extract_tensor_params(ni, njson);
        nodes_.push_back(std::move(ni));
    }
}

void QnnIRLoader::build_tensor_opid_map() {
    // Only set tensor_opids_ for non-param tensors.
    // Param tensors (is_param=true) are created with dedup in build_graph's
    // AppendInputParams section — they must NOT be in tensor_opids_ here,
    // otherwise build_graph will "reuse" IDs that were never created as nodes.
    for (const auto& [name, ti] : tensors_)
        if (!ti.is_param) tensor_opids_[name] = ti.id;
    // For load_net_json path: tensor_params aren't in tensors_ yet, so add them.
    // For load_cpp path: tensor_params are already in tensors_ (from Phase 3).
    for (const auto& node : nodes_) {
        for (const auto& tp : node.tensor_params) {
            if (!tensors_.count(tp.name))
                tensors_[tp.name] = tp;
        }
    }
}

uint32_t QnnIRLoader::load_net_json(const std::string& path) {
    JsonValue root = parse_json_file(path);
    const JsonValue& graph_json = root.at("graph");
    parse_tensors(graph_json);
    parse_nodes(graph_json);
    build_tensor_opid_map();
    return build_graph();
}

uint32_t QnnIRLoader::build_graph() {
    std::map<std::string, size_t> tensor_producer;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        for (const auto& out_name : nodes_[i].output_names)
            tensor_producer[out_name] = i;
    }

    struct CreateItem {
        uint32_t tensor_id;
        int kind; // 0=input, 1=const, 2=op, 3=output_sink
        std::string tensor_name;
        const QnnNodeInfo* node = nullptr;
    };
    std::vector<CreateItem> items;
    for (const auto& [name, ti] : tensors_) {
        if (ti.type == 0) {
            items.push_back({ti.id, 0, name, nullptr});
        } else if (ti.type == 4) {
            items.push_back({ti.id, 1, name, nullptr});
        } else if (ti.type == 3 || ti.type == 1) {
            auto it = tensor_producer.find(name);
            if (it != tensor_producer.end())
                items.push_back({ti.id, 2, name, &nodes_[it->second]});
        }
    }
    std::sort(items.begin(), items.end(),
              [](const CreateItem& a, const CreateItem& b) { return a.tensor_id < b.tensor_id; });

    // Pre-pass: build const_by_N_ map for wscale dedup.
    // Maps N → first op_id of a 1D const tensor with dims=[N], dt=562, N>1.
    // When FC/MatMul (without bias) needs a per-axis scale of dimension N,
    // QNN reuses the first existing [N] const tensor (e.g., LN bias).
    std::map<uint32_t, op_id_t> const_by_N;
    for (const auto& item : items) {
        if (item.kind != 1) continue;
        const QnnTensorInfo& ti = tensors_.at(item.tensor_name);
        if (ti.is_param) continue;
        if (ti.data_type != 562) continue;
        if (ti.dims.size() != 1 || ti.dims[0] <= 1) continue;
        uint32_t N = ti.dims[0];
        if (const_by_N.find(N) == const_by_N.end())
            const_by_N[N] = ti.id;
    }

    uint32_t op_count = 0;
    for (const auto& item : items) {
        const QnnTensorInfo& ti = tensors_.at(item.tensor_name);
        DType dt = map_dtype(ti.data_type);

        if (item.kind == 0) {
            OutputDef od;
            fill_output_def(od, ti.dims, dt);
            InputDef no_input{};
            gp_.append_node("Input", ti.id, &no_input, 0, &od, 1, nullptr);
            tensor_opids_[item.tensor_name] = ti.id;
        } else if (item.kind == 1) {
            // Skip tensor_params — they are created with dedup in kind==2
            if (ti.is_param) continue;
            OutputDef od;
            fill_output_def(od, ti.dims, dt);
            size_t elem_count = 1;
            for (auto d : ti.dims) elem_count *= d;
            size_t byte_size = elem_count * static_cast<size_t>(od.element_size);
            std::vector<uint8_t> cdata;
            if (!ti.data.empty()) {
                cdata = ti.data;
            } else {
                // Try to load from weights_ (loaded from .bin)
                // Tensor name for weight vars: extract from name (W, b)
                auto wit = weights_.find(item.tensor_name);
                if (wit != weights_.end() && !wit->second.empty()) {
                    cdata = wit->second;
                } else {
                    cdata.assign(byte_size, 0);
                }
            }
            gp_.append_const_node(ti.id, od, cdata.data(), cdata.size());
            tensor_opids_[item.tensor_name] = ti.id;
        } else if (item.kind == 2) {
            const QnnNodeInfo* node = item.node;
            OutputDef od;
            fill_output_def(od, ti.dims, dt);

            std::vector<InputDef> inputs;
            for (const auto& in_name : node->input_names) {
                auto it = tensor_opids_.find(in_name);
                if (it == tensor_opids_.end()) {
                    std::fprintf(stderr, "QnnIRLoader: input tensor '%s' not found for node '%s'\n",
                                 in_name.c_str(), node->name.c_str());
                    return 0;
                }
                InputDef id{};
                id.rank = static_cast<uint32_t>(it->second);
                id.dtype = 0;
                inputs.push_back(id);
            }

            // Create const nodes for tensor_params (perm, axes, stride, pad, dilation).
            // composeGraphs 阶段: 这些是独立 const 节点(用 tp.id, 低地址段),
            // 不放入 op inputs。HtpPrepare(do_prepare1) 注入阶段才追加到 inputs。
            std::vector<op_id_t> tp_ids;
            for (const auto& tp : node->tensor_params) {
                if (tensor_opids_.find(tp.name) != tensor_opids_.end()) {
                    tp_ids.push_back(tensor_opids_[tp.name]);
                    continue;
                }
                op_id_t param_id = tp.id;
                DType ptdt = map_dtype(tp.data_type);
                OutputDef pod;
                fill_output_def(pod, tp.dims, ptdt);
                std::vector<uint8_t> pdata;
                if (!tp.data.empty()) {
                    pdata = tp.data;
                } else {
                    size_t ec = 1;
                    for (auto d : tp.dims) ec *= d;
                    pdata.assign(ec * pod.element_size, 0);
                }
                gp_.append_const_node(static_cast<uint32_t>(param_id), pod, pdata.data(), pdata.size());
                // 参数 const 带名字(inject_htp_prepare_inputs 与 extra_info extractor
                // 都按名匹配 stride/pad_amount/dilation/perm)
                if (OpDef* pc = gp_.get_op_at(static_cast<uint32_t>(param_id)))
                    pc->name_tag = string_tag_t::map_str(tp.name.c_str());
                tensor_opids_[tp.name] = param_id;
                param_dedup_[tp.name] = param_id;
                tp_ids.push_back(param_id);
            }

            // composeGraphs 阶段: op inputs 只含 data 输入, 不注入 HtpPrepare 常量
            // (quant_marker/scale/output_rank 的注入挪到 do_prepare1)。
            // tensor_param ids 存到 OpDef::tensor_param_ids, 供 do_prepare1 注入。

            // 节点 id = tensor id; 若已被占(如该 op 自己的 tensor_param const
            // 恰好用同一 id —— 实测 X_0231 Transpose 与 perm const 撞 id=5 被
            // 静默丢弃), 自动后移到空闲 id(通用防碰撞)。
            op_id_t target = ti.id;
            while (gp_.get_op_at(target) != nullptr) target++;
            gp_.append_node(node->type, target,
                            inputs.data(), inputs.size(),
                            &od, 1, nullptr);
            // Set grouping = original node name (for before/after graph dump)
            OpDef* created = gp_.get_op_at(target);
            if (created) {
                created->grouping = node->name;
                created->tensor_param_ids = std::move(tp_ids);
            }
            tensor_opids_[item.tensor_name] = target;

            // Register this op as consumer of its tensor_params (for DCE).
            // tensor_params are not in op inputs, so append_node's consumer
            // registration misses them. Without this, DCE would delete
            // tensor_param const nodes (like Split's split_index) as
            // having no consumers.
            // NOTE: rebuild_consumers() at end of build_graph clears and
            // rebuilds consumers from inputs only, so we must re-register
            // tensor_param consumers AFTER rebuild_consumers().
            ++op_count;
        }
    }

    // Create Output sink node for APP_READ tensor.
    // composeGraphs 阶段: sink id 用固定 100(QNN SDK 约定, simple_linear tensor id 1-10 不冲突)。
    // 若 100 已被占用, 退到 max_tensor_id + 1。
    for (const auto& [name, ti] : tensors_) {
        if (ti.type == 1) {
            auto it = tensor_opids_.find(name);
            if (it == tensor_opids_.end()) continue;
            InputDef sink_input{};
            sink_input.rank = static_cast<uint32_t>(it->second);
            sink_input.dtype = 0;
            op_id_t out_id = 100;
            if (gp_.get_op_at(out_id) != nullptr) {
                op_id_t max_id = 0;
                for (const auto& [tname, tinfo] : tensors_)
                    if (tinfo.id > max_id) max_id = tinfo.id;
                out_id = max_id + 1;
            }
            gp_.append_node("Output", out_id, &sink_input, 1, nullptr, 0, nullptr);
            break;
        }
    }

    // Rebuild consumer lists: param const nodes (high IDs) were created
    // after the ops that reference them, so append_node's consumer
    // registration missed them. Rebuild from input connections.
    gp_.rebuild_consumers();

    // Register tensor_params consumers AFTER rebuild_consumers (通用)。
    // tensor_params 不在 op inputs 里, rebuild_consumers 看不到它们; 不注册
    // 的话 initial DCE 会把仍在使用的参数 const 删掉 —— 实测 Transpose 的
    // perm(以及 Conv 的 stride/pad/dilation)被删, 序列化/执行全丢参数。
    // (QNN before_graph 无 axes 是 dump 行为, 本仓以正确性为准: 全部注册)
    for (const auto& node : nodes_) {
        if (node.output_names.empty()) continue;
        auto oit = tensor_opids_.find(node.output_names[0]);
        if (oit == tensor_opids_.end()) continue;
        op_id_t op_id = oit->second;
        for (const auto& tp : node.tensor_params) {
            auto pit = tensor_opids_.find(tp.name);
            if (pit == tensor_opids_.end()) continue;
            OpDef* param_op = gp_.get_op_at(pit->second);
            if (!param_op) continue;
            auto& cs = param_op->consumers;
            if (std::find(cs.begin(), cs.end(), op_id) == cs.end())
                cs.push_back(op_id);
        }
    }

    return op_count;
}

// ============================================================================
// .cpp source parser
// ============================================================================

std::vector<std::string> QnnIRLoader::extract_functions(const std::string& src, const std::string& prefix) {
    std::vector<std::string> result;
    std::string needle = prefix;
    size_t pos = 0;
    while ((pos = src.find(needle, pos)) != std::string::npos) {
        size_t brace_start = src.find('{', pos);
        if (brace_start == std::string::npos) break;
        int depth = 1;
        size_t i = brace_start + 1;
        while (i < src.size() && depth > 0) {
            if (src[i] == '{') depth++;
            else if (src[i] == '}') depth--;
            i++;
        }
        result.push_back(src.substr(pos, i - pos));
        pos = i;
    }
    return result;
}

// Extract identifier after a field marker like ".type= QNN_TENSOR_TYPE_APP_WRITE"
static std::string extract_identifier(const std::string& func, const std::string& field) {
    size_t pos = func.find(field);
    if (pos == std::string::npos) return "";
    pos += field.size();
    while (pos < func.size() && (func[pos] == ' ' || func[pos] == '\t')) pos++;
    size_t start = pos;
    while (pos < func.size() && (std::isalnum(static_cast<unsigned char>(func[pos])) || func[pos] == '_')) pos++;
    return func.substr(start, pos - start);
}

std::string QnnIRLoader::extract_string_arg(const std::string& func, const std::string& field, size_t start) {
    size_t pos = func.find(field, start);
    if (pos == std::string::npos) return "";
    pos = func.find('"', pos + field.size());
    if (pos == std::string::npos) return "";
    size_t end = func.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return func.substr(pos + 1, end - pos - 1);
}

std::vector<uint32_t> QnnIRLoader::extract_dims_array(const std::string& func, const std::string& var_name) {
    std::vector<uint32_t> dims;
    std::string needle = var_name + "[] = {";
    size_t pos = func.find(needle);
    if (pos == std::string::npos) { needle = var_name + "[]={"; pos = func.find(needle); }
    if (pos == std::string::npos) return dims;
    size_t brace_end = func.find('}', pos);
    if (brace_end == std::string::npos) return dims;
    std::string content = func.substr(pos + needle.size(), brace_end - pos - needle.size());
    std::stringstream ss(content);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t");
        if (s == std::string::npos) continue;
        dims.push_back(static_cast<uint32_t>(std::stoul(tok.substr(s))));
    }
    return dims;
}

std::vector<std::string> QnnIRLoader::extract_string_list(const std::string& func, const std::string& var_name) {
    std::vector<std::string> result;
    std::string needle = var_name + "[] = {";
    size_t pos = func.find(needle);
    if (pos == std::string::npos) { needle = var_name + "[]={"; pos = func.find(needle); }
    if (pos == std::string::npos) return result;
    size_t brace_end = func.find('}', pos);
    if (brace_end == std::string::npos) return result;
    size_t i = pos + needle.size();
    while (i < brace_end) {
        size_t q1 = func.find('"', i);
        if (q1 == std::string::npos || q1 >= brace_end) break;
        size_t q2 = func.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 >= brace_end) break;
        result.push_back(func.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return result;
}

std::vector<uint8_t> QnnIRLoader::extract_inline_data(const std::string& func, const std::string& var_name) {
    std::vector<uint8_t> data;
    // Search for the variable declaration: var_name[] = { ... }
    // The array data is inline in the .cpp function body.
    // Must skip "dimensions_" + var_name, which also contains var_name[] as a substring.
    std::string needle = var_name + "[]";
    size_t pos = func.find(needle);
    while (pos != std::string::npos) {
        // Ensure the char before the match is not an identifier char
        // (skips "dimensions_conv1_pad_amount[]" matching "conv1_pad_amount[]")
        if (pos == 0 || (!std::isalnum(static_cast<unsigned char>(func[pos - 1])) && func[pos - 1] != '_')) {
            break;
        }
        pos = func.find(needle, pos + 1);
    }
    if (pos == std::string::npos) return data;
    // Find the opening brace after the assignment
    size_t brace_start = func.find('{', pos);
    if (brace_start == std::string::npos) return data;
    size_t brace_end = func.find('}', brace_start);
    if (brace_end == std::string::npos) return data;
    std::string content = func.substr(brace_start + 1, brace_end - brace_start - 1);
    std::stringstream ss(content);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t s = tok.find_first_not_of(" \t\n\r");
        if (s == std::string::npos) continue;
        uint32_t v = static_cast<uint32_t>(std::stoul(tok.substr(s)));
        uint8_t* p = reinterpret_cast<uint8_t*>(&v);
        data.insert(data.end(), p, p + 4);
    }
    return data;
}

std::string QnnIRLoader::map_qnn_tensor_type(const std::string& type_str) {
    if (type_str == "QNN_TENSOR_TYPE_APP_WRITE") return "input";
    if (type_str == "QNN_TENSOR_TYPE_APP_READ") return "output";
    if (type_str == "QNN_TENSOR_TYPE_NATIVE") return "native";
    if (type_str == "QNN_TENSOR_TYPE_STATIC") return "static";
    return "native";
}

DType QnnIRLoader::map_qnn_datatype(const std::string& dt_str) {
    if (dt_str == "QNN_DATATYPE_FLOAT_32") return DType::Float32;
    if (dt_str == "QNN_DATATYPE_FLOAT_16") return DType::Float16;
    if (dt_str == "QNN_DATATYPE_INT_32") return DType::Int32;
    if (dt_str == "QNN_DATATYPE_UINT_32") return DType::Int32;
    if (dt_str == "QNN_DATATYPE_SINT_8") return DType::Int8;
    if (dt_str == "QNN_DATATYPE_UINT_8") return DType::UInt8;
    return DType::Float32;
}

uint32_t QnnIRLoader::load_cpp(const std::string& cpp_path) {
    std::ifstream f(cpp_path);
    if (!f) { std::fprintf(stderr, "QnnIRLoader: cannot open %s\n", cpp_path.c_str()); return 0; }
    std::stringstream buf;
    buf << f.rdbuf();
    std::string src = buf.str();

    // ===== Phase 1: Parse all function definitions (no ID assignment) =====
    // addTensor_<name> → QnnTensorInfo (stored in tensor_defs)
    std::map<std::string, QnnTensorInfo> tensor_defs;
    auto tensor_funcs = extract_functions(src, "static ModelError_t addTensor_");
    for (const auto& func : tensor_funcs) {
        size_t fn_start = func.find("addTensor_");
        std::string fname = func.substr(fn_start + 10);
        size_t paren = fname.find('(');
        if (paren != std::string::npos) fname = fname.substr(0, paren);

        QnnTensorInfo ti;
        ti.name = fname;
        std::string type_str = extract_identifier(func, ".type=");
        ti.type = 0;
        if (type_str == "QNN_TENSOR_TYPE_APP_WRITE") ti.type = 0;
        else if (type_str == "QNN_TENSOR_TYPE_STATIC") ti.type = 4;
        else if (type_str == "QNN_TENSOR_TYPE_NATIVE") ti.type = 3;
        else if (type_str == "QNN_TENSOR_TYPE_APP_READ") ti.type = 1;

        std::string dt_str = extract_identifier(func, ".dataType=");
        if (dt_str == "QNN_DATATYPE_FLOAT_32") ti.data_type = 0x0232;
        else if (dt_str == "QNN_DATATYPE_UINT_32") ti.data_type = 0x0132;
        else if (dt_str == "QNN_DATATYPE_INT_32") ti.data_type = 0x0032;
        else if (dt_str == "QNN_DATATYPE_FLOAT_16") ti.data_type = 0x0332;
        else if (dt_str == "QNN_DATATYPE_SINT_8") ti.data_type = 0x0832;
        else if (dt_str == "QNN_DATATYPE_UINT_8") ti.data_type = 0x0732;
        else ti.data_type = 0x0232;

        std::string dim_var = "dimensions_" + fname;
        ti.dims = extract_dims_array(func, dim_var);

        if (func.find("BINVARSTART(") != std::string::npos) {
            size_t bv = func.find("BINVARSTART(");
            size_t rp = func.find(')', bv + 12);
            std::string var = func.substr(bv + 12, rp - bv - 12);
            if (weights_.find(var) == weights_.end()) weights_[var] = {};
        }
        tensor_defs[fname] = std::move(ti);
    }

    // addNode_<name> → parsed node + output tensor infos (stored in node_defs)
    struct ParsedNodeDef {
        QnnNodeInfo info;
        std::vector<QnnTensorInfo> output_tensors;
    };
    std::map<std::string, ParsedNodeDef> node_defs;

    auto node_funcs = extract_functions(src, "static ModelError_t addNode_");
    for (const auto& func : node_funcs) {
        size_t fn_start = func.find("addNode_");
        std::string nname = func.substr(fn_start + 8);
        size_t paren = nname.find('(');
        if (paren != std::string::npos) nname = nname.substr(0, paren);

        ParsedNodeDef pnd;
        pnd.info.name = nname;

        // Node type: find "qti.aisw", skip comment, then next quoted string
        {
            size_t pkg = func.find("\"qti.aisw\"");
            if (pkg != std::string::npos) {
                size_t q = func.find('"', pkg + 10);
                while (q != std::string::npos) {
                    size_t line_start = func.rfind('\n', q);
                    if (line_start == std::string::npos) line_start = 0;
                    std::string before = func.substr(line_start, q - line_start);
                    if (before.find("//") != std::string::npos) {
                        size_t eol = func.find('\n', q);
                        if (eol == std::string::npos) break;
                        q = func.find('"', eol);
                        continue;
                    }
                    size_t end_q = func.find('"', q + 1);
                    if (end_q != std::string::npos)
                        pnd.info.type = func.substr(q + 1, end_q - q - 1);
                    break;
                }
            }
        }

        // inputs
        std::string in_var = "inputs_" + nname;
        pnd.info.input_names = extract_string_list(func, in_var);

        // outputs: parse each output tensor's name + type + dtype + dims
        std::string out_var = "outputs_" + nname;
        size_t out_pos = func.find(out_var + "[] = {");
        if (out_pos == std::string::npos) out_pos = func.find(out_var + "[]={{");
        if (out_pos != std::string::npos) {
            size_t search = out_pos;
            while (search < func.size()) {
                size_t name_pos = func.find(".name= \"", search);
                if (name_pos == std::string::npos) break;
                size_t end_quote = func.find('"', name_pos + 8);
                if (end_quote == std::string::npos) break;
                std::string out_name = func.substr(name_pos + 8, end_quote - name_pos - 8);
                pnd.info.output_names.push_back(out_name);
                search = end_quote + 1;

                QnnTensorInfo oti;
                oti.name = out_name;
                std::string out_substr = func.substr(name_pos, 400);
                std::string t_type = extract_identifier(out_substr, ".type=");
                std::string t_dt = extract_identifier(out_substr, ".dataType=");
                oti.dims = extract_dims_array(func, "dimensions_" + out_name);
                if (t_dt == "QNN_DATATYPE_FLOAT_32") oti.data_type = 0x0232;
                else if (t_dt == "QNN_DATATYPE_UINT_32") oti.data_type = 0x0132;
                else if (t_dt == "QNN_DATATYPE_INT_32") oti.data_type = 0x0032;
                else if (t_dt == "QNN_DATATYPE_FLOAT_16") oti.data_type = 0x0332;
                else if (t_dt == "QNN_DATATYPE_SINT_8") oti.data_type = 0x0832;
                else if (t_dt == "QNN_DATATYPE_UINT_8") oti.data_type = 0x0732;
                else oti.data_type = 0x0232;
                if (t_type == "QNN_TENSOR_TYPE_APP_READ") oti.type = 1;
                else if (t_type == "QNN_TENSOR_TYPE_APP_WRITE") oti.type = 0;
                else if (t_type == "QNN_TENSOR_TYPE_STATIC") oti.type = 4;
                else oti.type = 3;
                pnd.output_tensors.push_back(std::move(oti));
            }
        }

        // tensor params (e.g. perm for Transpose)
        // Limit search to the params[] declaration block, which ends at
        // the inputs_[] or outputs_[] declaration (whichever comes first).
        std::string param_var = "params_" + nname;
        size_t param_pos = func.find(param_var + "[] = {");
        if (param_pos != std::string::npos) {
            // Find the end of params block: next declaration after param_pos
            size_t inputs_pos = func.find("inputs_" + nname, param_pos);
            size_t outputs_pos = func.find("outputs_" + nname, param_pos);
            size_t param_end = func.size();
            if (inputs_pos != std::string::npos) param_end = std::min(param_end, inputs_pos);
            if (outputs_pos != std::string::npos) param_end = std::min(param_end, outputs_pos);

            size_t search = param_pos;
            while (search < param_end) {
                size_t name_pos = func.find(".name=\"", search);
                if (name_pos == std::string::npos || name_pos >= param_end) break;
                size_t end_quote = func.find('"', name_pos + 7);
                if (end_quote == std::string::npos || end_quote >= param_end) break;
                search = end_quote + 1;

                // Only process TENSOR params (QNN_PARAMTYPE_TENSOR), not SCALAR
                // Check: is there a .tensorParam= between name and next .name?
                size_t tparam = func.find(".tensorParam=", search);
                size_t next_name = func.find(".name=", search);
                if (tparam == std::string::npos || (next_name != std::string::npos && tparam > next_name))
                    continue; // SCALAR param, skip

                size_t tname_pos = func.find(".name= \"", search);
                if (tname_pos == std::string::npos || tname_pos >= param_end) break;
                size_t tend_quote = func.find('"', tname_pos + 8);
                if (tend_quote == std::string::npos || tend_quote >= param_end) break;
                std::string tensor_name = func.substr(tname_pos + 8, tend_quote - tname_pos - 8);

                QnnTensorInfo pti;
                pti.name = tensor_name;
                pti.is_param = true;
                pti.type = 4;
                std::string dt_str = extract_identifier(func.substr(search, param_end - search), ".dataType=");
                if (dt_str == "QNN_DATATYPE_FLOAT_32") pti.data_type = 0x0232;
                else if (dt_str == "QNN_DATATYPE_UINT_32") pti.data_type = 0x0132;
                else if (dt_str == "QNN_DATATYPE_INT_32") pti.data_type = 0x0032;
                else if (dt_str == "QNN_DATATYPE_FLOAT_16") pti.data_type = 0x0332;
                else if (dt_str == "QNN_DATATYPE_SINT_8") pti.data_type = 0x0832;
                else if (dt_str == "QNN_DATATYPE_UINT_8") pti.data_type = 0x0732;
                else pti.data_type = 0x0232;
                pti.dims = extract_dims_array(func, "dimensions_" + tensor_name);
                std::vector<uint8_t> d = extract_inline_data(func, tensor_name);
                if (!d.empty()) pti.data = d;
                pnd.info.tensor_params.push_back(std::move(pti));
                search = tend_quote + 1;
            }
        }

        node_defs[nname] = std::move(pnd);
    }

    // ===== Phase 2: Parse QnnModel_composeGraphs for call order =====
    // Extract the function body and find all addTensor_*/addNode_* calls in order
    std::vector<std::pair<int, std::string>> call_order; // (0=addTensor, 1=addNode, func_name)
    {
        std::string needle = "QnnModel_composeGraphs";
        size_t pos = src.find(needle);
        if (pos != std::string::npos) {
            size_t brace_start = src.find('{', pos);
            if (brace_start != std::string::npos) {
                int depth = 1;
                size_t i = brace_start + 1;
                while (i < src.size() && depth > 0) {
                    if (src[i] == '{') depth++;
                    else if (src[i] == '}') depth--;
                    i++;
                }
                std::string body = src.substr(brace_start, i - brace_start);
                // Find all VALIDATE(addTensor_<X>( or VALIDATE(addNode_<Y>(
                size_t bpos = 0;
                while (bpos < body.size()) {
                    size_t at = body.find("addTensor_", bpos);
                    size_t an = body.find("addNode_", bpos);
                    size_t next = std::string::npos;
                    int kind = -1;
                    if (at != std::string::npos && (an == std::string::npos || at < an)) {
                        next = at; kind = 0;
                    } else if (an != std::string::npos) {
                        next = an; kind = 1;
                    }
                    if (next == std::string::npos) break;
                    std::string prefix = (kind == 0) ? "addTensor_" : "addNode_";
                    size_t name_start = next + prefix.size();
                    size_t name_end = body.find('(', name_start);
                    if (name_end == std::string::npos) break;
                    std::string fname = body.substr(name_start, name_end - name_start);
                    call_order.push_back({kind, fname});
                    bpos = name_end;
                }
            }
        }
    }

    // ===== Phase 3: Walk call order, assign IDs in QNN SDK order =====
    uint32_t next_id = 1;
    for (const auto& [kind, fname] : call_order) {
        if (kind == 0) {
            auto it = tensor_defs.find(fname);
            if (it == tensor_defs.end()) continue;
            QnnTensorInfo ti = it->second;
            ti.id = next_id++;
            tensors_[ti.name] = std::move(ti);
        } else {
            auto it = node_defs.find(fname);
            if (it == node_defs.end()) continue;
            ParsedNodeDef& pnd = it->second;

            for (auto& pti : pnd.info.tensor_params) {
                pti.id = next_id++;
                tensors_[pti.name] = pti;
            }
            for (auto& oti : pnd.output_tensors) {
                oti.id = next_id++;
                tensors_[oti.name] = oti;
            }
            nodes_.push_back(std::move(pnd.info));
        }
    }

    build_tensor_opid_map();
    return build_graph();
}

// ============================================================================
// .bin (TAR archive) weight loader
// ============================================================================

std::map<std::string, std::vector<uint8_t>> QnnIRLoader::load_weight_bin(const std::string& bin_path) {
    std::map<std::string, std::vector<uint8_t>> result;
    std::ifstream f(bin_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "QnnIRLoader: cannot open %s\n", bin_path.c_str()); return result; }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    size_t pos = 0;
    while (pos + 512 <= buf.size()) {
        // TAR header: name at [0,100), size octal at [124,136)
        std::string name(reinterpret_cast<const char*>(buf.data() + pos), 100);
        name = name.substr(0, name.find('\0'));
        if (name.empty()) break;

        // size field at offset 124, 12 bytes octal
        std::string size_str(reinterpret_cast<const char*>(buf.data() + pos + 124), 11);
        size_str = size_str.substr(0, size_str.find_first_of(" \0"));
        if (size_str.empty()) break;
        uint64_t fsize = std::stoul(size_str, nullptr, 8);

        // data starts at pos + 512
        if (pos + 512 + fsize > buf.size()) break;
        std::vector<uint8_t> data(buf.begin() + pos + 512, buf.begin() + pos + 512 + fsize);

        // strip ".raw" suffix -> var name (W.raw -> W)
        if (name.size() > 4 && name.substr(name.size() - 4) == ".raw")
            name = name.substr(0, name.size() - 4);
        result[name] = std::move(data);

        // advance to next 512-aligned block
        pos += 512 + ((fsize + 511) / 512) * 512;
        // TAR ends with two zero blocks
        bool all_zero = true;
        for (int i = 0; i < 512 && pos + i < buf.size(); i++) {
            if (buf[pos + i] != 0) { all_zero = false; break; }
        }
        if (all_zero) break;
    }
    return result;
}

uint32_t QnnIRLoader::load_qnn_ir(const std::string& cpp_path, const std::string& bin_path) {
    // 1. Load weights from .bin first
    weights_ = load_weight_bin(bin_path);

    // 2. Load graph structure from .cpp
    return load_cpp(cpp_path);
}

} // namespace hnnx
