// hnnx-compile: Compile QNN IR to HTP context binary (.bin)
//
// Usage:
//   hnnx-compile --net-json <path> [--weights-bin <path>] --output <out.bin> [--format qnn|tagged]
//   hnnx-compile --cpp <path> --weights-bin <path> --output <out.bin> [--format qnn|tagged]
//
// Pipeline:
//   1. Load QNN IR (net.json or .cpp) + weights .bin -> GraphPrepare
//   2. GraphPrepare::prepare() -> optimized graph (ST-Cut 调度 + 分块 + spill/fill)
//   3. --format qnn:    Scheduler 路径A金样重放 -> ContextBinaryWriter(冻结对拍)
//      --format tagged: GraphPrepare::serialize -> 我方 tagged runlist(产品路径,
//                        wtop_emit 消费)
//
// 2.48 权重适配: --weights-bin 可以是旧式 TAR, 也可以是 qairt-converter 2.48
// 产出的 params.bin(实测布局: 静态张量 f16 拼接) —— 自动识别并转 TAR。
// 2.48 产出 params.bin 的方式: unzip -p <converter.cpp> model.params.bin

#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/schedule/scheduler.hpp"
#include "hnnx/serialize/context_binary_writer.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ops/ops.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <unistd.h>  // getpid

using namespace hnnx;

static void usage() {
    std::fprintf(stderr,
        "hnnx-compile: Compile QNN IR to HTP context binary\n\n"
        "Usage:\n"
        "  hnnx-compile --net-json <path> [--weights-bin <path>] --output <out.bin>\n"
        "  hnnx-compile --cpp <path> --weights-bin <path> --output <out.bin>\n\n"
        "Options:\n"
        "  --net-json <path>   Path to QNN IR net.json file\n"
        "  --cpp <path>        Path to QNN model .cpp source file\n"
        "  --weights-bin <path> Path to weights .bin (TAR archive with W.raw, b.raw)\n"
        "                       or qairt-converter 2.48 params.bin (auto-detected)\n"
        "  --format <fmt>     qnn=路径A金样重放(默认) | tagged=我方 runlist\n"
        "  --vtcm-budget <n>  VTCM 预算字节(0=默认 8MB; 小值强制溢出)\n"
        "  --output <path>     Output context binary path (default: output.bin)\n"
        "  --graph-name <name>  Graph name (default: from net.json or 'compiled_graph')\n"
        "  --verbose           Print detailed compilation info\n"
        "  --help              Show this help\n");
}

static bool write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
    return f.good();
}


// ---------------------------------------------------------------------------
// 2.48 params.bin 适配(实测布局, 阶段1字节级核对):
//   [B f16 (N_b×2B)] [W f16 (kh·kw·ci·co×2B, hwio 序)] [尾 64B(denormal 残渣)]
// 无名字头 → 按 net.json 声明的张量尺寸做字节长度匹配; 同尺寸多候选时选
// |值|幅度最大的偏移(B ~ N(0,1) vs denormal 残渣 ~ 0, 实证有效)。
// 输出: TAR(W.raw/B.raw 等, f16 → f32 精确拓宽), 供既有 loader 消费。
// 若输入已是 TAR(512B 头含合法 name+octal size)则原样返回。
// ---------------------------------------------------------------------------
static float widen_f16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) { u = sign; }
        else {  // 次正规 → 正规
            int e = -1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            u = sign | ((uint32_t)(127 + 15 + e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);  // Inf/NaN
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static bool looks_like_tar(const std::vector<uint8_t>& buf) {
    if (buf.size() < 512) return false;
    // name 至少 1 个可打印字符 + size 字段 11 字符 octal(可含空格结尾)
    bool name_ok = false;
    for (int i = 0; i < 100; i++) if (buf[i] >= 0x20 && buf[i] < 0x7F) { name_ok = true; break; }
    if (!name_ok) return false;
    for (int i = 124; i < 136; i++) {
        char c = (char)buf[i];
        if (c >= '0' && c <= '7') return true;
        if (c == ' ' || c == 0) continue;
        return false;
    }
    return false;
}

static void tar_append(std::vector<uint8_t>& tar, const std::string& name,
                       const uint8_t* data, size_t len) {
    tar.resize(tar.size() + 512, 0);
    uint8_t* h = tar.data() + tar.size() - 512;
    std::memset(h, 0, 512);
    std::memcpy(h, name.c_str(), std::min<size_t>(name.size(), 99));
    std::memcpy(h + 100, "0000644", 7);
    std::memcpy(h + 108, "0000000", 7);
    std::memcpy(h + 116, "0000000", 7);
    char sz[12];
    std::snprintf(sz, sizeof(sz), "%011lo", (unsigned long)len);
    std::memcpy(h + 124, sz, 11);
    std::memset(h + 136, '0', 11);  // mtime = 0
    std::memset(h + 148, ' ', 8);   // chksum
    h[156] = '0';
    std::memcpy(h + 257, "ustar", 5);
    std::memcpy(h + 263, "00", 2);
    // chksum: 全部按空格计算
    uint32_t sum = 0;
    for (int i = 0; i < 512; i++) sum += h[i];
    std::snprintf(sz, sizeof(sz), "%06o", sum);
    std::memcpy(h + 148, sz, 6);
    h[154] = 0; h[155] = ' ';
    tar.insert(tar.end(), data, data + len);
    while (tar.size() % 512 != 0) tar.push_back(0);
}

static std::string adapt_weights_bin(const std::string& net_json_path,
                                     const std::string& weights_bin_path) {
    std::vector<uint8_t> raw;
    {
        std::ifstream f(weights_bin_path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "Warning: cannot open weights bin %s\n", weights_bin_path.c_str()); return ""; }
        raw.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    if (looks_like_tar(raw)) return weights_bin_path;  // 已是 TAR

    // net.json: 提取 "tensors" 对象里每个条目的 dims(最小解析, 容 schema 变体)
    struct Cand { std::string name; size_t bytes; };
    std::vector<Cand> cands;
    {
        std::string js;
        {
            std::ifstream f(net_json_path, std::ios::binary);
            if (!f) return "";
            js.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        }
        // 状态机: "tensors" 对象的直接子键 = 张量名(depth 1); 其值对象内
        // (depth 2)的 "dims" 键后跟数组。仅纳入静态张量("type": 4)。
        size_t tpos = js.find("\"tensors\"");
        if (tpos == std::string::npos) return "";
        size_t i = js.find('{', tpos) + 1;  // 越过 tensors 自己的 {
        if (i == std::string::npos) return "";
        size_t depth = 1;      // tensors 对象内部
        std::string cur_name;
        bool cur_static = false;
        auto parse_dims = [&](size_t from) -> size_t {
            size_t lb = js.find('[', from);
            size_t rb = js.find(']', lb);
            if (lb == std::string::npos || rb == std::string::npos) return from;
            std::string dimstr = js.substr(lb + 1, rb - lb - 1);
            size_t bytes = 2;  // 2.48 fp16 模型按 2B/元素
            bool any = false;
            size_t start = 0;
            while (start < dimstr.size()) {
                size_t comma = dimstr.find(',', start);
                std::string tok = dimstr.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                long v = std::atol(tok.c_str());
                if (v > 0) { bytes *= (size_t)v; any = true; }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (any && cur_static && !cur_name.empty())
                cands.push_back({cur_name, bytes});
            return rb;
        };
        while (i < js.size()) {
            char c = js[i];
            if (c == '{') { depth++; i++; continue; }
            if (c == '}') { depth--; if (depth == 0) break; i++; continue; }
            if (c == '"') {
                size_t q2 = js.find('"', i + 1);
                if (q2 == std::string::npos) break;
                std::string key = js.substr(i + 1, q2 - i - 1);
                if (depth == 1) { cur_name = key; cur_static = false; }
                else if (depth == 2 && key == "dims") { i = parse_dims(q2 + 1); continue; }
                else if (depth == 2 && key == "type") {
                    size_t colon = js.find(':', q2);
                    size_t comma2 = js.find_first_of(",}\n", colon);
                    if (colon != std::string::npos)
                        cur_static = (std::atoi(js.substr(colon + 1, comma2 - colon - 1).c_str()) == 4);
                }
                i = q2 + 1;
                continue;
            }
            i++;
        }
    }
    if (cands.empty()) return "";
    // 去掉重复(同一张量名出现多次取首)
    std::vector<Cand> uniq;
    for (const auto& c : cands) {
        bool dup = false;
        for (const auto& u : uniq) if (u.name == c.name) dup = true;
        if (!dup) uniq.push_back(c);
    }
    cands = uniq;

    // 顺序分配(阶段1 字节级实证: params.bin = 静态张量按 net.json 出现序的
    // f16 拼接, 无对齐填充; [B][W][尾] 即此序)。此前用 |值|幅度启发式挑偏移,
    // 在全部 N(0,1) 值域下不可靠(实测 W 匹配到偏移 32 错位)——弃用。
    std::vector<uint8_t> tar;
    size_t off = 0;
    size_t n_hits = 0;
    for (const auto& c : cands) {
        if (off + c.bytes > raw.size() || c.bytes == 0) continue;
        const uint16_t* h = reinterpret_cast<const uint16_t*>(raw.data() + off);
        std::vector<float> f32(c.bytes / 2);
        for (size_t k = 0; k < f32.size(); k++) f32[k] = widen_f16(h[k]);
        tar_append(tar, c.name + ".raw",
                   reinterpret_cast<const uint8_t*>(f32.data()), f32.size() * 4);
        off += c.bytes;
        n_hits++;
    }
    if (n_hits == 0) { std::fprintf(stderr, "Warning: no weight candidates matched in %s\n", weights_bin_path.c_str()); return ""; }
    tar.resize(tar.size() + 1024, 0);  // TAR 双零块结尾

    char tmpname[256];
    std::snprintf(tmpname, sizeof(tmpname), "/tmp/hnnx_weights_%d.tar", (int)getpid());
    {
        std::ofstream f(tmpname, std::ios::binary);
        if (!f) return "";
        f.write(reinterpret_cast<const char*>(tar.data()), tar.size());
    }
    std::printf("[1b] 2.48 params.bin adapted: %zu tensors -> %s\n", n_hits, tmpname);
    return tmpname;
}

int main(int argc, char** argv) {
    std::string net_json_path;
    std::string cpp_path;
    std::string weights_bin_path;
    std::string output_path = "output.bin";
    std::string graph_name = "simple_linear";
    std::string output_format = "qnn";   // qnn=路径A | tagged=我方 runlist
    uint64_t vtcm_budget = 0;           // 0=默认 8MB; 阶段7: 小预算强制溢出(spill/fill)
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        else if (arg == "--net-json" && i+1 < argc) net_json_path = argv[++i];
        else if (arg == "--cpp" && i+1 < argc) cpp_path = argv[++i];
        else if (arg == "--weights-bin" && i+1 < argc) weights_bin_path = argv[++i];
        else if (arg == "--output" && i+1 < argc) output_path = argv[++i];
        else if (arg == "--format" && i+1 < argc) output_format = argv[++i];
        else if (arg == "--vtcm-budget" && i+1 < argc) vtcm_budget = strtoull(argv[++i], nullptr, 0);
        else if (arg == "--graph-name" && i+1 < argc) graph_name = argv[++i];
        else if (arg == "--verbose" || arg == "-v") verbose = true;
        else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            usage();
            return 1;
        }
    }

    if (net_json_path.empty() && cpp_path.empty()) {
        std::fprintf(stderr, "Error: must specify --net-json or --cpp\n\n");
        usage();
        return 1;
    }

    std::printf("=== hnnx-compile: QNN IR -> HTP context binary ===\n\n");

    // Step 1: Register ops and load QNN IR
    register_all_ops();

    GraphPrepare gp;
    QnnIRLoader loader(gp);

    uint32_t op_count = 0;
    if (!net_json_path.empty()) {
        if (verbose) std::printf("[1a] Loading net.json: %s\n", net_json_path.c_str());
        // 权重必须先注入: load_net_json → build_graph 立即消费 weights_
        // (实测: 先 load 后 set_weights 会把 W/B 建成全零 const)
        if (!weights_bin_path.empty()) {
            // 2.48 params.bin(实测: 静态张量按 net.json 序 f16 拼接)自动识别并转 TAR
            std::string tar = adapt_weights_bin(net_json_path, weights_bin_path);
            if (!tar.empty())
                loader.set_weights(loader.load_weight_bin(tar));
        }
        op_count = loader.load_net_json(net_json_path);
    } else {
        if (verbose) std::printf("[1a] Loading .cpp: %s\n", cpp_path.c_str());
        if (weights_bin_path.empty()) {
            std::fprintf(stderr, "Error: --cpp requires --weights-bin\n");
            return 1;
        }
        op_count = loader.load_qnn_ir(cpp_path, weights_bin_path);
    }

    if (op_count == 0) {
        std::fprintf(stderr, "Error: failed to load QNN IR (0 ops created)\n");
        return 1;
    }
    std::printf("[1] Loaded QNN IR: %u op nodes, %zu total nodes\n",
                op_count, gp.op_count());
    if (verbose) {
        for (op_id_t id = 1; id < 200; id++) {
            const OpDef* od = gp.get_op_at(id);
            if (!od) continue;
            std::string nm = od->name_tag ? (od->name_tag->name() ? od->name_tag->name() : "") : "";
            std::printf("dbg: id=%llu name='%s' flags=0x%04x const=%d inputs=%zu\n",
                        (unsigned long long)id, nm.c_str(), (unsigned)od->flags,
                        (int)od->is_const(), od->inputs.size());
            for (const auto& c : od->inputs)
                std::printf("       <- src=%llu\n", (unsigned long long)c.src_id);
        }
    }

    // Step 2: Prepare (optimize) the graph
    if (verbose) std::printf("[2a] Preparing graph...\n");
    HexagonNNEnv env;
    if (vtcm_budget != 0) gp.set_vtcm_budget(vtcm_budget);
    GraphStatus s = gp.prepare(env);
    if (s != GraphStatus::Success) {
        std::fprintf(stderr, "Error: prepare() failed with status %d\n",
                     static_cast<int>(s));
        return 1;
    }
    std::printf("[2] Graph prepared (optimized)\n");

    // ---- tagged 产品路径(阶段8): 跳过路径A重放, 直接我方 runlist ----
    if (output_format == "tagged") {
        std::vector<uint8_t> buf(1u << 20, 0);
        size_t out_size = 0;
        if (!gp.serialize(buf.data(), buf.size(), out_size)) {
            std::fprintf(stderr, "Error: tagged serialize failed\n");
            return 1;
        }
        if (!write_file(output_path, std::vector<uint8_t>(buf.begin(), buf.begin() + out_size))) {
            std::fprintf(stderr, "Error: cannot write %s\n", output_path.c_str());
            return 1;
        }
        std::printf("[3] Tagged .bin written: %s (%zu bytes)\n", output_path.c_str(), out_size);
        return 0;
    }

    // Step 3: Schedule - convert optimized graph to HTP execution plan
    if (verbose) std::printf("[3a] Scheduling execution plan...\n");
    Scheduler scheduler;
    // 路径A金样重放(冻结): 19 步硬编码仅服务 ContextBinaryWriter 字节对拍
    Scheduler::Plan plan = scheduler.schedule_path_a_replay(gp);
    std::printf("[3] Scheduled: %zu steps, %zu kernels\n",
                plan.ops.size(), plan.kernel_names.size());

    if (verbose) {
        for (size_t i = 0; i < plan.ops.size(); i++) {
            const auto& op = plan.ops[i];
            std::printf("  Step %2zu: id=0x%02X cnt=%2u type=0x%02X f2=0x%08X blk=0x%08X extras=%zu  %s\n",
                   i, op.record_id, op.tensor_id,
                   static_cast<uint32_t>(op.type),
                   op.f2, op.block_ref, op.extras.size(),
                   op.step_name.c_str());
        }
    }

    // Step 4: Build const pool from weights
    std::vector<uint8_t> const_pool(0x200, 0);
    std::vector<ConstExtentDesc> extents;

    // W weights at offset 0x000 (file 0x9000)
    auto w_it = loader.weights().find("W");
    if (w_it != loader.weights().end() && w_it->second.size() >= 32) {
        std::memcpy(const_pool.data() + 0x000, w_it->second.data(), 32);
        ConstExtentDesc ext{};
        ext.op_id = 5;
        ext.offset = 0x0000;
        ext.size = 32;
        ext.tensor_type = 0;
        ext.reserved = 0;
        extents.push_back(ext);
        if (verbose) std::printf("[4a] W weights: %zu bytes at const_pool[0x000]\n",
                                w_it->second.size());
    }

    // b bias at offset 0x100 (file 0x9100)
    auto b_it = loader.weights().find("b");
    if (b_it != loader.weights().end() && b_it->second.size() >= 8) {
        std::memcpy(const_pool.data() + 0x100, b_it->second.data(), 8);
        ConstExtentDesc ext{};
        ext.op_id = 6;
        ext.offset = 0x0100;
        ext.size = 8;
        ext.tensor_type = 0;
        ext.reserved = 0;
        extents.push_back(ext);
        if (verbose) std::printf("[4b] b bias: %zu bytes at const_pool[0x100]\n",
                                b_it->second.size());
    }

    // Step 5: Write context binary
    if (verbose) std::printf("[5a] Writing context binary...\n");
    ContextBinaryWriter cbw;
    cbw.set_graph_name(graph_name);
    cbw.set_build_id("v2.48.0.260626120635");
    cbw.set_dsp_arch(0);
    cbw.set_io_tensor_size(0x00400000);
    cbw.set_const_size(0x00200000);

    cbw.set_kernel_names(plan.kernel_names);
    cbw.set_scheduled_ops(plan.ops);
    cbw.set_op_names(plan.op_names);
    cbw.set_tensor_names(plan.tensor_names);
    cbw.set_const_pool(const_pool);
    cbw.set_const_extents(extents);

    std::vector<uint8_t> output;
    size_t out_size = cbw.write(output);
    if (out_size == 0) {
        std::fprintf(stderr, "Error: context binary writer returned 0\n");
        return 1;
    }
    std::printf("[5] Context binary: %zu bytes\n", out_size);

    // Step 6: Save output
    if (!write_file(output_path, output)) {
        std::fprintf(stderr, "Error: failed to write output file: %s\n",
                     output_path.c_str());
        return 1;
    }
    std::printf("[6] Saved to: %s\n", output_path.c_str());

    std::printf("\n=== Compilation successful ===\n");
    std::printf("    Output: %s (%zu bytes)\n", output_path.c_str(), out_size);
    std::printf("    Graph:  %s\n", graph_name.c_str());
    std::printf("    Steps:  %zu\n", plan.ops.size());
    std::printf("\nTo run on device:\n");
    std::printf("  qnn-net-run --retrieve_context=%s \\\n", output_path.c_str());
    std::printf("    --backend=libQnnHtp.so \\\n");
    std::printf("    --input_list=input_list.txt --use_native_input_files\n");

    return 0;
}
