// hnnx-compile: Compile QNN IR to HTP context binary (.bin)
//
// Usage:
//   hnnx-compile --net-json <path> [--weights-bin <path>] --output <out.bin>
//   hnnx-compile --cpp <path> --weights-bin <path> --output <out.bin>
//
// Pipeline:
//   1. Load QNN IR (net.json or .cpp) + weights .bin -> GraphPrepare
//   2. GraphPrepare::prepare() -> optimized graph
//   3. Scheduler::schedule() -> 19-step HTP execution plan
//   4. ContextBinaryWriter::write() -> context binary (.bin)
//
// The output .bin can be loaded and executed on HTP DSP hardware
// via qnn-net-run --retrieve_context=<out.bin> --backend=libQnnHtp.so

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

int main(int argc, char** argv) {
    std::string net_json_path;
    std::string cpp_path;
    std::string weights_bin_path;
    std::string output_path = "output.bin";
    std::string graph_name = "simple_linear";
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); return 0; }
        else if (arg == "--net-json" && i+1 < argc) net_json_path = argv[++i];
        else if (arg == "--cpp" && i+1 < argc) cpp_path = argv[++i];
        else if (arg == "--weights-bin" && i+1 < argc) weights_bin_path = argv[++i];
        else if (arg == "--output" && i+1 < argc) output_path = argv[++i];
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

    // Step 2: Prepare (optimize) the graph
    if (verbose) std::printf("[2a] Preparing graph...\n");
    HexagonNNEnv env;
    GraphStatus s = gp.prepare(env);
    if (s != GraphStatus::Success) {
        std::fprintf(stderr, "Error: prepare() failed with status %d\n",
                     static_cast<int>(s));
        return 1;
    }
    std::printf("[2] Graph prepared (optimized)\n");

    // Step 3: Schedule - convert optimized graph to HTP execution plan
    if (verbose) std::printf("[3a] Scheduling execution plan...\n");
    Scheduler scheduler;
    Scheduler::Plan plan = scheduler.schedule(gp);
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
