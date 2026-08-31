// test_context_bin: generate a context binary for simple_linear using
// the Scheduler (dynamic compilation) and compare with the real SDK output.
//
// Model: input[1,4,3] -> Transpose -> Reshape -> FullyConnected(W[2,4]+b[2]) -> Reshape -> Transpose -> output[1,2,3]

#include "hnnx/serialize/context_binary_writer.hpp"
#include "hnnx/schedule/scheduler.hpp"
#include "hnnx/ir/types.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

using namespace hnnx;

static const char* REAL_BIN =
    "C:\\Users\\RQILIN\\Documents\\Default Project\\REQNN\\reference\\simple_linear_context.bin";

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

int main() {
    std::printf("=== Context Binary Writer Test (simple_linear, dynamic) ===\n\n");

    auto real = read_file(REAL_BIN);
    if (real.empty()) {
        std::printf("SKIP: real sample not found at %s\n", REAL_BIN);
        return 0;
    }
    std::printf("[0] Real sample: %zu bytes\n", real.size());

    // === Build tensor info for simple_linear ===
    // Tensor IDs: 0=graph_node, 1=input[1,4,3], 2=perm_in, 3=input_ncf[1,3,4],
    // 4=pre_reshape[3,4], 5=W[2,4], 6=b[2], 7=output_fc[3,2],
    // 8=output_ncf[1,3,2], 9=perm_out, 10=output[1,2,3]
    std::vector<TensorInfo> tensors(11);

    auto setup_tensor = [&](uint32_t id, const char* name, uint32_t rank,
                            std::initializer_list<uint64_t> dims, DType dt,
                            bool is_const = false, bool is_io = false) {
        tensors[id].tensor_id = id;
        tensors[id].name = name;
        tensors[id].rank = rank;
        tensors[id].dtype = dt;
        tensors[id].is_const = is_const;
        tensors[id].is_graph_io = is_io;
        uint32_t i = 0;
        for (auto d : dims) {
            if (i < 5) tensors[id].dims[i++] = d;
        }
    };

    setup_tensor(0, "graph_node", 0, {}, DType::Float32);
    setup_tensor(1, "input", 3, {1, 4, 3}, DType::Float32, false, true);
    setup_tensor(2, "perm_in", 1, {3}, DType::Int32, true);
    setup_tensor(3, "input_ncf", 3, {1, 3, 4}, DType::Float32);
    setup_tensor(4, "pre_reshape", 2, {3, 4}, DType::Float32);
    setup_tensor(5, "W", 2, {2, 4}, DType::Float32, true);
    setup_tensor(6, "b", 1, {2}, DType::Float32, true);
    setup_tensor(7, "output_fc", 2, {3, 2}, DType::Float32);
    setup_tensor(8, "output_ncf", 3, {1, 3, 2}, DType::Float32);
    setup_tensor(9, "perm_out", 1, {3}, DType::Int32, true);
    setup_tensor(10, "output", 3, {1, 2, 3}, DType::Float32, false, true);

    // === Run the scheduler to produce the 19-step execution plan ===
    Scheduler scheduler;
    std::vector<std::string> op_names = {
        "MatMul_0_post_reshape_transpose",
        "MatMul_0_post_reshape",
        "MatMul_0",
        "MatMul_0_pre_reshape",
        "input_ncf",
    };
    Scheduler::Plan plan = scheduler.schedule_simple_linear(tensors, op_names);

    std::printf("[1] Scheduler produced %zu steps\n", plan.ops.size());
    for (size_t i = 0; i < plan.ops.size(); i++) {
        const auto& op = plan.ops[i];
        std::printf("  Step %2zu: id=0x%02X cnt=%2u type=0x%02X f2=0x%08X blk=0x%08X extras=%zu  %s\n",
               i, op.record_id, op.tensor_id, static_cast<uint32_t>(op.type),
               op.f2, op.block_ref, op.extras.size(), op.step_name.c_str());
    }

    // === Build the context binary writer from the scheduler's plan ===
    ContextBinaryWriter cbw;
    cbw.set_graph_name("simple_linear");
    cbw.set_build_id("v2.48.0.260626120635");
    cbw.set_dsp_arch(0);
    cbw.set_io_tensor_size(0x00400000);
    cbw.set_const_size(0x00200000);

    cbw.set_kernel_names(plan.kernel_names);
    cbw.set_sz_record_value(54);  // Sz record payload[5] for simple_linear
    cbw.set_scheduled_ops(plan.ops);
    cbw.set_op_names(plan.op_names);
    cbw.set_tensor_names(plan.tensor_names);

    // Const pool: W (32 bytes) + b (8 bytes)
    // W is stored in COLUMN-MAJOR order: [1,3,5,7,2,4,6,8] for W[[1,2],[3,4],[5,6],[7,8]]
    // b = [0.1, 0.2]
    std::vector<uint8_t> const_pool(0x200, 0);
    float W[8] = {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    float b[2] = {0.1f, 0.2f};
    std::memcpy(const_pool.data() + 0x000, W, 32);
    std::memcpy(const_pool.data() + 0x100, b, 8);
    cbw.set_const_pool(const_pool);

    // Const extents
    std::vector<ConstExtentDesc> extents(2);
    extents[0].op_id = 5;
    extents[0].offset = 0x0000;
    extents[0].size = 32;
    extents[0].tensor_type = 0;
    extents[0].reserved = 0;
    extents[1].op_id = 6;
    extents[1].offset = 0x0100;
    extents[1].size = 8;
    extents[1].tensor_type = 0;
    extents[1].reserved = 0;
    cbw.set_const_extents(extents);

    // Const descriptor parameters: (tensor_id, X, Y, elem_size)
    // W (tid=5): X=N=3 (FC input rows), Y=M=2 (FC output count)
    // b (tid=6): X=K=4 (FC input cols), Y=N=3 (FC input rows)
    std::vector<ConstDescriptorParam> desc_params(2);
    desc_params[0] = {5, 3, 2, 4};  // W: X=N=3, Y=M=2, elem=4
    desc_params[1] = {6, 4, 3, 4};  // b: X=K=4, Y=N=3, elem=4
    cbw.set_const_descriptor_params(desc_params);

    // Preamble parameters: input[1,4,3], output[1,2,3], 7 non-const, max_tid=15
    PreambleParam pp{};
    pp.input_dim0 = 4;
    pp.input_dim1 = 3;
    pp.output_dim0 = 2;
    pp.output_dim1 = 3;
    pp.nonconst_count = 7;
    pp.max_tid = 15;
    pp.qnn_op_count = 5;  // 5 QNN ops: Transpose, Reshape, FC, Reshape, Transpose
    cbw.set_preamble_param(pp);

    // Trailer parameters
    TrailerParam tp{};
    tp.output_op_names = {"MatMul_0_post_reshape_transpose", "MatMul_0_post_reshape",
                          "MatMul_0", "MatMul_0_pre_reshape"};
    tp.input_op_names = {"input_ncf"};
    tp.output_tensor_name = "output";
    tp.input_tensor_name = "input";
    tp.output_tid = 10;
    tp.input_tid = 1;
    tp.input_rank = 3;
    tp.output_rank = 3;
    tp.output_dims = {1, 2, 3};
    tp.input_dims = {1, 4, 3};
    cbw.set_trailer_param(tp);

    // === Write ===
    std::vector<uint8_t> ours;
    size_t our_size = cbw.write(ours);
    std::printf("[2] Our output: %zu bytes\n", our_size);

    if (our_size == 0) {
        std::printf("FAILED: writer returned 0\n");
        return 1;
    }

    // === Compare with real sample ===
    std::printf("[3] Comparison:\n");
    std::printf("    Real: %zu bytes\n", real.size());
    std::printf("    Ours: %zu bytes\n", our_size);

    size_t cmp_len = std::min(our_size, real.size());
    int total_diffs = 0;
    int first_diff = -1;
    int last_diff = -1;
    for (size_t i = 0; i < cmp_len; i++) {
        if (ours[i] != real[i]) {
            if (first_diff < 0) first_diff = static_cast<int>(i);
            last_diff = static_cast<int>(i);
            total_diffs++;
        }
    }

    // System header check (skip contextBlob size at 0x20-0x27)
    int sys_diffs = 0;
    for (size_t i = 0; i < 0x1000; i++) {
        if (i >= 0x20 && i < 0x28) continue;
        if (i >= our_size || i >= real.size()) break;
        if (ours[i] != real[i]) sys_diffs++;
    }

    // opData check (0x7534 - 0x7898)
    int opdata_diffs = 0;
    int opdata_first = -1;
    for (size_t i = 0x7534; i < 0x7898 && i < cmp_len; i++) {
        if (ours[i] != real[i]) {
            if (opdata_first < 0) opdata_first = static_cast<int>(i);
            opdata_diffs++;
        }
    }

    // Const segment check (0x9000 - 0x9200)
    int const_diffs = 0;
    for (size_t i = 0x9000; i < 0x9200 && i < cmp_len; i++) {
        if (ours[i] != real[i]) const_diffs++;
    }

    std::printf("    System header: %d diffs (skip 0x20-0x27)\n", sys_diffs);
    std::printf("    opData:        %d diffs\n", opdata_diffs);
    if (opdata_first >= 0) {
        std::printf("    opData first diff at 0x%04X\n", opdata_first);
    }
    std::printf("    Const segment: %d diffs\n", const_diffs);
    std::printf("    Total: %d byte diffs in first %zu bytes\n", total_diffs, cmp_len);
    if (first_diff >= 0) {
        std::printf("    First diff at 0x%04X, last diff at 0x%04X\n", first_diff, last_diff);
    }

    // Show first 20 diffs in detail
    int shown = 0;
    for (size_t i = 0; i < cmp_len && shown < 20; i++) {
        if (ours[i] != real[i]) {
            std::printf("    [0x%04X] real=0x%02X ours=0x%02X\n",
                        static_cast<uint32_t>(i), real[i], ours[i]);
            shown++;
        }
    }

    // Save our .bin
    std::string out_path = "our_simple_linear_context.bin";
    std::ofstream of(out_path, std::ios::binary);
    of.write(reinterpret_cast<const char*>(ours.data()), our_size);
    of.close();
    std::printf("[4] Saved our .bin to %s (%zu bytes)\n", out_path.c_str(), our_size);

    bool pass = (total_diffs == 0);
    std::printf("\n=== %s ===\n", pass ? "PASSED (0 diffs)" : "PARTIAL (has diffs)");
    return pass ? 0 : 1;
}


