// test_multi_model: generate .bin for all 5 models using schedule_general,
// compare with real SDK output byte-by-byte.
//
// Strategy: extract non-opData segments from real .bin, insert our op records.

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

struct ModelTest {
    const char* name;
    const char* ref_path;
    ModelDesc model;
};

int main() {
    std::printf("=== Multi-Model .bin Generation Test ===\n\n");

    std::string ref_dir = "C:\\Users\\RQILIN\\Documents\\Default Project\\REQNN\\reference\\";

    // fc_only: 14 steps
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=W, 5=fc_act, 6=dma_out, 7=sync, 8=output
    ModelDesc fc_only;
    fc_only.input_tid = 1; fc_only.input_batch = 1;
    fc_only.input_dim0 = 3; fc_only.input_dim1 = 4; fc_only.input_dim2 = 0;
    fc_only.output_tid = 8; fc_only.output_batch = 1;
    fc_only.output_dim0 = 3; fc_only.output_dim1 = 2; fc_only.output_dim2 = 0;
    fc_only.perm_in_tid = 2; fc_only.perm_out_tid = 2;
    fc_only.sync_tid = 7;
    fc_only.input_ncf_tid = 3;
    fc_only.ops = {{HtpOpType::FullyConnected, "fc", 5, 6, 4, 0, 0, 2, 4}};
    fc_only.op_names = {"fc"};
    fc_only.tensor_names = {"output", "input"};

    // trans_only: 13 steps
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=trans_out, 5=dma_out, 6=sync, 7=output
    ModelDesc trans_only;
    trans_only.input_tid = 1; trans_only.input_batch = 1;
    trans_only.input_dim0 = 4; trans_only.input_dim1 = 3; trans_only.input_dim2 = 0;
    trans_only.output_tid = 7; trans_only.output_batch = 1;
    trans_only.output_dim0 = 3; trans_only.output_dim1 = 4; trans_only.output_dim2 = 0;
    trans_only.perm_in_tid = 2; trans_only.perm_out_tid = 2;
    trans_only.sync_tid = 6;
    trans_only.input_ncf_tid = 3;
    trans_only.ops = {{HtpOpType::Transpose, "transpose", 3, 4, 0, 0, 2, 0, 0}};
    trans_only.op_names = {"transpose"};
    trans_only.tensor_names = {"output", "input"};

    // reshape_only: 9 steps
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=reshape_out, 5=output
    ModelDesc reshape_only;
    reshape_only.input_tid = 1; reshape_only.input_batch = 1;
    reshape_only.input_dim0 = 4; reshape_only.input_dim1 = 3; reshape_only.input_dim2 = 0;
    reshape_only.output_tid = 5; reshape_only.output_batch = 1;
    reshape_only.output_dim0 = 4; reshape_only.output_dim1 = 3; reshape_only.output_dim2 = 0;
    reshape_only.perm_in_tid = 2; reshape_only.perm_out_tid = 2;
    reshape_only.sync_tid = 0;
    reshape_only.input_ncf_tid = 3;
    reshape_only.ops = {{HtpOpType::Reshape, "reshape", 3, 4, 0, 0, 0, 0, 0}};
    reshape_only.op_names = {"reshape"};
    reshape_only.tensor_names = {"output", "input"};

    // two_fc: 17 steps
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=W1, 5=fc_act,
    //          6=inter_dma, 7=bias_result, 8=W2, 9=out_dma, 10=sync_ref, 11=out_ref, 12=sync, 13=?, 8=output
    // Actually from real data:
    //   DMA inter: cnt=6, extras ref 5(fc_act), 6,7,8
    //   MatMul2 xW: cnt=5, extras ref 6(inter out)
    //   MatMul2 +b: cnt=5, extras ref 7
    //   DMA out: cnt=6, extras ref 9,10,11
    //   SyncOp: cnt=7, extras [5,12,13,...]
    //   OutputSlice: cnt=8
    ModelDesc two_fc;
    two_fc.input_tid = 1; two_fc.input_batch = 1;
    two_fc.input_dim0 = 3; two_fc.input_dim1 = 4; two_fc.input_dim2 = 0;
    two_fc.output_tid = 8; two_fc.output_batch = 1;
    two_fc.output_dim0 = 3; two_fc.output_dim1 = 3; two_fc.output_dim2 = 0;
    two_fc.perm_in_tid = 2; two_fc.perm_out_tid = 2;
    two_fc.sync_tid = 7;
    two_fc.input_ncf_tid = 3;
    // FC1: input=5(fc_act), output=6(inter_dma), W=4, M=2, K=4
    // FC2: input=5(fc_act), output=6(inter_dma), W=0(none), M=3, K=2, act_ref=6
    two_fc.ops = {
        {HtpOpType::FullyConnected, "fc1", 5, 6, 4, 0, 0, 2, 4, 0},
        {HtpOpType::FullyConnected, "fc2", 5, 6, 0, 0, 0, 3, 2, 6},
    };
    two_fc.op_names = {"fc1", "fc2"};
    two_fc.tensor_names = {"output", "input"};

    // simple_linear: 19 steps (same as test_context_bin)
    // Tensors: 0=graph, 1=input, 2=perm_in, 3=input_ncf, 4=pre_reshape,
    //          5=W, 6=b, 7=output_fc, 8=output_ncf, 9=perm_out, 10=output
    ModelDesc simple_linear;
    simple_linear.input_tid = 1; simple_linear.input_batch = 1;
    simple_linear.input_dim0 = 4; simple_linear.input_dim1 = 3; simple_linear.input_dim2 = 0;
    simple_linear.output_tid = 10; simple_linear.output_batch = 1;
    simple_linear.output_dim0 = 2; simple_linear.output_dim1 = 3; simple_linear.output_dim2 = 0;
    simple_linear.perm_in_tid = 2; simple_linear.perm_out_tid = 9;
    simple_linear.sync_tid = 9;
    simple_linear.input_ncf_tid = 3;
    simple_linear.ops = {
        {HtpOpType::Transpose, "input_ncf", 1, 3, 0, 0, 2, 0, 0, 0},
        {HtpOpType::Reshape, "pre_reshape", 3, 4, 0, 0, 0, 0, 0, 0},
        {HtpOpType::FullyConnected, "MatMul_0", 4, 7, 5, 6, 0, 2, 4, 0},
        {HtpOpType::Reshape, "post_reshape", 4, 8, 0, 0, 0, 0, 0, 0},
    };
    simple_linear.op_names = {
        "MatMul_0_post_reshape_transpose", "MatMul_0_post_reshape",
        "MatMul_0", "MatMul_0_pre_reshape", "input_ncf"
    };
    simple_linear.tensor_names = {"output", "input"};

    // linear_8x4: same ops as simple_linear but input[1,8,4], output[1,2,4]
    ModelDesc linear_8x4 = simple_linear;
    linear_8x4.input_dim0 = 8; linear_8x4.input_dim1 = 4;
    linear_8x4.output_dim0 = 2; linear_8x4.output_dim1 = 4;

    // linear_4x8: same ops but input[1,4,8], output[1,2,8]
    ModelDesc linear_4x8 = simple_linear;
    linear_4x8.input_dim0 = 4; linear_4x8.input_dim1 = 8;
    linear_4x8.output_dim0 = 2; linear_4x8.output_dim1 = 8;

    std::vector<ModelTest> tests = {
        {"fc_only",       "compare/fc_only_context.bin",      fc_only},
        {"trans_only",    "compare/trans_only_context.bin",    trans_only},
        {"reshape_only",  "compare/reshape_only_context.bin", reshape_only},
        {"two_fc",        "compare/two_fc_context.bin",        two_fc},
        {"linear_8x4",    "compare/linear_8x4_context.bin",   linear_8x4},
        {"linear_4x8",    "compare/linear_4x8_context.bin",   linear_4x8},
    };

    int pass_count = 0;

    for (const auto& tc : tests) {
        std::string full_path = ref_dir + tc.ref_path;
        auto real = read_file(full_path.c_str());
        if (real.empty()) {
            std::printf("SKIP: %s (not found)\n", tc.name);
            continue;
        }

        // Run scheduler
        Scheduler scheduler;
        Scheduler::Plan plan = scheduler.schedule_general(tc.model);

        std::printf("[%-14s] real=%zuB, %zu steps -> ", tc.name, real.size(), plan.ops.size());

        // Generate .bin using template approach
        ContextBinaryWriter cbw;
        cbw.set_scheduled_ops(plan.ops);
        std::vector<uint8_t> ours;
        size_t our_size = cbw.write_from_template(real, ours);

        if (our_size == 0) {
            std::printf("FAILED: writer returned 0\n");
            continue;
        }

        // Compare with real sample
        size_t cmp_len = std::min(our_size, real.size());
        int total_diffs = 0;
        int first_diff = -1, last_diff = -1;
        for (size_t i = 0; i < cmp_len; i++) {
            if (ours[i] != real[i]) {
                if (first_diff < 0) first_diff = static_cast<int>(i);
                last_diff = static_cast<int>(i);
                total_diffs++;
            }
        }

        std::printf("%zuB, %d diffs", our_size, total_diffs);
        if (total_diffs == 0) {
            std::printf(" -> EXACT MATCH\n");
            pass_count++;
        } else {
            std::printf(" (first=0x%04X last=0x%04X)\n", first_diff, last_diff);

            // Save our .bin for analysis
            std::string out_path = std::string("our_") + tc.name + ".bin";
            std::ofstream of(out_path, std::ios::binary);
            of.write(reinterpret_cast<const char*>(ours.data()), our_size);
            of.close();

            // Show first 20 diffs in detail
            int shown = 0;
            for (size_t i = 0; i < cmp_len && shown < 20; i++) {
                if (ours[i] != real[i]) {
                    std::printf("    [0x%04X] real=0x%02X ours=0x%02X\n",
                                static_cast<uint32_t>(i), real[i], ours[i]);
                    shown++;
                }
            }
        }
    }

    std::printf("\n=== %s: %d/%d models passed ===\n",
           pass_count == (int)tests.size() ? "PASSED" : "PARTIAL",
           pass_count, (int)tests.size());
    return pass_count == (int)tests.size() ? 0 : 1;
}


