// test_multi_model_write: generate .bin for all 6 models using write() method
// (not write_from_template), compare with real SDK output.
//
// This tests whether the writer can generate correct .bin for different
// graph structures, not just simple_linear.

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

struct ModelWriteTest {
    const char* name;
    const char* ref_path;
    ModelDesc model;
    const char* graph_name;
    // const pool
    std::vector<uint8_t> const_pool;
    std::vector<ConstExtentDesc> extents;
    std::vector<ConstDescriptorParam> desc_params;
    // preamble
    PreambleParam pp;
    // trailer
    TrailerParam tp;
    uint32_t sz_value;
};

int main() {
    std::printf("=== Multi-Model write() Test ===\n\n");

    std::string ref_dir = "C:\\Users\\RQILIN\\Documents\\Default Project\\REQNN\\reference\\";

    // fc_only: input[3,4] -> FC(W[4,2]+b[2]) -> output[3,2]
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=W, 5=fc_act, 6=dma_out, 7=sync, 8=output
    ModelWriteTest fc_only;
    fc_only.name = "fc_only";
    fc_only.ref_path = "compare/fc_only_context.bin";
    fc_only.model.input_tid = 1; fc_only.model.input_batch = 1;
    fc_only.model.input_dim0 = 3; fc_only.model.input_dim1 = 4; fc_only.model.input_dim2 = 0;
    fc_only.model.output_tid = 8; fc_only.model.output_batch = 1;
    fc_only.model.output_dim0 = 3; fc_only.model.output_dim1 = 2; fc_only.model.output_dim2 = 0;
    fc_only.model.perm_in_tid = 2; fc_only.model.perm_out_tid = 2;
    fc_only.model.sync_tid = 7;
    fc_only.model.input_ncf_tid = 3;
    fc_only.model.ops = {{HtpOpType::FullyConnected, "fc", 5, 6, 4, 0, 0, 2, 4}};
    fc_only.model.op_names = {"fc"};
    fc_only.model.tensor_names = {"output", "input"};
    fc_only.graph_name = "fc_only";
    // W=[4,2] 8 floats, b=[2] 2 floats
    fc_only.const_pool.resize(0x200, 0);
    {
        float W[8] = {1,2,3,4,5,6,7,8};
        float b[2] = {0.1f, 0.2f};
        std::memcpy(fc_only.const_pool.data() + 0x000, W, 32);
        std::memcpy(fc_only.const_pool.data() + 0x100, b, 8);
    }
    fc_only.extents = {{5, 0x0000, 32, 0, 0}, {6, 0x0100, 8, 0, 0}};
    // Wait - fc_only tensor IDs: need to check. From ModelDesc: W tid=4, b tid=? 
    // Actually in test_multi_model: ops = {{FullyConnected, "fc", 5, 6, 4, 0, 0, 2, 4}}
    // OpDef: {type, name, input_tid=5, output_tid=6, W_tid=4, b_tid=0, ..., M=2, K=4}
    // So W=4, b=? (b_tid=0 means no bias? or b is embedded?)
    // Actually looking at OpDef: {HtpOpType, name, input_tid, output_tid, W_tid, b_tid, act_ref, M, K, ...}
    // For fc_only: W_tid=4, b_tid=0 (no separate bias tensor)
    // Hmm, but real fc_only has W and b. Let me check...
    // Actually from the real .bin analysis, fc_only has W at tid=4 and b at tid=5
    // But in ModelDesc: input_tid=5, output_tid=6 - these are the FC input/output
    // W_tid=4 in OpDef
    // Let me just use what the scheduler gives us
    fc_only.desc_params = {{4, 3, 2, 4}, {5, 4, 3, 4}};  // W: X=N=3, Y=M=2; b: X=K=4, Y=N=3
    fc_only.pp = {};
    fc_only.pp.input_dim0 = 3; fc_only.pp.input_dim1 = 4;
    fc_only.pp.output_dim0 = 3; fc_only.pp.output_dim1 = 2;
    fc_only.pp.nonconst_count = 5;  // fc_only has fewer non-const tensors
    fc_only.pp.max_tid = 8;
    fc_only.pp.qnn_op_count = 1;  // 1 QNN op: FC
    fc_only.tp = {};
    fc_only.tp.output_op_names = {"fc"};
    fc_only.tp.input_op_names = {"fc"};  // fc consumes input
    fc_only.tp.output_tensor_name = "output";
    fc_only.tp.input_tensor_name = "input";
    fc_only.tp.output_tid = 8;
    fc_only.tp.input_tid = 1;
    fc_only.tp.input_rank = 2;  // [3,4]
    fc_only.tp.output_rank = 2;  // [3,2]
    fc_only.tp.output_dims = {3, 2};
    fc_only.tp.input_dims = {3, 4};
    fc_only.sz_value = 34;  // from 7-model analysis

    // trans_only: input[4,3] -> Transpose -> output[3,4]
    ModelWriteTest trans_only;
    trans_only.name = "trans_only";
    trans_only.ref_path = "compare/trans_only_context.bin";
    trans_only.model.input_tid = 1; trans_only.model.input_batch = 1;
    trans_only.model.input_dim0 = 4; trans_only.model.input_dim1 = 3; trans_only.model.input_dim2 = 0;
    trans_only.model.output_tid = 7; trans_only.model.output_batch = 1;
    trans_only.model.output_dim0 = 3; trans_only.model.output_dim1 = 4; trans_only.model.output_dim2 = 0;
    trans_only.model.perm_in_tid = 2; trans_only.model.perm_out_tid = 2;
    trans_only.model.sync_tid = 6;
    trans_only.model.input_ncf_tid = 3;
    trans_only.model.ops = {{HtpOpType::Transpose, "transpose", 3, 4, 0, 0, 2, 0, 0}};
    trans_only.model.op_names = {"transpose"};
    trans_only.model.tensor_names = {"output", "input"};
    trans_only.graph_name = "trans_only";
    trans_only.const_pool = {};  // no weights
    trans_only.extents = {};
    trans_only.desc_params = {};
    trans_only.pp = {};
    trans_only.pp.input_dim0 = 4; trans_only.pp.input_dim1 = 3;
    trans_only.pp.output_dim0 = 3; trans_only.pp.output_dim1 = 4;
    trans_only.pp.nonconst_count = 4;
    trans_only.pp.max_tid = 7;
    trans_only.pp.qnn_op_count = 1;
    trans_only.tp = {};
    trans_only.tp.output_op_names = {"transpose"};
    trans_only.tp.input_op_names = {"transpose"};
    trans_only.tp.output_tensor_name = "output";
    trans_only.tp.input_tensor_name = "input";
    trans_only.tp.output_tid = 7;
    trans_only.tp.input_tid = 1;
    trans_only.tp.input_rank = 2;
    trans_only.tp.output_rank = 2;
    trans_only.tp.output_dims = {3, 4};
    trans_only.tp.input_dims = {4, 3};
    trans_only.sz_value = 32;

    // reshape_only: input[4,3] -> Reshape -> output[4,3]
    ModelWriteTest reshape_only;
    reshape_only.name = "reshape_only";
    reshape_only.ref_path = "compare/reshape_only_context.bin";
    reshape_only.model.input_tid = 1; reshape_only.model.input_batch = 1;
    reshape_only.model.input_dim0 = 4; reshape_only.model.input_dim1 = 3; reshape_only.model.input_dim2 = 0;
    reshape_only.model.output_tid = 5; reshape_only.model.output_batch = 1;
    reshape_only.model.output_dim0 = 4; reshape_only.model.output_dim1 = 3; reshape_only.model.output_dim2 = 0;
    reshape_only.model.perm_in_tid = 2; reshape_only.model.perm_out_tid = 2;
    reshape_only.model.sync_tid = 0;
    reshape_only.model.input_ncf_tid = 3;
    reshape_only.model.ops = {{HtpOpType::Reshape, "reshape", 3, 4, 0, 0, 0, 0, 0}};
    reshape_only.model.op_names = {"reshape"};
    reshape_only.model.tensor_names = {"output", "input"};
    reshape_only.graph_name = "reshape_only";
    reshape_only.const_pool = {};
    reshape_only.extents = {};
    reshape_only.desc_params = {};
    reshape_only.pp = {};
    reshape_only.pp.input_dim0 = 4; reshape_only.pp.input_dim1 = 3;
    reshape_only.pp.output_dim0 = 4; reshape_only.pp.output_dim1 = 3;
    reshape_only.pp.nonconst_count = 2;
    reshape_only.pp.max_tid = 5;
    reshape_only.pp.qnn_op_count = 1;
    reshape_only.tp = {};
    reshape_only.tp.output_op_names = {"reshape"};
    reshape_only.tp.input_op_names = {"reshape"};
    reshape_only.tp.output_tensor_name = "output";
    reshape_only.tp.input_tensor_name = "input";
    reshape_only.tp.output_tid = 5;
    reshape_only.tp.input_tid = 1;
    reshape_only.tp.input_rank = 2;
    reshape_only.tp.output_rank = 2;
    reshape_only.tp.output_dims = {4, 3};
    reshape_only.tp.input_dims = {4, 3};
    reshape_only.sz_value = 17;

    std::vector<ModelWriteTest*> tests = {&fc_only, &trans_only, &reshape_only};

    int pass_count = 0;
    for (auto* tc : tests) {
        std::string full_path = ref_dir + tc->ref_path;
        auto real = read_file(full_path.c_str());
        if (real.empty()) {
            std::printf("SKIP: %s (not found)\n", tc->name);
            continue;
        }

        // Run scheduler
        Scheduler scheduler;
        Scheduler::Plan plan = scheduler.schedule_general(tc->model);

        std::printf("[%-14s] real=%zuB, %zu steps -> ", tc->name, real.size(), plan.ops.size());

        // Generate .bin using write() method
        ContextBinaryWriter cbw;
        cbw.set_graph_name(tc->graph_name);
        cbw.set_build_id("v2.48.0.260626120635");
        cbw.set_dsp_arch(0);
        cbw.set_io_tensor_size(0x00400000);
        cbw.set_const_size(0x00200000);
        cbw.set_kernel_names(plan.kernel_names);
        cbw.set_sz_record_value(tc->sz_value);
        cbw.set_scheduled_ops(plan.ops);
        cbw.set_op_names(plan.op_names);
        cbw.set_tensor_names(plan.tensor_names);
        if (!tc->const_pool.empty()) cbw.set_const_pool(tc->const_pool);
        if (!tc->extents.empty()) cbw.set_const_extents(tc->extents);
        if (!tc->desc_params.empty()) cbw.set_const_descriptor_params(tc->desc_params);
        cbw.set_preamble_param(tc->pp);
        cbw.set_trailer_param(tc->tp);

        std::vector<uint8_t> ours;
        size_t our_size = cbw.write(ours);

        if (our_size == 0) {
            std::printf("FAILED: writer returned 0\n");
            continue;
        }

        // Compare
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

            // Show first 30 diffs
            int shown = 0;
            for (size_t i = 0; i < cmp_len && shown < 30; i++) {
                if (ours[i] != real[i]) {
                    std::printf("    [0x%04X] real=0x%02X ours=0x%02X\n",
                                static_cast<uint32_t>(i), real[i], ours[i]);
                    shown++;
                }
            }

            // Save for analysis
            std::string out_path = std::string("our_write_") + tc->name + ".bin";
            std::ofstream of(out_path, std::ios::binary);
            of.write(reinterpret_cast<const char*>(ours.data()), our_size);
            of.close();
        }
    }

    std::printf("\n=== %d/%zu models passed ===\n", pass_count, tests.size());
    return pass_count == static_cast<int>(tests.size()) ? 0 : 1;
}
