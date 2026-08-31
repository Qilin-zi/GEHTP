// test_general_scheduler: verify schedule_general produces correct step counts
// for different op combinations (reshape_only, trans_only, fc_only, two_fc, simple_linear)

#include "hnnx/schedule/scheduler.hpp"
#include <cstdio>
#include <vector>
#include <string>

using namespace hnnx;

struct ExpectedStep {
    uint32_t counter;
    OpRecordType type;
    uint32_t f2;
};

struct TestCase {
    const char* name;
    ModelDesc model;
    int expected_steps;
    int expected_compute;
    int expected_memory;
    int expected_dma;
    int expected_sync;
};

int main() {
    std::printf("=== General Scheduler Step Count Test ===\n\n");

    // reshape_only: 9 steps (7 compute, 2 memory, 0 DMA, 0 sync)
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=reshape_out, 5=output
    ModelDesc reshape_only;
    reshape_only.input_tid = 1; reshape_only.input_batch = 1;
    reshape_only.input_dim0 = 4; reshape_only.input_dim1 = 3; reshape_only.input_dim2 = 0;
    reshape_only.output_tid = 5; reshape_only.output_batch = 1;
    reshape_only.output_dim0 = 4; reshape_only.output_dim1 = 3; reshape_only.output_dim2 = 0;
    reshape_only.perm_in_tid = 2; reshape_only.perm_out_tid = 2;
    reshape_only.sync_tid = 0;
    reshape_only.ops = {
        {HtpOpType::Reshape, "reshape", 3, 4, 0, 0, 0, 0, 0},
    };
    reshape_only.op_names = {"reshape"};
    reshape_only.tensor_names = {"output", "input"};

    // trans_only: 13 steps (8 compute, 3 memory, 1 DMA, 1 sync)
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=trans_out, 5=dma_out, 6=sync, 7=output
    ModelDesc trans_only;
    trans_only.input_tid = 1; trans_only.input_batch = 1;
    trans_only.input_dim0 = 4; trans_only.input_dim1 = 3; trans_only.input_dim2 = 0;
    trans_only.output_tid = 7; trans_only.output_batch = 1;
    trans_only.output_dim0 = 3; trans_only.output_dim1 = 4; trans_only.output_dim2 = 0;
    trans_only.perm_in_tid = 2; trans_only.perm_out_tid = 2;
    trans_only.sync_tid = 6;
    trans_only.ops = {
        {HtpOpType::Transpose, "transpose", 3, 4, 0, 0, 2, 0, 0},
    };
    trans_only.op_names = {"transpose"};
    trans_only.tensor_names = {"output", "input"};

    // fc_only: 14 steps (8 compute, 3 memory, 2 DMA, 1 sync)
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=W, 5=fc_act, 6=dma_out, 7=sync, 8=output
    ModelDesc fc_only;
    fc_only.input_tid = 1; fc_only.input_batch = 1;
    fc_only.input_dim0 = 3; fc_only.input_dim1 = 4; fc_only.input_dim2 = 0;
    fc_only.output_tid = 8; fc_only.output_batch = 1;
    fc_only.output_dim0 = 3; fc_only.output_dim1 = 2; fc_only.output_dim2 = 0;
    fc_only.perm_in_tid = 2; fc_only.perm_out_tid = 2;
    fc_only.sync_tid = 7;
    fc_only.ops = {
        {HtpOpType::FullyConnected, "fc", 5, 6, 4, 0, 0, 2, 4},
    };
    fc_only.op_names = {"fc"};
    fc_only.tensor_names = {"output", "input"};

    // two_fc: 17 steps (10 compute, 3 memory, 3 DMA, 1 sync)
    // Tensors: 0=graph, 1=input, 2=perm, 3=input_ncf, 4=W1, 5=fc_act, 6=inter_dma, 7=sync, 8=output
    ModelDesc two_fc;
    two_fc.input_tid = 1; two_fc.input_batch = 1;
    two_fc.input_dim0 = 3; two_fc.input_dim1 = 4; two_fc.input_dim2 = 0;
    two_fc.output_tid = 8; two_fc.output_batch = 1;
    two_fc.output_dim0 = 3; two_fc.output_dim1 = 3; two_fc.output_dim2 = 0;
    two_fc.perm_in_tid = 2; two_fc.perm_out_tid = 2;
    two_fc.sync_tid = 7;
    two_fc.ops = {
        {HtpOpType::FullyConnected, "fc1", 5, 6, 4, 0, 0, 2, 4},
        {HtpOpType::FullyConnected, "fc2", 5, 6, 0, 0, 0, 3, 2},
    };
    two_fc.op_names = {"fc1", "fc2"};
    two_fc.tensor_names = {"output", "input"};

    // simple_linear: 19 steps (11 compute, 3 memory, 4 DMA, 1 sync)
    // Tensors: 0=graph, 1=input, 2=perm_in, 3=input_ncf, 4=pre_reshape,
    //          5=W, 6=b, 7=output_fc, 8=output_ncf, 9=perm_out, 10=output
    ModelDesc simple_linear;
    simple_linear.input_tid = 1; simple_linear.input_batch = 1;
    simple_linear.input_dim0 = 4; simple_linear.input_dim1 = 3; simple_linear.input_dim2 = 0;
    simple_linear.output_tid = 10; simple_linear.output_batch = 1;
    simple_linear.output_dim0 = 2; simple_linear.output_dim1 = 3; simple_linear.output_dim2 = 0;
    simple_linear.perm_in_tid = 2; simple_linear.perm_out_tid = 9;
    simple_linear.sync_tid = 9;
    simple_linear.ops = {
        {HtpOpType::Transpose, "input_ncf", 1, 3, 0, 0, 2, 0, 0},
        {HtpOpType::Reshape, "pre_reshape", 3, 4, 0, 0, 0, 0, 0},
        {HtpOpType::FullyConnected, "MatMul_0", 4, 7, 5, 6, 0, 2, 4},
        {HtpOpType::Reshape, "post_reshape", 4, 8, 0, 0, 0, 0, 0},
    };
    simple_linear.op_names = {
        "MatMul_0_post_reshape_transpose", "MatMul_0_post_reshape",
        "MatMul_0", "MatMul_0_pre_reshape", "input_ncf"
    };
    simple_linear.tensor_names = {"output", "input"};

    std::vector<TestCase> tests = {
        {"reshape_only", reshape_only, 9, 7, 2, 0, 0},
        {"trans_only",   trans_only,   13, 8, 3, 1, 1},
        {"fc_only",      fc_only,      14, 8, 3, 2, 1},
        {"two_fc",       two_fc,       17, 10, 3, 3, 1},
        {"simple_linear", simple_linear, 19, 11, 3, 4, 1},
    };

    int pass_count = 0;
    int total_tests = 0;

    for (const auto& tc : tests) {
        total_tests++;
        Scheduler scheduler;
        Scheduler::Plan plan = scheduler.schedule_general(tc.model);

        int compute = 0, memory = 0, dma = 0, sync = 0;
        for (const auto& op : plan.ops) {
            if (op.type == OP_TYPE_COMPUTE) compute++;
            else if (op.type == OP_TYPE_MEMORY) memory++;
            else if (op.type == OP_TYPE_DMA) dma++;
            else if (op.type == OP_TYPE_SYNC) sync++;
        }

        bool step_count_ok = (int)plan.ops.size() == tc.expected_steps;
        bool compute_ok = compute == tc.expected_compute;
        bool memory_ok = memory == tc.expected_memory;
        bool dma_ok = dma == tc.expected_dma;
        bool sync_ok = sync == tc.expected_sync;
        bool all_ok = step_count_ok && compute_ok && memory_ok && dma_ok && sync_ok;

        if (all_ok) pass_count++;

        std::string status_str = step_count_ok ? "" : ("(expected " + std::to_string(tc.expected_steps) + ")");
        std::printf("%s: %s (%d steps, C=%d M=%d D=%d S=%d) %s\n",
               tc.name,
               all_ok ? "PASS" : "FAIL",
               (int)plan.ops.size(),
               compute, memory, dma, sync,
               status_str.c_str());

        // Print detailed records
        for (size_t i = 0; i < plan.ops.size(); i++) {
            const auto& op = plan.ops[i];
            std::printf("  [%2zu] cnt=%2u type=0x%02X f2=0x%08X blk=0x%08X extras=%zu  %s\n",
                   i, op.tensor_id, static_cast<uint32_t>(op.type),
                   op.f2, op.block_ref, op.extras.size(), op.step_name.c_str());
        }
        std::printf("\n");
    }

    std::printf("\n=== %s: %d/%d tests passed ===\n",
           pass_count == total_tests ? "PASSED" : "PARTIAL",
           pass_count, total_tests);
    return pass_count == total_tests ? 0 : 1;
}


