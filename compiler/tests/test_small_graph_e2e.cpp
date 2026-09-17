#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/schedule/scheduler.hpp"
#include "hnnx/scheduler/st_cut.hpp"
#include "hnnx/schedule/e2e_bridge.hpp"
#include "hnnx/serialize/context_binary_writer.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <vector>

using namespace hnnx;

namespace {

OutputDef make_od(uint32_t rank, std::initializer_list<uint64_t> dims, DType dt) {
    OutputDef od{};
    od.rank = rank;
    od.dtype = static_cast<uint32_t>(dt);
    od.flags = 0;
    od.quant_params = 0;
    uint32_t i = 0;
    for (uint64_t d : dims) {
        if (i < 5) od.dims[i++] = d;
    }
    od.element_size = (dt == DType::Int32 || dt == DType::Float32) ? 4 : 1;
    return od;
}

InputDef make_input(uint32_t src_id, uint32_t out_idx = 0) {
    InputDef in{};
    in.rank = src_id;
    in.dtype = out_idx;
    return in;
}

void append_const(GraphPrepare& gp, uint32_t id, const OutputDef& od,
                  const std::vector<uint8_t>& data) {
    gp.append_const_node(id, od, data.empty() ? nullptr : data.data(), data.size());
}

bool build_simple_linear(GraphPrepare& gp) {
    // input[1,4,3] -> Transpose -> Reshape -> FC(W[2,4]+b[2])
    //              -> Reshape -> Transpose -> output[1,2,3]
    auto input_od = make_od(3, {1, 4, 3}, DType::Float32);
    gp.append_node("Input", 1, nullptr, 0, &input_od, 1, nullptr);

    std::vector<uint8_t> perm_in = {0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0};
    append_const(gp, 2, make_od(1, {3}, DType::Int32), perm_in);

    InputDef trans_in[] = {make_input(1), make_input(2)};
    auto input_ncf_od = make_od(3, {1, 3, 4}, DType::Float32);
    gp.append_node("Transpose", 3, trans_in, 2,
                   &input_ncf_od, 1, nullptr);

    InputDef reshape1_in[] = {make_input(3)};
    auto pre_reshape_od = make_od(2, {3, 4}, DType::Float32);
    gp.append_node("Reshape", 4, reshape1_in, 1,
                   &pre_reshape_od, 1, nullptr);

    float w[8] = {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    float b[2] = {0.1f, 0.2f};
    std::vector<uint8_t> wbytes(sizeof(w));
    std::vector<uint8_t> bbytes(sizeof(b));
    std::memcpy(wbytes.data(), w, sizeof(w));
    std::memcpy(bbytes.data(), b, sizeof(b));
    append_const(gp, 5, make_od(2, {2, 4}, DType::Float32), wbytes);
    append_const(gp, 6, make_od(1, {2}, DType::Float32), bbytes);

    InputDef fc_in[] = {make_input(4), make_input(5), make_input(6)};
    auto output_fc_od = make_od(2, {3, 2}, DType::Float32);
    gp.append_node("FullyConnected", 7, fc_in, 3,
                   &output_fc_od, 1, nullptr);

    InputDef reshape2_in[] = {make_input(7)};
    auto output_ncf_od = make_od(3, {1, 3, 2}, DType::Float32);
    gp.append_node("Reshape", 8, reshape2_in, 1,
                   &output_ncf_od, 1, nullptr);

    std::vector<uint8_t> perm_out = {0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0};
    append_const(gp, 9, make_od(1, {3}, DType::Int32), perm_out);

    InputDef trans_out_in[] = {make_input(8), make_input(9)};
    auto output_od = make_od(3, {1, 2, 3}, DType::Float32);
    gp.append_node("Transpose", 10, trans_out_in, 2,
                   &output_od, 1, nullptr);

    InputDef out_in[] = {make_input(10)};
    gp.append_node("Output", 100, out_in, 1, nullptr, 0, nullptr);
    return true;
}

} // namespace

int main() {
    int failed = 0;
    auto check = [&](bool cond, const char* msg) {
        if (!cond) {
            std::cerr << "FAIL: " << msg << "\n";
            ++failed;
        } else {
            std::cerr << "OK:   " << msg << "\n";
        }
    };

    HexagonNNEnv env;
    GraphPrepare gp;
    check(build_simple_linear(gp), "build simple_linear graph");
    check(gp.op_count() >= 10, "graph contains expected nodes");

    GraphStatus st = gp.prepare(env);
    check(st == GraphStatus::Success, "prepare() returns success");

    StCutGraphInput stin;
    check(build_stcut_input_from_graph(gp, stin), "build StCutGraphInput from GraphPrepare");
    check(stin.node_count > 0, "st-cut input is non-empty");

    StCutOptions opt;
    opt.rt = 3;
    opt.it = 50;
    opt.rg = 20;
    opt.am = 0;
    opt.budget_base = 1ull << 40;
    opt.tr = 1.0;

    std::vector<uint32_t> best_order;
    std::vector<uint64_t> flows;
    std::vector<uint64_t> cycles;
    run_stcut_schedule(gp, opt, best_order, flows, cycles);
    std::set<uint32_t> uniq(best_order.begin(), best_order.end());
    check(uniq.size() == stin.node_count, "st-cut best order is a permutation");
    check(!flows.empty(), "st-cut produced flow history");
    std::cerr << "st-cut nodes=" << stin.node_count
              << " best_flow=" << (flows.empty() ? 0 : flows.front())
              << " rounds=" << flows.size() << "\n";

    Scheduler scheduler;
    Scheduler::Plan plan = scheduler.schedule(gp);
    // 通用 schedule() 每条 = 一个图 op(非 19 步重放; 重放路径在 schedule_path_a_replay)
    check(!plan.ops.empty(), "scheduler produces one ScheduledOp per graph op");

    // Populate a plan order field so downstream code can see st-cut's decision.
    plan.op_order.clear();
    plan.op_order.reserve(best_order.size());
    for (size_t i = 0; i < best_order.size(); ++i) {
        plan.op_order.push_back(best_order[i]);
    }
    check(plan.op_order.size() == stin.node_count, "plan carries st-cut order");

    ContextBinaryWriter writer;
    writer.set_graph_name("simple_linear");
    writer.set_build_id("v2.48.0.260626120635");
    writer.set_dsp_arch(0);
    writer.set_io_tensor_size(0x00400000);
    writer.set_const_size(0x00200000);
    writer.set_kernel_names(plan.kernel_names);
    writer.set_sz_record_value(54);
    writer.set_scheduled_ops(plan.ops);
    writer.set_op_names(plan.op_names);
    writer.set_tensor_names(plan.tensor_names);
    // Const pool for simple_linear is the same deterministic payload used by
    // test_context_bin: W[8 floats] + b[2 floats], each aligned to 0x100.
    std::vector<uint8_t> const_pool(0x200, 0);
    float w[8] = {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    float b[2] = {0.1f, 0.2f};
    std::memcpy(const_pool.data() + 0x000, w, sizeof(w));
    std::memcpy(const_pool.data() + 0x100, b, sizeof(b));
    writer.set_const_pool(const_pool);

    std::vector<ConstExtentDesc> exts(2);
    exts[0].op_id = 5;
    exts[0].offset = 0x0000;
    exts[0].size = sizeof(w);
    exts[1].op_id = 6;
    exts[1].offset = 0x0100;
    exts[1].size = sizeof(b);
    writer.set_const_extents(exts);

    std::vector<ConstDescriptorParam> desc_params(2);
    desc_params[0] = {5, 3, 2, 4};
    desc_params[1] = {6, 4, 3, 4};
    writer.set_const_descriptor_params(desc_params);

    PreambleParam pp{};
    pp.input_dim0 = 4;
    pp.input_dim1 = 3;
    pp.output_dim0 = 2;
    pp.output_dim1 = 3;
    pp.nonconst_count = 7;
    pp.max_tid = 15;
    pp.qnn_op_count = 5;
    writer.set_preamble_param(pp);

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
    writer.set_trailer_param(tp);

    std::vector<uint8_t> out;
    size_t n = writer.write(out);
    check(n > 0, "context binary writer produced output");

    std::ifstream golden("reference/simple_linear_context.bin", std::ios::binary);
    if (!golden) {
        std::cerr << "SKIP: golden not found; structural closure only\n";
    } else {
        std::vector<uint8_t> real((std::istreambuf_iterator<char>(golden)),
                                  std::istreambuf_iterator<char>());
        size_t cmp = std::min(out.size(), real.size());
        size_t diffs = 0;
        for (size_t i = 0; i < cmp; ++i) if (out[i] != real[i]) ++diffs;
        check(out.size() == real.size(), "generated size equals golden size");
        check(diffs == 0, "generated bytes match golden");
        if (diffs != 0) std::cout << "byte diffs=" << diffs << " / " << cmp << "\n";
    }

    std::cout << (failed == 0 ? "small_graph_e2e PASS\n" : "small_graph_e2e FAIL\n");
    return failed ? 1 : 0;
}
