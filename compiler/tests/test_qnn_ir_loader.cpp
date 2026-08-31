// test_qnn_ir_loader.cpp 鈥?Verify REQNN can parse QNN IR (net.json) and build
// a before-graph that matches the QNN SDK's simple_linear model structure.
//
// QNN SDK model (from simple_linear.cpp / simple_linear_net.json):
//   input[1,4,3] f32
//     -> Transpose(perm=[0,2,1]) -> input_ncf[1,3,4]
//     -> Reshape -> pre_reshape[3,4]
//     -> FullyConnected(W[2,4], b[2]) -> output_fc[3,2]
//     -> Reshape -> output_fc_ncf[1,3,2]
//     -> Transpose(perm=[0,2,1]) -> output[1,2,3]
//
// Tensor IDs (assigned by QNN in creation order):
//   1=input  2=perm_in  3=input_ncf  4=pre_reshape
//   5=W      6=b        7=output_fc  8=output_fc_ncf
//   9=perm_out  10=output

#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <cstring>
#include <cmath>

static const char* NET_JSON =
    "C:\\Users\\RQILIN\\Documents\\Default Project\\simple_linear_net.json";
static const char* CPP_PATH =
    "C:\\Users\\RQILIN\\Documents\\Default Project\\mlib_build\\jni\\simple_linear.cpp";
static const char* BIN_PATH =
    "C:\\Users\\RQILIN\\Documents\\Default Project\\simple_linear.bin";

static void check(bool cond, const std::string& msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); }
    std::cout << "  OK: " << msg << "\n";
}

int main() {
    using namespace hnnx;
    std::cout << "=== QNN IR Loader: before-graph verification ===\n\n";

    GraphPrepare gp;
    QnnIRLoader loader(gp);
    uint32_t ops = loader.load_net_json(NET_JSON);
    std::cout << "[1] Loaded net.json: " << ops << " op nodes created\n";
    check(ops == 5, "5 op nodes (Transpose+Reshape+FC+Reshape+Transpose)");

    // Total nodes: 5 ops + 4 const + Input + Output = 11
    size_t total = gp.op_count();
    std::cout << "[2] Total nodes: " << total << "\n";
    check(total == 11, "11 total nodes (5 op + 4 const + Input + Output)");

    // --- Verify each node by tensor ID ---
    std::cout << "\n[3] Verifying node structure:\n";

    // id=1: Input node, output [1,4,3] f32
    {
        OpDef* op = gp.get_op_at(1);
        check(op != nullptr, "id=1 Input exists");
        std::string name = op->name_tag ? op->name_tag->name() : "";
        check(name == "Input", "id=1 name=Input");
        check(op->output_def.rank == 3, "id=1 rank=3");
        check(op->output_def.dims[0] == 1, "id=1 dim[0]=1");
        check(op->output_def.dims[1] == 4, "id=1 dim[1]=4");
        check(op->output_def.dims[2] == 3, "id=1 dim[2]=3");
        check(op->output_def.dtype == 0, "id=1 dtype=Float32");
        check(op->inputs.size() == 0, "id=1 no inputs");
    }

    // id=2: perm const, [3] uint32, data=[0,2,1]
    {
        OpDef* op = gp.get_op_at(2);
        check(op != nullptr, "id=2 perm const exists");
        check(op->is_const(), "id=2 is const");
        check(op->output_def.rank == 1, "id=2 rank=1");
        check(op->output_def.dims[0] == 3, "id=2 dim[0]=3");
        check(op->const_data_size == 12, "id=2 data_size=12 (3*uint32)");
        check(op->const_data_offset != 0, "id=2 has const data");
    }

    // id=3: Transpose, input=[1], output [1,3,4] f32
    {
        OpDef* op = gp.get_op_at(3);
        check(op != nullptr, "id=3 Transpose exists");
        std::string name = op->name_tag ? op->name_tag->name() : "";
        check(name == "Transpose", "id=3 name=Transpose");
        check(op->inputs.size() == 1, "id=3 1 input");
        check(op->inputs[0].src_id == 1, "id=3 input src=1 (input)");
        check(op->output_def.rank == 3, "id=3 rank=3");
        check(op->output_def.dims[0] == 1, "id=3 dim[0]=1");
        check(op->output_def.dims[1] == 3, "id=3 dim[1]=3");
        check(op->output_def.dims[2] == 4, "id=3 dim[2]=4");
        check(op->output_def.dtype == 0, "id=3 dtype=Float32");
    }

    // id=4: Reshape, input=[3], output [3,4] f32
    {
        OpDef* op = gp.get_op_at(4);
        check(op != nullptr, "id=4 Reshape exists");
        std::string name = op->name_tag ? op->name_tag->name() : "";
        check(name == "Reshape", "id=4 name=Reshape");
        check(op->inputs.size() == 1, "id=4 1 input");
        check(op->inputs[0].src_id == 3, "id=4 input src=3 (input_ncf)");
        check(op->output_def.rank == 2, "id=4 rank=2");
        check(op->output_def.dims[0] == 3, "id=4 dim[0]=3");
        check(op->output_def.dims[1] == 4, "id=4 dim[1]=4");
    }

    // id=5: W const, [2,4] f32
    {
        OpDef* op = gp.get_op_at(5);
        check(op != nullptr, "id=5 W const exists");
        check(op->is_const(), "id=5 is const");
        check(op->output_def.rank == 2, "id=5 rank=2");
        check(op->output_def.dims[0] == 2, "id=5 dim[0]=2");
        check(op->output_def.dims[1] == 4, "id=5 dim[1]=4");
        check(op->output_def.dtype == 0, "id=5 dtype=Float32");
        check(op->const_data_size == 32, "id=5 data_size=32 (2*4*float32)");
    }

    // id=6: b const, [2] f32
    {
        OpDef* op = gp.get_op_at(6);
        check(op != nullptr, "id=6 b const exists");
        check(op->is_const(), "id=6 is const");
        check(op->output_def.rank == 1, "id=6 rank=1");
        check(op->output_def.dims[0] == 2, "id=6 dim[0]=2");
        check(op->const_data_size == 8, "id=6 data_size=8 (2*float32)");
    }

    // id=7: FullyConnected, inputs=[4,5,6], output [3,2] f32
    {
        OpDef* op = gp.get_op_at(7);
        check(op != nullptr, "id=7 FullyConnected exists");
        std::string name = op->name_tag ? op->name_tag->name() : "";
        check(name == "FullyConnected", "id=7 name=FullyConnected");
        check(op->inputs.size() == 3, "id=7 3 inputs (pre_reshape, W, b)");
        check(op->inputs[0].src_id == 4, "id=7 input[0]=4 (pre_reshape)");
        check(op->inputs[1].src_id == 5, "id=7 input[1]=5 (W)");
        check(op->inputs[2].src_id == 6, "id=7 input[2]=6 (b)");
        check(op->output_def.rank == 2, "id=7 rank=2");
        check(op->output_def.dims[0] == 3, "id=7 dim[0]=3");
        check(op->output_def.dims[1] == 2, "id=7 dim[1]=2");
    }

    // id=8: Reshape, input=[7], output [1,3,2] f32
    {
        OpDef* op = gp.get_op_at(8);
        check(op != nullptr, "id=8 Reshape exists");
        std::string name = op->name_tag ? op->name_tag->name() : "";
        check(name == "Reshape", "id=8 name=Reshape");
        check(op->inputs.size() == 1, "id=8 1 input");
        check(op->inputs[0].src_id == 7, "id=8 input src=7 (output_fc)");
        check(op->output_def.rank == 3, "id=8 rank=3");
        check(op->output_def.dims[0] == 1, "id=8 dim[0]=1");
        check(op->output_def.dims[1] == 3, "id=8 dim[1]=3");
        check(op->output_def.dims[2] == 2, "id=8 dim[2]=2");
    }

    // id=9: perm const, [3] uint32, data=[0,2,1]
    {
        OpDef* op = gp.get_op_at(9);
        check(op != nullptr, "id=9 perm const exists");
        check(op->is_const(), "id=9 is const");
        check(op->output_def.dims[0] == 3, "id=9 dim[0]=3");
        check(op->const_data_size == 12, "id=9 data_size=12");
    }

    // id=10: Transpose, input=[8], output [1,2,3] f32
    {
        OpDef* op = gp.get_op_at(10);
        check(op != nullptr, "id=10 Transpose exists");
        std::string name = op->name_tag ? op->name_tag->name() : "";
        check(name == "Transpose", "id=10 name=Transpose");
        check(op->inputs.size() == 1, "id=10 1 input");
        check(op->inputs[0].src_id == 8, "id=10 input src=8 (output_fc_ncf)");
        check(op->output_def.rank == 3, "id=10 rank=3");
        check(op->output_def.dims[0] == 1, "id=10 dim[0]=1");
        check(op->output_def.dims[1] == 2, "id=10 dim[1]=2");
        check(op->output_def.dims[2] == 3, "id=10 dim[2]=3");
    }

    // id=100: Output node, input=[10]
    {
        OpDef* op = gp.get_op_at(100);
        check(op != nullptr, "id=100 Output exists");
        std::string name = op->name_tag ? op->name_tag->name() : "";
        check(name == "Output", "id=100 name=Output");
        check(op->inputs.size() == 1, "id=100 1 input");
        check(op->inputs[0].src_id == 10, "id=100 input src=10 (output)");
    }

    // --- Verify graph topology matches QNN SDK ---
    std::cout << "\n[4] Verifying topology matches QNN SDK:\n";
    std::cout << "  input(1) -> Transpose(3) -> Reshape(4) -> FC(7) -> Reshape(8) -> Transpose(10) -> Output(100)\n";
    std::cout << "  W(5), b(6) -> FC(7)\n";
    std::cout << "  perm(2) -> Transpose(3), perm(9) -> Transpose(10)\n";
    check(gp.get_op_at(3)->inputs[0].src_id == 1, "Transpose(3) <- input(1)");
    check(gp.get_op_at(4)->inputs[0].src_id == 3, "Reshape(4) <- input_ncf(3)");
    check(gp.get_op_at(7)->inputs[0].src_id == 4, "FC(7) <- pre_reshape(4)");
    check(gp.get_op_at(7)->inputs[1].src_id == 5, "FC(7) <- W(5)");
    check(gp.get_op_at(7)->inputs[2].src_id == 6, "FC(7) <- b(6)");
    check(gp.get_op_at(8)->inputs[0].src_id == 7, "Reshape(8) <- output_fc(7)");
    check(gp.get_op_at(10)->inputs[0].src_id == 8, "Transpose(10) <- output_fc_ncf(8)");
    check(gp.get_op_at(100)->inputs[0].src_id == 10, "Output(100) <- output(10)");

    // --- Verify consumer lists ---
    std::cout << "\n[5] Verifying consumer lists:\n";
    check(gp.get_op_at(1)->consumers.size() == 1, "input(1) has 1 consumer");
    check(gp.get_op_at(1)->consumers[0] == 3, "input(1) -> Transpose(3)");
    check(gp.get_op_at(4)->consumers.size() == 1, "pre_reshape(4) has 1 consumer");
    check(gp.get_op_at(4)->consumers[0] == 7, "pre_reshape(4) -> FC(7)");
    check(gp.get_op_at(5)->consumers.size() == 1, "W(5) has 1 consumer");
    check(gp.get_op_at(5)->consumers[0] == 7, "W(5) -> FC(7)");
    check(gp.get_op_at(6)->consumers.size() == 1, "b(6) has 1 consumer");
    check(gp.get_op_at(6)->consumers[0] == 7, "b(6) -> FC(7)");

    // --- Summary ---
    std::cout << "\n[6] Before-graph summary:\n";
    std::cout << "  QNN SDK nodes:  Transpose, Reshape, FullyConnected, Reshape, Transpose\n";
    std::cout << "  REQNN  nodes:\n";
    // Print all nodes in ID order
    std::vector<std::pair<op_id_t, std::string>> summary;
    for (uint32_t id : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100}) {
        OpDef* op = gp.get_op_at(id);
        if (!op) continue;
        std::string name = op->name_tag ? op->name_tag->name() : "?";
        std::string ins = "(";
        for (size_t i = 0; i < op->inputs.size(); ++i) {
            if (i > 0) ins += ",";
            ins += std::to_string(op->inputs[i].src_id);
        }
        ins += ")";
        std::string shape = "[";
        for (uint32_t i = 0; i < op->output_def.rank; ++i) {
            if (i > 0) shape += ",";
            shape += std::to_string(op->output_def.dims[i]);
        }
        shape += "]";
        std::cout << "    id=" << id << " " << name << ins << " -> " << shape << "\n";
    }

    std::cout << "\n=== QNN IR Loader: before-graph PASSED ===\n";

    // ========================================================================
    // Part B: Load from .cpp + .bin (authoritative QNN IR)
    // ========================================================================
    std::cout << "\n=== QNN IR (.cpp + .bin): before-graph + real weights ===\n\n";

    GraphPrepare gp2;
    QnnIRLoader loader2(gp2);
    uint32_t ops2 = loader2.load_qnn_ir(CPP_PATH, BIN_PATH);
    std::cout << "[B1] Loaded .cpp + .bin: " << ops2 << " op nodes created\n";
    check(ops2 == 5, ".cpp path: 5 op nodes");

    // Verify tensor IDs match QNN SDK exactly (composeGraphs call order)
    std::cout << "\n[B1b] Verifying .cpp tensor IDs match QNN SDK composeGraphs order:\n";
    {
        // QNN SDK order: input(1), input_ncf_perm(2), input_ncf(3), pre_reshape(4),
        //   W(5), b(6), output_fc(7), output_fc_ncf(8), perm_out(9), output(10)
        auto& t = loader2.tensors();
        check(t.at("input").id == 1, ".cpp: input id=1");
        check(t.at("input_ncf_perm").id == 2, ".cpp: input_ncf_perm id=2");
        check(t.at("input_ncf").id == 3, ".cpp: input_ncf id=3");
        check(t.at("MatMul_0_pre_reshape").id == 4, ".cpp: pre_reshape id=4");
        check(t.at("W").id == 5, ".cpp: W id=5");
        check(t.at("b").id == 6, ".cpp: b id=6");
        check(t.at("output_fc").id == 7, ".cpp: output_fc id=7");
        check(t.at("output_fc_ncf").id == 8, ".cpp: output_fc_ncf id=8");
        check(t.at("MatMul_0_post_reshape_transpose_perm").id == 9, ".cpp: perm_out id=9");
        check(t.at("output").id == 10, ".cpp: output id=10");
    }

    // Verify W weight bytes from .bin (W = [1,3,5,7,2,4,6,8] in [2,4] layout)
    {
        auto& w = loader2.weights().at("W");
        std::cout << "[B2] W from .bin: " << w.size() << " bytes\n";
        check(w.size() == 32, "W size = 32 bytes (8 floats)");
        float wf[8];
        std::memcpy(wf, w.data(), 32);
        std::cout << "    W = [" << wf[0] << "," << wf[1] << "," << wf[2] << "," << wf[3]
                  << "," << wf[4] << "," << wf[5] << "," << wf[6] << "," << wf[7] << "]\n";
        check(wf[0] == 1.0f && wf[1] == 3.0f && wf[2] == 5.0f && wf[3] == 7.0f,
              "W row 0 = [1,3,5,7]");
        check(wf[4] == 2.0f && wf[5] == 4.0f && wf[6] == 6.0f && wf[7] == 8.0f,
              "W row 1 = [2,4,6,8]");
    }

    // Verify b weight bytes from .bin (b = [0.1, 0.2])
    {
        auto& b = loader2.weights().at("b");
        std::cout << "[B3] b from .bin: " << b.size() << " bytes\n";
        check(b.size() == 8, "b size = 8 bytes (2 floats)");
        float bf[2];
        std::memcpy(bf, b.data(), 8);
        std::cout << "    b = [" << bf[0] << "," << bf[1] << "]\n";
        // 0.1 and 0.2 in float32 have rounding, check approx
        bool b0_ok = std::abs(bf[0] - 0.1f) < 1e-6f;
        bool b1_ok = std::abs(bf[1] - 0.2f) < 1e-6f;
        check(b0_ok && b1_ok, "b = [0.1, 0.2]");
    }

    // Verify graph structure from .cpp matches net.json path
    std::cout << "\n[B4] Verifying .cpp graph structure matches:\n";
    check(gp2.get_op_at(1) != nullptr, ".cpp: id=1 exists");
    check(gp2.get_op_at(3) != nullptr, ".cpp: id=3 exists");
    check(gp2.get_op_at(7) != nullptr, ".cpp: id=7 exists");
    check(gp2.get_op_at(10) != nullptr, ".cpp: id=10 exists");

    // W (id should have real data in const pool)
    {
        // Find W node by checking all const nodes with 32 bytes
        OpDef* w_op = nullptr;
        gp2.for_each_op([&](OpDef* op) {
            if (op && op->is_const() && op->const_data_size == 32) w_op = op;
        });
        check(w_op != nullptr, ".cpp: W const node (32 bytes) exists");
        if (w_op) {
            check(w_op->const_data_offset != 0, ".cpp: W has const pool offset");
        }
    }

    std::cout << "\n=== QNN IR (.cpp + .bin): PASSED ===\n";
    return 0;
}


