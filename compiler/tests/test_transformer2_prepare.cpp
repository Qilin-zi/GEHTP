
#include "hnnx/ir/graph_prepare.hpp"
#include "test_paths.hpp"
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include <iostream>

int main() {
    using namespace hnnx;
    HexagonNNEnv env;
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    const char* CPP = TP_T2_CPP;
    const char* BIN = TP_T2_BIN;
    uint32_t ops = loader.load_qnn_ir(CPP, BIN);
    std::cout << "Loaded: " << ops << " ops, " << gp.op_count() << " nodes\n";
    std::cout << "Calling prepare()...\n";
    GraphStatus st = gp.prepare(env);
    std::cout << "prepare() returned: " << (int)st << "\n";
    if (st == GraphStatus::Success) {
        std::cout << "VTCM allocations: " << gp.get_vtcm_allocations().size() << "\n";
        std::cout << "SUCCESS: prepare() completed\n";
    } else {
        std::cout << "FAILED: prepare() returned error\n";
    }
    return (st == GraphStatus::Success) ? 0 : 1;
}
