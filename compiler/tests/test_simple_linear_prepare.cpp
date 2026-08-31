
// Linux: 测试数据仅存于 Windows 原开发机 → SKIP(计划: 隔离不移植)
#ifndef _WIN32
#include <cstdio>
int main() { std::printf("SKIP: Windows-only test data (Linux)\n"); return 0; }
#else
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/frontend/qnn_ir_loader.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include <iostream>

int main() {
    using namespace hnnx;
    HexagonNNEnv env;
    GraphPrepare gp;
    QnnIRLoader loader(gp);
    const char* CPP = "C:\\Users\\RQILIN\\Documents\\Default Project\\mlib_build\\jni\\simple_linear.cpp";
    const char* BIN = "C:\\Users\\RQILIN\\Documents\\Default Project\\simple_linear.bin";
    uint32_t ops = loader.load_qnn_ir(CPP, BIN);
    std::cout << "Loaded: " << ops << " ops, " << gp.op_count() << " nodes\n";
    GraphStatus st = gp.prepare(env);
    std::cout << "prepare(): " << (int)st << "\n";
    if (st == GraphStatus::Success) {
        auto& allocs = gp.get_vtcm_allocations();
        std::cout << "VTCM allocations (" << allocs.size() << "):\n";
        for (auto& [id, e] : allocs) {
            std::cout << "  op_id=" << id << " offset=" << e.offset
                      << " block_id=" << e.block_id << " spilled=" << e.spilled << "\n";
        }
    }
    return 0;
}

#endif // !_WIN32
