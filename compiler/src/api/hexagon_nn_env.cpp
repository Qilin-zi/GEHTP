#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <cstring>

namespace hnnx {

// HexagonNNEnv: environment for graph compilation
// Source: hexagon_nn_env.cc

HexagonNNEnv::HexagonNNEnv() {
    num_nsps_ = 1;
    soc_type_ = 0;
}

HexagonNNEnv::~HexagonNNEnv() = default;

} // namespace hnnx
