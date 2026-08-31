#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace hnnx {

class GraphPrepare;

// HexagonNNEnv: environment for graph compilation
// Source: hexagon_nn_env.cc
class HexagonNNEnv {
public:
    HexagonNNEnv();
    ~HexagonNNEnv();

    uint32_t num_nsps() const { return num_nsps_; }
    uint32_t soc_type() const { return soc_type_; }
    void set_num_nsps(uint32_t n) { num_nsps_ = n; }
    void set_soc_type(uint32_t t) { soc_type_ = t; }

private:
    uint32_t num_nsps_ = 1;
    uint32_t soc_type_ = 0;
};

// OpIoPtrs: input/output pointer structure for op factory
// Source: op_registry_prepare.cc
struct OpIoPtrs {
    GraphPrepare* graph_prepare = nullptr;  // +0x00
    void* reserved_08 = nullptr;
    void* reserved_10 = nullptr;
    void* reserved_18 = nullptr;
    void* reserved_20 = nullptr;
    void* opdef_ptr = nullptr;             // +0x28
    void* reserved_30 = nullptr;
};

// OpRef: reference to an op
struct OpRef {
    op_id_t op_id;
    uint32_t output_index;
};

// NspIdMap: NSP ID mapping for multi-NSP
struct NspIdMap {
    std::unordered_map<int, int> mapping;
};

} // namespace hnnx
