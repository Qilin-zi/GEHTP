#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <functional>

namespace hnnx {

// Cost model: hextimate_costsource.cc, cost_based_model.cc
// Uses CostSource::get_prediction_from_cost_model(op_name, Op*, grdep::OpDesc*, InferenceMode)
// Contains MLP (CostSource::MLP) and per-op cost tables
// Source: hextimate_costsource.cc, hextimate_simulator.cc

struct InferenceMode {
    enum Type {
        Performance,
        Power,
        Bandwidth,
    } type = Performance;
};

namespace grdep {
    struct OpDesc {
        std::string op_name;
        uint32_t op_type;
        std::vector<uint64_t> input_dims;
        std::vector<uint64_t> output_dims;
        uint32_t nsp_count;
        uint32_t vtcm_budget;
    };
}

namespace costbased {

class CostSource {
public:
    CostSource();
    ~CostSource();

    // Initialize for a specific SoC
    bool init_for_soc(const std::string& soc_type);

    // Get cost prediction for an op
    // Returns cycles estimate
    float get_prediction_from_cost_model(
        const std::string& op_name,
        const Op* op,
        const grdep::OpDesc* desc,
        InferenceMode mode) const;

    // MLP cost model
    struct MLP {
        std::vector<std::vector<float>> weights;
        std::vector<std::vector<float>> biases;
        // Phase 4.4: which layers use ReLU (hidden) vs linear (output)
        // Empty = all ReLU (legacy behavior); size = weights.size() for per-layer
        std::vector<bool> use_linear;  // true = linear (no ReLU), false = ReLU
        std::vector<float> predict(const std::vector<float>& features) const;
    };

    // Get features for an op
    // Phase 4.4: padded to fixed 8-feature width for MLP input
    std::vector<float> extract_features(
        const Op* op,
        const grdep::OpDesc* desc,
        InferenceMode mode) const;

    // Phase 4.4: calibration feature dump
    // Appends one CSV row per op to the file for offline MLP training.
    // Columns: op_type, in_dims..., out_dims..., nsp_count, vtcm_budget, mode, cost
    void dump_calibration_features(
        const std::vector<Op*>& ops,
        const std::string& csv_filename,
        InferenceMode mode = {}) const;

    // Phase 4.4: check if MLP is loaded
    bool has_mlp_model() const { return !mlp_model_.weights.empty(); }

    // Phase 4.4: get MLP model (for testing/inspection)
    const MLP& get_mlp_model() const { return mlp_model_; }

private:
    bool initialized_ = false;
    std::string soc_type_;
    MLP mlp_model_;
    std::unordered_map<std::string, float> cost_table_;

    void load_cost_table();
    void load_mlp_model();
};

// Cost-based scheduler: cost_based_scheduler.cc
class CostBasedScheduler {
public:
    CostBasedScheduler(CostSource& cost_source);
    ~CostBasedScheduler();

    // Schedule ops for optimal performance
    std::vector<Op*> schedule(
        const std::vector<Op*>& ops,
        const Graph* graph,
        InferenceMode mode) const;

private:
    CostSource& cost_source_;
};

// Hextimate simulator: hextimate_simulator.cc
class HextimateSimulator {
public:
    HextimateSimulator();
    ~HextimateSimulator();

    // Simulate execution and estimate cycles
    uint64_t simulate(
        const std::vector<Op*>& runlist,
        const Graph* graph,
        uint32_t num_nsps) const;

    // Chrome trace output
    void enable_chrome_trace(const std::string& filename);
    void disable_chrome_trace();

private:
    bool trace_enabled_ = false;
    std::string trace_filename_;
};

} // namespace costbased
} // namespace hnnx
