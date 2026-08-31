#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace hnnx {

// DP Sequencer: initial_sequencer_dp.cc, dp_group_graph.cc, dp_op_graph.cc
// Uses XGBoost decision trees (mlh::XGBDecisionTreeBranch) for heuristic selection
// Multi-level: SVF0, SVF1, SVF2 (Short Value Flow), LVF (Long Value Flow)

// SVF/LVF levels
enum class SequencerLevel {
    SVF0,
    SVF1,
    SVF2,
    LVF,
};

// DP Sequencer configuration (50+ seq_sf_* parameters)
struct SequencerConfig {
    // Algorithm selection
    std::string algo_selector;
    std::string resequencer;

    // SVF/LVF enable
    bool svf_en = true;
    bool svf_abort_en = false;

    // Parallelism
    uint32_t lvf_parallelism_cfg = 0;
    uint32_t svf0_parallelism_cfg = 0;
    uint32_t svf1_parallelism_cfg = 0;
    uint32_t svf2_parallelism_cfg = 0;
    uint32_t parallelism_pull_limit = 0;

    // TCM pressure mitigation
    bool mitigate_tcm_pressure = false;
    std::string tcm_pressure_dist;

    // MLH (XGBoost) configuration
    float mlh_svf_max_tcm_ratio = 0.0f;
    float mlh_lvf_max_tcm_ratio = 0.0f;
    std::vector<float> mlh_lvf_tcm_reduction_list;
    float mlh_ddr_ratio = 0.0f;
    std::string mlh_svf0_dma_cfg;
    std::string mlh_svf1_dma_cfg;
    std::string mlh_svf2_dma_cfg;
    std::string mlh_lvf_dma_cfg;
    bool mlh_parallelism_en = false;

    // MLH training
    bool mlh_training_mode = false;
    bool mlh_training_feature_dump = false;
    std::string mlh_training_csv = "network_and_device_ml_features.csv";
    bool mlh_update_precomputed_keys = false;

    // MLH debug
    bool debug_mlh_verify = false;
    bool debug_mlh_terminate = false;
    std::string debug_mlh_model;

    // DP optimization
    uint32_t dp_greedy_fallback_threshold = 0;
    bool dp_popular_groups_en = false;
    bool dp_splithist_based_offsets_en = false;
    uint32_t dp_sg_mapping_cost = 0;
    bool dp_sg_reorder = false;
    uint32_t dp_sg_reorder_set_sel = 0;
    uint32_t dp_reorder_cost = 0;
    uint32_t dp_sg_reorder_threshold = 0;
    bool dp_early_exit_en = false;

    // Heuristic selection
    float heuristic_select_confidence_threshold = 0.0f;

    // External sequencer (Python)
    bool external_sequencer = false;
    std::string selected_sequencer;
    std::string sequencer_py_path = "scripts/sequencer.py";

    // Scheduler config
    float sched_threshold_ratio = 1.0f;
    float sched_lower_threshold_ratio = 0.0f;
    uint32_t sched_timeout = 0;
    uint32_t sched_outer_timeout = 0;
    uint32_t sched_full_retries = 0;
    bool sched_afterburner = false;
    bool sched_abort_on_mistake = false;
    bool sched_early_out = false;
    bool sched_hint_depthwise = false;
    bool sched_delay_dma = false;
    uint32_t vtcm_retention = 0;
    uint32_t spill_fill_buffer_sizes = 0;
};

// DP Op Graph: dp_op_graph.cc
struct DPOpNode {
    op_id_t op_id;
    std::vector<DPOpNode*> predecessors;
    std::vector<DPOpNode*> successors;
    uint64_t vtcm_requirement;
    uint64_t ddr_requirement;
    uint32_t nsp_assignment;
    SequencerLevel level;
};

struct DPOpGraph {
    std::vector<DPOpNode> nodes;
    std::vector<std::vector<DPOpNode*>> groups; // subgroups

    // Feature extraction for MLH
    std::vector<float> get_op_stats(const std::vector<DPOpNode*>& nodes) const;
};

// XGBoost decision tree for MLH
// Source: mlh::XGBDecisionTreeBranch, mlh::DecisionTreeLeaf/Node/Branch
struct XGBDecisionTreeNode {
    int feature_index = -1;
    float threshold = 0.0f;
    int left_child = -1;
    int right_child = -1;
    float leaf_value = 0.0f;
    bool is_leaf = false;
};

class XGBDecisionTree {
public:
    XGBDecisionTree();
    ~XGBDecisionTree();

    float predict(const std::vector<float>& features) const;
    void load(const std::vector<XGBDecisionTreeNode>& nodes);

private:
    std::vector<XGBDecisionTreeNode> nodes_;
};

// MLH model: ml_heuristic.cc
class MLHModel {
public:
    MLHModel();
    ~MLHModel();

    // Select SVF level and DMA config based on op features
    struct MLHDecision {
        SequencerLevel level;
        std::string dma_config;
        uint32_t parallelism;
    };

    MLHDecision select(const std::vector<float>& features) const;

    // Greedy fallback when MLH model unavailable
    // Source: seq_sf_dp_greedy_fallback_threshold
    MLHDecision greedy_fallback(const DPOpGraph& graph, uint32_t threshold) const;

    // Training mode: dump features to CSV
    void dump_features(const std::vector<float>& features, const std::string& csv_path) const;

private:
    std::vector<XGBDecisionTree> trees_; // ensemble of trees
    bool model_loaded_ = false;
};

// DP Sequencer: initial_sequencer_dp.cc
class DPSequencer {
public:
    DPSequencer();
    ~DPSequencer();

    // Run the DP sequencer on a graph
    std::vector<op_id_t> sequence(
        const DPOpGraph& graph,
        const SequencerConfig& config,
        const MLHModel& mlh_model);

    // External Python sequencer
    std::vector<op_id_t> external_sequence(
        const DPOpGraph& graph,
        const std::string& script_path);

    // Reorder mode predictor
    // Source: dp_reorder_mode_selector_MODE_*_predictor
    enum class ReorderMode {
        MODE_1_1_2_1_0,
        MODE_1_1_1_1_1,
        MODE_2_4_4_4_0,
    };

    ReorderMode predict_reorder_mode(const DPOpGraph& graph) const;

private:
    SequencerConfig config_;
    MLHModel mlh_model_;

    // SVF/LVF multi-level sequencing
    std::vector<op_id_t> run_svf(
        const DPOpGraph& graph, SequencerLevel level,
        uint32_t parallelism, const std::string& dma_cfg);
    std::vector<op_id_t> run_lvf(
        const DPOpGraph& graph, uint32_t parallelism, const std::string& dma_cfg);
};

} // namespace hnnx
