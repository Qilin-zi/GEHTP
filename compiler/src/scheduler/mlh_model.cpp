#include "hnnx/scheduler/dp_sequencer.hpp"
#include <fstream>
#include <algorithm>

namespace hnnx {

// MLH model implementation
// Source: ml_heuristic.cc, mlh::XGBDecisionTreeBranch

// XGBoost decision tree prediction
// The MLH model uses XGBoost ensemble of decision trees to select
// SVF level (SVF0/1/2) or LVF and DMA configuration based on op features.
//
// Feature extraction: new_mlh::get_op_stats(DPOpGraph, vector<DPOpNode*>)
// Features include: op count, VTCM requirement, DDR requirement,
//   dependency depth, parallelism potential, TCM/DDR ratio
//
// Training: seq_sf_mlh_training_mode, seq_sf_mlh_training_feature_dump
// Dumps features to network_and_device_ml_features.csv

// The model weights are embedded in the binary but require extraction.
// When unavailable, use greedy fallback (seq_sf_dp_greedy_fallback_threshold).

XGBDecisionTree::XGBDecisionTree() = default;
XGBDecisionTree::~XGBDecisionTree() = default;
MLHModel::MLHModel() = default;
MLHModel::~MLHModel() = default;

float XGBDecisionTree::predict(const std::vector<float>& features) const {
    if (nodes_.empty() || features.empty()) return 0.0f;

    int current = 0;
    while (current >= 0 && current < static_cast<int>(nodes_.size())) {
        const auto& node = nodes_[current];
        if (node.is_leaf) return node.leaf_value;

        if (node.feature_index >= 0 &&
            static_cast<size_t>(node.feature_index) < features.size() &&
            features[node.feature_index] <= node.threshold) {
            current = node.left_child;
        } else {
            current = node.right_child;
        }
    }
    return 0.0f;
}

void XGBDecisionTree::load(const std::vector<XGBDecisionTreeNode>& nodes) {
    nodes_ = nodes;
}

// MLH model: ensemble of XGBoost trees
// The ensemble prediction is the sum of all tree predictions
// (standard XGBoost regression)
MLHModel::MLHDecision MLHModel::select(const std::vector<float>& features) const {
    MLHDecision decision;

    if (trees_.empty()) {
        // No model loaded: use default SVF0
        decision.level = SequencerLevel::SVF0;
        decision.dma_config = "default";
        decision.parallelism = 1;
        return decision;
    }

    // XGBoost ensemble: sum of all tree outputs
    float sum = 0.0f;
    for (const auto& tree : trees_) {
        sum += tree.predict(features);
    }

    // Map sum to SVF level selection
    // The model outputs a continuous value that maps to level selection
    if (sum < 0.25f) {
        decision.level = SequencerLevel::SVF0;
        decision.parallelism = 1;
    } else if (sum < 0.5f) {
        decision.level = SequencerLevel::SVF1;
        decision.parallelism = 2;
    } else if (sum < 0.75f) {
        decision.level = SequencerLevel::SVF2;
        decision.parallelism = 4;
    } else {
        decision.level = SequencerLevel::LVF;
        decision.parallelism = 8;
    }

    decision.dma_config = "mlh_selected";
    return decision;
}

MLHModel::MLHDecision MLHModel::greedy_fallback(const DPOpGraph& graph, uint32_t threshold) const {
    MLHDecision decision;

    // Greedy heuristic: if graph is small, use SVF0 (low overhead)
    // If graph is large, use LVF (better parallelism)
    if (graph.nodes.size() < threshold) {
        decision.level = SequencerLevel::SVF0;
        decision.parallelism = 1;
    } else {
        decision.level = SequencerLevel::LVF;
        decision.parallelism = 4;
    }
    decision.dma_config = "greedy";
    return decision;
}

void MLHModel::dump_features(const std::vector<float>& features, const std::string& csv_path) const {
    // Source: seq_sf_mlh_training_feature_dump
    // Append features to CSV file for offline ML training
    std::ofstream f(csv_path, std::ios::app);
    if (!f.is_open()) return;
    for (size_t i = 0; i < features.size(); ++i) {
        if (i > 0) f << ",";
        f << features[i];
    }
    f << "\n";
}

} // namespace hnnx
