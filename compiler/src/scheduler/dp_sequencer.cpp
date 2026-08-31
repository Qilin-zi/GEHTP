#include "hnnx/scheduler/dp_sequencer.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>

namespace hnnx {

// DPOpGraph feature extraction
// Source: new_mlh::get_op_stats(DPOpGraph, vector<DPOpNode*>)
std::vector<float> DPOpGraph::get_op_stats(const std::vector<DPOpNode*>& nodes) const {
    std::vector<float> features;

    // Features for MLH model:
    // - Number of ops in group
    // - Total VTCM requirement
    // - Total DDR requirement
    // - Average op cost
    // - Max dependency depth
    // - Parallelism potential
    // - TCM pressure ratio

    float total_vtcm = 0;
    float total_ddr = 0;
    float max_depth = 0;

    for (auto* node : nodes) {
        total_vtcm += static_cast<float>(node->vtcm_requirement);
        total_ddr += static_cast<float>(node->ddr_requirement);
        max_depth = std::max(max_depth, static_cast<float>(node->predecessors.size()));
    }

    features.push_back(static_cast<float>(nodes.size()));
    features.push_back(total_vtcm);
    features.push_back(total_ddr);
    features.push_back(max_depth);
    features.push_back(total_vtcm / (total_ddr + 1.0f));  // TCM/DDR ratio

    return features;
}

// XGBDecisionTree / MLHModel method implementations live in mlh_model.cpp
// (they were previously duplicated here; the duplicates broke linking because
//  both translation units are in htp_core).

// DPSequencer
DPSequencer::DPSequencer() = default;
DPSequencer::~DPSequencer() = default;

// Run the DP sequencer on a graph
// Source: initial_sequencer_dp.cc, "Starting DP Sequencer"
std::vector<op_id_t> DPSequencer::sequence(
    const DPOpGraph& graph,
    const SequencerConfig& config,
    const MLHModel& mlh_model) {

    config_ = config;

    // 1. Extract features for each op group
    // 2. Use MLH model to select SVF level and DMA config
    // 3. Run multi-level sequencing (SVF0/1/2/LVF)

    std::vector<op_id_t> result;

    // If no groups defined, treat all nodes as one group
    if (graph.groups.empty() && !graph.nodes.empty()) {
        std::vector<DPOpNode*> all_nodes;
        for (const auto& n : graph.nodes) all_nodes.push_back(const_cast<DPOpNode*>(&n));

        auto features = graph.get_op_stats(all_nodes);
        MLHModel::MLHDecision decision = mlh_model.select(features);

        if (decision.level == SequencerLevel::LVF) {
            auto seq = run_lvf(graph, decision.parallelism, decision.dma_config);
            result.insert(result.end(), seq.begin(), seq.end());
        } else {
            auto seq = run_svf(graph, decision.level, decision.parallelism, decision.dma_config);
            result.insert(result.end(), seq.begin(), seq.end());
        }
        return result;
    }

    for (const auto& group : graph.groups) {
        auto features = graph.get_op_stats(group);

        MLHModel::MLHDecision decision;
        if (config.mlh_training_mode) {
            mlh_model.dump_features(features, config.mlh_training_csv);
        }

        if (config.external_sequencer) {
            // Use external Python sequencer
            // Source: scripts/sequencer.py --graph=
            auto seq = external_sequence(graph, config.sequencer_py_path);
            result.insert(result.end(), seq.begin(), seq.end());
            break;
        }

        decision = mlh_model.select(features);

        if (decision.level == SequencerLevel::LVF) {
            auto seq = run_lvf(graph, decision.parallelism, decision.dma_config);
            result.insert(result.end(), seq.begin(), seq.end());
        } else {
            auto seq = run_svf(graph, decision.level, decision.parallelism, decision.dma_config);
            result.insert(result.end(), seq.begin(), seq.end());
        }
    }

    return result;
}

// External Python sequencer
// Source: external_sequencer, scripts/sequencer.py
std::vector<op_id_t> DPSequencer::external_sequence(
    const DPOpGraph& graph,
    const std::string& script_path) {
    // Call: python scripts/sequencer.py --graph=<graph_data>
    // Parse output for op ordering
    return {};
}

// Reorder mode predictor
// Source: dp_reorder_mode_selector_MODE_*_predictor
DPSequencer::ReorderMode DPSequencer::predict_reorder_mode(const DPOpGraph& graph) const {
    // Multiple predictor modes:
    // MODE_1_1_2_1_0
    // MODE_1_1_1_1_1
    // MODE_2_4_4_4_0
    // Selection based on graph characteristics
    return ReorderMode::MODE_1_1_1_1_1;
}

// SVF (Short Value Flow) sequencing
// Source: initial_sequencer_dp.cc, seq_sf_svf0/svf1/svf2
std::vector<op_id_t> DPSequencer::run_svf(
    const DPOpGraph& graph, SequencerLevel level,
    uint32_t parallelism, const std::string& dma_cfg) {

    // SVF (Short Value Flow) sequencing
    // Source: initial_sequencer_dp.cc, seq_sf_svf0/svf1/svf2
    // SVF0: lowest parallelism, simplest DMA
    // SVF1: medium parallelism, moderate DMA
    // SVF2: highest parallelism, complex DMA
    //
    // Algorithm: topological sort respecting predecessor constraints
    // Then apply DP optimization: sg reorder, popular groups, splithist offsets

    // Topological sort by predecessor relationship
    std::vector<op_id_t> result;
    std::unordered_map<op_id_t, int> in_degree;
    std::unordered_map<op_id_t, std::vector<op_id_t>> successors;

    for (const auto& node : graph.nodes) {
        in_degree[node.op_id] = static_cast<int>(node.predecessors.size());
        for (auto* succ : node.successors) {
            successors[node.op_id].push_back(succ->op_id);
        }
    }

    // Kahn's algorithm
    std::queue<op_id_t> q;
    for (const auto& node : graph.nodes) {
        if (in_degree[node.op_id] == 0) q.push(node.op_id);
    }

    while (!q.empty()) {
        op_id_t id = q.front();
        q.pop();
        result.push_back(id);
        for (op_id_t succ : successors[id]) {
            if (--in_degree[succ] == 0) q.push(succ);
        }
    }

    return result;
}

// LVF (Long Value Flow) sequencing
// Source: initial_sequencer_dp.cc, seq_sf_lvf_*
std::vector<op_id_t> DPSequencer::run_lvf(
    const DPOpGraph& graph, uint32_t parallelism, const std::string& dma_cfg) {
    // LVF: highest parallelism, most complex DMA
    // Used for large graphs with long dependency chains

    std::vector<op_id_t> result;
    for (const auto& node : graph.nodes) {
        result.push_back(node.op_id);
    }
    return result;
}

} // namespace hnnx
