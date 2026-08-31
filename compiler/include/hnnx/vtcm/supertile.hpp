#pragma once
#include "hnnx/ir/types.hpp"
#include "hnnx/vtcm/dp_group_graph.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace hnnx {

class GraphPrepare;

// SuperTile: 将多个连续 op 合并为一个 VTCM-resident 计算单元
// 反汇编确认: create_supertiles @ 0x1313ac0 [M36 修正; 旧址 0x13138d0 错] (5244B), make_one_supertile @ 0x1314d50 (591B)
// 文件: supertile.cc (字符串 @ 0x55b44e8), 日志行 267-406
//
// 核心思想 (行 922-927):
//   传统: Op1(tile0) → DDR → Op2(tile0) → DDR → ... 每个 tile 过 DDR
//   SuperTile: Op1(tile) → Op2(tile) 中间结果留 VTCM, 只最终结果回 DDR
//   num_dma_roundtrips 从 N 次降到 1 次

struct SuperTile {
    std::vector<op_id_t> op_ids;
    uint64_t vtcm_base = 0;       // VTCM 连续地址
    uint64_t ddr_base = 0;        // DDR 备份地址
    uint32_t num_dma_roundtrips = 1;  // ★ 从 N 次降到 1 次
    uint64_t vtcm_requirement = 0;
    uint64_t ddr_saved = 0;
};

// DP 求解中的 op 分组
struct SuperTileGroup {
    std::vector<op_id_t> op_ids;
    uint64_t vtcm_requirement = 0;
    uint64_t ddr_saved = 0;
    int split_point = -1;          // 若超 VTCM 预算的切分点
};

// split_history: make_one_supertile 递归切分历史
// 反汇编确认: get_split_history @ 0xf7e1f0 (25B) / 0xf88030 (200B)
struct SplitHistory {
    std::vector<int> split_points;
    int next_split_index = 0;
    int next_split(uint64_t total_vtcm, uint64_t vtcm_free) {
        if (next_split_index >= static_cast<int>(split_points.size())) return -1;
        return split_points[next_split_index++];
    }
};

// 代价适配器 (ARCH §11.3)
//   cost = DDR_saved(merge) - VTCM_penalty(merge)
class CostCalculatorAdapter {
public:
    CostCalculatorAdapter(uint64_t ddr_bandwidth, uint64_t vtcm_size);
    // 评估合并 op_ids 的代价 (负值=收益)
    double merge_cost(const std::vector<op_id_t>& op_ids, GraphPrepare& gp,
                      const DPGroupNode& group) const;
private:
    uint64_t ddr_bw_;
    uint64_t vtcm_size_;
};

// VTCM 约束适配器
//   can_merge(op1..opN) = sum(vtcm_need_i) <= vtcm_available
class TCMCalculatorAdapter {
public:
    explicit TCMCalculatorAdapter(uint64_t vtcm_size);
    bool can_merge(const std::vector<op_id_t>& op_ids, GraphPrepare& gp) const;
private:
    uint64_t vtcm_size_;
};

// SuperTile DP 求解器
// 反汇编确认: HeuristicDP (heuristic_dp.cc), dp[i][k] = min(dp[j][k-1] + cost(j+1..i))
class SuperTileSolver {
public:
    SuperTileSolver(GraphPrepare& gp, uint64_t vtcm_size,
                    const DPGroupGraph& group_graph);
    ~SuperTileSolver();

    // 运行 DP: 对每个 group 将 op 序列切分为若干 SuperTile
    std::vector<SuperTileGroup> solve();

    // 应用解: 创建 SuperTile 节点
    void apply(const std::vector<SuperTileGroup>& solution);

private:
    GraphPrepare& gp_;
    uint64_t vtcm_size_;
    const DPGroupGraph& group_graph_;
    CostCalculatorAdapter cost_calc_;
    TCMCalculatorAdapter tcm_calc_;
    std::vector<SuperTile> supertiles_;

    // 验证所有 op 可安全合并 (无外部 consumer)
    bool can_safely_merge(const std::vector<op_id_t>& op_ids) const;
    // 计算合并后 VTCM 需求
    uint64_t compute_vtcm_need(const std::vector<op_id_t>& op_ids) const;
    // 对单个 group 跑 DP
    std::vector<SuperTileGroup> solve_group(const DPGroupNode& group);
};

} // namespace hnnx
