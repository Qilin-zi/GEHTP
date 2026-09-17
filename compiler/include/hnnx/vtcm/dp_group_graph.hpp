#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <functional>

namespace hnnx {

class GraphPrepare;

// DPGroupGraph 数据结构 (反汇编确认: RTTI 12DPGroupGraph @ 0x469f69b,
//   继承 DPGraph<DPGroupNode>, DPNode<DPGroupNode>)
// 文件: dp_group_graph.cc (字符串 @ 0x469ed11), 代码区间 0x115000-0x1167xxx
//
// DPGroupNode 字段 (反汇编字符串 @ 0x39ad167):
//   "output_size+tcm+ddr+isolated_s_dim+s_dim_offset"
struct DPGroupNode {
    std::vector<op_id_t> op_ids;
    uint64_t vtcm_requirement = 0;   // tcm: 该 group 的 VTCM 需求
    uint64_t ddr_requirement = 0;    // ddr: 该 group 的 DDR 需求
    uint64_t output_size = 0;        // output_size: 输出 tensor 大小
    uint64_t isolated_s_dim = 0;     // isolated_s_dim: 孤立的 S 维切分量
    uint64_t s_dim_offset = 0;       // s_dim_offset: S 维偏移

    std::vector<uint32_t> predecessors;  // 前驱 group idx
    std::vector<uint32_t> successors;    // 后继 group idx

    bool is_merged = false;           // 是否由 merge 产生
    bool is_broken = false;            // 是否被 break up 切分过
    int32_t layer_id = -1;             // concat-tree layerId (行 346)
    std::string layer_name;            // op_id_based_layer_id 名 (行 631)

    bool is_input_group = false;       // input groups (行 631)
    bool is_popular = false;           // popular group (行 2453)
    uint64_t parent_max_spill = 0;     // parent_max_spill (行 2453)
};

// branch_linear_order 启发式 (反汇编确认: 行 2254/2293/2307)
//   "Setting branch linear order inside group breakup" (行 1066)
enum class BranchLinearOrder {
    LargestBranchFirst   = 0,  // 行 2254: largest_branch_first
    DfLevelIsinputName    = 1,  // 行 2293: df_level_isinput_name
    DfIoDistanceSizeName  = 2,  // 行 2307: df_io_distance_size_name
};

// DPGroupGraph: op 被分组为可独立分配 VTCM 的子图
// 反汇编确认: create_supertiles @ 0x1313ac0 [M36 修正] 消费此结构
class DPGroupGraph {
public:
    DPGroupGraph();
    ~DPGroupGraph();

    // ===== 构建主流程 (dp_group_graph.cc 8 步, 按日志行号) =====

    // Step 1: concat-tree group creation (行 330-353)
    //   在 concat_result_graph 查 OpID 的 layerId
    //   若数据可用 → concat-tree 分组; 否则 fallback 到 op_id based (行 353)
    //   input groups 强制 op_id_based_layer_id (行 631)
    void build_groups(GraphPrepare& gp);

    // Step 2: merge lower 32 groups (行 943)
    //   "num_post_merge_lower_32_groups" / "AFTER MERGE LOWER 32 GROUPS"
    void merge_lower_32_groups();

    // Step 3: branch linear order 选择 (行 1066, 2254-2307)
    //   "Setting branch linear order inside group breakup"
    void set_branch_linear_order(BranchLinearOrder order);

    // Step 4: TCM break up (行 1225, 1362)
    //   VTCM 超预算 → 切分 merged groups
    //   可能强制改变 branch heuristic (即使已锁定, 行 1225)
    //   若 linear order 失效 → 跳过 break merging (行 1362)
    bool tcm_break_up(uint64_t vtcm_budget);

    // Step 5: group breakup 主体 (0x11564d0, 19856 字节)
    //   break merged groups until fit
    //   失败兜底: "was not able to break any groups... exit and hope we fit" (行 1326)
    bool group_breakup(uint64_t vtcm_budget);

    // Step 6: popular groups 迁移 (行 2453-2649)
    //   "POPULAR_GROUPS: group: %s, parent_max_spill: %8zuB, num_moved_groups: %5zu"
    //   moving outputs (行 2493) → moving expanding groups (行 2526)
    //   计算 average dram footprint (行 2649)
    void popular_groups_migration(uint64_t vtcm_budget);

    // Step 7: DDR break up (行 1785-1802)
    //   "Warning: Forced to change branch heuristic due to DDR break up" (行 1785)
    bool ddr_break_up(uint64_t ddr_budget);

    // Step 8: split_merge_ops_into_predecessor_groups (RTTI 方法名 @ 0x469fabe)
    //   将 merge op 切分到前驱组
    void split_merge_ops_into_predecessor_groups();

    // ===== 全流程入口 =====
    void build(GraphPrepare& gp, uint64_t vtcm_budget, uint64_t ddr_budget);

    // ===== 调试 dump (反汇编确认: dp_group_graph.json 等) =====
    void dump_json(const std::string& filename) const;        // dp_group_graph.json
    void dump_json_with_inputs(const std::string& f) const;   // dp_group_graph_with_inputs.json
    void dump_json_unbroken(const std::string& f) const;      // dp_group_graph_unbroken.json

    // ===== 访问 =====
    const std::vector<DPGroupNode>& groups() const { return groups_; }
    std::vector<DPGroupNode>& groups() { return groups_; }
    size_t num_groups() const { return groups_.size(); }

private:
    std::vector<DPGroupNode> groups_;
    BranchLinearOrder branch_order_ = BranchLinearOrder::LargestBranchFirst;
    bool branch_order_locked_ = false;

    // Step 1 内部: concat-tree 分组尝试
    bool try_concat_tree_grouping(GraphPrepare& gp);
    // Step 1 内部: op_id based fallback (行 353)
    void op_id_based_grouping(GraphPrepare& gp);

    // 计算单个 group 的 vtcm/ddr/output_size 需求
    void compute_group_requirements(DPGroupNode& g, GraphPrepare& gp);
    // 建立 group 间依赖 (predecessors/successors)
    void build_group_adjacency(GraphPrepare& gp);
    // 切分单个 group (breakup 主体递归)
    bool break_single_group(uint32_t gidx, uint64_t vtcm_budget,
                            std::set<uint32_t>& visited);
    // 判断 op 是否可安全并入 group (无外部 consumer)
    bool can_merge_into_group(op_id_t op_id, const DPGroupNode& g,
                              GraphPrepare& gp) const;
};

} // namespace hnnx
