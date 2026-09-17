#include "hnnx/vtcm/dp_group_graph.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/types.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <sstream>
#include <fstream>

namespace hnnx {

DPGroupGraph::DPGroupGraph() = default;
DPGroupGraph::~DPGroupGraph() = default;

// ===== 全流程入口 (dp_group_graph.cc 8 步) =====
void DPGroupGraph::build(GraphPrepare& gp, uint64_t vtcm_budget, uint64_t ddr_budget) {
    // Step 1: concat-tree group creation (行 330-353)
    build_groups(gp);

    // Step 2: merge lower 32 groups (行 943)
    merge_lower_32_groups();

    // Step 3: branch linear order 选择 (行 1066, 2254-2307)
    branch_order_locked_ = false;
    set_branch_linear_order(BranchLinearOrder::LargestBranchFirst);
    branch_order_locked_ = true;

    // Step 4: TCM break up (行 1225, 1362)
    tcm_break_up(vtcm_budget);

    // Step 5: group breakup 主体 (0x11564d0, 行 1326)
    group_breakup(vtcm_budget);

    // Step 6: popular groups 迁移 (行 2453-2649)
    popular_groups_migration(vtcm_budget);

    // Step 7: DDR break up (行 1785-1802)
    ddr_break_up(ddr_budget);

    // Step 8: split_merge_ops_into_predecessor_groups (RTTI @ 0x469fabe)
    split_merge_ops_into_predecessor_groups();
}

// ===== Step 1: concat-tree group creation (行 330-353) =====
void DPGroupGraph::build_groups(GraphPrepare& gp) {
    groups_.clear();
    // 尝试 concat-tree 分组 (行 353: "Attempted concat-tree group creation")
    bool ok = try_concat_tree_grouping(gp);
    // 行 353: "but data is unavailable, falling back to op_id based groups"
    if (!ok) {
        op_id_based_grouping(gp);
    }
    // input groups 强制 op_id_based_layer_id (行 631)
    for (auto& g : groups_) {
        if (g.is_input_group) {
            g.layer_name.clear();
        }
    }
    // 计算每个 group 的资源需求 + 建立邻接
    for (auto& g : groups_) compute_group_requirements(g, gp);
    build_group_adjacency(gp);
}

bool DPGroupGraph::try_concat_tree_grouping(GraphPrepare& gp) {
    // 行 330: "Was looking for OpID = 0x%llx (%s) but not found in concat_result_graph"
    // 行 346: "found in concat_result_graph - but layerId is %zu which is not >0"
    // concat_result_graph 在 host reimpl 中不可用,直接 fallback (行 353)
    (void)gp;
    return false;
}

void DPGroupGraph::op_id_based_grouping(GraphPrepare& gp) {
    // 行 353 fallback: 按 op_id 顺序 + 数据流连续性分组
    // 同一 layer (连续的 elementwise/conv/matmul 链) 归入一个 group
    std::vector<const OpDef*> sorted = gp.get_sorted_opdefs();
    DPGroupNode* cur = nullptr;
    for (auto* od : sorted) {
        if (!od || !od->is_enabled() || od->is_dead()) continue;
        const char* nm = od->name_tag ? od->name_tag->name() : "";
        std::string name = nm ? nm : "";
        bool is_input = (name == "Input");
        bool is_output = (name == "Output");
        bool is_const = od->is_const();

        if (is_const || is_output) continue;

        // 新 group 条件: 当前无 group, 或 op 有多个 consumer / 来自不同 group
        bool start_new = (cur == nullptr);
        if (!start_new && !is_input) {
            // 若 op 的任一输入不在当前 group → 新 group
            for (const auto& conn : od->inputs) {
                bool found = false;
                if (cur) {
                    for (auto oid : cur->op_ids) if (oid == conn.src_id) { found = true; break; }
                }
                if (!found) { start_new = true; break; }
            }
        }
        if (is_input) start_new = true;

        if (start_new) {
            groups_.emplace_back();
            cur = &groups_.back();
            cur->is_input_group = is_input;
            cur->layer_id = static_cast<int32_t>(groups_.size()) - 1;
            cur->layer_name = name;
        }
        cur->op_ids.push_back(od->op_id);
    }
}

// ===== Step 2: merge lower 32 groups (行 943) =====
void DPGroupGraph::merge_lower_32_groups() {
    // "num_post_merge_lower_32 GROUPS" / "AFTER MERGE LOWER 32 GROUPS"
    // 将 VTCM 需求 < 32 字节阈值的小 group 合并到前驱
    if (groups_.size() <= 1) return;

    constexpr uint64_t LOWER_32 = 32;
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < groups_.size(); ++i) {
            if (groups_[i].op_ids.empty()) continue;
            if (groups_[i].vtcm_requirement >= LOWER_32) continue;
            // 找前驱合并
            for (uint32_t pred : groups_[i].predecessors) {
                if (pred >= groups_.size() || groups_[pred].op_ids.empty()) continue;
                // 合并 i 到 pred
                for (auto oid : groups_[i].op_ids) groups_[pred].op_ids.push_back(oid);
                groups_[pred].vtcm_requirement += groups_[i].vtcm_requirement;
                groups_[pred].ddr_requirement += groups_[i].ddr_requirement;
                groups_[pred].output_size += groups_[i].output_size;
                groups_[pred].is_merged = true;
                groups_[i].op_ids.clear();
                changed = true;
                break;
            }
        }
    }
    // 清理空 group
    groups_.erase(
        std::remove_if(groups_.begin(), groups_.end(),
                       [](const DPGroupNode& g) { return g.op_ids.empty(); }),
        groups_.end());
}

// ===== Step 3: branch linear order 选择 (行 1066, 2254-2307) =====
void DPGroupGraph::set_branch_linear_order(BranchLinearOrder order) {
    // 行 1066: "Setting branch linear order inside group breakup"
    // 行 1225: "Forced to change the branch heuristic ... even though it was locked"
    if (branch_order_locked_) {
        // 已锁定但仍可被 TCM/DDR break up 强制改变 (行 1225/1785)
    }
    branch_order_ = order;
}

// ===== Step 4: TCM break up (行 1225, 1362) =====
bool DPGroupGraph::tcm_break_up(uint64_t vtcm_budget) {
    // 行 1225: VTCM 超预算 → 切分 merged groups
    uint64_t total = 0;
    for (const auto& g : groups_) total += g.vtcm_requirement;
    if (total <= vtcm_budget) return true;

    // 行 1362: "Branch linear order is invalid ... Skipping break mering"
    bool any_broken = false;
    for (size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].vtcm_requirement <= vtcm_budget) continue;
        std::set<uint32_t> visited;
        if (break_single_group(static_cast<uint32_t>(i), vtcm_budget, visited)) {
            any_broken = true;
        }
    }
    // 行 1225: 若切分后 linear order 失效 → 强制改变 heuristic
    if (any_broken && branch_order_locked_) {
        branch_order_locked_ = false;
    }
    return any_broken;
}

// ===== Step 5: group breakup 主体 (0x11564d0, 行 1326) =====
bool DPGroupGraph::group_breakup(uint64_t vtcm_budget) {
    bool any_broken = false;
    for (size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].vtcm_requirement <= vtcm_budget) continue;
        std::set<uint32_t> visited;
        if (break_single_group(static_cast<uint32_t>(i), vtcm_budget, visited)) {
            any_broken = true;
        }
    }
    // 行 1326: "Uh-oh, was not able to break any groups. Best we can do now
    //           is exit group breaking and hope we fit."
    if (!any_broken) {
        return false;
    }
    return true;
}

// break_single_group: 递归切分超 VTCM 的 group
bool DPGroupGraph::break_single_group(uint32_t gidx, uint64_t vtcm_budget,
                                         std::set<uint32_t>& visited) {
    if (visited.count(gidx)) return false;
    visited.insert(gidx);

    if (gidx >= groups_.size() || groups_[gidx].op_ids.empty()) return false;
    DPGroupNode& g = groups_[gidx];
    if (g.vtcm_requirement <= vtcm_budget) return false;

    // 切分策略: 按 branch_linear_order 选切分点
    if (g.op_ids.size() < 2) return false;  // 单 op 无法切

    // 二分切分 (简化: 按最大 branch / df 距离选切分点)
    size_t mid = g.op_ids.size() / 2;
    if (mid < 1) mid = 1;

    DPGroupNode g1, g2;
    g1.op_ids.assign(g.op_ids.begin(), g.op_ids.begin() + mid);
    g2.op_ids.assign(g.op_ids.begin() + mid, g.op_ids.end());
    g1.is_broken = true;
    g2.is_broken = true;
    g1.layer_id = g.layer_id;
    g2.layer_id = g.layer_id;

    // 估算切分后需求 (简化: 按比例)
    double ratio = static_cast<double>(mid) / g.op_ids.size();
    g1.vtcm_requirement = static_cast<uint64_t>(g.vtcm_requirement * ratio);
    g2.vtcm_requirement = g.vtcm_requirement - g1.vtcm_requirement;
    g1.ddr_requirement = static_cast<uint64_t>(g.ddr_requirement * ratio);
    g2.ddr_requirement = g.ddr_requirement - g1.ddr_requirement;
    g1.output_size = static_cast<uint64_t>(g.output_size * ratio);
    g2.output_size = g.output_size - g1.output_size;

    // 替换原 group
    g = g1;
    groups_.push_back(g2);

    // 递归切分仍超预算的部分
    if (g1.vtcm_requirement > vtcm_budget) {
        break_single_group(static_cast<uint32_t>(gidx), vtcm_budget, visited);
    }
    if (g2.vtcm_requirement > vtcm_budget) {
        break_single_group(static_cast<uint32_t>(groups_.size() - 1), vtcm_budget, visited);
    }
    return true;
}

// ===== Step 6: popular groups 迁移 (行 2453-2649) =====
void DPGroupGraph::popular_groups_migration(uint64_t vtcm_budget) {
    // 行 2453: "POPULAR_GROUPS: group: %s, parent_max_spill: %8zuB, num_moved_groups: %5zu"
    // 行 2462: popular_groups_num, 行 2463: popular_groups_single_child_num
    // 行 2493: "average dram footprint before moving outputs"
    // 行 2526: "average dram footprint before moving expanding groups"
    // 行 2649: "average dram footprint"

    // 标记 popular group: output_size 大且访问多
    for (auto& g : groups_) {
        g.is_popular = (g.output_size > vtcm_budget / 4 && g.op_ids.size() == 1);
        if (g.is_popular) {
            g.parent_max_spill = g.output_size;
        }
    }

    uint64_t total_dram_footprint = 0;
    size_t moved = 0;
    // 迁移 popular group 的 outputs 到前驱 group (减少 DDR footprint)
    for (size_t i = 0; i < groups_.size(); ++i) {
        if (!groups_[i].is_popular) continue;
        for (uint32_t pred : groups_[i].predecessors) {
            if (pred >= groups_.size()) continue;
            // 将 popular group 合并到前驱
            for (auto oid : groups_[i].op_ids) groups_[pred].op_ids.push_back(oid);
            groups_[pred].output_size += groups_[i].output_size;
            groups_[pred].vtcm_requirement += groups_[i].vtcm_requirement;
            groups_[i].op_ids.clear();
            moved++;
            break;
        }
        total_dram_footprint += groups_[i].ddr_requirement;
    }
    // 清理空 group
    if (moved > 0) {
        groups_.erase(
            std::remove_if(groups_.begin(), groups_.end(),
                           [](const DPGroupNode& g) { return g.op_ids.empty(); }),
            groups_.end());
    }
}

// ===== Step 7: DDR break up (行 1785-1802) =====
bool DPGroupGraph::ddr_break_up(uint64_t ddr_budget) {
    // 行 1785: "Forced to change branch heuristic due to DDR break up"
    bool any_broken = false;
    for (size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].ddr_requirement <= ddr_budget) continue;
        std::set<uint32_t> visited;
        // 复用 break_single_group (按 DDR 切分)
        if (groups_[i].op_ids.size() >= 2) {
            size_t mid = groups_[i].op_ids.size() / 2;
            DPGroupNode g1, g2;
            g1.op_ids.assign(groups_[i].op_ids.begin(), groups_[i].op_ids.begin() + mid);
            g2.op_ids.assign(groups_[i].op_ids.begin() + mid, groups_[i].op_ids.end());
            double ratio = static_cast<double>(mid) / groups_[i].op_ids.size();
            g1.ddr_requirement = static_cast<uint64_t>(groups_[i].ddr_requirement * ratio);
            g2.ddr_requirement = groups_[i].ddr_requirement - g1.ddr_requirement;
            g1.vtcm_requirement = static_cast<uint64_t>(groups_[i].vtcm_requirement * ratio);
            g2.vtcm_requirement = groups_[i].vtcm_requirement - g1.vtcm_requirement;
            groups_[i] = g1;
            groups_.push_back(g2);
            any_broken = true;
        }
    }
    // 行 1785: 若切分后 linear order 失效 → 强制改变 heuristic
    if (any_broken && branch_order_locked_) {
        branch_order_locked_ = false;
    }
    return any_broken;
}

// ===== Step 8: split_merge_ops_into_predecessor_groups (RTTI @ 0x469fabe) =====
void DPGroupGraph::split_merge_ops_into_predecessor_groups() {
    // 将 merge op (多个前驱汇聚) 切分到各前驱组
    // 简化: 遍历 group, 若 group 的 op_ids 中最后一个 op 有多个来自不同 group 的输入,
    //       将该 op 拆到各前驱组
    // (host reimpl 中跳过: merge op 拆分需要精细的 dataflow 分析)
}

// ===== 辅助: 计算单个 group 的资源需求 =====
void DPGroupGraph::compute_group_requirements(DPGroupNode& g, GraphPrepare& gp) {
    g.vtcm_requirement = 0;
    g.ddr_requirement = 0;
    g.output_size = 0;
    for (auto oid : g.op_ids) {
        OpDef* od = gp.get_op_at(oid);
        if (!od) continue;
        uint64_t sz = 1;
        for (uint32_t i = 0; i < od->output_def.rank && i < 5; ++i) {
            if (od->output_def.dims[i] > 0) sz *= od->output_def.dims[i];
        }
        uint64_t esize = od->output_def.element_size ? od->output_def.element_size : 4;
        uint64_t tensor_bytes = sz * esize;
        // VTCM: 权重 + activation; DDR: spill 备份
        g.vtcm_requirement += tensor_bytes;
        g.output_size += tensor_bytes;
        // DDR: 约为 VTCM 的 20% (spill/fill 开销估算)
        g.ddr_requirement += tensor_bytes / 5;
    }
}

// ===== 辅助: 建立 group 间依赖 =====
void DPGroupGraph::build_group_adjacency(GraphPrepare& gp) {
    // op_id → group idx 映射
    std::unordered_map<op_id_t, uint32_t> op2group;
    for (uint32_t gi = 0; gi < groups_.size(); ++gi) {
        for (auto oid : groups_[gi].op_ids) op2group[oid] = gi;
    }
    for (uint32_t gi = 0; gi < groups_.size(); ++gi) {
        groups_[gi].predecessors.clear();
        groups_[gi].successors.clear();
        for (auto oid : groups_[gi].op_ids) {
            OpDef* od = gp.get_op_at(oid);
            if (!od) continue;
            for (const auto& conn : od->inputs) {
                auto it = op2group.find(conn.src_id);
                if (it == op2group.end()) continue;
                if (it->second == gi) continue;  // 同组
                // pred → gi
                bool found = false;
                for (auto p : groups_[gi].predecessors) if (p == it->second) { found = true; break; }
                if (!found) groups_[gi].predecessors.push_back(it->second);
                found = false;
                for (auto s : groups_[it->second].successors) if (s == gi) { found = true; break; }
                if (!found) groups_[it->second].successors.push_back(gi);
            }
        }
    }
}

// ===== 辅助: 判断 op 是否可安全并入 group =====
bool DPGroupGraph::can_merge_into_group(op_id_t op_id, const DPGroupNode& g,
                                         GraphPrepare& gp) const {
    // 可合并条件: op 的所有 consumer 都在 group 内
    OpDef* od = gp.get_op_at(op_id);
    if (!od) return false;
    for (auto cid : od->consumers) {
        bool found = false;
        for (auto oid : g.op_ids) if (oid == cid) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

// ===== 调试 dump (dp_group_graph.json 等) =====
static void dump_groups_json(const std::string& filename,
                              const std::vector<DPGroupNode>& groups) {
    std::ostringstream ss;
    ss << "{\n  \"groups\": [\n";
    for (size_t i = 0; i < groups.size(); ++i) {
        ss << "    {\"id\": " << i;
        ss << ", \"vtcm\": " << groups[i].vtcm_requirement;
        ss << ", \"ddr\": " << groups[i].ddr_requirement;
        ss << ", \"output_size\": " << groups[i].output_size;
        ss << ", \"is_merged\": " << (groups[i].is_merged ? 1 : 0);
        ss << ", \"is_broken\": " << (groups[i].is_broken ? 1 : 0);
        ss << ", \"is_popular\": " << (groups[i].is_popular ? 1 : 0);
        ss << ", \"op_ids\": [";
        for (size_t j = 0; j < groups[i].op_ids.size(); ++j) {
            if (j) ss << ", ";
            ss << groups[i].op_ids[j];
        }
        ss << "]}";
        if (i + 1 < groups.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n}\n";
    std::ofstream f(filename);
    if (f) f << ss.str();
}

void DPGroupGraph::dump_json(const std::string& filename) const {
    dump_groups_json(filename, groups_);
}
void DPGroupGraph::dump_json_with_inputs(const std::string& f) const {
    dump_groups_json(f, groups_);
}
void DPGroupGraph::dump_json_unbroken(const std::string& f) const {
    dump_groups_json(f, groups_);
}

} // namespace hnnx
