#include "hnnx/vtcm/supertile.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/types.hpp"
#include <algorithm>
#include <limits>
#include <cstring>

namespace hnnx {

// ===== CostCalculatorAdapter (ARCH §11.3) =====
CostCalculatorAdapter::CostCalculatorAdapter(uint64_t ddr_bandwidth, uint64_t vtcm_size)
    : ddr_bw_(ddr_bandwidth), vtcm_size_(vtcm_size) {}

double CostCalculatorAdapter::merge_cost(const std::vector<op_id_t>& op_ids,
                                          GraphPrepare& gp,
                                          const DPGroupNode& group) const {
    // cost = DDR_saved(merge) - VTCM_penalty(merge)
    if (op_ids.size() < 2) return 0.0;

    // DDR_saved = (intermediate_tensor_size / DDR_bandwidth) * access_count
    // 中间结果: op_ids[0..n-2] 的输出不再过 DDR
    uint64_t intermediate_size = 0;
    for (size_t i = 0; i + 1 < op_ids.size(); ++i) {
        OpDef* od = gp.get_op_at(op_ids[i]);
        if (!od) continue;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od->output_def.rank && d < 5; ++d) {
            if (od->output_def.dims[d] > 0) sz *= od->output_def.dims[d];
        }
        uint64_t esize = od->output_def.element_size ? od->output_def.element_size : 4;
        intermediate_size += sz * esize;
    }
    double ddr_saved = ddr_bw_ > 0
        ? static_cast<double>(intermediate_size) / static_cast<double>(ddr_bw_)
        : static_cast<double>(intermediate_size);

    // VTCM_penalty = merged_vtcm_requirement - max(individual_vtcm_requirements)
    uint64_t merged_vtcm = 0;
    uint64_t max_individual = 0;
    for (auto oid : op_ids) {
        OpDef* od = gp.get_op_at(oid);
        if (!od) continue;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od->output_def.rank && d < 5; ++d) {
            if (od->output_def.dims[d] > 0) sz *= od->output_def.dims[d];
        }
        uint64_t esize = od->output_def.element_size ? od->output_def.element_size : 4;
        uint64_t t = sz * esize;
        merged_vtcm += t;
        if (t > max_individual) max_individual = t;
    }
    double vtcm_penalty = static_cast<double>(merged_vtcm - max_individual);

    return vtcm_penalty - ddr_saved;  // 负值=收益(merge 好)
}

// ===== TCMCalculatorAdapter =====
TCMCalculatorAdapter::TCMCalculatorAdapter(uint64_t vtcm_size)
    : vtcm_size_(vtcm_size) {}

bool TCMCalculatorAdapter::can_merge(const std::vector<op_id_t>& op_ids,
                                       GraphPrepare& gp) const {
    uint64_t total = 0;
    for (auto oid : op_ids) {
        OpDef* od = gp.get_op_at(oid);
        if (!od) continue;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od->output_def.rank && d < 5; ++d) {
            if (od->output_def.dims[d] > 0) sz *= od->output_def.dims[d];
        }
        uint64_t esize = od->output_def.element_size ? od->output_def.element_size : 4;
        total += sz * esize;
    }
    return total <= vtcm_size_;
}

// ===== SuperTileSolver =====
SuperTileSolver::SuperTileSolver(GraphPrepare& gp, uint64_t vtcm_size,
                                  const DPGroupGraph& group_graph)
    : gp_(gp), vtcm_size_(vtcm_size), group_graph_(group_graph),
      cost_calc_(48000000000ULL, vtcm_size), tcm_calc_(vtcm_size) {}

SuperTileSolver::~SuperTileSolver() = default;

// ===== DP 求解主入口 =====
// 反汇编确认: create_supertiles @ 0x1313ac0 [M36 修正; 旧址 0x13138d0 错]
//   for (group : group_graph.groups) { HeuristicDP dp_solver(group, vtcm_size); ... }
std::vector<SuperTileGroup> SuperTileSolver::solve() {
    std::vector<SuperTileGroup> solution;
    for (const auto& group : group_graph_.groups()) {
        if (group.op_ids.empty()) continue;
        // supertile.cc 行 328: "Skip supertile ... Layer has few number of ops"
        if (group.op_ids.size() < 2) {
            SuperTileGroup stg;
            stg.op_ids = group.op_ids;
            stg.vtcm_requirement = group.vtcm_requirement;
            solution.push_back(stg);
            continue;
        }
        auto sub = solve_group(group);
        for (auto& s : sub) solution.push_back(std::move(s));
    }
    return solution;
}

// ===== 对单个 group 跑 DP: dp[i][k] = min(dp[j][k-1] + cost(j+1..i)) =====
std::vector<SuperTileGroup> SuperTileSolver::solve_group(const DPGroupNode& group) {
    const auto& ops = group.op_ids;
    size_t n = ops.size();
    if (n == 0) return {};

    // dp[i][k] = 把 ops[0..i] 分成 k 个 supertile 的最优代价
    // dp[i][k] = min over j<i of (dp[j][k-1] + cost(j+1..i))
    constexpr size_t MAX_K = 16;
    size_t kmax = std::min(n, MAX_K);

    // cost[j+1..i]: 合并 ops[j+1..i] 的代价
    auto cost_seg = [&](size_t a, size_t b) -> double {
        std::vector<op_id_t> seg(ops.begin() + a, ops.begin() + b + 1);
        // VTCM 超限 → 不可合并 → 无穷代价
        if (!tcm_calc_.can_merge(seg, gp_)) return std::numeric_limits<double>::infinity();
        return cost_calc_.merge_cost(seg, gp_, group);
    };

    std::vector<std::vector<double>> dp(n, std::vector<double>(kmax + 1, std::numeric_limits<double>::infinity()));
    std::vector<std::vector<int>> back(n, std::vector<int>(kmax + 1, -1));

    // base: dp[i][1] = cost(0..i)
    for (size_t i = 0; i < n; ++i) {
        dp[i][1] = cost_seg(0, i);
        back[i][1] = -1;  // j = -1 表示起点
    }
    // dp[i][k] = min(dp[j][k-1] + cost(j+1..i))
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 2; k <= kmax; ++k) {
            for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                if (dp[j][k - 1] == std::numeric_limits<double>::infinity()) continue;
                double c = cost_seg(j + 1, i);
                if (c == std::numeric_limits<double>::infinity()) continue;
                double total = dp[j][k - 1] + c;
                if (total < dp[i][k]) {
                    dp[i][k] = total;
                    back[i][k] = j;
                }
            }
        }
    }

    // 找最优 k: 最小代价
    size_t best_k = 1;
    double best_cost = dp[n - 1][1];
    for (size_t k = 2; k <= kmax; ++k) {
        if (dp[n - 1][k] < best_cost) {
            best_cost = dp[n - 1][k];
            best_k = k;
        }
    }

    // 回溯切分点
    std::vector<std::pair<size_t, size_t>> segs;  // [start, end]
    int i = static_cast<int>(n) - 1;
    int k = static_cast<int>(best_k);
    while (i >= 0 && k >= 1) {
        int j = back[i][k];
        int start = (j < 0) ? 0 : j + 1;
        segs.push_back({static_cast<size_t>(start), static_cast<size_t>(i)});
        if (j < 0) break;
        i = j;
        k -= 1;
    }
    std::reverse(segs.begin(), segs.end());

    // 构造 SuperTileGroup
    std::vector<SuperTileGroup> result;
    for (auto& [s, e] : segs) {
        SuperTileGroup stg;
        stg.op_ids.assign(ops.begin() + s, ops.begin() + e + 1);
        stg.vtcm_requirement = compute_vtcm_need(stg.op_ids);
        stg.split_point = -1;
        result.push_back(std::move(stg));
    }
    return result;
}

// ===== apply: 创建 SuperTile 节点 (make_one_supertile) =====
// 反汇编确认: make_one_supertile @ 0x1314d50 (591B)
void SuperTileSolver::apply(const std::vector<SuperTileGroup>& solution) {
    supertiles_.clear();
    for (const auto& stg : solution) {
        // 递归创建 (超 VTCM 则按 split_history 切分)
        SplitHistory hist;
        // 简化: 直接创建, 超限则二分
        if (stg.vtcm_requirement <= vtcm_size_ && can_safely_merge(stg.op_ids)) {
            SuperTile st;
            st.op_ids = stg.op_ids;
            st.vtcm_requirement = stg.vtcm_requirement;
            st.ddr_saved = stg.ddr_saved;
            st.num_dma_roundtrips = 1;  // ★ 从 N 次降到 1 次
            supertiles_.push_back(std::move(st));
        } else if (stg.op_ids.size() >= 2) {
            // 超限 → 二分切分 (make_one_supertile 递归)
            size_t mid = stg.op_ids.size() / 2;
            SuperTileGroup g1, g2;
            g1.op_ids.assign(stg.op_ids.begin(), stg.op_ids.begin() + mid);
            g2.op_ids.assign(stg.op_ids.begin() + mid, stg.op_ids.end());
            g1.vtcm_requirement = compute_vtcm_need(g1.op_ids);
            g2.vtcm_requirement = compute_vtcm_need(g2.op_ids);
            apply({g1, g2});
        }
    }
}

// ===== 验证所有 op 可安全合并 (无外部 consumer) =====
// 反汇编确认: supertile.cc 行 352-406 (duplicate offset / non-consecutive / too large)
bool SuperTileSolver::can_safely_merge(const std::vector<op_id_t>& op_ids) const {
    // 所有 op 的 consumer 必须在 op_ids 内 (或为 Output)
    std::set<op_id_t> idset(op_ids.begin(), op_ids.end());
    for (auto oid : op_ids) {
        OpDef* od = gp_.get_op_at(oid);
        if (!od) continue;
        for (auto cid : od->consumers) {
            if (idset.count(cid)) continue;
            // consumer 不在 supertile 内 → 检查是否为 Output
            OpDef* cod = gp_.get_op_at(cid);
            if (cod && cod->name_tag && cod->name_tag->name()) {
                std::string nm = cod->name_tag->name();
                if (nm == "Output") continue;
            }
            return false;  // 有外部 consumer → 不可合并
        }
    }
    return true;
}

// ===== 计算合并后 VTCM 需求 =====
uint64_t SuperTileSolver::compute_vtcm_need(const std::vector<op_id_t>& op_ids) const {
    uint64_t total = 0;
    for (auto oid : op_ids) {
        OpDef* od = gp_.get_op_at(oid);
        if (!od) continue;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od->output_def.rank && d < 5; ++d) {
            if (od->output_def.dims[d] > 0) sz *= od->output_def.dims[d];
        }
        uint64_t esize = od->output_def.element_size ? od->output_def.element_size : 4;
        total += sz * esize;
    }
    return total;
}

} // namespace hnnx
