#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace hnnx {

class GraphPrepare;
struct OpDef;

// ============================================================================
// PassManager —— 图优化 pass 的统一使能管理 + 结构匹配驱动框架(我方自建)。
//
// 与真实库的关系: 真实 libHtpPrepare 经 GraphOptInfo/matcher 注册表跑
// 优化规则(116×AUTOSPLIT + 105×TYPICAL_SLICE_REDUCE, 221 条语义是 RE
// 开放项 #7)。本框架不臆造该注册表, 而是自建"结构匹配"模型:
//
//   每个 pass = { matcher(OpDef*, GraphPrepare*) -> bool,
//                 rewrite(OpDef*, GraphPrepare*) -> int }
//
// PassManager 遍历图, 某 op 处的图结构与 matcher 匹配成功才在该 op 上
// 使能 rewrite(不硬编码哪个图跑哪个 pass; 图不匹配 = 图不动)。
//
// 使能策略(用户拍板): 所有 pass 默认全开; CLI --no-pass=<name> 关单 pass。
// 相位语义沿用 OptPhase(PHASE_0=常量折叠, PHASE_1=形状归一化, ...),
// 但不同于真实库的节点数阈值门(828/15k 图不因阈值跳过 pass ——
// 结构匹配天然无副作用), 阈值只保留为 RE 常量。
// ============================================================================

struct GraphPass {
    std::string name;    // pass 名(--no-pass 开关用, 全局唯一)
    uint32_t phase = 0;  // OptPhase 槽位
    std::function<bool(OpDef*, GraphPrepare*)> matcher;
    std::function<int(OpDef*, GraphPrepare*)> rewrite;
    bool enabled = true;  // 统一使能开关(默认全开)

    // 统计(每次 prepare 前 reset_stats 清零)
    uint32_t matched = 0;    // matcher 命中次数
    uint32_t rewritten = 0;  // rewrite 成功(变更>0)次数
};

class PassManager {
public:
    static PassManager& instance();

    // 注册 pass。同名重复注册返回 false(幂等, 不覆盖)。
    bool add_pass(GraphPass pass);

    // 统一使能管理。disable/enable 对未注册名也生效(先记开关,
    // 后注册的 pass 按开关落初始 enabled —— CLI 在 prepare/注册之前解析)。
    bool disable(const std::string& name);
    bool enable(const std::string& name);
    bool is_enabled(const std::string& name) const;

    // 按相位跑一遍: 收集 op 快照 → 逐 op × 该相位启用 pass 的 matcher
    // → 匹配则 rewrite。返回 rewrite 成功总数。
    // 相位级 fixpoint(DCE→order→CSE)由调用方 (run_optimize_passes) 驱动。
    uint32_t run_phase(GraphPrepare* gp, uint32_t phase);

    // 统计汇总(单行, hnnx_compile --opt-stats 输出)
    std::string stats_line() const;
    void reset_stats();
    // 记录最近一次 run_optimize_passes 的前后 op 条目数
    void note_counts(uint32_t before, uint32_t after);

    size_t pass_count() const { return passes_.size(); }

private:
    PassManager() = default;
    std::vector<GraphPass> passes_;
    // CLI 预置开关(名字集合; 未注册名也保留, add_pass 时消费)
    std::vector<std::string> disabled_pre_;
    std::vector<std::string> enabled_pre_;
    uint32_t last_before_ = 0;
    uint32_t last_after_ = 0;
};

// 注册全部内置 pass(每个 pass 一个文件, 各自提供 register_pass_* )。
// 由 run_optimize_passes 调用(幂等)。
void register_builtin_passes();

} // namespace hnnx
