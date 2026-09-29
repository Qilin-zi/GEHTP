#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <algorithm>
#include <cstdio>
#include <unordered_set>

namespace hnnx {

// 内置 pass 的注册函数(每个 pass 一个文件)
void register_pass_const_fold();
void register_pass_shape_fold();
void register_pass_fusion_rules();

PassManager& PassManager::instance() {
    static PassManager mgr;
    return mgr;
}

bool PassManager::add_pass(GraphPass pass) {
    for (auto& p : passes_) {
        if (p.name == pass.name) return false;  // 幂等: 已注册
    }
    // 消费 CLI 预置开关
    for (const auto& n : disabled_pre_) {
        if (n == pass.name) pass.enabled = false;
    }
    for (const auto& n : enabled_pre_) {
        if (n == pass.name) pass.enabled = true;
    }
    passes_.push_back(std::move(pass));
    return true;
}

bool PassManager::disable(const std::string& name) {
    bool found = false;
    for (auto& p : passes_) {
        if (p.name == name) { p.enabled = false; found = true; }
    }
    if (!found) {
        if (std::find(disabled_pre_.begin(), disabled_pre_.end(), name) == disabled_pre_.end())
            disabled_pre_.push_back(name);
    }
    return true;
}

bool PassManager::enable(const std::string& name) {
    for (auto& p : passes_) {
        if (p.name == name) p.enabled = true;
    }
    if (std::find(enabled_pre_.begin(), enabled_pre_.end(), name) == enabled_pre_.end())
        enabled_pre_.push_back(name);
    return true;
}

bool PassManager::is_enabled(const std::string& name) const {
    for (const auto& p : passes_) {
        if (p.name == name) return p.enabled;
    }
    return false;
}

uint32_t PassManager::run_phase(GraphPrepare* gp, uint32_t phase) {
    if (!gp) return 0;

    // 收集本 phase 的 enabled pass; 无则零开销返回(融合默认全关走这里)。
    std::vector<GraphPass*> active;
    for (auto& p : passes_) {
        if (p.phase == phase && p.enabled && p.matcher) active.push_back(&p);
    }
    if (active.empty()) return 0;

    // worklist 驱动的 greedy 重写(对齐 MLIR GreedyPatternRewriteDriver):
    // 种子 = 全部存活非 const op。命中改写后把 op 的输入 producer 与消费者
    // 压回 worklist, 使 const_fold 前向级联、shape_fold/fusion 改线后下游
    // 重访。与旧"一次快照 + 逐 op 单 pass + 单向 sweep"不同, 这里到 fixpoint,
    // 且不依赖 for_each_op 的 unordered_map 桶序。
    std::vector<op_id_t> worklist;
    std::unordered_set<op_id_t> in_worklist;
    auto push = [&](op_id_t id) {
        if (in_worklist.insert(id).second) worklist.push_back(id);
    };
    gp->for_each_op([&](OpDef* od) {
        if (od && od->is_enabled() && !od->is_dead() && !od->is_const())
            push(od->op_id);
    });

    uint32_t total = 0;
    while (!worklist.empty()) {
        op_id_t id = worklist.back();
        worklist.pop_back();
        in_worklist.erase(id);

        OpDef* op = gp->get_op_at(id);
        if (!op || !op->is_enabled() || op->is_dead() || op->is_const()) continue;

        for (auto* pass : active) {
            if (!pass->matcher(op, gp)) continue;
            pass->matched++;
            // 快照邻居 id 后再 rewrite: fold_op_to_const 经 replace_opdef_with_opconst
            // 同键替换会销毁旧 op(unique_ptr 重新赋值), 故 rewrite 之后不可再
            // 解引用 op。这里只拷 op_id 值, 安全。
            std::vector<op_id_t> src_ids;
            std::vector<op_id_t> consumer_ids;
            for (const auto& conn : op->inputs) src_ids.push_back(conn.src_id);
            consumer_ids = op->consumers;

            int changed = pass->rewrite ? pass->rewrite(op, gp) : 0;
            if (changed > 0) {
                pass->rewritten++;
                total += static_cast<uint32_t>(changed);
                for (op_id_t s : src_ids) push(s);
                for (op_id_t c : consumer_ids) push(c);
                break;  // 一次改写后让位; 邻居经 worklist 触发下一轮
            }
        }
    }
    return total;
}

std::string PassManager::stats_line() const {
    char buf[512];
    std::string s;
    for (const auto& p : passes_) {
        std::snprintf(buf, sizeof(buf), "%s=%u/%u ", p.name.c_str(),
                      p.matched, p.rewritten);
        s += buf;
    }
    if (s.empty()) s = "(no passes registered) ";
    std::snprintf(buf, sizeof(buf), "before=%u after=%u",
                  last_before_, last_after_);
    s += buf;
    return s;
}

void PassManager::note_counts(uint32_t before, uint32_t after) {
    last_before_ = before;
    last_after_ = after;
}

void PassManager::reset_stats() {
    for (auto& p : passes_) {
        p.matched = 0;
        p.rewritten = 0;
    }
}

void register_builtin_passes() {
    register_pass_const_fold();
    register_pass_shape_fold();
    register_pass_fusion_rules();
}

} // namespace hnnx
