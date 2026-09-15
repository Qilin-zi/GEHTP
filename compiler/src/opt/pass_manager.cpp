#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <algorithm>
#include <cstdio>

namespace hnnx {

// 内置 pass 的注册函数(每个 pass 一个文件)
void register_pass_const_fold();
void register_pass_shape_fold();

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
    uint32_t total = 0;

    // op 快照: rewrite 会改 opdef_map_(replace_opdef_with_opconst 同键替换
    // + 标 OP_DEAD), 遍历期间先收集 id 列表。
    std::vector<op_id_t> ids;
    gp->for_each_op([&](OpDef* od) {
        if (od && od->is_enabled() && !od->is_dead()) ids.push_back(od->op_id);
    });

    for (op_id_t id : ids) {
        OpDef* op = gp->get_op_at(id);
        if (!op || !op->is_enabled() || op->is_dead() || op->is_const()) continue;
        for (auto& pass : passes_) {
            if (pass.phase != phase || !pass.enabled) continue;
            if (!pass.matcher) continue;
            if (!pass.matcher(op, gp)) continue;
            pass.matched++;
            int changed = pass.rewrite ? pass.rewrite(op, gp) : 0;
            if (changed > 0) {
                pass.rewritten++;
                total += static_cast<uint32_t>(changed);
                break;  // 一个 op 每轮最多被一个 pass 重写
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
}

} // namespace hnnx
