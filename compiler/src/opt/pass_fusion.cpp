// pass_fusion —— 激活融合规则入注册制(默认关)
// =====================================================================
// 把 kLegacyFusionRules 的 8 条 (producer, consumer) → fused 规则逐条
// 注册为 GraphPass, 复用 optimization_passes.cpp 拆出的 fusion_match /
// fuse_pair。默认 enabled=false(非设备准入, 见 pass_fusion.hpp)。
// 这是唯一"可以注册但此前仍走 apply_fusion_rules 接口"的一类优化。
#include "hnnx/opt/pass_fusion.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"

#include <cctype>
#include <string>
#include <vector>

namespace hnnx {

namespace {

// producer/consumer 类型名 → 唯一 pass 名(小写, 与 const_fold/shape_fold
// 命名风格一致; --no-pass/--enable-pass 用裸名匹配)。
std::string fusion_pass_name(const char* producer, const char* consumer) {
    std::string s = "fusion_";
    s += producer;
    s += "_";
    s += consumer;
    for (auto& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

void register_one_fusion_rule(const FusionRule& rule) {
    GraphPass p;
    p.name = fusion_pass_name(rule.producer, rule.consumer);
    p.phase = PHASE_3;      // 12492: 激活融合槽(与 PHASE_0/PHASE_1 对齐)
    p.enabled = false;      // 默认关 —— 非设备准入
    p.matcher = [rule](OpDef* op, GraphPrepare* gp) -> bool {
        return fusion_match(rule, op, gp);
    };
    p.rewrite = [rule](OpDef* op, GraphPrepare* gp) -> int {
        return fuse_pair(gp, op, rule) ? 1 : 0;
    };
    PassManager::instance().add_pass(std::move(p));
}

} // namespace

void register_pass_fusion_rules() {
    for (const auto& rule : kLegacyFusionRules) {
        register_one_fusion_rule(rule);
    }
}

} // namespace hnnx
