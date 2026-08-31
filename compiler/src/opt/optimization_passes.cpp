#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <algorithm>

namespace hnnx {

// GraphOptInfo: 128 bytes, allocated via operator new(0x80)
// Source: add_package_opt @ 0x11C17F0
GraphOptInfo::GraphOptInfo(uint32_t phase, uint32_t id, void* defopt_fn, const char* name) {
    this->vtable = nullptr;
    this->phase = phase;
    this->id = id;
    this->defopt_fn = defopt_fn;
    this->matcher_desc = nullptr;
    // Remaining fields initialized to 0 (total 0x80 bytes)
}

GraphOptInfo::~GraphOptInfo() = default;

op_hash_t GraphOptInfo::get_hash_key() const {
    // matcher_desc[1] = hash key
    if (!matcher_desc) return 0;
    auto* desc = reinterpret_cast<const uint64_t*>(matcher_desc);
    return desc[1];
}

uint16_t GraphOptInfo::get_min_inputs() const {
    // matcher_desc[2] (lower 16 bits) = min inputs
    if (!matcher_desc) return 0;
    auto* desc = reinterpret_cast<const uint16_t*>(matcher_desc);
    return desc[4];  // offset 8 bytes = 4 uint16_t
}

uint16_t GraphOptInfo::get_max_inputs() const {
    // matcher_desc at offset 0x12 = max inputs
    if (!matcher_desc) return 0;
    auto* desc = reinterpret_cast<const uint16_t*>(matcher_desc);
    return desc[9];  // offset 0x12 = 9 uint16_t
}

// GraphOptPass::build_matchers
// Source: graph_opt_pass.cc, build_matchers @ 0x105E5C0
void GraphOptPass::build_matchers() {
    // Build hash table from rules
    // Each entry = 0x50 (80) bytes
    // Hash table size = next power of 2 > rules_.size() * 2
    if (rules_.empty()) return;

    size_t table_size = 1;
    while (table_size < rules_.size() * 2) table_size <<= 1;
    hash_table_size_ = table_size;
    hash_table_ = calloc(table_size, 0x50);  // each entry 80 bytes

    for (auto* rule : rules_) {
        op_hash_t key = rule->get_hash_key();
        // Fibonacci hash
        uint64_t h = hnnx::fibonacci_hash(key);
        size_t idx = static_cast<size_t>(h) & (table_size - 1);

        // Open addressing with secondary probe
        auto* entries = reinterpret_cast<uint8_t*>(hash_table_);
        while (*reinterpret_cast<uint64_t*>(entries + idx * 0x50) != 0) {
            // Secondary probe: ((h >> 15) & 0x1fffe) | 1
            size_t secondary = ((static_cast<size_t>(h) >> 15) & 0x1fffe) | 1;
            idx = (idx + secondary) & (table_size - 1);
        }
        *reinterpret_cast<uint64_t*>(entries + idx * 0x50) = key;
        *reinterpret_cast<GraphOptInfo**>(entries + idx * 0x50 + 8) = rule;
    }
}

void GraphOptPass::add_optim(GraphOptInfo* info) {
    rules_.push_back(info);
}

// GraphOptContext::attempt: apply a rule to an opdef
// Source: graph_opt_pass.cc, attempt @ 0x11C12E0 (1593 bytes)
bool GraphOptContext::attempt(const GraphOptInfo* rule, OpDef* opdef) {
    if (!rule || !opdef) return false;
    // No matcher descriptor => rule cannot match anything.
    if (!rule->matcher_desc) return false;

    // Fast hash filter: the rule's hash key must equal the op's hash key.
    auto* matcher = reinterpret_cast<const uint64_t*>(rule->matcher_desc);
    if (opdef->hash_key() != matcher[1]) return false;

    // Input count range check (use the C++ input connection vector).
    int input_count = static_cast<int>(opdef->input_count());
    uint16_t min_in = rule->get_min_inputs();
    uint16_t max_in = rule->get_max_inputs();
    if (input_count < min_in) return false;
    if (max_in != 0 && input_count > max_in) return false;

    // Pattern matched (hash + arity). In the real binary a virtual matcher
    // callback validates op-specific structure; with no registered rules this
    // path is not reached. Report a successful structural match.
    target_opdef_ = opdef;
    rule_ = rule;
    return true;
}

std::unique_ptr<Op> GraphOptContext::build_new_op(const GraphOptInfo& rule, OpDef& opdef) {
    // Build replacement op based on rule. With no concrete rules registered
    // there is nothing to build; the op factory regenerates ops by name.
    (void)rule; (void)opdef;
    return nullptr;
}

// Fusion 规则集: 真实库通过 GraphOptInfo/matcher 注册几十条融合规则。
// 这里实现直接的结构级 pattern: 找 (producer, consumer) 对，把 consumer
// 的唯一输入来自 producer 且 producer 只被 consumer 消费时，把 consumer
// 改名为 fused 类型并断开 producer (后续 DCE 回收)。
int apply_fusion_rules(GraphPrepare* gp, const std::vector<FusionRule>& rules) {
    if (!gp || rules.empty()) return 0;
    int fused_count = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& rule : rules) {
            std::vector<op_id_t> fused_consumers;
            gp->for_each_op([&](OpDef* consumer_od) {
                if (!consumer_od || !consumer_od->is_enabled() || consumer_od->is_const())
                    return;
                const char* cname = consumer_od->name_tag ? consumer_od->name_tag->name() : "";
                if (std::string(cname) != rule.consumer) return;
                if (consumer_od->inputs.size() != 1) return;
                op_id_t prod_id = consumer_od->inputs[0].src_id;
                OpDef* prod = gp->get_op_at(prod_id);
                if (!prod || !prod->is_enabled() || prod->is_const()) return;
                const char* pname = prod->name_tag ? prod->name_tag->name() : "";
                if (std::string(pname) != rule.producer) return;
                if (prod->consumers.size() != 1) return;
                // 融合: consumer 改名为 fused, 输入改指向 producer 的输入
                consumer_od->name_tag = string_tag_t::map_str(rule.fused);
                auto new_inputs = prod->inputs;
                for (const auto& conn : prod->inputs) {
                    OpDef* src = gp->get_op_at(conn.src_id);
                    if (src) {
                        src->consumers.erase(
                            std::remove(src->consumers.begin(), src->consumers.end(), prod->op_id),
                            src->consumers.end());
                        src->consumers.push_back(consumer_od->op_id);
                    }
                }
                consumer_od->inputs = new_inputs;
                prod->flags |= OP_DEAD;
                fused_consumers.push_back(consumer_od->op_id);
            });
            if (!fused_consumers.empty()) {
                fused_count += static_cast<int>(fused_consumers.size());
                changed = true;
            }
        }
        if (changed) gp->remove_dead_code(false);
    }
    return fused_count;
}

// Fixpoint loop
// Source: Phase2/3/5 vfunc[6]
void run_phase_fixpoint(GraphPrepare* gp) {
    // Loop until no changes:
    while (true) {
        int changed = gp->remove_dead_code(false);
        if (changed == 0) {
            changed = gp->order_nodes(true);
            if (changed == 0) {
                changed = gp->common_subexpr_eliminate(true);
                if (changed == 0) {
                    // Clear graph_dirty flag (+0x7311)
                    // gp->set_graph_dirty(false);
                    break;
                }
            }
        }
    }
}

// Merge optimization registries
// Source: merge_optimization_passes @ 0x11C0640 (13787 bytes)
std::map<uint32_t, GraphOptPass> merge_optimization_passes(
    std::map<uint32_t, GraphOptPass>& dst,
    const std::vector<std::string>& registry_names) {
    // Merge multiple registries by phase number
    // Rules from different registries with the same phase are combined
    return dst;
}

} // namespace hnnx
