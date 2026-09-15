// pass_shape_fold —— 形状归一化(PHASE_1)
// =====================================================================
// 模式 A: 恒等 Reshape/Flatten(输出 dims ≡ 输入 dims)—— 删除 + 消费者改线
//         (消费者看到的形状完全不变, 对任意消费者安全)。
// 模式 B: Reshape→Reshape 链(1:1)—— 链首并入链尾(链尾目标 dims 不变,
//         链尾按元素数读平铺数组)。
// 模式 C: 恒等 Transpose(全部 tensor_param const 为 0..n-1 序列)—— 同 A。
// 对应真实库 build_graph_deps 的 steal_output/erase_op(把平凡算子的输出
// 挪给下游并删除该算子, M04 实证), 我方按结构匹配实现。
#include "hnnx/opt/pass_shape_fold.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"

#include <algorithm>
#include <cstring>
#include <string>

namespace hnnx {

namespace {

// OutputDef dims 逐轴相等(含 rank)
bool od_equal(const OutputDef& a, const OutputDef& b) {
    if (a.rank != b.rank) return false;
    for (uint32_t i = 0; i < a.rank && i < 5; ++i)
        if (a.dims[i] != b.dims[i]) return false;
    return true;
}

// 全部 tensor_param const 都是恒等序列 0..n-1(int32)。无 tensor_param
// 时不判(perm 信息缺失 → 保守不折)。
bool all_params_identity(const GraphPrepare* gp, const OpDef* op) {
    if (op->tensor_param_ids.empty()) return false;
    for (op_id_t tpid : op->tensor_param_ids) {
        const OpDef* pc = gp->get_op_at(tpid);
        if (!pc || !pc->is_const() || pc->const_data_size == 0) return false;
        size_t n = pc->const_data_size / 4;
        if (n == 0 || n > 8) return false;  // perm 长度合理域外不判
        bool in_pool = pc->const_data_offset + pc->const_data_size <= gp->const_pool().size();
        if (!in_pool) return false;
        const int32_t* vals = reinterpret_cast<const int32_t*>(
            gp->const_pool().data() + pc->const_data_offset);
        for (size_t i = 0; i < n; i++)
            if (vals[i] != (int32_t)i) return false;
    }
    return true;
}

enum Pattern { P_NONE, P_IDENT_RESHAPE, P_RESHAPE_CHAIN, P_IDENT_TRANSPOSE };

Pattern match_pattern(OpDef* op, GraphPrepare* gp) {
    if (!op || !op->is_enabled() || op->is_dead() || op->is_const()) return P_NONE;
    if (!op->name_tag || !op->name_tag->name()) return P_NONE;
    std::string nm = op->name_tag->name();

    if (nm == "Reshape" || nm == "Flatten") {
        if (op->inputs.size() != 1) return P_NONE;
        const OpDef* src = gp->get_op_at(op->inputs[0].src_id);
        if (!src || !src->is_enabled() || src->is_dead()) return P_NONE;
        // B 优先: 链式合并(不要求恒等)
        if ((src->name_tag && src->name_tag->name() &&
             (std::string(src->name_tag->name()) == "Reshape" ||
              std::string(src->name_tag->name()) == "Flatten")) &&
            src->consumers.size() == 1) {
            return P_RESHAPE_CHAIN;
        }
        // A: 恒等
        if (od_equal(src->output_def, op->output_def)) return P_IDENT_RESHAPE;
        return P_NONE;
    }
    if (nm == "Transpose") {
        if (op->inputs.size() != 1) return P_NONE;
        const OpDef* src = gp->get_op_at(op->inputs[0].src_id);
        if (!src || !src->is_enabled() || src->is_dead()) return P_NONE;
        if (!od_equal(src->output_def, op->output_def)) return P_NONE;
        if (all_params_identity(gp, op)) return P_IDENT_TRANSPOSE;
        return P_NONE;
    }
    return P_NONE;
}

// 删除 op, 消费者改线到 producer(apply_fusion_rules 同款手工改线)
void rewire_consumers(GraphPrepare* gp, OpDef* op, OpDef* prod) {
    std::vector<op_id_t> cons = op->consumers;
    prod->consumers.erase(
        std::remove(prod->consumers.begin(), prod->consumers.end(), op->op_id),
        prod->consumers.end());
    for (op_id_t cid : cons) {
        OpDef* c = gp->get_op_at(cid);
        if (!c) continue;
        for (auto& conn : c->inputs) {
            if (conn.src_id == op->op_id) {
                conn.src_id = prod->op_id;
                conn.src_out_def = prod->output_def;
            }
        }
        prod->consumers.push_back(cid);
    }
    op->flags |= OP_DEAD;
}

// 链式合并: op(链尾)继承 src(链首)的输入, src 标 dead
void merge_chain(GraphPrepare* gp, OpDef* op, OpDef* src) {
    auto new_inputs = src->inputs;
    for (const auto& conn : src->inputs) {
        OpDef* s = gp->get_op_at(conn.src_id);
        if (s) {
            s->consumers.erase(
                std::remove(s->consumers.begin(), s->consumers.end(), src->op_id),
                s->consumers.end());
            s->consumers.push_back(op->op_id);
        }
    }
    op->inputs = new_inputs;
    src->flags |= OP_DEAD;
}

bool matcher(OpDef* op, GraphPrepare* gp) {
    return match_pattern(op, gp) != P_NONE;
}

int rewrite(OpDef* op, GraphPrepare* gp) {
    Pattern p = match_pattern(op, gp);
    if (p == P_NONE) return 0;
    const OpDef* src = gp->get_op_at(op->inputs[0].src_id);
    if (!src) return 0;
    switch (p) {
    case P_IDENT_RESHAPE:
    case P_IDENT_TRANSPOSE: {
        // src 是 const 指针, 改线需要非 const(维护 consumers)—— 用
        // get_op_at 的非 const 重载
        OpDef* prod = gp->get_op_at(op->inputs[0].src_id);
        if (!prod) return 0;
        rewire_consumers(gp, op, prod);
        return 1;
    }
    case P_RESHAPE_CHAIN: {
        OpDef* prod = gp->get_op_at(op->inputs[0].src_id);
        if (!prod) return 0;
        merge_chain(gp, op, prod);
        return 1;
    }
    default:
        return 0;
    }
}

} // namespace

void register_pass_shape_fold() {
    GraphPass p;
    p.name = "shape_fold";
    p.phase = PHASE_1;  // 10190: 形状归一化
    p.matcher = matcher;
    p.rewrite = rewrite;
    PassManager::instance().add_pass(std::move(p));
}

} // namespace hnnx
