// pass_fc_bias —— FullyConnected bias 折叠(PHASE_3)
// =====================================================================
// 锚定在 Eltwise_Binary(ADD) 上: 逆链找 [Reshape→] FullyConnected,
// bias const 并入 FC inputs[2], Add 的消费者改线到 FC/Reshape 输出,
// Add 标 dead(由 PHASE_3 后的 fixpoint DCE 回收)。
// 匹配/改写纪律见 pass_fc_bias.hpp。
//
// **注入形态**(do_prepare1 第 0 步 inject_htp_prepare_inputs 先于 pass):
// 真实管线里 FC 恒被追加输入 —— 无 bias 源图 FC 得 [act,W,wscale,quant_marker],
// 源图自带 bias FC 得 [act,W,bias,quant_marker]。故本 pass 的 inputs[2]
// 操作分两类:
//   - inputs.size()==2 (未走注入, 单测/裸 run_optimize_passes): 直接 append;
//   - inputs.size()>=3 (注入后): 替换 inputs[2], 仅当其是 (a) 注入合成
//     const (id ≥ 0x10000, wscale 等) / (b) 已等于本 bias / (c) 全零 const
//     —— 绝不覆盖疑似源图自带非零 bias(那种形状当前流不存在, 保守跳过)。
//
// **可信标记**: pass 在 fc->op_data 写 scalar "has_bias"=1。
// extract_fc_extra / host execute 只认这个 scalar —— 注入的 wscale const
// 恰好也是 N 元素, 纯形状启发会把 wscale 误判成 bias(test_wtop_emit US1
// 实锤), 形状启发已废弃。
#include "hnnx/opt/pass_fc_bias.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_extra.hpp"
#include "hnnx/ir/scalar_params.hpp"

#include <algorithm>
#include <string>

namespace hnnx {

namespace {

const char* op_name(const OpDef* od) {
    return (od && od->name_tag && od->name_tag->name()) ? od->name_tag->name() : "";
}

bool live(const OpDef* od) {
    return od && od->is_enabled() && !od->is_dead() && !od->is_const();
}

// 可信 bias 标记(与 extract_fc_extra / host execute 唯一认定口径)
bool has_bias_mark(const OpDef* fc) {
    auto sp = unpack_scalar_params(fc->op_data);
    const ScalarParam* p = scalar_get(sp, "has_bias");
    return p && p->is_numeric && p->value_num != 0;
}

void set_bias_mark(OpDef* fc) {
    auto sp = unpack_scalar_params(fc->op_data);
    for (auto& p : sp) {
        if (p.name == "has_bias") { p.is_numeric = true; p.value_num = 1; p.value_str.clear();
                                    fc->op_data = pack_scalar_params(sp); return; }
    }
    ScalarParam p; p.name = "has_bias"; p.is_numeric = true; p.value_num = 1;
    sp.push_back(std::move(p));
    fc->op_data = pack_scalar_params(sp);
}

// Eltwise_Binary 的 operation == ADD(0)。数字枚举直读; 字符串 "ADD" 兼容;
// 缺省(无 param) 与 extract_eltwise_extra 同口径按 0。
bool is_add_op(const OpDef* od) {
    auto sp = unpack_scalar_params(od->op_data);
    const ScalarParam* p = scalar_get(sp, "operation");
    if (!p) return true;
    if (p->is_numeric) return p->value_num == 0;
    return p->value_str.empty() || p->value_str == "ADD";
}

// output_def 有效 rank(压尾部 1 维)与元素数
uint32_t eff_rank(const OutputDef& d) {
    uint32_t r = d.rank;
    while (r > 0 && d.dims[r - 1] == 1) r--;
    return r;
}
uint64_t eff_elems(const OutputDef& d) {
    uint32_t r = eff_rank(d);
    uint64_t n = 1;
    for (uint32_t i = 0; i < r; ++i) n *= d.dims[i];
    return n;
}

// inputs[2] 现值是否可被 bias 替换(见文件头注释 (a)/(b)/(c))
bool slot_replaceable(const GraphPrepare* gp, const OpDef* fc, const OpDef* bias) {
    const OpDef* cur = gp->get_op_at(fc->inputs[2].src_id);
    if (!cur) return true;                       // 悬空引用, 替换无害
    if (cur->op_id == bias->op_id) return true;  // (b) 已是本 bias
    if (cur->op_id >= 0x10000) return true;      // (a) 注入合成 const (wscale 等)
    // (c) 全零 const (注入复用的零 scale): 读 const_pool 判定
    if (!(cur->is_const() || cur->const_data_size > 0) || cur->const_data_size == 0)
        return false;
    const auto& pool = gp->const_pool();
    if (cur->const_data_offset + cur->const_data_size > pool.size()) return false;
    const uint8_t* d = pool.data() + cur->const_data_offset;
    for (uint64_t i = 0; i < cur->const_data_size; ++i)
        if (d[i] != 0) return false;             // 非零 → 疑似源图 bias, 保守不覆盖
    return true;
}

struct Match {
    OpDef* add = nullptr;
    OpDef* reshape = nullptr;  // 可选; 无则 nullptr
    OpDef* fc = nullptr;
    OpDef* bias = nullptr;
};

Match match_pattern(OpDef* op, GraphPrepare* gp) {
    Match mt;
    if (!live(op)) return mt;
    if (std::string(op_name(op)) != "Eltwise_Binary") return mt;
    if (op->inputs.size() != 2) return mt;
    if (!is_add_op(op)) return mt;

    // 两输入中择一为 FC 链(可隔一层 Reshape), 另一为 bias const
    for (int side = 0; side < 2; ++side) {
        OpDef* chain = gp->get_op_at(op->inputs[side].src_id);
        OpDef* biasc = gp->get_op_at(op->inputs[1 - side].src_id);
        if (!live(chain) || !biasc) continue;
        if (!(biasc->is_const() || biasc->const_data_size > 0)) continue;

        OpDef* reshape = nullptr;
        OpDef* fc = chain;
        if (std::string(op_name(chain)) == "Reshape") {
            if (chain->inputs.size() != 1 || chain->consumers.size() != 1) continue;
            reshape = chain;
            fc = gp->get_op_at(chain->inputs[0].src_id);
            if (!live(fc)) continue;
        }
        // US3a: 链头接受 MatMul(形状/标志经 extract_matmul_extra 同源推导,
        // 与 emit/序列化唯一口径; serialized_extra 在 pass 期为空, 不能读)
        std::string fc_nm = op_name(fc);
        bool is_fc = fc_nm == "FullyConnected";
        bool is_mm = fc_nm == "MatMul";
        if (!is_fc && !is_mm) continue;
        if (fc->inputs.size() < 2) continue;           // 无权重, 畸形
        if (fc->consumers.size() != 1) continue;       // 链头只喂本链
        if (has_bias_mark(fc)) continue;               // 已折过(幂等)
        const OpDef* w = gp->get_op_at(fc->inputs[1].src_id);
        if (!w || !(w->is_const() || w->const_data_size > 0)) continue;
        if (fc->inputs.size() >= 3 && !slot_replaceable(gp, fc, biasc)) continue;

        // M/K/N + bias 长度 == N(严格 [N], 不猜广播)
        uint64_t m = 0, k = 0, n = 0;
        bool batched = false;
        if (is_fc) {
            const OutputDef& od = fc->output_def;
            uint32_t ork = eff_rank(od);
            if (ork < 1) continue;
            n = od.dims[ork - 1];
            m = 1;
            for (uint32_t i = 0; i + 1 < ork; ++i) m *= od.dims[i];
            uint32_t wr = eff_rank(w->output_def);
            if (wr < 1) continue;
            k = w->output_def.dims[wr - 1];            // [N,K] 内维
        } else {
            std::vector<uint8_t> ex = extract_matmul_extra(*gp, *fc);
            ExtraMatMul em{};
            if (ex.size() < sizeof(em)) continue;
            std::memcpy(&em, ex.data(), sizeof(em));
            m = em.m; k = em.k; n = em.n;
            batched = (em.batched != 0);
            if (batched) continue;   // 设备硬错防御: matmul_f16_bias 不支持 batched
        }
        if (m == 0 || k == 0 || n == 0) continue;
        if (eff_elems(biasc->output_def) != n) continue;

        mt.add = op; mt.reshape = reshape; mt.fc = fc; mt.bias = biasc;
        return mt;
    }
    return mt;
}

bool matcher(OpDef* op, GraphPrepare* gp) {
    return match_pattern(op, gp).add != nullptr;
}

int rewrite(OpDef* op, GraphPrepare* gp) {
    Match mt = match_pattern(op, gp);
    if (!mt.add) return 0;
    OpDef* tail = mt.reshape ? mt.reshape : mt.fc;   // Add 消费者改线目标

    // 1) bias 并入 FC inputs[2](append 或替换注入 wscale)
    if (mt.fc->inputs.size() < 3) {
        mt.fc->inputs.push_back({mt.bias->op_id, 0, mt.bias->output_def});
        mt.bias->consumers.push_back(mt.fc->op_id);
    } else if (mt.fc->inputs[2].src_id != mt.bias->op_id) {
        OpDef* old = gp->get_op_at(mt.fc->inputs[2].src_id);
        if (old) {
            old->consumers.erase(
                std::remove(old->consumers.begin(), old->consumers.end(), mt.fc->op_id),
                old->consumers.end());
        }
        mt.fc->inputs[2] = {mt.bias->op_id, 0, mt.bias->output_def};
        mt.bias->consumers.push_back(mt.fc->op_id);
    }
    // 2) 可信标记: extract_fc_extra / host execute 唯一认定口径
    set_bias_mark(mt.fc);

    // 3) Add 的消费者改线到 tail(形状不变: Add 输出 == tail 输出形状)
    std::vector<op_id_t> cons = mt.add->consumers;
    tail->consumers.erase(
        std::remove(tail->consumers.begin(), tail->consumers.end(), mt.add->op_id),
        tail->consumers.end());
    for (op_id_t cid : cons) {
        OpDef* c = gp->get_op_at(cid);
        if (!c) continue;
        for (auto& conn : c->inputs) {
            if (conn.src_id == mt.add->op_id) {
                conn.src_id = tail->op_id;
                conn.src_out_def = tail->output_def;
            }
        }
        tail->consumers.push_back(cid);
    }
    mt.add->flags |= OP_DEAD;
    return 1;
}

} // namespace

void register_pass_fc_bias() {
    GraphPass p;
    p.name = "fc_bias_fold";
    p.phase = PHASE_3;  // 12492: 融合槽(与 legacy fusion 同相位; 本规则默认开)
    p.matcher = matcher;
    p.rewrite = rewrite;
    PassManager::instance().add_pass(std::move(p));
}

} // namespace hnnx
