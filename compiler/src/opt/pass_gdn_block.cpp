// pass_gdn_block —— 009 US3: linear_attn 整块折叠为单发射 GdnBlock (默认关)
// =====================================================================
// 冻结契约 (specs/009-gdn-block-fusion, T007 边界 + T015 §[16]):
//   判别    grouping 含 "/linear_attn/" (containment), 同层共享层键前缀;
//   survivor 权重 matmul (in_proj*/out_proj) + beta/g 生产链 + 输出 norm 区
//           (rms_norm/z 门) + aux 投影链 (in_proj_a/b/z) + 胶水 op
//           (既不被可折成员消费、也不消费可折成员 —— 真实图 norm→out_proj
//            间的 Reshape_86 / Softplus 前的 Add / in_proj 前后 reshape);
//   锚      输出离开可折集的成员 → 就地改名 GdnBlock (op_id 不变, 块外
//           消费者零扰动), 接线 4 输入 [xqkv, conv_w, beta, g];
//   折      其余可折成员 flags |= OP_DEAD;
//   闭合    非锚成员被块外消费 / 多出口 / 接线源不全 / survivor 生产链
//           断裂 → 整层零改写进 FallbackRecord (禁静默)。
//
// token 双口径 (1010 net.json 校名 + 合成测试口径):
//   xqkv = FullyConnected/MatMul 且 grouping 含 "in_proj" 非 aux;
//   beta = "/beta/" 或 "/Sigmoid_1" (真实 005 终态图名);
//   g    = "/g/"    或 "/Softplus";
//   conv_w const = grouping 含 "conv_w" 或 "conv1d"。
// 覆盖率 C1 经 gdn_block_opt_stats() 输出 (hnnx_compile --opt-stats)。
#include "hnnx/opt/pass_gdn_block.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace hnnx {

namespace {

constexpr char kLayerMark[] = "/linear_attn/";
constexpr char kGdnName[] = "GdnBlock";

struct GdnStats {
    uint32_t recognized = 0;  // 识别出的 linear_attn 层数
    uint32_t lowered = 0;     // 折叠成功层数
    uint32_t fallback = 0;    // 跳过层数
    std::vector<std::string> reasons;
};
GdnStats g_stats;

void note_fallback(const std::string& layer, const std::string& why) {
    g_stats.fallback++;
    if (g_stats.reasons.size() < 12) g_stats.reasons.push_back(layer + ": " + why);
}

bool name_is(const OpDef* o, const char* n) {
    return o && o->name_tag && o->name_tag->name() &&
           std::strcmp(o->name_tag->name(), n) == 0;
}

// 层键 = "/model/…/linear_attn/"。真实图存在 "Reshape_post_/model/…" 类
// 转换器前缀噪声 —— 统一从 "/model/" 起算归一, 否则同层裂成两个伪层。
std::string layer_key(const std::string& g) {
    size_t pos = g.find(kLayerMark);
    if (pos == std::string::npos) return {};
    size_t end = pos + sizeof(kLayerMark) - 1;
    size_t m = g.find("/model/");
    if (m != std::string::npos && m < pos) return g.substr(m, end - m);
    return g.substr(0, end);
}
std::string layer_key(const OpDef* o) { return layer_key(o->grouping); }

bool is_weight_matmul(const OpDef* o) {
    return name_is(o, "FullyConnected") || name_is(o, "MatMul");
}
// survivor 权重投影 = in_proj*/out_proj 的 FC/MatMul。核内 act×act matmul
// (MatMul_2/3/5/6, GDN 状态更新, 1010 net.json 实证无权重 const 输入)必须
// 折 —— 全量 MatMul 判 survivor 会把层裂成 11 个出口。
bool is_proj_matmul(const OpDef* o, const std::string& g) {
    return is_weight_matmul(o) &&
           (g.find("in_proj") != std::string::npos ||
            g.find("out_proj") != std::string::npos);
}
// 真实图 aux 投影链: in_proj_a/b/z 整链 (FC + 前后 reshape 同 grouping 前缀)
bool is_aux_proj(const std::string& g) {
    return g.find("in_proj_a") != std::string::npos ||
           g.find("in_proj_b") != std::string::npos ||
           g.find("in_proj_z") != std::string::npos;
}
bool is_qkv_proj(const std::string& g) {
    return g.find("in_proj") != std::string::npos && !is_aux_proj(g);
}
bool is_beta_site(const std::string& g) {
    return g.find("/beta/") != std::string::npos ||
           g.find("/Sigmoid_1") != std::string::npos;
}
bool is_g_site(const std::string& g) {
    return g.find("/g/") != std::string::npos ||
           g.find("/Softplus") != std::string::npos;
}
// 输出 norm + z 门在块外 (T011 host 参考不含 rms_norm/z-gate)
bool is_norm_site(const std::string& g) {
    return g.find("/norm/") != std::string::npos ||
           g.find("rms_norm") != std::string::npos;
}
bool is_conv_w_const(const OpDef* o) {
    if (!o || !o->is_const()) return false;
    const std::string& g = o->grouping;
    return g.find("conv_w") != std::string::npos ||
           g.find("conv1d") != std::string::npos;
}

// 整层折叠。与触发 op 无关(任一成员触发都做全层); 失败 = 整层零改写。
int collapse_layer(OpDef* entry, GraphPrepare* gp) {
    const std::string key = layer_key(entry);
    if (key.empty() || !gp) return 0;

    std::vector<OpDef*> members;
    gp->for_each_op([&](OpDef* o) {
        if (!o || o->is_dead()) return;
        if (layer_key(o->grouping) != key) return;
        if (name_is(o, "Input") || name_is(o, "Output")) return;
        members.push_back(o);
    });
    if (members.empty()) return 0;
    for (OpDef* m : members)
        if (name_is(m, kGdnName)) return 0;  // 已折叠(幂等二跑零改写)
    g_stats.recognized++;

    std::unordered_map<op_id_t, size_t> midx;
    for (size_t i = 0; i < members.size(); i++) midx[members[i]->op_id] = i;
    auto member_idx = [&](op_id_t id) -> int {
        auto it = midx.find(id);
        return it == midx.end() ? -1 : (int)it->second;
    };

    // ---- survivor / 可折 二分 ----
    std::vector<char> surv(members.size(), 0);
    for (size_t i = 0; i < members.size(); i++) {
        OpDef* m = members[i];
        const std::string& g = m->grouping;
        surv[i] = is_conv_w_const(m) || m->is_const() || is_norm_site(g) ||
                  is_aux_proj(g) || is_proj_matmul(m, g) ||
                  is_beta_site(g) || is_g_site(g);
    }
    // 胶水吸收 (fixpoint): 不与可折集相邻的成员归 survivor。
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i = 0; i < members.size(); i++) {
            if (surv[i]) continue;
            OpDef* m = members[i];
            bool touches_fold = false;
            for (op_id_t c : m->consumers) {
                int ci = member_idx(c);
                OpDef* cc = gp->get_op_at(c);
                if (ci >= 0 && !surv[ci] && cc && !cc->is_dead()) touches_fold = true;
            }
            for (const auto& conn : m->inputs) {
                int pi = member_idx(conn.src_id);
                if (pi >= 0 && !surv[pi]) touches_fold = true;
            }
            if (!touches_fold) { surv[i] = 1; changed = true; }
        }
    }

    std::vector<OpDef*> fold;
    for (size_t i = 0; i < members.size(); i++)
        if (!surv[i]) fold.push_back(members[i]);
    if (fold.empty()) {
        note_fallback(key, "无可折成员");
        return 0;
    }

    // ---- 接线源识别 (须各恰一) ----
    OpDef *xqkv = nullptr, *beta = nullptr, *gdec = nullptr, *convw = nullptr;
    int n_qkv = 0, n_beta = 0, n_g = 0, n_convw = 0;
    for (size_t i = 0; i < members.size(); i++) {
        if (!surv[i]) continue;
        OpDef* m = members[i];
        if (m->is_const()) {
            if (is_conv_w_const(m)) { convw = m; n_convw++; }
            continue;
        }
        const std::string& g = m->grouping;
        if (is_aux_proj(g) || is_norm_site(g)) continue;
        if (is_weight_matmul(m) && is_qkv_proj(g)) { xqkv = m; n_qkv++; }
        else if (is_beta_site(g)) { beta = m; n_beta++; }
        else if (is_g_site(g)) { gdec = m; n_g++; }
    }
    if (n_convw != 1) {
        // 真实图回退: loader 给权重 const(kind==1) 不设 grouping → 不入层
        // 成员。conv_w = conv 成员(DepthWiseConv2d, 输入 0 = act)下标 ≥1 的
        // const 输入。
        n_convw = 0;
        convw = nullptr;
        for (OpDef* f : fold) {
            bool is_conv = name_is(f, "DepthWiseConv2d") ||
                           f->grouping.find("/conv1d") != std::string::npos;
            if (!is_conv) continue;
            for (size_t k = 1; k < f->inputs.size(); k++) {
                OpDef* p = gp->get_op_at(f->inputs[k].src_id);
                if (!p || p->is_dead() || !p->is_const()) continue;
                // 注入参数 const 名 = ".../Conv_2d_stride|pad_amount|dilation";
                // 权重 const(kind==1) 名 "$Const" 无 grouping; wscale/iscale
                // (inject_htp_prepare_inputs 的 per-axis scale)同名 "$Const", 但
                // 权重尺寸最大 —— 跳参数名取最大(1010 实测: 权重 49152B=
                // [1,4,1,6144]f16 > wscale 24576B > stride/pad/dilation 8/16/8B)。
                const char* pn = p->name_tag ? p->name_tag->name() : nullptr;
                if (pn && std::strstr(pn, "Conv_2d_")) continue;
                if (!convw || p->const_data_size > convw->const_data_size) convw = p;
            }
        }
        if (convw) n_convw = 1;
    }
    if (n_qkv != 1 || n_beta != 1 || n_g != 1 || n_convw != 1) {
        char why[96];
        std::snprintf(why, sizeof(why),
                      "接线源不全 qkv=%d beta=%d g=%d convw=%d", n_qkv, n_beta, n_g, n_convw);
        note_fallback(key, why);
        return 0;
    }

    // ---- 锚 = 恰一可折成员, 输出被可折集外存活 op 消费; 其余成员被块外
    //      消费即闭合破坏(与多出口同判) ----
    OpDef* anchor = nullptr;
    int n_anchor = 0;
    for (OpDef* f : fold) {
        for (op_id_t c : f->consumers) {
            OpDef* cc = gp->get_op_at(c);
            if (!cc || cc->is_dead()) continue;
            int ci = member_idx(c);
            if (ci < 0 || surv[ci]) { n_anchor++; anchor = f; break; }
        }
    }
    if (n_anchor != 1) {
        if (getenv("GEHTP_GDN_DBG")) {
            std::fprintf(stderr, "[anchor-dbg] %s n_anchor=%d\n", key.c_str(), n_anchor);
            int shown = 0;
            for (OpDef* f : fold) {
                for (op_id_t c : f->consumers) {
                    OpDef* cc = gp->get_op_at(c);
                    if (!cc || cc->is_dead()) continue;
                    int ci = member_idx(c);
                    if (ci < 0 || surv[ci]) {
                        if (shown++ < 8)
                            std::fprintf(stderr, "  exit: %u %s grp=%s -> cons %u %s grp=%.80s\n",
                                         f->op_id, f->name_tag?f->name_tag->name():"(-)",
                                         f->grouping.c_str(), c,
                                         cc->name_tag?cc->name_tag->name():"(-)",
                                         cc->grouping.c_str());
                        break;
                    }
                }
            }
        }
        note_fallback(key, n_anchor == 0 ? "无出口锚" : "旁路消费者/多出口(闭合破坏)");
        return 0;
    }

    // ---- survivor 生产链保全: survivor 的存活生产者不得被折(锚除外) ----
    for (size_t i = 0; i < members.size(); i++) {
        if (!surv[i]) continue;
        for (const auto& conn : members[i]->inputs) {
            int pi = member_idx(conn.src_id);
            if (pi >= 0 && !surv[pi] && members[pi] != anchor) {
                note_fallback(key, "survivor 生产链断裂(" +
                              std::to_string(members[i]->op_id) + " <- " +
                              std::to_string(conn.src_id) + ")");
                return 0;
            }
        }
    }

    // ---- 改写: 就地改名 + 四输入接线 + 其余成员 OP_DEAD ----
    const op_id_t a_id = anchor->op_id;
    std::vector<op_id_t> old_src;
    for (const auto& conn : anchor->inputs) old_src.push_back(conn.src_id);

    InputConn ic[4] = {{xqkv->op_id, 0}, {convw->op_id, 0},
                       {beta->op_id, 0}, {gdec->op_id, 0}};
    anchor->inputs.assign(ic, ic + 4);
    anchor->name_tag = string_tag_t::map_str(kGdnName);

    auto rm_consumer = [&](op_id_t src) {
        OpDef* p = gp->get_op_at(src);
        if (!p) return;
        p->consumers.erase(std::remove(p->consumers.begin(), p->consumers.end(), a_id),
                           p->consumers.end());
    };
    auto add_consumer = [&](op_id_t src) {
        OpDef* p = gp->get_op_at(src);
        if (!p) return;
        if (std::find(p->consumers.begin(), p->consumers.end(), a_id) == p->consumers.end())
            p->consumers.push_back(a_id);
    };
    const op_id_t keep[4] = {xqkv->op_id, convw->op_id, beta->op_id, gdec->op_id};
    for (op_id_t s : old_src) {
        if (std::find(keep, keep + 4, s) == keep + 4) rm_consumer(s);
    }
    for (op_id_t s : keep) add_consumer(s);

    int n_dead = 0;
    for (OpDef* f : fold) {
        if (f == anchor) continue;
        f->flags |= OP_DEAD;
        n_dead++;
    }
    g_stats.lowered++;
    return n_dead + 1;
}

bool gdn_matcher(OpDef* op, GraphPrepare*) {
    if (!op || op->is_dead() || op->is_const()) return false;
    if (name_is(op, "Input") || name_is(op, "Output")) return false;
    if (name_is(op, kGdnName)) return false;  // 幂等: 已折叠出口不再触发
    return op->grouping.find(kLayerMark) != std::string::npos;
}

} // namespace

void register_pass_gdn_block() {
    GraphPass p;
    p.name = "gdn_block";
    // PHASE_0 且注册最前(pass_manager.cpp register_builtin_passes 首位):
    // 块折叠是结构 pass, 必须先于 const_fold —— 后者会把全 const 输入的
    // beta/g 生产链折成 const 并被 DCE 清退(1010 实测 members 5/7, 接线
    // 源 beta=0/g=0), 生产链既失, 块边界无从识别。
    p.phase = PHASE_0;
    p.enabled = false;       // 默认关 —— 融合非设备准入, --enable-pass 打开
    p.matcher = gdn_matcher;
    p.rewrite = [](OpDef* op, GraphPrepare* gp) { return collapse_layer(op, gp); };
    PassManager::instance().add_pass(std::move(p));
}

std::string gdn_block_opt_stats() {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "gdn_block: recognized=%u lowered=%u fallback=%u coverage=%u/%u",
                  g_stats.recognized, g_stats.lowered, g_stats.fallback,
                  g_stats.lowered, g_stats.recognized);
    std::string s = buf;
    for (const auto& r : g_stats.reasons) s += "\n[gdn-fallback] " + r;
    return s;
}

} // namespace hnnx
