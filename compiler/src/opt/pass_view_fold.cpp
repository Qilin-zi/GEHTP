// pass_view_fold —— 视图链合并/吸收(PHASE_1, 默认开)
// =====================================================================
// 锚定在链**内层** op(producer)上: 唯一消费者 C 构成模式时改写 C。
// 锚在内层而非外层的理由: run_phase 改写后只把 anchor 的 src/consumers
// 压回 worklist —— 锚在内层, 改写的是消费者 C, C 必被重访, SS∘SS∘SS
// 长链得以级联合并(锚在外层时 C 不入队, 3 链漏 1)。
//
// 语义锚定(pass 与 host/设备唯一同源口径, 2026-09-30 核实):
//   - StridedSlice 效果 = ranges const 的每轴三元组 [b,e,s](交错布局,
//     loader 拍平 [[b,e,s]×rank]); host(ops.cpp)与设备(exec_slice)都不读
//     begin/end/new_axes/shrink 掩码(惰性元数据)。
//   - 轴系右对齐: ranges 秩 ≤ 输入秩, 前导轴隐含全区间。
//   - Transpose: in_coord[perm[d]] = out_coord[d](host ops.cpp 同款)。
//
// 合成后的 ranges 写成**新 const**(append_const_node, 名 "view_fold_ranges",
// find_param_const 按 "ranges" 子串命中): 不改写旧 const 池字节 —— CSE 可
// 能已让两个相同 ranges 共享一个 const 节点, 原地改会串味。
#include "hnnx/opt/pass_view_fold.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/scalar_params.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace hnnx {

namespace {

constexpr uint32_t MAX_AXES = 5;

const char* op_name(const OpDef* od) {
    return (od && od->name_tag && od->name_tag->name()) ? od->name_tag->name() : "";
}

bool live(const OpDef* od) {
    return od && od->is_enabled() && !od->is_dead() && !od->is_const();
}

uint64_t dims_elems(const OutputDef& d, uint32_t rank) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < rank && i < 5; ++i) n *= d.dims[i];
    return n;
}

// ---------------------------------------------------------------------------
// SliceSpec: 某一 SS op 在其输入帧上的具体区间(掩码已无视, 前导轴已补全)
// ---------------------------------------------------------------------------
struct SliceSpec {
    uint32_t rank = 0;              // 输入帧秩(= producer rank)
    int64_t b[MAX_AXES] = {0, 0, 0, 0, 0};
    int64_t e[MAX_AXES] = {0, 0, 0, 0, 0};
    int64_t s[MAX_AXES] = {1, 1, 1, 1, 1};
    int64_t dim[MAX_AXES] = {1, 1, 1, 1, 1};

    int64_t cnt(uint32_t ax) const {
        int64_t span = e[ax] > b[ax] ? e[ax] - b[ax] : 0;
        return (span + s[ax] - 1) / s[ax];
    }
    bool in_bounds(uint32_t ax) const { return b[ax] + (cnt(ax) - 1) * s[ax] < dim[ax]; }
    bool full_identity() const {
        for (uint32_t ax = 0; ax < rank; ++ax)
            if (b[ax] != 0 || s[ax] != 1 || cnt(ax) != dim[ax]) return false;
        return true;
    }
};

// ranges const 解析: inputs + tensor_param_ids 里名含 "ranges" 的 const
// (与 extract_strided_slice_extra / find_param_const 同口径)。
const OpDef* find_named_param(const GraphPrepare* gp, const OpDef* od, const char* key) {
    auto check = [&](op_id_t id) -> const OpDef* {
        const OpDef* s = gp->get_op_at(id);
        if (!s || !s->name_tag) return nullptr;
        const char* n = s->name_tag->name();
        return (n && std::strstr(n, key)) ? s : nullptr;
    };
    for (const auto& c : od->inputs)
        if (const OpDef* r = check(c.src_id)) return r;
    for (op_id_t tpid : od->tensor_param_ids)
        if (const OpDef* r = check(tpid)) return r;
    return nullptr;
}

bool read_i32_array(const GraphPrepare* gp, const OpDef* pc,
                    std::vector<int32_t>& out) {
    if (!pc || pc->const_data_size < 4 || (pc->const_data_size % 4) != 0) return false;
    const auto& pool = gp->const_pool();
    if (pc->const_data_offset + pc->const_data_size > pool.size()) return false;
    size_t n = pc->const_data_size / 4;
    out.resize(n);
    std::memcpy(out.data(), pool.data() + pc->const_data_offset, n * 4);
    return true;
}

// SS op → 具体区间(失败 = 不动该 op)。rank/输出元素数一致性都在这里门住。
bool parse_slice(const GraphPrepare* gp, const OpDef* ss, SliceSpec& spec) {
    if (std::string(op_name(ss)) != "StridedSlice") return false;
    if (ss->inputs.empty()) return false;
    const OpDef* src = gp->get_op_at(ss->inputs[0].src_id);
    if (!src || src->output_def.rank == 0 || src->output_def.rank > MAX_AXES) return false;

    const OpDef* pc = find_named_param(gp, ss, "ranges");
    std::vector<int32_t> rg;
    if (!read_i32_array(gp, pc, rg)) return false;
    if (rg.size() < 3 || (rg.size() % 3) != 0) return false;
    uint32_t rr = (uint32_t)(rg.size() / 3);              // ranges 秩
    if (rr == 0 || rr > MAX_AXES || rr > src->output_def.rank) return false;

    uint32_t R = src->output_def.rank;
    spec = SliceSpec{};
    spec.rank = R;
    uint32_t off = R - rr;                                 // 右对齐: 前导全区间
    for (uint32_t ax = 0; ax < R; ++ax) {
        spec.dim[ax] = src->output_def.dims[ax];
        if (ax < off) { spec.b[ax] = 0; spec.e[ax] = spec.dim[ax]; spec.s[ax] = 1; }
        else {
            uint32_t j = (ax - off) * 3;
            spec.b[ax] = rg[j]; spec.e[ax] = rg[j + 1]; spec.s[ax] = rg[j + 2];
        }
        if (spec.b[ax] < 0 || spec.s[ax] < 1) return false;      // 负步长/负起点: 发射侧本就硬错
        if (spec.cnt(ax) == 0) return false;                     // 空切片: 罕见, 保守跳过
        if (!spec.in_bounds(ax)) return false;                   // 依赖 clamp 语义的图不碰
    }
    uint64_t prod = 1;
    for (uint32_t ax = 0; ax < R; ++ax) prod *= (uint64_t)spec.cnt(ax);
    if (prod != dims_elems(ss->output_def, ss->output_def.rank)) return false;
    return true;
}

// 区间合成: outer ∘ inner(outer 消费 inner 输出)。要求同秩帧 + 逐轴对齐。
bool compose_slice(const SliceSpec& inner, const SliceSpec& outer, SliceSpec& out) {
    if (inner.rank != outer.rank) return false;
    for (uint32_t ax = 0; ax < inner.rank; ++ax) {
        if (outer.dim[ax] != inner.cnt(ax)) return false;        // outer 的输入维须是 inner 的输出维
        if (outer.b[ax] + (outer.cnt(ax) - 1) * outer.s[ax] >= inner.cnt(ax)) return false;
    }
    out = SliceSpec{};
    out.rank = inner.rank;
    for (uint32_t ax = 0; ax < inner.rank; ++ax) {
        out.dim[ax] = inner.dim[ax];
        out.b[ax] = inner.b[ax] + outer.b[ax] * inner.s[ax];
        out.s[ax] = inner.s[ax] * outer.s[ax];
        out.e[ax] = out.b[ax] + outer.cnt(ax) * out.s[ax];       // e' 使 ceil((e'-b')/s') == cnt_outer
    }
    return true;
}

// ---------------------------------------------------------------------------
// 图改写小件
// ---------------------------------------------------------------------------

// 掩码清零: 合成后的 ranges 已是具体值, 陈旧掩码只会误导阅读(host/设备
// 不读, 纯卫生动作)。
void zero_slice_masks(OpDef* ss) {
    auto sp = unpack_scalar_params(ss->op_data);
    bool changed = false;
    for (auto& p : sp) {
        if (p.name == "begin_mask" || p.name == "end_mask" ||
            p.name == "new_axes_mask" || p.name == "shrink_axes") {
            if (!(p.is_numeric && p.value_num == 0)) {
                p.is_numeric = true; p.value_num = 0; p.value_str.clear();
                changed = true;
            }
        }
    }
    if (changed) ss->op_data = pack_scalar_params(sp);
}

// 新 ranges const(不改旧池: CSE 可能共享)。名含 "ranges" 供 find_param_const 命中。
op_id_t make_ranges_const(GraphPrepare* gp, const SliceSpec& spec) {
    int32_t triples[MAX_AXES * 3];
    for (uint32_t ax = 0; ax < spec.rank; ++ax) {
        triples[ax * 3 + 0] = (int32_t)spec.b[ax];
        triples[ax * 3 + 1] = (int32_t)spec.e[ax];
        triples[ax * 3 + 2] = (int32_t)spec.s[ax];
    }
    OutputDef od{};
    od.rank = 2;
    od.dtype = static_cast<uint32_t>(DType::Int32);
    od.dims[0] = spec.rank;
    od.dims[1] = 3;
    od.element_size = 4;
    op_id_t id = gp->append_const_node(0, od,
                                       reinterpret_cast<const uint8_t*>(triples),
                                       spec.rank * 3 * 4);
    if (!id) return 0;
    OpDef* c = gp->get_op_at(id);
    if (c) c->name_tag = string_tag_t::map_str("view_fold_ranges");
    return id;
}

// 把 ss 的 ranges 引用换到新 const(tensor_param_ids 或 inputs, 哪里引用换哪里)。
bool rewire_ranges_ref(GraphPrepare* gp, OpDef* ss, op_id_t new_id) {
    const OpDef* old = find_named_param(gp, ss, "ranges");
    if (!old) return false;
    bool touched = false;
    for (auto& tpid : ss->tensor_param_ids)
        if (tpid == old->op_id) { tpid = new_id; touched = true; }
    for (auto& conn : ss->inputs)
        if (conn.src_id == old->op_id) {
            conn.src_id = new_id;
            conn.src_out_def = gp->get_op_at(new_id) ? gp->get_op_at(new_id)->output_def : conn.src_out_def;
            touched = true;
        }
    if (!touched) return false;
    OpDef* nc = gp->get_op_at(new_id);
    if (nc) nc->consumers.push_back(ss->op_id);
    OpDef* oc = gp->get_op_at(old->op_id);
    if (oc) {
        oc->consumers.erase(std::remove(oc->consumers.begin(), oc->consumers.end(), ss->op_id),
                            oc->consumers.end());
    }
    return true;
}

// op 死亡 + 其消费者全部改线到 producer(旁路)。要求形状一致(调用方保证)。
void bypass_op(GraphPrepare* gp, OpDef* op, OpDef* producer) {
    std::vector<op_id_t> cons = op->consumers;
    producer->consumers.erase(std::remove(producer->consumers.begin(),
                                          producer->consumers.end(), op->op_id),
                              producer->consumers.end());
    for (op_id_t cid : cons) {
        OpDef* c = gp->get_op_at(cid);
        if (!c) continue;
        bool fed = false;
        for (auto& conn : c->inputs) {
            if (conn.src_id == op->op_id) {
                conn.src_id = producer->op_id;
                conn.src_out_def = producer->output_def;
                fed = true;
            }
        }
        if (fed) producer->consumers.push_back(cid);
    }
    op->flags |= OP_DEAD;
}

// 把 inner(被吸收者)的死亡处理 + 消费者 c 的输入 0 号换到 underlying。
void absorb_into_consumer(GraphPrepare* gp, OpDef* inner, OpDef* c, OpDef* underlying) {
    underlying->consumers.erase(std::remove(underlying->consumers.begin(),
                                            underlying->consumers.end(), inner->op_id),
                                underlying->consumers.end());
    for (auto& conn : c->inputs) {
        if (conn.src_id == inner->op_id) {
            conn.src_id = underlying->op_id;
            conn.src_out_def = underlying->output_def;
        }
    }
    underlying->consumers.push_back(c->op_id);
    inner->flags |= OP_DEAD;
}

// 压前导 1 后形状等价(旁路门: host 按输出秩分解坐标, 前导 1 不改变
// C 序线性化, 但保守起见只允许前导 1 差异)。
bool shape_eq_mod_leading1(const OutputDef& a, const OutputDef& b) {
    uint32_t ra = a.rank, rb = b.rank;
    while (ra > 0 && a.dims[0] == 1) ra--;
    while (rb > 0 && b.dims[0] == 1) rb--;
    if (ra != rb) return false;
    // 压的是前导, 余下轴序不变
    for (uint32_t i = 0; i < ra; ++i)
        if (a.dims[a.rank - ra + i] != b.dims[b.rank - rb + i]) return false;
    return true;
}

// ---------------------------------------------------------------------------
// R1: SS∘SS 合并 + identity 删除
// ---------------------------------------------------------------------------
bool r1_pair(GraphPrepare* gp, OpDef* inner, OpDef*& outer, SliceSpec& merged, bool& identity) {
    if (std::string(op_name(inner)) != "StridedSlice") return false;
    if (inner->consumers.size() != 1) return false;
    outer = gp->get_op_at(inner->consumers[0]);
    if (!live(outer) || std::string(op_name(outer)) != "StridedSlice") return false;

    SliceSpec si, so;
    if (!parse_slice(gp, inner, si) || !parse_slice(gp, outer, so)) return false;
    if (!compose_slice(si, so, merged)) return false;
    identity = merged.full_identity();
    if (identity) {
        const OpDef* producer = gp->get_op_at(inner->inputs[0].src_id);
        if (!producer || !shape_eq_mod_leading1(outer->output_def, producer->output_def))
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// R2: Reshape 吸收进 SS(纯 1 维增删 + 单消费者)
// ---------------------------------------------------------------------------
bool pure_1dim_reshape(const OutputDef& in_od, const OutputDef& out_od) {
    if (dims_elems(in_od, in_od.rank) != dims_elems(out_od, out_od.rank)) return false;
    std::vector<uint32_t> si, so;
    for (uint32_t i = 0; i < in_od.rank && i < 5; ++i) if (in_od.dims[i] != 1) si.push_back(in_od.dims[i]);
    for (uint32_t i = 0; i < out_od.rank && i < 5; ++i) if (out_od.dims[i] != 1) so.push_back(out_od.dims[i]);
    return si == so;
}

bool r2_pair(GraphPrepare* gp, OpDef* reshape, OpDef*& ss, SliceSpec& merged, bool& identity) {
    if (std::string(op_name(reshape)) != "Reshape") return false;
    if (reshape->inputs.empty() || reshape->consumers.size() != 1) return false;
    ss = gp->get_op_at(reshape->consumers[0]);
    if (!live(ss) || std::string(op_name(ss)) != "StridedSlice") return false;

    const OpDef* underlying = gp->get_op_at(reshape->inputs[0].src_id);
    if (!live(underlying) || underlying->output_def.rank == 0 ||
        underlying->output_def.rank > MAX_AXES) return false;
    if (!pure_1dim_reshape(underlying->output_def, reshape->output_def)) return false;

    SliceSpec sss;                                            // ss 在 reshape 输出帧上的区间
    if (!parse_slice(gp, ss, sss)) return false;
    if (sss.rank != reshape->output_def.rank) return false;

    // 合并区间落到 underlying 帧: 非 1 轴按序对应(ss 的区间直传),
    // underlying 的 1 轴填全区间, reshape 输出帧新增的 1 轴要求 ss 平凡。
    merged = SliceSpec{};
    merged.rank = underlying->output_def.rank;
    uint32_t j = 0;                                           // reshape 输出帧游标(只推进非 1 轴)
    for (uint32_t u = 0; u < merged.rank; ++u) {
        merged.dim[u] = underlying->output_def.dims[u];
        if (merged.dim[u] == 1) { merged.b[u] = 0; merged.e[u] = 1; merged.s[u] = 1; continue; }
        while (j < sss.rank && reshape->output_def.dims[j] == 1) {
            // 输出帧新增 1 轴: ss 必须平凡([0:1:1]), 否则不可表示
            if (!(sss.b[j] == 0 && sss.s[j] == 1 && sss.cnt(j) == 1)) return false;
            j++;
        }
        if (j >= sss.rank || reshape->output_def.dims[j] != merged.dim[u]) return false;
        merged.b[u] = sss.b[j]; merged.e[u] = sss.e[j]; merged.s[u] = sss.s[j];
        j++;
    }
    for (uint32_t ax = 0; ax < merged.rank; ++ax)
        if (merged.cnt(ax) == 0 || !merged.in_bounds(ax)) return false;
    uint64_t prod = 1;
    for (uint32_t ax = 0; ax < merged.rank; ++ax) prod *= (uint64_t)merged.cnt(ax);
    if (prod != dims_elems(ss->output_def, ss->output_def.rank)) return false;

    identity = merged.full_identity();
    if (identity && !shape_eq_mod_leading1(ss->output_def, underlying->output_def)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// R3: Transpose∘Transpose 恒等对双删
// ---------------------------------------------------------------------------
bool read_perm(const GraphPrepare* gp, const OpDef* t, std::vector<int32_t>& perm) {
    const OpDef* pc = find_named_param(gp, t, "perm");
    std::vector<int32_t> v;
    if (!read_i32_array(gp, pc, v)) return false;
    if (v.empty() || v.size() > MAX_AXES) return false;
    perm = std::move(v);
    return true;
}

bool r3_pair(GraphPrepare* gp, OpDef* inner, OpDef*& outer) {
    if (std::string(op_name(inner)) != "Transpose") return false;
    if (inner->inputs.empty() || inner->consumers.size() != 1) return false;
    outer = gp->get_op_at(inner->consumers[0]);
    if (!live(outer) || std::string(op_name(outer)) != "Transpose") return false;

    std::vector<int32_t> p1, p2;
    if (!read_perm(gp, inner, p1) || !read_perm(gp, outer, p2)) return false;
    if (p1.size() != p2.size()) return false;
    for (int32_t v : p1) if (v < 0 || (size_t)v >= p1.size()) return false;
    for (int32_t v : p2) if (v < 0 || (size_t)v >= p2.size()) return false;
    for (size_t d = 0; d < p2.size(); ++d)                    // 复合恒等: perm1[perm2[d]] == d
        if (p1[(size_t)p2[d]] != (int32_t)d) return false;
    return true;
}

// ---------------------------------------------------------------------------
// matcher / rewrite(锚 = inner; 改写 = 唯一消费者)
// ---------------------------------------------------------------------------
bool matcher(OpDef* op, GraphPrepare* gp) {
    if (!live(op) || op->inputs.empty()) return false;
    SliceSpec merged; bool identity = false; OpDef* other = nullptr;
    if (r1_pair(gp, op, other, merged, identity)) return true;
    if (r2_pair(gp, op, other, merged, identity)) return true;
    if (r3_pair(gp, op, other)) return true;
    return false;
}

// 共用改写: merged 写进 ss(新 ranges + 掩码清零); identity 则整体旁路。
int apply_merge(GraphPrepare* gp, OpDef* ss, const SliceSpec& merged, bool identity,
                OpDef* absorbed /*inner, 待死*/) {
    if (identity) {
        OpDef* producer = gp->get_op_at(absorbed->inputs[0].src_id);
        if (!live(producer)) return 0;
        bypass_op(gp, ss, producer);
        absorbed->flags |= OP_DEAD;
        return 1;
    }
    op_id_t nid = make_ranges_const(gp, merged);
    if (!nid) return 0;
    if (!rewire_ranges_ref(gp, ss, nid)) return 0;
    zero_slice_masks(ss);
    absorb_into_consumer(gp, absorbed, ss, gp->get_op_at(absorbed->inputs[0].src_id));
    return 1;
}

int rewrite(OpDef* op, GraphPrepare* gp) {
    if (!live(op) || op->inputs.empty()) return 0;
    SliceSpec merged; bool identity = false; OpDef* other = nullptr;

    if (r1_pair(gp, op, other, merged, identity))
        return apply_merge(gp, other, merged, identity, op);

    if (r2_pair(gp, op, other, merged, identity))
        return apply_merge(gp, other, merged, identity, op);

    if (r3_pair(gp, op, other)) {
        OpDef* producer = gp->get_op_at(op->inputs[0].src_id);
        if (!live(producer)) return 0;
        bypass_op(gp, other, producer);                       // 外层旁路
        op->flags |= OP_DEAD;                                 // 内层同死
        return 1;
    }
    return 0;
}

} // namespace

void register_pass_view_fold() {
    GraphPass p;
    p.name = "view_fold";
    p.phase = PHASE_1;
    p.matcher = matcher;
    p.rewrite = rewrite;
    PassManager::instance().add_pass(std::move(p));
}

// ---------------------------------------------------------------------------
// T004 普查: 与 pass 同谓词的基数 dump(GEHTP_DUMP_VIEWS 钩子)
// ---------------------------------------------------------------------------
void dump_view_census(GraphPrepare* gp, std::FILE* out) {
    if (!gp || !out) return;
    uint32_t n_ss = 0, n_rs = 0, n_tp = 0;
    uint32_t r1 = 0, r2 = 0, r3 = 0, ss_ident = 0;
    uint32_t eb_pair = 0;
    std::map<int64_t, uint32_t> eb_hist;

    gp->for_each_op([&](OpDef* od) {
        if (!od || !od->is_enabled() || od->is_dead() || od->is_const()) return;
        std::string nm = op_name(od);
        if (nm == "StridedSlice") {
            n_ss++;
            SliceSpec m; bool idn = false; OpDef* o = nullptr;
            if (r1_pair(gp, od, o, m, idn)) r1++;
            SliceSpec sp;
            if (parse_slice(gp, od, sp) && sp.full_identity()) ss_ident++;
        } else if (nm == "Reshape") {
            n_rs++;
            SliceSpec m; bool idn = false; OpDef* o = nullptr;
            if (r2_pair(gp, od, o, m, idn)) r2++;
        } else if (nm == "Transpose") {
            n_tp++;
            OpDef* o = nullptr;
            if (r3_pair(gp, od, o)) r3++;
        } else if (nm == "Eltwise_Binary") {
            auto sp = unpack_scalar_params(od->op_data);
            const ScalarParam* p = scalar_get(sp, "operation");
            int64_t opc = p && p->is_numeric ? (int64_t)p->value_num : 0;
            if (od->inputs.size() == 2) {
                OpDef* a = gp->get_op_at(od->inputs[0].src_id);
                OpDef* b = gp->get_op_at(od->inputs[1].src_id);
                // 与 pass_eltwise_const 同口径: const 在任一侧, 允许前导 1 Reshape 包装
                auto cst_like = [&](OpDef* x) {
                    if (!x) return false;
                    if (x->is_const() || x->const_data_size > 0) return true;
                    if (std::string(op_name(x)) != "Reshape" || x->inputs.empty()) return false;
                    OpDef* u = gp->get_op_at(x->inputs[0].src_id);
                    return u && (u->is_const() || u->const_data_size > 0) &&
                           shape_eq_mod_leading1(u->output_def, x->output_def);
                };
                OpDef* inner = cst_like(b) ? a : (cst_like(a) ? b : nullptr);
                if (inner && live(inner) && std::string(op_name(inner)) == "Eltwise_Binary" &&
                    inner->consumers.size() == 1) {
                    auto sp2 = unpack_scalar_params(inner->op_data);
                    const ScalarParam* q = scalar_get(sp2, "operation");
                    int64_t opc2 = q && q->is_numeric ? (int64_t)q->value_num : -1;
                    if (opc == opc2) { eb_pair++; eb_hist[opc]++; }
                }
            }
        }
    });
    std::fprintf(out, "[view-census] ss=%u reshape=%u transpose=%u\n", n_ss, n_rs, n_tp);
    std::fprintf(out, "[view-census] r1_ss_ss=%u r2_reshape_absorb=%u r3_tt_ident=%u ss_identity=%u\n",
                 r1, r2, r3, ss_ident);
    std::fprintf(out, "[view-census] eb_same_type_pair=%u", eb_pair);
    for (auto& [k, v] : eb_hist) std::fprintf(out, " op%lld=%u", (long long)k, v);
    std::fprintf(out, "\n");
}

} // namespace hnnx
