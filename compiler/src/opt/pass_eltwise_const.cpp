// pass_eltwise_const_fold —— 同型 const 二元链合并(PHASE_1)
// pass_eltwise_div_to_mul —— Div→Mul(独立注册, 默认关)
// =====================================================================
// 锚定在链**内层** EB 上(run_phase 改写后重访消费者 → 级联):
//   R1 同型: ADD∘ADD → ADD(c1+c2), MUL∘MUL → MUL(c1·c2) —— 外层 EB 的
//      const 换成合成 c', 数据输入改接内层数据源, 内层标 dead;
//   R2 Div→Mul: EB(DIVIDE)(x, c) → EB(MULTIPLY)(x, 1/c), 原地改锚自身。
//
// **R2 默认关的实证(2026-09-30, d0f1784)**: 0.8B 真图 host 位级一致
// (除数全 2 的幂), 但设备跑出末 4 行(28-31) NaN、P1-only 同图无 NaN——
// 设备 EB 的 DIVIDE 与 MULTIPLY 执行路径行为不同(f16 溢出口径),
// host 位级一致不保证设备一致。故 R2 拆名 eltwise_div_to_mul 默认关,
// 仅 --enable-pass eltwise_div_to_mul 显式开启。
//
// 枚举锚定(ops.cpp:161, QNN 2.48): ADD=0 DIVIDE=2 MULTIPLY=13 SUBTRACT=18。
// host int32 语义(两输入 Int32 → 整数运算)下 Div→Mul 非恒等(截断除法),
// 跳过; ADD/MUL 整数合并在 int64 精确。
//
// 数值口径: f64 运算后写回 f32 池(host 执行按 f32 位模式读池); Div→Mul
// 的 1/c 用 f32 先算(与设备除法路径容差一致) → 整链 ≤1 ULP f16(SC-002)。
// 保守门: 双 const 元素数相等(标量∘标量 / [N]∘[N], 不猜广播提升);
// c 非零非 denormal; 单消费者门(内层旁路不合并)。
#include "hnnx/opt/pass_eltwise_const.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/scalar_params.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hnnx {

namespace {

constexpr int OP_ADD = 0;
constexpr int OP_DIVIDE = 2;
constexpr int OP_MULTIPLY = 13;

const char* op_name(const OpDef* od) {
    return (od && od->name_tag && od->name_tag->name()) ? od->name_tag->name() : "";
}

bool live(const OpDef* od) {
    return od && od->is_enabled() && !od->is_dead() && !od->is_const();
}

bool is_const_node(const OpDef* od) {
    return od && (od->is_const() || od->const_data_size > 0);
}

int eb_operation(const OpDef* od) {
    auto sp = unpack_scalar_params(od->op_data);
    const ScalarParam* p = scalar_get(sp, "operation");
    if (!p) return OP_ADD;                                   // 缺省与 host/extract 同口径
    if (p->is_numeric) return (int)p->value_num;
    return (int)std::strtol(p->value_str.c_str(), nullptr, 10);
}

void set_eb_operation(OpDef* od, int code) {
    auto sp = unpack_scalar_params(od->op_data);
    bool changed = false;
    for (auto& p : sp) {
        if (p.name == "operation") {
            p.is_numeric = true; p.value_num = (double)code; p.value_str.clear();
            changed = true;
        }
    }
    if (!changed) {
        ScalarParam p;
        p.name = "operation"; p.is_numeric = true; p.value_num = (double)code;
        sp.push_back(std::move(p));
    }
    od->op_data = pack_scalar_params(sp);
}

uint64_t elems_of(const OutputDef& d) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < d.rank && i < 5; ++i) n *= d.dims[i];
    return n;
}

// const 池读 f32 数组(执行侧同口径: 池内 f32 位模式)
bool read_const_f32(const GraphPrepare* gp, const OpDef* c, std::vector<float>& out) {
    if (!c || c->const_data_size < 4 || (c->const_data_size % 4) != 0) return false;
    const auto& pool = gp->const_pool();
    if (c->const_data_offset + c->const_data_size > pool.size()) return false;
    size_t n = c->const_data_size / 4;
    out.resize(n);
    std::memcpy(out.data(), pool.data() + c->const_data_offset, n * 4);
    return true;
}

op_id_t make_f32_const(GraphPrepare* gp, const std::vector<float>& vals) {
    OutputDef od{};
    od.rank = 1;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    od.dims[0] = (uint32_t)vals.size();
    od.element_size = 4;
    return gp->append_const_node(0, od,
                                 reinterpret_cast<const uint8_t*>(vals.data()),
                                 vals.size() * 4);
}

// 前导 1 等价(numpy 右对齐广播下语义相同): [N] ~ [1,N] ~ [1,1,N]。
bool leading1_equivalent(const OutputDef& a, const OutputDef& b) {
    uint32_t ra = a.rank, rb = b.rank;
    while (ra > 0 && a.dims[0] == 1) ra--;
    while (rb > 0 && b.dims[0] == 1) rb--;
    if (ra != rb) return false;
    for (uint32_t i = 0; i < ra; ++i)
        if (a.dims[a.rank - ra + i] != b.dims[b.rank - rb + i]) return false;
    return true;
}

// const 解包: EB 的 const 输入常经 Reshape 包装(0.8B net.json 实测 36+ 处
// div∘mul 链的外层 mul const 走 Reshape)。仅前导 1 增删可解(广播不变;
// [N]→[N,1] 会改变右对齐语义, 不碰)。返回底层 const, reshape 出参可空。
OpDef* unwrap_const(GraphPrepare* gp, OpDef* in) {
    if (!in) return nullptr;
    if (is_const_node(in)) return in;
    if (std::string(op_name(in)) != "Reshape" || in->inputs.empty()) return nullptr;
    OpDef* u = gp->get_op_at(in->inputs[0].src_id);
    if (!u || !is_const_node(u)) return nullptr;
    if (!leading1_equivalent(u->output_def, in->output_def)) return nullptr;
    return u;
}

// EB(两输入)的 {数据源, const} 拆分(const 侧解包 Reshape 包装);
// 两侧都 const 返回 false(留给 const_fold)。
bool split_data_const(GraphPrepare* gp, OpDef* eb, OpDef*& data, OpDef*& cst) {
    if (eb->inputs.size() != 2) return false;
    OpDef* a = gp->get_op_at(eb->inputs[0].src_id);
    OpDef* b = gp->get_op_at(eb->inputs[1].src_id);
    if (!a || !b) return false;
    OpDef* ca = unwrap_const(gp, a), * cb = unwrap_const(gp, b);
    if ((ca != nullptr) == (cb != nullptr)) return false;    // 双 const / 双数据
    data = ca ? b : a;
    cst = ca ? ca : cb;
    return true;
}

// EB 的 const 侧输入引用(可能是 Reshape 包装节点)替换为新 const。
void rewire_const_side(GraphPrepare* gp, OpDef* eb, op_id_t new_id) {
    OpDef* nc = gp->get_op_at(new_id);
    if (!nc) return;
    for (auto& conn : eb->inputs) {
        OpDef* cur = gp->get_op_at(conn.src_id);
        if (!cur) continue;
        bool cst_side = is_const_node(cur) ||
                        (std::string(op_name(cur)) == "Reshape" && !cur->inputs.empty() &&
                         is_const_node(gp->get_op_at(cur->inputs[0].src_id)));
        if (!cst_side) continue;
        conn.src_id = new_id;
        conn.src_out_def = nc->output_def;
        cur->consumers.erase(std::remove(cur->consumers.begin(),
                                         cur->consumers.end(), eb->op_id),
                             cur->consumers.end());
    }
    nc->consumers.push_back(eb->op_id);
}

// ---------------------------------------------------------------------------
// R2 Div→Mul(原地改锚)
// ---------------------------------------------------------------------------
int try_div_to_mul(OpDef* op, GraphPrepare* gp) {
    if (eb_operation(op) != OP_DIVIDE) return 0;
    if (op->output_def.dtype == static_cast<uint32_t>(DType::Int32)) return 0;  // 整数截断除法非恒等
    OpDef* data = nullptr; OpDef* cst = nullptr;
    if (!split_data_const(gp, op, data, cst)) return 0;
    if (cst->output_def.dtype == static_cast<uint32_t>(DType::Int32)) return 0;

    std::vector<float> cv;
    if (!read_const_f32(gp, cst, cv)) return 0;
    for (float v : cv) {
        if (v == 0.0f || std::fabs(v) < FLT_MIN) return 0;    // 零/denormal 不碰
    }
    std::vector<float> inv(cv.size());
    for (size_t i = 0; i < cv.size(); ++i) inv[i] = 1.0f / cv[i];
    op_id_t nid = make_f32_const(gp, inv);
    if (!nid) return 0;
    rewire_const_side(gp, op, nid);
    set_eb_operation(op, OP_MULTIPLY);
    return 1;
}

// ---------------------------------------------------------------------------
// R1 同型合并(锚 = 内层, 改写外层)
// ---------------------------------------------------------------------------
int try_same_type_merge(OpDef* op, GraphPrepare* gp) {
    int opc = eb_operation(op);
    if (opc != OP_ADD && opc != OP_MULTIPLY) return 0;
    if (op->consumers.size() != 1) return 0;
    OpDef* outer = gp->get_op_at(op->consumers[0]);
    if (!live(outer) || std::string(op_name(outer)) != "Eltwise_Binary") return 0;
    if (eb_operation(outer) != opc) return 0;

    OpDef* y = nullptr; OpDef* c1 = nullptr;
    if (!split_data_const(gp, op, y, c1)) return 0;
    OpDef* x = nullptr; OpDef* c2 = nullptr;
    if (!split_data_const(gp, outer, x, c2)) return 0;
    if (x != op) return 0;                                    // outer 的数据源须是本锚
    if (elems_of(c1->output_def) != elems_of(c2->output_def)) return 0;

    bool is_int = c1->output_def.dtype == static_cast<uint32_t>(DType::Int32) &&
                  c2->output_def.dtype == static_cast<uint32_t>(DType::Int32);
    std::vector<float> v1, v2;
    if (!is_int) {
        if (!read_const_f32(gp, c1, v1) || !read_const_f32(gp, c2, v2)) return 0;
        if (v1.size() != v2.size()) return 0;
        std::vector<float> merged(v1.size());
        for (size_t i = 0; i < v1.size(); ++i) {
            double r = (opc == OP_ADD)
                     ? (double)v1[i] + (double)v2[i]
                     : (double)v1[i] * (double)v2[i];
            merged[i] = (float)r;                             // f64 运算单次舍入
        }
        op_id_t nid = make_f32_const(gp, merged);
        if (!nid) return 0;
        rewire_const_side(gp, outer, nid);
    } else {
        // 整数路径: 池位模式按 i32 读, int64 精确
        std::vector<float> fa, fb;
        if (!read_const_f32(gp, c1, fa) || !read_const_f32(gp, c2, fb)) return 0;
        std::vector<int32_t> a(fa.size()), b(fb.size());
        std::memcpy(a.data(), fa.data(), fa.size() * 4);
        std::memcpy(b.data(), fb.data(), fb.size() * 4);
        if (a.size() != b.size()) return 0;
        std::vector<int32_t> merged(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            merged[i] = (opc == OP_ADD) ? (int32_t)((int64_t)a[i] + (int64_t)b[i])
                                        : (int32_t)((int64_t)a[i] * (int64_t)b[i]);
        OutputDef od{};
        od.rank = 1;
        od.dtype = static_cast<uint32_t>(DType::Int32);
        od.dims[0] = (uint32_t)merged.size();
        od.element_size = 4;
        op_id_t nid = gp->append_const_node(0, od,
                                            reinterpret_cast<const uint8_t*>(merged.data()),
                                            merged.size() * 4);
        if (!nid) return 0;
        OpDef* nc = gp->get_op_at(nid);
        if (nc) nc->name_tag = string_tag_t::map_str("eltwise_const_fold_i32");
        rewire_const_side(gp, outer, nid);
    }

    // outer 的数据输入改接 y, 内层死
    for (auto& conn : outer->inputs) {
        if (conn.src_id == op->op_id) {
            conn.src_id = y->op_id;
            conn.src_out_def = y->output_def;
        }
    }
    y->consumers.erase(std::remove(y->consumers.begin(), y->consumers.end(), op->op_id),
                       y->consumers.end());
    y->consumers.push_back(outer->op_id);
    op->flags |= OP_DEAD;
    return 1;
}

bool matcher_merge(OpDef* op, GraphPrepare* gp) {
    if (!live(op) || std::string(op_name(op)) != "Eltwise_Binary") return false;
    if (op->inputs.size() != 2 || op->consumers.size() != 1) return false;
    OpDef* outer = gp->get_op_at(op->consumers[0]);
    if (!live(outer) || std::string(op_name(outer)) != "Eltwise_Binary") return false;
    int opc = eb_operation(op);
    return (opc == OP_ADD || opc == OP_MULTIPLY) && eb_operation(outer) == opc;
}

bool matcher_div(OpDef* op, GraphPrepare* gp) {
    if (!live(op) || std::string(op_name(op)) != "Eltwise_Binary") return false;
    if (op->inputs.size() != 2 || eb_operation(op) != OP_DIVIDE) return false;
    OpDef* d = nullptr; OpDef* c = nullptr;
    return split_data_const(gp, op, d, c) && op->output_def.dtype != (uint32_t)DType::Int32;
}

} // namespace

void register_pass_eltwise_const() {
    GraphPass p;
    p.name = "eltwise_const_fold";
    p.phase = PHASE_1;
    p.matcher = matcher_merge;
    p.rewrite = [](OpDef* op, GraphPrepare* gp) -> int {
        if (!live(op) || std::string(op_name(op)) != "Eltwise_Binary") return 0;
        return try_same_type_merge(op, gp);
    };
    PassManager::instance().add_pass(std::move(p));
}

void register_pass_eltwise_div_to_mul() {
    GraphPass p;
    p.name = "eltwise_div_to_mul";
    p.phase = PHASE_1;
    p.enabled = false;                 // 设备 f16 溢出口径实证 NaN, 默认关
    p.matcher = matcher_div;
    p.rewrite = [](OpDef* op, GraphPrepare* gp) -> int {
        if (!live(op) || std::string(op_name(op)) != "Eltwise_Binary") return 0;
        return try_div_to_mul(op, gp);
    };
    PassManager::instance().add_pass(std::move(p));
}

} // namespace hnnx
