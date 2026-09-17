// pass_const_fold —— 常量折叠(求值版, PHASE_0)
// =====================================================================
// matcher: op 名在白名单内 && 输入全 const(OP_CONST 且 const_data_size>0)。
// rewrite: 输入从 const_pool_ 拓宽 f32(按 element_size, 与 execute_host 同款)
//   → TypicalOp::execute host 参考核求值 → 窄化回输出 dtype → fold_op_to_const。
// 铁律: 绝不只标 OP_CONST 不存值(见 pass_const_fold.hpp); 求值失败不折。
// 白名单 v1 = execute 分派已实现且经 M3c 逐 op 对拍的类型(layer_0 的
// ScatterNd 索引链靠 Gather+StridedSlice+Eltwise 组合折)。
#include "hnnx/opt/pass_const_fold.hpp"
#include "hnnx/opt/pass_manager.hpp"
#include "hnnx/opt/optimization_passes.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_registry.hpp"  // → hexagon_nn_env.hpp 的 OpIoPtrs(构建版, 勿引 op_io_ptrs.hpp 字节级版)
#include "hnnx/ops/ops.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace hnnx {

namespace {

// f32 → f16(round-to-nearest-even; 次正规/Inf/NaN 保真; 与设备 f16 写入语义一致)
static uint16_t narrow_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t exp = (u >> 23) & 0xFFu;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp == 0xFF) {                       // Inf/NaN
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    if (exp > 127 + 15) {                    // 溢出 → Inf
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp >= 127 - 14 + 1) {               // 正规
        uint32_t e = exp - 127 + 15;
        uint32_t m = mant >> 13;
        // round-to-nearest-even
        uint32_t rest = mant & 0x1FFFu;
        if (rest > 0x1000u || (rest == 0x1000u && (m & 1u))) {
            m++;
            if (m == 0x400u) { m = 0; e++; }
        }
        return (uint16_t)(sign | (e << 10) | m);
    }
    // 次正规
    int shift = 127 - 14 - (int)exp + 1;     // 隐含位下移量
    uint32_t m = (mant | 0x800000u) >> (13 + shift);
    uint32_t rest = (mant | 0x800000u) & ((1u << (13 + shift)) - 1);
    uint32_t half = 1u << (12 + shift);
    if (rest > half || (rest == half && (m & 1u))) m++;
    if (m >= 0x400u) m = 0x3FFu;             // 舍入进位进正规(容差级, 不推 e)
    return (uint16_t)(sign | m);
}

// 常量折叠白名单(execute 分派已实现; 不含 Conv/MatMul/FC/Norm 族 ——
// 权重路径的 op 不折, 保持既有槽约定)
bool in_whitelist(const std::string& nm) {
    static const std::vector<std::string> wl = {
        "Add", "Sub", "Mul", "Div",
        "Eltwise_Binary", "Eltwise_Unary", "ElementWiseNeuron", "Eltwise_Ternary",
        "Transpose", "StridedSlice", "Reshape", "Flatten", "Pad",
        "Concat", "Reduce", "Gather", "CumulativeSum", "ScatterNd",
    };
    return std::find(wl.begin(), wl.end(), nm) != wl.end();
}

// const 池 → f32 拓宽(与 GraphPrepare::execute_host 的物化同款)
static std::vector<float> widen_const(const GraphPrepare* gp, const OpDef* pc) {
    std::vector<float> buf;
    if (!pc || pc->const_data_size == 0) return buf;
    size_t es = pc->output_def.element_size;
    if (es == 0 || es > 4) es = 4;
    size_t elem_n = pc->const_data_size / es;
    buf.resize(elem_n, 0.0f);
    bool in_pool = pc->const_data_offset + pc->const_data_size <= gp->const_pool().size();
    const uint8_t* src = in_pool ? gp->const_pool().data() + pc->const_data_offset : nullptr;
    if (!src) return buf;
    if (es == 4) {
        std::memcpy(buf.data(), src, pc->const_data_size);
    } else if (es == 2) {
        for (size_t i = 0; i < elem_n; i++) {
            uint16_t h;
            std::memcpy(&h, src + i * 2, 2);
            uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, u;
            if (exp == 0) {
                if (mant == 0) u = sign;
                else {
                    int e = -1;
                    while (!(mant & 0x400)) { mant <<= 1; e--; }
                    u = sign | ((uint32_t)(114 + e) << 23) | ((mant & 0x3FF) << 13);
                }
            } else if (exp == 31) u = sign | 0x7F800000u | (mant << 13);
            else u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            float f;
            std::memcpy(&f, &u, 4);
            buf[i] = f;
        }
    } else {  // es == 1: bool/uint8 → float
        for (size_t i = 0; i < elem_n; i++) buf[i] = (float)src[i];
    }
    return buf;
}

// matcher: 白名单 + 输入全 const(有值)
bool matcher(OpDef* op, GraphPrepare* gp) {
    if (!op || !op->is_enabled() || op->is_dead() || op->is_const()) return false;
    if (!op->name_tag || !op->name_tag->name()) return false;
    std::string nm = op->name_tag->name();
    if (nm == "Input" || nm == "Output") return false;
    if (!in_whitelist(nm)) return false;
    if (op->inputs.empty()) return false;
    for (const auto& conn : op->inputs) {
        const OpDef* src = gp->get_op_at(conn.src_id);
        if (!src || !src->is_const() || src->const_data_size == 0) return false;
    }
    return true;
}

// rewrite: host 参考核求值 → 窄化 → fold_op_to_const
int rewrite(OpDef* op, GraphPrepare* gp) {
    // 1. 输入拓宽 f32(data 输入按各自 element_size; tensor_param 恒 4B 直拷)
    std::vector<std::vector<float>> widened;
    std::vector<const uint8_t*> in_bufs;
    std::vector<OutputDef> in_ods;
    widened.reserve(op->inputs.size() + op->tensor_param_ids.size());
    for (const auto& conn : op->inputs) {
        const OpDef* src = gp->get_op_at(conn.src_id);
        if (!src) return 0;  // 保守: 求值环境不全不折
        widened.push_back(widen_const(gp, src));
        in_bufs.push_back(reinterpret_cast<const uint8_t*>(widened.back().data()));
        in_ods.push_back(src->output_def);
    }
    for (op_id_t tpid : op->tensor_param_ids) {
        const OpDef* pc = gp->get_op_at(tpid);
        if (!pc || pc->const_data_size == 0) continue;
        std::vector<float> tbuf(pc->const_data_size / sizeof(float), 0.0f);
        if (pc->const_data_offset + pc->const_data_size <= gp->const_pool().size())
            std::memcpy(tbuf.data(), gp->const_pool().data() + pc->const_data_offset,
                        pc->const_data_size);
        widened.push_back(std::move(tbuf));
        in_bufs.push_back(reinterpret_cast<const uint8_t*>(widened.back().data()));
        in_ods.push_back(pc->output_def);
    }

    // 2. 输出 f32 缓冲(dims 乘积)
    size_t n = 1;
    for (uint32_t i = 0; i < op->output_def.rank && i < 5; ++i)
        n *= static_cast<size_t>(op->output_def.dims[i]);
    if (n == 0) n = 1;
    std::vector<float> out(n, 0.0f);

    // 3. host 参考核求值(与 execute_host 同款工厂 + 执行)
    OpIoPtrs io{};
    io.graph_prepare = gp;
    io.opdef_ptr = op;
    auto gen = OpRegistry::instance().generate(io, op->op_id);
    auto* typ = dynamic_cast<TypicalOp*>(gen.get());
    if (!typ) return 0;  // 白名单外/无构造 → 不折
    typ->execute(in_bufs, reinterpret_cast<uint8_t*>(out.data()),
                 op->output_def, in_ods);

    // 4. 窄化回输出 dtype(element_size: 4=f32 直拷, 2=f16, 1=bool)
    size_t es = op->output_def.element_size;
    if (es == 0 || es > 4) es = 4;
    std::vector<uint8_t> bytes(n * es, 0);
    if (es == 4) {
        std::memcpy(bytes.data(), out.data(), n * 4);
    } else if (es == 2) {
        for (size_t i = 0; i < n; i++) {
            uint16_t h = narrow_f16(out[i]);
            std::memcpy(bytes.data() + i * 2, &h, 2);
        }
    } else {
        for (size_t i = 0; i < n; i++) bytes[i] = (out[i] != 0.0f) ? 1 : 0;
    }

    return gp->fold_op_to_const(op, bytes.data(), bytes.size()) ? 1 : 0;
}

} // namespace

void register_pass_const_fold() {
    GraphPass p;
    p.name = "const_fold";
    p.phase = PHASE_0;  // 3000: 常量折叠
    p.matcher = matcher;
    p.rewrite = rewrite;
    PassManager::instance().add_pass(std::move(p));
}

} // namespace hnnx
