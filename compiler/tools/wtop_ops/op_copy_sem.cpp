// op_copy_sem.cpp — Reshape / Pad / ScatterNd / Cast 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

/* graph_prepare.cpp 匿名命名空间内的 find_param_const/read_param_u32 够不着,
 * 此处内置等价小助手 (按名子串匹配 inputs + tensor_param_ids)。 */
static const OpDef* cs_find_param(GraphPrepare& gp, const OpDef* od, const char* key) {
    for (const auto& c : od->inputs) {
        const OpDef* s = gp.get_op_at(c.src_id);
        if (!s || !s->name_tag) continue;
        const char* n = s->name_tag->name();
        if (n && std::strstr(n, key)) return s;
    }
    for (op_id_t tpid : od->tensor_param_ids) {
        const OpDef* s = gp.get_op_at(tpid);
        if (!s || !s->name_tag) continue;
        const char* n = s->name_tag->name();
        if (n && std::strstr(n, key)) return s;
    }
    return nullptr;
}

static std::vector<uint32_t> cs_read_u32(GraphPrepare& gp, const OpDef* p, size_t max_n) {
    std::vector<uint32_t> v;
    if (!p || p->const_data_size < 4) return v;
    size_t n = std::min(max_n, (size_t)p->const_data_size / 4);
    const auto& pool = gp.const_pool();
    if (p->const_data_offset == 0 || p->const_data_offset + p->const_data_size > pool.size())
        return v;
    v.resize(n);
    std::memcpy(v.data(), pool.data() + p->const_data_offset, n * 4);
    return v;
}

static int op_copy_sem(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    // 数据搬运/常量填充/状态更新语义 → 恒等(设备 M4 数值门兜底)
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od), 14});
    return 0;
}

/* Pad 真物化 (M4.5): 恒等拷贝在 temp 复用下把 pad 区留成垃圾
 * (0.8B GDN op43: pad 区被 CumSum 累加 → 门控全链毒化, logits cos 0.15)。
 * 支持形态: 单维增长 + 常量填充 (0.8B 全部 Pad: pad_amount 每维 [before,after]
 * 对, 实测全为 END-pad); 其余形态硬错误 (不再静默降级)。 */
static int op_pad(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    const OpDef* prod = od->inputs.empty() ? nullptr : gp.get_op_at(od->inputs[0].src_id);
    if (!prod || prod->output_def.rank == 0 || od->output_def.rank != prod->output_def.rank) {
        std::fprintf(stderr, "error: Pad op %llu 输入/输出 rank 不支持\n", (unsigned long long)od->op_id);
        return 1;
    }
    uint32_t r = od->output_def.rank;
    uint64_t in_d[5], out_d[5];
    for (uint32_t i = 0; i < r && i < 5; i++) {
        in_d[i] = prod->output_def.dims[i];
        out_d[i] = od->output_def.dims[i];
    }
    // 找唯一增长维
    int g = -1;
    for (uint32_t i = 0; i < r; i++) {
        if (in_d[i] == out_d[i]) continue;
        if (g >= 0 || out_d[i] < in_d[i]) {
            std::fprintf(stderr, "error: Pad op %llu 多维变化/缩维不支持\n", (unsigned long long)od->op_id);
            return 1;
        }
        g = (int)i;
    }
    if (g < 0) {  // 纯恒等: 退化为拷贝
        uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
        uint32_t out_t = em.fresh_temp(gp, od->op_id);
        em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od), 14});
        return 0;
    }
    // pad_amount pairs (右对齐到张量维): 校验 before/after 与形状一致
    uint64_t pad_before = 0;
    if (const OpDef* pa = cs_find_param(gp, const_cast<OpDef*>(od), "pad")) {
        auto v = cs_read_u32(gp, pa, 10);
        size_t np = v.size() / 2;
        if (np >= 1 && np <= (size_t)r) {
            for (uint32_t i = 0; i < r; i++) {
                long pi = (long)np - (long)r + (long)i;   // dim i 对应 pair
                uint32_t before = 0, after = 0;
                if (pi >= 0) { before = v[2 * pi]; after = v[2 * pi + 1]; }
                if ((int)i == g) {
                    if ((uint64_t)before + in_d[i] > out_d[i] ||
                        (uint64_t)before + in_d[i] + after != out_d[i]) {
                        std::fprintf(stderr, "error: Pad op %llu pad_amount 与形状不符\n", (unsigned long long)od->op_id);
                        return 1;
                    }
                    // pad_before 以元素计(乘尾维)
                    pad_before = before;
                    for (uint32_t j = (uint32_t)g + 1; j < r; j++) pad_before *= out_d[j];
                } else if (before != 0 || after != 0) {
                    std::fprintf(stderr, "error: Pad op %llu 非增长维有 pad\n", (unsigned long long)od->op_id);
                    return 1;
                }
            }
        }
    }
    // 常量值: ExtraPad(scheme+constant_value), 缺省 0
    double cv = 0.0;
    if (od->serialized_extra.size() >= sizeof(ExtraPad)) {
        ExtraPad e{};
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
        cv = e.constant_value;
    }
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint64_t outer = 1, in_inner = 1, out_inner = 1;
    for (int i = 0; i < g; i++) outer *= in_d[i];
    for (uint32_t i = (uint32_t)g; i < r; i++) { in_inner *= in_d[i]; out_inner *= out_d[i]; }
    union { float f; uint32_t u; } cvf;
    cvf.f = (float)cv;
    uint16_t pv16 = f32_to_f16_rne(cvf.f);
    uint32_t pv_bits;
    std::memcpy(&pv_bits, &pv16, 2);
    pv_bits &= 0xFFFFu;
    em.add_op(OP_PAD_F16, {x_t, out_t, (uint32_t)outer, (uint32_t)in_inner,
                           (uint32_t)out_inner, (uint32_t)pad_before, pv_bits});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_copy_sem{
    {"Reshape", op_copy_sem},
    {"Pad", op_pad},
    /* A3②: ScatterNd 已迁出 (op_scatter_nd.cpp 真语义发射, L0 cos 0.102 死刑判决) */
    {"Cast", op_copy_sem},
};
