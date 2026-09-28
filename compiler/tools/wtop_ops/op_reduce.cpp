// op_reduce.cpp — Reduce 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"
#include "hnnx/ir/scalar_params.hpp"

using namespace wtop;

namespace {

static uint32_t g_reduce_fallback = 0;

static int op_reduce(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    const OpDef* src_r = gp.get_op_at(od->inputs[0].src_id);
    const uint32_t irank = src_r ? src_r->output_def.rank : 0;
    const uint64_t* idims = src_r ? src_r->output_def.dims : nullptr;

    /* axis/subtype 解析序:
     * ① serialized_extra 有效 (axis≥0 且 reduce_type≤3) → 直接用
     *    (graph_prepare.extract_axis_extra 修复后, 新鲜 tagged.bin 走这里);
     * ② 陈旧 tagged.bin (axes tensor_param 丢失, extra=0xFFFFFFFF 哨兵) →
     *    绝对轴 3 兜底: 0.8B 全图 1170 个 Reduce 实证 — host dump 尺寸 1170/1170
     *    吻合 + op126 值级对拍 cos=1.0 (net.json axes=[3] 全系 ReduceSum)。
     *    "缺 axes=全轴归约"的 host 通用语义在此图不成立 (host 实证单轴)。 */
    int32_t axis = -1;
    uint32_t subtype = 0;
    ExtraAxis e{};
    bool extra_ok = false;
    if (od->serialized_extra.size() >= sizeof(e)) {
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
        extra_ok = (e.axis >= 0 && e.axis < 8 && e.reduce_type <= 3);
    }
    if (extra_ok) {
        axis = e.axis;
        subtype = e.reduce_type;
    } else {
        axis = (irank >= 4) ? 3 : (irank ? (int32_t)irank - 1 : 0);
        subtype = 0;
        if (!g_reduce_fallback)
            std::atexit([] {
                if (g_reduce_fallback)
                    std::fprintf(stderr,
                                 "warn: %u 个 Reduce 用绝对轴3兜底 (陈旧 tagged 缺 axes const)\n",
                                 g_reduce_fallback);
            });
        g_reduce_fallback++;
    }
    /* reduce_type: scalar 在 op_data 里时以它为准 (QNN 串直读) */
    {
        auto sp = unpack_scalar_params(od->op_data);
        const ScalarParam* prt = scalar_get(sp, "reduce_type");
        std::string rt = prt ? prt->value_str : "ReduceSum";
        if (rt == "ReduceSum" || rt == "SUM" || rt.empty()) subtype = 0;
        else if (rt == "ReduceMean" || rt == "AVG")   subtype = 1;
        else if (rt == "ReduceMax")                   subtype = 2;
        else if (rt == "ReduceMin")                   subtype = 3;
        else {
            std::fprintf(stderr, "error: Reduce op %llu reduce_type=%s 无映射\n",
                         (unsigned long long)od->op_id, rt.c_str());
            return 6;
        }
    }
    if (axis < 0 || (irank && axis >= (int32_t)irank)) {
        std::fprintf(stderr, "error: Reduce op %llu axis=%d 越界 (irank=%u, 拒发)\n",
                     (unsigned long long)od->op_id, (int)axis, irank);
        return 6;
    }

    /* dims 窗口: rank≤4 直推 (axis 绝对); rank>4 归一化为 axis=0 + 轴起后缀,
     * leading 维由 exec 的 outer=n/(L·inner) 展平吸收 — 单轴归约值恒正确 */
    uint32_t blob_axis = (uint32_t)axis;
    uint32_t dd[4] = {1u, 1u, 1u, 1u};
    if (irank <= 4) {
        for (uint32_t i = 0; i < 4; i++)
            dd[i] = (idims && i < irank) ? (uint32_t)idims[i] : 1u;
    } else {
        if (irank - (uint32_t)axis > 4) {
            std::fprintf(stderr, "error: Reduce op %llu axis=%d 后随维超窗 (irank=%u)\n",
                         (unsigned long long)od->op_id, (int)axis, irank);
            return 6;
        }
        for (uint32_t i = 0; i < 4; i++)
            dd[i] = (idims && (uint32_t)axis + i < irank) ? (uint32_t)idims[axis + i] : 1u;
        blob_axis = 0;
    }

    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint64_t n_in = src_r ? elems_of(src_r) : elems_of(od);
    em.add_op(OP_REDUCE_F16, {x_t, out_t, (uint32_t)n_in, blob_axis, subtype,
                              dd[0], dd[1], dd[2], dd[3]});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_reduce{
    {"Reduce", op_reduce},
};
