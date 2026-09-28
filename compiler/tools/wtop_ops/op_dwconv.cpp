// op_dwconv.cpp — DepthWiseConv2d 发射(算子三件套·host 发射侧; SSM 一维卷积)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_dwconv(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraDepthwiseConv e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
    /* kw 以权重实形重算 (QNN 布局 [1,k,1,C], 末维=ch): serialized_extra 是
     * hnnx_compile 期冻结的, 重跑 wtop_emit 不会刷新 — 0.8B GDN 旧 tagged.bin
     * 全 18 层 kw=1 (真核 4), 与 graph_prepare.cpp 提取器修复同式。 */
    if (w && w->output_def.rank >= 2) {
        uint32_t r = w->output_def.rank;
        uint64_t we = 1;
        for (uint32_t i = 0; i < r && i < 5; i++) we *= w->output_def.dims[i];
        uint64_t wch = w->output_def.dims[r - 1];
        uint32_t kw = wch ? (uint32_t)(we / wch) : 0;
        if (kw && kw != e.kw) {
            std::fprintf(stderr, "[op_dwconv] op %llu: kw %u→%u (权重实形重算)\n",
                         (unsigned long long)od->op_id, e.kw, kw);
            e.kw = kw;
        }
    }
    uint32_t w_s = w ? em.ensure_weight_slot(gp, w, wslots, od->grouping) : em.dummy_slot_id;
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint64_t ne = elems_of(od);
    uint32_t seq = 1, ch = (uint32_t)ne;
    if (od->output_def.rank >= 2) {
        seq = od->output_def.dims[od->output_def.rank - 2];
        ch = od->output_def.dims[od->output_def.rank - 1];
    }
    em.add_op(OP_CONV1D_SSM_F16, {x_t, w_s, out_t, seq, ch, e.kw});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_dwconv{
    {"DepthWiseConv2d", op_dwconv},
};
