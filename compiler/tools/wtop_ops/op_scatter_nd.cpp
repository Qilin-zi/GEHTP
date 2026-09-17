// op_scatter_nd.cpp — ScatterNd 真语义发射 (A3②; 算子三件套·host 发射侧)
// =====================================================================
// A3-① 判决 (docs/A3_COPY_SEM_AUDIT.md): ScatterNd 恒等拷贝 = L0 cos 0.102 死刑,
// 必须真语义上设备。发射 OP_SCATTER_ND_F16 (opcode 28, §7 预登记):
//   args = [data_ref, idx_s, upd_ref, out_t, n_out, rank, d0..d4, K, n_idx, block]
//   data/updates: f16 (ref 可 temp 或 0x8000|slot); idx: i32 (const → slot)
//   K=indices 末维 (≤5), n_idx=前导维积, block=updates_elems/n_idx
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_scatter_nd(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    if (od->inputs.size() < 3) {
        std::fprintf(stderr, "[op_scatter_nd] %s: inputs<3, 拒发 (不许静默恒等)\n", nm.c_str());
        return -1;
    }
    const OpDef* idxp = gp.get_op_at(od->inputs[1].src_id);
    const OpDef* updp = gp.get_op_at(od->inputs[2].src_id);

    uint32_t data_ref = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t idx_s   = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t upd_ref = em.src_ref(od->inputs[2], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t   = em.fresh_temp(gp, od->op_id);

    const uint32_t n_out = (uint32_t)elems_of(od);
    uint32_t rank = od->output_def.rank;
    if (rank > 5) {
        std::fprintf(stderr, "[op_scatter_nd] %s: rank %u>5, 拒发\n", nm.c_str(), rank);
        return -1;
    }
    uint32_t d[5] = {1, 1, 1, 1, 1};
    for (uint32_t i = 0; i < rank; ++i) d[i] = od->output_def.dims[i];

    /* K = indices 末维; n_idx = 前导维积; block = updates 元素/n_idx (与
     * host execute ScatterNd (ops.cpp) 及设备 exec 三方同口径) */
    uint32_t K = 1;
    uint64_t n_idx = 1;
    if (idxp && idxp->output_def.rank >= 1) {
        K = idxp->output_def.dims[idxp->output_def.rank - 1];
        for (uint32_t i = 0; i + 1 < idxp->output_def.rank; ++i)
            n_idx *= idxp->output_def.dims[i];
    }
    if (K == 0 || K > 5 || n_idx == 0) {
        std::fprintf(stderr, "[op_scatter_nd] %s: K=%u n_idx=%llu 非法, 拒发\n",
                     nm.c_str(), K, (unsigned long long)n_idx);
        return -1;
    }
    uint64_t upd_elems = 1;
    if (updp)
        for (uint32_t i = 0; i < updp->output_def.rank && i < 5; ++i)
            upd_elems *= updp->output_def.dims[i];
    const uint32_t block = (uint32_t)(upd_elems / n_idx);

    em.add_op(OP_SCATTER_ND_F16,
              {data_ref, idx_s, upd_ref, out_t, n_out, rank,
               d[0], d[1], d[2], d[3], d[4], K, (uint32_t)n_idx, block});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_scatter_nd{
    {"ScatterNd", op_scatter_nd},
    {"ScatterND", op_scatter_nd},
};
