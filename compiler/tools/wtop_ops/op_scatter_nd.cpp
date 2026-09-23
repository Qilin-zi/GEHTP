// op_scatter_nd.cpp — ScatterNd 真语义发射(算子三件套·host 发射侧; A3 暗雷收口)
//
// 背景: ScatterNd 此前走 op_copy_sem 恒等拷贝(OP_UNARY_F16 passthrough),
//   attn_iter 在设备上退化为 attn_pre(未真正迭代), 0.8B 1154 处全链数值错。
// 语义: out = data 拷贝; 对 n_idx 个坐标(每坐标 K 维, indices int32)写 upd[e]。
//   GDN attn_iter: data [16,64,64] f16, indices [n_idx, K≤5] int32 常量,
//   updates [n_idx] f16, block=1 (每步写一个衰减更新点)。
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_scatter_nd(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    if (od->inputs.size() < 3) {
        std::fprintf(stderr, "error: ScatterNd inputs<3 (op %llu)\n", (unsigned long long)od->op_id);
        return 2;
    }
    /* data(输入0)可为运行时 temp 或 const; indices(输入1)应为 int32 const 槽;
     * updates(输入2)为运行时 temp(衰减更新值)。 */
    uint32_t data_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t upd_t  = em.src_ref(od->inputs[2], gp.get_input_node_id(), gp, wslots);

    /* indices: const int32, 直接读 const_pool 落成 int32 槽(不转 f16)。
     * dims [.., K], n_idx = 前导维积。 */
    const OpDef* idx_od = gp.get_op_at(od->inputs[1].src_id);
    uint32_t K = 1, n_idx = 1;
    if (idx_od && idx_od->output_def.rank >= 1) {
        K = idx_od->output_def.dims[idx_od->output_def.rank - 1];
        for (uint32_t d = 0; d + 1 < idx_od->output_def.rank; ++d)
            n_idx *= idx_od->output_def.dims[d];
    }
    if (K == 0 || K > 5) K = 1;
    if (n_idx == 0) n_idx = 1;
    uint32_t idx_s = 0;
    if (idx_od && idx_od->const_data_size > 0) {
        const auto& pool = gp.const_pool();
        size_t off = idx_od->const_data_offset;
        size_t sz = idx_od->const_data_size;
        if (off + sz <= pool.size()) {
            /* indices int32 常量, 按 4B 直存为 int32 槽(不转 f16)。
             * 路由与 ensure_weight_slot 一致: 外置模式进 ext_weight_area。 */
            uint32_t len = (uint32_t)sz, cnt = (uint32_t)(sz / 4);
            idx_s = em.ext_weights ? em.add_ext_slot(len, cnt, pool.data() + off)
                                   : em.add_slot(len, cnt, pool.data() + off);
        }
    }
    if (idx_s == 0) {
        /* 非常量 indices(暂不支持) → 共享哑槽占位, 数值门兜底 */
        if (em.dummy_slot_id == 0) {
            std::vector<uint8_t> zeros(128, 0);
            em.dummy_slot_id = em.add_slot(128, 64, zeros.data());
        }
        idx_s = em.dummy_slot_id;
        std::fprintf(stderr, "warn: ScatterNd non-const indices (op %llu), 哑槽兜底\n",
                     (unsigned long long)od->op_id);
    }

    uint32_t out_t = em.fresh_temp(gp, od->op_id);

    /* data 全形 od[0..4]; od[0..K-1] = 最后 K 维 (索引参照),
     * od[K..r-1] = 前导批维 (输出总元素积)。 */
    const OpDef* data_od = gp.get_op_at(od->inputs[0].src_id);
    uint32_t odims[5] = {1, 1, 1, 1, 1};
    if (data_od) {
        uint32_t r = data_od->output_def.rank;
        uint32_t off = (r > K) ? (r - K) : 0;
        for (uint32_t k = 0; k < K && off + k < r; k++)
            odims[k] = data_od->output_def.dims[off + k];
        for (uint32_t k = 0; k < off && K + k < 5; k++)
            odims[K + k] = data_od->output_def.dims[k];
    }

    em.add_op(OP_SCATTER_ND_F16,
              {data_t, idx_s, upd_t, out_t, n_idx, K,
               odims[0], odims[1], odims[2], odims[3], odims[4], 0u});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_scatter_nd{
    {"ScatterNd", op_scatter_nd},
    {"ScatterND", op_scatter_nd},
};
