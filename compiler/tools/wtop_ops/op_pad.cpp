// op_pad.cpp — Pad 发射(算子三件套·host 发射侧; opcode 29)
// 恒等拷贝冒充(旧 op_copy_sem)在 GDN 是错的: [1,16,32]→[1,16,64] 前 pad 32,
// 恒等只拷 512 元素, 后半 512 读陈旧池字节 → exp 溢出 → 全 -inf。
// 真语义: out 全填 pad 值, 再按前 pad 偏移拷入 in。
#include "wtop_ops.hpp"
#include "hnnx/ir/scalar_params.hpp"

using namespace wtop;

namespace {

static int op_pad(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    const OpDef* src = (od->inputs.size() > 0) ? gp.get_op_at(od->inputs[0].src_id) : nullptr;
    if (!src) { std::fprintf(stderr, "error: pad src missing\n"); return 2; }
    uint32_t rk = od->output_def.rank;
    if (rk > 4) rk = 4;
    uint32_t id_[4] = {1, 1, 1, 1}, od_[4] = {1, 1, 1, 1};
    for (uint32_t i = 0; i < rk; i++) od_[i] = od->output_def.dims[i];
    /* 输入形状右对齐输出秩 (参考实现同款; 前导 1 填充) */
    {
        uint32_t srk = src->output_def.rank;
        uint32_t off = rk > srk ? rk - srk : 0;
        for (uint32_t i = 0; i < srk && off + i < 4; i++)
            id_[off + i] = src->output_def.dims[i];
    }
    /* pad_amount = inputs[1] const (int32 [n_axes,2] begin/end)。
     * pads 前向轴序对齐输出秩末段: pb[rank-n_axes + a] = begin[a] */
    uint32_t pb[4] = {0, 0, 0, 0};
    if (od->inputs.size() > 1) {
        const OpDef* pc = gp.get_op_at(od->inputs[1].src_id);
        if (pc && pc->const_data_size >= 8 && pc->const_data_size % 8 == 0) {
            const int32_t* p = reinterpret_cast<const int32_t*>(
                gp.const_pool().data() + pc->const_data_offset);
            uint32_t n_axes = (uint32_t)(pc->const_data_size / 8);
            uint32_t ax_off = (n_axes <= rk) ? rk - n_axes : 0;
            for (uint32_t a = 0; a < n_axes && ax_off + a < 4; a++)
                pb[ax_off + a] = (uint32_t)p[a * 2];
        }
    }
    uint16_t padv = 0;
    /* pad_constant_value 从 op_data 解 scalar params (默认 0; GDN 全 0) */
    {
        auto sp = hnnx::unpack_scalar_params(od->op_data);
        const auto* pv = hnnx::scalar_get(sp, "pad_constant_value");
        if (pv) {
            float v = (float)(pv->is_numeric ? pv->value_num
                                             : std::strtod(pv->value_str.c_str(), nullptr));
            padv = f32_to_f16_rne(v);
        }
    }
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_PAD_F16, {x_t, out_t, rk,
                           id_[0], id_[1], id_[2], id_[3],
                           od_[0], od_[1], od_[2], od_[3],
                           pb[0], pb[1], pb[2], pb[3], (uint32_t)padv});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_pad{
    {"Pad", op_pad},
};
