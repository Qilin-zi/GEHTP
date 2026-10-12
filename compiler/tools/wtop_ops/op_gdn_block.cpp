// op_gdn_block.cpp — GdnBlock 发射(算子三件套·host 发射侧; 009 块融合)
// 4 输入 [xqkv, conv_w, beta, g] → opcode 37 arity 5 [xqkv_t, conv_w_s,
// beta_t, g_t, y_t]。几何 (T/ntok/C/H/D/d_conv/chunk/vtcm_budget_bytes/
// state_idx/contract_ver) 走 serialized_extra (ExtraGdnBlock), 不占 args
// (args≤16 硬约束下几何外置 — 契约 oplist_parse.h OP_GDN_BLOCK_F16 注释)。
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_gdn_block(Emitter& em, GraphPrepare& gp, const OpDef* od,
                        std::map<uint64_t, uint32_t>& wslots,
                        const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    if (od->inputs.size() < 4) {
        std::fprintf(stderr, "error: GdnBlock 输入 %zu < 4 (契约 [xqkv,conv_w,beta,g])\n",
                     od->inputs.size());
        return 4;
    }
    // 几何校验: ExtraGdnBlock 必须已由 extract_gdn_block_extra 落 serialized_extra。
    // memcpy 供主循环 carve push 读 vtcm_budget_bytes (FR-004 闭式单源)。
    ExtraGdnBlock eg{};
    if (od->serialized_extra.size() < sizeof(eg)) {
        std::fprintf(stderr, "error: GdnBlock extra 缺失/过短 (%zu < %zu)\n",
                     od->serialized_extra.size(), sizeof(eg));
        return 4;
    }
    std::memcpy(&eg, od->serialized_extra.data(), sizeof(eg));

    // 4 输入引用: xqkv/beta/g 在真实图为 temp (in_proj 出值 / Sigmoid_1 /
    // Softplus), conv_w 为权重 const → 0x8000|slot。合成单测里 beta/g 也是
    // const → src_ref 统一按 0x8000|slot 落 (契约四套编码, 见 oplist_parse.h)。
    uint32_t xqkv_t   = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t conv_w_s = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t beta_t   = em.src_ref(od->inputs[2], gp.get_input_node_id(), gp, wslots);
    uint32_t g_t      = em.src_ref(od->inputs[3], gp.get_input_node_id(), gp, wslots);
    uint32_t y_t      = em.fresh_temp(gp, od->op_id);

    em.add_op(OP_GDN_BLOCK_F16, {xqkv_t, conv_w_s, beta_t, g_t, y_t});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_gdn_block{
    {"GdnBlock", op_gdn_block},
};
