// op_eltwise_binary.cpp — Eltwise_Binary 发射(算子三件套·host 发射侧;
// numpy 广播物化 + ADD/BINARY 编码)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_eltwise_binary(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    // operation 从 serialized_extra 读(0=ADD 1=SUB 2=MUL 3=DIV)
    uint32_t subtype = 0;
    if (od->serialized_extra.size() >= sizeof(ExtraEltwise)) {
        ExtraEltwise e;
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
        subtype = e.operation;
    }
    uint32_t a_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t b_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint64_t n = elems_of(od);
    /* numpy 广播: b 元素 < 输出时先物化为全尺寸 temp(带形状的
     * 逐轴广播), 保持 ADD/BINARY 纯元素语义 */
    const OpDef* bb = gp.get_op_at(od->inputs[1].src_id);
    uint64_t b_elems = bb ? elems_of(bb) : n;
    if (getenv("GEHTP_BDIAG")) {
        const OpDef* aa = gp.get_op_at(od->inputs[0].src_id);
        std::fprintf(stderr, "[bdiag] op %llu %s: od.rank=%u od.dims=",
                     (unsigned long long)od->op_id,
                     od->name_tag && od->name_tag->name() ? od->name_tag->name() : "?",
                     od->output_def.rank);
        for (uint32_t i = 0; i < od->output_def.rank && i < 8; i++)
            std::fprintf(stderr, "%u ", od->output_def.dims[i]);
        std::fprintf(stderr, "| a(%s).rank=%u dims=",
                     aa && aa->name_tag && aa->name_tag->name() ? aa->name_tag->name() : "?",
                     aa ? aa->output_def.rank : 0);
        if (aa) for (uint32_t i = 0; i < aa->output_def.rank && i < 8; i++)
            std::fprintf(stderr, "%u ", aa->output_def.dims[i]);
        std::fprintf(stderr, "| bb(%s).rank=%u dims=",
                     bb && bb->name_tag && bb->name_tag->name() ? bb->name_tag->name() : "?",
                     bb ? bb->output_def.rank : 0);
        if (bb) for (uint32_t i = 0; i < bb->output_def.rank && i < 8; i++)
            std::fprintf(stderr, "%u ", bb->output_def.dims[i]);
        std::fprintf(stderr, "| n=%llu b_elems=%llu\n",
                     (unsigned long long)n, (unsigned long long)b_elems);
    }
    /* 广播物化: a/b 任一输入元素数 < 输出即物化为全尺寸 temp
     * (numpy 广播; Expand 的 a [1,2,1,32,256]→[1,2,4,32,256] 同款) */
    {
        /* out_idx 6/7: 不与 op 自身输出(0)抢 tkey 注册 —
         * 否则覆盖 op_temp → 下游 src_ref 拿错 temp(probe 实锤) */
        auto fold4 = [](const OutputDef& odef, uint32_t d4[4]) {
            /* 折叠前导 1 后右对齐到 4 轴(设备契约; numpy 广播语义,
             * 与 host in_bc 同款) */
            uint32_t s = 0;
            while (s + 1 < odef.rank && odef.dims[s] == 1) s++;
            uint32_t rr = odef.rank - s;
            if (rr == 0) rr = 1;
            uint32_t take = rr < 4 ? rr : 4;
            uint32_t t[4] = {1, 1, 1, 1};
            for (uint32_t i = 0; i < take; i++)
                t[4 - take + i] = odef.dims[s + rr - take + i];
            for (int i = 0; i < 4; i++) d4[i] = t[i];
        };
        auto materialize = [&](uint32_t& t, const OpDef* src_op, uint32_t out_idx) {
            if (!src_op) return;
            uint64_t s_elems = elems_of(src_op);
            if (s_elems == 0 || s_elems >= n) return;
            uint32_t tmp = em.fresh_temp(gp, od->op_id, out_idx);
            uint32_t in_d[4] = {1, 1, 1, 1}, out_d[4] = {1, 1, 1, 1};
            fold4(src_op->output_def, in_d);
            fold4(od->output_def, out_d);
            em.add_op(OP_BROADCAST_F16, {t, tmp, (uint32_t)n, (uint32_t)s_elems,
                                         in_d[0], in_d[1], in_d[2], in_d[3],
                                         out_d[0], out_d[1], out_d[2], out_d[3]});
            t = tmp;
        };
        materialize(a_t, gp.get_op_at(od->inputs[0].src_id), 6);
        materialize(b_t, bb, 7);
    }
    uint32_t sub = qnn_binary_to_sub(subtype);
    if (sub == 0) em.add_op(OP_ADD_F16, {a_t, b_t, out_t, (uint32_t)n});
    else if (sub != 0xFFFFFFFFu)
        em.add_op(OP_BINARY_F16, {a_t, b_t, out_t, (uint32_t)n, sub});
    else
        em.add_op(OP_BINARY_F16, {a_t, b_t, out_t, (uint32_t)n, 0});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_eltwise_binary{
    {"Eltwise_Binary", op_eltwise_binary},
};
