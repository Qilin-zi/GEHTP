// op_eltwise_ternary.cpp — Eltwise_Ternary 发射(算子三件套·host 发射侧; SELECT)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_eltwise_ternary(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    // SELECT 语义: out = in0(cond) ? in1 : in2
    uint32_t c_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t a_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t b_t = em.src_ref(od->inputs[2], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    /* numpy 广播物化: 设备 exec SELECT 只做纯元素选择(loop 界 = true 元素数),
     * 任一输入元素数 < 输出时先物化为全尺寸 temp, 否则设备只写 scalar 元素
     * 余下读旧 temp 垃圾 → GDN 门控链全 nan(0.8B op105/op53/op55 实锤)。
     * cond [1,1,64,64]→[1,16,1,64,64](16 头广播) 也靠 OP_BROADCAST_F16 完成,
     * 设备 SELECT 的标量广播(≤1)覆盖不了部分广播。 */
    uint64_t n = elems_of(od);
    {
        auto fold4 = [](const OutputDef& odef, uint32_t d4[4]) {
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
        materialize(c_t, gp.get_op_at(od->inputs[0].src_id), 6);
        materialize(a_t, gp.get_op_at(od->inputs[1].src_id), 7);
        materialize(b_t, gp.get_op_at(od->inputs[2].src_id), 8);
    }
    em.add_op(OP_BINARY_F16, {c_t, a_t, b_t, out_t, 8});  // 8=SELECT(cond,a,b)
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_eltwise_ternary{
    {"Eltwise_Ternary", op_eltwise_ternary},
};
