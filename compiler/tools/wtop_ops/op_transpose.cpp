// op_transpose.cpp — Transpose 发射(算子三件套·host 发射侧)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_transpose(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    const OpDef* src = gp.get_op_at(od->inputs[0].src_id);
    if (!src) { std::fprintf(stderr, "error: transpose src missing\n"); return 2; }
    uint32_t rk = src->output_def.rank;
    if (rk > 5) { std::fprintf(stderr, "error: transpose rank %u > 5\n", rk); return 4; }
    uint32_t dims[5] = {1, 1, 1, 1, 1};
    for (uint32_t i = 0; i < rk; i++) dims[i] = src->output_def.dims[i];
    // 读完整 perm(最多 5 轴; 缺省单位)
    int32_t pv[5] = {0, 1, 2, 3, 4};
    uint32_t pc = 0;
    const OpDef* permc = (od->inputs.size() > 1) ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
    if (permc && permc->const_data_size >= 4 && permc->const_data_size % 4 == 0) {
        pc = permc->const_data_size / 4;
        if (pc > 5) { std::fprintf(stderr, "error: transpose perm 轴数 %u > 5\n", pc); return 4; }
        const int32_t* p = reinterpret_cast<const int32_t*>(
            gp.const_pool().data() + permc->const_data_offset);
        for (uint32_t i = 0; i < pc; i++) pv[i] = p[i];
    }
    if (pc == 0) pc = rk;  // 无 const: 单位 perm
    /* 秩对齐: 图 rank-4 填充 vs rank-3 perm 契约(L3 实测 [1,1,X,Y]
     * pc=3) —— 折前导 size-1 维至 perm 参照系。旧码 >=16B 门槛漏读
     * 12B perm, 回落"单位"实为 reverse(0x00010203 字节序), 因前两轴
     * 恒 1 恰好等价, M4.2 81/81 是侥幸通过。 */
    while (rk > pc && dims[0] == 1) {
        for (uint32_t i = 0; i + 1 < rk; i++) dims[i] = dims[i + 1];
        rk--;
    }
    if (rk != pc) {
        std::fprintf(stderr, "error: transpose perm 轴数 %u 与输入秩 %u 不可对齐\n", pc, rk);
        return 4;
    }
    /* rank>4: 折叠任意 size-1 轴(对转置线性布局无影响), perm 重映射。
     * M4.2b 首撞: [1,16,1,128,64] perm[0,1,2,4,3]。 */
    int keep[5]; uint32_t nd[4] = {1, 1, 1, 1}, nrk = 0;
    for (uint32_t ax = 0; ax < rk; ax++) {
        if (rk > 4 && dims[ax] == 1) { keep[ax] = -1; continue; }
        keep[ax] = (int)nrk;
        if (nrk < 4) nd[nrk] = dims[ax];
        nrk++;
    }
    if (nrk > 4) {
        std::fprintf(stderr, "error: transpose rank %u 无 size-1 轴可折\n", rk);
        return 4;
    }
    int32_t np[4] = {0, 1, 2, 3}; uint32_t npi = 0;
    for (uint32_t i = 0; i < rk && npi < 4; i++) {
        int32_t ax = pv[i];
        if (ax < 0 || ax >= (int32_t)rk) { npi = 0xFFFFFFFFu; break; }
        if (keep[ax] < 0) continue;
        np[npi++] = keep[ax];
    }
    if (npi != nrk) {
        std::fprintf(stderr, "error: transpose perm 与折叠不一致(npi=%u nrk=%u)\n", npi, nrk);
        return 4;
    }
    uint32_t perm = 0;
    for (uint32_t i = 0; i < nrk; i++) perm |= ((uint32_t)np[i] << (8 * i));
    uint32_t src_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    /* 统一走通用 N-D 转置(形状全参数化); opcode 10 的 4D NCHW
     * 契约(H/W/C 源自图输入)对 transformer 张量全错(probe 实锤) */
    em.add_op(OP_TRANSPOSE_GEN_F16,
              {src_t, out_t, nrk, nd[0], nd[1], nd[2], nd[3], perm});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_transpose{
    {"Transpose", op_transpose},
};
