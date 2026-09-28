// op_gather.cpp — Gather 发射(算子三件套·host 发射侧; embedding 表 + GDN state 回读)
#include "wtop_ops.hpp"
#include "hnnx/ir/scalar_params.hpp"

using namespace wtop;

namespace {

/* host ops.cpp Gather 同款的 leading-1 压缩: 剥前导 size-1 后 axis 必须
 * 同步左移 (VL 实锤: tbl [1,16,1,64,64] axis=3 → 压缩后真轴=2, 不左移
 * 取到末维 — L0 cos 1.0→0.53 回归根因) */
static uint32_t g_compact(const OpDef* d, uint64_t dd[5]) {
    uint32_t s = 0, r = d ? d->output_def.rank : 0;
    while (s + 1 < r && d->output_def.dims[s] == 1) s++;
    uint32_t w = 0;
    for (uint32_t i = 0; i < 5; i++) dd[i] = 1;
    for (uint32_t i = s; i < r && w < 5; i++, w++) dd[w] = d->output_def.dims[i];
    return w ? w : 1;
}

static int op_gather(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    const OpDef* tbl = od->inputs.size() > 0 ? gp.get_op_at(od->inputs[0].src_id) : nullptr;
    const OpDef* idx = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
    if (!tbl || !idx) {
        std::fprintf(stderr, "[op_gather] %s: 缺输入, 拒发\n", nm.c_str());
        return -1;
    }
    /* tbl 走 src_ref: GDN gather 读 ScatterNd/CumulativeSum 等计算 op 输出
     * (state 回读/门控), 不是权重 const — 直调 ensure_weight_slot 会打成
     * 128B 零槽 (0.8B 全图 1189 处实锤)。const → 0x8000|slot, temp → temp id */
    uint32_t tbl_s = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    /* idx: tagged const → int32 原样入槽 (ensure_weight_slot 会 f32→f16 洗零);
     * Input 节点(ids/pos) 经 src_ref → 输入槽 (烘焙 i32 原文 / EXT_IN 注入) */
    uint32_t idx_t;
    if (idx->const_data_size >= 4 &&
        od->inputs[1].src_id != em.main_input_id &&
        em.input_slots.find(od->inputs[1].src_id) == em.input_slots.end()) {
        idx_t = 0x8000u | em.ensure_i32_slot(gp, idx);
    } else {
        idx_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    }
    uint32_t out_t = em.fresh_temp(gp, od->op_id);

    /* axis: 从 serialized_extra(ExtraAxis, extract_axis_extra 打包)读 —— 与
     * op_concat/op_softmax/op_cumsum 同源。tagged.bin deserialize 不填 op_data
     * (serialize_opdef 只写 extra 尾随段), 直读 op_data 恒得 axis=0 → GDN
     * gather 轴错位 (op222 cos 0.0103 根因)。scalar 按 tbl 原始 rank 计,
     * 压缩后左移 (host 同式)。 */
    int axis = 0;
    {
        ExtraAxis e{};
        if (od->serialized_extra.size() >= sizeof(e)) {
            std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            axis = (int)e.axis;
        } else {
            auto sp = unpack_scalar_params(od->op_data);
            if (const ScalarParam* p = scalar_get(sp, "axis"))
                axis = (int)(p->is_numeric ? p->value_num
                                           : std::strtol(p->value_str.c_str(), nullptr, 10));
        }
    }
    uint64_t td[5], xd[5];
    uint32_t traw = tbl->output_def.rank;
    uint32_t irank = g_compact(tbl, td);
    uint32_t xrank = g_compact(idx, xd);
    if (axis < 0) axis += (int)traw;
    int stripped = ((int)traw > (int)irank) ? (int)traw - (int)irank : 0;
    axis -= stripped;
    if (axis < 0 || axis >= (int)irank) axis = 0;  /* host 同款钳位 */
    /* host ops.cpp Gather 逐字同款坐标分解上设备: out 坐标按 od(=td 把
     * axis 换成 xd)分解, idx 块取 r=tbl[axis], 余维直通。维度打包:
     * args[4] = axis | xrank<<8 | irank<<16, 后随 td[5]/xd[5] (紧凑形)。 */
    uint64_t ne = elems_of(od);
    uint32_t packed = (uint32_t)axis | ((uint32_t)xrank << 8) | ((uint32_t)irank << 16);
    em.add_op(OP_GATHER_F16, {tbl_s, idx_t, out_t, (uint32_t)ne, packed,
                              (uint32_t)td[0], (uint32_t)td[1], (uint32_t)td[2],
                              (uint32_t)td[3], (uint32_t)td[4],
                              (uint32_t)xd[0], (uint32_t)xd[1], (uint32_t)xd[2],
                              (uint32_t)xd[3], (uint32_t)xd[4]});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_gather{
    {"Gather", op_gather},
};
