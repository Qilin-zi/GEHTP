// op_strided_slice.cpp — StridedSlice 发射(算子三件套·host 发射侧;
// srk≤3 旧路径 + srk>3 秩对齐折叠, c5d4c2a 修复体)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_strided_slice(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    (void)nm;
    ExtraStridedSlice e{};
    if (od->serialized_extra.size() >= sizeof(e))
        std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    const OpDef* srcp = gp.get_op_at(od->inputs[0].src_id);
    if (!srcp) { std::fprintf(stderr, "error: slice src missing\n"); return 2; }
    uint32_t srk = srcp->output_def.rank;
    uint32_t n_out = (uint32_t)elems_of(od);
    std::vector<uint32_t> args{x_t, out_t, n_out, 0};

    if (srk <= 3) {
        /* srk≤3 旧路径(逐字不动): 设备契约 rank≤3 通用切片(begin/end/stride 各 3;
         * rank4 且 dim0=1 降 rank 的历史语义只存在于 srk>3 分支) */
        uint32_t rk = std::min<uint32_t>(e.rank, 4);
        if (rk == 4) rk = 3;
        args[3] = rk;
        uint32_t b0 = 0, b1 = 0, b2 = 0, e0 = 0, e1 = 0, e2 = 0, s0 = 1, s1 = 1, s2 = 1;
        const auto& pool = gp.const_pool();
        if (e.ranges_offset && e.ranges_offset + e.rank * 12 <= pool.size()) {
            const int32_t* rg = reinterpret_cast<const int32_t*>(pool.data() + e.ranges_offset);
            uint32_t ax0 = (e.rank == 4) ? 1u : 0u;  /* 降 rank: 跳 batch 轴 */
            /* ranges 布局 = 每轴三元组交错 [b,e,s]×rank(JSON 实测) */
            b0 = (uint32_t)rg[ax0 * 3 + 0]; e0 = (uint32_t)rg[ax0 * 3 + 1]; s0 = (uint32_t)rg[ax0 * 3 + 2];
            b1 = (uint32_t)rg[(ax0 + 1) * 3 + 0]; e1 = (uint32_t)rg[(ax0 + 1) * 3 + 1]; s1 = (uint32_t)rg[(ax0 + 1) * 3 + 2];
            b2 = (uint32_t)rg[(ax0 + 2) * 3 + 0]; e2 = (uint32_t)rg[(ax0 + 2) * 3 + 1]; s2 = (uint32_t)rg[(ax0 + 2) * 3 + 2];
        }
        args.push_back(b0); args.push_back(b1); args.push_back(b2);
        args.push_back(e0); args.push_back(e1); args.push_back(e2);
        args.push_back(s0); args.push_back(s1); args.push_back(s2);
        /* 输入 dims(轴 1..3, C 序); 二维 [ax0,ax1] 线性步长须 ax0*dims[1]+ax1 */
        uint32_t d0 = 1, d1 = 1, d2 = 1;
        {
            uint32_t ar = srk;
            while (ar > 0 && srcp->output_def.dims[ar - 1] == 1) ar--;
            if (ar >= 3) { d0 = srcp->output_def.dims[ar - 3]; d1 = srcp->output_def.dims[ar - 2]; d2 = srcp->output_def.dims[ar - 1]; }
            if (ar == 2) { d0 = srcp->output_def.dims[0]; d1 = srcp->output_def.dims[1]; d2 = 1; }
            if (ar == 1) { d0 = 1; d1 = 1; d2 = srcp->output_def.dims[0]; }
        }
        args.push_back(d0); args.push_back(d1); args.push_back(d2);
        em.add_op(OP_STRIDED_SLICE_F16, args);
        return 0;
    }

    /* srk>3(rank4/5): 先按 ranges 秩对齐(折前导 size-1 至 e.rank 参照系,
     * 与 Transpose 的 pc≠rk 同款——L0 实测 [1,1,6144,35] 配 rank-3 ranges),
     * 再按序折 size-1 且切片平凡的轴至 ≤3, ranges 同步。
     * GDN L0 实锤(例41 op207 PD 死): [1,16,1,64,64] ranges 5×3,
     * 旧码 ax0=(e.rank==4)?1:0 对 rank5 取轴 0..2 → 几何 (1,16,1) 与
     * n_out 10240 不符, 设备 exec_slice 按 n_out 线性扫 → 20× OOB 读,
     * PD 静默死(无 rc 无日志)。折叠须保持两条不变量:
     *   ① rank4 dim0=1 的旧产物逐字节不变(只折到 3 即停, 不多折);
     *   ② product(ceil((e-b)/s)) == n_out(发射期硬检查, 把 PD 死
     *      变成编译期诚实报错)。 */
    if (srk > 5 || e.rank > 5) {
        std::fprintf(stderr, "error: slice rank srk=%u e.rank=%u > 5\n", srk, e.rank);
        return 4;
    }
    if (e.rank == 0 || e.rank > srk) {
        std::fprintf(stderr, "error: slice ranges 秩 %u 与输入秩 %u 不配(特性门)\n", e.rank, srk);
        return 4;
    }
    uint32_t dims[5] = {1, 1, 1, 1, 1};
    for (uint32_t i = 0; i < srk; i++) dims[i] = srcp->output_def.dims[i];
    const auto& pool = gp.const_pool();
    int32_t rg[5][3];
    bool have_ranges = e.ranges_offset && e.ranges_offset + e.rank * 12 <= pool.size();
    for (uint32_t ax = 0; ax < 5; ax++) {
        if (have_ranges && ax < e.rank) {
            const int32_t* p = reinterpret_cast<const int32_t*>(pool.data() + e.ranges_offset);
            rg[ax][0] = p[ax * 3 + 0]; rg[ax][1] = p[ax * 3 + 1]; rg[ax][2] = p[ax * 3 + 2];
        } else {
            /* 无 ranges: 全轴全长(恒等切片) */
            rg[ax][0] = 0;
            rg[ax][1] = (ax < srk) ? (int32_t)dims[ax] : 1;
            rg[ax][2] = 1;
        }
    }
    /* 秩对齐: ranges 参照系 = 末尾 e.rank 轴, 折前导 size-1 至 srk==e.rank */
    while (srk > e.rank && dims[0] == 1) {
        for (uint32_t i = 0; i + 1 < srk; i++) dims[i] = dims[i + 1];
        srk--;
    }
    if (srk != e.rank) {
        std::fprintf(stderr, "error: slice ranges 秩 %u 与输入秩 %u 不可对齐(特性门)\n", e.rank, srk);
        return 4;
    }
    for (uint32_t ax = 0; ax < srk; ax++) {
        if (rg[ax][2] <= 0) {
            std::fprintf(stderr, "error: slice stride %d 非正(ax %u, 特性门)\n", rg[ax][2], ax);
            return 4;
        }
    }
    uint32_t nd[3] = {1, 1, 1}, nb[3] = {0, 0, 0}, ne[3] = {0, 0, 0}, ns[3] = {1, 1, 1};
    uint32_t nrk = 0, folded = 0;
    for (uint32_t ax = 0; ax < srk; ax++) {
        uint32_t dim = dims[ax];
        bool foldable = (dim == 1) && rg[ax][0] == 0 && rg[ax][1] <= 1 && rg[ax][2] == 1;
        /* 还需折 (srk - folded > 3) 且本轴可折才折 —— 不多折(保 rank4
         * dim0=1 旧编码), 不少折(rank5 折至 3) */
        if (foldable && (int)(srk - folded) > 3) {
            folded++;
            continue;
        }
        if (nrk >= 3) {
            std::fprintf(stderr, "error: slice srk=%u 折后秩>3(ax %u, 特性门)\n", srk, ax);
            return 4;
        }
        nd[nrk] = dim;
        nb[nrk] = (uint32_t)rg[ax][0]; ne[nrk] = (uint32_t)rg[ax][1]; ns[nrk] = (uint32_t)rg[ax][2];
        nrk++;
    }
    /* 一致性硬检查: product(ceil((e-b)/s)) == n_out */
    uint64_t prod = 1;
    for (uint32_t ax = 0; ax < nrk; ax++) {
        uint32_t span = (ne[ax] > nb[ax]) ? ne[ax] - nb[ax] : 0;
        prod *= (span + ns[ax] - 1) / ns[ax];
    }
    if (prod != n_out) {
        std::fprintf(stderr, "error: slice 几何 %llu != n_out %u (srk=%u, 特性门)\n",
                     (unsigned long long)prod, n_out, srk);
        return 4;
    }
    args[3] = nrk;
    args.push_back(nb[0]); args.push_back(nb[1]); args.push_back(nb[2]);
    args.push_back(ne[0]); args.push_back(ne[1]); args.push_back(ne[2]);
    args.push_back(ns[0]); args.push_back(ns[1]); args.push_back(ns[2]);
    args.push_back(nd[0]); args.push_back(nd[1]); args.push_back(nd[2]);
    em.add_op(OP_STRIDED_SLICE_F16, args);
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_strided_slice{
    {"StridedSlice", op_strided_slice},
};
