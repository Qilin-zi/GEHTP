// op_conv2d.cpp — Conv2d 发射(算子三件套·host 发射侧; im2col+GEMM 分块)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

static int op_conv2d(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared& sh) {
    (void)nm;
    // extra_info: 60B fixed + tiling 段
    if (od->serialized_extra.size() < sizeof(ExtraConv) + 16) {
        std::fprintf(stderr, "error: conv extra too short\n"); return 2;
    }
    ExtraConv e;
    std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
    uint32_t hdr[4];
    std::memcpy(hdr, od->serialized_extra.data() + sizeof(ExtraConv), 16);

    // 受支持几何门(特性门): dh=dw=1、s1、group=1、same-pad
    if (e.dh != 1 || e.dw != 1 || e.sh != 1 || e.sw != 1 || e.group != 1) {
        std::fprintf(stderr, "error: unsupported conv geometry (sh=%u sw=%u dh=%u dw=%u group=%u)\n",
                     e.sh, e.sw, e.dh, e.dw, e.group);
        return 4;
    }
    uint32_t ph = e.kh / 2, pw = e.kw / 2;
    if (e.ph_begin != ph || e.ph_end != ph || e.pw_begin != pw || e.pw_end != pw) {
        std::fprintf(stderr, "error: unsupported conv pad (%u,%u,%u,%u) vs same-pad (%u,%u)\n",
                     e.ph_begin, e.ph_end, e.pw_begin, e.pw_end, ph, pw);
        return 4;
    }

    uint32_t num_tiles = hdr[3];
    const uint32_t* descs = reinterpret_cast<const uint32_t*>(
        od->serialized_extra.data() + sizeof(ExtraConv) + 16);
    if (num_tiles == 0) {  // 旧流/未分块: 整图单 tile
        num_tiles = 1;
    }
    // 无分块时的整图 tile 描述(字段序同 ConvTileDesc 19×u32)
    const uint32_t full_desc[19] = {0, 0, sh.H, sh.W, 0, 0, sh.H, sh.W,
                                    e.kh, e.kw, e.sh, e.sw, e.ph_begin, e.pw_begin,
                                    sh.C, sh.C, 0, sh.C, 0};
    uint32_t src_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    uint32_t cols_t = em.fresh_temp(gp, 0xFFFFFFF0);  // 专用 cols 槽
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t* d = (hdr[3] == 0) ? full_desc : descs + t * 19;
        uint32_t iy0 = d[4], ix0 = d[5], ih = d[6], iw = d[7];
        uint32_t oy0 = d[0], ox0 = d[1], th = d[2], tw = d[3];
        uint32_t co0 = d[16], co_n = d[17];
        // im2col: 输入切片(含 halo) → cols [th*tw × K]
        em.add_op(OP_IM2COL, {src_t, cols_t, sh.H, sh.W, sh.C, e.kh, e.kw,
                              e.ph_begin, e.pw_begin, e.sh, e.sw,
                              iy0, ix0, th, tw});
        // GEMM: cols [M,K] @ W [K,N] → 输出 tile 写入全图 out_temp
        em.add_op(OP_CONV2D_F16, {cols_t, sh.w_slot, sh.b_slot, out_t,
                                  th * tw, e.kh * e.kw * sh.C, sh.C,
                                  oy0, ox0, sh.H, sh.W, co0, co_n});
    }
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_conv2d{
    {"Conv2d", op_conv2d},
};
