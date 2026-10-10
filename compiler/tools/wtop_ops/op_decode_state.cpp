// op_decode_state.cpp — T018 decode 图状态算子发射(host 发射侧)
//   DecodeFaceF16 → op15 rank1(EXT_IN 合并面 u16 视图切片)
//   KvAppendF16 → op22 / KvGatherKT,V → op23 (0x6000|state_idx, AUTO 哨兵)
//   Conv1dSsmStepF16 → op29(含 SiLU) / DeltaChunkF16 → op30(q_raw 图内 q̂ 直喂)
// 状态槽(WT_SLOT_STATE 描述符)与 EXT_IN 面别名由 wtop_emit 主流程发射。
#include "wtop_ops.hpp"
#include "hnnx/ir/scalar_params.hpp"

using namespace wtop;

namespace {

struct Sp {
    std::vector<ScalarParam> v;
    explicit Sp(const OpDef* od) : v(unpack_scalar_params(od->op_data)) {}
    uint32_t u(const char* k, uint32_t def) const {
        const ScalarParam* p = scalar_get(v, k);
        return p ? (uint32_t)p->as_int() : def;
    }
};

// decode_in i32 字流 → u16 视图 f16 位级切片(exec_slice rk=1: y[i]=x[b0+i])
static int op_decode_face(Emitter& em, GraphPrepare& gp, const OpDef* od,
                          std::map<uint64_t, uint32_t>& wslots, const std::string& nm,
                          const WtopEmitShared& sh) {
    (void)nm;
    Sp sp(od);
    uint32_t beg = sp.u("u16_begin", 0) + sh.face_u16_shift;
    uint32_t n_out = (uint32_t)elems_of(od);
    const OpDef* srcp = gp.get_op_at(od->inputs[0].src_id);
    if (!srcp) { std::fprintf(stderr, "error: decode face src missing\n"); return 2; }
    uint32_t d0 = sh.dyn_in_u16_total;
    if (!d0) {
        uint64_t el = 1;
        for (uint32_t i = 0; i < srcp->output_def.rank && i < 5; ++i)
            el *= (uint64_t)srcp->output_def.dims[i];
        d0 = (uint32_t)(el * 2);  // i32 输入 4B/元素 → u16 计
    }
    if (beg + n_out > d0) {
        std::fprintf(stderr, "error: decode face 窗口 [%u,%u) 越界(面 u16 长 %u)\n",
                     beg, beg + n_out, d0);
        return 4;
    }
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_STRIDED_SLICE_F16, {x_t, out_t, n_out, 1u,
                                     beg, 0u, 0u, beg + n_out, 0u, 0u,
                                     1u, 1u, 1u, d0, 1u, 1u});
    return 0;
}

// k_face 直通锚保 append→gather 拓扑序; 设备 op22 只写状态(无输出引用)
static int op_kv_append(Emitter& em, GraphPrepare& gp, const OpDef* od,
                        std::map<uint64_t, uint32_t>& wslots, const std::string& nm,
                        const WtopEmitShared&) {
    Sp sp(od);
    uint32_t k_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t v_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    em.fresh_temp(gp, od->op_id);
    em.add_op(OP_KV_APPEND_F16, {k_t, v_t,
                                 WT_REF_STATE_FLAG | decode_state_gid(nm, sp.u("state_idx", 0)),
                                 WT_KV_POS_AUTO});
    return 0;
}

// 状态 ring → KT/V 面(槽 0=跳面; AUTO=读计数器; 输出定宽 nslots)
static int op_kv_gather(Emitter& em, GraphPrepare& gp, const OpDef* od,
                        std::map<uint64_t, uint32_t>& wslots, const std::string& nm,
                        const WtopEmitShared&) {
    (void)gp; (void)wslots;
    Sp sp(od);
    bool kt = (nm == "KvGatherKT");
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_KV_GATHER_F16, {WT_REF_STATE_FLAG | decode_state_gid(nm, sp.u("state_idx", 0)),
                                 kt ? out_t : 0u, kt ? 0u : out_t,
                                 WT_KV_POS_AUTO,
                                 sp.u("n_heads", 8), sp.u("kv_heads", 2),
                                 sp.u("head_dim", 256)});
    return 0;
}

// 单步 depthwise causal conv(kernel 含 SiLU); w 槽 [d_conv, C] f16 同 op19 布局
static int op_conv1d_ssm_step(Emitter& em, GraphPrepare& gp, const OpDef* od,
                              std::map<uint64_t, uint32_t>& wslots, const std::string& nm,
                              const WtopEmitShared&) {
    Sp sp(od);
    uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t w_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t y_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_CONV1D_SSM_STEP_F16, {x_t, w_t, y_t,
                                       WT_REF_STATE_FLAG | decode_state_gid(nm, sp.u("state_idx", 0)),
                                       sp.u("C", 6144), sp.u("d_conv", 4)});
    return 0;
}

// 单步 gated delta rule(c=ntok=1; 状态 S=[H*D*D] f32 原地更新)
static int op_delta_chunk(Emitter& em, GraphPrepare& gp, const OpDef* od,
                          std::map<uint64_t, uint32_t>& wslots, const std::string& nm,
                          const WtopEmitShared&) {
    if (od->inputs.size() < 5) {
        std::fprintf(stderr, "error: DeltaChunkF16 inputs=%zu < 5\n", od->inputs.size());
        return 2;
    }
    Sp sp(od);
    uint32_t k_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t v_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
    uint32_t q_t = em.src_ref(od->inputs[2], gp.get_input_node_id(), gp, wslots);
    uint32_t be_t = em.src_ref(od->inputs[3], gp.get_input_node_id(), gp, wslots);
    uint32_t g_t = em.src_ref(od->inputs[4], gp.get_input_node_id(), gp, wslots);
    uint32_t y_t = em.fresh_temp(gp, od->op_id);
    em.add_op(OP_DELTA_CHUNK_F16, {k_t, v_t, q_t, be_t, g_t, y_t,
                                   WT_REF_STATE_FLAG | decode_state_gid(nm, sp.u("state_idx", 0)),
                                   1u, 1u,
                                   sp.u("H", 16), sp.u("D", 128),
                                   sp.u("q_raw", 1)});
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_decode_state{
    {"DecodeFaceF16", op_decode_face},
    {"KvAppendF16", op_kv_append},
    {"KvGatherKT", op_kv_gather},
    {"KvGatherV", op_kv_gather},
    {"Conv1dSsmStepF16", op_conv1d_ssm_step},
    {"DeltaChunkF16", op_delta_chunk},
};
