// op_matmul.cpp — FullyConnected / MatMul 发射(算子三件套·host 发射侧;
// W4A16 权重判定 + batched BMM flags)
#include "wtop_ops.hpp"

using namespace wtop;

namespace {

/* 008 T012: 权重 crouton 预打包 — 逐字节镜像 exec_matmul_f16_hmx 的 pack_wgt
 * 循环(oplist_exec.c)。SPLIT chunk = 独立紧凑布局 tile[(c*n_dot+kt)*1024]
 * (c 为 chunk 内列 tile 序), 源列号 = n0 + c*32 + n; 行主 [K,N_full] stride
 * = N_full。bit3 op 运行期按 DMA 线性搬此槽即得 wgt_cr。 */
static void pack_wgt_crouton_f16(const uint16_t* w, uint16_t* dst,
                                 uint32_t K, uint32_t N_full, uint32_t n0,
                                 uint32_t N_chunk, int t1) {
    uint32_t n_col = N_chunk / 32u, n_dot = K / 32u;
    for (uint32_t c = 0; c < n_col; ++c)
        for (uint32_t kt = 0; kt < n_dot; ++kt) {
            uint16_t* tile = dst + ((size_t)c * n_dot + kt) * 1024u;
            for (uint32_t k = 0; k < 32u; ++k)
                for (uint32_t n = 0; n < 32u; ++n) {
                    size_t src = t1 ? ((size_t)(n0 + c * 32u + n) * K + (kt * 32u + k))
                                    : ((size_t)(kt * 32u + k) * N_full + (n0 + c * 32u + n));
                    tile[(k / 2u) * 64u + 2u * n + (k & 1u)] = w[src];
                }
        }
}

static int op_matmul(Emitter& em, GraphPrepare& gp, const OpDef* od, std::map<uint64_t, uint32_t>& wslots, const std::string& nm, const WtopEmitShared&) {
    uint32_t a_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
    uint32_t out_t = em.fresh_temp(gp, od->op_id);
    const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
    /* B 为 const(权重)→ 槽; 运行时(q·kᵀ 的 k)→ temp 引用 */
    bool w_is_const = w && (w->is_const() || w->const_data_size > 0);
    uint32_t w_s = w_is_const
        ? (0x8000u | em.ensure_weight_slot(gp, w, wslots, od))
        : (od->inputs.size() > 1
               ? em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots)
               : em.dummy_slot_id);
    ExtraFc ef{};
    ExtraMatMul em2{};
    uint32_t m = 0, k = 0, nn = 0;
    if (nm == "FullyConnected") {
        if (od->serialized_extra.size() >= sizeof(ef))
            std::memcpy(&ef, od->serialized_extra.data(), sizeof(ef));
        m = ef.m; k = ef.k; nn = ef.n;
    } else {
        if (!od->serialized_extra.empty()) {
            size_t c = std::min(od->serialized_extra.size(), sizeof(em2));
            std::memcpy(&em2, od->serialized_extra.data(), c);
        }
        m = em2.m; k = em2.k; nn = em2.n;
    }
    if (m == 0 || k == 0 || nn == 0) {
        std::fprintf(stderr, "error: %s extra 未提取 (M/K/N=0)\n", nm.c_str());
        return 4;
    }
    /* FC bias 折叠(pass_fc_bias)配套: ExtraFc.has_bias 只认 pass 写入的
     * op_data scalar 标记(extract_fc_extra 同源口径; 注入的 N 元素 wscale
     * const 不再误判)。此处再核实 inputs[2] 确为 const 才发 BIAS opcode。
     * 005 US3a: MatMul 链头同标记同通道(em2.has_bias, 旧 bin 24B 无此
     * 字段按 0)。 */
    const OpDef* bias_op = nullptr;
    bool mm_mark = (nm == "MatMul") && em2.has_bias;
    bool has_bias = ((nm == "FullyConnected") ? (ef.has_bias != 0) : mm_mark) &&
                    od->inputs.size() > 2;
    if (nm == "MatMul" && od->serialized_extra.size() == 24u)
        has_bias = false;                     // 旧 bin: 无 has_bias 字段
    if (has_bias) {
        const OpDef* bp = gp.get_op_at(od->inputs[2].src_id);
        if (bp && (bp->is_const() || bp->const_data_size > 0)) bias_op = bp;
        else has_bias = false;
    }
    /* Q4_0 打包权重(K*N/2 字节)→ W4A16;f16 池权重 → MATMUL_F16
     * (float 图; probe 首撞: f16 权重被按 W4A16 执行直接失败) */
    /* 013 T011: 原生 W4A16 需 M%256==0 (M=32/128 实测 0xA002 失败, dd_worker
     * M%256 前置判)。M%256!=0 的打包权重回落 f16 路由(M<32→GEMV /
     * 32≤M<256→dense_f16), 绝不发必然失败的 W4A16。 */
    bool w4 = w_is_const && (w_s != em.dummy_slot_id) &&
              (m % 256u == 0u) &&
              (em.slots[w_s & 0x7FFFu].len == (k * nn / 2u));
    if (w4) {
        if (has_bias) {
            /* W4A16 的 bias 走权重内嵌表(exec_matmul 扫槽), 独立 bias 槽
             * 无通道, 静默丢 bias = 数值错 —— 硬报错, 待 W4A16+bias 通道 */
            std::fprintf(stderr, "error: FC has_bias + W4A16 权重组合暂不支持 (M=%u K=%u N=%u)\n",
                         m, k, nn);
            return 4;
        }
        /* 003 T015: W4A16 引用规范化。运行期 exec_matmul 的 act/w 均按裸
         * slot 索引取数 (b->slots[act_s]), act 另认 0x4000|temp 驻留形态;
         * 旧发射把 0x8000|slot / 裸 temp 原样传入 = slots 表越界读, 真实图
         * W4A16 从未走通 (仅合成 blob 用裸 slot 验证过)。act 三态:
         *   const 槽 → 裸 slot (烘焙字节, 零新 op)
         *   VTCM 驻留 temp → 0x4000|temp (生产者已直写池, 零新 op)
         *   DDR temp / 主输入(EXT_IN 注入) → OP_DMA 预置到池内 staging (--dma-runlist) */
        uint32_t w_raw = w_s & 0x7FFFu;
        uint32_t a_slot = (a_t & 0x8000u) ? (a_t & 0x7FFFu) : 0xFFFFFFFFu;
        bool act_is_const_slot = (a_slot != 0xFFFFFFFFu) && a_slot != 0u &&
                                 a_slot != em.dummy_slot_id;
        uint32_t act_arg;
        if (act_is_const_slot) {
            act_arg = a_slot;
        } else if (!(a_t & 0x8000u) && em.ddr_static &&
                   em.ddr_temp_vtcm.count(a_t) && em.ddr_temp_vtcm.at(a_t)) {
            act_arg = 0x4000u | a_t;
        } else {
            uint32_t act_bytes = m * k * 2u;
            uint32_t off = 0;
            int stg = em.alloc_vtcm_stage(act_bytes, &off);
            if (stg < 0 || !em.dma_runlist) {
                std::fprintf(stderr,
                    "error: W4A16 act 引用不可表达 (a_t=0x%X M=%u K=%u): 运行期仅认"
                    " 裸 slot / 0x4000|temp; DDR 源需 --dma-runlist + VTCM 池余量"
                    " (staging rc=%d dma_runlist=%d)\n",
                    a_t, m, k, stg, (int)em.dma_runlist);
                return 4;
            }
            uint32_t prev_eng = em.cur_engine;
            em.cur_engine = wt_annot_pack(0u, 1u, 0u);  /* 预置 DMA = 内存 op */
            em.add_op(OP_DMA, {a_t, 0x4000u | (uint32_t)stg, act_bytes,
                               0u, 0u, 1u, WT_DMA_FENCE_NONE, WT_DMA_FENCE_NONE});
            em.cur_engine = prev_eng;
            em.pending_stages.push_back({(uint32_t)(em.ops.size() - 1u), 0u,
                                         off, act_bytes});
            act_arg = 0x4000u | (uint32_t)stg;
        }
        /* 010 US1 (T008/T009): 权重 pin 决策显式下发 — 每个 W4A16 权重槽
         * 首次消费前发一条 OP_PIN[slot](去重)。运行期 xop_pin 在 carve 就绪
         * 时真搬一次并登记驻留; 后续同权重 matmul 走 exec_matmul 驻留追踪
         * 跳过重复 DDR→VTCM 搬 (FR-001/FR-002)。planned restage = OP_PIN 条数
         * = 去重权重槽数 (SC-003 对账源)。 */
        if (em.w4_pinned.insert({w_raw, true}).second) {
            uint32_t prev = em.cur_engine;
            em.cur_engine = wt_annot_pack(0u, 1u, 0u);  /* PIN = 内存/控制 op */
            em.add_op(OP_PIN, {w_raw});
            em.cur_engine = prev;
        }
        /* 013 T006: 9-arity 供给侧直连 — 编译期按闭式长度定位 bias/atbl/otbl
         * 三槽 (与运行期旧扫描同式: bias=(N/32)*512, atbl=8*(K/32)*4,
         * otbl=8*(N/32)*4), 直写 args[6..8] 裸 slot id (同 w_raw 约定)。三槽
         * 齐全 → 9 参 (运行期删扫描直读); 缺任一 → 6 参 legacy (运行期扫描)。
         * 终态由 T004 requant 生成三槽后按 id 直引 (此处长度定位 = 机制占位)。 */
        uint32_t bias_b = (nn / 32u) * 512u;
        uint32_t atbl_b = 8u * (k / 32u) * 4u;
        uint32_t otbl_b = 8u * (nn / 32u) * 4u;
        uint32_t bias_s = 0xFFFFFFFFu, atbl_s = 0xFFFFFFFFu, otbl_s = 0xFFFFFFFFu;
        for (uint32_t i = 0; i < em.slots.size(); i++) {
            uint32_t L = em.slots[i].len;
            if (bias_s == 0xFFFFFFFFu && L == bias_b) bias_s = i;
            else if (atbl_s == 0xFFFFFFFFu && L == atbl_b) atbl_s = i;
            else if (otbl_s == 0xFFFFFFFFu && L == otbl_b) otbl_s = i;
        }
        if (bias_s != 0xFFFFFFFFu && atbl_s != 0xFFFFFFFFu && otbl_s != 0xFFFFFFFFu)
            em.add_op(OP_MATMUL_W4A16,
                      {act_arg, w_raw, out_t, m, k, nn, bias_s, atbl_s, otbl_s});
        else
            em.add_op(OP_MATMUL_W4A16, {act_arg, w_raw, out_t, m, k, nn});
    } else {
        uint32_t flags = 0;
        if (nm == "MatMul")
            flags = (em2.transpose_in0 & 1u) | ((em2.transpose_in1 & 1u) << 1);
        else
            flags = 2u;  /* FC: 权重存 [N,K] */
        /* batched BMM: bit2 置位, 批数入高 16 位。批维 = 除末两维(M,N)外的
         * 全部前导维 —— 不得裁剪尾部 1 维: [1,8,1,1](attn q·kᵀ, M=N=1)会被
         * 剪掉真实批维 8 → Bn=1 → 执行器只写 1 元素, 其余 7 头读陈旧 temp
         * (seq=1 实锤: gid 4560 首发, 顺流污染全部 full-attn 层)。 */
        if (em2.batched) {
            flags |= 4u;
            uint32_t rk = od->output_def.rank;
            uint64_t bn = 1;
            for (uint32_t i = 0; i + 2 < rk; i++) bn *= od->output_def.dims[i];
            flags |= ((uint32_t)bn & 0xFFFFu) << 16;
        }
        /* 008 T012: f16 权重编译期 crouton 预打包。条件: --dma-runlist +
         * const 权重真槽 + 非 batched + M/K/N_chunk 32 全对齐 + 槽字节 =
         * K*N_full*2(f16 满形)。命中 → 镜像打包新槽 + flags bit3
         * (wgt_prepacked), 运行期 HMX 路径 DMA DDR→VTCM 线性预取替代
         * 标量散写 pack(探针: 39.85ms/op → ~0.3ms)。SPLIT 逐 chunk 独立
         * 打包(chunk 布局非全局 tile 子区); 查重键 = (权重 op_id,K,N_full,
         * n0,t1)。未命中静默维持原路径(零回归)。 */
        auto try_prepack = [&](uint32_t n0, uint32_t nc) -> uint32_t {
            if (!em.dma_runlist || !w_is_const || em2.batched) return 0;
            if (!(w_s & 0x8000u)) return 0;
            uint32_t src = w_s & 0x7FFFu;
            if (src == em.dummy_slot_id || src >= em.slots.size()) return 0;
            if ((m % 32u) || (k % 32u) || (nc % 32u)) return 0;
            const wt_slot& s = em.slots[src];
            if (s.len != k * nn * 2u) return 0;
            uint32_t t1 = (flags >> 1) & 1u;
            for (const auto& r : em.pp_recs)
                if (r.w_op_id == w->op_id && r.k == k && r.nn == nn && r.n0 == n0 && r.t1 == t1)
                    return r.pp_slot;
            const uint16_t* wbytes;
            if (s.addr == WT_SLOT_EXT_WGT) {
                if (s.offset + s.len > em.ext_weight_area.size()) return 0;
                wbytes = (const uint16_t*)(em.ext_weight_area.data() + s.offset);
            } else {
                if (s.offset + s.len > em.weight_area.size()) return 0;
                wbytes = (const uint16_t*)(em.weight_area.data() + s.offset);
            }
            std::vector<uint16_t> packed((size_t)nc * k);
            pack_wgt_crouton_f16(wbytes, packed.data(), k, nn, n0, nc, (int)t1);
            uint32_t pp = (s.addr == WT_SLOT_EXT_WGT)
                              ? em.add_ext_slot(nc * k * 2u, nc * k, (const uint8_t*)packed.data())
                              : em.add_slot(nc * k * 2u, nc * k, (const uint8_t*)packed.data());
            em.pp_recs.push_back({w->op_id, k, nn, n0, t1, pp});
            return pp;
        };
        const uint32_t kFlagWgtPP = 8u;  /* flags bit3 = wgt_prepacked */
        /* 002 US1: 超 VTCM 界的 f16 matmul 编译期 N 分块 (OP_MATMUL_F16_SPLIT)。
         * need 公式与 exec_matmul_f16_hmx 逐字节同口径; US2 前预算 = 运行时
         * VTCM total (16MB = 16777216, 设备 A2 query_VTCM 实锤), US2 后换
         * TAG_MEM_PLAN v2 的 scratch_size。
         * 条件: M/K/N 全 32 对齐 + 非 batched; 不满足则维持整 op (运行时
         * 超界回落标量, 现状行为零回归)。act/wgt/out 三引用跨 chunk 共享,
         * 仅 n0/N_chunk 两标量不同; 逐字节恒等五纪律见 spec FR-013。 */
        auto need_mm = [](uint32_t mm, uint32_t kk, uint32_t nn2) -> size_t {
            size_t a = (size_t)mm * kk * 2u;
            size_t w = (size_t)nn2 * kk * 2u;
            size_t o = (size_t)mm * nn2 * 2u;
            size_t off_w = (a + 2047u) & ~(size_t)2047u;
            size_t off_o = (off_w + w + 2047u) & ~(size_t)2047u;
            size_t off_s = (off_o + o + 2047u) & ~(size_t)2047u;
            return off_s + 2048u;
        };
        const size_t kVTCMTotal = 16u * 1024u * 1024u;
        /* 008 T006 预算单源化: 驻留池在场时 chunk 预算 = total − pool
         * (运行期 attach cap = vs − pool_size, oplist_exec.c f16 attach)。
         * p0 实测: pool 满额 16MB → arena≈0, 16MB 级大 chunk 运行时全部
         * attach 失败回落标量。池不在场(v4 惰性路径)保持 16MB 原口径。 */
        size_t vtcm_eff = kVTCMTotal;
        if (em.ddr_static && em.ddr_vtcm_cap > 0) {
            uint64_t pool_rounded = ((em.ddr_vtcm_cap + 1023) / 1024) * 1024;
            vtcm_eff = pool_rounded < (uint64_t)kVTCMTotal ? kVTCMTotal - (size_t)pool_rounded : 0;
        }
        if (!has_bias && !em2.batched && (m % 32u) == 0u && (k % 32u) == 0u && (nn % 32u) == 0u &&
            need_mm(m, k, nn) > vtcm_eff) {
            /* 二分找最大 32 倍数 N_c 使 need(m,k,N_c) ≤ 预算 (need 随 N 单调) */
            uint32_t N_chunk = 0, lo = 32u, hi = nn;
            while (lo <= hi) {
                uint32_t mid = lo + ((hi - lo) / 64u) * 32u;
                if (need_mm(m, k, mid) <= vtcm_eff) { N_chunk = mid; lo = mid + 32u; }
                else { hi = mid - 32u; }
            }
            if (N_chunk == 0) {
                /* 连最小 32 列 chunk 都超界 (act 自身超 VTCM 的极端形状) */
                std::fprintf(stderr, "warn: matmul M=%u K=%u N=%u 最小 chunk 也超 VTCM, 维持整 op\n",
                             m, k, nn);
            } else {
                /* 分块后每个 chunk 都满足 need_mm <= 16MB，必须保留
                 * HMX 计划；assign_engine 只看整 op 的形状，看不见这里的
                 * emit-side N 分块，否则 v3 会把所有 chunk 错标为 scalar。 */
                em.cur_engine = wt_annot_pack(2u, 1u, 0u);  /* HMX, 可回落 */
                for (uint32_t n0 = 0; n0 < nn; n0 += N_chunk) {
                    uint32_t nc = (N_chunk < nn - n0) ? N_chunk : (nn - n0);
                    uint32_t w_use = w_s, fl = flags;
                    uint32_t pp = try_prepack(n0, nc);
                    if (pp) { w_use = 0x8000u | pp; fl |= kFlagWgtPP; }
                    em.add_op(OP_MATMUL_F16_SPLIT, {a_t, w_use, out_t, m, k, nn, n0, nc, fl});
                }
                return 0;
            }
        }
        /* 013: Q4_0 权重槽的 M<32 GEMV 运行期走 HVX 列向量体
         * (exec_matmul_q4_0_gemv; n0/N 非 32 对齐手构 blob 兜底标量)。
         * 与 assign_engine m<32 计划 HVX 同口径, 无需覆盖 —— 仅形状/槽长
         * 落 Q4 判别 (长度即唯一判别: Q4_0 tile = (K/32)*(N/32)*640, 与
         * f16(k*n*2)/W4A16(k*n/2) 互斥) 时确保回落面可标量 (对齐兜底)。
         * 历史: 曾按槽长覆盖 SCALAR (GEMV 无 HVX 体时代), HVX 体落地后
         * 触发 planned SCALAR != actual HVX 硬门。 */
        if ((w_s & 0x8000u) && ((w_s & 0x7FFFu) < em.slots.size()) &&
            em.slots[w_s & 0x7FFFu].len == (size_t)(k / 32u) * (size_t)(nn / 32u) * 640u) {
            em.cur_engine = wt_annot_pack(1u, 1u, 0u);  /* Q4_0 GEMV = HVX, 可回落 */
        }
        if (has_bias) {
            /* OP_MATMUL_F16_BIAS (opcode 36): 尾部追加 bias 槽。
             * 005 US3b: 设备 HMX 路径已有 bias 通道(scale 区 out_bias,
             * SC-005 三判据 0 违例), 32 对齐形默认走 HMX —— US3a 的
             * 对齐硬报错已随 pass 门一并移除。 */
            uint32_t bias_s = 0x8000u | em.ensure_weight_slot(gp, bias_op, wslots, od, false);
            uint32_t w_use = w_s, fl = flags;
            uint32_t pp = try_prepack(0u, nn);
            if (pp) { w_use = 0x8000u | pp; fl |= kFlagWgtPP; }
            em.add_op(OP_MATMUL_F16_BIAS, {a_t, w_use, out_t, m, k, nn, fl, bias_s});
            return 0;
        }
        {
            uint32_t w_use = w_s, fl = flags;
            /* N=16 门缝: FC 权重 N<32 且 M/K 32 对齐时, 编译期把权重槽从
             * [N,K] 零填到 [round_up(N,32),K], flags bit4 = n_pad 标记。运行期
             * exec_matmul_f16_hmx 以 N_compute=round_up(N_full,32) 走 HMX, 输出
             * 窄解包只写前 N_full 列 (N_full<N 即识别 n_pad)。0.8B 36 个 SSM
             * in_proj_z FC (M=32 K=1024 N=16) 由此 scalar 11.26ms/op → HMX。
             * 仅 FC: t1=1 权重 [N,K] 行主序, 运行期 pack src=n*K+k 与 N_full 无关
             * (零列 padding 干净); MatMul t1=0 权重 [K,N] pack src 用 N_full 作
             * stride, 与 [K,32] 满形 padding 冲突, 故 n_pad 仅限 FC。 */
            if (nm == "FullyConnected" && !has_bias && (w_s & 0x8000u) &&
                (m % 32u) == 0u && (k % 32u) == 0u && nn > 0u && nn < 32u) {
                uint32_t src = w_s & 0x7FFFu;
                if (src < em.slots.size() && em.slots[src].len == (size_t)k * nn * 2u) {
                    const wt_slot& s = em.slots[src];
                    const uint16_t* wbytes = nullptr;
                    if (s.addr == WT_SLOT_EXT_WGT) {
                        if (s.offset + s.len <= em.ext_weight_area.size())
                            wbytes = (const uint16_t*)(em.ext_weight_area.data() + s.offset);
                    } else {
                        if (s.offset + s.len <= em.weight_area.size())
                            wbytes = (const uint16_t*)(em.weight_area.data() + s.offset);
                    }
                    if (wbytes) {
                        uint32_t np = (nn + 31u) & ~31u;
                        std::vector<uint16_t> padded((size_t)np * k, 0u);
                        for (uint32_t i = 0; i < nn; ++i)
                            std::memcpy(&padded[(size_t)i * k], wbytes + (size_t)i * k,
                                        (size_t)k * 2u);
                        uint32_t ps = (s.addr == WT_SLOT_EXT_WGT)
                            ? em.add_ext_slot(np * k * 2u, np * k, (const uint8_t*)padded.data())
                            : em.add_slot(np * k * 2u, np * k, (const uint8_t*)padded.data());
                        w_use = 0x8000u | ps;
                        fl |= 16u;  /* flags bit4 = n_pad */
                    }
                }
            }
            uint32_t pp = try_prepack(0u, nn);
            if (pp) { w_use = 0x8000u | pp; fl |= kFlagWgtPP; }
            em.add_op(OP_MATMUL_F16, {a_t, w_use, out_t, m, k, nn, fl});
        }
    }
    return 0;
}

} // namespace

static const wtop::OpRegistrar g_reg_op_matmul{
    {"FullyConnected", op_matmul},
    {"MatMul", op_matmul},
};
