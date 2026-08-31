// wtop_emit: GEHTP tagged .bin → WTOP blob (阶段 8, host 闭环枢纽)
//
// 用法:
//   wtop_emit --bin <tagged.bin> [--input-f16 <X.f16.raw>]
//             --out <blob.wtop> [--manifest <manifest.json>]
//
// 流程: deserialize .bin → 按 plan_order 遍历 → 逐 op 发射 WTOP:
//   Transpose        → OP_TRANSPOSE_F16
//   Conv2d           → 每 tile: OP_IM2COL(切片+halo) + OP_CONV2D_F16(纯 GEMM)
//   Eltwise_Binary   → OP_ADD_F16
//   溢出张量(SF 记录)→ OP_SPILL / OP_FILL(见 --spill-pool 说明)
// 受支持几何门: conv sh=sw=1、dh=dw=1、group=1、same-pad; 不满足报错退出
// (特性门非模型特判; 字段已参数化, 扩支持 = 解门)。
//
// 权重: 池内 [kh,kw,ci,co] f32 → f16(RNE)K×N 进 slot; 输入 slot 0 由
// --input-f16 提供(Level 1 外部槽: 阶段 9 引擎把该槽标为 external)。
#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/serialize/serializer.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// WTOP 契约(kernels/include/oplist_parse.h 的编译期副本语义; 头从 kernels 取)
#include "oplist_parse.h"

using namespace hnnx;

namespace {

struct ConvExtraInfoFixed {
    uint32_t sh, sw, ph_begin, ph_end, pw_begin, pw_end, dh, dw, group, kh, kw;
    uint64_t weight_src, bias_src;
};

// f32 → f16 (round-to-nearest-even, 软件实现; 与 numpy astype 语义一致)
uint16_t f32_to_f16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF) {  // Inf/NaN
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (((u >> 23) & 0xFF) == 0) {  // 零/次正规 f32
        return (uint16_t)sign;      // 次正规 f32 → 0 (量级远小于 f16 最小次正规的边角忽略)
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);  // 溢出 → Inf
    if (exp <= 0) {  // f16 次正规
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int32_t shift = 14 - exp;
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        half += (rem > (1u << (shift - 1))) || (rem == (1u << (shift - 1)) && (half & 1));
        return (uint16_t)(sign | half);
    }
    uint32_t half = (mant >> 13) & 0x3FFu;
    uint32_t rem = mant & 0x1FFFu;
    half += (rem > 0x1000u) || (rem == 0x1000u && (half & 1));
    if (half == 0x400u) { exp++; half = 0; }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half);
}

bool load_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz > 0 ? (size_t)sz : 0);
    if (sz > 0 && std::fread(out.data(), 1, (size_t)sz, f) != (size_t)sz) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

struct Emitter {
    std::vector<uint8_t> blob;
    std::vector<uint8_t> weight_area;   // 128B 对齐起点相对 blob 尾部
    std::vector<wt_slot> slots;
    std::vector<wt_op> ops;
    std::map<uint64_t, uint32_t> op_temp;  // op_id → temp id
    uint32_t next_temp = 0;

    uint32_t fresh_temp(uint64_t op_id) {
        uint32_t t = next_temp++;
        op_temp[op_id] = t;
        if (next_temp > 8) {
            std::fprintf(stderr, "error: >8 live temps (阶段8 契约上限)\n");
            std::exit(4);
        }
        return t;
    }
    // 输入节点的"数据源"= slot 0(0x8000|slot 编码, 见 oplist_parse.h 契约)
    uint32_t src_ref(uint64_t op_id, uint64_t input_node_id) {
        if (op_id == input_node_id) return 0x8000u | 0u;
        return temp_of(op_id);
    }
    uint32_t temp_of(uint64_t op_id) {
        auto it = op_temp.find(op_id);
        if (it == op_temp.end()) { std::fprintf(stderr, "error: no temp for op %llu\n",
                                                 (unsigned long long)op_id); std::exit(4); }
        return it->second;
    }
    uint32_t add_slot(uint32_t len, uint32_t count, const uint8_t* data) {
        wt_slot s{};
        s.len = len;
        s.count = count;
        s.offset = (uint32_t)weight_area.size();   // 128B 对齐由上层保证(每 slot 128 对齐写入)
        s.addr = 0;
        weight_area.insert(weight_area.end(), data, data + len);
        while (weight_area.size() % 128 != 0) weight_area.push_back(0);
        uint32_t id = (uint32_t)slots.size();
        slots.push_back(s);
        return id;
    }
    void add_op(uint16_t opcode, std::initializer_list<uint32_t> args) {
        wt_op o{};
        o.opcode = opcode;
        o.n_args = (uint16_t)args.size();
        uint32_t i = 0;
        for (uint32_t a : args) o.args[i++] = a;
        ops.push_back(o);
    }
};

} // namespace

int emit(const std::string& bin_path, const std::string& in_f16_path,
         const std::string& out_path, const std::string& manifest_path) {
    // 1. deserialize .bin
    std::vector<uint8_t> bin;
    if (!load_file(bin_path, bin)) { std::fprintf(stderr, "error: cannot open %s\n", bin_path.c_str()); return 2; }
    GraphPrepare gp;
    if (!gp.deserialize(bin.data(), bin.size())) { std::fprintf(stderr, "error: deserialize failed\n"); return 2; }

    // 2. 图级信息
    const OpDef* input_op = gp.get_op_at(gp.get_input_node_id());
    if (!input_op || input_op->output_def.rank < 4) { std::fprintf(stderr, "error: bad input node\n"); return 2; }
    uint32_t H = (uint32_t)input_op->output_def.dims[1];
    uint32_t W = (uint32_t)input_op->output_def.dims[2];
    uint32_t C = (uint32_t)input_op->output_def.dims[3];
    size_t input_elems = (size_t)H * W * C;

    Emitter em;

    // slot 0 = 输入 f16(NCHW); Level 1: 阶段9 引擎将其标为 external
    std::vector<uint8_t> in_f16;
    if (!in_f16_path.empty()) {
        if (!load_file(in_f16_path, in_f16)) { std::fprintf(stderr, "error: cannot open %s\n", in_f16_path.c_str()); return 2; }
        if (in_f16.size() != input_elems * 2) {
            std::fprintf(stderr, "error: input f16 size %zu != %zu\n", in_f16.size(), input_elems * 2);
            return 2;
        }
    } else {
        in_f16.assign(input_elems * 2, 0);
    }
    em.add_slot((uint32_t)in_f16.size(), (uint32_t)input_elems, in_f16.data());
    em.slots[0].addr = WT_SLOT_EXT_IN;  // Level 1: 输入槽标外部(wt_exec_run_io 注入)

    // 3. 权重槽(先扫 conv 的 W/B const)
    uint32_t w_slot = 0, b_slot = 0;
    bool have_w = false;
    {
        const OpDef* conv = nullptr;
        for (op_id_t id : gp.plan_order()) {
            const OpDef* od = gp.get_op_at(id);
            if (od && od->name_tag && std::string(od->name_tag->name() ? od->name_tag->name() : "") == "Conv2d")
                { conv = od; break; }
        }
        if (!conv) { std::fprintf(stderr, "error: no Conv2d in graph\n"); return 2; }
        if (conv->inputs.size() < 2) { std::fprintf(stderr, "error: conv missing weight input\n"); return 2; }
        const OpDef* w = gp.get_op_at(conv->inputs[1].src_id);
        if (!w) { std::fprintf(stderr, "error: weight const missing\n"); return 2; }
        // 元素数取 const_data_size/4(prepare 会把 rank-1 张量归一化成
        // rank-4 [1,1,1,N], dims 不可靠; 池内 f32 恒 4B/元素)
        size_t n_w = (size_t)w->const_data_size / 4;
        std::vector<uint16_t> w16(n_w);
        const float* wf = reinterpret_cast<const float*>(gp.const_pool().data() + w->const_data_offset);
        for (size_t i = 0; i < n_w; i++) w16[i] = f32_to_f16_rne(wf[i]);
        std::vector<uint8_t> wbytes(n_w * 2);
        std::memcpy(wbytes.data(), w16.data(), wbytes.size());
        w_slot = em.add_slot((uint32_t)wbytes.size(), (uint32_t)n_w, wbytes.data());
        have_w = true;

        if (conv->inputs.size() > 2) {
            const OpDef* b = gp.get_op_at(conv->inputs[2].src_id);
            if (b) {
                size_t n_b = (size_t)b->const_data_size / 4;  // 同上, size-based
                std::vector<uint16_t> b16(n_b);
                const float* bf = reinterpret_cast<const float*>(gp.const_pool().data() + b->const_data_offset);
                for (size_t i = 0; i < n_b; i++) b16[i] = f32_to_f16_rne(bf[i]);
                std::vector<uint8_t> bbytes(n_b * 2);
                std::memcpy(bbytes.data(), b16.data(), bbytes.size());
                b_slot = em.add_slot((uint32_t)bbytes.size(), (uint32_t)n_b, bbytes.data());
            }
        }
        (void)have_w;
    }

    // 4. 按 plan_order 发射 op
    std::vector<op_id_t> order = gp.plan_order();
    if (order.empty()) {
        for (op_id_t id = 1; id <= 10; id++) order.push_back(id);  // 兜底(常规图 id 1..10)
    }
    uint32_t spill_pool_slot = 0;  // 0 = 未创建
    for (op_id_t id : order) {
        const OpDef* od = gp.get_op_at(id);
        if (!od || od->is_const() || !od->name_tag) continue;
        std::string nm = od->name_tag->name() ? od->name_tag->name() : "";
        if (nm == "Input" || nm == "Output" || nm.empty()) continue;

        if (nm == "Transpose") {
            const OpDef* src = gp.get_op_at(od->inputs[0].src_id);
            if (!src) { std::fprintf(stderr, "error: transpose src missing\n"); return 2; }
            const OpDef* permc = (od->inputs.size() > 1) ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
            uint32_t perm = 0x00010203u;  // 缺省: 单位(字节序: 轴0..3 各 1 字节)
            if (permc && permc->const_data_size >= 16) {
                const int32_t* p = reinterpret_cast<const int32_t*>(
                    gp.const_pool().data() + permc->const_data_offset);
                perm = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            }
            uint32_t src_t = em.src_ref(src->op_id, gp.get_input_node_id());
            uint32_t out_t = em.fresh_temp(od->op_id);
            em.add_op(OP_TRANSPOSE_F16, {src_t, out_t, H, W, C, perm});
            continue;
        }

        if (nm == "Conv2d") {
            // extra_info: 60B fixed + tiling 段
            if (od->serialized_extra.size() < sizeof(ConvExtraInfoFixed) + 16) {
                std::fprintf(stderr, "error: conv extra too short\n"); return 2;
            }
            ConvExtraInfoFixed e;
            std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t hdr[4];
            std::memcpy(hdr, od->serialized_extra.data() + sizeof(ConvExtraInfoFixed), 16);

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
                od->serialized_extra.data() + sizeof(ConvExtraInfoFixed) + 16);
            if (num_tiles == 0) {  // 旧流/未分块: 整图单 tile
                num_tiles = 1;
            }
            // 无分块时的整图 tile 描述(字段序同 ConvTileDesc 19×u32)
            const uint32_t full_desc[19] = {0, 0, H, W, 0, 0, H, W,
                                            e.kh, e.kw, e.sh, e.sw, e.ph_begin, e.pw_begin,
                                            C, C, 0, C, 0};
            uint32_t src_t = em.src_ref(od->inputs[0].src_id, gp.get_input_node_id());
            uint32_t out_t = em.fresh_temp(od->op_id);
            uint32_t cols_t = em.fresh_temp(0xFFFFFFF0);  // 专用 cols 槽
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t* d = (hdr[3] == 0) ? full_desc : descs + t * 19;
                uint32_t iy0 = d[4], ix0 = d[5], ih = d[6], iw = d[7];
                uint32_t oy0 = d[0], ox0 = d[1], th = d[2], tw = d[3];
                uint32_t co0 = d[16], co_n = d[17];
                // im2col: 输入切片(含 halo) → cols [th*tw × K]
                em.add_op(OP_IM2COL, {src_t, cols_t, H, W, C, e.kh, e.kw,
                                      e.ph_begin, e.pw_begin, e.sh, e.sw,
                                      iy0, ix0, th, tw});
                // GEMM: cols [M,K] @ W [K,N] → 输出 tile 写入全图 out_temp
                em.add_op(OP_CONV2D_F16, {cols_t, w_slot, b_slot, out_t,
                                          th * tw, e.kh * e.kw * C, C,
                                          oy0, ox0, H, W, co0, co_n});
            }
            continue;
        }

        if (nm == "Eltwise_Binary") {
            uint32_t a_t = em.src_ref(od->inputs[0].src_id, gp.get_input_node_id());
            uint32_t b_t = em.src_ref(od->inputs[1].src_id, gp.get_input_node_id());
            uint32_t out_t = em.fresh_temp(od->op_id);
            em.add_op(OP_ADD_F16, {a_t, b_t, out_t, H * W * C});
            continue;
        }

        std::fprintf(stderr, "error: unsupported op '%s' in wtop_emit\n", nm.c_str());
        return 4;
    }

    // 5. spill/fill → OP_SPILL / OP_FILL(溢出张量; pool slot 惰性创建)
    for (const auto& r : gp.spill_fill_recs()) {
        if (spill_pool_slot == 0) {
            // DDR 池 slot: 尺寸 = 最大 ddr_offset+size, 128B 对齐
            uint64_t pool_end = 0;
            for (const auto& r2 : gp.spill_fill_recs())
                pool_end = std::max(pool_end, r2.ddr_offset + r2.size);
            pool_end = (pool_end + 127) & ~uint64_t(127);
            std::vector<uint8_t> zeros((size_t)pool_end, 0);
            spill_pool_slot = em.add_slot((uint32_t)pool_end, (uint32_t)pool_end / 2, zeros.data());
        }
        uint32_t t;
        if (r.op_id == gp.get_output_node_id()) {
            // Output 节点的张量 = 其输入生产者的 temp
            const OpDef* out = gp.get_op_at(r.op_id);
            t = (out && !out->inputs.empty())
                ? em.src_ref(out->inputs[0].src_id, gp.get_input_node_id())
                : 0x8000u;
        } else {
            t = em.src_ref(r.op_id, gp.get_input_node_id());
        }
        // SPILL: 张量 → 池; FILL: 池 → 张量(设备执行序由引擎按 op 序串行;
        // 输入节点的张量经 0x8000|slot 编码引用)
        em.add_op(OP_SPILL, {t, spill_pool_slot, (uint32_t)r.ddr_offset, (uint32_t)(r.size / 2)});
        em.add_op(OP_FILL, {spill_pool_slot, (uint32_t)r.ddr_offset, t, (uint32_t)(r.size / 2)});
    }

    // 6. 组装 blob
    {
        uint32_t n_slots = (uint32_t)em.slots.size();
        uint32_t n_ops = (uint32_t)em.ops.size();
        em.blob.resize(16 + n_slots * 16 + 1, 0);
        std::memcpy(em.blob.data(), "WTOP", 4);
        uint16_t ver = WT_BLOB_VER, eck = WT_ENDIAN_CHK;
        std::memcpy(em.blob.data() + 4, &ver, 2);
        std::memcpy(em.blob.data() + 6, &eck, 2);
        std::memcpy(em.blob.data() + 8, &n_slots, 4);
        std::memcpy(em.blob.data() + 12, &n_ops, 4);
        for (uint32_t i = 0; i < n_slots; i++) {
            uint8_t* s = em.blob.data() + 16 + i * 16;
            std::memcpy(s, &em.slots[i].len, 4);
            std::memcpy(s + 4, &em.slots[i].count, 4);
            std::memcpy(s + 8, &em.slots[i].offset, 4);
            std::memcpy(s + 12, &em.slots[i].addr, 4);
        }
        size_t p = 16 + n_slots * 16;
        em.blob.resize(p);
        for (const auto& o : em.ops) {
            size_t sz = 4 + o.n_args * 4;
            size_t old = em.blob.size();
            em.blob.resize(old + sz);
            std::memcpy(em.blob.data() + old, &o.opcode, 2);
            std::memcpy(em.blob.data() + old + 2, &o.n_args, 2);
            for (uint16_t a = 0; a < o.n_args; a++)
                std::memcpy(em.blob.data() + old + 4 + a * 4, &o.args[a], 4);
            p += sz;
        }
        size_t woff = (p + 127) & ~(size_t)127;
        em.blob.resize(woff + em.weight_area.size(), 0);
        std::memcpy(em.blob.data() + woff, em.weight_area.data(), em.weight_area.size());
    }

    // 7. 自校验 + 落盘
    {
        wt_blob wb{};
        int rc = wt_parse(em.blob.data(), em.blob.size(), &wb);
        if (rc != WT_OK) {
            std::fprintf(stderr, "error: self-validate failed: %s\n", wt_err_str(rc));
            return 3;
        }
        std::printf("WTOP OK: slots=%u ops=%u bytes=%zu\n", (unsigned)em.slots.size(), (unsigned)em.ops.size(), em.blob.size());
    }
    {
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str()); return 2; }
        std::fwrite(em.blob.data(), 1, em.blob.size(), f);
        std::fclose(f);
    }

    // 8. manifest
    if (!manifest_path.empty()) {
        FILE* f = std::fopen(manifest_path.c_str(), "w");
        if (f) {
            std::fprintf(f, "{\n");
            std::fprintf(f, "  \"input_slot\": 0,\n");
            std::fprintf(f, "  \"input_elems\": %zu,\n", input_elems);
            std::fprintf(f, "  \"output_temp\": %u,\n", em.next_temp > 0 ? em.next_temp - 1 : 0);
            std::fprintf(f, "  \"n_slots\": %u,\n  \"n_ops\": %u,\n",
                         (unsigned)em.slots.size(), (unsigned)em.ops.size());
            std::fprintf(f, "  \"opcodes\": [");
            for (size_t i = 0; i < em.ops.size(); i++)
                std::fprintf(f, "%s%u", i ? "," : "", (unsigned)em.ops[i].opcode);
            std::fprintf(f, "]\n}\n");
            std::fclose(f);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string bin_path, in_f16, out_path, manifest_path;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--bin") bin_path = next();
        else if (a == "--input-f16") in_f16 = next();
        else if (a == "--out") out_path = next();
        else if (a == "--manifest") manifest_path = next();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (bin_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: wtop_emit --bin <tagged.bin> [--input-f16 <f16.raw>] --out <blob.wtop> [--manifest <json>]\n");
        return 2;
    }
    return emit(bin_path, in_f16, out_path, manifest_path);
}
