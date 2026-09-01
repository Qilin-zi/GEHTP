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
#include "hnnx/ir/op_extra.hpp"
#include "hnnx/ir/types.hpp"
#include "hnnx/serialize/serializer.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
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

// GGUF 匹配条目(TSV 行; M2c)
struct GgufEntry { uint64_t file_offset; uint64_t nbytes; uint32_t type; std::vector<uint64_t> dims; };

// 字符串按分隔符切分
static std::vector<std::string> split_str(const std::string& st, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= st.size()) {
        size_t pp = st.find(sep, start);
        out.push_back(st.substr(start, pp == std::string::npos ? std::string::npos : pp - start));
        if (pp == std::string::npos) break;
        start = pp + 1;
    }
    return out;
}

// f16 → f32(反量化用)
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) { u = sign; }
        else {
            int e = -1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            u = sign | ((uint32_t)(127 + 15 + e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Emitter;

struct Emitter {
    std::vector<uint8_t> blob;
    std::vector<uint8_t> weight_area;   // 128B 对齐起点相对 blob 尾部
    std::vector<wt_slot> slots;
    std::vector<wt_op> ops;
    std::map<uint64_t, uint32_t> op_temp;  // (op_id<<32 | out_idx) → temp id
    uint32_t next_temp = 0;
    std::vector<uint32_t> free_temps;      // 活性分析释放的 temp(M2)
    std::map<uint64_t, uint32_t> last_use; // 复合键 → 最后消费位置(plan_order 下标)

    static uint64_t tkey(uint64_t op_id, uint32_t out_idx) {
        return (op_id << 32) | out_idx;
    }
    // 活性分析: 每条 temp 的生命期 = 生产位置 .. 最后消费位置
    void compute_last_use(const std::vector<op_id_t>& order, const GraphPrepare& gp) {
        for (size_t pos = 0; pos < order.size(); pos++) {
            const OpDef* od = gp.get_op_at(order[pos]);
            if (!od) continue;
            for (const auto& c : od->inputs)
                last_use[tkey(c.src_id, c.out_idx)] = static_cast<uint32_t>(pos);
        }
    }
    uint64_t dbg_released = 0, dbg_calls = 0, dbg_miss = 0, dbg_kept = 0;
    void release_at(uint64_t key, size_t pos) {
        dbg_calls++;
        auto it = op_temp.find(key);
        if (it == op_temp.end()) { dbg_miss++; return; }
        auto lu = last_use.find(key);
        if (lu != last_use.end() && lu->second > pos) { dbg_kept++; return; }  // 还有消费者
        free_temps.push_back(it->second);
        op_temp.erase(it);
        dbg_released++;
    }
    uint32_t fresh_temp(uint64_t op_id, uint32_t out_idx = 0) {
        uint32_t t;
        if (!free_temps.empty()) { t = free_temps.back(); free_temps.pop_back(); }
        else { t = next_temp++; }
        op_temp[tkey(op_id, out_idx)] = t;
        if (next_temp > 32768) {
            std::fprintf(stderr, "error: >32768 live temps (live=%zu free=%zu released=%llu calls=%llu miss=%llu kept=%llu)\n",
                         op_temp.size(), free_temps.size(),
                         (unsigned long long)dbg_released, (unsigned long long)dbg_calls,
                         (unsigned long long)dbg_miss, (unsigned long long)dbg_kept);
            { int n = 0;
              for (auto& [k, t] : op_temp) {
                auto lu = last_use.find(k);
                std::fprintf(stderr, "  key op=%llu out=%u -> temp=%u lu=%u\n",
                             (unsigned long long)(k >> 32), (unsigned)(k & 0xFFFFFFFFu), t,
                             lu != last_use.end() ? lu->second : 0xFFFFFFFFu);
                if (++n >= 8) break;
              } }
            std::exit(4);
        }
        return t;
    }
// GGUF 供给上下文(M2c; emit 启动时设置)
    const std::map<std::string, GgufEntry>* gguf_tab = nullptr;
    const std::vector<uint8_t>* gguf_bytes = nullptr;
    bool use_gguf = false;
    uint32_t gguf_hits = 0, gguf_miss = 0;

    // GGUF Q4_0 → tile-major repack(tile 契约 = 例15 ref_dequant_q4_0_tile):
    // tile 640B = [4×128B 列对 quants][32 fp16 scales@+512][pad];
    // 列对(c,c+1) 32B: byte r = lo(列c nib r) | hi(列c+1 nib r)<<4;
    // GGUF 块(n 列, kt)= 18B [2B d][16B nibbles: 元素0..15=lo, 16..31=hi]。
    // tile 的 per-row scale 与 GGUF per-block d 分组不同 → 反量化 f32 后按行重量化。
    static std::vector<uint8_t> repack_q4_0_tiles(const uint8_t* src, size_t nbytes,
                                                  size_t K, size_t N) {
        size_t n_k_tiles = K / 32, n_col_tiles = N / 32;
        std::vector<float> w(K * N);
        for (size_t b = 0; b < nbytes / 18; b++) {
            uint16_t d_raw;
            std::memcpy(&d_raw, src + b * 18, 2);
            float d = f16_to_f32(d_raw);
            const uint8_t* nib = src + b * 18 + 2;
            for (int j = 0; j < 16; j++) {
                w[(b * 32) + j] = d * ((nib[j] & 0xF) - 8);
                w[(b * 32) + 16 + j] = d * ((nib[j] >> 4) - 8);
            }
        }
        std::vector<uint8_t> out(n_col_tiles * n_k_tiles * 640, 0);
        for (size_t ct = 0; ct < n_col_tiles; ct++)
            for (size_t kt = 0; kt < n_k_tiles; kt++) {
                uint8_t* tile = out.data() + (ct * n_k_tiles + kt) * 640;
                for (size_t r = 0; r < 32; r++) {
                    float mx = 0.0f;
                    for (size_t c = 0; c < 32; c++)
                        mx = std::max(mx, std::fabs(w[(kt * 32 + r) * N + ct * 32 + c]));
                    float d = mx / 7.0f;
                    uint16_t d16 = f32_to_f16_rne(d);
                    std::memcpy(tile + 512 + r * 2, &d16, 2);
                    for (size_t c = 0; c < 32; c++) {
                        float v = w[(kt * 32 + r) * N + ct * 32 + c];
                        int q = (int)std::lround(v / d) + 8;
                        if (q < 0) q = 0;
                        if (q > 15) q = 15;
                        uint8_t& byte = tile[(c / 2) * 32 + r];
                        if (c & 1) byte |= (uint8_t)(q << 4);
                        else byte = (uint8_t)q;
                    }
                }
            }
        return out;
    }

    // 权重消费 op 的 const 输入 → 权重槽(按 const id 去重; GGUF 供给优先)
    // consumer_grp = 消费节点的原始名(loader 只给计算 op 设 grouping,
    // 权重 const 自身没有; TSV 键 = 消费节点名)
    uint32_t ensure_weight_slot(GraphPrepare& gp, const OpDef* w,
                                std::map<uint64_t, uint32_t>& wslots,
                                const std::string& consumer_grp = "") {
        auto it = wslots.find(w->op_id);
        if (it != wslots.end()) return it->second;
        if (use_gguf && gguf_tab && gguf_bytes) {
            auto mit = gguf_tab->find(w->grouping);
            if (mit == gguf_tab->end() && !consumer_grp.empty())
                mit = gguf_tab->find(consumer_grp);
            if (mit != gguf_tab->end()) {
                const GgufEntry& e = mit->second;
                if (e.file_offset + e.nbytes <= gguf_bytes->size()) {
                    const uint8_t* src = gguf_bytes->data() + e.file_offset;
                    uint32_t id;
                    size_t K = e.dims[0], NN = 1;
                    for (size_t i = 1; i < e.dims.size(); i++) NN *= e.dims[i];
                    if (e.type == 2 && K % 32 == 0 && NN % 32 == 0) {
                        // Q4_0 → tile-major(32×32 tile 契约)
                        auto tile = repack_q4_0_tiles(src, e.nbytes, K, NN);
                        id = add_slot((uint32_t)tile.size(), (uint32_t)(K * NN), tile.data());
                    } else if (e.type == 2) {
                        // 小维度(<32)不走 tile: 反量化 f16 直存
                        std::vector<float> wf2(K * NN);
                        for (size_t b = 0; b < e.nbytes / 18; b++) {
                            uint16_t d_raw;
                            std::memcpy(&d_raw, src + b * 18, 2);
                            float d = f16_to_f32(d_raw);
                            const uint8_t* nib = src + b * 18 + 2;
                            for (int j = 0; j < 16; j++) {
                                wf2[(b * 32) + j] = d * ((nib[j] & 0xF) - 8);
                                wf2[(b * 32) + 16 + j] = d * ((nib[j] >> 4) - 8);
                            }
                        }
                        std::vector<uint16_t> w16(K * NN);
                        for (size_t i = 0; i < K * NN; i++) w16[i] = f32_to_f16_rne(wf2[i]);
                        std::vector<uint8_t> wb(K * NN * 2);
                        std::memcpy(wb.data(), w16.data(), wb.size());
                        id = add_slot((uint32_t)wb.size(), (uint32_t)(K * NN), wb.data());
                    } else {  // F32 → f16
                        size_t n4 = e.nbytes / 4;
                        std::vector<uint16_t> w16(n4);
                        const float* wf = reinterpret_cast<const float*>(src);
                        for (size_t i = 0; i < n4; i++) w16[i] = f32_to_f16_rne(wf[i]);
                        std::vector<uint8_t> wb(n4 * 2);
                        std::memcpy(wb.data(), w16.data(), wb.size());
                        id = add_slot((uint32_t)wb.size(), (uint32_t)n4, wb.data());
                    }
                    wslots[w->op_id] = id;
                    gguf_hits++;
                    return id;
                }
            }
            gguf_miss++;
        }
        size_t n = (size_t)w->const_data_size / 4;
        if (n == 0) {
            // 空 const(记录在但池数据被 DCE 删): 最小零槽占位(M4 数值门兜底)
            std::fprintf(stderr, "warn: empty weight const (op %llu), 零槽占位\n",
                         (unsigned long long)w->op_id);
            std::vector<uint8_t> zeros(128, 0);
            uint32_t id = add_slot(128, 64, zeros.data());
            wslots[w->op_id] = id;
            return id;
        }
        std::vector<uint16_t> w16(n);
        const float* wf = reinterpret_cast<const float*>(gp.const_pool().data() + w->const_data_offset);
        for (size_t i = 0; i < n; i++) w16[i] = f32_to_f16_rne(wf[i]);
        std::vector<uint8_t> wbytes(n * 2);
        std::memcpy(wbytes.data(), w16.data(), wbytes.size());
        uint32_t id = add_slot((uint32_t)wbytes.size(), (uint32_t)n, wbytes.data());
        wslots[w->op_id] = id;
        return id;
    }
    // 输入节点 = slot 0; const 生产者 = 权重槽(0x8000|slot 编码);
    // 幻影占位(空名非 const, 如第二输入 position_ids) = 专用零槽
    // (M2 结构闭环; M4 改真注入: 幻影槽标 EXT_IN 由 host 供给)
    uint32_t dummy_slot_id = 0;  // 共享哑槽(幻影/参数 const/缺失生产者引用)
    uint32_t src_ref(const InputConn& c, uint64_t input_node_id, GraphPrepare& gp,
                     std::map<uint64_t, uint32_t>& wslots) {
        uint64_t op_id = c.src_id;
        if (op_id == input_node_id) return 0x8000u | 0u;
        const OpDef* p = gp.get_op_at(op_id);
        // 有池数据即按 const 槽(is_const 标志对 loader 的 tensor_param const 不可靠)
        if (p && (p->is_const() || p->const_data_size > 0)) {
            uint32_t s = ensure_weight_slot(gp, p, wslots);
            return 0x8000u | s;
        }
        bool param_const_name = false;
        if (p && p->name_tag && p->name_tag->name()) {
            const char* nm2 = p->name_tag->name();
            static const char* suf[] = {"_shape", "_axes", "_pad_amount", "_ranges"};
            for (const char* sfx : suf) {
                size_t l = std::strlen(sfx), n2 = std::strlen(nm2);
                if (n2 >= l && std::strcmp(nm2 + n2 - l, sfx) == 0) param_const_name = true;
            }
        }
        if (!p || param_const_name || !p->name_tag || !p->name_tag->name() || !p->name_tag->name()[0]) {
            // 生产者缺失(DCE 已删)/参数 const/幻影占位 → 共享哑槽(惰性创建,
            // 保持无幻影图的槽布局不变) (M2 结构闭环; M4 真注入)
            if (dummy_slot_id == 0) {
                std::vector<uint8_t> zeros(128, 0);
                dummy_slot_id = add_slot(128, 64, zeros.data());
            }
            return 0x8000u | dummy_slot_id;
        }
        return temp_of_dbg(op_id, c.out_idx, gp);
    }
    uint32_t temp_of(uint64_t op_id, uint32_t out_idx = 0) {
        auto it = op_temp.find(tkey(op_id, out_idx));
        if (it == op_temp.end()) { std::fprintf(stderr, "error: no temp for op %llu\n",
                                                 (unsigned long long)op_id); std::exit(4); }
        return it->second;
    }
    uint64_t cur_op = 0;
    uint32_t temp_of_dbg(uint64_t op_id, uint32_t out_idx, GraphPrepare& gp) {
        auto it = op_temp.find(tkey(op_id, out_idx));
        if (it == op_temp.end()) {
            const OpDef* od = gp.get_op_at(op_id);
            const OpDef* cur = gp.get_op_at(cur_op);
            std::fprintf(stderr, "error: no temp for op %llu (%s rank=%u is_const=%d const_size=%zu); consumer op %llu (%s)\n",
                         (unsigned long long)op_id,
                         od && od->name_tag && od->name_tag->name() ? od->name_tag->name() : "?",
                         od ? od->output_def.rank : 0,
                         od ? (od->is_const() ? 1 : 0) : 0,
                         od ? od->const_data_size : 0,
                         (unsigned long long)cur_op,
                         cur && cur->name_tag && cur->name_tag->name() ? cur->name_tag->name() : "?");
            std::exit(4);
        }
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
    void add_op(uint16_t opcode, const std::vector<uint32_t>& args) {
        wt_op o{};
        o.opcode = opcode;
        o.n_args = (uint16_t)args.size();
        uint32_t i = 0;
        for (uint32_t a : args) o.args[i++] = a;
        ops.push_back(o);
    }
};

}

// 权重消费 op 的 const 输入下标(weight input 位置)
static int weight_input_index(const std::string& nm) {
    if (nm == "FullyConnected" || nm == "MatMul" || nm == "Conv2d" ||
        nm == "DepthWiseConv2d" || nm == "RmsNorm") return 1;
    if (nm == "Gather") return 0;  // embedding 表
    return -1;
}

// op 输出元素数(output_def 全维乘积)
static uint64_t elems_of(const OpDef* od) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < od->output_def.rank && i < 5; ++i)
        n *= (uint64_t)od->output_def.dims[i];
    return n;
}

int emit(const std::string& bin_path, const std::string& in_f16_path,
         const std::string& out_path, const std::string& manifest_path,
         const std::string& gguf_path, const std::string& match_path) {
    // 1. deserialize .bin
    std::vector<uint8_t> bin;
    if (!load_file(bin_path, bin)) { std::fprintf(stderr, "error: cannot open %s\n", bin_path.c_str()); return 2; }
    GraphPrepare gp;
    if (!gp.deserialize(bin.data(), bin.size())) { std::fprintf(stderr, "error: deserialize failed\n"); return 2; }

    // 2. 图级信息
    const OpDef* input_op = gp.get_op_at(gp.get_input_node_id());
    if (!input_op || input_op->output_def.rank == 0) { std::fprintf(stderr, "error: bad input node\n"); return 2; }
    // 输入形状/元素数: 任意 rank; dtype 决定字节/元素(f32/f16→2B 外部, int32→4B)
    size_t input_elems = 1;
    for (uint32_t i = 0; i < input_op->output_def.rank && i < 5; ++i)
        input_elems *= (size_t)input_op->output_def.dims[i];
    uint32_t input_dtype = input_op->output_def.dtype;
    size_t input_elem_bytes = (input_dtype == 1 || input_dtype == 0x21) ? 2 : 4;
    // NCHW 便捷(conv 路径沿用)
    uint32_t H = input_op->output_def.rank >= 4 ? (uint32_t)input_op->output_def.dims[1] : 1;
    uint32_t W = input_op->output_def.rank >= 4 ? (uint32_t)input_op->output_def.dims[2] : 1;
    uint32_t C = input_op->output_def.rank >= 4 ? (uint32_t)input_op->output_def.dims[3] : 1;

    Emitter em;

    // slot 0 = 输入 f16(NCHW); Level 1: 阶段9 引擎将其标为 external
    std::vector<uint8_t> in_f16;
    size_t input_bytes = input_elems * input_elem_bytes;
    if (!in_f16_path.empty()) {
        if (!load_file(in_f16_path, in_f16)) { std::fprintf(stderr, "error: cannot open %s\n", in_f16_path.c_str()); return 2; }
        if (in_f16.size() != input_bytes) {
            std::fprintf(stderr, "error: input size %zu != %zu (elems=%zu × %zuB)\n",
                         in_f16.size(), input_bytes, input_elems, input_elem_bytes);
            return 2;
        }
    } else {
        in_f16.assign(input_bytes, 0);
    }
    em.add_slot((uint32_t)in_f16.size(), (uint32_t)input_elems, in_f16.data());
    em.slots[0].addr = WT_SLOT_EXT_IN;  // Level 1: 输入槽标外部(wt_exec_run_io 注入)

    // 2b. GGUF 供给(M2c): 读 TSV 匹配表 + GGUF 文件
    std::map<std::string, GgufEntry> gguf_map;
    std::vector<uint8_t> gguf_data;
    bool have_gguf = false;
    if (!gguf_path.empty() && !match_path.empty()) {
        std::ifstream mf(match_path);
        std::string line;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == 'n') continue;
            std::vector<std::string> cols = split_str(line, '\t');
            if (cols.size() < 6) continue;
            GgufEntry e{};
            e.type = (uint32_t)std::strtoul(cols[2].c_str(), nullptr, 10);
            e.file_offset = std::strtoull(cols[3].c_str(), nullptr, 10);
            e.nbytes = std::strtoull(cols[4].c_str(), nullptr, 10);
            for (auto& d : split_str(cols[5], ','))
                e.dims.push_back(std::strtoull(d.c_str(), nullptr, 10));
            gguf_map[cols[0]] = e;
        }
        if (!load_file(gguf_path, gguf_data)) {
            std::fprintf(stderr, "error: cannot open %s\n", gguf_path.c_str());
            return 2;
        }
        have_gguf = true;
        std::printf("[gguf] loaded %zu entries, %zu bytes\n", gguf_map.size(), gguf_data.size());
    }
    if (have_gguf) {
        em.use_gguf = true;
        em.gguf_tab = &gguf_map;
        em.gguf_bytes = &gguf_data;
    }

    // 3. 通用权重槽收集(M2): 按 plan_order 遍历, 对权重消费 op 的 const 输入
    //     建 f16 槽(按 const id 去重, tie 权重共享一槽)
    std::map<uint64_t, uint32_t> wslots;  // const op_id → slot
    {
        std::vector<op_id_t> pre_order = gp.plan_order();
        for (op_id_t id : pre_order) {
            const OpDef* od = gp.get_op_at(id);
            if (!od || od->is_const() || !od->name_tag) continue;
            std::string nm = od->name_tag->name() ? od->name_tag->name() : "";
            int wi = weight_input_index(nm);
            if (wi < 0 || od->inputs.size() <= (size_t)wi) continue;
            const OpDef* w = gp.get_op_at(od->inputs[wi].src_id);
            if (!w || w->const_data_size == 0) continue;
            em.ensure_weight_slot(gp, w, wslots, od->grouping);
        }
    }
    // conv 分支的便捷引用(无 conv 图时为 0, conv 分支不会触发)
    uint32_t w_slot = 0, b_slot = 0;
    {
        const OpDef* conv = nullptr;
        for (op_id_t id : gp.plan_order()) {
            const OpDef* od = gp.get_op_at(id);
            if (od && od->name_tag && std::string(od->name_tag->name() ? od->name_tag->name() : "") == "Conv2d")
                { conv = od; break; }
        }
        if (conv) {
            if (conv->inputs.size() < 2) { std::fprintf(stderr, "error: conv missing weight input\n"); return 2; }
            const OpDef* w = gp.get_op_at(conv->inputs[1].src_id);
            if (!w) { std::fprintf(stderr, "error: weight const missing\n"); return 2; }
            w_slot = em.ensure_weight_slot(gp, w, wslots, conv->grouping);
            if (conv->inputs.size() > 2) {
                const OpDef* b = gp.get_op_at(conv->inputs[2].src_id);
                if (b) b_slot = em.ensure_weight_slot(gp, b, wslots, conv->grouping);
            }
        }
    }

    // 4. 按 plan_order 发射 op
    std::vector<op_id_t> order = gp.plan_order();
    if (order.empty()) {
        for (op_id_t id = 1; id <= 10; id++) order.push_back(id);  // 兜底(常规图 id 1..10)
    }
    em.compute_last_use(order, gp);
    // Output 引用的 temp 必须活到循环外(唯一真实跨循环引用)
    {
        const OpDef* out = gp.get_op_at(gp.get_output_node_id());
        if (out && !out->inputs.empty())
            em.last_use[Emitter::tkey(out->inputs[0].src_id, out->inputs[0].out_idx)] =
                static_cast<uint32_t>(order.size());
    }
    // spill/fill 不钉 temp: 0.8B 默认预算下全图溢出 → 上千 spill 记录,
    // 钉住会把整个活跃集抬到 16879。spill 段(循环后)用保留 temp 引用。
    uint32_t spill_pool_slot = 0;  // 0 = 未创建
    for (size_t pos = 0; pos < order.size(); pos++) {
        op_id_t id = order[pos];
        em.cur_op = id;
        if (pos < 8) {
            const OpDef* dbg = gp.get_op_at(id);
            std::fprintf(stderr, "[dbg] pos=%zu id=%llu name=%s const=%d\n", pos,
                         (unsigned long long)id,
                         dbg && dbg->name_tag && dbg->name_tag->name() ? dbg->name_tag->name() : "?",
                         dbg && dbg->is_const());
        }
        const OpDef* od = gp.get_op_at(id);
        if (!od || od->is_const() || !od->name_tag) continue;
        std::string nm = od->name_tag->name() ? od->name_tag->name() : "";
        if (nm == "Input" || nm == "Output" || nm.empty()) continue;
        // loader 合成的 tensor_param const(非 is_const 标志): *_shape/*_axes/
        // *_pad_amount/*_ranges 后缀, 无发射语义
        {
            static const char* suf[] = {"_shape", "_axes", "_pad_amount", "_ranges"};
            bool is_param = false;
            for (const char* sfx : suf) {
                size_t l = std::strlen(sfx);
                if (nm.size() >= l && nm.compare(nm.size() - l, l, sfx) == 0) is_param = true;
            }
            if (is_param) continue;
        }
        do {

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
            uint32_t src_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            em.add_op(OP_TRANSPOSE_F16, {src_t, out_t, H, W, C, perm});
            break;
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
            uint32_t src_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
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
            break;
        }

        if (nm == "Eltwise_Binary") {
            // operation 从 serialized_extra 读(0=ADD 1=SUB 2=MUL 3=DIV)
            uint32_t subtype = 0;
            if (od->serialized_extra.size() >= sizeof(ExtraEltwise)) {
                ExtraEltwise e;
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
                subtype = e.operation;
            }
            uint32_t a_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t b_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            if (subtype == 0) em.add_op(OP_ADD_F16, {a_t, b_t, out_t, H * W * C});
            else em.add_op(OP_BINARY_F16, {a_t, b_t, out_t, H * W * C, subtype});
            break;
        }

        // ---- M2: 0.8B 20 型新分支(参数全从 serialized_extra 读) ----

        if (nm == "Eltwise_Unary" || nm == "ElementWiseNeuron") {
            ExtraEltwise e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od), e.operation});
            break;
        }

        if (nm == "Eltwise_Ternary") {
            // SELECT 语义: out = in0(cond) ? in1 : in2
            uint32_t c_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t a_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
            uint32_t b_t = em.src_ref(od->inputs[2], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            em.add_op(OP_BINARY_F16, {c_t, a_t, b_t, out_t, 8});  // 8=SELECT(cond,a,b)
            break;
        }

        if (nm == "Softmax") {
            ExtraAxis e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            uint64_t n_elems = elems_of(od);
            uint32_t ax = (e.axis < 0) ? od->output_def.rank - 1 : (uint32_t)e.axis;
            uint64_t row_w = 1;
            if (ax < od->output_def.rank) row_w = od->output_def.dims[ax];
            em.add_op(OP_SOFTMAX_F16, {x_t, out_t, (uint32_t)(n_elems / row_w), (uint32_t)row_w});
            break;
        }

        if (nm == "RmsNorm") {
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
            uint32_t w_s = w ? em.ensure_weight_slot(gp, w, wslots, od->grouping) : em.dummy_slot_id;
            uint32_t out_t = em.fresh_temp(od->op_id);
            em.add_op(OP_RMSNORM_F16, {x_t, w_s, out_t, (uint32_t)elems_of(od)});
            break;
        }

        if (nm == "FullyConnected" || nm == "MatMul") {
            uint32_t a_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
            uint32_t w_s = w ? em.ensure_weight_slot(gp, w, wslots, od->grouping) : em.dummy_slot_id;
            ExtraFc ef{};
            ExtraMatMul em2{};
            uint32_t m = 0, k = 0, nn = 0;
            if (nm == "FullyConnected") {
                if (od->serialized_extra.size() >= sizeof(ef))
                    std::memcpy(&ef, od->serialized_extra.data(), sizeof(ef));
                m = ef.m; k = ef.k; nn = ef.n;
            } else {
                if (od->serialized_extra.size() >= sizeof(em2))
                    std::memcpy(&em2, od->serialized_extra.data(), sizeof(em2));
                m = em2.m; k = em2.k; nn = em2.n;
            }
            if (m == 0 || k == 0 || nn == 0) {
                std::fprintf(stderr, "error: %s extra 未提取 (M/K/N=0)\n", nm.c_str());
                return 4;
            }
            em.add_op(OP_MATMUL_W4A16, {a_t, w_s, out_t, m, k, nn});
            break;
        }

        if (nm == "Gather") {
            const OpDef* tbl = od->inputs.size() > 0 ? gp.get_op_at(od->inputs[0].src_id) : nullptr;
            uint32_t tbl_s = tbl ? em.ensure_weight_slot(gp, tbl, wslots, od->grouping) : em.dummy_slot_id;
            uint32_t idx_t = em.src_ref(od->inputs[1], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            uint32_t row_bytes = 0;
            if (tbl && tbl->output_def.rank >= 1)
                row_bytes = (uint32_t)tbl->output_def.dims[tbl->output_def.rank - 1] * 2;
            em.add_op(OP_GATHER_F16, {tbl_s, idx_t, out_t, (uint32_t)elems_of(od), row_bytes});
            break;
        }

        if (nm == "Reshape" || nm == "Pad" || nm == "ScatterNd" || nm == "Cast") {
            // 数据搬运/常量填充/状态更新语义 → 恒等(设备 M4 数值门兜底)
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            em.add_op(OP_UNARY_F16, {x_t, out_t, (uint32_t)elems_of(od), 0xFFFFFFFFu});
            break;
        }

        if (nm == "DepthWiseConv2d") {
            ExtraDepthwiseConv e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            const OpDef* w = od->inputs.size() > 1 ? gp.get_op_at(od->inputs[1].src_id) : nullptr;
            uint32_t w_s = w ? em.ensure_weight_slot(gp, w, wslots, od->grouping) : em.dummy_slot_id;
            uint32_t out_t = em.fresh_temp(od->op_id);
            uint64_t ne = elems_of(od);
            uint32_t seq = 1, ch = (uint32_t)ne;
            if (od->output_def.rank >= 2) {
                seq = od->output_def.dims[od->output_def.rank - 2];
                ch = od->output_def.dims[od->output_def.rank - 1];
            }
            em.add_op(OP_CONV1D_SSM_F16, {x_t, w_s, out_t, seq, ch, e.kw});
            break;
        }

        if (nm == "Concat") {
            ExtraAxis e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t out_t = em.fresh_temp(od->op_id);
            std::vector<uint32_t> args;
            for (size_t i = 0; i < 8; i++)
                args.push_back(i < od->inputs.size()
                    ? em.src_ref(od->inputs[i], gp.get_input_node_id(), gp, wslots) : 0u);
            args.push_back(out_t);
            uint32_t ax_c = (uint32_t)(e.axis < 0 ? od->output_def.rank - 1 : e.axis);
            args.push_back(ax_c);
            args.push_back((uint32_t)od->inputs.size());
            args.push_back((uint32_t)elems_of(od));
            // 每段 axis 维尺寸(≤4 段): 从各输入生产者 output_def 取
            for (size_t i = 0; i < 4; i++) {
                uint32_t sz = 0;
                if (i < od->inputs.size()) {
                    const OpDef* src = gp.get_op_at(od->inputs[i].src_id);
                    if (src && ax_c < src->output_def.rank)
                        sz = src->output_def.dims[ax_c];
                }
                args.push_back(sz);
            }
            em.add_op(OP_CONCAT_F16, args);
            break;
        }

        if (nm == "StridedSlice") {
            ExtraStridedSlice e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            // 设备契约: rank≤3 通用切片(begin/end/stride 各 3; rank4 且 dim0=1 降 rank)
            uint32_t rk = std::min<uint32_t>(e.rank, 3);
            std::vector<uint32_t> args{x_t, out_t, (uint32_t)elems_of(od), rk};
            const auto& pool = gp.const_pool();
            uint32_t b0 = 0, b1 = 0, b2 = 0, e0 = 0, e1 = 0, e2 = 0, s0 = 1, s1 = 1, s2 = 1;
            if (e.ranges_offset && e.ranges_offset + e.rank * 12 <= pool.size()) {
                const int32_t* rg = reinterpret_cast<const int32_t*>(pool.data() + e.ranges_offset);
                if (rk >= 1) { b0 = (uint32_t)rg[0]; e0 = (uint32_t)rg[e.rank]; s0 = (uint32_t)rg[2 * e.rank]; }
                if (rk >= 2) { b1 = (uint32_t)rg[1]; e1 = (uint32_t)rg[e.rank + 1]; s1 = (uint32_t)rg[2 * e.rank + 1]; }
                if (rk >= 3) { b2 = (uint32_t)rg[2]; e2 = (uint32_t)rg[e.rank + 2]; s2 = (uint32_t)rg[2 * e.rank + 2]; }
            }
            args.push_back(b0); args.push_back(b1); args.push_back(b2);
            args.push_back(e0); args.push_back(e1); args.push_back(e2);
            args.push_back(s0); args.push_back(s1); args.push_back(s2);
            em.add_op(OP_STRIDED_SLICE_F16, args);
            break;
        }

        if (nm == "Split") {
            ExtraSplit e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            std::vector<uint32_t> args{x_t};
            for (size_t i = 0; i < 8; i++) {
                uint32_t t = i < (size_t)e.num_splits ? em.fresh_temp(od->op_id, (uint32_t)i) : 0u;
                args.push_back(t);
            }
            args.push_back((uint32_t)(e.axis < 0 ? 0 : e.axis));
            args.push_back(e.num_splits);
            for (size_t i = 0; i < 4; i++)
                args.push_back(i < (size_t)e.num_splits ? e.sizes[i] : 0u);
            em.add_op(OP_SPLIT_F16, args);
            break;
        }

        if (nm == "Reduce") {
            ExtraAxis e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            {
                const OpDef* src_r = gp.get_op_at(od->inputs[0].src_id);
                uint64_t n_in = src_r ? elems_of(src_r) : elems_of(od);
                std::vector<uint32_t> args{x_t, out_t, (uint32_t)n_in,
                                           (uint32_t)e.axis, e.reduce_type};
                for (uint32_t i = 0; i < 4; i++)
                    args.push_back(src_r && i < src_r->output_def.rank
                                       ? src_r->output_def.dims[i] : 1u);
                em.add_op(OP_REDUCE_F16, args);
            }
            break;
        }

        if (nm == "CumulativeSum") {
            ExtraAxis e{};
            if (od->serialized_extra.size() >= sizeof(e))
                std::memcpy(&e, od->serialized_extra.data(), sizeof(e));
            uint32_t x_t = em.src_ref(od->inputs[0], gp.get_input_node_id(), gp, wslots);
            uint32_t out_t = em.fresh_temp(od->op_id);
            uint64_t ne = elems_of(od);
            uint32_t ax = (e.axis < 0) ? od->output_def.rank - 1 : (uint32_t)e.axis;
            uint32_t rows = 1, n_ax = 1;
            if (ax < od->output_def.rank) {
                rows = (uint32_t)(ne / od->output_def.dims[ax]);
                n_ax = od->output_def.dims[ax];
            }
            em.add_op(OP_CUMSUM_F32, {x_t, out_t, rows, n_ax, ax, e.exclusive, e.reverse});
            break;
        }

        if (nm == "Input" || nm == "Output" || nm.empty()) continue;

        std::fprintf(stderr, "error: unsupported op '%s' in wtop_emit\n", nm.c_str());
        return 4;
        } while (0);
        // 活性: 本 op 的输入已消费, 释放其 temp
        for (const auto& c : od->inputs)
            em.release_at(Emitter::tkey(c.src_id, c.out_idx), pos);
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
                ? em.src_ref(out->inputs[0], gp.get_input_node_id(), gp, wslots)
                : 0x8000u;
        } else {
            // M2 结构闭环: spill 段引用保留 temp(真实溢出语义 M4 重做)
            t = em.fresh_temp(0xFFFFFFFEu);
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
        wt_blob* wb = new wt_blob{};
        int rc = wt_parse(em.blob.data(), em.blob.size(), wb);
        if (rc != WT_OK) {
            std::fprintf(stderr, "error: self-validate failed: %s (blob=%zu weight_area=%zu)\n",
                         wt_err_str(rc), em.blob.size(), em.weight_area.size());
            uint64_t max_end = 0;
            for (auto& sl : em.slots)
                max_end = std::max<uint64_t>(max_end, (uint64_t)sl.offset + sl.len);
            std::fprintf(stderr, "  max slot end=%llu n_slots=%zu\n",
                         (unsigned long long)max_end, em.slots.size());
            return 3;
        }
        std::printf("WTOP OK: slots=%u ops=%u bytes=%zu (gguf hits=%u miss=%u)\n",
                    (unsigned)em.slots.size(), (unsigned)em.ops.size(), em.blob.size(),
                    em.gguf_hits, em.gguf_miss);

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
    std::string bin_path, in_f16, out_path, manifest_path, gguf_path, match_path;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--bin") bin_path = next();
        else if (a == "--input-f16") in_f16 = next();
        else if (a == "--out") out_path = next();
        else if (a == "--manifest") manifest_path = next();
        else if (a == "--gguf") gguf_path = next();
        else if (a == "--match") match_path = next();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (bin_path.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: wtop_emit --bin <tagged.bin> [--input-f16 <f16.raw>] --out <blob.wtop> [--manifest <json>] [--gguf <g> --match <tsv>]\n");
        return 2;
    }
    return emit(bin_path, in_f16, out_path, manifest_path, gguf_path, match_path);
}
