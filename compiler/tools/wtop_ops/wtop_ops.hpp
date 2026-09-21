// wtop_ops.hpp — wtop_emit 算子发射共享层(算子三件套·host 发射侧)
//
// Emitter/公共工具/注册表从 wtop_emit.cpp 拆出; 每算子实现一文件
// (tools/wtop_ops/op_<name>.cpp), 文件尾 OpRegistrar 静态对象构造期
// 自注册进 op_registry()(与设备侧 g_op_exec_table 同构)。
// 加新算子 = 新建一个 op_<name>.cpp + CMakeLists 源列表加一行,
// 不动 wtop_emit 主循环。
#ifndef WTOP_OPS_HPP
#define WTOP_OPS_HPP

#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_extra.hpp"
#include "hnnx/ir/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// WTOP 契约(kernels/include/oplist_parse.h 的编译期副本语义; 头从 kernels 取)
#include "oplist_parse.h"

namespace wtop {

using namespace hnnx;

// f32 → f16 (round-to-nearest-even, 软件实现; 与 numpy astype 语义一致)
inline uint16_t f32_to_f16_rne(float f) {
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

// f16 → f32(反量化用)
inline float f16_to_f32(uint16_t h) {
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
            u = sign | ((uint32_t)(114 + e) << 23) | (mant << 13);
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

// GGUF 匹配条目(TSV 行; M2c)
struct GgufEntry { uint64_t file_offset; uint64_t nbytes; uint32_t type; std::vector<uint64_t> dims; };

struct Emitter {
    std::vector<uint8_t> blob;
    std::vector<uint8_t> weight_area;   // 128B 对齐起点相对 blob 尾部
    std::vector<wt_slot> slots;
    std::vector<wt_op> ops;
    std::map<uint64_t, uint32_t> op_temp;  // (op_id<<32 | out_idx) → temp id
    uint32_t next_temp = 0;
    std::vector<uint32_t> free_temps;      // 活性分析释放的 temp(M2)
    std::map<uint64_t, uint32_t> last_use; // 复合键 → 最后消费位置(plan_order 下标)

    // 第7步阶段一: 静态 DDR 偏移表(emit 层注入)
    bool ddr_static = false;
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> ddr_op_map;  // op_id → {offset, size}
    std::unordered_map<uint64_t, bool> ddr_op_vtcm;                          // op_id → in_vtcm
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> ddr_temp_tab;          // temp_id → {offset, size}
    std::map<uint32_t, bool> ddr_temp_vtcm;                                  // temp_id → in_vtcm
    uint64_t ddr_static_cap = 0;   // 编译期静态区大小(表外 temp 池尾 bump)
    uint64_t ddr_vtcm_cap = 0;     // VTCM 驻留池大小(0=无驻留)
    uint64_t bump_reserve = 0;     // 表外 bump 预留(字节; M2 起由 TAG_MEM_PLAN 供给)
    std::set<uint64_t> supply_slots;  // W4A16 供给槽去重 (高32位=种类)
    std::map<uint64_t, uint32_t> supply_slot_ids;  // 种类键 → 已发槽 id
    // W4A16 kernel 格式: bias/scale 槽内容随权重而定 (per-weight-op,
    // 不能按尺寸去重 — 同 N 不同权重内容不同), 由 ensure_weight_slot 发射
    // 后按 op_id 挂账, op_matmul 取用
    std::map<uint64_t, uint32_t> w4_bias_slots;
    std::map<uint64_t, uint32_t> w4_scale_slots;
    // kernel-格式权重槽独立缓存 (与 wslots 分离): tie 权重被 GATHER 与
    // GEMM 共享时 f16 槽与 kernel 槽必须并存, 各自消费方取各自格式
    std::map<uint64_t, uint32_t> w4_wslots;
    // 阶段二: 溢出张量 → SPILL/FILL 插桩
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> ddr_spill_map;  // op_id → {溢出区偏移, size}
    uint32_t spill_pool_slot = 0;

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
    uint32_t fresh_temp(GraphPrepare& gp, uint64_t op_id, uint32_t out_idx = 0) {
        uint32_t t;
        if (!free_temps.empty()) { t = free_temps.back(); free_temps.pop_back(); }
        else { t = next_temp++; }
        op_temp[tkey(op_id, out_idx)] = t;
        if (ddr_static) {
            /* 静态模式: 表内(op_id 命中且 out_idx==0)登记编译期偏移;
             * 编译期 size 按 od->output_def 算, 运行时广播物化后可能更大 —
             * size 不符时不得指静态偏移(设备 temp_get 按运行时 bytes 检查
             * off+bytes > cap → NULL → ref fail; probe op_id=146 实锤:
             * output_def 4096B 但广播物化后输出 8192 元素 = 16384B) */
            auto dit = ddr_op_map.find(op_id);
            const OpDef* od = gp.get_op_at(op_id);
            uint64_t elems = 1;
            for (uint32_t i = 0; od && i < od->output_def.rank && i < 5; ++i)
                elems *= (uint64_t)od->output_def.dims[i];
            if (elems == 0) elems = 1;
            uint64_t es = od && od->output_def.element_size ? od->output_def.element_size : 4;
            uint64_t need = elems * es;
            if (dit != ddr_op_map.end() && out_idx == 0 &&
                ddr_temp_tab.find(t) == ddr_temp_tab.end() &&
                dit->second.second >= need) {
                ddr_temp_tab[t] = dit->second;
                auto vit = ddr_op_vtcm.find(op_id);
                if (vit != ddr_op_vtcm.end() && vit->second)
                    ddr_temp_vtcm[t] = true;
            }
        }
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
    // 路线B(4B 外置权重): 权重字节不进 blob, 累积到 ext_weight_area(独立
    // model.weights.bin), 权重 slot 记 EXT_WGT + 相对该区偏移。
    bool ext_weights = false;
    std::vector<uint8_t> ext_weight_area;   // 128B 对齐, emit 末尾落盘

    // 外置权重槽: 同 add_slot 但数据进 ext_weight_area, addr=EXT_WGT。
    // slot.offset = 相对外部权重区基址(设备 slot_ptr 据此定位)。
    uint32_t add_ext_slot(uint32_t len, uint32_t count, const uint8_t* data) {
        wt_slot s{};
        s.len = len;
        s.count = count;
        s.offset = (uint32_t)ext_weight_area.size();
        s.addr = WT_SLOT_EXT_WGT;
        ext_weight_area.insert(ext_weight_area.end(), data, data + len);
        while (ext_weight_area.size() % 128 != 0) ext_weight_area.push_back(0);
        uint32_t id = (uint32_t)slots.size();
        slots.push_back(s);
        return id;
    }

    // 路线B 外置权重源: emit --weights-bin 直读(params.bin f32), 按
    // const_data_offset 从此外部缓冲取权重字节(替代空 const_pool)。
    const std::vector<uint8_t>* ext_weights_bin = nullptr;

// GGUF 供给上下文(M2c; emit 启动时设置)
    const std::map<std::string, GgufEntry>* gguf_tab = nullptr;
    const std::vector<uint8_t>* gguf_bytes = nullptr;
    bool use_gguf = false;
    uint32_t gguf_hits = 0, gguf_miss = 0;

    // GGUF Q4_0 → W4A16 HMX kernel 格式 (闭包权威 = /4090disk2/htpw4a16_v81
    // prepare_owned_inputs.py; 110 板 256³ 位恒等验证):
    //   f32 反量化 [K,N] → 每列 int8[-7,7] (scale = max|col|/7, 全零列→1) →
    //   wt: pack_w4_kblock32_nmajor_k4_lohi = nib(w+8)&0xF, kb(外)×N32×
    //       kg(4)×n×kr(4) lohi 字节 → K*N/2, 整区 XOR 0x88 (4-bit 补码)
    //   bias: pack_native_a16_bias = (N/32)*512, 每列 eff=-128*Σw (i32 LE)
    //       + 控制字 0x5524/0x8040/0x0092/0x4000 (u16 LE)
    //   scale: 每列 f16 (kernel 固定 ÷7 域, 出面反量化用)
    static void pack_w4a16_kernel(const uint8_t* src, size_t nbytes, size_t K, size_t N,
                                  std::vector<uint8_t>& wt_out,
                                  std::vector<uint8_t>& bias_out,
                                  std::vector<uint8_t>& scale_out) {
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
        std::vector<int8_t> wq(K * N);
        scale_out.assign(N * 2, 0);
        for (size_t n = 0; n < N; n++) {
            float mx = 0.0f;
            for (size_t k = 0; k < K; k++)
                mx = std::max(mx, std::fabs(w[k * N + n]));
            float S = mx > 0.0f ? mx / 7.0f : 1.0f;
            uint16_t s16 = f32_to_f16_rne(S);
            std::memcpy(&scale_out[n * 2], &s16, 2);
            for (size_t k = 0; k < K; k++) {
                int q = (int)std::lround(w[k * N + n] / S);
                if (q < -7) q = -7;
                if (q > 7) q = 7;
                wq[k * N + n] = (int8_t)q;
            }
        }
        wt_out.assign(K * N / 2, 0);
        size_t o = 0;
        for (size_t kb = 0; kb < K / 32; kb++)
            for (size_t n_base = 0; n_base < N; n_base += 32)
                for (size_t kg = 0; kg < 4; kg++) {
                    size_t k_base = kb * 32 + kg * 8;
                    for (size_t n = n_base; n < n_base + 32; n++)
                        for (size_t kr = 0; kr < 4; kr++) {
                            uint32_t lo = ((uint32_t)(uint8_t)wq[(k_base + kr) * N + n] + 8u) & 0xFu;
                            uint32_t hi = ((uint32_t)(uint8_t)wq[(k_base + kr + 4) * N + n] + 8u) & 0xFu;
                            wt_out[o++] = (uint8_t)((lo | (hi << 4)) ^ 0x88u);
                        }
                }
        bias_out.assign((N / 32) * 512, 0);
        static const uint8_t cw[8] = {0x24, 0x55, 0x40, 0x80, 0x92, 0x00, 0x00, 0x40};
        for (size_t nt = 0; nt < N / 32; nt++)
            for (int parity = 0; parity < 2; parity++) {
                size_t half = (size_t)parity * 256;
                for (int lane = 0; lane < 16; lane++) {
                    size_t col = nt * 32 + (size_t)parity + (size_t)(2 * lane);
                    size_t base = nt * 512 + half + (size_t)(8 * lane);
                    std::memcpy(&bias_out[base], cw, 8);
                    int32_t sum_w = 0;
                    for (size_t k = 0; k < K; k++) sum_w += (int32_t)wq[k * N + col];
                    int32_t eff = -128 * sum_w;
                    std::memcpy(&bias_out[base + 128], &eff, 4);
                }
            }
    }

    // 权重消费 op 的 const 输入 → 权重槽(按 const id 去重; GGUF 供给优先)
    // consumer_grp = 消费节点的原始名(loader 只给计算 op 设 grouping,
    // 权重 const 自身没有; TSV 键 = 消费节点名)
    uint32_t ensure_weight_slot(GraphPrepare& gp, const OpDef* w,
                                std::map<uint64_t, uint32_t>& wslots,
                                const std::string& consumer_grp = "",
                                bool kernel_fmt = false) {
        /* kernel-格式调用方用独立缓存 (tie 权重两种格式并存, 互不遮蔽) */
        std::map<uint64_t, uint32_t>& cache = kernel_fmt ? w4_wslots : wslots;
        auto it = cache.find(w->op_id);
        if (it != cache.end()) return it->second;
        /* 路线B: 外置模式权重进 ext_weight_area(addr=EXT_WGT), 内置进 blob */
        auto slot_for = [&](uint32_t len, uint32_t cnt, const uint8_t* d) {
            return ext_weights ? add_ext_slot(len, cnt, d) : add_slot(len, cnt, d);
        };
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
                    if (e.type == 2 && K % 32 == 0 && NN % 32 == 0 && kernel_fmt && getenv("GEHTP_TILE")) {
                        // Q4_0 → W4A16 HMX kernel 格式 (wt + 折叠 bias + 列 scale;
                        // 仅 GEMM 消费方 — GATHER 表等仍走 f16 反量化, 否则
                        // 嵌入表被 nibble 打包 → gather 读垃圾 token)
                        std::vector<uint8_t> wtb, biasb, scaleb;
                        pack_w4a16_kernel(src, e.nbytes, K, NN, wtb, biasb, scaleb);
                        id = slot_for((uint32_t)wtb.size(), (uint32_t)(K * NN), wtb.data());
                        uint32_t bid = slot_for((uint32_t)biasb.size(), (uint32_t)(biasb.size() / 2u), biasb.data());
                        uint32_t sid = slot_for((uint32_t)scaleb.size(), (uint32_t)NN, scaleb.data());
                        cache[w->op_id] = id;
                        w4_bias_slots[w->op_id] = bid;
                        w4_scale_slots[w->op_id] = sid;
                        gguf_hits++;
                        return id;
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
                        id = slot_for((uint32_t)wb.size(), (uint32_t)(K * NN), wb.data());
                    } else {  // F32 → f16
                        size_t n4 = e.nbytes / 4;
                        std::vector<uint16_t> w16(n4);
                        const float* wf = reinterpret_cast<const float*>(src);
                        for (size_t i = 0; i < n4; i++) w16[i] = f32_to_f16_rne(wf[i]);
                        std::vector<uint8_t> wb(n4 * 2);
                        std::memcpy(wb.data(), w16.data(), wb.size());
                        id = slot_for((uint32_t)wb.size(), (uint32_t)n4, wb.data());
                    }
                    cache[w->op_id] = id;
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
            uint32_t id = slot_for(128, 64, zeros.data());
            cache[w->op_id] = id;
            return id;
        }
        std::vector<uint16_t> w16(n);
        /* 权重字节源始终从 const_pool 读 (const_data_offset 是 const_pool 内偏移)。
         * 路线B 仅改变槽路由: slot_for 已按 ext_weights 选 add_ext_slot/add_slot。
         * ext_weights_bin(params.bin) 的字节布局与 const_pool 不同, 不能作为源。 */
        const float* wf = reinterpret_cast<const float*>(
            gp.const_pool().data() + w->const_data_offset);
        for (size_t i = 0; i < n; i++) w16[i] = f32_to_f16_rne(wf[i]);
        std::vector<uint8_t> wbytes(n * 2);
        std::memcpy(wbytes.data(), w16.data(), wbytes.size());
        uint32_t id = slot_for((uint32_t)wbytes.size(), (uint32_t)n, wbytes.data());
        cache[w->op_id] = id;
        return id;
    }
    // 输入节点 = slot 0; const 生产者 = 权重槽(0x8000|slot 编码);
    // 幻影占位(空名非 const, 如第二输入 position_ids) = 专用零槽
    // (M2 结构闭环; M4 改真注入: 幻影槽标 EXT_IN 由 host 供给)
    uint32_t dummy_slot_id = 0;  // 共享哑槽(幻影/参数 const/缺失生产者引用)
    // 多输入图: 各 Input 节点 → 槽(主输入=EXT_IN 槽 0; 其余=固化 const 槽)
    std::map<uint64_t, uint32_t> input_slots;
    uint64_t main_input_id = 0;  /* = in_ids[0](与 EXT_IN 槽分配一致) */
    uint32_t src_ref(const InputConn& c, uint64_t input_node_id, GraphPrepare& gp,
                     std::map<uint64_t, uint32_t>& wslots) {
        uint64_t op_id = c.src_id;
        (void)input_node_id;
        /* 主输入判定必须与槽分配(in_ids 升序首项)一致:
         * gp.get_input_node_id() 可能是别的 Input(cos 等) */
        if (op_id == main_input_id) return 0x8000u | 0u;
        {
            auto isl = input_slots.find(op_id);
            if (isl != input_slots.end()) return 0x8000u | isl->second;
        }
        const OpDef* p = gp.get_op_at(op_id);
        // 有池数据即按 const 槽(is_const 标志对 loader 的 tensor_param const 不可靠)
        if (p && (p->is_const() || p->const_data_size > 0)) {
            uint32_t s = ensure_weight_slot(gp, p, wslots);
            return 0x8000u | s;
        }
        bool param_const_name = false;
        if (p && p->name_tag && p->name_tag->name()) {
            const char* nm2 = p->name_tag->name();
            static const char* suf[] = {"_shape", "_axes", "_pad_amount", "_ranges", "_perm", "_multiples", "_split_index", "_dilation", "_stride"};
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
    std::vector<uint64_t> emitted_ids;  /* 与 ops 同序的 op_id(add_op 记录) */
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
        emitted_ids.push_back(cur_op);
        wt_op o{};
        o.opcode = opcode;
        o.n_args = (uint16_t)args.size();
        uint32_t i = 0;
        for (uint32_t a : args) o.args[i++] = a;
        ops.push_back(o);
    }
    void add_op(uint16_t opcode, const std::vector<uint32_t>& args) {
        emitted_ids.push_back(cur_op);
        wt_op o{};
        o.opcode = opcode;
        o.n_args = (uint16_t)args.size();
        uint32_t i = 0;
        for (uint32_t a : args) o.args[i++] = a;
        ops.push_back(o);
    }
};

/* 跨 handler 共享的 emit() 函数域状态(重构前 Conv2d 分支隐式捕获):
 * H/W/C = 图输入 NCHW 几何; w_slot/b_slot = 首个 Conv2d 的权重/偏置槽
 * (op 循环前预登记, 槽序敏感勿移)。 */
struct WtopEmitShared {
    uint32_t H = 1, W = 1, C = 1;
    uint32_t w_slot = 0, b_slot = 0;
};

/* ===================== 算子发射注册表 =====================
 * 每 op 一个 handler(QNN OpPackage 同构): 从 IR OpDef 读语义(serialized_extra/
 * dims/const 池), 编码为设备 opcode + args 并接线 slot/temp 引用。
 * 返回 0=成功; 非 0=emit 致命错误码(原样上抛)。 */
using WtopOpHandler = int (*)(Emitter&, GraphPrepare&, const OpDef*,
                              std::map<uint64_t, uint32_t>&, const std::string&, const WtopEmitShared&);

// 注册表本体(Meyers 单例; op_<name>.cpp 的 OpRegistrar 构造期填充)
inline std::unordered_map<std::string, WtopOpHandler>& op_registry() {
    static std::unordered_map<std::string, WtopOpHandler> t;
    return t;
}
struct OpRegistrar {
    explicit OpRegistrar(std::initializer_list<std::pair<const char*, WtopOpHandler>> entries) {
        for (const auto& e : entries) op_registry()[e.first] = e.second;
    }
};

// W4A16 kernel-格式权重消费方注册 (同 OpRegistrar 机制: 发射器自身注册,
// emit 预收集遍查同一张表决定 kernel_fmt — 单一真相源, 不做名字硬编码)
inline std::unordered_set<std::string>& op_w4_registry() {
    static std::unordered_set<std::string> t;
    return t;
}
struct OpW4Registrar {
    explicit OpW4Registrar(std::initializer_list<const char*> names) {
        for (const char* n : names) op_w4_registry().insert(n);
    }
};
inline bool wants_w4_kernel_weight(const std::string& nm) {
    return op_w4_registry().count(nm) > 0;
}

// 权重消费 op 的 const 输入下标(weight input 位置)
inline int weight_input_index(const std::string& nm) {
    if (nm == "FullyConnected" || nm == "MatMul" || nm == "Conv2d" ||
        nm == "DepthWiseConv2d" || nm == "RmsNorm") return 1;
    if (nm == "Gather") return 0;  // embedding 表
    return -1;
}

// op 输出元素数(output_def 全维乘积)
inline uint64_t elems_of(const OpDef* od) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < od->output_def.rank && i < 5; ++i)
        n *= (uint64_t)od->output_def.dims[i];
    return n;
}

/* QNN 2.48 operation 枚举 → 设备 exec 内部 subtype
 * (oplist_parse.h OP_UNARY_F16/OP_BINARY_F16 契约; 与 2.48 QnnOpDef.h 对齐) */
inline uint32_t qnn_binary_to_sub(uint32_t op) {
    switch (op) {
    case 0:  return 0;   /* ADD */
    case 18: return 1;   /* SUBTRACT */
    case 13: return 2;   /* MULTIPLY */
    case 2:  return 3;   /* DIVIDE */
    default: return 0xFFFFFFFFu;
    }
}
inline uint32_t qnn_unary_to_sub(uint32_t op) {
    switch (op) {
    case 8:  return 0;   /* NEG */
    case 5:  return 1;   /* EXP */
    case 15: return 2;   /* SQRT */
    case 12: return 3;   /* RSQRT */
    case 7:  return 4;   /* LOG */
    case 0:  return 5;   /* ABS */
    case 14: return 6;   /* SIN */
    case 4:  return 7;   /* COS */
    default: return 0xFFFFFFFFu;
    }
}
inline uint32_t qnn_neuron_to_sub(uint32_t op) {
    switch (op) {
    case 6:  return 8;   /* SIGMOID */
    case 8:  return 9;   /* TANH */
    case 1:  return 10;  /* GELU */
    case 4:  return 11;  /* RELU */
    case 3:  return 12;  /* HARD_SWISH → SWISH 近似 */
    case 7:  return 13;  /* SOFTPLUS (设备侧 case 13; 缺失=恒等→cumsum 爆炸) */
    default: return 0xFFFFFFFFu;
    }
}

} // namespace wtop

#endif // WTOP_OPS_HPP
