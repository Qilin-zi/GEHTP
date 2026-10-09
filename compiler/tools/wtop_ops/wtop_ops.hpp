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
#include <string>
#include <unordered_map>
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
    std::vector<uint32_t> op_annot;     // 每 op 一条引擎注解 (v3 footer, 平行于 ops)
    struct CarveEntry {
        uint32_t op_index;
        uint32_t vtcm_offset;
        uint32_t size_bytes;
        uint32_t residency_start;
        uint32_t residency_end;
    };
    struct PendingCarve {
        uint32_t op_index;
        uint32_t producer_pos;
        uint32_t lifetime_end_pos;
        uint32_t vtcm_offset;
        uint32_t size_bytes;
    };
    std::vector<PendingCarve> pending_carves; // graph-order lifetimes, resolved after emit
    std::vector<CarveEntry> carve_entries; // v3 optional VTCM carve footer
    uint32_t cur_engine = 0;            // add_op 时写入 op_annot 的打包注解 u32
    std::map<uint64_t, uint32_t> op_temp;  // (op_id<<32 | out_idx) → temp id
    std::map<uint64_t, uint32_t> i32_slots;  // int32 索引 const 独立缓存(键 op_id)
                                             // 与 wslots(权重槽)分离: 同一 const 节点
                                             // 既当索引又当权重时, 权重槽 f32→f16 洗白
                                             // 与索引槽原样字节必须各占一槽, 不能撞键
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
    // 阶段二: 溢出张量 → SPILL/FILL 插桩
    std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> ddr_spill_map;  // op_id → {溢出区偏移, size}
    uint32_t spill_pool_slot = 0;

    /* 003 T015: DDR→VTCM 预置 staging。W4A16 act 源在 DDR (计算 temp / EXT_IN
     * 注入) 而运行期 exec_matmul 只认裸 slot / 0x4000|temp 两种形态 → 编译期
     * 在 VTCM 池内 planner 驻留区之上 bump 一块 2048 对齐 staging 区, 发
     * OP_DMA 预置 + matmul act 引用 0x4000|stg。TEMPOFF/ carve 双登记复用
     * 既有表结构 (ddr_temp_tab/ddr_temp_vtcm + pending_stages→carve_entries)。 */
    bool dma_runlist = false;
    struct StageRec {
        uint32_t op_index;   /* 预置 DMA op 的发射下标 (carve 记录锚点+窗口起点) */
        uint32_t end_pos;    /* 消费 op 的图序位+1 (主循环回填, 合并时映射边界) */
        uint32_t vtcm_offset;
        uint32_t size_bytes;
    };
    std::vector<StageRec> pending_stages;
    size_t stage_mark = 0;                 /* 主循环已回填 end_pos 的水位 */
    uint32_t vtcm_stage_top = 0xFFFFFFFFu; /* 惰性: staging 起点 (planner 驻留区上沿对齐 2048) */
    // 池内 bump 一块 staging 区, 登记为静态 VTCM temp, 返回 temp id (失败 -1:
    // 无 mem plan / 无 VTCM 池 / 池放不下 / temp id 耗尽)。基线取 ddr_op_map
    // 全量计划态上沿 (非物化态: 首个 W4A16 到达时高偏移驻留 temp 可能尚未物化)。
    int alloc_vtcm_stage(uint32_t bytes, uint32_t* off_out) {
        if (!ddr_static || ddr_vtcm_cap == 0) return -1;
        if (next_temp > 32768u) return -1;
        if (vtcm_stage_top == 0xFFFFFFFFu) {
            uint64_t mx = 0;
            for (const auto& [oid, offsz] : ddr_op_map) {
                auto vit = ddr_op_vtcm.find(oid);
                if (vit != ddr_op_vtcm.end() && vit->second)
                    mx = std::max<uint64_t>(mx, (uint64_t)offsz.first + offsz.second);
            }
            vtcm_stage_top = (uint32_t)((mx + 2047u) & ~2047u);
        }
        if ((uint64_t)vtcm_stage_top + bytes > ddr_vtcm_cap) return -1;
        uint32_t t = next_temp++;
        ddr_temp_tab[t] = {vtcm_stage_top, bytes};
        ddr_temp_vtcm[t] = true;
        *off_out = vtcm_stage_top;
        vtcm_stage_top += (bytes + 2047u) & ~2047u;
        return (int)t;
    }

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
        const OpDef* od = gp.get_op_at(op_id);
        uint64_t elems = 1;
        for (uint32_t i = 0; od && i < od->output_def.rank && i < 5; ++i)
            elems *= (uint64_t)od->output_def.dims[i];
        if (elems == 0) elems = 1;
        uint64_t es = od && od->output_def.element_size ? od->output_def.element_size : 4;
        uint64_t need = elems * es;
        /* 静态模式 temp id 回收规则: TEMPOFF/carve 按 temp id 一对一定址,
         * 已静态登记(ddr_temp_tab)的 id 只能复用给"计划区完全一致 + 同域
         * (VTCM/DDR) + 放得下"的后继张量 —— 否则两个同时存活张量叠同一段
         * 静态区 = 编译期踩踏 (t2l 实锤: temp1 gid14 区 [512,1024) 回收给
         * gid18, 与 gid10 同区, 生命期 [43,49)∩[44,46) 重叠)。
         * 非静态 id 任意复用; 无可回收 id 时发新号(0.8B 16k op 靠同区
         * 复用把 id 压回并发水位, 不能一刀切禁回收)。 */
        auto recyclable = [&](uint32_t c) {
            if (!ddr_static || !ddr_temp_tab.count(c)) return true;
            auto dit0 = ddr_op_map.find(op_id);
            if (out_idx != 0 || dit0 == ddr_op_map.end()) return false;
            if (!(dit0->second == ddr_temp_tab.at(c))) return false;
            if (dit0->second.second < need) return false;
            bool cv = ddr_temp_vtcm.count(c) && ddr_temp_vtcm.at(c);
            bool ov = ddr_op_vtcm.count(op_id) && ddr_op_vtcm.at(op_id);
            return cv == ov;
        };
        uint32_t t;
        size_t pick = free_temps.size();
        for (size_t k = free_temps.size(); k-- > 0;) {
            if (recyclable(free_temps[k])) { pick = k; break; }
        }
        if (pick < free_temps.size()) {
            t = free_temps[pick];
            free_temps[pick] = free_temps.back();
            free_temps.pop_back();
        } else {
            t = next_temp++;
        }
        op_temp[tkey(op_id, out_idx)] = t;
        if (ddr_static) {
            /* 静态模式: 表内(op_id 命中且 out_idx==0)登记编译期偏移;
             * 编译期 size 按 od->output_def 算, 运行时广播物化后可能更大 —
             * size 不符时不得指静态偏移(设备 temp_get 按运行时 bytes 检查
             * off+bytes > cap → NULL → ref fail; probe op_id=146 实锤:
             * output_def 4096B 但广播物化后输出 8192 元素 = 16384B) */
            auto dit = ddr_op_map.find(op_id);
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
    /* 007 US2 (T013): 权重域报告。domain w4/f16; reason 枚举 = none |
     * shape_misalign | source_missing | consumer_gate(对齐 Q4 但非 matmul
     * 消费者/batched → by-design f16, spec 三枚举的扩展, 见
     * evidence/us2_route_recon.md)。 */
    struct W4DomainRecord {
        std::string layer, gguf_name;
        const char* domain, *reason;
        uint64_t size_bytes;
    };
    std::vector<W4DomainRecord> w4_records;
    std::vector<std::string> w4_missing;   // FR-003: matmul 权重 miss 名单(收尾硬错误)
    std::map<uint64_t, bool> wslot_is_q4;  // const op_id → w4 tile 槽(复用冲突防护)
    /* tie 权重槽去重: GGUF 张量身份(file_offset,nbytes) → slot。embed Gather 与
     * lm_head MatMul 是不同图 const 但同一 GGUF 张量(token_embd tie), tile 字节
     * 只由源决定 → 共槽(0.8B 省 2×151.5MB, SC-003 预算的关键) */
    std::map<std::pair<uint64_t, uint64_t>, uint32_t> gguf_w4_dedup;

    // GGUF Q4_0 → 640B/tile 纯重排(007 US2 Route A; payload 576B 零重量化,
    // 量化 nibble 与 per-block d 数值原样保留 — 仅做 nibble 面序置换; +64B
    // pad 对齐 hvhx_v2_weight_aligned_tile_size(Q4_0)=640, 保 HVX 反量化
    // 128B scale 读不越界)。
    // nibble 面序仲裁(007 US2 实测, token_embd 行 0 对拍 HF F32 embed):
    //   本 GGUF = 平面序(B): qs[j] lo=值 j(0..15), hi=值 j+16 → cos 0.996;
    //   对偶序(A) 解读 cos 0.0066 = 乱码。HVX dequant task(ggml 移植)按
    //   对偶序(A)解读 tile 字节: byte[i] lo=值 2i, hi=值 2i+1 → repack 做
    //   B→A nibble 置换, 消费端(matmul HVX/gather 标量)统一 A 序。
    // tile(ct,kt) @ (ct*n_k + kt)*640:
    //   quants byte[i*32 + n] = [值(2i) | 值(2i+1)<<4] (A 序, i=k/2);
    //   scales = 块 d f16 原样 @ tile+512+2n (32 个, 后 64B pad 0)。
    // 设备侧消费 = oplist_exec.c exec_matmul_f16_hmx(HVX 反量化) /
    // exec_gather(行级 dequant), 槽长判别 len == (K/32)*(N/32)*640。
    static std::vector<uint8_t> repack_q4_0_tiles(const uint8_t* src, size_t nbytes,
                                                  size_t K, size_t N) {
        (void)nbytes;
        size_t n_k = K / 32, n_ct = N / 32;
        std::vector<uint8_t> out(n_ct * n_k * 640, 0);
        for (size_t ct = 0; ct < n_ct; ct++)
            for (size_t kt = 0; kt < n_k; kt++) {
                uint8_t* tile = out.data() + (ct * n_k + kt) * 640;
                for (size_t r = 0; r < 32; r++) {
                    const uint8_t* blk = src + ((ct * 32 + r) * n_k + kt) * 18;
                    std::memcpy(tile + 512 + 2 * r, blk, 2);  // d 原样
                    const uint8_t* qs = blk + 2;
                    for (size_t i = 0; i < 8; i++) {          // 值 2i/2i+1 在 lo 面
                        uint8_t lo = qs[2 * i] & 0xFu;
                        uint8_t hi = qs[2 * i + 1] & 0xFu;
                        tile[i * 32 + r] = lo | (uint8_t)(hi << 4);
                    }
                    for (size_t i = 8; i < 16; i++) {         // 值 2i/2i+1 在 hi 面
                        uint8_t lo = qs[2 * i - 16] >> 4;
                        uint8_t hi = qs[2 * i + 1 - 16] >> 4;
                        tile[i * 32 + r] = lo | (uint8_t)(hi << 4);
                    }
                }
            }
        return out;
    }

    // GGUF Q4_0 → f16 槽反量化(T014 兜底: 形状未对齐 / 非 Q4 消费者)。
    // 平面序(B, 见 repack_q4_0_tiles 面序仲裁): qs[j] lo=值 j, hi=值 j+16。
    static std::vector<uint8_t> dequant_q4_0_f16(const uint8_t* src, size_t nbytes) {
        std::vector<uint16_t> w16(nbytes / 18 * 32);
        for (size_t b = 0; b < nbytes / 18; b++) {
            uint16_t d_raw;
            std::memcpy(&d_raw, src + b * 18, 2);
            float d = f16_to_f32(d_raw);
            const uint8_t* nib = src + b * 18 + 2;
            for (int j = 0; j < 16; j++) {
                w16[b * 32 + j] = f32_to_f16_rne(d * ((nib[j] & 0xF) - 8));
                w16[b * 32 + 16 + j] = f32_to_f16_rne(d * ((nib[j] >> 4) - 8));
            }
        }
        std::vector<uint8_t> wb(w16.size() * 2);
        std::memcpy(wb.data(), w16.data(), wb.size());
        return wb;
    }

    // 007 US2 Route A 准入门: 返回 0=否 / 1=Gather(embed 表, 设备 exec_gather
    // 行级 dequant 消费) / 2=对齐非 batched FullyConnected/MatMul(设备
    // exec_matmul_f16_hmx Q4 分支消费, 标量路=硬错误)。
    static int q4_consumer_kind(const OpDef* consumer) {
        if (!consumer || !consumer->name_tag || !consumer->name_tag->name()) return 0;
        const char* n = consumer->name_tag->name();
        if (!std::strcmp(n, "Gather")) return 1;
        if (std::strcmp(n, "FullyConnected") && std::strcmp(n, "MatMul")) return 0;
        uint32_t m = 0, k = 0, nn = 0, batched = 0;
        if (n[0] == 'F') {
            ExtraFc ef{};
            if (consumer->serialized_extra.size() < sizeof(ef)) return 0;
            std::memcpy(&ef, consumer->serialized_extra.data(), sizeof(ef));
            m = ef.m; k = ef.k; nn = ef.n;
        } else {
            ExtraMatMul em2{};
            if (consumer->serialized_extra.empty()) return 0;
            size_t c = std::min(consumer->serialized_extra.size(), sizeof(em2));
            std::memcpy(&em2, consumer->serialized_extra.data(), c);
            m = em2.m; k = em2.k; nn = em2.n; batched = em2.batched;
        }
        if (m && k && nn && !batched &&
            (m % 32u == 0u) && (k % 32u == 0u) && (nn % 32u == 0u)) return 2;
        return 0;
    }

    // 权重消费 op 的 const 输入 → 权重槽(按 const id 去重; GGUF 供给优先)
    // consumer = 消费节点(Q4 准入按消费者类型/形状裁决; loader 只给计算 op
    // 设 grouping, 权重 const 自身没有; TSV 键 = 消费节点名)
    uint32_t ensure_weight_slot(GraphPrepare& gp, const OpDef* w,
                                std::map<uint64_t, uint32_t>& wslots,
                                const OpDef* consumer = nullptr) {
        const std::string consumer_grp = consumer ? consumer->grouping : "";
        const int q4k = q4_consumer_kind(consumer);
        auto it = wslots.find(w->op_id);
        if (it != wslots.end()) {
            /* 已建 w4 tile 槽又被非 Q4 消费者复用(tie 权重 M 不齐等) → 设备侧
             * 该消费者无 Q4 路径, 编译期即报(同 wt_parse rc 纪律, 不等上板死) */
            if (q4k == 0 && wslot_is_q4.count(w->op_id) && wslot_is_q4[w->op_id]) {
                std::fprintf(stderr,
                             "error: Q4 tile 槽 (op %llu) 被非 Q4 消费者 %s 复用, "
                             "需拆分权重 const\n",
                             (unsigned long long)w->op_id, consumer_grp.c_str());
                std::exit(4);
            }
            return it->second;
        }
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
                    const std::string layer =
                        consumer_grp.empty() ? w->grouping : consumer_grp;
                    if (e.type == 2 && q4k != 0 && K % 32 == 0 && NN % 32 == 0) {
                        /* 007 US2: Q4_0 → 640B/tile 纯重排(量化零损); 设备
                         * exec_matmul_f16_hmx / exec_gather 的 Q4 分支按槽长
                         * 判别消费。tie(同 GGUF 张量多图 const)共槽 */
                        const auto dkey = std::make_pair(e.file_offset, e.nbytes);
                        auto dit = gguf_w4_dedup.find(dkey);
                        if (dit != gguf_w4_dedup.end()) {
                            wslots[w->op_id] = dit->second;
                            wslot_is_q4[w->op_id] = true;
                            w4_records.push_back({layer, mit->first, "w4", "dedup", 0});
                            gguf_hits++;
                            return dit->second;
                        }
                        auto tile = repack_q4_0_tiles(src, e.nbytes, K, NN);
                        id = slot_for((uint32_t)tile.size(), (uint32_t)(K * NN), tile.data());
                        gguf_w4_dedup[dkey] = id;
                        w4_records.push_back({layer, mit->first, "w4", "none", tile.size()});
                        wslot_is_q4[w->op_id] = true;
                    } else if (e.type == 2) {
                        /* Q4_0 但非 tile 候选(形状未对齐/非 Q4 消费者/batched)
                         * → 反量化 f16 槽(T014 兜底, 报告列名) */
                        auto wb = dequant_q4_0_f16(src, e.nbytes);
                        id = slot_for((uint32_t)wb.size(), (uint32_t)(K * NN), wb.data());
                        w4_records.push_back({layer, mit->first, "f16",
                                              (K % 32 == 0 && NN % 32 == 0)
                                                  ? "consumer_gate" : "shape_misalign",
                                              wb.size()});
                    } else {  // F32 → f16 (by-design, 非 Q4 候选)
                        size_t n4 = e.nbytes / 4;
                        std::vector<uint16_t> w16(n4);
                        const float* wf = reinterpret_cast<const float*>(src);
                        for (size_t i = 0; i < n4; i++) w16[i] = f32_to_f16_rne(wf[i]);
                        std::vector<uint8_t> wb(n4 * 2);
                        std::memcpy(wb.data(), w16.data(), wb.size());
                        id = slot_for((uint32_t)wb.size(), (uint32_t)n4, wb.data());
                        w4_records.push_back({layer, mit->first, "f16", "none", wb.size()});
                    }
                    wslots[w->op_id] = id;
                    gguf_hits++;
                    return id;
                }
            }
            /* FR-003: matmul(准入门=2) 权重不在匹配表 = 表不全/GGUF 不配套,
             * 硬错误; 其余消费者 miss 走 const_pool f16(数值正确, 仅体积回退)
             * 并记录 source_missing 供报告点名 */
            if (q4k == 2) {
                w4_missing.push_back(consumer_grp.empty() ? w->grouping : consumer_grp);
                w4_records.push_back({consumer_grp.empty() ? w->grouping : consumer_grp,
                                      "-", "f16", "source_missing", 0});
            }
            gguf_miss++;
        }
        size_t n = (size_t)w->const_data_size / 4;
        if (n == 0) {
            // 空 const(记录在但池数据被 DCE 删): 最小零槽占位(M4 数值门兜底)
            std::fprintf(stderr, "warn: empty weight const (op %llu, name=%s, rank=%u), 零槽占位\n",
                         (unsigned long long)w->op_id,
                         (w->name_tag && w->name_tag->name()) ? w->name_tag->name() : "?",
                         w->output_def.rank);
            std::vector<uint8_t> zeros(128, 0);
            uint32_t id = slot_for(128, 64, zeros.data());
            wslots[w->op_id] = id;
            return id;
        }
        /* es=2 池宽度双约定判别: 权重/物化 const = f32 拓宽(4B/elem, 下方
         * f32→f16 洗回); const_fold 自产 const = f16-native(2B/elem, 字节
         * 即设备面, 原样入槽)。size == 2*elems 与 4*elems 互斥, 判别无歧义 */
        {
            size_t dims_elems = 1;
            for (uint32_t i = 0; i < w->output_def.rank && i < 5; ++i)
                dims_elems *= (size_t)w->output_def.dims[i];
            if (w->output_def.element_size == 2 && dims_elems > 0 &&
                (size_t)w->const_data_size == dims_elems * 2 &&
                w->const_data_offset + dims_elems * 2 <= gp.const_pool().size()) {
                const uint8_t* src = gp.const_pool().data() + w->const_data_offset;
                uint32_t id = slot_for((uint32_t)(dims_elems * 2), (uint32_t)dims_elems, src);
                wslots[w->op_id] = id;
                return id;
            }
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
        wslots[w->op_id] = id;
        return id;
    }
    /* int 索引 const 原样入槽 (ScatterNd/Gather 索引等; 设备按 int32 直读)。
     * 走 ensure_weight_slot 会被 f32→f16 洗白: 小整数 int32 按位当 denormal
     * f32 → 全零 (0.8B GDN state ScatterNd 索引槽全零实锤 — 16 个更新全写
     * 同一坐标 (0,0,0,0,0), state 路径常数化, 层间彻底解耦)。 */
    uint32_t ensure_i32_slot(GraphPrepare& gp, const OpDef* w) {
        auto it = i32_slots.find(w->op_id);
        if (it != i32_slots.end()) return it->second;
        size_t nb = (size_t)w->const_data_size;
        uint32_t id;
        if (nb >= 4 && w->const_data_offset + nb <= gp.const_pool().size()) {
            const uint8_t* src = gp.const_pool().data() + w->const_data_offset;
            id = add_slot((uint32_t)nb, (uint32_t)(nb / 4), src);
        } else {
            std::vector<uint8_t> zeros(128, 0);
            id = add_slot(128, 64, zeros.data());
        }
        i32_slots[w->op_id] = id;
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
        // 有池数据即按 const 槽(is_const 标志对 loader 的 tensor_param const 不可靠)。
        // consumer = cur_op(主循环发射位上的消费 op): q4 准入/复用防护按真实
        // 消费者裁决 —— 不带 consumer 会被判 kind 0, 把 Gather(embed 表) 等
        // 合法 w4 消费误杀在复用防护门上
        if (p && (p->is_const() || p->const_data_size > 0)) {
            const OpDef* cons = cur_op && cur_op != op_id ? gp.get_op_at(cur_op) : nullptr;
            uint32_t s = ensure_weight_slot(gp, p, wslots, cons);
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
        op_annot.push_back(cur_engine);
    }
    void add_op(uint16_t opcode, const std::vector<uint32_t>& args) {
        emitted_ids.push_back(cur_op);
        wt_op o{};
        o.opcode = opcode;
        o.n_args = (uint16_t)args.size();
        uint32_t i = 0;
        for (uint32_t a : args) o.args[i++] = a;
        ops.push_back(o);
        op_annot.push_back(cur_engine);
    }
};

/* T003 成本模型 (R4): 编译期给 op 指派计算单元 + 回落策略, 打包 v3 注解 u32。
 * OpDef 不携带 store type (RESOURCE_HVX/HMX 在 mid-level IR Op 上, 序列化时
 * 被丢弃), 故从 serialized_extra (matmul 的 M/K/N) + output_def (elemwise
 * 形状) 重推, 口径镜像运行期 exec_matmul_f16 / exec_unary 的 gate:
 *   matmul 家族: 非 batched 且 M/K/N 全 %32 且 need_mm ≤ 16MB → HMX; 否则 scalar
 *   conv: HMX (exec_conv2d_hmx tile 几何)
 *   elemwise 家族: 输出元素数 %32 → HVX; 否则 scalar
 *   其余/控制/内存 op: scalar (无加速单元)
 * 回落: 恒允许 (fallback_allowed=1) 回落 scalar (数值敏感禁止回落由 US3 细化)。
 */
inline uint32_t assign_engine(const OpDef* od, const std::string& nm) {
    uint32_t engine = 0u;          /* WT_ENG_SCALAR */
    uint32_t fb_allowed = 1u, fb_engine = 0u;
    /* 决策留痕: GEHTP_DEBUG_ENGINE=1 时逐 op 打印引擎+原因 —— 运行期"为什么
     * 全标量"的定位入口(例: t2l matmul M=8 不 32 对齐→scalar, 一行即判)。 */
    char why[160];
    auto need_mm = [](uint32_t m, uint32_t k, uint32_t n) -> size_t {
        size_t a = (size_t)m * k * 2u, w = (size_t)n * k * 2u, o = (size_t)m * n * 2u;
        size_t off_w = (a + 2047u) & ~(size_t)2047u;
        size_t off_o = (off_w + w + 2047u) & ~(size_t)2047u;
        return ((off_o + o + 2047u) & ~(size_t)2047u) + 2048u;
    };
    const size_t kVTCM = 16u * 1024u * 1024u;
    if (nm == "MatMul" || nm == "FullyConnected") {
        uint32_t m = 0, k = 0, n = 0;
        bool has_bias = false;
        if (nm == "FullyConnected") {
            ExtraFc ef{};
            if (od->serialized_extra.size() >= sizeof(ef))
                std::memcpy(&ef, od->serialized_extra.data(), sizeof(ef));
            m = ef.m; k = ef.k; n = ef.n; has_bias = ef.has_bias != 0;
        } else {
            ExtraMatMul em{};
            if (!od->serialized_extra.empty()) {   /* min-size: 旧 bin 24B 无 has_bias */
                size_t c = std::min(od->serialized_extra.size(), sizeof(em));
                std::memcpy(&em, od->serialized_extra.data(), c);
            }
            m = em.m; k = em.k; n = em.n; has_bias = em.has_bias != 0;
        }
        if (has_bias)
            snprintf(why, sizeof why, "has_bias (HMX bias path unverified)");
        else if ((m % 32u) || (k % 32u) || (n % 32u))
            snprintf(why, sizeof why, "shape %ux%ux%u not 32-aligned", m, k, n);
        else if (need_mm(m, k, n) > kVTCM)
            snprintf(why, sizeof why, "scratch need %llu > VTCM %llu (m%u k%u n%u)",
                     (unsigned long long)need_mm(m, k, n),
                     (unsigned long long)kVTCM, m, k, n);
        else {
            engine = 2u;           /* WT_ENG_HMX; batched runtime loops per batch */
            snprintf(why, sizeof why, "m%u k%u n%u 32-aligned, need %llu <= 16MB",
                     m, k, n, (unsigned long long)need_mm(m, k, n));
        }
    } else if (nm == "Conv2d") {
        engine = 2u;               /* WT_ENG_HMX */
        snprintf(why, sizeof why, "conv has HMX path");
    } else if (nm == "Add" || nm == "Eltwise_Binary" || nm == "Eltwise_Unary" ||
               nm == "ElementWiseNeuron" || nm == "Softmax") {
        /* 003 精度修正: 仅这些 op 名有默认开启的 HVX 内核。其余 elemwise/内存 op
         * (Transpose/Gather/Concat/Reshape/Split/Reduce/Pad/StridedSlice/Ternary)
         * 无 HVX 内核, RmsNorm 的 HVX 内核默认关(精度 0.5× 偏差未修) —— 一律 scalar,
         * 勿虚报 HVX 致运行期回落/静默替换 (FR-004/FR-008)。 */
        uint64_t n = 1;
        for (uint32_t i = 0; i < od->output_def.rank && i < 5; ++i)
            n *= (uint64_t)od->output_def.dims[i];
        if ((n % 32u) == 0u) {
            engine = 1u;           /* WT_ENG_HVX */
            snprintf(why, sizeof why, "n=%llu 32-aligned", (unsigned long long)n);
        } else {
            engine = 0u;           /* WT_ENG_SCALAR */
            snprintf(why, sizeof why, "n=%llu not 32-aligned", (unsigned long long)n);
        }
    } else {
        snprintf(why, sizeof why, "no fast kernel for '%s'", nm.c_str());
    }
    if (getenv("GEHTP_DEBUG_ENGINE")) {
        static const char* eng_name[] = {"SCALAR", "HVX", "HMX"};
        std::fprintf(stderr, "[engine] op %llu %-18s -> %-6s (%s)\n",
                     (unsigned long long)od->op_id, nm.c_str(),
                     eng_name[engine & 3u], why);
    }
    return wt_annot_pack(engine, fb_allowed, fb_engine);
}

/* 跨 handler 共享的 emit() 函数域状态(重构前 Conv2d 分支隐式捕获):
 * H/W/C = 图输入 NCHW 几何; w_slot/b_slot = 首个 Conv2d 的权重/偏置槽
 * (op 循环前预登记, 槽序敏感勿移)。 */
struct WtopEmitShared {
    uint32_t H = 1, W = 1, C = 1;
    uint32_t w_slot = 0, b_slot = 0;
};

/* 007 US1: broadcast 形态判定(孪生) — 正典 = specs/007-perf-sprint-1/evidence/
 * broadcast_census.md "判定规则"节; 设备侧孪生 = kernels/src/runtime/
 * oplist_exec.c broadcast_shape_class (判据互指, 勿单边改)。
 * 1=整块(重复/铺标量) 2=行循环(拷行/铺行) 3=通用退化(strided)。
 * 输入 = emit 进 args 的 fold4 后 4D in_d/out_d(与设备读到的完全同一数组)。 */
inline int broadcast_shape_class(const uint32_t in_d[4], const uint32_t out_d[4])
{
    uint32_t r = 4;
    while (r > 0 && out_d[r - 1] == 1) r--;
    if (r == 0) r = 1;
    uint32_t ir = 4;
    if (ir > r) ir = r;
    if (in_d[ir - 1] != out_d[r - 1] && in_d[ir - 1] != 1) return 3; /* 末轴 wrap */
    uint32_t lead = r - ir;
    int outer_bcast = 1;
    for (uint32_t d = 0; d + 1 < ir; d++) {
        if (in_d[d] == 1) continue;               /* 外轴广播 */
        if (in_d[d] != out_d[d + lead]) return 3; /* 外轴 wrap */
        outer_bcast = 0;
    }
    return outer_bcast ? 1 : 2;
}

/* 注入的 OP_BROADCAST_F16 子 op 引擎注解 (007 P1 修正: 父 op assign_engine
 * 的 n%32 判据看不见 broadcast 自身形状 —— 按形态分派: 类 1/2 → HVX 纯搬运,
 * 类 3 → scalar 计划(合法非回落, dev=0); fb_allowed 恒 1 沿 003 契约)。 */
inline uint32_t broadcast_engine_annot(const uint32_t in_d[4], const uint32_t out_d[4])
{
    return wt_annot_pack(broadcast_shape_class(in_d, out_d) == 3 ? 0u : 1u, 1u, 0u);
}

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
    case 3:  return 4;   /* EQUAL → bool 0/1 (0.8B pad 掩码; 静默 ADD 兜底=设备全 -inf 实锤) */
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
    case 7:  return 13;  /* SOFTPLUS (GDN 门控 ×18; 稳定化公式见 exec_unary case 13) */
    default: return 0xFFFFFFFFu;
    }
}

} // namespace wtop

#endif // WTOP_OPS_HPP
