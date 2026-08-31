// ============================================================================
// M36 反汇编保真实现 —— 见 include/hnnx/vtcm/create_supertiles_m36.hpp 头注
// 证据基线: audit_verify/reports/M36_create_supertiles_disasm.md (2026-08-28)
// 每段逻辑的 [0x地址] 指向 real_so/libHtpPrepare.so 的指令；
// 未逐指令确认处一律注明"遗留"，禁止臆测补齐。
// ============================================================================
#include "hnnx/vtcm/create_supertiles_m36.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace hnnx {

// ============================================================================
// OpIdTable —— 三个解码函数的字面移植
// ============================================================================

// [0x12face0 全指令]
//   12face6: shr 0xa                → idx = id>>10
//   12facfc: cmp rcx(idx-1), rax(n); jbe error
//   12fad01: tag = entries[idx-1]   (1 基下标: -0x4(%r8,%rsi,4))
//   12fad0e: (tag & 0xF8000000)==0x78000000 → 远跳
//   12fad16: delta = tag & 0xFFFFF; 越界查 idx+delta-1；idx += delta；重读 tag
//   12fad31: bit30 且 bit31(负) → 越界查 idx 后返回
uint64_t OpIdTable::resolve_table_index(uint64_t id, bool* error) const {
    auto fail = [&]() { if (error) *error = true; return 0ull; };
    if (error) *error = false;
    if (id == 0) return 0;                       // [0x12face1 test rsi; je → xor esi]
    uint64_t idx = id >> 10;                     // [0x12face6]
    const uint64_t n = static_cast<uint64_t>(entries.size());
    if (n <= idx - 1) return fail();             // [0x12facff jbe 0x12fad4e]
    uint32_t tag = entries[idx - 1];             // [0x12fad01]
    if ((tag & 0xF8000000u) == FAR_MARKER) {     // [0x12fad0e]
        uint64_t delta = tag & 0xFFFFFu;         // [0x12fad16]
        if (n <= idx + delta - 1) return fail(); // [0x12fad24 cmp; jbe]
        idx += delta;                            // [0x12fad29]
        tag = entries[idx - 1];                  // [0x12fad2c]
    }
    if ((tag & NEG_BIT30) && (tag & NEG_BIT31)) {
        if (n <= idx) return fail();             // [0x12fad3d cmp rsi,rax; jbe]
    }
    return idx;                                  // [0x12fad42]
}

// [0x12fad60 全指令]
//   与 resolve_table_index 同构，差别仅两处：
//   12fad84: esi = id & 0x3ff                      (chunk 序号保留)
//   12fadb2-0x12fadb9: idx += delta; **rsi |= delta<<10**
//     —— 注意：结果用的是 delta 本身左移，不是 (idx+delta)。远跳标记的低 20 位
//     存的是目标表项的绝对下标（越界检查却按 idx+delta 算）——字面保留此行为。
uint64_t OpIdTable::resolve_full_id(uint64_t id, bool* error) const {
    auto fail = [&]() { if (error) *error = true; return 0ull; };
    if (error) *error = false;
    if (id == 0) return 0;                       // [0x12fad64]
    uint64_t idx = id >> 10;                      // [0x12fad69]
    const uint64_t n = static_cast<uint64_t>(entries.size());
    if (n <= idx - 1) return fail();              // [0x12fad7f]
    uint64_t low = id & CHUNK_MASK;               // [0x12fad84]
    uint32_t tag = entries[idx - 1];              // [0x12fad8a]
    if ((tag & 0xF8000000u) == FAR_MARKER) {      // [0x12fad97]
        uint64_t delta = tag & 0xFFFFFu;          // [0x12fad9f]
        if (n <= idx + delta - 1) return fail();  // [0x12fadad]
        idx += delta;                             // [0x12fadb2]
        low |= delta << 10;                       // [0x12fadb5-0x12fadb9]
        tag = entries[idx - 1];                   // [0x12fadbc]
    }
    if ((tag & NEG_BIT30) && (tag & NEG_BIT31)) {
        if (n <= idx) return fail();              // [0x12fadcd]
    }
    return low;                                   // [0x12fadd2]
}

// [0x12faf20 全指令]
//   12faf21: r8=0（默认返回 0）
//   12faf73: bit30 清 → 返回 0
//   12faf7a: 负 tag（bit30+bit31）→ 越界查后读 entries[idx]（下一项）原值 [0x12faf90]
//   12faf7e: 正常 → tag & 0x3FFFFF
uint32_t OpIdTable::extract_field(uint64_t id, bool* error) const {
    auto fail = [&]() { if (error) *error = true; return 0u; };
    if (error) *error = false;
    if (id == 0) return 0;                       // [0x12faf27 je → 12faf86 返回 r8=0]
    uint64_t idx = id >> 10;                      // [0x12faf29]
    const uint64_t n = static_cast<uint64_t>(entries.size());
    if (n <= idx - 1) return fail();              // [0x12faf42]
    uint32_t tag = entries[idx - 1];              // [0x12faf44]
    if ((tag & 0xF8000000u) == FAR_MARKER) {      // [0x12faf51]
        uint64_t delta = tag & 0xFFFFFu;         // [0x12faf59]
        if (n <= idx + delta - 1) return fail(); // [0x12faf66]
        idx += delta;                            // [0x12faf6b]
        tag = entries[idx - 1];                  // [0x12faf6e]
    }
    if ((tag & NEG_BIT30) == 0) return 0;        // [0x12faf78 je → 返回 r8=0]
    if (tag & NEG_BIT31) {                       // [0x12faf7c js]
        if (n <= idx) return fail();             // [0x12faf8b]
        return entries[idx];                     // [0x12faf90] 下一项原值
    }
    return tag & 0x3FFFFFu;                      // [0x12faf7e]
}

// [0x12fadf0] 仅前 12B 指令级：
//   cmp rdx,rsi; jne → al=1；a==b → 0；a==0 分支跳 0x12fae7c 未 dump（遗留）
int OpIdTable::compare_ids(uint64_t a, uint64_t b) {
    if (a == b) return 0;                        // [0x12fadf6]
    return 1;                                    // [0x12fadfa]（a==0 特例未证）
}

void OpIdTable::grow_for(uint64_t max_id) {
    // 读取端 1 基: idx = id>>10 (≥1) → entries[idx-1]。因此 id 空间 (k+1)<<10
    // 由 entries[k] 覆盖；表大小须 ≥ max_id>>10。
    // 真表按 0x2000B（0x800 项）步长增长；此处按需增长等价覆盖。
    const uint64_t need = max_id >> 10;
    if (entries.size() < need) entries.resize(need, 0);
}
void OpIdTable::set_plain(uint64_t id, uint32_t tag) {
    // id < 0x400（idx=0）时读取端会下溢 — 真代码读 entries[-1]（越界未防），
    // 本实现直接忽略写入；读取端对该区间报 error（见 resolve_* 的 n<=idx-1）。
    if ((id >> 10) == 0) return;
    grow_for(id);
    entries[(id >> 10) - 1] = tag;               // 1 基: 覆盖 id 空间 (k+1)<<10 的是 entries[k]
}
void OpIdTable::set_far(uint64_t id, uint64_t target_entry_index) {
    set_plain(id, FAR_MARKER | static_cast<uint32_t>(target_entry_index & 0xFFFFFu));
}
void OpIdTable::set_negative(uint64_t id) {
    set_plain(id, NEG_BIT30 | NEG_BIT31);
}

// ============================================================================
// autothread_size [0x10c3690 全指令]
//   tbl @0x39b7650 = {1, 8, 8, 32}（文件字节实测，M36 §0.2）
//   round_up: ((M+q-1) - (M+q-1)%M)   [原式字面，非 (x+M-1)&~(M-1) 的位技巧]
// ============================================================================
uint32_t autothread_size(uint32_t num_threads, uint32_t size_field, unsigned idx) {
    static const uint32_t tbl[4] = {1, 8, 8, 32};   // @0x39b7650
    if (idx > 3) idx = 3;                            // 真代码无此钳位（表只有 4 项，越界未证）
    if (num_threads < 2) return size_field;          // [T<2 分支]
    const uint32_t M = tbl[idx];
    const uint64_t q = ((uint64_t)size_field + num_threads - 1) / num_threads;  // ceil
    const uint64_t ru = (M + q - 1) - (M + q - 1) % M;                          // round_up
    return static_cast<uint32_t>(std::min<uint64_t>(ru, size_field));
}

// ============================================================================
// insert_spill_fill [0x106d7d0 38B 全指令]
//   本 so 构建为永久桩：qnndsp_log 后 mov $0xffffffff,%eax; ret
// ============================================================================
int insert_spill_fill_stub(std::string* log_out) {
    if (log_out) *log_out =
        "insert_spill_fill.cc:17::ERROR:insert_spill_fill not supported";  // fmt@0x462adf1, file@0x462ae1f
    return -1;                                        // [0x106d7d0 尾]
}

// ============================================================================
// fc5910 —— 依赖记录有效切片数 [全指令，见头注]
// ============================================================================
uint32_t dep_effective_num_slices(const SupertileDepRecord& rec,
                                  const std::vector<uint32_t>& chunk_slice_counts,
                                  uint32_t self_slicing_count) {
    const uint32_t flags = rec.flags;
    if (flags & 0x200000u) {                          // [0xfc5921 testl $0x200000]
        // v = get_supertiled_ops_info(this, op) [0xfc5951]；
        // 字节长度 = 元素数*16（每元素 16B: Op* + 辅助 u64）
        const uint64_t bytes = static_cast<uint64_t>(chunk_slice_counts.size()) * 16;
        if (bytes <= 0x1f)                            // [0xfc5968 cmpq $0x1f; jbe → throw]
            throw std::runtime_error("num_internal_threads");  // 串@0x4623207 [0xfc59a9]
        if (rec.extra_bits_2 & 0x40u) {               // [0xfc596e testb $0x40,0x2(%r15)]
            uint32_t sum = 0;                         // [0xfc5990 循环: 逐元素 self_slicing_num_slices]
            for (uint32_t c : chunk_slice_counts) sum += c;
            return sum;
        }
        return static_cast<uint32_t>(bytes / 16);     // [0xfc5975 shrq $4]
    }
    if (flags & 0x400000u) {                          // [0xfc592e testl $0x400000]
        return self_slicing_count;                    // [0xfc5945 尾调 self_slicing_num_slices]
    }
    return 1;                                         // [0xfc5928 movl $1]
}

// ============================================================================
// fc5af0 —— supertile per-chunk 依赖注册 [头部已证段全指令]
// ============================================================================
SupertileDepRegStatus register_supertiled_dep(
    std::vector<GrdepOpRecord>& op_records, uint32_t rec_op_index,
    uint32_t count, uint64_t chunk_id,
    std::vector<std::vector<uint64_t>>& chunk_id_store) {
    if (count <= 1) return SupertileDepRegStatus::EarlyExit;      // [0xfc5b1a cmpl $1; jbe]
    if (rec_op_index == 0 || rec_op_index > op_records.size())
        // 真代码 (idx-1)*0xd0 对 idx=0 会 u32 回绕成巨数野读（未防御）；此处按
        // BadResourceFlags 收敛并留痕 —— 与真行为差异已声明
        return SupertileDepRegStatus::BadResourceFlags;
    const uint64_t f = op_records[rec_op_index - 1].resource_flags;  // [0xfc5b3d +0x20]
    const uint64_t q = f & 0xc;                                      // [0xfc5b42]
    const bool power = (q == 4 || q == 8);  // (q-1)&q==0 且 q!=0 ⇔ q∈{4,8} [0xfc5b45-0xfc5b4a]
    const bool nonzero = q != 0;                                     // [0xfc5b52]
    if (!(power && nonzero))
        return SupertileDepRegStatus::BadResourceFlags;  // → ERROR 3457, grdep_main.cc@0x46229f9 [0xfc5c02]
    // [0xfc5ba6] operator new(count*8) 建 u64 数组；尾部(0xfc5c3a 后)写法未逐指令
    // （遗留#5）。按 create 循环已证的 id 步进语义，填 chunk_id..chunk_id+count-1。
    if (chunk_id_store.size() < op_records.size()) chunk_id_store.resize(op_records.size());
    auto& slot = chunk_id_store[rec_op_index - 1];
    slot.clear();
    slot.reserve(count);
    for (uint32_t k = 0; k < count; ++k) slot.push_back(chunk_id + k);
    return SupertileDepRegStatus::Ok;
}

// ============================================================================
// Phase 3 除数搜索
// ============================================================================
uint32_t supertile_find_divisor(uint64_t c, uint32_t b, uint32_t cnt_start) {
    // [0x131415e-0x13143ce 区间] cnt 从候选向下递减直到 (c*b)%cnt==0。
    // 操作数 c/b 的实体归属未逐指令钉死（遗留：c=尺寸累计、b=成员数是当前读法）
    const uint64_t product = c * static_cast<uint64_t>(b);
    uint32_t cnt = cnt_start ? cnt_start : 1;
    while (cnt > 1 && (product % cnt) != 0) --cnt;
    return cnt;
}

uint32_t supertile_next_chunk_seq(uint32_t seq) {
    // create 循环: id+1 且 & 0x3ff 回绕；跨 0x400 倍数处跳过 [已证]；
    // 0x7ff 哨兵跳过为半证（"skip 0x7ff"）—— 遗留，行为按字面保留
    uint32_t n = seq + 1;
    if ((n & 0x3ffu) == 0) n += 1;      // 跳过 0x400 倍数
    if (n == 0x7ffu) n += 1;            // 半证哨兵
    return n;
}

// ============================================================================
// decode_tag_triples —— 0x12fa730 尾部 [0x12fa880-0x12fab07 指令级]
//   根三元组（进入时用本 id 的表项 tag）:
//     type  = (tag >> 27) & 7            [0x12fa8f6]
//     value = (tag & 0x7FFFFFF) + 1      [0x12fa89d → 0x12fa8f2]（无符号位分支）
//     seq   = id & 0x3ff                 [0x12fa909, 种子存 0x18(%rsp) @0x12fa8a7]
//     链种子 next = entries[idx] 原值（0 基下一项）[0x12fa896 movl (%r8,%rbx,4)]
//       —— 加载点在 dump 起点之前，归属半证
//   循环体（每步一条）:
//     idx = next>>10; 越界 n<=idx-1 → 0xcf27a0        [0x12fa960-0x12fa96e]
//     seq = next & 0x3ff                               [0x12fa974]
//     tag = entries[idx-1]                             [0x12fa97d]
//     远跳 tag: idx += delta; seq |= delta<<10; 重读   [0x12fa990-0x12fa9b1]
//     bit30 清 → next = 0（终链）                      [0x12fa9be]
//       bit31 置: value = (tag & 0x7FFFFFF)+1          [0x12fa930-0x12fa937]
//       bit31 清: value = (tag>>22 & 0x1f)+1           [0x12fa9ec → 0x12fa937]
//     bit30 置:
//       bit31 清: next = tag & 0x3FFFFF; value 同上 field+1 [0x12fa9e4-0x12fa9f4]
//       bit31 置: 越界 n<=idx → 错; next = entries[idx] 原值
//                 value = (tag & 0x7FFFFFF)+1          [0x12fa9fc-0x12faa09]
//     type = (tag >> 27) & 7                            [0x12fa93a-0x12fa93d]
//   输出次序: 本实现按访问序（根在前）。真代码先数链长（0x12faa68-0x12faafb
//   预扫），negl ebp 后从槽 count-1 倒序写 [0x12fab12/0x12fa8fe] —— 数组物理
//   次序相反，已声明差异；下游消费方向未证（遗留）。
//   遗留: 追加原语 0x12fb8a0 内部；尾部结果记录 {终值, 终表项下标}
//         [0x12faa0e-0x12faa40] 不建模；预扫入口 0x12fa823 在 dump 外。
// ============================================================================
size_t decode_tag_triples(const OpIdTable& t, uint64_t id,
                          std::vector<SupertileTagTriple>& out, bool* error) {
    auto fail = [&]() { if (error) *error = true; return out.size(); };
    if (error) *error = false;
    const uint64_t n = static_cast<uint64_t>(t.entries.size());

    // ---- 根表项解析（远跳处理与循环体同式；根侧远跳在 dump 外，半证）----
    uint64_t idx = id >> 10;
    if (id == 0 || n <= idx - 1) return fail();        // idx==0 → idx-1 回绕巨数 → 越界
    uint32_t tag = t.entries[idx - 1];
    uint64_t seq = id & 0x3ff;
    if ((tag & 0xF8000000u) == OpIdTable::FAR_MARKER) {
        const uint64_t delta = tag & 0xFFFFFu;
        if (n <= idx + delta - 1) return fail();
        idx += delta;
        seq |= delta << 10;                            // [0x12fa9ae 同式]
        tag = t.entries[idx - 1];
    }
    // 链种子 = 下一表项原值（0 基 entries[idx]），读取前越界查
    // [0x12fa88d cmp rbx,rcx; jbe 错误 / 0x12fa896 movl (%r8,%rbx,4)]
    if (n <= idx) return fail();
    uint64_t next = t.entries[idx];
    // 根三元组: value 恒 payload27+1（不看符号位）[0x12fa89d/0x12fa8f2]
    out.push_back({(tag >> 27) & 7u, (tag & 0x7FFFFFFu) + 1u,
                   static_cast<uint32_t>(seq)});

    // ---- 链循环 [0x12fa960-0x12faa09] ----
    while (next != 0) {
        uint64_t i2 = next >> 10;                      // [0x12fa963]
        if (n <= i2 - 1) return fail();                // [0x12fa96e]（含 next<0x400 回绕）
        uint64_t seq2 = next & 0x3ff;                  // [0x12fa974]
        uint32_t tg = t.entries[i2 - 1];               // [0x12fa97d]
        if ((tg & 0xF8000000u) == OpIdTable::FAR_MARKER) {   // [0x12fa98e]
            const uint64_t delta = tg & 0xFFFFFu;
            if (n <= i2 + delta - 1) return fail();    // [0x12fa9a1]
            i2 += delta;                               // [0x12fa9a7]
            seq2 |= delta << 10;                       // [0x12fa9aa-0x12fa9ae]
            tg = t.entries[i2 - 1];                    // [0x12fa9b1]
        }
        uint32_t value;
        if ((tg & OpIdTable::NEG_BIT30) == 0) {        // [0x12fa9b6 test → bit30 清]
            next = 0;                                  // 终链 [0x12fa9be]
            value = (tg & 0x80000000u)                 // bit31 定形态 [0x12fa9c9 jns]
                        ? (tg & 0x7FFFFFFu) + 1u       // [0x12fa930-0x12fa937]
                        : ((tg >> 22) & 0x1fu) + 1u;   // [0x12fa9ec-0x12fa9f1 → +1]
        } else if ((tg & 0x80000000u) == 0) {          // bit30 置、正 [0x12fa9e2 jns]
            next = tg & 0x3FFFFFu;                     // [0x12fa9e4-0x12fa9e6]
            value = ((tg >> 22) & 0x1fu) + 1u;
        } else {                                       // bit30+bit31 同置 [0x12fa9fc]
            if (n <= i2) return fail();                // [0x12fa9ff]
            next = t.entries[i2];                      // [0x12faa05] 下一表项原值
            value = (tg & 0x7FFFFFFu) + 1u;            // [jmp 0x12fa930]
        }
        out.push_back({(tg >> 27) & 7u, value, static_cast<uint32_t>(seq2)});
    }
    return out.size();
}

// ============================================================================
// create_supertiles 三阶段主体 [0x1313ac0]
// ============================================================================
std::vector<SupertileGroupResult> create_supertiles_disasm(
    const OpIdTable& id_table,
    const std::vector<SupertileCandidate>& ops,
    const CreateSupertilesConfig& cfg,
    CreateSupertilesStats* stats) {
    if (stats) stats->errors.clear();

    // ---- Phase 1 [0x1313c58 起]: map<pair<resolved_id, tag>, vec<u32>> 分组 ----
    // resolved_id 经 resolve_full_id（真函数调 0x12fad60 [0x13155f5 同族调用点]）
    // 记录门 [反编译 rec[0]/rec[8]/rec[0x98] 三非零]；名字黑名单 [0x1313b95 六连 cmp，
    //   6 个 map_str 字面量见 k_supertile_name_blacklist]；资格 [反编译谓词]：
    //   (hmx_count>1 && flags&8) || (hvx_count>1 && (flags&0x110004)==4)
    std::map<std::pair<uint64_t, uint32_t>, std::vector<const SupertileCandidate*>> groups;
    if (stats) stats->groups_in = ops.size();
    for (const auto& c : ops) {
        if (!c.record_valid) continue;      // rec[0]/rec[8]/rec[0x98] 门
        bool in_blacklist = false;          // 名字黑名单（空串=禁用占位）
        for (const char* bn : cfg.blacklist_names)
            if (*bn && c.op_name && std::strcmp(c.op_name, bn) == 0) { in_blacklist = true; break; }
        if (in_blacklist) continue;
        const bool hmx_ok = (cfg.hmx_count > 1) && (c.resource_flags & 0x8u);
        const bool hvx_ok = (cfg.hvx_count > 1) && ((c.resource_flags & 0x110004u) == 4u);
        if (!hmx_ok && !hvx_ok) continue;   // 资格不过（线程数 + 记录资源位联合判定）
        bool err = false;
        const uint64_t rid = id_table.resolve_full_id(c.op_id, &err);
        if (err) {
            if (stats) stats->errors.push_back("id-table resolve error @op " +
                                               std::to_string(c.op_id));
            continue;
        }
        groups[{rid, c.tag}].push_back(&c);
    }

    // ---- Phase 2 [0x1313efc-0x13141a0]: 单元素组 erase；组间按 resolved id 稳定排序 ----
    // （≥129 元素的 nothrow-halving scratch 是分配策略，不影响序，不单独建模）
    std::vector<std::pair<uint64_t, uint32_t>> order;
    order.reserve(groups.size());
    for (auto& [key, vec] : groups) {
        if (vec.empty()) {                  // 空组 → ERROR 406 分支（map 路径不会产生，防御）
            if (stats) stats->errors.push_back(
                "supertile.cc:406::ERROR:unexpected value size for grouping");  // fmt@0x55b4541
            continue;
        }
        if (vec.size() == 1) continue;      // 单元素组 → erase
        order.push_back(key);
    }
    // map 本身按 key 有序 ⇒ stable_sort 语义自动满足（等价实现，注释留痕）

    // ---- Phase 3 [0x131415e-0x13143ce]: 预算 + 除数 + chunk id ----
    std::vector<SupertileGroupResult> out;
    out.reserve(order.size());
    for (auto& key : order) {
        auto& vec = groups[key];
        SupertileGroupResult r;
        r.resolved_id = key.first;
        r.tag = key.second;

        // 预算: 哈希表 gp+0x6e40/0x6e48（键 id>>10，命中 entry[8]==hash &&
        // entry[0x10]==hash）→ gp[0x74c0]，否则 gp[0x5fd8]<<10
        const bool hit = std::find(cfg.hash_hit_ids.begin(), cfg.hash_hit_ids.end(),
                                   key.first >> 10) != cfg.hash_hit_ids.end();
        r.budget = hit ? cfg.budget_hash_hit : (cfg.budget_default);

        uint64_t total_bytes = 0;
        for (auto* c : vec) total_bytes += c->tensor_bytes;   // vtable +0x60/+0xa0, tensor+0xd0

        // 层级连续性判定（调 0x12fad60）：谓词未逐指令钉死（遗留），不实现，留痕:
        // TODO(M36 遗留): level continuity check

        uint32_t cnt = 1;
        if (r.budget != 0 && total_bytes > r.budget) {
            // 超预算 → 除数搜索决定 chunk 数（起点候选未钉死，取组员数，遗留声明）
            cnt = supertile_find_divisor(total_bytes,
                                         static_cast<uint32_t>(vec.size()),
                                         static_cast<uint32_t>(vec.size()));
            if (cnt < 1) cnt = 1;
        }

        // chunk 建组 [0x12fa220]: (entry_index<<10)|seq；seq 步进见 supertile_next_chunk_seq
        // entry_index 是 op-id 间接表的表项下标（resolve_table_index 的 idx，如
        // id 0x400 → idx 1），不是 resolved_id 的高位 —— 两组数在非远跳路径不同源
        bool idx_err = false;
        const uint64_t entry_index =
            id_table.resolve_table_index(vec[0]->op_id, &idx_err);
        if (idx_err && stats) stats->errors.push_back(
            "entry-index resolve error @op " + std::to_string(vec[0]->op_id));
        uint32_t seq = 0;                    // 起始 seq=0 为读法（未逐指令钉死，遗留）
        const size_t per = (vec.size() + cnt - 1) / cnt;
        for (uint32_t k = 0; k < cnt; ++k) {
            SupertileChunk ch;
            ch.chunk_id = (entry_index << 10) | seq;
            const size_t beg = static_cast<size_t>(k) * per;
            const size_t end = std::min(vec.size(), beg + per);
            uint64_t cbytes = 0;
            for (size_t i = beg; i < end; ++i) {
                ch.member_ops.push_back(vec[i]->op_id);
                cbytes += vec[i]->tensor_bytes;
            }
            ch.tensor_bytes = cbytes;
            if (!ch.member_ops.empty()) {
                r.chunks.push_back(std::move(ch));
                seq = supertile_next_chunk_seq(seq);   // 下一 chunk 序号
            }
        }
        if (stats) stats->chunks_out += r.chunks.size();
        out.push_back(std::move(r));
    }
    if (stats) stats->groups_kept = out.size();
    return out;
}

} // namespace hnnx
