// M36 冒烟测试: create_supertiles 反汇编保真实现
// 证据基线: audit_verify/reports/M36_create_supertiles_disasm.md
#include "hnnx/vtcm/create_supertiles_m36.hpp"
#include <cstdio>
#include <cassert>
#include <stdexcept>

using namespace hnnx;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

static void test_opid_table() {
    OpIdTable t;
    // 表项 1 覆盖 id 0x400..0x7ff（id>>10==1）；普通 tag 带 bit30+payload
    t.set_plain(0x400, 0x40000000u | 123u);      // entry[1]（id>>10==1 命中）
    bool err = false;
    CHECK(t.resolve_table_index(0x400, &err) == 1 && !err, "resolve_table_index plain -> 1");
    CHECK(t.resolve_full_id(0x405, &err) == 5 && !err, "resolve_full_id keeps low 10 bits (0x405 -> 5)");
    CHECK(t.extract_field(0x405, &err) == 123 && !err, "extract_field payload 123 (tag & 0x3FFFFF)");
    CHECK(t.resolve_table_index(0, &err) == 0 && !err, "id==0 short-circuits to 0 [0x12face1]");
    CHECK(t.resolve_full_id(0, &err) == 0, "resolve_full_id(0) == 0");

    // 远跳: entries[1] (覆盖 id 0x800..0xbff) 存 0x78000000|3；
    // idx(2) += delta(3) → 5 → 读 entries[4]，其覆盖 id 空间 0x1400..0x17ff
    t.set_far(0x800, 3);
    t.set_plain(0x1400, 0x40000000u | 777u);      // entries[4]
    CHECK(t.resolve_table_index(0x800, &err) == 5 && !err, "far jump idx 2 -> 5 (idx += delta)");
    // resolve_full_id 远跳: 结果 = (delta<<10)|(id&0x3ff) [0x12fadb5-0x12fadb9]
    CHECK(t.resolve_full_id(0x807, &err) == (3ull << 10 | 7) && !err,
          "resolve_full_id far: (delta<<10)|seq");

    // 负 tag: bit30+bit31 → 越界保护后返回当前 idx
    t.set_negative(0x1000);                       // entries[3] (id 0x1000>>10 = 4)
    CHECK(t.resolve_table_index(0x1000, &err) == 4 && !err, "negative tag returns idx");
    // extract_field 负 tag → 读下一表项原值 [0x12faf90] → entries[4] = 0x40000000|777
    CHECK(t.extract_field(0x1000, &err) == (0x40000000u | 777u) && !err,
          "extract_field negative tag reads next entry raw");
    // bit30 清 → 恒 0 [0x12faf78]
    t.set_plain(0x1800, 0x00000123u);             // entries[5], 无 bit30
    CHECK(t.extract_field(0x1800, &err) == 0, "extract_field without bit30 -> 0");

    // 越界 → error 置位
    t.resolve_table_index(0xf00000, &err);
    CHECK(err, "out-of-range id sets error [0x12facff]");
    CHECK(OpIdTable::compare_ids(7, 7) == 0 && OpIdTable::compare_ids(7, 8) == 1,
          "compare_ids a==b->0 else 1 [0x12fadf0 前 12B]");
}

static void test_autothread() {
    // tbl@0x39b7650 = {1,8,8,32} [0x10c3690]
    CHECK(autothread_size(1, 100, 1) == 100, "T<2 -> S untouched");
    // T=4,S=100,idx=1: q=25, M=8, round_up(25,8)=32, min(32,100)=32
    CHECK(autothread_size(4, 100, 1) == 32, "T=4 S=100 idx=1 -> 32");
    // T=8,S=17,idx=3: q=3(ceil 17/8=3), M=32, round_up(3,32)=32, min(32,17)=17
    CHECK(autothread_size(8, 17, 3) == 17, "round_up clamped by min(.,S)");
    // T=2,S=5,idx=0: q=3, M=1, ru=3, min(3,5)=3
    CHECK(autothread_size(2, 5, 0) == 3, "M=1 row: plain ceil");
}

static void test_stub_and_descs() {
    std::string log;
    CHECK(insert_spill_fill_stub(&log) == -1, "insert_spill_fill stub returns -1 [0x106d7d0]");
    CHECK(log.find("not supported") != std::string::npos, "stub logs fmt@0x462adf1");
    auto d1 = make_dma_checkpoint_op(1, 0x42, true);
    auto d0 = make_dma_checkpoint_op(2, 0x43, false);
    CHECK(static_cast<uint32_t>(d1.vtable_class) == 0x5ec2488u &&
          static_cast<uint32_t>(d0.vtable_class) == 0x5ec2568u,
          "b!=0 -> Wait@0x5ec2488, b==0 -> Set@0x5ec2568 [M36 修正]");
    CHECK(make_sync_op_desc(9).new_id == 9 && SyncOpDesc::VTABLE == 0x5ec31f8u,
          "SyncOp vptr 0x5ec31f8 [0xdac440]");
}

static void test_dep_helpers() {
    // fc5910: 0x200000 + 无 0x40 → 元素数
    SupertileDepRecord r1{0x200000, 0x00, 1};
    std::vector<uint32_t> three{1, 1, 1};
    CHECK(dep_effective_num_slices(r1, three, 9) == 3, "supertiled no-0x40 -> count [0xfc5975]");
    // 0x40 → 累加
    SupertileDepRecord r2{0x200000, 0x40, 1};
    std::vector<uint32_t> counts{2, 3, 5};
    CHECK(dep_effective_num_slices(r2, counts, 9) == 10, "supertiled 0x40 -> sum [0xfc5990]");
    // 少于 2 元素 → throw num_internal_threads [0xfc59a9]
    bool threw = false;
    try { dep_effective_num_slices(r1, {7}, 9); }
    catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "num_internal_threads";
    }
    CHECK(threw, "short vector throws num_internal_threads [0xfc59a9]");
    // 0x400000 → self_slicing_count；无标志 → 1
    SupertileDepRecord r3{0x400000, 0, 1}, r4{0x0, 0, 1};
    CHECK(dep_effective_num_slices(r3, {}, 6) == 6, "self-slicing tail-call value [0xfc5945]");
    CHECK(dep_effective_num_slices(r4, {}, 6) == 1, "no flags -> 1 [0xfc5928]");

    // fc5af0: count<=1 早退 [0xfc5b1a]
    std::vector<GrdepOpRecord> recs(4);
    std::vector<std::vector<uint64_t>> store;
    CHECK(register_supertiled_dep(recs, 1, 1, 0x2000, store) == SupertileDepRegStatus::EarlyExit,
          "count<=1 early exit");
    // resource_flags & 0xc 必须 ∈ {4,8} [0xfc5b42-0xfc5b5a]
    recs[0].resource_flags = 4;   // ok
    recs[1].resource_flags = 12;  // 非幂 → 拒
    recs[2].resource_flags = 8;   // ok
    CHECK(register_supertiled_dep(recs, 2, 3, 0x2000, store) == SupertileDepRegStatus::BadResourceFlags,
          "flags&0xc==12 rejected (ERROR 3457 path)");
    CHECK(register_supertiled_dep(recs, 1, 3, 0x2000, store) == SupertileDepRegStatus::Ok,
          "flags&0xc==4 accepted");
    CHECK(store[0].size() == 3 && store[0][0] == 0x2000 && store[0][2] == 0x2002,
          "chunk id array count entries (content 遗留#5, sequential fill)");
    CHECK(register_supertiled_dep(recs, 3, 2, 0x3000, store) == SupertileDepRegStatus::Ok,
          "flags&0xc==8 accepted");
}

static void test_create_supertiles() {
    // 表: entry1 覆盖 id 0x400-0x7ff，payload 11
    OpIdTable t;
    t.set_plain(0x400, 0x40000000u | 11u);
    // 同组 4 op（同 resolved 0x400 空间、同 tag），张量字节合计 2000，预算 1000
    CreateSupertilesConfig cfg;
    cfg.budget_default = 1000;
    cfg.budget_hash_hit = 5000;
    cfg.hmx_count = 4;              // gp+0x5fe8 计数 > 1 → HMX 路径开
    // 分组键 = 完整 resolved id（低 10 位保留）→ 同一 id 的重复引用才成组
    // （真 map 值是 run 位置 custom_vec<u32>；M36 Phase1）
    // resource_flags & 0x8 = HMX 资格位 [反编译谓词]
    std::vector<SupertileCandidate> ops = {
        {0x400, 7, 500, "q::Conv2d", 0x8, true},
        {0x400, 7, 500, "q::Conv2d", 0x8, true},
        {0x400, 7, 500, "q::Conv2d", 0x8, true},
        {0x400, 7, 500, "q::Conv2d", 0x8, true},
    };
    CreateSupertilesStats st;
    auto res = create_supertiles_disasm(t, ops, cfg, &st);
    CHECK(res.size() == 1 && res[0].budget == 1000, "one group, default budget [gp[0x5fd8]<<10]");
    CHECK(st.groups_in == 4 && st.groups_kept == 1, "4 ops in, 1 group kept");
    // total 2000 > 1000 → 除数搜索: vec 4 元素, product=8000, cnt 从 4 下探:
    // 8000%4==0 → cnt=4 chunks
    CHECK(res[0].chunks.size() == 4, "over-budget -> divisor 4 chunks");
    // chunk id: (entry_index<<10)|seq, seq 步进 0,1,2,3
    bool ids_ok = true;
    for (size_t i = 0; i < res[0].chunks.size(); ++i)
        if (res[0].chunks[i].chunk_id != ((1ull << 10) | i)) ids_ok = false;
    CHECK(ids_ok, "chunk ids = (entry<<10)|seq [0x12fa220]");
    uint64_t sum = 0;
    for (auto& c : res[0].chunks) sum += c.tensor_bytes;
    CHECK(sum == 2000, "chunk bytes sum to group total (vtable +0x60/+0xa0 accumulate)");

    // 预算内 → 单 chunk
    cfg.budget_default = 4000;
    res = create_supertiles_disasm(t, ops, cfg, &st);
    CHECK(res.size() == 1 && res[0].chunks.size() == 1 && res[0].chunks[0].member_ops.size() == 4,
          "within budget -> single chunk");

    // 哈希命中 → budget_hash_hit（键 = resolved>>10 = 1）
    cfg.budget_default = 1000;
    cfg.hash_hit_ids = {0ull};   // 哈希键 = rid>>10 = 0
    res = create_supertiles_disasm(t, ops, cfg, &st);
    CHECK(res.size() == 1 && res[0].budget == 5000 && res[0].chunks.size() == 1,
          "hash-hit budget gp[0x74c0] [gp+0x6e40/0x6e48]");
    cfg.hash_hit_ids.clear();

    // ---- M36b 修正后的 Phase1 门 ----
    // 记录门: rec[0]/rec[8]/rec[0x98] 任一为 0 → 剔除
    std::vector<SupertileCandidate> bad_rec = ops;
    bad_rec[1].record_valid = false;
    bad_rec[2].record_valid = false;
    bad_rec[3].record_valid = false;
    res = create_supertiles_disasm(t, bad_rec, cfg, &st);
    CHECK(res.empty(), "record_valid=false members dropped, singleton erased [0x1313b41 区]");

    // 名字黑名单: 默认 6 项 [0x710806-0x710865]，命中即剔除
    std::vector<SupertileCandidate> conc = {
        {0x400, 7, 500, "q::Concat", 0x8, true},
        {0x400, 7, 500, "q::Concat", 0x8, true},
    };
    res = create_supertiles_disasm(t, conc, cfg, &st);
    CHECK(res.empty(), "q::Concat blacklisted [0x1313b95-0x1313bc2]");
    std::vector<SupertileCandidate> sp = {
        {0x400, 7, 500, "q::Slice_contig.tcm", 0x8, true},
        {0x400, 7, 500, "q::*InputSlicePad", 0x8, true},
    };
    res = create_supertiles_disasm(t, sp, cfg, &st);
    CHECK(res.empty(), "slice/pad names blacklisted (6-name set)");
    cfg.blacklist_names = {""};   // 禁用黑名单 → 同名可入组
    res = create_supertiles_disasm(t, conc, cfg, &st);
    CHECK(res.size() == 1, "empty-name entry disables blacklist");
    cfg.blacklist_names = {k_supertile_name_blacklist, k_supertile_name_blacklist + 6};

    // 资格门: (hmx_count>1 && flags&8) || (hvx_count>1 && (flags&0x110004)==4)
    cfg.hmx_count = 1;            // 单 HMX 线程 → HMX 路径关
    res = create_supertiles_disasm(t, ops, cfg, &st);
    CHECK(res.empty(), "hmx_count<=1 blocks flags&0x8-only members [gp+0x5fe8]");
    cfg.hmx_count = 4;
    std::vector<SupertileCandidate> no_flag = {
        {0x400, 7, 500, "q::Conv2d", 0x4, true},   // HVX 位但 hvx_count==0
        {0x400, 7, 500, "q::Conv2d", 0x4, true},
    };
    res = create_supertiles_disasm(t, no_flag, cfg, &st);
    CHECK(res.empty(), "HVX-flag member blocked while hvx_count<=1 [gp+0x5fe0]");
    cfg.hvx_count = 4;            // HVX 路径开 → flags 0x4 通过 ((0x4&0x110004)==4)
    res = create_supertiles_disasm(t, no_flag, cfg, &st);
    CHECK(res.size() == 1, "hvx_count>1 && flags&0x110004==4 admitted");
    cfg.hvx_count = 0;
    // 注意: 资格门是字面位测试（flags & 0x8），非幂校验 —— 0xC 含 0x8 位照样
    // 过 HMX 路径（与 fc5af0 的 &0xc∈{4,8} 幂校验是两个不同函数）
    std::vector<SupertileCandidate> both_bits = {
        {0x400, 7, 500, "q::Conv2d", 0xC, true},
        {0x400, 7, 500, "q::Conv2d", 0xC, true},
    };
    res = create_supertiles_disasm(t, both_bits, cfg, &st);
    CHECK(res.size() == 1, "flags 0xC passes HMX path: plain bit-test, not power check");
    std::vector<SupertileCandidate> no_bit = {
        {0x400, 7, 500, "q::Conv2d", 0x10, true},  // bit4: HMX(0x8)/HVX(0x4) 均无
        {0x400, 7, 500, "q::Conv2d", 0x10, true},
    };
    res = create_supertiles_disasm(t, no_bit, cfg, &st);
    CHECK(res.empty(), "flags without 0x8/0x4 bits match neither predicate");

    // 单元素组 → erase（不输出）
    res = create_supertiles_disasm(t, {{0x400, 7, 100, "q::Conv2d", 0x8, true}}, cfg, &st);
    CHECK(res.empty() && st.groups_kept == 0, "singleton group erased [Phase2]");

    // 不同 tag → 不同组
    std::vector<SupertileCandidate> mixed = {
        {0x400, 7, 500, "q::Conv2d", 0x8, true}, {0x400, 7, 500, "q::Conv2d", 0x8, true},
        {0x400, 8, 500, "q::Conv2d", 0x8, true}, {0x400, 8, 500, "q::Conv2d", 0x8, true},
    };
    res = create_supertiles_disasm(t, mixed, cfg, &st);
    CHECK(res.size() == 2, "distinct tags -> distinct groups [pair<resolved,tag> key]");
}

// ---- decode_tag_triples [0x12fa880-0x12fab07] ----
static void test_decode_tag_triples() {
    // 表布局（entries 0 基）:
    //   [0] 根 tag（覆盖 id 0x400-0x7ff）
    //   [1] 根的链种子 = 表项原值；id 0xC00-0xFFF → 读 [2]
    //   [2] bit30 置的正 tag（type=3, field=22）→ 链续 payload22=0x1000 → 读 [3]
    //   [3] bit30/31 全清 → 终链
    OpIdTable t;
    t.grow_for(0x17ff);
    t.entries[0] = 0x19400123u;      // type=3, field=5, payload=0x123
    t.entries[1] = 0xC05u;           // 种子 id（>>10 = 3 → 读 entries[2]）
    t.entries[2] = 0x5D801000u;      // bit30 置、正: type=3, field=22, payload22=0x1000
    t.entries[3] = 0x00000045u;      // 终链（bit30 清、正）
    std::vector<SupertileTagTriple> out;
    bool err = false;
    size_t n = decode_tag_triples(t, 0x400, out, &err);
    // 根: type=3, value=(tag&0x7FFFFFF)+1=0x1400124, seq=0  [0x12fa89d/0x12fa8f2]
    CHECK(!err && n == 3 && out[0].type == 3 && out[0].value == 0x1400124u && out[0].seq == 0,
          "root triple: payload27+1, type=(tag>>27)&7");
    // 链 1 (id 0xC05): bit30 置正 → value=field+1=23, next=0x1000, seq=5
    CHECK(out[1].type == 3 && out[1].value == 23u && out[1].seq == 5u,
          "bit30-set positive link: value=(tag>>22&0x1f)+1 [0x12fa9ec]");
    // 链 2 (id 0x1000): bit30 清正 → 终链, value=1, seq=0
    CHECK(out[2].type == 0 && out[2].value == 1u && out[2].seq == 0u,
          "terminal link: bit30 clear -> next=0 [0x12fa9be]");

    // 负 tag（bit30 清、bit31 置）→ value = payload27+1 [0x12fa930]
    OpIdTable t2;
    t2.grow_for(0xfff);
    t2.entries[0] = 0x09400077u;    // 根: type=1
    t2.entries[1] = 0xC05u;         // 种子 id → 读 entries[2]
    t2.entries[2] = 0x8000002Du;    // bit31 置、bit30 清 → 负终链
    out.clear();
    n = decode_tag_triples(t2, 0x400, out, &err);
    CHECK(!err && n == 2 && out[1].value == 0x2Eu && out[1].seq == 5u,
          "negative terminal link: value=(tag&0x7FFFFFF)+1");

    // bit30+bit31 同置 → next = entries[idx] 原值 [0x12fa9fc-0x12faa05]
    //   根 id 0x1400 → entries[4]; 种子 entries[5]=0x1C09 → 读 entries[6]
    //   entries[6] bit30+31 → next = entries[7] 原值 = 0x2005（作 id 读回 entries[7] → 终链）
    OpIdTable t3;
    t3.grow_for(0x23ff);
    t3.entries[4] = 0x8000C30u;     // 根 tag（type=1）
    t3.entries[5] = 0x1C09u;        // 种子 id → 读 entries[6]
    t3.entries[6] = 0xDC00002Au;    // bit30+bit31, type=3, payload27=0x400002A
    t3.entries[7] = 0x2005u;        // 充当 next id；又作为 tag 读回 → 终链
    out.clear();
    n = decode_tag_triples(t3, 0x1400, out, &err);
    CHECK(!err && n == 3 && out[1].type == 3 && out[1].value == 0x400002Bu && out[1].seq == 9u,
          "bit30+bit31 link: value=payload27+1, next=entries[idx] raw");
    CHECK(out[2].seq == 5u && out[2].value == 1u, "raw-next id 0x2005 terminates chain");

    // 远跳（循环体 [0x12fa990-0x12fa9b1]）: seq |= delta<<10
    OpIdTable t4;
    t4.grow_for(0x2bff);
    t4.entries[0] = 0x00000101u;    // 根 tag（种子见表项 [1]）
    t4.entries[1] = 0x2003u;        // 种子 id: >>10=8 → 读 entries[7]
    t4.entries[7] = 0x78000002u;    // 远跳 delta=2 → idx 8+2=10 → 读 entries[9]
    t4.entries[9] = 0x00000007u;    // 终链
    out.clear();
    n = decode_tag_triples(t4, 0x400, out, &err);
    CHECK(!err && n == 2 && out[1].seq == (2u << 10 | 3u) && out[1].value == 1u,
          "far link composes seq |= delta<<10 [0x12fa9ae]");

    // 越界 → error（真代码走 0xcf27a0；本实现置 *error 并返回已有条数，已声明差异）
    OpIdTable t5;
    t5.grow_for(0x7ff);             // 1 项表: entries[1] 不存在 → 种子越界查
    t5.entries[0] = 0x00000001u;
    out.clear();
    decode_tag_triples(t5, 0x400, out, &err);
    CHECK(err && out.empty(), "seed load bounds check fires before root emit [0x12fa88d]");
    OpIdTable t6;
    t6.grow_for(0xbff);             // 2 项表（indices 0-1）
    t6.entries[0] = 0x00000001u;
    t6.entries[1] = 0x7ffffu;       // 种子 id → idx=0x200 越界（表仅 2 项）
    out.clear();
    decode_tag_triples(t6, 0x400, out, &err);
    CHECK(err && out.size() == 1, "out-of-range chain id sets error [0x12fa96e]");
}

static void test_chunk_seq_wrap() {
    // 0x3fe → +1 = 0x3ff（不跨界）→ 再 +1 = 0x400 → &0x3ff==0 → 跳到 0x401
    CHECK(supertile_next_chunk_seq(1) == 2, "plain advance");
    CHECK(supertile_next_chunk_seq(0x3fe) == 0x3ff, "0x3fe -> 0x3ff");
    CHECK(supertile_next_chunk_seq(0x3ff) == 0x401, "0x3ff skips 0x400 multiple");
    CHECK(supertile_find_divisor(8000, 4, 4) == 4, "divisor stays when (c*b)%cnt==0");
    CHECK(supertile_find_divisor(8000, 4, 3) == 2, "divisor walks down 3->2 (8000%3!=0, 8000%2==0)");
}

int main() {
    test_opid_table();
    test_autothread();
    test_stub_and_descs();
    test_dep_helpers();
    test_create_supertiles();
    test_decode_tag_triples();
    test_chunk_seq_wrap();
    if (g_fail) { std::printf("\n%d FAILURE(S)\n", g_fail); return 1; }
    std::printf("\nALL M36 SMOKE TESTS PASSED\n");
    return 0;
}
