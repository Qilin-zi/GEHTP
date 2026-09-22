// test_g4_sfcd_roundtrip: P4 SFCD spill/fill 写侧 ↔ 读侧 round-trip 自洽门
//
// 覆盖矩阵（GEHTP_BACKPORT_PLAN §2 / GEHTP_MEMPLAN_GAP_CLOSURE §P4）:
//   §2.1 tcm 块记录两词头（占位回填 (pool<<16)|nblocks + blob 偏移；终结器 {0, blob_off}）
//   §2.2 wait 记录窄/宽式判定（窄: val≤0xff && mb≤0xFFFFFF；宽: 对形态 low16=n<<1）
//   §2.3 find_peak_tcm_usage 残差语义（blocks = bytes>>11；net = running − arr[2n−1]）
//   §2.4 常量池 ID（G4_SPILLFILL_POOL_ID=2 等）
//   §2.6 读侧针脚（mb 十进制 / tcm 0x%X 大写 / :2095 错误文本）
//   §C/§E 分配面（三槽填法、池 2、shared/far 位、oversize 抛错、DLBC setup）
//   §D checkpoint op 双 vtable + insert_spill_fill 永久桩语义
//
// 主门: 写侧序列化 → g4_dump_sfcd 回读 → 头/记录/字段逐项一致，尾校验无 bad_length。
#include "hnnx/vtcm/spillfill_g4.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hnnx;

static int failed = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); failed++; } \
    else { std::printf("  OK: %s\n", msg); } \
} while(0)

static bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------- §2.4/§C/§E
static void test_alloc_surface() {
    std::printf("== §2.4/§C/§E 分配面 ==\n");
    CHECK(G4_SPILLFILL_POOL_ID == 2, "G4_SPILLFILL_POOL_ID == 2");
    CHECK(G4_COPYLESS_WEIGHTS_POOL == 0xf, "G4_COPYLESS_WEIGHTS_POOL == 0xf");
    CHECK(G4_POOL_SHARED == 0x1 && G4_POOL_FAR == 0x10, "shared/far 位 = 0x1/0x10");

    // 多域三槽：逐槽 64K 取整
    {
        G4FancyAllocator fa;
        const uint32_t peaks[3] = {0x18000, 0x8000, 0};
        uint64_t total = g4_dlbc_spill_fill_setup(fa, peaks, /*is_multi_nsp=*/true);
        // slots = {0x20000, 0x10000, 0}; total = 0x20000+0x10000 + 2*0x10000
        CHECK(total == 0x50000, "dlbc multi: total = 0x50000");
        CHECK(fa.pools.size() == 1 && fa.pools[0].size == 0x50000,
              "dlbc multi: 池 2 描述符 size");
        CHECK(fa.spillfill_pool == 1, "dlbc multi: +0xa0 句柄 = 下标+1");
        CHECK(fa.slot_sizes[0] == 0x20000 && fa.slot_sizes[1] == 0x10000 &&
              fa.slot_sizes[2] == 0, "dlbc multi: 三槽回存 +0xa8/+0xac/+0xb0");
    }
    // 单域 argmax：p2 不严格大于 max(p0,p1) → p1>p0 → 槽 1
    {
        G4FancyAllocator fa;
        const uint32_t peaks[3] = {0x8000, 0x18000, 0x9000};
        uint64_t total = g4_dlbc_spill_fill_setup(fa, peaks, false);
        CHECK(total == 0x30000, "dlbc single: total = 0x20000+0x10000");
        CHECK(fa.slot_sizes[0] == 0 && fa.slot_sizes[1] == 0x20000 &&
              fa.slot_sizes[2] == 0, "dlbc single: 仅获胜槽 1");
    }
    // 单域 p2 严格最大 → 槽 2
    {
        uint32_t slots[3];
        const uint32_t peaks[3] = {0x8000, 0x9000, 0x18000};
        g4_fill_slots_single(peaks, slots);
        CHECK(slots[0] == 0 && slots[1] == 0 && slots[2] == 0x20000,
              "single argmax: p2 严格大 → 槽 2");
    }
    // 三峰值全 0 → 不分配
    {
        G4FancyAllocator fa;
        const uint32_t peaks[3] = {0, 0, 0};
        CHECK(g4_dlbc_spill_fill_setup(fa, peaks, true) == 0 &&
              fa.pools.empty(), "dlbc: 全零不分配");
    }
    // oversize → throw runtime_error("oversize mem pool")
    {
        G4FancyAllocator fa;
        uint32_t sizes[3] = {0xFFFF0000u, 0x10000, 0};
        bool threw = false;
        try { g4_set_spillfill_size(fa, sizes); }
        catch (const std::runtime_error& e) {
            threw = std::strcmp(e.what(), "oversize mem pool") == 0;
        }
        CHECK(threw, "set_spillfill_size: total > 0xFFFFFF00 抛 oversize");
    }
    // shared 尺寸：0 → 0；64K 取整 + SHARED 位；env 阈值 → FAR 位
    {
        G4FancyAllocator fa;
        CHECK(g4_set_spillfill_shared_size(fa, 0, true, 1) == 0,
              "shared size 0 → 返回 0");
        CHECK(g4_set_spillfill_shared_size(fa, 0x18000, false, 0) == 0x20000,
              "shared 64K 取整");
        CHECK(fa.shared_size_290 == 0x18000, "+0x290 存原值");
        CHECK((fa.pools.back().flags & G4_POOL_SHARED) != 0 &&
              (fa.pools.back().flags & G4_POOL_FAR) == 0, "SHARED 置位/FAR 未置");
        CHECK(g4_set_spillfill_shared_size(fa, 2u << 20, true, 1) == (2u << 20),
              "shared 2MB");
        CHECK((fa.pools.back().flags & G4_POOL_FAR) != 0,
              "sz >= env_mb<<20 → FAR 置位");
        CHECK(g4_can_mempool_be_far(fa.pools.back(), true, 1), "can_mempool_be_far");
        CHECK(g4_is_shared_spillfill(2, 0, true), "is_shared_spillfill: pool2 未置位 env");
        CHECK(!g4_is_shared_spillfill(2, G4_POOL_SHARED, true),
              "is_shared_spillfill: 已置 SHARED → false");
        CHECK(!g4_is_shared_spillfill(3, 0, true), "is_shared_spillfill: pool!=2 → false");
    }
    // 薄适配器
    {
        G4FancyAllocator g4;
        G4PoolDesc pd[2]; pd[0].size = 0x1000; pd[1].size = 0x2000; pd[1].flags = 1;
        const uint32_t slots[3] = {0x10000, 0, 0x20000};
        g4_fancy_fill(g4, pd, 2, 2, slots, 0x9000);
        CHECK(g4.pools.size() == 2 && g4.pools[1].flags == 1 &&
              g4.spillfill_pool == 2 && g4.slot_sizes[2] == 0x20000 &&
              g4.shared_size_290 == 0x9000, "g4_fancy_fill 薄适配器逐项");
    }
}

// ------------------------------------------------------------------- §2.3 残差
static void test_peak_residual() {
    std::printf("== §2.3 find_peak_tcm_usage 残差 ==\n");
    const uint32_t arr[4] = {0x4000, 0x2000, 0x1000, 0x800};   // 2n = 4
    G4PeakTcmUsage u = g4_find_peak_tcm_usage(arr, 4);
    // running = 0x7800（递增剖面 peak = 末值）→ 0x7800>>11 = 0xF
    CHECK(u.peak_blocks == 0xF, "peak blocks = bytes>>11");
    // :78 net = running − arr[2n−1] = 0x7800 − 0x800 = 0x7000 → 0xE
    CHECK(u.net_blocks == 0xE, ":78 残差 = running − arr[2n−1]");
    G4PeakTcmUsage z = g4_find_peak_tcm_usage(nullptr, 0);
    CHECK(z.peak_blocks == 0 && z.net_blocks == 0, "空剖面 → 0");
}

// ------------------------------------------------- §2.1/§2.2 写→读 round-trip
static void test_roundtrip_construct() {
    std::printf("== §E construct_sfcd → g4_dump_sfcd round-trip ==\n");
    G4SlcArea area;
    area.area_id = 0x1234;
    area.is_fill = false;
    area.dma_checkpoint = 7;

    G4SlcRecord r0;                               // tcm 块记录（含可合并子项）
    r0.rec_type = G4_REC_SPILLFILL;
    r0.ddr_pool = G4_SPILLFILL_POOL_ID;
    r0.sf_offset = 0x400;
    r0.copies = {{0x1000, 0x800, 5},              // 与下条 tcm 连续 + hints 同 → 合并
                 {0x1800, 0x400, 5},
                 {0xABC0, 0x1000, 7}};            // hints 异 → 不合并
    area.records.push_back(r0);

    G4SlcRecord r1;                               // 窄式 wait（val ≤ 0xff）
    r1.rec_type = G4_REC_WAITFOR;
    r1.pairs = {{3, 7}, {0xABC, 0xFF}};
    area.records.push_back(r1);

    G4SlcRecord r2;                               // 宽式 wait（val > 0xff）
    r2.rec_type = G4_REC_WAITFOR;
    r2.pairs = {{5, 0x100}};
    area.records.push_back(r2);

    G4SlcRecord r3;                               // set_progress
    r3.rec_type = G4_REC_SETPROGRESS;
    r3.mb_idx = 9;
    r3.value_to_set = 0x1234;
    area.records.push_back(r3);

    const std::vector<uint32_t> words = g4_construct_sfcd(area);

    // ---- 写侧词级逐项（§2.1/§2.2 打包钉死）----
    const uint32_t expect[] = {
        64, 7, 4,                                 // 头: total=17*4-4, ckpt, 4 记录
        (2u << 16) | 2, 0x400,                    // §2.1 两词头（合并后 nblocks=2）
        0x1000, 0xC00,                            //   子项 {w=tcm|type0, len}（已合并）
        0xABC0, 0x1000,
        0x80000002, (7u << 24) | 3, (0xFFu << 24) | 0xABC,  // 窄式 wait
        0x80010002, 5, 0x100,                     // 宽式 wait（bit16, low16=n<<1）
        0x81000009, 0x1234,                       // set_progress
    };
    CHECK(words.size() == sizeof(expect) / sizeof(expect[0]), "写侧词数 = 17");
    CHECK(std::memcmp(words.data(), expect, sizeof expect) == 0,
          "写侧词缓冲逐项 == 期望（§2.1/§2.2/头回填）");

    // ---- 读侧回读（spill 向）----
    G4SfcdWalk walk;
    const std::string dump = g4_dump_sfcd(words.data(), /*is_fill=*/false,
                                          /*verbose=*/false, &walk);
    CHECK(walk.hdr.total_bytes == 64 && walk.hdr.checkpoint_index == 7 &&
          walk.hdr.record_count == 4, "回读头: total/ckpt/记录数");
    CHECK(!walk.zero_records && !walk.bad_length && walk.errors.empty(),
          "回读: 无 zero/bad_length/:2095");
    CHECK(walk.cursor_bytes == 4 * words.size(), "回读游标 == 缓冲字节数");
    CHECK(walk.blocks_total == 0xC00 + 0x1000, "回读 blocks_total = Σ 子项 len");

    // 字段逐项（文本针脚 §2.6）
    CHECK(contains(dump, "---- SFCD for spill @"), "头行 spill 标签");
    CHECK(contains(dump, "    checkpoint_index = 7\n"), "checkpoint_index 行");
    CHECK(contains(dump, "     2 spill to mempool 2, offset 0x400\n"),
          "tcm 块记录行 (pool<<16)|nblocks 回填正确");
    CHECK(contains(dump, "0x400 <- tcm 0x1000  3072 bytes\n"), "子项 0 行");
    CHECK(contains(dump, "0x1000 <- tcm 0xABC0  4096 bytes\n"),
          "子项 1 行（游标 64B 步进 + tcm %%X 大写）");
    CHECK(contains(dump, " wait for mb[3] >= 7\n"), "窄式 wait 对 0");
    CHECK(contains(dump, " wait for mb[2748] >= 255\n"), "窄式 wait 对 1（mb 十进制）");
    CHECK(contains(dump, " wait for mb[5] >= 256\n"), "宽式 wait 对");
    CHECK(contains(dump, "     set_progress: mb[9] := 4660\n"), "set_progress 行");
    CHECK(contains(dump, "    ---> total 0x1C00 (7168) bytes\n"), "汇总行");
    CHECK(!contains(dump, "Bad length"), "尾校验无 Bad length");

    // ---- 读侧回读（fill 向：is_fill 由调用方参数决定）----
    const std::string fdump = g4_dump_sfcd(words.data(), true, false, nullptr);
    CHECK(contains(fdump, "---- SFCD for fill @") &&
          contains(fdump, "2 fill from mempool 2") && contains(fdump, "-> tcm"),
          "fill 向 fmt/箭头翻转");

    // ---- verbose 元组行 ----
    const std::string vdump = g4_dump_sfcd(words.data(), false, true, nullptr);
    CHECK(contains(vdump, "        (2, 0x400, 0x1000, 0xc00, 0x0),  #"),
          "verbose tcm 元组行（type nibble=0）");
    CHECK(contains(vdump, "        ('wait_for_progress', 3, 7),"), "verbose wait 元组");
    CHECK(contains(vdump, "        ('set_progress', 9, 4660),"), "verbose set 元组");
}

// ------------------------------------------- §2.1 终结器/空记录 + §2.2 分块
static void test_roundtrip_writer_direct() {
    std::printf("== §E writer 直写: 终结器/空块记录/wait 分块 ==\n");
    {
        G4SfcdWriter w;
        const size_t rec = w.begin_block_record(0x800);   // nblocks=0 合法空记录
        w.end_block_record(rec, 4);
        w.push_terminator(0x800);                          // {0, blob_off} @0x1012460
        w.finish_header();
        CHECK(w.words.size() == 3 + 2 + 2 && w.words[2] == 2, "空记录+终结器词数/记录数");
        CHECK(w.words[3] == (4u << 16) && w.words[5] == 0 && w.words[6] == 0x800,
              "终结器 = push {0, blob_off}");
        G4SfcdWalk walk;
        const std::string dump = g4_dump_sfcd(w.words.data(), false, false, &walk);
        CHECK(walk.hdr.record_count == 2 && !walk.bad_length,
              "回读: 终结器计一条记录且无 bad_length");
        CHECK(contains(dump, "0 spill to mempool 4, offset 0x800") &&
              contains(dump, "0 spill to mempool 0, offset 0x800"),
              "空块记录/终结器回读行（nblocks==0 合法）");
    }
    {
        // 窄式低 16 位容量分块: n = 0x10000 → 0xFFFF + 1 两条记录
        std::vector<G4SfcdWaitEntry> pairs(0x10000, {1, 2});
        G4SfcdWriter w;
        w.write_waits(pairs.data(), pairs.size());
        w.finish_header();
        CHECK(w.records == 2, "窄式分块: 0x10000 条 → 2 记录");
        CHECK(w.words[3] == 0x8000FFFFu && w.words[3 + 1 + 0xFFFF] == 0x80000001u,
              "窄式分块块头");
        G4SfcdWalk walk;
        g4_dump_sfcd(w.words.data(), false, false, &walk);
        CHECK(walk.hdr.record_count == 2 && !walk.bad_length && walk.errors.empty(),
              "窄式分块回读自洽");
    }
    {
        // 宽式分块: n = 0x8000（val 超 0xff）→ 0x7FFF + 1 两条记录
        std::vector<G4SfcdWaitEntry> pairs(0x8000, {1, 0x200});
        G4SfcdWriter w;
        w.write_waits(pairs.data(), pairs.size());
        w.finish_header();
        CHECK(w.records == 2, "宽式分块: 0x8000 对 → 2 记录");
        CHECK(w.words[3] == (0x80010000u | (0x7FFFu << 1)), "宽式首块 low16 = n<<1");
        G4SfcdWalk walk;
        g4_dump_sfcd(w.words.data(), false, false, &walk);
        CHECK(walk.hdr.record_count == 2 && !walk.bad_length, "宽式分块回读自洽");
    }
    {
        // 窄↔宽边界: mb > 0xFFFFFF 亦强制宽式
        G4SfcdWaitEntry p[1] = {{0x1000000, 1}};
        G4SfcdWriter w;
        w.write_waits(p, 1);
        CHECK((w.words[3] & 0x10000u) != 0, "mb > 0xFFFFFF → 宽式");
    }
    {
        // 写侧契约违例
        G4SfcdWriter w;
        bool threw = false;
        try { w.begin_block_record(0); w.push_block_entry(0x1001, 0, 0x40); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "tcm_offset 未 16B 对齐 → invalid_argument");
        threw = false;
        try { w.end_block_record(0, 0x8000); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "pool > 0x7FFF（w0 须 < 0x80000000）→ invalid_argument");
    }
}

// ------------------------------------------------------ 负例: :2095 / 零记录
static void test_negative() {
    std::printf("== 负例: :2095 非法头 / 零记录 ==\n");
    {
        const uint32_t buf[] = {8, 0, 1, 0x82000000, 0};
        G4SfcdWalk walk;
        const std::string dump = g4_dump_sfcd(buf, false, false, &walk);
        CHECK(walk.errors.size() == 1 &&
              contains(walk.errors[0],
                       "grdep_spillfill.cc:2095::ERROR:Bad SFCD record header 82000000"),
              ":2095 错误入 walk");
        CHECK(contains(dump, "!!!!  Bad SFCD record header 82000000"),
              ":2095 文件侧双写");
        CHECK(!walk.bad_length, ":2095 立即返回，无尾校验");
    }
    {
        const uint32_t buf[] = {8, 0, 0};
        G4SfcdWalk walk;
        const std::string dump = g4_dump_sfcd(buf, false, false, &walk);
        CHECK(walk.zero_records && contains(dump, "?? zero records??"),
              "零记录早退");
    }
    {
        // mgroup 校验负例
        G4SlcArea area;
        G4SlcRecord bad; bad.rec_type = 3;
        area.records.push_back(bad);
        std::string err;
        CHECK(!g4_fill_mgroup_check(area, err) && contains(err, "bad rec_type"),
              "mgroup 校验: 非法 rec_type");
        area.records.clear();
        G4SlcRecord r; r.rec_type = G4_REC_SPILLFILL; r.ddr_pool = 2;
        r.copies = {{0x1001, 0x40, 0}};
        area.records.push_back(r);
        CHECK(!g4_fill_mgroup_check(area, err), "mgroup 校验: tcm 未对齐");
        bool threw = false;
        try { g4_construct_sfcd(area); }
        catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw, "construct_sfcd 拒发射非法区域");
    }
}

// --------------------------------------------------------------------- §D / §B
static void test_checkpoint_and_slc() {
    std::printf("== §D checkpoint op / 桩 + §B slc 一致性 ==\n");
    G4CheckpointOp set_op = g4_make_dma_checkpoint_op(0, 42, 9, true);
    G4CheckpointOp wait_op = g4_make_dma_checkpoint_op(0, 43, 9, false);
    CHECK(set_op.vtable == 0x5ec2488ull && wait_op.vtable == 0x5ec2568ull,
          "双 vtable 0x5ec2488(set)/0x5ec2568(wait)");
    CHECK(set_op.mb_idx == 9 && set_op.value == 0 && !set_op.owned &&
          set_op.opid == 42, "checkpoint op 字段");
    std::string log;
    CHECK(g4_insert_spill_fill(log) == -1 &&
          contains(log, "insert_spill_fill not supported"),
          "insert_spill_fill 永久桩语义（log 后 return -1）");

    // 同一区域同时喂 §B 序列化与 §E 写侧，两路径字段同源
    G4SlcArea area;
    area.area_id = 0x1234;
    area.dma_checkpoint = 7;
    G4SlcRecord r; r.rec_type = G4_REC_SETPROGRESS; r.mb_idx = 1; r.value_to_set = 2;
    area.records.push_back(r);
    const auto fields = g4_serialize_slc_area(area);
    bool saw_name = false, saw_ckpt = false;
    for (const auto& f : fields) {
        if (f.is_string && f.sval == "SLC_spillfill_area_0x1234") saw_name = true;
        if (f.key == "dma_checkpoint" && f.value_tag == (int)G4_TAG_I32_DMA_CHECKPOINT &&
            f.uval == 7) saw_ckpt = true;
    }
    CHECK(saw_name && saw_ckpt, "§B 序列化: 区域名/dma_checkpoint(tag 0xe) 同源");
}

int main() {
    test_alloc_surface();
    test_peak_residual();
    test_roundtrip_construct();
    test_roundtrip_writer_direct();
    test_negative();
    test_checkpoint_and_slc();
    if (failed) { std::fprintf(stderr, "FAILED: %d checks\n", failed); return 1; }
    std::printf("ALL PASS\n");
    return 0;
}
