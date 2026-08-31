// M37 冒烟测试: phys_alloc_in_runlist 反汇编保真实现
// 证据基线: audit_verify/reports/M37_phys_alloc_in_runlist_disasm.md
#include "hnnx/vtcm/phys_alloc_m37.hpp"
#include <cstdio>

using namespace hnnx;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

static bool seq_eq(const std::vector<op_id_t>& got, std::initializer_list<op_id_t> want) {
    return got == std::vector<op_id_t>(want);
}

static void test_empty_and_gates() {
    // count==0 → return 0 [0xf72b0b → 0xf72e1b]
    PhysAllocRunlistResult r = phys_alloc_in_runlist_disasm({}, {}, {});
    CHECK(r.rc == 0 && r.called_ids.empty(), "empty runlist returns 0 [0xf72b0b]");

    // 门 1 (this+0x6208==0) 关 → 成员表被无视 [0xf72b23]
    PhysAllocRunlistConfig cfg;
    cfg.supertile_mode = false;
    cfg.supertile_entries = {{7, {7, 8, 9}, {false, false, false}}};
    std::unordered_map<op_id_t, PhysAllocOp> ops;
    r = phys_alloc_in_runlist_disasm({{7, 0}}, cfg, ops);
    CHECK(r.rc == 0 && seq_eq(r.called_ids, {7}),
          "gate this+0x6208==0 skips fan-out [0xf72b23]");

    // 门 2 (桶数==0 ⇔ 空表) 关 → 同样直走本体 [0xf72b47]
    cfg.supertile_mode = true;
    cfg.supertile_entries.clear();
    r = phys_alloc_in_runlist_disasm({{7, 0}}, cfg, ops);
    CHECK(r.rc == 0 && seq_eq(r.called_ids, {7}),
          "empty table (bucket_count==0) skips fan-out [0xf72b47]");

    // id 未命中 → 不扇出 [0xf72ca5 未达]
    cfg.supertile_entries = {{100, {100, 101}, {false, false}}};
    r = phys_alloc_in_runlist_disasm({{7, 0}}, cfg, ops);
    CHECK(r.rc == 0 && seq_eq(r.called_ids, {7}), "id not in table -> no fan-out");
}

static void test_fanout_order() {
    // 命中: 成员先（表序、跳 null），本体后 [0xf72d97-0xf72dcc → 0xf72ce0]
    PhysAllocRunlistConfig cfg;
    cfg.supertile_mode = true;
    cfg.supertile_entries = {{7, {7, 8, 9}, {false, true, false}}};  // 8 号成员为 null
    std::unordered_map<op_id_t, PhysAllocOp> ops;
    auto r = phys_alloc_in_runlist_disasm({{7, 0}, {9, 0}}, cfg, ops);
    CHECK(r.rc == 0 && seq_eq(r.called_ids, {7, 9, 7, 9}),
          "members first (null skipped [0xf72dc1]), then self [0xf72ce0]");
    // ↑ 条目7: 成员 7,9 → 本体 7；表无条目9 → op 9 只调本体

    // 空成员 vector → 只调本体
    cfg.supertile_entries = {{7, {}, {}}};
    r = phys_alloc_in_runlist_disasm({{7, 0}}, cfg, ops);
    CHECK(r.rc == 0 && seq_eq(r.called_ids, {7}), "empty member vector -> self only");
}

static void test_error_short_circuit() {
    PhysAllocRunlistConfig cfg;      // 两门全关：纯本体路径
    std::unordered_map<op_id_t, PhysAllocOp> ops;
    // 本体返回非 0 → 日志 + 短路 [0xf72cec-0xf72e4d]
    auto r = phys_alloc_in_runlist_disasm({{1, 0}, {2, 5}, {3, 0}}, cfg, ops);
    CHECK(r.rc == 5 && seq_eq(r.called_ids, {1, 2}),
          "nonzero alloc_rc stops the walk after the failing op");
    CHECK(r.error_log.find("graph_prepare.cc:2173") != std::string::npos &&
          r.error_log.find("op 2") != std::string::npos,
          "error log fmt@0x461d95b, file@0x461dff6 [0xf72e48]");

    // 成员的失败被静默丢弃（成员无 rc 字段可携带；本体继续）[0xf72dcf 无 test]
    cfg.supertile_mode = true;
    cfg.supertile_entries = {{1, {9}, {false}}};
    PhysAllocOp failing_member{9, 77};
    ops.emplace(9, failing_member);  // 记录级：成员调用只登记 id，不检查返回值
    r = phys_alloc_in_runlist_disasm({{1, 0}}, cfg, ops);
    CHECK(r.rc == 0 && seq_eq(r.called_ids, {9, 1}),
          "member return values discarded [0xf72dcf]");

    // 全成功 → rc 0（条目1 扇出成员 9，故 4 次调用）
    r = phys_alloc_in_runlist_disasm({{1, 0}, {2, 0}, {3, 0}}, cfg, ops);
    CHECK(r.rc == 0 && seq_eq(r.called_ids, {9, 1, 2, 3}) && r.error_log.empty(),
          "all-success returns 0 [0xf72e1b]");
}

static void test_log_text() {
    // fmt 逐字节: '%s:2173::ERROR:could not allocate memory for op %llx!!\n\n'
    const std::string s = m37_alloc_error_log(0x1234);
    CHECK(s == "graph_prepare.cc:2173::ERROR:could not allocate memory for op 1234!!\n\n",
          "log text matches fmt@0x461d95b byte-for-byte");
}

int main() {
    test_empty_and_gates();
    test_fanout_order();
    test_error_short_circuit();
    test_log_text();
    if (g_fail) { std::printf("\n%d FAILURE(S)\n", g_fail); return 1; }
    std::printf("\nALL M37 SMOKE TESTS PASSED\n");
    return 0;
}
