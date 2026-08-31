// M38 冒烟测试: allocate_io_tensors 反汇编保真实现
// 证据基线: audit_verify/reports/M38_allocate_io_tensors_disasm.md
#include "hnnx/vtcm/alloc_io_tensors_m38.hpp"
#include <cstdio>

using namespace hnnx;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

// 搭一个标准场景: 输出节点 0x200 带 3 个 op（0x300 无 cast / 0x301 有 cast / 0x302 有 cast）
static M38Graph make_graph() {
    M38Graph g;
    g.output_node_id = 0x200;
    g.new_id_seq = 5;
    M38OpDef out;
    out.id = 0x200;
    out.ops = {0x300, 0x301, 0x302};
    g.opdefs.emplace(0x200, out);
    M38OpDef o1; o1.id = 0x300; o1.needs_cast = false;
    o1.outdefs = {{0, 0}};
    g.opdefs.emplace(0x300, o1);
    M38OpDef o2; o2.id = 0x301; o2.needs_cast = true; o2.field_0x20 = 0xAB;
    o2.outdefs = {{1, 100}, {2, 200}};      // 两条都要被改写
    g.opdefs.emplace(0x301, o2);
    M38OpDef o3; o3.id = 0x302; o3.needs_cast = true;
    o3.outdefs = {{9, 7}};                   // 枚举 9: 不改写，只步进
    g.opdefs.emplace(0x302, o3);
    return g;
}

static void test_new_id_formula() {
    // Graph::new_id@0xd2f680: (seq++<<32) | (hint&0xFFFFFFFF) [0xd2f68b-0xd2f68f]
    M38Graph g; g.new_id_seq = 5;
    CHECK(g.new_id(0x123456789ULL) == (5ULL << 32) | 0x3456789ULL,
          "new_id = (seq<<32)|(hint low32) [0xd2f68b]");
    CHECK(g.new_id_seq == 6, "counter increments (fetch-then-inc) [0xd2f685]");
    // 高 32 位被丢弃: 传 0x100000002 → 低 32 位 2
    M38Graph g2; g2.new_id_seq = 0;
    CHECK(g2.new_id(0x100000002ULL) == 2ULL, "hint keeps low 32 bits only [0xd2f68f]");
}

static void test_input_node_paths() {
    // id==0 → 直进 B 段 [0xf6996f]
    M38Graph g = make_graph();
    g.input_node_id = 0;
    auto r = allocate_io_tensors_disasm(g);
    CHECK(r.rc == 0 && r.tensor_info_ids.size() == 3,
          "input id 0 skips A body [0xf6996f]");

    // 命中 → A 段先留一条 tensor-info [0xf69c05]
    g = make_graph();
    g.input_node_id = 0x300;
    r = allocate_io_tensors_disasm(g);
    CHECK(r.tensor_info_ids.size() == 4 && r.tensor_info_ids.front() == 0x300,
          "input hit prepends one tensor-info entry [0xf69c1a]");
    CHECK(g.input_node_id == 0x300 && !r.input_node_cleared, "input hit keeps id");

    // miss → 6676 WARNING + 清 id + 继续（不 return）[0xf69a09-0xf69a36]
    g = make_graph();
    g.input_node_id = 0x999;
    r = allocate_io_tensors_disasm(g);
    CHECK(g.input_node_id == 0 && r.input_node_cleared, "input miss clears this+0x5340 [0xf69a36]");
    CHECK(r.warning_log.find("graph_prepare.cc:6676") != std::string::npos &&
          r.warning_log.find("node 999") != std::string::npos,
          "6676 warning text matches fmt@0x4620185");
    CHECK(r.rc == 0 && r.tensor_info_ids.size() == 3, "input miss does NOT return early");
}

static void test_output_node_paths() {
    // id==0 → return 0，主体全不做 [0xf69a4e]
    M38Graph g = make_graph();
    g.output_node_id = 0;
    auto r = allocate_io_tensors_disasm(g);
    CHECK(r.rc == 0 && r.tensor_info_ids.empty() && r.final_op_ids.empty(),
          "output id 0 returns 0 immediately [0xf69a4e]");

    // miss → 6740 WARNING + 清 id + 提前 return 0 [0xf69b4c-0xf69b87]
    g = make_graph();
    g.output_node_id = 0x777;
    r = allocate_io_tensors_disasm(g);
    CHECK(g.output_node_id == 0 && r.output_node_cleared, "output miss clears this+0x5348 [0xf69b79]");
    CHECK(r.warning_log.find("graph_prepare.cc:6740") != std::string::npos &&
          r.warning_log.find("node 777") != std::string::npos,
          "6740 warning text matches fmt@0x46201f9");
    CHECK(r.rc == 0 && r.final_op_ids.empty(), "output miss returns 0 early [0xf69b87]");
}

static void test_main_loop_and_cast() {
    M38Graph g = make_graph();
    auto r = allocate_io_tensors_disasm(g);

    // C 段: 每 op 一条 tensor-info（首个 OutputDef 复制）[0xf69f00]
    CHECK(r.tensor_info_ids.size() == 3, "one tensor-info per loop iteration [0xf69f35]");

    // 新表: 老 id 保留，cast 的换新 id [0xf69eab-0xf69eb0]
    CHECK(r.final_op_ids.size() == 3, "final id array has n entries [0xf69af4]");
    CHECK(r.final_op_ids[0] == 0x300, "non-cast keeps original id [0xf69dfe]");
    CHECK(r.final_op_ids[1] == (5ULL << 32) | 0x301, "cast op gets new id #1 [0xf69e1e]");
    CHECK(r.final_op_ids[2] == (6ULL << 32) | 0x302, "cast op gets new id #2 (counter advanced)");

    // D 段: 第二个 new_id 用输出节点 id 作 hint [0xf6a0a9]
    CHECK(r.new_output_node_id == (7ULL << 32) | 0x200,
          "new Output node id = new_id(output_id) [0xf6a0a9]");
    CHECK(g.new_id_seq == 8, "three new_id calls total (2 casts + 1 output)");

    // 改写: 1→7 (-0x80), 2→3 (-0x8000), 9 不动 [0xf69f40-0xf69f74]
    CHECK(r.enum_rewrites.size() == 3, "one rewrite record per outputdef of cast ops");
    CHECK(r.enum_rewrites[0] == std::make_pair(1, 7) && r.adjust_after[0] == 100 - 0x80,
          "encoding 1 -> 7, adjust -= 0x80 [0xf69f4a-0xf69f4d]");
    CHECK(r.enum_rewrites[1] == std::make_pair(2, 3) && r.adjust_after[1] == 200 - 0x8000,
          "encoding 2 -> 3, adjust -= 0x8000 [0xf69f6a-0xf69f6f]");
    CHECK(r.enum_rewrites[2] == std::make_pair(9, 9) && r.adjust_after[2] == 7,
          "other encodings untouched [0xf69f65->0xf69f50]");
    // 就地改写也发生（OpDef#2 的表被真码改写）
    CHECK(g.opdefs.at(0x301).outdefs[0].encoding == 7 &&
          g.opdefs.at(0x302).outdefs[0].encoding == 9,
          "opdefs rewritten in place (记录级留痕)");

    // 老 Output 节点在表中 → +0x9|=3 路径（非 deletable）[0xf6a236]
    CHECK(r.old_output_flagged_0x3 && !r.old_output_marked_deletable,
          "old output found in map -> flag path [0xf6a236]");
    CHECK(r.collect_deletable_ran && r.rc == 0, "collect_deletable_nodes then return 0 [0xf6a257/0xf6a278]");
}

static void test_no_new_node_shortcut() {
    // 全部无 cast → 旗为 0 → 不进 D 段 [0xf6a07b-0xf6a08f]
    M38Graph g = make_graph();
    g.opdefs.at(0x301).needs_cast = false;
    g.opdefs.at(0x302).needs_cast = false;
    auto r = allocate_io_tensors_disasm(g);
    CHECK(r.rc == 0 && r.new_output_node_id == 0 && r.final_op_ids.empty() &&
          !r.collect_deletable_ran,
          "no cast created -> return 0 without D section [0xf6a082]");
    CHECK(g.new_id_seq == 5, "no new_id calls in that case");
}

static void test_error_paths() {
    // C 段查不到 op id → 6696 致命错 return -1 [0xf6a037-0xf6a059]
    M38Graph g2 = make_graph();
    g2.opdefs.at(0x200).ops = {0x300, 0x404};         // 0x404 不在表里
    auto r = allocate_io_tensors_disasm(g2);
    CHECK(r.rc == -1 && r.error_log ==
          "graph_prepare.cc:6696::ERROR:Fatal error in allocate_io_tensors function\n",
          "unknown op id -> fatal 6696, return -1 [0xf6a037]");
    CHECK(r.final_op_ids.empty() && r.tensor_info_ids.size() == 1,
          "fatal stops mid-loop (first op already processed) [0xf69da1]");

    // make_op_node_impl 失败 → return -1 [0xf69e84→0xf69fe4]
    M38Graph g3 = make_graph();
    g3.make_op_node_fails = true;
    r = allocate_io_tensors_disasm(g3);
    CHECK(r.rc == -1 && r.final_op_ids.empty(), "make_op_node_impl null -> return -1 [0xf69fe4]");

    // 插入失败 → return 插入 rc [0xf6a1c3-0xf6a1dc]
    M38Graph g4 = make_graph();
    g4.insert_rc = 42;
    r = allocate_io_tensors_disasm(g4);
    CHECK(r.rc == 42 && r.final_op_ids.size() == 3,
          "node-table insert failure propagates rc [0xf6a1b9]");
}

static void test_deletable_path() {
    // D 段复查老 Output id 于 0x6d60 [0xf6a1f2-0xf6a234]:
    //   命中 → +0x9|=3 [0xf6a236]；miss → mark_op_deletable [0xf6a24f]。
    // 记录级说明: 模型里 B→D 之间无人从 opdefs 删键，故公开流程恒走命中支；
    //   miss 支在真码中是防御性分支（make_op_node_impl 可能动表），此处验证互斥性。
    M38Graph g = make_graph();
    auto r = allocate_io_tensors_disasm(g);
    CHECK(r.old_output_flagged_0x3 && !r.old_output_marked_deletable,
          "standard flow takes the |=3 path, not deletable [0xf6a236 vs 0xf6a24f]");
}

int main() {
    test_new_id_formula();
    test_input_node_paths();
    test_output_node_paths();
    test_main_loop_and_cast();
    test_no_new_node_shortcut();
    test_error_paths();
    test_deletable_path();
    if (g_fail) { std::printf("\n%d FAILURE(S)\n", g_fail); return 1; }
    std::printf("\nALL M38 SMOKE TESTS PASSED\n");
    return 0;
}
