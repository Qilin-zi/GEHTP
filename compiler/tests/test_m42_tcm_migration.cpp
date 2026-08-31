// M42 冒烟测试: tcm_migration 外壳（预算三分之四规则 / 3472-3459 判决 / 错误路径）
// 证据基线: audit_verify/reports/M42_tcm_migration_disasm.md
#include "hnnx/vtcm/tcm_migration_m42.hpp"
#include <cstdio>

using namespace hnnx;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

// ---- P1 预算规则 ------------------------------------------------------------
static void test_three_quarters() {
    // (size/4)*3 [0x1321bd4-0x1321bda]
    CHECK(m42_three_quarters(0x00100000) == 0x000c0000, "1MB -> 768KB (3/4) [0x1321bd7]");
    CHECK(m42_three_quarters(0x80000) == 0x60000, "512KB -> 384KB");
    // 先除后乘：非 4 倍数
    CHECK(m42_three_quarters(0x80003u) == 0x60000u, "0x80003: (0x80003/4)*3 = 0x60000 (div first)");
    CHECK(m42_three_quarters(0x80003u) != (0x80003u * 3u / 4u), "div-first differs from mul-first");
    CHECK(m42_three_quarters(7) == 3, "7 -> 1*3 = 3");
    // 配置块 +0x28/+0x2c [0x1321bcd/0x1321bda]
    M42BudgetConfig bc;
    CHECK(reinterpret_cast<uint8_t*>(&bc.full) == &bc.head[0x28],
          "config layout: full at +0x28");
}

// ---- popcount / 单比特 ------------------------------------------------------
static void test_popcount() {
    CHECK(m42_popcount(0) == 0, "popcount(0)=0 [0x13235ec]");
    CHECK(m42_popcount(0x8000ULL) == 1, "popcount single bit");
    CHECK(m42_popcount(0x3000ULL) == 2, "popcount two bits");
    CHECK(m42_popcount(~0ULL) == 64, "popcount all");
    CHECK(m42_at_most_one_bit(0x4000ULL), "at-most-one-bit true [0x13235f0]");
    CHECK(!m42_at_most_one_bit(0x5000ULL), "at-most-one-bit false -> ja multi-bit path");
}

// ---- crouton 码匹配 ---------------------------------------------------------
static void test_layout_match() {
    CHECK(m42_layout_code_match(0x101, 0x101), "both flagged, same low byte [0x1322c99]");
    // 0x101 低字节=0x01, 0x1ff 低字节=0xff → 仅比低字节 → 失配 [0x1322c9c]
    CHECK(!m42_layout_code_match(0x101, 0x1ff), "flagged pair compares low bytes only [0x1322c9c]");
    CHECK(m42_layout_code_match(0x0042, 0x0042), "both small, equal [0x1322cb4]");
    CHECK(!m42_layout_code_match(0x101, 0x0042), "mixed flag -> mismatch [0x1322ca0-0x1322cb2]");
    CHECK(!m42_layout_code_match(0x0042, 0x0101), "mixed flag reverse -> mismatch");
}

// ---- 标准钩子 ----------------------------------------------------------------
static uint64_t g_est;
static uint64_t g_tensor_sz[2];   // [0]=before, [1]=after；tensor_size 按调用序取
static int g_sz_call;
static int g_priority;

static uint64_t h_est(M42Ctx&, M42Op& op) { return op.estimate ? op.estimate : g_est; }
static uint64_t h_sz(M42Ctx&, M42TensorRec&, bool) {
    return g_sz_call < 2 ? g_tensor_sz[g_sz_call++] : g_tensor_sz[1];
}
static int h_prio() { return g_priority; }

static void reset_hooks(M42Hooks& hk) {
    g_est = 0; g_sz_call = 0; g_tensor_sz[0] = g_tensor_sz[1] = 0; g_priority = 0;
    hk.estimate_op = h_est;
    hk.tensor_size = h_sz;
    hk.log_priority = h_prio;
    hk.transform = nullptr;
}

// 构造 A 支可通过的 op（码匹配 + kind 0xa）
static M42Op make_op(const char* name, uint64_t demand) {
    M42Op op;
    op.kind = 0xa;
    op.name = name;
    op.opid = 0xabcdULL;
    M42TensorRec t;
    t.tensor_kind = 0xa;
    t.code1 = 0x0042;             // ctx.default_code1 = 0x0042（两小码相等）
    op.tensors.push_back(t);
    // demand 通过 before/after 差值表达: after - before = demand
    g_tensor_sz[0] = 0x100;
    g_tensor_sz[1] = 0x100 + demand;
    g_sz_call = 0;
    return op;
}

// ---- P6 判决: 3472 / 3459 ----------------------------------------------------
static void test_verdict() {
    M42Ctx ctx; M42BudgetConfig bc; M42Trace tr; M42Hooks hk;
    reset_hooks(hk);
    ctx.default_code1 = 0x0042;

    // 需求 > 全量 → 3472 + errcount++ [0x1322f8b-0x1322fbd]
    ctx.ops.push_back(make_op("Conv0", 0x10000));
    m42_tcm_migration(ctx, bc, hk, 0x8000 /*tile*/, false, tr);
    CHECK(tr.log.find("3472") != std::string::npos &&
          tr.log.find("Conv0") != std::string::npos &&
          tr.log.find("(0x8000)") != std::string::npos &&
          tr.log.find("0x10000") != std::string::npos,
          "demand>full logs 3472 with budget+demand args [0x1322f95]");
    CHECK(ctx.errcount == 1, "3472 increments errcount [0x1322fbd]");
    CHECK(tr.early_exit_mode == 0x11, "errcount -> 1829 early exit mode 0x11 [0x132315c]");
    CHECK(tr.log.find("1829") != std::string::npos, "1829 logged at boundary [0x1323120]");
    CHECK(tr.convert_stages.empty(), "early exit skips convert pipeline");
    // 计时点成对（before + 尾部）[0x1321a29/0x13231a5]
    CHECK(tr.time_points.size() == 2 &&
          tr.time_points[0] == "prepare_before_tcm_migration" &&
          tr.time_points[1] == "prepare_tcm_migration",
          "time point pair even on early exit");

    // 需求 <= 全量 且优先级>0 → 3459 不计数 [0x1322fc7-0x1322ff2]
    M42Ctx ctx2; M42BudgetConfig bc2; M42Trace tr2;
    ctx2.default_code1 = 0x0042;
    g_priority = 1;
    ctx2.ops.push_back(make_op("Conv1", 0x8000));
    m42_tcm_migration(ctx2, bc2, hk, 0x8000, false, tr2);
    CHECK(tr2.log.find("3459") != std::string::npos && tr2.log.find("Conv1") != std::string::npos,
          "demand<=full + priority>0 logs 3459 [0x1322fd5]");
    CHECK(ctx2.errcount == 0 && tr2.early_exit_mode < 0,
          "3459 does not count, no early exit [0x1322fc7]");
    CHECK(tr2.convert_stages.size() == 4, "good path runs 4-stage convert [0x1323bee-0x1323c0c]");

    // 需求 <= 全量 且优先级==0 → 完全无日志
    M42Ctx ctx3; M42BudgetConfig bc3; M42Trace tr3;
    ctx3.default_code1 = 0x0042;
    g_priority = 0;
    ctx3.ops.push_back(make_op("Conv2", 0x100));
    m42_tcm_migration(ctx3, bc3, hk, 0x8000, false, tr3);
    CHECK(tr3.log.empty(), "no log when fits and priority==0");
}

// ---- 码失配的张量不计入累计 ---------------------------------------------------
static void test_code_mismatch_excluded() {
    M42Ctx ctx; M42BudgetConfig bc; M42Trace tr; M42Hooks hk;
    reset_hooks(hk);
    ctx.default_code1 = 0x0042;
    M42Op op; op.kind = 0xa; op.name = "X"; op.opid = 1;
    M42TensorRec t; t.tensor_kind = 0xa; t.code1 = 0x0101; // 失配（单边 >=0x100）
    op.tensors.push_back(t);
    g_tensor_sz[0] = 0; g_tensor_sz[1] = 0x90000;          // 若计入必超预算
    ctx.ops.push_back(op);
    m42_tcm_migration(ctx, bc, hk, 0x8000, false, tr);
    CHECK(ctx.errcount == 0 && tr.log.find("3472") == std::string::npos,
          "layout-mismatched tensor excluded from acc [0x1322c89-0x1322cee]");
    // kind & -2 != 0xa 的张量不计入 [0x1322d35]
    M42Ctx ctx4; M42BudgetConfig bc4; M42Trace tr4;
    ctx4.default_code1 = 0x0042;
    M42Op op4; op4.kind = 0xa; op4.name = "Y"; op4.opid = 2;
    M42TensorRec t4; t4.tensor_kind = 0x9; t4.code1 = 0x0042; // 0x9 & -2 = 8 != 0xa
    op4.tensors.push_back(t4);
    g_tensor_sz[0] = 0; g_tensor_sz[1] = 0x90000;
    ctx4.ops.push_back(op4);
    M42Trace tr5;
    m42_tcm_migration(ctx4, bc4, hk, 0x8000, false, tr5);
    CHECK(ctx4.errcount == 0, "tensor kind &-2 != 0xa excluded [0x1322d35]");
}

// ---- P2/P3 早退 --------------------------------------------------------------
static void test_early_exits() {
    // TLS 门 → 1819 [0x1321bf6]
    {
        M42Ctx ctx; M42BudgetConfig bc; M42Trace tr; M42Hooks hk;
        reset_hooks(hk);
        hk.tls_busy = []() { return true; };
        m42_tcm_migration(ctx, bc, hk, 0x8000, false, tr);
        CHECK(tr.log.find("1819") != std::string::npos, "TLS busy logs 1819 [0x1321bf6]");
        CHECK(tr.early_exit_mode == 0x11 && !tr.scan_ran,
              "TLS gate exits before scan [0x1321c67]");
        CHECK(ctx.errcount == 0, "TLS path leaves errcount untouched");
    }
    // scan 置错误 → 1824 [0x1321c80]
    {
        M42Ctx ctx; M42BudgetConfig bc; M42Trace tr; M42Hooks hk;
        reset_hooks(hk);
        hk.scan_input_graph = [](M42Ctx& c) -> bool { c.errcount = 2; return false; };
        m42_tcm_migration(ctx, bc, hk, 0x8000, false, tr);
        CHECK(tr.log.find("1824") != std::string::npos, "scan failure logs 1824 [0x1321c80]");
        CHECK(tr.early_exit_mode == 0x11, "scan failure early exit");
    }
}

// ---- B 支：op 门走估计器，不走张量累计 ----------------------------------------
static void test_branch_b() {
    M42Ctx ctx; M42BudgetConfig bc; M42Trace tr; M42Hooks hk;
    reset_hooks(hk);
    ctx.default_code1 = 0x0042;
    M42Op op; op.kind = 0xa; op.f5d = 1; op.f5c = 1;       // B 支门 [0x1322b5f-0x1322b69]
    op.name = "B0"; op.opid = 7; op.estimate = 0x20000;
    M42TensorRec t; t.tensor_kind = 0xa; t.code1 = 0x0042;
    op.tensors.push_back(t);
    g_tensor_sz[0] = 0; g_tensor_sz[1] = 0x90000;          // 若走 A 支必 3472
    ctx.ops.push_back(op);
    m42_tcm_migration(ctx, bc, hk, 0x8000, false, tr);
    CHECK(tr.log.find("3472") == std::string::npos,
          "branch B uses estimator, not tensor acc [0x1322b94-0x1322ba3]");
    // B 支估计超限 → 变换（记录级）：此处仅验证无 3472/3459（B 支无日志锚点）
    CHECK(ctx.errcount == 0, "branch B over-budget has no counter (transform only)");
}

// ---- 类型门 ------------------------------------------------------------------
static void test_kind_gate() {
    M42Ctx ctx; M42BudgetConfig bc; M42Trace tr; M42Hooks hk;
    reset_hooks(hk);
    ctx.default_code1 = 0x0042;
    M42Op op; op.kind = 3; op.name = "K"; op.opid = 9;      // kind-0xa = 大 → P4 跳过
    M42TensorRec t; t.tensor_kind = 0xa; t.code1 = 0x0042;
    op.tensors.push_back(t);
    g_tensor_sz[0] = 0; g_tensor_sz[1] = 0x90000;
    ctx.ops.push_back(op);
    // 注意 P6 对 kind=3 无显式门（真码头门在 +0x5e/+0x5f，记录级）；模型仍会走 A 支 → 应当 3472
    m42_tcm_migration(ctx, bc, hk, 0x8000, false, tr);
    CHECK(ctx.errcount == 1, "P4 kind gate is P4-only; P6 still judges [0x1322504]");
}

int main() {
    test_three_quarters();
    test_popcount();
    test_layout_match();
    test_verdict();
    test_code_mismatch_excluded();
    test_early_exits();
    test_branch_b();
    test_kind_gate();
    if (g_fail) { std::printf("\n%d FAILURE(S)\n", g_fail); return 1; }
    std::printf("\nALL M42 SMOKE TESTS PASSED\n");
    return 0;
}
