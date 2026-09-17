// M36c 冒烟测试: tile 子系统（TileShapeBase / declare_tiling_rule）反汇编保真实现
// 证据基线: audit_verify/reports/M36c_tile_shape_subsystem_disasm.md
#include "hnnx/vtcm/tile_shape_m36c.hpp"
#include <cstdio>

using namespace hnnx;

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

// 标准场景: 基准 op (dtype 0x11) + 两个命名张量 "Activations"/"Weights"
static void setup(M36cTileShapeBase& ts, M36cOpDef& base, M36cOpDef& act, M36cOpDef& wt,
                  M36cCtx& ctx) {
    base.dtype = 0x11;
    act.dtype = 0x22;  act.shape = {1, 8, 16, 3};
    wt.dtype = 0x33;   wt.shape = {4, 3, 3, 8};
    ts.base = &base;
    ts.names = {"Activations", "Weights"};
    ts.oprefs[0] = &act;
    ts.oprefs[1] = &wt;
    ts.ctx = &ctx;
    ctx.hw_vtcm_size = 0x00100000;      // 1MB 硬件 VTCM
    ctx.vtcm_tile_size = 0x00080000;    // 512KB tile 预算
}

static void test_resolve() {
    M36cTileShapeBase ts; M36cOpDef base, act, wt; M36cCtx ctx;
    setup(ts, base, act, wt, ctx);
    // "*" 通配 → 基准对象 [0x138f95f-0x138f970]
    CHECK(ts.get("*") == &base, "wildcard '*' returns base [0x138f970]");
    // 命名命中 → OpRef 表解引用 [0x138f9d3-0x138f9e1]
    CHECK(ts.get("Activations") == &act && ts.get("Weights") == &wt,
          "named lookup walks OpRef table [0x138f990-0x138f9e1]");
    // miss → 888 ERROR [0x138f9af-0x138f9cb]
    CHECK(ts.get("Bias") == nullptr, "miss returns null (record-level)");
    CHECK(ts.log.find("tiling_registration.cc:888") != std::string::npos &&
          ts.log.find("Bias") != std::string::npos,
          "888 error text matches fmt@0x55ba472");
    // 顺序敏感: 名字表是线性扫（首个命中即返回）
    ts.names = {"Weights", "Activations", "Weights"};
    ts.oprefs[0] = &wt; ts.oprefs[1] = &act; ts.oprefs[2] = &base;
    CHECK(ts.get("Weights") == &wt, "linear scan takes first match [0x138f990]");
}

static void test_options() {
    M36cTileShapeBase ts; M36cOpDef base, act, wt; M36cCtx ctx;
    setup(ts, base, act, wt, ctx);
    ctx.options["central_tiler_frac"] = 75;
    // tcm_size → 硬件事务 [0x138fab0]
    CHECK(ts.option_uint("tcm_size") == 0x00100000,
          "tcm_size -> nn_os_vtcm_get_hardware_size [0x138fab0]");
    // tcm_size_for_tiling → get_vtcm_tile_size [0x138fad3]
    CHECK(ts.option_uint("tcm_size_for_tiling") == 0x00080000,
          "tcm_size_for_tiling -> get_vtcm_tile_size [0x138fad3]");
    // 普通键 → ctx+0x54d0 选项表 [0x138fada-0x138faf0]
    CHECK(ts.option_uint("central_tiler_frac") == 75,
          "other keys hit options map at ctx+0x54d0");
    // miss → WARNING + 0（记录级缺省）[0x138fa30-0x138fa60]
    CHECK(ts.option_int("no_such_key") == 0, "missing key returns default 0");
    CHECK(ts.log.find("no_such_key") != std::string::npos, "missing key logs WARNING");
}

static void test_gen_perf_shape() {
    M36cTileShapeBase ts; M36cOpDef base, act, wt; M36cCtx ctx;
    setup(ts, base, act, wt, ctx);
    // 正常: {4,w,x,y,z} [0x1391d65-0x1391d7a]
    M36cTinyVector r = ts.gen_perf_shape(2, 4, 8, 16);
    CHECK(r.count == 4 && r.v[0] == 2 && r.v[1] == 4 && r.v[2] == 8 && r.v[3] == 16,
          "gen_perf_Shape -> {4,w,x,y,z} [0x1391d6c-0x1391d76]");
    // minimize_tiling: ctx+0x5554>0 → 全零 [0x1391d27/0x1391d51]
    ctx.flag_5554 = 1;
    CHECK(ts.minimize_tiling(), "minimize_tiling = ctx+0x5554 > 0 [0x1391d27]");
    M36cTinyVector z = ts.gen_perf_shape(2, 4, 8, 16);
    M36cTinyVector zero; zero.count = 4;
    CHECK(z == zero, "minimize zeroes the shape [0x1391d53-0x1391d64]");
}

static void test_build() {
    M36cTileShapeBase ts; M36cOpDef base, act, wt; M36cCtx ctx;
    setup(ts, base, act, wt, ctx);
    M36cTinyVector tv; tv.count = 4; tv.v[0] = 1; tv.v[1] = 2; tv.v[2] = 3; tv.v[3] = 4;
    // crouton("Activations", TV): 解析 + dtype=+0x4c [0x1391819]
    auto b = ts.build("crouton", "Activations", tv);
    CHECK(b.resolved && b.layout == "crouton" && b.dtype == 0x22,
          "crouton(name,TV) resolves and takes dtype from +0x4c [0x1391819]");
    // weights("*", TV): 通配到基准
    auto b2 = ts.build("weights", "*", tv);
    CHECK(b2.resolved && b2.dtype == 0x11, "weights('*') uses base dtype");
    // miss: resolved=false + 888 记录
    auto b3 = ts.build("flat", "Nothing", tv);
    CHECK(!b3.resolved && ts.log.find("Nothing") != std::string::npos,
          "flat(miss) records 888 and returns unresolved");
}

static void test_declare_rule() {
    M36cTileShapeBase ts; M36cOpDef base, act, wt; M36cCtx ctx;
    setup(ts, base, act, wt, ctx);
    std::vector<M36cTilingRule> reg;
    // 注册门: flags+0xd & 0x48 → 不注册 [0x138f8dc]
    M36cOpDef bad; bad.flags_0xd = 0x48;
    CHECK(m36c_declare_tiling_rule(ts, 7, "ConvLayer", bad, reg) == 0 && reg.empty(),
          "0x48-flagged holder is not registered [0x138f8dc]");
    // 正常注册: 0x20B 记录 {fn, id, holder, name} [0x138f8e2-0x138f914]
    M36cOpDef good; good.flags_0xd = 0x00;
    CHECK(m36c_declare_tiling_rule(ts, 7, "ConvLayer", good, reg) == 0 && reg.size() == 1,
          "clean holder appends one 0x20B record [0x138f8f2]");
    CHECK(reg[0].id == 7 && reg[0].holder == &good && reg[0].name == "ConvLayer" &&
          reg[0].shape_fn != nullptr,
          "record fields {+0 fn, +8 id, +0x10 holder, +0x18 name} [0x138f8f5-0x138f905]");
    // 返回值恒 0 [0x138f944]
    M36cOpDef bad2; bad2.flags_0xd = 0x08;   // 只 0x08 也命中门（0x48=0x40|0x08）
    CHECK(m36c_declare_tiling_rule(ts, 8, "x", bad2, reg) == 0 && reg.size() == 1,
          "gate is bit-test 0x48 (0x08 alone triggers) [0x138f8dc]");
}

int main() {
    test_resolve();
    test_options();
    test_gen_perf_shape();
    test_build();
    test_declare_rule();
    if (g_fail) { std::printf("\n%d FAILURE(S)\n", g_fail); return 1; }
    std::printf("\nALL M36c SMOKE TESTS PASSED\n");
    return 0;
}
