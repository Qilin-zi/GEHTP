// ============================================================================
// test_st_cut.cpp — st_cut 重实现（M31）冒烟测试
// 覆盖：建图层原语（connect_nodes/槽互指）、洪泛+权重搬运、aux 曲线、评分器、
//       验证器（合法/非法序）、重放账本、develop_schedule 终段原地重写、
//       full_schedule 重试循环（轮次/择优/恢复最优序）。
// 目标：无崩溃 + 工作序恒为排列 + 基本不变量成立（非精确数值比对——概要级
//       提案器/标记原语的数值行为不与 .so 对表）。
// ============================================================================

#include "hnnx/scheduler/st_cut.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <set>

using namespace hnnx;

static int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do { if (!(cond)) { std::printf("FAIL: %s (line %d)\n", msg, __LINE__); ++g_fail; } } while (0)

// ---- 1. 建图层原语：替换对建立（0x1300cc0）----
static void test_connect_nodes()
{
    StCutContext ctx;
    ctx.mode = 0x40;                                   // 建表期 [12fba54]
    for (uint32_t i = 0; i < 4; ++i) stcut_register_node(ctx, i, 0);

    uint32_t a = stcut_connect_nodes(ctx, 0, 1, 1000, nullptr);
    uint32_t b = ctx.weight_table[a].idx;              // A.idx=B 互指链
    // A.canon=dst / A.idx=B / B.weight=0x5f5e0ff / B.flags=mode|4 [1300dc5-1300e13]
    CHECK(ctx.weight_table[a].canon == 1, "slot A canon=dst");
    CHECK(ctx.weight_table[b].canon == 0, "slot B canon=src");
    CHECK(ctx.weight_table[b].idx == a, "A/B 互指（idx 链闭合）");
    CHECK(ctx.weight_table[b].weight == STCUT_MAX_WEIGHT, "slot B weight=0x5f5e0ff");
    CHECK((ctx.weight_table[b].flags & 4) != 0, "slot B flags=mode|4");
    // 四向建链 [1300ed8-1300f07]
    CHECK(std::find(ctx.add_set[0].begin(), ctx.add_set[0].end(), a) != ctx.add_set[0].end(),
          "add集[src] 含 A");
    CHECK(std::find(ctx.add_set[1].begin(), ctx.add_set[1].end(), b) != ctx.add_set[1].end(),
          "add集[dst] 含 B");
    CHECK(std::find(ctx.subtract_set[0].begin(), ctx.subtract_set[0].end(), b) != ctx.subtract_set[0].end(),
          "subtract集[src] 含 B");
    CHECK(std::find(ctx.subtract_set[1].begin(), ctx.subtract_set[1].end(), a) != ctx.subtract_set[1].end(),
          "subtract集[dst] 含 A");

    // 命中路径：重复 connect 累加权重 [1301008-130101d]
    uint32_t a2 = stcut_connect_nodes(ctx, 0, 1, 500, nullptr);
    CHECK(a2 == a, "命中返回同一槽");
    CHECK(ctx.weight_table[a].weight == 1500, "累加权重 1000+500");

    std::printf("test_connect_nodes OK\n");
}

// ---- 2. 洪泛收集 + 迭代驱动（0x1309230/0x13096b0/0x1309810）----
static void test_flood_and_iterate()
{
    // 链式流网：S=0 → 1 → 2 → T=3，各边权重 10
    StCutContext ctx;
    ctx.mode = 0x40;
    for (uint32_t i = 0; i < 4; ++i) stcut_register_node(ctx, i, 0);
    stcut_connect_nodes(ctx, 0, 1, 10, nullptr);
    stcut_connect_nodes(ctx, 1, 2, 10, nullptr);
    stcut_connect_nodes(ctx, 2, 3, 10, nullptr);

    uint64_t collected = stcut_flood_collect(ctx, 0, 3);   // [1309230]
    CHECK(collected > 0, "洪泛应收集到中间槽");

    uint64_t moved = stcut_iterate(ctx, 0, 3);             // [1309810]
    CHECK(moved > 0, "迭代驱动应搬运出权重");
    CHECK(ctx.call_count == 1, "迭代驱动计数 [ctx+0x10]");

    std::printf("test_flood_and_iterate OK (collected=%llu moved=%llu)\n",
                (unsigned long long)collected, (unsigned long long)moved);
}

// ---- 3. aux 前缀和 + 峰值评分（0x130cea0 / 0x13065f0）----
static void test_aux_and_peak()
{
    StCutGraphInput in;
    in.node_count = 4;
    in.initial_order = { 0, 1, 2, 3 };
    in.relations = { { 0, 1, 100, false }, { 1, 2, 200, false }, { 2, 3, 50, false } };

    StCutContext ctx;
    stcut_build_initial_state(ctx, in);
    // grain(id) = Σ add集[id] 槽权重 [0x130ccd0]：
    //   add集[0]={A01}→100；add集[1]={B01,A12}→300；add集[2]={B12,A23}→250；add集[3]={B23}→50
    CHECK(stcut_grain_of(ctx, 0) == 100, "grain[0]=100");
    CHECK(stcut_grain_of(ctx, 1) == 300, "grain[1]=100+200=300");
    CHECK(stcut_grain_of(ctx, 2) == 250, "grain[2]=200+50=250");

    std::vector<uint64_t> aux;
    stcut_aux_fill(ctx, aux, 0, 4);                        // [130cea0(0,n) 全量]
    CHECK(aux.size() == 4, "aux 长度 = n");
    CHECK(aux[0] == 100 && aux[3] == 700, "aux 前缀和曲线");

    uint64_t peak = stcut_measure_peak(ctx);               // [13065f0]
    CHECK(peak == 700, "峰值 = 曲线终值（单调链）");

    std::printf("test_aux_and_peak OK (peak=%llu)\n", (unsigned long long)peak);
}

// ---- 4. 验证器（0x1306a20）：合法序过、前驱倒置序挂 ----
// 注意：验证器走快照邻接且跳 flags&4 边（M28）——约束序的是软关系，非硬依赖。
static void test_validate()
{
    StCutGraphInput in;
    in.node_count = 3;
    in.initial_order = { 0, 1, 2 };
    in.relations = { { 0, 1, 10, false }, { 1, 2, 10, false } };   // 软链 0→1→2

    StCutContext ctx;
    stcut_build_initial_state(ctx, in);
    stcut_snapshot_five(ctx);                              // 验证器走快照邻接（ctx+0x1f0）

    ctx.working_order = { 0, 1, 2 };
    CHECK(stcut_validate_order(ctx), "拓扑序合法");
    ctx.working_order = { 1, 0, 2 };
    CHECK(!stcut_validate_order(ctx), "前驱倒置应判 Bad Schedule");
    ctx.working_order = { 2, 1, 0 };
    CHECK(!stcut_validate_order(ctx), "全倒序应判 Bad Schedule");
    ctx.working_order = { 0, 2, 1 };
    CHECK(!stcut_validate_order(ctx), "尾双倒置应判 Bad Schedule");

    std::printf("test_validate OK\n");
}

// ---- 5. 重放账本（0x13080a0）：type0 置权重 / type2 移出 / type4 重加 ----
// 重放为「倒序弹出」（撤销语义）——逐条单独应用以观察各 case 的效果。
static void test_replay()
{
    StCutContext ctx;
    ctx.mode = 0x40;
    for (uint32_t i = 0; i < 3; ++i) stcut_register_node(ctx, i, 0);
    uint32_t a = stcut_connect_nodes(ctx, 0, 1, 700, nullptr);

    std::vector<StCutReplayRec> log;

    log.push_back({ a, 123, 0 });                         // type0 置权重
    stcut_replay_apply(ctx, log);
    CHECK(ctx.weight_table[a].weight == 123, "type0 置权重");

    log.push_back({ a, 0, 2 });                           // type2 移出 add 集
    stcut_replay_apply(ctx, log);
    CHECK(std::find(ctx.add_set[0].begin(), ctx.add_set[0].end(), a) == ctx.add_set[0].end(),
          "type2 移出 add 集（宿主=weight_table[A.idx].canon=src）");

    log.push_back({ a, 0, 4 });                           // type4 重加 add 集
    stcut_replay_apply(ctx, log);
    CHECK(std::find(ctx.add_set[0].begin(), ctx.add_set[0].end(), a) != ctx.add_set[0].end(),
          "type4 重加 add 集");

    std::printf("test_replay OK\n");
}

// ---- 6. develop_schedule 单轮：journal 分裂 + 终段原地重写 ----
static void test_develop_schedule()
{
    StCutGraphInput in;
    in.node_count = 8;
    in.initial_order.resize(8);
    std::iota(in.initial_order.begin(), in.initial_order.end(), 0u);
    for (uint32_t i = 0; i + 1 < 8; ++i) in.relations.push_back({ i, i + 1, 100, false });
    in.relations.push_back({ 0, 7, 9000, false });        // 长命张量

    StCutContext ctx;
    stcut_build_initial_state(ctx, in);
    ctx.threshold_base = 2 * STCUT_GRAIN_BYTES;           // [ctx+0x278]（语义未解，测试取小值）
    stcut_snapshot_five(ctx);

    std::vector<uint64_t> aux;
    stcut_aux_fill(ctx, aux, 0, 8);
    std::vector<uint32_t> out_vec;
    stcut_develop_schedule(ctx, 0, 7, out_vec, aux, 100000, 4);   // [0x130d3e0]

    CHECK(ctx.journal.size() > 3, "主循环产生分裂子记录（初始 3 条之上）");
    std::set<uint32_t> uniq(ctx.working_order.begin(), ctx.working_order.end());
    CHECK(uniq.size() == 8, "终段原地重写后仍为排列 [130e79b]");
    // journal 区间守恒：活记录区间总长 = 8（父 span−=cut_count [130b698]，子 = cut_count）
    uint64_t total_span = 0;
    for (auto &r : ctx.journal)
        if (r.link_fwd != 0) total_span += r.span_len;
    CHECK(total_span == 8, "父/子区间长守恒");

    std::printf("test_develop_schedule OK: journal=%zu span_total=%llu\n",
                ctx.journal.size(), (unsigned long long)total_span);
}

// ---- 7. full_schedule 重试循环端到端（M27 十二步）----
static void test_full_schedule()
{
    // 12 节点：硬依赖链 0→1→…→11（flags&4，Kahn 终段消费），叠加软共享（验证器消费）
    StCutGraphInput in;
    in.node_count = 12;
    in.initial_order.resize(12);
    std::iota(in.initial_order.begin(), in.initial_order.end(), 0u);
    for (uint32_t i = 0; i + 1 < 12; ++i)
        in.relations.push_back({ i, i + 1, 300, false });         // 软链（验证器约束）
    for (uint32_t i = 0; i + 3 < 12; ++i)
        in.relations.push_back({ i, i + 3, 5000, false });        // 软共享（跨 3 层）
    in.relations.push_back({ 0, 11, 8000, false });               // 长命张量（全程占用）
    in.relations.push_back({ 5, 6, 400, true });                  // 硬依赖（Kahn 消费）

    StCutContext ctx;
    stcut_build_initial_state(ctx, in);
    ctx.threshold_base = 4 * STCUT_GRAIN_BYTES;           // [ctx+0x278]（语义未解，测试取小值）
    ctx.config.threshold_mult = 1.0;                      // [[ctx+0x270]+0x128]
    ctx.config.emit_terminal = true;                      // [+0x110]
    ctx.config.restart_gate = 0;                          // [+0x158] 关随机重启（可复现）
    ctx.config.deep_copy_gate = false;                    // [+0x160]

    StCutOptions opt;                                     // SCHED_OPTIONS（this+0x55d4…）
    opt.rt = 3;                                           // 最大轮数 [+0x55dc]
    opt.it = 50;                                          // 预算分子乘子 [+0x55d8]
    opt.rg = 20;                                          // 选项#1 → 0x1306750 [+0x55d4]
    opt.am = 0;                                           // Bad Schedule 不 fatal [+0x55e8]
    opt.budget_base = 1ull << 40;                         // this+0x6830
    opt.tr = 1.0;                                         // TR [+0x55f8]

    uint64_t peak_before = stcut_measure_peak(ctx);       // 循环前 delay_dma 基准

    std::vector<uint32_t> best;
    std::vector<uint64_t> flows, cycles;
    stcut_full_schedule(ctx, opt, in.initial_order, best, flows, cycles, 0, 11);

    CHECK(!best.empty(), "最优序非空（退出恢复快照#1 [13044cb]）");
    CHECK(flows.size() == cycles.size() && flows.size() >= 1, "FLOWS/CYCLES 逐轮记录");
    CHECK(best.size() == 12, "最优序长度守恒");
    std::set<uint32_t> uniq(best.begin(), best.end());
    CHECK(uniq.size() == 12, "最优序是排列（无丢无重）");
    for (uint32_t v : best) CHECK(v < 12, "节点号在界内");

    std::printf("test_full_schedule OK: peak_before=%llu best_flow=%llu rounds=%zu\n",
                (unsigned long long)peak_before,
                (unsigned long long)(flows.empty() ? 0 : *std::min_element(flows.begin(), flows.end())),
                flows.size());
    std::printf("  flows:");
    for (uint64_t f : flows) std::printf(" %llu", (unsigned long long)f);
    std::printf("\n  best order:");
    for (uint32_t v : best) std::printf(" %u", v);
    std::printf("\n");
}

// ---- 8. stcut_read_nodes 候选表填充（M33）+ 带组的 develop 不变式 ----
static void test_read_nodes_groups()
{
    // 8 节点链 + 两组强关联（组 0xaaa 覆盖 5/1/3（乱序给排序断言），组 0xbbb 覆盖 6/2）
    StCutGraphInput in;
    in.node_count = 8;
    in.initial_order.resize(8);
    std::iota(in.initial_order.begin(), in.initial_order.end(), 0u);
    for (uint32_t i = 0; i + 1 < 8; ++i) in.relations.push_back({ i, i + 1, 100, false });
    in.relations.push_back({ 0, 7, 9000, false });
    in.groups[0xaaa] = { 5, 1, 3 };
    in.groups[0xbbb] = { 6, 2 };

    StCutContext ctx;
    stcut_build_initial_state(ctx, in);

    // 槽位/表长：register_node 级联 resize [13013bc-1301416]
    CHECK(ctx.cand_tables.size() == 8, "cand_tables 槽数 = 节点数");
    // 成员表排序写回 [12fcbf0-12fcc51]
    CHECK(ctx.relation_groups[0xaaa] == (std::vector<uint32_t>{ 1, 3, 5 }), "组成员表按 id 升序写回");
    CHECK(ctx.relation_groups[0xbbb] == (std::vector<uint32_t>{ 2, 6 }), "组 0xbbb 排序");
    // 指派 [12fcc79]：同组成员表一致；未入组节点为空
    CHECK(ctx.cand_tables[1] == ctx.cand_tables[5] && ctx.cand_tables[1].size() == 3,
          "同组节点共享成员表（0xaaa，3 员）");
    CHECK(ctx.cand_tables[2].size() == 2 && ctx.cand_tables[6].size() == 2, "组 0xbbb 两员");
    CHECK(ctx.cand_tables[0].empty() && ctx.cand_tables[4].empty() && ctx.cand_tables[7].empty(),
          "未入组节点表为空");
    // 组登记 [12fc916 → ctx+0x258]
    CHECK(ctx.group_list_keys.size() == 2, "组表收集两条（ctx+0x258）");

    // 越界成员告警路 [12fcac8 "Node index %u is larger than size %zu"]
    StCutGraphInput bad = in;
    bad.groups[0xccc] = { 2, 99 };
    StCutContext ctx2;
    stcut_build_initial_state(ctx2, bad);
    CHECK(ctx2.relation_groups[0xccc] == (std::vector<uint32_t>{ 2 }), "越界成员被剔除（99 ≥ 8）");
    // 节点 2 同属 0xbbb/0xccc：map 按键升序遍历，后组覆盖先组（.so 同为后写胜 [12fcc79]）
    CHECK(ctx2.cand_tables[2] == (std::vector<uint32_t>{ 2 }), "跨组节点取末组（0xccc 覆盖 0xbbb）");

    // 带组跑 develop：提案器外部/通用路消费 cand_tables 后不变式必须保持
    StCutContext ctx3;
    stcut_build_initial_state(ctx3, in);
    ctx3.threshold_base = 2 * STCUT_GRAIN_BYTES;
    stcut_snapshot_five(ctx3);
    std::vector<uint64_t> aux;
    stcut_aux_fill(ctx3, aux, 0, 8);
    std::vector<uint32_t> out_vec;
    stcut_develop_schedule(ctx3, 0, 7, out_vec, aux, 100000, 4);
    std::set<uint32_t> uniq(ctx3.working_order.begin(), ctx3.working_order.end());
    CHECK(uniq.size() == 8, "带组 develop 终态仍为排列");
    uint64_t total_span = 0;
    for (auto &r : ctx3.journal)
        if (r.link_fwd != 0) total_span += r.span_len;
    CHECK(total_span == 8, "带组 develop 区间长守恒");

    std::printf("test_read_nodes_groups OK: journal=%zu span_total=%llu\n",
                ctx3.journal.size(), (unsigned long long)total_span);
}

// ---- 9. 标记原语层语义（M34：0x1305130 软边正向 / 0x1308c50 &4 反向）----
static void test_mark_primitives()
{
    // 链 0→1→2→3（软）：seed=0 → 层号 2,3,4,5（首层 2 [13051c3]，每层 +1 [13051ee]）
    {
        StCutGraphInput in;
        in.node_count = 5;                                 // 4 号孤立（未达 → mark=0）
        in.initial_order = { 0, 1, 2, 3, 4 };
        in.relations = { { 0, 1, 10, false }, { 1, 2, 10, false }, { 2, 3, 10, false } };
        StCutContext ctx;
        stcut_build_initial_state(ctx, in);
        stcut_mark_levels(ctx, 0);
        CHECK(ctx.mark_table[0] == 2, "seed 层=2（首层不加 [13051da]）");
        CHECK(ctx.mark_table[1] == 3 && ctx.mark_table[2] == 4 && ctx.mark_table[3] == 5,
              "链上逐层 +1");
        CHECK(ctx.mark_table[4] == 0, "未达节点 mark=0");
    }
    // 菱形 0→{1,2}→3：1/2 同层 3；3 等双前驱定层 → 层 4（就绪门 [130529b]）
    {
        StCutGraphInput in;
        in.node_count = 4;
        in.initial_order = { 0, 1, 2, 3 };
        in.relations = { { 0, 1, 10, false }, { 0, 2, 10, false },
                         { 1, 3, 10, false }, { 2, 3, 10, false } };
        StCutContext ctx;
        stcut_build_initial_state(ctx, in);
        stcut_mark_levels(ctx, 0);
        CHECK(ctx.mark_table[1] == 3 && ctx.mark_table[2] == 3, "分叉同层");
        CHECK(ctx.mark_table[3] == 4, "汇点等双前驱（软前驱全定层才就绪）");
    }
    // 硬边与软边同层（M34 修正：hard 不落 A&4——A&4 会令 0x1308c50 就绪门自锁）：
    //   0→1 软、0→2 硬，seed=0 → 1/2 同层 3
    {
        StCutGraphInput in;
        in.node_count = 3;
        in.initial_order = { 0, 1, 2 };
        in.relations = { { 0, 1, 10, false }, { 0, 2, 10, true } };
        StCutContext ctx;
        stcut_build_initial_state(ctx, in);
        stcut_mark_levels(ctx, 0);
        CHECK(ctx.mark_table[0] == 2 && ctx.mark_table[1] == 3, "软链照常分层");
        CHECK(ctx.mark_table[2] == 3, "硬边同层参与分层（A 侧不置 4）");
    }
    // 0x1308c50 &4 反向孪生：软链 0→1→2，seed=2 → 沿 B 半边（canon=前驱）反走
    //   层号：2→2，1→3，0→4；写外部表，ctx+0x128 不动 [1308c7a memset 外部表]
    {
        StCutGraphInput in;
        in.node_count = 3;
        in.initial_order = { 0, 1, 2 };
        in.relations = { { 0, 1, 10, false }, { 1, 2, 10, false } };
        StCutContext ctx;
        stcut_build_initial_state(ctx, in);
        stcut_mark_levels(ctx, 0);                        // 先正向打标（ctx+0x128 = 2,3,4）
        std::vector<uint16_t> marks = ctx.mark_table;      // [130a36a-130a382 assign 副本]
        for (auto &m : marks) m = 0x5a5a;                  // 填哨兵验证 memset
        stcut_level_mark_bfs(ctx, 2, marks);               // [130a391 种子=arg3 反向]
        CHECK(marks[2] == 2 && marks[1] == 3 && marks[0] == 4, "&4 反向分层（B 半边 canon=前驱）");
        CHECK(ctx.mark_table[0] == 2 && ctx.mark_table[2] == 4, "外部表写入，ctx+0x128 保留正向结果");
    }
    // ★中点种子结构性死锁（M34 结论，正测）：链 0→1→2，twin seed=1。
    //   就绪门（循环1 [1308dc2]）要求 node1 的 &4 对端——subtract集[1] 里 B(1→2)
    //   的伙伴 A.canon = 后继 2——先定层；而后继 2 只能从 1 的反向波里发现
    //   （[1308f72] child=canon 沿 B 半边走前驱）→ 波不含 2，node1 永不就绪。
    //   .so 侧同构无界（[1308d16] 循环条件只有 queue 非空，无层上限）；
    //   正向原语 0x1305130 链中点种子同构死锁（等前向不可达的软前驱）。
    //   概要级调用方传 max_levels 截断（.so 无此参——发散点，已记录文档）。
    {
        StCutGraphInput in;
        in.node_count = 3;
        in.initial_order = { 0, 1, 2 };
        in.relations = { { 0, 1, 10, false }, { 1, 2, 10, false } };
        StCutContext ctx;
        stcut_build_initial_state(ctx, in);
        std::vector<uint16_t> marks(ctx.node_count(), 0);
        stcut_level_mark_bfs(ctx, 1, marks, 3);       // 有界：3 层后强制截断
        CHECK(marks[0] == 0 && marks[1] == 0 && marks[2] == 0,
              "中点种子波死等后继（无层可写），护栏截断后全 0");
        // 对照：极性本身已由上一块覆盖——seed=2 时 1@3 早于 0@4 定层，
        //   正说明就绪门只认 &4 对端（后继）[1308dc2]，非 &4 的 A 槽不构成依赖。
    }

    std::printf("test_mark_primitives OK\n");
}

// ---- 10. random_restart：撤销一刀，区间归还链前向 [0x130bc70 全解] ----
static void test_random_restart()
{
    StCutGraphInput in;
    in.node_count = 8;
    in.initial_order.resize(8);
    std::iota(in.initial_order.begin(), in.initial_order.end(), 0u);
    for (uint32_t i = 0; i + 1 < 8; ++i) in.relations.push_back({ i, i + 1, 100, false });
    in.relations.push_back({ 0, 7, 9000, false });        // 长命张量

    StCutContext ctx;
    stcut_build_initial_state(ctx, in);
    ctx.threshold_base = 2 * STCUT_GRAIN_BYTES;
    stcut_snapshot_five(ctx);

    std::vector<uint64_t> aux;
    stcut_aux_fill(ctx, aux, 0, 8);
    std::vector<uint32_t> out_vec;
    stcut_develop_schedule(ctx, 0, 7, out_vec, aux, 100000, 4);

    // 选一条链前向有效、span>0 的子记录
    size_t pick = SIZE_MAX;
    for (size_t i = 3; i < ctx.journal.size(); ++i) {
        const StCutRec &r = ctx.journal[i];
        if (r.link_fwd != 0 && r.link_fwd < ctx.journal.size() && r.span_len > 0) { pick = i; break; }
    }
    CHECK(pick != SIZE_MAX, "存在可重启子记录");

    const uint64_t span_old = ctx.journal[pick].span_len;
    const size_t fwd_idx = ctx.journal[pick].link_fwd;
    const uint64_t fwd_span_old = ctx.journal[fwd_idx].span_len;
    const uint16_t fwd_id = ctx.journal[fwd_idx].id;
    std::vector<uint32_t> range_nodes;                     // rec 旧区间节点（归属改写前）
    for (uint64_t k = 0; k < span_old; ++k)
        range_nodes.push_back(ctx.working_order[ctx.journal[pick].range_start + k]);

    stcut_random_restart(ctx, pick);

    const StCutRec &rec = ctx.journal[pick];
    CHECK(rec.link_fwd == 0 && rec.link_back == 0, "重启后记录判死（@0/@8 同清）[130bcc8]");
    CHECK(rec.span_len == 0, "区间清零 [130c8d7]");
    CHECK(rec.add_entry == 0 && rec.sub_entry == 0 && rec.field24 == 0,
          "端点三字段清零 [130cac2/130caca/130cad1]");
    CHECK(ctx.journal[fwd_idx].span_len == fwd_span_old + span_old,
          "区间并入链前向 [130c8d2]");
    bool no_inlink = true;
    for (auto &r : ctx.journal)
        if ((&r != &rec) && (r.link_fwd == pick || r.link_back == pick)) no_inlink = false;
    CHECK(no_inlink, "双向跨接后无记录再指向被撤销者 [130bcb3/130bcc0]");
    bool owned = true;
    for (uint32_t n : range_nodes)
        if (ctx.ownership[n] != fwd_id) owned = false;
    CHECK(owned, "旧区间节点归属改记 fwd.id（u16）[130c83b-130c8c6]");
    uint64_t total_span = 0;
    for (auto &r : ctx.journal)
        if (r.link_fwd != 0) total_span += r.span_len;
    CHECK(total_span == 8, "活记录区间长守恒（8）");

    std::printf("test_random_restart OK: pick=%zu fwd=%zu span=%llu\n",
                pick, fwd_idx, (unsigned long long)span_old);
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);           // 无缓冲：挂点可见
    test_connect_nodes();
    test_flood_and_iterate();
    test_aux_and_peak();
    test_validate();
    test_replay();
    test_develop_schedule();
    test_full_schedule();
    test_read_nodes_groups();
    test_mark_primitives();
    test_random_restart();

    if (g_fail == 0) std::printf("\nALL ST_CUT TESTS PASSED\n");
    else             std::printf("\n%d CHECK(S) FAILED\n", g_fail);
    return g_fail ? 1 : 0;
}
