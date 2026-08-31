// test_cp_solver.cpp — oracle-solver regression tests.
//
// Known-answer anchors come from US20240386237 FIG.6/7 (worked out by hand in
// patents_qnn/CP_SOLVER_WORKED_EXAMPLE.md):
//   例 B (m1=2): the ONLY optimum under C={1,2,1,1,1} is spill t2 after slot
//        3 / fill at slot 5 -> DDR = 2, peak = 3. (The doc's "symmetric
//        spill-t4" note would need C_v4 >= 2, contradicting its own C
//        assignment — with C_v4=1 it is not a legal solution, so the exact
//        event placement below is deterministic.)
//   例 A (m1=1): the doc's remat solution (discard t2 after slot 3, remat v2
//        at slot 5 with t1 held open) violates Eq.6 at slot 3 when M=3
//        (t1+t2+t3 = 4 > 3) — it only becomes legal at M>=4. Tests below
//        cover both the honest M=3 optimum (spill, DDR=2) and the M=4 remat
//        solution (DDR=0, compute=6=W).
//
// Graph (idx 0..4 = v1..v5):  v1 -> v2 -> v3 -> v4 -> v5, v2 -> v5
// m = {m1, 1, 2, 1, 1}, w = 1, M = 3, W = 6.

#include "hnnx/vtcm/cp_solver.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

using namespace hnnx;

static int tests_passed = 0;
static int tests_failed = 0;
static void check(bool cond, const char* msg) {
    if (cond) {
        tests_passed++;
    } else {
        tests_failed++;
        std::printf("FAIL: %s\n", msg);
    }
}

static cp::CPProblem fig6(uint64_t m1, uint64_t M = 3, uint64_t W = 6) {
    cp::CPProblem p;
    p.nodes = {
        {1, m1, 1, 1, false}, // v1 (C=1)
        {2, 1, 1, 2, false},  // v2 (C=2: spill/fill or remat possible)
        {3, 2, 1, 1, false},  // v3
        {4, 1, 1, 1, false},  // v4
        {5, 1, 1, 1, false},  // v5
    };
    p.edges = {{0, 1}, {1, 2}, {2, 3}, {1, 4}, {3, 4}};
    p.M = M;
    p.W = W;
    p.topo_hint = {1, 2, 3, 4, 5};
    return p;
}

static cp::CPProblem remat_fixture() {
    // same topology as fig6, m = {1,2,2,2,1}, M=5, W=6: naive slot 4
    // (t2+t3+t4 = 6) overflows, spill costs 4, remat v2 @5 fits -> DDR 0
    cp::CPProblem p;
    p.nodes = {{1, 1, 1, 1, false}, {2, 2, 1, 2, false}, {3, 2, 1, 1, false},
               {4, 2, 1, 1, false}, {5, 1, 1, 1, false}};
    p.edges = {{0, 1}, {1, 2}, {2, 3}, {1, 4}, {3, 4}};
    p.M = 5;
    p.W = 6;
    p.topo_hint = {1, 2, 3, 4, 5};
    return p;
}

static const cp::CPTask* find_task(const cp::CPSolution& s, cp::CPTask::Kind k, op_id_t v) {
    for (const auto& t : s.seq)
        if (t.kind == k && t.v == v) return &t;
    return nullptr;
}

// ── 1. FIG.6 例 B: spill t2 is forced, DDR = 2, optimal ──
static void test_fig6_b() {
    cp::CPProblem p = fig6(2);
    cp::CPProblem keep = p;
    cp::CPSolver solver(std::move(p), cp::CPOptions{});
    cp::CPSolution r = solver.solve();
    check(r.feasible, "fig6b: feasible");
    check(r.optimal, "fig6b: proven optimal");
    check(r.ddr_bytes == 2, "fig6b: DDR == 2 (spill+fill of m=1 tensor)");
    check(r.peak_resident == 3, "fig6b: peak == 3 == M");
    check(r.spill_count == 1, "fig6b: exactly one spill");
    check(r.remat_count == 0, "fig6b: no remat");
    check(r.total_compute == 5, "fig6b: compute == 5 (single pass)");
    std::string why;
    check(cp::CPSolver::verify(keep, r, &why), "fig6b: verify passes");
    if (!r.feasible) return;
    const cp::CPTask* po = find_task(r, cp::CPTask::Kind::PAGE_OUT, 2);
    const cp::CPTask* pi = find_task(r, cp::CPTask::Kind::PAGE_IN, 2);
    check(po && po->slot == 3, "fig6b: PAGE_OUT(v2) after slot 3");
    check(pi && pi->slot == 5, "fig6b: PAGE_IN(v2) at slot 5");
    check(r.segments[1][0].spill_end && r.segments[1][0].s == 2 && r.segments[1][0].e == 3,
          "fig6b: v2 seg0 = [2,3] spill");
    check(r.segments[1][1].active && r.segments[1][1].fill_start && r.segments[1][1].s == 5,
          "fig6b: v2 seg1 = fill @5");
    check(r.segments[1][0].ddr_offset == r.segments[1][1].ddr_offset,
          "fig6b: Eq.9 fill address == spill address");
    check(r.placement_degraded == 0, "fig6b: all segments placed");
}

// ── 2. FIG.6 例 A ──
// The worked example's §2.2 only checks slot 5 of the remat solution
// (t1+t4+t2' = 3 <= 3) and misses slot 3: holding t1 open for the remat
// makes slot 3 residency t1+t2+t3 = 1+1+2 = 4 > M = 3 — Eq.6 kills the
// remat path at M=3 regardless of m1. At M=3 the honest optimum is the
// same spill as 例 B; the doc's intended remat solution needs M >= 4.
static void test_fig6_a() {
    // 2a. m1=1, M=3: remat still capacity-blocked -> spill, DDR = 2
    cp::CPProblem p = fig6(1);
    cp::CPProblem keep = p;
    cp::CPSolver solver(std::move(p), cp::CPOptions{});
    cp::CPSolution r = solver.solve();
    check(r.feasible, "fig6a: feasible");
    check(r.optimal, "fig6a: proven optimal");
    check(r.ddr_bytes == 2, "fig6a: DDR == 2 (remat blocked by Eq.6 at slot 3)");
    check(r.remat_count == 0, "fig6a: no remat at M=3");
    check(r.total_compute == 5, "fig6a: compute == 5 (single pass)");
    check(r.peak_resident == 3, "fig6a: peak == 3 == M");
    std::string why;
    check(cp::CPSolver::verify(keep, r, &why), "fig6a: verify passes");

    // 2b. m1=1, M=4: naive single pass fits (slot 4 = t2+t3+t4 = 4) — the
    // optimum keeps t2 resident: DDR 0 with NO remat, compute 5
    cp::CPProblem p4 = fig6(1, 4);
    cp::CPProblem keep4 = p4;
    cp::CPSolver s4(std::move(p4), cp::CPOptions{});
    cp::CPSolution r4 = s4.solve();
    check(r4.feasible, "fig6a-m4: feasible");
    check(r4.optimal, "fig6a-m4: proven optimal");
    check(r4.ddr_bytes == 0, "fig6a-m4: DDR == 0 (single pass fits)");
    check(r4.remat_count == 0, "fig6a-m4: no remat needed");
    check(r4.total_compute == 5, "fig6a-m4: compute == 5");
    check(r4.peak_resident == 4, "fig6a-m4: peak == 4 == M (slot 4)");
    std::string why4;
    check(cp::CPSolver::verify(keep4, r4, &why4), "fig6a-m4: verify passes");

    // 2c. remat is STRICTLY optimal: m = {1,2,2,2,1}, M=5 (see remat_fixture).
    // (On the original fig6 numbers remat can never win: slot 4 would need
    // t1+t3+t4 <= M < t2+t3+t4, and slot 5 t1+t4+t2' <= M — unsatisfiable
    // for M = 3 with any m1, which is why 2a degenerates to the spill.)
    cp::CPProblem pr = remat_fixture();
    cp::CPProblem keepr = pr;
    cp::CPSolver sr(std::move(pr), cp::CPOptions{});
    cp::CPSolution rr = sr.solve();
    check(rr.feasible, "remat: feasible");
    check(rr.optimal, "remat: proven optimal");
    check(rr.ddr_bytes == 0, "remat: DDR == 0 beats the 4-byte spill");
    check(rr.remat_count == 1, "remat: one remat");
    check(rr.total_compute == 6, "remat: compute == 6 == W (budget saturated)");
    check(rr.peak_resident == 5, "remat: peak == 5 == M");
    std::string whyr;
    check(cp::CPSolver::verify(keepr, rr, &whyr), "remat: verify passes");
    const cp::CPTask* rm = find_task(rr, cp::CPTask::Kind::REMAT, 2);
    check(rm && rm->slot == 5, "remat: REMAT(v2) at slot 5");
}

// ── 3. invariants across fixed graphs x 4 modes ──
struct Fixture {
    const char* name;
    cp::CPProblem p;
    bool expect_feasible_full;
};

static void run_invariants() {
    std::vector<Fixture> fx;
    {
        cp::CPProblem chain; // 1->2->3->4->5, m=2 each
        chain.nodes = {{1, 2, 1, 2, false}, {2, 2, 1, 2, false}, {3, 2, 1, 2, false},
                       {4, 2, 1, 2, false}, {5, 2, 1, 2, false}};
        chain.edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
        chain.M = 6;
        chain.W = 20;
        chain.topo_hint = {1, 2, 3, 4, 5};
        fx.push_back({"chain5", chain, true});
    }
    {
        cp::CPProblem chain_tight = fig6(2); // reuses the tight fixture
        fx.push_back({"fig6-tight", chain_tight, true});
    }
    {
        cp::CPProblem diamond;
        diamond.nodes = {{1, 2, 1, 2, false}, {2, 2, 1, 2, false},
                         {3, 2, 1, 2, false}, {4, 2, 1, 1, false}};
        diamond.edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        diamond.M = 6;
        diamond.W = 10;
        diamond.topo_hint = {1, 2, 3, 4};
        fx.push_back({"diamond", diamond, true});
    }
    {
        cp::CPProblem skip; // 1->2, 2->3, 1->3 (skip connection)
        skip.nodes = {{1, 1, 1, 2, false}, {2, 2, 1, 2, false}, {3, 1, 1, 1, false}};
        skip.edges = {{0, 1}, {1, 2}, {0, 2}};
        skip.M = 3;
        skip.W = 10;
        skip.topo_hint = {1, 2, 3};
        fx.push_back({"skip", skip, true});
    }
    {
        cp::CPProblem iso; // no edges at all
        iso.nodes = {{1, 2, 1, 2, false}, {2, 2, 1, 2, false}, {3, 2, 1, 2, false}};
        iso.M = 4;
        iso.W = 10;
        iso.topo_hint = {1, 2, 3};
        fx.push_back({"isolated", iso, true});
    }
    {
        cp::CPProblem big; // single tensor larger than M
        big.nodes = {{1, 10, 1, 2, false}};
        big.M = 5;
        big.W = 10;
        big.topo_hint = {1};
        fx.push_back({"too-big", big, false});
    }

    const cp::CPOptions::Mode modes[] = {
        cp::CPOptions::Mode::FULL, cp::CPOptions::Mode::PAGING_ONLY,
        cp::CPOptions::Mode::REMAT_ONLY, cp::CPOptions::Mode::SEQ_ONLY};
    for (auto& f : fx) {
        for (cp::CPOptions::Mode md : modes) {
            cp::CPOptions o;
            o.mode = md;
            cp::CPProblem copy = f.p;
            cp::CPSolver solver(std::move(copy), o);
            cp::CPSolution r = solver.solve();
            char buf[128];
            std::snprintf(buf, sizeof buf, "%s/%d: feasible flag", f.name, static_cast<int>(md));
            check(r.feasible == f.expect_feasible_full || !r.feasible, buf);
            std::snprintf(buf, sizeof buf, "%s/%d: abort_reason set when infeasible", f.name,
                          static_cast<int>(md));
            check(r.feasible || !r.abort_reason.empty(), buf);
            if (!r.feasible) continue;
            std::string why;
            std::snprintf(buf, sizeof buf, "%s/%d: verify", f.name, static_cast<int>(md));
            check(cp::CPSolver::verify(f.p, r, &why), buf);
            std::snprintf(buf, sizeof buf, "%s/%d: peak <= M", f.name, static_cast<int>(md));
            check(r.peak_resident <= f.p.M, buf);
            std::snprintf(buf, sizeof buf, "%s/%d: compute <= W", f.name, static_cast<int>(md));
            check(r.total_compute <= f.p.W, buf);
        }
    }
}

// ── 4. degenerate knob orderings (patent L1144) ──
static void test_knobs() {
    // ddr(FULL) <= ddr(PAGING_ONLY) on 例 B
    {
        cp::CPOptions full, pg;
        pg.mode = cp::CPOptions::Mode::PAGING_ONLY;
        cp::CPProblem p1 = fig6(2), p2 = fig6(2);
        cp::CPSolver s1(std::move(p1), full), s2(std::move(p2), pg);
        auto r1 = s1.solve(), r2 = s2.solve();
        check(r1.feasible && r2.feasible, "knob: paging/full feasible on fig6b");
        check(r1.ddr_bytes <= r2.ddr_bytes, "knob: ddr(FULL) <= ddr(PAGING_ONLY)");
        check(r1.ddr_bytes == 2 && r2.ddr_bytes == 2, "knob: fig6b both need the single spill");
    }
    // compute(REMAT_ONLY) >= compute(FULL) on the remat fixture (both = 6)
    {
        cp::CPOptions full, rm;
        rm.mode = cp::CPOptions::Mode::REMAT_ONLY;
        cp::CPProblem p1 = remat_fixture(), p2 = remat_fixture();
        cp::CPSolver s1(std::move(p1), full), s2(std::move(p2), rm);
        auto r1 = s1.solve(), r2 = s2.solve();
        check(r1.feasible && r2.feasible, "knob: remat-only feasible on remat fixture");
        check(r2.total_compute >= r1.total_compute, "knob: compute(REMAT_ONLY) >= compute(FULL)");
        check(r2.ddr_bytes == 0, "knob: remat-only never pages");
    }
    // REMAT_ONLY at M=3 is genuinely infeasible: the only relief for slot 4
    // is closing t2, and reopening it needs t1 held -> slot 3 overflows
    {
        cp::CPOptions rm;
        rm.mode = cp::CPOptions::Mode::REMAT_ONLY;
        cp::CPProblem p = fig6(1);
        cp::CPSolver s(std::move(p), rm);
        auto r = s.solve();
        check(!r.feasible, "knob: remat-only infeasible on fig6a(M=3)");
    }
    // SEQ_ONLY: no spill, no remat; peak(SEQ) >= peak(FULL) with slack M
    {
        cp::CPOptions full, sq;
        sq.mode = cp::CPOptions::Mode::SEQ_ONLY;
        cp::CPProblem p1 = fig6(1, 5), p2 = fig6(1, 5); // M=5: sequencing alone fits
        cp::CPSolver s1(std::move(p1), full), s2(std::move(p2), sq);
        auto r1 = s1.solve(), r2 = s2.solve();
        check(r1.feasible && r2.feasible, "knob: seq feasible with M=5");
        check(r2.spill_count == 0 && r2.remat_count == 0, "knob: seq-only never spills/remats");
        check(r2.ddr_bytes == 0, "knob: seq-only zero DDR");
        check(r2.peak_resident >= r1.peak_resident, "knob: peak(SEQ_ONLY) >= peak(FULL)");
    }
    // SEQ_ONLY on the tight fixture (M=3) is genuinely infeasible
    {
        cp::CPOptions sq;
        sq.mode = cp::CPOptions::Mode::SEQ_ONLY;
        cp::CPProblem p = fig6(2);
        cp::CPSolver s(std::move(p), sq);
        auto r = s.solve();
        check(!r.feasible, "knob: seq-only infeasible on fig6b (peak 4 > M 3)");
    }
}

// ── 5. A/B oracle vs FancyAllocator (greedy approximation) ──
static void test_oracle_ab() {
    std::vector<fa::FancyAllocator::AllocRequest> reqs = {
        {1, 2, 0, 1}, {2, 1, 1, 4}, {3, 2, 2, 3}, {4, 1, 3, 4}, {5, 1, 4, 4},
    };
    fa::FancyAllocator alloc;
    auto g = alloc.allocate_with_lifetime(reqs, 3, 1);
    uint64_t greedy_spilled_m = 0;
    bool any = false;
    for (const auto& r : reqs) {
        auto it = g.find(r.op_id);
        if (it != g.end() && it->second.spilled) {
            greedy_spilled_m += r.size;
            any = true;
        }
    }
    check(any, "oracle: fancy spills something on the tight fixture");
    check(greedy_spilled_m >= 1, "oracle: spilled bytes >= 1");

    cp::CPOptions o;
    std::vector<std::pair<op_id_t, op_id_t>> edges = {{1, 2}, {2, 3}, {3, 4}, {2, 5}, {4, 5}};
    cp::CPProblem p = cp::build_problem_from_requests(reqs, edges, 3, {1, 2, 3, 4, 5}, o);
    check(p.M == 3, "oracle: budget wired to M");
    check(p.W >= 5, "oracle: W >= sum(w) (slack rule)");
    cp::CPProblem keep = p;
    cp::CPSolver solver(std::move(p), o);
    cp::CPSolution r = solver.solve();
    check(r.feasible, "oracle: cp feasible");
    check(r.peak_resident <= 3, "oracle: cp peak <= M (always)");
    check(r.ddr_bytes <= 2 * greedy_spilled_m,
          "oracle: cp DDR <= greedy round-trip cost (2 * spilled bytes)");
    check(r.ddr_bytes == 2, "oracle: cp finds the DDR=2 optimum from requests");
}

// ── 6. bridge smoke: alloc results + spill/fill plans ──
static void test_bridge() {
    std::vector<fa::FancyAllocator::AllocRequest> reqs = {
        {1, 2, 0, 1}, {2, 1, 1, 4}, {3, 2, 2, 3}, {4, 1, 3, 4}, {5, 1, 4, 4},
    };
    std::vector<std::pair<op_id_t, op_id_t>> edges = {{1, 2}, {2, 3}, {3, 4}, {2, 5}, {4, 5}};
    std::vector<op_id_t> topo = {1, 2, 3, 4, 5};
    cp::CPOptions o;
    cp::CPProblem p = cp::build_problem_from_requests(reqs, edges, 3, topo, o);
    cp::CPProblem keep = p;
    cp::CPSolver solver(std::move(p), o);
    cp::CPSolution r = solver.solve();
    check(r.feasible && r.optimal, "bridge: solved optimal");
    check(r.placement_degraded == 0, "bridge: first-fit placed every segment");

    auto allocs = cp::to_alloc_results(r, keep, 3);
    check(allocs.size() == 5, "bridge: one AllocResult per request");
    uint64_t max_end = 0;
    for (const auto& kv : allocs) {
        check(!kv.second.spilled, "bridge: no degraded placement");
        max_end = std::max(max_end, kv.second.offset + 1);
    }
    check(max_end <= 3, "bridge: offsets fit the budget");
    // co-live tensors must not share offset ranges
    {
        struct Iv {
            uint32_t s, e;
            uint64_t off, sz;
        };
        std::vector<Iv> ivs;
        for (uint32_t v = 0; v < keep.n(); v++) {
            const auto& s0 = r.segments[v][0];
            const auto& s1 = r.segments[v][1];
            uint32_t topo_pos = v; // topo_hint == index order in this fixture
            ivs.push_back({s0.s, s0.e, s0.vtcm_offset, keep.nodes[v].m});
            if (s1.active)
                ivs.push_back({s1.s, s1.e, s1.vtcm_offset, keep.nodes[v].m});
            (void)topo_pos;
        }
        bool clash = false;
        for (size_t i = 0; i < ivs.size() && !clash; i++)
            for (size_t j = i + 1; j < ivs.size() && !clash; j++) {
                const bool time_ov = !(ivs[i].e < ivs[j].s || ivs[j].e < ivs[i].s);
                const bool space_ov =
                    !(ivs[i].off + ivs[i].sz <= ivs[j].off || ivs[j].off + ivs[j].sz <= ivs[i].off);
                if (time_ov && space_ov) clash = true;
            }
        check(!clash, "bridge: no time+space overlap among placed segments");
    }

    auto plans = cp::to_spill_fill_plans(r, keep);
    check(plans.size() == 1, "bridge: exactly one spill/fill plan (Eq.14)");
    if (plans.size() == 1) {
        const auto& pl = plans[0];
        check(pl.op_id == 2, "bridge: plan targets v2");
        check(pl.spill_position == 2, "bridge: spill after last seg0 consumer (v3, topo 2)");
        check(pl.fill_position == 4, "bridge: fill before first seg1 consumer (v5, topo 4)");
        check(pl.fill_position > pl.spill_position, "bridge: fill after spill");
        check(pl.size == 1, "bridge: size == m(v2)");
    }
}

// ── 7. determinism + greedy fallback + empty ──
static void test_misc() {
    // determinism: identical runs -> identical solutions
    cp::CPProblem p1 = fig6(2), p2 = fig6(2);
    cp::CPSolver s1(std::move(p1), cp::CPOptions{}), s2(std::move(p2), cp::CPOptions{});
    auto r1 = s1.solve(), r2 = s2.solve();
    check(r1.ddr_bytes == r2.ddr_bytes && r1.seq.size() == r2.seq.size(), "misc: deterministic");
    bool same = r1.seq.size() == r2.seq.size();
    for (size_t i = 0; same && i < r1.seq.size(); i++)
        same = r1.seq[i].kind == r2.seq[i].kind && r1.seq[i].v == r2.seq[i].v &&
               r1.seq[i].slot == r2.seq[i].slot;
    check(same, "misc: identical task sequences across runs");

    // greedy fallback: max_exact_nodes=0 forces the degenerate path
    cp::CPOptions o;
    o.max_exact_nodes = 0;
    cp::CPProblem p3 = fig6(2);
    cp::CPProblem keep = p3;
    cp::CPSolver s3(std::move(p3), o);
    auto r3 = s3.solve();
    check(r3.feasible, "misc: greedy feasible on fig6b");
    check(!r3.optimal, "misc: greedy never claims optimal");
    check(r3.abort_reason.find("exact-size-limit") != std::string::npos,
          "misc: greedy carries the skip reason");
    check(r3.ddr_bytes == 2, "misc: greedy reaches DDR=2 on fig6b");
    std::string why;
    check(cp::CPSolver::verify(keep, r3, &why), "misc: greedy solution verifies");

    // empty problem
    cp::CPProblem empty;
    cp::CPSolver s4(std::move(empty), cp::CPOptions{});
    auto r4 = s4.solve();
    check(r4.feasible && r4.optimal, "misc: empty problem trivially optimal");
}

// ── 8. env parsing ──
static void test_env() {
    cp::CPOptions o;
    unsetenv("HNNX_VTCM_ALLOCATOR");
    check(!cp::parse_cp_options_from_env(&o), "env: default (unset) stays on fancy");
    setenv("HNNX_VTCM_ALLOCATOR", "fancy", 1);
    check(!cp::parse_cp_options_from_env(&o), "env: fancy stays greedy");
    setenv("HNNX_VTCM_ALLOCATOR", "cp", 1);
    check(cp::parse_cp_options_from_env(&o) && o.mode == cp::CPOptions::Mode::FULL &&
              o.respect_topo_order,
          "env: cp -> FULL, topo respected");
    setenv("HNNX_VTCM_ALLOCATOR", "cp-reorder", 1);
    check(cp::parse_cp_options_from_env(&o) && !o.respect_topo_order, "env: cp-reorder frees Eq.8");
    setenv("HNNX_VTCM_ALLOCATOR", "cp-paging", 1);
    setenv("HNNX_CP_MAX_NODES", "7", 1);
    setenv("HNNX_CP_NODE_LIMIT", "12345", 1);
    check(cp::parse_cp_options_from_env(&o) && o.mode == cp::CPOptions::Mode::PAGING_ONLY &&
              o.max_exact_nodes == 7 && o.node_limit == 12345,
          "env: knobs parsed");
    setenv("HNNX_VTCM_ALLOCATOR", "garbage", 1);
    check(!cp::parse_cp_options_from_env(&o), "env: unknown selector falls back to fancy");
    unsetenv("HNNX_VTCM_ALLOCATOR");
    unsetenv("HNNX_CP_MAX_NODES");
    unsetenv("HNNX_CP_NODE_LIMIT");
}

int main() {
    test_fig6_b();
    test_fig6_a();
    run_invariants();
    test_knobs();
    test_oracle_ab();
    test_bridge();
    test_misc();
    test_env();
    std::printf("test_cp_solver: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
