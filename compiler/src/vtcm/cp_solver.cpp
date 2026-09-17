// @file cp_solver.cpp — implementation of hnnx::cp (formal model in cp_solver.hpp).
//
// Layout:
//   §1 small helpers
//   §2 CPSolver::Impl — preprocessing, DFS branch-and-bound (exact), extraction
//   §3 degenerate greedy fallback (n > max_exact_nodes, or guardrail abort
//       without incumbent)
//   §4 CPSolver public API + verify()
//   §5 bridging adapters + env parsing
//
// Search structure (one recursion per boundary; tasks occupy slots 1,2,3,...):
//   dfs(boundary t)   [tasks 1..t placed]
//     A. forced closes — open dead tensors (no future consumption possible)
//     B. canonical close branching — LIVE0 tensors consumed at slot t may
//        close (discard -> CLOSED_FREE | spill -> CLOSED_DDR | keep open)
//     C. task at slot t+1 — COMPUTE / REMAT / PAGE_IN in deterministic order;
//        a task that overflows capacity branches over eviction victims
//        (early close). Evictions only ever happen at an overflow boundary:
//        an eviction elsewhere can be delayed to the first subsequent
//        overflow without changing cost or feasibility, so this restriction
//        covers at least one optimal solution (dominance).
//
// Why closes elsewhere are not branched: by dominance a segment ends at the
// slot of the last consumption it serves, EXCEPT when capacity forces an
// early eviction — and that is exactly the overflow branch.

#include "hnnx/vtcm/cp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hnnx {
namespace cp {
namespace {

constexpr uint64_t kAlign = 128;          // VTCM placement alignment (fancy path default)
constexpr uint64_t kUnplaced = ~0ull;     // sentinel: first-fit failed
constexpr uint64_t kBytesPerW = 2048;     // BYTES cost-model granularity
constexpr uint64_t kNodeTimeCheck = 4096; // time-limit check cadence

uint64_t align_up(uint64_t x, uint64_t a) {
    return (x + a - 1) / a * a;
}

bool env_u64(const char* name, uint64_t* out) {
    const char* s = std::getenv(name);
    if (!s || !*s) return false;
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 0);
    if (errno != 0 || end == s) return false;
    *out = v;
    return true;
}

bool env_double(const char* name, double* out) {
    const char* s = std::getenv(name);
    if (!s || !*s) return false;
    errno = 0;
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (errno != 0 || end == s || v < 0.0) return false;
    *out = v;
    return true;
}

} // namespace

// ── §2 CPSolver::Impl ─────────────────────────────────────────────────────

struct CPSolver::Impl {
    CPProblem p;
    CPOptions o;
    CPOptions::Objective obj = CPOptions::Objective::DDR_BYTES;

    // idx-space preprocessing
    std::vector<uint64_t> m, w;
    std::vector<uint32_t> C;
    std::vector<std::vector<uint32_t>> preds, cons; // edge (u,v): v consumes u
    std::vector<uint32_t> outdeg;
    std::vector<int32_t> topo_rank; // idx -> rank (all nodes ranked)
    std::vector<uint32_t> topo_seq; // rank -> idx
    std::vector<uint8_t> may_serve_remat_v;
    uint64_t sum_w = 0;

    enum class St : uint8_t { UNCOMPUTED, LIVE0, CLOSED_DDR, CLOSED_FREE, LIVE1, DONE };
    std::vector<St> st;
    std::vector<uint32_t> unsched, last_cons, seg_start;
    // undo stack for last_cons saves (place/unplace of consumer tasks)
    std::vector<std::vector<uint32_t>> lc_undo;
    uint32_t ncomputed = 0, next_topo = 0;
    uint64_t resident = 0, spill_b = 0, fill_b = 0, compute_used = 0, computed_w = 0, peak = 0;
    uint32_t remats = 0, spills = 0;
    uint32_t t = 0; // boundary: tasks 1..t placed

    enum LK : uint8_t { L_COMPUTE, L_REMAT, L_FILL, L_SPILL, L_CLOSE };
    struct Ent {
        uint32_t slot; // tasks: the slot they occupy; closes: boundary slot
        LK k;
        uint32_t v;
        uint32_t e; // closes only: canonical segment end max(seg_start, last_cons)
    };
    std::vector<Ent> log;

    bool have_inc = false;
    uint64_t inc_k0 = 0, inc_k1 = 0, inc_k2 = 0;
    std::vector<Ent> inc_log;
    uint64_t inc_ddr = 0, inc_peak = 0, inc_compute = 0;
    uint32_t inc_remats = 0, inc_spills = 0, inc_slots = 0;

    uint64_t nodes = 0;
    bool aborted = false;
    std::string abort_reason;
    std::chrono::steady_clock::time_point t0;
    uint64_t frontier_lb = ~0ull;

    // ── preprocessing ──

    void preprocess() {
        const uint32_t n = p.n();
        m.resize(n);
        w.resize(n);
        C.resize(n);
        preds.assign(n, {});
        cons.assign(n, {});
        outdeg.assign(n, 0);
        topo_rank.assign(n, -1);
        may_serve_remat_v.assign(n, 0);
        sum_w = 0;
        for (uint32_t v = 0; v < n; v++) {
            m[v] = p.nodes[v].m;
            w[v] = p.nodes[v].w ? p.nodes[v].w : 1;
            C[v] = p.nodes[v].C ? p.nodes[v].C : 1;
            if (p.nodes[v].no_remat) C[v] = 1;
            if (o.mode == CPOptions::Mode::SEQ_ONLY) C[v] = 1;
            sum_w += w[v];
        }
        for (const auto& ed : p.edges) {
            if (ed.first >= n || ed.second >= n || ed.first == ed.second) continue;
            preds[ed.second].push_back(ed.first);
            cons[ed.first].push_back(ed.second);
        }
        for (uint32_t v = 0; v < n; v++) {
            auto& ps = preds[v];
            std::sort(ps.begin(), ps.end());
            ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
            auto& cs = cons[v];
            std::sort(cs.begin(), cs.end());
            cs.erase(std::unique(cs.begin(), cs.end()), cs.end());
            outdeg[v] = static_cast<uint32_t>(cs.size());
        }
        // topo ranks from hint; unranked nodes appended in index order
        topo_seq.reserve(n);
        std::vector<uint8_t> ranked(n, 0);
        for (const op_id_t id : p.topo_hint)
            for (uint32_t v = 0; v < n; v++)
                if (!ranked[v] && p.nodes[v].id == id) {
                    ranked[v] = 1;
                    topo_rank[v] = static_cast<int32_t>(topo_seq.size());
                    topo_seq.push_back(v);
                }
        for (uint32_t v = 0; v < n; v++)
            if (!ranked[v]) {
                topo_rank[v] = static_cast<int32_t>(topo_seq.size());
                topo_seq.push_back(v);
            }
        if (remat_allowed_mode(o.mode)) {
            for (uint32_t u = 0; u < n; u++)
                for (uint32_t c : cons[u])
                    if (C[c] >= 2) {
                        may_serve_remat_v[u] = 1;
                        break;
                    }
        }
        obj = o.mode == CPOptions::Mode::REMAT_ONLY ? CPOptions::Objective::TOTAL_COMPUTE
             : o.mode == CPOptions::Mode::SEQ_ONLY  ? CPOptions::Objective::PEAK_RESIDENT
                                                    : CPOptions::Objective::DDR_BYTES;
    }

    static bool remat_allowed_mode(CPOptions::Mode md) {
        return md == CPOptions::Mode::FULL || md == CPOptions::Mode::REMAT_ONLY;
    }
    static bool paging_allowed_mode(CPOptions::Mode md) {
        return md == CPOptions::Mode::FULL || md == CPOptions::Mode::PAGING_ONLY;
    }
    bool remat_allowed() const { return remat_allowed_mode(o.mode); }
    bool paging_allowed() const { return paging_allowed_mode(o.mode); }
    bool future_need(uint32_t u) const { return unsched[u] > 0 || may_serve_remat_v[u] != 0; }
    bool preds_open(uint32_t v) const {
        for (uint32_t u : preds[v])
            if (st[u] != St::LIVE0 && st[u] != St::LIVE1) return false;
        return true;
    }

    // ── dead-branch detection + lower bounds (O(n) scans; n <= 50 exact) ──

    bool dead_state() const {
        if (!remat_allowed()) {
            // a discard-closed tensor with future consumers can never reopen
            for (uint32_t u = 0, n = p.n(); u < n; u++)
                if (st[u] == St::CLOSED_FREE && (unsched[u] > 0 || may_serve_remat_v[u])) return true;
        }
        if (o.mode != CPOptions::Mode::SEQ_ONLY) {
            // compute floor: every node still computes + inevitable reopens.
            // LB soundness: only unsched[u] > 0 is an *unconditional* need —
            // may_serve_remat_v is conditional on a consumer actually rematting,
            // so counting it here can prune the true optimum (C=2 everywhere).
            uint64_t forced = sum_w - computed_w + compute_used;
            if (remat_allowed())
                for (uint32_t u = 0, n = p.n(); u < n; u++)
                    if (st[u] == St::CLOSED_FREE && unsched[u] > 0) forced += w[u];
            if (forced > p.W) return true;
        }
        return false;
    }

    uint64_t lb_objective() const {
        switch (obj) {
            case CPOptions::Objective::DDR_BYTES: {
                uint64_t lb = spill_b + fill_b;
                for (uint32_t u = 0, n = p.n(); u < n; u++)
                    if (st[u] == St::CLOSED_DDR && unsched[u] > 0) lb += m[u]; // fill inevitable
                return lb;
            }
            case CPOptions::Objective::TOTAL_COMPUTE: {
                uint64_t lb = sum_w - computed_w + compute_used;
                for (uint32_t u = 0, n = p.n(); u < n; u++)
                    if (st[u] == St::CLOSED_FREE && unsched[u] > 0) lb += w[u];
                return lb;
            }
            default:
                return peak > resident ? peak : resident;
        }
    }

    // Lower bound on the lexicographic tie-break key k1 (see leaf()): needed so
    // the prune below cannot kill equal-k0 branches that would still improve
    // k1 (e.g. the same DDR=0 with fewer remat compute cycles).
    uint64_t lb_tiebreak() const {
        switch (obj) {
            case CPOptions::Objective::DDR_BYTES: { // k1 = compute
                uint64_t lb = sum_w - computed_w + compute_used;
                if (remat_allowed())
                    for (uint32_t u = 0, n = p.n(); u < n; u++)
                        if (st[u] == St::CLOSED_FREE && unsched[u] > 0) lb += w[u];
                return lb;
            }
            case CPOptions::Objective::TOTAL_COMPUTE: { // k1 = ddr
                uint64_t lb = spill_b + fill_b;
                for (uint32_t u = 0, n = p.n(); u < n; u++)
                    if (st[u] == St::CLOSED_DDR && unsched[u] > 0) lb += m[u];
                return lb;
            }
            default:
                return t; // k1 = slot count
        }
    }

    // prune iff this subtree cannot strictly improve the incumbent
    // lexicographically (k0, k1); k2 (slots) is not bound — accepted loss.
    bool prune_by_bound() const {
        if (!have_inc) return false;
        const uint64_t lb0 = lb_objective();
        if (lb0 != inc_k0) return lb0 > inc_k0;
        return lb_tiebreak() >= inc_k1;
    }

    void note_frontier() {
        if (!aborted) return;
        const uint64_t lb = lb_objective();
        if (lb < frontier_lb) frontier_lb = lb;
    }

    bool time_over() {
        if (aborted) return true;
        if ((nodes & (kNodeTimeCheck - 1)) != 0) return false;
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - t0).count() * 1000.0 >= o.time_limit_ms) {
            aborted = true;
            if (abort_reason.empty()) abort_reason = "timeout";
            return true;
        }
        return false;
    }

    // ── state transitions (all logged; undo mirrors exactly) ──

    void close_l0(uint32_t u, bool to_ddr) {
        // segment end = the boundary it is closed at: the tensor truly
        // occupied VTCM through slot t (evictions included)
        resident -= m[u];
        st[u] = to_ddr ? St::CLOSED_DDR : St::CLOSED_FREE;
        if (to_ddr) {
            spill_b += m[u];
            spills++;
            log.push_back({t, L_SPILL, u, t});
        } else {
            log.push_back({t, L_CLOSE, u, t});
        }
    }
    void reopen_l0(uint32_t u, bool was_ddr) {
        resident += m[u];
        st[u] = St::LIVE0;
        if (was_ddr) {
            spill_b -= m[u];
            spills--;
        }
        log.pop_back();
    }

    struct Cand {
        uint8_t kind; // 0 COMPUTE, 1 REMAT, 2 PAGE_IN
        uint32_t v;
        int32_t rnk;
        bool operator<(const Cand& other) const {
            if (rnk != other.rnk) return rnk < other.rnk;
            if (kind != other.kind) return kind < other.kind;
            return v < other.v;
        }
    };

    void place(const Cand& c, uint32_t slot) {
        const uint32_t v = c.v;
        const uint64_t mv = m[v];
        switch (c.kind) {
            case 0: { // COMPUTE
                st[v] = St::LIVE0;
                seg_start[v] = slot;
                resident += mv;
                compute_used += w[v];
                computed_w += w[v];
                ncomputed++;
                if (o.respect_topo_order) next_topo++;
                std::vector<uint32_t> undo;
                for (uint32_t u : preds[v]) {
                    undo.push_back(last_cons[u]);
                    unsched[u]--;
                    last_cons[u] = slot;
                }
                lc_undo.push_back(std::move(undo));
                log.push_back({slot, L_COMPUTE, v, 0});
                break;
            }
            case 1: { // REMAT (reopens a discard-closed tensor)
                st[v] = St::LIVE1;
                seg_start[v] = slot;
                resident += mv;
                compute_used += w[v];
                remats++;
                std::vector<uint32_t> undo;
                for (uint32_t u : preds[v]) {
                    undo.push_back(last_cons[u]);
                    last_cons[u] = slot;
                }
                lc_undo.push_back(std::move(undo));
                log.push_back({slot, L_REMAT, v, 0});
                break;
            }
            default: { // PAGE_IN (reopens a spill-closed tensor)
                st[v] = St::LIVE1;
                seg_start[v] = slot;
                resident += mv;
                fill_b += mv;
                lc_undo.push_back({});
                log.push_back({slot, L_FILL, v, 0});
                break;
            }
        }
        if (resident > peak) peak = resident;
    }

    void unplace(const Cand& c, uint32_t slot) {
        const uint32_t v = c.v;
        const uint64_t mv = m[v];
        switch (c.kind) {
            case 0:
                for (size_t i = 0; i < preds[v].size(); i++) {
                    unsched[preds[v][i]]++;
                    last_cons[preds[v][i]] = lc_undo.back()[i];
                }
                lc_undo.pop_back();
                if (o.respect_topo_order) next_topo--;
                ncomputed--;
                computed_w -= w[v];
                compute_used -= w[v];
                resident -= mv;
                st[v] = St::UNCOMPUTED;
                break;
            case 1:
                for (size_t i = 0; i < preds[v].size(); i++)
                    last_cons[preds[v][i]] = lc_undo.back()[i];
                lc_undo.pop_back();
                remats--;
                compute_used -= w[v];
                resident -= mv;
                st[v] = St::CLOSED_FREE;
                break;
            default:
                lc_undo.pop_back();
                fill_b -= mv;
                resident -= mv;
                st[v] = St::CLOSED_DDR;
                break;
        }
        log.pop_back();
    }

    // ── exact DFS ──

    void dfs() {
        if (aborted) {
            note_frontier();
            return;
        }
        nodes++;
        if (nodes > o.node_limit) {
            aborted = true;
            if (abort_reason.empty()) abort_reason = "node-limit";
            note_frontier();
            return;
        }
        if (time_over()) {
            note_frontier();
            return;
        }

        // A. forced closes of dead open tensors (dominance: close ASAP).
        //    Recorded with pre-close state so undo is exact (LIVE0 vs LIVE1).
        std::vector<std::pair<uint32_t, St>> closed_dead;
        for (uint32_t u = 0, n = p.n(); u < n; u++) {
            if ((st[u] == St::LIVE0 || st[u] == St::LIVE1) && !future_need(u)) {
                closed_dead.emplace_back(u, st[u]);
                resident -= m[u];
                st[u] = St::DONE;
                log.push_back({t, L_CLOSE, u, t});
            }
        }

        if (ncomputed == p.n()) {
            leaf();
        } else if (!dead_state() && !prune_by_bound()) {
            // B. canonical close branching over tensors consumed at slot t.
            //    LIVE1 (reopened) tensors with no remaining consumers MUST also
            //    get a close branch here — otherwise they squat residency until
            //    the leaf (search residency > extracted residency, so the
            //    search can falsely declare capacity infeasible).
            std::vector<uint32_t> cands;
            if (t > 0) {
                for (uint32_t u = 0, n = p.n(); u < n; u++) {
                    const bool live0 = st[u] == St::LIVE0;
                    const bool live1_done = st[u] == St::LIVE1 && unsched[u] == 0;
                    if ((live0 || live1_done) && last_cons[u] == t) cands.push_back(u);
                }
                std::sort(cands.begin(), cands.end(), [this](uint32_t a, uint32_t b) {
                    if (topo_rank[a] != topo_rank[b]) return topo_rank[a] < topo_rank[b];
                    return a < b;
                });
            }
            close_rec(cands, 0);
        }

        for (auto it = closed_dead.rbegin(); it != closed_dead.rend(); ++it) {
            resident += m[it->first];
            st[it->first] = it->second;
            log.pop_back();
        }
    }

    void close_rec(const std::vector<uint32_t>& cands, size_t i) {
        if (aborted) {
            note_frontier();
            return;
        }
        nodes++;
        if (i == cands.size()) {
            task_phase();
            return;
        }
        const uint32_t u = cands[i];
        if (st[u] == St::LIVE1) {
            // both retention segments spent: finalize (frees residency, no
            // reopen possible) or keep (only useful to feed a later REMAT of
            // a consumer — may_serve_remat_v gated by the caller).
            resident -= m[u];
            st[u] = St::DONE;
            log.push_back({t, L_CLOSE, u, t});
            close_rec(cands, i + 1);
            log.pop_back();
            st[u] = St::LIVE1;
            resident += m[u];
            if (aborted) return;
            close_rec(cands, i + 1); // keep
            return;
        }
        // discard close (reopen only via REMAT)
        if (remat_allowed()) {
            close_l0(u, false);
            close_rec(cands, i + 1);
            reopen_l0(u, false);
            if (aborted) return;
        }
        // keep open
        close_rec(cands, i + 1);
        if (aborted) return;
        // spill close (reopen only via PAGE_IN)
        if (paging_allowed() && C[u] >= 2 && future_need(u)) {
            close_l0(u, true);
            close_rec(cands, i + 1);
            reopen_l0(u, true);
        }
    }

    void task_phase() {
        if (aborted) {
            note_frontier();
            return;
        }
        nodes++;
        const uint32_t slot = t + 1;
        std::vector<Cand> cs;
        for (uint32_t v = 0, n = p.n(); v < n; v++) {
            const int32_t r = topo_rank[v];
            if (st[v] == St::UNCOMPUTED) {
                if (o.respect_topo_order &&
                    (next_topo >= topo_seq.size() || topo_seq[next_topo] != v))
                    continue;
                if (!preds_open(v)) continue;
                if (compute_used + w[v] > p.W) continue;
                cs.push_back({0, v, r});
            } else if (st[v] == St::CLOSED_FREE) {
                if (!remat_allowed() || C[v] < 2 || !future_need(v)) continue;
                if (!preds_open(v)) continue;
                if (compute_used + w[v] > p.W) continue;
                cs.push_back({1, v, r});
            } else if (st[v] == St::CLOSED_DDR) {
                if (!paging_allowed() || C[v] < 2 || !future_need(v)) continue;
                cs.push_back({2, v, r});
            }
        }
        std::sort(cs.begin(), cs.end());
        static const bool cp_trace = getenv("HNNX_CP_TRACE") != nullptr;
        if (cp_trace) {
            std::fprintf(stderr, "task_phase t=%u slot=%u ncomp=%u res=%llu/%llu cs:",
                         t, slot, ncomputed, (unsigned long long)resident,
                         (unsigned long long)p.M);
            for (const Cand& c : cs)
                std::fprintf(stderr, " %u/k%u", c.v, c.kind);
            std::fprintf(stderr, " st:");
            for (uint32_t v = 0, n = p.n(); v < n; v++)
                std::fprintf(stderr, " %u:%u", v, (unsigned)st[v]);
            std::fprintf(stderr, "\n");
        }
        for (const Cand& c : cs) {
            if (resident + m[c.v] <= p.M) {
                place(c, slot);
                const uint32_t saved_t = t;
                t = slot;
                dfs();
                t = saved_t;
                unplace(c, slot);
            } else {
                evict_and_place(c, slot);
            }
            if (aborted) {
                note_frontier();
                return;
            }
        }
    }

    void evict_and_place(const Cand& c, uint32_t slot) {
        nodes++;
        // victims: open LIVE0 tensors, not consumed at this boundary (their
        // close was a Phase-B branch), not preds of c (must stay open at slot);
        // plus fully-consumed LIVE1 zombies (finalize frees them outright)
        std::vector<uint32_t> vict;
        for (uint32_t u = 0, n = p.n(); u < n; u++) {
            if (st[u] == St::LIVE1) {
                if (unsched[u] != 0) continue; // segments spent: not reopenable
                if (std::find(preds[c.v].begin(), preds[c.v].end(), u) != preds[c.v].end())
                    continue;
                vict.push_back(u);
                continue;
            }
            if (st[u] != St::LIVE0 || last_cons[u] == t) continue;
            if (std::find(preds[c.v].begin(), preds[c.v].end(), u) != preds[c.v].end()) continue;
            const bool can_discard = remat_allowed() && C[u] >= 2;
            const bool can_spill = paging_allowed() && C[u] >= 2 && future_need(u);
            if (can_discard || can_spill) vict.push_back(u);
        }
        std::sort(vict.begin(), vict.end(), [this](uint32_t a, uint32_t b) {
            if (m[a] != m[b]) return m[a] > m[b]; // largest frees most
            if (topo_rank[a] != topo_rank[b]) return topo_rank[a] < topo_rank[b];
            return a < b;
        });
        evict_rec(c, slot, vict, 0);
    }

    void evict_rec(const Cand& c, uint32_t slot, const std::vector<uint32_t>& vict, size_t i) {
        if (aborted) {
            note_frontier();
            return;
        }
        nodes++;
        if (resident + m[c.v] <= p.M) {
            place(c, slot);
            const uint32_t saved_t = t;
            t = slot;
            dfs();
            t = saved_t;
            unplace(c, slot);
            return;
        }
        if (i == vict.size()) return;
        const uint32_t u = vict[i];
        if (st[u] == St::LIVE1) { // finalize zombie (no reopen possible)
            resident -= m[u];
            st[u] = St::DONE;
            log.push_back({t, L_CLOSE, u, t});
            evict_rec(c, slot, vict, i + 1);
            log.pop_back();
            st[u] = St::LIVE1;
            resident += m[u];
            if (aborted) return;
            evict_rec(c, slot, vict, i + 1); // skip u
            return;
        }
        if (remat_allowed() && C[u] >= 2) { // discard evict (DDR-free)
            close_l0(u, false);
            evict_rec(c, slot, vict, i + 1);
            reopen_l0(u, false);
            if (aborted) return;
        }
        if (paging_allowed() && C[u] >= 2 && future_need(u)) { // spill evict
            close_l0(u, true);
            evict_rec(c, slot, vict, i + 1);
            reopen_l0(u, true);
            if (aborted) return;
        }
        evict_rec(c, slot, vict, i + 1); // skip u
    }

    void leaf() {
        // close all remaining open segments (canonical discard — nothing can
        // consume them after the last COMPUTE)
        std::vector<std::pair<uint32_t, St>> opened;
        for (uint32_t u = 0, n = p.n(); u < n; u++)
            if (st[u] == St::LIVE0 || st[u] == St::LIVE1) {
                opened.emplace_back(u, st[u]);
                resident -= m[u];
                st[u] = St::DONE;
                log.push_back({t, L_CLOSE, u, t});
            }
        const uint64_t ddr = spill_b + fill_b;
        uint64_t k0, k1, k2;
        switch (obj) {
            case CPOptions::Objective::TOTAL_COMPUTE: k0 = compute_used; k1 = ddr; k2 = t; break;
            case CPOptions::Objective::PEAK_RESIDENT: k0 = peak; k1 = t; k2 = ddr; break;
            default: k0 = ddr; k1 = compute_used; k2 = t; break;
        }
        if (!have_inc || k0 < inc_k0 || (k0 == inc_k0 && (k1 < inc_k1 || (k1 == inc_k1 && k2 < inc_k2)))) {
            have_inc = true;
            inc_k0 = k0;
            inc_k1 = k1;
            inc_k2 = k2;
            inc_log = log;
            inc_ddr = ddr;
            inc_peak = peak;
            inc_compute = compute_used;
            inc_remats = remats;
            inc_spills = spills;
            inc_slots = t;
        }
        for (auto it = opened.rbegin(); it != opened.rend(); ++it) {
            resident += m[it->first];
            st[it->first] = it->second;
            log.pop_back();
        }
    }

    // ── extraction: event log -> CPSolution (shared by exact + greedy) ──

    CPSolution extract_from(const std::vector<Ent>& lg) {
        const uint32_t n = p.n();
        CPSolution s;
        s.segments.assign(n, {});
        s.feasible = true;
        std::vector<uint8_t> open_seg(n, 255);
        for (const Ent& e : lg) {
            switch (e.k) {
                case L_COMPUTE:
                    s.segments[e.v][0].s = e.slot;
                    open_seg[e.v] = 0;
                    break;
                case L_REMAT:
                case L_FILL:
                    s.segments[e.v][1].s = e.slot;
                    s.segments[e.v][1].fill_start = (e.k == L_FILL);
                    s.segments[e.v][1].active = true;
                    open_seg[e.v] = 1;
                    break;
                case L_SPILL: {
                    CPSegment& sg = s.segments[e.v][open_seg[e.v]];
                    sg.e = e.e;
                    sg.spill_end = true;
                    open_seg[e.v] = 255;
                    break;
                }
                case L_CLOSE:
                    s.segments[e.v][open_seg[e.v]].e = e.e;
                    open_seg[e.v] = 255;
                    break;
            }
        }
        for (uint32_t v = 0; v < n; v++) s.segments[v][0].active = true;

        s.seq.clear();
        s.ddr_bytes = 0;
        s.total_compute = 0;
        s.remat_count = 0;
        s.spill_count = 0;
        for (const Ent& e : lg) {
            CPTask::Kind k;
            switch (e.k) {
                case L_COMPUTE: k = CPTask::Kind::COMPUTE; break;
                case L_REMAT:   k = CPTask::Kind::REMAT; break;
                case L_FILL:    k = CPTask::Kind::PAGE_IN; break;
                case L_SPILL:   k = CPTask::Kind::PAGE_OUT; break;
                default: continue;
            }
            CPTask task;
            task.kind = k;
            task.v = p.nodes[e.v].id;
            task.bytes = m[e.v];
            task.slot = e.slot;
            s.seq.push_back(task);
            if (e.k == L_SPILL || e.k == L_FILL) s.ddr_bytes += m[e.v];
            if (e.k == L_COMPUTE || e.k == L_REMAT) s.total_compute += w[e.v];
            if (e.k == L_REMAT) s.remat_count++;
            if (e.k == L_SPILL) s.spill_count++;
        }

        place_offsets(s, lg);
        s.peak_resident = recompute_peak(s);
        s.nodes_explored = nodes;
        return s;
    }

    // first-fit VTCM placement over the interval set + DDR arena bump.
    // A second segment first tries its tensor's first-segment offset (free by
    // Eq.5 ordering). Failed placement -> offset sentinel + degraded counter.
    // Alignment is 128 (fancy-path default) shrunk to fit tiny unit-test
    // budgets (down to 1 for M=3 fixtures with byte-sized tensors).
    void place_offsets(CPSolution& s, const std::vector<Ent>& lg) {
        uint64_t a = kAlign;
        // shrink while the alignment either exceeds M or inflates any tensor
        // (real-path sizes are 128-multiples -> a stays at kAlign; byte-sized
        // unit fixtures fall through to 1 so first-fit matches the Eq.6 proof)
        auto inflates = [&](uint64_t al) {
            for (uint32_t v = 0, n = p.n(); v < n; v++)
                if (m[v] % al != 0) return true;
            return false;
        };
        while (a > 1 && (a > p.M || inflates(a))) a >>= 1;
        struct P {
            uint32_t v, idx;
        };
        std::vector<P> order;
        for (uint32_t v = 0, n = p.n(); v < n; v++)
            for (uint32_t i = 0; i < 2; i++)
                if (s.segments[v][i].active) order.push_back({v, i});
        std::sort(order.begin(), order.end(), [this, &s](const P& a2, const P& b) {
            const CPSegment& A = s.segments[a2.v][a2.idx];
            const CPSegment& B = s.segments[b.v][b.idx];
            if (A.s != B.s) return A.s < B.s;
            if (m[a2.v] != m[b.v]) return m[a2.v] > m[b.v];
            return a2.v < b.v;
        });
        struct Placed {
            uint32_t s, e;
            uint64_t off, sz;
        };
        std::vector<Placed> done;
        for (const P& seg : order) {
            CPSegment& S = s.segments[seg.v][seg.idx];
            S.vtcm_offset = kUnplaced;
            std::vector<uint64_t> cands{0};
            if (seg.idx == 1 && s.segments[seg.v][0].vtcm_offset != kUnplaced)
                cands.push_back(s.segments[seg.v][0].vtcm_offset);
            for (const Placed& q : done) {
                cands.push_back(q.off);
                cands.push_back(align_up(q.off + q.sz, a));
            }
            std::sort(cands.begin(), cands.end());
            cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
            const uint64_t sz = align_up(m[seg.v], a);
            for (uint64_t off : cands) {
                if (off + sz > p.M) continue;
                bool clash = false;
                for (const Placed& q : done) {
                    const bool time_ov = !(S.e < q.s || q.e < S.s);
                    const bool space_ov = !(off + sz <= q.off || q.off + q.sz <= off);
                    if (time_ov && space_ov) {
                        clash = true;
                        break;
                    }
                }
                if (!clash) {
                    S.vtcm_offset = off;
                    break;
                }
            }
            if (S.vtcm_offset == kUnplaced) {
                s.placement_degraded++;
            } else {
                done.push_back({S.s, S.e, S.vtcm_offset, sz});
                if (S.vtcm_offset + sz > s.vtcm_arena_used) s.vtcm_arena_used = S.vtcm_offset + sz;
            }
        }
        // DDR arena: spill events in log order (Eq.14 => single version, no
        // slot contention); the fill of the same tensor reuses the address.
        uint64_t cur = 0;
        std::vector<uint64_t> ddr_off(p.n(), kUnplaced);
        for (const Ent& e : lg) {
            if (e.k != L_SPILL) continue;
            ddr_off[e.v] = cur;
            cur += m[e.v];
            s.segments[e.v][0].ddr_offset = ddr_off[e.v];
            if (s.segments[e.v][1].active && s.segments[e.v][1].fill_start)
                s.segments[e.v][1].ddr_offset = ddr_off[e.v];
        }
        s.ddr_arena_used = cur;
    }

    uint64_t recompute_peak(const CPSolution& s) const {
        uint64_t pk = 0;
        const uint32_t max_slot = 2 * p.n() + 2;
        for (uint32_t slot = 1; slot <= max_slot; slot++) {
            uint64_t r = 0;
            for (uint32_t v = 0, n = p.n(); v < n; v++) {
                bool open = false;
                for (uint32_t i = 0; i < 2; i++) {
                    const CPSegment& S = s.segments[v][i];
                    if (S.active && S.s <= slot && slot <= S.e) open = true;
                }
                if (open) r += m[v];
            }
            if (r > pk) pk = r;
        }
        return pk;
    }

    // ── §3 degenerate greedy fallback ──
    //
    // Topo sweep: canonical minimal intervals, discard at last consumer,
    // largest-first spill eviction when the *next* slot's forced residency
    // would overflow (the FancyAllocator-flavored approximation). Produces an
    // event log and reuses extract_from(); optimal is always false and
    // abort_reason carries why the exact search was skipped.

    CPSolution solve_greedy(const char* why) {
        const uint32_t n = p.n();
        CPSolution bad;
        bad.feasible = false;
        bad.abort_reason = why;
        auto fail = [&](const char* extra) {
            bad.abort_reason = std::string(why) + "+" + extra;
            return bad;
        };

        std::vector<Ent> glog;
        std::vector<uint8_t> live(n, 0), paged(n, 0), spilled(n, 0);
        std::vector<uint32_t> last_rank(n, 0), crank(n, 0);
        for (uint32_t r = 0; r < n; r++) crank[topo_seq[r]] = r;
        for (const auto& ed : p.edges) {
            const uint32_t u = ed.first, v = ed.second;
            if (crank[v] > last_rank[u]) last_rank[u] = crank[v];
        }
        uint64_t res = 0, cu = 0;
        uint32_t sl = 0; // task-slot counter: Eq.8 — one task per slot, FILL included
        for (uint32_t r = 0; r < n; r++) {
            const uint32_t c = topo_seq[r];
            // reopen paged preds (each FILL takes its own slot)
            for (uint32_t u : preds[c]) {
                if (live[u]) continue;
                if (paged[u]) {
                    live[u] = 1;
                    paged[u] = 0;
                    res += m[u];
                    glog.push_back({++sl, L_FILL, u, 0});
                } else {
                    return fail("greedy-infeasible"); // discard-closed pred needed again
                }
            }
            // compute c — tensors consumed at slot sl stay resident through sl
            live[c] = 1;
            res += m[c];
            cu += w[c];
            if (cu > p.W) return fail("greedy-W");
            glog.push_back({++sl, L_COMPUTE, c, 0});
            if (res > p.M) return fail("greedy-infeasible"); // present slot unfixable
            // boundary sl: discard-close tensors whose last consumer was just computed
            for (uint32_t u = 0; u < n; u++) {
                if (live[u] && !paged[u] && last_rank[u] == r && crank[u] <= r) {
                    live[u] = 0;
                    res -= m[u];
                    glog.push_back({sl, L_CLOSE, u, sl});
                }
            }
            // one-step lookahead (topo rank r+1): forced residency
            if (r + 1 < n) {
                const uint32_t nxt = topo_seq[r + 1];
                uint64_t forced = res + m[nxt];
                for (uint32_t u : preds[nxt])
                    if (!live[u] && paged[u]) forced += m[u];
                if (forced > p.M) {
                    if (!paging_allowed()) return fail("greedy-infeasible");
                    std::vector<uint32_t> vict;
                    for (uint32_t u = 0; u < n; u++) {
                        if (!live[u] || spilled[u]) continue;    // Eq.14: spill once max
                        if (last_rank[u] <= r + 1) continue;      // consumed at r+1 or done
                        if (std::find(preds[nxt].begin(), preds[nxt].end(), u) != preds[nxt].end())
                            continue;
                        if (C[u] < 2) continue;
                        vict.push_back(u);
                    }
                    std::sort(vict.begin(), vict.end(),
                              [this](uint32_t a, uint32_t b) { return m[a] > m[b]; });
                    for (uint32_t u : vict) {
                        if (forced <= p.M) break;
                        live[u] = 0;
                        paged[u] = 1;
                        spilled[u] = 1;
                        forced -= m[u];
                        res -= m[u];
                        glog.push_back({sl, L_SPILL, u, sl});
                    }
                    if (forced > p.M) return fail("greedy-infeasible");
                }
            }
        }
        // close survivors at the final boundary
        for (uint32_t u = 0; u < n; u++)
            if (live[u]) glog.push_back({sl, L_CLOSE, u, sl});
        CPSolution s = extract_from(glog);
        s.optimal = false;
        s.abort_reason = why;
        s.nodes_explored = nodes;
        std::string vwhy;
        if (!CPSolver::verify(p, s, &vwhy)) {
            s.feasible = false;
            s.abort_reason += "+verify-failed: " + vwhy;
        }
        return s;
    }
};

// ── §4 CPSolver public ────────────────────────────────────────────────────

CPSolver::CPSolver(CPProblem problem, CPOptions options) : impl_(new Impl) {
    impl_->p = std::move(problem);
    impl_->o = options;
    impl_->preprocess();
}

CPSolver::~CPSolver() = default;

CPSolution CPSolver::solve() {
    Impl& I = *impl_;
    const uint32_t n = I.p.n();
    CPSolution out;
    if (n == 0) {
        out.feasible = true;
        out.optimal = true;
        return out;
    }
    if (I.p.W < I.sum_w) {
        out.abort_reason = "W below sum(w): no solution can compute every node";
        return out;
    }
    for (uint32_t v = 0; v < n; v++) {
        if (I.m[v] > I.p.M) {
            out.abort_reason = "tensor m exceeds M (can never be resident)";
            return out;
        }
    }

    if (n > I.o.max_exact_nodes) return I.solve_greedy("exact-size-limit");

    // init search state
    I.st.assign(n, Impl::St::UNCOMPUTED);
    I.unsched = I.outdeg;
    I.last_cons.assign(n, 0);
    I.seg_start.assign(n, 0);
    I.lc_undo.clear();
    I.ncomputed = 0;
    I.next_topo = 0;
    I.resident = I.spill_b = I.fill_b = I.compute_used = I.computed_w = I.peak = 0;
    I.remats = I.spills = 0;
    I.t = 0;
    I.log.clear();
    I.have_inc = false;
    I.nodes = 0;
    I.aborted = false;
    I.abort_reason.clear();
    I.frontier_lb = ~0ull;
    I.t0 = std::chrono::steady_clock::now();

    I.dfs();

    if (I.have_inc) {
        out = I.extract_from(I.inc_log);
        if (I.aborted) {
            out.optimal = false;
            out.abort_reason = I.abort_reason;
            out.bound_gap = 1.0;
            if (I.frontier_lb != ~0ull && I.frontier_lb < I.inc_k0 && I.inc_k0 > 0)
                out.bound_gap = static_cast<double>(I.inc_k0 - I.frontier_lb) /
                                static_cast<double>(I.inc_k0);
        } else {
            out.optimal = true;
            out.abort_reason.clear();
            out.bound_gap = 0.0;
        }
    } else if (I.aborted) {
        return I.solve_greedy(I.abort_reason.c_str());
    } else {
        out.abort_reason = "infeasible: search tree exhausted";
        return out;
    }

    // independent re-verification (Eq.4-14 + objective consistency)
    std::string why;
    if (!verify(I.p, out, &why)) {
        out.feasible = false;
        out.optimal = false;
        out.abort_reason = "verify-failed: " + why;
    }
    return out;
}

bool CPSolver::verify(const CPProblem& p, const CPSolution& s, std::string* why) {
    auto fail = [why](const char* msg) {
        if (why) *why = msg;
        return false;
    };
    const uint32_t n = p.n();
    if (s.segments.size() != n) return fail("segments size mismatch");
    if (!s.feasible) return fail("solution marked infeasible");

    // id -> idx + per-node predicates
    std::unordered_map<op_id_t, uint32_t> idx;
    std::vector<std::vector<uint32_t>> pr(n);
    std::vector<uint64_t> m(n), w(n);
    for (uint32_t v = 0; v < n; v++) {
        idx[p.nodes[v].id] = v;
        m[v] = p.nodes[v].m;
        w[v] = p.nodes[v].w ? p.nodes[v].w : 1;
    }
    for (const auto& ed : p.edges)
        if (ed.first < n && ed.second < n) pr[ed.second].push_back(ed.first);

    // task census (Eq.12: exactly one COMPUTE per node)
    std::vector<uint32_t> n_comp(n, 0), n_remat(n, 0), n_pin(n, 0), n_pout(n, 0);
    std::vector<uint32_t> comp_slot(n, 0), remat_slot(n, 0), pin_slot(n, 0), pout_slot(n, 0);
    uint32_t last_start = 0;
    uint64_t ddr = 0, compute = 0;
    for (const CPTask& t : s.seq) {
        auto it = idx.find(t.v);
        if (it == idx.end()) return fail("seq references unknown tensor id");
        const uint32_t v = it->second;
        switch (t.kind) {
            case CPTask::Kind::COMPUTE:
                if (++n_comp[v] > 1) return fail("Eq.12: more than one COMPUTE");
                comp_slot[v] = t.slot;
                compute += w[v];
                if (t.slot <= last_start) return fail("Eq.8: start slots not increasing");
                last_start = t.slot;
                break;
            case CPTask::Kind::REMAT:
                if (++n_remat[v] > 1) return fail("more than one REMAT (C_v=2 bound)");
                remat_slot[v] = t.slot;
                compute += w[v];
                if (t.slot <= last_start) return fail("Eq.8: start slots not increasing");
                last_start = t.slot;
                break;
            case CPTask::Kind::PAGE_IN:
                if (++n_pin[v] > 1) return fail("more than one PAGE_IN");
                pin_slot[v] = t.slot;
                ddr += m[v];
                if (t.slot <= last_start) return fail("Eq.8: start slots not increasing");
                last_start = t.slot;
                break;
            case CPTask::Kind::PAGE_OUT:
                if (++n_pout[v] > 1) return fail("Eq.14: more than one PAGE_OUT");
                pout_slot[v] = t.slot;
                ddr += m[v];
                if (t.slot < last_start) return fail("PAGE_OUT precedes current start");
                break;
        }
    }
    for (uint32_t v = 0; v < n; v++)
        if (n_comp[v] != 1) return fail("Eq.12: node not computed exactly once");

    // segment structure vs seq (Eq.4/5/9)
    for (uint32_t v = 0; v < n; v++) {
        const CPSegment& s0 = s.segments[v][0];
        const CPSegment& s1 = s.segments[v][1];
        if (!s0.active) return fail("first segment inactive (Eq.12)");
        if (s0.s != comp_slot[v]) return fail("seg0 start != COMPUTE slot");
        if (s0.e < s0.s) return fail("Eq.4: seg0 e < s");
        if (s0.spill_end != (n_pout[v] == 1)) return fail("seg0.spill_end vs PAGE_OUT mismatch");
        if (s1.active) {
            if (n_remat[v] + n_pin[v] != 1) return fail("seg1 active but no reopen task");
            if (s1.fill_start != (n_pin[v] == 1)) return fail("seg1.fill_start vs PAGE_IN mismatch");
            if (s1.s != (n_pin[v] ? pin_slot[v] : remat_slot[v])) return fail("seg1 start mismatch");
            if (s1.e < s1.s) return fail("Eq.4: seg1 e < s");
            if (s0.e > s1.s) return fail("Eq.5: seg0 overlaps seg1");
            if (s1.fill_start && !s0.spill_end) return fail("Eq.9: fill without prior spill");
            if (s1.fill_start && s1.ddr_offset != s0.ddr_offset)
                return fail("Eq.9: fill DDR address != spill address");
        } else if (n_remat[v] || n_pin[v]) {
            return fail("reopen task without active seg1");
        }
    }

    // Eq.7 + Eq.6 + peak
    uint64_t pk = 0;
    const uint32_t max_slot = 2 * n + 2;
    for (uint32_t slot = 1; slot <= max_slot; slot++) {
        uint64_t r = 0;
        for (uint32_t v = 0; v < n; v++)
            for (uint32_t i = 0; i < 2; i++) {
                const CPSegment& S = s.segments[v][i];
                if (S.active && S.s <= slot && slot <= S.e) r += m[v];
            }
        if (r > pk) pk = r;
    }
    if (pk > p.M) return fail("Eq.6: residency exceeds M");
    if (pk != s.peak_resident) return fail("peak_resident mismatch");
    // consumption coverage per edge (Eq.7; fill-opened segments are exempt
    // only at their own open — the reopened tensor's consumers below)
    for (uint32_t v = 0; v < n; v++) {
        for (uint32_t slot_i = 0; slot_i < 2; slot_i++) {
            const uint32_t slot = slot_i == 0 ? comp_slot[v] : remat_slot[v];
            if (slot_i == 1 && slot == 0) continue;
            for (uint32_t u : pr[v]) {
                bool covered = false;
                for (uint32_t i = 0; i < 2; i++) {
                    const CPSegment& S = s.segments[u][i];
                    if (S.active && S.s <= slot && slot <= S.e) covered = true;
                }
                if (!covered) return fail("Eq.7: predecessor segment does not cover consumer");
            }
        }
    }
    // Eq.10 + objectives
    if (compute > p.W) return fail("Eq.10: compute budget exceeded");
    if (ddr != s.ddr_bytes) return fail("ddr_bytes mismatch");
    if (compute != s.total_compute) return fail("total_compute mismatch");
    uint32_t rm = 0, sp = 0;
    for (const CPTask& t : s.seq) {
        if (t.kind == CPTask::Kind::REMAT) rm++;
        if (t.kind == CPTask::Kind::PAGE_OUT) sp++;
    }
    if (rm != s.remat_count || sp != s.spill_count) return fail("counters mismatch");
    return true;
}

// ── §5 bridging adapters + env parsing ────────────────────────────────────

bool parse_cp_options_from_env(CPOptions* out) {
    if (!out) return false;
    const char* sel = std::getenv("HNNX_VTCM_ALLOCATOR");
    if (!sel || !*sel || std::strcmp(sel, "fancy") == 0) return false;
    CPOptions o;
    if (std::strcmp(sel, "cp") == 0) {
        o.mode = CPOptions::Mode::FULL;
    } else if (std::strcmp(sel, "cp-paging") == 0) {
        o.mode = CPOptions::Mode::PAGING_ONLY;
    } else if (std::strcmp(sel, "cp-remat") == 0) {
        o.mode = CPOptions::Mode::REMAT_ONLY;
    } else if (std::strcmp(sel, "cp-seq") == 0) {
        o.mode = CPOptions::Mode::SEQ_ONLY;
    } else if (std::strcmp(sel, "cp-reorder") == 0) {
        o.mode = CPOptions::Mode::FULL;
        o.respect_topo_order = false;
    } else {
        return false; // unknown selector: stay on the default greedy path
    }
    uint64_t u = 0;
    if (env_u64("HNNX_CP_NODE_LIMIT", &u) && u > 0) o.node_limit = u;
    if (env_u64("HNNX_CP_TIME_LIMIT_MS", &u) && u > 0) o.time_limit_ms = static_cast<uint32_t>(u);
    if (env_u64("HNNX_CP_MAX_NODES", &u) && u > 0) o.max_exact_nodes = static_cast<uint32_t>(u);
    const char* cm = std::getenv("HNNX_CP_COST_MODEL");
    if (cm && *cm) {
        if (std::strcmp(cm, "uniform") == 0) o.cost_model = CPOptions::CostModel::UNIFORM;
        else if (std::strcmp(cm, "bytes") == 0) o.cost_model = CPOptions::CostModel::BYTES;
        else if (std::strcmp(cm, "table") == 0) o.cost_model = CPOptions::CostModel::COST_TABLE;
    }
    o.objective = o.mode == CPOptions::Mode::REMAT_ONLY ? CPOptions::Objective::TOTAL_COMPUTE
                : o.mode == CPOptions::Mode::SEQ_ONLY    ? CPOptions::Objective::PEAK_RESIDENT
                                                         : CPOptions::Objective::DDR_BYTES;
    *out = o;
    return true;
}

CPProblem build_problem_from_requests(
    const std::vector<fa::FancyAllocator::AllocRequest>& requests,
    const std::vector<std::pair<op_id_t, op_id_t>>& edges,
    size_t vtcm_budget, const std::vector<op_id_t>& topo_order,
    const CPOptions& opts) {
    CPProblem pr;
    std::unordered_map<op_id_t, uint32_t> idx;
    pr.nodes.reserve(requests.size());
    for (const auto& r : requests) {
        if (idx.count(r.op_id)) continue;
        CPTensor t;
        t.id = r.op_id;
        t.m = r.size;
        switch (opts.cost_model) {
            case CPOptions::CostModel::UNIFORM:
                t.w = 1;
                break;
            default: // BYTES; COST_TABLE falls back until a CostSource is wired
                t.w = 1 + (r.size ? (r.size - 1) / kBytesPerW : 0);
                break;
        }
        t.C = opts.C_default;
        idx[r.op_id] = static_cast<uint32_t>(pr.nodes.size());
        pr.nodes.push_back(t);
    }
    pr.edges.reserve(edges.size());
    for (const auto& e : edges) {
        auto a = idx.find(e.first);
        auto b = idx.find(e.second);
        if (a == idx.end() || b == idx.end() || a->second == b->second) continue;
        pr.edges.emplace_back(a->second, b->second);
    }
    pr.M = vtcm_budget;
    uint64_t sum_w = 0;
    for (const auto& t : pr.nodes) sum_w += t.w;
    double slack = 0.25;
    env_double("HNNX_CP_W_SLACK", &slack);
    pr.W = sum_w + static_cast<uint64_t>(static_cast<double>(sum_w) * slack);
    if (pr.W < sum_w) pr.W = sum_w;
    pr.topo_hint = topo_order;
    return pr;
}

std::unordered_map<op_id_t, fa::FancyAllocator::AllocResult>
to_alloc_results(const CPSolution& sol, const CPProblem& p, size_t budget, size_t alignment) {
    (void)budget;
    (void)alignment;
    std::unordered_map<op_id_t, fa::FancyAllocator::AllocResult> out;
    if (sol.segments.size() != p.nodes.size()) return out;
    uint32_t block = 1;
    for (uint32_t v = 0; v < p.nodes.size(); v++) {
        const CPSegment& s0 = sol.segments[v][0];
        fa::FancyAllocator::AllocResult r;
        if (s0.vtcm_offset == kUnplaced) {
            r.spilled = true; // first-fit degraded: same fallback semantics as greedy
        } else {
            r.offset = s0.vtcm_offset;
            r.block_id = block++;
            r.spilled = false;
        }
        out[p.nodes[v].id] = r;
    }
    return out;
}

std::vector<CPSpillFillPlan> to_spill_fill_plans(const CPSolution& sol, const CPProblem& p) {
    std::vector<CPSpillFillPlan> out;
    if (sol.segments.size() != p.nodes.size()) return out;
    // topo index by id + predicate index
    std::unordered_map<op_id_t, size_t> topo_pos;
    for (size_t i = 0; i < p.topo_hint.size(); i++) topo_pos[p.topo_hint[i]] = i;
    std::unordered_map<op_id_t, uint32_t> idx;
    std::vector<std::vector<uint32_t>> pr(p.nodes.size());
    for (uint32_t v = 0; v < p.nodes.size(); v++) idx[p.nodes[v].id] = v;
    for (const auto& e : p.edges)
        if (e.second < p.nodes.size()) pr[e.second].push_back(e.first);
    // consumption events from seq: slot -> (consumer, pred)
    struct Cons {
        uint32_t slot;
        uint32_t consumer;
    };
    std::vector<std::vector<Cons>> by_pred(p.nodes.size());
    for (const CPTask& t : sol.seq) {
        if (t.kind != CPTask::Kind::COMPUTE && t.kind != CPTask::Kind::REMAT) continue;
        auto it = idx.find(t.v);
        if (it == idx.end()) continue;
        for (uint32_t u : pr[it->second])
            by_pred[u].push_back({t.slot, it->second});
    }
    for (uint32_t v = 0; v < p.nodes.size(); v++) {
        const CPSegment& s0 = sol.segments[v][0];
        const CPSegment& s1 = sol.segments[v][1];
        if (!s0.spill_end || !s1.active || !s1.fill_start) continue;
        // spill after the last consumer served by seg0, fill before the first
        // consumer served by seg1
        const Cons* last0 = nullptr;
        const Cons* first1 = nullptr;
        for (const Cons& c : by_pred[v]) {
            if (c.slot <= s0.e && (!last0 || c.slot > last0->slot)) last0 = &c;
            if (c.slot >= s1.s && (!first1 || c.slot < first1->slot)) first1 = &c;
        }
        if (!last0 || !first1) continue; // defensive: spill without service
        CPSpillFillPlan plan;
        auto pos_of = [&](uint32_t consumer) -> size_t {
            const auto it2 = topo_pos.find(p.nodes[consumer].id);
            return it2 == topo_pos.end() ? 0 : it2->second;
        };
        plan.spill_position = pos_of(last0->consumer);
        plan.fill_position = pos_of(first1->consumer);
        plan.vtcm_offset = s1.vtcm_offset == kUnplaced ? 0 : s1.vtcm_offset;
        plan.vtcm_offset_spill = s0.vtcm_offset == kUnplaced ? 0 : s0.vtcm_offset;
        plan.ddr_offset = s0.ddr_offset;
        plan.size = p.nodes[v].m;
        plan.double_buffered = false;
        plan.op_id = p.nodes[v].id;
        out.push_back(plan);
    }
    return out;
}

} // namespace cp
} // namespace hnnx
