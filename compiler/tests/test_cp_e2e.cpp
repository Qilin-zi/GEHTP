// test_cp_e2e.cpp — end-to-end wiring of the CP oracle branch.
//
// GraphPrepare::vtcm_lifetime_alloc gains an env-selected branch:
//   HNNX_VTCM_ALLOCATOR unset/"fancy"  -> greedy path, byte-identical behavior
//   HNNX_VTCM_ALLOCATOR=cp / cp-paging -> CP oracle over the SAME requests,
//                                         verified solution -> allocations +
//                                         spill/fill plans + OpEmitter records
// This test drives the branch through the full prepare() pipeline.
//
// Pressure fixture (graph B): Abs -> Ceiling -> Cos -> Clamp -> Cast -> Concat
// with m = {1MB, 1.25MB, 1.25MB, 128KB, 128KB, 1.125MB} and a skip edge
// Abs->Concat. VTCM budget = min(4MB, 4MB) x 0.75 = 3MB.
// Every edge pair fits (t2+t3 = 2.5MB <= 3MB), but the Cos slot forces
// {t1 (skip tensor, future Concat need), t2 (consumed), t3 (output)} =
// 3.5MB > 3MB: t1 MUST leave VTCM mid-chain. In cp-paging mode (remat
// forbidden) that forces a spill+fill pair -> non-empty plans; the
// record-level Phase B pass then emits 2 DmaOpInfos per plan through
// OpEmitter::insert_spill_fill_pair.

#include "hnnx/ir/graph_prepare.hpp"
#include "hnnx/ir/op_registry.hpp"
#include "hnnx/api/hexagon_nn_env.hpp"
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/schedule/scheduler.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace hnnx {
void register_all_ops();
}

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

using namespace hnnx;

static OutputDef vec_out(size_t elems) {
    OutputDef od{};
    od.rank = 1;
    od.dtype = static_cast<uint32_t>(DType::Float32);
    od.dims[0] = elems;
    return od;
}

// Graph B: Input(5) -> Abs(10) -> Ceiling(20) -> Cos(30) -> Clamp(40) -> Cast(50)
//          Concat(60) <- [10 (skip), 50]; Output(70) <- 60.
template <typename F>
static void with_env(const char* allocator, F&& body) {
    if (allocator) setenv("HNNX_VTCM_ALLOCATOR", allocator, 1);
    else unsetenv("HNNX_VTCM_ALLOCATOR");
    unsetenv("HNNX_CP_NODE_LIMIT");
    unsetenv("HNNX_CP_TIME_LIMIT_MS");
    unsetenv("HNNX_CP_MAX_NODES");
    unsetenv("HNNX_CP_W_SLACK");
    unsetenv("HNNX_CP_COST_MODEL");
    body();
    unsetenv("HNNX_VTCM_ALLOCATOR");
}

static GraphPrepare* make_pressure_graph() {
    GraphPrepare* gp = new GraphPrepare();
    InputDef in5{};
    in5.rank = 5;
    in5.dtype = 0;
    OutputDef o5 = vec_out(1024); // 4KB
    gp->append_node("Input", 5, nullptr, 0, &o5, 1, nullptr);

    struct Spec {
        const char* name;
        uint32_t id;
        uint32_t src;
        size_t out_elems;
    };
    const Spec specs[] = {
        {"Abs", 10, 5, 262144},      // 1MB    (skip tensor, re-consumed by Concat)
        {"Ceiling", 20, 10, 327680}, // 1.25MB
        {"Cos", 30, 20, 327680},     // 1.25MB (pressure slot: 1+1.25+1.25 > 3MB)
        {"Clamp", 40, 30, 32768},    // 128KB
        {"Cast", 50, 40, 32768},     // 128KB
    };
    for (const auto& s : specs) {
        InputDef in{};
        in.rank = s.src;
        in.dtype = 0;
        OutputDef od = vec_out(s.out_elems);
        gp->append_node(s.name, s.id, &in, 1, &od, 1, nullptr);
    }
    InputDef i60[2];
    i60[0].rank = 10; // skip: Concat re-consumes Abs's 1MB output
    i60[0].dtype = 0;
    i60[1].rank = 50;
    i60[1].dtype = 0;
    OutputDef o60 = vec_out(262144 + 32768); // 1.125MB
    gp->append_node("Concat", 60, i60, 2, &o60, 1, nullptr);
    InputDef i70{};
    i70.rank = 60;
    i70.dtype = 0;
    gp->append_node("Output", 70, &i70, 1, nullptr, 0, nullptr);
    return gp;
}

int main() {
    register_all_ops();
    HexagonNNEnv env;
    env.set_num_nsps(1);
    env.set_soc_type(75);

    const uint64_t budget = static_cast<uint64_t>(4 * 1024 * 1024 / 4 * 3); // 3MB

    // ── 1. default (fancy): CP branch must stay dormant ──
    {
        GraphPrepare* gp = make_pressure_graph();
        GraphStatus st = gp->prepare(env);
        check(st == GraphStatus::Success, "e2e/fancy: prepare OK");
        check(!gp->is_cp_allocator_active(), "e2e/fancy: cp branch dormant");
        check(!gp->get_vtcm_allocations().empty(), "e2e/fancy: allocations populated");
        check(gp->get_cp_op_emitter() == nullptr, "e2e/fancy: no emitter records");
        const size_t n_fancy = gp->get_vtcm_allocations().size();
        check(n_fancy > 0, "e2e/fancy: nonzero allocation count");
        std::printf("fancy: %zu allocations\n", n_fancy);
        delete gp;
    }

    // ── 2. cp mode: verified solution wired into allocations ──
    with_env("cp", [&] {
        GraphPrepare* gp = make_pressure_graph();
        GraphStatus st = gp->prepare(env);
        check(st == GraphStatus::Success, "e2e/cp: prepare OK");
        if (!gp->is_cp_allocator_active()) {
            // acceptable only if the solver proved infeasible and fell back
            check(!gp->get_vtcm_allocations().empty(),
                  "e2e/cp: fallback still produced allocations");
            std::printf("cp: fell back to greedy\n");
        } else {
            const auto& allocs = gp->get_vtcm_allocations();
            const auto& sol = gp->get_cp_solution();
            check(sol.feasible, "e2e/cp: solution feasible");
            check(sol.peak_resident <= budget, "e2e/cp: peak within budget");
            check(allocs.size() == sol.segments.size() ||
                      allocs.size() <= sol.segments.size(),
                  "e2e/cp: allocation count consistent with solution");
            uint64_t max_off = 0;
            for (const auto& [id, e] : allocs) {
                if (!e.spilled && e.offset > max_off) max_off = e.offset;
                (void)id;
            }
            check(max_off < budget, "e2e/cp: offsets within budget");
            const auto& plans = gp->get_cp_spill_fill_plans();
            for (const auto& pl : plans)
                check(pl.fill_position > pl.spill_position, "e2e/cp: fill after spill");
            std::printf("cp: ddr=%llu spill=%u remat=%u plans=%zu\n",
                        (unsigned long long)sol.ddr_bytes, sol.spill_count, sol.remat_count,
                        plans.size());
            // relief structure: either DDR-free (remat/fit) or paired plans
            check(sol.ddr_bytes == 0 || plans.size() > 0 || sol.spill_count > 0,
                  "e2e/cp: pressure resolved by remat or paging");
        }
        delete gp;
    });

    // ── 3. cp-paging: remat forbidden -> spill+fill pair MUST appear, and the
    //      record-level Phase B pass emits 2 DmaOpInfos per plan ──
    with_env("cp-paging", [&] {
        GraphPrepare* gp = make_pressure_graph();
        GraphStatus st = gp->prepare(env);
        check(st == GraphStatus::Success, "e2e/cp-paging: prepare OK");
        check(gp->is_cp_allocator_active(), "e2e/cp-paging: cp branch active");
        if (gp->is_cp_allocator_active()) {
            const auto& sol = gp->get_cp_solution();
            const auto& plans = gp->get_cp_spill_fill_plans();
            check(sol.spill_count >= 1, "e2e/cp-paging: forced spill on pressure graph");
            check(!plans.empty(), "e2e/cp-paging: spill/fill plans non-empty");
            const OpEmitter* em = gp->get_cp_op_emitter();
            check(em != nullptr, "e2e/cp-paging: OpEmitter records present");
            if (em && !plans.empty()) {
                check(em->get_emitted_ops().size() == 2 * plans.size(),
                      "e2e/cp-paging: 2 DMA records per plan");
                check(em->validate_spill_fill(), "e2e/cp-paging: emitter validation");
                // each plan's pair shares the DDR address (Eq.9) and the fill
                // destination is inside the budget
                const auto& ops = em->get_emitted_ops();
                for (size_t i = 0; i + 1 < ops.size(); i += 2) {
                    const auto& spill = ops[i];
                    const auto& fill = ops[i + 1];
                    check(spill.type == DmaOpType::Spill && fill.type == DmaOpType::Fill,
                          "e2e/cp-paging: spill/fill record types");
                    check(spill.dst_offset == fill.src_offset,
                          "e2e/cp-paging: Eq.9 spill dst == fill src (same DDR address)");
                    check(fill.dst_offset < budget, "e2e/cp-paging: fill dst within budget");
                    check(spill.size == fill.size && spill.size > 0,
                          "e2e/cp-paging: pair sizes match");
                }
            }
            std::printf("cp-paging: ddr=%llu plans=%zu emitter=%zu\n",
                        (unsigned long long)sol.ddr_bytes, plans.size(),
                        em ? em->get_emitted_ops().size() : 0);
        }
        delete gp;
    });

    // ── 4. Phase B serialization level: Scheduler::apply_cp_spill_fill ──
    // Plan-layer fixture: two compute steps per graph tensor in topo order,
    // each referencing its tensor by id — the anchoring contract
    // apply_cp_spill_fill resolves positions against.
    static const uint32_t topo_ids[8] = {5, 10, 20, 30, 40, 50, 60, 70};
    auto linear_plan = [&]() {
        Scheduler::Plan plan;
        uint32_t step = 0;
        for (size_t t = 0; t < 8; t++) {
            TensorInfo ti{};
            ti.tensor_id = topo_ids[t];
            ti.rank = 1;
            plan.tensors.push_back(ti);
            for (int k = 0; k < 2; k++) {
                ScheduledOp op;
                op.record_id = compute_record_id(step++);
                op.tensor_id = topo_ids[t];
                op.type = OP_TYPE_COMPUTE;
                op.step_name = "node" + std::to_string(t);
                plan.ops.push_back(op);
            }
        }
        return plan;
    };
    auto is_cp_step = [](const ScheduledOp& op) {
        return op.step_name == "CpSpill" || op.step_name == "CpFill";
    };
    auto record_ids_valid = [](const Scheduler::Plan& p) {
        for (size_t i = 0; i < p.ops.size(); i++)
            if (p.ops[i].record_id != compute_record_id((uint32_t)i)) return false;
        return true;
    };

    with_env("cp-paging", [&] {
        GraphPrepare* gp = make_pressure_graph();
        gp->prepare(env);
        const auto& plans = gp->get_cp_spill_fill_plans();
        check(plans.size() == 1, "e2e/plan: single forced spill plan");
        if (plans.size() == 1) {
            Scheduler::Plan plan = linear_plan();
            // hand-built layout: consumer at topo t owns steps 2t, 2t+1, so
            // spill lands at 2*spill_position+2 (after last seg0 consumer)
            // and fill at 2*fill_position+1 (before first seg1 consumer,
            // shifted one past the earlier spill insertion).
            const size_t spill_exp = 2 * plans[0].spill_position + 2;
            const size_t fill_exp = 2 * plans[0].fill_position + 1;
            Scheduler sched;
            sched.apply_cp_spill_fill(plan, *gp);
            check(plan.ops.size() == 16 + 2, "e2e/plan: one DMA pair inserted");
            check(fill_exp > spill_exp, "e2e/plan: fill after spill");
            if (plan.ops.size() == 18) {
                const ScheduledOp& sp = plan.ops[spill_exp];
                const ScheduledOp& fl = plan.ops[fill_exp];
                check(sp.step_name == "CpSpill" && fl.step_name == "CpFill",
                      "e2e/plan: DMA pair at consumer boundaries");
                check(sp.type == OP_TYPE_DMA && fl.type == OP_TYPE_DMA,
                      "e2e/plan: pair type OP_TYPE_DMA");
                check(sp.tensor_id == (uint32_t)plans[0].op_id &&
                          fl.tensor_id == (uint32_t)plans[0].op_id,
                      "e2e/plan: pair carries the paged tensor id");
                check(sp.kernel_name == "@DmaCheckpointSet",
                      "e2e/plan: checkpoint kernel name");
                // Eq.9: shared DDR address; tentative extras {ddr, vtcm, size}
                check(sp.extras.size() == 3 && fl.extras.size() == 3,
                      "e2e/plan: extras arity");
                if (sp.extras.size() == 3 && fl.extras.size() == 3) {
                    check(sp.extras[0] == fl.extras[0] &&
                              sp.extras[0] == (uint32_t)plans[0].ddr_offset,
                          "e2e/plan: Eq.9 shared DDR address");
                    check(sp.extras[2] == fl.extras[2] &&
                              sp.extras[2] == (uint32_t)plans[0].size,
                          "e2e/plan: pair sizes match plan");
                    check(sp.extras[1] == (uint32_t)plans[0].vtcm_offset_spill &&
                              fl.extras[1] == (uint32_t)plans[0].vtcm_offset,
                          "e2e/plan: per-end VTCM offsets");
                }
                check((sp.block_ref & 0xF000u) == 0x1000u &&
                          (fl.block_ref & 0xF000u) == 0x1000u,
                      "e2e/plan: VTCM block_ref prefix");
                check(sp.f2 != fl.f2 && sp.f2 >= 0x20u && fl.f2 >= 0x20u,
                      "e2e/plan: distinct DMA tags clear of pattern range");
            }
            size_t cp_count = 0;
            for (const auto& op : plan.ops)
                if (is_cp_step(op)) cp_count++;
            check(cp_count == 2, "e2e/plan: exactly one pair, no strays");
            check(record_ids_valid(plan), "e2e/plan: record_id renumbered");
        }
        delete gp;
    });

    // ── 5. non-CP regression: apply is a no-op, plan unchanged ──
    {
        GraphPrepare* gp = make_pressure_graph();
        gp->prepare(env); // fancy (env unset)
        Scheduler::Plan plan = linear_plan();
        std::vector<uint32_t> ids_before;
        std::vector<std::string> names_before;
        for (const auto& op : plan.ops) {
            ids_before.push_back(op.record_id);
            names_before.push_back(op.step_name);
        }
        Scheduler sched;
        sched.apply_cp_spill_fill(plan, *gp);
        check(plan.ops.size() == 16, "e2e/plan-fancy: zero insertion");
        bool identical = plan.ops.size() == 16;
        for (size_t i = 0; identical && i < plan.ops.size(); i++)
            identical = plan.ops[i].record_id == ids_before[i] &&
                        plan.ops[i].step_name == names_before[i];
        check(identical, "e2e/plan-fancy: byte-identical plan");
        delete gp;
    }

    // ── 6. schedule() integration: cp plans surface in the scheduled plan,
    //      fancy plans stay untouched ──
    {
        GraphPrepare* gp_fancy = make_pressure_graph();
        gp_fancy->prepare(env);
        Scheduler s_fancy;
        auto pf = s_fancy.schedule(*gp_fancy);
        size_t cp_fancy = 0;
        for (const auto& op : pf.ops)
            if (is_cp_step(op)) cp_fancy++;
        check(cp_fancy == 0, "e2e/sched: fancy schedule has no CP ops");
        check(record_ids_valid(pf), "e2e/sched: fancy record_id pattern");
        delete gp_fancy;
    }
    with_env("cp-paging", [&] {
        GraphPrepare* gp = make_pressure_graph();
        gp->prepare(env);
        const size_t n_plans = gp->get_cp_spill_fill_plans().size();
        Scheduler s;
        auto pc = s.schedule(*gp);
        size_t cp_ops = 0, spill_idx = SIZE_MAX, fill_idx = 0;
        for (size_t i = 0; i < pc.ops.size(); i++) {
            if (pc.ops[i].step_name == "CpSpill") {
                cp_ops++;
                spill_idx = i;
            } else if (pc.ops[i].step_name == "CpFill") {
                cp_ops++;
                fill_idx = i;
            }
        }
        check(n_plans > 0 && cp_ops == 2 * n_plans,
              "e2e/sched: cp schedule carries 2 DMA ops per plan");
        check(fill_idx > spill_idx, "e2e/sched: fill ordered after spill");
        check(record_ids_valid(pc), "e2e/sched: cp record_id renumbered");
        delete gp;
    });

    // ── 7. fancy regression: two fancy runs agree (default path untouched) ──
    {
        std::vector<std::string> sigs;
        for (int run = 0; run < 2; run++) {
            unsetenv("HNNX_VTCM_ALLOCATOR");
            GraphPrepare* gp = make_pressure_graph();
            gp->prepare(env);
            std::string sig;
            for (const auto& [id, e] : gp->get_vtcm_allocations()) {
                sig += std::to_string(id) + ":" + std::to_string(e.offset) + ":" +
                       std::to_string(e.block_id) + ":" + (e.spilled ? "S" : "R") + ";";
            }
            sigs.push_back(sig);
            delete gp;
        }
        check(sigs[0] == sigs[1], "e2e/fancy: deterministic across runs");
    }

    std::printf("test_cp_e2e: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
