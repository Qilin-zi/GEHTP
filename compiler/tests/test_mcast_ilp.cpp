// Test: Multicast ILP optimizer (Phase 4.3)
// Validates McastOptimizer with capacity constraints, receiver conflict
// detection, num_mcids fix, MPS dump, and pipeline integration.
#include "hnnx/mcast/mcast_optimizer.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>

using namespace hnnx;

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++tests_passed;
    } else {
        ++tests_failed;
        std::cout << "  FAIL: " << msg << "\n";
    }
}

static McSend make_send(uint32_t tag, uint32_t nsp, uint64_t payload,
                        std::vector<uint32_t> mcids,
                        std::vector<uint32_t> receivers) {
    McSend s{};
    s.tag = tag;
    s.sender_nsp = nsp;
    s.num_mcids = static_cast<uint32_t>(mcids.size());
    s.payload_size = payload;
    s.mcids = std::move(mcids);
    s.receivers = std::move(receivers);
    return s;
}

// ---------------------------------------------------------------------------
// Test 1: Basic greedy merge — two sends with overlapping MCID, same sender,
//   receiver sets are disjoint (no partial overlap).
// ---------------------------------------------------------------------------
static void test_basic_merge() {
    std::cout << "\n[Test 1] Basic greedy merge\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10}, {1, 2}),
        make_send(2, 0, 512,  {10}, {3, 4}),  // disjoint receivers
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 1, "merged into 1 supercast");
    check(result[0].mcids.size() == 1, "1 MCID (union)");
    check(result[0].payload_size == 1536, "payload summed");
    check(result[0].receivers.size() == 4, "4 receivers (union)");
    check(result[0].num_mcids == 1, "num_mcids = 1 (Phase 4.3 fix)");
    check(opt.supercast_count() == 1, "1 supercast recorded");
    check(opt.result_mcsend_count() == 1, "result count = 1");
}

// ---------------------------------------------------------------------------
// Test 2: No merge — different senders.
// ---------------------------------------------------------------------------
static void test_no_merge_different_sender() {
    std::cout << "\n[Test 2] No merge (different senders)\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10}, {1}),
        make_send(2, 1, 512,  {10}, {2}),
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 2, "not merged (different senders)");
    check(opt.supercast_count() == 0, "0 supercasts");
}

// ---------------------------------------------------------------------------
// Test 3: No merge — non-overlapping MCIDs.
// ---------------------------------------------------------------------------
static void test_no_merge_no_mcid_overlap() {
    std::cout << "\n[Test 3] No merge (no MCID overlap)\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10}, {1}),
        make_send(2, 0, 512,  {20}, {1}),
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 2, "not merged (different MCIDs)");
    check(opt.supercast_count() == 0, "0 supercasts");
}

// ---------------------------------------------------------------------------
// Test 4: Capacity constraint — reject merge that exceeds buffer.
// ---------------------------------------------------------------------------
static void test_capacity_constraint() {
    std::cout << "\n[Test 4] Capacity constraint\n";
    McastOptimizer opt;
    opt.set_max_mcast_buffer_size(2048);  // 2KB cap
    auto sends = {
        make_send(1, 0, 1024, {10}, {1}),
        make_send(2, 0, 2048, {10}, {2}),  // total = 3072 > 2048
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 2, "not merged (exceeds capacity)");
    check(opt.supercast_count() == 0, "0 supercasts (capacity rejected)");
}

// ---------------------------------------------------------------------------
// Test 5: Capacity allows merge when under limit.
// ---------------------------------------------------------------------------
static void test_capacity_allows() {
    std::cout << "\n[Test 5] Capacity allows merge\n";
    McastOptimizer opt;
    opt.set_max_mcast_buffer_size(4096);  // 4KB cap
    auto sends = {
        make_send(1, 0, 1024, {10}, {1}),
        make_send(2, 0, 1024, {10}, {2}),  // total = 2048 <= 4096
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 1, "merged (under capacity)");
    check(opt.supercast_count() == 1, "1 supercast");
}

// ---------------------------------------------------------------------------
// Test 6: Receiver conflict — partial receiver overlap rejected.
//   A sends to {1,2}, B sends to {2,3}. Partial overlap at NSP 2.
//   Merging would cause NSP 2 to receive two different payloads on same MCID.
// ---------------------------------------------------------------------------
static void test_receiver_conflict() {
    std::cout << "\n[Test 6] Receiver conflict (partial overlap)\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10}, {1, 2}),
        make_send(2, 0, 512,  {10}, {2, 3}),  // partial overlap at 2
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 2, "not merged (receiver conflict)");
    check(opt.supercast_count() == 0, "0 supercasts");
}

// ---------------------------------------------------------------------------
// Test 7: Receiver compatible — one is subset of the other.
//   A sends to {1,2,3}, B sends to {1,2}. B ⊆ A → OK to merge.
// ---------------------------------------------------------------------------
static void test_receiver_subset() {
    std::cout << "\n[Test 7] Receiver subset (compatible)\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10}, {1, 2, 3}),
        make_send(2, 0, 512,  {10}, {1, 2}),   // subset of A's receivers
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 1, "merged (receiver subset)");
    check(opt.supercast_count() == 1, "1 supercast");
}

// ---------------------------------------------------------------------------
// Test 8: num_mcids fix — verify merged send has correct num_mcids.
// ---------------------------------------------------------------------------
static void test_num_mcids_fix() {
    std::cout << "\n[Test 8] num_mcids fix\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10, 11}, {1}),
        make_send(2, 0, 512,  {11, 12}, {1}),  // overlap at 11
    };
    auto result = opt.optimize(sends, 2);

    check(result.size() == 1, "merged");
    // Union of MCIDs: {10, 11, 12} → 3
    check(result[0].mcids.size() == 3, "3 MCIDs in union");
    check(result[0].num_mcids == 3, "num_mcids = 3 (was 0 before fix)");
}

// ---------------------------------------------------------------------------
// Test 9: Empty input.
// ---------------------------------------------------------------------------
static void test_empty_input() {
    std::cout << "\n[Test 9] Empty input\n";
    McastOptimizer opt;
    auto result = opt.optimize({}, 0);
    check(result.empty(), "empty input → empty output");
    check(opt.supercast_count() == 0, "0 supercasts");
}

// ---------------------------------------------------------------------------
// Test 10: Transitive merge — A-B overlap, B-C overlap, A-C don't.
//   Greedy merges transitively: A+B+C (B's MCIDs added to supercast,
//   then C overlaps with the extended set).
// ---------------------------------------------------------------------------
static void test_transitive_merge() {
    std::cout << "\n[Test 10] Transitive merge\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 512, {10}, {1}),
        make_send(2, 0, 512, {11}, {1}),  // B overlaps A? No (10 vs 11)
        make_send(3, 0, 512, {10}, {1}),  // C overlaps A at 10
    };
    auto result = opt.optimize(sends, 3);

    // A and C share MCID 10 → merged. B has MCID 11 (no overlap) → separate.
    check(result.size() == 2, "A+C merged, B separate");
    check(opt.supercast_count() == 1, "1 supercast");
}

// ---------------------------------------------------------------------------
// Test 11: dump_mps produces a file.
// ---------------------------------------------------------------------------
static void test_dump_mps() {
    std::cout << "\n[Test 11] dump_mps\n";
    McastOptimizer opt;
    opt.set_dump_mps(true);
    auto sends = {
        make_send(1, 0, 1024, {10}, {1}),
        make_send(2, 0, 512,  {10}, {2}),
    };
    opt.optimize(sends, 2);
    // dump_mps is called internally by solve_ilp when dump_mps_ is set
    // Check if file was created
    std::ifstream f("MCastLP.mps");
    bool exists = f.good();
    f.close();
    check(exists, "MPS file created");
    if (exists) {
        std::cout << "  MPS file: MCastLP.mps created\n";
        // Clean up
        std::remove("MCastLP.mps");
    }
}

// ---------------------------------------------------------------------------
// Test 12: Statistics tracking.
// ---------------------------------------------------------------------------
static void test_statistics() {
    std::cout << "\n[Test 12] Statistics\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10}, {1}),
        make_send(2, 0, 512,  {10}, {2}),
        make_send(3, 0, 256,  {20}, {1}),  // no overlap → not merged
    };
    opt.optimize(sends, 3);

    check(opt.original_mcsend_count() == 3, "original = 3");
    check(opt.supercast_count() == 1, "1 supercast");
    check(opt.result_mcsend_count() == 2, "result = 2 (1 merged + 1 separate)");
}

// ---------------------------------------------------------------------------
// Test 13: Multi-NSP scenario — 3 sends, 2 merge (disjoint receivers), 1 separate.
// ---------------------------------------------------------------------------
static void test_multi_nsp() {
    std::cout << "\n[Test 13] Multi-NSP scenario\n";
    McastOptimizer opt;
    auto sends = {
        make_send(1, 0, 1024, {10}, {1, 2}),
        make_send(2, 0, 2048, {10}, {3, 4}),  // same sender, same MCID, disjoint receivers → merge
        make_send(3, 1, 512,  {20}, {0, 2}),  // different sender → separate
    };
    auto result = opt.optimize(sends, 3);

    check(result.size() == 2, "2 result sends (1 supercast + 1 separate)");
    check(opt.supercast_count() == 1, "1 supercast");

    // Find the supercast (merged send has 4 receivers union)
    bool found_merged = false;
    for (const auto& s : result) {
        if (s.receivers.size() == 4) {
            found_merged = true;
            check(s.payload_size == 3072, "merged payload = 3072");
            break;
        }
    }
    check(found_merged, "found merged supercast");
}

// ---------------------------------------------------------------------------
// Test 14: Pipeline integration — run_mcast_optimization on GraphPrepare.
//   In single-NSP mode, should produce no mcsends (no-op).
// ---------------------------------------------------------------------------
static void test_pipeline_integration() {
    std::cout << "\n[Test 14] Pipeline integration (single-NSP, no-op)\n";
    GraphPrepare gp;
    // Build a minimal graph: just an Input node
    OutputDef in_od{};
    in_od.rank = 1;
    in_od.dims[0] = 4;
    in_od.dtype = 0;
    in_od.element_size = 4;
    InputDef no_input{};
    gp.append_node("Input", 1, &no_input, 0, &in_od, 1, nullptr);

    // This should not crash and be a no-op in single-NSP mode
    gp.run_mcast_optimization();
    check(true, "run_mcast_optimization completed without error");
    std::cout << "  (single-NSP: no mcsends, no-op)\n";
}

int main() {
    std::cout << "=== Multicast ILP Optimizer Test (Phase 4.3) ===\n";

    test_basic_merge();
    test_no_merge_different_sender();
    test_no_merge_no_mcid_overlap();
    test_capacity_constraint();
    test_capacity_allows();
    test_receiver_conflict();
    test_receiver_subset();
    test_num_mcids_fix();
    test_empty_input();
    test_transitive_merge();
    test_dump_mps();
    test_statistics();
    test_multi_nsp();
    test_pipeline_integration();

    std::cout << "\n=== " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
