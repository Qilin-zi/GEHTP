// Test: VTCM lifetime-end sort + identical-interval grouping (反汇编路径)
// Validates FancyAllocator::allocate_with_lifetime against several graph
// topologies. Algorithm matches libHtpPrepare.so x86_64 v2.48 disassembly:
//   1. Sort by life_end DESC, life_begin ASC, op_id ASC
//   2. Group by identical (life_begin, life_end)
//   3. Groups placed sequentially (bump), no cross-group reuse
//   4. Within group: sequential bump for members
//   5. Spill if over budget
#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/ir/types.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using fa::FancyAllocator;
using hnnx::op_id_t;

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

static std::unordered_map<op_id_t, FancyAllocator::AllocResult>
run_alloc(const std::vector<FancyAllocator::AllocRequest>& reqs,
          size_t budget, size_t alignment = 128) {
    FancyAllocator alloc;
    return alloc.allocate_with_lifetime(reqs, budget, alignment);
}

// ---------------------------------------------------------------------------
// Test 1: Diamond graph — A produces B and C, both feed D.
//   Lifetimes (topo indices):
//     A: [0, 0]   B: [1, 2]   C: [1, 2]   D: [2, 2]
//   Sort by life_end DESC, life_begin ASC, op_id ASC:
//     B[1,2], C[1,2], D[2,2], A[0,0]
//   Groups: G1={B,C} (same [1,2]), G2={D}, G3={A}
//   Placement (sequential bump):
//     G1: B@0, C@1024 (within-group bump), region [0, 2048)
//     G2: D@2048, region [2048, 3072)
//     G3: A@3072, region [3072, 4096)
//   Used = 4096, saved = 0 (no cross-group reuse in this function)
// ---------------------------------------------------------------------------
static void test_diamond_reuse() {
    std::cout << "\n[Test 1] Diamond graph\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 1024, 0, 0},  // A
        {2, 1024, 1, 2},  // B
        {3, 1024, 1, 2},  // C
        {4, 1024, 2, 2},  // D
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    check(results.count(1) && !results[1].spilled, "A not spilled");
    check(results.count(2) && !results[2].spilled, "B not spilled");
    check(results.count(3) && !results[3].spilled, "C not spilled");
    check(results.count(4) && !results[4].spilled, "D not spilled");

    // B and C have identical lifetime [1,2] → same group, sequential bump
    check(results[2].offset != results[3].offset, "B and C different offsets (same group, bump)");

    // Sort order: B[1,2] first → B gets offset 0
    check(results[2].offset == 0, "B at offset 0 (first in sort order)");

    // C is in same group as B → C at offset 1024 (within-group bump)
    check(results[3].offset == 1024, "C at offset 1024 (within-group bump from B)");

    // D[2,2] is a different group → placed after G1
    check(results[4].offset == 2048, "D at offset 2048 (new group after G1)");

    // A[0,0] is last (life_end=0 is smallest) → placed after D
    check(results[1].offset == 3072, "A at offset 3072 (last in sort order)");

    check(alloc.total_vtcm_used() == 4 * 1024, "VTCM used = 4 blocks (no cross-group reuse)");
    check(alloc.vtcm_saved_by_reuse() == 0, "saved = 0 (cross-group reuse is RuntimeAllocator's job)");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 2: Linear chain A→B→C→D.
//   Lifetimes: A[0,1], B[1,2], C[2,3], D[3,3]
//   Sort by life_end DESC, life_begin ASC:
//     C[2,3] (end=3, begin=2), D[3,3] (end=3, begin=3), B[1,2], A[0,1]
//   Groups: each unique → 4 groups
//   Placement (sequential bump): C@0, D@1024, B@2048, A@3072
//   Used = 4096, saved = 0
// ---------------------------------------------------------------------------
static void test_linear_chain_reuse() {
    std::cout << "\n[Test 2] Linear chain\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 1024, 0, 1},  // A
        {2, 1024, 1, 2},  // B
        {3, 1024, 2, 3},  // C
        {4, 1024, 3, 3},  // D
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    // Sort: C[2,3] first (end=3, begin=2), then D[3,3] (end=3, begin=3), B, A
    check(results[3].offset == 0, "C at offset 0 (end=3, begin=2, first in sort)");
    check(results[4].offset == 1024, "D at offset 1024 (end=3, begin=3, second)");
    check(results[2].offset == 2048, "B at offset 2048 (end=2)");
    check(results[1].offset == 3072, "A at offset 3072 (end=1, last in DESC sort)");

    // All different offsets (no two share a lifetime group)
    check(results[1].offset != results[2].offset, "A != B");
    check(results[2].offset != results[3].offset, "B != C");
    check(results[3].offset != results[4].offset, "C != D");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
    check(alloc.total_vtcm_used() == 4 * 1024, "used = 4 blocks");
    check(alloc.vtcm_saved_by_reuse() == 0, "saved = 0 (sequential bump, no cross-group reuse)");
}

// ---------------------------------------------------------------------------
// Test 3: All overlapping — every tensor has identical lifetime [0,5].
//   All in one group → within-group sequential bump.
//   Sort by op_id: 1, 2, 3, 4
//   Placement: 1@0, 2@512, 3@1024, 4@1536
//   Used = 2048, saved = 0
// ---------------------------------------------------------------------------
static void test_no_reuse_all_overlap() {
    std::cout << "\n[Test 3] All-same-lifetime (one group)\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 512,  0, 5},
        {2, 512,  0, 5},
        {3, 512,  0, 5},
        {4, 512,  0, 5},
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    // All same lifetime → one group, sequential bump
    check(results[1].offset == 0, "op1 at offset 0");
    check(results[2].offset == 512, "op2 at offset 512");
    check(results[3].offset == 1024, "op3 at offset 1024");
    check(results[4].offset == 1536, "op4 at offset 1536");

    // All different offsets
    check(results[1].offset != results[2].offset, "1 != 2");
    check(results[1].offset != results[3].offset, "1 != 3");
    check(results[1].offset != results[4].offset, "1 != 4");
    check(results[2].offset != results[3].offset, "2 != 3");
    check(results[2].offset != results[4].offset, "2 != 4");
    check(results[3].offset != results[4].offset, "3 != 4");

    check(alloc.total_vtcm_used() == 4 * 512, "used = 4*512");
    check(alloc.vtcm_saved_by_reuse() == 0, "no savings");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 4: Budget overflow — 3 tensors of 1024 bytes, all [0,2] (one group).
//   Budget 2048 fits only 2 → 1 spilled (whole group spills together).
// ---------------------------------------------------------------------------
static void test_spill_on_overflow() {
    std::cout << "\n[Test 4] Budget overflow spill\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 1024, 0, 2},
        {2, 1024, 0, 2},
        {3, 1024, 0, 2},
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 2048);

    // One group [0,2], total = 3072 > budget 2048 → whole group spills
    int placed = 0, spilled = 0;
    for (const auto& [id, r] : results) {
        if (r.spilled) ++spilled;
        else ++placed;
    }
    check(placed == 0, "0 placed (whole group overflows budget)");
    check(spilled == 3, "3 spilled (whole group spills together)");

    std::cout << "  placed=" << placed << ", spilled=" << spilled << "\n";
}

// ---------------------------------------------------------------------------
// Test 5: Identical lifetime grouping with size variation.
//   Two tensors with same lifetime [1,1] but different sizes → one group.
//   One tensor with different lifetime [0,0] → another group.
//   Sort by life_end DESC: [1,1] group first, then [0,0] group.
// ---------------------------------------------------------------------------
static void test_identical_lifetime_grouping() {
    std::cout << "\n[Test 5] Identical lifetime grouping\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 2048, 0, 0},  // A: life_end=0
        {2, 512,  1, 1},  // B: life_end=1
        {3, 512,  1, 1},  // C: life_end=1 (same as B)
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    // Sort: B[1,1], C[1,1] (life_end=1, first), then A[0,0] (life_end=0)
    // Group G1 = {B, C} (identical [1,1]), G2 = {A}
    // G1 placed first: B@0, C@512 (within-group bump)
    // G2 placed after: A@1024 (aligned: 512→512, so G1 ends at 1024)

    check(results[2].offset == 0, "B at offset 0 (first in sort order)");
    check(results[3].offset == 512, "C at offset 512 (same group as B, bump)");
    check(results[2].offset != results[3].offset, "B and C different (same group, bump)");

    // A is in a different group, placed after G1
    check(results[1].offset != results[2].offset, "A different from B (different group)");
    check(results[1].offset != results[3].offset, "A different from C (different group)");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 6: get_allocation query + empty input.
// ---------------------------------------------------------------------------
static void test_query_and_empty() {
    std::cout << "\n[Test 6] Query + empty input\n";
    FancyAllocator alloc;

    auto empty = alloc.allocate_with_lifetime({}, 1 << 20);
    check(empty.empty(), "empty input -> empty output");

    check(alloc.get_allocation(999) == nullptr, "query unallocated -> nullptr");

    std::vector<FancyAllocator::AllocRequest> reqs = {
        {42, 256, 0, 0},
    };
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);
    const auto* res = alloc.get_allocation(42);
    check(res != nullptr, "get_allocation(42) found");
    check(res && !res->spilled, "op 42 not spilled");
    check(res && res->offset == 0, "op 42 at offset 0");

    std::cout << "  query OK\n";
}

// ---------------------------------------------------------------------------
// Test 7: Alignment — all offsets aligned to 128 bytes.
// ---------------------------------------------------------------------------
static void test_alignment() {
    std::cout << "\n[Test 7] Alignment (128 bytes)\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 100, 0, 0},
        {2, 200, 1, 2},
        {3, 50,  3, 3},
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    for (const auto& [id, r] : results) {
        check(r.offset % 128 == 0, "offset aligned to 128");
    }
    std::cout << "  all offsets 128-aligned\n";
}

int main() {
    std::cout << "=== VTCM Lifetime-End Sort + Grouping Test ===\n";

    test_diamond_reuse();
    test_linear_chain_reuse();
    test_no_reuse_all_overlap();
    test_spill_on_overflow();
    test_identical_lifetime_grouping();
    test_query_and_empty();
    test_alignment();

    std::cout << "\n=== " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
