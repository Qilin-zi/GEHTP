// Test: VTCM lifetime-aware first-fit reuse (allocate_with_lifetime)
// Validates FancyAllocator::allocate_with_lifetime against several graph
// topologies. Algorithm (T5 first-fit reuse):
//   1. Sort by life_begin ASC, life_end DESC, op_id ASC
//   2. Group by identical (life_begin, life_end)
//   3. Groups processed in life_begin order: dead ranges (life_end < current
//      life_begin) reclaimed into free_list; new group first-fits a free
//      range, else bumps; over budget spills the whole group.
//   4. Within group: sequential bump for members (must coexist)
//   5. Reuse safety asserted by validate_no_overlap (aborts on overlap)
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
//   Lifetimes (topo indices): A[0,0]  B[1,2]  C[1,2]  D[2,2]
//   Sort by life_begin ASC: A[0,0], B[1,2], C[1,2], D[2,2]
//   Groups: G1={A}, G2={B,C}, G3={D}
//   Placement (first-fit reuse):
//     G1: A@0            (bump),            active=[A]
//     G2: B@1024, C@2048 (A dead, 1024<2048 no fit → bump)
//     G3: D@0            (reuses A's freed [0,1024))
//   Used = 3072, saved = 1024 (D reuses A)
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

    // life_begin ASC: A[0,0] first → A at offset 0
    check(results[1].offset == 0, "A at offset 0 (first in life_begin order)");

    // B, C same group [1,2] → bump after A
    check(results[2].offset == 1024, "B at offset 1024");
    check(results[3].offset == 2048, "C at offset 2048 (within-group bump from B)");

    // D[2,2] reuses A's freed range [0,1024) (A dead: 0 < 2)
    check(results[4].offset == 0, "D at offset 0 (reuses A)");
    check(results[4].offset == results[1].offset, "D shares offset with A (non-overlapping)");

    check(alloc.total_vtcm_used() == 3 * 1024, "VTCM used = 3 blocks (D reuses A)");
    check(alloc.vtcm_saved_by_reuse() == 1024, "saved = 1024");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 2: Linear chain A→B→C→D.
//   Lifetimes: A[0,1], B[1,2], C[2,3], D[3,3]
//   Sort by life_begin ASC: A, B, C, D (each unique → 4 groups)
//   Placement (first-fit reuse):
//     A@0 (bump), B@1024 (A dead at le=1 == B.lb=1 → NOT reclaimed, strict <)
//     C@0 (reuses A: le=1 < lb=2), D@1024 (reuses B: le=2 < lb=3)
//   Used = 2048, saved = 2048 (C reuses A, D reuses B)
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

    check(results[1].offset == 0, "A at offset 0");
    check(results[2].offset == 1024, "B at offset 1024 (A not reclaimed: le=1 !< lb=1)");

    // C reuses A (A[0,1] dead: 1 < 2); D reuses B (B[1,2] dead: 2 < 3)
    check(results[3].offset == 0, "C at offset 0 (reuses A)");
    check(results[4].offset == 1024, "D at offset 1024 (reuses B)");
    check(results[3].offset == results[1].offset, "C shares A (non-overlapping)");
    check(results[4].offset == results[2].offset, "D shares B (non-overlapping)");

    // Adjacent lifetimes must not share: A[0,1] vs B[1,2] overlap-free but touching
    check(results[1].offset != results[2].offset, "A != B (touching at idx 1, strict <)");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
    check(alloc.total_vtcm_used() == 2 * 1024, "used = 2 blocks");
    check(alloc.vtcm_saved_by_reuse() == 2 * 1024, "saved = 2048");
}

// ---------------------------------------------------------------------------
// Test 3: All overlapping — every tensor has identical lifetime [0,5].
//   All in one group → within-group sequential bump (must coexist).
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
//   A[0,0] size 2048, B[1,1] size 512, C[1,1] size 512.
//   Groups: G1={A}, G2={B,C}. B and C reuse A's freed range.
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

    // G1={A}: A@0 (2048). G2={B,C}: A dead (le=0 < lb=1) → B,C reuse A's range.
    check(results[1].offset == 0, "A at offset 0");
    check(results[2].offset == 0, "B at offset 0 (reuses A)");
    check(results[3].offset == 512, "C at offset 512 (within-group bump from B)");
    check(results[2].offset != results[3].offset, "B and C different (same group, bump)");

    // B shares A's offset (A[0,0] and B[1,1] non-overlapping); C does not share B
    check(results[2].offset == results[1].offset, "B shares A (non-overlapping)");
    check(results[3].offset != results[1].offset, "C != A (within-group, must coexist)");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
    check(alloc.total_vtcm_used() == 2048, "used = 2048 (max of A vs B+C)");
    check(alloc.vtcm_saved_by_reuse() == 1024, "saved = 1024 (B reuses A)");
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

// ---------------------------------------------------------------------------
// Test 8: Maximal reuse — three non-overlapping tensors all share one offset.
//   A[0,0], B[1,1], C[2,2]: each dies before the next starts.
// ---------------------------------------------------------------------------
static void test_maximal_reuse() {
    std::cout << "\n[Test 8] Maximal reuse (chain of disjoint lifetimes)\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 1024, 0, 0},
        {2, 1024, 1, 1},
        {3, 1024, 2, 2},
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    // All pairwise non-overlapping → all share offset 0
    check(results[1].offset == 0, "A at offset 0");
    check(results[2].offset == 0, "B at offset 0 (reuses A)");
    check(results[3].offset == 0, "C at offset 0 (reuses B)");

    check(alloc.total_vtcm_used() == 1024, "peak = 1 block (maximal reuse)");
    check(alloc.vtcm_saved_by_reuse() == 2 * 1024, "saved = 2048");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 9: Overlapping lifetimes must NOT share.
//   A[0,3], B[1,2]: B lives inside A's range → must be placed apart.
// ---------------------------------------------------------------------------
static void test_overlap_no_share() {
    std::cout << "\n[Test 9] Overlapping lifetimes never share\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 1024, 0, 3},  // A: long-lived
        {2, 1024, 1, 2},  // B: lives inside A
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    check(results[1].offset == 0, "A at offset 0");
    check(results[2].offset == 1024, "B at offset 1024 (A alive, no reuse)");
    check(results[1].offset != results[2].offset, "A and B different offsets");

    check(alloc.total_vtcm_used() == 2 * 1024, "used = 2 blocks (both live)");
    check(alloc.vtcm_saved_by_reuse() == 0, "saved = 0");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 10: Free-list coalescing — two adjacent freed holes merge, letting a
//   big group fit that fits in neither alone.
//   A[0,1]512, B[1,2]512, C[2,2]512, D[3,3]1024.
//   Without coalescing: D can't fit in either 512 hole → bump @1024 (used=2048).
//   With coalescing: [0,512)+[512,1024) → [0,1024) → D@0 (used=1024).
// ---------------------------------------------------------------------------
static void test_freelist_coalescing() {
    std::cout << "\n[Test 10] Free-list coalescing (adjacent holes merge)\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 512,  0, 1},  // A
        {2, 512,  1, 2},  // B
        {3, 512,  2, 2},  // C
        {4, 1024, 3, 3},  // D: needs the merged hole
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    check(results[1].offset == 0,    "A at offset 0");
    check(results[2].offset == 512,  "B at offset 512 (A alive at lb=1)");
    check(results[3].offset == 0,    "C at offset 0 (reuses A)");
    check(results[4].offset == 0,    "D at offset 0 (merged [0,512)+[512,1024))");
    check(alloc.total_vtcm_used() == 1024, "used = 1024 (no false-overflow bump)");
    check(alloc.vtcm_saved_by_reuse() == 1536, "saved = 1536");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 11: Carve remainder persists while its donor is still alive, then
//   merges with a later free.
//   A[0,2]1024 (long-lived), B[1,1]512, C[2,3]256, D[3,4]256, E[4,4]512.
//   C@1024 carves B's hole (A alive → no merge); remainder [1280,1536)
//   survives; D@0 carves freed A; E@256 merges D-remainder [256,1024)
//   with freed C [1024,1280) and leftover [1280,1536) → [256,1536).
// ---------------------------------------------------------------------------
static void test_carve_remainder_and_triple_merge() {
    std::cout << "\n[Test 11] Carve remainder + merge across live tenant\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 1024, 0, 2},  // A: long-lived
        {2, 512,  1, 1},  // B
        {3, 256,  2, 3},  // C: carves B's hole while A alive
        {4, 256,  3, 4},  // D: carves freed A
        {5, 512,  4, 4},  // E: merges D-remainder + freed C + B-leftover
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    check(results[1].offset == 0,    "A at offset 0");
    check(results[2].offset == 1024, "B at offset 1024 (A alive)");
    check(results[3].offset == 1024, "C at offset 1024 (carves B; A alive → no merge)");
    check(results[4].offset == 0,    "D at offset 0 (carves freed A)");
    check(results[5].offset == 256,  "E at offset 256 (merged [256,1024)+[1024,1536))");
    check(alloc.total_vtcm_used() == 1536, "used = 1536");
    check(alloc.vtcm_saved_by_reuse() == 1024, "saved = 1024");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

// ---------------------------------------------------------------------------
// Test 12: Address-ordered first-fit — when several holes fit, the LOWEST
//   address wins regardless of reclaim/insertion order.
//   A[0,2]128, B[1,1]128, C[2,9]128, D[3,3]128, E[4,4]384, F[5,5]128, G[6,6]128.
//   At G: holes are [256,640) (E's, reclaimed first) and [0,128) (F's,
//   reclaimed second), separated by live C@128. Both fit 128; sorted
//   first-fit must pick [0,128), not the first-inserted [256,640).
// ---------------------------------------------------------------------------
static void test_first_fit_lowest_address() {
    std::cout << "\n[Test 12] First-fit picks lowest-address hole\n";
    std::vector<FancyAllocator::AllocRequest> reqs = {
        {1, 128, 0, 2},  // A
        {2, 128, 1, 1},  // B
        {3, 128, 2, 9},  // C: long-lived tenant between future holes
        {4, 128, 3, 3},  // D
        {5, 384, 4, 4},  // E: big, bumps to 256
        {6, 128, 5, 5},  // F
        {7, 128, 6, 6},  // G: two non-touching holes fit; must pick lowest
    };
    FancyAllocator alloc;
    auto results = alloc.allocate_with_lifetime(reqs, 1 << 20);

    check(results[3].offset == 128, "C at offset 128 (reuses B)");
    check(results[4].offset == 0,   "D at offset 0 (reuses A)");
    check(results[5].offset == 256, "E at offset 256 (bump past live C)");
    check(results[6].offset == 0,   "F at offset 0 (reuses D)");
    check(results[7].offset == 0,   "G at offset 0 (lowest hole, not first-inserted)");
    check(alloc.total_vtcm_used() == 640, "used = 640");

    std::cout << "  VTCM used: " << alloc.total_vtcm_used()
              << ", saved: " << alloc.vtcm_saved_by_reuse() << "\n";
}

int main() {
    std::cout << "=== VTCM Lifetime-Aware First-Fit Reuse Test ===\n";

    test_diamond_reuse();
    test_linear_chain_reuse();
    test_no_reuse_all_overlap();
    test_spill_on_overflow();
    test_identical_lifetime_grouping();
    test_query_and_empty();
    test_alignment();
    test_maximal_reuse();
    test_overlap_no_share();
    test_freelist_coalescing();
    test_carve_remainder_and_triple_merge();
    test_first_fit_lowest_address();

    std::cout << "\n=== " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
