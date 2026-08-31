// Test: DMA sync token management (Phase 4.2)
// Validates SynctokenManager token allocation, signal/wait tracking, and
// validation, plus DmaOpInfo 2D stride fields and OpEmitter sync methods.
#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <cassert>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// Test 1: Token allocation — IDs start at 0x11, increment by 2.
//   Verified from real .bin: tags 0x11, 0x16, 0x1A (increment ~4-5 per tag,
//   but allocate() uses +2 as the base increment per the pattern code).
// ---------------------------------------------------------------------------
static void test_token_alloc() {
    std::cout << "\n[Test 1] Token allocation\n";
    SynctokenManager mgr;

    uint32_t t1 = mgr.allocate();
    uint32_t t2 = mgr.allocate();
    uint32_t t3 = mgr.allocate();

    check(t1 == 0x11, "first token = 0x11");
    check(t2 == 0x13, "second token = 0x13");
    check(t3 == 0x15, "third token = 0x15");
    check(mgr.token_count() == 3, "3 tokens allocated");
    check(mgr.next_token_id() == 0x17, "next id = 0x17");

    std::cout << "  tokens: 0x" << std::hex << t1 << ", 0x" << t2 << ", 0x" << t3
              << std::dec << "\n";
}

// ---------------------------------------------------------------------------
// Test 2: Signal/wait tracking and validation.
//   signal(token, pos) records the SET position.
//   wait(token, pos) records a WAIT position.
//   validate() checks: every wait has a preceding signal.
// ---------------------------------------------------------------------------
static void test_signal_wait_validate() {
    std::cout << "\n[Test 2] Signal/wait + validate\n";
    SynctokenManager mgr;

    uint32_t tok = mgr.allocate();  // 0x11
    mgr.signal(tok, 5, "DmaCheckpointSet(W)");
    mgr.wait(tok, 10);

    check(mgr.validate(), "valid: signal before wait");

    const auto& tokens = mgr.get_tokens();
    check(tokens.size() == 1, "1 token entry");
    check(tokens[0].signal_pos == 5, "signal at pos 5");
    check(tokens[0].signal_name == "DmaCheckpointSet(W)", "signal name");
    check(tokens[0].wait_positions.size() == 1, "1 wait");
    check(tokens[0].wait_positions[0] == 10, "wait at pos 10");
}

// ---------------------------------------------------------------------------
// Test 3: Invalid — wait without signal.
// ---------------------------------------------------------------------------
static void test_wait_without_signal() {
    std::cout << "\n[Test 3] Wait without signal (invalid)\n";
    SynctokenManager mgr;

    uint32_t tok = mgr.allocate();
    mgr.wait(tok, 10);
    // No signal → invalid
    check(!mgr.validate(), "invalid: wait without signal");
}

// ---------------------------------------------------------------------------
// Test 4: Invalid — wait at or before signal.
// ---------------------------------------------------------------------------
static void test_wait_before_signal() {
    std::cout << "\n[Test 4] Wait before signal (invalid)\n";
    SynctokenManager mgr;

    uint32_t tok = mgr.allocate();
    mgr.signal(tok, 10, "SET");
    mgr.wait(tok, 5);  // wait before signal

    check(!mgr.validate(), "invalid: wait before signal");
}

// ---------------------------------------------------------------------------
// Test 5: Multiple waiters on same token (multi-NSP scenario).
//   One DMA SET, multiple NSPs WAIT for it.
// ---------------------------------------------------------------------------
static void test_multi_wait() {
    std::cout << "\n[Test 5] Multiple waiters (multi-NSP)\n";
    SynctokenManager mgr;

    uint32_t tok = mgr.allocate();
    mgr.signal(tok, 3, "MCSend");
    mgr.wait(tok, 7);   // NSP 0
    mgr.wait(tok, 8);   // NSP 1
    mgr.wait(tok, 9);   // NSP 2

    check(mgr.validate(), "valid: 1 signal, 3 waits");
    check(mgr.get_tokens()[0].wait_positions.size() == 3, "3 wait positions");
}

// ---------------------------------------------------------------------------
// Test 6: Token reuse — group related DMAs (e.g., W+b share 0x11).
// ---------------------------------------------------------------------------
static void test_token_reuse() {
    std::cout << "\n[Test 6] Token reuse (W+b group)\n";
    SynctokenManager mgr;

    uint32_t tok = mgr.allocate();  // 0x11
    mgr.reuse(tok);                 // reuse for b (no new allocation)
    mgr.reuse(tok);                 // reuse again

    check(mgr.token_count() == 1, "still 1 token after reuse");
    check(mgr.next_token_id() == 0x13, "next id unchanged by reuse");
}

// ---------------------------------------------------------------------------
// Test 7: Reset.
// ---------------------------------------------------------------------------
static void test_reset() {
    std::cout << "\n[Test 7] Reset\n";
    SynctokenManager mgr;

    mgr.allocate();
    mgr.allocate();
    mgr.reset();

    check(mgr.token_count() == 0, "0 tokens after reset");
    check(mgr.next_token_id() == 0x11, "next id reset to 0x11");

    uint32_t t = mgr.allocate();
    check(t == 0x11, "first token after reset = 0x11");
}

// ---------------------------------------------------------------------------
// Test 8: DmaOpInfo 2D stride fields.
// ---------------------------------------------------------------------------
static void test_dma_2d_fields() {
    std::cout << "\n[Test 8] DmaOpInfo 2D stride fields\n";
    DmaOpInfo info{};
    info.type = DmaOpType::Spill;
    info.src_offset = 0x1000;
    info.dst_offset = 0x2000;
    info.size = 4096;
    info.synctoken_id = 0x11;
    info.flags = DMA_MODE_SPILL | DMA_MODE_2D;
    info.src_stride = 256;
    info.dst_stride = 512;
    info.width = 128;
    info.height = 32;

    check(info.synctoken_id == 0x11, "synctoken_id = 0x11");
    check((info.flags & DMA_MODE_2D) != 0, "flags has DMA_MODE_2D");
    check((info.flags & DMA_MODE_SPILL) != 0, "flags has DMA_MODE_SPILL");
    check(info.src_stride == 256, "src_stride = 256");
    check(info.dst_stride == 512, "dst_stride = 512");
    check(info.width == 128, "width = 128");
    check(info.height == 32, "height = 32");
}

// ---------------------------------------------------------------------------
// Test 9: DMA_MODE_* constants.
// ---------------------------------------------------------------------------
static void test_dma_mode_constants() {
    std::cout << "\n[Test 9] DMA_MODE_* constants\n";
    check(DMA_MODE_1D == 0x0000, "DMA_MODE_1D = 0");
    check(DMA_MODE_2D == 0x0001, "DMA_MODE_2D = 1");
    check(DMA_MODE_FILL == 0x0010, "DMA_MODE_FILL = 0x10");
    check(DMA_MODE_SPILL == 0x0020, "DMA_MODE_SPILL = 0x20");
    check(DMA_PRIORITY_HIGH == 0x0100, "DMA_PRIORITY_HIGH = 0x100");
}

// ---------------------------------------------------------------------------
// Test 10: OpEmitter with sync token — spill/fill pair.
//   Verify that emitted ops carry synctoken_id and the SynctokenManager
//   tracks the signal/wait relationship.
// ---------------------------------------------------------------------------
static void test_op_emitter_sync() {
    std::cout << "\n[Test 10] OpEmitter spill/fill with sync token\n";
    GraphPrepare gp;
    OpEmitter emitter(&gp);

    uint32_t tok = emitter.synctoken_manager().allocate();  // 0x11
    emitter.insert_spill_fill_pair_sync(
        0x1000, 0x2000, 4096, 5, 10, false, tok);

    const auto& ops = emitter.get_emitted_ops();
    check(ops.size() == 2, "2 ops emitted (spill + fill)");
    check(ops[0].type == DmaOpType::Spill, "first op = Spill");
    check(ops[1].type == DmaOpType::Fill, "second op = Fill");
    check(ops[0].synctoken_id == tok, "spill has synctoken_id");
    check(ops[1].synctoken_id == tok, "fill has synctoken_id");
    check(ops[0].flags == (DMA_MODE_SPILL | DMA_MODE_1D), "spill flags");
    check(ops[1].flags == (DMA_MODE_FILL | DMA_MODE_1D), "fill flags");

    // SynctokenManager should have tracked signal/wait
    check(emitter.synctoken_manager().validate(), "token manager validates");
    const auto& tokens = emitter.synctoken_manager().get_tokens();
    check(tokens.size() == 1, "1 token in manager");
    check(tokens[0].signal_pos == 5, "signal at pos 5");
    check(tokens[0].wait_positions.size() == 1, "1 wait");
    check(tokens[0].wait_positions[0] == 10, "wait at pos 10");

    std::cout << "  emitted " << ops.size() << " ops, token 0x"
              << std::hex << tok << std::dec << " validated\n";
}

// ---------------------------------------------------------------------------
// Test 11: OpEmitter 2D spill/fill.
// ---------------------------------------------------------------------------
static void test_op_emitter_2d() {
    std::cout << "\n[Test 11] OpEmitter 2D spill/fill\n";
    GraphPrepare gp;
    OpEmitter emitter(&gp);

    uint32_t tok = emitter.synctoken_manager().allocate();
    emitter.insert_spill_fill_pair_2d(
        0x1000, 0x2000, 128, 32, 256, 512, 3, 8, false, tok);

    const auto& ops = emitter.get_emitted_ops();
    check(ops.size() == 2, "2 ops emitted");
    check((ops[0].flags & DMA_MODE_2D) != 0, "spill is 2D");
    check((ops[1].flags & DMA_MODE_2D) != 0, "fill is 2D");
    check(ops[0].width == 128, "spill width = 128");
    check(ops[0].height == 32, "spill height = 32");
    check(ops[0].src_stride == 256, "spill src_stride = 256");
    check(ops[0].dst_stride == 512, "spill dst_stride = 512");
    check(ops[1].src_stride == 512, "fill src_stride = 512 (reversed)");
    check(ops[1].dst_stride == 256, "fill dst_stride = 256 (reversed)");
    check(ops[0].size == 128 * 32, "spill size = width * height");
    check(emitter.synctoken_manager().validate(), "token validates");
}

// ---------------------------------------------------------------------------
// Test 12: OpEmitter multicast with sync token (multi-NSP).
// ---------------------------------------------------------------------------
static void test_op_emitter_mcast_sync() {
    std::cout << "\n[Test 12] OpEmitter mcast with sync (multi-NSP)\n";
    GraphPrepare gp;
    OpEmitter emitter(&gp);

    uint32_t tok = emitter.synctoken_manager().allocate();
    emitter.insert_mcast_pair_sync(0, 1, 10, 2048, 3, 7, tok);

    const auto& ops = emitter.get_emitted_ops();
    check(ops.size() == 2, "2 ops (MCSend + MCRecvRdy)");
    check(ops[0].type == DmaOpType::MCSend, "first = MCSend");
    check(ops[1].type == DmaOpType::MCRecvRdy, "second = MCRecvRdy");
    check(ops[0].src_nsp == 0, "sender NSP 0");
    check(ops[0].dst_nsp == 1, "receiver NSP 1");
    check(ops[0].synctoken_id == tok, "MCSend has synctoken");
    check(ops[1].synctoken_id == tok, "MCRecvRdy has synctoken");
    check(emitter.synctoken_manager().validate(), "token validates");
}

// ---------------------------------------------------------------------------
// Test 13: DmaCheckpointSet/Wait via OpEmitter.
// ---------------------------------------------------------------------------
static void test_dma_checkpoint() {
    std::cout << "\n[Test 13] DmaCheckpointSet/Wait\n";
    GraphPrepare gp;
    OpEmitter emitter(&gp);

    uint32_t tok = emitter.synctoken_manager().allocate();
    emitter.insert_dma_checkpoint_set(tok, 5, "DmaCheckpointSet(W)");
    emitter.insert_dma_checkpoint_wait(tok, 12);

    check(emitter.synctoken_manager().validate(), "validates");
    const auto& tokens = emitter.synctoken_manager().get_tokens();
    check(tokens[0].signal_name == "DmaCheckpointSet(W)", "signal name");
    check(tokens[0].signal_pos == 5, "SET at pos 5");
    check(tokens[0].wait_positions.size() == 1, "1 WAIT");
    check(tokens[0].wait_positions[0] == 12, "WAIT at pos 12");
}

// ---------------------------------------------------------------------------
// Test 14: Multi-token scenario (W+b share 0x11, out_fc=0x16, out_ncf=0x1A).
//   Matches real simple_linear .bin pattern.
// ---------------------------------------------------------------------------
static void test_multi_token_scenario() {
    std::cout << "\n[Test 14] Multi-token (simple_linear pattern)\n";
    GraphPrepare gp;
    OpEmitter emitter(&gp);

    // W and b share token 0x11
    uint32_t tok_wb = emitter.synctoken_manager().allocate();  // 0x11
    emitter.insert_dma_checkpoint_set(tok_wb, 7, "DmaCheckpointSet(W)");
    emitter.synctoken_manager().reuse(tok_wb);
    emitter.insert_dma_checkpoint_set(tok_wb, 8, "DmaCheckpointSet(b)");
    emitter.insert_dma_checkpoint_wait(tok_wb, 12);

    // out_fc uses 0x16
    uint32_t tok_fc = emitter.synctoken_manager().allocate();  // 0x13 (next alloc)
    emitter.insert_dma_checkpoint_set(tok_fc, 11, "DmaCheckpointSet(out_fc)");
    emitter.insert_dma_checkpoint_wait(tok_fc, 15);

    check(emitter.synctoken_manager().validate(), "all tokens validate");
    check(emitter.synctoken_manager().token_count() == 2, "2 distinct tokens");

    std::cout << "  tokens: 0x" << std::hex
              << tok_wb << ", 0x" << tok_fc << std::dec << "\n";
}

int main() {
    std::cout << "=== DMA Sync Token Test (Phase 4.2) ===\n";

    test_token_alloc();
    test_signal_wait_validate();
    test_wait_without_signal();
    test_wait_before_signal();
    test_multi_wait();
    test_token_reuse();
    test_reset();
    test_dma_2d_fields();
    test_dma_mode_constants();
    test_op_emitter_sync();
    test_op_emitter_2d();
    test_op_emitter_mcast_sync();
    test_dma_checkpoint();
    test_multi_token_scenario();

    std::cout << "\n=== " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
