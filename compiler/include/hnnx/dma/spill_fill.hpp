#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace fa { class FancyAllocator; }

namespace hnnx {

// DMA Spill/Fill system: grdep_spillfill.cc, grdep_fillcombine.cc, insert_spillfill.cc
// dynamic_dma.cc, portable_dma.cc
// Op emitter: op_emitter at 0x1048320 (28264 bytes, 152694 chars decompiled)
// Inserts @Spill, @Fill, @MCSend, @MCRecvRdy, @MCRecvDone, @Spill with DB, @Fill with DB
// ChunkPreloadOp: DMA preload (serialize_oplist.cc:418)
//
// [反汇编证据 libHtpPrepare.so x86_64-linux-clang v2.48.40.260702, 逐指令, M36]
// DmaCheckpointSet/Wait 由 make_dma_checkpoint_op(y, j, b) @0xd95ac0 创建:
//   - b != 0 → Op::Op 后 vptr=0x5ec2488, [8]=j(u32), [0x10]=0, insert_op(...,false)
//              0x5ec2488 的 typeinfo = DmaCheckpointWait
//   - b == 0 → 同上但 vptr=0x5ec2568, typeinfo = DmaCheckpointSet
//   （注意：bool 参数选择的是 Wait！与旧注释相反，M36 证据 audit_verify/asm/
//    _ZN12GraphPrepare22make_dma_checkpoint_opEyjb...asm + vtable typeinfo 追踪）
//   serialize 槽位两 vtable 各不相同：Wait = 0xd96b30/0xd96b60/0xd96b70,
//   Set = 0xd96c70/0xd96ca0/0xd96cb0 (vtable +0/+8/+0x30)。
//   旧注 "serialize_internal 相同 (@0xd969a0)" 有误：0xd969a0 不在任何一方的槽位。
// 文档字符串 @0x039B2690: WAIT="Waits for a previously started DMA to complete
//   before proceeding" (逐字节验证); @0x039B26D0 实测为空串 (SET 无 doc 串)
enum class DmaOpType {
    Spill,
    Fill,
    SpillWithDB,    // Double-buffered spill
    FillWithDB,     // Double-buffered fill
    MCSend,
    MCRecvRdy,
    MCRecvDone,
    MCRecvDone2,
    ChunkPreload,
    MSyncPost,
    MSyncWait,
    HVXSpawn,       // fork/join
};

// DMA mode flags (Phase 4.2, from PortableDMA decompilation)
// Source: mid_level_ir.hpp:666-670, portable_dma.cc
constexpr uint16_t DMA_MODE_1D       = 0x0000;
constexpr uint16_t DMA_MODE_2D       = 0x0001;   // 2D strided transfer
constexpr uint16_t DMA_MODE_FILL    = 0x0010;   // DDR → VTCM
constexpr uint16_t DMA_MODE_SPILL   = 0x0020;   // VTCM → DDR
constexpr uint16_t DMA_PRIORITY_HIGH = 0x0100;   // high-priority DMA

struct DmaOpInfo {
    DmaOpType type;
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t size;
    uint32_t src_nsp;
    uint32_t dst_nsp;
    uint32_t mcid;
    uint64_t payload_size;
    bool is_multicast;
    bool double_buffered;
    // Phase 4.2: sync token + 2D stride (from PortableDMA @ portable_dma.cc)
    uint32_t synctoken_id = 0;  // 0 = no sync token (unsynchronized)
    uint16_t flags = 0;          // DMA_MODE_1D/2D/FILL/SPILL/PRIORITY
    uint16_t config = 0;        // priority, channel config
    // 2D transfer (only used when flags & DMA_MODE_2D):
    uint32_t src_stride = 0;
    uint32_t dst_stride = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// ===== SynctokenManager (Phase 4.2) =====
// Source: synctoken_manager.cc, make_dma_checkpoint_op @0xd95ac0 [M36 修正]
//
// Compile-time DMA sync token planner. The real QNN device uses sync tokens
// to coordinate DMA operations:
//   - DmaCheckpointSet: records a DMA completion tag in a table (producer)
//   - DmaCheckpointWait: waits for a previously started DMA (consumer)
//
// In the compiler, SynctokenManager:
//   1. Allocates unique token IDs (starting at 0x11, matching real .bin)
//   2. Tracks which DMA op SETS each token (producer position)
//   3. Tracks which ops WAIT for each token (consumer positions)
//   4. Validates that every WAIT has a preceding SET (signal_pos < wait_pos)
//   5. Groups related DMA ops under the same token (e.g., W+b share 0x11)
class SynctokenManager {
public:
    SynctokenManager();

    // Allocate a new sync token ID. IDs start at 0x11 (verified from .bin).
    uint32_t allocate();

    // Reuse an existing token ID (for grouping related DMAs, e.g., W+b).
    // No-op if the ID hasn't been allocated yet.
    void reuse(uint32_t id);

    // Record that a DMA op at position `pos` signals (SETs) token `id`.
    void signal(uint32_t id, size_t pos, const std::string& name = "");

    // Record that an op at position `pos` waits for token `id`.
    void wait(uint32_t id, size_t pos);

    // Validate: every wait has a preceding signal (signal_pos < wait_pos).
    // Returns false if any wait is unmatched or out of order.
    bool validate() const;

    // Token assignment record
    struct TokenEntry {
        uint32_t token_id = 0;
        size_t signal_pos = SIZE_MAX;        // position of the SET op
        std::string signal_name;            // human-readable (e.g., "DmaCheckpointSet(W)")
        std::vector<size_t> wait_positions;  // positions of WAIT ops
    };

    const std::vector<TokenEntry>& get_tokens() const { return tokens_; }

    // Statistics
    size_t token_count() const { return tokens_.size(); }
    uint32_t next_token_id() const { return next_id_; }

    // Reset to initial state
    void reset();

private:
    uint32_t next_id_ = 0x11;  // DMA tags start at 0x11 (verified from .bin)
    std::vector<TokenEntry> tokens_;

    TokenEntry* find_token(uint32_t id);
};

// Op emitter: inserts DMA ops into the op list
// Source: serialize_oplist.cc, grdep_spillfill.cc
class OpEmitter {
public:
    OpEmitter(GraphPrepare* gp);
    ~OpEmitter();

    // Insert a DMA op at a specific position
    void emit_dma_op(const DmaOpInfo& info, size_t position);

    // Insert ChunkPreloadOp (DMA preload)
    // Source: serialize_oplist.cc:418
    void insert_preload_op(size_t position, size_t prev_position);

    // Insert spill/fill pair
    void insert_spill_fill_pair(
        uint64_t vtcm_offset, uint64_t ddr_offset, uint64_t size,
        size_t spill_pos, size_t fill_pos, bool double_buffered);

    // Insert spill/fill pair with sync token (Phase 4.2)
    // The synctoken_id is set on both spill and fill ops; a DmaCheckpointSet
    // is emitted after the spill, and DmaCheckpointWait before the fill.
    void insert_spill_fill_pair_sync(
        uint64_t vtcm_offset, uint64_t ddr_offset, uint64_t size,
        size_t spill_pos, size_t fill_pos, bool double_buffered,
        uint32_t synctoken_id);

    // Insert 2D spill/fill pair (Phase 4.2)
    // For strided transfers (e.g., tiled data with non-contiguous rows).
    void insert_spill_fill_pair_2d(
        uint64_t vtcm_offset, uint64_t ddr_offset,
        uint32_t width, uint32_t height,
        uint32_t src_stride, uint32_t dst_stride,
        size_t spill_pos, size_t fill_pos, bool double_buffered,
        uint32_t synctoken_id = 0);

    // Insert multicast send/recv
    void insert_mcast_pair(
        uint32_t src_nsp, uint32_t dst_nsp, uint32_t mcid,
        uint64_t payload_size, size_t send_pos, size_t recv_pos);

    // Insert multicast send/recv with sync token (Phase 4.2)
    void insert_mcast_pair_sync(
        uint32_t src_nsp, uint32_t dst_nsp, uint32_t mcid,
        uint64_t payload_size, size_t send_pos, size_t recv_pos,
        uint32_t synctoken_id);

    // Insert DmaCheckpointSet/Wait (Phase 4.2)
    // SET: records DMA completion tag in table (producer side)
    // WAIT: waits for a previously started DMA (consumer side)
    // Source: make_dma_checkpoint_op @0xd95ac0 [M36 修正: vtable 0x5ec2488=Wait, 0x5ec2568=Set]
    void insert_dma_checkpoint_set(uint32_t synctoken_id, size_t pos,
                                   const std::string& name = "");
    void insert_dma_checkpoint_wait(uint32_t synctoken_id, size_t pos);

    // Insert sync operations
    void insert_msync_post(size_t pos);
    void insert_msync_wait(size_t pos);

    // HVX thread fork/join
    void insert_hvx_spawn(size_t fork_pos, size_t join_pos);

    // Spill/fill sanity check: grdep_sanity.cc
    bool validate_spill_fill() const;

    // Fill combine: grdep_fillcombine.cc - merge adjacent fills
    void combine_fills();

    // Source destructive: grdep_src_destructive.cc
    void link_source_destructive_operands(const std::vector<uint32_t>& tags);

    // Access emitted ops (for inspection/testing)
    const std::vector<DmaOpInfo>& get_emitted_ops() const { return emitted_ops_; }

    // Sync token manager access (Phase 4.2)
    SynctokenManager& synctoken_manager() { return synctoken_mgr_; }
    const SynctokenManager& synctoken_manager() const { return synctoken_mgr_; }

private:
    GraphPrepare* gp_;
    std::vector<DmaOpInfo> emitted_ops_;
    SynctokenManager synctoken_mgr_;  // Phase 4.2

    // Spill-fill buffer sizing
    uint64_t spill_fill_buffer_size_ = 0;
    uint64_t min_spill_fill_buffer_size() const;
};

// Spill/fill timing calculator
// Determines when to spill/fill based on VTCM pressure
class SpillFillScheduler {
public:
    SpillFillScheduler();
    ~SpillFillScheduler();

    // Calculate optimal spill/fill timing
    struct SpillFillPlan {
        size_t spill_position;
        size_t fill_position;
        uint64_t vtcm_offset;
        uint64_t ddr_offset;
        uint64_t size;
        bool double_buffered;
    };

    std::vector<SpillFillPlan> plan(
        const std::vector<Op*>& runlist,
        const fa::FancyAllocator& allocator,
        size_t vtcm_size);

    // Calculate minimum spill-fill buffer size
    // Source: "Minimum Required Spill-Fill Buffer Size is %llu"
    uint64_t min_buffer_size(
        const std::vector<Op*>& runlist,
        const fa::FancyAllocator& allocator) const;

private:
    // VTCM pressure tracking
    struct VtcmPressurePoint {
        size_t op_index;
        uint64_t current_usage;
        uint64_t peak_usage;
    };

    std::vector<VtcmPressurePoint> compute_pressure_curve(
        const std::vector<Op*>& runlist,
        const fa::FancyAllocator& allocator) const;
};

} // namespace hnnx
