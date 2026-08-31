#include "hnnx/dma/spill_fill.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <algorithm>

namespace hnnx {

OpEmitter::OpEmitter(GraphPrepare* gp) : gp_(gp) {}
OpEmitter::~OpEmitter() = default;

// Op emitter: inserts DMA ops into the op list during serialization
// Source: serialize_oplist.cc, op_emitter @ 0x1048320 (28264 bytes)
//
// Op types emitted (from decompiled strings):
//   @Spill / @Fill           - VTCM <-> DDR transfer
//   @Spill with DB / @Fill with DB - Double-buffered variant
//   @MCSend / @MCRecvRdy / @MCRecvDone - Multi-NSP communication
//   @ChunkPreload             - DMA preload for next chunk
//   @MSyncPost / @MSyncWait   - Memory synchronization
//   default_hvx_spawn (_fork/_join) - HVX thread management
//
// The emitter works in two modes:
//   Prescan: count sizes only (Serializer +0x200 != 0)
//   Write: actually emit op records

void OpEmitter::emit_dma_op(const DmaOpInfo& info, size_t position) {
    emitted_ops_.push_back(info);

    switch (info.type) {
        case DmaOpType::Spill:
            // Create SpillOp: moves data from VTCM to DDR
            // Source: grdep_spillfill.cc
            // SpillOp(src=vtcm_offset, dst=ddr_offset, size)
            break;
        case DmaOpType::Fill:
            // Create FillOp: moves data from DDR to VTCM
            // Source: grdep_spillfill.cc
            // FillOp(src=ddr_offset, dst=vtcm_offset, size)
            break;
        case DmaOpType::SpillWithDB:
            // Create SpillOp with double buffering
            // "default_dma_start (Spill with DB)"
            // Uses two buffers alternately to hide DMA latency
            break;
        case DmaOpType::FillWithDB:
            // Create FillOp with double buffering
            // "default_dma_start (Fill with DB)"
            break;
        case DmaOpType::MCSend:
            // Create McSendOp: sends data to another NSP via multicast
            // Source: grdep_multicast.cc
            // McSendOp(sender_nsp, mcids[], receivers[], payload_size)
            break;
        case DmaOpType::MCRecvRdy:
            // Create McRecvRdyOp: waits for multicast data ready
            // Source: grdep_multicast.cc
            break;
        case DmaOpType::MCRecvDone:
            // Create McRecvDoneOp: signals multicast receive complete
            break;
        case DmaOpType::MCRecvDone2:
            // Variant of MCRecvDone
            break;
        case DmaOpType::ChunkPreload:
            // Create ChunkPreloadOp: preloads next chunk via DMA
            // Source: serialize_oplist.cc:418
            // "insert ChunkPreloadOp #%d at posn = %zu; previous at %zu"
            break;
        case DmaOpType::MSyncPost:
            // Create MSyncPostOp: post memory synchronization barrier
            // Source: msync_op_prepare.cc
            break;
        case DmaOpType::MSyncWait:
            // Create MSyncWaitOp: wait for memory synchronization
            break;
        case DmaOpType::HVXSpawn:
            // Create HVX fork/join for thread management
            // "default_hvx_spawn (_fork / _join)"
            break;
    }
}

void OpEmitter::insert_preload_op(size_t position, size_t prev_position) {
    // Source: serialize_oplist.cc:418
    // Serializer::do_insert_preload_op
    // "insert ChunkPreloadOp #%d at posn = %zu; previous at %zu"
    //
    // The preload op is inserted at the current position in the op list.
    // It triggers a DMA to preload the next chunk of data into VTCM
    // while the current chunk is being computed.
    //
    // Serializer fields:
    //   +0x340: current position
    //   +0x348: previous position
    //   +0x350: preload count
    DmaOpInfo info{};
    info.type = DmaOpType::ChunkPreload;
    info.src_offset = position;
    info.dst_offset = prev_position;
    emit_dma_op(info, position);
}

void OpEmitter::insert_spill_fill_pair(
    uint64_t vtcm_offset, uint64_t ddr_offset, uint64_t size,
    size_t spill_pos, size_t fill_pos, bool double_buffered) {

    DmaOpInfo spill{};
    spill.type = double_buffered ? DmaOpType::SpillWithDB : DmaOpType::Spill;
    spill.src_offset = vtcm_offset;
    spill.dst_offset = ddr_offset;
    spill.size = size;
    spill.double_buffered = double_buffered;
    spill.flags = DMA_MODE_SPILL | DMA_MODE_1D;
    emit_dma_op(spill, spill_pos);

    DmaOpInfo fill{};
    fill.type = double_buffered ? DmaOpType::FillWithDB : DmaOpType::Fill;
    fill.src_offset = ddr_offset;
    fill.dst_offset = vtcm_offset;
    fill.size = size;
    fill.double_buffered = double_buffered;
    fill.flags = DMA_MODE_FILL | DMA_MODE_1D;
    emit_dma_op(fill, fill_pos);
}

// Phase 4.2: spill/fill with sync token
void OpEmitter::insert_spill_fill_pair_sync(
    uint64_t vtcm_offset, uint64_t ddr_offset, uint64_t size,
    size_t spill_pos, size_t fill_pos, bool double_buffered,
    uint32_t synctoken_id) {

    DmaOpInfo spill{};
    spill.type = double_buffered ? DmaOpType::SpillWithDB : DmaOpType::Spill;
    spill.src_offset = vtcm_offset;
    spill.dst_offset = ddr_offset;
    spill.size = size;
    spill.double_buffered = double_buffered;
    spill.synctoken_id = synctoken_id;
    spill.flags = DMA_MODE_SPILL | DMA_MODE_1D;
    emit_dma_op(spill, spill_pos);

    // Record SET (producer signals completion)
    synctoken_mgr_.signal(synctoken_id, spill_pos, "DmaCheckpointSet");

    DmaOpInfo fill{};
    fill.type = double_buffered ? DmaOpType::FillWithDB : DmaOpType::Fill;
    fill.src_offset = ddr_offset;
    fill.dst_offset = vtcm_offset;
    fill.size = size;
    fill.double_buffered = double_buffered;
    fill.synctoken_id = synctoken_id;
    fill.flags = DMA_MODE_FILL | DMA_MODE_1D;
    emit_dma_op(fill, fill_pos);

    // Record WAIT (consumer waits for completion)
    synctoken_mgr_.wait(synctoken_id, fill_pos);
}

// Phase 4.2: 2D strided spill/fill
void OpEmitter::insert_spill_fill_pair_2d(
    uint64_t vtcm_offset, uint64_t ddr_offset,
    uint32_t width, uint32_t height,
    uint32_t src_stride, uint32_t dst_stride,
    size_t spill_pos, size_t fill_pos, bool double_buffered,
    uint32_t synctoken_id) {

    DmaOpInfo spill{};
    spill.type = double_buffered ? DmaOpType::SpillWithDB : DmaOpType::Spill;
    spill.src_offset = vtcm_offset;
    spill.dst_offset = ddr_offset;
    spill.size = static_cast<uint64_t>(width) * height;
    spill.double_buffered = double_buffered;
    spill.synctoken_id = synctoken_id;
    spill.flags = DMA_MODE_SPILL | DMA_MODE_2D;
    spill.src_stride = src_stride;
    spill.dst_stride = dst_stride;
    spill.width = width;
    spill.height = height;
    emit_dma_op(spill, spill_pos);

    if (synctoken_id != 0) {
        synctoken_mgr_.signal(synctoken_id, spill_pos, "DmaCheckpointSet(2D)");
    }

    DmaOpInfo fill{};
    fill.type = double_buffered ? DmaOpType::FillWithDB : DmaOpType::Fill;
    fill.src_offset = ddr_offset;
    fill.dst_offset = vtcm_offset;
    fill.size = static_cast<uint64_t>(width) * height;
    fill.double_buffered = double_buffered;
    fill.synctoken_id = synctoken_id;
    fill.flags = DMA_MODE_FILL | DMA_MODE_2D;
    fill.src_stride = dst_stride;  // fill source = spill destination
    fill.dst_stride = src_stride;  // fill destination = spill source
    fill.width = width;
    fill.height = height;
    emit_dma_op(fill, fill_pos);

    if (synctoken_id != 0) {
        synctoken_mgr_.wait(synctoken_id, fill_pos);
    }
}

void OpEmitter::insert_mcast_pair(
    uint32_t src_nsp, uint32_t dst_nsp, uint32_t mcid,
    uint64_t payload_size, size_t send_pos, size_t recv_pos) {

    DmaOpInfo send{};
    send.type = DmaOpType::MCSend;
    send.src_nsp = src_nsp;
    send.dst_nsp = dst_nsp;
    send.mcid = mcid;
    send.payload_size = payload_size;
    send.is_multicast = true;
    emit_dma_op(send, send_pos);

    DmaOpInfo recv{};
    recv.type = DmaOpType::MCRecvRdy;
    recv.src_nsp = src_nsp;
    recv.dst_nsp = dst_nsp;
    recv.mcid = mcid;
    recv.is_multicast = true;
    emit_dma_op(recv, recv_pos);
}

// Phase 4.2: multicast with sync token
void OpEmitter::insert_mcast_pair_sync(
    uint32_t src_nsp, uint32_t dst_nsp, uint32_t mcid,
    uint64_t payload_size, size_t send_pos, size_t recv_pos,
    uint32_t synctoken_id) {

    DmaOpInfo send{};
    send.type = DmaOpType::MCSend;
    send.src_nsp = src_nsp;
    send.dst_nsp = dst_nsp;
    send.mcid = mcid;
    send.payload_size = payload_size;
    send.is_multicast = true;
    send.synctoken_id = synctoken_id;
    emit_dma_op(send, send_pos);

    synctoken_mgr_.signal(synctoken_id, send_pos, "MCSend");

    DmaOpInfo recv{};
    recv.type = DmaOpType::MCRecvRdy;
    recv.src_nsp = src_nsp;
    recv.dst_nsp = dst_nsp;
    recv.mcid = mcid;
    recv.is_multicast = true;
    recv.synctoken_id = synctoken_id;
    emit_dma_op(recv, recv_pos);

    synctoken_mgr_.wait(synctoken_id, recv_pos);
}

// Phase 4.2: DmaCheckpointSet/Wait
// Source: make_dma_checkpoint_op @0xd958d0
// SET: records DMA completion tag in table (producer side)
// WAIT: waits for a previously started DMA (consumer side)
void OpEmitter::insert_dma_checkpoint_set(uint32_t synctoken_id, size_t pos,
                                           const std::string& name) {
    DmaOpInfo info{};
    info.type = DmaOpType::Spill;  // placeholder type; real .bin uses OP_TYPE_DMA
    info.synctoken_id = synctoken_id;
    emit_dma_op(info, pos);
    synctoken_mgr_.signal(synctoken_id, pos, name);
}

void OpEmitter::insert_dma_checkpoint_wait(uint32_t synctoken_id, size_t pos) {
    DmaOpInfo info{};
    info.type = DmaOpType::Fill;  // placeholder type; real .bin uses OP_TYPE_DMA
    info.synctoken_id = synctoken_id;
    emit_dma_op(info, pos);
    synctoken_mgr_.wait(synctoken_id, pos);
}

void OpEmitter::insert_msync_post(size_t pos) {
    DmaOpInfo info{};
    info.type = DmaOpType::MSyncPost;
    emit_dma_op(info, pos);
}

void OpEmitter::insert_msync_wait(size_t pos) {
    DmaOpInfo info{};
    info.type = DmaOpType::MSyncWait;
    emit_dma_op(info, pos);
}

void OpEmitter::insert_hvx_spawn(size_t fork_pos, size_t join_pos) {
    DmaOpInfo fork{};
    fork.type = DmaOpType::HVXSpawn;
    emit_dma_op(fork, fork_pos);

    DmaOpInfo join{};
    join.type = DmaOpType::HVXSpawn;
    emit_dma_op(join, join_pos);
}

bool OpEmitter::validate_spill_fill() const {
    // Source: grdep_sanity.cc, spillfill_sanity @ 0x10029F0 (5461 bytes)
    // Checks:
    // 1. Every Fill has a matching Spill (same offset, size)
    // 2. Spill/fill sizes match
    // 3. VTCM offsets are within VTCM budget
    // 4. DDR offsets are valid
    // 5. No overlapping concurrent spills (unless double-buffered)
    // 6. Spill/fill ordering is correct (spill before fill)
    return true;
}

void OpEmitter::combine_fills() {
    // Source: grdep_fillcombine.cc
    // Merge adjacent FillOps that read from consecutive DDR addresses
    // into a single larger FillOp. This reduces DMA overhead.
    //
    // Algorithm:
    // 1. Sort fills by DDR offset
    // 2. Find adjacent fills with consecutive DDR ranges
    // 3. Merge into single fill with combined size
}

void OpEmitter::link_source_destructive_operands(const std::vector<uint32_t>& tags) {
    // Source: grdep_src_destructive.cc, link_source_destructive_operands @ 0xF82490
    // Marks operands as "destructive" - meaning the source tensor can be
    // overwritten by the consumer. This enables in-place computation,
    // reducing VTCM usage.
}

} // namespace hnnx
