#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>

#include "hnnx/ir/op_id.hpp" // hnnx::op_id_t (原经 types.hpp 传递)

// Op 双世界桥 (同 ser_ops_interface.hpp): 真身为全局类 (mangling
// _ZN2fa14FancyAllocator20allow_tensor_overlapEPK2Op)。旧族 TU 用 hnnx::Op ——
// 不引入全局前置, 免 `using namespace hnnx` 歧义。注: 旧族定义 TU
// (src/vtcm/fancy_allocator.cpp) 导出符号暂为 PKN4hnnx2OpE, M35 fa 真层级
// 统一时回归 PK2Op。
#if defined(HNNX_SER_PRECISE)
class Op;
#define HNNX_OP_T ::Op
#else
namespace hnnx {
class Op;
} // namespace hnnx
#define HNNX_OP_T hnnx::Op
#endif
namespace hnnx {
class VtcmCacheInstance;
class GraphDeps;
class OpDef;
class Serializer; // _ZNK2fa14FancyAllocator15setup_heap_infoERN4hnnx10SerializerEm
} // namespace hnnx
// (types.php 依赖移除: 本头亦须可入精确族 TU; 唯一 types 产物 hnnx::OpDef 改前置,
//  旧族 TU 经 graph_deps.hpp 等先含 types.hpp 而得完整定义)

namespace fa {

// fa14::FancyAllocator - VTCM memory allocator
// Source: vtcm_alloc.cc, fa_alloc.cc (libHtpPrepare.so x86_64)
//
// Memory reuse has two mechanisms (both verified from disassembly):
//
// 1. allow_tensor_overlap (per-Op producer-consumer reuse)
//    反汇编 @ 0x13a14b0 (1157B)
//    Called per-Op in topological order. Enumerates the Op's input tensor
//    block IDs, calls force_contiguous to register/allocate blocks in a
//    Fibonacci hash table, then calls link_blocks to connect the first
//    input's block to the output's block — enabling producer-consumer
//    VTCM reuse (output overwrites input's VTCM after input is consumed).
//
// 2. allocate_with_lifetime (batch lifetime-based reuse)
//    反汇编路径:
//      set_lifetimes          @ 0xf49960 (interval_set 构建)
//      allocate_tcm_blocks_internal @ 0x13b29d0
//      sort_blocks_by_reverse_lifetime_end @ 0x13b21b0
//      allow_tensor_overlap   @ 0x13a14b0 (组内顺序 bump)
//    Algorithm:
//      (a) 按 life_end 降序、life_begin 升序、op_id 升序排序
//          (sort_blocks_by_reverse_lifetime_end)
//      (b) (life_begin, life_end) 完全相同的连续 tensor 归入同一组
//          (allocate_tcm_blocks_internal 主循环 @ 0x13b3c40)
//      (c) 组内: 顺序 bump 分配 offset (同生命周期必须共存)
//      (d) 组间: 事件扫描复用 life_end < 当前组 life_begin 的已过期区域
//      (e) 超预算 → spill 到 DDR

class FancyAllocator {
public:
    FancyAllocator();
    ~FancyAllocator();

    // ===== Lifetime-aware allocation (反汇编路径 allocate_tcm_blocks_internal) =====
    // Allocates VTCM for a batch of tensors using lifetime-end sort + identical
    // interval grouping + event-sweep reuse. Tensors are sorted by life_end
    // descending (reverse_lifetime_end), then grouped by identical (life_begin,
    // life_end). Within a group, members are sequentially bump-allocated (they
    // are co-live and cannot share offsets). Across groups, regions whose
    // life_end < current group's life_begin are reusable (event-sweep).
    // Returns a map op_id -> {offset, block_id, spilled}.
    // Tensors that don't fit the budget get spilled (spilled=true).
    struct AllocRequest {
        hnnx::op_id_t op_id;
        size_t size;          // tensor size in bytes
        uint32_t life_begin; // first use (topological index)
        uint32_t life_end;   // last use (topological index)
    };
    struct AllocResult {
        uint64_t offset = 0;      // VTCM byte offset (only valid if !spilled)
        uint32_t block_id = 0;    // block index (for .bin block_ref field)
        bool spilled = false;     // true if placed on DDR instead of VTCM
    };
    std::unordered_map<hnnx::op_id_t, AllocResult>
    allocate_with_lifetime(const std::vector<AllocRequest>& requests,
                           size_t vtcm_budget, size_t alignment = 128);

    // Query a single op's allocation result (returns nullptr if not allocated).
    const AllocResult* get_allocation(hnnx::op_id_t op_id) const;

    // Statistics: actual VTCM bytes consumed (excludes reuse-saved space).
    uint64_t total_vtcm_used() const { return lifetime_used_; }
    // Bytes saved by lifetime reuse = (sum of all tensor sizes) - total_vtcm_used.
    // Equals 0 when no reuse happened (pure bump).
    uint64_t vtcm_saved_by_reuse() const { return lifetime_saved_; }

    // Overlap allocation results (populated by allow_tensor_overlap_opdef)
    struct OverlapAlloc {
        uint64_t offset = 0;     // VTCM byte offset
        uint32_t block_id = 0;   // block ID
        bool spilled = false;     // true if placed on DDR
        bool reused = false;      // true if VTCM offset shared with another block
        uint32_t linked_to = 0;   // block_id this is linked to (0 = none)
    };
    const std::unordered_map<hnnx::op_id_t, OverlapAlloc>& overlap_allocs() const {
        return overlap_allocs_;
    }
    uint64_t overlap_vtcm_used() const { return overlap_next_offset_; }
    uint64_t overlap_saved() const { return overlap_saved_; }

    // Core allocation (legacy bump allocator, kept for compatibility)
    void* allocate(size_t size, size_t alignment);
    void deallocate(void* ptr);

    // ===== Per-Op producer-consumer reuse (反汇编 @ 0x13a14b0, 1157B) =====
    // Called per-Op in topological order. Enumerates the Op's input/output
    // tensor block IDs, calls force_contiguous + link_blocks to enable
    // VTCM reuse between producer and consumer tensors.
    // If mode_ != 0 (serialization/prescan), returns immediately.
    void allow_tensor_overlap(const HNNX_OP_T* op); // .so: PK2Op (全局)

    // REQNN-adapted version: takes OpDef + GraphDeps instead of binary Op*.
    // Implements the same algorithm as the disassembled allow_tensor_overlap:
    //   1. Enumerate input block IDs from OpDef::inputs (src_op_id for each input)
    //   2. Allocate output block via force_contiguous
    //   3. link_blocks(first_input_block, output_block) for producer-consumer reuse
    void allow_tensor_overlap_opdef(const hnnx::OpDef& opdef,
                                    const hnnx::GraphDeps& deps,
                                    uint32_t topo_idx);

    // force_contiguous: register/allocate VTCM blocks for a batch of tensors.
    // 反汇编 @ 0x13a01b0 (2311B)
    // For each tensor: extracts block_id, computes Fibonacci hash (×0x740f1de9),
    // probes open-addressing hash table, registers or reuses blocks based on
    // lifetime overlap. Returns the primary (first) block_id.
    uint32_t force_contiguous(const uint32_t* block_ids, size_t count,
                              const uint32_t* sizes,
                              const uint32_t* life_begins,
                              const uint32_t* life_ends);

    // link_blocks: establish reuse link between two blocks.
    // 反汇编 @ 0x13a0ac0 (2542B)
    // If id1 == id2, return. If same size: share VTCM offset.
    // If different sizes: larger block's offset shared with smaller.
    // Adds freed blocks to free list for future reuse.
    void link_blocks(uint32_t id1, uint32_t id2);

    // VTCM-specific
    void setup_heap_info(hnnx::Serializer& ser, uint64_t total_size);
    void* get_ws_metadata();
    int check_total_allocation(uint64_t* limit, uint64_t used, uint64_t extra, uint64_t const_size);

    // MC recv block allocation
    void force_contiguous_allocate_mcrecv_blocks(const hnnx::VtcmCacheInstance& vtcm,
                                                  const std::vector<uint32_t>& tags);

    // ===== TCM block allocation main entry (反汇编 @ 0x13B29D0, 12155B) =====
    // allocate_tcm_blocks_internal: the core VTCM scratch allocator.
    // 反汇编确认 (vtcm_alloc.cc:2425/2429 错误检查 + STAT 日志):
    //   1. Sort blocks by reverse lifetime end (call sort_blocks_by_reverse_lifetime_end)
    //   2. For each block in sorted order:
    //      a. Compute Fibonacci hash: block_id × 0x740f1de9 (imul @ 0x13b3208)
    //      b. Probe hash table at [this+0x130] (open addressing, step=(h>>15)&0x1fffe|1)
    //      c. If found: reuse existing block's VTCM offset
    //      d. If not found: insert into hash table, allocate new VTCM offset
    //   3. Call obtain_loc_of_tcm_blocks to compute total cached weights with padding
    //   4. Check: reserving + pinned < TCM budget (行 2425 ERROR check)
    //   5. Log STAT: total_cached_weights_with_padding (行 2429)
    //   6. Store total in [this+0xd8] (= overlap_next_offset_) and [this+0x358]
    //
    // REQNN adaptation: GraphDeps provides block list + lifetimes;
    //   block_hash_ / block_table_ / overlap_allocs_ store results.
    struct TcmAllocOptions {
        bool pin_persistent = true;   // pin persistent blocks (weights) in VTCM
        uint32_t nsp_id = 0;          // target NSP
    };
    void allocate_tcm_blocks_internal(const hnnx::GraphDeps& deps,
                                       const std::vector<uint32_t>& block_ids,
                                       const TcmAllocOptions& opts);

    // obtain_loc_of_tcm_blocks: compute total VTCM bytes used by a set of blocks.
    // 反汇编 @ 0x13B24C0 (359B)
    // Sums (block_id + offset) & mask for each block — used for stats/overflow check.
    uint64_t obtain_loc_of_tcm_blocks(const std::vector<uint32_t>& block_ids,
                                        uint32_t mask) const;

    // Pool management
    // make_persistent_pools: classify allocated blocks into persistent pools.
    // 反汇编 @ 0xF453D0 (8477B, fa_alloc.cc)
    // Algorithm (from disassembly):
    //   1. For each allocated block, call find_persistent to check if it's a
    //      constant/weight block (vs scratch/activation)
    //   2. gather_const_blocks: collect all persistent blocks into a list
    //   3. apply_weight_compression: compress weights if applicable
    //   4. partition_switchable_blocks + make_pools_for_switchable: handle
    //      switchable constants (dynamic weight loading)
    //   5. allocate_new_pool (×3): create pools for
    //      - persistent (weights/constants, NSP-specific)
    //      - replaceable (swappable weights)
    //      - scratch (activations)
    //   6. Error checks: 行 1296/1301 "block not found" / "missing_blocks"
    //      行 2910 "placeholder pool_id mapping error"
    //      行 1451/1455 second-pass block lookup errors
    //   7. qnndsp_log stats: pool count, total persistent bytes
    struct PersistentPool {
        uint32_t pool_id = 0;
        uint64_t base_offset = 0;
        uint64_t total_size = 0;
        uint32_t block_count = 0;
        bool is_replaceable = false;
        uint32_t nsp_id = 0;
    };
    void make_persistent_pools();
    const std::vector<PersistentPool>& persistent_pools() const { return persistent_pools_; }
    void make_replaceable_persistent_pool();
    void serialize_pools(hnnx::Serializer& ser);
    void serialize_replaceable_mempool(hnnx::Serializer& ser);

    // Block mapping
    void map_plain_block(const void* block);
    void map_plain_block_to_pool_offs(const void* block, uint32_t* pool_id, uint64_t* offset);
    void* rewrite_to_physical_offset(uint64_t virtual_offset);
    void set_placeholder_mapping(const void* placeholder, void* actual);
    void get_blocks_pool_and_location(const void* block, uint32_t* pool, uint64_t* location);
    void pointers_to_poolid_offs();
    void* get_gather_desc_for_const_mempool(uint32_t pool_id);

    // Mode
    void set_mode(int mode);

    // Set VTCM budget for overlap allocation
    void set_overlap_budget(uint64_t budget) { overlap_budget_ = budget; }

    // Public read access to block table (for finalize step in GraphPrepare)
    const std::unordered_map<uint32_t, uint32_t>& block_id_to_idx() const {
        return block_id_to_idx_;
    }

private:
    // Memory regions
    struct MemBlock {
        uint64_t offset;
        uint64_t size;
        uint32_t pool_id;
        bool is_free;
        bool is_mc_cacheable_shared;
    };

    struct Pool {
        uint64_t base_offset;
        uint64_t size;
        std::vector<MemBlock> blocks;
    };

    std::vector<Pool> pools_;
    std::unordered_map<const void*, MemBlock*> block_map_;
    int mode_ = 0;

    // Taken ranges for lifetime overlap analysis
    struct TakenRange {
        uint64_t start;       // VTCM byte offset
        uint64_t end;         // offset + size
        int op_index;
        uint32_t life_begin;  // first use (topological index)
        uint32_t life_end;    // last use (topological index)
    };
    std::vector<TakenRange> taken_ranges_;

    void find_live_taken_ranges(int start_idx, int end_idx);

    // Lifetime-aware allocation state (populated by allocate_with_lifetime)
    std::unordered_map<hnnx::op_id_t, AllocResult> lifetime_allocs_;
    uint64_t lifetime_used_ = 0;   // high-water mark of VTCM after placement
    uint64_t lifetime_saved_ = 0;  // bytes saved by reuse (sum sizes - used)

    // ===== Overlap allocation state (反汇编 @ 0x13a14b0 / 0x13a01b0 / 0x13a0ac0) =====

    // Block table: per-block metadata (mirrors [this+0xe0] in .so)
    // Each entry tracks size, lifetime, VTCM offset, and link status.
    struct BlockEntry {
        uint32_t block_id = 0;      // unique block ID (= op_id-based)
        uint32_t size = 0;          // tensor size in bytes
        uint32_t life_begin = 0;    // first topological use
        uint32_t life_end = 0;      // last topological use
        uint8_t  flag = 0;          // 0=free, 1=used, 2=linked/reused
        uint64_t vtcm_offset = 0;   // assigned VTCM offset (0 = not yet assigned)
        uint32_t linked_to = 0;     // block_id this reuses (0 = none)
    };
    std::vector<BlockEntry> block_table_;
    std::unordered_map<uint32_t, uint32_t> block_id_to_idx_;  // block_id → index in block_table_

    // Fibonacci hash table for block lookup (mirrors [this+0x130] in .so)
    // Open addressing, 12-byte entries: block_id(4) + size(4) + flag(1) + pad(3)
    // Hash = (block_id × 0x740f1de9) folded to 32 bits
    // Probe step = (hash >> 15) & 0x1fffe | 1 (always odd)
    struct BlockHashEntry {
        uint32_t block_id = 0;  // 0 = empty slot
        uint32_t size = 0;     // block size
        uint8_t  flag = 0;     // 0=empty, 1=occupied, 2=tombstone
    };
    std::vector<BlockHashEntry> block_hash_;
    uint32_t block_hash_mask_ = 0;    // capacity - 1
    uint32_t block_hash_count_ = 0;  // occupied slot count

    // Free lists per size class (mirrors [this+0x190] in .so)
    // Size class = floor(log2(align_up(size, 128)))
    // free_lists_[class] = list of block_ids that are free and can be reused
    std::unordered_map<uint32_t, std::vector<uint32_t>> free_lists_;

    // Allocation state
    uint32_t next_block_id_ = 1;      // next block ID to assign
    uint64_t overlap_next_offset_ = 0;  // bump pointer for VTCM offsets
    uint64_t overlap_saved_ = 0;       // bytes saved by reuse
    uint64_t overlap_budget_ = 0;      // VTCM budget for overlap allocation

    // Overlap allocation results (op_id → OverlapAlloc)
    std::unordered_map<hnnx::op_id_t, OverlapAlloc> overlap_allocs_;

    // Persistent pools (populated by make_persistent_pools)
    std::vector<PersistentPool> persistent_pools_;

    // Hash table helpers (Fibonacci hash, open addressing)
    static uint32_t fib_hash_block(uint32_t key);
    void block_hash_insert(uint32_t block_id, uint32_t size);
    BlockHashEntry* block_hash_find(uint32_t block_id);
    void block_hash_grow();
    static uint32_t size_to_class(uint32_t size);
    void free_list_add(uint32_t block_id);
    uint32_t free_list_try_pop(uint32_t size_class, uint32_t life_begin, uint32_t life_end);
};

} // namespace fa

namespace hnnx {

// VTCM budget constants (verified from libHtpPrepare.so decompilation)
// Source: do_prepare1 @ 0xf66550, sap_reduce_bandwidth @ 0xf6ee78
// 0x400000 = 4MB hard upper limit (spill to DDR if exceeded)
// 0x3fe8000000000000 = 0.75 as IEEE double (3/4 usable, 1/4 for system/stack)
constexpr uint64_t VTCM_4MB_LIMIT = 0x400000;
constexpr double VTCM_USABLE_FACTOR = 0.75;

// VtcmCacheInstance wraps VTCM allocation for a specific NSP
class VtcmCacheInstance {
public:
    VtcmCacheInstance(uint32_t nsp_id, size_t vtcm_size);
    ~VtcmCacheInstance();

    size_t size() const { return vtcm_size_; }
    // Actual usable VTCM = min(size, 4MB) × 0.75
    // Source: sap_reduce_bandwidth @ 0xf6ee78 (mulsd 0.5a68(%r15), %xmm1)
    size_t usable_size() const;
    uint32_t nsp_id() const { return nsp_id_; }
    fa::FancyAllocator& allocator() { return allocator_; }

private:
    uint32_t nsp_id_;
    size_t vtcm_size_;
    fa::FancyAllocator allocator_;
};

} // namespace hnnx
