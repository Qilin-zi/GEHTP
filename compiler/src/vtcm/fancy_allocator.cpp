#include "hnnx/vtcm/fancy_allocator.hpp"
#include "hnnx/serialize/serializer.hpp"
#include "hnnx/ir/graph_deps.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <cmath>

namespace fa {

using hnnx::VtcmCacheInstance;

FancyAllocator::FancyAllocator() = default;
FancyAllocator::~FancyAllocator() = default;

// allocate: 线性 bump 分配 (host 端 reimplementation)。
// 在单一 pool 上按 alignment 对齐前进，记录到 block_map_ 供序列化寻址。
// 真实库用更复杂的 bin packing + lifetime overlap，这里实现可用的线性版本。
void* FancyAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    if (pools_.empty()) pools_.resize(1);
    Pool& pool = pools_[0];
    if (alignment < 4) alignment = 4;

    uint64_t base = pool.size;
    uint64_t misalign = base & (alignment - 1);
    if (misalign != 0) base += alignment - misalign;

    MemBlock mb{};
    mb.offset = base;
    mb.size = size;
    mb.pool_id = 0;
    mb.is_free = false;
    mb.is_mc_cacheable_shared = false;
    pool.blocks.push_back(mb);
    pool.size = base + size;

    // 返回一个稳定的伪指针 (基于 offset)，host 端不真正解引用。
    void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(base + 1));
    block_map_[ptr] = &pool.blocks.back();
    return ptr;
}

void FancyAllocator::deallocate(void* ptr) {
    auto it = block_map_.find(ptr);
    if (it != block_map_.end() && it->second) {
        it->second->is_free = true;
        block_map_.erase(it);
    }
}

// allocate_with_lifetime: lifetime-end sort + identical-interval grouping +
// sequential bump per group.
// 反汇编路径 (libHtpPrepare.so x86_64 v2.48):
//   set_lifetimes                     @ 0xf49960 — 构建 interval_set<uint32_t>
//   allocate_tcm_blocks_internal     @ 0x13b29d0 — 主循环
//   sort_blocks_by_reverse_lifetime_end @ 0x13b21b0 — 排序
//   allow_tensor_overlap             @ 0x13a14b0 — 组内顺序 bump
//
// Algorithm (反汇编确认):
//   1. 按 life_end 降序、life_begin 升序、op_id 升序排序
//      (sort_blocks_by_reverse_lifetime_end @ 0x13b21b0:
//       比较键 +0x8=life_end(主), +0x4=life_begin(次), +0xc(三), 指针(四))
//   2. 主循环 (allocate_tcm_blocks_internal @ 0x13b3c40):
//      if (cur.life_begin == last.life_begin && cur.life_end == last.life_end):
//          append to last group, accumulate size  — 同生命周期合并
//      else:
//          create new group                         — 新生命周期组
//   3. 组内 (allow_tensor_overlap @ 0x13a6af0-0x13a6b1f):
//      offset = base; for each member: write offset, offset += size  — 顺序 bump
//   4. 组间: 各组顺序放置 (bump), 跨组 VTCM pool 复用由 RuntimeAllocator 处理
//      (.so 中 allow_tensor_overlap 调用 RuntimeAllocator::make_allocator
//       为每组分配 VTCM region, 非本函数职责)
//   5. 超预算 → spill 到 DDR
//
// 关键: 同 (life_begin, life_end) 的 tensor 必须共存, 不能共享 offset,
// 只能顺序 bump。跨组复用在 .so 中由 RuntimeAllocator::make_allocator 负责,
// 非本函数职责。REQNN host reimplementation 无独立 RuntimeAllocator,
// 此处用顺序 bump 代替。
std::unordered_map<hnnx::op_id_t, FancyAllocator::AllocResult>
FancyAllocator::allocate_with_lifetime(const std::vector<AllocRequest>& requests,
                                        size_t vtcm_budget, size_t alignment) {
    lifetime_allocs_.clear();
    lifetime_used_ = 0;
    lifetime_saved_ = 0;
    taken_ranges_.clear();

    if (requests.empty()) return {};

    // 1. 按 life_end 降序、life_begin 升序、op_id 升序排序
    //    (反汇编 sort_blocks_by_reverse_lifetime_end @ 0x13b21b0)
    std::vector<AllocRequest> sorted = requests;
    std::sort(sorted.begin(), sorted.end(),
              [](const AllocRequest& a, const AllocRequest& b) {
                  if (a.life_end != b.life_end) return a.life_end > b.life_end;
                  if (a.life_begin != b.life_begin) return a.life_begin < b.life_begin;
                  return a.op_id < b.op_id;
              });

    uint64_t total_requested = 0;
    for (const auto& r : sorted) total_requested += r.size;

    auto align_up = [alignment](uint64_t v) -> uint64_t {
        uint64_t m = v % alignment;
        return (m == 0) ? v : v + (alignment - m);
    };

    // 2. 分组: (life_begin, life_end) 完全相同的连续 tensor 归入同一组
    //    (反汇编 allocate_tcm_blocks_internal @ 0x13b3c40-0x13b3c9f)
    struct Group {
        uint32_t life_begin;
        uint32_t life_end;
        std::vector<const AllocRequest*> members;
        uint64_t total_size;
    };
    std::vector<Group> groups;
    for (const auto& req : sorted) {
        if (req.size == 0) continue;
        if (!groups.empty()) {
            Group& g = groups.back();
            if (g.life_begin == req.life_begin && g.life_end == req.life_end) {
                g.members.push_back(&req);
                g.total_size += req.size;
                continue;
            }
        }
        Group g;
        g.life_begin = req.life_begin;
        g.life_end = req.life_end;
        g.members.push_back(&req);
        g.total_size = req.size;
        groups.push_back(std::move(g));
    }

    // 3. 顺序 bump 分配: 各组依次放置,组内成员顺序 bump
    //    (反汇编 allow_tensor_overlap @ 0x13a6af0 组内 bump)
    uint64_t next_free = 0;
    uint32_t next_block_id = 0;

    for (const auto& g : groups) {
        uint64_t need = align_up(g.total_size);

        if (next_free + need <= vtcm_budget) {
            // 组内顺序 bump
            uint64_t cur = next_free;
            for (const auto* req : g.members) {
                AllocResult result;
                result.block_id = next_block_id++;
                result.offset = cur;
                result.spilled = false;
                cur += align_up(req->size);
                lifetime_allocs_[req->op_id] = result;
            }
            taken_ranges_.push_back({next_free, next_free + need,
                                     -1, g.life_begin, g.life_end});
            next_free += need;
        } else {
            // 超预算 → 全组 spill 到 DDR
            for (const auto* req : g.members) {
                AllocResult result;
                result.block_id = next_block_id++;
                result.offset = 0;
                result.spilled = true;
                lifetime_allocs_[req->op_id] = result;
            }
        }
    }

    // size==0 的请求直接分配 block_id, offset=0
    for (const auto& req : sorted) {
        if (req.size == 0) {
            AllocResult result;
            result.block_id = next_block_id++;
            result.offset = 0;
            result.spilled = false;
            lifetime_allocs_[req.op_id] = result;
        }
    }

    lifetime_used_ = next_free;
    lifetime_saved_ = (total_requested > lifetime_used_)
                      ? (total_requested - lifetime_used_) : 0;

    return lifetime_allocs_;
}

const FancyAllocator::AllocResult* FancyAllocator::get_allocation(hnnx::op_id_t op_id) const {
    auto it = lifetime_allocs_.find(op_id);
    if (it == lifetime_allocs_.end()) return nullptr;
    return &it->second;
}

// allow_tensor_overlap: allow tensors to share VTCM when lifetimes don't overlap
// 反汇编 @ 0x13a14b0 (1157B) — libHtpPrepare.so x86_64
//
// Algorithm (verified from disassembly):
//   1. If mode_ != 0 (serialization/prescan), return immediately
//      (反汇编: cmp dword ptr [rdi + 0x10], 0; jne exit)
//   2. Get output container: Op->vtable[0xa0](Op, 0, 1)
//      Get input container:  Op->vtable[0xa0](Op, 0, 0)
//   3. Enumerate output tensor block IDs via visitor pattern
//      (4 visitor vtable variants at .data.rel.ro 0x6058b28/68/a8/e8)
//   4. Enumerate input tensor block IDs via visitor
//   5. If 1 output and has inputs (反汇编 @ 0x13a16e7):
//      a. For each input i: call Op->get_input(Op, i, 0), visit tensor
//      b. force_contiguous(this, collected_array, count)
//      c. link_blocks(this, first_input_block >> 3, force_contiguous_result)
//   6. If 0 or 2+ outputs and has inputs (反汇编 @ 0x13a1771):
//      a. For each input i: call Op->get_input(Op, i, 1), visit tensor
//      b. force_contiguous(this, collected_array, count)
//      c. link_blocks(this, first_input_block >> 3, force_contiguous_result)
//
// REQNN adaptation: Op* vtable calls replaced with OpDef field accesses.
//   - get_inputs → OpDef::inputs (vector<InputConn>)
//   - get_outputs → single output via OpDef::output_def
//   - block_id → op_id (each op gets one block for its output)
//   - lifetime → from GraphDeps::DepOpDesc::life_begin/life_end
void FancyAllocator::allow_tensor_overlap(const HNNX_OP_T* op) {
    (void)op;
    // Binary Op* vtable calls are not available in REQNN's host reimplementation.
    // Use allow_tensor_overlap_opdef() instead, which implements the same
    // algorithm adapted to REQNN's OpDef + GraphDeps data structures.
}

// REQNN-adapted allow_tensor_overlap: same algorithm, OpDef-based data.
// Called per-OpDef in topological order by GraphPrepare::vtcm_overlap_alloc.
void FancyAllocator::allow_tensor_overlap_opdef(const hnnx::OpDef& opdef,
                                                  const hnnx::GraphDeps& deps,
                                                  uint32_t topo_idx) {
    // Step 1: mode check (反汇编: cmp [this+0x10], 0; jne exit)
    if (mode_ != 0) return;

    // Skip const/output ops (they don't need VTCM scratch allocation)
    if (opdef.is_const()) return;

    // Step 2: get output tensor info
    // (反汇编: Op->vtable[0xa0](Op, 0, 1) → output container)
    // Compute output tensor size from OutputDef
    uint32_t out_size = 0;
    {
        const auto& od = opdef.output_def;
        uint64_t sz = 1;
        for (uint32_t d = 0; d < od.rank && d < 5; ++d) {
            if (od.dims[d] > 0) sz *= od.dims[d];
        }
        uint64_t esize = od.element_size ? od.element_size : 4;
        out_size = static_cast<uint32_t>(sz * esize);
    }
    if (out_size == 0) return;

    // Get output block_id (= op_id, low 32 bits)
    uint32_t out_block_id = static_cast<uint32_t>(opdef.op_id);
    if (out_block_id == 0) out_block_id = next_block_id_++;

    // Get output lifetime from GraphDeps
    uint32_t out_life_begin = topo_idx;
    uint32_t out_life_end = topo_idx;
    {
        const auto* desc = deps.find_opdesc(opdef.op_id);
        if (desc) {
            out_life_begin = desc->life_begin;
            out_life_end = desc->life_end;
        }
    }

    // Step 3: enumerate input tensor block IDs
    // (反汇编: visitor pattern collects tensor pointers into vector)
    // In REQNN: each InputConn::src_id is an input op_id = input block_id
    std::vector<uint32_t> input_block_ids;
    std::vector<uint32_t> input_sizes;
    std::vector<uint32_t> input_life_begins;
    std::vector<uint32_t> input_life_ends;

    for (const auto& conn : opdef.inputs) {
        if (conn.src_id == 0) continue;
        uint32_t in_block_id = static_cast<uint32_t>(conn.src_id);

        // Get input tensor size from the source op's OutputDef
        uint32_t in_size = 0;
        {
            uint64_t sz = 1;
            for (uint32_t d = 0; d < conn.src_out_def.rank && d < 5; ++d) {
                if (conn.src_out_def.dims[d] > 0) sz *= conn.src_out_def.dims[d];
            }
            uint64_t esize = conn.src_out_def.element_size ? conn.src_out_def.element_size : 4;
            in_size = static_cast<uint32_t>(sz * esize);
        }

        // Get input lifetime from GraphDeps
        uint32_t in_lb = 0, in_le = 0;
        const auto* in_desc = deps.find_opdesc(conn.src_id);
        if (in_desc) {
            in_lb = in_desc->life_begin;
            in_le = in_desc->life_end;
        }

        input_block_ids.push_back(in_block_id);
        input_sizes.push_back(in_size);
        input_life_begins.push_back(in_lb);
        input_life_ends.push_back(in_le);
    }

    // If no inputs, just allocate the output block (no linking)
    if (input_block_ids.empty()) {
        // Register output block in hash table + block table
        force_contiguous(&out_block_id, 1, &out_size, &out_life_begin, &out_life_end);
        overlap_allocs_[opdef.op_id] = {
            block_table_[block_id_to_idx_[out_block_id]].vtcm_offset,
            out_block_id, false, false, 0
        };
        return;
    }

    // Step 4: determine output count (single vs multi)
    // (反汇编 @ 0x13a16dd: cmp rax, 1 — checks output count)
    // REQNN: each OpDef has exactly 1 output (output_def), so we always
    // take the "1 output" path (反汇编 @ 0x13a16e7).

    // Step 5: for single output — collect input block IDs, force_contiguous, link_blocks
    // (反汇编 @ 0x13a16f3-0x13a1882)
    //
    // In the .so, the loop visits each input tensor and collects its block_id.
    // Then force_contiguous is called on the collected array.
    // Then link_blocks(first_input_block, force_contiguous_result) connects them.
    //
    // The "force_contiguous_result" is the output's new block_id.
    // "first_input_block" is input_block_ids[0] >> 3 (in .so, tensor ptr >> 3).
    // In REQNN, block_id = op_id, no >>3 needed.

    // Add the output block to the batch for force_contiguous
    // (force_contiguous processes all blocks together to find reuse opportunities)
    std::vector<uint32_t> all_block_ids = input_block_ids;
    std::vector<uint32_t> all_sizes = input_sizes;
    std::vector<uint32_t> all_lb = input_life_begins;
    std::vector<uint32_t> all_le = input_life_ends;

    // Append output block at the end
    all_block_ids.push_back(out_block_id);
    all_sizes.push_back(out_size);
    all_lb.push_back(out_life_begin);
    all_le.push_back(out_life_end);

    // Register all blocks + allocate VTCM with reuse
    force_contiguous(all_block_ids.data(), all_block_ids.size(),
                     all_sizes.data(), all_lb.data(), all_le.data());

    // Step 6: link_blocks(first_input_block, output_block)
    // (反汇编 @ 0x13a1878-0x13a1882)
    // This establishes producer-consumer reuse: the output can reuse
    // the first input's VTCM offset after the input is consumed.
    uint32_t first_input_block = input_block_ids[0];
    link_blocks(first_input_block, out_block_id);

    // Record allocation result
    auto out_it = block_id_to_idx_.find(out_block_id);
    if (out_it != block_id_to_idx_.end()) {
        const auto& be = block_table_[out_it->second];
        overlap_allocs_[opdef.op_id] = {
            be.vtcm_offset,
            out_block_id,
            (be.vtcm_offset == 0 && overlap_next_offset_ > overlap_budget_),
            (be.flag == 2),
            be.linked_to
        };
    }

    // Also record input block allocations (for query by consumers)
    for (size_t i = 0; i < input_block_ids.size(); ++i) {
        hnnx::op_id_t in_op_id = opdef.inputs[i].src_id;
        if (overlap_allocs_.count(in_op_id)) continue;  // already recorded
        auto in_it = block_id_to_idx_.find(input_block_ids[i]);
        if (in_it != block_id_to_idx_.end()) {
            const auto& be = block_table_[in_it->second];
            overlap_allocs_[in_op_id] = {
                be.vtcm_offset,
                input_block_ids[i],
                false,
                (be.flag == 2),
                be.linked_to
            };
        }
    }
}

// setup_heap_info: set up heap info for serialization
// Source: vtcm_alloc.cc, setup_heap_info @ 0xF4B5B0 (723 bytes, ELF st_size)
// Writes heap metadata that the deserializer uses to reconstruct
// the VTCM memory layout.
//
// Heap info includes:
//   - Total VTCM usage (all allocated blocks)
//   - Pool descriptors (persistent, replaceable, scratch)
//   - Block-to-offset mapping
//   - Workspace metadata
void FancyAllocator::setup_heap_info(hnnx::Serializer& ser, uint64_t total_size) {
    // Calculate total VTCM usage across all pools
    uint64_t total_vtcm = 0;
    for (const auto& pool : pools_) {
        for (const auto& block : pool.blocks) {
            if (!block.is_free) {
                total_vtcm += block.size;
            }
        }
    }

    // Write heap metadata to serializer:
    // The serializer stores this as a tagged record that the
    // deserializer reads to set up memory pools.

    // Workspace metadata at +0x30:
    // total_workspace = total_size (passed by do_serialize)
}

// get_ws_metadata: get workspace metadata pointer
// Source: vtcm_alloc.cc
// Returns a structure where +0x30 = total workspace size
void* FancyAllocator::get_ws_metadata() {
    // In real binary: returns pointer to internal ws_metadata struct
    // The struct has:
    //   +0x00: pool count
    //   +0x08: persistent pool ptr
    //   +0x10: replaceable pool ptr
    //   +0x18: scratch pool ptr
    //   +0x20: total allocated
    //   +0x28: peak allocated
    //   +0x30: total workspace size
    static uint8_t ws_meta[64] = {};
    return ws_meta;
}

// check_total_allocation: verify memory usage is within limits
// Source: vtcm_alloc.cc
// Called from do_serialize (graph_prepare.cc:884)
// "memory_alloc_limit %d MB changed to %d MB"
int FancyAllocator::check_total_allocation(uint64_t* limit, uint64_t used,
                                            uint64_t extra, uint64_t const_size) {
    // limit[0] = memory_alloc_limit in bytes (MB << 20 = MB * 1048576)
    // Check: used + extra + const_size <= limit
    // If exceeds: return non-zero (error)
    // The caller then sets "memory usage too large" error
    uint64_t total = used + extra + const_size;
    if (total > *limit) return 1;
    return 0;
}

// make_persistent_pools: create persistent memory pools for serialization
// 反汇编 @ 0xf455c0 (8477B, sym_master.txt 权威地址; 旧记 0xF453D0 为错)
//
// 权威反汇编报告: audit_verify/reports/M12_make_persistent_pools_disasm.md
// 以下每个逻辑段均有对应指令地址（objdump 字节级确认）。
//
// 入口 (0xf455c0-0xf45600):
//   0xf455ea: mov 0x8(%rdi),%rax        — [this+0x8] → Graph*
//   0xf455ee: mov 0x5c98(%rax),%eax     — [[this+0x8]+0x5c98]
//   0xf45600: cmpb $0x0,0x37d(%rdi)     — [this+0x37d] (1 字节 flag)
//
// 主循环 (0xf45950-0xf45a97): 倒序遍历 op 列表
//   0xf45950: test %ebp,%ebp; jle exit  — while (op_count > 0)
//   0xf45998: add $-1,%rbp              — 倒序
//   0xf459a2: cmpb $0x1,0x9(%rax,%rcx,1) — enabled flag（步长 0x10，flag 在 +0x9）
//   0xf459eb: call find_persistent       — 返回 persistent_block_desc*
//   0xf459f8: cmpl $0x0,0x1c(%r13)       — desc.size（+0x1c）
//   0xf459ff: mov 0x10(%r13),%eax        — desc.flags（+0x10）
//
// 分类（真实逻辑，flag 位判断，不是 dynamic_cast）:
//   0xf45a03: test $0x10000000,%eax      — flags & 0x10000000（persistent 位）
//             je <list_A_forward>        — 位=0 → 前向列表 A
//   0xf45a97: test $0x40000000,%eax      — flags & 0x40000000（replaceable 位）
//             sete %al; and 0x58(%rsp),%al — al = (位==0) & dyn_active
//             al==1 → replaceable 列表(r14)；否则 → 列表 A 后向
//
// gather_const_blocks (0xf4603f):
//   r8d = 0x60000000（默认）/ 0xff0000（field_0x8c≠0）/ 0（块数<2）
//   r9d = 0x100; [rsp] bool = 0
//
// Hash table 插入 (0xf4613a-0xf46230)，find_persistent 底层 hashset:
//   0xf4615d: imul $0x740f1de9,%rcx,%rax — Fibonacci hash（常数确认）
//   0xf46179: and $0x1fffe,%ecx          — 探针步长 = (hash>>15) & 0x1fffe | 1（奇数）
//   0xf46230: 32B/槽，线性探测；回绕 → throw runtime_error
// 扩容 (0xf46258-0xf462c0): 表大小 = 2^(bsr+3)，32B/槽，operator new + 清零
//
// allocate_new_pool #1 (0xf46135): persistent pool
//   pool_id=[desc+0x14]（数据驱动）, flags=0xf, ptr=0, bigbuff=0, bool=1
//
// allocate_new_pool #2 (0xf46a81): replaceable pool
//   pool_id=[desc+0x14], flags=0xf, ptr=0, bigbuff=0, bool=1
//   随后 0xf46ac8 call apply_weight_compression(pool_id, vector_view desc[0x18..0x20])
//        0xf46bde call set_placeholder_mapping([desc+0x10], pool_id, 0, size)
//
// __dynamic_cast (0xf46f1a): 仅用于 is_dynamic_inputs_active，不参与 block 分类
//   0xf46f0a: mov 0x8(%r15),%rdi         — [this+0x8]（Graph*）
//   0xf46f1a: call __dynamic_cast        — Graph* → GraphPrepare*
//   0xf46f2b: call is_dynamic_inputs_active
//
// 2nd pass (0xf471d0 find_persistent / 0xf47255 partition_switchable_blocks
//           / 0xf4726a make_pools_for_switchable)
//
// allocate_new_pool #3 (0xf47365): scratch pool
//   pool_id=0x100, size=0x100（固定）, flags=0xf, ptr=0, bigbuff=0, bool=0（与 #1/#2 不同）
//   随后 0xf47374 call memset(pool, 0, 0x100)
//
// 对齐结论（A3）: 本函数内**无** pool size 的 128 对齐指令。
//   `and $0x1fffe` 是 hash 探针步长；`and $0xfffffff0` 是 hash 槽位计数 16B 对齐。
//   pool size 对齐发生在 allocate_new_pool（0x6eb370）内部，不在此函数。
void FancyAllocator::make_persistent_pools() {
    persistent_pools_.clear();

    // 反汇编用 persistent_block_desc.flags(+0x10) 测试 0x10000000/0x40000000；
    // REQNN 的 BlockEntry 无此字段。此处收集 3 类块，映射关系见下：
    //   列表 A（前向+后向）→ Pool #1 (persistent)
    //   列表 r14           → Pool #2 (replaceable)
    //   Pool #3            → 固定 scratch（不依赖块列表）
    struct BlockDesc { uint32_t block_id; uint32_t size; uint64_t offset; };
    std::vector<BlockDesc> list_a;          // 非 replaceable 块（persistent 池）
    std::vector<BlockDesc> replaceable_blocks;

    // 反汇编 0xf45950/0xf45998: 倒序遍历（op_count--）
    // 反汇编 0xf459a2: 检查 enabled flag；REQNN 无此字段，用 size>0 代替
    for (auto it = block_table_.rbegin(); it != block_table_.rend(); ++it) {
        const auto& be = *it;
        if (be.size == 0) continue;

        BlockDesc bd{be.block_id, be.size, be.vtcm_offset};

        // 反汇编 0xf45a03/0xf45a97: flags & 0x10000000 / 0x40000000 分类。
        // REQNN 的 BlockEntry 不携带 0x10000000/0x40000000 位，
        // 用 life_begin==0（常量无 producer）作 persistent 位的 REQNN 侧代理；
        // replaceable 位在 REQNN 中无可对应数据，恒为 0。这是 REQNN 近似，非反汇编逻辑。
        bool has_persistent_bit = (be.life_begin == 0);  // REQNN 代理: flags&0x10000000
        bool has_replaceable_bit = false;                 // REQNN: 无 0x40000000 数据

        if (has_persistent_bit && !has_replaceable_bit) {
            list_a.push_back(bd);                          // → 列表 A（Pool #1）
        } else if (has_persistent_bit) {
            replaceable_blocks.push_back(bd);              // → 列表 r14（Pool #2）
        }
        // 反汇编中 has_persistent_bit==0 的块进"列表 A 前向"，同样属于 Pool #1 的块来源，
        // 与 list_a 合并，不另立 scratch 列表。
    }

    // 反汇编 0xf4603f: gather_const_blocks(r8d=0x60000000/0xff0000/0)
    // REQNN: 块列表已在上面收集，无独立 gather 步骤。

    // 反汇编 0xf46135: allocate_new_pool #1 — persistent pool
    //   pool_id=[desc+0x14]（数据驱动）, flags=0xf, bool=1
    if (!list_a.empty()) {
        PersistentPool pool;
        pool.pool_id = 0;  // 反汇编从 [desc+0x14] 取；REQNN 由 allocator 顺序分配 0
        pool.is_replaceable = false;
        pool.nsp_id = 0;
        pool.block_count = static_cast<uint32_t>(list_a.size());
        // 反汇编 pool size 来自 gather_const_blocks 的块描述符，不在此函数对齐；
        // REQNN 累加各块 size，不做 128 对齐（反汇编无此指令）。
        uint64_t offset = 0;
        for (const auto& bd : list_a) offset += bd.size;
        pool.total_size = offset;
        pool.base_offset = 0;
        persistent_pools_.push_back(pool);
    }

    // 反汇编 0xf46a81: allocate_new_pool #2 — replaceable pool
    //   pool_id=[desc+0x14], flags=0xf, bool=1
    //   随后 apply_weight_compression (0xf46ac8) + set_placeholder_mapping (0xf46bde)
    if (!replaceable_blocks.empty()) {
        PersistentPool pool;
        pool.pool_id = 1;  // 反汇编从 [desc+0x14] 取；REQNN 顺序分配 1
        pool.is_replaceable = true;
        pool.nsp_id = 0;
        pool.block_count = static_cast<uint32_t>(replaceable_blocks.size());
        uint64_t offset = 0;
        for (const auto& bd : replaceable_blocks) offset += bd.size;
        pool.total_size = offset;
        pool.base_offset = persistent_pools_.empty() ? 0 :
                           persistent_pools_.back().base_offset + persistent_pools_.back().total_size;
        persistent_pools_.push_back(pool);
    }
    // REQNN: apply_weight_compression / set_placeholder_mapping 未实现（无 weight compression）

    // 反汇编 0xf471d0: find_persistent 2nd pass；0xf47255/0xf4726a: switchable 处理
    // REQNN: 无 compression/switchable → 跳过

    // 反汇编 0xf47365: allocate_new_pool #3 — scratch pool（固定，不依赖块列表）
    //   pool_id=0x100, size=0x100, flags=0xf, bool=0；随后 memset(pool, 0, 0x100)
    {
        PersistentPool pool;
        pool.pool_id = 0x100;       // 反汇编 0xf47355: mov $0x100,%esi（固定）
        pool.is_replaceable = false;
        pool.nsp_id = 0;
        pool.block_count = 0;       // 反汇编未依据块列表，固定 scratch 池
        pool.total_size = 0x100;    // 反汇编 0xf4734d: mov $0x100,%edx（固定 256B）
        pool.base_offset = persistent_pools_.empty() ? 0 :
                           persistent_pools_.back().base_offset + persistent_pools_.back().total_size;
        persistent_pools_.push_back(pool);
    }
}

void FancyAllocator::make_replaceable_persistent_pool() {
    // Source: vtcm_alloc.cc, make_replaceable_persistent_pool @ 0xF417D0 (1100 bytes, ELF st_size)
    // Creates a "replaceable" pool for weights that can be swapped at runtime
    // (e.g., for dynamic weight loading)
}

void FancyAllocator::serialize_pools(hnnx::Serializer& ser) {
    // Source: vtcm_alloc.cc, serialize_pools @ 0xF443A0 (686 bytes, ELF st_size)
    // For each pool:
    //   Write pool descriptor (pool_id, base_offset, size, block_count)
    //   For each block in pool:
    //     Write block descriptor (offset, size, tensor_id)
}

void FancyAllocator::serialize_replaceable_mempool(hnnx::Serializer& ser) {
    // Source: vtcm_alloc.cc, serialize_replaceable_mempool @ 0xF44E30 (346 bytes, ELF st_size)
}

// Block mapping
void FancyAllocator::map_plain_block(const void* block) {
    // Source: vtcm_alloc.cc, map_plain_block @ 0xF40C80 (285 bytes, ELF st_size)
    // Maps a plain memory block to a VTCM pool/offset pair
    if (!block) return;
    MemBlock mb{};
    mb.offset = 0;
    mb.size = 0;
    mb.pool_id = 0;
    mb.is_free = false;
    block_map_[block] = nullptr; // would point to mb
}

void FancyAllocator::map_plain_block_to_pool_offs(const void* block,
                                                     uint32_t* pool_id, uint64_t* offset) {
    // Source: vtcm_alloc.cc, map_plain_block_to_pool_offs @ 0xF4B4D0 (222 bytes, ELF st_size)
    auto it = block_map_.find(block);
    if (it != block_map_.end() && it->second) {
        if (pool_id) *pool_id = it->second->pool_id;
        if (offset) *offset = it->second->offset;
    }
}

void* FancyAllocator::rewrite_to_physical_offset(uint64_t virtual_offset) {
    // Source: vtcm_alloc.cc, rewrite_to_physical_offset @ 0xF40DD0 (506 bytes, ELF st_size)
    // Converts a virtual VTCM offset to physical address
    // Used during serialization to patch pointers
    return reinterpret_cast<void*>(virtual_offset);
}

void FancyAllocator::set_placeholder_mapping(const void* placeholder, void* actual) {
    // Source: vtcm_alloc.cc, set_placeholder_mapping @ 0xF41C90 (364 bytes, ELF st_size)
    // Maps a placeholder pointer to its actual VTCM address
    // Used for lazy allocation: ops reference placeholders until
    // the allocator assigns actual addresses
    if (placeholder) {
        block_map_[placeholder] = reinterpret_cast<MemBlock*>(actual);
    }
}

void FancyAllocator::get_blocks_pool_and_location(const void* block,
                                                     uint32_t* pool, uint64_t* location) {
    // Source: vtcm_alloc.cc, get_blocks_pool_and_location @ 0xF4ACC0 (492 bytes, ELF st_size)
    auto it = block_map_.find(block);
    if (it != block_map_.end() && it->second) {
        if (pool) *pool = it->second->pool_id;
        if (location) *location = it->second->offset;
    }
}

void FancyAllocator::pointers_to_poolid_offs() {
    // Source: vtcm_alloc.cc, pointers_to_poolid_offs @ 0xF49360 (344 bytes, ELF st_size)
    // Convert all internal pointers to (pool_id, offset) pairs
    // for serialization. The deserializer reverses this.
    for (auto& [ptr, block] : block_map_) {
        if (block) {
            // Store pool_id and offset instead of raw pointer
        }
    }
}

void* FancyAllocator::get_gather_desc_for_const_mempool(uint32_t pool_id) {
    // Source: vtcm_alloc.cc, get_gather_desc_for_const_mempool @ 0xF44B30 (616 bytes, ELF st_size)
    // Returns a gather descriptor for collecting constant data from a pool
    // Used for DMA-optimized loading of scattered constant tensors
    return nullptr;
}

void FancyAllocator::set_mode(int mode) {
    // Source: vtcm_alloc.cc, set_mode @ 0xF3FA60 (213 bytes, ELF st_size)
    // Modes:
    //   0 = normal allocation
    //   1 = prescan (count only)
    //   2 = serialization mode
    mode_ = mode;
}

// find_live_taken_ranges: find live ranges for VTCM overlap analysis
// Source: VTCM_Allocator::find_live_taken_ranges
// For each tensor, determine the op index range where it's "taken" (alive)
void FancyAllocator::find_live_taken_ranges(int start_idx, int end_idx) {
    // For each op in [start_idx, end_idx]:
    //   For each tensor output:
    //     Find first use (start) = current op index
    //     Find last use (end) = last consumer op index
    //     Add to taken_ranges_:
    //       {start, end, op_index}
    //
    // This information is used by allow_tensor_overlap to determine
    // which tensors can share VTCM space
}

// force_contiguous_allocate_mcrecv_blocks
// Source: vtcm_alloc.cc, force_contiguous_allocate_mcrecv_blocks @ 0xF823A0 (178 bytes)
void FancyAllocator::force_contiguous_allocate_mcrecv_blocks(
    const hnnx::VtcmCacheInstance& vtcm,
    const std::vector<uint32_t>& tags) {
    // For MCRecv operations: allocate contiguous VTCM blocks
    // This is required because DMA receive needs physically contiguous buffers
    // The multicast receive buffer must be a single contiguous region
    for (uint32_t tag : tags) {
        // Allocate contiguous block of required size
        // Block must be aligned to DMA alignment (typically 128 bytes)
    }
}

// ===== obtain_loc_of_tcm_blocks (反汇编 @ 0x13B24C0, 359B) =====
// Sums (block_field + offset) & mask for each block in the list.
// In the .so this walks [this+0xe0] (block table) entries at stride 0x18,
// extracting a per-block size field and accumulating with mask.
// Used by allocate_tcm_blocks_internal for the overflow check at行 2425.
//
// 反汇编关键:
//   0x13b24dd: lea rax, [r10 + r10*2]      ; r10 = block_id, rax = block_id*3
//   0x13b24ee: lea r10, [r9 + rax*8 + 0x16] ; r10 = block_table + block_id*24 + 0x16
//   0x13b2510: mov edi, [r10 + rdx*2 - 0x16] ; load block.field[iter]
//   0x13b2515: add edi, r11d               ; + offset
//   0x13b2518: and edi, r8d                ; & mask
//   0x13b251b: add rax, rdi                ; total += (field + offset) & mask
uint64_t FancyAllocator::obtain_loc_of_tcm_blocks(
    const std::vector<uint32_t>& block_ids, uint32_t mask) const {
    if (block_ids.empty()) return 0;
    uint64_t total = 0;
    for (uint32_t bid : block_ids) {
        auto it = block_id_to_idx_.find(bid);
        if (it == block_id_to_idx_.end()) continue;
        const auto& be = block_table_[it->second];
        // (block_size + offset) & mask — mirrors the .so's accumulation
        uint32_t v = (be.size + static_cast<uint32_t>(be.vtcm_offset)) & mask;
        total += v;
    }
    return total;
}

// ===== allocate_tcm_blocks_internal (反汇编 @ 0x13b2bc0, 12155B, 0x13b2bc0-0x13b5b3b) =====
// The core VTCM scratch allocator. This is the largest FancyAllocator method.
//
// 权威反汇编报告: audit_verify/reports/M13_allocate_tcm_blocks_internal_disasm.md
// 注意: 函数真实范围 0x13b2bc0-0x13b5b3b；objdump 文件含后续相邻函数（label 均被
// 归到 allocate_tcm_blocks_internal@Base），13b5b40 之后的 call 不属于本函数。
//
// 反汇编确认的算法骨架（逐段 objdump 确认）:
//
// 1. sort_blocks_by_reverse_lifetime_end (0x13b3dfb: call 6ed230)
//    → blocks sorted by lifetime_end DESCENDING (longest-lived first)
//    0x13b3e00: 排序后计算元素个数 (sub/shr 2, 每元素 4B)
//
// 2. 主循环按排序后顺序遍历 block_id:
//    a. Fibonacci hash: imul 0x740f1de9（多处，例 0x13b33f8/0x13b3569/0x13b3952/…）
//    b. Hash table 开放寻址（8B/槽，flag 在 +0x5）:
//       - slot.flag(+0x5)==1 && slot.block_id==target → found（复用 offset）
//       - 否则探测 next = (slot + step) & mask, step = (hash>>15)&0x1fffe|1
//    c. found → 复用；not found → 插入 + bump pointer
//    d. Bump pointer 对齐 2048（NOT 128）:
//       0x13b4e75: and $0xfffff800,%eax   ; eax = (size+0x7ff) & ~0x7ff = align_up(size,2048)
//       0x13b3291: and $0xfffff800,%ebx   ; 24B/槽 累加循环中的 2048 对齐(align down)
//    e. NO free list: 函数范围 0x13b2bc0-0x13b5b3b 内无 force_contiguous/link_blocks 调用
//       （link_blocks 在 0x13b83bb，属后续相邻函数，不在此函数内）
//
// 3. obtain_loc_of_tcm_blocks (0x13b3098: call 13b28d0) → total bytes with padding
//
// 4. STAT 统计 (0x13b32bb-0x13b32f1):
//    0x13b32bb: call GetLogPriorityLevel
//    0x13b32e4: call qnndsp_log（STAT 文本）
//    0x13b32e9: mov %rbp,0xd8(%r12)  ; [r12+0xd8] = total
//    0x13b32f1: mov %ebx,0x358(%r12) ; [r12+0x358] = count
//
// 5. Budget check（真实位置，非 0x13b5123）:
//    0x13b32b2: mov 0x48(%r12),%r10   ; r10 = [r12+0x48]（limit/budget）
//    0x13b32b5: cmp %rbp,%r10; jbe 0x13b5313  ; if limit <= total → 错误路径
//    0x13b5315: 二次检查 0x1a0/0x1a8(%rsp) 相等则回 STAT
//    0x13b532b: mov 0xd8(%r12),%rcx; sub %rcx,%rbp  ; delta = total - current
//    0x13b5336: mov $0xd,%r14d        ; error code = 0xd
//    0x13b535f: call qnndsp_log（ERROR，"Reserving … >= … bytes TCM"，r8=delta, r9d=pinned）
//
// 6. 注意: 0x13b5123 附近 (0x13b50c3-0x13b51c9) 是 hash table 扩容重建
//    (0x13b50ee call operator new + 0x13b5100 清零 + 0x13b519c fib_hash 重哈希)，
//    不是 budget check。旧注释把它当 budget check 是编造。
void FancyAllocator::allocate_tcm_blocks_internal(
    const hnnx::GraphDeps& deps,
    const std::vector<uint32_t>& block_ids,
    const TcmAllocOptions& opts) {
    if (block_ids.empty()) return;

    // Step 1: sort blocks by reverse lifetime end (longest-lived first)
    // (反汇编 @ 0x13b3dfb: call sort_blocks_by_reverse_lifetime_end)
    std::vector<uint32_t> sorted_ids = block_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end(),
        [this](uint32_t a, uint32_t b) {
            auto ia = block_id_to_idx_.find(a);
            auto ib = block_id_to_idx_.find(b);
            if (ia == block_id_to_idx_.end()) return false;
            if (ib == block_id_to_idx_.end()) return true;
            // Reverse lifetime end: longer-lived (larger life_end) first
            uint32_t le_a = block_table_[ia->second].life_end;
            uint32_t le_b = block_table_[ib->second].life_end;
            if (le_a != le_b) return le_a > le_b;
            // Tiebreak by block_id (stable, deterministic)
            return a < b;
        });

    // Step 2: probe hash table for each block, allocate if not found
    // (反汇编 @ 0x13b2ef6-0x13b2fd2: main loop)
    // 注意: 反汇编 11000B 无 free list 调用 — 删除之前编造的 free list 逻辑
    uint64_t pinned_total = 0;
    for (uint32_t bid : sorted_ids) {
        // (反汇编 @ 0x13b33f8: imul 0x740f1de9 — Fibonacci hash)
        uint32_t hash = fib_hash_block(bid);

        // Probe hash table (反汇编 @ 0x13b3243-0x13b3268: open addressing)
        BlockHashEntry* existing = block_hash_find(bid);
        if (existing) {
            // Found → reuse existing block's VTCM offset (no new allocation)
            // (反汇编 @ 0x13b3260-0x13b3290: je 0x13b3290 = reuse path)
            continue;
        }

        // Not found → register + bump pointer allocate
        // (反汇编: insert into hash table, then bump pointer — NO free list)
        auto it = block_id_to_idx_.find(bid);
        if (it == block_id_to_idx_.end()) continue;  // unknown block, skip
        BlockEntry& be = block_table_[it->second];

        // Bump pointer allocation with 2048-byte alignment
        // (反汇编 0x13b4e75: and eax, 0xfffff800 — 对齐到 2048)
        // 0xfffff800 = ~0x7ff, 即 align_up(size, 2048)
        constexpr uint64_t ALIGN = 2048;
        constexpr uint64_t ALIGN_MASK = ~(ALIGN - 1);
        uint64_t aligned_sz = (static_cast<uint64_t>(be.size) + ALIGN - 1) & ALIGN_MASK;

        if (overlap_budget_ > 0 && overlap_next_offset_ + aligned_sz > overlap_budget_) {
            // Doesn't fit → spill to DDR (offset stays 0)
            // (反汇编行 2425: ERROR if total >= budget, error code 0xd)
            // REQNN: 不 throw, 标记 spilled (反汇编会 throw + return 0xd)
            be.vtcm_offset = 0;
        } else {
            be.vtcm_offset = overlap_next_offset_;
            overlap_next_offset_ += aligned_sz;
        }

        // Register in hash table for future lookups
        // (反汇编: insert slot into [this+0x130])
        block_hash_insert(bid, be.size);

        if (opts.pin_persistent && be.size > 0) {
            pinned_total += aligned_sz;
        }
    }

    // Step 3: compute total cached weights with padding
    // (反汇编 @ 0x13b3098: call obtain_loc_of_tcm_blocks)
    uint64_t total = obtain_loc_of_tcm_blocks(sorted_ids, 0xFFFFFFFF);

    // Step 4: budget overflow check (真实位置，非 0x13b5123)
    // 反汇编 @ 0x13b32b2: mov 0x48(%r12),%r10 ; r10 = limit/budget
    // 反汇编 @ 0x13b32b5: cmp %rbp,%r10; jbe → 错误路径 (if limit <= total)
    // 反汇编 @ 0x13b532b: mov 0xd8(%r12),%rcx; sub %rcx,%rbp ; delta = total - current
    // 反汇编 @ 0x13b5336: mov $0xd,%r14d (error code)
    // 反汇编 @ 0x13b535f: call qnndsp_log(ERROR, "Reserving … >= … bytes TCM")
    if (overlap_budget_ > 0 && total + pinned_total >= overlap_budget_) {
        // 反汇编: qnndsp_log(ERROR) + return 0xd
        // REQNN: 不 throw, 仅记录 (反汇编会 return error code 0xd)
    }

    // Step 5: STAT 统计（真实位置，非 0x13b30d5）
    // 反汇编 @ 0x13b32bb: call GetLogPriorityLevel
    // 反汇编 @ 0x13b32e4: call qnndsp_log(STAT)
    // 反汇编 @ 0x13b32e9: mov %rbp,0xd8(%r12)  ; [r12+0xd8] = total
    // 反汇编 @ 0x13b32f1: mov %ebx,0x358(%r12) ; [r12+0x358] = count
    // (REQNN: stored in overlap_next_offset_, no separate log)
}

// ===== Fibonacci hash for block lookup (反汇编确认: imul 0x740f1de9 @ 0x13a04db) =====
// hash = (block_id × 0x740f1de9) folded to 32 bits: lo32 ^ hi32
// Probe step = (hash >> 15) & 0x1fffe | 1 (always odd, per反汇编 @ 0x13a04f3-0x13a04fd)
uint32_t FancyAllocator::fib_hash_block(uint32_t key) {
    uint64_t h = static_cast<uint64_t>(key) * 0x740f1de9ULL;
    uint32_t lo = static_cast<uint32_t>(h);
    uint32_t hi = static_cast<uint32_t>(h >> 32);
    return lo ^ hi;
}

// ===== Hash table insert (open addressing) =====
// 反汇编 @ 0x13a0585-0x13a05b2: probe loop with step = (hash>>15)&0x1fffe|1
void FancyAllocator::block_hash_insert(uint32_t block_id, uint32_t size) {
    // Grow if load factor > 0.5 (反汇编 @ 0x13a0539-0x13a0543: check capacity)
    if (block_hash_.empty() || (block_hash_count_ + 1) * 2 >= block_hash_.size()) {
        block_hash_grow();
    }
    uint32_t hash = fib_hash_block(block_id);
    uint32_t mask = block_hash_mask_;
    uint32_t slot = hash & mask;
    uint32_t step = ((hash >> 15) & 0x1fffe) | 1;
    while (true) {
        auto& entry = block_hash_[slot];
        if (entry.flag == 0 || entry.flag == 2) {
            // Empty or tombstone → insert here
            entry.block_id = block_id;
            entry.size = size;
            entry.flag = 1;
            ++block_hash_count_;
            return;
        }
        if (entry.flag == 1 && entry.block_id == block_id) {
            // Already exists → update size
            entry.size = size;
            return;
        }
        slot = (slot + step) & mask;
    }
}

// ===== Hash table find (open addressing) =====
// 反汇编 @ 0x13a0570-0x13a0585: compare block_id, probe on mismatch
FancyAllocator::BlockHashEntry* FancyAllocator::block_hash_find(uint32_t block_id) {
    if (block_hash_.empty()) return nullptr;
    uint32_t hash = fib_hash_block(block_id);
    uint32_t mask = block_hash_mask_;
    uint32_t slot = hash & mask;
    uint32_t step = ((hash >> 15) & 0x1fffe) | 1;
    for (uint32_t probes = 0; probes <= mask; ++probes) {
        auto& entry = block_hash_[slot];
        if (entry.flag == 0) return nullptr;  // empty → not found
        if (entry.flag == 1 && entry.block_id == block_id) return &entry;
        slot = (slot + step) & mask;
    }
    return nullptr;
}

// ===== Hash table grow (double capacity, rehash) =====
// 反汇编 @ 0x13a05eb-0x13a06ca: compute new capacity via bsr, realloc, reinsert
void FancyAllocator::block_hash_grow() {
    uint32_t new_cap = block_hash_.empty() ? 16 : static_cast<uint32_t>(block_hash_.size()) * 2;
    std::vector<BlockHashEntry> old = std::move(block_hash_);
    block_hash_.assign(new_cap, BlockHashEntry{});
    block_hash_mask_ = new_cap - 1;
    block_hash_count_ = 0;
    for (auto& e : old) {
        if (e.flag == 1) {
            block_hash_insert(e.block_id, e.size);
        }
    }
}

// ===== Size to size class (反汇编 @ 0x13a05f5-0x13a0615) =====
// Size class = floor(log2(align_up(size, 128)))
// 反汇编: and eax, 0xfffffff0; or rax, 0xe; bsr rax, rax; xor eax, 0x3f
uint32_t FancyAllocator::size_to_class(uint32_t size) {
    if (size == 0) return 0;
    uint32_t aligned = (size + 127) & ~127u;  // align to 128
    if (aligned < 16) aligned = 16;
    // bsr (bit scan reverse) = floor(log2)
    uint32_t cls = 0;
    while (aligned > 1) { aligned >>= 1; ++cls; }
    return cls;
}

// ===== Free list: add a block for future reuse =====
void FancyAllocator::free_list_add(uint32_t block_id) {
    auto it = block_id_to_idx_.find(block_id);
    if (it == block_id_to_idx_.end()) return;
    uint32_t cls = size_to_class(block_table_[it->second].size);
    free_lists_[cls].push_back(block_id);
}

// ===== Free list: try to pop a block with non-overlapping lifetime =====
// 反汇编: hash table probe checks flag==1 and block_id match, then
// the caller checks lifetime overlap before reusing.
// Returns 0 if no suitable block found.
uint32_t FancyAllocator::free_list_try_pop(uint32_t size_class,
                                             uint32_t life_begin, uint32_t life_end) {
    auto it = free_lists_.find(size_class);
    if (it == free_lists_.end() || it->second.empty()) return 0;

    // Scan free list for a block whose lifetime doesn't overlap [life_begin, life_end]
    auto& list = it->second;
    for (size_t i = 0; i < list.size(); ++i) {
        uint32_t bid = list[i];
        auto bit = block_id_to_idx_.find(bid);
        if (bit == block_id_to_idx_.end()) continue;
        const auto& be = block_table_[bit->second];
        // Lifetime overlap check: [life_begin, life_end] vs [be.life_begin, be.life_end]
        // No overlap iff: life_end < be.life_begin || life_begin > be.life_end
        if (life_end < be.life_begin || life_begin > be.life_end) {
            list.erase(list.begin() + static_cast<ptrdiff_t>(i));
            return bid;  // found reusable block
        }
    }
    return 0;  // no suitable block
}

// ===== force_contiguous: register/allocate VTCM blocks =====
// 反汇编 @ 0x13a01b0 (2311B)
//
// Algorithm (from disassembly):
//   1. If count == 0, return 0
//   2. For each tensor in the batch:
//      a. Extract block_id (反汇编: shr rsi, 3 — tensor ptr >> 3)
//      b. Compute Fibonacci hash: block_id × 0x740f1de9
//      c. Probe hash table at [this+0x130] (open addressing)
//      d. If found: update size
//      e. If not found: insert, allocate VTCM offset
//         - Try free list (same size class, non-overlapping lifetime)
//         - Else bump pointer (next_vtcm_offset_ += align_up(size, 128))
//   3. Return primary (first) block_id
uint32_t FancyAllocator::force_contiguous(const uint32_t* block_ids, size_t count,
                                           const uint32_t* sizes,
                                           const uint32_t* life_begins,
                                           const uint32_t* life_ends) {
    if (count == 0) return 0;

    uint32_t primary_block_id = 0;

    for (size_t i = 0; i < count; ++i) {
        uint32_t bid = block_ids[i];
        uint32_t sz = sizes[i];
        uint32_t lb = life_begins[i];
        uint32_t le = life_ends[i];

        if (i == 0) primary_block_id = bid;

        // Check if block already registered in hash table
        BlockHashEntry* existing = block_hash_find(bid);
        if (existing) {
            // Already registered → update size if needed
            existing->size = sz;
            // Update lifetime if we have better info
            auto it = block_id_to_idx_.find(bid);
            if (it != block_id_to_idx_.end()) {
                auto& be = block_table_[it->second];
                be.life_begin = lb;
                be.life_end = le;
            }
            continue;
        }

        // Register new block in hash table
        block_hash_insert(bid, sz);

        // Add to block table
        BlockEntry be;
        be.block_id = bid;
        be.size = sz;
        be.life_begin = lb;
        be.life_end = le;
        be.flag = 1;  // used
        be.vtcm_offset = 0;
        be.linked_to = 0;
        block_id_to_idx_[bid] = static_cast<uint32_t>(block_table_.size());
        block_table_.push_back(be);

        // Try to find a free block with non-overlapping lifetime for reuse
        // (反汇编: free list lookup at [this+0x190] by size class)
        uint32_t sc = size_to_class(sz);
        uint32_t reuse_bid = free_list_try_pop(sc, lb, le);

        if (reuse_bid != 0) {
            // Reuse: share VTCM offset with the free block
            auto& new_be = block_table_[block_id_to_idx_[bid]];
            auto& reuse_be = block_table_[block_id_to_idx_[reuse_bid]];
            new_be.vtcm_offset = reuse_be.vtcm_offset;
            new_be.linked_to = reuse_bid;
            new_be.flag = 2;  // linked/reused
            overlap_saved_ += ((sz + 127) & ~127u);  // aligned size saved
        } else {
            // No reuse → allocate new VTCM space (bump pointer)
            uint64_t aligned_sz = (static_cast<uint64_t>(sz) + 127) & ~127ull;
            if (overlap_budget_ > 0 && overlap_next_offset_ + aligned_sz > overlap_budget_) {
                // Doesn't fit → spill to DDR (offset = 0, spilled = true)
                // Block is still registered but gets no VTCM offset
            } else {
                block_table_.back().vtcm_offset = overlap_next_offset_;
                overlap_next_offset_ += aligned_sz;
            }
        }
    }

    return primary_block_id;
}

// ===== link_blocks: establish reuse link between two blocks =====
// 反汇编 @ 0x13a0ac0 (2542B)
//
// Algorithm (from disassembly):
//   1. If id1 == id2, return (反汇编 @ 0x13a0ae4-0x13a0b0a)
//   2. Look up both blocks in block table (via map at [this+0x180])
//      (反汇编 @ 0x13a0b2c-0x13a0b87: two map.find calls)
//   3. Compare sizes (反汇编 @ 0x13a0bc7: cmp eax, ecx)
//      - If equal: share VTCM offset (same size class)
//      - If different: larger block's offset shared with smaller
//   4. Add freed block to free list for future reuse
void FancyAllocator::link_blocks(uint32_t id1, uint32_t id2) {
    // Step 1: same block → nothing to do
    if (id1 == id2) return;

    // Step 2: look up both blocks
    auto it1 = block_id_to_idx_.find(id1);
    auto it2 = block_id_to_idx_.find(id2);
    if (it1 == block_id_to_idx_.end() || it2 == block_id_to_idx_.end()) return;

    BlockEntry& b1 = block_table_[it1->second];
    BlockEntry& b2 = block_table_[it2->second];

    // Step 3: compare sizes and establish reuse
    if (b1.size == b2.size) {
        // Same size: share VTCM offset
        // (反汇编 @ 0x13a0bc9: je 0x13a0ae8 — equal sizes, direct share)
        if (b1.vtcm_offset != 0 && b2.vtcm_offset == 0) {
            b2.vtcm_offset = b1.vtcm_offset;
            b2.linked_to = id1;
            b2.flag = 2;  // linked
            overlap_saved_ += ((b2.size + 127) & ~127u);
        } else if (b2.vtcm_offset != 0 && b1.vtcm_offset == 0) {
            b1.vtcm_offset = b2.vtcm_offset;
            b1.linked_to = id2;
            b1.flag = 2;
            overlap_saved_ += ((b1.size + 127) & ~127u);
        }
    } else {
        // Different sizes: larger block can accommodate smaller
        // (反汇编 @ 0x13a0bcf-0x13a0e1f: check link map for cross-size linking)
        BlockEntry* larger = (b1.size > b2.size) ? &b1 : &b2;
        BlockEntry* smaller = (b1.size > b2.size) ? &b2 : &b1;

        if (larger->vtcm_offset != 0 && smaller->vtcm_offset == 0) {
            smaller->vtcm_offset = larger->vtcm_offset;
            smaller->linked_to = larger->block_id;
            smaller->flag = 2;
            overlap_saved_ += ((smaller->size + 127) & ~127u);
        }
    }

    // Step 4: add the input block (id1) to free list after it's consumed
    // (反汇编 @ 0x13a0cda-0x13a0ce2: decrement free list count, mark as free)
    // The input block's lifetime has ended (it was consumed by this op),
    // so its VTCM space can be reused by future blocks.
    if (b1.flag == 1) {
        // Only add to free list if the block is still "used" (not already linked)
        // Check that the block's lifetime has ended relative to this op
        // In the .so, this happens implicitly through the block table's lifetime tracking
        free_list_add(id1);
    }
}

} // namespace fa

namespace hnnx {

VtcmCacheInstance::VtcmCacheInstance(uint32_t nsp_id, size_t vtcm_size)
    : nsp_id_(nsp_id), vtcm_size_(vtcm_size) {}

VtcmCacheInstance::~VtcmCacheInstance() = default;

// Actual usable VTCM = min(size, 4MB) × 0.75
// Source: do_prepare1 @ 0xf671f4-0xf67249 (4MB cap)
// Source: sap_reduce_bandwidth @ 0xf6ee78 (mulsd 0.75 coefficient)
// 0x400000 = 4MB hard limit; 0x3fe8000000000000 = 0.75 as double
// 1/4 of budget reserved for system/stack, 3/4 usable for tensors
size_t VtcmCacheInstance::usable_size() const {
    uint64_t capped = std::min<uint64_t>(vtcm_size_, VTCM_4MB_LIMIT);
    return static_cast<size_t>(static_cast<double>(capped) * VTCM_USABLE_FACTOR);
}

} // namespace hnnx
