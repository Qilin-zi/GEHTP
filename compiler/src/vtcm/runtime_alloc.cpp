// ============================================================================
// fa::RuntimeAllocator —— 分配器中间层 (P3 行为保持重构, 2026-09-21)
// 规范: docs/GEHTP_BACKPORT_PLAN.md §1; 布局定稿见 runtime_alloc.hpp 头注。
// 本文件只含: ctor/dtor 锚点、make_allocator 槽位工厂 (§1.2)、三槽访问
// (§1.5② 越界抛 std::out_of_range)。跨组复用算法属 P5+ (GAP_CLOSURE §P3)。
// ============================================================================
#include "hnnx/vtcm/runtime_alloc.hpp"
#include "hnnx/vtcm/fancy_allocator.hpp" // make_allocator fancy 分支需完整类型
#include <stdexcept>

namespace fa {

RuntimeAllocator::RuntimeAllocator() = default;

RuntimeAllocator::RuntimeAllocator(uint32_t mode, HNNX_GRAPH_T &graph) : graph_(&graph), mode_(mode) {}

RuntimeAllocator::~RuntimeAllocator() = default;

// ---- §1.2 make_allocator 槽位工厂 (@0xf4e3b0) -------------------------------
// .so: fancy ? new(0x3a0)+FancyAllocator : new(0xb8)+RuntimeAllocator,
//      两分支 Mode 均传 esi=0 (AllocVirtual)。
RuntimeAllocator *RuntimeAllocator::make_allocator(HNNX_GRAPH_T &graph, bool fancy)
{
    if (fancy) return new FancyAllocator(MODE_ALLOC_VIRTUAL, graph);
    return new RuntimeAllocator(MODE_ALLOC_VIRTUAL, graph);
}

// 槽位变体: 写入 *(Allocator**)this = p 并返回 this (@0xf4e3b0 写槽面);
// 旧占用者经虚析构删除 (= .so 调用方 "vptr+0x08 deleting dtor" 面)。
RuntimeAllocator **RuntimeAllocator::make_allocator_at(RuntimeAllocator **slot, HNNX_GRAPH_T &graph, bool fancy)
{
    if (slot == nullptr) return nullptr;
    RuntimeAllocator *p = make_allocator(graph, fancy);
    RuntimeAllocator *old = *slot;
    if (old != nullptr) delete old; // 虚析构 → 派生 deleting dtor
    *slot = p;
    return slot;
}

// ---- §2.5 三槽访问; 越界按 §1.5② 冷径抛 std::out_of_range("vector") --------
// (.so 冷径 @0xd8ed50: __cxa_allocate_exception + _ZTISt12out_of_range)
uint32_t RuntimeAllocator::slot_size(unsigned i) const
{
    if (i >= SLOT_COUNT) throw std::out_of_range("vector");
    return slot_sizes_[i];
}

void RuntimeAllocator::set_slot_size(unsigned i, uint32_t v)
{
    if (i >= SLOT_COUNT) throw std::out_of_range("vector");
    slot_sizes_[i] = v;
}

} // namespace fa
