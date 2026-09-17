// hnnx::Crate 记录分配 — M34 过渡 TU (待 Crate 里程碑整体替换)
//
// · bump_record_count: 精确 —— 全部调用点为 `addq $1, 0x40(crate)`
//   (canonical_shape @0x12f7f4c / crated_shape @0x12f7cc4 / Shape<4>::
//   deserialize @0xd94648 / VariadicOpBase::assign_input_pointers @0x139fcb1)。
//
// · add_record_slot: 签名/返回结构精确 (sret 24B {未读, slot@+8, status@+0x10});
//   arena 内部算法为过渡实现。真身 @0xcf3ff0 (0x2b0 字节) 基于 chunk 链
//   (头藏于分配区前 8 字节, {u32@0, u32@4, u32@8}), 自由表于 +0x18/+0x20,
//   原始模式标志 +0x30, 0x10000 chunk 启发 —— 连同 chunkhdr::allocate
//   (@0xcf3ea0) / allocate_bulk (@0xcf4730) / move_to_free (@0xcf4550) 属
//   Crate 里程碑。过渡 bump 状态占用 crate+0x48/+0x50 (真 sizeof(Crate)=0x48
//   之外): 仅测试用伪造 Graph 缓冲区场景合法。
#include "hnnx/ir/tensor_base.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
struct bump_ctl { // 过渡 bump 状态 (crate+0x48)
    uint64_t cursor;
    uint64_t limit;
};
} // namespace

namespace hnnx {

Crate::record_slot_result Crate::add_record_slot(size_t bytes, size_t align)
{
    char *const self = reinterpret_cast<char *>(this);
    bump_ctl *const ctl = reinterpret_cast<bump_ctl *>(self + 0x48);
    if (ctl->limit == 0) {
        std::fprintf(stderr, "REQNN Crate::add_record_slot: 过渡 bump 区未初始化 (crate+0x48)\n");
        std::abort();
    }
    uintptr_t const p = (ctl->cursor + align - 1) & ~uintptr_t(align - 1);
    if (p + bytes > ctl->limit) {
        std::fprintf(stderr, "REQNN Crate::add_record_slot: 过渡 bump 区耗尽 (%zu 字节)\n", bytes);
        std::abort();
    }
    ctl->cursor = p + bytes;
    return record_slot_result{nullptr, reinterpret_cast<void **>(p), 0};
}

void Crate::bump_record_count()
{
    ++*reinterpret_cast<uint64_t *>(reinterpret_cast<char *>(this) + 0x40);
}

} // namespace hnnx
