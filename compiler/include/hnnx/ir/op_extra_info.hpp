#pragma once
// ============================================================================
// op_extra_info.hpp — hnnx::OpExtraInfo (SDK op_extra_info.h)
//
// M33 自 op.hpp 拆出 (原文不变): deserz.hpp 需 OpExtraInfo 完整类型作
// Deserz+0x80 成员, 而 op.hpp 又包含 tensor_base.hpp → 后者含 deserz.hpp,
// 不拆则成环。
//
// 构造 @0xd4e010 证据: 栈上仅初始化 {id@0, chkpts@8}=16B, op_tag 不写
// (SDK 同样不初始化; Deserz ctor 对 +0x90 同样不触及)。
// get_op_store_type/get_serialize_flags 读 OpExtraAttrib 基类后首个位域单元
// @+0x18 的 16 位字并测 bit0 (movzwl 0x18(%rax); test $1) = for_hlx
// ============================================================================
#include <cstdint>
#include <utility>

typedef unsigned long long OpId; // interface_defs.h:17 (mangling y)

namespace hnnx {

struct OpExtraInfo {
    using Chkpts = std::pair<int, int>;

    OpId id;          // +0x00 (id() 直接返回)
    Chkpts chkpts;    // +0x08 (set_chkpts 整体 64 位写入 @0x10bc872)
    const char *op_tag; // +0x10 (构造不初始化)
    explicit OpExtraInfo(OpId id_in) : id(id_in), chkpts(-1, -1) {}
    OpExtraInfo(OpId id_in, int cg, int dc) : id(id_in), chkpts(cg, dc) {}
    OpExtraInfo() : OpExtraInfo(0) {}

    bool valid() const { return id != 0; };
    void clear() { id = 0; };
};
static_assert(sizeof(OpExtraInfo) == 24); // 0x18 (x86-64: 8+8+8)

} // namespace hnnx
