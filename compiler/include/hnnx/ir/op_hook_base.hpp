#pragma once
// ============================================================================
// hnnx::OpHookBase —— SDK op_hook_base.h 精确复刻 (M32)。
//
// 用途: Op 构造期挂钩 (ctor hook)。无可数据成员、无析构 —— 整个对象就是
// 一个 vptr, 可 constexpr 构造。全部方法 const 且返回 GraphStatus。
// 多个 hook 可层叠: 依次调用, 返回 NotApplicable 则试下一个。
//
// 证据 (libHtpPrepare.so):
//   导出: _ZNK4hnnx10OpHookBase15pre_output_prepERKNS_8OpIoPtrsER2Op @0xd4e270
//         _ZNK4hnnx10OpHookBase12pre_allocateERKNS_8OpIoPtrsER2Op  @0xd4e280
//   vtable _ZTVN4hnnx10OpHookBaseE @0x5ebeab8 (重定位钉死):
//         [+0x10] → pre_output_prep, [+0x18] → pre_allocate
//   两个默认体均为 5 字节 `movl $0x64,%eax; ret` —— 返回 NotApplicable(100),
//   与 SDK 头注释 "the 'default' methods do nothing and return
//   GraphNotApplicable" 一致。
//   pmf 编码 (调用方 TypicalOpUtil::assign_input_pointers/output_allocate):
//         pre_output_prep: {1,0}  (vtable 字节偏移 1*8 = +0x10, 槽 0)
//         pre_allocate:    {9,0}  (vtable 字节偏移 9*8-0x40 = +0x18, 槽 1;
//                                  Itanium pmf 首字 = vtable 偏移+1)
//   ophook_func @0xd351a0 (OpIoPtrs 受保护成员, 见 op_io_ptrs.hpp)。
// ============================================================================

#include "status.hpp"

class Op;

namespace hnnx {

class OpIoPtrs;

class OpHookBase {
  public:
    virtual GraphStatus pre_output_prep(OpIoPtrs const &, Op &) const; // @0xd4e270 → NotApplicable
    virtual GraphStatus pre_allocate(OpIoPtrs const &, Op &) const; // @0xd4e280 → NotApplicable
};

} // namespace hnnx
