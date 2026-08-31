// hnnx::throw_dcrate_seg_overflow @0xcf5240 — 逐指令
//
//   push rax; __cxa_allocate_exception(8); 异常对象首 8 字节 = vptr
//   (0x5ebde58+0x10); __cxa_throw(obj, typeinfo(GOT 0x623f180), 0)
//   → 即 throw dcrate_seg_overflow_error(); (空类, 8B = 纯 vptr)。
//   what() 文本未解码 (vtable 域 ICF 于 _ZTV5Graph+0x200 邻域), 本侧占位。
#include "hnnx/serialize/deserz.hpp"

namespace hnnx {

[[noreturn]] void throw_dcrate_seg_overflow()
{
    throw dcrate_seg_overflow_error();
}

char const *dcrate_seg_overflow_error::what() const noexcept
{
    return "hnnx::dcrate_seg_overflow_error"; // .so 文本未解码 —— 占位
}

} // namespace hnnx
