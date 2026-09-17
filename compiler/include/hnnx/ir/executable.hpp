#pragma once
// ============================================================================
// hnnx::Executable 字节级重实现 —— Op 的唯一基类
// libHtpPrepare.so 2.48.40.260702, x86_64-linux-clang, stripped
//
// 证据链 (全部字节级, 无推断):
//   * _ZTVN4hnnx10ExecutableE @0x5ebd458, size 0x48 = 7 槽:
//       +0x00 __cxa_pure_virtual                 → execute() 纯虚
//       +0x08 _ZNK4hnnx10Executable7compileER5Graph        @0xd0eb00 (10B)
//       +0x10 0xd0e110 (2B 跳板: mov (%rdi),%rax; mov 0x8(%rax),%rax; jmp *%rax)
//       +0x18 0xd0e120 (与 +0x10 逐字节相同的跳板)
//       +0x20 _ZNK4hnnx10Executable30check_constraint_for_recompileER5Graph @0xd0eb10 (3B)
//       +0x28 0xd0eb30 (1B: retq)                → D1
//       +0x30 0xd0eb40 (2B: ud2)                 → D0 (抽象类, 不可达陷阱)
//   * _ZTIN4hnnx10ExecutableE = __class_type_info → 无基类的根类
//   * 静态成员导出符号:
//       _ZN4hnnx10Executable6vtableEPKS0_       @0xd0eae0 (4B: mov (%rdi),%rax; ret)
//       _ZN4hnnx10Executable15execute_addressEPKS0_ @0xd0eaf0 (7B: mov (%rdi),%rax; mov (%rax),%rax; ret)
//       _ZN4hnnx10Executable14no_op_functionEPKvP5GraphNS_3OsSE @0xd0eb20 (3B: xor %eax,%eax; ret)
//     null_item() 内联证据 @0x10bd060:
//       mov GOT(0x623f038=no_op_function),%rax; xor %edx,%edx; ret → {no_op_function, nullptr}
//   * Op vtable (op.hpp) 前 7 槽与此表逐槽一致 (Op 仅覆盖 +0x30 D0=自己的 ud2 @0x10bd040,
//     D1 与基类同址 0xd0eb30 —— 平凡空析构代码折叠)。
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <utility>

#include "status.hpp" // ::GraphStatus struct (SDK graph_status.h; M32 重写)

class Graph;

namespace hnnx {

// list_type.h —— enum hnnx::ListType (compile 第 3 重载参数)
enum ListType { MainList, VecList, MtxList, EltList };
constexpr unsigned ListTypeCount = EltList + 1;

// ----------------------------------------------------------------------------
// hnnx::OsS = op_slice_spec —— 32 位位域 (可经单寄存器传入汇编执行体)
//   bits [13: 0] m_nslices | [15:14] m_resources | [31:16] m_slice_idx
// (SDK executable.h:29-86 原文位布局; m_resources 必须占据低 16 位的高 2 位)
// ----------------------------------------------------------------------------
struct OsS {
  protected:
    unsigned m_nslices : 14;
    unsigned m_resources : 2;
    unsigned m_slice_idx : 16;

  public:
    OsS(OsS const &) = default;
    OsS &operator=(OsS const &) = default;
    constexpr OsS() : m_nslices(1), m_resources(0), m_slice_idx(0) {}
    constexpr OsS(unsigned const n, unsigned const i) : m_nslices(n), m_resources(0), m_slice_idx(i) {}
    constexpr OsS(unsigned const r, unsigned const n, unsigned const i)
        : m_nslices(n), m_resources(r), m_slice_idx(i)
    {
    }
    void set_nslices(unsigned n) { m_nslices = n; }
    constexpr unsigned num_slices() const { return m_nslices; }
    constexpr unsigned slice_idx() const { return m_slice_idx; }
    hnnx::ListType resources() const { return static_cast<hnnx::ListType>(m_resources); }

    unsigned as_uint32() const
    {
        union {
            OsS ss;
            unsigned as_u;
        } uu = {*this};
        return uu.as_u;
    }

    void from_uint32(unsigned x)
    {
        union {
            unsigned u;
            OsS ss;
        } uu = {x};
        *this = uu.ss;
    }
};
using op_slice_spec = OsS;

typedef volatile uint32_t *counter_t;
typedef volatile uint32_t *counter_nc_t;

// ----------------------------------------------------------------------------
// class hnnx::Executable —— execute() 必须位于 vtable 槽 0 (SDK 原注释:
// "THE execute() VIRTUAL FUNCTION MUST BE THE 0th THING IN THE VTABLE")
// ----------------------------------------------------------------------------
class Executable {
  public:
    static constexpr unsigned MAX_OP_HVX_SLICES = 8;
    static constexpr unsigned MAX_OP_HMX_SLICES = 2;
    static constexpr unsigned MAX_ALL_OP_SLICES = MAX_OP_HVX_SLICES + MAX_OP_HMX_SLICES + 1;

    using FuncType = GraphStatus (*)(const void *, Graph *, hnnx::op_slice_spec);
    using ItemType = std::pair<FuncType, const void *>;
    struct alignas(16) ExecType { // 4 指针, 对齐 16
        FuncType funcp;
        const void *datap;
        counter_t gate_cp;
        counter_t done_cp;
        ExecType(FuncType const f, const void *const d, counter_t const gc, counter_t const dc)
            : funcp(f), datap(d), gate_cp(gc), done_cp(dc)
        {
        }
        ExecType &operator=(ExecType const &rhs) = default;
        ExecType() : funcp{}, datap{}, gate_cp{}, done_cp{} {}
    };

    // +0x00 (纯虚, 强制槽 0)
    virtual GraphStatus execute(Graph * /*gr*/, hnnx::op_slice_spec) const noexcept = 0;

    // +0x08 @0xd0eb00 (10B): mov (%rdi),%rax; mov (%rax),%rax; mov %rdi,%rdx; ret
    //   → 返回 {vtable[0] 即 execute 地址, this}
    virtual ItemType compile(Graph & /*graph_in*/) const
    {
        return ItemType{reinterpret_cast<FuncType>(
                            reinterpret_cast<void *const *>(hnnx::Executable::vtable(this))[0]),
                        static_cast<const void *>(this)};
    }
    // +0x10 @0xd0e110 跳板: 取 vtable[1] 尾跳 → 默认忽略自切片, 直接 compile(graph_in)
    virtual ItemType compile(Graph &graph_in, unsigned /*nslices*/) const { return this->compile(graph_in); }
    // +0x18 @0xd0e120 同一跳板 → 默认忽略 list_type
    virtual ItemType compile(Graph &graph_in, hnnx::ListType) const { return this->compile(graph_in); }
    // +0x20 @0xd0eb10 (3B: xor %eax,%eax; ret) → 默认 false
    virtual bool check_constraint_for_recompile(Graph & /*graph_in*/) const { return false; }
    // +0x28 D1 @0xd0eb30 (ret) / +0x30 D0 @0xd0eb40 (ud2, 抽象类不可达)
    virtual ~Executable() = default;

    // @0xd0eae0 (4B: mov (%rdi),%rax; ret) → 返回 vptr 值(即指向槽 0 的指针)
    static const size_t *vtable(Executable const *e) { return *reinterpret_cast<size_t const *const *>(e); }
    // @0xd0eaf0 (7B) → vtable(e)[0] = execute 函数地址
    static size_t execute_address(Executable const *e) { return hnnx::Executable::vtable(e)[0]; }

    // @0xd0eb20 (3B: xor %eax,%eax; ret) → GraphStatus::Success
    static GraphStatus no_op_function(const void *, Graph *, hnnx::op_slice_spec)
    {
        return GraphStatus::Success;
    }
    // 内联证据 @0x10bd060: {GOT(no_op_function), nullptr}
    static ItemType null_item() { return ItemType{no_op_function, nullptr}; }
};

// execute_item (SDK 原内联, 无独立符号)
inline GraphStatus execute_item(Graph *graph_in, Executable::ExecType const &itemt)
{
    return (*itemt.funcp)(itemt.datap, graph_in, op_slice_spec{});
}

} // namespace hnnx
