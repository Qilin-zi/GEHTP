// ============================================================================
// Interface / PlainInterface / ScaleOffsetInterface / ifc_method_table
// 精确反演自 libHtpPrepare.so (QNN 2.48.40.260702) — M34
//
// 字节级证据 (全部 objdump 直证):
//   · ifc_method_table @0x5ec7fd0, sizeof 0x210 = 11 条目 × 0x30 (.data.rel.ro,
//     无动态重定位 —— 函数地址为烤入文件镜像的绝对值)。
//     符号 _ZN4hnnx16ifc_method_tableE (GOT 0x623f050 GLOB_DAT 直证)。
//   · 条目布局: IfcExemplar exemplar@0 (dtype_info 4B + 4B pad);
//     read_float@8; write_float@0x10; get_qparms@0x18; ifc_hash@0x20;
//     ifc_compare@0x28。InterfaceRef::methods_p 即条目首址 (get_refobj
//     @0xdcd390: leaq table+dt*0x30)。
//   · thunk 块 @0xdcd040..0xdcd38e (局部符号, 内部链接):
//       0xdcd040 read  (UNKNOWN → 0.0f)         0xdcd050 write (UNKNOWN → nop)
//       0xdcd060 qparms(plain → &null_parms)     0xdcd070 qparms(quant → this+4)
//       0xdcd080 hash (quant)                    0xdcd0e0 compare (quant)
//       0xdcd120/130 read/write QInt16           0xdcd180/1a0 QInt32
//       0xdcd1d0/1e0 QInt8                       0xdcd230/290 Float16
//       0xdcd350/360 BFloat16
//   · Interface 无 vptr (constexpr 构造, 全局单例); sizeof(PlainInterface)=4,
//     sizeof(ScaleOffsetInterface)=16 (dtype_info@0, offset@+4, scale@+8,
//     scale_recip@+0xc —— 与 get_qparms thunk 0xdcd070 `lea 0x4(%rdi)` 及
//     ifc_hash 0xdcd080 读 +4/+8 逐字节互证)。
//   · Interface::null_parms @0x39b5ea4 (.rodata): {0, 1.0f, 1.0f}
//     (GOT 0x623f480 GLOB_DAT, 符号 _ZN9Interface10null_parmsE)。
//   · dtype_info 位域 (LE): {elbytes:8; dtype:8; is_quant:1; is_float:1;
//     is_signed:1} —— 表中 11 个 exemplar 字节序列直证。
// ============================================================================
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "../ir/op_def.hpp" // 全局 ::DType (.so 值域: UNKNOWN=0..BFloat16=10)
#include "../ir/tensor_base.hpp" // hnnx::InterfaceRef / 全局前置

// ---------------------------------------------------------------------------
// struct dtype_info — 全局作用域 (SDK dtype.h 原文位域)
// ---------------------------------------------------------------------------
struct dtype_info {
    unsigned elbytes : 8;
    unsigned dtype : 8;
    unsigned is_quant : 1;
    unsigned is_float : 1;
    unsigned is_signed : 1;
};
static_assert(sizeof(dtype_info) == 4);

// ---------------------------------------------------------------------------
// class Interface — 全局作用域, 无 vptr (符号 _ZNK9Interface10get_refobjEv
// @0xdcd390 直证非虚; 所有方法经 ifc_method_table 分发)
// ---------------------------------------------------------------------------
class Interface {
  public:
    // qparms 与 ScaleOffsetInterface 的 +4..+0xc 三字段逐字节同构
    // (thunk 0xdcd070 直接返回 this+4); 命名空间级孪生见 hnnx::ifc_qparms
    using qparms = hnnx::ifc_qparms;
    static_assert(sizeof(qparms) == 12);

    using read_float_fp = float (*)(Interface const *, void const *) noexcept;
    using write_float_fp = void (*)(Interface const *, void *, float) noexcept;
    using get_qparms_fp = qparms const *(*)(Interface const *) noexcept;
    using ifc_hash_fp = uint32_t (*)(Interface const *) noexcept;
    using ifc_compare_fp = int (*)(Interface const *, Interface const *) noexcept;

    static unsigned constexpr N_types = 11; // ZZ_LAST_DTYPE (dtype_enum.h)

    // @0x39b5ea4 (.rodata): {0, 1.0f, 1.0f}
    static qparms const null_parms;

    constexpr dtype_info const &get_dt_info() const noexcept { return dtinfo; }
    constexpr unsigned element_size() const noexcept { return dtinfo.elbytes; }
    constexpr ::DType get_dtype() const noexcept { return (::DType)dtinfo.dtype; }
    constexpr bool is_quantized() const noexcept { return dtinfo.is_quant != 0; }

    // get_refobj @0xdcd390 精确复刻:
    //   dt = dtinfo.dtype; dt >= 0xb → 0 (clamp);
    //   e = &hnnx::ifc_method_table[dt];
    //   if (dtinfo.dtype != e->exemplar.dtype || dtinfo.elbytes != e->exemplar.elbytes)
    //       → qnndsp_log 告警 (构造不失败);
    //   return {e, this};
    hnnx::InterfaceRef get_refobj() const;

  protected:
    // thunk 0xdcd060: 返回 &null_parms
    static qparms const *get_null_qparms(Interface const *) noexcept { return &null_parms; }

    alignas(4) dtype_info dtinfo; // +0x00 (4B)

    constexpr Interface() noexcept : dtinfo{} {}
    constexpr explicit Interface(dtype_info const di) noexcept : dtinfo(di) {}

    ~Interface() = default;
    // 派生类 (IfcExemplar/PlainInterface/ScaleOffsetInterface) 经 protected
    // ctor 构造 —— 与 .so constexpr 单例构造一致, 无需 friend。
};

// ---------------------------------------------------------------------------
// PlainInterface<T> / ScaleOffsetInterface<T> — 全局模板 (SDK interface.h)
// ---------------------------------------------------------------------------
class NullInterface final : public Interface {
  public:
    constexpr NullInterface() noexcept : Interface() {}
};

template <typename T> class dtype_info_for; // SDK dtype.h: 每类型特化

template <typename T> class PlainInterface final : public Interface {
  public:
    static constexpr ::DType dtype = dtype_info_for<T>::value;
    constexpr PlainInterface() noexcept : Interface(dtype_info_for<T>::info()) {}
    // interface.h:549 原文 — DTypeScaleOff(dtype) = {dtype, 1.0f, 0}
    // (ConcreteTensor::get_dtype_intfc @vcall+0x48 → gen_output_def @0xdac84e
    //  读 rax 高 32 位 = scale 的调用链互证: plain 张量 stepsize 恒 1.0f)
    static ::DTypeScaleOff get_dtype_scaleoff() noexcept { return ::DTypeScaleOff(dtype, 1.0f, 0); }
    // interface.h:551-554 原文 — 同型 plain 接口全相等
    static int compare(PlainInterface const &) noexcept { return 0; }
    static float get_scale() noexcept { return 1.0f; }
    static float get_scale_recip() noexcept { return 1.0f; }
    static int32_t get_offset() noexcept { return 0; }
};

template <typename T> class ScaleOffsetInterface final : public Interface {
  public:
    static constexpr ::DType dtype = dtype_info_for<T>::value;
    constexpr ScaleOffsetInterface() noexcept : Interface(dtype_info_for<T>::info()), offset(0), scale(1.0f),
                                               scale_recip(1.0f) {}
    // from_odef @0xd13b90 族: {dtinfo; def+0x48→offset; def+0x4c→scale;
    // 1.0f/scale→recip}; scale==0 且 !isnan 时 GetLogPriorityLevel()>=0xb
    // → qnndsp_log(0xb, ...) (构造不失败)。
    constexpr ScaleOffsetInterface(int32_t const offset_in, float const scale_in) noexcept
            : Interface(dtype_info_for<T>::info()), offset(offset_in), scale(scale_in),
              scale_recip(1.0f / scale_in)
    {
    }

    int32_t offset;      // +0x04
    float scale;         // +0x08
    float scale_recip;   // +0x0c

    // interface.h:795 原文 — DTypeScaleOff(dtype, qparms) = {dtype, scale, offset}
    ::DTypeScaleOff get_dtype_scaleoff() const noexcept { return ::DTypeScaleOff(dtype, scale, offset); }
    // interface.h:703-716 原文 (SOIfcBase::compare): 先 offset 后 scale;
    // compare_eq 同字段 (与 .so 0x1340480 / 0xdcd0e0 逐位一致)
    int compare(ScaleOffsetInterface const &rhs) const noexcept
    {
        if (offset != rhs.offset) return (offset < rhs.offset) ? -1 : 1;
        if (scale != rhs.scale) return (scale < rhs.scale) ? -1 : 1;
        return 0;
    }
    bool compare_eq(ScaleOffsetInterface const &rhs) const noexcept
    {
        return offset == rhs.offset && scale == rhs.scale;
    }
    // interface.h:718-720 原文
    float get_scale() const noexcept { return scale; }
    float get_scale_recip() const noexcept { return scale_recip; }
    int32_t get_offset() const noexcept { return offset; }
};
static_assert(sizeof(PlainInterface<float>) == 4);
static_assert(sizeof(ScaleOffsetInterface<unsigned char>) == 16);

// dtype_info_for<T> 特化 —— .so 表中 11 个条目的 dtype_info 全集
template <> class dtype_info_for<unsigned char> {
  public:
    static constexpr ::DType value = ::DType::QUInt8;
    static constexpr dtype_info info() noexcept
    {
        return dtype_info{1u, unsigned(::DType::QUInt8), 1u, 0u, 0u};
    }
};
template <> class dtype_info_for<unsigned short> {
  public:
    static constexpr ::DType value = ::DType::QUInt16;
    static constexpr dtype_info info() noexcept
    {
        return dtype_info{2u, unsigned(::DType::QUInt16), 1u, 0u, 0u};
    }
};
template <> class dtype_info_for<short> {
  public:
    static constexpr ::DType value = ::DType::QInt16;
    static constexpr dtype_info info() noexcept
    {
        return dtype_info{2u, unsigned(::DType::QInt16), 1u, 0u, 1u};
    }
};
template <> class dtype_info_for<float> {
  public:
    static constexpr ::DType value = ::DType::Float32;
    static constexpr dtype_info info() noexcept
    {
        return dtype_info{4u, unsigned(::DType::Float32), 0u, 1u, 0u};
    }
};
template <> class dtype_info_for<int> {
  public:
    static constexpr ::DType value = ::DType::Int32;
    static constexpr dtype_info info() noexcept
    {
        return dtype_info{4u, unsigned(::DType::Int32), 0u, 0u, 1u};
    }
};
template <> class dtype_info_for<long> { // Int64 (elbytes 8, signed)
  public:
    static constexpr ::DType value = ::DType::Int64;
    static constexpr dtype_info info() noexcept
    {
        return dtype_info{8u, unsigned(::DType::Int64), 0u, 0u, 1u};
    }
};

namespace hnnx {

// ---------------------------------------------------------------------------
// IfcExemplar — 表内 8 字节标识槽 (dtype_info + pad; 量化全量参数不在表中,
// 由 from_odef/from_deser 在栈上构建 16B exemplar 后经 from_exemplar 规范化)
// ---------------------------------------------------------------------------
class IfcExemplar final : public ::Interface {
  public:
    constexpr IfcExemplar() noexcept : Interface() {}
    constexpr explicit IfcExemplar(::dtype_info const di) noexcept : Interface(di) {}
};

// ---------------------------------------------------------------------------
// intfc_methods / ifc_method_table — 单条目 0x30 = 11 × 0x30 = 0x210
// ---------------------------------------------------------------------------
struct intfc_methods {
    IfcExemplar exemplar;                    // +0x00 (8B)
    ::Interface::read_float_fp read_float;    // +0x08
    ::Interface::write_float_fp write_float;  // +0x10
    ::Interface::get_qparms_fp get_qparms;    // +0x18
    ::Interface::ifc_hash_fp ifc_hash;        // +0x20
    ::Interface::ifc_compare_fp ifc_compare;  // +0x28
};
static_assert(sizeof(intfc_methods) == 0x30);

using ifc_method_table_t = std::array<intfc_methods, ::Interface::N_types>;

// 定义于 src/tens/interface_table.cpp —— 值域逐字节对表 @0x5ec7fd0
extern ifc_method_table_t const ifc_method_table;

// ---------------------------------------------------------------------------
// InterfaceRef 方法族 — interface.h:216-262 原文, 经 methods_p 分发
// ---------------------------------------------------------------------------
inline ::Interface::qparms const *InterfaceRef::get_qparms() const
{
    auto const *const m = static_cast<intfc_methods const *>(methods_p);
    return m->get_qparms(static_cast<::Interface const *>(intfc_p));
}
inline float InterfaceRef::get_scale() const { return get_qparms()->scale; }
inline float InterfaceRef::get_scale_recip() const { return get_qparms()->scale_recip; }
inline int32_t InterfaceRef::get_offset() const { return get_qparms()->offset; }
// interface.h:223-226: ifc_hash 为空 (plain 族) 时按 0; 尾部异或 dtype ——
// 与 from_exemplar 的 BST 键 ((scale位<<1) ^ (offset·0x10661)) ^ dt 恒等
inline uint32_t InterfaceRef::interface_hash() const noexcept
{
    auto const *const m = static_cast<intfc_methods const *>(methods_p);
    uint32_t const h = m->ifc_hash ? m->ifc_hash(static_cast<::Interface const *>(intfc_p)) : 0;
    return h ^ uint32_t(m->exemplar.get_dtype());
}
inline ::DType InterfaceRef::get_dtype() const noexcept
{
    return static_cast<intfc_methods const *>(methods_p)->exemplar.get_dtype();
}
inline unsigned InterfaceRef::element_size() const noexcept
{
    return static_cast<intfc_methods const *>(methods_p)->exemplar.element_size();
}
inline bool InterfaceRef::is_quantized() const noexcept
{
    return static_cast<intfc_methods const *>(methods_p)->exemplar.is_quantized();
}
// interface.h:246-251: 表条目不同 → 按条目地址 (即按 dtype) 定序;
// 同表同对象 → 0; 同表异对象 → ifc_compare (plain 恒 0)
inline int InterfaceRef::compare(InterfaceRef const &rhs) const noexcept
{
    if (methods_p != rhs.methods_p) return (methods_p < rhs.methods_p) ? -1 : 1;
    if (intfc_p == rhs.intfc_p) return 0;
    auto const *const m = static_cast<intfc_methods const *>(methods_p);
    auto const fp = m->ifc_compare;
    return (fp == nullptr) ? 0 : (*fp)(static_cast<::Interface const *>(intfc_p),
                                       static_cast<::Interface const *>(rhs.intfc_p));
}
inline bool InterfaceRef::compare_eq(InterfaceRef const &rhs) const noexcept
{
    if (methods_p != rhs.methods_p) return false;
    if (intfc_p == rhs.intfc_p) return true;
    auto const *const m = static_cast<intfc_methods const *>(methods_p);
    auto const fp = m->ifc_compare;
    return (fp == nullptr) ? true
                           : (*fp)(static_cast<::Interface const *>(intfc_p),
                                   static_cast<::Interface const *>(rhs.intfc_p)) == 0;
}

// canonical_instance_for<IFC>() — SDK interface.h 原文: 表内单例地址
template <typename IFC> ::Interface const *canonical_instance_for() noexcept
{
    return &ifc_method_table[unsigned(IFC::dtype)].exemplar;
}

// ---------------------------------------------------------------------------
// make_interface<IFC> — from_odef / from_deser / from_exemplar 三入口
//   · from_odef (Plain)   : 返回表内单例 (PlainFloat = &table[4].exemplar
//                           = 0x5ec7fd0+0xc0; Float16 = +0x180)
//   · from_odef (Quant)   : 栈建 16B exemplar {dtinfo, def+0x48, def+0x4c,
//                           1/scale} → from_exemplar(graph, ex)
//   · from_exemplar       : key = ifc_hash(ex) ^ dt = ((scale_bits·2) ^
//                           (u32(offset)·0x10661)) ^ dt; Graph+0x5448 BST
//                           查找/插入 (命中返回既有, 未命中 crate 记录新建)
//   · from_deser (Quant)  : deserialize_shared_obj_func 去重; 新建走
//                           DCrate 游标 +0x10 / Crate::emplace_explicit
//                           (size_align_code 0x42); 字段 {0x10101, u32 offset,
//                           f32 scale, 1/scale} @0xd022f0/0xd02320
// ---------------------------------------------------------------------------
class Deserz; // hnnx::Deserz (Graph 为全局类, 见 ir/graph.hpp)

template <typename IFC> class make_interface;

template <typename T> class make_interface<PlainInterface<T>> {
  public:
    using Interface_t = PlainInterface<T>;
    static ::Interface const *from_odef(::Graph &, ::OutputDef const &);
    static ::Interface const *from_deser(Deserz &, ::Interface const **slot);
    static ::Interface const *from_exemplar(::Graph &, ::Interface const &);
};

template <typename T> class make_interface<ScaleOffsetInterface<T>> {
  public:
    using Interface_t = ScaleOffsetInterface<T>;
    static ::Interface const *from_odef(::Graph &, ::OutputDef const &);
    static ::Interface const *from_deser(Deserz &, ::Interface const **slot);
    // from_exemplar @0x1340280: Graph+0x5448 规范化 BST
    static ::Interface const *from_exemplar(::Graph &, ::ScaleOffsetInterface<T> const &);
};

} // namespace hnnx
