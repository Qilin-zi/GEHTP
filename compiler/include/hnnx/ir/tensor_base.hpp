#pragma once
// ============================================================================
// Tensor 抽象基类字节级重实现 (libHtpPrepare.so 2.48.40.260702, x86_64-linux-clang)
//
// 证据链 (全部字节级, 无推断):
//   * _ZTV6Tensor @0x60579a8 (0xE8 字节 → vptr 槽 +0x00..+0xD0 共 27 个), 由
//     .rela.dyn 全量重建: R_X86_64_RELATIVE(内部函数) + R_X86_64_64(外部符号,
//     纯虚槽 = __cxa_pure_virtual) 逐槽解析; _ZTS6Tensor = "6Tensor" @0x55b5ebc
//     → Tensor 为全局作用域类 (无 N4hnnx…E 前缀)。
//   * 18 个导出方法 + 12 个基类具体虚函数逐指令反汇编 (地址见各处注释)。
//   * SDK 头 qnn_ori_include/hnnx/sdk/core/tens/tensor_base.h 的声明序与 27 槽
//     1:1 对应 (证明编译时未定义 PREPARE_DISABLED —— get_shape_info 槽存在)。
//   * OpDef 侧调用方交叉验证 (M27): +0x18 删除析构 / +0x38 find_content_hash /
//     +0x68 set_dims / +0x78 raw_data / +0x90 total_storage_bytes / +0xc8 比较槽。
//
// sizeof(Tensor) == 8: 仅 vptr, 无任何数据成员 (三个构造函数均为空函数体;
//   0x13430b0 处基类 D0 = ud2 —— 抽象类不可删除, 实际删除走派生类 vtable +0x18;
//   本重实现按 C++ 语义生成常规 D0, 对一切合法程序行为等价, 仅代码字节不同)。
//
// ---- vtable 槽位表 (vptr 相对) -------------------------------------------
//  +0x00 interface()                    纯虚
//  +0x08 true_name()                    @0xcfff00 = (vptr[-1]=typeinfo)->name()
//  +0x10 ~Tensor D1                     @0xd00490 = ret (空析构体)
//  +0x18 ~Tensor D0                     @0x13430b0 = ud2 (抽象类陷阱)
//  +0x20 rank()                         纯虚
//  +0x28 dim(size_t)                    纯虚
//  +0x30 get_dims()                     纯虚
//  +0x38 find_content_hash()            @0x13414d0 = (u32)(uintptr)typeid(*this).name()
//  +0x40 element_addr(...)              纯虚 (protected)
//  +0x48 get_dtype_intfc()              纯虚
//  +0x50 get_shape_info()               @0xdcd410 = sret 清 24B = 空 std::string
//  +0x58 get_tensor_info()              @0xdac8e0 = 0
//  +0x60 get_tensor_format_code()       @0xdac8d0 = 0
//  +0x68 set_dims(size_t const[])       纯虚
//  +0x70 set_dims(Tensor const&)        纯虚
//  +0x78 raw_data()                     纯虚
//  +0x80 set_raw_data_despite_danger()  @0xdcd430 = ret (release 构建中 assert 被编译掉)
//  +0x88 total_storage_elements()       纯虚
//  +0x90 total_storage_bytes()          纯虚
//  +0x98 enum_memory_blocks()           纯虚
//  +0xa0 get_checksum()                 @0xdcd440 = 0
//  +0xa8 read_tile()                    @0xdcf220 = fprintf+fflush+throw
//  +0xb0 write_tile()                   @0xdcf2a0 = fprintf+fflush+throw
//  +0xb8 tile_support_bits()            @0xdcf320 = 0
//  +0xc0 allocate_func(...)             纯虚 (protected)
//  +0xc8 compare_sametype(...)          纯虚 (protected)
//  +0xd0 clone_util(...)                纯虚 (protected)
// ============================================================================
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>

#include "op_def.hpp" // ::OutputDef (0x50), ::DType, hnnx::MAX_DIMENSIONS;
#include "hnnx/serialize/deserz.hpp" // Deserz 真类 (顶层包含; 原处嵌于 namespace 内致 hnnx::std 污染)
                      // 全局前置 class Tensor (本文件随后给出完整定义)

class Op;    // 全局作用域 (mangling: _ZN6TensorC2EPK2Op)
class Graph; // 全局作用域 (allocator.h: class Graph;)
class Interface; // 全局作用域 (tens/interface.hpp 定义; .so: _ZN9Interface…)

namespace hnnx {
using Interface = ::Interface; // .so 中 Interface 在全局作用域
class SerOpsInterface;
class Allocator;
class Deserz;
class Crate;
class MemBlockEnumerator;
// fa::RuntimeAllocator 之全局前置声明由 hnnx/serialize/deserz.hpp 提供
} // namespace hnnx

using SIdx = long; // padding.h:22 — typedef long SIdx (x86-64 → 8 字节)

namespace hnnx {

// interface.h:57-63 原文 — qparms 三元组 {offset, scale, scale_recip} (12B)。
// 以命名空间级孪生形式先行定义 (::Interface 在 tens/interface.hpp 才完整,
// InterfaceRef 方法签名需要此类型; Interface::qparms 是它的别名)。
struct ifc_qparms {
    int32_t offset;    // +0x00
    float scale;       // +0x04
    float scale_recip; // +0x08
};
static_assert(sizeof(ifc_qparms) == 12);

// interface.h:197-262 — get_dtype_intfc/element_addr 涉及的接口引用对 (16B)。
// 方法族经 methods_p (intfc_methods 表条目) 分发; 定义在 tens/interface.hpp
// (依赖 intfc_methods 完整定义, 此处仅声明)。
struct InterfaceRef {
    void const *methods_p; // +0x00
    void const *intfc_p;   // +0x08

    // interface.h:218-220 原文 (get_qparms 分发; plain → &null_parms)
    ifc_qparms const *get_qparms() const;
    float get_scale() const;
    float get_scale_recip() const;
    int32_t get_offset() const;
    // interface.h:223-226: h = ifc_hash ? ifc_hash(intfc_p) : 0; return h ^ dt。
    // (即 from_exemplar 的 BST 键: 量化时 = ((scale位<<1) ^ (offset·0x10661)) ^ dt)
    uint32_t interface_hash() const noexcept;
    ::DType get_dtype() const noexcept;
    unsigned element_size() const noexcept;
    bool is_quantized() const noexcept;
    // interface.h:246-251: 表不同 → 按表地址序 (即按 dtype 序); 同表同对象 → 0;
    // 同表异对象 → ifc_compare (plain 恒 0)。
    int compare(InterfaceRef const &rhs) const noexcept;
    bool compare_eq(InterfaceRef const &rhs) const noexcept;
};

// allocator.h:21 — 枚举值经 SDK 头钉死 (Plain=0/TCM=1/UnCached=2/XXX_LAST=3)
enum class MemoryClass { Plain, TCM, UnCached, XXX_LAST_MEMORY_TYPE, Default = Plain };

// allocator.h:44-47 原文 — Tensor::allocate 的选项位
enum AllocOptions {
    AllocOpts_packed = 0x1 // allocation will be packed
};

// shape.h:54-64 — ShapeFlag/ShapeFlags; persistent_clone @0x1341211 仅测试 bit0
enum class ShapeFlag { none = 0, constant = 1, uncached = 2 };
struct ShapeFlags {
    uint16_t flags;                                                // +0x00
  private:
    uint16_t padding[sizeof(size_t) / sizeof(uint16_t) - 1]; // +0x02 显式填充 (8B 对齐)

  public:
    ShapeFlags() noexcept : flags(0) { padding[0] = padding[1] = padding[2] = 0; }
    explicit ShapeFlags(ShapeFlag f) noexcept : flags(uint16_t(f)) { padding[0] = padding[1] = padding[2] = 0; }
    bool is_const_memory() const { return (flags & uint16_t(ShapeFlag::constant)) != 0; }
    bool is_uncached_memory() const { return (flags & uint16_t(ShapeFlag::uncached)) != 0; }
};
static_assert(sizeof(ShapeFlags) == 8);

} // namespace hnnx

// interface.h:350 — get_dtype_intfc() 返回值; ABI: rax=[dtype|scale], edx=offset
// (12B 平凡结构 = 两个 eightbyte 均为 INTEGER 类 → rax:rdx 返回)。
// 证据: gen_output_def @0xdac84e call *0x48 后 mov %edx,%r12d / mov %rax,%rbp /
//       shr $0x20,%rbx。
// 作用域说明: 返回类型不入 mangling, 无作用域证据; 依使用方 (全局类 Tensor) 置于
// 全局作用域。
struct DTypeScaleOff {
    ::DType dtype; // +0x00
    float scale;       // +0x04
    int32_t offset;    // +0x08
    // interface.h:354-357 原文三构造 (12B 平凡结构, 不入 mangling)
    DTypeScaleOff(::DType dt, float sc, int zo) noexcept : dtype(dt), scale(sc), offset(zo) {}
    explicit DTypeScaleOff(::DType dt) noexcept : DTypeScaleOff(dt, 1.0f, 0) {}
    DTypeScaleOff() noexcept : DTypeScaleOff(::DType::UNKNOWN) {}
};
static_assert(sizeof(DTypeScaleOff) == 12);

// RuntimeAllocator 为全局 fa:: (mangling N2fa16RuntimeAllocator); 原嵌于 hnnx
// 内与 deserz.hpp 的全局前置声明分裂 —— M33 移至全局 (M35 完成全树统一)。
namespace fa {
// 两个外部符号 (经 @plt 调用)
struct RuntimeAllocator {
    void const *map_block_reference(unsigned off, unsigned size) const noexcept; // @0xd8d640 (jj)
    void deserialize_blocks(hnnx::Deserz &dctx, void const **table, size_t nblocks); // @0xd8d6e0
};
} // namespace fa

namespace hnnx {

// ---------------------------------------------------------------------------
// hnnx::blockid_set_t = minObj::hashset<void*, true, findhash<unsigned long>>
//   (get_memory_blocks 形参的 mangled 名)。sizeof = 0x38;
//   空表构造 = {+0x00=0, +0x08=0xffffffff(仅 4 字节 movl), +0x0C..0x37=0}
//   —— get_memory_blocks(int) @0x134111b-0x1341134 逐字节写入。
//   其余字段的容器语义 (桶计数/哈希函数等) 未在 Tensor 任务范围内解码, 不臆造;
//   emplace 仅声明 (定义由库侧提供)。
// ---------------------------------------------------------------------------
class blockid_set_t {
  public:
    blockid_set_t() noexcept // = 反汇编观察到的逐字节初始化: 整体清 0, +0x08 置 -1
    {
        memset(this, 0, sizeof(*this));
        mask = 0xffffffffu;
    }
    void emplace(void *id); // MemBlockEnumToSet 使用

  private:
    void **buckets;               // +0x00
    uint32_t mask;                // +0x08 (空表 = -1)
    uint32_t rest_[0x0C / 4 - 1]; // +0x0C
    uint64_t tail_[4];            // +0x10..0x37 (语义未解码, 整体清零)
    friend class ::Tensor;        // 仅为布局核对
};
static_assert(sizeof(blockid_set_t) == 0x38);

// ---------------------------------------------------------------------------
// hnnx::Deserz —— M33 起以真实类 (serialize/deserz.hpp, sizeof 0xd8) 取代
// 旧 _opq[0xa8] 镜像。原调用点钉死的偏移与成员逐一对应:
//   +0x00 vptr; vtable 槽 +0x10 = fill_buffer (旧名 fetch_more/refill,
//             调用点 0xdac907 / 0xdd932e / 0xddbe3e)
//   +0x18 allocator (fa::RuntimeAllocator*)   (0xdac9b3 / 0xdacc1b)
//   +0x20/+0x28 d_crate.nextp/limitp = bump 游标/上限 (0xdacaa1 / 0xdacad2)
//   +0x30 d_crate.cratep (hnnx::Crate*)        (0xdacaf1)
//   +0x50 full_deser (共享上下文*; 其 +0x11d0/+0x11d8 为 vector begin/end,
//             0xdacb6b-0xdacb78)
//   +0x68 bufp 读游标  +0x70 buf_limit 读末尾  (0xdaca0f)
//   +0x9c format_version 压缩格式标志          (0xdaca02)
//   +0xa0 seg_fixup_state (共享块表回退处理器子对象, 0xdacc95 → 0xdcd9d0)
// 兼容访问面 (is_compressed/read_cursor/scratch_ptr/bump_cursor/shared_ctx/
// shared_subobject/fetch_more/...) 已并入真实类声明。
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// hnnx::Crate::add_record_slot(size_t, size_t) @0xcf3ff0 — 返回 ≥0x14 字节结构
//   (sret); 调用点 0xdacaf1/0xdacbf1 仅使用 sret+0x08 (槽指针) 与 sret+0x10
//   (int 状态)。size_t 覆盖: 实参 nblocks*8; 对齐实参 8。
//   arena 内部算法属 Crate 范畴, 此处只声明。
// ---------------------------------------------------------------------------
class Crate {
  public:
    struct record_slot_result { // 24B → x86-64 MEMORY 类 → sret, 与调用点一致
        void *observed_unused_0; // +0x00 (两处调用点均未读)
        void **slot;             // +0x08 ← 0x10(%rsp)
        int32_t status;          // +0x10 ← 0x18(%rsp); >= 0 成功
    };
    record_slot_result add_record_slot(size_t bytes, size_t align); // @0xcf3ff0
    // M32: 成功分配后调用方自增 [+0x40] 记录计数 (VariadicOpBase::
    //   assign_input_pointers @0x139fcb1: addq $1,0x40(%rbp); 该 +0x40 即
    //   Graph 内嵌 crate(+0xd0) 的 Graph+0x110 计数器, 见 graph.hpp 互证注)
    void bump_record_count(); // +0x40 自增
};

// ---------------------------------------------------------------------------
// hnnx::Allocator 镜像 (allocator.h 声明序 → 槽位 D1@+0x00, D0@+0x08,
// allocate_n@+0x10, allocate_persistent_blocks@+0x18 —— 与全部克隆族调用点
// `call *0x18` 一致; 0x12f9430 调用点读 +0x08 = graph)。
// ---------------------------------------------------------------------------
enum class AllocatorMode { AllocVirtual, AllocPhysical, AllocTemp, AllocTempEnd, AllocComplete };

class Allocator {
  public:
    static constexpr unsigned MIN_ALIGN = 256;
    static constexpr unsigned MAX_ALIGN = 256;
    static constexpr unsigned TCM_ALLOC_ALIGN = 2048;
    static void *vacant() { return (void *)2; } // 特殊 "空槽" 值

    Allocator(AllocatorMode mode_in, ::Graph &graph_in) : graph(graph_in), mode(mode_in) {}
    virtual ~Allocator() = 0;

    ::Graph &graph; // +0x08 (vptr 之后第一个成员)

    virtual void allocate_n(void **arrp, size_t n, size_t block_size, size_t alignment, MemoryClass memclass,
                            unsigned options, DType dtype) = 0; // +0x10

    // persistent_options (allocator.h:116) —— 语义由 SDK 注释钉死:
    //   allnew=1: 指针表视为垃圾, 全部新分配; zoneB=2: A/B 区引用计数;
    //   incref=4: 覆盖 allnew, 全部有效持久块 +1; decref=8: 覆盖 incref/allnew,
    //             全部 -1, 归零释放, 不更新指针表; infinite=16: 新块计数置巨值。
    enum persistent_options {
        allnew = 1u,
        zoneB = 2u,
        incref = 4u,
        decref = 8u,
        infinite = 16u,
    };
    virtual void allocate_persistent_blocks(void **table, size_t nblocks, size_t block_size, size_t alignment,
                                            unsigned options) = 0; // +0x18
    virtual void set_mode(AllocatorMode new_mode) = 0;              // +0x20
    virtual void set_tcm_pool(void *base, size_t size) = 0;
    virtual void set_largest_memory_alloc_size(size_t size) = 0;

    AllocatorMode get_mode() const noexcept { return mode; }

  private:
    AllocatorMode mode; // SDK: get_mode() 读取; 相对偏移未在 Tensor 范围钉死
};
inline Allocator::~Allocator() = default;

// ---------------------------------------------------------------------------
// 内存块枚举 (block_enumeration.h —— SDK 原文级)
// ---------------------------------------------------------------------------
class MemBlockEnumerator {
  public:
    virtual ~MemBlockEnumerator() {}
    MemBlockEnumerator() {}
    MemBlockEnumerator(MemBlockEnumerator const &) = delete;
    MemBlockEnumerator(MemBlockEnumerator &&) = delete;
    MemBlockEnumerator &operator=(MemBlockEnumerator const &) = delete;
    MemBlockEnumerator &operator=(MemBlockEnumerator &&) = delete;

    // +0x10 —— 张量侧回调; memclass<0 表示类别未指定
    virtual void supply_blocks_func(::Tensor const *tensp, int memclass, void *const *ptr, size_t num) = 0;
    void supply_blocks(::Tensor const *tensp, void *const *ptr, size_t num)
    {
        supply_blocks_func(tensp, -1, ptr, num);
    }
    void supply_blocks(::Tensor const *tensp, MemoryClass mc, void *const *ptr, size_t num)
    {
        supply_blocks_func(tensp, int(mc), ptr, num);
    }
};

class MemBlockEnumToSet final : public MemBlockEnumerator {
    blockid_set_t &m_set; // +0x08
    int m_memclass_sel;   // +0x10
  public:
    explicit MemBlockEnumToSet(blockid_set_t &s, int mclass_sel = -1) : m_set(s), m_memclass_sel(mclass_sel) {}
    MemBlockEnumToSet(blockid_set_t &s, MemoryClass mc) : m_set(s), m_memclass_sel(int(mc)) {}
    void supply_blocks_func(::Tensor const *, int memclass, void *const *ptr, size_t num) override
    {
        if (m_memclass_sel >= 0 && memclass >= 0 && m_memclass_sel != memclass) return;
        for (size_t i = 0; i < num; i++) {
            if (ptr[i] != Allocator::vacant()) m_set.emplace(ptr[i]);
        }
    }
};
static_assert(sizeof(MemBlockEnumToSet) == 0x18); // _ZTVN4hnnx17MemBlockEnumToSetE @0x623f1a8

template <typename ENFUNC> class MemBlockEnumWrapper : public MemBlockEnumerator {
    ENFUNC m_enfunc;
    void supply_blocks_func(::Tensor const *tensp, int memclass, void *const *ptr, size_t num) override
    {
        m_enfunc(tensp, memclass, ptr, num);
    }
  public:
    explicit MemBlockEnumWrapper(ENFUNC &&ef) : m_enfunc(std::move(ef)) {}
    explicit MemBlockEnumWrapper(ENFUNC const &ef) : m_enfunc(ef) {}
};

template <typename REPLFUNC> class MemBlockReplBlockWrapper : public MemBlockEnumerator {
    REPLFUNC m_replfunc;
    void supply_blocks_func(::Tensor const *tensp, int /*memclass*/, void *const *ptr, size_t num) override
    {
        for (size_t i = 0; i < num; i++) {
            void *newblk = m_replfunc(tensp, ptr[i]);
            const_cast<void *&>(ptr[i]) = newblk;
        }
    }
  public:
    explicit MemBlockReplBlockWrapper(REPLFUNC &&ef) : m_replfunc(std::move(ef)) {}
    explicit MemBlockReplBlockWrapper(REPLFUNC const &ef) : m_replfunc(ef) {}
};

// ---------------------------------------------------------------------------
// 三个无名局部函数 (stripped, 符号名不可知; 签名由调用点钉死, 名字为占位)
// ---------------------------------------------------------------------------

// @0x12f9430 (紧随 Shape<1>::canonical_shape @0x12f93e0): persistent_clone
//   @0x1341238 调用, 参数 (allocator->graph /*rdi*/, *info.shapepp /*rsi*/,
//   rank() 低 32 位 /*edx*/, 1 /*cl*/), 返回值写回 *info.shapepp。
ShapeFlags *unnamed_12f9430_shape_persist(::Graph &g, ShapeFlags const *sf, unsigned rank, bool incref);

// @0xdcd9d0 (Interface::get_refobj+0x640): 共享槽为空时的回退物化器。
//   调用点 0xdacc95: (this = dctx+0xa0 子对象, idx /*rsi*/, cnt /*edx*/,
//   blockp_loc /*rcx*/) → bool; true 时已写入 *blockp_loc。
bool unnamed_dcd9d0_shared_blocktable(void *deserz_a0_subobject, unsigned idx, unsigned cnt, void ***blockp_loc);

// @0xcf5240 (graph_crate+0xad0): 抛出 hnnx::dcrate_seg_overflow_error
//   (typeinfo 0x5ebde40, vtable 0x5ebde58, 8 字节纯 vptr 异常对象)。
//   bump 游标越界路径 (0xdacccc) 调用, 不返回。
[[noreturn]] void unnamed_cf5240_dcrate_overflow_throw();

} // namespace hnnx

// ============================================================================
// class Tensor — 全局作用域 (符号 _ZN6Tensor… 无命名空间限定)
// ============================================================================
class Tensor {
  public:
    enum class clone_mode {
        duplicate,
        UNUSED_persistent,
    };
    // dims() 按名取维 (SDK 原文)
    enum dimensions { BATCH, HEIGHT, WIDTH, DEPTH, CHANNEL };

    // Op::allocate_generic @0x10bcfe0 调用受保护虚槽 +0xc0 (allocate_func)
    //   → SDK 侧 Op 拥有该访问权 (字节证据, 非布局变更)
    friend class ::Op;

    // +0x00
    virtual hnnx::InterfaceRef interface() const noexcept = 0;

    // +0x08 @0xcfff00: mov (%rdi),%rax; mov -0x8(%rax),%rax; mov 0x8(%rax),%rax; ret
    //   = vptr[-1] (typeinfo) → +0x08 (名字指针)
    virtual const char *true_name() const noexcept { return typeid(*this).name(); }

    explicit Tensor(const Op * /*producer_in*/) noexcept {}
    explicit Tensor(hnnx::Deserz & /*dctx*/) noexcept {}
    Tensor(const Tensor & /*old*/, hnnx::Allocator * /*allocator*/, clone_mode) noexcept {}

    // +0x10 D1 @0xd00490 = ret; +0x18 D0 @0x13430b0 = ud2 (见文件头注)
    virtual ~Tensor() {}

    Tensor(Tensor const &) = delete;
    Tensor(Tensor &&) = delete;
    Tensor &operator=(Tensor const &) = delete;
    Tensor &operator=(Tensor &&) = delete;

    // +0x20
    virtual size_t rank() const noexcept = 0;
    // +0x28
    virtual size_t dim(size_t index) const noexcept = 0;
    // +0x30
    virtual std::pair<size_t const *, size_t> get_dims() const noexcept = 0;

    // +0x38 @0x13414d0: mov (%rdi),%rax; mov -0x8(%rax),%rax; mov 0x8(%rax),%eax; ret
    //   —— 取 typeinfo 名字指针的低 32 位
    virtual uint32_t find_content_hash() const noexcept { return (uint32_t)(uintptr_t)typeid(*this).name(); }

  protected:
    // +0x40 (iref 非空时 *iref = interface() 的结果)
    virtual void *element_addr(size_t rank, SIdx const coords_in[], hnnx::InterfaceRef *iref = nullptr) const
        noexcept = 0;

    template <typename... T>
    static std::array<size_t, sizeof...(T)> dims_extractor(std::pair<size_t const *, size_t> const &dims_r,
                                                            T... indices)
    {
        auto const read_dim = [&dims_r](unsigned i) -> size_t { return (i < dims_r.second) ? dims_r.first[i] : 1; };
        return {read_dim(indices)...};
    }
    template <unsigned R>
    static std::array<size_t, R> dims_extractor_all(std::pair<size_t const *, size_t> const &dims_r)
    {
        std::array<size_t, R> result{};
        for (unsigned i = 0; i < R; i++) result[i] = (i < dims_r.second) ? dims_r.first[i] : 1;
        return result;
    }

  public:
    // ---------- 非虚内联访问器 (SDK 原文, 无 .so 代码) ----------
    void const *element_ptr(size_t rank, const SIdx coords[]) const
    {
        return (void const *)element_addr(rank, coords);
    }
    void *element_ptr(size_t rank, const SIdx coords[]) { return element_addr(rank, coords); }

    std::tuple<size_t, size_t, size_t, size_t> get_dims_4() const
    {
        size_t const *ptr = nullptr;
        size_t n = 0;
        std::tie(ptr, n) = get_dims();
        if (n != 4) throw std::runtime_error("rank not 4");
        return std::make_tuple(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
    std::tuple<size_t, size_t> get_dims_1_2() const
    {
        size_t const *ptr = nullptr;
        size_t n = 0;
        std::tie(ptr, n) = get_dims();
        if (n < 3) throw std::runtime_error("rank not >=3");
        return std::make_tuple(ptr[1], ptr[2]);
    }
    std::array<size_t, 4> dims() const { return dims_extractor_all<4>(get_dims()); }
    template <typename... T> std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        return dims_extractor(get_dims(), indices...);
    }

    // +0x48
    virtual DTypeScaleOff get_dtype_intfc() const noexcept = 0;
    ::DType get_dtype() const { return get_dtype_intfc().dtype; }
    float interface_scale() const { return get_dtype_intfc().scale; }
    int32_t interface_offset() const { return get_dtype_intfc().offset; }

    // ::OutputDef Tensor::gen_output_def() const — @0xdac840 精确复刻:
    //   vcall+0x48 → rax=[dtype|scale], edx=offset; vcall+0x20 → rank(写入 +0x00);
    //   dtype → +0x04; max_sizes 8×u64 清零 (+0x08..0x47); offset → +0x48;
    //   scale(rax>>32) → +0x4c; for (i<rank) vcall+0x28 dim(i) → max_sizes[i]。
    ::OutputDef gen_output_def() const
    {
        ::OutputDef od;
        const DTypeScaleOff di = get_dtype_intfc();
        od.rank = (uint32_t)rank();
        od.dtype = di.dtype;
        od.zero_offset = di.offset;
        od.stepsize = di.scale;
        for (uint32_t i = 0; i < od.rank; i++) od.max_sizes[i] = dim(i);
        return od;
    }

    // get_tensor_format_code 的位域 (tensor_base.h 原文注释表格)
    static constexpr unsigned tformat_dtype_shift = 0u, tformat_dtype_mask = 0xFu;
    static constexpr unsigned tformat_log2sz_shift = 6u, tformat_log2sz_mask = 3u;
    static constexpr unsigned tformat_rank_shift = 8u, tformat_rank_mask = 0xFu;
    static constexpr unsigned tformat_is_tcm = 1u << 16u;
    static constexpr unsigned tformat_is_quantized = 1u << 17u;
    static constexpr unsigned tformat_is_indirect = 1u << 18u;
    static constexpr unsigned tformat_is_chunked = 1u << 19u;
    static constexpr unsigned tformat_is_not_flat = 1u << 20u;
    static constexpr unsigned tformat_is_singular = 1u << 21u;
    static constexpr unsigned tformat_tmode_shift = 28u, tformat_tmode_mask = 0xFu;

    // +0x50 @0xdcd410: sret 清 24 字节 = 空 libc++ std::string
    virtual std::string get_shape_info() const { return {}; }
    // +0x58 @0xdac8e0: xor eax,eax; ret
    virtual uint32_t get_tensor_info() const noexcept { return 0; }
    // +0x60 @0xdac8d0: xor eax,eax; ret
    virtual uint32_t get_tensor_format_code() const noexcept { return 0; }

    // +0x68 / +0x70
    virtual bool set_dims(const size_t dims[]) = 0;
    virtual bool set_dims(const Tensor &prototype) = 0;

    void allocate(hnnx::Allocator &allocator, unsigned options = 0) { allocate_func(allocator, options); }

    // +0x78
    virtual void *raw_data() noexcept = 0;
    void const *raw_data_const() const noexcept { return const_cast<Tensor *>(this)->raw_data(); }
    // +0x80 @0xdcd430: 单条 ret —— SDK 头中的
    //   assert(!"Invalid to set raw pointer on this type of tensor")
    //   在 release 构建中被编译掉, 是无条件空操作。
    virtual void set_raw_data_despite_danger(void * /*buffer*/) noexcept {}
    // +0x88
    virtual size_t total_storage_elements() const = 0;
    // +0x90
    virtual size_t total_storage_bytes() const = 0;

    const char *truetype() const noexcept { return typeid(*this).name(); }

    // get_memory_blocks(set&, int) — @0x13410a0 精确复刻: 栈上构造
    //   MemBlockEnumToSet{vptr=_ZTV…+0x10 @0x623f1b8, set=&blocklist @+0x8,
    //   mc_sel @+0x10}, vcall +0x98。
    void get_memory_blocks(hnnx::blockid_set_t &blocklist, int mc_sel = -1) const
    {
        hnnx::MemBlockEnumToSet enumer(blocklist, mc_sel);
        enum_memory_blocks(enumer);
    }
    void get_memory_blocks(hnnx::blockid_set_t &blocklist, hnnx::MemoryClass mc) const
    {
        get_memory_blocks(blocklist, int(mc));
    }
    // get_memory_blocks(int) — @0x1341100: sret 就地按上述字节图案构造空表,
    //   走同一路径后按值返回 (rbx 保持 sret 指针)。
    hnnx::blockid_set_t get_memory_blocks(int mc_sel = -1) const
    {
        hnnx::blockid_set_t blocklist;
        get_memory_blocks(blocklist, mc_sel);
        return blocklist;
    }
    hnnx::blockid_set_t get_memory_blocks(hnnx::MemoryClass mc) const { return get_memory_blocks(int(mc)); }

    // +0x98
    virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const = 0;

    template <typename ENFUNC> void enum_memory_blocks_withfunc(ENFUNC &&ef) const
    {
        hnnx::MemBlockEnumWrapper<std::remove_reference_t<ENFUNC>> enumer(std::forward<ENFUNC>(ef));
        this->enum_memory_blocks(enumer);
    }
    template <typename REPLFUNC> void replace_memory_blocks_withfunc(REPLFUNC &&rf) const
    {
        hnnx::MemBlockReplBlockWrapper<std::remove_reference_t<REPLFUNC>> enumer(std::forward<REPLFUNC>(rf));
        this->enum_memory_blocks(enumer);
    }
    template <typename MAPTYPE> void replace_memory_blocks_withmap(MAPTYPE const &map) const
    {
        replace_memory_blocks_withfunc([&map](Tensor const *, void *oldid) {
            auto found_at = map.find(oldid);
            return (found_at != map.end()) ? found_at->second : oldid;
        });
    }

    void serialize(hnnx::SerOpsInterface & /*sctx*/) const; // SDK: sctx.tensor_serialize(this);
                                                            // 定义依赖 serops 头, 此处仅声明

    // ---- 克隆族 (clone_util vcall +0xd0 的通用包装) ----

    // persistent_clone @0x13411b0 精确复刻:
    //   ret 置空; newblocks = clone_util(allocator, &ret, &info, nullptr) (vcall+0xd0);
    //   newblocks 非空时:
    //     ① info.shapepp 非空且 (*shapepp)->flags bit0 (constant) 为 0:
    //          *shapepp = unnamed_12f9430(allocator->graph /*+0x8*/, *shapepp,
    //                                     rank() 低 32 位, /*incref=*/true);
    //     ② 无条件 allocate_persistent_blocks(newblocks, info.nblocks, info.blocksize,
    //          /*align=*/8, allnew | (zoneb ? zoneB : 0))     ← nblocks==0 也调用!
    //     ③ 仅当 info.nblocks != 0: for (i<nblocks)
    //          memcpy(newblocks[i], info.blkptrs[i], info.blocksize);
    //   异常路径: ret 中指针经 vtable+0x18 (D0) 删除后 _Unwind_Resume。
    std::unique_ptr<Tensor> persistent_clone(hnnx::Allocator *allocator, bool zoneb = false) const
    {
        std::unique_ptr<Tensor> ret;
        tensor_blockinfo info;
        void **const newblocks = clone_util(allocator, &ret, &info, nullptr);
        if (newblocks != nullptr) {
            if (info.shapepp != nullptr && ((*info.shapepp)->flags & uint16_t(hnnx::ShapeFlag::constant)) == 0) {
                *info.shapepp = hnnx::unnamed_12f9430_shape_persist(allocator->graph, *info.shapepp,
                                                                    (unsigned)rank(), true);
            }
            allocator->allocate_persistent_blocks(newblocks, info.nblocks, info.blocksize, 8,
                                                  unsigned(hnnx::Allocator::allnew) |
                                                          (zoneb ? unsigned(hnnx::Allocator::zoneB) : 0u));
            for (size_t i = 0; i < info.nblocks; i++)
                memcpy(newblocks[i], info.blkptrs[i], info.blocksize);
        }
        return ret;
    }
    std::unique_ptr<Tensor> persistent_clone_Op(hnnx::Allocator *allocator) const
    {
        return persistent_clone(allocator, true);
    }

    // shallow_clone_Op @0x13412f0: 克隆后仅 incref —— options = zoneB|incref = 6
    std::unique_ptr<Tensor> shallow_clone_Op(hnnx::Allocator *allocator) const
    {
        std::unique_ptr<Tensor> ret;
        tensor_blockinfo info;
        void **const newblocks = clone_util(allocator, &ret, &info, nullptr);
        if (newblocks != nullptr)
            allocator->allocate_persistent_blocks(newblocks, info.nblocks, info.blocksize, 8, 6u);
        return ret;
    }

    // persistent_decref @0x13413a0: 不克隆 (tensp=null → info.blkptrs 指原表);
    //   nblocks != 0 时 options = decref | (zoneb ? zoneB : 0)
    void persistent_decref(hnnx::Allocator *allocator, bool zoneb = false) const
    {
        tensor_blockinfo info;
        clone_util(allocator, nullptr, &info, nullptr);
        if (info.nblocks != 0)
            allocator->allocate_persistent_blocks(info.blkptrs, info.nblocks, info.blocksize, 8,
                                                  unsigned(hnnx::Allocator::decref) |
                                                          (zoneb ? unsigned(hnnx::Allocator::zoneB) : 0u));
    }
    void persistent_decref_Op(hnnx::Allocator *allocator) const { persistent_decref(allocator, true); }

    std::unique_ptr<Tensor> duplicate_clone(hnnx::Allocator *allocator) const
    {
        return reallocate_clone(allocator, true);
    }
    // reallocate_clone @0x1341420 精确复刻:
    //   dup  → clone_util(allocator, &ret, nullptr, od)           (不填 info)
    //   !dup → newblocks = clone_util(allocator, &ret, &info, od);
    //          newblocks 非空时 memset(newblocks, 0, info.nblocks * 8) (块表清零)
    std::unique_ptr<Tensor> reallocate_clone(hnnx::Allocator *allocator, bool dup = false,
                                             const ::OutputDef *od = nullptr) const
    {
        std::unique_ptr<Tensor> ret;
        if (dup) {
            clone_util(allocator, &ret, nullptr, od);
        } else {
            tensor_blockinfo info;
            void **const newblocks = clone_util(allocator, &ret, &info, od);
            if (newblocks != nullptr) memset(newblocks, 0, info.nblocks * 8);
        }
        return ret;
    }

    // compare (SDK 头内联原文): 同型 → compare_sametype; 否则按 typeid 序 -1/+1
    int compare(const Tensor *rhs) const
    {
        std::type_info const &lhs_type = typeid(*this);
        std::type_info const &rhs_type = typeid(*rhs);
        if (lhs_type == rhs_type) return compare_sametype(rhs);
        return lhs_type.before(rhs_type) ? -1 : 1;
    }

    // +0xa0 @0xdcd440: xorl %eax,%eax; ret (零扩展 → 64 位 0)
    virtual uint64_t get_checksum() const noexcept { return 0; }

    // +0xa8 @0xdcf220:
    //   rbx = *(0x623f228) (stderr 的 GOT 间接); FILE* = *rbx;
    //   fprintf(stderr, "unsupported: read tile on tensor type: %s\n" /*0x39b5f20*/,
    //           typeid(*this).name());  fflush(stderr);
    //   throw std::runtime_error("error");  /* 0x461aa69 */
    virtual void const *read_tile(unsigned /*flags*/, void * /*buffer*/, size_t /*b*/, int /*h*/, int /*w*/,
                                  int /*d*/) const
    {
        fprintf(stderr, "unsupported: read tile on tensor type: %s\n", typeid(*this).name());
        fflush(stderr);
        throw std::runtime_error("error");
    }
    // +0xb0 @0xdcf2a0: 同上, 格式串 "unsupported: write tile on tensor type: %s\n" /*0x39b5f4b*/
    virtual void write_tile(unsigned /*flags*/, void const * /*buffer*/, size_t /*b*/, int /*h*/, int /*w*/, int /*d*/)
    {
        fprintf(stderr, "unsupported: write tile on tensor type: %s\n", typeid(*this).name());
        fflush(stderr);
        throw std::runtime_error("error");
    }

    static constexpr unsigned tile_8bit = 1;
    static constexpr unsigned tile_16bit = 2;
    static constexpr unsigned tile_32bit = 4;
    static constexpr unsigned tile_any = (1 + 2 + 4);
    static constexpr unsigned tile_fast = 16;
    static constexpr unsigned tile_direct = 32;

    // +0xb8 @0xdcf320: xorl %eax,%eax; ret
    virtual unsigned tile_support_bits() const noexcept { return 0; }
    bool tile_support() const { return (tile_support_bits() & tile_any) != 0; }
    bool tile_support_fast() const { return (tile_support_bits() & tile_fast) != 0; }
    bool tile_support_direct() const { return (tile_support_bits() & tile_direct) != 0; }
    // write_tile_strategy: SDK 中定义于 tile_extract.h, 不在本次范围。

    // ---- 静态哈希工具 ----

    // build_hash @0x1343030 精确复刻 (h = previous; n 为带符号 int, n<=0 原样返回):
    //   for (i = 0; i < n; i++) h = h * 0x112531 ^ (u32)dims[i];
    //   (xor 32 位内存操作数 —— 取 size_t 低 32 位; .so 4 路展开纯为优化)
    static constexpr uint32_t build_hash(const size_t *dims, int n, uint32_t previous) noexcept
    {
        uint32_t h = previous;
        for (int i = 0; i < n; i++) h = h * 0x112531u ^ (uint32_t)dims[i];
        return h;
    }

    // content_hash_data @0x1342d40 精确复刻。
    //  marker = is_float ? 0x80000000 : 0 (0x1342d65: edx<<31); 命中 → 按 0 计 (±0 同哈希)。
    //  分档:
    //   nbytes == 0  → 0x982184da                     (0x1342fcc)
    //   nbytes ≤ 4   → w = 小端零拼装 (逐字节 shl/or, 0x1342d7e-0x1342db4);
    //                  (w==marker) ? 0x478d474a : w + 0x478d474a  (0x1342ff1 cmovne)
    //   4 < nbytes ≤ 64 → nw = (nbytes+3)>>2 个 32 位字;
    //       指针未 4 对齐或 nbytes%4≠0 → 先 buf[nw-1]=0 再 memcpy(buf,data,nbytes)
    //       (0x1342ef4-0x1342f0b) —— 即尾字零填充; h 从 0x982184da 起,
    //       h = h*0x6291e319 + norm(w)                (单链, M1=0x6291e319)
    //   64 < nbytes ≤ 0x407 → 覆盖 (nbytes>>3)*2 个字 = 前 nbytes&~7 字节, 尾 0..7 字节丢弃;
    //       h 同上 (码流为双 lane: h_even/h_odd, 乘数 M2=0x2fc75871,
    //       终结合 h_even*M1 + h_odd @0x1342ee7 —— 因 M2 == M1² mod 2³²,
    //       与单链逐项相等, 已代数+数值双验证)
    //   nbytes > 0x407 → 三个 256 字节窗口依次馈入同一累加链:
    //       [0, 256) ∪ [((n-0x80)>>1)&~0x7f, +256) ∪ [(n-0x100)&~0x3f, +256)
    static uint32_t content_hash_data(void const *data, size_t nbytes, bool is_float) noexcept
    {
        const uint32_t marker = is_float ? 0x80000000u : 0u;
        if (nbytes == 0) return 0x982184da;
        if (nbytes <= 4) {
            uint32_t w = 0;
            memcpy(&w, data, nbytes); // 小端零填充
            return (w == marker) ? 0x478d474au : w + 0x478d474au;
        }
        uint32_t h = 0x982184da;
        const auto feed_range = [&](uint32_t const *p, size_t nwords) noexcept {
            for (size_t i = 0; i < nwords; i++) {
                uint32_t w = p[i];
                if (w == marker) w = 0;
                h = h * 0x6291e319u + w;
            }
        };
        if (nbytes <= 64) {
            uint32_t buf[16];
            const size_t nw = (nbytes + 3) >> 2;
            if ((reinterpret_cast<uintptr_t>(data) & 3u) != 0 || (nbytes & 3u) != 0) {
                buf[nw - 1] = 0; // 尾字清零 (0x1342ef7)
                memcpy(buf, data, nbytes);
                feed_range(buf, nw);
            } else {
                feed_range(static_cast<uint32_t const *>(data), nw);
            }
            return h;
        }
        const unsigned char *base = static_cast<unsigned char const *>(data);
        if (nbytes <= 0x407) {
            feed_range(reinterpret_cast<uint32_t const *>(base), (nbytes >> 3) << 1);
        } else {
            feed_range(reinterpret_cast<uint32_t const *>(base), 64);                                       // 窗 1
            feed_range(reinterpret_cast<uint32_t const *>(base + (((nbytes - 0x80) >> 1) & ~size_t(0x7f))),
                       64);                                                                              // 窗 2
            feed_range(reinterpret_cast<uint32_t const *>(base + ((nbytes - 0x100) & ~size_t(0x3f))), 64); // 窗 3
        }
        return h;
    }

    // content_hash_data_indirect @0x1342cc0 精确复刻:
    //   nblocks == 0 → 原样返回 inhash (0x1342cca je)
    //   h = inhash * 0x5103031 ^ chd(blocks[0], …);  nblocks == 1 → 返回
    //   nblocks >= 5 → h = h * 0x5103031 ^ chd(blocks[nblocks>>1], …)
    //   末块: h = h * 0x5103031 ^ chd(blocks[nblocks-1], …) → 返回
    static uint32_t content_hash_data_indirect(uint32_t inhash, void **blocks, unsigned nblocks, size_t blockbytes,
                                               bool is_float) noexcept
    {
        uint32_t h = inhash;
        if (nblocks == 0) return h;
        h = h * 0x5103031u ^ content_hash_data(blocks[0], blockbytes, is_float);
        if (nblocks == 1) return h;
        if (nblocks >= 5) h = h * 0x5103031u ^ content_hash_data(blocks[nblocks >> 1], blockbytes, is_float);
        h = h * 0x5103031u ^ content_hash_data(blocks[nblocks - 1], blockbytes, is_float);
        return h;
    }

    // ---- tensor_blockinfo (SDK 结构原文; 0x38 字节, 布局经克隆族四函数
    //      的栈访问 (rsp+8 起: +0x00/+0x08/+0x10/+0x18/+0x20) 逐字段钉死) ----
    struct tensor_blockinfo {
        void **blkptrs;                            // +0x00 块表 (克隆时指向原张量表)
        hnnx::ShapeFlags const **shapepp;          // +0x08 形状字段地址 (克隆时指向克隆内字段;
                                                   //   persistent_clone @0x1341243 写 *shapepp
                                                   //   → 槽位可写, 内层指针不带 const)
        hnnx::Interface const *const *interfacepp; // +0x10 接口字段地址
        size_t nblocks;                            // +0x18
        size_t blocksize;                          // +0x20
        ::DType dtype;                         // +0x28
        hnnx::MemoryClass mclass;                  // +0x2C
        bool is_indirect;                          // +0x30
        bool is_chunked;                           // +0x31
        bool is_variable_block;                    // +0x32
        void setup(::DType dt = ::DType::UNKNOWN,
                   hnnx::MemoryClass mc = hnnx::MemoryClass::Default) noexcept
        {
            blkptrs = nullptr;
            shapepp = nullptr;
            interfacepp = nullptr;
            nblocks = 0;
            blocksize = 0;
            dtype = dt;
            mclass = mc;
            is_indirect = false;
            is_chunked = false;
            is_variable_block = false;
        }
        tensor_blockinfo() noexcept { setup(); }
    };
    static_assert(sizeof(tensor_blockinfo) == 0x38);

    void get_tensor_blockinfo(tensor_blockinfo *infop) const { clone_util(nullptr, nullptr, infop); }

    // ---- 反序列化 (静态) ----

    // deserialize_block_pointer @0xdac8f0 精确复刻。
    // 读字原语: cursor(+0x68) >= end(+0x70) 时先 vcall+0x10 refill; 字 = 小端 u32。
    // 压缩 (dctx+0x9c != 0):
    //   w1 == 0         → nullptr                     (0xdac927 je → xor eax)
    //   w1 <= 0x3fffffff → off = w1>>16, size = (w1&0xffff)<<6   (无保留字)
    //   w1 >  0x3fffffff → off = w1 & 0x0fffffff, size = w2      (再读一字)
    // 经典:
    //   (w1 & 2) == 0 → 先无条件丢弃一个保留字 (0xdac93c-0xdac94f, 在 ==1 判定之前!);
    //                   w1 == 1 → nullptr; 否则 off = w1>>22, size = (w1<<6)&0xfffff00
    //   (w1 & 2) != 0 → size = w2 (读一字); 再丢一个保留字 (0xdac9ab); off = w1>>3
    // 末尾: (*(fa::RuntimeAllocator**)(dctx+0x18))->map_block_reference(off, size) 尾跳
    static void *deserialize_block_pointer(hnnx::Deserz &dctx)
    {
        const auto read_u32 = [&dctx]() noexcept -> uint32_t {
            if (dctx.read_cursor() >= dctx.read_end()) dctx.fetch_more();
            uint32_t w;
            memcpy(&w, dctx.read_cursor(), 4);
            dctx.set_read_cursor(dctx.read_cursor() + 4);
            return w;
        };
        const auto skip_u32 = [&dctx]() noexcept {
            if (dctx.read_cursor() >= dctx.read_end()) dctx.fetch_more();
            dctx.set_read_cursor(dctx.read_cursor() + 4);
        };
        uint32_t off, size;
        const uint32_t w1 = read_u32();
        if (dctx.is_compressed()) {
            if (w1 == 0) return nullptr;
            if (w1 <= 0x3fffffffu) {
                size = (w1 & 0xffffu) << 6;
                off = w1 >> 16;
            } else {
                off = w1 & 0x0fffffffu;
                size = read_u32();
            }
        } else if ((w1 & 2u) == 0) {
            skip_u32(); // 保留字 (先于 ==1 判定被消费)
            if (w1 == 1) return nullptr;
            off = w1 >> 22;
            size = (w1 << 6) & 0xfffff00u;
        } else {
            size = read_u32();
            skip_u32(); // 第三个保留字
            off = w1 >> 3;
        }
        return const_cast<void *>(dctx.runtime_allocator()->map_block_reference(off, size));
    }

    template <typename T> // T = storage_type (SDK 原文模板)
    static void deserialize_blocktable(hnnx::Deserz &dctx, T **&blockptr, unsigned const nblocks)
    {
        deserialize_blocktable_generic(dctx, (void ***)&blockptr, nblocks);
    }

    // deserialize_blocktable_generic @0xdac9e0 精确复刻。
    //  共同原语:
    //   bump 分配 (cursor=*(dctx+0x20)): 非 0 → 对齐 8; cursor+nblocks*8 超过
    //     *(dctx+0x28) → hnnx::unnamed_cf5240_dcrate_overflow_throw() (抛
    //     dcrate_seg_overflow_error, 0xdacccc); 否则 *(dctx+0x20) += nblocks*8,
    //     表 = 对齐后 cursor。cursor == 0 → 走 crate:
    //     Crate::add_record_slot(bytes=nblocks*8, align=8) (sret: slot@+8, status@+0x10);
    //     status >= 0 时 *(crate+0x40) += 1 (0xdacbf8 —— 计数器在 Crate 对象上);
    //     表 = result.slot。nblocks == 0 → 表 = nullptr。
    //  压缩路径 (dctx+0x9c != 0):
    //   w = read_u32(); nblocks = (w>>14) & 0x3fff;
    //   if (w & 0x08002000) { if ((~w & 0x3fff)==0) skip_u32();   // nblocks 满域时其值在下一字
    //                        if (nblocks==0x3fff) nblocks = read_u32(); }
    //   if (w & 0x40000000) {  // 复用共享槽 (0xdacb25)
    //       ref = read_u32();
    //       (int)ref >= 0 → idx = ref>>16, cnt = ref&0xffff;
    //       否则           idx = ref&0x7fffffff, cnt = read_u32();
    //       vec = *(dctx+0x50) 的 +0x11d0(begin)/+0x11d8(end);
    //       (end-begin)>>3 <= idx → throw std::runtime_error("link index")   /*0x39b3194*/
    //       slotp = &begin[idx];
    //       nblocks != 0 → 表 = bump/crate 分配; deserialize_blocks; *slotp = 表;
    //                      *blockp_loc = 表 + cnt;
    //       nblocks == 0 且 *slotp 非空 → *blockp_loc = *slotp + cnt;
    //       nblocks == 0 且 *slotp 为空 → unnamed_dcd9d0(dctx+0xa0, idx, cnt,
    //                      blockp_loc) @0xdcd9d0; false → throw "link sequence" /*0x39b319f*/;
    //                      true → 返回 (*blockp_loc 已由被调方写入)。
    //   } else {  // 全新表 (0xdaca94)
    //       表 = (nblocks != 0) ? bump/crate 分配 : nullptr;
    //       deserialize_blocks(*(dctx+0x18), dctx, 表, nblocks);   // nblocks==0 也调用
    //       *blockp_loc = 表;
    //   }
    //  经典路径 (dctx+0x9c == 0, 用参数 nblocks):
    //   表 = (nblocks != 0) ? bump/crate 分配 : nullptr;
    //   deserialize_blocks(*(dctx+0x18), dctx, 表, nblocks);       // 无条件
    //   skip_u32();                                                // 尾部保留字 (0xdacc57)
    //   *blockp_loc = 表;
    static void deserialize_blocktable_generic(hnnx::Deserz &dctx, void ***blockp_loc, unsigned nblocks_in)
    {
        const auto read_u32 = [&dctx]() noexcept -> uint32_t {
            if (dctx.read_cursor() >= dctx.read_end()) dctx.fetch_more();
            uint32_t w;
            memcpy(&w, dctx.read_cursor(), 4);
            dctx.set_read_cursor(dctx.read_cursor() + 4);
            return w;
        };
        const auto skip_u32 = [&dctx]() noexcept {
            if (dctx.read_cursor() >= dctx.read_end()) dctx.fetch_more();
            dctx.set_read_cursor(dctx.read_cursor() + 4);
        };
        // bump/crate 表分配 (0xdaca94-0xdacb1b 与 0xdacb98-0xdacc02 同一逻辑)
        const auto alloc_table = [&dctx](unsigned nblocks) noexcept -> void ** {
            unsigned char *cursor = dctx.bump_cursor();
            if (cursor != nullptr) {
                cursor = reinterpret_cast<unsigned char *>((reinterpret_cast<uintptr_t>(cursor) + 7) & ~uintptr_t(7));
                unsigned char *const next = cursor + nblocks * 8;
                if (next > dctx.bump_limit()) hnnx::unnamed_cf5240_dcrate_overflow_throw(); // 不返回
                dctx.set_bump_cursor(next);
                return reinterpret_cast<void **>(cursor);
            }
            if (nblocks == 0) return nullptr;
            hnnx::Crate *const crate = dctx.crate();
            auto const r = crate->add_record_slot(nblocks * 8, 8);
            if (r.status >= 0) {
                uint64_t cnt;
                memcpy(&cnt, reinterpret_cast<unsigned char *>(crate) + 0x40, 8);
                ++cnt;
                memcpy(reinterpret_cast<unsigned char *>(crate) + 0x40, &cnt, 8);
            }
            return r.slot;
        };
        // (end-begin)>>3 为有符号除 8; 两指针均来自同一 vector, 语义即元素数
        const auto shared_vec = [&dctx]() noexcept -> std::pair<void **, void **> {
            unsigned char *const ctx = static_cast<unsigned char *>(dctx.shared_context());
            void *b, *e;
            memcpy(&b, ctx + 0x11d0, 8);
            memcpy(&e, ctx + 0x11d8, 8);
            return {static_cast<void **>(b), static_cast<void **>(e)};
        };

        if (dctx.is_compressed()) {
            const uint32_t w = read_u32();
            unsigned nblocks = (w >> 14) & 0x3fffu;
            if (w & 0x08002000u) {
                if ((~w & 0x3fffu) == 0) skip_u32();
                if (nblocks == 0x3fff) nblocks = read_u32();
            }
            if (w & 0x40000000u) {
                const uint32_t ref = read_u32();
                uint32_t idx, cnt;
                if ((int32_t)ref >= 0) {
                    cnt = ref & 0xffffu;
                    idx = ref >> 16;
                } else {
                    idx = ref & 0x7fffffffu;
                    cnt = read_u32();
                }
                auto const [vb, ve] = shared_vec();
                if (size_t(ve - vb) <= idx) throw std::runtime_error("link index"); // 0x39b3194
                void **const slotp = vb + idx;
                if (nblocks != 0) {
                    void **const table = alloc_table(nblocks);
                    dctx.runtime_allocator()->deserialize_blocks(dctx, (void const **)table, nblocks);
                    *slotp = table;
                    *blockp_loc = table + cnt;
                } else if (void *const base = *slotp) {
                    *blockp_loc = static_cast<void **>(base) + cnt;
                } else if (!hnnx::unnamed_dcd9d0_shared_blocktable(dctx.shared_subobject(), idx, cnt,
                                                                   blockp_loc)) {
                    throw std::runtime_error("link sequence"); // 0x39b319f
                }
                return;
            }
            void **const table = alloc_table(nblocks);
            dctx.runtime_allocator()->deserialize_blocks(dctx, (void const **)table, nblocks);
            *blockp_loc = table;
            return;
        }
        // 经典路径
        void **const table = alloc_table(nblocks_in);
        dctx.runtime_allocator()->deserialize_blocks(dctx, (void const **)table, nblocks_in);
        skip_u32();
        *blockp_loc = table;
    }

  protected:
    // +0xc0
    virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) = 0;
    // +0xc8
    virtual int compare_sametype(const Tensor *rhs) const = 0;
    // +0xd0 — 克隆多面手 (SDK 注释原文语义):
    //   tensp 非空 → 生成克隆存入 *tensp; infop 非空 → 填 *infop;
    //   两者皆非空 → infop->blkptrs 指原张量的块表, 返回值 = 新张量块表指针
    //   (无块则为 null)。tensp 为 null 时 allocator 可为 null。
    virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                              tensor_blockinfo *infop, const ::OutputDef *od = nullptr) const = 0;
};

static_assert(sizeof(Tensor) == 8, "Tensor 仅 vptr, 无数据字段");

// 槽位速查 (vptr 相对偏移; 供派生类/调用方核对, 与文件头注释一致)
namespace TensorVtableSlots {
[[maybe_unused]] constexpr unsigned interface = 0x00, true_name = 0x08, D1 = 0x10, D0 = 0x18, rank = 0x20,
                  dim = 0x28,
                  get_dims = 0x30, find_content_hash = 0x38, element_addr = 0x40, get_dtype_intfc = 0x48,
                  get_shape_info = 0x50, get_tensor_info = 0x58, get_tensor_format_code = 0x60,
                  set_dims_arr = 0x68, set_dims_tensor = 0x70, raw_data = 0x78,
                  set_raw_data_despite_danger = 0x80, total_storage_elements = 0x88,
                  total_storage_bytes = 0x90, enum_memory_blocks = 0x98, get_checksum = 0xa0,
                  read_tile = 0xa8, write_tile = 0xb0, tile_support_bits = 0xb8, allocate_func = 0xc0,
                  compare_sametype = 0xc8, clone_util = 0xd0;
} // namespace TensorVtableSlots
