#pragma once
// ============================================================================
// class Op 字节级重实现 (含 ChkptStoreType/OpStoreType/flags/hnnx::type_info/
// OpInfo/OpExtraInfo(OpExtraAttrib)/uptr_Tensor 配套类型)
// libHtpPrepare.so 2.48.40.260702, x86_64-linux-clang, stripped
//
// 证据链 (全部字节级, 无推断; 未知类以精确签名+地址标注):
//   * 类身份: 全局作用域 (mangling _ZN2Op…), 抽象基类。
//     _ZTI2Op 经 GOT 0x623fde0 → __dynamic_cast(this, Op, MetaOpBase, 0) @0x10bce4e。
//   * _ZTV2Op @0x60532b0 (GOT 0x623fdb0), 对象区 0xC0 = 22 槽 =
//     hnnx::Executable 7 槽继承 + Op 自有 15 槽; 后接 0x0/_ZTIN4hnnx17MemBlockEnumToSetE
//     (下一 vtable 的 offset-to-top/typeinfo) → 边界精确。
//   * sizeof(Op) == 8: 构造 _ZN2OpC2ER5Graphy @0xd4e010 (91B) 仅写 vptr
//     (GOT 0x623fdb0+0x10) 后调 Graph::set_extra_info(this, {my_id,-1}),
//     无任何成员写入; 派生 ConstWrapperOp 仅增 uptr_Tensor(16B)@+8。
//   * 槽序 (vptr 相对; 声明序与 .so 逐槽吻合, 见类体内逐槽注释):
//     +0x00 execute(纯虚) +0x08/0x10/0x18 compile×3 +0x20 check_constraint_for_recompile
//     +0x28 D1(0xd0eb30 ret, 与基类同址=平凡析构折叠) +0x30 D0(0x10bd040 ud2, 抽象类不可达)
//     +0x38 clear +0x40 prepare(纯虚) +0x48 allocate(纯虚) +0x50 set_input
//     +0x58 is_valid(纯虚) +0x60 num_inputs_outputs(纯虚) +0x68 true_name
//     +0x70 get_flag_word +0x78 get_docs +0x80 true_func +0x88 get_type_info
//     +0x90 enumerate_blocks +0x98 serialize(纯虚) +0xa0 get_input_output(纯虚)
//     +0xa8 swap_output —— 6 个自有纯虚 (40/48/58/60/98/a0) 与 SDK 声明完全一致。
//   * OpId = unsigned long long (interface_defs.h:17; mangling y)。
// ============================================================================
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "executable.hpp" // hnnx::Executable (基类, 7 槽)
#include "graph.hpp"      // Graph (set_extra_info/get_extra_info/is_hmx_threaded 声明)
#include "op_def.hpp"     // hnnx::uptr_Op, hnnx::OpIoPtrs 前置, OpDef, qnndsp_log
#include "tensor_base.hpp" // Tensor, MemBlockEnumerator, MemBlockEnumToSet, blockid_set_t, Allocator
#include "status.hpp"      // GraphStatus (经 status.hpp 与 GCP 旧族 types.hpp 解耦)

using hnnx::GraphStatus; // Op 的 prepare/allocate/execute 返回类型 (SDK op.h 同名引用)

class Op;
class Tensor;

typedef unsigned long long OpId; // interface_defs.h:17 (mangling y)

namespace hnnx {
class Serializer; // 前置: serialize_internal 参数 (布局未反演, 见方法注释)
}

// ---------------------------------------------------------------------------
// ChkptStoreType — 全局枚举 (op.h:58)
// ---------------------------------------------------------------------------
enum ChkptStoreType {
    ChkptNormal = 0,  // N, M
    ChkptNone = 1,    // (-1 or 0), -1
    ChkptNoGate = 2,  // (-1 or 0), N
    ChkptNoDone = 3,  // N, -1
    ChkptFlagShift = 2,
    ChkptOpFlagMask = 0x3,
    ChkptFlagMask = ((1 << ChkptFlagShift) - 1),
};

// ---------------------------------------------------------------------------
// OpStoreType — 全局枚举 (serialize_defs.h:16)
// ---------------------------------------------------------------------------
enum OpStoreType {
    OpStoreFg,   // 主线程前台
    OpStoreVec,  // HVX (vector)
    OpStoreMtx,  // HMX (matrix)
    OpStoreElt,  // HLX (element-wise long vector)
};

// ---------------------------------------------------------------------------
// flags (flags.h) —— Flags_word + Flags 位枚举 + hnnx::flags_for 机制
// ---------------------------------------------------------------------------
typedef unsigned long Flags_word;
enum class Flags : unsigned {
    IS_CONST = 0,                    // output doesn't change
    INHIBIT_CONST_PROP,              // do not const-propagate this Op
    RESOURCE_HVX,                    // bit 2 (0x4)
    RESOURCE_HMX,                    // bit 3 (0x8)
    RESOURCE_HLX,                    // bit 4 (0x10)
    IS_DMA,
    FOR_HVX,
    FOR_HMX,
    FOR_HLX,
    FOR_DMA,
    MOVE_EARLY,
    MOVE_LATE,
    NULL_EXEC,                       // bit 12 (0x1000) — opclone_auto 判定位
    IS_SPILL,
    IS_FILL,
    INPLACE_NOP,
    IS_COPY,
    IS_SYNC,
    IS_SEND,
    IS_RECV,
    IS_PADZAP,
    IS_PRELOAD,
    CAN_BE_SRC_DESTRUCTIVE,
    IS_WEIGHT_FOR_BIT_REARRANGE,
    IS_PREDICATE_MARKER_TRUE,
    IS_PREDICATE_MARKER_FALSE,
    RESOURCE_EXCLUSIVE,
    IS_GATHER_DMA,
    XXX_LAST_FLAG
};
static_assert(static_cast<int>(Flags::XXX_LAST_FLAG) <= 64, "Too many flags");

namespace hnnx {

template <Flags... idxs>
constexpr Flags_word flagval_generate = ((Flags_word(1) << static_cast<unsigned>(idxs)) | ... | 0);

template <typename T, int S> struct flags_for_t {
    constexpr static Flags_word value = 0;
};
template <typename T, int S = 1> constexpr Flags_word flags_for = flags_for_t<T, S>::value;

inline constexpr bool test_flag_for(Flags_word w, Flags which)
{
    return (w >> static_cast<unsigned>(which) & 1u) != 0;
}
inline constexpr bool test_flag_and(Flags_word w, Flags which_a, Flags which_b)
{
    if ((w >> static_cast<unsigned>(which_a) & 1u) == 0) return false;
    return (w >> static_cast<unsigned>(which_b) & 1u) != 0;
}

#define DOCS_UNSET ""
template <typename T> [[maybe_unused]] static constexpr const char *docs_for() { return DOCS_UNSET; }

// ---------------------------------------------------------------------------
// hnnx::type_info (qhpi_type_info.h:37) —— {bool is_std; union{两个指针}}
// get_type_info 默认体 @0xd0e1e0 逐字节钉死布局: [out]=1, [out+8]=vptr[-1]
// ---------------------------------------------------------------------------
struct type_info {
    bool is_std;
    union {
        const std::type_info *std_type_info;
        const void *foreign_type_info; // QHPI_Kernel* (本 .so 未启用 QHPI)
    };
    type_info(const hnnx::type_info &std_type) = default;
    type_info(hnnx::type_info &&std_type) = default;
    type_info(const std::type_info &std_type) : is_std(true), std_type_info(&std_type) {}
    type_info(const void *kernel) : is_std(false), foreign_type_info(kernel) {}
    ~type_info() {}
    type_info &operator=(type_info &&other) noexcept = default;
    void *value() const { return is_std ? (void *)std_type_info : (void *)foreign_type_info; }
    const char *name() const { return std_type_info->name(); }
    bool operator==(const type_info ty) const
    {
        if (is_std != ty.is_std) return false;
        if (is_std) return *std_type_info == *ty.std_type_info;
        return foreign_type_info == ty.foreign_type_info;
    }
    bool operator!=(const type_info ty) const { return !(*this == ty); }
    type_info &operator=(const type_info &ty)
    {
        is_std = ty.is_std;
        if (is_std)
            std_type_info = ty.std_type_info;
        else
            foreign_type_info = ty.foreign_type_info;
        return *this;
    }
};
#if defined(_MSC_VER)
static_assert(sizeof(hnnx::type_info) == 0x10);
#endif

// ---------------------------------------------------------------------------
// hnnx::cost_function_t (cost_funcs.h:22) —— {funcp@0, val@8}
// Op::cost @0x10bcf4b: rcx=[entry+0]; rdi=entry; rsi=&graph; rdx=this; jmp *rcx
//   → funcp 首参即 cost_function_t 自身
// ---------------------------------------------------------------------------
class cost_function_t {
  public:
    using inner_func_t = float (*)(cost_function_t const &, const ::Graph &, ::Op const *);
    inner_func_t funcp;
    float val;

    cost_function_t(cost_function_t const &) = default;
    cost_function_t &operator=(cost_function_t const &) = default;
    constexpr explicit cost_function_t(float val_in) : funcp(simple_cost_function), val(val_in) {}
    constexpr cost_function_t(inner_func_t f, float val_in) : funcp(f), val(val_in) {}
    constexpr cost_function_t() noexcept : funcp(simple_cost_function), val(0.0f) {}
    static float simple_cost_function(cost_function_t const &self, const ::Graph &, ::Op const *)
    {
        return self.val;
    }
    float operator()(const ::Graph &g, ::Op const *op) const { return funcp(*this, g, op); }
};

// ---------------------------------------------------------------------------
// hnnx::OpInfo (op_info.h:31) —— op_info_map 的映射值
// true_name 读 [0x28] (type_tag.data()); cost 调 [0x0]; clone 用 [0x20] (op_factory)
// (hnnx::OpIoPtrs 已由 op_def.hpp 前置声明为 struct)
// ---------------------------------------------------------------------------
using OpFactory = uptr_Op (*)(OpIoPtrs const &, const OpId);

class OpInfo {
  private: // 显式标签: 布局断言 (#define private public 探针) 依赖之
    cost_function_t cost;             // +0x00 (funcp@0, val@8)
    Flags_word flags;                 // +0x10
    bool is_external_flag;            // +0x18
    OpFactory op_factory;             // +0x20
    const std::string_view type_tag;  // +0x28 (data@0x28, size@0x30)

  public:
    OpInfo(cost_function_t cost_in, Flags_word flags_in, OpFactory op_factory_in, bool is_external_in,
           const std::string_view type_tag_in)
        : cost(cost_in), flags(flags_in), is_external_flag(is_external_in), op_factory(op_factory_in),
          type_tag(type_tag_in)
    {
    }
    ~OpInfo() = default;

    Flags_word get_flags() const { return flags; }
    cost_function_t const &get_cost() const { return cost; }
    bool is_external() const { return is_external_flag; }
    const char *get_type_tag() const { return type_tag.data(); }
    OpFactory get_op_factory() const { return op_factory; }
};

// 映射本体属于未反演区段; 查找接口经 true_name/cost 调用点钉死 (@plt 0x6ed9a0)
OpInfo const *op_info_map_lookup(std::type_index tind);

// ---------------------------------------------------------------------------
// hnnx::OpExtraInfo —— M33 起拆至 op_extra_info.hpp (deserz.hpp 依赖完整类型)
// ---------------------------------------------------------------------------
#include "op_extra_info.hpp"

struct OpExtraAttrib : public OpExtraInfo {
    // 以下字段仅在 prepare 期有效。位域分配 (Itanium ABI, 单元 32 位):
    //   单元@0x18: for_hlx(bit0) | num_scratch_outputs(bit1-5) | self_slicing_op_nslices(bit6-9)
    //   单元@0x1c: predicate_offset_sense(24)   单元@0x20: switched_op_attr(24)
    //   单元@0x24: switched_op_info(24)         @0x28: switched_op_num_alias
    //   @0x30: switched_op_alias_coding (vector)
    bool for_hlx : 1;                         // HVX op to be moved to HLX ← movzwl+test $1 的目标
    unsigned int num_scratch_outputs : 5;
    unsigned int self_slicing_op_nslices : 4; // 0 means just 1 slice; otherwise >= 2
    unsigned int predicate_offset_sense : 24; // predicate_offset.23::sense.1
    unsigned int switched_op_attr : 24;       // extra:1::switch_info:u3::num_actual:u16
    unsigned int switched_op_info : 24;       // switch_info_payload.24
    unsigned int switched_op_num_alias = 0;   // number of Alias: table_len - num_actual
    std::vector<uint32_t> switched_op_alias_coding;

    OpExtraAttrib() : OpExtraInfo() { clear_fields(); }
    OpExtraAttrib(OpExtraInfo const &baseval) : OpExtraInfo(baseval) { clear_fields(); }
    explicit OpExtraAttrib(OpId id_in) : OpExtraInfo(id_in) { clear_fields(); }
    OpExtraAttrib(OpId id_in, int cg, int dc) : OpExtraInfo(id_in, cg, dc) { clear_fields(); }

    void clear_fields()
    {
        for_hlx = false;
        num_scratch_outputs = 0;
        self_slicing_op_nslices = 0;
        predicate_offset_sense = 0;
        switched_op_attr = 0;
        switched_op_info = 0;
        switched_op_num_alias = 0;
        switched_op_alias_coding.clear();
    }
};

// ---------------------------------------------------------------------------
// hnnx::uptr_Tensor —— uptr_DWDI<Tensor> 的 16B 布局 {Tensor*@0; 删除器字@8}
// steal_output @0x10bcb57 初始化 {0,0}; 失败清理 @0x10bcb78: 读 ptr, 清零,
//   ptr 非空且字@8==0 时调 Tensor 虚槽 +0x18 (D0) —— 即 delete。
// ---------------------------------------------------------------------------
struct uptr_Tensor {
    ::Tensor *ptr = nullptr;    // +0x00 (:: 限定 —— types.hpp 另有占位 hnnx::Tensor)
    unsigned char dw = 0;       // +0x08: 0=归属(重置时经 D0 槽析构), 非 0=借用
    uptr_Tensor() = default;
    explicit operator bool() const { return ptr != nullptr; }
    void reset()
    {
        ::Tensor *const p = ptr;
        ptr = nullptr;
        if (p != nullptr && dw == 0) delete p; // delete → Tensor 虚槽 +0x18 (D0)
    }
    ::Tensor *release()
    {
        ::Tensor *const p = ptr;
        ptr = nullptr;
        return p;
    }
};
#if defined(_MSC_VER)
static_assert(sizeof(uptr_Tensor) == 0x10);
#endif

} // namespace hnnx

// .so 导出的包装类名串 —— is_const/true_name 的指针比较目标:
//   is_const @0xd4e1d0:  n == GOT[_ZTIN4hnnx14ConstWrapperOpE(0x623f9b0)]->name()
//                          || GOT[_ZTIN4hnnx14ShapeWrapperOpE(0x623f508)]->name()
//   true_name @0x10bc7b0: n == GOT(0x623f150)=_ZTSN4hnnx14ShapeWrapperOpE → "Shape"
//   true_name @0x10bc7b9: n == GOT(0x623f418)=_ZTSN4hnnx14ConstWrapperOpE → "Const"
// (typeinfo+8 的 name 指针即 _ZTS 串地址, 两种写法运行时等值)
extern char const _ZTSN4hnnx14ShapeWrapperOpE[];
extern char const _ZTSN4hnnx14ConstWrapperOpE[];

// ---------------------------------------------------------------------------
// StandardCosts (op.h:302) —— cost 查表 miss 的回退值
// ---------------------------------------------------------------------------
struct StandardCosts {
    static constexpr float GLACIAL = 0x1.0p48;  // 2**48 — @0x39bc73c = 281474976710656.0f
    static constexpr float SNAIL = 0x1.0p32;    // 2**32
    static constexpr float FAST = 0x1.0p8;      // 256
    static constexpr float FREE = 0x1.0p-64;
    static constexpr float DISABLE = 0x1.0p50;  // 2**50
};

// ============================================================================
// class Op —— 全局作用域, 抽象, : public hnnx::Executable (单继承, 22 槽)
// ============================================================================
class Op : public hnnx::Executable {
  public:
    Op() {};
    // @0xd4e010 (91B): vptr = _ZTV2Op+0x10 (编译器生成) 后仅调
    //   Graph::set_extra_info(this, OpExtraInfo{my_id, chkpts=(-1,-1)}); 无成员写入。
    //   (栈保护字节为编译产物)
    Op(Graph &graph_in, unsigned long long int my_id_in)
    {
        graph_in.set_extra_info(this, hnnx::OpExtraInfo(my_id_in));
    }
    explicit Op(hnnx::Deserz &dctx) { (void)dctx; }
    Op(Op const &) = delete;
    Op &operator=(Op const &) = delete;

    // M32: OpIoPtrs (2) 型 ctor @0xf85070 与 get_output_for_cloned_op @0xf853f0
    //   直调受保护虚槽 +0xa0 (get_input_output)。.so 为机器码无访问控制,
    //   镜像以友元等价 (与 Graph 对 OpIoPtrs 的 +0x1d8 直读友元同源证据)。
    friend class hnnx::OpIoPtrs;

    // +0x28 D1 (0xd0eb30, 与 Executable::D1 同址 —— 平凡空析构代码折叠);
    // +0x30 D0 (0x10bd040 = ud2, 抽象类不可达陷阱)
    virtual ~Op() = default;

    // +0x38 @0xd0e1a0 (1B: ret) —— 默认空实现
    virtual void clear(Graph *graph_in) { (void)graph_in; }

    // +0x40 纯虚
    virtual GraphStatus prepare(hnnx::OpIoPtrs const &, bool tcm_available) = 0;
    // +0x48 纯虚
    virtual GraphStatus allocate(Graph &graph_in) = 0;

    // @0x10bc840 (28B): rdi/rsi 互换后 call get_extra_info@plt; 返回 [eax...]
    //   = ((OpExtraAttrib const&)graph.get_extra_info(this)).id
    OpId id(const Graph &graph_in) const noexcept { return graph_in.get_extra_info(this).id; }

    // @0x10bc7f0 (71B) —— 无分支 cmov 链的等价开关形式:
    //   d=chkpts.second, f=chkpts.first (均取 32 位无符号):
    //   d==0xFFFFFFFF: f∈{0,0xFFFFFFFF} → ChkptNone(1), 否则 ChkptNoDone(3)
    //   其他 d:        f∈{0,0xFFFFFFFF} → ChkptNoGate(2), 否则 ChkptNormal(0)
    //   (边界值 0xFFFFFFFEFFFFFFFF 恰为 {first=-1,second=-2} → NoGate, 已覆盖)
    ChkptStoreType get_chkpt_store_type(const Graph &graph_in) const
    {
        auto const &cp = graph_in.get_extra_info(this).chkpts;
        uint32_t const d = (uint32_t)cp.second;
        uint32_t const f = (uint32_t)cp.first;
        if (d == 0xFFFFFFFFu) return (f == 0u || f == 0xFFFFFFFFu) ? ChkptNone : ChkptNoDone;
        return (f == 0u || f == 0xFFFFFFFFu) ? ChkptNoGate : ChkptNormal;
    }

    // @0x10bc9b0 (114B):
    //   f = get_flag_word() (虚槽+0x70); attrib = get_extra_info(this) @+0x18 位域
    //   f & 0x4 (RESOURCE_HVX): !for_hlx → Vec(1); for_hlx → Elt(3)
    //   否则 (f & 0x10 (RESOURCE_HLX) || for_hlx) → Elt(3)
    //   否则 graph.is_hmx_threaded() && f & 0x8 (RESOURCE_HMX) → Mtx(2); 其余 Fg(0)
    OpStoreType get_op_store_type(const Graph &gr) const
    {
        Flags_word const f = this->get_flag_word();
        bool const for_hlx = gr.get_extra_info(this).for_hlx;
        if (f & (Flags_word(1) << unsigned(Flags::RESOURCE_HVX))) {
            if (!for_hlx) return OpStoreVec;
            return OpStoreElt;
        }
        if ((f & (Flags_word(1) << unsigned(Flags::RESOURCE_HLX))) || for_hlx) return OpStoreElt;
        if (gr.is_hmx_threaded())
            return (f & (Flags_word(1) << unsigned(Flags::RESOURCE_HMX))) ? OpStoreMtx : OpStoreFg;
        return OpStoreFg;
    }
    // flags → OpStoreType: (flags >> 2) & 3 (静态内联, 调用点折叠, 无独立符号)
    static OpStoreType get_op_store_type(uint32_t flags)
    {
        return OpStoreType((flags >> ChkptFlagShift) & ChkptOpFlagMask);
    };

    // @0x10bc860 (24B): get_extra_info(非 const) 返回值 [8] ← 64 位 pair 整体写入
    void set_chkpts(Graph &graph_in, const std::pair<int, int> chkpts)
    {
        graph_in.get_extra_info(this).chkpts = chkpts;
    }
    void set_chkpts(Graph &graph_in, int gate, int done)
    {
        set_chkpts(graph_in, std::make_pair(gate, done));
    }

    // bool is_input: true=输入(pair.first) / false=输出(pair.second)
    //   (allocate_generic @0x10bcff4 直接用 second; enumerate_blocks_generic @0x10bcf88
    //    cmovne rdx←rax: is_input→first —— 两处交叉钉死)
    const Tensor *get_input(size_t which) const { return this->get_input_output(which, true); }
    const Tensor *get_output(size_t which) const { return this->get_input_output(which, false); }

    // +0x50 @0xd0e1b0 (3B: xor %eax,%eax; ret) —— 默认 false
    virtual bool set_input(size_t which, const Tensor *tensor)
    {
        (void)which;
        (void)tensor;
        return false;
    }

    // +0x58 纯虚
    virtual bool is_valid() const noexcept = 0; // Is this op valid in this situation?

    // 无 .so 符号 (内联或未发射) —— 仅声明
    void dependence_resolved() noexcept;

    // @0xd4e1d0 (42B): n = typeid(*this).name();
    //   n == _ZTIN4hnnx14ConstWrapperOpE->name() → true
    //   n == _ZTIN4hnnx14ShapeWrapperOpE->name() → true
    bool is_const() const noexcept
    {
        const char *const n = typeid(*this).name();
        return n == _ZTSN4hnnx14ConstWrapperOpE || n == _ZTSN4hnnx14ShapeWrapperOpE;
    }

    // +0x60 纯虚 —— {num_inputs(first), num_outputs(second)}, 双寄存器返回
    virtual std::pair<size_t, size_t> num_inputs_outputs() const = 0;
    size_t num_outputs() const { return this->num_inputs_outputs().second; }
    size_t num_inputs() const { return this->num_inputs_outputs().first; }

    // +0x68 @0x10bc770 (109B):
    //   info = op_info_map_lookup(type_index(typeid(*this)));
    //   info 且 info->type_tag.data() 非空 → 返回之;
    //   否则 n = typeid(*this).name(); strstr(n,"Wrapper") 命中时:
    //     n == _ZTSN4hnnx14ShapeWrapperOpE → "Shape" (@0x398c480)
    //     n == _ZTSN4hnnx14ConstWrapperOpE → "Const" (@0x3967675)
    //   否则返回 n
    virtual const char *true_name() const
    {
        if (hnnx::OpInfo const *info = hnnx::op_info_map_lookup(std::type_index(typeid(*this))))
            if (const char *tag = info->get_type_tag()) return tag;
        const char *const n = typeid(*this).name();
        if (strstr(n, "Wrapper") != nullptr) {
            if (n == _ZTSN4hnnx14ShapeWrapperOpE) return "Shape";
            if (n == _ZTSN4hnnx14ConstWrapperOpE) return "Const";
        }
        return n;
    }

    // +0x70 @0xd4e390 (3B: xor %eax,%eax; ret) → hnnx::flags_for<Op> (主模板值 0)
    virtual Flags_word get_flag_word() const { return hnnx::flags_for<Op>; }
    // +0x78 @0xd0e1d0 (7B: lea 0x4628a0e("") → rax; ret) → hnnx::docs_for<Op>
    virtual const char *get_docs() const { return hnnx::docs_for<Op>(); }

    // +0x80 @0x10bc7e0 (12B): vptr[-1] → +0x8 → 即 typeid(*this).name()
    virtual const char *true_func() const noexcept { return typeid(*this).name(); }

    // @0x10bcca0 (8B): mov (%rdi),%rax; mov -0x8(%rax),%rax; ret → &typeid(*this)
    std::type_info const *get_type_extended() const { return &typeid(*this); }

    // +0x88 @0xd0e1e0 (17B, sret): [out]=1; [out+8]=vptr[-1] → type_info(typeid(*this))
    virtual hnnx::type_info get_type_info() const { return hnnx::type_info(typeid(*this)); }

    bool get_flag(Flags flag) const { return hnnx::test_flag_for(this->get_flag_word(), flag); }
    bool get_flag_and(Flags flag0, Flags flag1) const
    {
        return hnnx::test_flag_and(this->get_flag_word(), flag0, flag1);
    }

    // legacy 接口: input_output_blocks(true, mc_sel)/(..., false, ...) 的内联包装
    hnnx::blockid_set_t input_blocks(int mc_sel = -1) const { return this->input_output_blocks(true, mc_sel); }
    hnnx::blockid_set_t input_blocks(hnnx::MemoryClass mc) const { return this->input_output_blocks(true, int(mc)); }
    hnnx::blockid_set_t output_blocks(int mc_sel = -1) const { return this->input_output_blocks(false, mc_sel); }
    hnnx::blockid_set_t output_blocks(hnnx::MemoryClass mc) const
    {
        return this->input_output_blocks(false, int(mc));
    }

    // +0x90 @0xd0e200 (1B: ret) —— 默认空实现; 参数 is_input 见 get_input 注
    virtual void enumerate_blocks(hnnx::MemBlockEnumerator &en, bool is_input) const { (void)en; (void)is_input; }
    void enumerate_input_blocks(hnnx::MemBlockEnumerator &en) const { this->enumerate_blocks(en, true); }
    void enumerate_output_blocks(hnnx::MemBlockEnumerator &en) const { this->enumerate_blocks(en, false); }
    // enumerate_*_blocks_withfunc (MemBlockEnumWrapper 模板族) —— MemBlockEnumWrapper
    // 未反演, 略; SDK op.h:173-185。

    // +0x98 纯虚
    virtual void serialize(hnnx::SerOpsInterface &) const = 0;

    // tensor_deserializer_register_func: 静态 constexpr, 依赖 deserialize_tensor_tuple
    // (未反演), 略; SDK op.h:188-196。

    // @0x10bcf30 (64B):
    //   info = op_info_map_lookup(type_index(typeid(*this)));
    //   info → tail-jmp [info+0] (cost.funcp), 实参 (info 的 cost, graph, this);
    //   否则返回 @0x39bc73c 常量 = 281474976710656.0f = 2^48 = StandardCosts::GLACIAL
    float cost(const Graph &graph_in) const
    {
        if (hnnx::OpInfo const *info = hnnx::op_info_map_lookup(std::type_index(typeid(*this))))
            return info->get_cost().funcp(info->get_cost(), graph_in, this);
        return StandardCosts::GLACIAL;
    }

    // 'clone_mode' for Op::clone
    enum op_clonemode {
        opclone_auto, // opclone_dup if op has NULL_EXEC, otherwise opclone_realloc
        opclone_realloc, // when duplicating the output tensors, zero all block ids and reallocate
        opclone_dup // duplicate output with same block ids; and suppress ctor hooks.
    };

    // @0x10bccb0 (639B) —— 仅声明; 完整反汇编结论 (供 OpIoPtrs 反演后落地):
    //   1. mode==opclone_auto: mode = (get_flag_word() & 0x1000 /*NULL_EXEC*/)
    //        ? opclone_dup(2) : opclone_realloc(1)   [sbb 编码: 2-CF]
    //   2. clonemode 字 = (mode==opclone_dup)?1:0; op_def 非空 →
    //        OpIoPtrs io(g,this,new_id,op_def,clonemode) (5 参)
    //        否则 OpIoPtrs io(g,this,new_id,clonemode) (4 参)
    //   3. io 内部字 @+0x18 非零 → 失败短路 (OpIoPtrs 未反演, 语义待定)
    //   4. t = as_type ?: typeid(*this); info = op_info_map_lookup(t);
    //      info 且 [info+0x20](op_factory) → sret 调 factory(&out, io, new_id);
    //      out 非空 → out->prepare(io, true /*tcm_available*/, 虚槽+0x40);
    //        返回 OK(0) → 成功返回 out; 否则 out 置空且不再告警
    //   5. 失败时 __dynamic_cast(this, _ZTI2Op, MetaOpBase, 0) 命中 →
    //        调 MetaOpBase 虚槽 +0xb0(&out, this, g, new_id) 作第二机制
    //   6. 仍空且 bl (告警许可位 —— 0x10bcdca: lookup 返回后无条件置 1; 仅
    //        "工厂真正运行过"(info 且 factory 非空) 而产物为空/prepare 失败时
    //        在 0x10bce31 清 0。查表 miss 或 factory 为空均保持 1) →
    //        GetLogPriorityLevel()>0 时 qnndsp_log(1, "%s:317:WARNING:Op::clone
    //        on unsupported Op type", "op_prepare.cc", "")
    //   7. 异常路径: out 置空; 部分产物经虚槽 +0x30 (D0) 析构; ~OpIoPtrs 清理
    hnnx::uptr_Op clone(Graph &graph_in, OpId, op_clonemode opclonemode = opclone_auto,
                        std::type_info const *as_type = nullptr, const OpDef *op_def_in = nullptr) const;

    // @0x10bcb50 (116B): t 置空; swap_output(which, t) 失败 → t.reset()
    //   (reset: 读 ptr、清零, ptr 非空且字@8==0 → Tensor 虚槽 +0x18 析构)
    hnnx::uptr_Tensor steal_output(size_t which)
    {
        hnnx::uptr_Tensor t;
        if (!this->swap_output(which, t)) t.reset();
        return t;
    }
    // @0x10bcbd0 (21B): 显式 null-this 检查 (test %rdi,%rdi; je → false) 后尾跳虚槽 +0xa8
    //   (经整数转换比较以保留该字节级行为; 直接写 this==nullptr 会被编译器按 UB 优化掉)
    bool install_output(size_t which, hnnx::uptr_Tensor &&val)
    {
        if (reinterpret_cast<uintptr_t>(this) == 0) return false;
        return this->swap_output(which, val);
    }

  protected:
    // +0xa0 纯虚 —— which=序号, is_input: true=输入/false=输出
    virtual Tensor const *get_input_output(size_t which, bool is_input) const = 0;

    // swap_output 是 steal_output/install_output 的底层:
    //   进参 val 空 → 视作 steal; val 非空 → set_output (已占用则失败)
    // +0xa8 @0xd4e200 (3B: xor %eax,%eax; ret) —— 默认不支持, false
    virtual bool swap_output(size_t which, hnnx::uptr_Tensor &val)
    {
        (void)which;
        (void)val;
        return false;
    }

    // @0x10bcab0 (69B): for i<n: inputs_p[i]->虚槽+0x98 (enum_memory_blocks)(&en)
    void enumerate_op_input_blocks(hnnx::MemBlockEnumerator &en, Tensor const *const *inputs_p,
                                   unsigned n) const
    {
        for (unsigned i = 0; i < n; ++i) inputs_p[i]->enum_memory_blocks(en);
    }
    // @0x10bcb00 (69B): 同上, 数组元素为 16B uptr_Tensor (步长 0x10, 取首成员 Tensor*)
    void enumerate_op_output_blocks(hnnx::MemBlockEnumerator &en, hnnx::uptr_Tensor const *outputs_p,
                                    unsigned n) const
    {
        for (unsigned i = 0; i < n; ++i) outputs_p[i].ptr->enum_memory_blocks(en);
    }
    template <typename VIN, typename VOUT>
    [[gnu::always_inline]] inline void enumerate_op_blocks(hnnx::MemBlockEnumerator &en, VIN const &vinputs,
                                                           VOUT const &voutputs, bool is_input) const
    {
        if (is_input) {
            this->enumerate_op_input_blocks(en, vinputs.data(), vinputs.size());
        } else {
            this->enumerate_op_output_blocks(en, voutputs.data(), voutputs.size());
        }
    }

    // @0x10bcbf0 (163B): out(0x38B blockid_set_t) 逐字节初始化 {0,-1,0...0} (同
    //   tensor_base.hpp blockid_set_t 构造); 栈上 MemBlockEnumToSet{vptr, &out, mc_sel}
    //   (GOT 0x623f1b8 = _ZTVN4hnnx17MemBlockEnumToSetE); 调虚槽 +0x90
    //   enumerate_blocks(en, is_input) 收集; 返回 out
    hnnx::blockid_set_t input_output_blocks(bool is_input, int mc_sel) const
    {
        hnnx::blockid_set_t ret;
        hnnx::MemBlockEnumToSet en(ret, mc_sel);
        this->enumerate_blocks(en, is_input);
        return ret;
    }

    // @0x10bcf70 (98B):
    //   n = num_inputs_outputs(); count = is_input ? n.first : n.second (cmovne)
    //   for i<count: get_input_output(i, is_input)->虚槽+0x98 (enum_memory_blocks)(&en)
    void enumerate_blocks_generic(hnnx::MemBlockEnumerator &en, bool is_input) const
    {
        std::pair<size_t, size_t> const n = this->num_inputs_outputs();
        size_t const count = is_input ? n.first : n.second;
        for (size_t i = 0; i < count; ++i) this->get_input_output(i, is_input)->enum_memory_blocks(en);
    }

    // @0x10bcfe0 (89B):
    //   n = num_inputs_outputs().second (输出数, 直接用 rdx)
    //   for i<n: get_input_output(i, false)->虚槽+0xc0 allocate_func(alloc, 0)
    //   返回 GraphStatus::Success (xor %eax,%eax)
    GraphStatus allocate_generic(hnnx::Allocator *alloc = nullptr)
    {
        size_t const n = this->num_inputs_outputs().second;
        for (size_t i = 0; i < n; ++i) {
            Tensor *const t = const_cast<Tensor *>(this->get_input_output(i, false));
            t->allocate_func(*alloc, 0);
        }
        return GraphStatus::Success;
    }

    // @0x10bc880 (299B) —— 仅声明; 完整反汇编结论 (供 Serializer 反演后落地):
    //   graph = *(Serializer+0xa0); ei = graph.get_extra_info(this);
    //   容量检查/增长经 Serializer 虚槽 +0xa0 (flush), 游标 @+0xb8, 上限 @+0xb0:
    //   写 ei.id (8B);
    //   first=(int)ei.chkpts, second=(int)(ei.chkpts>>32);
    //   ChkptNormal: 写 first 再写 second (两 4B);
    //   ChkptNone:   不写;
    //   ChkptNoGate: 越界检查后写 second (1×4B);
    //   ChkptNoDone: 越界检查后写 first  (1×4B, 经 0x10bc98f)
    void serialize_internal(hnnx::Serializer &sctx, ChkptStoreType st) const;

  public:
    // @0x10bca30 (113B) —— 所有出口汇合 orl %r14d(st)。导出符号, 序列化侧
    //   (serialize.cc 等) 经 PLT 调用 → 公有:
    //   f = get_flag_word(); for_hlx = get_extra_info(this).for_hlx
    //   f & 0x4 (RESOURCE_HVX):   for_hlx ? 0xc : 0x4
    //   否则 (f & 0x10 (RESOURCE_HLX) || for_hlx): 0xc
    //   否则 graph.is_hmx_threaded(): f & 0x8 (RESOURCE_HMX)
    //   否则 0;   最终 return base | st (chkpt 类型占低 2 位, 标志占 bit2-3)
    uint32_t get_serialize_flags(const Graph &gr, ChkptStoreType st) const
    {
        Flags_word const f = this->get_flag_word();
        bool const for_hlx = gr.get_extra_info(this).for_hlx;
        uint32_t base;
        if (f & (Flags_word(1) << unsigned(Flags::RESOURCE_HVX))) {
            base = for_hlx ? 0xc : 0x4;
        } else if ((f & (Flags_word(1) << unsigned(Flags::RESOURCE_HLX))) || for_hlx) {
            base = 0xc;
        } else if (gr.is_hmx_threaded()) {
            base = uint32_t(f & (Flags_word(1) << unsigned(Flags::RESOURCE_HMX)));
        } else {
            base = 0;
        }
        return base | uint32_t(st);
    }
};

#if defined(_MSC_VER)
static_assert(sizeof(Op) == 8);
#endif // vptr 之外无数据成员 (构造 @0xd4e010 无成员写入)
static_assert(std::is_abstract<Op>::value);

// ---------------------------------------------------------------------------
// class ConstWrapperOp (op.h:341+) —— Op 的首个已知派生:
//   { uptr_Tensor owned_tensor @+0x10 }; 构造/细节未反演, 此处仅前置供引用。
// ---------------------------------------------------------------------------
namespace hnnx {
class ConstWrapperOp : public ::Op {
    ::hnnx::uptr_Tensor owned_tensor; // +0x10 (16B) → sizeof(ConstWrapperOp)=0x18

  public:
    ConstWrapperOp(::Graph &graph_in, OpId my_id_in, const ::OpDef *op_def_in);
    ConstWrapperOp(::Graph &graph_in, OpId my_id_in, ::hnnx::uptr_Tensor owned_tensor_in);
    explicit ConstWrapperOp(::hnnx::Deserz &dctx);
};
} // namespace hnnx
