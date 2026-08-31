#pragma once
// ============================================================================
// ser_ops_interface.hpp — hnnx::SerOpsInterface 完整虚面 + OpSerHandle
//
// M33 字节级证据 (libHtpPrepare.so, x86_64):
//   * Serializer vtable @0x6057558 (_ZTVN4hnnx10SerializerE, GOT 0x623f070):
//     vptr(=符号+0x10) 起连续 18 槽 = 本接口全部虚函数, 槽序与 SDK
//     serialize_oplist.h 声明序逐一对应 (.rela.dyn R_X86_64_64 重定位逐槽核对):
//       +0x00 op_serialize_func   @0x12ed280 (导出)
//       +0x08 op_for_tensor_func  @0x12ed640
//       +0x10 prescan_ops_func    @0x12ea480
//       +0x18 graph_io_tensors    @0x12ec070
//       +0x20 checkpoints_table   @0x12ec740
//       +0x28 before_runlists     @0x12ec7c0
//       +0x30 after_non_runlist   @0x12ec7d0
//       +0x38 after_runlist       @0x12ec7e0
//       +0x40 serialize_op        @0x12ec820
//       +0x48 tensor_serialize    @0x12edb30
//       +0x50 shape_serialize     @0x12f51f0
//       +0x58 op_special          @0x12ed6e0
//       +0x60 spcl_done           @0x12ed740
//       +0x68 spcl_add_u32        @0x12ed8d0
//       +0x70 spcl_add_sized_vec  @0x12ed980
//       +0x78 spcl_fill_nullptr   @0x12edb00
//       +0x80 spcl_add_in_tensor  @0x12eda30
//       +0x88 spcl_add_out_tensor @0x12eda60
//   * 槽 0 无虚析构先行 (M32 已证: VariadicOpBase::serialize @0x139fc40 尾调
//     vptr 槽 0, 7 实参)。
//   * FileSerializer (匿名命名空间本地类, typeinfo 名 N4hnnx14FileSerializerE
//     @0x55b3833) 的本地 vtable @0x6057630 同 18 槽 + 后继 7 槽 —— 派生面一致。
// ============================================================================
#include <array>
#include <cstdint>
#include <utility>

// 双世界桥: Op/Tensor 真身为全局类 (mangling _ZN6Tensor…, _ZN6TensorC2EPK2Op)。
// 精确族 TU (-DHNNX_SER_PRECISE, 如 serializer.cpp) 用全局名 → 导出符号与 .so
// 一致; 旧族 TU (types.hpp 的 hnnx::Op/hnnx::Tensor 近似) 用 hnnx 名 —— 不引入
// 全局声明, 免 `using namespace hnnx` 歧义。两世界仅声明名不同, 虚表/布局同。
#if defined(HNNX_SER_PRECISE)
class Op;
class Tensor;
#define HNNX_OP_T ::Op
#define HNNX_TENSOR_T ::Tensor
#else
namespace hnnx {
class Op; // 旧族近似名 (types.hpp 完整定义的先声明; 若先于 types.hpp 亦合法)
class Tensor;
} // namespace hnnx
#define HNNX_OP_T hnnx::Op
#define HNNX_TENSOR_T hnnx::Tensor
#endif

namespace hnnx {

struct uptr_Tensor;
class Checkpoints;
struct ShapeFlags;
class OpSerHandle;

// ---------------------------------------------------------------------------
// op 序列化模式 (SDK serialize_oplist.h; M32 实测槽 0 第 6 参取值):
//   TypicalOpIoBase::serialize 传 0, VariadicOpBase::serialize 传 1。
//   5..257 为保留码 (SDK 注释); foreign=4 为外框架 op。
// ---------------------------------------------------------------------------
enum opMODE {
    opMODE_typical = 0,
    opMODE_variadic = 1,
    opMODE_foreign = 4,
};

// tensMODE: serialize_single_tensor_pointer / tensor_serialize 的张量类别
enum tensMODE {
    tensMODE_fail = 0,
    tensMODE_general = 1,
    tensMODE_shape = 2,
    tensMODE_scalar = 3,
};

// SerializeOpFlagMask (SDK serialize_defs.h): serialize_op 第 2 参 op_seqno 的
// 高 4 位标志掩码。
constexpr unsigned SerializeOpFlagMask = 0xf0000000u;
// OP_SEQNO_*: 序号字自身的特殊值
constexpr unsigned OP_SEQNO_FIRST = 0xffffffffu;
constexpr unsigned OP_SEQNO_LAST = 0xfffffffeu;
constexpr unsigned OP_SEQNO_NONE = 0xfffffffdu;
// OP_EXTATTR_*: serialize_op 标志位携带的 op 属性 (位段)
constexpr unsigned OP_EXTATTR_SELF_SLICING = 0x1;
constexpr unsigned OP_EXTATTR_PREDICATE = 0x2;
constexpr unsigned OP_EXTATTR_SWITCHED = 0x3;

// ---------------------------------------------------------------------------
// SerOpsInterface —— 序列化回调接口 (纯虚; 无数据成员, sizeof = 8)
// ---------------------------------------------------------------------------
class SerOpsInterface {
    friend class OpSerHandle; // 句柄方法经 protected 槽落记

  public:
    // +0x00: 典型/变长 op 的 IO 序列化 (M32 命名 ser_op_typical, M33 恢复真名)。
    //   variadic_flag = opMODE_*; extra = 附加类别位。
    virtual void op_serialize_func(HNNX_OP_T const *op, unsigned n_in, HNNX_TENSOR_T const *const *in_tens,
                                   unsigned n_out, uptr_Tensor const *out_tens, unsigned variadic_flag,
                                   unsigned extra) = 0;

    // SDK 拼写包装 (serialize_oplist.h): sctx.op_typical(op, inputs, outputs)
    // —— 槽 0 的 3 参便捷形 (variadic_flag=0, extra=0)。模板成员不入虚表。
    template <unsigned N_IN, unsigned N_OUT>
    void op_typical(HNNX_OP_T const *op, std::array<HNNX_TENSOR_T const *, N_IN> const &inputs,
                    std::array<uptr_Tensor, N_OUT> const &outputs)
    {
        op_serialize_func(op, N_IN, inputs.data(), N_OUT, outputs.data(), 0, 0);
    }
    // +0x08: 无 IO 描述的 op (仅落 op 本体记录)
    virtual void op_for_tensor_func(HNNX_OP_T const *op, unsigned n_out, uptr_Tensor const *out_tens) = 0;
    // +0x10: runlist 预扫描 (last = 最后一批)
    virtual void prescan_ops_func(HNNX_OP_T *const *seq_of_ops, unsigned n_ops, bool last = false) = 0;
    // +0x18: 图级输入输出张量
    virtual void graph_io_tensors(unsigned n_in, uptr_Tensor const *in_tensors, unsigned n_out,
                                  uptr_Tensor const *out_tensors, bool input_only) = 0;
    // +0x20: 检查点表
    virtual void checkpoints_table(Checkpoints const &) = 0;
    // +0x28: runlist 前计数 (norun/main/vector/mtx/segdesc)
    virtual void before_runlists(unsigned nops_norun, unsigned nops_main, unsigned nops_vector,
                                 unsigned nops_mtx, unsigned n_runlist_seg_descs) = 0;
    // +0x30: 非 runlist 段结束
    virtual void after_non_runlist() = 0;
    // +0x38: runlist 段结束
    virtual void after_runlist() = 0;
    // +0x40: 单 op 序列化 (op_seqno 含 SerializeOpFlagMask 标志位)
    virtual void serialize_op(HNNX_OP_T const &, unsigned op_seqno) = 0;
    // +0x48: 张量定义
    virtual void tensor_serialize(HNNX_TENSOR_T const *tens) = 0;
    // +0x50: 形状数组 (rank 个)
    virtual void shape_serialize(ShapeFlags const *basep, unsigned rank) = 0;
    // +0x58: 特殊 op 记录句柄 (见 OpSerHandle)
    virtual OpSerHandle op_special(HNNX_OP_T const *op) = 0;

  protected:
    // +0x60..+0x88: 仅经 OpSerHandle 调用
    virtual void spcl_done(OpSerHandle &) = 0; // ~OpSerHandle 触发
    virtual void spcl_add_u32(OpSerHandle &, uint32_t const *p, unsigned n) = 0;
    virtual void spcl_add_sized_vec(OpSerHandle &, uint32_t const *data, bool extra) = 0;
    virtual void spcl_fill_nullptr(OpSerHandle &, unsigned n) = 0;
    virtual void spcl_add_in_tensor(OpSerHandle &, HNNX_TENSOR_T const *) = 0;
    virtual void spcl_add_out_tensor(OpSerHandle &, uptr_Tensor const &) = 0;
    // 工厂: 派生类 (Serializer::op_special @0x12ed6e0) 构造句柄
    OpSerHandle make_opser_handle(unsigned info);
};

// ---------------------------------------------------------------------------
// OpSerHandle —— op_special 返回的记录累积句柄 (SDK 布局: {owner&, info}, 0x10)
// 调用序列契约 (SDK): ≤63 个 data_u32 词 → 至多 1 次 sized_vec → ≤15 次
// tensor_in → ≤7 次 tensor_out; fill_nullptr 任意时刻。
// M23 曾以 vector 逼近其行为 (op_ser_handle.hpp); M33 恢复真身 (仅两成员,
// 词直接累积进 owner 即 Serializer 的 op 记录区 +0xb8)。
// ---------------------------------------------------------------------------
class OpSerHandle {
    friend class SerOpsInterface;

  protected:
    SerOpsInterface &owner; // +0x00
    unsigned info;          // +0x08 (op_special 的类别字)
    OpSerHandle(SerOpsInterface &owner_in, unsigned info_in) : owner(owner_in), info(info_in) {}

  public:
    inline ~OpSerHandle() { owner.spcl_done(*this); }
    OpSerHandle(const OpSerHandle &) = delete;
    OpSerHandle &operator=(const OpSerHandle &) = delete;
    // 移动: 转移所有权, 源句柄失效 (SDK 通过重新初始化实现)
    OpSerHandle(OpSerHandle &&from) noexcept : owner(from.owner), info(from.info) { from.info = 0xffffffffu; }

    OpSerHandle &data_u32(uint32_t const *p, unsigned n)
    {
        owner.spcl_add_u32(*this, p, n);
        return *this;
    }
    OpSerHandle &data_u32(uint32_t v)
    {
        owner.spcl_add_u32(*this, &v, 1);
        return *this;
    }
    OpSerHandle &sized_vec(uint32_t const *data, bool extra = false)
    {
        owner.spcl_add_sized_vec(*this, data, extra);
        return *this;
    }
    OpSerHandle &tensor_in(HNNX_TENSOR_T const *t)
    {
        owner.spcl_add_in_tensor(*this, t);
        return *this;
    }
    OpSerHandle &tensor_out(uptr_Tensor const &t)
    {
        owner.spcl_add_out_tensor(*this, t);
        return *this;
    }
    OpSerHandle &fill_nullptr(unsigned n)
    {
        owner.spcl_fill_nullptr(*this, n);
        return *this;
    }
    unsigned get_info() const { return info; }
};

inline OpSerHandle SerOpsInterface::make_opser_handle(unsigned info) { return OpSerHandle(*this, info); }

} // namespace hnnx
