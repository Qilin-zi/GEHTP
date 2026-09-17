#pragma once
// ============================================================================
// hnnx::MetaOpBase —— SDK meta_op_base.h 精确复刻 (M32)。
//
// 用途: 为 Op 的全部纯虚方法提供空默认实现, 使内部 Op (如 PreloadOp)
// 可以只覆写需要的槽。
//
// 证据 (libHtpPrepare.so):
//   * Op::clone @0x10bccb0 步骤 5: `__dynamic_cast(this, _ZTI2Op,
//     <MetaOpBase typeinfo>, 0)` 命中 → 经虚槽 +0xb0 调
//     clone_meta(&sret, this, g, new_id) —— 即 MetaOpBase 在 Op 的
//     22 槽 (0xb0 字节) vtable 之后新增的第 23 个槽 (0 基索引 22);
//   * 类型名字符串 "N4hnnx10MetaOpBaseE" @0x39aca38 (紧邻
//     "N4hnnx17SpecialPrepOpBaseE"/"N4hnnx10OpHookBaseE" 的类型名表);
//     _ZTV/_ZTI 对象未导出 (隐藏可见性), 由派生类 vtable 隐式物化;
//   * 默认体均为 SDK 头注释所标值 (prepare/allocate→Success,
//     is_valid→false, num_inputs_outputs→{0,0}, get_input_output→nullptr,
//     serialize→{}, clone_meta→空 uptr_Op); .so 中无独立导出符号
//     (nm/relocs 全文 0 命中), 佐证这些默认以弱定义/内联形式进入各派生
//     类 vtable 而非独立函数。
// ============================================================================

#include "op.hpp"

class Graph;
class Tensor;

namespace hnnx {

class OpIoPtrs;
class Deserz;

class MetaOpBase : public Op {
  public:
    MetaOpBase(){};
    MetaOpBase(Graph &graph_in, unsigned long long int my_id_in) : Op(graph_in, my_id_in) {}
    explicit MetaOpBase(hnnx::Deserz &dctx) : Op(dctx) {}

    virtual GraphStatus prepare(hnnx::OpIoPtrs const &, bool tcm_available) override; // → Success
    virtual GraphStatus allocate(Graph &graph_in) override; // → Success

    virtual bool is_valid() const noexcept override; // → false
    virtual std::pair<size_t, size_t> num_inputs_outputs() const override; // → {0,0}
    virtual Tensor const *get_input_output(size_t which, bool is_input) const override; // → nullptr
    virtual void serialize(hnnx::SerOpsInterface &) const override; // → {}
    virtual uptr_Op clone_meta(Graph &graph_in, OpId new_opid) const; // → uptr_Op(nullptr)
};

} // namespace hnnx
