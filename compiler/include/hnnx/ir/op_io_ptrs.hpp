#pragma once
// ============================================================================
// hnnx::OpIoPtrs —— libHtpPrepare.so 精确反演 (M32)。
//
// 用途 (SDK op_io_ptrs.h 注释):
//   (1) 基于 OpDef 造新 Op (op_def_map 信息);
//   (2) 造另一个 Op 的克隆并换新 ID。
//   (1) 型 ctor 从 OpDef 取 new_op_id; (2) 型 ctor 带参数。
//   两类 ctor 都收集输入 Tensor const* 数组; (1) 另收集 OutputDef const* 数组
//   并只记录输出数。克隆时 ctor hook 被抑制。
//
// 布局 sizeof 0x78 (5 个 ctor 的存储初始化序列逐一钉死):
//   +0x00 GraphPrepare *graphp_p      +0x08 Graph *graph_p (同一对象, 编译期异型)
//   +0x10 OpId new_op_id              +0x18 GraphStatus stat (初值 -1=ErrorFatal)
//   +0x20 OpHookBase const *ophook_ptr +0x28 unsigned num_out
//   +0x30 OpDef const *op_def         +0x38 Op const *op_to_clone (恰一非空)
//   +0x40 clonemode clone_mode (u32)  +0x48 vector<Tensor const*> in_tensors
//   +0x60 vector<OutputDef const*> out_defs
// clonemode: output_realloc=0, output_dup=1, output_steal=2。
// 析构: 隐式 (vector 自毁); ctor 失败展开路径手删 vector 缓冲 (@0xf85189:
//   先 out_defs @+0x68/0x70 再 in_tensors @+0x50/0x58 —— 逆声明序)。
//
// 导出符号 (全部):
//   C2/C1 (GraphPrepare&,OpDef const*)                   @0xf84a10
//   C2/C1 (GraphPrepare&,Op const*,OpId,clonemode)       @0xf85070
//   C2/C1 (Graph&,Op const*,OpId,clonemode)              @0xf851d0 → bad_cast 后委托
//   C2/C1 (Graph&,Op const*,OpId,OpDef const*,cm)        @0x131be70 → 委托 OpDef 型
//   C2/C1 (GraphPrepare&,Op const*,OpId,OpDef const*,cm) @0x131bed0 → 委托 OpDef 型
//   set_op_slicing(Op const&,unsigned) const             @0xf85230
//   get_output_for_cloned_op(unsigned) const             @0xf853f0
//   allocator() const                                    @0xf85540
//   full_allocator() const                               @0xf85550
//   ophook_func(pmf,Op&) const                           @0xd351a0
//
// 本文件同时落地 Op::clone @0x10bccb0 (Op 完整定义在 op.hpp, 此处经 inline
// 成员定义补体 —— 规避 op.hpp ↔ op_io_ptrs.hpp 循环包含)。
// ============================================================================

#include "hnnx/ir/status.hpp"
#include "hnnx/ir/op_hook_base.hpp"
#include "hnnx/ir/graph_prepare_base.hpp"
#include "hnnx/ir/op.hpp"
#include "hnnx/ir/meta_op_base.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

struct OutputDef;

namespace hnnx {

class Allocator;

class OpIoPtrs {
    GraphPrepare *graphp_p; // +0x00
    Graph *graph_p; // +0x08 — 指向同一对象, 此处编译期类型不同
    OpId new_op_id; // +0x10
    GraphStatus stat; // +0x18 (初值 ErrorFatal)
    OpHookBase const *ophook_ptr = nullptr; // +0x20
    unsigned num_out; // +0x28

  public:
    enum clonemode { output_realloc, output_dup, output_steal };
    OpDef const *op_def; // +0x30 —— 恰一非空 ...
    Op const *op_to_clone; // +0x38 —— ...的另一者为空
    clonemode clone_mode = output_realloc; // +0x40
    std::vector<Tensor const *> in_tensors; // +0x48
    std::vector<OutputDef const *> out_defs; // +0x60

    // 布局断言须在完整类语境 (成员函数体内) —— 类内直接 static_assert 时
    // OpIoPtrs 尚不完整, offsetof 不可用。
    static void layout_static_asserts()
    {
        static_assert(offsetof(OpIoPtrs, graphp_p) == 0x00 && offsetof(OpIoPtrs, graph_p) == 0x08 &&
                              offsetof(OpIoPtrs, new_op_id) == 0x10 && offsetof(OpIoPtrs, stat) == 0x18 &&
                              offsetof(OpIoPtrs, ophook_ptr) == 0x20 && offsetof(OpIoPtrs, num_out) == 0x28 &&
                              offsetof(OpIoPtrs, op_def) == 0x30 && offsetof(OpIoPtrs, op_to_clone) == 0x38 &&
                              offsetof(OpIoPtrs, clone_mode) == 0x40 && offsetof(OpIoPtrs, in_tensors) == 0x48 &&
                              offsetof(OpIoPtrs, out_defs) == 0x60 && sizeof(OpIoPtrs) == 0x78,
                      "OpIoPtrs 布局必须为 0x78");
    }

    // ------------------------------------------------------------------
    // (1) 型 ctor @0xf84a10 精确复刻:
    //   全字段先置初值 (stat=-1, clone_mode=output_realloc, 两 vector 空);
    //   new_op_id = def->id(+0x20); op_def = def;
    //   遍历 def->input_defs (OpRef 8B):
    //     t = g.get_tensor(src, 0);
    //     t 空 → opid_alias_map(@g+0x6db8) 查表 (hashmap::find @0xf925d0,
    //           元素 24B, (ret-begin)/24 索引折算, 哨兵 [g+0x6de0]):
    //           命中 → t = g.get_tensor(elem.val@+8, elem.u32@+0x10);
    //     仍空 → qnndsp_log(0, "%s:6897::ERROR:Can't find input tensor from
    //           node %llX to node %llX\n" @0x46203ab, "graph_prepare.cc",
    //           src, def->id) 并以 stat 保持 ErrorFatal 返回;
    //     否则 in_tensors.push_back(t);
    //   stat = g.collect_multi_outputdef(def, out_defs, &alias_map, nullptr);
    //   num_out = (unsigned)out_defs.size();
    OpIoPtrs(GraphPrepare &g, OpDef const *def)
        : graphp_p(&g), graph_p(&g), new_op_id(def->id), stat(GraphStatus::ErrorFatal), num_out(0),
          op_def(def), op_to_clone(nullptr)
    {
        for (OpRef const &ref : def->input_defs) {
            Tensor const *t = g.get_tensor(ref.input_id, 0);
            if (t == nullptr) {
                GraphPrepare::opid_alias_entry alias;
                if (g.find_opid_alias(ref.input_id, alias))
                    t = g.get_tensor(alias.target_id, alias.output_idx);
            }
            if (t == nullptr) {
                qnndsp_log(0, "%s:6897::ERROR:Can't find input tensor from node %llX to node %llX\n",
                           "graph_prepare.cc", ref.input_id, def->id);
                return; // stat 保持 ErrorFatal
            }
            this->in_tensors.push_back(t);
        }
        this->stat = g.collect_multi_outputdef(def, this->out_defs, g.opid_alias_map_raw(), nullptr);
        this->num_out = (unsigned)this->out_defs.size();
    }

    // (2) 型 ctor @0xf85070 精确复刻:
    //   nio = op->num_inputs_outputs() (虚槽+0x60); num_out = nio.second;
    //   n_in = nio.first; in_tensors.resize(n_in) (增长调 0xcf85d0, 收缩改 end);
    //   for i<n_in: t = op->get_input_output(i, true) (虚槽+0xa0);
    //     空 → qnndsp_log(0, "%s:6921::ERROR:Fatal error in %s function\n"
    //            @0x46203ef, "graph_prepare.cc", "OpIoPtrs", "") 后返回
    //            (stat 保持 -1);
    //     in_tensors[i] = t;
    //   stat = Success。
    OpIoPtrs(GraphPrepare &g, Op const *op_to_clone_in, OpId new_op_id_in,
             clonemode clone_mode_in = output_realloc)
        : graphp_p(&g), graph_p(&g), new_op_id(new_op_id_in), stat(GraphStatus::ErrorFatal),
          ophook_ptr(nullptr), num_out(0), op_def(nullptr), op_to_clone(op_to_clone_in),
          clone_mode(clone_mode_in)
    {
        std::pair<size_t, size_t> const nio = op_to_clone_in->num_inputs_outputs();
        this->num_out = (unsigned)nio.second;
        size_t const n_in = nio.first;
        if (n_in != 0) {
            this->in_tensors.resize(n_in);
            for (size_t i = 0; i < n_in; ++i) {
                Tensor const *const t = op_to_clone_in->get_input_output(i, true);
                if (t == nullptr) {
                    qnndsp_log(0, "%s:6921::ERROR:Fatal error in %s function\n", "graph_prepare.cc",
                               "OpIoPtrs", "");
                    return; // stat 保持 ErrorFatal
                }
                this->in_tensors[i] = t;
            }
        }
        this->stat = GraphStatus::Success;
    }

    // @0xf851d0: 引用下行转换后整体委托 (Graph 必须"就是"GraphPrepare):
    //   __dynamic_cast(&g, _ZTI5Graph [GOT 0x623fae0], _ZTI12GraphPrepare
    //   [GOT 0x623f0a0], 0) 空结果 → __cxa_bad_cast。
    OpIoPtrs(Graph &g, Op const *op_to_clone_in, OpId new_op_id_in, clonemode clone_mode_in = output_realloc)
        : OpIoPtrs(dynamic_cast<GraphPrepare &>(g), op_to_clone_in, new_op_id_in, clone_mode_in)
    {
    }

    // @0x131be70: 同样 bad_cast 检查后委托给 (1) 型 OpDef ctor —— 注意
    // new_op_id_in 参数被丢弃 (OpDef ctor 内以 def->id 覆写 +0x10,
    // 0x131bea3-0x131beb9 仅补写 op_to_clone/+0x38 与 clone_mode/+0x40)。
    OpIoPtrs(Graph &g, Op const *op_to_clone_in, OpId /*new_op_id_in 被丢弃*/,
             OpDef const *def, clonemode clone_mode_in = output_dup)
        : OpIoPtrs(dynamic_cast<GraphPrepare &>(g), def)
    {
        this->op_to_clone = op_to_clone_in;
        this->clone_mode = clone_mode_in;
    }

    // @0x131bed0: 无需转换, 直调 OpDef ctor 后补写两字段。
    OpIoPtrs(GraphPrepare &g, Op const *op_to_clone_in, OpId /*new_op_id_in 被丢弃*/,
             OpDef const *def, clonemode clone_mode_in = output_dup)
        : OpIoPtrs(g, def)
    {
        this->op_to_clone = op_to_clone_in;
        this->clone_mode = clone_mode_in;
    }

    GraphStatus status() const { return stat; }

    Graph &graph() const { return *graph_p; }
    GraphPrepare &graphp() const { return *graphp_p; }

    // @0xf85540 (12B): mov 0x8(%rdi),%rax; mov 0x1d8(%rax),%rax; ret
    //   = *(Allocator**)((char*)graph_p + 0x1d8) —— 直接读 Graph::allocator 字段。
    Allocator &allocator() const { return *this->graph_p->allocator; } // Graph+0x1d8

    // @0xf85550 (12B): mov (%rdi),%rdi; jmp *0x1c0(vptr) —— 虚调用 GraphPrepare 槽。
    ::fa::FancyAllocator &full_allocator() const { return this->graphp_p->get_full_allocator(); }

    // @0xf85230 (8B): mov (%rdi),%rdi; jmp set_self_slicing_num_slices@plt
    void set_op_slicing(Op const &op, unsigned n_slices) const
    {
        this->graphp_p->set_self_slicing_num_slices(&op, n_slices);
    }

    void add_ophook(OpHookBase const *hookp) { ophook_ptr = hookp; }

    // hook 经 .ophook(mfp, Op&) 调用, mfp 为 OpHookBase 成员指针。
    // 抑制规则: 克隆模式且非 output_realloc 时直接 Success。
    GraphStatus ophook(GraphStatus (OpHookBase::*mfp)(OpIoPtrs const &, Op &) const, Op &target_op) const
    {
        // hooks are suppressed if we are cloning an op, in a mode other than realloc.
        if (is_clone_mode() && clone_mode != output_realloc) return GraphStatus::Success;
        return (ophook_ptr != nullptr) ? ophook_func(mfp, target_op) : GraphStatus::Success;
    }

    OpId get_id() const { return new_op_id; }
    size_t n_inputs() const { return in_tensors.size(); }
    size_t n_outputs() const { return num_out; }

    inline bool is_clone_mode() const { return op_to_clone != nullptr; }

    // @0xf853f0 精确复刻:
    //   clone_mode==output_steal → result = op_to_clone->steal_output(idx)
    //     (@0xf85433 连同 dw 标志字节一起拷贝 —— 被偷张量保留其借用位) 后即返回;
    //   否则 alloc = graph_p->allocator (Graph+0x1d8);
    //     orig = op_to_clone->get_input_output(idx, false) (虚槽+0xa0);
    //     op_def 非空 → 先 conditionally_validate_single_quant (@0x6f0980) 再取
    //       &op_def->m_first_outputdef (OpDef+0x48) 作模板;
    //     result.ptr = orig->reallocate_clone(alloc, cm==output_dup, 模板?)
    //       (非虚 @0x1341420), dw 置 0;
    //     cm != output_dup 时补 allocate_func(*alloc, 0) (Tensor 虚槽+0xc0)。
    //   异常着陆垫 (@0xf85508): 结果置空, reallocate_clone 临时产物经 Tensor
    //     虚槽+0x18 (D0) 析构后 _Unwind_Resume —— 以下以 unique_ptr 临时对象等价实现。
    uptr_Tensor get_output_for_cloned_op(unsigned idx) const
    {
        uptr_Tensor result;
        if (this->clone_mode == output_steal) {
            // .so 为机器码直调 (无访问控制); const 成员内经 const_cast 镜像
            result = const_cast<Op *>(this->op_to_clone)->steal_output(idx); // 连 dw 一起拷贝
            return result;
        }
        hnnx::Allocator *const alloc = this->graph_p->allocator; // Graph+0x1d8
        bool const dup = this->clone_mode == output_dup;
        OutputDef const *def = nullptr;
        std::unique_ptr<Tensor> tmp;
        if (this->op_def != nullptr) {
            Tensor const *const orig = this->op_to_clone->get_input_output(idx, false);
            this->op_def->conditionally_validate_single_quant();
            def = &this->op_def->get_outputdef<false>(); // +0x48 内嵌模板
            tmp = orig->reallocate_clone(alloc, dup, def);
        } else {
            Tensor const *const orig = this->op_to_clone->get_input_output(idx, false);
            tmp = orig->reallocate_clone(alloc, dup, nullptr);
        }
        result.dw = 0;
        if (this->clone_mode != output_dup) // realloc: 对克隆体重新分配存储
            tmp->allocate(*alloc, 0); // Tensor 虚槽 +0xc0
        result.ptr = tmp.release();
        return result;
    }

  protected:
    // @0xd351a0 精确复刻:
    //   (ophook_ptr->*mfp)(*this, target_op) 之后, 返回值 100 (NotApplicable)
    //   被归一化为 0 (Success) —— xorl %ecx,%ecx; cmpl $0x64,%eax; cmovel。
    //   (pmf 首字 &1 判虚; 虚寻址 [vptr + word0 - 1]: word0=1→槽0
    //    pre_output_prep, word0=9→槽1 pre_allocate; this 调整字相加后作对象。)
    GraphStatus ophook_func(GraphStatus (OpHookBase::*mfp)(OpIoPtrs const &, Op &) const,
                            Op &target_op) const
    {
        GraphStatus r = (ophook_ptr->*mfp)(*this, target_op);
        if (r == GraphStatus::NotApplicable) r = GraphStatus::Success;
        return r;
    }
};

} // namespace hnnx

// ============================================================================
// Op::clone @0x10bccb0 —— 完整落地 (声明在 op.hpp, 此处给出 inline 体)。
//
//   1. mode==opclone_auto: flags=get_flag_word() (虚槽+0x70) & 0x1000
//      (NULL_EXEC) → opclone_dup(2) 否则 opclone_realloc(1)
//      (sbb 编码: r8=2-CF);
//   2. cm = (mode==opclone_dup); op_def 非空 → 5 参 OpIoPtrs(Graph&,this,id,
//      def,cm) @0x131be70, 否则 4 参 (Graph&,this,id,cm) @0xf851d0
//      —— 两路都传同一布尔量, output_steal(2) 永不为 Op::clone 所传;
//   3. io.stat != Success → 清毁 io 两 vector 后返回空 uptr_Op;
//   4. ti = as_type ?: typeid(*this) (vptr[-1]); info =
//      op_info_map_lookup(ti); noisy=bl=true;
//      info 且 info->op_factory(+0x20) → out = factory(io, new_id) (拷贝
//      ptr+dw); out 非空且 out->prepare(io, true /*tcm_available*/, 虚槽
//      +0x40) 返回 0 → 直接进入收尾 (成功); 否则 out.op 置空;
//      此后 noisy=false (仅工厂真正运行过才免告警);
//   5. out 仍空 → __dynamic_cast(this, _ZTI2Op [GOT 0x623fde0],
//      _ZTIN4hnnx12MetaOpBaseE @0x5ebe990 [隐藏 __si_class_type_info, 名
//      "N4hnnx10MetaOpBaseE" @0x39aca30, base=_ZTI2Op], 0) 命中 →
//      out = mob->clone_meta(graph, new_id) (虚槽 +0xb0), 非空即收尾返回;
//   6. 仍空且 noisy 且 GetLogPriorityLevel()>0 →
//      qnndsp_log(1, "%s:317:WARNING:Op::clone on unsupported Op type\n"
//      @0x4696a98, "op_prepare.cc", "");
//   7. 收尾 (0x10bcd5b): 逆序毁 io (先 out_defs 后 in_tensors), 返回 out。
//   异常着陆垫: out 置空; 产物经 dw==0 判定后走 Op 虚槽+0x30 (D0) 析构;
//     重抛。以下以 uptr_Op::reset() + 栈展开等价实现。
// ============================================================================
int GetLogPriorityLevel(); // @plt 0x6f3480 (全局 C 符号)

inline hnnx::uptr_Op Op::clone(Graph &graph_in, OpId new_opid, op_clonemode opclonemode_in,
                               std::type_info const *as_type, OpDef const *op_def_in) const
{
    if (opclonemode_in == opclone_auto)
        opclonemode_in = (this->get_flag_word() & 0x1000 /*NULL_EXEC*/) ? opclone_dup : opclone_realloc;
    hnnx::OpIoPtrs::clonemode const cm =
            (opclonemode_in == opclone_dup) ? hnnx::OpIoPtrs::output_dup : hnnx::OpIoPtrs::output_realloc;

    hnnx::uptr_Op out;
    std::optional<hnnx::OpIoPtrs> io; // .so 中两 ctor 共用同一栈槽 (rsp+8)
    if (op_def_in != nullptr)
        io.emplace(graph_in, this, new_opid, op_def_in, cm);
    else
        io.emplace(graph_in, this, new_opid, cm);

    if (io->status() != GraphStatus::Success) return out; // 失败短路, io 自毁

    std::type_info const *const ti = (as_type != nullptr) ? as_type : &typeid(*this);
    hnnx::OpInfo const *const info = hnnx::op_info_map_lookup(std::type_index(*ti));
    bool noisy = true;
    try {
        if (info != nullptr && info->get_op_factory() != nullptr) {
            out = info->get_op_factory()(*io, new_opid);
            if (out.op != nullptr) {
                if (out.op->prepare(*io, true) == GraphStatus::Success) return out; // 成功收尾
                out.op = nullptr; // prepare 失败
            }
            noisy = false;
        }
        if (out.op == nullptr) {
            if (hnnx::MetaOpBase const *const mob = dynamic_cast<hnnx::MetaOpBase const *>(this)) {
                out = mob->clone_meta(graph_in, new_opid);
                if (out.op != nullptr) return out; // 第二机制成功
            }
            if (noisy && GetLogPriorityLevel() > 0)
                qnndsp_log(1, "%s:317:WARNING:Op::clone on unsupported Op type\n", "op_prepare.cc", "");
        }
    } catch (...) {
        out.reset(); // dw==0 时经 Op 虚槽 +0x30 (D0) 析构
        throw;
    }
    return out;
}
