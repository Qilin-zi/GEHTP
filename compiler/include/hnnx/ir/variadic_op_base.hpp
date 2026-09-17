#pragma once
// ============================================================================
// hnnx::VariadicOpBase —— 变长输入/输出 Op 基类, libHtpPrepare.so 精确反演 (M32)。
// 无 SDK 头对应 (qnn_ori_include 仅有 typical_op.h) —— 全部方法签名取自导出
// 符号 Mangling, 行为取自逐指令反汇编 (M32_VariadicOpBase_all.asm)。
//
// 导出符号:
//   D1/D2 @0x139f9e0   D0 @0x139fa50 (ud2 —— 抽象类不可达陷阱)
//   allocate @0x139fa60   num_inputs_outputs @0x139fab0
//   get_input_output @0x139fac0   set_input @0x139fae0 (抛 range_error("cratevec"))
//   swap_output @0x139fb50   is_valid @0x139fbf0   enumerate_blocks @0x139fc00
//   serialize @0x139fc20   assign_input_pointers @0x139fc50
//   output_allocate_fixed @0x139fe80   deserialize_helper @0xddbe10
//   vtable _ZTVN4hnnx14VariadicOpBaseE (GOT 0x623f590 —— 已导出)。
//   ctor 无导出符号 (完全内联进派生类构造) —— 本镜像给 protected 缺省形,
//   访问级别/形参列为后续补证项。
//
// 对象布局 (全部方法体访址钉死, sizeof 0x28):
//   +0x00 vptr(Op 22 槽族)
//   +0x08 Tensor const  **inputs        +0x10 unsigned num_inputs (u32)
//   +0x18 uptr_Tensor   *outputs        +0x20 unsigned num_outputs(u32)
// ============================================================================

#include "hnnx/ir/typical_op_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace hnnx {

// .so 0xcf5240: __cxa_allocate_exception(8) 后抛 8 字节 std::exception 派生对象
// (vtable @0x5ebde58, typeinfo 基 _ZTISt9exception; 其 name 槽位字串与
//  DCrate::emplace0<ConcreteTensor<QUint16Crouton>> 的 __func 名字串相邻,
//  精确类名待 M33 Deserz 反演)。noreturn; 测试 TU 提供抛出体。
void deserz_scratch_overflow();

class VariadicOpBase : public Op {
  protected:
    // 字段名无符号证据 (无 SDK 头) —— 依 TypicalOpIoBase io.inputs()/outputs()
    // 命名传统取的镜像名; 偏移由全部方法体访址钉死。
    Tensor const **inputs = nullptr; // +0x08
    unsigned num_inputs = 0;         // +0x10 (u32; 序列化字低 16 位)
    uptr_Tensor *outputs = nullptr;  // +0x18
    unsigned num_outputs = 0;        // +0x20 (u32; 序列化字高 16 位)

    // 布局断言须在完整类语境 (成员函数体内) —— 类内直接 static_assert 时
    // VariadicOpBase 尚不完整, offsetof 不可用。
    static void layout_static_asserts()
    {
        static_assert(offsetof(VariadicOpBase, inputs) == 0x08 && offsetof(VariadicOpBase, num_inputs) == 0x10 &&
                              offsetof(VariadicOpBase, outputs) == 0x18 &&
                              offsetof(VariadicOpBase, num_outputs) == 0x20 && sizeof(VariadicOpBase) == 0x28,
                      "VariadicOpBase 布局必须为 0x28");
    }

    // ctor 已内联进派生类 (无导出符号)。deserialize_helper 会写满四个字段,
    // 与 "字段不初始化、由 helper 填充" 的 .so 语义一致。
    VariadicOpBase() = default;
    explicit VariadicOpBase(Deserz &dctx) : Op(dctx) {}

  public:
    // @0x139f9e0 (D2/D1): 置 vptr 后逆序 reset 全部输出槽
    //   (每槽: 先清 ptr, 非空且 dw==0 时经 Tensor 虚槽 +0x18 delete)。
    //   基类 Op 无成员析构 (平凡, 内联为空)。
    // D0 @0x139fa50 为 ud2 (抽象类陷阱) —— 本镜像类因 prepare 纯虚未覆盖
    //   而抽象, 编译器生成的 D0 同样不可达。
    ~VariadicOpBase() override
    {
        for (size_t i = this->num_outputs; i-- > 0;) this->outputs[i].reset();
    }

    // @0x139fa60: num_outputs==0 → 直接 Success; 否则逐输出
    //   outputs[i].ptr->allocate_func(*graph.[+0x1d8], 0) (虚槽 +0xc0,
    //   每次迭代重读分配器字段)。返回 Success。
    GraphStatus allocate(Graph &graph_in) override
    {
        for (unsigned i = 0; i < this->num_outputs; ++i)
            this->outputs[i].ptr->allocate(*graph_in.allocator, 0);
        return GraphStatus::Success;
    }

    // @0x139fab0: {num_inputs, num_outputs} (两 u32 零扩展)。
    std::pair<size_t, size_t> num_inputs_outputs() const override
    {
        return {this->num_inputs, this->num_outputs};
    }

    // @0x139fac0: is_input ? inputs[which] : outputs[which].ptr —— 无边界检查。
    Tensor const *get_input_output(size_t which, bool is_input) const override
    {
        return is_input ? this->inputs[which] : this->outputs[which].ptr;
    }

    // @0x139fae0: which < num_inputs (无符号) 时 inputs[which] = tensor, true;
    //   否则抛 std::range_error("cratevec") (16B 异常, 串 @0x55bacac,
    //   typeinfo _ZTISt11range_error / D1 GOT 0x623fe78/0x623ff70)。
    bool set_input(size_t which, Tensor const *tensor) override
    {
        if (which < this->num_inputs) {
            this->inputs[which] = tensor;
            return true;
        }
        throw std::range_error("cratevec");
    }

    // @0x139fb50: which >= num_outputs → false; value 与槽两者皆非空 → false;
    //   否则 16B 交换 (ptr+dw) 后 true。.so 内两处 "旧值归属则 delete" 检查
    //   均落在已清零的槽/临时上 (不可达死代码), 净语义即纯交换。
    bool swap_output(size_t which, uptr_Tensor &value) override
    {
        if (which >= this->num_outputs) return false;
        uptr_Tensor &out = this->outputs[which];
        if (value && out) return false;
        std::swap(out, value);
        return true;
    }

    // @0x139fbf0: movb $1,%al —— 恒真。
    bool is_valid() const noexcept override { return true; }

    // @0x139fc00: 尾跳 Op::enumerate_op_input_blocks(en, inputs, (unsigned)n)
    //   或 Op::enumerate_op_output_blocks(en, outputs, (unsigned)n)。
    void enumerate_blocks(MemBlockEnumerator &en, bool is_input) const override
    {
        if (is_input)
            this->enumerate_op_input_blocks(en, this->inputs, this->num_inputs);
        else
            this->enumerate_op_output_blocks(en, this->outputs, this->num_outputs);
    }

    // @0x139fc20: 单发 SerOpsInterface 虚槽 0, 7 参
    //   (this, n_in, inputs, n_out, outputs, 变长标志=1, 附加=0)。
    void serialize(SerOpsInterface &sctx) const override
    {
        sctx.op_serialize_func(this, this->num_inputs, this->inputs, this->num_outputs, this->outputs, 1, 0);
    }

    // @0x139fc50 精确复刻:
    //   crate = graph_crate(iop.graph());
    //   bytes = iop.in_tensors 字节长度 ([+0x50]-[+0x48]);
    //   bytes != 0 时:
    //     r = crate->add_record_slot(bytes, 8); r.status >= 0 时 crate 计数 [+0x40]++;
    //     this->inputs = r.slot; n = bytes/8; memset 槽区 0; this->num_inputs = n;
    //     for i<n: t = iop.in_tensors[i]; 空 →
    //       qnndsp_log(0, "%s:90::ERROR:Bad output, my id=%llx input #%d\n"
    //       @0x55bac65, "variadic_op_preapare.cc" @0x55bac94,
    //       iop.graph().get_extra_info(this).id, i, "" @0x4628a0e) 且返回 -1;
    //       否则 inputs[i] = t;
    //   返回 0 (Success)。
    GraphStatus assign_input_pointers(OpIoPtrs const &op_io_ptrs)
    {
        Graph &g = op_io_ptrs.graph();
        size_t const bytes = op_io_ptrs.in_tensors.size() * sizeof(Tensor const *);
        if (bytes != 0) {
            Crate *const crate = graph_crate(g);
            auto r = crate->add_record_slot(bytes, 8);
            if (r.status >= 0) crate->bump_record_count();
            this->inputs = reinterpret_cast<Tensor const **>(static_cast<void *>(r.slot));
            size_t const n = bytes / 8;
            std::memset(this->inputs, 0, bytes);
            this->num_inputs = (unsigned)n;
            for (size_t i = 0; i < n; ++i) {
                Tensor const *const t = op_io_ptrs.in_tensors[i];
                if (t == nullptr) {
                    qnndsp_log(0, "%s:90::ERROR:Bad output, my id=%llx input #%d\n",
                               "variadic_op_preapare.cc", g.get_extra_info(this).id, (int)i, "");
                    return GraphStatus::ErrorFatal;
                }
                this->inputs[i] = t;
            }
        }
        return GraphStatus::Success;
    }

    // @0x139fe80 (void): 输出区一次性重建。
    //   n_out = iop.num_out ([+0x28]); 为 0 直接返回;
    //   crate = graph_crate(iop.graph());
    //   r = crate->add_record_slot(n_out*16, 8); status>=0 → 计数++;
    //   this->outputs = r.slot; memset 0; this->num_outputs = n_out;
    //   n = min(num_fixed, n_out); n==0 返回;
    //   克隆 (iop.op_to_clone 非空): for i<n: 槽空时
    //     t = iop.get_output_for_cloned_op(i); 旧值归属则析构 (槽刚清零,
    //     死分支); 接管 ptr+dw;
    //   新建: for i<n: t = gen[i](this, *iop.out_defs[i], iop.graph());
    //     旧值归属则析构 (同上死分支); dw = 0。
    void output_allocate_fixed(OpIoPtrs const &op_io_ptrs, unsigned num_fixed,
                               tensor_generate_fp const *out_gen_functions)
    {
        unsigned const n_out = (unsigned)op_io_ptrs.n_outputs();
        if (n_out == 0) return;
        Graph &g = op_io_ptrs.graph();
        Crate *const crate = graph_crate(g);
        size_t const bytes = size_t(n_out) * sizeof(uptr_Tensor);
        auto r = crate->add_record_slot(bytes, 8);
        if (r.status >= 0) crate->bump_record_count();
        this->outputs = reinterpret_cast<uptr_Tensor *>(r.slot);
        std::memset(this->outputs, 0, bytes);
        this->num_outputs = n_out;
        unsigned const n = std::min(num_fixed, n_out);
        if (n == 0) return;
        if (op_io_ptrs.op_to_clone != nullptr) {
            for (unsigned i = 0; i < n; ++i) {
                uptr_Tensor t = op_io_ptrs.get_output_for_cloned_op(i);
                uptr_Tensor &slot = this->outputs[i];
                Tensor *const oldp = slot.ptr;
                slot.ptr = t.ptr;
                if (oldp != nullptr && slot.dw == 0) delete oldp;
                slot.dw = t.dw;
                t.ptr = nullptr;
            }
        } else {
            for (unsigned i = 0; i < n; ++i) {
                std::unique_ptr<Tensor> t = out_gen_functions[i](this, *op_io_ptrs.out_defs[i], g);
                uptr_Tensor &slot = this->outputs[i];
                Tensor *const oldp = slot.ptr;
                slot.ptr = t.release();
                if (oldp != nullptr && slot.dw == 0) delete oldp;
                slot.dw = 0;
            }
        }
    }

    // @0xddbe10 (void; 第二参数 j 在 .so 体内从未被读 —— 保留形参):
    //   word = 读 u32 (cursor>=end 时先 refill; 无 is_compressed 门控);
    //   n_in = word 低 16 位;
    //   n_in != 0 时 (输入区分配, 两径):
    //     快径: sp = [d+0x20] 非空 → 8 对齐; 越过 [d+0x28] 时调
    //       deserz_scratch_overflow() (0xcf5240 抛异常); 否则 [d+0x20] 前推;
    //     慢径: [d+0x30] crate → add_record_slot(n_in*8, 8) + 计数++;
    //     this->inputs = 槽; memset 0; this->num_inputs = n_in;
    //     deserz_fixup_input_ptrs([d+0x50]+0xf0, d, inputs, n_in);
    //   word >= 0x10000 时 (n_out = word>>16, 输出区同两径, 每元 16B):
    //     this->outputs = 槽; memset 0; this->num_outputs = n_out;
    //     for i<n_out: outputs[i] = deserialize_tensor(d) (旧值归属则析构
    //       —— 槽刚清零, 死分支)。
    //   (asm 循环界呈 max(n_out,1) 形态; word>=0x10000 ⇒ n_out>=1, 等价。)
    void deserialize_helper(Deserz &dctx, unsigned /*j —— .so 体内未读*/)
    {
        unsigned char *cur = dctx.read_cursor();
        if (cur >= dctx.read_end()) cur = dctx.refill();
        uint32_t const word = *reinterpret_cast<uint32_t const *>(cur);
        dctx.set_read_cursor(cur + 4);

        unsigned const n_in = word & 0xFFFFu;
        if (n_in != 0) {
            unsigned char *slot;
            if (unsigned char *sp = dctx.scratch_ptr()) {
                sp = reinterpret_cast<unsigned char *>((reinterpret_cast<uintptr_t>(sp) + 7) & ~uintptr_t(7));
                unsigned char *const next = sp + size_t(n_in) * sizeof(Tensor const *);
                if (next > dctx.scratch_end()) deserz_scratch_overflow();
                dctx.set_scratch_ptr(next);
                slot = sp;
            } else {
                Crate *const crate = dctx.scratch_crate();
                auto r = crate->add_record_slot(size_t(n_in) * sizeof(Tensor const *), 8);
                if (r.status >= 0) crate->bump_record_count();
                slot = reinterpret_cast<unsigned char *>(r.slot);
            }
            this->inputs = reinterpret_cast<Tensor const **>(slot);
            std::memset(slot, 0, size_t(n_in) * sizeof(Tensor const *));
            this->num_inputs = n_in;
            deserz_fixup_input_ptrs(static_cast<char *>(dctx.shared_ctx()) + 0xf0, dctx, this->inputs, n_in);
        }

        if (word >= 0x10000u) {
            unsigned const n_out = word >> 16;
            unsigned char *slot;
            if (unsigned char *sp = dctx.scratch_ptr()) {
                sp = reinterpret_cast<unsigned char *>((reinterpret_cast<uintptr_t>(sp) + 7) & ~uintptr_t(7));
                unsigned char *const next = sp + size_t(n_out) * sizeof(uptr_Tensor);
                if (next > dctx.scratch_end()) deserz_scratch_overflow();
                dctx.set_scratch_ptr(next);
                slot = sp;
            } else {
                Crate *const crate = dctx.scratch_crate();
                auto r = crate->add_record_slot(size_t(n_out) * sizeof(uptr_Tensor), 8);
                if (r.status >= 0) crate->bump_record_count();
                slot = reinterpret_cast<unsigned char *>(r.slot);
            }
            this->outputs = reinterpret_cast<uptr_Tensor *>(slot);
            std::memset(slot, 0, size_t(n_out) * sizeof(uptr_Tensor));
            this->num_outputs = n_out;
            for (unsigned i = 0; i < n_out; ++i) {
                uptr_Tensor t = deserialize_tensor(dctx);
                uptr_Tensor &o = this->outputs[i];
                Tensor *const oldp = o.ptr;
                o.ptr = t.ptr;
                if (oldp != nullptr && o.dw == 0) delete oldp;
                o.dw = t.dw;
                t.ptr = nullptr;
            }
        }
    }
};

} // namespace hnnx
