#pragma once
// ============================================================================
// hnnx::TypicalOpUtil / TypIoIo<N_OUT,N_IN> / TypicalOpIoBase<N_OUT,N_IN>
//   —— libHtpPrepare.so 精确反演 (M32)。
//
// 类层级 (SDK typical_op.h):
//   Op → TypicalOpUtil → TypicalOpIoBase<N_OUT,N_IN> → TypicalOpIo<Ftype>
//                              (M32 止于此; TypicalOpIO<Ftype>/TypicalOp<F>
//                               需 ArgsTuples 机制, 列为后续里程碑)
//
// TypicalOpUtil (非模板工具基类) 导出符号:
//   D1/D2 @0x139a930  D0 @0x1399140 (ud2 族? 见下)
//   get_flag_word @0x1399150   do_allocate @0x1399170
//   assign_input_pointers @0x13991c0   output_create @0x1399270
//   output_scratch_create @0x1399440   output_allocate @0x13995d0
//   output_allocate_with_scratch @0x1399630
//   do_deserialize @0xdd92f0
//   vtable _ZTVN4hnnx13TypicalOpUtilE (GOT 0x623fda8)。
// TypicalOpIoBase 显式实例化 9 组 (extern template; .so 均有弱符号):
//   <0,1> <1,0> <1,1> <1,2> <1,3> <1,4> <1,5> <1,6> <2,1>
//   <1,1> 代表体: ctor(OpIoPtrs) @0x1399a30 / ctor(Deserz) @0x1399a60 /
//   swap_output @0x1399ad0 / set_input @0x1399b90 / get_input_output @0x1399b70 /
//   check_szal_base @0x1399b60 (mov al,1 —— x86 宿主恒真) /
//   allocate @0x1399bc0 / is_valid @0x1399be0。
//
// 对象布局 (以 <1,1> ctor 初始化序列钉死, sizeof 0x20):
//   +0x00 vptr   +0x08 inputs_arr[N_IN] (Tensor const*, 不初始化!)
//   +0x08+8*N_IN outputs_arr[N_OUT] (uptr_Tensor 16B, ctor 置 {0,0})
// ============================================================================

#include "hnnx/ir/op_io_ptrs.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

// SerOpsInterface 18 槽真面在 serialize/ser_ops_interface.hpp (M33)。
// Op/Tensor 双世界桥 (HNNX_OP_T/HNNX_TENSOR_T) 亦由该头提供。
#include "hnnx/serialize/ser_ops_interface.hpp"

namespace hnnx {

class Deserz;

// tensor_generate_fp (SDK op_utils.h:81)
typedef std::unique_ptr<Tensor> (*tensor_generate_fp)(Op const *, OutputDef const &, Graph &);
// dt_rank_pair (SDK op_utils.h:111)
struct dt_rank_pair {
    DType dt;
    unsigned rank;
};

// (M32 曾在此声明最小面 SerOpsInterface —— 槽 0 占位名 ser_op_typical;
//  M33 已由 serialize/ser_ops_interface.hpp 的 18 槽真面取代, 槽 0 真名
//  op_serialize_func; op_typical 成员包装亦随真头。)

// hnnx::deserialize_tensor (SDK deserialize_tensors.h; 实现待 M33)
uptr_Tensor deserialize_tensor(Deserz &dctx);

// M32 边界: Deserz 输入指针修正 (strip 函数 @0xcfae60, 调用点 do_deserialize
//   @0xdd9357 与 VariadicOpBase::deserialize_helper @0xddbe8b:
//   rdi = [d+0x50]+0xf0, rsi = &d, rdx = inputs, ecx = n_in)。M33 展开。
void deserz_fixup_input_ptrs(void *sharedctx_plus_0xf0, Deserz &d, Tensor const **inputs, unsigned n_in);

//
// TypicalOpUtil —— 非模板工具方法集
//
class TypicalOpUtil : public Op {
  protected:
    explicit TypicalOpUtil(OpIoPtrs const &ioptrs) : Op(ioptrs.graph(), ioptrs.get_id()) {}
    explicit TypicalOpUtil(Deserz &dctx) : Op(dctx) {}
    TypicalOpUtil(const TypicalOpUtil &) = delete;
    TypicalOpUtil &operator=(const TypicalOpUtil &) = delete;
    TypicalOpUtil(TypicalOpUtil &&) = delete;
    TypicalOpUtil &operator=(TypicalOpUtil &&) = delete;
    virtual ~TypicalOpUtil() override = default; // D1 @0x139a930 (空体); D0 @0x1399140

    // @0x1399150 (48B): info = op_info_map_lookup(*[vptr-1]); 返回
    //   info ? info->get_flags() ([+0x10]) : 0。
    virtual Flags_word get_flag_word() const override
    {
        OpInfo const *const info = op_info_map_lookup(std::type_index(typeid(*this)));
        return info != nullptr ? info->get_flags() : 0;
    }
    virtual const char *get_docs() const override { return hnnx::docs_for<TypicalOpUtil>(); }

    // @0xdd92f0 精确复刻:
    //   if (![d+0x9c]) { // 非压缩
    //     cur = read_cursor; if (cur >= end) cur = refill()(虚槽+0x10);
    //     set_read_cursor(cur + 4);   // 跳过 4 字节 (值不使用)
    //   }
    //   if (n_in) deserz_fixup_input_ptrs([d+0x50]+0xf0, d, inputs, n_in);
    //   if (n_out) for i: outputs[i] = deserialize_tensor(d) (uptr 交换语义)。
    void do_deserialize(Deserz &dctx, size_t n_in, Tensor const **inputs, size_t n_out, uptr_Tensor *outputs)
    {
        if (!dctx.is_compressed()) { // [d+0x9c]
            unsigned char *cur = dctx.read_cursor();
            if (cur >= dctx.read_end()) cur = dctx.refill();
            dctx.set_read_cursor(cur + 4);
        }
        if (n_in != 0) deserz_fixup_input_ptrs(static_cast<char *>(dctx.shared_ctx()) + 0xf0, dctx, inputs,
                                               (unsigned)n_in);
        // 槽位由 ctor 刚清零 —— .so 即纯 16B 搬运, 无旧值析构分支。
        for (size_t i = 0; i < n_out; ++i) {
            uptr_Tensor t = deserialize_tensor(dctx);
            uptr_Tensor &slot = outputs[i];
            slot.ptr = t.ptr;
            slot.dw = t.dw;
            t.ptr = nullptr;
        }
    }

    // @0x1399170: for i<n_out: outputs[i].ptr->allocate_func(*graph.[+0x1d8], 0)
    //   (虚槽 +0xc0); 返回 Success。
    GraphStatus do_allocate(Graph &graph_in, size_t n_out, uptr_Tensor *outputs)
    {
        for (size_t i = 0; i < n_out; ++i) outputs[i].ptr->allocate(*graph_in.allocator, 0);
        return GraphStatus::Success;
    }

    // @0x13991c0 精确复刻:
    //   for i<n: t = iop.in_tensors[i]; t 空 →
    //     qnndsp_log(0, "%s:55::ERROR:Bad output, my id=%llx inp #%d\n"
    //     @0x55ba6f3, "typical_op_prepare.cc" @0x55ba720,
    //     iop.graph().get_extra_info(this).id, (int)i) 且返回 -1 (ErrorFatal);
    //   inputs[i] = t; 返回 Success。
    GraphStatus assign_input_pointers(OpIoPtrs const &op_io_ptrs, size_t n_inputs, Tensor const **inputs)
    {
        for (size_t i = 0; i < n_inputs; ++i) {
            Tensor const *const t = op_io_ptrs.in_tensors[i];
            if (t == nullptr) {
                qnndsp_log(0, "%s:55::ERROR:Bad output, my id=%llx inp #%d\n", "typical_op_prepare.cc",
                           op_io_ptrs.graph().get_extra_info(this).id, (int)i);
                return GraphStatus::ErrorFatal;
            }
            inputs[i] = t;
        }
        return GraphStatus::Success;
    }
    GraphStatus set_input_pointer(size_t which, Tensor const *input); // 无独立导出体 (全内联)

    // @0x1399270 精确复刻:
    //   克隆路径 (iop.op_to_clone 非空): for i<n: 槽已占用则跳过; 否则
    //     t = iop.get_output_for_cloned_op(i); 接管 (连 dw 一起)。
    //   新建路径 (n != 0): for i<n: 槽空时 t = gen[i](this, *iop.out_defs[i],
    //     iop.graph()) (tensor_generate_fp, 按输出索引取函数指针); dw 置 0。
    //   两条路径都只写空槽 —— 无旧值析构分支。
    //   返回 Success。
    GraphStatus output_create(OpIoPtrs const &op_io_ptrs, size_t num_outputs, uptr_Tensor *outputs,
                              tensor_generate_fp const *out_gen_functions)
    {
        if (op_io_ptrs.op_to_clone != nullptr) { // [iop+0x38]
            for (unsigned i = 0; i < num_outputs; ++i) {
                if (outputs[i].ptr != nullptr) continue; // 已占位
                uptr_Tensor t = op_io_ptrs.get_output_for_cloned_op(i);
                uptr_Tensor &slot = outputs[i];
                slot.ptr = t.ptr;
                slot.dw = t.dw; // 被克隆/被偷张量的借用位一并保留
                t.ptr = nullptr;
            }
        } else if (num_outputs != 0) {
            Graph &g = op_io_ptrs.graph();
            for (unsigned i = 0; i < num_outputs; ++i) {
                if (outputs[i].ptr != nullptr) continue;
                OutputDef const *const def = op_io_ptrs.out_defs[i]; // [iop+0x60] 向量
                std::unique_ptr<Tensor> t = out_gen_functions[i](this, *def, g);
                uptr_Tensor &slot = outputs[i];
                slot.ptr = t.release();
                slot.dw = 0; // 生成张量归属
            }
        }
        return GraphStatus::Success;
    }

    // @0x1399440 精确复刻:
    //   栈上 OutputDef 模板 (0x50B): max_sizes 偶数维=1 奇数维=0
    //   (movaps 0x399d310 = {01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00}
    //   → +0x08/+0x18/+0x28/+0x38 四组 {1,0}), stepsize=1.0f (+0x4c);
    //   for i<n_scratch: 槽已占用则跳过; 否则
    //     template.rank(+0x00) ← dt_rank_vals[i].rank ([+i*8+4]),
    //     template.dtype(+0x04) ← dt_rank_vals[i].dt ([+i*8]),
    //     t = gen[i](this, template, gr); 旧值归属则析构; dw=0。
    //   收尾: ea = gr.get_extra_info(this); 16 位字 @+0x18 =
    //     (old & ~0x3F) | ((n_scratch & 0x1F) << 1)   // bits1-5 = scratch 数
    //   返回 Success。
    GraphStatus output_scratch_create(Graph &gr, size_t num_scratch_outputs, uptr_Tensor *scratch_outputs,
                                      tensor_generate_fp const *out_gen_functions, dt_rank_pair const *dt_rank_vals)
    {
        OutputDef tdef{};
        for (size_t d = 0; d < MAX_DIMENSIONS; d += 2) {
            tdef.max_sizes[d] = 1; // {1,0}×4 组 (movaps 常量 0x399d310)
            tdef.max_sizes[d + 1] = 0;
        }
        tdef.stepsize = 1.0f;
        for (unsigned i = 0; i < num_scratch_outputs; ++i) {
            if (scratch_outputs[i].ptr != nullptr) continue;
            tdef.rank = dt_rank_vals[i].rank; // 模板 +0x00 ← pair[+4]
            tdef.dtype = dt_rank_vals[i].dt; // 模板 +0x04 ← pair[+0]
            std::unique_ptr<Tensor> t = out_gen_functions[i](this, tdef, gr);
            uptr_Tensor &slot = scratch_outputs[i];
            slot.ptr = t.release();
            slot.dw = 0; // 生成张量归属
        }
        OpExtraAttrib &ea = gr.get_extra_info(this);
        ea.num_scratch_outputs = (unsigned char)(num_scratch_outputs & 0x1F); // 位域 bits1-5
        return GraphStatus::Success;
    }

    // @0x13995d0 精确复刻:
    //   output_create(...);
    //   hr = iop.ophook(pre_allocate /*pmf {9,0}*/, *this) —— 抑制规则
    //     (克隆且非 output_realloc → Success; 无 hook → Success) 内嵌于 ophook;
    //     hr != Success → 直接返回 hr (否决);
    //   尾跳 this->allocate(iop.graph()) (虚槽 +0x48)。
    GraphStatus output_allocate(OpIoPtrs const &op_io_ptrs, size_t num_outputs, uptr_Tensor *outputs,
                                tensor_generate_fp const *out_gen_functions)
    {
        output_create(op_io_ptrs, num_outputs, outputs, out_gen_functions);
        GraphStatus const hr = op_io_ptrs.ophook(&OpHookBase::pre_allocate, *this);
        if (hr != GraphStatus::Success) return hr;
        return this->allocate(op_io_ptrs.graph()); // 虚槽 +0x48 (尾跳)
    }

    // @0x1399630 精确复刻:
    //   n_real = num_outputs - num_scratch_outputs;
    //   克隆: output_create(…, num_outputs 全量, …);
    //   否则: output_create(…, n_real, …); num_scratch 非零再
    //     output_scratch_create(graph, num_scratch, outputs+n_real,
    //                           gen+n_real, dt_rank_vals);
    //   随后与 output_allocate 相同的 hook + 尾跳虚 allocate。
    GraphStatus output_allocate_with_scratch(OpIoPtrs const &op_io_ptrs, size_t num_outputs,
                                             size_t num_scratch_outputs, uptr_Tensor *outputs,
                                             tensor_generate_fp const *out_gen_functions,
                                             dt_rank_pair const *dt_rank_vals)
    {
        size_t const n_real = num_outputs - num_scratch_outputs;
        if (op_io_ptrs.op_to_clone != nullptr) {
            output_create(op_io_ptrs, num_outputs, outputs, out_gen_functions); // 克隆: 全量
        } else {
            output_create(op_io_ptrs, n_real, outputs, out_gen_functions);
            if (num_scratch_outputs != 0)
                output_scratch_create(op_io_ptrs.graph(), num_scratch_outputs, outputs + n_real,
                                      out_gen_functions + n_real, dt_rank_vals);
        }
        GraphStatus const hr = op_io_ptrs.ophook(&OpHookBase::pre_allocate, *this); // 抑制内嵌
        if (hr != GraphStatus::Success) return hr;
        return this->allocate(op_io_ptrs.graph());
    }
};

//
// TypIoIo<N_OUT, N_IN> —— 输入/输出数组载体 (0 输出/0 输入特化共享空数组)
//
template <unsigned N_OUT, unsigned N_IN> struct TypIoIo {
    // 输入指针。保持为泛型 Tensor*,prepare 时检查, execute 时静态下转。
    std::array<const Tensor *, N_IN> inputs_arr;
    // 输出所有权 (uptr)。execute 时静态下转。
    std::array<uptr_Tensor, N_OUT> outputs_arr;

    std::array<const Tensor *, N_IN> &inputs() { return inputs_arr; }
    std::array<const Tensor *, N_IN> const &inputs() const { return inputs_arr; }
    std::array<uptr_Tensor, N_OUT> &outputs() { return outputs_arr; }
    std::array<uptr_Tensor, N_OUT> const &outputs() const { return outputs_arr; }
};

extern std::array<uptr_Tensor, 0> typical_op_0_outputs;
extern std::array<const Tensor *, 0> typical_op_0_inputs;

// 0 输出特化: std::array<uptr_Tensor,0> 在部分实现中仍占 1 槽, 以共享空数组
// 彻底消除 (SDK 同注释)。
template <unsigned N_IN> struct TypIoIo<0, N_IN> {
    std::array<const Tensor *, N_IN> inputs_arr;

    std::array<const Tensor *, N_IN> &inputs() { return inputs_arr; }
    std::array<const Tensor *, N_IN> const &inputs() const { return inputs_arr; }
    std::array<uptr_Tensor, 0> &outputs() { return typical_op_0_outputs; }
    std::array<uptr_Tensor, 0> const &outputs() const { return typical_op_0_outputs; }
};
template <unsigned N_OUT> struct TypIoIo<N_OUT, 0> {
    std::array<uptr_Tensor, N_OUT> outputs_arr;

    std::array<const Tensor *, 0> &inputs() { return typical_op_0_inputs; }
    std::array<const Tensor *, 0> const &inputs() const { return typical_op_0_inputs; }
    std::array<uptr_Tensor, N_OUT> &outputs() { return outputs_arr; }
    std::array<uptr_Tensor, N_OUT> const &outputs() const { return outputs_arr; }
};
template <> struct TypIoIo<0, 0> {
    std::array<const Tensor *, 0> &inputs() { return typical_op_0_inputs; }
    std::array<const Tensor *, 0> const &inputs() const { return typical_op_0_inputs; }
    std::array<uptr_Tensor, 0> &outputs() { return typical_op_0_outputs; }
    std::array<uptr_Tensor, 0> const &outputs() const { return typical_op_0_outputs; }
};

//
// TypicalOpIoBase<N_OUT, N_IN>
//
template <unsigned N_OUT, unsigned N_IN> class TypicalOpIoBase : public TypicalOpUtil {
    static constexpr size_t n_inputs = N_IN;
    static constexpr size_t n_outputs = N_OUT;

  protected:
    // @0x1399a30 (<1,1>): Op(iop.graph(), iop.get_id()) 后置 vptr; 仅初始化
    //   outputs_arr (inputs_arr 不初始化 —— SDK 语义同)。
    explicit TypicalOpIoBase(OpIoPtrs const &ioptrs) : TypicalOpUtil(ioptrs) {}
    // @0x1399a60 (<1,1>): 同上初始化, 随后
    //   do_deserialize(d, n_in, io.inputs().data(), n_out, io.outputs().data())
    //   (n==0 时必须用 .data() 而非 &arr[0] —— 后者在 0 长时 abort)。
    explicit TypicalOpIoBase(Deserz &dctx) : TypicalOpUtil(dctx)
    {
        do_deserialize(dctx, n_inputs, io.inputs().data(), n_outputs, io.outputs().data());
    }

    // @0x1399ad0 (<1,1>): which < n_outputs 且 (value 空 或 槽空) 时
    //   16 字节 std::swap(ptr,dw) —— 两方向都不析构 (被换出者由调用方管理);
    //   否则 false。
    virtual bool swap_output(size_t which, uptr_Tensor &value) override
    {
        if (which < n_outputs) {
            uptr_Tensor &out = io.outputs()[which];
            if (!value || !out) {
                std::swap(out, value);
                return true;
            }
        }
        return false;
    }
    // @0x1399b60 (<1,1>): movb $1,%al —— x86 宿主构建恒真 (bake 尺寸检查为
    //   Hexagon 专用)。非虚成员 (rdi 即 this, 未用)。
    bool check_szal_base() { return true; }

  public:
    TypIoIo<n_outputs, n_inputs> io; // 输入与输出指针 (+0x08 起)

  protected:
    // @0x1399b70 (<1,1>): 输入 → this+8+which*8; 输出 → this+0x10+which*0x10
    //   (即 TypIoIo 数组直寻址)。
    virtual const Tensor *get_input_output(size_t which, bool is_input) const override
    {
        if (is_input) {
            assert(which < n_inputs);
            return io.inputs()[which];
        } else {
            assert(which < n_outputs);
            return io.outputs()[which].ptr; // .get()
        }
    }

  public:
    // @0x1399b90 (<1,1>): std::swap(io.inputs().at(which), tensor); true
    virtual bool set_input(size_t which, const Tensor *tensor) override
    {
        assert(which < n_inputs);
        std::swap(io.inputs().at(which), tensor);
        return true;
    }

    virtual std::pair<size_t, size_t> num_inputs_outputs() const override { return {n_inputs, n_outputs}; }

    // @0x1399bc0 (<1,1>): n_outputs>0 → do_allocate(graph, n, outputs.data())
    //   (内联展开为 outputs[0].ptr->allocate_func(*graph.[+0x1d8], 0));
    //   n_outputs==0 → Success。
    virtual GraphStatus allocate(Graph &graph_in) override
    {
        if constexpr (n_outputs > 0) {
            return do_allocate(graph_in, n_outputs, io.outputs().data());
        } else {
            return GraphStatus::Success;
        }
    }

    // @0x1399be0 (<1,1>): mov al,1 —— 恒真。
    virtual bool is_valid() const noexcept override { return true; }

    virtual void enumerate_blocks(MemBlockEnumerator &en, bool is_input) const override
    {
        // Op 的模板方法:
        enumerate_op_blocks(en, io.inputs(), io.outputs(), is_input);
    }

    virtual void serialize(SerOpsInterface &sctx) const override
    {
        sctx.op_serialize_func(this, n_inputs, io.inputs().data(), n_outputs, io.outputs().data(), 0, 0);
    }
};

// .so 中显式实例化的 9 组 (weak 符号; 本镜像以隐式实例化等价覆盖)
extern template class TypicalOpIoBase<0, 1>;
extern template class TypicalOpIoBase<1, 0>;
extern template class TypicalOpIoBase<1, 1>;
extern template class TypicalOpIoBase<1, 2>;
extern template class TypicalOpIoBase<1, 3>;
extern template class TypicalOpIoBase<1, 4>;
extern template class TypicalOpIoBase<1, 5>;
extern template class TypicalOpIoBase<1, 6>;
extern template class TypicalOpIoBase<2, 1>;

} // namespace hnnx
