#pragma once
// ============================================================================
// OpDef 家族字节级重实现 (libHtpPrepare.so 2.48.40.260702, x86_64-linux-clang)
//
// 依据: audit_verify/asm/_ZN[K]5OpDef*、_ZN4hnnx1[1-5]OpDef_*、_ZN14OutputDefPatch*
//       共 20 个函数逐条反汇编解码；字段偏移经调用方交叉验证:
//         GraphPrepare::get_split_history @0xf7e3e0   读 +0x10
//         GraphPrepare::truegraph_n_outputs @0xf881c0 读 +0x4c/+0x90
//         GraphPrepare::delete_opdef_if_no_refs @0xf881f0 读/写 +0x8 (bit0)
//       SDK 头 (qnn_ori_include/hnnx/sdk/core/op_def.h, interface_defs.h,
//       dtype_enum.h) 全部声明与反汇编字节布局吻合。
//
// 布局总表 (vptr = _ZTV… + 0x10, Itanium ABI):
//   OpDef            sizeof=0xA0
//     +0x00 vptr
//     +0x08 uint16 flags          (OpDefFlags 基类, 非空基类优化后置于 vptr 后)
//     +0x0A uint16 opstr_hashval
//     +0x0C pad(4)
//     +0x10 splithist_t splithist (u64)
//     +0x18 GraphPrepare* graphref
//     +0x20 OpId id (u64)
//     +0x28 opname_tag_t opstr    (注册表节点指针; 节点+0x10 = u16 hash)
//     +0x30 vector<OpRef> input_defs (begin/end/cap, 24B)
//     +0x48 OutputDef m_first_outputdef (0x50=80B)
//     +0x98 unique_ptr<vector<OutputDef>> m_rest_outputdefs
//   OpDef_ConstBase  sizeof=0xA8  +0xA0 mutable uint32 content_hash (+pad4)
//   OpDef_Const      sizeof=0xB0  +0xA8 unique_ptr<Tensor> const_data
//   OpDef_Shape      sizeof=0xA8  (无新增字段)
//   OutputDef        sizeof=0x50  rank@0(u32) dtype@4(u32) max_sizes[8]@8(u64×8)
//                               zero_offset@0x48(i32) stepsize@0x4C(float)
//   OutputDefPatch   sizeof=0x78  见下
//
// OpDef vtable 槽位 (vptr 相对):
//   +0x00 generate, +0x08 const_data_ptr, +0x10 const_data_len,
//   +0x18 get_tensor, +0x20 release_memory, +0x28 D1, +0x30 D0,
//   +0x38 find_content_hash (OpDef_ConstBase 纯虚)
// ============================================================================
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

class Tensor; // 全局作用域! 证据: _ZTV6Tensor / _ZTS6Tensor="6Tensor" /
              // 全部 _ZN6Tensor… 均无 N4hnnx…E 前缀 (M28 修正, 原误置于 hnnx 内)
class GraphPrepare; // 全局 (mangling: _ZN12GraphPrepare…; SDK op_def.h:174)
class OpDef;        // 全局 (mangling: _ZN5OpDef13exact_same_asERKS_; SDK op_def.h:174)
struct OutputDefPatch; // 全局 (mangling: _ZN5OpDef13set_outputdefERK14OutputDefPatch)
struct OpRef;       // 全局 (mangling: _Z…RK5OpRef; SDK op_def.h:174)
struct InputDef;    // 全局 (mangling: _ZN5Graph11append_nodeE…PK8InputDef)

namespace hnnx {
class Allocator; // hnnx::Allocator (PN4hnnx9AllocatorE)
}

// ---------------------------------------------------------------------------
// DType — 值表出自 SDK dtype_enum.h, 与 .so 中字面量 0xFE/0xFF 完全一致。
// 全局作用域 (SDK dtype_enum.h: PUSH_VISIBILITY 顶层; TensorSclrDTIL5DType4EE)。
// 注意方向 (曾被弄反, 已用三处代码语义钉死):
//   None  = 254 = 0xFE  空输出:  exact_same_as 两边均 None 时跳过 OutputDef
//                          比较 (0x10b8ea0: cmp $0xfe → 直接 true);
//                          make_output_exemplar 无来源且 dtype==None → 抛
//                          "no outputs"; truegraph_n_outputs 对 None 返回 0。
//   Multi = 255 = 0xFF  多输出:  compare_eq/compare_constbase 对 Multi 提前
//                          返回 (形状无意义); truegraph_n_outputs 对 Multi
//                          返回 zero_offset (输出数存于 zero_offset)。
// ---------------------------------------------------------------------------
enum class DType : uint32_t {
    UNKNOWN = 0,
    QUInt8 = 1,
    QUInt16 = 2,
    QInt16 = 3,
    Float32 = 4,
    Int32 = 5,
    QInt32 = 6,
    QInt8 = 7,
    Float16 = 8,
    Int64 = 9,
    BFloat16 = 10,
    None = 254,  // OpDef 空输出专用, 不对外
    Multi = 255, // OpDef 多输出专用, 不对外
};

constexpr uint32_t MAX_DIMENSIONS = 8; // interface_defs.h

// ---------------------------------------------------------------------------
// OutputDef — sizeof 0x50。
// 证据: set_outputdef @0x10ba2c0 五条 movups 共拷 0x50;
//       find_basic_hash @0x10b8430 哈希循环按 +0x50+i*8 取 max_sizes 低 32 位;
//       向量元素大小 80 = ((end-begin)>>4)*0xCCCC..CD mod 2^64 精确除法
//       (get_outputdef @0x10b8f3b: 5n·M = n(4·2^64+1) ≡ n, M=0xCCCCCCCCCCCCCCCD)。
// ---------------------------------------------------------------------------
struct OutputDef {
    uint32_t rank = 0;                      // +0x00
    DType dtype = DType::UNKNOWN;           // +0x04
    uint64_t max_sizes[MAX_DIMENSIONS] = {}; // +0x08 (size_t[8])
    int32_t zero_offset = 0;                // +0x48
    float stepsize = 0.0f;                  // +0x4C
};
#if defined(_MSC_VER)
static_assert(sizeof(OutputDef) == 0x50);
#endif

// InputDef — SDK interface_defs.h: "must be the same layout as struct input"
// (C API hexagon_nn_append_node 的输入描述)。
struct InputDef {
    uint32_t input_id = 0;   // +0x00
    uint32_t output_idx = 0; // +0x04
};
#if defined(_MSC_VER)
static_assert(sizeof(InputDef) == 8);
#endif

namespace hnnx {

// hnnx::compare_eq(OutputDef, OutputDef) @0x10b9000 — 精确复刻:
//   dtype 不等 → false; zero_offset 不等 → false; dtype==Multi → true;
//   stepsize (ucomiss, NaN≠NaN → false); rank; max_sizes 低 32 位逐维。
inline bool compare_eq(const ::OutputDef &a, const ::OutputDef &b)
{
    if (a.dtype != b.dtype) return false;
    if (a.zero_offset != b.zero_offset) return false;
    if (a.dtype == ::DType::Multi) return true; // 0x10b9012
    if (!(a.stepsize == b.stepsize)) return false; // NaN 情形为 false
    if (a.rank != b.rank) return false;
    for (uint32_t i = 0; i < a.rank; i++)
        if ((uint32_t)a.max_sizes[i] != (uint32_t)b.max_sizes[i]) return false; // 32 位比较 (mov eax)
    return true;
}

// ---------------------------------------------------------------------------
// splithist_t — splithist.h: 单个 u64; SPLITPER=1024。
// 证据: get_split_history @0xf7e3e0 读 OpDef+0x10。
// ---------------------------------------------------------------------------
struct splithist_t {
    uint64_t val = 0;
};

// ---------------------------------------------------------------------------
// opname_tag_t — 实为指向字符串注册表节点的指针 (string_registry2.h)。
// 本类反汇编只证明两件事: map_str 返回该指针; 节点+0x10 处是 u16 opname 哈希
// (OpDef_Const ctor @0x10b8a79: movzwl 0x10(%r13))。节点其余布局未从本族
// 函数确认, 故仅声明已证字段。
// ---------------------------------------------------------------------------
struct string_key_node {
    uint64_t _unverified_0x00 = 0; // 注册表节点头 (未从本类反汇编确认)
    uint64_t _unverified_0x08 = 0;
    uint16_t hashval16 = 0; // +0x10: opname_hash = h*0x381+byte (u16 截断)
};
using opname_tag_t = string_key_node *;

// 进程级注册表 (弱符号, 由 string registry 模块提供; .so 中为
// _ZN4hnnx12string_tag_t7map_strEPKc @plt)
opname_tag_t string_tag_map_str(const char *s);

} // namespace hnnx

// ---------------------------------------------------------------------------
// OpRef — 单 u64 input_id (向量步长 8: exact_same_as @0x10b8dca shr $3)。
// 全局作用域 (mangling: _Z…RK5OpRef; SDK op_def.h:174)。
// ---------------------------------------------------------------------------
struct OpRef {
    uint64_t input_id = 0;
};

inline uint16_t find_opname_hash(std::string_view sv)
{ // opname_tag.h: opname_hash_impl — h*0x381+byte, &0xFFFF (16 位环绕)
    uint32_t h = 0;
    for (char c : sv) h = h * 0x381u + (uint8_t)c;
    return (uint16_t)h;
}

// ---------------------------------------------------------------------------
// OpDefFlags — 全局作用域 (SDK op_def.h:47; OpDef 为全局 5OpDef)。
// OpDef 的非空基类, 恰好落在 OpDef+0x08 (flags) 与 +0x0A (hashval)。
// flag_init @0xf90ef0 (于 make_output_exemplar 内联展开, esi=n_in, edx=n_out):
//   取 "::" 后首字符 (无 "::" 则取首字符; 空或仅 "::" → 0), 然后
//   '#'→const; n_out==0→retain|volatile; n_in==0||'*'→volatile;
//   "$Out"→retain|dummy_out; 否则 0。
// ---------------------------------------------------------------------------
class OpDefFlags {
public:
    constexpr static uint32_t BIT_deleted = 1;        // 已删, 待移除
    constexpr static uint32_t BIT_hidden = 2;         // 已被替换/隐藏
    constexpr static uint32_t BIT_const = 4;          // OpDef_ConstBase 或常量输出
    constexpr static uint32_t BIT_volatile = 8;       // 输入不变输出也变
    constexpr static uint32_t BIT_retain = 16;        // 不做死代码删除
    constexpr static uint32_t BIT_dummy_out = 32;     // $Out 节点
    constexpr static uint32_t BIT_constbase = 64;     // 当且仅当 OpDef_ConstBase
    constexpr static uint32_t BIT_in_constmap = 128;  // 位于 graph.const_map
    constexpr static uint32_t BIT_fake_unsigned = 256;
    constexpr static uint32_t BIT_custom_op = 512;
    constexpr static uint32_t BIT_mux_condition = 1024;

protected:
    uint16_t opstr_hashval = 0; // +0x0A (相对 OpDef)

public:
    uint16_t flags = 0; // +0x08 (相对 OpDef)

    static int flag_init(std::string_view opstr, int n_in, int n_out)
    {
        char c0;
        auto found = opstr.find("::");
        if (found != std::string_view::npos && !(opstr.substr(found + 2).empty())) {
            c0 = opstr[found + 2];
        } else if (found == std::string_view::npos && opstr.size() > 0) {
            c0 = opstr[0];
        } else {
            return 0;
        }
        return (c0 == '#') ? (int)BIT_const
               : (n_out == 0) ? (int)(BIT_retain | BIT_volatile)
               : (n_in == 0 || c0 == '*') ? (int)BIT_volatile
               : (opstr == "$Out") ? (int)(BIT_retain | BIT_dummy_out)
                                   : 0;
    }
};

// ---------------------------------------------------------------------------
// 外部依赖 (由 graph/tensor 层提供, 此处只声明; 均已在 .so 中定位):
// ---------------------------------------------------------------------------
namespace hnnx {
struct OpIoPtrs; // hnnx::OpIoPtrs (mangling: N4hnnx8OpIoPtrs)
} // namespace hnnx

class Op;    // 全局 (mangling: PK2Op / _ZN6TensorC2EPK2Op)

namespace hnnx {
struct uptr_Op { // 16B: {Op* op @0; 删除器字 @8}  (generate sret)
    // Op::clone 反汇编 (@0x10bce03/0x10bcf07): 重置时读字 @8, ==0 才经虚槽 +0x30 (D0) 析构
    //   → 0 = 归属; 非 0 = 借用 (与 hnnx::uptr_Tensor 同布局, 见 op.hpp)
    ::Op *op = nullptr; // :: 限定 —— types.hpp (GCP 族) 另有占位 hnnx::Op, 勿被遮蔽
    unsigned char dw = 0;
    // M32: reset() 语义即 Op::clone 异常着陆垫 (@0x10bcf0b: cmpb dw; D0 析构)
    void reset()
    {
        ::Op *const p = op;
        op = nullptr;
        if (p != nullptr && dw == 0) delete p; // delete → Op 虚槽 +0x30 (D0)
    }
};
uptr_Op op_factory_generate(const OpIoPtrs &, uint64_t id); // @plt 0x6edb50

// Tensor 侧被 OpDef 族调用的虚槽 (Tensor vptr 相对):
//   +0x18 删除析构(D0)   +0x38 find_content_hash   +0x68 set_dims(max_sizes*)
//   +0x78 const_data_ptr +0x80 set_data_ptr        +0x90 data_len
//   +0xc8 tensor_compare(this, other) → 3 路 int
struct TensorVtableOps { // 仅描述 OpDef 视角, 供重实现挂接 (无 .so 对应符号)
    uint32_t (*find_content_hash)(const ::Tensor *) = nullptr;
    const uint8_t *(*const_data_ptr)(const ::Tensor *) = nullptr;
    size_t (*data_len)(const ::Tensor *) = nullptr;
    int (*tensor_compare)(const ::Tensor *, const ::Tensor *) = nullptr;
};
} // namespace hnnx

void qnndsp_log(uint32_t level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
// 全局 C 风格符号 qnndsp_log @0xd4b2a0 (无 _ZN 前缀)

// ---------------------------------------------------------------------------
// OpDef — sizeof 0xA0; 全局作用域 (mangling: _ZN5OpDef…; SDK op_def.h:252)
// ---------------------------------------------------------------------------
class OpDef : public OpDefFlags {
protected:
    hnnx::splithist_t splithist;                              // +0x10
    GraphPrepare *graphref;                                   // +0x18
    class ForConst {}; // ConstBase 专用构造标记

    OpDef(GraphPrepare &graph_in, uint64_t my_id_in, hnnx::opname_tag_t opstr_in, const OutputDef &odef,
          ForConst)
        : OpDefFlags(), splithist(), graphref(&graph_in), id(my_id_in), opstr(opstr_in), input_defs(),
          m_first_outputdef(odef)
    { // flags = const|constbase —— OpDef_Const ctor @0x10b8a73 直接 movw $0x44
        flags = BIT_const | BIT_constbase;
        opstr_hashval = opstr_in ? opstr_in->hashval16 : 0;
    }

public:
    uint64_t id = 0;                                         // +0x20
    hnnx::opname_tag_t opstr = nullptr;                            // +0x28
    std::vector<OpRef> input_defs;                           // +0x30
protected:
    OutputDef m_first_outputdef;                             // +0x48
    std::unique_ptr<std::vector<OutputDef>> m_rest_outputdefs; // +0x98

public:
    // ---- 主构造 (OpDef ctor, 在各调用点内联; 字段序见 OpDef_Const ctor
    //      @0x10b880a-0x10b8878 的基类段) ----
    OpDef(GraphPrepare &graph_in, uint64_t my_id_in, hnnx::opname_tag_t opstr_in, std::vector<OpRef> &&input_defs_in,
          const OutputDef *output_defs_in, size_t num_quant_params)
        : splithist(), graphref(&graph_in), id(my_id_in), opstr(opstr_in), input_defs(std::move(input_defs_in)),
          m_first_outputdef(), m_rest_outputdefs(nullptr)
    {
        flags = (uint16_t)flag_init(opstr_name(opstr_in), (int)input_defs.size(), output_defs_in ? 1 : 0);
        opstr_hashval = opstr_in ? opstr_in->hashval16 : 0;
        if (output_defs_in == nullptr) {
            m_first_outputdef.dtype = DType::None;
            return;
        }
        m_first_outputdef = *output_defs_in;
        if (num_quant_params <= 1) return;
        m_rest_outputdefs = std::make_unique<std::vector<OutputDef>>(output_defs_in + 1,
                                                                    output_defs_in + num_quant_params);
    }
    OpDef(GraphPrepare &g, uint64_t id_, hnnx::opname_tag_t t, std::vector<OpRef> &&in, const OutputDef *out,
          size_t nqp, hnnx::splithist_t sl)
        : OpDef(g, id_, t, std::move(in), out, nqp)
    {
        splithist = sl;
    }
    OpDef(const OpDef &) = delete;
    OpDef(OpDef &&) = default;
    OpDef &operator=(const OpDef &) = delete;
    OpDef &operator=(OpDef &&) = delete;
    virtual ~OpDef() = default; // 基类析构: 删 m_rest 与 input_defs (D1 @0x10b8bac-)

    GraphPrepare &graph() const { return *graphref; }

    // ---- splithist 存取 ----
    hnnx::splithist_t get_splithist() const { return splithist; }
    void set_splithist(hnnx::splithist_t v) { splithist = v; }
    void set_splithist(const OpDef &other) { splithist = other.splithist; }

    void change_opstr_internal(hnnx::opname_tag_t new_opstr)
    {
        opstr = new_opstr;
        opstr_hashval = new_opstr->hashval16;
    }

    // ---- 输出访问 ----
    // get_outputdef(unsigned) @0x10b8f30:
    //   j==0 → &m_first; 否则 j 必须 < 1+m_rest->size() 且 m_rest 非空,
    //   越界抛 "quant_index must be < num_quant_params" (0x469684e);
    //   返回 &(*m_rest)[j-1]。
    const OutputDef &get_outputdef(unsigned quant_index) const
    {
        if (quant_index == 0) return m_first_outputdef;
        if (m_rest_outputdefs == nullptr || quant_index >= 1 + m_rest_outputdefs->size())
            throw std::runtime_error("quant_index must be < num_quant_params");
        return (*m_rest_outputdefs)[quant_index - 1];
    }
    template <bool ValidateSingleQuant = true> const OutputDef &get_outputdef() const
    {
        if constexpr (ValidateSingleQuant) conditionally_validate_single_quant();
        return m_first_outputdef;
    }

    // get_outputdefs @0x10b99b0: 有 m_rest → reserve(n+1); push first; 追加
    //   rest 全部; 无 m_rest → 单元素 {m_first} (分配 0x50 + memcpy)。
    std::vector<OutputDef> get_outputdefs() const
    {
        if (m_rest_outputdefs) {
            std::vector<OutputDef> ret;
            ret.reserve(m_rest_outputdefs->size() + 1);
            ret.push_back(m_first_outputdef);
            ret.insert(ret.end(), m_rest_outputdefs->begin(), m_rest_outputdefs->end());
            return ret;
        }
        return std::vector<OutputDef>(1, m_first_outputdef);
    }

    // set_outputdef(const OutputDef&) @0x10ba2c0: m_rest 非空抛
    //   "Cannot set outputdef with multiple quant params" (0x46968a9);
    //   先校验 single quant, 再整块 0x50 覆盖 m_first。
    void set_outputdef(const OutputDef &v)
    {
        if (m_rest_outputdefs) throw std::runtime_error("Cannot set outputdef with multiple quant params");
        conditionally_validate_single_quant();
        m_first_outputdef = v;
    }

    // set_outputdef(const OutputDefPatch&) @0x10ba170 — 逐字段条件覆盖:
    //   rank_valid→rank; dtype_valid→dtype;
    //   per_dim_mode(+0x60)→仅覆盖各 set_dim[i]; 否则 all_dims(+0x50)→
    //   memcpy 0x40 整组 max_sizes; stepsize_valid→stepsize; zo_valid→zero_offset。
    void set_outputdef(const OutputDefPatch &p);

    // set_outputdefs @0x10b9e70: 元素数必须 == num_quant_params, 否则抛
    //   "Count of outputdefs does not match number of quants" (0x4696875);
    //   v[0]→m_first, 其余 assign 进 m_rest。
    void set_outputdefs(const std::vector<OutputDef> &v)
    {
        if (m_rest_outputdefs) {
            if (v.size() != 1 + m_rest_outputdefs->size())
                throw std::runtime_error("Count of outputdefs does not match number of quants");
            m_first_outputdef = v[0];
            m_rest_outputdefs->assign(v.begin() + 1, v.end());
        } else {
            if (v.size() != 1) throw std::runtime_error("Count of outputdefs does not match number of quants");
            m_first_outputdef = v[0];
        }
    }

    // conditionally_validate_single_quant @0x10ba360:
    //   graph->multi_quant_transformed_to_single_quant() 为真且 m_rest 非空时:
    //   qnndsp_log(0, "%s:504::ERROR:OpDef 0x%016llX has multiple quant params\n",
    //              "op_def.cc", id), 抛
    //   "Expectation: All multi-quants ops should have been transformed to single-quant ops"
    void conditionally_validate_single_quant() const;

    size_t n_inputs() const { return input_defs.size(); }
    // truegraph_n_outputs @0xf881c0 交叉验证的语义:
    //   dtype==Multi → n_outputs = zero_offset; dtype==None → 0; 否则 1。
    size_t n_outputs() const { return get_outputdef<false>().dtype == DType::None ? 0 : 1; }
    bool has_outputs() const { return !(get_outputdef<false>().dtype == DType::None); }
    bool is_graph_sink() const { return !has_outputs(); }
    bool has_multiple_outputs() const { return get_outputdef<false>().dtype == DType::Multi; }
    OpRef reference() const { return OpRef{id}; }
    bool has_multiple_quant_params() const { return m_rest_outputdefs != nullptr; }
    size_t num_quant_params() const { return 1 + (has_multiple_quant_params() ? m_rest_outputdefs->size() : 0); }

    // make_output_exemplar @0x10b8110:
    //   m_rest 非空 → 抛 "Cannot make output exemplar with multiple quant params"(0x4696788);
    //   (!size_from||!outp_from) && dtype==None → 抛 "no outputs"(0x46967bf);
    //   tmp=m_first; size_from→rank+max_sizes[8]; outp_from→dtype/zo/stepsize;
    //   opstr = map_str("_")(0x39ba716); flags = flag_init("_",0,1) = volatile;
    //   splithist 继承自 this; graphref/id 复制; input_defs={}; m_rest=null。
    OpDef make_output_exemplar(const OutputDef *size_from, const OutputDef *outp_from) const
    {
        if (m_rest_outputdefs)
            throw std::runtime_error("Cannot make output exemplar with multiple quant params");
        if ((!size_from || !outp_from) && m_first_outputdef.dtype == DType::None)
            throw std::runtime_error("no outputs");
        conditionally_validate_single_quant();
        OutputDef tmp = m_first_outputdef;
        if (size_from) {
            tmp.rank = size_from->rank;
            memcpy(tmp.max_sizes, size_from->max_sizes, sizeof(tmp.max_sizes));
        }
        if (outp_from) {
            tmp.dtype = outp_from->dtype;
            tmp.zero_offset = outp_from->zero_offset;
            tmp.stepsize = outp_from->stepsize;
        }
        hnnx::opname_tag_t tag = hnnx::string_tag_map_str("_");
        OpDef ret(*graphref, id, tag, std::vector<OpRef>(), &tmp, 1, splithist);
        return ret;
    }
    OpDef make_output_exemplar(const OutputDef *from) const { return make_output_exemplar(from, from); }
    OpDef make_output_exemplar() const { return make_output_exemplar(nullptr, nullptr); }

    // ---- 虚函数 (vtable 槽位序) ----
    virtual hnnx::uptr_Op generate(const hnnx::OpIoPtrs &io) const
    { // @0x10b8c40: return hnnx::op_factory_generate(io, this->id);
        return hnnx::op_factory_generate(io, id);
    }
    virtual const uint8_t *const_data_ptr() const { return nullptr; } // 默认 @0x10ba4b0: xor eax,eax;ret
    virtual size_t const_data_len() const { return 0; }               // 默认 @0x10ba4c0
    virtual const Tensor *get_tensor() const { return nullptr; } // 默认 @0x10ba4d0 (xor eax,eax;ret)
    virtual void release_memory() {}                                  // 默认 @0x10ba4e0: ret

    // ---- 比较 ----
    static bool compare_less(const OpDef &lhs, const OpDef &rhs); // @0x10b9060
    static bool compare_eq(const OpDef &lhs, const OpDef &rhs);   // @0x10b9480
    bool exact_same_as(const OpDef &rhs) const;                   // @0x10b8da0

    static const char *opstr_name(hnnx::opname_tag_t t) { return t ? (const char *)t->_unverified_0x00 : ""; }
};

// ---------------------------------------------------------------------------
// exact_same_as @0x10b8da0:
//   opstr 指针不等 → false; 输入数不等 → false; 任一 OpRef 不等 → false;
//   要求两边 dtype 的 None 性一致 (0x10b8e17: setne/sete 对偶, 两边均 None 或
//   均非 None 才继续); num_quant_params 不等 → false;
//   dtype==None → true (跳过全部 OutputDef 比较);
//   否则逐 quant: get_outputdef(j) 比较 rank/dtype/zero_offset/stepsize
//   (ucomiss, NaN→false) 与 max_sizes 全 64 位 (mov rsi, 与 compare_eq 的
//   32 位不同!)。
// ---------------------------------------------------------------------------
inline bool OpDef::exact_same_as(const OpDef &rhs) const
{
    if (opstr != rhs.opstr) return false;
    if (input_defs.size() != rhs.input_defs.size()) return false;
    for (size_t i = 0; i < input_defs.size(); i++)
        if (input_defs[i].input_id != rhs.input_defs[i].input_id) return false;
    const bool a_none = m_first_outputdef.dtype != DType::None;
    const bool b_none = rhs.m_first_outputdef.dtype == DType::None;
    if (a_none == b_none) return false; // None 性不一致
    const unsigned nq = (unsigned)num_quant_params();
    if (nq != rhs.num_quant_params()) return false;
    if (m_first_outputdef.dtype == DType::None) return true; // 0x10b8ea5
    for (unsigned j = 0; j < nq; j++) {
        const OutputDef &pa = get_outputdef(j);
        const OutputDef &pb = rhs.get_outputdef(j);
        if (pa.rank != pb.rank || pa.dtype != pb.dtype) return false;
        if (pa.zero_offset != pb.zero_offset) return false;
        if (!(pa.stepsize == pb.stepsize)) return false;
        for (uint32_t d = 0; d < pa.rank; d++)
            if (pa.max_sizes[d] != pb.max_sizes[d]) return false; // 全 64 位
    }
    return true;
}

// ---------------------------------------------------------------------------
// compare_less @0x10b9060 — CSE 序 (严格弱序):
//   同一对象 → false;
//   a 常量(4) 且 b 非常量 → true; a 非常量且 b 常量 → false;
//   双常量: a constbase(0x40) 且 b 非 → true; 反之 false;
//          双 constbase → compare_constbase(a,b) < 0 (shr $0x1f);
//   否则: opstr 指针序 (setb)。
// ---------------------------------------------------------------------------
namespace hnnx { // hnnx::compare_constbase (mangling: _ZN4hnnx17compare_constbase…)
int compare_constbase(const class OpDef_ConstBase &lhs, const class OpDef_ConstBase &rhs);
} // namespace hnnx

inline bool OpDef::compare_less(const OpDef &lhs, const OpDef &rhs)
{
    if (&lhs == &rhs) return false;
    const bool a_const = lhs.flags & BIT_const, b_const = rhs.flags & BIT_const;
    if (a_const) {
        if (!b_const) return true;
        const bool a_cb = lhs.flags & BIT_constbase, b_cb = rhs.flags & BIT_constbase;
        if (a_cb) {
            if (!b_cb) return true;
            return hnnx::compare_constbase((const hnnx::OpDef_ConstBase &)lhs,
                                           (const hnnx::OpDef_ConstBase &)rhs) < 0;
        }
        if (b_cb) return false;
    } else if (b_const) {
        return false;
    }
    return lhs.opstr < rhs.opstr; // 指针比较 @0x10b908f setb
}

// ---------------------------------------------------------------------------
// OpDef_ConstBase — +0xA0 content_hash (mutable)。
// hnnx:: (mangling: _ZN4hnnx15OpDef_ConstBase… / _ZNK4hnnx11OpDef_Shape…; SDK op_def.h:427 起)
// ---------------------------------------------------------------------------
namespace hnnx {

class OpDef_ConstBase : public OpDef {
protected:
    OpDef_ConstBase(::GraphPrepare &g, uint64_t id_, opname_tag_t t, const ::OutputDef &od)
        : ::OpDef(g, id_, t, od, ForConst{})
    {
    }

public:
    mutable uint32_t content_hash = 0; // +0xA0; 0 表示未算 (哈希永不为 0, 为 0 时存 1)

    uint32_t get_content_hash() const { return content_hash == 0 ? get_content_hash_func() : content_hash; }
    void invalidate_content_hash() { content_hash = 0; }
    bool has_content_hash() const { return content_hash != 0; }

    // get_content_hash_func @0x10b8400:
    //   h = find_content_hash(); content_hash = h ? h : 1; return content_hash;
    uint32_t get_content_hash_func() const
    {
        uint32_t h = find_content_hash();
        content_hash = h ? h : 1;
        return content_hash;
    }

protected:
    // find_basic_hash @0x10b8430 (OpDef_Shape::find_content_hash @0x10b86b0
    // 与之逐指令相同):
    //   validate; n=min(rank,8); h=(opstr_hashval<<16)|n;
    //   for i<n: h = h*0x103011 ^ (u32)max_sizes[i];
    //   step_bits = memcpy(&u32, &stepsize, 4);
    //   return (((zero_offset*0x41201) ^ (step_bits*2)) * 0x104411)
    //          ^ (dtype*0x501239) ^ h;
    uint32_t find_basic_hash() const noexcept
    {
        const ::OutputDef &od = m_first_outputdef;
        uint32_t n = od.rank < 8 ? od.rank : 8;
        uint32_t h = ((uint32_t)opstr_hashval << 16) | n;
        for (uint32_t i = 0; i < n; i++) h = h * 0x103011u ^ (uint32_t)od.max_sizes[i];
        uint32_t step_bits;
        memcpy(&step_bits, &od.stepsize, 4);
        return (((uint32_t)od.zero_offset * 0x41201u) ^ (step_bits * 2u)) * 0x104411u
               ^ ((uint32_t)od.dtype * 0x501239u) ^ h;
    }
    virtual uint32_t find_content_hash() const noexcept = 0;
};

// ---------------------------------------------------------------------------
// compare_constbase @0x10b9260 — 3 路比较 (返回 -1/0/+1):
//   ① 备忘哈希 (双方 content_hash==0 则 vcall+0x38 求值并按"永不为0"规则存);
//      不等 → 哈希序 (setae: >= → +1)。
//   ② dynamic_cast<const OpDef_Const*>: a 是 Const 而 b 不是 → +1, 反之 -1
//      (Const 因带 tensor 数据排后)。
//   ③ opstr 节点指针不等 → 指针序。
//   ④ 深比较 (双方 validate): dtype 不等 → 有符号序 (setge);
//      zero_offset 不等 → 有符号序; dtype==Multi → 跳到 ⑥;
//      stepsize (ucomiss 双跳: 不等时 b<=a → +1 否则 -1);
//      rank 不等 → 无符号序 (setae);
//      max_sizes[i] 不等 → 低 32 位有符号序 (setge, cmpl)。
//   ⑤ 全部 OutputDef 相等且非 Const → 0。
//   ⑥ 双 Const: ta=get_tensor(), tb=…; 任一空 → this 指针序 tie-break;
//      typeinfo->name 指针不等 → 其序; 相等 → tail-call ta 的 vtable+0xc8
//      (tensor_compare)。
// ---------------------------------------------------------------------------
int compare_constbase_impl(const OpDef_ConstBase &lhs, const OpDef_ConstBase &rhs);

inline int compare_constbase(const OpDef_ConstBase &lhs, const OpDef_ConstBase &rhs)
{
    return compare_constbase_impl(lhs, rhs);
}

// compare_constbase_eq @0x10b9920:
//   双方备忘哈希相等 && compare_constbase(...)==0
inline bool compare_constbase_eq(const OpDef_ConstBase &lhs, const OpDef_ConstBase &rhs)
{
    uint32_t ha = lhs.content_hash ? lhs.content_hash : lhs.get_content_hash_func();
    uint32_t hb = rhs.content_hash ? rhs.content_hash : rhs.get_content_hash_func();
    if (ha != hb) return false;
    return compare_constbase(lhs, rhs) == 0;
}

// ---------------------------------------------------------------------------
// OpDef_Const — +0xA8 const_data; opstr = "$Const" (0x398c486)。
// ---------------------------------------------------------------------------
class OpDef_Const : public OpDef_ConstBase {
public:
    std::unique_ptr<Tensor> const_data; // +0xA8

    // ---- ctor(unique_ptr<Tensor>&&) @0x10b8a30 ----
    //   tag=map_str("$Const"); od = tensor->gen_output_def();
    //   flags=0x44(const|constbase); m_first=od; 其余同基类; content_hash=0;
    //   const_data = tensor.release()。
    OpDef_Const(::GraphPrepare &graph_in, uint64_t my_id_in, std::unique_ptr<::Tensor> tensor_in);

    // ---- ctor(const ::OutputDef&, const uint8_t*, size_t) @0x10b87d0 ----
    //   基类同上 (m_first=od); 然后:
    //   datalen!=0 且 od.rank==0 (标量) →
    //       tmp = tensor_generator_scalar(nullptr, od, data, datalen);
    //   失败或未走 → tmp = graph 内部工厂 (@0xd129e0, 未导出);
    //       成功则 tmp->vtable+0x80(data), tmp->vtable+0x68(&od.max_sizes),
    //       persistent_clone(&tmp, graph->allocator(+0x1d8), false), 删临时。
    //   彻底失败: qnndsp_log(0,"%s:146::ERROR:OpDef_Const failed to generate
    //       tensor\n","op_def.cc",""), const_data 留空 (不抛)。
    OpDef_Const(::GraphPrepare &graph_in, uint64_t my_id_in, const ::OutputDef &output_def, const uint8_t *data_in,
                size_t len);

    ~OpDef_Const() override; // D1 @0x10b8b30
    const uint8_t *const_data_ptr() const override; // @0x10b8d20: const_data? vcall+0x78 : nullptr
    size_t const_data_len() const override;         // @0x10b8d40: const_data->vcall+0x90 (无空检查!)
    uptr_Op generate(const OpIoPtrs &) const override; // @0x10b8c60: {new ConstWrapperOp(*graphref,id,this), false}
    const ::Tensor *get_tensor() const override { return const_data.get(); } // vtable+0x18 @0x10ba5f0
    void release_memory() override;                 // @0x10b8d50

protected:
    // find_content_hash @0x10b8550:
    //   basic = find_basic_hash();
    //   th = const_data->vcall+0x38 (无空检查);
    //   return ((th ^ 0x11111111) * 0x10991) ^ basic;
    uint32_t find_content_hash() const noexcept override;
};

// ---------------------------------------------------------------------------
// OpDef_Shape — opstr = "$Shape"; 无新字段 (sizeof 0xA8)。
// ---------------------------------------------------------------------------
class OpDef_Shape : public OpDef_ConstBase {
public:
    OpDef_Shape(::GraphPrepare &graph_in, uint64_t my_id_in, const ::OutputDef &output_def)
        : OpDef_ConstBase(graph_in, my_id_in, string_tag_map_str("$Shape"), output_def)
    {
    }
    const uint8_t *const_data_ptr() const override { return nullptr; }
    size_t const_data_len() const override { return 0; }
    uptr_Op generate(const OpIoPtrs &) const override; // @0x10b8cc0: {new ShapeWrapperOp(*graphref,id,this), false}

protected:
    // @0x10b86b0: 逐指令等于 find_basic_hash (仅基础哈希, 无 tensor 项)
    uint32_t find_content_hash() const noexcept override { return find_basic_hash(); }
};

} // namespace hnnx

// ---------------------------------------------------------------------------
// OutputDefPatch — sizeof 0x78; 全局作用域 (mangling: RK14OutputDefPatch;
// SDK op_def.h:201)。布局由 set_outputdef(Patch) @0x10ba170、
// new_dtype @0x10ba430、stepsize_zeroOffset @0x10ba3f0、set_dim_max_size
// @0x10ba470 四条反汇编联合钉死:
//   +0x00 u32 rank         +0x04 u8 rank_set
//   +0x08 DType dtype      +0x0C u8 dtype_set
//   +0x10 u64 max_sizes[8] (+0x40 字节)
//   +0x50 u8 max_sizes_present   (set_dim_max_size 首次触发时清零数组并置 1)
//   +0x51..0x57 pad
//   +0x58 u8 set_dim[8]
//   +0x60 u8 patch_maxsizes_partially
//   +0x64 i32 zero_offset  +0x68 u8 zero_offset_set
//   +0x6C f32 stepsize     +0x70 u8 stepsize_set
// ---------------------------------------------------------------------------
struct OutputDefPatch {
    uint32_t rank = 0;                  // +0x00
    bool rank_set = false;              // +0x04
    DType dtype = DType::UNKNOWN;       // +0x08
    bool dtype_set = false;             // +0x0C
    uint64_t max_sizes[MAX_DIMENSIONS] = {}; // +0x10
    bool max_sizes_present = false;     // +0x50
    uint8_t _pad_0x51[7] = {};          // +0x51..0x57 填充 (反汇编无任何读写)
    bool set_dim[MAX_DIMENSIONS] = {};  // +0x58
    bool patch_maxsizes_partially = false; // +0x60
    int32_t zero_offset = 0;            // +0x64
    bool zero_offset_set = false;       // +0x68
    float stepsize = 0.0f;              // +0x6C
    bool stepsize_set = false;          // +0x70

    // @0x10ba3f0: 其余字段全清 (含各 valid 位), 只置 stepsize/zero_offset
    static OutputDefPatch stepsize_zeroOffset(float stepsize, int32_t zero_offset)
    {
        OutputDefPatch p;
        p.stepsize = stepsize;
        p.stepsize_set = true;
        p.zero_offset = zero_offset;
        p.zero_offset_set = true;
        return p;
    }
    // @0x10ba430: 只置 dtype
    static OutputDefPatch new_dtype(DType d)
    {
        OutputDefPatch p;
        p.dtype = d;
        p.dtype_set = true;
        return p;
    }
    // @0x10ba470: partial=1; 若 max_sizes 尚未 present → 整组清零并置 present;
    //   随后 max_sizes[dim]=max_size, set_dim[dim]=1
    void set_dim_max_size(size_t dim, uint64_t max_size)
    {
        patch_maxsizes_partially = true;
        if (!max_sizes_present) {
            memset(max_sizes, 0, sizeof(max_sizes));
            max_sizes_present = true;
        }
        max_sizes[dim] = max_size;
        set_dim[dim] = true;
    }
};
#if defined(_MSC_VER)
static_assert(sizeof(OutputDefPatch) == 0x78);
#endif

// set_outputdef(Patch) 依据 @0x10ba170 的精确语义:
//   m_rest 非空 → 抛 (同上); 先 validate;
//   rank_set→rank; dtype_set→dtype;
//   patch_maxsizes_partially → 仅覆盖 set_dim[i] 的维度;
//   否则 max_sizes_present → memcpy 0x40 全组;
//   stepsize_set→stepsize (0x94); zero_offset_set→zero_offset (0x90)。
inline void OpDef::set_outputdef(const OutputDefPatch &p)
{
    if (m_rest_outputdefs) throw std::runtime_error("Cannot set outputdef with multiple quant params");
    conditionally_validate_single_quant();
    OutputDef &od = m_first_outputdef;
    if (p.rank_set) od.rank = p.rank;
    if (p.dtype_set) od.dtype = p.dtype;
    if (p.patch_maxsizes_partially) {
        for (uint32_t i = 0; i < MAX_DIMENSIONS; i++)
            if (p.set_dim[i]) od.max_sizes[i] = p.max_sizes[i];
    } else if (p.max_sizes_present) {
        memcpy(od.max_sizes, p.max_sizes, sizeof(od.max_sizes));
    }
    if (p.stepsize_set) od.stepsize = p.stepsize;
    if (p.zero_offset_set) od.zero_offset = p.zero_offset;
}
// (文件止; 原 namespace hnnx 全局包裹已按 mangling 证据拆分 —— 见 M29 修正清单)
