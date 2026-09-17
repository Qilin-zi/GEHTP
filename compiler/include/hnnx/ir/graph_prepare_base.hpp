#pragma once
// ============================================================================
// ::GraphPrepare —— M32 最小面 (OpIoPtrs 直接触及的成员; 完整类待 M36)。
//
// 证据 (libHtpPrepare.so):
//   * GraphPrepare : Graph 公开继承 —— OpIoPtrs(Graph&,…) ctor @0xf851d0 与
//     5 参 (Graph&,…) ctor @0x131be70 均先做引用下行转换
//     __dynamic_cast(&g, _ZTI5Graph[GOT 0x623fae0], _ZTI12GraphPrepare
//     [GOT 0x623f0a0], 0), 空结果 → __cxa_bad_cast;
//   * get_tensor(OpId, unsigned long) const —— _ZNK12GraphPrepare10get_tensorEym
//     @0xf84e10 (OpIoPtrs(GraphPrepare,OpDef const*) ctor @0xf84a10 直调);
//   * collect_multi_outputdef(OpDef const*, vector<OutputDef const*>&,
//     minObj::hashmap<u64,pair<u64,unsigned>,false,findhash<u64>>*,
//     vector<__map_const_iterator<…__value_type<u64,OpDefPtr>…>>*) const ——
//     _ZNK12GraphPrepare23collect_multi_outputdefE… @0xf7adb0;
//     (后两参的 libc++ 内部模板类型在 libstdc++ 下不可拼写, M32 以 void* 为
//     ABI 边界并在调用点保持实参语义: 第 3 参恒 &opid_alias_map(@+0x6db8),
//     第 4 参恒 nullptr @0xf84bd3 —— M36 以真实类型替换)
//   * set_self_slicing_num_slices(Op const*, unsigned) —— @0xf85240,
//     OpIoPtrs::set_op_slicing @0xf85230 的尾跳目标
//     (体内先调 Graph::get_extra_info(@plt 0x6f3240) 再写位域);
//   * get_full_allocator —— _ZNK12GraphPrepare18get_full_allocatorEv @0xf849e0:
//     mov 0x1d8(%rdi) → __dynamic_cast(ptr, _ZTIN2fa16RuntimeAllocator
//     [GOT 0x623f068], _ZTIN2fa14FancyAllocator [GOT 0x623efe0], 0);
//     空 → __cxa_bad_cast。即:
//       return dynamic_cast<fa::FancyAllocator&>(
//           *static_cast<fa::RuntimeAllocator*>(Graph::allocator /*+0x1d8*/));
//     (Graph 基类同槽 @0xd35140 则恒抛 std::runtime_error("wide crouton
//     not supported") —— 16B 异常, 串 @0x39a942a)
//   * opid_alias_map @+0x6db8 —— Graph 0x6C00 之外的 GraphPrepare 扩展区;
//     元素 24B {key u64@0; val.pair u64@8; val.u32@0x10}, 末哨兵 @+0x6de0,
//     向量基址 @+0x6dd8 (OpIoPtrs ctor 直读; hashmap::find @0xf925d0)。
// ============================================================================

#include "hnnx/ir/graph.hpp"

#include <vector>

class OpDef;
struct OutputDef;
class Tensor;
class Op;

namespace fa {
struct FancyAllocator;
struct RuntimeAllocator;
} // namespace fa

typedef unsigned long long OpId; // 与 op.hpp:47 同一全局 typedef (相容重声明)

class GraphPrepare : public Graph {
  protected:
    // 构造占位: GraphPrepare : Graph 必经 Graph 主构造建立基类 (59 槽 vptr/
    //   +0x1d8 allocator 等); .so 中 GraphPrepare 自有构造的精确签名/体内
    //   初始化 (含 +0x6db8 别名表) 属 M36 —— 此处仅给出同形转发, 无自有逻辑。
    GraphPrepare(HexagonNNEnv &env, unsigned graph_id_in, Graph::alloc_sel sel, optionpair const *opts,
                 unsigned nopts)
        : Graph(env, graph_id_in, sel, opts, nopts)
    {
    }

  public:
    // ---- opid_alias_map (@+0x6db8) 查表边界 -------------------------------
    // .so 行为: hashmap::find(src) 命中 → get_tensor(elem.val@+8, elem.u32@+0x10)。
    // M32 以成对接口暴露 (M36 以真实 minObj::hashmap 成员替换)。
    struct opid_alias_entry {
        OpId target_id; // 元素 val 的 u64 @+8
        unsigned output_idx; // 元素 val 的 u32 @+0x10
    };
    bool find_opid_alias(OpId src, opid_alias_entry &out) const;
    // 取 [this+0x6db8] 的不透明地址 (collect_multi_outputdef 第 3 实参)。
    void *opid_alias_map_raw() const;

    // ---- OpIoPtrs 直接调用的成员 ------------------------------------------
    Tensor const *get_tensor(OpId op_id, unsigned long output_idx) const; // @0xf84e10

    GraphStatus collect_multi_outputdef(OpDef const *def, std::vector<OutputDef const *> &out_defs,
                                        void *alias_map /* =opid_alias_map_raw() */,
                                        void *iter_vec /* =nullptr */) const; // @0xf7adb0

    void set_self_slicing_num_slices(Op const *op, unsigned n_slices); // @0xf85240

    // +0x1c0 槽的 GraphPrepare 覆写 (@0xf849e0 → 见文件头注)。
    //   体需要 fa::RuntimeAllocator/fa::FancyAllocator 完整类型 (两级 cast),
    //   而 fa 族完整定义在 M35 —— 此处仅声明, 镜像体在测试/库侧提供:
    //     return dynamic_cast<fa::FancyAllocator&>(
    //         *static_cast<fa::RuntimeAllocator*>(this->allocator));
    fa::FancyAllocator &get_full_allocator() const override;
};
