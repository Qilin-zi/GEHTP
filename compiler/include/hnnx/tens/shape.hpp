// ============================================================================
// Shape<Rank> — 全局作用域 (符号 _ZN5ShapeILm4EE… 直证), M34
//
// 字节级证据:
//   · 布局: hnnx::ShapeFlags@+0 (8B); dims[Rank]@+8; max_dims[Rank]@+8+8R;
//     pad[Rank](u8)@+8+16R —— Shape<4>::deserialize @0xd94110 构造点
//     (0xd94552-0xd9464d) 逐字段写入序直证; shplen() = 8 + 16R + R
//     (R=4 → 0x4c); crate 记录分配 align8(shplen) (R=4 → 0x50)。
//   · canonical_shape: 逐字 FNV —— h = w0·0x12401D1; h = (h+wₖ)·0x12401D1
//     (k=1..N-2); h += w_{N-1} (末字只加不乘!) + 逐秩盐; N = ⌈shplen/4⌉
//     (R=4 → 19 字, 恰无尾部填充参与; 其余秩末字含结构填充字节)。
//     盐 (六秩实测): R1 0x48D26249 / R2 0x18619B8A / R3 0x3179DF4B /
//     R4 0x8065DD8C / R5 0x0B93B1DD / R6 0xC2C91AEE。
//   · 规范化容器 (逐秩独立, Graph 内偏移 R→0x53b0+0x18(R-1), 步长 0x18):
//     libc++ __tree 布局容器 {begin@0, end.root@+8, size@+0x10} =
//     std::multimap<u32, Shape const*> (节点 new(0x30) {l,r,up,black@0x18,
//     key@0x20, value@0x28}); 7 处调用点偏移直证 (R1@0x12f92d2→+0x53b0 …
//     R6@0x12f8782→+0x5428)。插入助手 @0x12f9630 (七秩共享静态, 提示式
//     多重插入), 再平衡 @0x868c50 (libc++ __tree_balance_after_insert 逐
//     指令转录于 src/tens/shape.cpp)。
//   · OutputDef 重载: 栈建 Shape{flags=0, dims=max=max_sizes[0..R), pad=0}
//     (R1 @0x12f93fa 读 def+8; R5 @0x12f845a..0x12f8482 读 def+8..+0x28)
//     → 尾调用 canonical_shape。
//   · ShapeFlag 重载 @0x12f7f70: 整形栈拷贝 (0x4c) + flags16 覆盖
//     (movw %dx,(rsp)) → 尾调用 canonical_shape。
//   · deserialize 位格式 (0xd94110 全解):
//     首字 w0: (w0 & 0xffff0000) == 0xcccc0000 → flags16 = w0 & 0xffff,
//     控制字 cw = 读 w1; 否则 flags16 = 0, cw = w0。
//     每轴 g∈{0..Rank-1} 占 cw 位 4g: mode=(cw>>4g)&3, max覆盖字=cw&(4<<4g),
//     pad覆盖字=cw&(8<<4g):
//       mode 0 → dim=1, max=1, pad=0 (无字)
//       mode 1 → 一字 {dim:16, Δ:8, pad:8}; max = dim + Δ
//       mode 2 → 一字 {dim:24, pad:8}; max = dim
//       mode 3 → 一字全宽 dim = max, pad = 0
//     依序: [mode 字] → [max 覆盖字] → [pad 覆盖字]。
//     构造: Deserz DCrate 游标 (+0x20) 非 0 → align8 + shplen, 超 +0x28
//     上限 → 抛 dcrate_seg_overflow; 游标 0 → cratep(+0x30)::
//     add_record_slot(align8(shplen), 8)。flags 8B = flags16 | 0 (高位清零,
//     0xd94559-0xd94560 双存直证)。status≥0 时 crate 计数 [+0x40]++。
//     入口: deserialize_shared_obj_func 去重 (rdx==0 → 返回既有 rax;
//     否则新指针写入 *rdx = 共享表项)。
// ============================================================================
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "../ir/op_def.hpp"       // ::OutputDef (max_sizes[8]@+8)
#include "../ir/tensor_base.hpp"  // hnnx::ShapeFlags / hnnx::ShapeFlag

class Graph;
namespace hnnx {
class Deserz;
class Serializer;
} // namespace hnnx

template <size_t Rank> class Shape : public hnnx::ShapeFlags {
  public:
    std::array<size_t, Rank> dims;      // +0x08
    std::array<size_t, Rank> max_dims;  // +0x08 + 8·Rank
    std::array<uint8_t, Rank> pad;      // +0x08 + 16·Rank

    Shape() noexcept : dims{}, max_dims{}, pad{} {}
    explicit Shape(std::array<size_t, Rank> const &dims_in) noexcept : dims(dims_in), max_dims(dims_in), pad{} {}
    Shape(std::array<size_t, Rank> const &dims_in, std::array<size_t, Rank> const &max_in) noexcept
            : dims(dims_in), max_dims(max_in), pad{}
    {
    }
    Shape(Shape const &ref, hnnx::ShapeFlag const newflags) noexcept
            : hnnx::ShapeFlags(newflags), dims(ref.dims), max_dims(ref.max_dims), pad(ref.pad)
    {
    }

    // 序列化长度 (SDK shape.h 原文): flags 8 + dims 8R + max 8R + pad R
    unsigned shplen() const noexcept
    {
        return unsigned(sizeof(hnnx::ShapeFlags) + 16 * Rank + Rank);
    }

    // ---- 规范化 (Graph+0x53b0+0x18(R-1) multimap) ----
    static Shape const *canonical_shape(Graph &graph, Shape const &shp);
    static Shape const *canonical_shape(Graph &graph, Shape const &shp, hnnx::ShapeFlag flags);
    static Shape const *canonical_shape(Graph &graph, ::OutputDef const &def);
    // ---- crate 持久形 (不查重, 直接 crate 新建) ----
    static Shape const *crated_shape(Graph &graph, Shape const &shp);
    // ---- 反序列化 (shared-obj 去重; 见文件头位格式) ----
    static Shape const *deserialize(hnnx::Deserz &dctx, Shape const **slot);

    // ---- 序列化 (导出于 .so; 实现属序列化族 —— 待 M-序列化补全) ----
    void serialize(hnnx::Serializer &ser) const;
    // get_shape_info @0x12f7020 (R4) 返回类型未解码 —— 暂不声明
};

#if defined(_MSC_VER)
static_assert(sizeof(Shape<4>) == 0x50);
#endif // 8 对齐: shplen 0x4c → sizeof 80
#if defined(_MSC_VER)
static_assert(sizeof(hnnx::ShapeFlags) + 16 * 4 + 4 == 0x4c);
#endif // shplen 公式 (R4)
