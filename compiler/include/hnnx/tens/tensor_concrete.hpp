// ============================================================================
// LayoutTensor / ConcreteTensor — tensor_concrete.h 精确移植, M34-C3
//
// 字节级证据 (objdump 直证; 全量详见 audit_verify/reports/M20_LayoutTensor_disasm.md
// 与 M34 报告):
//   · 对象布局 (M20 §3.1): vptr@0x00 / Interface* @0x08 / Shape<Rank>* @0x10 /
//     blocktab-or-bulk_data@0x18; sizeof(LayoutTensor) = 0x20 (DCrate 张量槽
//     32B/align8, size_align_code 0x43 @0xd0215e-0xd021b2)。
//   · 三 noinline 构造:
//     主 @db5280 (Crouton_8): vptr → ifc=fn(graph,def) → 栈 Shape{flags=0,
//       dims=max_sizes, max_dims=TLayout::pad(max_sizes), pad=0} → canonical_shape
//       → nblocks = maxd0*(maxd1/8)*(maxd2/8)*(maxd3/32); 0 → blocktab=null;
//       否则 graph_crate → add_record_slot(nblocks*8, 8) → status≥0 时
//       ++crate[0x40] (db537b) → memset(slot,0,nblocks*8)。
//     deser @db5410: ifc=fn(dctx,&ifc) → Shape::deserialize(dctx,&shape) →
//       blocktab=null → nblocks = classic(dctx[0x9c]==0) ? num_blocks : 1 →
//       尾跳 Tensor::deserialize_blocktable_generic。
//     clone @db5590: ifc 直传 (rcx, 不调 factory) → shape 共享 (old[0x10] 原样) →
//       nblocks → graph_crate(alloc->graph /*+0x8*/) → add_record_slot(nblocks*8,8)
//       → status≥0 时 ++crate[0x40] (db5623) → memcpy(slot, old.blocktab, nblocks*8)
//       浅拷贝; clone_mode(r8) 全程未读。
//   · interface 工厂签名: 主构造 2 参 (Graph&, OutputDef const&) (db52ae:
//     mov %rcx,%rdi; mov %rdx,%rsi; call *%r8); deser 构造 2 参
//     (Deserz&, Interface const**) (@plt 0x6edb00 =
//     _ZN12LayoutTensorIN5Ldefs6Flat_8EEC2ERN4hnnx6DeserzEPFPK9InterfaceS5_PS8_E)。
//   · change_shapepad_impl @dc8f70: pad 字段 uint8 截断 (movb 只取低字节);
//     max_dims = layout.pad(pad_coords(dims, pad)) 上取整; realloc_blocktab;
//     shape = canonical_shape(新)。
//   · realloc_blocktab @dc90a0: nblocks > old_nblocks 且 shape 变化 → 重新
//     make_blocktab (已清零); 否则 memset(blocktab,0,nblocks*8)。
//   · compare_sametype_layout @db5be0: dims 字典序 → (is_padded) max_dims
//     字典序 → 逐块 memcmp(blocklen = block_total*sizeof)。
//   · allocate_layout @db5920: allocate_n(blocktab, nblocks, blocksize,
//     align = is_indirect ? blocksize : min(256,elbytes), mclass, options,
//     get_dtype()) 7 参; is_singular 时 options |= AllocOpts_packed。
//   · find_content_hash_layout @db5a80: build_hash(dims,Rank,hash_in) [无
//     Rank*0x102401 异或 —— 源内死代码, 编译器已消除] → (is_padded)
//     build_hash(max_dims) → mem.find_content_hash。
//   · contig 族 (M20 §5.11): element_addr 纯 row-major; find_content_hash =
//     content_hash_data(bulk,len,is_float) ^ mulu32_modular(h,0x223131);
//     compare = 单次 memcmp; nblocks=1; align=min(256,elbytes)。
//   · ConcreteTensor::find_content_hash (tensor_concrete.h:1022 原文):
//     interface().interface_hash() ^ mulu32_modular(unsigned(dtype),0x107301)
//     → find_content_hash_layout(h, dtype_traits::is_float)。
//   · vtable 槽位 = tensor_base.hpp 27 槽表 (vptr 相对), 全部 override 对齐。
// 类结构/命名: SDK qnn_ori_include/hnnx/sdk/core/tens/tensor_concrete.h 原文;
//   .so mangling: LayoutTensor/ConcreteTensor/layout_mem_* 为全局作用域
//   (_ZN12LayoutTensorIN5Ldefs9Crouton_8EE…, _ZTV14ConcreteTensorIN5Tdefs…),
//   indirect_layout_mem 在 hnnx 内。
// ============================================================================
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// SDK 通用宏 (各公共头): 内层强制内联
#define ALWAYSINLINE inline __attribute__((always_inline))

#include "../ir/graph.hpp"       // hnnx::graph_crate 声明 (graph.hpp:154)
#include "../ir/tensor_base.hpp" // ::Tensor / hnnx::Allocator / DTypeScaleOff / 静态哈希族
#include "interface.hpp"         // ::Interface 族 + hnnx::make_interface
#include "shape.hpp"             // ::Shape<Rank>
#include "tensor_defs.hpp"       // dtype_traits / Tdefs (checksum_bytes 见下方前置)
#include "tensor_layouts.hpp"    // 布局族 + NoPadding/Padding

// hnnx::checksum_bytes — 定义于 ir/types.hpp (该头含 GCP 旧世界 hnnx::OpDef 族,
// 与 op_def.hpp 的 hnnx::OpDef_Const 重定义冲突, 故此处仅前置声明)
namespace hnnx {
uint64_t checksum_bytes(uint64_t prev, uint8_t const *bytes, unsigned n);
} // namespace hnnx

// ---------------------------------------------------------------------------
// hnnx::ptr_to_stdarray (SDK 通用辅助)
// ---------------------------------------------------------------------------
namespace hnnx {
template <size_t Rank, typename T> static inline std::array<T, Rank> ptr_to_stdarray(T const *ptr)
{
    std::array<T, Rank> ret{};
    for (size_t i = 0; i < Rank; ++i) ret[i] = ptr[i];
    return ret;
}
} // namespace hnnx

// ---------------------------------------------------------------------------
// RankedTensor<TRank> — tensor_base.h:558 原文 (rank() final; 其余全继承)
// ---------------------------------------------------------------------------
template <unsigned TRank> class RankedTensor : public Tensor {
  public:
    struct traits {
        static constexpr unsigned Rank = TRank;
    };
    explicit RankedTensor(const Op *producer_in) noexcept : Tensor(producer_in) {}
    explicit RankedTensor(hnnx::Deserz &dctx) noexcept : Tensor(dctx) {}
    RankedTensor(const RankedTensor &old, hnnx::Allocator *allocator, clone_mode cmode) noexcept
        : Tensor(old, allocator, cmode)
    {
    }
    static constexpr auto Rank = TRank;
    virtual size_t rank() const noexcept override final { return Rank; }
};

// ---------------------------------------------------------------------------
// layout_mem_contig<STYPE, TLayout, Pad_t> — 连续张量内存面
// ---------------------------------------------------------------------------
template <typename STYPE, typename TLayout, typename Pad_t> struct layout_mem_contig {
    static constexpr unsigned Rank = TLayout::Rank;
    using Shape_t = Shape<Rank>;
    using storage_type = STYPE;
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};
    static constexpr bool is_singular = std::is_same<TLayout, SingularMemoryLayout<Rank>>::value;

    storage_type *bulk_data;

    layout_mem_contig(Shape_t const * /*shp*/, Graph & /*graph_in*/) : bulk_data() {}

    // duplicate clone from another
    layout_mem_contig(Shape_t const * /*shp*/, layout_mem_contig const &other, hnnx::Allocator * /*alloc*/,
                      Tensor::clone_mode /*cmode*/)
        : bulk_data(other.bulk_data)
    {
    }

    // construct from deserialize
    layout_mem_contig(Shape_t const *, hnnx::Deserz &dctx)
        : bulk_data((storage_type *)Tensor::deserialize_block_pointer(dctx))
    {
    }

    void *raw_data() const noexcept { return (void *)bulk_data; }

    void set_raw_data_despite_danger(void *buffer) noexcept { bulk_data = static_cast<storage_type *>(buffer); }

    void *element_addr(Shape_t const *shp, size_t rank, SIdx const coords_in[],
                       std::array<size_t, Rank> const &valid_dims) const noexcept
    {
        (void)rank;
        static_assert(!is_singular);
        std::array<size_t, Rank> const padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        size_t const offset = layout.linear_offset(padded_coords, valid_dims);
        return (void *)&bulk_data[offset];
    }

    void **get_block_list_ptr() const
    {
        return (void **)&bulk_data; // contig: 块表 = bulk_data 字段本身
    }
    static size_t get_block_list_len(Shape_t const * /*shp*/) { return 1; }
    static size_t get_elements_per_block(Shape_t const *shp)
    {
        if (is_singular) return 1;
        size_t prod = 1;
        for (size_t i = 0; i < Rank; ++i) prod *= shp->max_dims[i];
        return prod;
    }
    storage_type **block_ptr_addr(Shape_t const * /*shape*/, std::array<SIdx, Rank> /*coords*/) const
    {
        return &bulk_data;
    }
    void realloc_blocktab(hnnx::Allocator * /*alloc*/, Shape_t const * /*old_shape*/,
                          Shape_t const * /*new_shape*/)
    {
        bulk_data = nullptr;
    }

    int compare_memory(Shape_t const *shp, layout_mem_contig const &rhs) const
    {
        size_t const len = get_elements_per_block(shp) * sizeof(storage_type);
        return memcmp(bulk_data, rhs.bulk_data, len);
    }
    uint32_t find_content_hash(Shape_t const *shp, uint32_t oldhash, bool is_float) const
    {
        // M20 §5.11 @db0667: content_hash_data ^ mulu32_modular(oldhash, 0x223131)
        size_t const len = get_elements_per_block(shp) * sizeof(storage_type);
        uint64_t const acc = uint64_t(oldhash) * 0x223131ull;
        return Tensor::content_hash_data(bulk_data, len, is_float) ^ uint32_t(acc);
    }
};

// ---------------------------------------------------------------------------
// hnnx::indirect_layout_mem — make_blocktab 族 (tensor_concrete.h:135-155 原文)
// ---------------------------------------------------------------------------
namespace hnnx {

inline void **make_blocktab(size_t n_blocks, ::Graph &graph_in)
{
    // @db534e..db5392: graph_crate → add_record_slot(n*8, 8) → status≥0 时
    // ++crate[0x40] (db537b) → memset(slot, 0, n*8)
    Crate *const crate_p = graph_crate(graph_in);
    if (n_blocks == 0) return nullptr; // db5356: 0 块 → null (add_record_slot 不调)
    Crate::record_slot_result r = crate_p->add_record_slot(n_blocks * 8, 8);
    if (r.status >= 0) crate_p->bump_record_count();
    void **const slot = r.slot; // sret+0x08 槽字段的值即记录地址
    memset(slot, 0, n_blocks * 8);
    return slot;
}

template <typename CRATE> // Crate or DCrate
void **make_blocktab_for_overwrite(size_t const n_blocks, CRATE *const crate_p)
{
    return crate_p->template alloc_array<void *>(n_blocks);
}

inline int compare_indirect_blocks(void **ptr_a, void **ptr_b, size_t nblocks, size_t blocklen)
{
    for (size_t i = 0; i < nblocks; i++) {
        int const cmp = memcmp(ptr_a[i], ptr_b[i], blocklen);
        if (cmp != 0) return cmp;
    }
    return 0;
}
} // namespace hnnx

// ---------------------------------------------------------------------------
// layout_mem_indirect<STYPE, TLayout, Pad_t> — 间接 (分块) 张量内存面
// ---------------------------------------------------------------------------
template <typename STYPE, typename TLayout, typename Pad_t> struct layout_mem_indirect {
    static constexpr unsigned Rank = TLayout::Rank;
    using Shape_t = Shape<Rank>;
    using storage_type = STYPE;
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};

    storage_type **blocktab;

    // construct table
    layout_mem_indirect(Shape_t const *shp, Graph &graph_in)
        : blocktab((storage_type **)hnnx::make_blocktab(layout.num_blocks(shp->max_dims), graph_in))
    {
    }
    // duplicate clone from another (@db5590: add_record_slot + bump + memcpy, 不清零)
    layout_mem_indirect(Shape_t const *shp, layout_mem_indirect const &other, hnnx::Allocator *alloc,
                        Tensor::clone_mode /*cmode*/)
        : blocktab()
    {
        size_t const nblocks = layout.num_blocks(shp->max_dims);
        hnnx::Crate *crate_p = hnnx::graph_crate(alloc->graph);
        if (nblocks == 0) return; // db55fe: 0 块 → blocktab 保持 null
        hnnx::Crate::record_slot_result r = crate_p->add_record_slot(nblocks * 8, 8);
        if (r.status >= 0) crate_p->bump_record_count();
        blocktab = (storage_type **)r.slot; // 槽字段值即记录地址 (db5638 memcpy 源)
        memcpy(blocktab, other.blocktab, sizeof(void *) * nblocks);
    }
    // construct from deserialize (@db5410: classic ? num_blocks : 1)
    layout_mem_indirect(Shape_t const *shp, hnnx::Deserz &dctx) : blocktab()
    {
        // 非 classic 格式下因延迟指针解析读不到 shape 对象; 1 是 "未知" 值。
        unsigned const nblocks = dctx.classic_format() ? unsigned(layout.num_blocks(shp->max_dims)) : 1;
        Tensor::deserialize_blocktable(dctx, blocktab, nblocks);
    }

    void *raw_data() const noexcept { return (void *)blocktab[0]; }
    void set_raw_data_despite_danger(void * /*buffer*/) noexcept
    {
        // SDK 原文 assert(!"Invalid to set raw pointer on this type of tensor");
        // release 构建 (0xdcd430) 中编译为空操作 —— 与 Tensor 基类行为一致。
    }

    void *element_addr(Shape_t const *shp, size_t rank, SIdx const coords_in[],
                       std::array<size_t, Rank> const & /*valid_dims*/) const noexcept
    {
        (void)rank;
        std::array<size_t, Rank> const padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        // @db5af0 (Crouton_8): block_offset = (Hp&7)<<8 | (Wp&7)<<5 | (Dp&0x1f);
        //                   block_index  = 行主序 over [N, H/8, W/8, D/32]
        size_t const block_offset = layout.chunk_offset(padded_coords, shp->max_dims);
        size_t const block_idx = layout.chunk_index(padded_coords, shp->max_dims);
        return (void *)&blocktab[block_idx][block_offset];
    }

    void **get_block_list_ptr() const
    {
        return (void **)blocktab; // 间接: 块表首址
    }
    static size_t get_block_list_len(Shape_t const *shp) { return layout.num_blocks(shp->max_dims); }
    static size_t get_elements_per_block(Shape_t const *shp) { return layout.block_total(shp->max_dims); }

    storage_type **block_ptr_addr(Shape_t const *shape, std::array<SIdx, Rank> coords) const
    {
        std::array<size_t, Rank> const padded_coords = pad.pad_coords(coords, shape->pad);
        size_t const block_idx = layout.chunk_index(padded_coords, shape->max_dims);
        return &blocktab[block_idx];
    }
    // @dc90a0: shape 实际变化且 nblocks 增大 → 重新分配 (已清零); 否则仅清零
    void realloc_blocktab(hnnx::Allocator *alloc, Shape_t const *old_shape, Shape_t const *new_shape)
    {
        size_t const nblocks = layout.num_blocks(new_shape->max_dims);
        if (old_shape != new_shape) {
            size_t const old_nblocks = layout.num_blocks(old_shape->max_dims);
            if (nblocks > old_nblocks) { // need reallocate.
                blocktab = (storage_type **)hnnx::make_blocktab(nblocks, alloc->graph);
                return; // already zeroed
            }
        }
        ::memset(blocktab, 0, nblocks * sizeof(void *));
    }

    int compare_memory(Shape_t const *shp, layout_mem_indirect const &rhs) const
    {
        size_t const nblocks = layout.num_blocks(shp->max_dims);
        size_t const blocklen = sizeof(storage_type) * layout.block_total(shp->max_dims);
        return hnnx::compare_indirect_blocks((void **)blocktab, (void **)rhs.blocktab, nblocks, blocklen);
    }
    uint32_t find_content_hash(Shape_t const *shp, uint32_t oldhash, bool is_float) const
    {
        size_t const nblocks = layout.num_blocks(shp->max_dims);
        size_t const blocklen = sizeof(storage_type) * layout.block_total(shp->max_dims);
        return Tensor::content_hash_data_indirect(oldhash, (void **)blocktab, unsigned(nblocks), blocklen,
                                                  is_float);
    }
};

// ---------------------------------------------------------------------------
// BlockTableAccessor<Linfo> — tensor_concrete.h:278-385 原文
// (对象布局 @0xdc91f0: blktab@0x00, blkdims@0x08, blkstrides@0x28, margin@0x48)
// ---------------------------------------------------------------------------
template <typename Linfo> class LayoutTensor;

template <typename Linfo> class BlockTableAccessor {
  protected:
    static constexpr unsigned Rank = Linfo::Rank;
    using storage_type = typename Linfo::storage_type;
    using pointer_type = storage_type *;
    using TLayout = typename Linfo::Tlayout;
    using Pad_t = typename Linfo::Pad_t;
    static_assert(Linfo::is_indirect && Linfo::is_chunked);
    // tensor_traits<LayoutTensor<Linfo>>::is_variable_block / indirect_ranks 原文
    static constexpr unsigned indirect_ranks_v = TLayout::indirect_ranks;
    static constexpr bool is_variable_block_v = indirect_ranks_v < Rank;
    pointer_type *blktab;                // the base of the block table
    std::array<size_t, Rank> blkdims;    // dims of the block table in blocks
    std::array<size_t, Rank> blkstrides; // 'strides' (dim i 的步长是 blkstrides[i+1];
                                         //   dim RANK-1 为 1; blkstrides[0] 是总长)
    std::array<unsigned, Rank> margin;   // margin offset
    // support for 'variable_block':
    static constexpr unsigned n_blockshape_dims = is_variable_block_v ? Rank - indirect_ranks_v : 0;
    std::array<size_t, n_blockshape_dims> blockshape_dims;

  public:
    explicit BlockTableAccessor(LayoutTensor<Linfo> const &tens);

    static constexpr unsigned rank() { return Rank; }

    size_t blocktab_len() const { return blkstrides[0]; }
    pointer_type *blocktab_ptr() const { return blktab; }
    size_t blocktab_blocksize() const
    {
        std::array<size_t, Rank> dummy_shape{};
        // the shape will be ignored (unless is_variable_block).
        // (SDK 原文此处写作 Rank - blockshape_dims —— 依值即 n_blockshape_dims 的笔误)
        std::copy_n(blockshape_dims.begin(), n_blockshape_dims, dummy_shape.begin() + (Rank - n_blockshape_dims));
        return TLayout::block_total(dummy_shape);
    }
    size_t blocktab_blocksize_bytes() const { return blocktab_blocksize() * sizeof(storage_type); }

    size_t blocktab_dim(int i) const { return blkdims[i]; }
    size_t blocktab_dim_stride(int i) const { return (i < int(Rank) - 1) ? blkstrides[i + 1] : 1; }

    // block_ptr_address(b,h,w,d) and block_ptr accept element coordinates.
    template <typename... ind_types> pointer_type *block_ptr_address(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return block_ptr_calc(coords);
    }
    template <typename... ind_types> pointer_type &block_ptr(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *block_ptr_calc(coords);
    }
    // blktab(b,h,w,d) accepts *block* coords
    template <typename... ind_types> pointer_type &blocktab(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *blktab_ptr_calc(coords);
    }
    // same_table_shape: the shape of the table is the same as the 'other'.
    bool same_table_shape(BlockTableAccessor const &other) const
    {
        for (int i = 0; i < int(Rank); i++)
            if (blkdims[i] != other.blkdims[i]) return false;
        return true;
    }
    // 'same_layout' means the same table shape and the same padding offset.
    bool same_layout(BlockTableAccessor const &other) const
    {
        if (!same_table_shape(other)) return false;
        for (int i = 0; i < int(Rank); i++)
            if (margin[i] != other.margin[i]) return false;
        return true;
    }

  protected:
    pointer_type *block_ptr_calc(std::array<SIdx, Rank> const &coords) const
    {
        size_t sum = 0;
        for (int i = 0; i < int(Rank); i++) {
            unsigned blk = TLayout::ChunkSizes[i];
            unsigned idx = unsigned((coords[i] + SIdx(margin[i]) + (blk - 1)) / blk);
            sum += idx * ((i < int(Rank) - 1) ? blkstrides[i + 1] : 1);
        }
        return blktab + sum;
    }
    pointer_type *blktab_ptr_calc(std::array<SIdx, Rank> const &coords) const
    {
        size_t sum = size_t(coords[Rank - 1]);
        for (int i = 0; i < int(Rank) - 1; i++) {
            sum += size_t(coords[i]) * blkstrides[i + 1];
        }
        return blktab + sum;
    }
};

// ---------------------------------------------------------------------------
// 本文件局部辅助 (SDK 侧位于 tensor_base.h:253-297 / conversions.h:510 /
// tensor_concrete.h:19-24; 因 FlatMemoryLayout 在本头才完整, 不能挂到 Tensor 上)
// ---------------------------------------------------------------------------
namespace hnnx {
// conversions.h:510 原文 — u32 模乘 (仅为绕过 UBSan 的 unsigned 溢出包装)
static inline constexpr unsigned mulu32_modular(unsigned const a, unsigned const b) noexcept { return a * b; }
// tensor_concrete.h:19-24 原文
template <size_t N> static inline size_t product_of_array(std::array<size_t, N> const &arr)
{
    return std::accumulate(arr.cbegin(), arr.cend(), size_t(1), std::multiplies<size_t>());
}
// serialize_oplist.h:114 — tensMODE_general = 1 (与 ser_ops_interface.hpp 一致)
static constexpr unsigned tensMODE_general_v = 1;
} // namespace hnnx

// Tensor::formatcode_for_interface<IFC> — tensor_base.h:254-264 原文
template <typename IFC> static constexpr uint32_t formatcode_for_interface()
{
    constexpr ::DType dt = IFC::dtype;
    uint32_t result = unsigned(dt);
    constexpr unsigned elbytes = sizeof(typename dtype_traits<dt>::element_type);
    constexpr unsigned log2sz = (elbytes == 8) ? 3 : (elbytes == 4) ? 2 : (elbytes == 2) ? 1 : 0;
    static_assert(elbytes == (1u << log2sz));
    result |= log2sz << Tensor::tformat_log2sz_shift;
    if (dtype_traits<dt>::is_quant) result |= Tensor::tformat_is_quantized;
    return result;
}
// Tensor::formatcode_for_general<TRAITS> — tensor_base.h:265-279 原文
template <typename TRAITS> static constexpr uint32_t formatcode_for_general()
{
    constexpr unsigned rankval = TRAITS::rank;
    uint32_t result = formatcode_for_interface<typename TRAITS::interface_type>();
    result |= (rankval << Tensor::tformat_rank_shift);
    if (TRAITS::memclass == hnnx::MemoryClass::TCM) result |= Tensor::tformat_is_tcm;
    if (TRAITS::is_indirect) result |= Tensor::tformat_is_indirect;
    if (TRAITS::is_chunked) result |= Tensor::tformat_is_chunked;
    if (!std::is_base_of<FlatMemoryLayout<rankval>, typename TRAITS::layout_type>::value) {
        result |= Tensor::tformat_is_not_flat;
    }
    if (TRAITS::is_singular) result |= Tensor::tformat_is_singular;
    return (hnnx::tensMODE_general_v << Tensor::tformat_tmode_shift) | result;
}
// Tensor::pack_tensor_info — tensor_base.h:291-299 原文
static constexpr uint32_t pack_tensor_info(::DType type, uint32_t rank, hnnx::MemoryClass mclass)
{
    uint32_t tinfo = 0x10;
    tinfo |= static_cast<uint32_t>(type) & 0xFu;
    tinfo |= (rank & 0xFu) << 8u;
    tinfo |= (static_cast<uint32_t>(mclass) & 0xF) << 16u;
    tinfo |= hnnx::tensMODE_general_v << Tensor::tformat_tmode_shift;
    return tinfo;
}

// ---------------------------------------------------------------------------
// LayoutTensor<Linfo> — tensor_concrete.h:407-870 原文
// ---------------------------------------------------------------------------
template <typename Linfo> class LayoutTensor : public RankedTensor<Linfo::Rank> {
  protected:
    using BaseRT = RankedTensor<Linfo::Rank>;
    static constexpr unsigned Rank = Linfo::Rank;
    using storage_type = typename Linfo::storage_type;
    using TLayout = typename Linfo::Tlayout;
    using Pad_t = typename Linfo::Pad_t;
    static constexpr bool is_chunked = Linfo::is_chunked;
    static_assert(is_chunked == (TLayout::chunk_total > 1));
    static constexpr bool is_indirect = Linfo::is_indirect;
    static constexpr bool is_padded = !std::is_same<Pad_t, NoPadding<Rank>>::value;

    static_assert(!(is_indirect && !is_chunked), "non-chunked layouts can't be indirect");

    ::Interface const *const interface_ptr; // +0x08 — 共享 Interface 实例
    using Shape_t = Shape<Rank>;

  public:
    Shape_t const *shape; // +0x10 — 共享形状 (canonical_shape 规范化)
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};
    // SDK 原文: shape->get_shape_info() (shape 侧 @0x12f7020 未解码 —— 暂以空串占位)
    std::string get_shape_info() const override { return {}; }

  protected: // interface, then shape, then mem
    using layout_mem_t = std::conditional_t<is_indirect, layout_mem_indirect<storage_type, TLayout, Pad_t>,
                                            layout_mem_contig<storage_type, TLayout, Pad_t>>;
    layout_mem_t mem; // +0x18

  public:
    struct traits {
        using storage_type = LayoutTensor::storage_type;
        using raw_type = LayoutTensor::storage_type; // result from get_raw()
        static constexpr unsigned rank = Rank;
        static constexpr bool is_indirect = LayoutTensor::is_indirect;
        static constexpr bool is_chunked = LayoutTensor::is_chunked;
        static constexpr bool is_singular = std::is_same<TLayout, SingularMemoryLayout<Rank>>::value;
        static constexpr bool has_padding = !std::is_same<Pad_t, NoPadding<Rank>>::value;
        static constexpr unsigned indirect_ranks = TLayout::indirect_ranks;
        static constexpr bool is_variable_block = is_indirect && (indirect_ranks < Rank);
        using pad_type = Pad_t;
        using layout_type = TLayout;
        using layouttensor_type = LayoutTensor;
    };

  protected:
    static constexpr bool is_singular = traits::is_singular;
    // 主构造的 shape 构建 (@db5302: 栈 Shape{flags=0, dims=max_sizes,
    // max_dims=TLayout::pad(max_sizes), pad=0} → canonical_shape)
    static Shape<Rank> const *init_shape_p(Graph &graph_in, const ::OutputDef &def)
    {
        // (目标平台 uint64_t == size_t; 本机 Apple 工具链下二者为不同整数类型,
        //  以指针转换保持 SDK 原文的 ptr_to_stdarray 调用形态)
        using msp_t = size_t const *;
        Shape_t const shp(hnnx::ptr_to_stdarray<Rank, size_t>(msp_t(&def.max_sizes[0])),
                          TLayout::pad(hnnx::ptr_to_stdarray<Rank, size_t>(msp_t(&def.max_sizes[0]))));
        if constexpr (is_singular) {
            if (std::find_if(shp.dims.begin(), shp.dims.end(), [](size_t d) { return d != 1; }) != shp.dims.end()) {
                throw std::runtime_error("singular tensor with shape not 1's");
            }
        }
        return Shape_t::canonical_shape(graph_in, shp);
    }
    // only used in the deserialize ctor
    ::Interface const *&interface_ptr_ref() { return const_cast<::Interface const *&>(interface_ptr); }
    // ctors are marked noinline; otherwise they just get inlined
    // into all the ConcreteTensor ctors, which isn't really helpful.
    [[gnu::noinline]] LayoutTensor(const Op *producer_in, const ::OutputDef &def, Graph &graph_in,
                                   ::Interface const *(*ifc_maker)(Graph &, ::OutputDef const &))
        : BaseRT(producer_in), interface_ptr((*ifc_maker)(graph_in, def)), //
          shape(init_shape_p(graph_in, def)), //
          mem(shape, graph_in)
    {
    }
    using interface_deser_func = ::Interface const *(*)(hnnx::Deserz &, ::Interface const **);
    [[gnu::noinline]] LayoutTensor(hnnx::Deserz &dctx, interface_deser_func const ifc_deser_fp)
        : BaseRT(dctx), interface_ptr((*ifc_deser_fp)(dctx, &interface_ptr_ref())),
          shape(Shape_t::deserialize(dctx, &shape)), mem(shape, dctx)
    {
    }
    // clone ctor (@db5590: ifc 直传; shape 共享; cmode 全程未读)
    [[gnu::noinline]] LayoutTensor(const LayoutTensor &old, hnnx::Allocator *allocator,
                                   ::Interface const *ifc_in, Tensor::clone_mode cmode)
        : BaseRT(old, allocator, cmode), interface_ptr(ifc_in), shape(old.shape), mem(shape, old.mem, allocator, cmode)
    {
    }

  public:
    virtual size_t dim(size_t index) const noexcept override final { return shape->dims[index]; }
    std::array<size_t, Rank> const &dims() const { return shape->dims; }
    template <typename... T> std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        return Tensor::dims_extractor(get_dims(), indices...);
    }
    virtual std::pair<size_t const *, size_t> get_dims() const noexcept final
    {
        return std::pair<size_t const *, size_t>(&shape->dims[0], Rank);
    }
    virtual bool set_dims(const size_t dims[]) override final { return false; }
    virtual bool set_dims(const Tensor &prototype) override final
    {
        (void)prototype;
        return false;
    }
    // 'interface()' needs to be overriden in ConcreteTensor
    float interface_scale() const { return this->interface().get_scale(); }
    float interface_scale_recip() const { return this->interface().get_scale_recip(); }
    int32_t interface_offset() const { return this->interface().get_offset(); }

    // for direct access to bulk_data, in contiguous tensors only
    std::conditional_t<is_indirect, void, storage_type *&> data_ptr()
    {
        if constexpr (!is_indirect) {
            return mem.bulk_data;
        }
    }
    std::conditional_t<is_indirect, void, storage_type *const &> data_ptr() const
    {
        if constexpr (!is_indirect) {
            return mem.bulk_data;
        }
    }

    // block table access
    storage_type **blocktab_ptr() const { return (storage_type **)mem.get_block_list_ptr(); }
    storage_type *&blocktab_at(size_t i)
    {
        if constexpr (!is_indirect) {
            assert(i == 0);
            return *(storage_type **)mem.get_block_list_ptr();
        } else {
            return ((storage_type **)mem.get_block_list_ptr())[i];
        }
    }
    storage_type *const &blocktab_at(size_t i) const
    {
        if constexpr (!is_indirect) {
            assert(i == 0);
            return *(storage_type **)mem.get_block_list_ptr();
        } else {
            return ((storage_type **)mem.get_block_list_ptr())[i];
        }
    }
    size_t blocktab_len() const { return mem.get_block_list_len(shape); }
    size_t blocktab_blocksize() const { return mem.get_elements_per_block(shape); }
    size_t blocktab_blocksize_bytes() const { return mem.get_elements_per_block(shape) * sizeof(storage_type); }

    virtual size_t total_storage_bytes() const final override
    {
        return total_storage_elements() * sizeof(storage_type);
    }
    virtual size_t total_storage_elements() const final override
    {
        size_t const total_elements = hnnx::product_of_array(shape->max_dims);
        return total_elements;
    }
    virtual void *raw_data() noexcept override final { return mem.raw_data(); }
    virtual void set_raw_data_despite_danger(void *buffer) noexcept override final
    {
        mem.set_raw_data_despite_danger(buffer);
    }

  protected:
    // Underlying code for change_{shape,pad,shape_pad} (@dc8f70:
    // pad 字段 uint8 截断; max_dims = layout.pad(pad_coords(dims, pad)))
    void change_shapepad_impl(hnnx::Allocator &allocator, size_t const *const p_new_dims,
                              size_t const *const p_new_pads = nullptr) // optional pads
    {
        Shape_t newshape = *shape; // copy old shape
        if (p_new_dims) {
            for (size_t i = 0; i < Rank; i++)
                newshape.dims[i] = p_new_dims[i];
        }
        if (p_new_pads) {
            for (size_t i = 0; i < Rank; i++)
                newshape.pad[i] = uint8_t(p_new_pads[i]);
        }
        newshape.max_dims = layout.pad(pad.pad_coords(newshape.dims, newshape.pad));
        // nake a persistent copy of new shape
        Shape_t const *const new_shape_p = Shape_t::canonical_shape(allocator.graph, newshape);
        // new_shape_p will be same pointer as shape, if shape wasn't changed. realloc_blocktab
        // checks for that.
        mem.realloc_blocktab(&allocator, shape, new_shape_p);
        shape = new_shape_p;
    }

  public:
    // change the padding; and reallocate blocktab if it's larger as a result.
    // in any case, all of the block pointers are zeroed.
    void change_pad(std::array<size_t, Rank> const &new_pad, hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, nullptr, new_pad.data());
    }
    void change_shape(std::array<size_t, Rank> const &new_dims, hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, new_dims.data());
    }
    // special entry point for use by 'generic' change_shape operation.
    void change_shape_arr(size_t const *const p_new_dims, hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, p_new_dims);
    }
    void change_shape_pad(std::array<size_t, Rank> const &new_dims, std::array<size_t, Rank> const &new_pad,
                          hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, new_dims.data(), new_pad.data());
    }

    template <typename... ind_types> storage_type const *const *block_ptr_address(ind_types... inds) const
    {
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return mem.block_ptr_addr(shape, coords);
    }
    template <typename... ind_types> storage_type *const *block_ptr_address(ind_types... inds)
    {
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return mem.block_ptr_addr(shape, coords);
    }
    template <typename... ind_types> storage_type const *block_ptr(ind_types... inds) const
    {
        return *block_ptr_address(inds...);
    }
    template <typename... ind_types> storage_type *block_ptr(ind_types... inds)
    {
        return *block_ptr_address(inds...);
    }

    std::conditional_t<is_indirect, BlockTableAccessor<Linfo>, void> blocktable_accessor() const
    {
        if constexpr (is_indirect) {
            return BlockTableAccessor<Linfo>(*this);
        }
    }

    // this only makes sense for indirect tensors.
    std::conditional_t<is_indirect, std::array<size_t, Linfo::Rank>, void> tile_strides() const
    {
        if constexpr (is_indirect) {
            std::array<size_t, Linfo::Rank> ret = {0};
            ret[Linfo::Rank - 1] = 1;
            for (int i = int(Linfo::Rank) - 2; i >= 0; i--) {
                ret[i] = ret[i + 1] * (shape->max_dims[i + 1] / layout.ChunkSizes[i + 1]);
            }
            return ret;
        }
    }

    // get_raw_addr(...) on this class gives a storage_type *.
    template <typename... ind_types> storage_type const *get_raw_addr(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (storage_type const *)element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> storage_type *get_raw_addr(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (storage_type *)element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> storage_type const &get_raw(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(storage_type const *)element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> storage_type &get_raw(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(storage_type *)element_addr0(Rank, coords.data());
    }
    // read_tile/write_tile/tile_support_bits (SDK 定义于 tile_extract.h) —— 该翻译
    // 单元尚未反汇编, 此处不覆盖, 沿用 Tensor 基类默认 (打印 + throw)。

    LayoutTensor &layout_base() { return *this; }
    LayoutTensor const &layout_base() const { return *this; }

    // checksum for debug
    [[gnu::noinline]] virtual uint64_t get_checksum() const noexcept override
    {
        uint64_t chk = 0;
        if constexpr (Rank == 4) {
            auto [batch, heights, width, depth] = this->get_dims_4();
            if (batch && heights && width && depth) {
                storage_type const x0 = *(storage_type const *)this->get_raw_addr(0, 0, 0, 0);
                for (size_t b = 0; b < batch; b++) {
                    for (size_t h = 0; h < heights; h++) {
                        for (size_t w = 0; w < width; w++) {
                            for (size_t d = 0; d < depth; d++) {
                                storage_type x = *(storage_type const *)this->get_raw_addr(b, h, w, d);
                                x ^= x0;
                                union {
                                    storage_type as_x;
                                    uint8_t as_byte[sizeof(storage_type)];
                                } uu = {x};
                                chk = hnnx::checksum_bytes(chk, uu.as_byte, sizeof(storage_type));
                            }
                        }
                    }
                }
                chk ^= x0;
            }
        }
        return chk;
    }

  protected:
    // element_addr is delegated to the particular specialization of layout_mem
    ALWAYSINLINE void *element_addr0(size_t rank, const SIdx coords_in[]) const noexcept
    {
        if constexpr (!is_singular) {
            return mem.element_addr(shape, rank, coords_in, shape->dims);
        } else {
            return mem.raw_data();
        }
    }

    // This is called from ConcreteTensor::compare_sametype to fully compare two tensors
    // which are already known to be the same type (and have same interface)
    [[gnu::noinline]] int compare_sametype_layout(LayoutTensor const *rhs) const
    {
        if (shape->dims != rhs->shape->dims) {
            return std::lexicographical_compare(shape->dims.begin(), shape->dims.end(), rhs->shape->dims.begin(),
                                                rhs->shape->dims.end())
                           ? -1
                           : 1;
        }
        if (is_padded) {
            if (shape->max_dims != rhs->shape->max_dims) {
                return std::lexicographical_compare(shape->max_dims.begin(), shape->max_dims.end(),
                                                    rhs->shape->max_dims.begin(), rhs->shape->max_dims.end())
                               ? -1
                               : 1;
            }
        }
        // compare memory now (delegate to layout_mem).
        return mem.compare_memory(shape, rhs->mem);
    }
    // allocation and enumeration.
    [[gnu::noinline]] void allocate_layout(hnnx::Allocator &allocator, unsigned options, hnnx::MemoryClass mclass)
    {
        // get the pointer to block table; and number of entries in it.
        void **const blocktab = this->mem.get_block_list_ptr();
        size_t const nblocks = this->mem.get_block_list_len(this->shape);
        size_t const blocksize = sizeof(storage_type) * this->mem.get_elements_per_block(this->shape);
        size_t const align = traits::is_indirect ? blocksize : std::min(size_t(256), sizeof(storage_type));
        if constexpr (traits::is_singular) {
            options |= unsigned(hnnx::AllocOpts_packed);
        }
        allocator.allocate_n(blocktab, nblocks, blocksize, align, mclass, options, this->get_dtype());
    }
    [[gnu::noinline]] void enum_memory_blocks_layout(hnnx::MemBlockEnumerator &en, hnnx::MemoryClass mclass) const
    {
        void **const blocktab = this->mem.get_block_list_ptr();
        size_t const nblocks = this->mem.get_block_list_len(this->shape);
        en.supply_blocks(this, mclass, (void *const *)blocktab, nblocks);
    }
    // called from find_content_hash in the ConcreteTensor class. hash_in includes
    // hash of dtype and interface. (@db5a80: 首行 Rank*0x102401 异或是源内死代码,
    // 已被编译器消除 —— 保留为注释)
    [[gnu::noinline]] uint32_t find_content_hash_layout(uint32_t hash_in, bool is_float) const noexcept
    {
        // uint32_t h = hash_in ^ (Linfo::Rank * 0x102401u);   // dead code
        uint32_t h = Tensor::build_hash(shape->dims.data(), int(Linfo::Rank), hash_in);
        if (is_padded) {
            h = Tensor::build_hash(shape->max_dims.data(), int(Linfo::Rank), h);
        }
        return mem.find_content_hash(shape, h, is_float);
    }
};

// ---------------------------------------------------------------------------
// BlockTableAccessor<Linfo> 构造 — tensor_concrete.h:305-318 原文
// (需 LayoutTensor 完整; @dc91f0 对象布局: blktab@0, blkdims@+8, blkstrides@+0x28,
//  margin@+0x48)
// ---------------------------------------------------------------------------
template <typename Linfo>
BlockTableAccessor<Linfo>::BlockTableAccessor(LayoutTensor<Linfo> const &tens) : blktab(tens.blocktab_ptr())
{
    Shape<Rank> const &shp = *tens.shape;
    size_t allprod = 1;
    for (int i = int(Rank) - 1; i >= 0; --i) {
        unsigned const blk = unsigned(TLayout::ChunkSizes[i]);
        size_t const blkdim = shp.max_dims[i] / blk;
        allprod *= blkdim;
        blkdims[i] = blkdim;
        margin[i] = shp.pad[i];
        blkstrides[i] = allprod;
    }
    // if any dims affect the block size, copy those out.
    std::copy_n(shp.max_dims.begin() + (Rank - n_blockshape_dims), n_blockshape_dims, blockshape_dims.begin());
}

// ---------------------------------------------------------------------------
// ConcreteTensor<Tinfo> — tensor_concrete.h:841-1070 原文
// (operator() Accessor 族与 code_to_type_name/TensorTypeStruct 未随本里程碑
//  移植: 前者依赖 Interface::Accessor 模型, 后者属类型注册表)
// ---------------------------------------------------------------------------
template <typename Tinfo> class ConcreteTensor : public LayoutTensor<typename Tinfo::Lconfig> {
  protected:
    using Interface_t = typename Tinfo::Interface_t;
    using Layout_t = typename Tinfo::Tlayout;
    using Pad_t = typename Tinfo::Pad_t;
    static constexpr ::DType dtype = Interface_t::dtype; // dtype_of_type<Interface_t>()
    static constexpr bool is_indirect = Tinfo::is_indirect;
    static constexpr unsigned Rank = Layout_t::Rank;
    using BaseLayout = LayoutTensor<typename Tinfo::Lconfig>;
    static constexpr bool is_singular = BaseLayout::traits::is_singular;
    using BaseRT = typename BaseLayout::BaseRT;

    static_assert(Rank == BaseLayout::Rank && is_indirect == BaseLayout::traits::is_indirect &&
                          std::is_same<Layout_t, typename BaseLayout::traits::layout_type>::value &&
                          std::is_same<Pad_t, typename BaseLayout::traits::pad_type>::value,
                  "incompatible base class for ConcreteTensor");

    Interface_t const *interface_typed() const
    {
        return static_cast<Interface_t const *>(this->interface_ptr);
    }

  public:
    const char *true_name() const noexcept override { return Tinfo::typetag; }
    using element_type = typename dtype_traits<dtype>::element_type;

    struct traits : public BaseLayout::traits {
        static constexpr ::DType dtype = ConcreteTensor::dtype;
        using element_type = typename dtype_traits<dtype>::element_type;
        using raw_type = element_type; // result from get_raw()
        using interface_type = Interface_t;
        static constexpr hnnx::MemoryClass memclass = Tinfo::memclass;
    };
    //
    //  - build for given shape, attached to given producer.
    //  - pass the nase class ctor a specialized ctor, it uses to make the interface
    //   from the output def.
    ConcreteTensor(const Op *producer_in, const ::OutputDef &def, Graph &graph_in)
        : BaseLayout(producer_in, def, graph_in, hnnx::make_interface<Interface_t>::from_odef)
    {
    }
    ConcreteTensor(const Op *producer_in, const ::OutputDef &def, Graph &graph_in, element_type *data_in)
        : BaseLayout(producer_in, def, graph_in, hnnx::make_interface<Interface_t>::from_odef)
    {
        this->mem.set_raw_data_despite_danger((void *)data_in);
    }
    //   - deserialize. Note that dctx contains a graph ref.
    explicit ConcreteTensor(hnnx::Deserz &dctx) : BaseLayout(dctx, &hnnx::make_interface<Interface_t>::from_deser) {}

    //    - 'clone duplicate' of the given tensor. Note that cmode is ignored.
    ConcreteTensor(const ConcreteTensor &old, hnnx::Allocator *allocator, Tensor::clone_mode cmode,
                   const ::OutputDef *def = nullptr)
        : BaseLayout(old, allocator,
                     (def ? hnnx::make_interface<Interface_t>::from_odef(allocator->graph, *def) : old.interface_ptr),
                     cmode)
    {
    }

    virtual ::DTypeScaleOff get_dtype_intfc() const noexcept override
    {
        return interface_typed()->get_dtype_scaleoff();
    }

    virtual hnnx::InterfaceRef interface() const noexcept override final
    {
        return interface_typed()->get_refobj();
    }
    float interface_scale() const { return interface_typed()->get_scale(); }
    float interface_scale_recip() const { return interface_typed()->get_scale_recip(); }
    int32_t interface_offset() const { return interface_typed()->get_offset(); }

    ALWAYSINLINE element_type const *element_ptr(size_t rank, const SIdx coords[]) const
    {
        return (element_type const *)this->element_addr0(rank, coords);
    }
    ALWAYSINLINE element_type *element_ptr(size_t rank, const SIdx coords[])
    {
        return (element_type *)this->element_addr0(rank, coords);
    }

    // Some methods return the same thing as in LayoutTensor, but
    // with the type being element_type instead of storage_type.
    std::conditional_t<is_indirect, void, element_type *&> data_ptr()
    {
        if constexpr (!is_indirect) {
            return (element_type *&)this->mem.bulk_data;
        }
    }
    std::conditional_t<is_indirect, void, element_type *const &> data_ptr() const
    {
        if constexpr (!is_indirect) {
            return (element_type *const &)this->mem.bulk_data;
        }
    }

    // block table access
    element_type **blocktab_ptr() const { return (element_type **)this->mem.get_block_list_ptr(); }
    element_type *&blocktab_at(size_t i) { return (element_type *&)BaseLayout::blocktab_at(i); }
    element_type *const &blocktab_at(size_t i) const { return (element_type *const &)BaseLayout::blocktab_at(i); }

    template <typename... ind_types> element_type const *const *block_ptr_address(ind_types... inds) const
    {
        return (element_type const *const *)BaseLayout::block_ptr_address(inds...);
    };
    template <typename... ind_types> element_type *const *block_ptr_address(ind_types... inds)
    {
        return (element_type *const *)BaseLayout::block_ptr_address(inds...);
    };
    template <typename... ind_types> element_type const *block_ptr(ind_types... inds) const
    {
        return *this->block_ptr_address(inds...);
    }
    template <typename... ind_types> element_type *block_ptr(ind_types... inds)
    {
        return *this->block_ptr_address(inds...);
    }

    // direct access methods.
    template <typename... ind_types> element_type const &get_raw(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(element_type const *)this->element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> element_type &get_raw(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(element_type *)this->element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> element_type const *get_raw_addr(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (element_type const *)this->element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> element_type *get_raw_addr(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (element_type *)this->element_addr0(Rank, coords.data());
    }
    virtual uint32_t get_tensor_format_code() const noexcept override
    {
        return formatcode_for_general<traits>();
    }

    virtual uint32_t get_tensor_info() const noexcept override
    {
        return pack_tensor_info(traits::dtype, uint32_t(Rank), traits::memclass);
    }
    // allocation and enumeration.
    virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) override final
    {
        this->allocate_layout(allocator, options, traits::memclass);
    }
    virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &en) const override
    {
        this->enum_memory_blocks_layout(en, traits::memclass);
    }
    // hash the dtype and interface, and let find_content_hash_layout do the rest.
    virtual uint32_t find_content_hash() const noexcept override final
    {
        uint32_t const h = interface().interface_hash() ^ hnnx::mulu32_modular(unsigned(dtype), 0x107301);
        static constexpr bool is_float = dtype_traits<dtype>::is_float;
        return this->find_content_hash_layout(h, is_float);
    }

  protected:
    // because this (may) need to return an "InterfaceRef" via iref pointer, it's defined
    // here in the 'Concrete' class, but it uses the non-virtual 'element_addr0'
    // in the LayoutTensor base class to find the address, and adds the InterfaceRef if
    // requested.
    virtual void *element_addr(size_t rank, const SIdx coords_in[],
                               hnnx::InterfaceRef *const iref = nullptr) const noexcept final override
    {
        if (iref) *iref = interface_typed()->get_refobj();
        return this->element_addr0(rank, coords_in);
    }
    virtual int compare_sametype(const Tensor *rhs_in) const override
    {
        // compare the interface, and then all the rest is done in compare_sametype_layout.
        auto *rhs = static_cast<ConcreteTensor const *>(rhs_in);
        int const icmp = interface_typed()->compare(*rhs->interface_typed());
        if (icmp != 0) return icmp;
        return this->compare_sametype_layout(rhs);
    }

    virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                              Tensor::tensor_blockinfo *infop,
                              const ::OutputDef *od /* nullptr */) const override
    {
        void **retval = nullptr;
        ConcreteTensor const *newtens = nullptr;
        if (tensp) {
            if constexpr (!dtype_traits<dtype>::is_quant) od = nullptr;
            *tensp = std::make_unique<ConcreteTensor>(*this, allocator, Tensor::clone_mode::duplicate, od);
            newtens = static_cast<ConcreteTensor const *>(tensp->get());
            retval = (void **)newtens->mem.get_block_list_ptr();
        }
        if (infop) {
            infop->setup(traits::dtype, traits::memclass);
            infop->blkptrs = (void **)this->mem.get_block_list_ptr();
            // pretend that a pointer to Shape<Rank> is really a pointer to its base class ShapeFlags
            // we provide a pointer to the shape field in the cloned tensor, if applicable; otherwise in 'this'.
            infop->shapepp = (hnnx::ShapeFlags const **)&(newtens ? newtens : this)->shape;
            infop->interfacepp = &(newtens ? newtens : this)->interface_ptr;
            infop->nblocks = this->mem.get_block_list_len(this->shape);
            infop->blocksize = sizeof(element_type) * this->mem.get_elements_per_block(this->shape);
            infop->is_indirect = this->is_indirect;
            infop->is_chunked = traits::is_chunked;
            infop->is_variable_block = traits::is_variable_block;
            return retval;
        }
        return nullptr;
    }
};
