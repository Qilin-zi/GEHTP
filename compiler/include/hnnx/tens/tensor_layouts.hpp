// ============================================================================
// 布局层 — memory_layout.h + padding.h 全局模板精确移植, M34-C3
//
// 字节级证据 (objdump 直证, 详见 audit_verify/reports/M20_LayoutTensor_disasm.md):
//   · ChunkSizes 的反汇编直接证据: blocktab_len/element_addr 内恒见
//     max_dims[1]>>3 / max_dims[2]>>3 / max_dims[3]>>5 (Crouton_8, chunk 1,8,8,32,
//     块字节恒 0x800); Crouton_16 为 >>3 / >>2 / >>5 (R4Crouton2Layout chunk 1,8,2,32
//     ×尾部 2,2 → dim2 有效 4, dc9710 shr $0x2 直证)。
//   · TLayout::pad 内联点 (主构造 @db5302): dim0 不取整 (chunk=1)、dim1/dim2
//     round8、dim3 round32 —— 与 R4CroutonLayout::pad 逐位一致。
//   · FlatMemoryLayout::pad = dims 恒等 (NoPadding 族, 无任何取整指令)。
//   · num_blocks (Crouton_8) @db5326: maxd0*(maxd1/8)*(maxd2/8)*(maxd3/32)。
//   · 主构造/克隆构造 make_blocktab: graph_crate → Crate::add_record_slot(nblocks*8, 8)
//     → memset(slot, 0, nblocks*8) (db534e..db5392 / db55f7..db5638);
//     覆写变体 (clone) 不清零, 直接 memcpy 指针表 (db5638)。
//   · element_addr (间接) @db5af0: 块内偏移 (Hp&7)<<8 | (Wp&7)<<5 | (Dp&0x1f),
//     块号 = ((Np*(maxd1/8) + Hp/8)*(maxd2/8) + Wp/8)*(maxd3/32) + Dp/32。
//   · element_addr (contig) @db0680: 纯 row-major
//     ((c0*d1 + c1)*d2 + c2)*d3 + c3。
// 类结构/命名: SDK qnn_ori_include/hnnx/sdk/core/memory_layout.h + padding.h 原文。
// ============================================================================
#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "../ir/tensor_base.hpp" // SIdx (= long); ::Tensor 前置族

using Idx = size_t; // padding.h:16 — typedef size_t Idx

// ---------------------------------------------------------------------------
// padding.h 原文 — NoPadding / Padding (全局模板)
// ---------------------------------------------------------------------------
template <Idx Rank> class NoPadding {
  public:
    static constexpr unsigned int is_padded = 0;
    template <typename PadT>
    inline constexpr std::array<Idx, Rank> pad_coords(const std::array<Idx, Rank> &coords,
                                                      const std::array<PadT, Rank> &left_padding) const
    {
        (void)left_padding;
        std::array<Idx, Rank> ret{coords};
        return ret;
    }
    template <typename PadT>
    inline constexpr std::array<Idx, Rank> pad_coords(const std::array<SIdx, Rank> &coords,
                                                      const std::array<PadT, Rank> &left_padding) const
    {
        (void)left_padding;
        std::array<Idx, Rank> ret{};
        for (size_t i = 0; i < Rank; ++i) ret[i] = Idx(coords[i]);
        return ret;
    }
};

template <Idx Rank> class Padding {
  public:
    static constexpr unsigned int is_padded = 1;
    template <typename PadT>
    inline const std::array<Idx, Rank> pad_coords(const std::array<Idx, Rank> &coords,
                                                  const std::array<PadT, Rank> &left_padding) const
    {
        std::array<Idx, Rank> ret{};
        for (size_t i = 0; i < Rank; ++i) ret[i] = left_padding[i] + coords[i];
        return ret;
    }
    template <typename PadT>
    inline const std::array<Idx, Rank> pad_coords(const std::array<SIdx, Rank> &coords,
                                                  const std::array<PadT, Rank> &left_padding) const
    {
        std::array<Idx, Rank> ret{};
        for (size_t i = 0; i < Rank; ++i) ret[i] = left_padding[i] + Idx(coords[i]);
        return ret;
    }
};

namespace hnnx {

// memory_layout.h:30 原文
static inline constexpr bool is_power_of_two(unsigned long in)
{
    return (in > 0) && ((in & (in - 1)) == 0);
}

// memory_layout.h:47-53 原文 (pre-C++17 std::array constexpr 规避)
template <typename T, size_t Rank, size_t... I>
static inline constexpr std::array<T, Rank> make_stdarray_helper(const T val, std::index_sequence<I...>)
{
    std::array<T, Rank> out = {((void)I, val)...};
    return out;
}
template <typename T, size_t Rank> static inline constexpr std::array<T, Rank> make_stdarray(const T val)
{
    return make_stdarray_helper<T, Rank>(val, std::make_index_sequence<Rank>{});
}

} // namespace hnnx

// ---------------------------------------------------------------------------
// IChunkedMemoryLayout — 递归模板 (memory_layout.h:79-225 原文, 全局作用域)
// ---------------------------------------------------------------------------
template <size_t... Stuff> struct IChunkedMemoryLayout;

// 递归基: 最小 chunk = 1 元素
template <size_t RankVal, size_t IndirRanks> struct IChunkedMemoryLayout<RankVal, IndirRanks> {
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = IndirRanks;
    static constexpr std::array<size_t, Rank> ChunkSizes = hnnx::make_stdarray<size_t, Rank>(1);
    static constexpr size_t chunk_total = 1;
    static constexpr bool is_valid_chunk = true;
    static constexpr unsigned inner_dim = 99; // invalid value

    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> &padded_coords,
                                                const std::array<size_t, Rank> &dims_total)
    {
        (void)padded_coords;
        (void)dims_total;
        return 0;
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> &padded_coords,
                                                 const std::array<size_t, Rank> &dims_total)
    {
        (void)padded_coords;
        (void)dims_total;
        return 0;
    }
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> &padded_coords,
                                               const std::array<size_t, Rank> &dims_total, size_t offset = 0)
    {
        (void)padded_coords;
        (void)dims_total;
        return offset;
    }
    static inline constexpr std::array<size_t, Rank> pad_dims(const std::array<size_t, Rank> dims_in)
    {
        return dims_in;
    }
};

// 递归步: 追加 (Dim, ChunkSize) 一级
template <size_t RankVal, size_t IndirRanks, size_t Dim, size_t ChunkSize, size_t... Rest>
struct IChunkedMemoryLayout<RankVal, IndirRanks, Dim, ChunkSize, Rest...> {
    using Smaller = IChunkedMemoryLayout<RankVal, IndirRanks, Rest...>;
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = IndirRanks;
    static_assert(Dim < RankVal);
    static_assert(RankVal >= IndirRanks);
    static_assert((ChunkSize == 0) || hnnx::is_power_of_two(ChunkSize));
    static_assert((ChunkSize == 0) || Smaller::is_valid_chunk);
    static constexpr bool is_valid_chunk = ((ChunkSize > 0) && (Smaller::is_valid_chunk));
    // inner_dim: 最后一个以 chunksize 0 出现的 (最快变化的"外维")
    static constexpr unsigned inner_dim = ((ChunkSize == 0) && Smaller::is_valid_chunk) ? Dim : Smaller::inner_dim;
    static constexpr std::array<size_t, Rank> embiggen_chunksize(const std::array<size_t, Rank> smaller_chunksize)
    {
        std::array<size_t, Rank> out = smaller_chunksize;
        if (ChunkSize) std::get<Dim>(out) *= ChunkSize;
        return out;
    }
    static constexpr std::array<size_t, Rank> ChunkSizes = embiggen_chunksize(Smaller::ChunkSizes);
    static constexpr size_t chunk_total = ChunkSize ? Smaller::chunk_total * ChunkSize : Smaller::chunk_total;

    /* 块内偏移 (memory_layout.h:137-158 原文) */
    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> &padded_coords,
                                                const std::array<size_t, Rank> &dims_total, //
                                                size_t block_off = 0)
    {
        if constexpr (ChunkSize > 0) {
            const size_t smaller_offset = Smaller::chunk_offset(padded_coords, dims_total);
            const size_t dim_coord = padded_coords[Dim];
            const size_t smaller_idx = dim_coord / std::get<Dim>(Smaller::ChunkSizes);
            const size_t thischunk_smaller_idx = smaller_idx % ChunkSize;
            const size_t smaller_chunk_total = Smaller::chunk_total;
            return block_off * chunk_total + thischunk_smaller_idx * smaller_chunk_total + smaller_offset;
        } else if constexpr (Dim < IndirRanks) {
            // 间接维, 不参与块内偏移
            size_t const chunk_off = Smaller::chunk_offset(padded_coords, dims_total, block_off);
            return chunk_off;
        } else {
            // 块内跨 chunk 偏移
            block_off *= std::get<Dim>(dims_total) / std::get<Dim>(ChunkSizes);
            block_off += std::get<Dim>(padded_coords) / std::get<Dim>(ChunkSizes);
            size_t const chunk_off = Smaller::chunk_offset(padded_coords, dims_total, block_off);
            return chunk_off;
        }
    }
    /* 块表索引 (memory_layout.h:164-180 原文) */
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> &padded_coords,
                                               const std::array<size_t, Rank> &dims_total, size_t offset = 0)
    {
        if constexpr (is_valid_chunk) {
            return offset;
        } else if constexpr (Dim >= IndirRanks) {
            return Smaller::chunk_index(padded_coords, dims_total, offset);
        } else {
            offset *= std::get<Dim>(dims_total) / std::get<Dim>(ChunkSizes);
            offset += std::get<Dim>(padded_coords) / std::get<Dim>(ChunkSizes);
            size_t const chunk_idx = Smaller::chunk_index(padded_coords, dims_total, offset);
            return chunk_idx;
        }
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> &padded_coords,
                                                 const std::array<size_t, Rank> &dims_total)
    {
        const size_t offset = chunk_offset(padded_coords, dims_total);
        const size_t index = chunk_index(padded_coords, dims_total);
        return index * chunk_total + offset;
    }
    static inline std::array<size_t, Rank> pad(const std::array<size_t, Rank> dims_in)
    {
        std::array<size_t, Rank> newdims;
        for (size_t i = 0; i < Rank; i++) {
            auto dim_chunk_size = ChunkSizes[i];
            newdims[i] = ((dims_in[i] + (dim_chunk_size - 1)) & (~(dim_chunk_size - 1)));
        }
        return newdims;
    }
    // 块表长度
    static inline size_t num_blocks(const std::array<size_t, Rank> max_dims)
    {
        size_t blocks = 1;
        for (size_t i = 0; i < IndirRanks; i++) {
            auto dim_chunk_size = ChunkSizes[i];
            blocks *= max_dims[i] / dim_chunk_size;
        }
        return blocks;
    }
    // 块内元素总数
    static inline size_t block_total(const std::array<size_t, Rank> max_dims)
    {
        size_t blocks = 1;
        for (size_t i = 0; i < Rank; i++) {
            auto const dim_chunk_size = ChunkSizes[i];
            if (i < IndirRanks)
                blocks *= dim_chunk_size;
            else
                blocks *= max_dims[i];
        }
        return blocks;
    }
};

// ChunkedMemoryLayout = IDirRanks = Rank (memory_layout.h:224 原文)
template <size_t Rank, size_t... Etc> //
using ChunkedMemoryLayout = IChunkedMemoryLayout<Rank, Rank, Etc...>;

// ---------------------------------------------------------------------------
// FlatMemoryLayout (memory_layout.h:231-263 原文)
// ---------------------------------------------------------------------------
template <size_t RankVal> struct FlatMemoryLayout {
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = RankVal; // to be consistent; only applies when chunk_total > 1.
    static constexpr std::array<size_t, Rank> ChunkSizes = hnnx::make_stdarray<size_t, Rank>(1);
    static constexpr size_t chunk_total = 1;
    static constexpr unsigned inner_dim = Rank - 1;
    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> &padded_coords,
                                                const std::array<size_t, Rank> &dims_total)
    {
        (void)padded_coords;
        (void)dims_total;
        return 0;
    }
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> &padded_coords,
                                               const std::array<size_t, Rank> &dims_total)
    {
        size_t offset = padded_coords[0];
        for (size_t i = 1; i < Rank; i++) {
            offset = offset * dims_total[i] + padded_coords[i];
        }
        return offset;
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> &padded_coords,
                                                 const std::array<size_t, Rank> &dims_total)
    {
        return chunk_index(padded_coords, dims_total);
    }
    static inline constexpr std::array<size_t, Rank> pad(const std::array<size_t, Rank> dims_in) { return dims_in; }

    static inline size_t num_blocks(const std::array<size_t, Rank> max_dims)
    {
        size_t blocks = max_dims[0];
        for (size_t i = 1; i < Rank; i++) {
            blocks *= max_dims[i];
        }
        return blocks;
    }
};
using R4FlatMemoryLayout = FlatMemoryLayout<4>; // NHWC
using R5FlatMemoryLayout = FlatMemoryLayout<5>; // NHWDC
using R6FlatMemoryLayout = FlatMemoryLayout<6>;

// ---------------------------------------------------------------------------
// SingularMemoryLayout (memory_layout.h:268-296 原文) — 维度被忽略的单值布局
// ---------------------------------------------------------------------------
template <size_t RankVal> struct SingularMemoryLayout {
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = RankVal;
    static constexpr std::array<size_t, Rank> ChunkSizes = hnnx::make_stdarray<size_t, Rank>(1);
    static constexpr size_t chunk_total = 1;
    static constexpr unsigned inner_dim = Rank - 1;
    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> &padded_coords,
                                                const std::array<size_t, Rank> &dims_total)
    {
        (void)padded_coords;
        (void)dims_total;
        return 0;
    }
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> &padded_coords,
                                               const std::array<size_t, Rank> &dims_total)
    {
        (void)padded_coords;
        (void)dims_total;
        return 0;
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> &padded_coords,
                                                 const std::array<size_t, Rank> &dims_total)
    {
        (void)padded_coords;
        (void)dims_total;
        return 0;
    }
    static inline constexpr std::array<size_t, Rank> pad(const std::array<size_t, Rank> dims_in) { return dims_in; }
    static inline size_t num_blocks(const std::array<size_t, Rank> max_dims)
    {
        (void)max_dims;
        return 1;
    }
};
using R4SingularMemoryLayout = SingularMemoryLayout<4>;

// ---------------------------------------------------------------------------
// 具名布局族 (memory_layout.h:300-346 原文, 逐逗号参数)
// ---------------------------------------------------------------------------
class R4NCHWMemoryLayout : public ChunkedMemoryLayout<4, 0, 0, 3, 0, 2, 0, 1, 0> {}; // NCHW
class R4Depth32MemoryLayout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 3, 0, 2, 0, 2, 4, 3, 32> {};

// Croutons for HMX, YYYXXXDDDDD chunks —— M34 三代表之一 (Crouton_8)
class R4CroutonLayout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 8, 3, 32> {};
// Croutons for HMX, YXXXXXDDDDD chunks (wide aspect ratio)
class R4WideCroutonLayout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 2, 2, 32, 3, 32> {};

// Croutons for HMX, YYYXDDDDDXX chunks
class R4Crouton4x1Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 2, 3, 32, 2, 4> {};

// Croutons for HMX, YYXXDDDDDYX chunks
class R4Crouton2x2Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 4, 2, 4, 3, 32, 1, 2, 2, 2> {};

// Croutons for HMX, YYXXDDDDDYX chunks (wide aspect ratio)
class R4WideCrouton2x2Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 2, 16, 3, 32, 1, 2, 2, 2> {};

// Croutons2 for HMX, 8x4x32 chunks where the data is 16b —— Crouton_16 布局
class R4Crouton2Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 2, 3, 32, 2, 2> {};

// AR4 8*32==256 deep 1H
using R4DeepAR4_16bLayout = IChunkedMemoryLayout<4, 3, 0, 0, 1, 0, 2, 0, 3, 0, 2, 2, 3, 32, 2, 2>;
// AR8 chunk format 8x32 == 256 elements / 512B
using R4DeepAR8_16bLayout = IChunkedMemoryLayout<4, 3, 0, 0, 1, 0, 2, 0, 3, 0, 2, 4, 3, 32, 2, 2>;

class R4Crouton4Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 2, 3, 32> {};

class R4WideCrouton4Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 2, 2, 8, 3, 32> {};

class R4Weights8x4Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 3, 0, 2, 0, 0, 8, 1, 4, 2, 16, 3, 32, 2, 2> {};

// ---------------------------------------------------------------------------
// Ldefs::stype_for (tensor_definitions.h:14-28 原文)
// ---------------------------------------------------------------------------
namespace Ldefs {
template <unsigned elbytes> struct stype_for;
template <> struct stype_for<1> {
    typedef uint8_t type;
};
template <> struct stype_for<2> {
    typedef uint16_t type;
};
template <> struct stype_for<4> {
    typedef uint32_t type; // NN_UINT32_T
};
template <> struct stype_for<8> {
    typedef uint64_t type; // NN_UINT64_T
};
} // namespace Ldefs
