#include "hnnx/tiling/tiler.hpp"
#include "hnnx/ops/ops.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace hnnx {

Tiler::Tiler() = default;
Tiler::~Tiler() = default;

// SimpleTiler: basic tiling strategy
// Source: simple_tiler.cc
SimpleTiler::SimpleTiler() = default;
SimpleTiler::~SimpleTiler() = default;

std::vector<TileInfo> SimpleTiler::generate_tiles(
    const Op* op, const Graph* graph, const TilingConfig& config) {

    // Source: simple_tiler.cc
    // Generate tiles by splitting each dimension according to config
    //
    // For a Conv op with output [N, H, W, C]:
    //   Split N by conv_batch_tiling
    //   Split H by conv_height_tiling
    //   Split W by conv_width_tiling
    //   Split C by conv_channel_tiling
    //
    // Total tiles = batch_tiles × height_tiles × width_tiles × channel_tiles

    std::vector<TileInfo> tiles;

    // Determine output dimensions from op
    // In real binary: reads from Op's OutputDef at +0x48
    // For now: use config to determine tile count

    uint32_t batch_tiles   = config.conv_batch_tiling > 0 ? config.conv_batch_tiling : 1;
    uint32_t height_tiles  = config.conv_height_tiling > 0 ? config.conv_height_tiling : 1;
    uint32_t width_tiles   = config.conv_width_tiling > 0 ? config.conv_width_tiling : 1;
    uint32_t channel_tiles = config.conv_channel_tiling > 0 ? config.conv_channel_tiling : 1;

    // Assume default output dims if not available
    uint32_t out_n = 1, out_h = 224, out_w = 224, out_c = 32;

    uint32_t tile_h = (out_h + height_tiles - 1) / height_tiles;
    uint32_t tile_w = (out_w + width_tiles - 1) / width_tiles;
    uint32_t tile_c = (out_c + channel_tiles - 1) / channel_tiles;

    for (uint32_t bn = 0; bn < batch_tiles; ++bn) {
        for (uint32_t bh = 0; bh < height_tiles; ++bh) {
            for (uint32_t bw = 0; bw < width_tiles; ++bw) {
                for (uint32_t bc = 0; bc < channel_tiles; ++bc) {
                    TileInfo tile;
                    tile.shape.rank = 4;
                    tile.shape.dims[0] = 1;
                    tile.shape.dims[1] = tile_h;
                    tile.shape.dims[2] = tile_w;
                    tile.shape.dims[3] = tile_c;
                    tile.nsp_id = 0;
                    tile.vtcm_offset = 0;
                    tile.ddr_offset = 0;
                    tile.is_contiguous = true;
                    tile.needs_dma = (batch_tiles * height_tiles * width_tiles * channel_tiles > 1);
                    tiles.push_back(tile);
                }
            }
        }
    }

    return tiles;
}

uint64_t SimpleTiler::estimate_cost(
    const Op* op, const std::vector<TileInfo>& tiles) const {
    // Cost = sum of tile computation + DMA overhead
    uint64_t total = 0;
    for (const auto& tile : tiles) {
        if (tile.needs_dma) {
            total += tile.ddr_offset / 1024;  // DMA cost
        }
    }
    return total;
}

// Supertiler: merge adjacent tiles for DMA efficiency
// Source: supertile.cc
Supertiler::Supertiler() = default;
Supertiler::~Supertiler() = default;

std::vector<TileInfo> Supertiler::generate_tiles(
    const Op* op, const Graph* graph, const TilingConfig& config) {

    // First generate base tiles using SimpleTiler
    SimpleTiler base;
    auto base_tiles = base.generate_tiles(op, graph, config);

    // Then merge adjacent tiles
    return merge_tiles(base_tiles);
}

uint64_t Supertiler::estimate_cost(
    const Op* op, const std::vector<TileInfo>& tiles) const {
    // Super-tiles have lower DMA overhead due to merging
    return SimpleTiler().estimate_cost(op, tiles) * 0.8;
}

std::vector<TileInfo> Supertiler::merge_tiles(const std::vector<TileInfo>& tiles) const {
    if (tiles.size() <= 1) return tiles;

    std::vector<TileInfo> merged;
    std::vector<bool> used(tiles.size(), false);

    for (size_t i = 0; i < tiles.size(); ++i) {
        if (used[i]) continue;

        TileInfo current = tiles[i];
        used[i] = true;

        // Try to merge with adjacent tiles
        for (size_t j = i + 1; j < tiles.size(); ++j) {
            if (used[j]) continue;
            if (tiles[j].nsp_id == current.nsp_id &&
                tiles[j].is_contiguous == current.is_contiguous) {
                // Merge
                current.ddr_offset = std::min(current.ddr_offset, tiles[j].ddr_offset);
                used[j] = true;
            }
        }
        merged.push_back(current);
    }

    return merged;
}

// ConvTiler: 从 op 的真实 OutputDef 读 N,H,W,C 按 config 切分。
std::vector<TileInfo> ConvTiler::generate_tiles(const Op* op, const Graph*,
                                                  const TilingConfig& config) {
    std::vector<TileInfo> tiles;
    const OutputDef* od = nullptr;
    auto* top = dynamic_cast<const TypicalOp*>(op);
    if (top) od = &top->cached_out_def;

    uint32_t out_n = 1, out_h = 224, out_w = 224, out_c = 32;
    if (od && od->rank >= 4) {
        out_n = static_cast<uint32_t>(od->dims[0]);
        out_h = static_cast<uint32_t>(od->dims[1]);
        out_w = static_cast<uint32_t>(od->dims[2]);
        out_c = static_cast<uint32_t>(od->dims[3]);
    }
    uint32_t ht = config.conv_height_tiling > 0 ? config.conv_height_tiling : 1;
    uint32_t wt = config.conv_width_tiling > 0 ? config.conv_width_tiling : 1;
    uint32_t tile_h = (out_h + ht - 1) / ht;
    uint32_t tile_w = (out_w + wt - 1) / wt;
    for (uint32_t bh = 0; bh < ht; ++bh) {
        for (uint32_t bw = 0; bw < wt; ++bw) {
            TileInfo t{};
            t.shape.rank = 4;
            t.shape.dims[0] = out_n;
            t.shape.dims[1] = (bh == ht - 1) ? (out_h - bh * tile_h) : tile_h;
            t.shape.dims[2] = (bw == wt - 1) ? (out_w - bw * tile_w) : tile_w;
            t.shape.dims[3] = out_c;
            t.is_contiguous = true;
            t.needs_dma = (ht * wt > 1);
            tiles.push_back(t);
        }
    }
    return tiles;
}
uint64_t ConvTiler::estimate_cost(const Op* op, const std::vector<TileInfo>& tiles) const {
    uint64_t total = 0;
    for (const auto& t : tiles) {
        uint64_t mac = uint64_t(t.shape.dims[1]) * t.shape.dims[2] * t.shape.dims[3] * 9;
        total += mac + (t.needs_dma ? 128 : 0);
    }
    return total;
}

// MatMulTiler: 对 [M,K]@[K,N]=[M,N] 按 M,N 切分 (K 通常不切以减少部分和)。
std::vector<TileInfo> MatMulTiler::generate_tiles(const Op* op, const Graph*,
                                                     const TilingConfig& config) {
    std::vector<TileInfo> tiles;
    const OutputDef* od = nullptr;
    auto* top = dynamic_cast<const TypicalOp*>(op);
    if (top) od = &top->cached_out_def;

    uint32_t m = 1, n = 1;
    if (od && od->rank >= 2) {
        m = static_cast<uint32_t>(od->dims[0]);
        n = static_cast<uint32_t>(od->dims[1]);
    }
    // 默认 M,N 各切 2 块 (可被 config 覆盖: 复用 conv_height/width 字段)
    uint32_t mt = config.conv_height_tiling > 0 ? config.conv_height_tiling : 2;
    uint32_t nt = config.conv_width_tiling > 0 ? config.conv_width_tiling : 2;
    uint32_t tile_m = (m + mt - 1) / mt;
    uint32_t tile_n = (n + nt - 1) / nt;
    for (uint32_t bi = 0; bi < mt; ++bi) {
        for (uint32_t bj = 0; bj < nt; ++bj) {
            TileInfo t{};
            t.shape.rank = 2;
            t.shape.dims[0] = (bi == mt - 1) ? (m - bi * tile_m) : tile_m;
            t.shape.dims[1] = (bj == nt - 1) ? (n - bj * tile_n) : tile_n;
            t.is_contiguous = true;
            t.needs_dma = (mt * nt > 1);
            tiles.push_back(t);
        }
    }
    return tiles;
}
uint64_t MatMulTiler::estimate_cost(const Op* op, const std::vector<TileInfo>& tiles) const {
    uint64_t total = 0;
    for (const auto& t : tiles) {
        total += uint64_t(t.shape.dims[0]) * t.shape.dims[1] * 64;  // 估算 K=64
    }
    return total;
}

// TileDistributor: distribute tiles across NSPs
// Source: tile_distribute.cc
TileDistributor::TileDistributor() = default;
TileDistributor::~TileDistributor() = default;

std::vector<std::vector<TileInfo>> TileDistributor::distribute(
    const std::vector<TileInfo>& tiles, uint32_t num_nsps) const {

    std::vector<std::vector<TileInfo>> distribution(num_nsps);

    // Round-robin distribution (basic strategy)
    // More sophisticated: balance by VTCM usage and DMA cost
    for (size_t i = 0; i < tiles.size(); ++i) {
        uint32_t nsp = static_cast<uint32_t>(i % num_nsps);
        distribution[nsp].push_back(tiles[i]);
    }

    return distribution;
}

void TileDistributor::print_stats(
    const std::vector<std::vector<TileInfo>>& distribution) const {
    // Source: tile_distribute_stats.cc
}

// TileExtractor: extract tile data from source tensor
// Source: tile_extract.cc
TileExtractor::TileExtractor() = default;
TileExtractor::~TileExtractor() = default;

void TileExtractor::extract(
    const void* src_data, const TileShape& src_shape,
    void* dst_data, const TileInfo& tile) {
    // Copy tile data from source to destination
    // Handle strided access for non-contiguous tiles
}

// TileConformer: conform tile shape to hardware constraints
// Source: tile_conforming.cc
TileConformer::TileConformer() = default;
TileConformer::~TileConformer() = default;

TileShape TileConformer::conform(
    const TileShape& shape, DType dtype, uint32_t nsp_id) const {
    TileShape result = shape;

    // Conform to HVX/HMX alignment requirements:
    // - 128-byte alignment for HVX
    // - Specific dimension requirements for HMX
    // - VTCM alignment for DMA

    switch (dtype) {
        case DType::Float16:
        case DType::Int16:
            // 2-byte elements, align to 64 elements (128 bytes)
            for (int i = 0; i < 5; ++i) {
                result.dims[i] = (result.dims[i] + 63) & ~63u;
            }
            break;
        case DType::Int8:
        case DType::UInt8:
            // 1-byte elements, align to 128 elements
            for (int i = 0; i < 5; ++i) {
                result.dims[i] = (result.dims[i] + 127) & ~127u;
            }
            break;
        case DType::Int32:
        case DType::Float32:
            // 4-byte elements, align to 32 elements
            for (int i = 0; i < 5; ++i) {
                result.dims[i] = (result.dims[i] + 31) & ~31u;
            }
            break;
        default:
            break;
    }

    return result;
}

// TilingRegistry: singleton registry for tilers
// Source: tiling_registration.cc
TilingRegistry& TilingRegistry::instance() {
    static TilingRegistry registry;
    return registry;
}

void TilingRegistry::register_tiler(const std::string& op_name, std::unique_ptr<Tiler> tiler) {
    tilers_[op_name] = std::move(tiler);
}

Tiler* TilingRegistry::get_tiler(const std::string& op_name) const {
    auto it = tilers_.find(op_name);
    return it != tilers_.end() ? it->second.get() : nullptr;
}

void TilingRegistry::register_callback(
    const std::string& event,
    std::function<void(const TileInfo&)> callback) {
    callbacks_[event].push_back(std::move(callback));
}

// Conv tile cost computation
// Source: conv_tile_cost.cc
uint64_t compute_conv_tile_cost(
    const TileShape& input_shape,
    const TileShape& output_shape,
    const TileShape& weight_shape,
    uint32_t stride, uint32_t padding) {

    // Cost model for convolution tiling
    // Based on: MAC operations + DMA overhead + VTCM pressure

    uint64_t mac_ops = 1;
    for (int i = 0; i < 5; ++i) {
        mac_ops *= output_shape.dims[i];
    }
    mac_ops *= weight_shape.dims[0] * weight_shape.dims[1] * weight_shape.dims[2];

    // DMA cost: input + weight load
    uint64_t input_bytes = 1;
    for (int i = 0; i < 5; ++i) input_bytes *= input_shape.dims[i];

    uint64_t weight_bytes = 1;
    for (int i = 0; i < 5; ++i) weight_bytes *= weight_shape.dims[i];

    return mac_ops + (input_bytes + weight_bytes) / 1024;
}

} // namespace hnnx
