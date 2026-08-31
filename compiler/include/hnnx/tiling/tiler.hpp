#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

namespace hnnx {

// Tiling system: tiler.cc, simple_tiler.cc, supertile.cc, tile_*.cc
// 183+ strings related to tiling, 259 decompiled functions
// Handles: conv tiling, tile distribution, tile extraction, tile conforming,
//          tile specified shape, tiling callbacks, tiling info, tiling registration

struct TileShape {
    uint32_t dims[5];
    uint32_t rank;
};

struct TileInfo {
    TileShape shape;
    uint32_t nsp_id;
    uint32_t vtcm_offset;
    uint64_t ddr_offset;
    bool is_contiguous;
    bool needs_dma;
};

struct TilingConfig {
    uint32_t conv_batch_tiling = 1;
    uint32_t conv_width_tiling = 1;
    uint32_t conv_height_tiling = 1;
    uint32_t conv_channel_tiling = 1;
    bool central_tiling = false;
    uint32_t central_tiler_conv_batch = 0;
    uint32_t central_tiler_conv_width = 0;
    bool force_conv_fusion = false;
    bool conv_output_dynamic_rescaling = false;
};

// Base tiler: tiler.cc
class Tiler {
public:
    Tiler();
    virtual ~Tiler();

    // Generate tiles for an op
    virtual std::vector<TileInfo> generate_tiles(
        const Op* op,
        const Graph* graph,
        const TilingConfig& config) = 0;

    // Get tiling cost estimate
    virtual uint64_t estimate_cost(
        const Op* op,
        const std::vector<TileInfo>& tiles) const = 0;

    // Get tiler name
    virtual const char* name() const = 0;
};

// Simple tiler: simple_tiler.cc
class SimpleTiler : public Tiler {
public:
    SimpleTiler();
    ~SimpleTiler() override;

    std::vector<TileInfo> generate_tiles(
        const Op* op,
        const Graph* graph,
        const TilingConfig& config) override;

    uint64_t estimate_cost(
        const Op* op,
        const std::vector<TileInfo>& tiles) const override;

    const char* name() const override { return "SimpleTiler"; }
};

// Supertile: supertile.cc - combines multiple tiles for better DMA efficiency
class Supertiler : public Tiler {
public:
    Supertiler();
    ~Supertiler() override;

    std::vector<TileInfo> generate_tiles(
        const Op* op,
        const Graph* graph,
        const TilingConfig& config) override;

    uint64_t estimate_cost(
        const Op* op,
        const std::vector<TileInfo>& tiles) const override;

    const char* name() const override { return "Supertiler"; }

private:
    // Super-tile merges adjacent tiles with same dimensions
    std::vector<TileInfo> merge_tiles(const std::vector<TileInfo>& tiles) const;
};

// ConvTiler: Conv 专用 tiler，从 op 的真实 OutputDef 读取 N,H,W,C 并按
// config 切分 height/width (batch/channel 通常不切)。
class ConvTiler : public Tiler {
public:
    std::vector<TileInfo> generate_tiles(const Op*, const Graph*,
                                         const TilingConfig&) override;
    uint64_t estimate_cost(const Op*, const std::vector<TileInfo>&) const override;
    const char* name() const override { return "ConvTiler"; }
};

// MatMulTiler: MatMul 专用 tiler，对 M,K,N 按 VTCM 友好的块切分。
class MatMulTiler : public Tiler {
public:
    std::vector<TileInfo> generate_tiles(const Op*, const Graph*,
                                         const TilingConfig&) override;
    uint64_t estimate_cost(const Op*, const std::vector<TileInfo>&) const override;
    const char* name() const override { return "MatMulTiler"; }
};

// Tile distribution: tile_distribute.cc
class TileDistributor {
public:
    TileDistributor();
    ~TileDistributor();

    // Distribute tiles across NSPs
    std::vector<std::vector<TileInfo>> distribute(
        const std::vector<TileInfo>& tiles,
        uint32_t num_nsps) const;

    // Statistics: tile_distribute_stats.cc
    void print_stats(const std::vector<std::vector<TileInfo>>& distribution) const;
};

// Tile extraction: tile_extract.cc
class TileExtractor {
public:
    TileExtractor();
    ~TileExtractor();

    // Extract tile data from source tensor
    void extract(
        const void* src_data, const TileShape& src_shape,
        void* dst_data, const TileInfo& tile);
};

// Tile conforming: tile_conforming.cc
class TileConformer {
public:
    TileConformer();
    ~TileConformer();

    // Conform tile shape to hardware constraints
    TileShape conform(
        const TileShape& shape,
        DType dtype,
        uint32_t nsp_id) const;
};

// Tiling registration: tiling_registration.cc
class TilingRegistry {
public:
    static TilingRegistry& instance();

    void register_tiler(const std::string& op_name, std::unique_ptr<Tiler> tiler);
    Tiler* get_tiler(const std::string& op_name) const;

    // Callback registration: tiling_callbacks.cc
    void register_callback(const std::string& event,
                          std::function<void(const TileInfo&)> callback);

private:
    std::unordered_map<std::string, std::unique_ptr<Tiler>> tilers_;
    std::unordered_map<std::string, std::vector<std::function<void(const TileInfo&)>>> callbacks_;
};

// Conv tile cost: conv_tile_cost.cc
uint64_t compute_conv_tile_cost(
    const TileShape& input_shape,
    const TileShape& output_shape,
    const TileShape& weight_shape,
    uint32_t stride, uint32_t padding);

} // namespace hnnx
