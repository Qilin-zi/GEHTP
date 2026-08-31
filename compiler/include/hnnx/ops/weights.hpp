#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>
#include <string>

namespace hnnx {

// Weight processing system
// Source: compose_weights.cc, conv_weights.cc, scatter_conv_weight.cc
// wtshare_*.cc (7 files), weight_dump.cc

struct WeightDescriptor {
    DType dtype;
    uint32_t rank;
    uint64_t dims[5];
    uint64_t element_size;
    uint64_t total_size;
    bool is_const;
    bool is_shared;
    bool is_composed;
    uint32_t wtshare_tag;
};

class WeightProcessor {
public:
    WeightProcessor() = default;
    ~WeightProcessor() = default;

    // Compose weights from multiple sources
    // Source: compose_weights.cc
    std::vector<uint8_t> compose(const std::vector<std::vector<uint8_t>>& parts,
                                  const WeightDescriptor& desc) const;

    // Convert conv weights to target format
    // Source: conv_weights.cc, conv_weights_opts.cc
    std::vector<uint8_t> convert_conv_weights(const void* weights,
                                                const WeightDescriptor& src_desc,
                                                const WeightDescriptor& dst_desc) const;

    // Scatter conv weights for multi-NSP
    // Source: scatter_conv_weight.cc
    std::vector<std::vector<uint8_t>> scatter_conv_weights(
        const void* weights, size_t total_size,
        uint32_t num_nsps) const;

    // Weight sharing metadata
    // Source: wtshare_metadata.cc, wtshare_blockrefs.cc
    struct WtshareMetadata {
        uint32_t tag;
        uint32_t nsp_id;
        uint64_t offset;
        uint64_t size;
        std::string blockref;
    };
    std::vector<WtshareMetadata> compute_sharing_plan(
        const std::vector<WeightDescriptor>& weights,
        uint32_t num_nsps) const;

    // Weight serialization for pickle
    // Source: wtshare_pickleread.cc, wtshare_deser.cc
    std::vector<uint8_t> serialize_for_pickle(
        const void* weights, size_t size,
        const WeightDescriptor& desc) const;

    std::vector<uint8_t> deserialize_from_pickle(
        const void* data, size_t size,
        WeightDescriptor& out_desc) const;

    // Dump weights for debugging
    // Source: weight_dump.cc
    void dump(const std::string& filename, const void* weights,
              size_t size, const WeightDescriptor& desc) const;
};

} // namespace hnnx
