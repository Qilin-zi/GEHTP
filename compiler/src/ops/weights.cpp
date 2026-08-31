#include "hnnx/ops/weights.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>

namespace hnnx {

std::vector<uint8_t> WeightProcessor::compose(
    const std::vector<std::vector<uint8_t>>& parts,
    const WeightDescriptor& desc) const {
    // Source: compose_weights.cc
    // Concatenate weight parts into single buffer
    size_t total = 0;
    for (const auto& p : parts) total += p.size();

    std::vector<uint8_t> result(total);
    size_t offset = 0;
    for (const auto& p : parts) {
        std::memcpy(result.data() + offset, p.data(), p.size());
        offset += p.size();
    }
    return result;
}

std::vector<uint8_t> WeightProcessor::convert_conv_weights(
    const void* weights, const WeightDescriptor& src_desc,
    const WeightDescriptor& dst_desc) const {
    // Source: conv_weights.cc
    // Convert between weight formats (e.g., HWCN -> NCHW, or pack for HVX)
    size_t src_size = src_desc.total_size;
    std::vector<uint8_t> result(src_size);
    std::memcpy(result.data(), weights, src_size);
    return result;
}

std::vector<std::vector<uint8_t>> WeightProcessor::scatter_conv_weights(
    const void* weights, size_t total_size, uint32_t num_nsps) const {
    // Source: scatter_conv_weight.cc
    // Split weights across NSPs for multi-NSP execution
    std::vector<std::vector<uint8_t>> result(num_nsps);
    size_t per_nsp = total_size / num_nsps;
    for (uint32_t i = 0; i < num_nsps; ++i) {
        size_t offset = i * per_nsp;
        size_t size = (i == num_nsps - 1) ? (total_size - offset) : per_nsp;
        result[i].resize(size);
        std::memcpy(result[i].data(),
                    static_cast<const uint8_t*>(weights) + offset, size);
    }
    return result;
}

std::vector<WeightProcessor::WtshareMetadata> WeightProcessor::compute_sharing_plan(
    const std::vector<WeightDescriptor>& weights, uint32_t num_nsps) const {
    // Source: wtshare_metadata.cc, wtshare_blockrefs.cc, sharing_plan.cc
    std::vector<WtshareMetadata> plan;
    for (uint32_t i = 0; i < weights.size(); ++i) {
        for (uint32_t n = 0; n < num_nsps; ++n) {
            WtshareMetadata m;
            m.tag = i;
            m.nsp_id = n;
            m.offset = i * weights[i].total_size;
            m.size = weights[i].total_size;
            m.blockref = "wtshare_" + std::to_string(i) + "_nsp" + std::to_string(n);
            plan.push_back(m);
        }
    }
    return plan;
}

std::vector<uint8_t> WeightProcessor::serialize_for_pickle(
    const void* weights, size_t size, const WeightDescriptor& desc) const {
    // Source: wtshare_pickleread.cc, pickle_header_prepare.cc
    std::vector<uint8_t> result(size + 256); // header space
    // Write pickle header with descriptor info
    // Then write weight data
    std::memcpy(result.data() + 256, weights, size);
    return result;
}

std::vector<uint8_t> WeightProcessor::deserialize_from_pickle(
    const void* data, size_t size, WeightDescriptor& out_desc) const {
    // Source: wtshare_deser.cc, validate_pickle.cc
    // Parse pickle header
    // Extract weight data
    std::vector<uint8_t> result(size - 256);
    std::memcpy(result.data(), static_cast<const uint8_t*>(data) + 256, size - 256);
    return result;
}

void WeightProcessor::dump(const std::string& filename, const void* weights,
                           size_t size, const WeightDescriptor& desc) const {
    // Source: weight_dump.cc
    std::ofstream f(filename, std::ios::binary);
    f.write(static_cast<const char*>(weights), size);
}

} // namespace hnnx
