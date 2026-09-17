#pragma once
#include "hnnx/ir/types.hpp"
#include <cstdint>
#include <vector>

namespace hnnx {

// Quantization system
// Source: quantize.cc, requantize.cc, dequantize.cc
// 466 strings related to quantization in .rodata

enum class QuantType : uint32_t {
    None       = 0,
    PerTensor  = 1,
    PerChannel = 2,
    PerAxis    = 3,
    Block      = 4,
    BwAxis     = 5,  // QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET_MAPPED
};

struct QuantParams {
    QuantType type = QuantType::None;
    float scale = 1.0f;
    int32_t offset = 0;
    uint32_t bitwidth = 8;
    uint32_t axis = 0;
    std::vector<float> per_channel_scales;
    std::vector<int32_t> per_channel_offsets;
    bool is_signed = true;
};

class Quantizer {
public:
    Quantizer() = default;
    ~Quantizer() = default;

    // Quantize float data to int8/int16/etc
    // Source: quantize.cc
    std::vector<int8_t> quantize_int8(const float* data, size_t count,
                                       const QuantParams& params) const;
    std::vector<int16_t> quantize_int16(const float* data, size_t count,
                                         const QuantParams& params) const;
    std::vector<int32_t> quantize_int32(const float* data, size_t count,
                                         const QuantParams& params) const;

    // Dequantize int data to float
    // Source: dequantize.cc
    std::vector<float> dequantize(const void* data, size_t count,
                                    DType dtype, const QuantParams& params) const;

    // Requantize: convert between quantization parameters
    // Source: requantize.cc, requantize_opt.cc
    std::vector<int8_t> requantize(const void* input, size_t count,
                                    DType in_dtype, const QuantParams& in_params,
                                    const QuantParams& out_params) const;

    // Compute optimal quantization parameters
    QuantParams compute_params(const float* data, size_t count,
                                DType target_dtype, QuantType type,
                                uint32_t axis = 0) const;

    // Validate quantization parameters
    // Source: "Invalid per-axis quantization passed"
    bool validate(const QuantParams& params) const;
};

} // namespace hnnx
