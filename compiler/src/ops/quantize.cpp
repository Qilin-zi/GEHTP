#include "hnnx/ops/quantize.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace hnnx {

std::vector<int8_t> Quantizer::quantize_int8(const float* data, size_t count,
                                               const QuantParams& params) const {
    std::vector<int8_t> result(count);
    if (params.type == QuantType::PerTensor) {
        for (size_t i = 0; i < count; ++i) {
            float q = std::round(data[i] / params.scale) + params.offset;
            q = std::clamp(q, -128.0f, 127.0f);
            result[i] = static_cast<int8_t>(q);
        }
    } else if (params.type == QuantType::PerChannel || params.type == QuantType::PerAxis) {
        for (size_t i = 0; i < count; ++i) {
            uint32_t ch = (i / (count / params.per_channel_scales.size())) % params.per_channel_scales.size();
            float scale = params.per_channel_scales[ch];
            float offset = static_cast<float>(params.per_channel_offsets[ch]);
            float q = std::round(data[i] / scale) + offset;
            q = std::clamp(q, -128.0f, 127.0f);
            result[i] = static_cast<int8_t>(q);
        }
    }
    return result;
}

std::vector<int16_t> Quantizer::quantize_int16(const float* data, size_t count,
                                                 const QuantParams& params) const {
    std::vector<int16_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        float q = std::round(data[i] / params.scale) + params.offset;
        q = std::clamp(q, -32768.0f, 32767.0f);
        result[i] = static_cast<int16_t>(q);
    }
    return result;
}

std::vector<int32_t> Quantizer::quantize_int32(const float* data, size_t count,
                                                 const QuantParams& params) const {
    std::vector<int32_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        float q = std::round(data[i] / params.scale) + params.offset;
        result[i] = static_cast<int32_t>(q);
    }
    return result;
}

std::vector<float> Quantizer::dequantize(const void* data, size_t count,
                                            DType dtype, const QuantParams& params) const {
    std::vector<float> result(count);
    if (dtype == DType::Int8) {
        const int8_t* p = static_cast<const int8_t*>(data);
        for (size_t i = 0; i < count; ++i) {
            result[i] = (static_cast<float>(p[i]) - params.offset) * params.scale;
        }
    } else if (dtype == DType::Int16) {
        const int16_t* p = static_cast<const int16_t*>(data);
        for (size_t i = 0; i < count; ++i) {
            result[i] = (static_cast<float>(p[i]) - params.offset) * params.scale;
        }
    } else if (dtype == DType::Int32) {
        const int32_t* p = static_cast<const int32_t*>(data);
        for (size_t i = 0; i < count; ++i) {
            result[i] = (static_cast<float>(p[i]) - params.offset) * params.scale;
        }
    } else if (dtype == DType::UInt8) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < count; ++i) {
            result[i] = (static_cast<float>(p[i]) - params.offset) * params.scale;
        }
    }
    return result;
}

std::vector<int8_t> Quantizer::requantize(const void* input, size_t count,
                                            DType in_dtype, const QuantParams& in_params,
                                            const QuantParams& out_params) const {
    auto float_data = dequantize(input, count, in_dtype, in_params);
    return quantize_int8(float_data.data(), count, out_params);
}

QuantParams Quantizer::compute_params(const float* data, size_t count,
                                        DType target_dtype, QuantType type,
                                        uint32_t axis) const {
    QuantParams params;
    params.type = type;
    params.axis = axis;

    if (target_dtype == DType::Int8) {
        params.bitwidth = 8;
    } else if (target_dtype == DType::Int16) {
        params.bitwidth = 16;
    } else if (target_dtype == DType::Int32) {
        params.bitwidth = 32;
    }

    if (type == QuantType::PerTensor) {
        float min_val = *std::min_element(data, data + count);
        float max_val = *std::max_element(data, data + count);
        float range = std::max(std::abs(min_val), std::abs(max_val));
        if (params.bitwidth == 8) {
            params.scale = range / 127.0f;
            params.offset = 0;
        } else if (params.bitwidth == 16) {
            params.scale = range / 32767.0f;
            params.offset = 0;
        }
        params.is_signed = true;
    }

    return params;
}

bool Quantizer::validate(const QuantParams& params) const {
    if (params.scale <= 0.0f) return false;
    if (params.bitwidth < 4 || params.bitwidth > 32) return false;
    if (params.type == QuantType::PerChannel || params.type == QuantType::PerAxis) {
        if (params.per_channel_scales.empty()) return false;
        for (float s : params.per_channel_scales) {
            if (s <= 0.0f) return false;
        }
    }
    return true;
}

} // namespace hnnx
