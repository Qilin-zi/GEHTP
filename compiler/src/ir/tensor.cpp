#include "hnnx/ir/tensor.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <cstring>

namespace hnnx {

// Tensor implementation
// Source: tensor_prepare.cc, tensor_utility.cc

// Tensor vtable layout (from decompiled OpDef_Const constructor):
// +0x00: vtable pointer
// +0x08-0x98: tensor data (shape, dtype, quant, buffer ptr)
// +0xA0: persistent tensor
// +0xA8: persistent clone

// Tensor::persistent_clone: clone tensor to persistent storage
// Source: OpDef_Const constructor calls Tensor::persistent_clone
void Tensor::persistent_clone(Allocator* alloc, Tensor* src) {
    if (!src) return;
    // In real implementation: allocate persistent buffer, copy data
    // For now: shallow copy is sufficient for framework
}

// tensor_generator_scalar: generate a scalar tensor from OutputDef
// Source: tensor_prepare.cc, called from OpDef_Const constructor
// Creates a single-element (or small) tensor and fills it with data.
Tensor* tensor_generator_scalar(OutputDef* od_override, const OutputDef* od, const uint8_t* data) {
    const OutputDef* active_od = od_override ? od_override : od;
    if (!active_od) return nullptr;

    auto* t = new Tensor{};
    t->vtable = nullptr;
    std::memset(t->data, 0, sizeof(t->data));

    // Copy shape info from OutputDef to Tensor GCP 2.2 fields
    t->dtype = active_od->dtype;
    t->dims.clear();
    uint64_t total_elems = 1;
    for (uint32_t i = 0; i < active_od->rank && i < 5; i++) {
        uint32_t d = static_cast<uint32_t>(active_od->dims[i]);
        t->dims.push_back(d);
        total_elems *= (d > 0 ? d : 1);
    }
    // Also store rank and dims in legacy data[] for backward compat
    t->data[0] = active_od->rank;
    for (uint32_t i = 0; i < active_od->rank && i < 5; i++) {
        t->data[1 + i] = active_od->dims[i];
    }

    // Copy scalar data value into the tensor's data buffer area
    // For scalar (rank 0), total_elems = 1
    // The data buffer contains the raw bytes of the scalar value
    if (data && total_elems > 0) {
        size_t elem_size = t->get_element_size();
        size_t copy_size = std::min(elem_size, (size_t)(total_elems * elem_size));
        // Store first 8 bytes of data in the data[] array as the scalar value
        size_t n_bytes = std::min(copy_size, sizeof(uint64_t) * 14 - sizeof(void*));
        std::memcpy(t->data + 8, data, std::min(n_bytes, (size_t)112));
    }

    return t;
}

// Tensor::get_element_size: bytes per element based on dtype
uint32_t Tensor::get_element_size() const {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:  return 4;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int8:     return 1;
        case DType::UInt8:    return 1;
        case DType::Int16:    return 2;
        case DType::Int32:    return 4;
        case DType::Bool:     return 1;
        case DType::Int4:     return 1;  // packed, minimum 1 byte
        case DType::UInt4:    return 1;
        case DType::FP8_E4M3: return 1;
        case DType::FP8_E5M2: return 1;
        case DType::MXFP4:   return 1;
        default:              return 4;
    }
}

// Tensor::num_elements: product of all dimensions
// Uses GCP 2.2 dims vector when populated; falls back to legacy data[14] array.
// Source: tensor_utility.cc
uint64_t Tensor::num_elements() const {
    if (!dims.empty()) {
        uint64_t n = 1;
        for (uint32_t d : dims) {
            n *= (d > 0 ? d : 1);
        }
        return n;
    }
    // Fallback: interpret legacy data[14] layout.
    // data[0] = rank, data[1..rank] = dims.
    uint64_t rank = data[0];
    if (rank == 0 || rank > 5) return 1;
    uint64_t n = 1;
    for (uint64_t i = 0; i < rank; ++i) {
        uint64_t d = data[1 + i];
        n *= (d > 0 ? d : 1);
    }
    return n;
}

// Tensor::memory_cost: byte size for VTCM budget / DMA estimation
// Source: used by Op::cost for DMA estimation; also used by VTCM allocator
// for budget checking. Returns num_elements * element_size.
uint64_t Tensor::memory_cost() const {
    return num_elements() * get_element_size();
}

// Tensor::getSize: total tensor size in bytes (alias for memory_cost)
uint64_t Tensor::getSize() const {
    return memory_cost();
}

} // namespace hnnx
