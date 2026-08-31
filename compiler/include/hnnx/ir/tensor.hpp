#pragma once
#include "hnnx/ir/types.hpp"

namespace hnnx {

class Allocator;

// Tensor: represents a multi-dimensional array
// Source: tensor_prepare.cc, tensor_utility.cc
// Layout from OpDef_Const constructor:
//   +0x00: vtable
//   +0x08-0x98: tensor data (dims, dtype, quant, buffer ptr)
struct Tensor {
    void* vtable;             // +0x00
    uint64_t data[14];        // +0x08-0x98: dims/dtype/quant/buffer ptr

    // GCP 2.2 supplementary fields (real library Tensor metadata)
    uint32_t id = 0;                    // tensor ID
    std::vector<uint32_t> dims;         // shape
    uint32_t dtype = 0;                 // DType
    StorageClass storage_class = DDR;   // storage category
    uint8_t  bank_mask = 0;             // 8-bit bank occupancy mask
    uint32_t vtcm_offset = 0;           // VTCM offset (2KB aligned)
    uint32_t lifetime_start = 0;        // first producer op index
    uint32_t lifetime_end = 0;          // last consumer op index
    std::vector<op_id_t> producers;     // producer ops
    std::vector<op_id_t> consumers;     // consumer ops
    bool is_output = false;
    bool has_side_effect = false;

    static void persistent_clone(Allocator* alloc, Tensor* src);
    uint64_t memory_cost() const;   // byte size for VTCM budget / DMA estimation
    uint64_t num_elements() const;  // product of dimensions
    uint64_t getSize() const;       // num_elements * element_size
    uint32_t get_element_size() const; // bytes per element based on dtype
    bool isVTCM() const { return storage_class == VTCM || storage_class == VTCM_PERSISTENT; }
};

Tensor* tensor_generator_scalar(OutputDef* od_override, const OutputDef* od, const uint8_t* data);

} // namespace hnnx
