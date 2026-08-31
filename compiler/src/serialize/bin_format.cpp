#include "hnnx/serialize/serializer.hpp"
#include <cstring>

namespace hnnx {

// Bin format implementation
// Source: serialize_oplist.cc, graph_auxdata.cc, const_extent_serialize.cc

// .bin format structure (from do_serialize decompilation):
//
// [Config tagged records]
//   - io_dma_bypass (0xEF4D)
//   - spill_fill_instead (0x4453)
//   - extended_udma (0xD446)
//   - io_tensors_config (0xE347)
//   - extra_config (0xD352)
//   - multicast_config (0xD349)
//
// [Runlist segment descriptors]
//   - num_segments (0x5248)
//   - vec_runlist_count (0x524C)
//   - runlist_segment_desc (0x5647)
//   - self_slicing config (0xC953, 0x4650)
//
// [Extra config]
//   - 0xC955, 0xCF55
//   - profiling (0x5350)
//   - mc_cacheable (0x5453)
//
// 0xFA0000FA  -- segment separator
//
// [serialize_io write mode]
//   - IO tensor data
//
// [Runlist ops]
//   - Op::serialize_internal for each op
//
// 0xBEEFF00D  -- end marker

// Tag encoding: (tag & 0xFFFF | tag << 16) ^ 0xFFFF
uint32_t encode_bin_tag(uint32_t tag) {
    return ((tag & 0xFFFF) | (tag << 16)) ^ 0xFFFF;
}

// Tag decoding: inverse of encode
uint32_t decode_bin_tag(uint32_t encoded) {
    uint32_t decoded = encoded ^ 0xFFFF;
    // The low 16 bits and high 16 bits should match
    return decoded & 0xFFFF;
}

// Make plan for deserialization by segments
// Source: make_plan_for_deser_by_segments @ 0x7ED200
// 把 runlist 按 ops_per_segment 切成 num_segments 段，记录每段在序列化
// 流中的字节偏移，供大图增量加载。
void make_plan_for_deser_by_segments(Serializer* ser, int num_segments, uint32_t ops_per_segment) {
    if (!ser || num_segments <= 0) return;
    // 每段记录: [seg_idx 4B][op_count 4B][byte_offset 8B][reserved 8B] = 24B
    std::vector<uint8_t> buf(static_cast<size_t>(num_segments) * 24, 0);
    size_t off = 0;
    uint64_t cur_offset = ser->current_position();
    for (int i = 0; i < num_segments; ++i) {
        uint32_t seg_idx = static_cast<uint32_t>(i);
        uint32_t oc = ops_per_segment;
        std::memcpy(buf.data() + off, &seg_idx, 4); off += 4;
        std::memcpy(buf.data() + off, &oc, 4); off += 4;
        std::memcpy(buf.data() + off, &cur_offset, 8); off += 8;
        off += 8;  // reserved
        cur_offset += ops_per_segment * 64;  // 估算: 每段约 ops*64 字节
    }
    ser->write_tagged_record(0x5347, buf.data(), static_cast<int>(buf.size()));
}

// Blocktable serialization
// Source: blocktable_encode.cc, blocktable_reduce.cc
// The block table maps VTCM block IDs to physical offsets
void serialize_blocktable(Serializer& ser, const void* blocktable, size_t size) {
    // Write block table as tagged record
    // Each entry: [block_id][pool_id][offset][size]
}

// QP record serialization (quantization parameters)
// Source: serialize_qp_record @ 0x7EDE60
void serialize_qp_record(Serializer& ser, const std::vector<uint32_t>& qp_ids) {
    // For each quantization parameter set:
    //   Write scale, offset, bitwidth, axis info
}

// Serialize internal (generic)
// Source: serialize_internal @ 0x7ED750
void serialize_internal(Serializer& ser, const void* data, size_t size) {
    ser.serialize_fwrite(data, size, true);
}

// Get serialization indices for objects
// Source: get_serialization_indices_for_objects @ 0x7EC700
// Maps object pointers to sequential indices for cross-referencing
void get_serialization_indices(const void** objects, size_t count, uint32_t* indices) {
    for (size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<uint32_t>(i);
    }
}

// Deserialize block pointer
// Source: deserialize_block_pointer @ 0x7ECBC0
void* deserialize_block_pointer(Deserializer& deser) {
    // Read block index from stream
    // Look up in block table
    // Return pointer to deserialized block
    return nullptr;
}

// gpe_serialize_to_mem: GPE (Graph Prepared Executable) serialization
// Source: gpe_serialize_to_mem @ 0x7EDB40
void gpe_serialize_to_mem(Serializer& ser, const void* gpe_data, size_t size) {
    // Serialize GPE blob to memory buffer
    // GPE contains the compiled graph ready for execution
}

} // namespace hnnx
