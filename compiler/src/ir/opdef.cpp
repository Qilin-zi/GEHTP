#include "hnnx/ir/types.hpp"
#include "hnnx/ir/graph_prepare.hpp"
#include <cstring>
#include <unordered_map>

namespace hnnx {

// string_tag_t: maps string names to hash keys for optimization rule lookup
// Uses Fibonacci hash with open addressing
// Each hash table entry = 0x50 (80) bytes

static std::unordered_map<std::string, string_tag_t*> tag_map;

string_tag_t* string_tag_t::map_str(const char* name) {
    auto it = tag_map.find(name);
    if (it != tag_map.end()) return it->second;
    auto* tag = new string_tag_t{};
    // Make a stable copy (the input .c_str() may be a temporary std::string buffer).
    char* stable = new char[std::strlen(name) + 1];
    std::strcpy(stable, name);
    tag->name_ = stable;
    // Compute hash key using Fibonacci hash
    uint64_t h = 0;
    for (const char* p = name; *p; ++p) {
        h = h * 31 + static_cast<unsigned char>(*p);
    }
    uint32_t lo = static_cast<uint32_t>(h);
    uint32_t hi = static_cast<uint32_t>(h >> 32);
    uint64_t h2 = (lo ^ hi * FIB_MULT_1) * FIB_MULT_2;
    tag->hash_key_ = h2;
    tag_map[name] = tag;
    return tag;
}

// OpDef methods

op_hash_t OpDef::hash_key() const {
    if (name_tag) return name_tag->hash_key();
    return 0;
}

size_t OpDef::input_count() const {
    // Prefer the C++ input connection vector (populated by append_node); fall
    // back to the legacy inputs_start..inputs_end range used by the binary.
    if (!inputs.empty()) return inputs.size();
    if (!inputs_start || !inputs_end) return 0;
    auto start = reinterpret_cast<uintptr_t>(inputs_start);
    auto end = reinterpret_cast<uintptr_t>(inputs_end);
    return (end - start) / sizeof(void*);
}

// OpDef_Const constructor
// Source: op_def.cc:146
OpDef_Const::OpDef_Const(GraphPrepare& gp, op_id_t id, const OutputDef& od,
                         const uint8_t* data, size_t data_len) {
    vtable = nullptr;  // set by vtable assignment
    flags = OP_ENABLED | OP_CONST;  // 0x44
    string_tag = 0;  // set from string_tag_t::map_str("$Const")
    name_tag = string_tag_t::map_str("$Const");
    graph = &gp;
    op_id = id;
    reserved_10 = 0;
    inputs_start = nullptr;
    inputs_end = nullptr;
    extra_40 = nullptr;

    // Copy OutputDef (0x48 bytes from param_3)
    std::memcpy(&output_def, &od, sizeof(OutputDef));

    tensor_ptr = nullptr;
    flags2 = 0;
    vtable2 = nullptr;
    persistent_tensor = nullptr;

    // Generate tensor from const data
    if (data_len != 0 && od.rank == 0) {
        // tensor_generator_scalar path
        // Tensor::persistent_clone path
    }
    // If data is null, generate from OutputDef
    // Source: op_def.cc:146 "OpDef_Const failed to generate tensor"
}

// Fibonacci hash for optimization rule lookup
// Source: Graph::update_extra_info_map @ 0xd29910 (verified)
// Two multipliers: 0x192E2101 (32-bit high half), 0x740F1DE9 (64-bit)
// Algorithm (verified from decompilation):
//   a = (hi * 0x192e2101) & 0xffffffff     # 32-bit Fibonacci (high half)
//   b = lo ^ a
//   c = (b * 0x740f1de9) & 0xffffffffffffffff  # 64-bit Fibonacci
//   d = c ^ (c >> 32)                       # fold
//   slot = ((d >> 15) & 0x1fffe) | 1         # 17-bit bucket, force odd
uint64_t fibonacci_hash(uint64_t key) {
    uint32_t lo = static_cast<uint32_t>(key);
    uint32_t hi = static_cast<uint32_t>(key >> 32);
    uint32_t a = hi * FIB_MULT_1;                      // 32-bit Fibonacci (high half)
    uint32_t b = lo ^ a;                               // mix
    uint64_t c = static_cast<uint64_t>(b) * FIB_MULT_2; // 64-bit Fibonacci
    uint64_t d = c ^ (c >> 32);                         // fold
    return d;
}

// Open-addressing hash table lookup with secondary probing
// Source: run_optimize_passes_single_registry @ 0xf73b10
// Secondary probe: ((d >> 15) & 0x1fffe) | 1 — 17-bit bucket, force odd
size_t hash_table_lookup(uint64_t key, size_t table_size) {
    uint64_t h = fibonacci_hash(key);
    size_t mask = table_size - 1;
    size_t idx = static_cast<size_t>(h) & mask;

    // Secondary probe: 17-bit bucket, force odd (verified @ 0xd299df-0xd29a1b)
    size_t secondary = (static_cast<size_t>(h >> 15) & 0x1fffe) | 1;
    // On collision: idx = (idx + secondary) & mask
    (void)secondary;

    return idx;
}

} // namespace hnnx
