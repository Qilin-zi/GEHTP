/**
 * @file weight_pack.h
 * @brief HTP weight repacking: row-major ggml blocks → tile-major crouton format
 *
 * HMX (Hexagon Matrix eXtension) GEMM consumes weights in a tiled "crouton"
 * layout, not the row-major ggml block layout. This module ports the pure-scalar
 * repack functions from ggml-hexagon (no HVX/Q6 intrinsics) so the host-side
 * compiler can pack weights into the crate blob that the DSP consumes.
 *
 * Tile layout (Q8_0, 1088 bytes = one 32×32 weight tile):
 *   - 1024 bytes of quantized values, arranged as 16 column-pair stripes.
 *     Each stripe is 64 bytes: 32 rows × 2 columns (even/odd interleaved):
 *       [row0_col0, row0_col1, row1_col0, row1_col1, ...]
 *   - 64 bytes of F16 scales (32 rows × 1 scale each).
 *
 * Q4_0 tile (576 bytes):
 *   - 512 bytes of 4-bit quants (16 stripes × 32 bytes, high/low nibble = even/odd col)
 *   - 64 bytes of F16 scales
 *
 * Source: /disk2/temp02/ggml-hexagon/ggml-hexagon.cpp:478-529 (q4_0), 701-747 (q8_0).
 * Vendored block structs from ggml-common.h.
 */

#ifndef QNN_HTP_WEIGHT_PACK_H
#define QNN_HTP_WEIGHT_PACK_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace qnn {
namespace htp {

// --- Tile size constants (from htp/matmul-ops.h) ---
static constexpr size_t WEIGHT_TILE_SIZE_Q8_0 = 1088;  // 1024 quants + 64 scales
static constexpr size_t WEIGHT_TILE_SIZE_Q4_0 = 576;   // 512 quants + 64 scales
static constexpr uint32_t WEIGHT_TILE_ROWS = 32;
static constexpr uint32_t WEIGHT_TILE_COLS = 32;

// --- ggml-compatible quantization block structures ---
// ggml_half = IEEE 754 binary16, stored as uint16_t (host endianness).
using ggml_half = uint16_t;

#define QK8_0 32
// block_q8_0: 1 F16 scale + 32 int8 quants = 34 bytes (matches ggml-common.h).
struct block_q8_0 {
    ggml_half d;       // F16 delta (scale)
    int8_t qs[QK8_0];  // 32 quantized values
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

#define QK4_0 32
// block_q4_0: 1 F16 scale + 16 bytes of packed 4-bit quants = 18 bytes.
struct block_q4_0 {
    ggml_half d;       // F16 delta (scale)
    uint8_t qs[QK4_0 / 2];  // 32 values packed 2-per-byte (low nibble first)
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// Round n up to the next multiple of m (m must be > 0).
static inline uint32_t hex_round_up(uint32_t n, uint32_t m) {
    return m * ((n + m - 1) / m);
}

// Repack row-major Q8_0 weights into tile-major crouton format.
//   src: ne1 rows × (ne0/32) column-blocks of block_q8_0, row-major.
//   dst: destination buffer (must hold tiled_size(ne0, ne1) bytes).
//   ne0: K dimension (must be multiple of 32).
//   ne1: N dimension (output rows; padded to 32 internally).
// Returns total bytes written.
size_t repack_q8_0_tiled(const block_q8_0* src, uint8_t* dst,
                         uint32_t ne0, uint32_t ne1);

// Repack row-major Q4_0 weights into tile-major crouton format.
size_t repack_q4_0_tiled(const block_q4_0* src, uint8_t* dst,
                         uint32_t ne0, uint32_t ne1);

// Inverse: unpack tile-major back to row-major block layout (for round-trip tests).
void unpack_tiled_q8_0(block_q8_0* dst, const uint8_t* src,
                       uint32_t ne0, uint32_t ne1);

void unpack_tiled_q4_0(block_q4_0* dst, const uint8_t* src,
                       uint32_t ne0, uint32_t ne1);

// Total packed size for a ne0×ne1 weight matrix (Q8_0).
static inline size_t tiled_size_q8_0(uint32_t ne0, uint32_t ne1) {
    uint32_t n_col_tiles = hex_round_up(ne1, 32) / 32;
    uint32_t n_k_tiles = hex_round_up(ne0, 32) / 32;
    return static_cast<size_t>(n_col_tiles) * n_k_tiles * WEIGHT_TILE_SIZE_Q8_0;
}

static inline size_t tiled_size_q4_0(uint32_t ne0, uint32_t ne1) {
    uint32_t n_col_tiles = hex_round_up(ne1, 32) / 32;
    uint32_t n_k_tiles = hex_round_up(ne0, 32) / 32;
    return static_cast<size_t>(n_col_tiles) * n_k_tiles * WEIGHT_TILE_SIZE_Q4_0;
}

// --- Pair-interleave crouton position formula (from hvxhmx_libs hmx_crouton.h) ---
// pos(row, col) = (row/2)*64 + 2*col + (row&1)
// Each 128B vector (64 fp16) stores a pair of adjacent rows, element-interleaved.
static inline uint32_t crouton_pos(uint32_t row, uint32_t col) {
    return (row / 2u) * 64u + 2u * col + (row & 1u);
}

// Convert one Q8_0 tile (1088B) from tile-major to crouton pair-interleave.
// Rearranges the 1024 quant bytes; the 64B scale region at offset 1024 is copied unchanged.
void tile_major_to_crouton_q8_0(const uint8_t* src_tile, uint8_t* dst_tile);

// Inverse: crouton pair-interleave → tile-major.
void crouton_to_tile_major_q8_0(const uint8_t* src_tile, uint8_t* dst_tile);

// Convert 32×32 fp16 (1024 elements = 2048 bytes) from row-major to crouton pair-interleave.
void rowmajor_to_crouton_fp16(const uint16_t* src, uint16_t* dst);

// Inverse: crouton pair-interleave → row-major fp16.
void crouton_to_rowmajor_fp16(const uint16_t* src, uint16_t* dst);

// Full pipeline: ggml row-major Q8_0 blocks → crouton pair-interleave tiles.
// Each tile = 1088B (1024 crouton quants + 64B scales), same total size as tile-major.
size_t repack_q8_0_crouton(const block_q8_0* src, uint8_t* dst,
                           uint32_t ne0, uint32_t ne1);

// Inverse: unpack crouton pair-interleave tiles → row-major Q8_0 blocks (for round-trip test).
void unpack_crouton_q8_0(block_q8_0* dst, const uint8_t* src,
                         uint32_t ne0, uint32_t ne1);

// --- F16 helpers (host-side, software conversion for tests) ---
// IEEE 754 binary16 ↔ binary32 conversion (bit-twiddling, no hardware dependency).
uint16_t f32_to_f16_bits(float f);
float f16_bits_to_f32(uint16_t h);

// ============================================================================
// Crate record: the pickle-body section that carries packed weight bytes.
// Format: [n_slots:u32][n_slots × 16B slot][blob].
//   slot = [len:u32][count:u32][offset:u32][addr:u32]
//     len    = byte length of this entry's packed weight data
//     count  = 1 (one tensor per slot)
//     offset = byte offset of this entry's data WITHIN the blob (after slots)
//     addr   = 0 (DSP runtime fill — device virtual address after mmap)
// The blob is the 4-byte-aligned concatenation of all entries' packed bytes.
// ============================================================================

enum class ConvWeightLayout : uint8_t {
    T1_Default = 0,  // stride-1 (and any non-T2/T5 shape): h asc, kw desc
    T2_Stride2 = 1,  // stride-2 3x3 (SpaceToDepth path): (h%2,kw%2) parity
                     // classes row-major; h asc, kw desc within class
    T5_KW0Last = 2,  // 5x5 stride-1: [(h,kw) kw=4..1] then [(h,0)]
    Natural = 3,     // inconv-style: K-major flat rows padded to an even row
                     // count, no tap reorder
};

enum class WeightQuant : uint32_t {
    Q8_0 = 0,
    Q4_0 = 1,
    FP16 = 2,
};

struct CrateEntry {
    uint32_t tensor_id;
    WeightQuant quant;
    uint32_t ne0;
    uint32_t ne1;
    std::vector<uint8_t> packed;
    // Transform applied by pack (E-B conclusion, T-E4). Only meaningful for
    // WeightQuant::FP16 conv entries; ggml quant entries keep the default.
    ConvWeightLayout layout = ConvWeightLayout::T1_Default;
};

struct CrateSlot {
    uint32_t len;
    uint32_t count;
    uint32_t offset;
    uint32_t addr;
};

std::vector<uint8_t> build_crate_record(const std::vector<CrateEntry>& entries);

bool parse_crate_record(const uint8_t* data, size_t size,
                        std::vector<CrateSlot>& slots,
                        std::vector<uint8_t>& blob);

CrateEntry make_crate_entry_q8_0(uint32_t tensor_id, uint32_t ne0, uint32_t ne1,
                                 const block_q8_0* src);
CrateEntry make_crate_entry_q4_0(uint32_t tensor_id, uint32_t ne0, uint32_t ne1,
                                 const block_q4_0* src);
CrateEntry make_crate_entry_q8_0_crouton(uint32_t tensor_id, uint32_t ne0, uint32_t ne1,
                                          const block_q8_0* src);

// ============================================================================
// Phase E T-E4: QNN conv weight packing — crouton interleave-32 tile repack.
//
// Byte layout RE'd from the InceptionV3 golden blob (E-A experiment); the
// reference implementation with full 190/190 golden verification is
// tools/weight_layout_pack.py. Rules:
//   - FP16 values preserved; tile = 32 rows (cin) x 32 cols (cout), rows AND
//     cols zero-padded to 32; pair-interleave: hw[32m+2c]=row(2m)[c],
//     hw[32m+2c+1]=row(2m+1)[c]. 2048 B per tile.
//   - emit order: n0 (cout banks of 32) -> k0 (cin banks of 32) -> tap order.
//   - raw layout: v[((h*KW+kw)*CIN+cin)*COUT + cout].
// ============================================================================

size_t conv_weights_packed_size(uint32_t KH, uint32_t KW, uint32_t CIN,
                                uint32_t COUT, ConvWeightLayout layout);

// Full weight stream: all n0 banks concatenated (no bias interleaving; blob
// assembly is a scheduler-order concern, T-D8/Q2).
size_t pack_conv_weights_fp16(const uint16_t* src, uint32_t KH, uint32_t KW,
                              uint32_t CIN, uint32_t COUT,
                              ConvWeightLayout layout, uint8_t* dst);

// Single n0 bank (the 32-cout slice [n0, min(n0+32,COUT))) — the granularity
// at which the golden interleaves stride-2 bias chunks.
size_t pack_conv_weights_fp16_bank(const uint16_t* src, uint32_t KH, uint32_t KW,
                                   uint32_t CIN, uint32_t COUT,
                                   ConvWeightLayout layout, uint32_t n0,
                                   uint8_t* dst);

// bias: fp16 v -> u32 (v << 16), 4 B per channel.
size_t pack_conv_bias_fp16(const uint16_t* src, uint32_t n, uint8_t* dst);

CrateEntry make_crate_entry_conv_fp16(uint32_t tensor_id, uint32_t KH, uint32_t KW,
                                      uint32_t CIN, uint32_t COUT,
                                      ConvWeightLayout layout, const uint16_t* src);

} // namespace htp
} // namespace qnn

#endif // QNN_HTP_WEIGHT_PACK_H
