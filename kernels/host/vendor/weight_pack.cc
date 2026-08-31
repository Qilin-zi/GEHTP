/**
 * @file weight_pack.cc
 * @brief HTP weight repacking implementations (ported from ggml-hexagon.cpp).
 *
 * Pure scalar C++ — no HVX/Q6 intrinsics, fully portable to x86_64 host.
 * The tile layout matches what HMX GEMM expects on the DSP.
 */

#include "weight_pack.h"

#include <cstring>
#include <vector>

namespace qnn {
namespace htp {

// ============================================================================
// F16 ↔ F32 conversion (IEEE 754 bit manipulation).
// Used only for test validation (real weight data carries pre-quantized F16
// scales; the repack never converts them).
// ============================================================================
uint16_t f32_to_f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3ff;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x400;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        mant = (mant + (1u << (shift - 1))) >> shift;
        return static_cast<uint16_t>(sign | mant);
    }
    if (exp == 0xff) exp = 0x1f;
    else if (exp > 0x1e) { exp = 0x1f; mant = 0; }
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

float f16_bits_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            int shift = 0;
            uint32_t m = mant;
            while ((m & 0x400) == 0) { m <<= 1; shift++; }
            m &= 0x3ff;
            exp = -14 - shift + 127;
            u = sign | (exp << 23) | (m << 13);
        }
    } else if (exp == 0x1f) {
        u = sign | 0x7f800000 | (mant << 13);
    } else {
        exp = exp - 15 + 127;
        u = sign | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ============================================================================
// repack_q8_0_tiled — row-major block_q8_0 → tile-major crouton (1088B/tile).
//
// Ported from /disk2/temp02/ggml-hexagon/ggml-hexagon.cpp:701-747.
// ne0 = K (must be multiple of 32), ne1 = N (padded to 32 internally).
// ============================================================================
size_t repack_q8_0_tiled(const block_q8_0* src, uint8_t* dst,
                         uint32_t ne0, uint32_t ne1) {
    const uint32_t ne0_padded = hex_round_up(ne0, 32);
    const uint32_t ne1_padded = hex_round_up(ne1, 32);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const uint32_t n_k_blocks = ne0 / 32;  // src blocks per row

    for (uint32_t ct = 0; ct < n_col_tiles; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            uint8_t* tile_dst = dst + (static_cast<size_t>(ct) * n_k_tiles + kt) * WEIGHT_TILE_SIZE_Q8_0;

            // 16 column-pair stripes, each 64 bytes = 32 rows × 2 cols.
            for (uint32_t cp = 0; cp < 16; cp++) {
                const uint32_t col0 = cp * 2;
                const uint32_t col1 = col0 + 1;
                for (uint32_t row = 0; row < 32; row++) {
                    const uint32_t r = ct * 32 + row;
                    const block_q8_0* b =
                        (r < ne1 && kt < n_k_blocks) ? &src[r * n_k_blocks + kt] : nullptr;
                    tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                }
            }

            // 32 F16 scales at tile offset 1024.
            ggml_half* scale_dst = reinterpret_cast<ggml_half*>(tile_dst + 1024);
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                scale_dst[row] = (r < ne1 && kt < n_k_blocks)
                                 ? src[r * n_k_blocks + kt].d : 0;
            }
        }
    }
    return static_cast<size_t>(n_col_tiles) * n_k_tiles * WEIGHT_TILE_SIZE_Q8_0;
}

// ============================================================================
// repack_q4_0_tiled — row-major block_q4_0 → tile-major crouton (576B/tile).
//
// Ported from /disk2/temp02/ggml-hexagon/ggml-hexagon.cpp:478-529.
// Q4_0 packs 32 values as 16 bytes (2 per byte, low nibble = even index).
// The tile re-packs them into 16 stripes × 32 bytes with even/odd column
// in low/high nibble.
// ============================================================================
size_t repack_q4_0_tiled(const block_q4_0* src, uint8_t* dst,
                         uint32_t ne0, uint32_t ne1) {
    const uint32_t ne0_padded = hex_round_up(ne0, 32);
    const uint32_t ne1_padded = hex_round_up(ne1, 32);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const uint32_t n_k_blocks = ne0 / 32;

    for (uint32_t ct = 0; ct < n_col_tiles; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            uint8_t* tile_dst = dst + (static_cast<size_t>(ct) * n_k_tiles + kt) * WEIGHT_TILE_SIZE_Q4_0;

            // Unpack each row's 32 quants (0..15) into a temp buffer.
            uint8_t tile_quants[32][32];
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                if (r < ne1 && kt < n_k_blocks) {
                    const block_q4_0* x = &src[r * n_k_blocks + kt];
                    for (uint32_t i = 0; i < QK4_0 / 2; i++) {
                        tile_quants[row][i] = x->qs[i] & 0x0f;          // even index (low nibble)
                        tile_quants[row][i + QK4_0 / 2] = x->qs[i] >> 4; // odd index (high nibble)
                    }
                } else {
                    std::memset(tile_quants[row], 8, 32);  // pad with 8 (=0 in symmetric q4)
                }
            }

            // Pack 16 column-pair stripes; each stripe = 32 rows × 1 byte (high=odd, low=even).
            for (uint32_t cp = 0; cp < 16; cp++) {
                for (uint32_t row = 0; row < 32; row++) {
                    tile_dst[cp * 32 + row] =
                        (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                }
            }

            // 32 F16 scales at tile offset 512.
            ggml_half* scale_dst = reinterpret_cast<ggml_half*>(tile_dst + 512);
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                scale_dst[row] = (r < ne1 && kt < n_k_blocks)
                                 ? src[r * n_k_blocks + kt].d : 0;
            }
        }
    }
    return static_cast<size_t>(n_col_tiles) * n_k_tiles * WEIGHT_TILE_SIZE_Q4_0;
}

// ============================================================================
// Inverse functions (tile-major → row-major) for round-trip validation.
// ============================================================================
void unpack_tiled_q8_0(block_q8_0* dst, const uint8_t* src,
                       uint32_t ne0, uint32_t ne1) {
    const uint32_t ne0_padded = hex_round_up(ne0, 32);
    const uint32_t ne1_padded = hex_round_up(ne1, 32);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const uint32_t n_k_blocks = ne0 / 32;

    for (uint32_t ct = 0; ct < n_col_tiles; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            const uint8_t* tile_src = src + (static_cast<size_t>(ct) * n_k_tiles + kt) * WEIGHT_TILE_SIZE_Q8_0;
            for (uint32_t cp = 0; cp < 16; cp++) {
                const uint32_t col0 = cp * 2;
                const uint32_t col1 = col0 + 1;
                for (uint32_t row = 0; row < 32; row++) {
                    const uint32_t r = ct * 32 + row;
                    if (r < ne1 && kt < n_k_blocks) {
                        block_q8_0& b = dst[r * n_k_blocks + kt];
                        b.qs[col0] = static_cast<int8_t>(tile_src[cp * 64 + 2 * row + 0]);
                        b.qs[col1] = static_cast<int8_t>(tile_src[cp * 64 + 2 * row + 1]);
                    }
                }
            }
            const ggml_half* scale_src = reinterpret_cast<const ggml_half*>(tile_src + 1024);
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                if (r < ne1 && kt < n_k_blocks) {
                    dst[r * n_k_blocks + kt].d = scale_src[row];
                }
            }
        }
    }
}

void unpack_tiled_q4_0(block_q4_0* dst, const uint8_t* src,
                       uint32_t ne0, uint32_t ne1) {
    const uint32_t ne0_padded = hex_round_up(ne0, 32);
    const uint32_t ne1_padded = hex_round_up(ne1, 32);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const uint32_t n_k_blocks = ne0 / 32;

    for (uint32_t ct = 0; ct < n_col_tiles; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            const uint8_t* tile_src = src + (static_cast<size_t>(ct) * n_k_tiles + kt) * WEIGHT_TILE_SIZE_Q4_0;
            // Unpack tile bytes → tile_quants[32][32].
            uint8_t tile_quants[32][32];
            for (uint32_t cp = 0; cp < 16; cp++) {
                for (uint32_t row = 0; row < 32; row++) {
                    uint8_t byte = tile_src[cp * 32 + row];
                    tile_quants[row][2 * cp] = byte & 0x0f;
                    tile_quants[row][2 * cp + 1] = (byte >> 4) & 0x0f;
                }
            }
            // Reconstruct block qs[i]: low nibble = tile_quants[row][i],
            // high nibble = tile_quants[row][i + 16] (inverse of unpack_q4_0_quants).
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                if (r < ne1 && kt < n_k_blocks) {
                    block_q4_0& b = dst[r * n_k_blocks + kt];
                    for (uint32_t i = 0; i < QK4_0 / 2; i++) {
                        b.qs[i] = (tile_quants[row][i] & 0x0f) |
                                  ((tile_quants[row][i + QK4_0 / 2] & 0x0f) << 4);
                    }
                }
            }
            const ggml_half* scale_src = reinterpret_cast<const ggml_half*>(tile_src + 512);
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                if (r < ne1 && kt < n_k_blocks) {
                    dst[r * n_k_blocks + kt].d = scale_src[row];
                }
            }
        }
    }
}

// ============================================================================
// Tile-major ↔ crouton pair-interleave conversion.
//
// Tile-major layout (from repack_q8_0_tiled): 1024 quant bytes organized as
// 16 column-pair stripes × 64 bytes. Element (row, col) is at byte:
//   tile_major_pos(row, col) = (col/2)*64 + row*2 + (col%2)
//
// Crouton pair-interleave layout: Element (row, col) is at byte:
//   crouton_pos(row, col) = (row/2)*64 + 2*col + (row&1)
//
// The 64-byte scale region (32 F16 scales at tile offset 1024) is identical
// in both layouts (one per row, consecutive).
// ============================================================================

static inline uint32_t tile_major_pos(uint32_t row, uint32_t col) {
    return (col / 2u) * 64u + row * 2u + (col & 1u);
}

void tile_major_to_crouton_q8_0(const uint8_t* src_tile, uint8_t* dst_tile) {
    // Rearrange 1024 quant bytes from tile-major to crouton pair-interleave.
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 32; col++) {
            dst_tile[crouton_pos(row, col)] = src_tile[tile_major_pos(row, col)];
        }
    }
    // Copy 64-byte scale region unchanged.
    std::memcpy(dst_tile + 1024, src_tile + 1024, 64);
}

void crouton_to_tile_major_q8_0(const uint8_t* src_tile, uint8_t* dst_tile) {
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 32; col++) {
            dst_tile[tile_major_pos(row, col)] = src_tile[crouton_pos(row, col)];
        }
    }
    std::memcpy(dst_tile + 1024, src_tile + 1024, 64);
}

void rowmajor_to_crouton_fp16(const uint16_t* src, uint16_t* dst) {
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 32; col++) {
            dst[crouton_pos(row, col)] = src[row * 32 + col];
        }
    }
}

void crouton_to_rowmajor_fp16(const uint16_t* src, uint16_t* dst) {
    for (uint32_t row = 0; row < 32; row++) {
        for (uint32_t col = 0; col < 32; col++) {
            dst[row * 32 + col] = src[crouton_pos(row, col)];
        }
    }
}

// ============================================================================
// repack_q8_0_crouton — row-major block_q8_0 → crouton pair-interleave tiles.
// Same output size as repack_q8_0_tiled (1088B per tile), but quant bytes
// are in crouton pair-interleave order instead of tile-major stripe order.
// ============================================================================
size_t repack_q8_0_crouton(const block_q8_0* src, uint8_t* dst,
                           uint32_t ne0, uint32_t ne1) {
    const uint32_t ne0_padded = hex_round_up(ne0, 32);
    const uint32_t ne1_padded = hex_round_up(ne1, 32);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const uint32_t n_k_blocks = ne0 / 32;

    for (uint32_t ct = 0; ct < n_col_tiles; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            uint8_t* tile_dst = dst + (static_cast<size_t>(ct) * n_k_tiles + kt) * WEIGHT_TILE_SIZE_Q8_0;

            // Write quants directly in crouton pair-interleave order.
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                const block_q8_0* b_row =
                    (r < ne1 && kt < n_k_blocks) ? &src[r * n_k_blocks + kt] : nullptr;

                for (uint32_t col = 0; col < 32; col++) {
                    tile_dst[crouton_pos(row, col)] = b_row ? b_row->qs[col] : 0;
                }
            }

            // 32 F16 scales at tile offset 1024 (same as tile-major).
            ggml_half* scale_dst = reinterpret_cast<ggml_half*>(tile_dst + 1024);
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                scale_dst[row] = (r < ne1 && kt < n_k_blocks)
                                 ? src[r * n_k_blocks + kt].d : 0;
            }
        }
    }
    return static_cast<size_t>(n_col_tiles) * n_k_tiles * WEIGHT_TILE_SIZE_Q8_0;
}

void unpack_crouton_q8_0(block_q8_0* dst, const uint8_t* src,
                         uint32_t ne0, uint32_t ne1) {
    const uint32_t ne0_padded = hex_round_up(ne0, 32);
    const uint32_t ne1_padded = hex_round_up(ne1, 32);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const uint32_t n_k_blocks = ne0 / 32;

    for (uint32_t ct = 0; ct < n_col_tiles; ct++) {
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            const uint8_t* tile_src = src + (static_cast<size_t>(ct) * n_k_tiles + kt) * WEIGHT_TILE_SIZE_Q8_0;

            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                if (r < ne1 && kt < n_k_blocks) {
                    block_q8_0& b = dst[r * n_k_blocks + kt];
                    for (uint32_t col = 0; col < 32; col++) {
                        b.qs[col] = static_cast<int8_t>(tile_src[crouton_pos(row, col)]);
                    }
                }
            }
            const ggml_half* scale_src = reinterpret_cast<const ggml_half*>(tile_src + 1024);
            for (uint32_t row = 0; row < 32; row++) {
                const uint32_t r = ct * 32 + row;
                if (r < ne1 && kt < n_k_blocks) {
                    dst[r * n_k_blocks + kt].d = scale_src[row];
                }
            }
        }
    }
}

// ============================================================================
// Crate record build/parse.
// ============================================================================
static inline void put_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}

static inline uint32_t get_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

std::vector<uint8_t> build_crate_record(const std::vector<CrateEntry>& entries) {
    std::vector<uint8_t> buf;
    const uint32_t n_slots = static_cast<uint32_t>(entries.size());
    put_u32(buf, n_slots);

    // First pass: compute blob offsets (4-byte aligned each).
    std::vector<uint32_t> offsets(n_slots);
    uint32_t cursor = 0;
    for (uint32_t i = 0; i < n_slots; i++) {
        offsets[i] = cursor;
        cursor += static_cast<uint32_t>(entries[i].packed.size());
        // 4-byte align
        cursor = (cursor + 3u) & ~3u;
    }

    // Slot table: [len][count=1][offset][addr=0].
    for (uint32_t i = 0; i < n_slots; i++) {
        put_u32(buf, static_cast<uint32_t>(entries[i].packed.size()));
        put_u32(buf, 1);
        put_u32(buf, offsets[i]);
        put_u32(buf, 0);  // addr: runtime fill
    }

    // Blob: concatenated packed bytes, each 4-byte aligned.
    for (uint32_t i = 0; i < n_slots; i++) {
        const auto& pk = entries[i].packed;
        buf.insert(buf.end(), pk.begin(), pk.end());
        while (buf.size() % 4 != 0) buf.push_back(0);
    }
    return buf;
}

bool parse_crate_record(const uint8_t* data, size_t size,
                        std::vector<CrateSlot>& slots,
                        std::vector<uint8_t>& blob) {
    slots.clear();
    blob.clear();
    if (size < 4) return false;
    size_t pos = 0;
    const uint32_t n_slots = get_u32(data + pos);
    pos += 4;
    if (n_slots > (size - pos) / 16) return false;  // slot table overflow

    slots.resize(n_slots);
    for (uint32_t i = 0; i < n_slots; i++) {
        slots[i].len    = get_u32(data + pos + 0);
        slots[i].count  = get_u32(data + pos + 4);
        slots[i].offset = get_u32(data + pos + 8);
        slots[i].addr   = get_u32(data + pos + 12);
        pos += 16;
    }
    // Remaining bytes = blob.
    if (pos <= size) {
        blob.assign(data + pos, data + size);
    }
    return true;
}

CrateEntry make_crate_entry_q8_0(uint32_t tensor_id, uint32_t ne0, uint32_t ne1,
                                 const block_q8_0* src) {
    CrateEntry e;
    e.tensor_id = tensor_id;
    e.quant = WeightQuant::Q8_0;
    e.ne0 = ne0;
    e.ne1 = ne1;
    e.packed.resize(tiled_size_q8_0(ne0, ne1));
    repack_q8_0_tiled(src, e.packed.data(), ne0, ne1);
    return e;
}

CrateEntry make_crate_entry_q4_0(uint32_t tensor_id, uint32_t ne0, uint32_t ne1,
                                 const block_q4_0* src) {
    CrateEntry e;
    e.tensor_id = tensor_id;
    e.quant = WeightQuant::Q4_0;
    e.ne0 = ne0;
    e.ne1 = ne1;
    e.packed.resize(tiled_size_q4_0(ne0, ne1));
    repack_q4_0_tiled(src, e.packed.data(), ne0, ne1);
    return e;
}

CrateEntry make_crate_entry_q8_0_crouton(uint32_t tensor_id, uint32_t ne0, uint32_t ne1,
                                          const block_q8_0* src) {
    CrateEntry e;
    e.tensor_id = tensor_id;
    e.quant = WeightQuant::Q8_0;
    e.ne0 = ne0;
    e.ne1 = ne1;
    e.packed.resize(tiled_size_q8_0(ne0, ne1));
    repack_q8_0_crouton(src, e.packed.data(), ne0, ne1);
    return e;
}

// ============================================================================
// Phase E T-E4: conv weight packing (crouton interleave-32).
// Transliteration of tools/weight_layout_pack.py (190/190 golden-verified).
// ============================================================================

namespace {

struct TapOrder {
    uint32_t h[64];
    uint32_t kw[64];
    uint32_t count;
};

TapOrder conv_tap_order(ConvWeightLayout layout, uint32_t KH, uint32_t KW) {
    TapOrder t{};
    uint32_t n = 0;
    auto push = [&](uint32_t hh, uint32_t kk) {
        t.h[n] = hh;
        t.kw[n] = kk;
        n++;
    };
    switch (layout) {
    case ConvWeightLayout::T2_Stride2:
        for (uint32_t rh = 0; rh < 2; rh++) {
            for (uint32_t rw = 0; rw < 2; rw++) {
                for (uint32_t h = 0; h < KH; h++) {
                    if (h % 2 != rh) continue;
                    for (uint32_t k = KW; k-- > 0;) {
                        if (k % 2 == rw) push(h, k);
                    }
                }
            }
        }
        break;
    case ConvWeightLayout::T5_KW0Last:
        for (uint32_t h = 0; h < KH; h++)
            for (uint32_t k = KW; k-- > 1;) push(h, k);
        for (uint32_t h = 0; h < KH; h++) push(h, 0);
        break;
    case ConvWeightLayout::Natural:
    case ConvWeightLayout::T1_Default:
    default:
        for (uint32_t h = 0; h < KH; h++)
            for (uint32_t k = KW; k-- > 0;) push(h, k);
        break;
    }
    t.count = n;
    return t;
}

// One 32x32 fp16 tile (2048 B). rows = cin bank, cols = cout bank, both
// zero-padded to 32; pair-interleaved as hw[32m+2c]=row(2m)[c],
// hw[32m+2c+1]=row(2m+1)[c].
void emit_conv_tile32(const uint16_t* v, uint32_t KW, uint32_t CIN, uint32_t COUT,
                      uint32_t h, uint32_t kw, uint32_t n0, uint32_t k0,
                      uint16_t* dst) {
    uint16_t rows[32][32];
    const uint32_t ncols = (n0 + 32 <= COUT) ? 32 : COUT - n0;
    uint32_t nrows = 0;
    for (uint32_t cin = k0; cin < k0 + 32 && cin < CIN; cin++, nrows++) {
        const size_t base =
            ((static_cast<size_t>(h) * KW + kw) * CIN + cin) * COUT + n0;
        uint16_t* r = rows[nrows];
        for (uint32_t i = 0; i < ncols; i++) r[i] = v[base + i];
        for (uint32_t i = ncols; i < 32; i++) r[i] = 0;
    }
    for (; nrows < 32; nrows++)
        for (uint32_t c = 0; c < 32; c++) rows[nrows][c] = 0;
    for (uint32_t m = 0; m < 16; m++) {
        uint16_t* pair = dst + m * 64;
        for (uint32_t c = 0; c < 32; c++) {
            pair[2 * c] = rows[2 * m][c];
            pair[2 * c + 1] = rows[2 * m + 1][c];
        }
    }
}

} // namespace

size_t conv_weights_packed_size(uint32_t KH, uint32_t KW, uint32_t CIN,
                                uint32_t COUT, ConvWeightLayout layout) {
    if (layout == ConvWeightLayout::Natural) {
        const uint32_t total = KH * KW * CIN * COUT;
        uint32_t nrows = (total + 31) / 32;
        if (nrows & 1) nrows++;
        return static_cast<size_t>(nrows) * 64;
    }
    const uint32_t nb = (COUT + 31) / 32;
    const uint32_t kb = (CIN + 31) / 32;
    return static_cast<size_t>(nb) * kb * (KH * KW) * 2048;
}

size_t pack_conv_weights_fp16_bank(const uint16_t* src, uint32_t KH, uint32_t KW,
                                   uint32_t CIN, uint32_t COUT,
                                   ConvWeightLayout layout, uint32_t n0,
                                   uint8_t* dst) {
    if (layout == ConvWeightLayout::Natural) {
        // Natural mode ignores banks entirely (flat stream).
        return 0;
    }
    const TapOrder t = conv_tap_order(layout, KH, KW);
    uint16_t* out = reinterpret_cast<uint16_t*>(dst);
    for (uint32_t k0 = 0; k0 < CIN; k0 += 32) {
        for (uint32_t i = 0; i < t.count; i++) {
            emit_conv_tile32(src, KW, CIN, COUT, t.h[i], t.kw[i], n0, k0, out);
            out += 1024;
        }
    }
    return reinterpret_cast<uint8_t*>(out) - dst;
}

size_t pack_conv_weights_fp16(const uint16_t* src, uint32_t KH, uint32_t KW,
                              uint32_t CIN, uint32_t COUT,
                              ConvWeightLayout layout, uint8_t* dst) {
    if (layout == ConvWeightLayout::Natural) {
        const size_t total = static_cast<size_t>(KH) * KW * CIN * COUT;
        uint32_t nrows = static_cast<uint32_t>((total + 31) / 32);
        if (nrows & 1) nrows++;
        std::vector<uint16_t> rows(static_cast<size_t>(nrows) * 32, 0);
        for (size_t i = 0; i < total; i++) rows[i] = src[i];
        uint16_t* out = reinterpret_cast<uint16_t*>(dst);
        for (uint32_t m = 0; m < nrows / 2; m++) {
            const uint16_t* a = rows.data() + (2 * m) * 32;
            const uint16_t* b = rows.data() + (2 * m + 1) * 32;
            for (uint32_t c = 0; c < 32; c++) {
                out[2 * c] = a[c];
                out[2 * c + 1] = b[c];
            }
            out += 64;
        }
        return static_cast<size_t>(nrows) * 64;
    }
    const TapOrder t = conv_tap_order(layout, KH, KW);
    uint16_t* out = reinterpret_cast<uint16_t*>(dst);
    for (uint32_t n0 = 0; n0 < COUT; n0 += 32) {
        for (uint32_t k0 = 0; k0 < CIN; k0 += 32) {
            for (uint32_t i = 0; i < t.count; i++) {
                emit_conv_tile32(src, KW, CIN, COUT, t.h[i], t.kw[i], n0, k0, out);
                out += 1024;
            }
        }
    }
    return reinterpret_cast<uint8_t*>(out) - dst;
}

size_t pack_conv_bias_fp16(const uint16_t* src, uint32_t n, uint8_t* dst) {
    uint32_t* out = reinterpret_cast<uint32_t*>(dst);
    for (uint32_t i = 0; i < n; i++) {
        out[i] = static_cast<uint32_t>(src[i]) << 16;
    }
    return static_cast<size_t>(n) * 4;
}

CrateEntry make_crate_entry_conv_fp16(uint32_t tensor_id, uint32_t KH, uint32_t KW,
                                      uint32_t CIN, uint32_t COUT,
                                      ConvWeightLayout layout, const uint16_t* src) {
    CrateEntry e{};
    e.tensor_id = tensor_id;
    e.quant = WeightQuant::FP16;
    e.ne0 = COUT;
    e.ne1 = KH * KW * CIN;
    e.layout = layout;
    e.packed.resize(conv_weights_packed_size(KH, KW, CIN, COUT, layout));
    pack_conv_weights_fp16(src, KH, KW, CIN, COUT, layout, e.packed.data());
    return e;
}

} // namespace htp
} // namespace qnn
