/* w4a16_quant.c — W4A16 act/out 量化面函数 (平台无关, 闭包权威公式)
 *
 * 来源: prepare_owned_inputs.py pack_a16_crouton16_row4_surface +
 *       float_ref.py a16 域契约 (scale=1/32767, offset=-32768)。
 * 铁律: a16 域里 q=0 代表 real=-1.0 — 零行 pad 必须填 32768 (零值点)。
 */
#include <stdint.h>
#include <string.h>

#include "w4a16_quant.h"

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t ex = (h >> 10) & 0x1Fu, mn = h & 0x3FFu;
    uint32_t u;
    if (ex == 0) u = sign;
    else if (ex == 31) u = sign | 0x7F800000u | (mn << 13);
    else u = sign | ((ex - 15 + 127) << 23) | (mn << 13);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint16_t f32_to_f16_rne(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sgn = (u >> 16) & 0x8000u;
    int32_t ex = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mn = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF)
        return (uint16_t)(sgn | 0x7C00u | (mn ? 0x200u : 0u));
    if (((u >> 23) & 0xFF) == 0) return (uint16_t)sgn;
    if (ex >= 31) return (uint16_t)(sgn | 0x7C00u);
    uint32_t half;
    if (ex <= 0) {
        if (ex < -10) return (uint16_t)sgn;
        mn |= 0x800000u;
        int32_t sh = 14 - ex;
        half = mn >> sh;
        uint32_t rm = mn & ((1u << sh) - 1);
        half += (rm > (1u << (sh - 1))) || (rm == (1u << (sh - 1)) && (half & 1));
        return (uint16_t)(sgn | half);
    }
    half = (mn >> 13) & 0x3FFu;
    uint32_t rm = mn & 0x1FFFu;
    half += (rm > 0x1000u) || (rm == 0x1000u && (half & 1));
    if (half == 0x400u) { ex++; half = 0; }
    if (ex >= 31) return (uint16_t)(sgn | 0x7C00u);
    return (uint16_t)(sgn | ((uint32_t)ex << 10) | half);
}

float w4a16_act_scale(const uint16_t* a_f16, uint32_t n) {
    float mx = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float v = f16_to_f32(a_f16[i]);
        if (v < 0.0f) v = -v;
        if (v > mx) mx = v;
    }
    return mx > 0.0f ? mx : 1.0f;
}

uint16_t w4a16_quant_f16(uint16_t a_f16, float scale) {
    float v = f16_to_f32(a_f16) / scale * 32767.0f;
    if (v > 32767.0f) return 65535u;   /* round(v) 已 ≥ 32768 → 钳 */
    if (v < -32768.0f) return 0u;      /* round(v) 已 ≤ -32769 → 钳 */
    /* round-half-away 纯 C (libc roundf 设备运行时未证, 不用) */
    float r = v < 0.0f ? v - 0.5f : v + 0.5f;
    return (uint16_t)((int32_t)r + 32768);
}

void w4a16_pack_crouton(const uint16_t* lin, uint32_t m, uint32_t k, uint16_t* surf) {
    uint32_t n_kt = k / 32, n_m32 = m / 32;
    uint32_t out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++) {
            uint32_t k_base = kt * 32;
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    uint32_t row1 = row0 + 1;
                    for (uint32_t c = 0; c < 32; c++) {
                        surf[out++] = lin[(size_t)row0 * k + k_base + c];
                        surf[out++] = lin[(size_t)row1 * k + k_base + c];
                    }
                }
        }
}

void w4a16_unpack_crouton(const uint16_t* surf, uint32_t m, uint32_t k, uint16_t* lin) {
    uint32_t n_kt = k / 32, n_m32 = m / 32;
    uint32_t out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t kt = 0; kt < n_kt; kt++) {
            uint32_t k_base = kt * 32;
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    uint32_t row1 = row0 + 1;
                    for (uint32_t c = 0; c < 32; c++) {
                        lin[(size_t)row0 * k + k_base + c] = surf[out++];
                        lin[(size_t)row1 * k + k_base + c] = surf[out++];
                    }
                }
        }
}

void w4a16_dequant_out(const uint16_t* lin_q, uint32_t m, uint32_t n,
                       float act_scale, const uint16_t* scale_f16, uint16_t* out_f16) {
    const float inv = 1.0f / 32767.0f;
    for (uint32_t row = 0; row < m; row++)
        for (uint32_t col = 0; col < n; col++) {
            int32_t aq = (int32_t)lin_q[(size_t)row * n + col] - 32768;
            float S = f16_to_f32(scale_f16[col]);
            float v = act_scale * S * (float)aq * inv;
            out_f16[(size_t)row * n + col] = f32_to_f16_rne(v);
        }
}

void w4a16_dequant_crouton(const uint16_t* surf, uint32_t m_pad, uint32_t n,
                           uint32_t m_out, float act_scale,
                           const uint16_t* scale_f16, uint16_t* out_f16) {
    const float inv = 1.0f / 32767.0f;
    uint32_t n_nt = n / 32, n_m32 = m_pad / 32;
    uint32_t out = 0;
    for (uint32_t phase = 0; phase < 8; phase++)
        for (uint32_t nt = 0; nt < n_nt; nt++) {
            uint32_t n_base = nt * 32;
            for (uint32_t g = 0; g < n_m32; g++)
                for (uint32_t rp = 0; rp < 2; rp++) {
                    uint32_t row0 = g * 32 + phase * 4 + rp * 2;
                    uint32_t row1 = row0 + 1;
                    for (uint32_t c = 0; c < 32; c++) {
                        uint16_t q0 = surf[out++], q1 = surf[out++];
                        if (row0 < m_out) {
                            float S = f16_to_f32(scale_f16[n_base + c]);
                            float v = act_scale * S * (float)((int32_t)q0 - 32768) * inv;
                            out_f16[(size_t)row0 * n + n_base + c] = f32_to_f16_rne(v);
                        }
                        if (row1 < m_out) {
                            float S = f16_to_f32(scale_f16[n_base + c]);
                            float v = act_scale * S * (float)((int32_t)q1 - 32768) * inv;
                            out_f16[(size_t)row1 * n + n_base + c] = f32_to_f16_rne(v);
                        }
                    }
                }
        }
}
