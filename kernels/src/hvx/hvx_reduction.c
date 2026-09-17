/*
 * hvx_reduction.c — HVX 沿 depth 维归约 (argminmax / find_max / top1 / sum)
 * Module: hvx-reduction
 * Math:   per-row 树形归约 (vror+vmax/min), element 0 = 全局极值 + 首匹配 idx
 * Note:   支持 u8 (d≤128) / u16 (d≤64). 详见 docs/api_hvx_reduction.md.
 */
#include "hvx_reduction.h"
#include "hvxhmx_types.h"

#if defined(__HVX__) || defined(__hexagon__)
#include <hvx_hexagon_protos.h>
#include <hexagon_types.h>

/* byte 树形归约: 128→1, element 0 = 全局 max/min (unsigned) */
static inline uint8_t hvx_reduc_max_ub(HVX_Vector v) {
    v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 2));
    v = Q6_Vub_vmax_VubVub(v, Q6_V_vror_VR(v, 1));
    union { HVX_Vector v; uint8_t b[128]; } u; u.v = v; return u.b[0];
}
static inline uint8_t hvx_reduc_min_ub(HVX_Vector v) {
    v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 2));
    v = Q6_Vub_vmin_VubVub(v, Q6_V_vror_VR(v, 1));
    union { HVX_Vector v; uint8_t b[128]; } u; u.v = v; return u.b[0];
}
/* half 树形归约: 64→1, vror 字节数全偶 (保 halfword 对齐) */
static inline uint16_t hvx_reduc_max_uh(HVX_Vector v) {
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vuh_vmax_VuhVuh(v, Q6_V_vror_VR(v, 2));
    union { HVX_Vector v; uint16_t h[64]; } u; u.v = v; return u.h[0];
}
static inline uint16_t hvx_reduc_min_uh(HVX_Vector v) {
    v = Q6_Vuh_vmin_VuhVuh(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vuh_vmin_VuhVuh(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vuh_vmin_VuhVuh(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vuh_vmin_VuhVuh(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vuh_vmin_VuhVuh(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vuh_vmin_VuhVuh(v, Q6_V_vror_VR(v, 2));
    union { HVX_Vector v; uint16_t h[64]; } u; u.v = v; return u.h[0];
}

/* byte 单行 argminmax: data→union (0 pad max / 0xFF pad min) → 归约 → 标量 idx */
static inline void hvx_row_argminmax_b(const uint8_t *row, uint32_t d,
                                       uint8_t *mx, uint32_t *mxi,
                                       uint8_t *mn, uint32_t *mni) {
    union { HVX_Vector v; uint8_t b[128]; } umx, umn;
    for (uint32_t i = 0; i < d; ++i) { umx.b[i] = row[i]; umn.b[i] = row[i]; }
    for (uint32_t i = d; i < 128; ++i) { umx.b[i] = 0; umn.b[i] = 0xFF; }
    *mx = hvx_reduc_max_ub(umx.v);
    *mn = hvx_reduc_min_ub(umn.v);
    uint32_t i;
    for (i = 0; i < d; ++i) if (umx.b[i] == *mx) { *mxi = i; break; }
    for (i = 0; i < d; ++i) if (umn.b[i] == *mn) { *mni = i; break; }
}
static inline void hvx_row_findmax_b(const uint8_t *row, uint32_t d,
                                     uint8_t *mx, uint32_t *mxi) {
    union { HVX_Vector v; uint8_t b[128]; } umx;
    for (uint32_t i = 0; i < d; ++i) umx.b[i] = row[i];
    for (uint32_t i = d; i < 128; ++i) umx.b[i] = 0;
    *mx = hvx_reduc_max_ub(umx.v);
    for (uint32_t i = 0; i < d; ++i) if (umx.b[i] == *mx) { *mxi = i; break; }
}
static inline void hvx_row_argminmax_h(const uint16_t *row, uint32_t d,
                                       uint16_t *mx, uint32_t *mxi,
                                       uint16_t *mn, uint32_t *mni) {
    union { HVX_Vector v; uint16_t h[64]; } umx, umn;
    for (uint32_t i = 0; i < d; ++i) { umx.h[i] = row[i]; umn.h[i] = row[i]; }
    for (uint32_t i = d; i < 64; ++i) { umx.h[i] = 0; umn.h[i] = 0xFFFF; }
    *mx = hvx_reduc_max_uh(umx.v);
    *mn = hvx_reduc_min_uh(umn.v);
    uint32_t i;
    for (i = 0; i < d; ++i) if (umx.h[i] == *mx) { *mxi = i; break; }
    for (i = 0; i < d; ++i) if (umn.h[i] == *mn) { *mni = i; break; }
}
static inline void hvx_row_findmax_h(const uint16_t *row, uint32_t d,
                                     uint16_t *mx, uint32_t *mxi) {
    union { HVX_Vector v; uint16_t h[64]; } umx;
    for (uint32_t i = 0; i < d; ++i) umx.h[i] = row[i];
    for (uint32_t i = d; i < 64; ++i) umx.h[i] = 0;
    *mx = hvx_reduc_max_uh(umx.v);
    for (uint32_t i = 0; i < d; ++i) if (umx.h[i] == *mx) { *mxi = i; break; }
}
#endif

void hvhx_argminmax_depth_crouton_b(const uint8_t * __restrict__ in,
                                    uint32_t hw, uint32_t d,
                                    hvhx_argminmax_t * __restrict__ out)
{
    if (d == 0) return;
    if (d > HVHX_VEC_ELEM_U8) d = HVHX_VEC_ELEM_U8;
#if defined(__HVX__) || defined(__hexagon__)
    for (uint32_t r = 0; r < hw; ++r)
        hvx_row_argminmax_b(in + r * d, d,
                            &out[r].max_val, &out[r].max_idx,
                            &out[r].min_val, &out[r].min_idx);
#else
    for (uint32_t r = 0; r < hw; ++r) {
        const uint8_t *row = in + r * d;
        uint8_t best = 0, mn = UINT8_MAX; uint32_t best_i = 0, mni = 0;
        for (uint32_t i = 0; i < d; ++i) {
            if (row[i] > best) { best = row[i]; best_i = i; }
            if (row[i] < mn)   { mn = row[i];   mni = i; }
        }
        out[r].max_val = best; out[r].max_idx = best_i;
        out[r].min_val = mn;   out[r].min_idx = mni;
    }
#endif
}

void hvhx_argminmax_depth_crouton_h(const uint16_t * __restrict__ in,
                                    uint32_t hw, uint32_t d,
                                    hvhx_argminmax_t * __restrict__ out)
{
    if (d == 0) return;
    if (d > HVHX_VEC_ELEM_U16) d = HVHX_VEC_ELEM_U16;
#if defined(__HVX__) || defined(__hexagon__)
    for (uint32_t r = 0; r < hw; ++r)
        hvx_row_argminmax_h(in + r * d, d,
                            &out[r].max_val, &out[r].max_idx,
                            &out[r].min_val, &out[r].min_idx);
#else
    for (uint32_t r = 0; r < hw; ++r) {
        const uint16_t *row = in + r * d;
        uint16_t mx = 0, mn = UINT16_MAX; uint32_t mxi = 0, mni = 0;
        for (uint32_t i = 0; i < d; ++i) {
            if (row[i] > mx) { mx = row[i]; mxi = i; }
            if (row[i] < mn) { mn = row[i]; mni = i; }
        }
        out[r].max_val = mx; out[r].max_idx = mxi;
        out[r].min_val = mn; out[r].min_idx = mni;
    }
#endif
}

void hvhx_argminmax_depth_dLE32_crouton_b(const uint8_t * __restrict__ in,
                                          uint32_t hw, uint32_t d,
                                          hvhx_argminmax_t * __restrict__ out)
{
    if (d > 32) d = 32;
    hvhx_argminmax_depth_crouton_b(in, hw, d, out);
}

void hvhx_argminmax_depth_flat_h(const uint16_t * __restrict__ in,
                                 uint32_t hw, uint32_t d,
                                 hvhx_argminmax_t * __restrict__ out)
{
    if (d == 0) return;
    if (d > HVHX_VEC_ELEM_U16) d = HVHX_VEC_ELEM_U16;
#if defined(__HVX__) || defined(__hexagon__)
    for (uint32_t r = 0; r < hw; ++r)
        hvx_row_argminmax_h(in + r * d, d,
                            &out[r].max_val, &out[r].max_idx,
                            &out[r].min_val, &out[r].min_idx);
#else
    for (uint32_t r = 0; r < hw; ++r) {
        const uint16_t *row = in + r * d;
        uint16_t mx = 0, mn = UINT16_MAX; uint32_t mxi = 0, mni = 0;
        for (uint32_t i = 0; i < d; ++i) {
            if (row[i] > mx) { mx = row[i]; mxi = i; }
            if (row[i] < mn) { mn = row[i]; mni = i; }
        }
        out[r].max_val = mx; out[r].max_idx = mxi;
        out[r].min_val = mn; out[r].min_idx = mni;
    }
#endif
}

void hvhx_argminmax_depth_short_b(const uint8_t * __restrict__ in,
                                  uint32_t hw, uint32_t d,
                                  hvhx_argminmax_t * __restrict__ out)
{
    if (d == 0) return;
#if defined(__HVX__) || defined(__hexagon__)
    if (d <= HVHX_VEC_ELEM_U8) {
        for (uint32_t r = 0; r < hw; ++r)
            hvx_row_argminmax_b(in + r * d, d,
                                &out[r].max_val, &out[r].max_idx,
                                &out[r].min_val, &out[r].min_idx);
        return;
    }
#endif
    for (uint32_t r = 0; r < hw; ++r) {
        const uint8_t *row = in + r * d;
        uint8_t mx = 0, mn = UINT8_MAX; uint32_t mxi = 0, mni = 0;
        for (uint32_t i = 0; i < d; ++i) {
            if (row[i] > mx) { mx = row[i]; mxi = i; }
            if (row[i] < mn) { mn = row[i]; mni = i; }
        }
        out[r].max_val = mx; out[r].max_idx = mxi;
        out[r].min_val = mn; out[r].min_idx = mni;
    }
}

void hvhx_find_max_and_index_in_depth_b(const uint8_t * __restrict__ in,
                                        uint32_t hw, uint32_t d,
                                        uint32_t *idx, uint8_t *max_val)
{
    if (d == 0) return;
    if (d > HVHX_VEC_ELEM_U8) d = HVHX_VEC_ELEM_U8;
#if defined(__HVX__) || defined(__hexagon__)
    for (uint32_t r = 0; r < hw; ++r)
        hvx_row_findmax_b(in + r * d, d, &max_val[r], &idx[r]);
#else
    for (uint32_t r = 0; r < hw; ++r) {
        const uint8_t *row = in + r * d;
        uint8_t best = 0; uint32_t best_i = 0;
        for (uint32_t i = 0; i < d; ++i)
            if (row[i] > best) { best = row[i]; best_i = i; }
        idx[r] = best_i; max_val[r] = best;
    }
#endif
}

void hvhx_find_max_and_index_in_depth_h(const uint16_t * __restrict__ in,
                                        uint32_t hw, uint32_t d,
                                        uint32_t *idx, uint16_t *max_val)
{
    if (d == 0) return;
    if (d > HVHX_VEC_ELEM_U16) d = HVHX_VEC_ELEM_U16;
#if defined(__HVX__) || defined(__hexagon__)
    for (uint32_t r = 0; r < hw; ++r)
        hvx_row_findmax_h(in + r * d, d, &max_val[r], &idx[r]);
#else
    for (uint32_t r = 0; r < hw; ++r) {
        const uint16_t *row = in + r * d;
        uint16_t mx = 0; uint32_t mxi = 0;
        for (uint32_t i = 0; i < d; ++i)
            if (row[i] > mx) { mx = row[i]; mxi = i; }
        idx[r] = mxi; max_val[r] = mx;
    }
#endif
}

void hvhx_top1_qu8_dLE32_cr2flt(const uint8_t * __restrict__ in,
                                uint32_t hw, uint32_t d,
                                hvhx_top1_t * __restrict__ out)
{
    if (d == 0) return;
    if (d > 32) d = 32;
#if defined(__HVX__) || defined(__hexagon__)
    for (uint32_t r = 0; r < hw; ++r)
        hvx_row_findmax_b(in + r * d, d, &out[r].val, &out[r].idx);
#else
    for (uint32_t r = 0; r < hw; ++r) {
        const uint8_t *row = in + r * d;
        uint8_t best = 0; uint32_t best_i = 0;
        for (uint32_t i = 0; i < d; ++i)
            if (row[i] > best) { best = row[i]; best_i = i; }
        out[r].val = best; out[r].idx = best_i;
    }
#endif
}

void hvhx_reducesum_depth_u8(const uint8_t * __restrict__ in,
                             uint32_t hw, uint32_t d,
                             uint32_t * __restrict__ sum_u32)
{
    if (d == 0) return;
    if (d > HVHX_VEC_ELEM_U8) d = HVHX_VEC_ELEM_U8;
    for (uint32_t r = 0; r < hw; ++r) {
        const uint8_t *row = in + r * d;
        uint32_t s = 0;
        for (uint32_t i = 0; i < d; ++i) s += row[i];
        sum_u32[r] = s;
    }
}

/* case_N 累加 (host + DSP 共用同标量) */
#define DEFINE_REDUCE_SUM_U8(CASE_N)                                       \
void hvhx_reduce_sum_u8_case_##CASE_N(const uint8_t *in,                   \
                                      uint32_t hw, uint8_t *out)           \
{                                                                          \
    for (uint32_t r = 0; r < hw; ++r) {                                    \
        uint32_t s = 0;                                                    \
        for (uint32_t i = 0; i < (CASE_N); ++i)                            \
            s += in[r * (CASE_N) + i];                                     \
        out[r] = HVHX_SAT_U8(s);                                           \
    }                                                                      \
}

#define DEFINE_REDUCE_SUM_U16(CASE_N)                                      \
void hvhx_reduce_sum_u16_case_##CASE_N(const uint16_t *in,                \
                                       uint32_t hw, uint16_t *out)        \
{                                                                          \
    for (uint32_t r = 0; r < hw; ++r) {                                    \
        uint32_t s = 0;                                                    \
        for (uint32_t i = 0; i < (CASE_N); ++i)                            \
            s += in[r * (CASE_N) + i];                                     \
        out[r] = HVHX_SAT_U16(s);                                          \
    }                                                                      \
}

DEFINE_REDUCE_SUM_U8(1)
DEFINE_REDUCE_SUM_U8(2)
DEFINE_REDUCE_SUM_U8(4)
DEFINE_REDUCE_SUM_U8(6)
DEFINE_REDUCE_SUM_U8(8)

DEFINE_REDUCE_SUM_U16(1)
DEFINE_REDUCE_SUM_U16(2)
DEFINE_REDUCE_SUM_U16(4)
DEFINE_REDUCE_SUM_U16(6)
DEFINE_REDUCE_SUM_U16(8)
