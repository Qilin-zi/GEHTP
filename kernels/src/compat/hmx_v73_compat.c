/*
 * hmx_v73_compat.c — V73 back-compat 薄 wrapper (119 个 kernel)
 * Module: compat
 * Duty:   每个 wrapper 尾调对应 V81 族函数 (V73 是 V81 子集, 数值语义一致)
 * Note:   符号命名 hmx_v73_<kernel>. 详见 docs/compat_layer.md.
 */
#include "hmx_kernels.h"

/* ============================================================
 *  Family macros — one instantiation per kernel
 * ============================================================ */

/* u8 × u8 → u8 (convbbb: activation.ub + weight.b → sat.u8) */
#define BBB(name) \
    void hmx_v73_##name(const uint8_t *a, const uint8_t *w, \
                        const int32_t *b, uint8_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convbbb(a, w, b, o, M, K, N); \
    }

/* u8 × u8 → u8, sparsity variant (sparsity param between wgt and bias, ignored) */
#define BBB_SP(name) \
    void hmx_v73_##name(const uint8_t *a, const uint8_t *w, \
                        const uint8_t *sp, const int32_t *b, uint8_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        (void)sp; \
        hmx_convbbb(a, w, b, o, M, K, N); \
    }

/* u8 × i16 → u8 (convbcb / convbnb / convbnh: activation.ub + weight.c/.n → sat.u8) */
#define I16W_U8(name) \
    void hmx_v73_##name(const uint8_t *a, const int16_t *w, \
                        const int32_t *b, uint8_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convbnb(a, w, b, o, M, K, N); \
    }

/* u8 × i16 → u8, sparsity variant */
#define I16W_U8_SP(name) \
    void hmx_v73_##name(const uint8_t *a, const int16_t *w, \
                        const uint8_t *sp, const int32_t *b, uint8_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        (void)sp; \
        hmx_convbnb(a, w, b, o, M, K, N); \
    }

/* fp16 × fp16 → fp16 (convf16) */
#define F16(name) \
    void hmx_v73_##name(const __fp16 *a, const __fp16 *w, \
                        const __fp16 *b, __fp16 *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convf16(a, w, b, o, M, K, N); \
    }

/* u8 × i8 → u16 (convhbh: activation.ub + weight.b → sat.uh) */
#define HBH(name) \
    void hmx_v73_##name(const uint8_t *a, const int8_t *w, \
                        const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convhbh(a, w, b, o, M, K, N); \
    }

#define HBH_SP(name) \
    void hmx_v73_##name(const uint8_t *a, const int8_t *w, \
                        const uint8_t *sp, const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        (void)sp; \
        hmx_convhbh(a, w, b, o, M, K, N); \
    }

/* u8 × i16 → u16 (convhch / convhnh: activation.ub + weight.c/.n → sat.uh) */
#define H16(name) \
    void hmx_v73_##name(const uint8_t *a, const int16_t *w, \
                        const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convhnh(a, w, b, o, M, K, N); \
    }

#define H16_SP(name) \
    void hmx_v73_##name(const uint8_t *a, const int16_t *w, \
                        const uint8_t *sp, const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        (void)sp; \
        hmx_convhnh(a, w, b, o, M, K, N); \
    }

/* u8 × i8 → u16, :2x2 format (convhhh) */
#define HHH(name) \
    void hmx_v73_##name(const uint8_t *a, const int8_t *w, \
                        const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convhhh(a, w, b, o, M, K, N); \
    }

#define HHH_SP(name) \
    void hmx_v73_##name(const uint8_t *a, const int8_t *w, \
                        const uint8_t *sp, const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        (void)sp; \
        hmx_convhhh(a, w, b, o, M, K, N); \
    }

/* element-wise add: out = max(0, a + b + bias) */
#define ADD() \
    void hmx_v73_add(const __fp16 *a, const __fp16 *b, \
                     const __fp16 *bias, __fp16 *o, \
                     uint32_t M, uint32_t N) { \
        hmx_add(a, b, bias, o, M, N); \
    }

/* depthwise conv u8 × u8 → u8 */
#define DWBBB(name) \
    void hmx_v73_##name(const uint8_t *a, const uint8_t *w, \
                        const int32_t *b, uint8_t *o, \
                        uint32_t H, uint32_t W, uint32_t C) { \
        hmx_dwconvbbb(a, w, b, o, H, W, C); \
    }

/* ============================================================
 *  Instantiations (119 kernels)
 * ============================================================ */

/* --- convbbb (17) --- */
BBB(convbbb1x1deep_stride1)
BBB_SP(convbbb1x1deep_stride1_sparsity)
BBB(convbbb1x1_stride1)
BBB_SP(convbbb1x1_stride1_sparsity)
BBB(convbbb1x1_stride1_unaligned)
BBB_SP(convbbb1x1_stride1_unaligned_sparsity)
BBB(convbbb1xN_stride2)
BBB_SP(convbbb1xN_stride2_sparsity)
BBB(convbbb_dilate_stride1)
BBB(convbbbNx1_stride2)
BBB_SP(convbbbNx1_stride2_sparsity)
BBB(convbbb_stride1)
BBB(convbbb_stride1_aligned)
BBB_SP(convbbb_stride1_aligned_sparsity)
BBB_SP(convbbb_stride1_sparsity)
BBB(convbbb_stride2)
BBB_SP(convbbb_stride2_sparsity)

/* --- convbcb (2) --- */
I16W_U8(convbcb1x1deep_stride1)
I16W_U8(convbcb1x1_stride1)

/* --- convbnb (11) --- */
I16W_U8(convbnb1x1deep_stride1)
I16W_U8_SP(convbnb1x1deep_stride1_sparsity)
I16W_U8(convbnb1x1_stride1)
I16W_U8_SP(convbnb1x1_stride1_sparsity)
I16W_U8(convbnb1x1_stride1_unaligned)
I16W_U8_SP(convbnb1x1_stride1_unaligned_sparsity)
I16W_U8(convbnb_dilate_stride1)
I16W_U8(convbnb_stride1)
I16W_U8_SP(convbnb_stride1_sparsity)
I16W_U8(convbnb_stride2)
I16W_U8_SP(convbnb_stride2_sparsity)

/* --- convbnh (1) — u8×i16→u8, same math as convbnb --- */
I16W_U8(convbnh1x1_stride1)

/* --- convf16 (10) --- */
F16(convf16_1x1_stride1)
F16(convf16_1x1_stride1_unaligned)
F16(convf16_1xN_stride2)
F16(convf16_5x5_stride1)
F16(convf16_dilate_stride1)
F16(convf16_Nx1_stride2)
F16(convf16_NxN_stride1)
F16(convf16_stride1)
F16(convf16_stride1_aligned)
F16(convf16_stride2)

/* --- convhbh (25) --- */
HBH(convhbh1x1deep_lp_stride1)
HBH(convhbh1x1deep_stride1)
HBH_SP(convhbh1x1deep_stride1_sparsity)
HBH(convhbh1x1_lp_stride1)
HBH(convhbh1x1_stride1)
HBH_SP(convhbh1x1_stride1_sparsity)
HBH(convhbh1x1_stride1_unaligned)
HBH_SP(convhbh1x1_stride1_unaligned_sparsity)
HBH(convhbh1xN_stride2)
HBH_SP(convhbh1xN_stride2_sparsity)
HBH(convhbh_5x5_stride1)
HBH_SP(convhbh_5x5_stride1_sparsity)
HBH(convhbh_dilate_stride1)
HBH(convhbhNx1_stride2)
HBH_SP(convhbhNx1_stride2_sparsity)
HBH(convhbh_NxN_stride1)
HBH_SP(convhbh_NxN_stride1_sparsity)
HBH(convhbh_NxN_stride1_unaligned)
HBH_SP(convhbh_NxN_stride1_unaligned_sparsity)
HBH(convhbh_stride1)
HBH(convhbh_stride1_aligned)
HBH_SP(convhbh_stride1_aligned_sparsity)
HBH_SP(convhbh_stride1_sparsity)
HBH(convhbh_stride2)
HBH_SP(convhbh_stride2_sparsity)

/* --- convhch (10) --- */
H16(convhch1x1deep_stride1)
H16(convhch1x1_stride1)
H16(convhch1xN_stride2)
H16(convhch_5x5_stride1)
H16(convhchNx1_stride2)
H16(convhch_NxN_stride1)
H16(convhch_NxN_stride1_unaligned)
H16(convhch_stride1)
H16(convhch_stride1_aligned)
H16(convhch_stride2)

/* --- convhhh (19) --- */
HHH(convhhh1x1_stride1)
HHH_SP(convhhh1x1_stride1_sparsity)
HHH(convhhh1x1_stride1_unaligned)
HHH_SP(convhhh1x1_stride1_unaligned_sparsity)
HHH(convhhh1xN_stride2)
HHH_SP(convhhh1xN_stride2_sparsity)
HHH(convhhh_5x5_stride1)
HHH_SP(convhhh_5x5_stride1_sparsity)
HHH(convhhh_dilate_stride1)
HHH(convhhhNx1_stride2)
HHH_SP(convhhhNx1_stride2_sparsity)
HHH(convhhh_NxN_stride1)
HHH_SP(convhhh_NxN_stride1_sparsity)
HHH(convhhh_stride1)
HHH(convhhh_stride1_aligned)
HHH_SP(convhhh_stride1_aligned_sparsity)
HHH_SP(convhhh_stride1_sparsity)
HHH(convhhh_stride2)
HHH_SP(convhhh_stride2_sparsity)

/* --- convhnh (22) --- */
H16(convhnh1x1deep_lp_stride1)
H16(convhnh1x1deep_stride1)
H16(convhnh1x1_lp_stride1)
H16(convhnh1x1_stride1)
H16_SP(convhnh1x1_stride1_sparsity)
H16(convhnh1xN_stride2)
H16_SP(convhnh1xN_stride2_sparsity)
H16(convhnh_5x5_stride1)
H16_SP(convhnh_5x5_stride1_sparsity)
H16(convhnh_dilate_stride1)
H16(convhnhNx1_stride2)
H16_SP(convhnhNx1_stride2_sparsity)
H16(convhnh_NxN_stride1)
H16_SP(convhnh_NxN_stride1_sparsity)
H16(convhnh_NxN_stride1_unaligned)
H16_SP(convhnh_NxN_stride1_unaligned_sparsity)
H16(convhnh_stride1)
H16(convhnh_stride1_aligned)
H16_SP(convhnh_stride1_aligned_sparsity)
H16_SP(convhnh_stride1_sparsity)
H16(convhnh_stride2)
H16_SP(convhnh_stride2_sparsity)

/* --- add (1) --- */
ADD()

/* --- dwconvbbb (1) --- */
DWBBB(dwconvbbb1x1_stride1)
