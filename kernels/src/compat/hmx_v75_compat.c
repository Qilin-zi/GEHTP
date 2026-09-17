/*
 * hmx_v75_compat.c — V75 back-compat 薄 wrapper (6 个 kernel)
 * Module: compat
 * Duty:   每个 wrapper 尾调对应 V81 族函数 (V75 是 V81 子集, 数值语义一致)
 * Note:   符号命名 hmx_v75_<kernel>. 详见 docs/compat_layer.md.
 */
#include "hmx_kernels.h"

/* u8 × u8 → u16 (convbbh: activation.ub + weight.b → sat.uh) */
#define BBH(name) \
    void hmx_v75_##name(const uint8_t *a, const uint8_t *w, \
                        const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convbbh(a, w, b, o, M, K, N); \
    }

/* u8 × i8 → u16 (convhbh) */
#define HBH(name) \
    void hmx_v75_##name(const uint8_t *a, const int8_t *w, \
                        const int32_t *b, uint16_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convhbh(a, w, b, o, M, K, N); \
    }

/* --- convbbh (1) --- */
BBH(convbbh1x1_stride1)

/* --- convhbh (5) --- */
HBH(convhbh1x1deep_stride1)
HBH(convhbh1x1_stride1)
HBH(convhbh_dilate_stride1)
HBH(convhbh_stride1_aligned)
HBH(convhbh_stride2)
