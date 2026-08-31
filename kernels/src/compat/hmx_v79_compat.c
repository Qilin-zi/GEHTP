/*
 * hmx_v79_compat.c — V79 back-compat 薄 wrapper (6 个 kernel)
 * Module: compat
 * Duty:   每个 wrapper 尾调对应 V81 族函数 (V79 是 V81 子集, 数值语义一致)
 * Note:   符号命名 hmx_v79_<kernel>. 详见 docs/compat_layer.md.
 */
#include "hmx_kernels.h"

/* u8 × i16 → u8 (convbnb: activation.ub + weight.n → sat.u8) */
#define BNB(name) \
    void hmx_v79_##name(const uint8_t *a, const int16_t *w, \
                        const int32_t *b, uint8_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        hmx_convbnb(a, w, b, o, M, K, N); \
    }

#define BNB_SP(name) \
    void hmx_v79_##name(const uint8_t *a, const int16_t *w, \
                        const uint8_t *sp, const int32_t *b, uint8_t *o, \
                        uint32_t M, uint32_t K, uint32_t N) { \
        (void)sp; \
        hmx_convbnb(a, w, b, o, M, K, N); \
    }

/* --- convbnb (6) --- */
BNB(convbnb_stride1)
BNB(convbnb_stride1_aligned)
BNB_SP(convbnb_stride1_aligned_sparsity)
BNB_SP(convbnb_stride1_sparsity)
BNB(convbnb_stride2)
BNB_SP(convbnb_stride2_sparsity)
