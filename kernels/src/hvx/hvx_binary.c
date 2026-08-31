/*
 * hvx_binary.c — V2 逐元素二元运算封装
 * Module: v2-binary
 * 源:    ggmlHTPV3E htp/hvx-base.h (hvx_vec_{add,sub,mul}_f32_f32)
 *        + htp/hvx-inverse.h (hvx_vec_inverse_f32, for div)
 */
#include "hvxhmx_v2_binary.h"
#include "hvxhmx_types.h"

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hvx-utils.h"
#endif

#define VLEN_F32 32u

#if HVX_V2_KERNELS_ENABLED
#define V2_BIN_LOOP(dst, a, b, n, VEC_OP, SCALAR_EXPR)                       \
    do {                                                                      \
        const uint32_t _nvec = (n) / VLEN_F32;                                \
        const uint32_t _tail = (n) - _nvec * VLEN_F32;                        \
        HVX_Vector       * __restrict__ _dv = (HVX_Vector *)(dst);            \
        const HVX_Vector * __restrict__ _av = (const HVX_Vector *)(a);        \
        const HVX_Vector * __restrict__ _bv = (const HVX_Vector *)(b);        \
        for (uint32_t _i = 0; _i < _nvec; ++_i)                               \
            _dv[_i] = VEC_OP(_av[_i], _bv[_i]);                               \
        for (uint32_t _i = 0; _i < _tail; ++_i) {                             \
            uint32_t _j = _nvec * VLEN_F32 + _i;                              \
            float x = ((const float *)(a))[_j];                               \
            float y = ((const float *)(b))[_j];                               \
            ((float *)(dst))[_j] = (SCALAR_EXPR);                             \
        }                                                                     \
    } while (0)
#endif

void hvhx_v2_add_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_BIN_LOOP(dst, a, b, n, hvx_vec_add_f32_f32, (x + y));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] + b[i];
#endif
}

void hvhx_v2_sub_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_BIN_LOOP(dst, a, b, n, hvx_vec_sub_f32_f32, (x - y));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] - b[i];
#endif
}

void hvhx_v2_mul_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    V2_BIN_LOOP(dst, a, b, n, hvx_vec_mul_f32_f32, (x * y));
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] * b[i];
#endif
}

void hvhx_v2_div_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n)
{
    /* div = a * inverse(b); inverse 是两步, 不套 V2_BIN_LOOP. */
#if HVX_V2_KERNELS_ENABLED
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ av = (const HVX_Vector *)a;
    const HVX_Vector * __restrict__ bv = (const HVX_Vector *)b;
    for (uint32_t i = 0; i < nvec; ++i)
        dv[i] = hvx_vec_mul_f32_f32(av[i], hvx_vec_inverse_f32(bv[i]));
    for (uint32_t i = 0; i < tail; ++i) {
        uint32_t j = nvec * VLEN_F32 + i;
        dst[j] = a[j] / b[j];
    }
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] / b[i];
#endif
}

void hvhx_v2_add_scalar_f32(float * __restrict__ dst, const float * __restrict__ a,
                             float s, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    HVX_Vector vs = hvx_vec_splat_f32(s);
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ av = (const HVX_Vector *)a;
    for (uint32_t i = 0; i < nvec; ++i) dv[i] = hvx_vec_add_f32_f32(av[i], vs);
    for (uint32_t i = 0; i < tail; ++i) dst[nvec * VLEN_F32 + i] = a[nvec * VLEN_F32 + i] + s;
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] + s;
#endif
}

void hvhx_v2_mul_scalar_f32(float * __restrict__ dst, const float * __restrict__ a,
                             float s, uint32_t n)
{
#if HVX_V2_KERNELS_ENABLED
    HVX_Vector vs = hvx_vec_splat_f32(s);
    const uint32_t nvec = n / VLEN_F32;
    const uint32_t tail = n - nvec * VLEN_F32;
    HVX_Vector       * __restrict__ dv = (HVX_Vector *)dst;
    const HVX_Vector * __restrict__ av = (const HVX_Vector *)a;
    for (uint32_t i = 0; i < nvec; ++i) dv[i] = hvx_vec_mul_f32_f32(av[i], vs);
    for (uint32_t i = 0; i < tail; ++i) dst[nvec * VLEN_F32 + i] = a[nvec * VLEN_F32 + i] * s;
#else
    for (uint32_t i = 0; i < n; ++i) dst[i] = a[i] * s;
#endif
}
