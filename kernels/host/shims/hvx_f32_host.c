/* hvx_f32_host.c — host 构建的 HVX f32 内核标量实现
 *
 * 设备侧这些内核在 src/hvx/*.c (HVX 向量体 + 标量尾)。host 无 HVX,
 * 用与 oplist_exec.c 标量路径同算法的 C 实现补齐链接面, 供 wt_host_exec
 * 执行 WTOP blob 时走 xop_binary/unary/softmax/rmsnorm 的策略 B f32 快路径。
 * 注意: 语义只要求 host 自洽 (host vs host 差分), 不承诺与设备逐 ULP 一致。
 */
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---- 与 oplist_exec.c:387/411 同款 f16 转换 (RNE, 次正规展开) ---- */
static float h_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            uint32_t m = man;
            int e = -1;
            while (!(m & 0x400u)) { m <<= 1; e++; }
            bits = sign | ((uint32_t)(127 - 15 + 1 - e - 1) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
static uint16_t h_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) return (uint16_t)(sign);           /* 次正规截断 (同 411-416 面) */
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man >> 12) & 1u) half += ((man & 0xfffu) || (half & 1u)) ? 1u : 0u;
    return (uint16_t)half;
}

void hvhx_v2_cvt_f16_to_f32(float * __restrict__ dst,
                            const uint16_t * __restrict__ src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = h_f16_to_f32(src[i]);
}
void hvhx_v2_cvt_f32_to_f16(uint16_t * __restrict__ dst,
                            const float * __restrict__ src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = h_f32_to_f16(src[i]);
}
void hvhx_v2_add_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}
void hvhx_v2_sub_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = a[i] - b[i];
}
void hvhx_v2_mul_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = a[i] * b[i];
}
void hvhx_v2_div_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = (b[i] != 0.0f) ? a[i] / b[i] : 0.0f;
}
void hvhx_v2_exp_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = expf(src[i]);
}
void hvhx_v2_sigmoid_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = 1.0f / (1.0f + expf(-src[i]));
}
void hvhx_v2_tanh_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = tanhf(src[i]);
}
void hvhx_v2_silu_f32(float * __restrict__ dst, const float * __restrict__ src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) dst[i] = src[i] / (1.0f + expf(-src[i]));
}
void hvhx_v2_softmax_f32(const float * __restrict__ src,
                         float * __restrict__ dst,
                         float * __restrict__ pad, uint32_t n) {
    (void)pad;   /* 设备侧向量填充面, 标量不需要 */
    float mx = -INFINITY;
    for (uint32_t i = 0; i < n; i++) if (src[i] > mx) mx = src[i];
    double s = 0.0;
    for (uint32_t i = 0; i < n; i++) { dst[i] = expf(src[i] - mx); s += dst[i]; }
    for (uint32_t i = 0; i < n; i++) dst[i] = (float)(dst[i] / s);
}
void hvhx_v2_rms_norm_mul_f32_rows(const float * __restrict__ src,
                                   const float * __restrict__ weight,
                                   float       * __restrict__ dst,
                                   uint32_t n_rows, uint32_t row_size, float eps) {
    for (uint32_t r = 0; r < n_rows; r++) {
        const float* x = src + (size_t)r * row_size;
        float* y = dst + (size_t)r * row_size;
        double s = 0.0;
        for (uint32_t c = 0; c < row_size; c++) s += (double)x[c] * x[c];
        float inv = 1.0f / sqrtf((float)(s / row_size) + eps);
        for (uint32_t c = 0; c < row_size; c++) y[c] = x[c] * inv * weight[c];
    }
}

/* W-P3 UserDMA 累加器: host 无 DMA 计数, 恒零 */
void dc_dma_reset(void) {}
void dc_dma_get(int64_t* us, int64_t* bytes) { *us = 0; *bytes = 0; }
