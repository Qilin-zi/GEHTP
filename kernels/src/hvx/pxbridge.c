/* pxbridge.c — U13 精度桥 (见 include/pxbridge.h) */
#include "pxbridge.h"
#include "gdn_sm.h"
#include <math.h>

static inline int32_t pxb_code(float x, float scale) {
    float u = roundf(x / scale);
    if (u < 0.f) u = 0.f;
    if (u > 65535.f) u = 65535.f;
    return (int32_t)u - 32768;
}

int16_t pxb_f32_to_i16(float x, float scale) { return (int16_t)pxb_code(x, scale); }

float pxb_i16_to_f32(int16_t q, float scale) {
    uint32_t u = (uint32_t)(((uint32_t)(int32_t)q) + 32768u) & 0xFFFFu;
    return (float)u * scale;
}

int16_t pxb_f16_to_i16(int16_t h, float scale) {
    return pxb_f32_to_i16(gdn_f16_to_f32(h), scale);
}

int16_t pxb_i16_to_f16(int16_t q, float scale) {
    return gdn_f32_to_f16(pxb_i16_to_f32(q, scale));
}

void pxb_f32_to_i16_v(const float* x, int16_t* q, uint32_t n, float scale) {
    for (uint32_t i = 0; i < n; i++) q[i] = pxb_f32_to_i16(x[i], scale);
}
void pxb_i16_to_f32_v(const int16_t* q, float* x, uint32_t n, float scale) {
    for (uint32_t i = 0; i < n; i++) x[i] = pxb_i16_to_f32(q[i], scale);
}
void pxb_f16_to_i16_v(const int16_t* h, int16_t* q, uint32_t n, float scale) {
    for (uint32_t i = 0; i < n; i++) q[i] = pxb_f16_to_i16(h[i], scale);
}
void pxb_i16_to_f16_v(const int16_t* q, int16_t* h, uint32_t n, float scale) {
    for (uint32_t i = 0; i < n; i++) h[i] = pxb_i16_to_f16(q[i], scale);
}
