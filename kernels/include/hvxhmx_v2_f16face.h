/*
 * hvxhmx_v2_f16face.h — GEHTP oplist 执行器 f16 面 HVX 接线族 (W1)
 * 全部函数: f16 直收直发, 数学 = f32 中间路径; scratch 由调用者提供 (128B 对齐)。
 * 详见 hvx_f16face.c 头部注记。
 */
#ifndef HVXHMX_V2_F16FACE_H
#define HVXHMX_V2_F16FACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* f16↔f32 向量转换 (IEEE RNE; 任意地址安全) */
void gehtp_hvx_cvt_f16_to_f32(float *dst, const uint16_t *src, uint32_t n);
void gehtp_hvx_cvt_f32_to_f16(uint16_t *dst, const float *src, uint32_t n);

/* 原生 f16 乘: dst[i] = f32_to_f16(f16_to_f32(a[i]) * f16_to_f32(b[i])) */
void gehtp_hvx_mul_f16(uint16_t *dst, const uint16_t *a, const uint16_t *b, uint32_t n);

/* 一元族 (subtype 同 exec_unary 编号): 0 neg/1 exp/2 sqrt/3 rsqrt/4 log/
 * 8 sigmoid/9 tanh/10 gelu_tanh/12 silu; 其余 subtype 返回 -1 (回落标量)。 */
int  gehtp_hvx_unary_f16(uint16_t *dst, const uint16_t *src, uint32_t n, uint32_t subtype,
                         float *si, float *so);

/* SiLU f16 面 */
void gehtp_hvx_silu_f16(uint16_t *dst, const uint16_t *src, uint32_t n,
                        float *si, float *so);

/* Softmax f16 面: rows × n, n 须 32 倍数; pad = 行内 scratch (≥n f32) */
void gehtp_hvx_softmax_f16(uint16_t *dst, const uint16_t *src, uint32_t rows, uint32_t n,
                           float *si, float *so, float *pad);

/* RMSNorm×gamma f16 面: [m,rw]; bias 可为 NULL */
void gehtp_hvx_rmsnorm_mul_f16(uint16_t *dst, const uint16_t *src,
                               const uint16_t *w, const uint16_t *bias,
                               uint32_t m, uint32_t rw, float eps,
                               float *si, float *sw, float *so);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V2_F16FACE_H */
