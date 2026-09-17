/* pxbridge.h — U13 精度桥 (f32 ↔ f16 ↔ INT16 对称量化)
 *
 * V81 HMX INT16 契约 (AIMET 0xc26 教训): 激活必须 ufixed16, offset 恒 -32768
 * (AIMET min/max 自适应 offset 会被引擎拒 0xc26):
 *   raw_unsigned = q + 32768 ∈ [0, 65535]   (q 为 int16 位型; 单极, x ≥ 0)
 *   编码:  q  = clamp(round(x/scale), 0, 65535) - 32768   (负输入钳到零码)
 *   解码:  x' = (q + 32768) * scale
 *   零 ⇔ q == 0x8000 (精确); "对称" 指 offset 与容器零点重合
 *
 * f16 转换复用 gdn_sm 的软实现 (同库内依赖, 位级一致)。
 */
#ifndef HVXHMX_V23_PXBRIDGE_H
#define HVXHMX_V23_PXBRIDGE_H

#include <stdint.h>

int16_t pxb_f32_to_i16(float x, float scale);   /* 对称编码 (上面契约) */
float   pxb_i16_to_f32(int16_t q, float scale); /* 对称解码 */
int16_t pxb_f16_to_i16(int16_t h, float scale); /* f16 位型 → i16 位型 */
int16_t pxb_i16_to_f16(int16_t q, float scale); /* i16 位型 → f16 位型 */

/* 批量 (标量实现, 正确性单元) */
void pxb_f32_to_i16_v(const float* x, int16_t* q, uint32_t n, float scale);
void pxb_i16_to_f32_v(const int16_t* q, float* x, uint32_t n, float scale);
void pxb_f16_to_i16_v(const int16_t* h, int16_t* q, uint32_t n, float scale);
void pxb_i16_to_f16_v(const int16_t* q, int16_t* h, uint32_t n, float scale);

#endif
