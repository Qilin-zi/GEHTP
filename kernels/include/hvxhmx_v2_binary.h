/*
 * hvxhmx_v2_binary.h — V2 逐元素二元运算 (HVX 向量化)
 * =====================================================================
 * 源自 ggmlHTPV3E htp/hvx-base.h 的 hvx_vec_{add,sub,mul}_f32_f32.
 *
 * 所有函数: dst[i] = a[i] ⊕ b[i], i ∈ [0, n), f32 in/out.
 * add/sub/mul 走 HVX 向量原语; div = a * inverse(b) (HVX 近似倒数).
 *
 * 对齐: 128B 对齐性能最佳; 尾部自动标量.
 */
#ifndef HVXHMX_V2_BINARY_H
#define HVXHMX_V2_BINARY_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void hvhx_v2_add_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n);
void hvhx_v2_sub_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n);
void hvhx_v2_mul_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n);
void hvhx_v2_div_f32(float * __restrict__ dst,
                     const float * __restrict__ a, const float * __restrict__ b, uint32_t n);

/* 标量右操作数: dst[i] = a[i] ⊕ s — 常见 (bias add / scale mul). */
void hvhx_v2_add_scalar_f32(float * __restrict__ dst, const float * __restrict__ a,
                             float s, uint32_t n);
void hvhx_v2_mul_scalar_f32(float * __restrict__ dst, const float * __restrict__ a,
                             float s, uint32_t n);

/* ---- f16 面直收直发(oplist 执行器契约; 数学与标量 f32 中间路径位级等价) ----
 * dst[i] = f32_to_f16(f16_to_f32(a[i]) + f16_to_f32(b[i])), i ∈ [0, n).
 * 向量主体: f16→f32(HVX vmpy 位级保真) → f32 加 → f16(RNE)。
 * 首/尾未对齐段与 <VLEN 余量走标量(与 oplist_exec 标量逐字节同)。
 * 对齐: 任意地址安全; 128B 对齐时全向量化。 */
void hvhx_v2_add_f16(uint16_t * __restrict__ dst,
                     const uint16_t * __restrict__ a, const uint16_t * __restrict__ b,
                     uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_BINARY_H */
