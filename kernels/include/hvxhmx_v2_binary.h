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

/* f16 <-> f32 整面转换 (HVX 快路径 + 标量尾) — oplist 执行器 f16 直收直发用。
 * dst/src 任意地址安全; 128B 对齐时全向量化。位级等价于标量 RNE 参考。 */
void hvhx_v2_cvt_f16_to_f32(float * __restrict__ dst,
                            const uint16_t * __restrict__ src, uint32_t n);
void hvhx_v2_cvt_f32_to_f16(uint16_t * __restrict__ dst,
                            const float * __restrict__ src, uint32_t n);

/* ---- Q4_0 反量化 + crouton pack (013; w4 线反量化税 12.5s → HVX) ----
 * 输入: 每 tile 640B = 512B quant + 128B scale(仅前 64B 用, 32 f16 d_n)。
 *       quant 字节 qt[i*32+n] = [hi=q(2i+1,n)][lo=q(2i,n)] (i=K 对, n=oc 列)。
 * 数学: out[pos(k,n)] = f32_to_f16(f16_to_f32(d_n) * (q(k,n)-8)), RNE。
 * 输出: 每 tile 1024 f16 crouton (16×64f16), pos(k,n)=(k>>1)*64+2n+(k&1)。
 * 与 oplist_exec 标量 T017 LUT 分支逐位同 (f32 中间 + RNE)。
 * lane 序 = pair-interleave (crouton), 非 ggml core_dot row-major pair。
 * dst/src 均按 tile 连续 (c-major × kt 序), n_tiles = n_col*n_dot。 */
void hvhx_v2_dequant_q4_0_crouton(uint16_t * __restrict__ dst,
                                  const uint8_t * __restrict__ src,
                                  uint32_t n_tiles);

/* ---- broadcast 纯搬运原语 (007 US1; 位级 = 标量逐元素拷贝, 无数值语义) ----
 * fill: dst[i] = s;  copy: dst[i] = src[i], i ∈ [0, n)。
 * HVX 主体 + 首/尾标量; 任意地址安全。host 路径标量。 */
void hvhx_v2_fill_f16(uint16_t * __restrict__ dst, uint16_t s, uint32_t n);
void hvhx_v2_copy_f16(uint16_t * __restrict__ dst,
                      const uint16_t * __restrict__ src, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_BINARY_H */
