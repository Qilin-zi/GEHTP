/*
 * hvx_lookup_unpack.h — V81 HVX 表查和权重解包
 * =====================================================================
 *  - table_lookup: 256 元素 LUT, out[i] = table[in[i]]
 *  - unpack_weights: 4-bit 压缩 → u8 展开
 *
 * 数值:
 *  - table_lookup 用于 exp(·), 1/x, tanh(·) 等非线性函数的硬件 LUT.
 *  - unpack_weights 用于 LDI-4 量化方案 (QAT).
 */
#ifndef HVXHMX_HVX_LOOKUP_UNPACK_H
#define HVXHMX_HVX_LOOKUP_UNPACK_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------
 *  Table lookup
 * ------------------------------------------------------------- */
void hvhx_table_lookup_flat_u8(const uint8_t *in,
                               const uint8_t *table,
                               uint8_t *out,
                               uint32_t n);

void hvhx_table_lookup_crouton_u8(const uint8_t *in,
                                  const uint8_t *table,
                                  uint32_t batches,
                                  uint8_t *out);

/* -------------------------------------------------------------
 *  Weight unpack (4-bit → 8-bit)
 *   LDI-4: 每字节存 2 个 4-bit 权重
 *   out[2i+0] = (in[i] >> 4) & 0x0F
 *   out[2i+1] =  in[i]       & 0x0F
 *
 *  sparsity: 2:4 结构稀疏标记, 若启用需额外输入掩码.
 * ------------------------------------------------------------- */
void hvhx_unpack_weights(const uint8_t *in, uint8_t *out, uint32_t n);

void hvhx_unpack_custom_weights(const uint8_t *in,
                                const uint8_t *sparsity_mask,  /* 可 NULL */
                                uint8_t *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_HVX_LOOKUP_UNPACK_H */
