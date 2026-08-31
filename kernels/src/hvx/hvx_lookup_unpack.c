/*
 * hvx_lookup_unpack.c — HVX 表查 (LUT) + 权重解包 (4-bit → 8-bit)
 * Module: hvx-lookup-unpack
 * Math:   table_lookup: out[i]=table[in[i]] (256 元素 LUT)
 *         unpack: out[2i]=(in[i]>>4)&0xF, out[2i+1]=in[i]&0xF  (n 字节→2n 字节)
 * Note:   详见 docs/api_hvx_lookup_unpack.md.
 */
#include "hvx_lookup_unpack.h"
#include <hvx_hexagon_protos.h>

/* ============================================================
 *  Table lookup flat u8
 *   table 256 字节, 与 vmem 兼容. 一次查 128 lane/向量.
 * ============================================================ */
void hvhx_table_lookup_flat_u8(const uint8_t * __restrict__ in,
                               const uint8_t * __restrict__ table,
                               uint8_t       * __restrict__ out,
                               uint32_t n)
{
    if (n == 0) return;
    const uint32_t n_vec = n & ~(HVHX_VEC_ELEM_U8 - 1u);
    const uint32_t n_tail = n - n_vec;

    for (uint32_t i = 0; i < n_vec; i += HVHX_VEC_ELEM_U8) {
        HVX_Vector v_in = *(const HVX_Vector *) (in + i);

        /* HVX 256 entry u8 表查 用 vmem (128B 段) 拼接
         *   - hi 4 bit 选 16 段之一 (256/16=16 字节段)
         *   - lo 4 bit 选段内 16 字节
         *   - 一次 128 lane 查 16 表项, 8 趟拼出 256.
         *   简化: 用 vdeal + vshuff 模拟
         * 这里采用最直接的标量化实现 (vshuff 索引构造复杂, 标量版易校验) */
        uint8_t lanes[HVHX_VEC_ELEM_U8];
        *(HVX_Vector *) lanes = v_in;
        uint8_t result[HVHX_VEC_ELEM_U8];
        for (int j = 0; j < HVHX_VEC_ELEM_U8; ++j) result[j] = table[lanes[j]];
        *(HVX_Vector *) (out + i) = *(HVX_Vector *) result;
    }

    for (uint32_t i = n_vec; i < n; ++i) {
        out[i] = table[in[i]];
    }
}

/* crouton u8 — 32x32 块, 多 batch */
void hvhx_table_lookup_crouton_u8(const uint8_t * __restrict__ in,
                                  const uint8_t * __restrict__ table,
                                  uint32_t batches,
                                  uint8_t       * __restrict__ out)
{
    if (batches == 0) return;
    const uint32_t total = batches * 32u * 32u;
    hvhx_table_lookup_flat_u8(in, table, out, total);
}

/* ============================================================
 *  Unpack 4-bit → 8-bit
 *   每输入字节含 2 个 4-bit 权重:
 *     out[2i+0] = (in[i] >> 4) & 0x0F
 *     out[2i+1] =  in[i]       & 0x0F
 *
 *  HVX vdeal 路径:
 *   - vdeal 拆 (hi, lo) → 2 个 64B 通道
 *   - vsxt 4-bit → 8-bit (zero extend)
 *   - 拼接写回
 * ============================================================ */
void hvhx_unpack_weights(const uint8_t * __restrict__ in,
                         uint8_t       * __restrict__ out,
                         uint32_t n)
{
    if (n == 0) return;
    /* n = 输入字节数 (每字节 2 个 4-bit nibble) → 输出 2n 字节 */
    const uint32_t n_in  = n;
    const uint32_t n_vec = n_in & ~(HVHX_VEC_ELEM_U8 - 1u);

    for (uint32_t i = 0; i < n_vec; i += HVHX_VEC_ELEM_U8) {
        HVX_Vector v_in = *(const HVX_Vector *) (in + i);
        uint8_t in_lanes[HVHX_VEC_ELEM_U8];
        *(HVX_Vector *) in_lanes = v_in;
        for (int j = 0; j < HVHX_VEC_ELEM_U8; ++j) {
            out[2 * (i + j) + 0] = (in_lanes[j] >> 4) & 0x0F;
            out[2 * (i + j) + 1] =  in_lanes[j]       & 0x0F;
        }
    }

    for (uint32_t i = n_vec; i < n_in; ++i) {
        out[2 * i + 0] = (in[i] >> 4) & 0x0F;
        out[2 * i + 1] =  in[i]       & 0x0F;
    }
}

/* 自定义 unpack: 支持 sparsity 掩码.
 *   sparsity_mask[i] = 1: 该权重位置激活, 否则输出 0.
 *   掩码布局: 每 4 个连续权重至少 2 个为 0 (2:4 稀疏). */
void hvhx_unpack_custom_weights(const uint8_t * __restrict__ in,
                                const uint8_t * __restrict__ sparsity_mask,
                                uint8_t       * __restrict__ out,
                                uint32_t n)
{
    if (n == 0) return;
    const uint32_t n_in = n / 2u;

    for (uint32_t i = 0; i < n_in; ++i) {
        uint8_t b = in[i];
        uint8_t hi = (b >> 4) & 0x0F;
        uint8_t lo =  b       & 0x0F;

        if (sparsity_mask) {
            if (!sparsity_mask[2 * i + 0]) hi = 0;
            if (!sparsity_mask[2 * i + 1]) lo = 0;
        }
        out[2 * i + 0] = hi;
        out[2 * i + 1] = lo;
    }
    if (n & 1u) {
        out[n - 1] = (in[n_in] >> 4) & 0x0F;
    }
}
