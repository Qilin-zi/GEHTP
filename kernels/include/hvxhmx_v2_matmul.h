/*
 * hvxhmx_v2_matmul.h — V2 HMX tiled GEMM + 量化权重 dequant
 * =====================================================================
 * 源自 ggmlHTPV3E htp/hmx-mm-kernels-tiled.h + matmul-ops.h.
 *
 * 数据布局 (HMX tile-major interleaved):
 *   - activation: [n_row_tiles × n_dot_tiles × 1024] fp16, 由
 *                 hmx_interleave_rows_to_tiles() 从 row-major fp16 生成.
 *   - weight:     [n_col_tiles × n_dot_tiles × 1024] fp16, 由
 *                 hmx_interleave_cols_to_tiles() 或 dequant 生成.
 *   - output:     [n_row_tiles × n_col_tiles × 1024] fp16 tile-major.
 *
 * 调用流程 (量化 LLM 权重):
 *   1. hmx_runtime_setup()                          // 一次性
 *   2. hvhx_v2_dequant_tiled_to_fp16(q_wgt, w_fp16, N, K, type)
 *      // 或 fp16 权重: hmx_interleave_cols_to_tiles(...)
 *   3. hmx_interleave_rows_to_tiles(act_tiles, act_rowmajor, M, K, K, 0, M)
 *   4. hvhx_v2_hmx_gemm_dot_fp16(out_tiles, act_tiles, w_fp16,
 *                                scales, M/32, N/32, K/32)
 *   5. transfer_output_chunk_fp16_to_fp32(...) // tile→row-major (internal header)
 *
 * 约束: M, K, N 均 32 倍数 (HMX tile 边界). dst/act/wgt 2KB(VTCM)/128B(DDR) 对齐.
 */
#ifndef HVXHMX_V2_MATMUL_H
#define HVXHMX_V2_MATMUL_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  量化权重类型 (映射 ggml HTP_TYPE_*)
 * ============================================================ */
typedef enum {
    HVHX_V2_WT_F16    = 1,   /* fp16 权重 (仅 interleave, 无 dequant) */
    HVHX_V2_WT_Q4_0   = 2,
    HVHX_V2_WT_Q4_1   = 3,
    HVHX_V2_WT_Q8_0   = 4,
    HVHX_V2_WT_IQ4_NL = 5,
    HVHX_V2_WT_MXFP4  = 6,
} hvhx_v2_weight_type_t;

/* ============================================================
 *  Tile 尺寸 helper (tile-major 几何)
 * ============================================================ */

/** @return 单 tile 字节数 (Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4); F16 返回 0. */
static inline uint32_t hvhx_v2_weight_tile_size(hvhx_v2_weight_type_t t)
{
    switch (t) {
        case HVHX_V2_WT_Q4_0:  case HVHX_V2_WT_IQ4_NL: return 576;
        case HVHX_V2_WT_Q4_1:                            return 640;
        case HVHX_V2_WT_Q8_0:                            return 1088;
        case HVHX_V2_WT_MXFP4:                           return 544;
        default: return 0;
    }
}

/** @return 128B 对齐后的 tile 字节数. */
static inline uint32_t hvhx_v2_weight_aligned_tile_size(hvhx_v2_weight_type_t t)
{
    switch (t) {
        case HVHX_V2_WT_Q4_0:  case HVHX_V2_WT_IQ4_NL:
        case HVHX_V2_WT_Q4_1:  case HVHX_V2_WT_MXFP4:   return 640;
        case HVHX_V2_WT_Q8_0:                            return 1152;
        default: return 0;
    }
}

/* ============================================================
 *  量化权重 dequant → tile-major fp16 (单线程, 全量)
 *
 *  把 cooker Pass2 产出的 tile-major 量化权重量 (n_col_tiles × n_k_tiles
 *  个 aligned_tile_size 块, 连续存放) 解量化为 tile-major fp16
 *  (每 tile 1024 个 fp16, 连续存放). 单线程顺序处理全部 tile.
 *
 * @param in      量化权重, tile-major [n_col_tiles × n_k_tiles × aligned_tile_size]
 * @param out     输出 fp16, tile-major [n_col_tiles × n_k_tiles × 1024], 2KB 对齐
 * @param n_cols  N (输出列数, 必须 32 倍数)
 * @param k       K (归约维, 必须 32 倍数)
 * @param type    量化类型
 * @return 0 成功, -1 参数非法
 * ============================================================ */
int hvhx_v2_dequant_tiled_to_fp16(const void    * __restrict__ in,
                                  __fp16        * __restrict__ out,
                                  uint32_t n_cols, uint32_t k,
                                  hvhx_v2_weight_type_t type);

/* ============================================================
 *  HMX crouton GEMM (tile-major fp16 in/out, 需 VTCM)
 *
 *  C[M,N] += A[M,K]·B[K,N], 其中 A/B/C 均为 tile-major interleaved fp16.
 *  内部: hmx_enable_execution + core_dot_chunk_fp16.
 *
 * @param out     C, tile-major [n_row_tiles × n_col_tiles × 1024] fp16, VTCM/2KB 对齐
 * @param act     A, tile-major [n_row_tiles × n_dot_tiles × 1024] fp16
 * @param wgt     B, tile-major [n_col_tiles × n_dot_tiles × 1024] fp16
 * @param scales  HMX bias 区 (256B, 2KB 对齐); 全 0 = scale 1.0
 *                注: HMX bias 寄存器, fp16 per-col scale. 详见 hmx_common.h.
 * @param n_row_tiles  = M/32
 * @param n_col_tiles  = N/32
 * @param n_dot_tiles  = K/32 (≤32 走 short 路径, >32 自动分批)
 *
 *  dot 版: 每次清累加器 (zero_init 语义). mma 版见下.
 * ============================================================ */
void hvhx_v2_hmx_gemm_dot_fp16(__fp16       * __restrict__ out,
                               const __fp16 * __restrict__ act,
                               const __fp16 * __restrict__ wgt,
                               const __fp16 * __restrict__ scales,
                               uint32_t n_row_tiles,
                               uint32_t n_col_tiles,
                               uint32_t n_dot_tiles);

/* ============================================================
 *  HMX crouton MMA (tile-major, 累加到现有 C)
 *
 *  与 dot 版同, 但 zero_init=0 时先把 C 现有 tile 读回累加 (C += A·B).
 *  eye_tile 用于 HMX "load existing accumulator" 路径, 通常传全 1.0 的 2KB tile.
 *
 * @param zero_init  1 = 清零后累加 (等价 dot); 0 = 累加到 C 现有值
 * @param eye_tile   2KB fp16 tile, 全 1.0 (仅 zero_init=0 用); 传 NULL 则强制 zero_init
 * ============================================================ */
void hvhx_v2_hmx_gemm_mma_fp16(__fp16       * __restrict__ C,
                               const __fp16 * __restrict__ A,
                               const __fp16 * __restrict__ B,
                               const __fp16 * __restrict__ col_scales,
                               const __fp16 * __restrict__ eye_tile,
                               uint32_t n_row_tiles,
                               uint32_t n_col_tiles,
                               uint32_t n_dot_tiles,
                               int zero_init);

/* ============================================================
 *  HMX tile-major ↔ row-major I/O 变换 (transfer_*)
 *
 *  HMX tile 布局: 每 tile = 32×32 = 1024 fp16, 存为 16 个 HVX_Vector.
 *    vector r1 (0..15) 打包 2 行: fp16[0..31] = tile 行 2*r1,
 *                                  fp16[32..63] = tile 行 2*r1+1.
 *  这两个函数在 tile-major interleaved 与 row-major 之间转换,
 *  是 GEMM 输入激活准备 + 输出回写的标准 I/O 内核.
 * ============================================================ */

/**
 * @brief HMX tile-major fp16 输出 → row-major fp32 (+ 可选残差 src2).
 *
 * @param dst         row-major fp32 输出 [n_rows × dst_stride]
 * @param src2        f32 残差 [n_rows × src2_stride]; NULL = 不加残差
 * @param vtcm_src    tile-major fp16 HMX 输出 [(start_row+n_rows)/32 × n_cols/32 × 1024]
 * @param start_row   全局起始行 (应 32 对齐以匹配 tile 行边界)
 * @param n_rows      要写出的行数
 * @param n_cols      tile 列数 (必须 32 倍数)
 * @param dst_stride  dst 行步长 (float 单位)
 * @param src2_stride src2 行步长 (float 单位)
 * @param dst_cols    dst 实际列数 (写出 min(n_cols, dst_cols) 列)
 *
 * @note src: hmx-mm-kernels-tiled.h:769. host 路径标量等价 (同 tile 布局).
 */
void hvhx_v2_transfer_output_fp16_to_fp32(
    float         * __restrict__ dst,
    const float   * __restrict__ src2,
    const __fp16  * __restrict__ vtcm_src,
    uint32_t start_row, uint32_t n_rows, uint32_t n_cols,
    uint32_t dst_stride, uint32_t src2_stride, uint32_t dst_cols);

/**
 * @brief row-major fp32 激活 → HMX tile-major fp16.
 *
 * @param vtcm_dst  tile-major fp16 输出 [n_rows_padded/32 × k_block/32 × 1024]
 * @param src       row-major fp32 输入 [n_rows × k_stride]
 * @param n_rows    行数 (< n_rows_padded 的尾部行补 0)
 * @param k_block   padded K (必须 32 倍数, = n_col_tiles × 32)
 * @param k_stride  src 行步长 (float 单位)
 * @param k_valid   实际有效列数 (≤ k_block; 不足部分补 0)
 *
 * @note src: hmx-mm-kernels-tiled.h:866. host 路径标量等价.
 */
void hvhx_v2_transfer_activation_fp32_to_fp16(
    __fp16        * __restrict__ vtcm_dst,
    const float   * __restrict__ src,
    uint32_t n_rows, uint32_t k_block, uint32_t k_stride, uint32_t k_valid);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_MATMUL_H */
