/*
 * hmx_matmul.c — V2 HMX tiled GEMM + 量化权重 dequant 封装
 * Module: v2-matmul
 * 源:    ggmlHTPV3E htp/hmx-mm-kernels-tiled.h (dequant task + core_dot/mma)
 *        + matmul-ops.h (tile 常量) + hex-fastdiv.h (fastdiv_values)
 *
 * 封装策略:
 *   - dequant: 把 tiled_dequantize_state_t 构造 + 类型分发 + 单线程全量调用
 *     打包成一个干净的 (in, out, n_cols, k, type) 入口. ctx/traces 传 NULL
 *     (task 函数不解引用).
 *   - GEMM dot/mma: 套 hmx_enable_execution (HMX 执行权限) 后直接调 core_*.
 */
#include "hvxhmx_v2_matmul.h"
#include "hvxhmx_runtime.h"   /* hmx_enable_execution / hmx_disable_execution */
#include "hvxhmx_types.h"

#if defined(__HVX__) || defined(__hexagon__)
#define HVX_V2_KERNELS_ENABLED 1
#include "internal/hmx-mm-kernels-tiled.h"   /* dequant task + core_dot/mma */
#include "internal/hmx-utils.h"
#include "internal/hex-fastdiv.h"
#endif

#include <string.h>

/* ============================================================
 *  量化权重 dequant (单线程, 全量 tile)
 * ============================================================ */
int hvhx_v2_dequant_tiled_to_fp16(const void    * __restrict__ in,
                                  __fp16        * __restrict__ out,
                                  uint32_t n_cols, uint32_t k,
                                  hvhx_v2_weight_type_t type)
{
#if HVX_V2_KERNELS_ENABLED
    if (in == NULL || out == NULL) return -1;
    if (n_cols == 0 || k == 0)     return -1;
    if ((n_cols % HTP_MM_HMX_TILE_N_COLS) != 0 || (k % HTP_MM_HMX_TILE_N_COLS) != 0)
        return -1;
    if (type == HVHX_V2_WT_F16)    return -1;   /* F16 走 interleave, 无 dequant */

    uint32_t n_k_tiles   = k / HTP_MM_HMX_TILE_N_COLS;
    uint32_t n_col_tiles = n_cols / HTP_MM_HMX_TILE_N_COLS;
    uint32_t n_tot_tiles = n_col_tiles * n_k_tiles;

    /* 映射到 ggml 内部 weight_type 编码 (HTP_TYPE_*, htp-ops.h) */
    int wt;
    switch (type) {
        case HVHX_V2_WT_Q4_0:   wt = 2;  break;  /* HTP_TYPE_Q4_0   */
        case HVHX_V2_WT_Q4_1:   wt = 3;  break;  /* HTP_TYPE_Q4_1   */
        case HVHX_V2_WT_Q8_0:   wt = 8;  break;  /* HTP_TYPE_Q8_0   */
        case HVHX_V2_WT_IQ4_NL: wt = 20; break;  /* HTP_TYPE_IQ4_NL */
        case HVHX_V2_WT_MXFP4:  wt = 39; break;  /* HTP_TYPE_MXFP4  */
        default: return -1;
    }

    tiled_dequantize_state_t st;
    memset(&st, 0, sizeof(st));
    st.dst               = out;
    st.src               = (const uint8_t *)in;
    st.n_k_tiles         = n_k_tiles;
    st.n_k_tiles_div     = init_fastdiv_values(n_k_tiles);
    st.n_tot_tiles       = n_tot_tiles;
    st.n_tiles_per_task  = n_tot_tiles;
    st.n_tasks           = 1;
    st.n_cols            = n_cols;
    st.k_block           = k;
    st.row_stride        = (size_t)n_k_tiles *
                           hvhx_v2_weight_aligned_tile_size(type);
    st.weight_type       = (uint32_t)wt;
    st.tile_size         = hvhx_v2_weight_tile_size(type);
    st.aligned_tile_size = hvhx_v2_weight_aligned_tile_size(type);
    st.ctx               = NULL;
    st.traces            = NULL;

    switch (type) {
        case HVHX_V2_WT_Q4_0:
            dequantize_tiled_weight_to_fp16_task_q4_0(&st, 0, n_tot_tiles); break;
        case HVHX_V2_WT_Q4_1:
            dequantize_tiled_weight_to_fp16_task_q4_1(&st, 0, n_tot_tiles); break;
        case HVHX_V2_WT_Q8_0:
            dequantize_tiled_weight_to_fp16_task_q8_0(&st, 0, n_tot_tiles); break;
        case HVHX_V2_WT_IQ4_NL:
            dequantize_tiled_weight_to_fp16_task_iq4_nl(&st, 0, n_tot_tiles); break;
        case HVHX_V2_WT_MXFP4:
            dequantize_tiled_weight_to_fp16_task_mxfp4(&st, 0, n_tot_tiles); break;
        default: return -1;
    }
    return 0;
#else
    (void)in; (void)out; (void)n_cols; (void)k; (void)type;
    return -1;   /* host: 无 HVX, dequant 不可用 */
#endif
}

/* ============================================================
 *  HMX crouton GEMM (dot, zero-init)
 * ============================================================ */
void hvhx_v2_hmx_gemm_dot_fp16(__fp16       * __restrict__ out,
                               const __fp16 * __restrict__ act,
                               const __fp16 * __restrict__ wgt,
                               const __fp16 * __restrict__ scales,
                               uint32_t n_row_tiles,
                               uint32_t n_col_tiles,
                               uint32_t n_dot_tiles)
{
#if HVX_V2_KERNELS_ENABLED
    hmx_enable_execution();
    core_dot_chunk_fp16(out, act, wgt, scales,
                        n_row_tiles, n_col_tiles, n_dot_tiles);
    hmx_disable_execution();
#else
    (void)out; (void)act; (void)wgt; (void)scales;
    (void)n_row_tiles; (void)n_col_tiles; (void)n_dot_tiles;
#endif
}

/* ============================================================
 *  HMX crouton MMA (可累加到现有 C)
 * ============================================================ */
void hvhx_v2_hmx_gemm_mma_fp16(__fp16       * __restrict__ C,
                               const __fp16 * __restrict__ A,
                               const __fp16 * __restrict__ B,
                               const __fp16 * __restrict__ col_scales,
                               const __fp16 * __restrict__ eye_tile,
                               uint32_t n_row_tiles,
                               uint32_t n_col_tiles,
                               uint32_t n_dot_tiles,
                               int zero_init)
{
#if HVX_V2_KERNELS_ENABLED
    hmx_enable_execution();
    /* eye_tile 必须非空才能走 accumulate 路径; 否则强制 zero_init */
    if (eye_tile == NULL) zero_init = 1;
    core_mma_chunk_fp16(C, A, B, col_scales, eye_tile,
                        n_row_tiles, n_col_tiles, n_dot_tiles,
                        zero_init ? true : false);
    hmx_disable_execution();
#else
    (void)C; (void)A; (void)B; (void)col_scales; (void)eye_tile;
    (void)n_row_tiles; (void)n_col_tiles; (void)n_dot_tiles; (void)zero_init;
#endif
}

/* ============================================================
 *  tile-major ↔ row-major I/O 变换 (hmx-mm-kernels-tiled.h:769,866)
 * ============================================================ */
void hvhx_v2_transfer_output_fp16_to_fp32(
    float         * __restrict__ dst,
    const float   * __restrict__ src2,
    const __fp16  * __restrict__ vtcm_src,
    uint32_t start_row, uint32_t n_rows, uint32_t n_cols,
    uint32_t dst_stride, uint32_t src2_stride, uint32_t dst_cols)
{
#if HVX_V2_KERNELS_ENABLED
    transfer_output_chunk_fp16_to_fp32(dst, src2, vtcm_src,
                                       start_row, n_rows, n_cols,
                                       dst_stride, src2_stride, dst_cols);
#else
    /* 标量参考: tile[r0,c0] 的 vector r1 打包 2 行,
     * fp16[0..31] = tile 行 2*r1, fp16[32..63] = tile 行 2*r1+1. */
    const uint32_t n_col_tiles  = n_cols / 32;
    const uint32_t tile_row_st  = n_col_tiles * 1024;
    const uint32_t limit_c      = (n_cols < dst_cols) ? n_cols : dst_cols;
    for (uint32_t r = 0; r < n_rows; ++r) {
        const uint32_t gr   = start_row + r;
        const uint32_t r0   = gr / 32;
        const uint32_t wt   = gr % 32;
        const uint32_t r1   = wt / 2;
        const uint32_t half = wt % 2;
        const __fp16 *row_base = vtcm_src + r0 * tile_row_st;
        for (uint32_t c = 0; c < limit_c; ++c) {
            const uint32_t c0 = c / 32;
            const uint32_t lc = c % 32;
            const __fp16 *tile = row_base + c0 * 1024;
            float val = (float)tile[r1 * 64 + half * 32 + lc];
            if (src2) val += src2[r * src2_stride + c];
            dst[r * dst_stride + c] = val;
        }
    }
#endif
}

void hvhx_v2_transfer_activation_fp32_to_fp16(
    __fp16        * __restrict__ vtcm_dst,
    const float   * __restrict__ src,
    uint32_t n_rows, uint32_t k_block, uint32_t k_stride, uint32_t k_valid)
{
#if HVX_V2_KERNELS_ENABLED
    transfer_activation_chunk_fp32_to_fp16(vtcm_dst, src, n_rows,
                                           k_block, k_stride, k_valid);
#else
    const uint32_t n_col_tiles = k_block / 32;
    for (uint32_t gr = 0; gr < n_rows; ++gr) {
        const uint32_t r0     = gr / 32;
        const uint32_t wt     = gr % 32;
        const uint32_t r1half = wt / 2;
        const uint32_t half   = wt % 2;
        for (uint32_t c = 0; c < k_valid; ++c) {
            const uint32_t c0 = c / 32;
            const uint32_t lc = c % 32;
            const uint32_t tile_idx = r0 * n_col_tiles + c0;
            __fp16 *tile = vtcm_dst + tile_idx * 1024;
            tile[r1half * 64 + half * 32 + lc] = (__fp16)src[gr * k_stride + c];
        }
    }
#endif
}
