/*
 * hvxhmx_v2_rope.h — V2 Rotary Position Embedding (RoPE)
 * =====================================================================
 * 源自 ggmlHTPV3E htp/rope-ops.c 的核心数学内核:
 *   - rope_yarn_one / rope_yarn_ramp / rope_corr_dims  (YaRN 标量参数)
 *   - rope_cache_init  (HVX 向量化 cos/sin 表构建, V79+ fast path)
 *   - hvx_rope_neox_f32_aa  (split-half 旋转, GPT-NeoX 风格)
 *   - hvx_rope_f32_aa       (interleaved 旋转, 原始 LLaMA 风格)
 *
 * 数据模型:
 *   theta_cache = [ne0] 个 f32, 布局 [cos0, sin0, cos1, sin1, ...].
 *   由 hvhx_v2_rope_cache_init() 一次性构建, 同一 position 的所有 head 共用.
 *
 * 外层驱动 (DMA / 多线程 / 多 batch / position 索引) 在 ggmlHTPV3E 里与
 * tensor 模型耦合, 此处只暴露已验证的单行数学内核 + cache 构建.
 *
 * 数学:
 *   theta_scale = powf(freq_base, -2.0f / n_dims)
 *   NEOX (split-half):  x[i],x[i+he] = rotate(x[i], x[i+he], θ[i]), he = ne/2
 *   Normal (interleaved): x[2i],x[2i+1] = rotate(x[2i], x[2i+1], θ[2i])
 */
#ifndef HVXHMX_V2_ROPE_H
#define HVXHMX_V2_ROPE_H

#include "hvxhmx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  RoPE mode (与 ggml GML_ROPE_TYPE_* 对齐)
 * ============================================================ */
typedef enum {
    HVHX_V2_ROPE_NORMAL = 0,   /* interleaved (原始 LLaMA) */
    HVHX_V2_ROPE_NEOX   = 2,   /* split-half (GPT-NeoX / Llama2+) */
    HVHX_V2_ROPE_MROPE  = 8,   /* multi-dimensional (Qwen2-VL) */
    HVHX_V2_ROPE_VISION = 24,  /* vision (Qwen2-VL) */
    HVHX_V2_ROPE_IMROPE = 40,  /* interleaved mrope */
} hvhx_v2_rope_mode_t;

/* ============================================================
 *  YaRN 参数 helpers (标量, host-safe)
 * ============================================================ */

/**
 * @brief 计算 YaRN correction dims (ext_factor != 0 时用).
 * @param n_dims     旋转维数
 * @param n_ctx_orig 原始 context 长度
 * @param freq_base  base 频率 (通常 10000)
 * @param beta_fast  YaRN fast 参数 (通常 32)
 * @param beta_slow  YaRN slow 参数 (通常 1)
 * @param dims[2]    输出 [start, end] correction 区间
 */
void hvhx_v2_rope_corr_dims(int n_dims, int n_ctx_orig, float freq_base,
                            float beta_fast, float beta_slow,
                            float dims[2]);

/* ============================================================
 *  theta_cache 构建 (YaRN-scaled cos/sin 表)
 * ============================================================ */

/**
 * @brief 构建 theta_cache: cache[2i] = cos(θ_i)*mscale, cache[2i+1] = sin(θ_i)*mscale.
 *        DSP 路径: V79+ + ext_factor==0 走全向量化 fast path (32 pair/iter).
 *        Host/通用路径: 标量 rope_yarn_one 循环.
 * @param theta_base    当前 position 的基础 theta = pos[i2] (整数位置)
 * @param freq_scale    频率缩放 (1.0 / (freq_base/original_base) 或 rope_scaling)
 * @param freq_factors  可选 per-dim 频率因子 [ne0/2], NULL = 全 1
 * @param corr_dims     YaRN correction dims (由 hvhx_v2_rope_corr_dims 计算)
 * @param ne0           head_dim (cache 长度 = ne0)
 * @param ext_factor    外推因子 (0 = 纯插值)
 * @param mscale        attention 因子 (attn_factor, YaRN 用)
 * @param cache         输出 f32 [ne0], 128B 对齐性能最佳
 * @param theta_scale   = powf(freq_base, -2.0f/n_dims)
 */
void hvhx_v2_rope_cache_init(float theta_base, float freq_scale,
                             const float *freq_factors, const float corr_dims[2],
                             uint32_t ne0, float ext_factor, float mscale,
                             float *cache, float theta_scale);

/* ============================================================
 *  单行旋转内核 (f32, 128B 对齐性能最佳)
 * ============================================================ */

/**
 * @brief NEOX 风格单行旋转: split-half.
 *        src 布局: [he][he] (前半 real, 后半 imag), he = n_dims/2.
 *        cache:    [n_dims] (cos0,sin0,...,cos_{he-1},sin_{he-1}).
 *        out[i]    = src[i]*cos[i] - src[i+he]*sin[i]
 *        out[i+he] = src[i]*sin[i] + src[i+he]*cos[i],  i ∈ [0, he)
 * @param dst  输出 f32 [n_dims]
 * @param src  输入 f32 [n_dims]
 * @param n_dims  旋转维数 (偶数)
 * @param theta_cache  cos/sin 表 [n_dims]
 */
void hvhx_v2_rope_neox_f32(float * __restrict__ dst,
                            const float * __restrict__ src,
                            uint32_t n_dims,
                            const float * __restrict__ theta_cache);

/**
 * @brief Normal (interleaved) 风格单行旋转: 原始 LLaMA.
 *        pair (src[2i], src[2i+1]) 用 θ[2i] 旋转.
 * @param dst, src  f32 [n_dims]
 * @param n_dims    旋转维数 (偶数)
 * @param theta_cache  cos/sin 表 [n_dims]
 */
void hvhx_v2_rope_normal_f32(float * __restrict__ dst,
                              const float * __restrict__ src,
                              uint32_t n_dims,
                              const float * __restrict__ theta_cache);

/**
 * @brief 便捷: 完整一行 (n_dims 旋转 + tail 拷贝).
 *        n_dims < ne0 时, dst[n_dims..ne0] = src[n_dims..ne0] (直传).
 * @param dst, src  f32 [ne0]
 * @param n_dims    旋转维数
 * @param ne0       行完整长度 (head_dim, 可能 > n_dims)
 * @param theta_cache  cos/sin 表 [ne0]
 * @param mode      HVHX_V2_ROPE_NEOX 或 HVHX_V2_ROPE_NORMAL
 */
void hvhx_v2_rope_row_f32(float * __restrict__ dst,
                           const float * __restrict__ src,
                           uint32_t n_dims, uint32_t ne0,
                           const float * __restrict__ theta_cache,
                           hvhx_v2_rope_mode_t mode);

#ifdef __cplusplus
}
#endif

#endif /* HVXHMX_V2_ROPE_H */
