# V2 — LLM 算子层 API

V2 在 V1 (runtime/hmx/hvx/compat) 之上新增 LLM 推理算子层 (已并入 `src/hmx|hvx/` + 
`include/*.h`, internal 在 `include/internal/`), 源自 ggmlHTPV3E 的已验证 HVX/HMX kernel. 一行引入:

```c
#include "v2/hvxhmx_v2.h"
```

## 设计约定

- **公共符号** 统一 `hvhx_v2_*` 前缀, `extern "C"` 保护, 仅依赖 `hvxhmx_types.h`.
- **DSP 路径** (`#if defined(__HVX__) || defined(__hexagon__)`): 调 V81 HVX/HMX
  向量化 kernel (源自 ggmlHTPV3E htp/*.h).
- **Host 路径** (`#else`): 标量 libm fallback, 数值与 DSP 一致 (供 host 回归对比).
- **内部 kernel** (`include/internal/*.h`): 仅在 `.c` 的 DSP 分支 include,
  HVX 原语不泄漏到公共接口.
- **tile-major interleaved** (HMX): 见 [data_layout.md](data_layout.md).
  32×32×2B = 2048B tile, 1024 elements, VTCM 2KB 对齐.

## 模块总览

| 优先级 | 模块 | 头文件 | 符号数 |
|--------|------|--------|--------|
| P0 | Norm | `hvxhmx_v2_norm.h` | 6 |
| P0 | MatMul (HMX tiled GEMM + dequant + transfer) | `hvxhmx_v2_matmul.h` | 5 |
| P1 | Flash Attention | `hvxhmx_v2_flash_attn.h` | 6 |
| P1 | RoPE | `hvxhmx_v2_rope.h` | 5 |
| P2 | Unary (exp/log/sqrt/sin/silu/gelu/...) | `hvxhmx_v2_unary.h` | 22 |
| P2 | Binary (add/sub/mul/div) | `hvxhmx_v2_binary.h` | 6 |
| P2 | Softmax | `hvxhmx_v2_softmax.h` | 2 |
| P2 | SSM (solve_tri + GDN + ssm_conv) | `hvxhmx_v2_ssm.h` | 8 |
| | **合计** | | **60** |

---

## P0 — Norm

RMSNorm / LayerNorm / L2Norm, 每行一个 head_dim.

```c
void hvhx_v2_rms_norm_mul_f32_rows(const float *src, const float *weight,
                                    float *dst, uint32_t n_rows, uint32_t row_size, float eps);
```

LLM 主路径: `src` = [n_rows × row_size], `weight` = [row_size], `dst` = 归一化后 × weight.

## P0 — MatMul (HMX tiled GEMM + 量化 dequant)

tile-major interleaved 数据流: cooker → dequant → interleave → gemm → transfer.

```c
/* 1. 量化权重 dequant → tile-major fp16 (一次性) */
hvhx_v2_dequant_tiled_to_fp16(q_wgt, w_fp16, N, K, HVHX_V2_WT_Q4_0);

/* 2. 激活 row-major → tile-major (用 internal/hmx-utils.h 的 interleave) */

/* 3. HMX crouton GEMM: C[M,N] = A[M,K]·B[K,N] */
hvhx_v2_hmx_gemm_dot_fp16(C_tiles, A_tiles, B_tiles, scales,
                           M/32, N/32, K/32);
```

约束: M/K/N 为 32 倍数. 需先 `hmx_runtime_setup()` (V1 runtime).

**tile ↔ row-major I/O 变换** (transfer_*):
```c
/* HMX tile-major fp16 输出 → row-major fp32 (+ 可选残差 src2) */
hvhx_v2_transfer_output_fp16_to_fp32(dst, src2, vtcm_src, start_row,
                                      n_rows, n_cols, dst_stride,
                                      src2_stride, dst_cols);

/* row-major fp32 激活 → HMX tile-major fp16 */
hvhx_v2_transfer_activation_fp32_to_fp16(vtcm_dst, src, n_rows,
                                         k_block, k_stride, k_valid);
```
tile 布局: 每 32×32 tile = 1024 fp16, vector r1 (0..15) 打包 2 行
(fp16[0..31] = tile 行 2·r1, fp16[32..63] = tile 行 2·r1+1).

**组成 recipe**: 完整 GEMM 驱动 (VTCM chunking + DMA + 多线程) 在 ggmlHTPV3E
matmul-ops.c 里与 tensor 模型耦合, 此处暴露已验证的 crouton 级 GEMM 内核
(`core_dot_chunk_fp16` / `core_mma_chunk_fp16`) + dequant + tile 常量
+ tile↔row-major I/O 变换 (`transfer_*`).

## P1 — Flash Attention

online-softmax tile 级 building blocks, 调用方编排外层 KV 循环.

```c
/* HVX 单行 Q·K^T 点积 (缩放) */
float score;
hvhx_v2_fa_qk_dot_f16(q_row, k_row, head_dim, scale, &score);

/* HVX attn·V 累加: y += v * attn_weight */
hvhx_v2_fa_attn_v_mad_f16(out_acc, v_row, &attn_weight, head_dim);

/* HMX 32×32 tile 级 (高吞吐) */
hvhx_v2_hmx_fa_qk_dot_tile(q_tiles, k_tiles, s_tile, n_dot_tiles);   /* head_dim/32 */
hvhx_v2_hmx_fa_o_update_tile(d_diag, o_rc, p_tiles, v_tiles, o_out, n_col_tiles);
hvhx_v2_hmx_fa_o_norm_tile(d_diag, o_rc, o_out);   /* 收尾 /exp_sum */
```

`d_diag` = 1/row_exp_sum broadcast tile; `o_rc` = 当前 O tile. HMX tile 内核
**不** 内部调 `hmx_enable_execution` — 外层统一管理 HMX 锁.

ALiBi: `hvhx_v2_alibi_slopes(kv_head, G, n_head_log2, m0, m1, out32)` → 32 个 slope.

## P1 — RoPE

```c
float theta_scale = powf(freq_base, -2.0f / n_dims);
float corr[2];  hvhx_v2_rope_corr_dims(n_dims, n_ctx_orig, freq_base, 32, 1, corr);

/* 每个新 position: 重建 cos/sin 表 (同 position 的所有 head 共用) */
hvhx_v2_rope_cache_init(pos, freq_scale, freq_factors, corr,
                         head_dim, ext_factor, attn_factor,
                         theta_cache, theta_scale);

/* 逐行旋转 (NEOX 或 Normal) */
hvhx_v2_rope_row_f32(dst_row, src_row, n_dims, head_dim,
                      theta_cache, HVHX_V2_ROPE_NEOX);
```

YaRN fast path (V79+ + ext_factor==0): 全向量化 32 pair/iter.

## P2 — Unary

```c
hvhx_v2_exp_f32(dst, src, n);
hvhx_v2_sigmoid_f32(dst, src, n);
hvhx_v2_tanh_f32(dst, src, n);
hvhx_v2_sin_f32(dst, src, n);
hvhx_v2_scale_f32(dst, src, n, 0.125f);
float sum = hvhx_v2_reduce_sum_f32(src, n);
```

覆盖: exp/neg_exp/log/sqrt/rsqrt/sigmoid/tanh/sin/cos/pow/pow_const_base/
inverse/sqr/floor/truncate/scale/scale_offset/reduce_sum.

**激活 (act-ops.c 数学)**:
```c
hvhx_v2_silu_f32      (dst, src, n);   /* x · sigmoid(x) */
hvhx_v2_gelu_f32      (dst, src, n);   /* x · sigmoid(1.702·x)  (quick 近似) */
hvhx_v2_gelu_tanh_f32 (dst, src, n);   /* 0.5·x·(1+tanh(√(2/π)·x·(1+0.044715·x²))) */
hvhx_v2_softplus_f32  (dst, src, n);   /* log(1+exp(x)) */
```
HVX 路径: vec 级 building block 组合 (sigmoid 经 exp guard 防溢出).

## P2 — Binary

```c
hvhx_v2_add_f32(dst, a, b, n);
hvhx_v2_mul_f32(dst, a, b, n);
hvhx_v2_div_f32(dst, a, b, n);         /* a * inverse(b), HVX 近似倒数 */
hvhx_v2_add_scalar_f32(dst, a, bias, n);
```

## P2 — Softmax

```c
/* fused online softmax (max→exp→sum→scale), n 为 32 倍数 */
hvhx_v2_softmax_f32(src, dst, pad, n);   /* pad = VTCM scratch [n] */

/* FA mask 预处理: dst = src*scale + mask*slope */
hvhx_v2_softmax_mask_f32(src, dst, n, scale, mask, slope);
```

非 32 倍数 / 非对齐: 自动走标量 fallback.

## P2 — SSM

```c
/* 下三角前代解 L·X = B 的第 row 行 (GDN/linear attn 用) */
hvhx_v2_solve_tri_row_f32(L_row, B_row, X, row, k, col0, coln, inv_diag);

/* gated delta net 4/8 状态门控更新 + dot 积 */
float sums[4];
hvhx_v2_gdn_mul_dot4_f32(s0, s1, s2, s3, forget_gate, query, n, sums);
/* dst_k[i] *= mul[i]; sums[k] = Σ dst_k[i]·dot[i] */
```

**depthwise 1D causal conv (Mamba/Jamba 前导卷积, ssm-conv.c)**:
```c
/* 全量 row-major 标量 (任意尺寸, 正确性基准) */
hvhx_v2_ssm_conv_f32(src0, src1, dst, d_inner, n_t, n_s, ncs, d_conv, apply_silu);

/* HVX 32-channel tile 级 (已转置 channel-contiguous 布局, 高吞吐) */
hvhx_v2_ssm_conv_dot32_f32(dst32, x_row, x_stride, w_row, w_stride, d_conv, apply_silu);

/* 32×32 f32 转置 (dot32 的前置变换) */
hvhx_v2_transpose_32x32_f32(m32x32);
```
数学: `dst[i1,i2,i3] = Σ_{j} src0[i2+j,i1,i3] · src1[j,i1]`. dot32 内核体
取自 ssm-conv.c:318-345 (Q6_Vqf32 累加 + 可选 silu 融合).

---

## 构建

```bash
cd /disk1/V81Dev/hvxhmx_libsV2
./build_libs.sh
# → lib/libhvxhmx_v2.so          (124 KB, 321 T symbols, vgather=0)
# → lib/libhvxhmx_v2.signed.so   (SWIV 签名版)
```

工具链: hexagon-clang 19.0.07, flags `-mv81 -O2 -mhvx -mhvx-length=128B -mhmx`.

## 源映射

| V2 模块 | ggmlHTPV3E 源 |
|---------|---------------|
| norm | htp/hvx-norm.h |
| matmul | htp/hmx-mm-kernels-tiled.h + matmul-ops.h (GEMM + dequant + transfer_*) |
| flash_attn | htp/hvx-fa-kernels.h + hmx-fa-kernels.h + hvx-flash-attn.h |
| rope | htp/rope-ops.c (核心内核提取) |
| unary | htp/hvx-{exp,log,sqrt,sigmoid,sin-cos,pow,inverse,floor,scale,arith,reduce}.h + act-ops.c (silu/gelu/softplus 数学) |
| binary | htp/hvx-base.h (vec add/sub/mul) |
| softmax | htp/softmax-ops.c (hvx_fast_softmax_f32 + prep) |
| ssm | htp/solve-tri-ops.c + gated-delta-net-ops.c + ssm-conv.c (内核提取到 internal/ssm-kernels.h) |
