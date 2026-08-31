/* gdn_sm.h — MODULE B 递归状态机 kernel 族 (B1 conv / B2 delta-rule / B3 solve-tri)
 *
 * 状态一律外置 (调用方持有), kernel 只做纯函数步进。
 * 数值契约 (host numpy 对拍已到 1e-14, 见 host/proto_chunk.py):
 *   B2 chunk 形式 (每 head, 独立推导 + per-token oracle 逐形对拍):
 *     G_j = Σ_{t<=j} g_t (inclusive cumsum)
 *     L[j,i] = exp(G_j-G_i) * (k_j·k_i) * β_j        (严格下三角)
 *     b_j    = β_j * (v_j - exp(G_j) * S0 k_j)
 *     u      = (I+L)^{-1} b                          ← B3 显式调用
 *     y_t    = exp(G_t)*(S0 q_t) + Σ_{i<=t} exp(G_t-G_i)(q_t·k_i) u_i
 *     S'     = exp(G_C)*S0 + Σ_i exp(G_C-G_i) u_i k_i^T
 *   per-token oracle (ggmlHTPV3E pp-thread 同式):
 *     S *= exp(g); yk=S k; δ=β(v-yk); S += δ⊗k; y = (S q)/√D
 *   B1 conv (ssm-conv.c 同式): y_t = silu(Σ_{i<d_conv} w[i,ch]*x[t-d_conv+1+i,ch])
 *
 * f16 = IEEE half 存 int16_t; kernel 内部 f32 计算; 状态一律 f32。
 */
#ifndef GDN_SM_H
#define GDN_SM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 状态布局 (DoD-2 状态文档, 与 ggmlHTPV3E 对应) ----
 * conv_state_t: win[(d_conv-1) * d_inner] f32, win[i*d_inner+ch] = 滑窗第 i 个输入
 *               (ggml: conv_states ne=[d_conv-1, d_inner], 同序)
 * rec_state_t:  s[h * d*d] f32, head h 的 S[j*d+n], j=v 维行, n=k 维列
 *               (ggmlHTPV3E pp-thread: s_work + row*S_v, 行=v 维, 同序)
 */
typedef struct { int d_inner, d_conv; float* win; } conv_state_t;
typedef struct { int h, d; float* s; }               rec_state_t;

/* ---- B1 conv1d (d_conv 因果卷积 + SiLU) ---- */
int conv1d_step_f16 (conv_state_t* st, const int16_t* w_f16, const int16_t* x, int16_t* y);
int conv1d_block_f16(conv_state_t* st, const int16_t* w_f16, const int16_t* x, int16_t* y, int m);

/* ---- B3 solve-tri: T = (I+L)^{-1}, L 严格下三角 C×C f16 ---- */
int solve_tri_f16(const int16_t* L, int16_t* T, int c);

/* ---- B2 delta-rule chunk 更新 (32 head 批) ----
 * 缓冲布局 [h][ntok][dim], 调用方传 base+t0*dim (只加 token 偏移, head 步幅内部算)
 * c = 本 chunk 实际 token 数 (<= C_MAX, 尾块可不对齐); ntok = 缓冲的完整 token 跨度
 * 内部: 每 head 构建 L (含 decay mask + β), 调 solve_tri_f16, 回代 u, 出 y, 推 S */
#define GDN_C_MAX 64
int delta_chunk_f16(rec_state_t* st, const int16_t* k, const int16_t* v,
                    const int16_t* q, const int16_t* beta, const int16_t* g,
                    int16_t* y, int c, int ntok);

/* ---- 标量 f32 oracle (host/device 同源, 判定真值) ---- */
void ref_conv_step (conv_state_t* st, const float* w, const float* x, float* y);
void ref_conv_block(conv_state_t* st, const float* w, const float* x, float* y, int m);
void ref_delta_token(rec_state_t* st, const float* k, const float* v,
                     const float* q, float beta, float g, float* y);
void ref_solve_tri (const float* L, float* T, int c);   /* T=(I+L)^{-1} 回代 */

/* f16 工具 */
float  gdn_f16_to_f32(int16_t h);
int16_t gdn_f32_to_f16(float f);

/* 可复现输入 (host python 同 LCG, 免资产传输) */
uint32_t gdn_lcg_next(uint32_t* st);
float    gdn_lcg_norm(uint32_t* st);      /* ~N(0,0.5) 近似 (和差法) */

#ifdef __cplusplus
}
#endif
#endif /* GDN_SM_H */
