/* gdn_tree.h — U14 树形 GDN (gated delta net over tree topology)
 *
 * 线性形式 (gdn_sm chunk 数学直接推广, 路径 cumsum 换祖先三角系):
 *   Pg[i]   = g_i + Pg[parent(i)]                 (路径衰减和, root=0)
 *   L[j][i] = exp(Pg_j−Pg_i)·(k_j·k_i)·β_j        (i ∈ Anc(j), 严格下三角)
 *   b_j     = β_j·(v_j − exp(Pg_j)·S0 k_j)
 *   u       = (I+L)^{-1} b                        (复用 ref_solve_tri)
 *   y_j     = s·[ exp(Pg_j)(S0 q_j) + Σ_{i∈Anc(j)∪{j}} exp(Pg_j−Pg_i)(q_j·k_i) u_i ]
 *   S_a     = exp(Pg_a)·S0 + Σ_{j∈Anc(a)∪{a}} exp(Pg_a−Pg_j)·u_j k_j^T
 *   (s = 1/√D, 与 gdn_sm per-token oracle 同标度)
 *
 * 拓扑契约: parent[0] == -1 (单根); parent[i] ∈ [-1, i-1]; 升序即拓扑序。
 * 状态一律 f32 ([t][d][d]), 输入 f16, kernel 内部 f32。
 */
#ifndef HVXHMX_V23_GDN_TREE_H
#define HVXHMX_V23_GDN_TREE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 设备串行递归 (f16 输入): states 调用方给 [t][d][d] f32。
 * 逐节点: S_i = e^{g_i}·S_{parent}; δ=β(v−S_i k); S_i += δ⊗k; y=(S_i q)/√D。
 * 返回 0; -1 = 拓扑非法 (parent 越界 / parent[0] != -1) */
int gdn_tree_serial_f16(const float* S0, float* states, int d, int t,
                        const int* parent, const int16_t* k, const int16_t* v,
                        const int16_t* q, const int16_t* beta, const int16_t* g16,
                        int16_t* y);

/* oracle: 串行递归 f32 (与设备 kernel 同算法, 判定真值) */
void ref_delta_tree(const float* S0, int d, int t, const int* parent,
                    const float* k, const float* v, const float* q,
                    const float* beta, const float* g,
                    float* y, float* states);

/* oracle: 闭式解 f32 (祖先三角系; 与串行递归在 f32 内逐形一致) */
void ref_tree_closed(const float* S0, int d, int t, const int* parent,
                     const float* k, const float* v, const float* q,
                     const float* beta, const float* g,
                     float* y, float* states);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_GDN_TREE_H */
