/* gemm_dispatch.h — U17 MatMul 三路由决策 (例 17/18 的决策层抽象)
 *
 * 决策表 (纯函数, host 可对拍):
 *   M % 256 == 0 && M >= 256        → GR_W4A16    (HMX W4 引擎; 单 invoke 仅
 *                                                   M=256, M>256 拆块拼接)
 *   M < 32                          → GR_SMALLM   (pad-256 复用 W4 引擎, 取前 M 行;
 *                                                   例 18 结论: M=1 与 256 同价)
 *   其余 (32..255 / 非 256 倍数)     → GR_DENSE_F16 (f16 便携 kernel, f32 累加)
 *
 * 边界: M=1 SMALLM / M=32 DENSE / M=128 DENSE / M=256 W4A16 / M=512 W4A16
 */
#ifndef HVXHMX_V23_GEMM_DISPATCH_H
#define HVXHMX_V23_GEMM_DISPATCH_H

#include <stdint.h>
#include "dc_parts.h"

enum gemm_route { GR_W4A16 = 0, GR_SMALLM = 1, GR_DENSE_F16 = 2 };

int         gemm_route_for(uint32_t m, uint32_t k, uint32_t n);
const char* gemm_route_name(int r);

/* GR_DENSE_F16 执行体: c[m][n] = Σ_k a[m][k]·w[k][n] (f16 入, f32 累加, f16 出) */
void gemm_f16_dense(const int16_t* a, const int16_t* w, int16_t* c,
                    uint32_t m, uint32_t k, uint32_t n);

/* GR_SMALLM 执行体: 线性 f16 激活前 m 行 → pad-256 (crouton 重编码)
 * → dc_w4 引擎 → 解码取前 m 线性行。e: 已 carve(M=256,K,N) 且 wt/bias 已
 * stage。act_lin = m*k 个 f16 (row-major); out_lin = m*n 个 f16。返回 0 成功。 */
int gemm_smallm_pad256(struct dc_w4* e, const int16_t* act_lin, uint32_t m,
                       int16_t* out_lin);

/* GR_W4A16 执行体: m 为 256 倍数, 每 256 行块一次 invoke + 逐块解码拼接。
 * kernel ABI 单 invoke 固定 M=256 (descriptor n_tiles_pow2=32/m_total_minus_step=8
 * → M-loop 8×32 行, 常量段逐字保留不可改), M>256 必须本层拆块 — 512 面
 * 单 invoke 直发高位行错误 (例 31 G2b 实测)。e 同上 (carve M=256)。 */
int gemm_w4a16_m256(struct dc_w4* e, const int16_t* act_lin, uint32_t m,
                    int16_t* out_lin);

/* crouton16_row4 编解码 (row-major f16 位型 ⇔ HMX 面; rows%32==0, cols%32==0) */
void gemm_crouton_encode(const uint16_t* lin, uint16_t* surf,
                         uint32_t rows, uint32_t cols);
void gemm_crouton_decode(const uint16_t* surf, uint16_t* lin,
                         uint32_t rows, uint32_t cols);

#endif
