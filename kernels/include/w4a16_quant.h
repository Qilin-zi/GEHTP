/* w4a16_quant.h — W4A16 act/out 量化面函数 (平台无关; host 单测 + 设备 lib 同源)
 *
 * 闭包权威 = /4090disk2/htpw4a16_v81 prepare_owned_inputs.py + float_ref.py:
 *   act a16 域: u16 对称, q = round(a/A_s·32767) + 32768 (real=(q-32768)/32767)
 *   crouton16_row4 面: pack_a16_crouton16_row4_surface 公式
 *   out 反量化: f16 = A_s·S[n]·(q-32768)/32767 (S=权重列 scale 槽, 已含 /7)
 */
#ifndef W4A16_QUANT_H
#define W4A16_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* DDR f16 [n] → max|a| (GEMM 级 act scale; 全零 → 1.0) */
float w4a16_act_scale(const uint16_t* a_f16, uint32_t n);

/* f16 → a16 域 u16 (scale = max|a|; 越界钳 [0,65535]) */
uint16_t w4a16_quant_f16(uint16_t a_f16, float scale);

/* u16 线性 [m,k] → crouton16_row4 面 (m,k 必须 %32; 闭包 pack 公式:
 * row4_phase(8) 外 × kt × m32_group × row_pair(2), 相邻两行成对) */
void w4a16_pack_crouton(const uint16_t* lin, uint32_t m, uint32_t k, uint16_t* surf);

/* crouton16_row4 面 → u16 线性 [m,k] (pack 的精确逆) */
void w4a16_unpack_crouton(const uint16_t* surf, uint32_t m, uint32_t k, uint16_t* lin);

/* u16 线性输出 [m,n] → f16: A_s·S[n]·(q-32768)/32767 (S 为 f16 槽, 已含 /7;
 * out_row_bytes = 目标行跨度字节, 分块写全宽 N_full 时传 N_full*2) */
void w4a16_dequant_out(const uint16_t* lin_q, uint32_t m, uint32_t n,
                       float act_scale, const uint16_t* scale_f16, uint16_t* out_f16,
                       uint32_t out_row_bytes);

/* crouton 面 (m_pad×n) → 前 m_out 行 f16 (crouton 序直读, 无中间缓冲;
 * 行≥m_out 的 pad 行丢弃; out_row_bytes 同上) */
void w4a16_dequant_crouton(const uint16_t* surf, uint32_t m_pad, uint32_t n,
                           uint32_t m_out, float act_scale,
                           const uint16_t* scale_f16, uint16_t* out_f16,
                           uint32_t out_row_bytes);

#ifdef __cplusplus
}
#endif

#endif
