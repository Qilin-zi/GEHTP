/*
 * hvx_activation.c — HVX 激活函数 (HardSwish Q12 + PReLU)
 * Module: hvx-activation
 * Math:   HardSwish: f(x)=x*clamp(x+3,0,6)/6  (Q12 定点, ≤2 LSB)
 *         PReLU:     f(x)= x>0 ? x : slope*x   (u8 偏移二进制, 零点 0x80)
 * Note:   详见 docs/api_hvx_activation.md.
 */
#include "hvx_activation.h"
#include <hvx_hexagon_protos.h>

/* Q12 偏移常量 */
#define HVHX_Q12_ONE       (1 << 12)              /* 1.0 in Q12      */
#define HVHX_Q12_THREE     (3 << 12)              /* 3.0 in Q12      */
#define HVHX_Q12_SIX       (6 << 12)              /* 6.0 in Q12      */
#define HVHX_HARDSWISH_SHIFT  14                   /* x * cl: Q24 → Q10  */
                                                    /* +2 = Q12          */

/* ============================================================
 *  HardSwish flat u16
 * ============================================================ */
void hvhx_hardswish_flat_u16(const uint16_t * __restrict__ in,
                             uint16_t       * __restrict__ out,
                             uint32_t n)
{
    if (n == 0) return;
    const uint32_t n_vec  = n & ~(HVHX_VEC_ELEM_U16 - 1u);

#if defined(__HVX__) || defined(__hexagon__)
    HVX_Vector v_off   = Q6_Vh_vsplat_R(HVHX_Q12_THREE);
    HVX_Vector v_lo    = Q6_Vh_vsplat_R(0);
    HVX_Vector v_hi    = Q6_Vh_vsplat_R(HVHX_Q12_SIX);
    /* 1/6 在 Q14 = round(16384/6) = 2731.
     * 注意: vmpyiwh 只填偶 word lane (半速率 DV 指令), 不能用于
     * 全 lane word×scalar. 改用 vmpyiewuh(word × uh-broadcast). */
    HVX_Vector v_recip6 = Q6_Vh_vsplat_R(2731);

    for (uint32_t i = 0; i < n_vec; i += HVHX_VEC_ELEM_U16) {
        HVX_Vector v_x  = *(const HVX_Vector *) (in + i);

        /* 1. x + 3, clamp [0, 6<<12] */
        HVX_Vector v_xo = Q6_Vh_vadd_VhVh(v_x, v_off);
        HVX_Vector v_c  = Q6_Vh_vmin_VhVh(v_hi, v_xo);
        v_c             = Q6_Vh_vmax_VhVh(v_lo, v_c);

        /* 2. vmpy(x, cl) → 32-bit pair (Q24), signed 16x16→32 */
        HVX_VectorPair v_mp = Q6_Ww_vmpy_VhVh(v_x, v_c);
        HVX_Vector p_even = Q6_V_lo_W(v_mp);  /* 偶 lane: 32-bit 乘积 */
        HVX_Vector p_odd  = Q6_V_hi_W(v_mp);  /* 奇 lane */

        /* 3. /24576 = /4096/6 = vasr(12) 后乘 1/6(Q14=2731) 再 vasr(14) */
        HVX_Vector ve = Q6_Vw_vasr_VwR(p_even, 12);
        HVX_Vector vo = Q6_Vw_vasr_VwR(p_odd,  12);
        ve = Q6_Vw_vmpyie_VwVuh(ve, v_recip6);
        vo = Q6_Vw_vmpyie_VwVuh(vo, v_recip6);
        ve = Q6_Vw_vasr_VwR(ve, 14);
        vo = Q6_Vw_vasr_VwR(vo, 14);

        /* 4. 收窄 2×32-word → 64-halfword (饱和), arg 顺序: odd, even */
        HVX_Vector v_r = Q6_Vh_vasr_VwVwR_sat(vo, ve, 0);

        *(HVX_Vector *) (out + i) = v_r;
    }
#else
    for (uint32_t i = 0; i < n_vec; ++i) {
        out[i] = (uint16_t) hvhx_hardswish_scalar((int16_t) in[i]);
    }
#endif

    for (uint32_t i = n_vec; i < n; ++i) {
        out[i] = (uint16_t) hvhx_hardswish_scalar((int16_t) in[i]);
    }
}

/* ============================================================
 *  HardSwish crouton 32x32 (u16)
 *   - 输入按 crouton 32x32 块布局 (HW 维连续, 1 个 channel)
 *   - 多 batch 平行 (channels = batches)
 *   - 直接对每个元素套用 HardSwish, 但保持原内存布局
 *
 *  实现: 与 flat 路径等价, 但参数上是 32 行 × (batches*32) 列的 tensor
 * ============================================================ */
void hvhx_hardswish_crouton_u16(const uint16_t * __restrict__ in,
                                uint32_t batches,
                                uint16_t       * __restrict__ out)
{
    if (batches == 0) return;
    const uint32_t total = batches * 32u * 32u;
    hvhx_hardswish_flat_u16(in, out, total);
}

/* ============================================================
 *  PReLU u8
 *   in[i] (u8) 视为 [-128, 127] (s8 域)
 *   slope_q7  : 1..255  → 0.0078..0.996
 *   out[i] = in[i] if in[i] (s8) > 0
 *          = (in[i] * slope) >> 7   (s8 域)
 * ============================================================ */
void hvhx_prelu_u8(const uint8_t * __restrict__ in,
                   uint8_t slope_q7,
                   uint8_t       * __restrict__ out,
                   uint32_t n)
{
    if (n == 0) return;
    const uint32_t n_vec = n & ~(HVHX_VEC_ELEM_U8 - 1u);

#if defined(__HVX__) || defined(__hexagon__)
    /* 0x80 = u8 零点 (s8↔u8 转换) */
    HVX_Vector v_zp    = Q6_Vb_vsplat_R(0x80);
    HVX_Vector v_slope = Q6_Vh_vsplat_R(slope_q7);

    for (uint32_t i = 0; i < n_vec; i += HVHX_VEC_ELEM_U8) {
        HVX_Vector v_in = *(const HVX_Vector *) (in + i);

        /* s8 = u8 - 0x80 */
        HVX_Vector v_s8 = Q6_Vb_vsub_VbVb(v_in, v_zp);

        /* s8 → s16 pair (sign extend). |s8*slope| ≤ 128*255=32640 < 32768,
         * 故 s16 乘积不溢出, 可直接用 vmpyih (取低 16 位) */
        HVX_VectorPair v_s16p = Q6_Wh_vsxt_Vb(v_s8);
        HVX_Vector v_pl = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(v_s16p), v_slope);
        HVX_Vector v_ph = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(v_s16p), v_slope);

        /* 2×64 s16 → 128 s8, >>7 饱和. vasrhb(Vu,Vv,R): out[2k]=Vv, [2k+1]=Vu.
         * want even←v_pl(even s8), odd←v_ph(odd s8) → Vv=v_pl, Vu=v_ph */
        HVX_Vector v_byte = Q6_Vb_vasr_VhVhR_sat(v_ph, v_pl, 7);

        /* 还原零点 (byte 加, 自然 mod 256) */
        HVX_Vector v_neg_u8 = Q6_Vb_vadd_VbVb(v_byte, v_zp);

        /* u8 > 0x80 (s8>0) → 原值; 否则 → 缩放后 */
        HVX_VectorPred q_pos = Q6_Q_vcmp_gt_VubVub(v_in, v_zp);
        HVX_Vector v_out = Q6_V_vmux_QVV(q_pos, v_in, v_neg_u8);

        *(HVX_Vector *) (out + i) = v_out;
    }
#else
    for (uint32_t i = 0; i < n_vec; ++i) {
        out[i] = hvhx_prelu_scalar_u8((int8_t) (in[i] - 0x80), slope_q7);
    }
#endif

    for (uint32_t i = n_vec; i < n; ++i) {
        out[i] = hvhx_prelu_scalar_u8((int8_t) (in[i] - 0x80), slope_q7);
    }
}
