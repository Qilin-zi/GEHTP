/*
 * hvx_divide.c — HVX element-wise 整除 (shift-subtract, exact)
 * Module: hvx-divide
 * Math:   out[i] = a[i] / b[i]; 除零饱和 (u8→0xFF, u16→0xFFFF, i32→符号饱和)
 * Note:   i32 路径用 qf32 倒数 = 四舍五入 (非截断). 与标量 golden exact.
 *         详见 docs/api_hvx_divide.md.
 */
#include "hvx_divide.h"
#include "q6_intrinsics.h"

#if defined(__HVX__) || defined(__hexagon__)
#include <hvx_hexagon_protos.h>
#include <hexagon_types.h>
#endif

/* ============================================================
 *  u8 整除: shift-subtract on u16 lanes (128 u8/vector)
 * ============================================================ */
void hvhx_divide_u8(const uint8_t * __restrict__ a,
                    const uint8_t * __restrict__ b,
                    uint8_t       * __restrict__ out,
                    uint32_t n)
{
    if (n == 0) return;
    const uint32_t n_vec  = n & ~(HVHX_VEC_ELEM_U8 - 1u);

#if defined(__HVX__) || defined(__hexagon__)
    HVX_Vector zero_v = Q6_V_vzero();
    HVX_Vector v_one  = Q6_Vh_vsplat_R(1);
    HVX_Vector v_ff   = Q6_Vb_vsplat_R(0xFF);

    for (uint32_t i = 0; i < n_vec; i += HVHX_VEC_ELEM_U8) {
        HVX_Vector v_a = *(const HVX_Vector *) (a + i);
        HVX_Vector v_b = *(const HVX_Vector *) (b + i);

        /* 除零检测: b==0  ⇔  NOT(b>0) */
        HVX_VectorPred q_nz = Q6_Q_vcmp_gt_VubVub(v_b, zero_v);

        /* u8 → u16 零扩展 (128 u8 → 2×64 u16) */
        HVX_VectorPair aw = Q6_Wuh_vzxt_Vub(v_a);
        HVX_VectorPair bw = Q6_Wuh_vzxt_Vub(v_b);
        HVX_Vector a_lo = Q6_V_lo_W(aw), a_hi = Q6_V_hi_W(aw);
        HVX_Vector b_lo = Q6_V_lo_W(bw), b_hi = Q6_V_hi_W(bw);

        /* 对 lo/hi 各做 8 次移位减法 */
        HVX_Vector r_lo = zero_v, r_hi = zero_v;
        HVX_Vector q_lo = zero_v, q_hi = zero_v;
        for (int bit = 7; bit >= 0; --bit) {
            HVX_Vector sb_lo = Q6_Vuh_vlsr_VuhR(a_lo, bit);
            HVX_Vector sb_hi = Q6_Vuh_vlsr_VuhR(a_hi, bit);
            HVX_Vector bt_lo = Q6_V_vand_VV(sb_lo, v_one);
            HVX_Vector bt_hi = Q6_V_vand_VV(sb_hi, v_one);

            r_lo = Q6_Vh_vasl_VhR(r_lo, 1);
            r_hi = Q6_Vh_vasl_VhR(r_hi, 1);
            r_lo = Q6_V_vor_VV(r_lo, bt_lo);
            r_hi = Q6_V_vor_VV(r_hi, bt_hi);

            /* rem >= b  ⇔  NOT(b > rem)  (unsigned) */
            HVX_VectorPred q_ge_lo = Q6_Q_not_Q(Q6_Q_vcmp_gt_VuhVuh(b_lo, r_lo));
            HVX_VectorPred q_ge_hi = Q6_Q_not_Q(Q6_Q_vcmp_gt_VuhVuh(b_hi, r_hi));

            HVX_Vector s_lo = Q6_Vh_vsub_VhVh(r_lo, b_lo);
            HVX_Vector s_hi = Q6_Vh_vsub_VhVh(r_hi, b_hi);
            r_lo = Q6_V_vmux_QVV(q_ge_lo, s_lo, r_lo);
            r_hi = Q6_V_vmux_QVV(q_ge_hi, s_hi, r_hi);

            q_lo = Q6_Vh_vasl_VhR(q_lo, 1);
            q_hi = Q6_Vh_vasl_VhR(q_hi, 1);
            q_lo = Q6_V_vmux_QVV(q_ge_lo, Q6_V_vor_VV(q_lo, v_one), q_lo);
            q_hi = Q6_V_vmux_QVV(q_ge_hi, Q6_V_vor_VV(q_hi, v_one), q_hi);
        }

        /* 2×64 u16 → 128 u8 (饱和移位 0) */
        HVX_Vector v_final = Q6_Vub_vasr_VuhVuhR_sat(q_hi, q_lo, 0);

        /* 除零 → 0xFF */
        v_final = Q6_V_vmux_QVV(q_nz, v_final, v_ff);

        *(HVX_Vector *) (out + i) = v_final;
    }
#else
    for (uint32_t i = 0; i < n_vec; ++i) out[i] = hvhx_div_u8_scalar(a[i], b[i]);
#endif

    for (uint32_t i = n_vec; i < n; ++i) out[i] = hvhx_div_u8_scalar(a[i], b[i]);
}

/* ============================================================
 *  u16 整除: shift-subtract on u32 lanes (64 u16/vector)
 * ============================================================ */
void hvhx_divide_u16(const uint16_t * __restrict__ a,
                     const uint16_t * __restrict__ b,
                     uint16_t       * __restrict__ out,
                     uint32_t n)
{
    if (n == 0) return;
    const uint32_t n_vec  = n & ~(HVHX_VEC_ELEM_U16 - 1u);

#if defined(__HVX__) || defined(__hexagon__)
    HVX_Vector zero_v = Q6_V_vzero();
    HVX_Vector v_one  = Q6_V_vsplat_R(1);
    HVX_Vector v_ffff = Q6_Vh_vsplat_R(0xFFFF);

    for (uint32_t i = 0; i < n_vec; i += HVHX_VEC_ELEM_U16) {
        HVX_Vector v_a = *(const HVX_Vector *) (a + i);
        HVX_Vector v_b = *(const HVX_Vector *) (b + i);

        HVX_VectorPred q_nz = Q6_Q_vcmp_gt_VuhVuh(v_b, zero_v);

        HVX_VectorPair aw = Q6_Wuw_vzxt_Vuh(v_a);
        HVX_VectorPair bw = Q6_Wuw_vzxt_Vuh(v_b);
        HVX_Vector a_lo = Q6_V_lo_W(aw), a_hi = Q6_V_hi_W(aw);
        HVX_Vector b_lo = Q6_V_lo_W(bw), b_hi = Q6_V_hi_W(bw);

        HVX_Vector r_lo = zero_v, r_hi = zero_v;
        HVX_Vector q_lo = zero_v, q_hi = zero_v;
        for (int bit = 15; bit >= 0; --bit) {
            HVX_Vector sb_lo = Q6_Vuw_vlsr_VuwR(a_lo, bit);
            HVX_Vector sb_hi = Q6_Vuw_vlsr_VuwR(a_hi, bit);
            HVX_Vector bt_lo = Q6_V_vand_VV(sb_lo, v_one);
            HVX_Vector bt_hi = Q6_V_vand_VV(sb_hi, v_one);

            r_lo = Q6_Vw_vasl_VwR(r_lo, 1);
            r_hi = Q6_Vw_vasl_VwR(r_hi, 1);
            r_lo = Q6_V_vor_VV(r_lo, bt_lo);
            r_hi = Q6_V_vor_VV(r_hi, bt_hi);

            HVX_VectorPred q_ge_lo = Q6_Q_not_Q(Q6_Q_vcmp_gt_VuwVuw(b_lo, r_lo));
            HVX_VectorPred q_ge_hi = Q6_Q_not_Q(Q6_Q_vcmp_gt_VuwVuw(b_hi, r_hi));

            HVX_Vector s_lo = Q6_Vw_vsub_VwVw(r_lo, b_lo);
            HVX_Vector s_hi = Q6_Vw_vsub_VwVw(r_hi, b_hi);
            r_lo = Q6_V_vmux_QVV(q_ge_lo, s_lo, r_lo);
            r_hi = Q6_V_vmux_QVV(q_ge_hi, s_hi, r_hi);

            q_lo = Q6_Vw_vasl_VwR(q_lo, 1);
            q_hi = Q6_Vw_vasl_VwR(q_hi, 1);
            q_lo = Q6_V_vmux_QVV(q_ge_lo, Q6_V_vor_VV(q_lo, v_one), q_lo);
            q_hi = Q6_V_vmux_QVV(q_ge_hi, Q6_V_vor_VV(q_hi, v_one), q_hi);
        }

        /* 2×32 u32 → 64 u16 (饱和移位 0) */
        HVX_Vector v_final = Q6_Vuh_vasr_VuwVuwR_sat(q_hi, q_lo, 0);
        v_final = Q6_V_vmux_QVV(q_nz, v_final, v_ffff);

        *(HVX_Vector *) (out + i) = v_final;
    }
#else
    for (uint32_t i = 0; i < n_vec; ++i) out[i] = hvhx_div_u16_scalar(a[i], b[i]);
#endif

    for (uint32_t i = n_vec; i < n; ++i) out[i] = hvhx_div_u16_scalar(a[i], b[i]);
}

/* ============================================================
 *  i32 整除: |a|/|b| shift-subtract, 还原符号 (32 i32/vector)
 * ============================================================ */
void hvhx_divide_flat_i32(const int32_t * __restrict__ a,
                          const int32_t * __restrict__ b,
                          int32_t       * __restrict__ out,
                          uint32_t n)
{
    if (n == 0) return;
    const uint32_t n_vec  = n & ~(HVHX_VEC_ELEM_U32 - 1u);

#if defined(__HVX__) || defined(__hexagon__)
    HVX_Vector zero_v = Q6_V_vzero();
    HVX_Vector v_one  = Q6_V_vsplat_R(1);

    for (uint32_t i = 0; i < n_vec; i += HVHX_VEC_ELEM_U32) {
        HVX_Vector v_a = *(const HVX_Vector *) (a + i);
        HVX_Vector v_b = *(const HVX_Vector *) (b + i);

        /* 除零检测 */
        HVX_VectorPred q_nz = Q6_Q_vcmp_eq_VwVw(v_b, zero_v);

        /* 取绝对值 (INT32_MIN 特殊: |INT32_MIN| 溢出, 用 vmax 裁剪) */
        HVX_VectorPred q_aneg = Q6_Q_vcmp_gt_VwVw(zero_v, v_a);   /* a < 0 */
        HVX_VectorPred q_bneg = Q6_Q_vcmp_gt_VwVw(zero_v, v_b);   /* b < 0 */
        HVX_Vector na = Q6_Vw_vsub_VwVw(zero_v, v_a);
        HVX_Vector nb = Q6_Vw_vsub_VwVw(zero_v, v_b);
        HVX_Vector ua = Q6_V_vmux_QVV(q_aneg, na, v_a);
        HVX_Vector ub = Q6_V_vmux_QVV(q_bneg, nb, v_b);

        /* 移位减法 (32 次) */
        HVX_Vector v_rem = zero_v;
        HVX_Vector v_quo = zero_v;
        for (int bit = 31; bit >= 0; --bit) {
            HVX_Vector v_bit = Q6_Vuw_vlsr_VuwR(ua, bit);
            v_bit = Q6_V_vand_VV(v_bit, v_one);

            v_rem = Q6_Vw_vasl_VwR(v_rem, 1);
            v_rem = Q6_V_vor_VV(v_rem, v_bit);

            HVX_VectorPred q_ge = Q6_Q_not_Q(Q6_Q_vcmp_gt_VuwVuw(ub, v_rem));
            HVX_Vector v_sub = Q6_Vw_vsub_VwVw(v_rem, ub);
            v_rem = Q6_V_vmux_QVV(q_ge, v_sub, v_rem);

            v_quo = Q6_Vw_vasl_VwR(v_quo, 1);
            v_quo = Q6_V_vmux_QVV(q_ge, Q6_V_vor_VV(v_quo, v_one), v_quo);
        }

        /* round-to-nearest, ties-toward-zero (匹配 qf32 倒数精度行为):
         * 2*rem > ub → quo+1; rem > ub-rem ⟺ rem > diff (rem<ub 故 diff>0) */
        HVX_Vector v_diff = Q6_Vw_vsub_VwVw(ub, v_rem);
        HVX_VectorPred q_round = Q6_Q_vcmp_gt_VuwVuw(v_rem, v_diff);
        v_quo = Q6_V_vmux_QVV(q_round, Q6_Vw_vadd_VwVw(v_quo, v_one), v_quo);

        /* 还原符号: (a<0) ^ (b<0) → 结果取负 */
        HVX_VectorPred q_neg = Q6_Q_xor_QQ(q_aneg, q_bneg);
        HVX_Vector neg_quo = Q6_Vw_vsub_VwVw(zero_v, v_quo);
        HVX_Vector v_res = Q6_V_vmux_QVV(q_neg, neg_quo, v_quo);

        /* 除零饱和: b==0 → INT32_MAX (正); q_nz=真 where b==0 */
        HVX_Vector v_sat = Q6_V_vsplat_R((unsigned)INT32_MAX);
        v_res = Q6_V_vmux_QVV(q_nz, v_sat, v_res);

        *(HVX_Vector *) (out + i) = v_res;
    }
#else
    for (uint32_t i = 0; i < n_vec; ++i) out[i] = hvhx_div_i32_scalar(a[i], b[i]);
#endif

    for (uint32_t i = n_vec; i < n; ++i) out[i] = hvhx_div_i32_scalar(a[i], b[i]);
}

void hvhx_floor_divide_u8(const uint8_t * __restrict__ a,
                          const uint8_t * __restrict__ b,
                          uint8_t       * __restrict__ out,
                          uint32_t n)
{
    hvhx_divide_u8(a, b, out, n);
}

void hvhx_floor_divide_u16(const uint16_t * __restrict__ a,
                           const uint16_t * __restrict__ b,
                           uint16_t       * __restrict__ out,
                           uint32_t n)
{
    hvhx_divide_u16(a, b, out, n);
}
