/*
 * q6_intrinsics.h — Q6 / HVX intrinsic 集中声明
 * =====================================================================
 * 仅 DSP 编译时激活 (-mvhx, -mhmx), host 编译跳过.
 * 直接 include <hvx_hexagon_protos.h> 即可拿到全部真实声明;
 * 这里维护一份"我们用到的"子集 + 使用注释, 避免散落定义.
 *
 * 真实 V81 (hexagon-clang 19.0.07) 关键 intrinsic 速查:
 *   - 类型: HVX_Vector / HVX_VectorPair / HVX_VectorPred
 *   - splat: Q6_Vb_vsplat_R, Q6_Vh_vsplat_R, Q6_Vw_vsplat_R (按字段)
 *   - vzero: Q6_V_vzero()
 *   - vadd : Q6_Vb_vadd_VbVb[_sat], Q6_Vh_vadd_VhVh[_sat], Q6_Vw_vadd_VwVw[_sat]
 *   - vsub : Q6_Vb_vsub_VbVb[_sat], Q6_Vh_vsub_VhVh[_sat]
 *   - vmul : Q6_Ww_vmpy_VhRh (h × scalar → pair-word), Q6_Ww_vmpy_VhVuh (h × uh)
 *           Q6_Ww_vmpyacc_WwVhRh, Q6_Ww_vmpyacc_WwVhRh_sat
 *   - vmpa : Q6_Vh_vmpa_VhVhVhPh_sat, Q6_Vh_vmpa_VhVhVuhPuh_sat (Word64 Ph/Puh)
 *   - vasl : Q6_Vh_vasl_VhVh, Q6_Vh_vasl_VhR
 *   - vasr : Q6_Vh_vasr_VhR, Q6_Vb_vasr_VhVhR_sat, Q6_Vb_vasr_VhR_sat
 *   - vlut : Q6_Vh_vlut4_VuhPh (Word64 Ph = pair-half scalar)
 *   - vcmp : Q6_Q_vcmp_eq_VbVb, Q6_Q_vcmp_eq_VhVh, Q6_Q_vcmp_eq_VwVw
 *           Q6_Q_vcmp_gt_VbVb, Q6_Q_vcmp_gt_VhVh, Q6_Q_vcmp_gt_VwVw
 *           Q6_Q_vcmp_gt_VubVub, Q6_Q_vcmp_gt_VuhVuh, Q6_Q_vcmp_gt_VuwVuw
 *           unsigned eq 需用 vcmp_gt_vubvub 双向: gt(x,y) & gt(y,x) 等价于 eq
 *   - vmux : Q6_V_vmux_QVV
 *   - vzxt : Q6_Wuh_vzxt_Vub (u8→u16 pair zero-extend)
 *   - vsxt : Q6_Wh_vsxt_Vb (s8→s16 pair sign-extend)
 *   - pair : Q6_W_vcombine_VV, Q6_V_lo_W, Q6_V_hi_W
 *   - vnorm: Q6_Vh_vnormamt_Vh
 *   - vdeal: Q6_Vh_vdeal_Vh (16-bit deal; 8-bit 用 vshuff+vzxt 组合)
 *   - vshuff: Q6_Vb_vshuff_Vb, Q6_Vb_vshuffe_VbVb, Q6_Vb_vshuffo_VbVb
 *   - vabsdiff: Q6_Vub_vabsdiff_VubVub
 *
 * 真实 declare 在 target/hexagon/include/hvx_hexagon_protos.h
 */
#ifndef HVXHMX_Q6_INTRINSICS_H
#define HVXHMX_Q6_INTRINSICS_H

#if defined(__HVX__) || defined(__HEXAGON__)
  /* DSP 编译: 直接 include 真实头 */
  #include <hexagon_types.h>
  #include <hvx_hexagon_protos.h>
  #include <hmx_hexagon_protos.h>
#endif

#endif /* HVXHMX_Q6_INTRINSICS_H */
