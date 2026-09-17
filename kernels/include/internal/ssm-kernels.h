#ifndef SSM_KERNELS_H
#define SSM_KERNELS_H
/*
 * ssm-kernels.h — 从 ggmlHTPV3E htp/ 原样提取的 SSM 内核 (HVX 向量化).
 * 源文件:
 *   - solve-tri-ops.c       : solve_tri_row_{scalar,hvx} + hvx_load_partial_f32
 *   - gated-delta-net-ops.c : gdn_{mul,add_scaled,mul_scalar}_dot{4,8}_f32
 *
 * 仅在 DSP (HVX) 路径包含. 依赖 hvx-utils.h (全 hvx-* 数学头).
 */
#include "hvx-utils.h"

/* ============================================================
 *  下三角前代解 (逐行) — 源 solve-tri-ops.c
 * ============================================================ */
static inline void solve_tri_row_scalar(const float * A_row,
                                        const float * B_row,
                                        float *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        float         inv_diag) {
    for (uint32_t col = col0; col < col0 + coln; ++col) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

static inline HVX_Vector hvx_load_partial_f32(const float * src, uint32_t n) {
    HVX_Vector v = *((const HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * sizeof(float));
    return Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

static inline void solve_tri_row_hvx(const float * A_row,
                                     const float * B_row,
                                     float *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     float         inv_diag) {
    const bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    for (uint32_t t = 0; t < row; ++t) {
        const float   a         = A_row[t];
        const float * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((const HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    const float * b_row_col = B_row + col0;
    float *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((const HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((void *) x_out_col, coln * sizeof(float), out_v);
}

/* ============================================================
 *  Gated Delta Net — 源 gated-delta-net-ops.c (L117-569, 原样)
 * ============================================================ */
static inline void gdn_mul_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, const float * restrict mul,
        const float * restrict dot, uint32_t n, float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm   = hvx_vmem(mul + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

static inline void gdn_add_scaled_dot4_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, const float * restrict src,
        const float * restrict scale, const float * restrict dot, uint32_t n,
        float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs = hvx_vmem(src + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
    }

    HVX_Vector_x4 acc = { .v = { acc0, acc1, acc2, acc3 } };
    hvx_vec_store_u(sums, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(acc));
}

static inline void gdn_mul_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const float * restrict mul, const float * restrict dot, uint32_t n,
        float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vm = hvx_vmem(mul + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + i * epv), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + i * epv), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + i * epv), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + i * epv), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + i * epv), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + i * epv), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + i * epv), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + i * epv), vm);

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vm = hvx_vmem(mul + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_mul_f32_f32(hvx_vmemu(dst0 + off), vm);
        HVX_Vector out1 = hvx_vec_mul_f32_f32(hvx_vmemu(dst1 + off), vm);
        HVX_Vector out2 = hvx_vec_mul_f32_f32(hvx_vmemu(dst2 + off), vm);
        HVX_Vector out3 = hvx_vec_mul_f32_f32(hvx_vmemu(dst3 + off), vm);
        HVX_Vector out4 = hvx_vec_mul_f32_f32(hvx_vmemu(dst4 + off), vm);
        HVX_Vector out5 = hvx_vec_mul_f32_f32(hvx_vmemu(dst5 + off), vm);
        HVX_Vector out6 = hvx_vec_mul_f32_f32(hvx_vmemu(dst6 + off), vm);
        HVX_Vector out7 = hvx_vec_mul_f32_f32(hvx_vmemu(dst7 + off), vm);

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    hvx_vec_store_u(sums + 0, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accA));
    hvx_vec_store_u(sums + 4, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accB));
}

static inline void gdn_add_scaled_dot8_f32(float * restrict dst0, float * restrict dst1,
        float * restrict dst2, float * restrict dst3, float * restrict dst4,
        float * restrict dst5, float * restrict dst6, float * restrict dst7,
        const float * restrict src, const float * restrict scale,
        const float * restrict dot, uint32_t n, float * restrict sums) {
    HVX_Vector acc0 = Q6_V_vzero();
    HVX_Vector acc1 = Q6_V_vzero();
    HVX_Vector acc2 = Q6_V_vzero();
    HVX_Vector acc3 = Q6_V_vzero();
    HVX_Vector acc4 = Q6_V_vzero();
    HVX_Vector acc5 = Q6_V_vzero();
    HVX_Vector acc6 = Q6_V_vzero();
    HVX_Vector acc7 = Q6_V_vzero();
    const HVX_Vector scale0 = hvx_vec_splat_f32(scale[0]);
    const HVX_Vector scale1 = hvx_vec_splat_f32(scale[1]);
    const HVX_Vector scale2 = hvx_vec_splat_f32(scale[2]);
    const HVX_Vector scale3 = hvx_vec_splat_f32(scale[3]);
    const HVX_Vector scale4 = hvx_vec_splat_f32(scale[4]);
    const HVX_Vector scale5 = hvx_vec_splat_f32(scale[5]);
    const HVX_Vector scale6 = hvx_vec_splat_f32(scale[6]);
    const HVX_Vector scale7 = hvx_vec_splat_f32(scale[7]);

    const uint32_t epv = 128 / sizeof(float);
    const uint32_t nvec = n / epv;
    const uint32_t nloe = n % epv;
    for (uint32_t i = 0; i < nvec; ++i) {
        HVX_Vector vs = hvx_vmem(src + i * epv);
        HVX_Vector vdot = hvx_vmem(dot + i * epv);

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + i * epv), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + i * epv), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + i * epv), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + i * epv), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + i * epv), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + i * epv), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + i * epv), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + i * epv), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vmemu(dst0 + i * epv) = out0;
        hvx_vmemu(dst1 + i * epv) = out1;
        hvx_vmemu(dst2 + i * epv) = out2;
        hvx_vmemu(dst3 + i * epv) = out3;
        hvx_vmemu(dst4 + i * epv) = out4;
        hvx_vmemu(dst5 + i * epv) = out5;
        hvx_vmemu(dst6 + i * epv) = out6;
        hvx_vmemu(dst7 + i * epv) = out7;

        acc0 = hvx_vec_add_f32_f32(acc0, hvx_vec_mul_f32_f32(out0, vdot));
        acc1 = hvx_vec_add_f32_f32(acc1, hvx_vec_mul_f32_f32(out1, vdot));
        acc2 = hvx_vec_add_f32_f32(acc2, hvx_vec_mul_f32_f32(out2, vdot));
        acc3 = hvx_vec_add_f32_f32(acc3, hvx_vec_mul_f32_f32(out3, vdot));
        acc4 = hvx_vec_add_f32_f32(acc4, hvx_vec_mul_f32_f32(out4, vdot));
        acc5 = hvx_vec_add_f32_f32(acc5, hvx_vec_mul_f32_f32(out5, vdot));
        acc6 = hvx_vec_add_f32_f32(acc6, hvx_vec_mul_f32_f32(out6, vdot));
        acc7 = hvx_vec_add_f32_f32(acc7, hvx_vec_mul_f32_f32(out7, vdot));
    }

    if (nloe) {
        const uint32_t off = nvec * epv;
        HVX_Vector vs = hvx_vmem(src + off);
        HVX_Vector vdot = hvx_vmem(dot + off);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(nloe * sizeof(float));
        HVX_Vector zero = Q6_V_vzero();

        HVX_Vector out0 = hvx_vec_add_f32_f32(hvx_vmemu(dst0 + off), hvx_vec_mul_f32_f32(vs, scale0));
        HVX_Vector out1 = hvx_vec_add_f32_f32(hvx_vmemu(dst1 + off), hvx_vec_mul_f32_f32(vs, scale1));
        HVX_Vector out2 = hvx_vec_add_f32_f32(hvx_vmemu(dst2 + off), hvx_vec_mul_f32_f32(vs, scale2));
        HVX_Vector out3 = hvx_vec_add_f32_f32(hvx_vmemu(dst3 + off), hvx_vec_mul_f32_f32(vs, scale3));
        HVX_Vector out4 = hvx_vec_add_f32_f32(hvx_vmemu(dst4 + off), hvx_vec_mul_f32_f32(vs, scale4));
        HVX_Vector out5 = hvx_vec_add_f32_f32(hvx_vmemu(dst5 + off), hvx_vec_mul_f32_f32(vs, scale5));
        HVX_Vector out6 = hvx_vec_add_f32_f32(hvx_vmemu(dst6 + off), hvx_vec_mul_f32_f32(vs, scale6));
        HVX_Vector out7 = hvx_vec_add_f32_f32(hvx_vmemu(dst7 + off), hvx_vec_mul_f32_f32(vs, scale7));

        hvx_vec_store_u(dst0 + off, nloe * sizeof(float), out0);
        hvx_vec_store_u(dst1 + off, nloe * sizeof(float), out1);
        hvx_vec_store_u(dst2 + off, nloe * sizeof(float), out2);
        hvx_vec_store_u(dst3 + off, nloe * sizeof(float), out3);
        hvx_vec_store_u(dst4 + off, nloe * sizeof(float), out4);
        hvx_vec_store_u(dst5 + off, nloe * sizeof(float), out5);
        hvx_vec_store_u(dst6 + off, nloe * sizeof(float), out6);
        hvx_vec_store_u(dst7 + off, nloe * sizeof(float), out7);

        acc0 = hvx_vec_add_f32_f32(acc0, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out0, vdot), zero));
        acc1 = hvx_vec_add_f32_f32(acc1, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out1, vdot), zero));
        acc2 = hvx_vec_add_f32_f32(acc2, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out2, vdot), zero));
        acc3 = hvx_vec_add_f32_f32(acc3, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out3, vdot), zero));
        acc4 = hvx_vec_add_f32_f32(acc4, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out4, vdot), zero));
        acc5 = hvx_vec_add_f32_f32(acc5, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out5, vdot), zero));
        acc6 = hvx_vec_add_f32_f32(acc6, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out6, vdot), zero));
        acc7 = hvx_vec_add_f32_f32(acc7, Q6_V_vmux_QVV(mask, hvx_vec_mul_f32_f32(out7, vdot), zero));
    }

    HVX_Vector_x4 accA = { .v = { acc0, acc1, acc2, acc3 } };
    HVX_Vector_x4 accB = { .v = { acc4, acc5, acc6, acc7 } };
    hvx_vec_store_u(sums + 0, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accA));
    hvx_vec_store_u(sums + 4, 4 * sizeof(float), hvx_vec_reduce_sum_f32x4(accB));
}

/* ============================================================
 *  SSM depthwise 1D causal conv — 源 ssm-conv.c
 *
 *  数学: dst[i1, i2, i3] = Σ_{j=0}^{d_conv-1} src0[i2+j, i1, i3] * src1[j, i1]
 *  (Mamba/Jamba 状态空间模型前导卷积, 每个 channel i1 一个 d_conv 抽头滤波器)
 * ============================================================ */

/* In-register 32×32 fp32 transpose (5-stage HVX vshuff butterfly).
 * 源: ssm-conv.c:140-186. m[i] = 行 i 的 32 个 f32 (一个 HVX_Vector).
 * 转置后 m[i] = 列 i 的 32 个 f32. 纯布局工具, 通用. */
static inline void hvx_transpose_32x32_f32(HVX_Vector m[32]) {
    HVX_Vector tmp[32];

    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[2*i + 1], m[2*i], -4);
        tmp[2*i + 0] = Q6_V_lo_W(p);
        tmp[2*i + 1] = Q6_V_hi_W(p);
    }
    for (int b = 0; b < 32; b += 4) {
        HVX_VectorPair p0 = Q6_W_vshuff_VVR(tmp[b + 2], tmp[b + 0], -8);
        HVX_VectorPair p1 = Q6_W_vshuff_VVR(tmp[b + 3], tmp[b + 1], -8);
        m[b + 0] = Q6_V_lo_W(p0); m[b + 1] = Q6_V_hi_W(p0);
        m[b + 2] = Q6_V_lo_W(p1); m[b + 3] = Q6_V_hi_W(p1);
    }
    for (int b = 0; b < 32; b += 8) {
        for (int i = 0; i < 4; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(m[b + i + 4], m[b + i], -16);
            tmp[b + 2*i + 0] = Q6_V_lo_W(p);
            tmp[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }
    for (int b = 0; b < 32; b += 16) {
        for (int i = 0; i < 8; ++i) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(tmp[b + i + 8], tmp[b + i], -32);
            m[b + 2*i + 0] = Q6_V_lo_W(p);
            m[b + 2*i + 1] = Q6_V_hi_W(p);
        }
    }
    for (int i = 0; i < 16; ++i) {
        HVX_VectorPair p = Q6_W_vshuff_VVR(m[i + 16], m[i], -64);
        tmp[2 * i + 0]   = Q6_V_lo_W(p);
        tmp[2 * i + 1]   = Q6_V_hi_W(p);
    }
    for (int i = 0; i < 32; ++i) {
        m[i] = tmp[i];
    }
}

/* HVX 32-channel depthwise conv dot — ssm-conv.c:318-345 提取的内核体.
 * 前置条件: src0/src1 已被调用方转置为 channel-contiguous 布局 (见 hvx_transpose_32x32_f32),
 *   使每个 HVX_Vector = 32 个连续 channel 在给定 (time, conv_tap) 处的值.
 *
 * @param res       输出 1 个 HVX_Vector (32 个 channel 的 conv 结果)
 * @param x_base    转置后的 src0 基址: x_base[j*x_stride] = 时间偏移 (t+j) 处的 32 channel 向量
 * @param x_stride  连续时间偏移间距 (float 单位, 典型 = d_inner_tile)
 * @param w_base    转置后的 src1 基址: w_base[j*w_stride] = 第 j 个 conv 抽头的 32 channel 权重
 * @param w_stride  连续抽头间距 (float 单位, 典型 = d_inner_stride)
 * @param d_conv    卷积核长度 (抽头数)
 * @param apply_silu 1 = 写 silu(conv_out) 代替 conv_out (Mamba fused Stage C.3) */
static inline void ssm_conv_dot32_hvx(HVX_Vector * __restrict__ res,
                                      const float  * __restrict__ x_base, uint32_t x_stride,
                                      const float  * __restrict__ w_base, uint32_t w_stride,
                                      uint32_t d_conv, int apply_silu) {
    HVX_Vector acc = hvx_vec_splat_f32(0.0f);
    for (uint32_t j = 0; j < d_conv; ++j) {
        HVX_Vector x = *(const HVX_Vector *)(x_base + j * x_stride);
        HVX_Vector w = *(const HVX_Vector *)(w_base + j * w_stride);
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
    }
    HVX_Vector r = Q6_Vsf_equals_Vqf32(acc);
    if (apply_silu) {
        /* sig/r 必须 128B 对齐栈槽: hvx_sigmoid_f32_aa 用 aligned vmem,
         * HVX_UVector 类型只保证 1B 对齐 → 曾在此 CX_FAULT 杀 PD. */
        HVX_Vector sig;
        hvx_sigmoid_f32_aa((uint8_t *)&sig, (const uint8_t *)&r, 32);
        r = Q6_Vsf_vmpy_VsfVsf(r, sig);
    }
    *res = r;
}

#endif /* SSM_KERNELS_H */
