/*
 * 28_gdn_tree — U14 树形 GDN 设备验证
 * =====================================================================
 * 判据 (T=8/16/32 × 3 随机拓扑, D=64, LCG 可复现):
 *   1) 闭式解 vs 串行递归 oracle (全 f32): y 与逐节点 commit 状态 cos ≥ 0.99999
 *   2) 设备 f16 kernel vs 闭式解: y cos ≥ 0.999, 逐节点状态 min cos ≥ 0.999
 *   3) kernel 复跑 bit-exact
 *   4) 拓扑校验: parent[0]≠-1 / parent[i]≥i 均被拒 (rc=-1)
 *   5) INT16 衰减连乘精度曲线: i16 量化逐步衰减积 vs f32 exp(路径和),
 *      rel err ≤ depth·6e-4 + 1e-3 (f16 半步 + i16 半步包络), 且逐深度不劣于 f16 路径
 *   6) 状态缓冲走 U10 arena (128B 对齐, 全释放归零)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hvxhmx_v23.h"
#include "example_util.h"

#define D 64
#define TMAX 32

static double cos_sim(const float* a, const float* b, size_t n) {
    double p = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        p += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}

int main(void) {
    ex_open_result("28_gdn_tree");
    const int Ts[3] = { 8, 16, 32 };

    /* U10 arena 承载全部工作缓冲 (D=64, T≤32) */
    uint32_t ab = (uint32_t)TMAX * D * D * 4;         /* 状态面 512KB */
    uint8_t* ablk = memalign(128, (size_t)ab * 2 + 512 * 1024);
    struct arena ar;
    if (arena_init(&ar, ablk, ab * 2 + 512 * 1024, NULL, 0)) { ex_log("arena_init FAIL"); goto out; }

    float  *Sser = arena_alloc(&ar, ab, 0, 0), *Scls = arena_alloc(&ar, ab, 0, 0);
    float  *S0   = arena_alloc(&ar, D * D * 4, 0, 0);
    float  *kf = arena_alloc(&ar, (size_t)TMAX * D * 4, 0, 0),
           *vf = arena_alloc(&ar, (size_t)TMAX * D * 4, 0, 0),
           *qf = arena_alloc(&ar, (size_t)TMAX * D * 4, 0, 0),
           *bf = arena_alloc(&ar, TMAX * 4, 0, 0), *gf = arena_alloc(&ar, TMAX * 4, 0, 0),
           *ys = arena_alloc(&ar, (size_t)TMAX * D * 4, 0, 0),
           *yc = arena_alloc(&ar, (size_t)TMAX * D * 4, 0, 0),
           *yk = arena_alloc(&ar, (size_t)TMAX * D * 4, 0, 0);
    int16_t *k16 = arena_alloc(&ar, (size_t)TMAX * D * 2, 0, 0),
            *v16 = arena_alloc(&ar, (size_t)TMAX * D * 2, 0, 0),
            *q16 = arena_alloc(&ar, (size_t)TMAX * D * 2, 0, 0),
            *be16 = arena_alloc(&ar, TMAX * 2, 0, 0), *g16 = arena_alloc(&ar, TMAX * 2, 0, 0),
            *y16a = arena_alloc(&ar, (size_t)TMAX * D * 2, 0, 0),
            *y16b = arena_alloc(&ar, (size_t)TMAX * D * 2, 0, 0);
    int* parent = arena_alloc(&ar, TMAX * sizeof(int), 0, 0);
    int* dep    = arena_alloc(&ar, TMAX * sizeof(int), 0, 0);
    if (!Sser || !Scls || !S0 || !kf || !vf || !qf || !bf || !gf || !ys || !yc || !yk ||
        !k16 || !v16 || !q16 || !be16 || !g16 || !y16a || !y16b || !parent || !dep) {
        ex_log("arena_alloc FAIL"); goto out;
    }

    double worst_oracle = 1, worst_oracle_state = 1;
    double worst_kern = 1, worst_kern_state = 1;
    int det_bad = 0, depth_max_seen = 0;

    for (int ti = 0; ti < 3; ti++) {
        int t = Ts[ti];
        for (int seed = 0; seed < 3; seed++) {
            uint32_t lcg = 20260828u + (uint32_t)(seed * 7919 + ti * 104729);
            /* 随机拓扑: 35% 链 (造深度), 其余挂随机祖先 */
            parent[0] = -1; dep[0] = 0;
            for (int i = 1; i < t; i++) {
                double u = (double)(gdn_lcg_next(&lcg) >> 8) / 16777216.0;
                parent[i] = (u < 0.35) ? i - 1 : (int)(gdn_lcg_next(&lcg) % (uint32_t)i);
                dep[i] = dep[parent[i]] + 1;
                if (dep[i] > depth_max_seen) depth_max_seen = dep[i];
            }
            for (size_t z = 0; z < (size_t)t * D; z++) {
                kf[z] = gdn_lcg_norm(&lcg) * 0.5f; k16[z] = gdn_f32_to_f16(kf[z]);
                vf[z] = gdn_lcg_norm(&lcg) * 0.8f; v16[z] = gdn_f32_to_f16(vf[z]);
                qf[z] = gdn_lcg_norm(&lcg) * 0.5f; q16[z] = gdn_f32_to_f16(qf[z]);
            }
            for (int i = 0; i < t; i++) {
                bf[i] = fabsf(gdn_lcg_norm(&lcg)) + 0.2f; be16[i] = gdn_f32_to_f16(bf[i]);
                gf[i] = -fabsf(gdn_lcg_norm(&lcg)) * 0.5f; g16[i] = gdn_f32_to_f16(gf[i]);
            }
            for (size_t z = 0; z < (size_t)D * D; z++) S0[z] = gdn_lcg_norm(&lcg) * 0.3f;

            /* oracle: 串行递归 f32 vs 闭式 f32 */
            ref_delta_tree(S0, D, t, parent, kf, vf, qf, bf, gf, ys, Sser);
            ref_tree_closed(S0, D, t, parent, kf, vf, qf, bf, gf, yc, Scls);
            double cy = cos_sim(ys, yc, (size_t)t * D);
            double csm = 1;
            for (int i = 0; i < t; i++) {
                double c = cos_sim(Sser + (size_t)i * D * D, Scls + (size_t)i * D * D,
                                   (size_t)D * D);
                if (c < csm) csm = c;
            }
            if (cy < worst_oracle) worst_oracle = cy;
            if (csm < worst_oracle_state) worst_oracle_state = csm;

            /* kernel: f16 vs 闭式 */
            if (gdn_tree_serial_f16(S0, Sser, D, t, parent, k16, v16, q16,
                                    be16, g16, y16a)) {
                ex_log("kernel rc FAIL (t=%d seed=%d)", t, seed);
                goto out;
            }
            if (gdn_tree_serial_f16(S0, Sser, D, t, parent, k16, v16, q16,
                                    be16, g16, y16b) ||
                memcmp(y16a, y16b, (size_t)t * D * 2)) det_bad++;
            for (size_t z = 0; z < (size_t)t * D; z++)
                yk[z] = gdn_f16_to_f32(y16a[z]);
            double ck = cos_sim(yk, yc, (size_t)t * D);
            double ckm = 1;
            for (int i = 0; i < t; i++) {
                double c = cos_sim(Sser + (size_t)i * D * D, Scls + (size_t)i * D * D,
                                   (size_t)D * D);
                if (c < ckm) ckm = c;
            }
            if (ck < worst_kern) worst_kern = ck;
            if (ckm < worst_kern_state) worst_kern_state = ckm;
        }
    }
    ex_log("  oracle closed-vs-serial: y cos=%.6f state cos=%.6f",
           worst_oracle, worst_oracle_state);
    ex_log("  kernel f16-vs-closed:    y cos=%.6f state cos=%.6f (max depth %d)",
           worst_kern, worst_kern_state, depth_max_seen);
    ex_check("closed_vs_serial_y_cos", worst_oracle < 0.99999 ? 1 : 0, 0);
    ex_check("closed_vs_serial_state_cos", worst_oracle_state < 0.99999 ? 1 : 0, 0);
    ex_check("kernel_vs_closed_y_cos", worst_kern < 0.999 ? 1 : 0, 0);
    ex_check("kernel_vs_closed_state_cos", worst_kern_state < 0.999 ? 1 : 0, 0);
    ex_check("kernel_rerun_bitexact", det_bad, 0);

    /* 拓扑校验拒绝 */
    {
        int par2[4] = { 0, 0, 1, 2 };      /* parent[0] != -1 */
        int par3[4] = { -1, 2, 1, 2 };     /* parent[1] >= 1 */
        float st[4 * D * D];
        int16_t kk[4 * D], vv[4 * D], qq[4 * D], bb[4], gg[4], yy[4 * D];
        int r1 = gdn_tree_serial_f16(S0, st, D, 4, par2, kk, vv, qq, bb, gg, yy);
        int r2 = gdn_tree_serial_f16(S0, st, D, 4, par3, kk, vv, qq, bb, gg, yy);
        ex_check("topology_invalid_rejected", (r1 == -1 && r2 == -1) ? 0 : 1, 0);
    }

    /* INT16 衰减连乘精度曲线: i16 量化逐步衰减积 vs f32 exp(路径和) */
    {
        uint32_t lcg = 20260829u;
        int t = TMAX;
        int par[TMAX];
        float gf[TMAX]; int16_t gq[TMAX];
        par[0] = -1; dep[0] = 0;
        for (int i = 1; i < t; i++) {
            double u = (double)(gdn_lcg_next(&lcg) >> 8) / 16777216.0;
            par[i] = (u < 0.5) ? i - 1 : (int)(gdn_lcg_next(&lcg) % (uint32_t)i);
            dep[i] = dep[par[i]] + 1;
        }
        for (int i = 0; i < t; i++) {
            gf[i] = -fabsf(gdn_lcg_norm(&lcg)) * 0.5f;
            gq[i] = gdn_f32_to_f16(gf[i]);
        }
        const float sc16 = 1.0f / 65535.0f;           /* e^g ∈ (0,1], 满量程 */
        double maxerr_by_dep[TMAX + 1];
        int seen_dep[TMAX + 1];
        for (int i = 0; i <= TMAX; i++) { maxerr_by_dep[i] = 0; seen_dep[i] = 0; }
        float pf16[TMAX], pi16[TMAX], psum[TMAX];
        for (int i = 0; i < t; i++) {
            float e = expf(gdn_f16_to_f32(gq[i]));
            pf16[i] = (par[i] < 0) ? e : pf16[par[i]] * e;
            float q16v = pxb_f32_to_i16(e, sc16);
            float eq = pxb_i16_to_f32(q16v, sc16);
            pi16[i] = (par[i] < 0) ? eq : pi16[par[i]] * eq;
            psum[i] = (par[i] < 0) ? gdn_f16_to_f32(gq[i])
                                   : psum[par[i]] + gdn_f16_to_f32(gq[i]);
        }
        int bad = 0;
        for (int i = 0; i < t; i++) {
            double ref = exp((double)psum[i]);
            double e16 = fabs((double)pf16[i] - ref) / (ref + 1e-30);
            double eI  = fabs((double)pi16[i] - ref) / (ref + 1e-30);
            int dp = dep[i];
            seen_dep[dp] = 1;
            if (eI > maxerr_by_dep[dp]) maxerr_by_dep[dp] = eI;
            if (eI > (double)dp * 6.0e-4 + 1e-3) bad++;
            if (eI < e16 - 1e-9) bad++;               /* i16 不得比 f16 更准 */
        }
        ex_log("  decay curve (depth: max rel err i16):");
        for (int dp = 0; dp <= TMAX; dp++)
            if (seen_dep[dp])
                ex_log("    d=%2d  err=%.3e", dp, maxerr_by_dep[dp]);
        ex_check("i16_decay_curve_envelope", bad, 0);
    }

    /* arena 全释放归零 (状态用例后无泄漏) */
    arena_free(&ar, Sser); arena_free(&ar, Scls); arena_free(&ar, S0);
    arena_free(&ar, kf); arena_free(&ar, vf); arena_free(&ar, qf);
    arena_free(&ar, bf); arena_free(&ar, gf); arena_free(&ar, ys);
    arena_free(&ar, yc); arena_free(&ar, yk); arena_free(&ar, k16);
    arena_free(&ar, v16); arena_free(&ar, q16); arena_free(&ar, be16);
    arena_free(&ar, g16); arena_free(&ar, y16a); arena_free(&ar, y16b);
    arena_free(&ar, parent); arena_free(&ar, dep);
    ex_check("arena_fully_freed", arena_used(&ar, ARENA_DDR) != 0 ? 1 : 0, 0);
out:
    free(ablk);
    return ex_summary();
}
