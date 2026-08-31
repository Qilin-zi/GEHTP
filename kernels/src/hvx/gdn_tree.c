/* gdn_tree.c — U14 设备串行树形递归 kernel (见 include/gdn_tree.h) */
#include "gdn_tree.h"
#include "gdn_sm.h"
#include <math.h>
#include <stdlib.h>

int gdn_tree_serial_f16(const float* S0, float* states, int d, int t,
                        const int* parent, const int16_t* k, const int16_t* v,
                        const int16_t* q, const int16_t* beta, const int16_t* g16,
                        int16_t* y) {
    if (t < 1 || d < 1 || !S0 || !states || !parent || !k || !v || !q ||
        !beta || !g16 || !y)
        return -1;
    if (parent[0] != -1) return -1;
    for (int i = 1; i < t; i++)
        if (parent[i] < -1 || parent[i] >= i) return -1;

    float* kf = (float*)malloc((size_t)d * sizeof(float));
    float* vf = (float*)malloc((size_t)d * sizeof(float));
    float* qf = (float*)malloc((size_t)d * sizeof(float));
    float* yk = (float*)malloc((size_t)d * sizeof(float));
    if (!kf || !vf || !qf || !yk) { free(kf); free(vf); free(qf); free(yk); return -1; }
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < t; i++) {
        for (int n = 0; n < d; n++) {
            kf[n] = gdn_f16_to_f32(k[(size_t)i * d + n]);
            vf[n] = gdn_f16_to_f32(v[(size_t)i * d + n]);
            qf[n] = gdn_f16_to_f32(q[(size_t)i * d + n]);
        }
        float a  = expf(gdn_f16_to_f32(g16[i]));
        float bi = gdn_f16_to_f32(beta[i]);
        const float* sp = (parent[i] < 0) ? S0 : states + (size_t)parent[i] * d * d;
        float* si = states + (size_t)i * d * d;

        for (int j = 0; j < d; j++) {                 /* S_i = e^{g} S_par; yk = S_i k */
            const float* sr = sp + (size_t)j * d;
            float* dr = si + (size_t)j * d;
            float s = 0.0f;
            for (int n = 0; n < d; n++) { dr[n] = sr[n] * a; s += dr[n] * kf[n]; }
            yk[j] = s;
        }
        for (int j = 0; j < d; j++) {                 /* S_i += δ⊗k; y = (S_i q)/√d */
            float dj = bi * (vf[j] - yk[j]);
            float* dr = si + (size_t)j * d;
            float s = 0.0f;
            for (int n = 0; n < d; n++) { dr[n] += dj * kf[n]; s += dr[n] * qf[n]; }
            y[(size_t)i * d + j] = gdn_f32_to_f16(s * scale);
        }
    }
    free(kf); free(vf); free(qf); free(yk);
    return 0;
}
