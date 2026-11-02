/* check_w4stream_host.c — 018 G2 host 门: w4_stream host 标量镜像
 * vs q4_0_gemv 标量基线 vs f32 金标, 三向 cos + max_rel。
 *
 * 与 pack_w4stream_ab.cc 同确定性生成器 (激活/tile 权重逐位同)。
 * 编译: gcc -O2 -std=gnu11 -I../include check_w4stream_host.c -o /tmp/check_w4s -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t sh = (uint32_t)(14 - e);
        uint32_t half = m >> (sh + 1);
        uint32_t rest = m & ((1u << sh) - 1u);
        if ((m >> sh) & 1u) half += (rest || (half & 1u)) ? 1u : 0u;
        return (uint16_t)(sign | half);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
    if ((m >> 12) & 1u) h += (((m & 0xfffu) || (h & 1u)) ? 1u : 0u);
    return h;
}
static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (!e && !m) bits = s;
    else if (e == 0x1Fu) bits = s | 0x7F800000u | (m << 13);
    else if (!e) { int8_t sh = 1; while (!(m & 0x400u)) { m <<= 1; sh++; }
        bits = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | ((m & 0x3FFu) << 13); }
    else bits = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* 与 pack_w4stream_ab.cc 逐位同 */
static void gen_act(uint16_t* a, uint32_t K) {
    for (uint32_t k = 0; k < K; k++)
        a[k] = f32_to_f16((float)((int)(k % 29u) - 14) * 0.25f);
}
static void gen_w_q4(uint8_t* w, uint32_t K, uint32_t N) {
    uint32_t n_dot = K / 32u, n_ct = N / 32u;
    for (uint32_t ct = 0; ct < n_ct; ct++)
        for (uint32_t kt = 0; kt < n_dot; kt++) {
            uint8_t* tile = w + ((size_t)ct * n_dot + kt) * 640u;
            for (uint32_t i = 0; i < 16u; i++)
                for (uint32_t r = 0; r < 32u; r++)
                    tile[i * 32u + r] = (uint8_t)(ct * 31u + kt * 17u + i * 7u + r * 11u + i * r);
            for (uint32_t r = 0; r < 32u; r++) {
                uint16_t d = f32_to_f16(0.02f + (float)((ct + r) % 7u) * 0.005f);
                tile[512u + 2u * r] = (uint8_t)(d & 0xffu);
                tile[512u + 2u * r + 1u] = (uint8_t)(d >> 8);
            }
            for (uint32_t p = 512u + 64u; p < 640u; p++) tile[p] = 0;
        }
}

static double cos_sim(const float* a, const float* b, size_t n) {
    double p = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        p += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
    }
    return p / (sqrt(na) * sqrt(nb) + 1e-30);
}
static double max_rel(const float* a, const float* b, size_t n) {
    double mr = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - b[i]);
        double r = d / (fabs((double)b[i]) + 1e-6);
        if (r > mr) mr = r;
    }
    return mr;
}

/* f32 金标: y[n] = Σ_kt d_w[kt][n%32] × Σ_j (v-8)×af[kt*32+j] (f16 激活精确) */
static void gold_f32(const uint16_t* a, const uint8_t* w, float* y, uint32_t K, uint32_t N) {
    uint32_t n_dot = K / 32u;
    for (uint32_t n = 0; n < N; n++) {
        uint32_t ct = n / 32u, r = n % 32u;
        float acc = 0.f;
        for (uint32_t kt = 0; kt < n_dot; kt++) {
            const uint8_t* tile = w + ((size_t)ct * n_dot + kt) * 640u;
            const uint16_t* drow = (const uint16_t*)(const void*)(tile + 512u);
            float d = f16_to_f32(drow[r]);
            for (uint32_t i = 0; i < 16u; i++) {
                uint8_t b = tile[i * 32u + r];
                uint32_t k0 = kt * 32u + 2u * i, k1 = k0 + 1u;
                acc += f16_to_f32(a[k0]) * d * (float)((int)(b & 0xFu) - 8);
                acc += f16_to_f32(a[k1]) * d * (float)((int)(b >> 4) - 8);
            }
        }
        y[n] = acc;
    }
}

/* 基线镜像 (q4_0_gemv 标量): f16 激活, per-kt d_w 乘, f32 累加 — 同 gold
 * 公式但逐元素直接乘 (与设备基线数值同构) */
static void base_q4gemv(const uint16_t* a, const uint8_t* w, float* y, uint32_t K, uint32_t N) {
    gold_f32(a, w, y, K, N);  /* 公式同 gold (基线与金标同算法) */
}

/* w4_stream host 镜像 (oplist_exec.c #else 分支同式): q8 激活 + int 累加 */
static void w4stream_host(const uint16_t* a, const uint8_t* w, float* y, uint32_t K, uint32_t N) {
    uint32_t n_dot = K / 32u;
    static float af[8192], d_a[256];
    static int8_t aq8[8192];
    for (uint32_t k = 0; k < K; k++) af[k] = f16_to_f32(a[k]);
    for (uint32_t kt = 0; kt < n_dot; kt++) {
        float mx = 0.f;
        for (uint32_t j = 0; j < 32u; j++) {
            float v = fabsf(af[kt * 32u + j]);
            if (v > mx) mx = v;
        }
        float d = (mx > 0.f) ? mx / 127.f : 1.f;
        d_a[kt] = f16_to_f32(f32_to_f16(d));
        float inv = (mx > 0.f) ? 1.f / d : 0.f;
        for (uint32_t j = 0; j < 32u; j++) {
            int q = (int)lrintf(af[kt * 32u + j] * inv);
            aq8[kt * 32u + j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        }
    }
    for (uint32_t n = 0; n < N; n++) {
        uint32_t ct = n / 32u, r = n % 32u;
        float acc = 0.f;
        for (uint32_t kt = 0; kt < n_dot; kt++) {
            const uint8_t* tile = w + ((size_t)ct * n_dot + kt) * 640u;
            const uint16_t* drow = (const uint16_t*)(const void*)(tile + 512u);
            float d_w = f16_to_f32(drow[r]);
            int isum = 0;
            for (uint32_t i = 0; i < 16u; i++) {
                uint8_t b = tile[i * 32u + r];
                isum += ((int)(b & 0xFu) - 8) * (int)aq8[kt * 32u + 2u * i];
                isum += ((int)(b >> 4) - 8) * (int)aq8[kt * 32u + 2u * i + 1u];
            }
            acc += d_w * d_a[kt] * (float)isum;
        }
        y[n] = acc;
    }
}

int main(void) {
    const uint32_t shapes[2][2] = {{1024u, 2048u}, {1024u, 16384u}};
    int nfail = 0;
    for (int s = 0; s < 2; s++) {
        uint32_t K = shapes[s][0], N = shapes[s][1];
        uint16_t* a = malloc(K * 2);
        uint8_t* w = malloc((size_t)(K / 32) * (N / 32) * 640);
        float* yg = malloc(N * 4);
        float* yb = malloc(N * 4);
        float* ys = malloc(N * 4);
        gen_act(a, K);
        gen_w_q4(w, K, N);
        gold_f32(a, w, yg, K, N);
        base_q4gemv(a, w, yb, K, N);
        w4stream_host(a, w, ys, K, N);
        double c_bs = cos_sim(yb, ys, N), c_sg = cos_sim(ys, yg, N);
        double mr_sg = max_rel(ys, yg, N);
        printf("[K=%u N=%u] cos(base,stream)=%.6f  cos(stream,gold)=%.6f  max_rel(stream,gold)=%.4f\n",
               K, N, c_bs, c_sg, mr_sg);
        int ok = (c_bs >= 0.9999) && (c_sg >= 0.9999);
        printf("  %s G2 host 门 (cos ≥ 0.9999)\n", ok ? "PASS" : "FAIL");
        if (!ok) nfail++;
        free(a); free(w); free(yg); free(yb); free(ys);
    }
    printf("%s\n", nfail ? "FAIL" : "ALL PASS");
    return nfail ? 1 : 0;
}
