/* test_kv_ssm_ops.c — 004 T016/T017 host 合成验证 (不链 DSP runtime)
 *
 * 1) KV ring (真 kvcache.c, host qurt 桩): append 8 pos → GQA 展开往返逐字节
 *    (展开体 = oplist_exec.c exec_kv_gather 内环逐字移植; 期望面独立下标重建)
 * 2) conv1d_step_f16 ×3 vs 独立 oracle (窗序 + SiLU, f16 权重同源解码)
 * 3) delta_chunk_f16: 跨 2 chunk (c=2→c=1) vs 单 chunk (c=3) 状态/输出一致性
 *    + vs ref_delta_token per-token oracle 链 (43_gdn_sm_08b 同款判据)
 *
 * 构建: gcc -O2 -std=gnu11 -I kernels/host -I kernels/include kernels/host/test_kv_ssm_ops.c \
 *         kernels/src/runtime/kvcache.c kernels/src/runtime/fence.c \
 *         kernels/src/hvx/gdn_kern.c kernels/src/hvx/gdn_ref.c -o /tmp/test_kv_ssm -lm
 */
#include <math.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kvcache.h"
#include "gdn_sm.h"

/* ---- 独立 f16 (IEEE binary16) 转换: 测试 oracle 侧自用, 勿与被测共用 ---- */
static float my_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (!exp) { bits = man ? 0 : sign; if (man) return 0.0f; }
    else bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
static uint16_t my_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) return (uint16_t)sign;  /* 测试值域无 subnormal, 粗截断够用 */
    uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
    if ((man >> 12) & 1u) h += (uint16_t)(((man & 0xfffu) || (h & 1u)) ? 1 : 0);
    return h;
}

/* LCG (与 gdn_ref 同范式但独立种子流) */
static uint32_t lcg_state = 0x00400421u;
static float frnd(float lo, float hi) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    float u = (float)(lcg_state >> 8) / 16777216.0f;
    return lo + u * (hi - lo);
}

static int n_fail = 0;
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { printf("FAIL %s:%d " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); n_fail++; } \
} while (0)

/* ===================== 1) KV append→gather 往返 ===================== */
/* exec_kv_gather 内环逐字移植 (mock: kvc 来自真 kvcache.c) */
static int gather_expand(struct kvc* c, uint16_t* kT, uint16_t* vo,
                         uint32_t n_kv, uint32_t n_heads, uint32_t kv_heads, uint32_t hd) {
    uint32_t rep = n_heads / kv_heads;
    for (uint32_t p = 0; p < n_kv; p++) {
        int slot = kvc_lookup(c, p);
        if (slot < 0) return -1;
        const uint16_t* kf = (const uint16_t*)kvc_k_face(c, (uint32_t)slot);
        const uint16_t* vf = (const uint16_t*)kvc_v_face(c, (uint32_t)slot);
        for (uint32_t h = 0; h < n_heads; h++) {
            const uint16_t* ksrc = kf + (h / rep) * hd;
            const uint16_t* vsrc = vf + (h / rep) * hd;
            uint16_t* kdst = kT + ((size_t)h * hd) * n_kv;
            uint16_t* vdst = vo + (size_t)h * n_kv * hd;
            for (uint32_t d = 0; d < hd; d++) kdst[(size_t)d * n_kv + p] = ksrc[d];
            memcpy(vdst + (size_t)p * hd, vsrc, hd * 2u);
        }
    }
    return 0;
}

static void test_kv_roundtrip(void) {
    enum { KVH = 2, HD = 256, NH = 8, NPOS = 8 };   /* 0.8B full-attn 几何 */
    uint32_t slot_bytes = KVH * HD * 2u * 2u;       /* 2048B, 128 倍数 */
    uint32_t nslots = 8;
    uint8_t* base = memalign(128, (size_t)slot_bytes * nslots);
    uint32_t* posmap = malloc((size_t)nslots * 4u);
    struct kvc c;
    CHECK(kvc_init(&c, base, slot_bytes, nslots, posmap) == 0, "kvc_init");

    uint16_t* ksrc = malloc((size_t)NPOS * KVH * HD * 2u);
    uint16_t* vsrc = malloc((size_t)NPOS * KVH * HD * 2u);
    for (uint32_t i = 0; i < NPOS * KVH * HD; i++) {
        ksrc[i] = my_f32_to_f16(frnd(-1.5f, 1.5f));
        vsrc[i] = my_f32_to_f16(frnd(-1.5f, 1.5f));
    }
    for (uint32_t p = 0; p < NPOS; p++)
        CHECK(kvc_append(&c, ksrc + (size_t)p * KVH * HD, vsrc + (size_t)p * KVH * HD, p) >= 0,
              "append pos %u", p);

    uint16_t* kT = malloc((size_t)NH * HD * NPOS * 2u);
    uint16_t* vo = malloc((size_t)NH * NPOS * HD * 2u);
    CHECK(gather_expand(&c, kT, vo, NPOS, NH, KVH, HD) == 0, "gather");

    /* 期望面: 独立下标重建 (repeat_interleave: 头 h ← kv 头 h/rep) */
    for (uint32_t p = 0; p < NPOS; p++)
        for (uint32_t h = 0; h < NH; h++) {
            uint32_t kv = h / (NH / KVH);
            for (uint32_t d = 0; d < HD; d++) {
                CHECK(kT[((size_t)h * HD + d) * NPOS + p] == ksrc[(size_t)p * KVH * HD + kv * HD + d],
                      "kT h=%u d=%u p=%u", h, d, p);
                CHECK(vo[(size_t)h * NPOS * HD + (size_t)p * HD + d] == vsrc[(size_t)p * KVH * HD + kv * HD + d],
                      "vo h=%u p=%u d=%u", h, p, d);
            }
        }

    /* miss 必炸: 未写的位置 lookup = -1 */
    CHECK(kvc_lookup(&c, NPOS) == -1, "pos %u 应 miss", NPOS);
    /* 环回绕: append 满 8 槽后 pos 8 覆写槽 0 → pos 0 变 miss (位置动态语义) */
    uint16_t dummy[512] = {0};
    kvc_append(&c, dummy, dummy, NPOS);
    CHECK(kvc_lookup(&c, 0) == -1, "回绕后 pos 0 应 miss");
    CHECK(kvc_lookup(&c, NPOS) >= 0, "回绕后 pos %u 应命中", NPOS);

    free(base); free(posmap); free(ksrc); free(vsrc); free(kT); free(vo);
    printf("KV roundtrip: done\n");
}

/* ===================== 2) conv1d_step_f16 vs oracle ===================== */
static void test_conv_step(void) {
    enum { C = 96, DC = 4, T = 3 };
    uint16_t* w = malloc((size_t)DC * C * 2u);
    uint16_t x[T][/*C*/ 96];
    for (int i = 0; i < DC * C; i++) w[i] = my_f32_to_f16(frnd(-0.8f, 0.8f));
    for (int t = 0; t < T; t++)
        for (int ch = 0; ch < C; ch++) x[t][ch] = my_f32_to_f16(frnd(-2.0f, 2.0f));

    float* win = calloc((size_t)(DC - 1) * C, 4);   /* 独立 oracle 窗 (0 起) */
    conv_state_t st = { C, DC, NULL };
    st.win = calloc((size_t)(DC - 1) * C, 4);       /* kernel 窗 */
    uint16_t y[C];

    for (int t = 0; t < T; t++) {
        CHECK(conv1d_step_f16(&st, w, x[t], y) == 0, "step %d", t);
        for (int ch = 0; ch < C; ch++) {
            float s = 0.0f;
            for (int i = 0; i < DC - 1; i++)
                s += my_f16_to_f32(w[(size_t)i * C + ch]) * win[(size_t)i * C + ch];
            s += my_f16_to_f32(w[(size_t)(DC - 1) * C + ch]) * my_f16_to_f32(x[t][ch]);
            float oracle = s / (1.0f + expf(-s));
            float got = my_f16_to_f32(y[ch]);
            CHECK(fabsf(got - oracle) <= 0.02f + 0.02f * fabsf(oracle),
                  "t=%d ch=%d got=%f oracle=%f", t, ch, got, oracle);
        }
        /* oracle 窗推进: win[0..DC-3]=win[1..DC-2], 末行=f32(x[t]) (kernel 存 f32) */
        memmove(win, win + C, (size_t)(DC - 2) * C * 4);
        for (int ch = 0; ch < C; ch++)
            win[(size_t)(DC - 2) * C + ch] = my_f16_to_f32(x[t][ch]);
        /* 独立复核: kernel 窗与 oracle 窗应完全一致 (f32 位级) */
        CHECK(memcmp(st.win, win, (size_t)(DC - 1) * C * 4) == 0, "win mismatch t=%d", t);
    }
    free(w); free(win); free(st.win);
    printf("conv step: done\n");
}

/* ===================== 3) delta chunk 跨步一致性 ===================== */
static void test_delta_chunks(void) {
    enum { H = 4, D = 16, NTOK = 3 };
    /* 输入缓冲 (两套同值): [H][3][D] 单 chunk 用 / 2+1 拆 chunk 用 */
    size_t kvq = (size_t)H * NTOK * D;
    uint16_t *k = malloc(kvq * 2), *v = malloc(kvq * 2), *q = malloc(kvq * 2);
    uint16_t *beta = malloc((size_t)H * NTOK * 2), *g = malloc((size_t)H * NTOK * 2);
    for (size_t i = 0; i < kvq; i++) {
        k[i] = my_f32_to_f16(frnd(-1.0f, 1.0f));
        v[i] = my_f32_to_f16(frnd(-1.0f, 1.0f));
        q[i] = my_f32_to_f16(frnd(-1.0f, 1.0f));
    }
    for (size_t i = 0; i < (size_t)H * NTOK; i++) {
        beta[i] = my_f32_to_f16(frnd(0.2f, 1.0f));
        g[i] = my_f32_to_f16(frnd(-0.3f, 0.05f));   /* 衰减为负/小正 */
    }

    /* A: 单 chunk c=3 */
    float* SA = calloc((size_t)H * D * D, 4);
    rec_state_t stA = { H, D, SA };
    uint16_t* yA = malloc(kvq * 2);
    CHECK(delta_chunk_f16(&stA, k, v, q, beta, g, yA, NTOK, NTOK, 0) == 0, "delta A");

    /* B: 拆 2 chunk: c=2 (ntok=3 缓冲但只算前 2) → c=1 (独立 1-token 缓冲) */
    float* SB = calloc((size_t)H * D * D, 4);
    rec_state_t stB = { H, D, SB };
    uint16_t* yB = malloc(kvq * 2);
    CHECK(delta_chunk_f16(&stB, k, v, q, beta, g, yB, 2, NTOK, 0) == 0, "delta B0");
    uint16_t *k1 = malloc((size_t)H * D * 2), *v1 = malloc((size_t)H * D * 2),
             *q1 = malloc((size_t)H * D * 2);
    uint16_t *b1 = malloc((size_t)H * 2), *g1 = malloc((size_t)H * 2);
    uint16_t* yB1 = malloc((size_t)H * D * 2);
    for (uint32_t h = 0; h < H; h++) {   /* 第 3 token 切成 [H][1][D] 形 */
        memcpy(k1 + (size_t)h * D, k + ((size_t)h * NTOK + 2) * D, (size_t)D * 2);
        memcpy(v1 + (size_t)h * D, v + ((size_t)h * NTOK + 2) * D, (size_t)D * 2);
        memcpy(q1 + (size_t)h * D, q + ((size_t)h * NTOK + 2) * D, (size_t)D * 2);
        b1[h] = beta[h * NTOK + 2]; g1[h] = g[h * NTOK + 2];
    }
    CHECK(delta_chunk_f16(&stB, k1, v1, q1, b1, g1, yB1, 1, 1, 0) == 0, "delta B1");
    for (uint32_t h = 0; h < H; h++)
        memcpy(yB + ((size_t)h * NTOK + 2) * D, yB1 + (size_t)h * D, (size_t)D * 2);

    /* 判据 1: 跨 chunk 输出/状态 vs 单 chunk (chunk 数学精确, 容 f16 舍入) */
    double ss = 0, sa = 0, sb = 0, sdiff = 0;
    float maxdy = 0;
    for (size_t i = 0; i < kvq; i++) {
        float a = my_f16_to_f32(yA[i]), b2 = my_f16_to_f32(yB[i]);
        ss += (double)a * b2; sa += (double)a * a; sb += (double)b2 * b2;
        float d = fabsf(a - b2); if (d > maxdy) maxdy = d;
    }
    double cos_y = ss / (sqrt(sa) * sqrt(sb) + 1e-30);
    float maxds = 0;
    for (size_t i = 0; i < (size_t)H * D * D; i++) {
        float d = fabsf(SA[i] - SB[i]); if (d > maxds) maxds = d;
        sdiff += fabs((double)SA[i] - (double)SB[i]);
    }
    CHECK(cos_y > 0.9999, "y cos(A,B)=%.8f (maxdy=%g)", cos_y, maxdy);
    CHECK(maxds < 5e-3, "S max|dS|=%g (L1=%.3g)", maxds, sdiff);

    /* 判据 2: 单 chunk vs per-token oracle (ref_delta_token 把 st->s 当单 head
     * [d,d] 面用 → 逐 head 独立 rec_state_t, s 指向 SO + h*D*D) */
    float* SO = calloc((size_t)H * D * D, 4);
    double so = 0, aa = 0, oo = 0;
    for (uint32_t h = 0; h < H; h++) {
        rec_state_t stO = { (int)H, (int)D, SO + (size_t)h * D * D };
        for (uint32_t t = 0; t < NTOK; t++) {
            float yo[D], kf[D], vf[D], qf[D];
            for (uint32_t n = 0; n < D; n++) {
                kf[n] = my_f16_to_f32(k[((size_t)h * NTOK + t) * D + n]);
                vf[n] = my_f16_to_f32(v[((size_t)h * NTOK + t) * D + n]);
                qf[n] = my_f16_to_f32(q[((size_t)h * NTOK + t) * D + n]);
            }
            ref_delta_token(&stO, kf, vf, qf,
                            my_f16_to_f32(beta[h * NTOK + t]),
                            my_f16_to_f32(g[h * NTOK + t]), yo);
            for (uint32_t n = 0; n < D; n++) {
                float a = my_f16_to_f32(yA[((size_t)h * NTOK + t) * D + n]);
                so += (double)a * yo[n]; aa += (double)a * a; oo += (double)yo[n] * yo[n];
            }
        }
    }
    double cos_o = so / (sqrt(aa) * sqrt(oo) + 1e-30);
    CHECK(cos_o > 0.9999, "y cos(A,oracle)=%.8f", cos_o);

    free(k); free(v); free(q); free(beta); free(g);
    free(SA); free(SB); free(SO); free(yA); free(yB); free(yB1);
    free(k1); free(v1); free(q1); free(b1); free(g1);
    printf("delta chunks: cos(yA,yB)=%.6f maxdS=%g cos(yA,oracle)=%.6f\n", cos_y, maxds, cos_o);
}

int main(void) {
    test_kv_roundtrip();
    test_conv_step();
    test_delta_chunks();
    if (n_fail) { printf("== %d FAIL ==\n", n_fail); return 1; }
    printf("== all green ==\n");
    return 0;
}
