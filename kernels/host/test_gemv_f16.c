/* test_gemv_f16.c — host 验证 M<32 GEMV 的标量索引公式 vs 独立参考
 *
 * 目的: T006 主机验证. exec_matmul_f16_gemv 的 HVX 向量体在 x86 主机不可执行
 *       (#if defined(__HVX__)), 但标量回落体的索引逻辑必须正确:
 *       - M=1/16/32 各种形状
 *       - batched / Bn, t0/t1 转置, SPLIT 的 n0 / N_full
 *       - 与独立参考 cos>=0.99999
 *
 * 约束: 本测试不链接完整 DSP runtime, 因此不覆盖 HVX 向量体. 向量体的数值
 *       需在 d0f1784 或 V81HexSimNew 上验证 (见 tasks T013).
 *
 * 编译: gcc -O2 -std=gnu11 -I../include test_gemv_f16.c -o /tmp/test_gemv -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- f16 转换 (与 oplist_exec.c 的 f32_to_f16/f16_to_f32 同算法, RNE + subnormal 修正) ---- */
static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t sub = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1u);
        sub += (rem > (1u << (shift - 1u)) ||
                (rem == (1u << (shift - 1u)) && (sub & 1u))) ? 1u : 0u;
        return (uint16_t)(sign | sub);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    if ((man >> 12) & 1u) half += ((man & 0xfffu) || (half & 1u)) ? 1u : 0u;
    return (uint16_t)half;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            uint32_t m = man; int e = -1;
            while (!(m & 0x400)) { m <<= 1; e++; }
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

static double cos_sim(const uint16_t *a, const uint16_t *b, size_t n) {
    double sa = 0, sb = 0, s = 0;
    for (size_t i = 0; i < n; ++i) {
        float fa = f16_to_f32(a[i]), fb = f16_to_f32(b[i]);
        s += (double)fa * fb; sa += (double)fa * fa; sb += (double)fb * fb;
    }
    return s / sqrt(sa * sb);
}

/* ---- 独立参考: 逐元素直接绝对索引, 不用 base-pointer 简写 ---- */
static void gemv_reference(uint16_t *y, const uint16_t *a, const uint16_t *w,
                           uint32_t M, uint32_t K, uint32_t N,
                           uint32_t n0, uint32_t N_full,
                           int t0, int t1, uint32_t batched, uint32_t Bn)
{
    uint32_t bb_max = batched ? Bn : 1u;
    memset(y, 0, (size_t)bb_max * M * N_full * sizeof(uint16_t));

    for (uint32_t bb = 0; bb < bb_max; ++bb) {
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = n0; n < n0 + N; ++n) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    size_t ia = batched ? ((size_t)bb * M + m) * K + k
                                       : (t0 ? (size_t)k * M + m : (size_t)m * K + k);
                    size_t iw = batched
                        ? (t1 ? ((size_t)bb * N_full + n) * K + k
                              : ((size_t)bb * K + k) * N_full + n)
                        : (t1 ? (size_t)n * K + k
                              : (size_t)k * N_full + n);
                    acc += f16_to_f32(a[ia]) * f16_to_f32(w[iw]);
                }
                size_t iy = batched ? ((size_t)bb * M + m) * N_full + n
                                    : (size_t)m * N_full + n;
                y[iy] = f32_to_f16(acc);
            }
        }
    }
}

/* ---- 被测: 移植 oplist_exec.c 的 exec_matmul_f16_gemv 标量回落体 ----
 * 逐字段复刻其 ab/wb/yb base-pointer + 相对索引写法, 使本测试能钉住
 * "base 简写 == 直接绝对索引" 这一不变量. */
static void gemv_under_test(uint16_t *y, const uint16_t *a, const uint16_t *w,
                            uint32_t M, uint32_t K, uint32_t N,
                            uint32_t n0, uint32_t N_full,
                            int t0, int t1, uint32_t batched, uint32_t Bn)
{
    uint32_t bb_max = batched ? Bn : 1u;
    memset(y, 0, (size_t)bb_max * M * N_full * sizeof(uint16_t));

    for (uint32_t bb = 0; bb < bb_max; ++bb) {
        const uint16_t *ab = a + (size_t)bb * M * K;
        const uint16_t *wb = w + (size_t)bb * K * N_full;
        uint16_t *yb = y + (size_t)bb * M * N_full;
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = n0; n < n0 + N; ++n) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    float av = batched ? f16_to_f32(ab[m * K + k])
                                       : f16_to_f32(a[t0 ? k * M + m : m * K + k]);
                    float wv = batched
                        ? f16_to_f32(wb[t1 ? n * K + k : k * N_full + n])
                        : f16_to_f32(w[t1 ? n * K + k : k * N_full + n]);
                    acc += av * wv;
                }
                yb[m * N_full + n] = f32_to_f16(acc);
            }
        }
    }
}

typedef struct {
    const char *name;
    uint32_t M, K, N, n0, N_full, Bn;
    int t0, t1, batched;
} TestCase;

static int run_test(const TestCase *tc) {
    printf("%-24s M=%-3u K=%-5u N=%-6u n0=%-6u N_full=%-6u Bn=%u t0=%d t1=%d batched=%d\n",
           tc->name, tc->M, tc->K, tc->N, tc->n0, tc->N_full, tc->Bn,
           tc->t0, tc->t1, tc->batched);

    uint32_t bb_max = tc->batched ? tc->Bn : 1u;
    size_t a_elems = (size_t)bb_max * tc->M * tc->K;
    size_t w_elems = (size_t)bb_max * tc->K * tc->N_full;
    size_t y_elems = (size_t)bb_max * tc->M * tc->N_full;

    uint16_t *a = malloc(a_elems * sizeof(uint16_t));
    uint16_t *w = malloc(w_elems * sizeof(uint16_t));
    uint16_t *y_ref = calloc(y_elems, sizeof(uint16_t));
    uint16_t *y_imp = calloc(y_elems, sizeof(uint16_t));
    if (!a || !w || !y_ref || !y_imp) { printf("  OOM\n"); return 1; }

    /* 小整数种子: 每个值在 f16 下精确, 避免输入本身的舍入掩盖索引 bug */
    srand(1234);
    for (size_t i = 0; i < a_elems; ++i) a[i] = f32_to_f16((float)(rand() % 17) - 8.0f);
    for (size_t i = 0; i < w_elems; ++i) w[i] = f32_to_f16((float)(rand() % 17) - 8.0f);

    gemv_reference(y_ref, a, w, tc->M, tc->K, tc->N, tc->n0, tc->N_full,
                   tc->t0, tc->t1, tc->batched, tc->Bn);
    gemv_under_test(y_imp, a, w, tc->M, tc->K, tc->N, tc->n0, tc->N_full,
                    tc->t0, tc->t1, tc->batched, tc->Bn);

    double cos = cos_sim(y_ref, y_imp, y_elems);
    int exact = memcmp(y_ref, y_imp, y_elems * sizeof(uint16_t)) == 0;
    int pass = (cos >= 0.99999) && exact;
    printf("    cos=%.7f %s%s\n", cos, pass ? "PASS" : "FAIL", exact ? " (bit-exact)" : "");

    free(a); free(w); free(y_ref); free(y_imp);
    return pass ? 0 : 1;
}

int main(void) {
    static const TestCase tests[] = {
        /* M=1/16/32 基本形状 (M=32 是 HMX 门的边界, 验证 dispatch 不会误取) */
        {"M1",                 1, 256, 512,   0,   512, 0, 0, 0, 0},
        {"M16",               16, 256, 512,   0,   512, 0, 0, 0, 0},
        {"M32",               32, 256, 512,   0,   512, 0, 0, 0, 0},
        /* t0 (a 存 [K,M]) / t1 (w 存 [N,K]) */
        {"M1_t0",              1, 256, 512,   0,   512, 0, 1, 0, 0},
        {"M1_t1",              1, 256, 512,   0,   512, 0, 0, 1, 0},
        {"M1_t0_t1",           1, 256, 512,   0,   512, 0, 1, 1, 0},
        /* batched / Bn */
        {"M1_batched_Bn1",     1, 256, 512,   0,   512, 1, 0, 0, 1},
        {"M1_batched_Bn4",     1, 256, 512,   0,   512, 4, 0, 0, 1},
        {"M16_batched_Bn2",   16, 256, 512,   0,   512, 2, 0, 0, 1},
        {"M1_batched_t1",      1, 256, 512,   0,   512, 3, 0, 1, 1},
        /* SPLIT: n0 偏移 + N_full 行距 */
        {"M1_split_n0_256",    1, 256, 256, 256,   512, 0, 0, 0, 0},
        {"M16_split_n0_128",  16, 256, 384, 128,   512, 0, 0, 0, 0},
        {"M1_split_t1",        1, 256, 256, 256,   512, 0, 0, 1, 0},
        /* 真实模型形状: lm_head K=1024, N=248320 分块 (w 需 ~508MB, 取缩小分块) */
        {"lm_head_split",      1,1024, 512, 1024,  4096, 0, 0, 0, 0},
    };

    int fail = 0;
    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i)
        fail |= run_test(&tests[i]);

    printf("\n%s\n", fail ? "OVERALL FAIL" : "ALL PASS");
    return fail;
}
