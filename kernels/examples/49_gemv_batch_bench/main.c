/*
 * 49_gemv_batch_bench — HVX batched-GEMV 微基准 (lm_head M=32 GEMV 缺口裁决)
 * =====================================================================
 * 背景 (OP35 裁决): lm_head prefill M=32×K1024×N_full=248320 分 32 chunk,
 * 每 chunk 7904 列 HMX 13.5ms = 37.6 GFLOP/s (峰值 0.3%), 全 lm_head 433ms。
 * M=32 = 恰 1 HMX 行 tile, 权重流喂不饱 MAC。候选 = HVX batched-GEMV:
 * 权重向量读一次, 同列块同时点积 G 行 (生产 GEMV 是 m 外层, 权重读 M 次)。
 *
 * 裁决问题:
 *   Q1 mpyacc (Q6_Wsf_vmpyacc_WsfVhfVhf, v81) 复合体 (vmpy.qf32+2×vadd)
 *      的持续发射率 —— 每 chunk 复合数恒定 ≈ 4.04M → 计算下界=4.04M/rate。
 *   Q2 G=4/8/16 相对 G=1(现状结构) 的实测增益 (权重流量 32/G 倍降)。
 *   Q3 权重布局代价: chunk 连续 [K,7936] vs 全矩阵跨步 (row stride=N_full)。
 *   Q4 两种布局的纯流读带宽 (128B 粒度)。
 *   Q5 f16 内层累加变体 (免 qf32↔sf 转换对, 每复合 ~4 条): 指令地板
 *      能否压到 13.5ms HMX 之下, 精度代价多少 (64-k 冲洗 f32)。
 *
 * 纪律 (htp-kernel-measurement): 预热 2 + 计时 8 取中位数 (绝不取 min);
 * PCYCLE+us 双计 → 时钟自检 cyc/us; 先守门后报数:
 *   G1) G∈{4,8,16} 输出 vs G1 逐字节一致 (逐元素 k 升序 f32 累加序不变)
 *   G2) G1 输出 vs 标量 f32 参考 rel-norm < 1e-3 (f16 存储舍入口径)
 *   G3) 流读探针 dst 逐字节 == src (搬运真实性)
 *   G4) f16acc vs G1 rel-norm < 2% (量化精度代价, 非位级)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <hexagon_types.h>
#include "hvxhmx_runtime.h"
#include "internal/hvx-base.h"
#include "example_util.h"

/* ---- 形状: lm_head 单 chunk 口径 (N 取 124×64=7936 免尾列) ---- */
#define MM 32u
#define KK 1024u
#define NB_BLK 124u            /* 7936 / 64 */
#define NC (NB_BLK * 64u)      /* 7936 chunk 连续面板 */
#define NV_BLK 32u             /* VTCM 面板 2048 列 (W 4MB 驻留) */
#define NV (NV_BLK * 64u)
#define NFULL 248320u          /* 全矩阵行步长 (元素) */
#define REPS 8
#define WARM 2

static uint32_t lcg = 20261013u;
static uint32_t lcg_n(void) { lcg = lcg * 1664525u + 1013904223u; return lcg; }
static uint16_t lcg_f16(void) {
    uint32_t r = lcg_n();
    return (uint16_t)(0x3800u | ((r >> 5) & 0x3FFu) | ((r & 0xFu) << 10));
    /* 指数位 ~[2^-9,2^?) 附近 + 随机尾数: 无 denormal, K=1024 dot 无溢出 */
}
static float f16_2_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = s;                       /* ±0 */
        else {                                       /* subnormal → 规格化 */
            e = 127u - 15u + 1u;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            m &= 0x3FFu;
            bits = s | (e << 23) | (m << 13);
        }
    } else if (e == 0x1Fu) {
        bits = s | 0x7F800000u | (m << 13);          /* inf/NaN */
    } else {
        bits = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ---- batched-GEMV: A[32,K] 行主 × W 行主 [K,N](row 步长 wstride 元素),
 * Y[32,N]。G 行共享一次权重向量读 (n 块 × m 组 两层外循环)。
 * 与生产 exec_matmul_f16_gemv !t0&&!t1 体同款向量体/收尾 (mpyacc 连续
 * 两半布局, shuff 收尾), 仅循环嵌套不同。 ---- */
#define GEMV_BATCH(GI)                                                        \
static void gemv_b##GI(const uint16_t* a, const uint16_t* w, uint16_t* y,     \
                       uint32_t K, uint32_t N, uint32_t nblk, size_t wstride) {\
    for (uint32_t nb = 0; nb < nblk; ++nb) {                                  \
        uint32_t n0 = nb * 64u;                                               \
        for (uint32_t g0 = 0; g0 < MM; g0 += GI) {                            \
            HVX_VectorPair acc[GI];                                           \
            for (uint32_t g = 0; g < GI; ++g) acc[g] = Q6_W_vzero();          \
            for (uint32_t k = 0; k < K; ++k) {                                \
                HVX_Vector wv = hvx_vmemu(w + (size_t)k * wstride + n0);      \
                for (uint32_t g = 0; g < GI; ++g) {                           \
                    HVX_Vector av = Q6_Vh_vsplat_R(a[(size_t)(g0 + g) * K + k]);\
                    acc[g] = hvx_vec_mpyacc_f32_f16(acc[g], av, wv);          \
                }                                                             \
            }                                                                 \
            for (uint32_t g = 0; g < GI; ++g)                                 \
                hvx_vmemu(y + (size_t)(g0 + g) * N + n0) =                    \
                    hvx_vec_f32_to_f16_shuff(Q6_V_lo_W(acc[g]), Q6_V_hi_W(acc[g]));\
        }                                                                     \
    }                                                                         \
}
GEMV_BATCH(1)
GEMV_BATCH(4)
GEMV_BATCH(8)
GEMV_BATCH(16)


/* ---- f16 内层累加变体 (G=4): 内层 64-k 段用 f16 向量累加 (vmpy+vadd
 * 每复合 ~4 条, 免 qf32↔sf 转换对), 每 64 k 冲洗进该行 f32 链 (K=1024 整除)。
 * 精度代价由守门量化 (vs G1 rel-norm), 非位级。 ---- */
static void gemv_f16acc4(const uint16_t* a, const uint16_t* w, uint16_t* y,
                         uint32_t K, uint32_t N, uint32_t nblk, size_t wstride) {
    for (uint32_t nb = 0; nb < nblk; ++nb) {
        uint32_t n0 = nb * 64u;
        for (uint32_t g0 = 0; g0 < MM; g0 += 4u) {
            HVX_Vector fl0 = Q6_V_vzero(), fh0 = Q6_V_vzero();
            HVX_Vector fl1 = Q6_V_vzero(), fh1 = Q6_V_vzero();
            HVX_Vector fl2 = Q6_V_vzero(), fh2 = Q6_V_vzero();
            HVX_Vector fl3 = Q6_V_vzero(), fh3 = Q6_V_vzero();
            for (uint32_t kp = 0; kp < K; kp += 64u) {
                HVX_Vector ah0 = Q6_V_vzero(), ah1 = Q6_V_vzero();
                HVX_Vector ah2 = Q6_V_vzero(), ah3 = Q6_V_vzero();
                for (uint32_t kk = 0; kk < 64u; ++kk) {
                    uint32_t k = kp + kk;
                    HVX_Vector wv = hvx_vmemu(w + (size_t)k * wstride + n0);
                    ah0 = Q6_Vhf_vadd_VhfVhf(ah0, Q6_Vhf_vmpy_VhfVhf(
                            Q6_Vh_vsplat_R(a[(size_t)(g0 + 0u) * K + k]), wv));
                    ah1 = Q6_Vhf_vadd_VhfVhf(ah1, Q6_Vhf_vmpy_VhfVhf(
                            Q6_Vh_vsplat_R(a[(size_t)(g0 + 1u) * K + k]), wv));
                    ah2 = Q6_Vhf_vadd_VhfVhf(ah2, Q6_Vhf_vmpy_VhfVhf(
                            Q6_Vh_vsplat_R(a[(size_t)(g0 + 2u) * K + k]), wv));
                    ah3 = Q6_Vhf_vadd_VhfVhf(ah3, Q6_Vhf_vmpy_VhfVhf(
                            Q6_Vh_vsplat_R(a[(size_t)(g0 + 3u) * K + k]), wv));
                }
                HVX_VectorPair p0 = hvx_vec_f16_to_f32(ah0);
                HVX_VectorPair p1 = hvx_vec_f16_to_f32(ah1);
                HVX_VectorPair p2 = hvx_vec_f16_to_f32(ah2);
                HVX_VectorPair p3 = hvx_vec_f16_to_f32(ah3);
                fl0 = hvx_vec_add_f32_f32(fl0, Q6_V_lo_W(p0));
                fh0 = hvx_vec_add_f32_f32(fh0, Q6_V_hi_W(p0));
                fl1 = hvx_vec_add_f32_f32(fl1, Q6_V_lo_W(p1));
                fh1 = hvx_vec_add_f32_f32(fh1, Q6_V_hi_W(p1));
                fl2 = hvx_vec_add_f32_f32(fl2, Q6_V_lo_W(p2));
                fh2 = hvx_vec_add_f32_f32(fh2, Q6_V_hi_W(p2));
                fl3 = hvx_vec_add_f32_f32(fl3, Q6_V_lo_W(p3));
                fh3 = hvx_vec_add_f32_f32(fh3, Q6_V_hi_W(p3));
            }
            hvx_vmemu(y + (size_t)(g0 + 0u) * N + n0) = hvx_vec_f32_to_f16_shuff(fl0, fh0);
            hvx_vmemu(y + (size_t)(g0 + 1u) * N + n0) = hvx_vec_f32_to_f16_shuff(fl1, fh1);
            hvx_vmemu(y + (size_t)(g0 + 2u) * N + n0) = hvx_vec_f32_to_f16_shuff(fl2, fh2);
            hvx_vmemu(y + (size_t)(g0 + 3u) * N + n0) = hvx_vec_f32_to_f16_shuff(fl3, fh3);
        }
    }
}

/* ---- 计时: 中位数 ---- */
static uint64_t med_u64(uint64_t* a, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
    return a[n / 2];
}
typedef void (*gemv_fn)(const uint16_t*, const uint16_t*, uint16_t*,
                        uint32_t, uint32_t, uint32_t, size_t);

static volatile uint32_t g_sink = 0;

/* 跑一个变体: 报 cyc/us + GFLOP/s + 权重流 GB/s。返回中位 us。 */
static int64_t run_variant(const char* tag, gemv_fn fn, const uint16_t* a,
                           const uint16_t* w, uint16_t* y, uint32_t N,
                           uint32_t nblk, size_t wstride, uint32_t G) {
    uint64_t cyc[REPS]; int64_t us[REPS];
    for (int r = 0; r < WARM; ++r)
        fn(a, w, y, KK, N, nblk, wstride);
    for (int r = 0; r < REPS; ++r) {
        uint64_t c0 = HAP_perf_get_pcycles();
        int64_t t0 = HAP_perf_get_time_us();
        fn(a, w, y, KK, N, nblk, wstride);
        cyc[r] = HAP_perf_get_pcycles() - c0;
        us[r] = HAP_perf_get_time_us() - t0;
    }
    uint64_t mc = med_u64(cyc, REPS);
    int64_t mu = med_u64((uint64_t*)us, REPS);
    g_sink ^= y[0] ^ y[(size_t)(MM - 1) * N + N - 1];
    double flops = 2.0 * MM * (double)KK * nblk * 64.0;
    double wbytes = (double)(MM / G) * nblk * KK * 128.0;
    double gfs = mu > 0 ? flops / (double)mu / 1000.0 : 0.0;
    double gbs = mu > 0 ? wbytes / (double)mu / 1000.0 : 0.0;
    double cyc_kstep = (double)mc / ((double)nblk * (MM / G) * KK);
    ex_log("%-24s G=%-2u %8llu cyc %7lld us  %7.1f GF/s  %6.1f GB/s  %.2f cyc/kstep",
           tag, G, (unsigned long long)mc, (long long)mu, gfs, gbs, cyc_kstep);
    return mu;
}

/* ---- Q1: mpyacc 纯发射率 (8 独立链, 操作数驻寄存器, 源向量 VTCM 载一次) ---- */
static double bench_mpyacc_rate(const uint16_t* wvtcm) {
    HVX_Vector x = hvx_vmemu(wvtcm);
    HVX_Vector yv = hvx_vmemu(wvtcm + 64);
    HVX_VectorPair a[8];
    for (int i = 0; i < 8; ++i) a[i] = Q6_W_vzero();
    const uint32_t ITERS = 4096u;
    uint64_t cyc[REPS];
    for (int r = 0; r < WARM + REPS; ++r) {
        int timed = r >= WARM;
        uint64_t c0 = HAP_perf_get_pcycles();
        for (uint32_t i = 0; i < ITERS; ++i)
            for (int g = 0; g < 8; ++g)
                a[g] = hvx_vec_mpyacc_f32_f16(a[g], x, yv);
        uint64_t dc = HAP_perf_get_pcycles() - c0;
        if (timed) cyc[r - WARM] = dc;
        else for (int g = 0; g < 8; ++g) a[g] = Q6_W_vzero();
    }
    uint16_t sink[64] __attribute__((aligned(128)));
    hvx_vmemu(sink) = hvx_vec_f32_to_f16_shuff(Q6_V_lo_W(a[0]), Q6_V_hi_W(a[0]));
    g_sink ^= sink[0];
    uint64_t mc = med_u64(cyc, REPS);
    double per = (double)mc / (double)(ITERS * 8);
    ex_log("mpyacc_rate        8链×%u it: %llu cyc → %.3f cyc/mpyacc (%.2f instr/cyc)",
           ITERS, (unsigned long long)mc, per, 1.0 / per);
    return 1.0 / per;
}

/* ---- Q4: 纯流读 (8 独立 xor 链防死码防串行, 测带宽非延迟) ---- */
static uint64_t stream_read(const uint16_t* w, uint32_t nblk, size_t wstride,
                            uint16_t* dst_chk) {
    (void)nblk;
    HVX_Vector x0 = Q6_V_vzero(), x1 = Q6_V_vzero(), x2 = Q6_V_vzero(), x3 = Q6_V_vzero();
    HVX_Vector x4 = Q6_V_vzero(), x5 = Q6_V_vzero(), x6 = Q6_V_vzero(), x7 = Q6_V_vzero();
    uint64_t cyc[REPS]; int64_t us[REPS];
    for (int r = 0; r < WARM + REPS; ++r) {
        int timed = r >= WARM;
        uint64_t c0 = HAP_perf_get_pcycles();
        int64_t t0 = HAP_perf_get_time_us();
        for (uint32_t k = 0; k < KK; ++k) {
            const uint16_t* row = w + (size_t)k * wstride;
            x0 = Q6_V_vxor_VV(x0, hvx_vmemu(row + 0u));
            x1 = Q6_V_vxor_VV(x1, hvx_vmemu(row + 64u));
            x2 = Q6_V_vxor_VV(x2, hvx_vmemu(row + 128u));
            x3 = Q6_V_vxor_VV(x3, hvx_vmemu(row + 192u));
            x4 = Q6_V_vxor_VV(x4, hvx_vmemu(row + 256u));
            x5 = Q6_V_vxor_VV(x5, hvx_vmemu(row + 320u));
            x6 = Q6_V_vxor_VV(x6, hvx_vmemu(row + 384u));
            x7 = Q6_V_vxor_VV(x7, hvx_vmemu(row + 448u));
        }
        uint64_t dc = HAP_perf_get_pcycles() - c0;
        int64_t du = HAP_perf_get_time_us() - t0;
        if (timed) { cyc[r - WARM] = dc; us[r - WARM] = du; }
    }
    HVX_Vector acc = Q6_V_vxor_VV(Q6_V_vxor_VV(Q6_V_vxor_VV(x0, x1), Q6_V_vxor_VV(x2, x3)),
                                  Q6_V_vxor_VV(Q6_V_vxor_VV(x4, x5), Q6_V_vxor_VV(x6, x7)));
    hvx_vmemu(dst_chk) = acc;
    g_sink ^= dst_chk[0];
    uint64_t mc = med_u64(cyc, REPS);
    int64_t mu = med_u64((uint64_t*)us, REPS);
    double bytes = (double)KK * 8u * 128.0;
    ex_log("stream_read stride=%zu (1KB/k): %llu cyc %lld us  %.1f GB/s  %.2f cyc/128B",
           wstride, (unsigned long long)mc, (long long)mu,
           mu > 0 ? bytes / (double)mu / 1000.0 : 0.0,
           (double)mc / (double)(KK * 8u));
    return (uint64_t)mu;
}

int main(void) {
    ex_open_result("49_gemv_batch_bench");

    /* ---- 面: VTCM 驻留 (Q1/Q2 计算上限) + DDR 两种布局 (Q3/Q4) ---- */
    if (hmx_runtime_setup(16u * 1024u * 1024u) != 0) {
        ex_log("!! hmx_runtime_setup"); ex_summary(); return 1;
    }
    uint8_t* vtcm = (uint8_t*)hmx_runtime_get_vtcm_base();
    if (!vtcm) { ex_log("!! no vtcm"); ex_summary(); return 1; }
    uint16_t* w_v = (uint16_t*)(vtcm + 2048u);            /* [K,NV] 驻留 4MB */

    uint16_t* a = memalign(128, (size_t)MM * KK * 2u);            /* [32,K] */
    uint16_t* w_c = memalign(128, (size_t)KK * NC * 2u);          /* chunk 连续 [K,7936] */
    uint16_t* w_s = memalign(128, (size_t)KK * NFULL * 2u);       /* 全矩阵跨步窗 */
    uint16_t* y1 = memalign(128, (size_t)MM * NC * 2u);
    uint16_t* y4 = memalign(128, (size_t)MM * NC * 2u);
    uint16_t* y8 = memalign(128, (size_t)MM * NC * 2u);
    uint16_t* y16 = memalign(128, (size_t)MM * NC * 2u);
    uint16_t* yv = memalign(128, (size_t)MM * NV * 2u);
    uint16_t* chk = memalign(128, 128u);
    if (!a || !w_c || !w_s || !y1 || !y4 || !y8 || !y16 || !yv || !chk) {
        ex_log("!! alloc (w_s=%.0fMB)", (double)KK * NFULL * 2 / 1048576.0);
        ex_summary(); return 1;
    }
    ex_log("alloc ok (w_s 全矩阵 %.0fMB, 只触 chunk 窗)",
           (double)KK * NFULL * 2 / 1048576.0);
    for (uint32_t i = 0; i < MM * KK; i++) a[i] = lcg_f16();
    for (uint32_t i = 0; i < (uint32_t)KK * NC; i++) w_c[i] = lcg_f16();
    for (uint32_t k = 0; k < KK; k++) {                     /* 跨步窗 = 同 w_c 内容 */
        uint16_t* row = w_s + (size_t)k * NFULL;
        for (uint32_t n = 0; n < NC; n++) row[n] = w_c[(size_t)k * NC + n];
    }
    for (uint32_t i = 0; i < (uint32_t)KK * NV; i++) w_v[i] = lcg_f16();
    qurt_mem_cache_clean((qurt_addr_t)w_v, (size_t)KK * NV * 2u,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)a, (size_t)MM * KK * 2u,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)w_c, (size_t)KK * NC * 2u,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)w_v, (size_t)KK * NV * 2u,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    /* w_s 只清触过的每行 chunk 窗 (免 508MB VA 大范围 clean 踩未映射页) */
    for (uint32_t k = 0; k < KK; k++)
        qurt_mem_cache_clean((qurt_addr_t)(w_s + (size_t)k * NFULL),
                             (size_t)NC * 2u,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    /* ---- 守门 G2: G1 vs 标量 f32 参考 (前 4 列块窗口, rel-norm) ---- */
    gemv_b1(a, w_c, y1, KK, NC, NB_BLK, NC);
    {
        double sse = 0.0, srr = 0.0;
        uint32_t WCOLS = 256u;
        for (uint32_t m = 0; m < MM; m++)
            for (uint32_t n = 0; n < WCOLS; n++) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < KK; k++)
                    acc += f16_2_f32(a[(size_t)m * KK + k]) *
                           f16_2_f32(w_c[(size_t)k * NC + n]);
                float got = f16_2_f32(y1[(size_t)m * NC + n]);
                double d = (double)got - (double)acc;
                sse += d * d;
                srr += (double)acc * (double)acc;
            }
        double nrm = srr > 0.0 ? __builtin_sqrt(sse / srr) : 1.0;
        ex_log("G2 scalar-ref rel-norm=%.5g (f16 存储舍入口径)", nrm);
        ex_check("G2_g1_vs_scalar_ref", nrm < 1e-3 ? 0 : 1, 0);
    }
    /* ---- 守门 G1: G∈{4,8,16} vs G1 逐字节一致 ---- */
    gemv_b4(a, w_c, y4, KK, NC, NB_BLK, NC);
    gemv_b8(a, w_c, y8, KK, NC, NB_BLK, NC);
    gemv_b16(a, w_c, y16, KK, NC, NB_BLK, NC);
    ex_check("G1_g4_vs_g1_byteexact", memcmp(y4, y1, (size_t)MM * NC * 2u), 0);
    ex_check("G1_g8_vs_g1_byteexact", memcmp(y8, y1, (size_t)MM * NC * 2u), 0);
    ex_check("G1_g16_vs_g1_byteexact", memcmp(y16, y1, (size_t)MM * NC * 2u), 0);
    /* ---- 守门 G4: f16acc 精度代价 (vs G1 f32 链, rel-norm, 非位级) ---- */
    gemv_f16acc4(a, w_c, y4, KK, NC, NB_BLK, NC);
    {
        double sse = 0.0, srr = 0.0;
        for (uint32_t m = 0; m < MM; m++)
            for (uint32_t n = 0; n < NC; n++) {
                double r = (double)f16_2_f32(y1[(size_t)m * NC + n]);
                double g = (double)f16_2_f32(y4[(size_t)m * NC + n]);
                sse += (g - r) * (g - r);
                srr += r * r;
            }
        double nrm = srr > 0.0 ? __builtin_sqrt(sse / srr) : 1.0;
        ex_log("G4 f16acc(64-flush) vs G1 rel-norm=%.5g", nrm);
        ex_check("G4_f16acc_relnorm_2pct", nrm < 0.02 ? 0 : 1, 0);
    }

    /* ---- Q1: mpyacc 发射率 ---- */
    double rate = bench_mpyacc_rate(w_v);
    ex_log("→ mpyacc 计算下界 ≈ %.2f M-cyc/chunk (每 chunk 4.03M 条, cyc→ms 用时钟自检折算)",
           4.03 / rate);

    /* ---- Q4: 纯流读两种布局 ---- */
    stream_read(w_c, NB_BLK, NC, chk);
    stream_read(w_s, NB_BLK, NFULL, chk);

    /* ---- Q2/Q3: batched-GEMV 全变体 ---- */
    ex_log("-- W@VTCM 驻留 (NV=2048 面板, 计算上限; ×%.2f 折算 chunk)",
           (double)NB_BLK / (double)NV_BLK);
    run_variant("vtcm G=1", gemv_b1, a, w_v, yv, NV, NV_BLK, NV, 1);
    run_variant("vtcm G=4", gemv_b4, a, w_v, yv, NV, NV_BLK, NV, 4);
    run_variant("vtcm G=8", gemv_b8, a, w_v, yv, NV, NV_BLK, NV, 8);
    run_variant("vtcm G=16", gemv_b16, a, w_v, yv, NV, NV_BLK, NV, 16);
    ex_log("-- W@DDR chunk 连续 [K,7936] (emit 侧 chunk 槽行主布局)");
    run_variant("ddr-c G=1 (现状结构)", gemv_b1, a, w_c, y1, NC, NB_BLK, NC, 1);
    run_variant("ddr-c G=4", gemv_b4, a, w_c, y4, NC, NB_BLK, NC, 4);
    run_variant("ddr-c G=8", gemv_b8, a, w_c, y8, NC, NB_BLK, NC, 8);
    run_variant("ddr-c G=16", gemv_b16, a, w_c, y16, NC, NB_BLK, NC, 16);
    ex_log("-- W@DDR 全矩阵跨步 stride=248320 (供给侧零改动布局)");
    run_variant("ddr-s G=1", gemv_b1, a, w_s, y1, NC, NB_BLK, NFULL, 1);
    run_variant("ddr-s G=4", gemv_b4, a, w_s, y4, NC, NB_BLK, NFULL, 4);
    run_variant("ddr-s G=8", gemv_b8, a, w_s, y8, NC, NB_BLK, NFULL, 8);
    run_variant("ddr-s G=16", gemv_b16, a, w_s, y16, NC, NB_BLK, NFULL, 16);
    ex_log("-- f16 内层累加 G=4 (vmpy+vadd 每复合 ~4 条, 64-k 冲洗 f32)");
    run_variant("f16acc G=4 w@VTCM", gemv_f16acc4, a, w_v, yv, NV, NV_BLK, NV, 4);
    run_variant("f16acc G=4 w@ddr-c", gemv_f16acc4, a, w_c, y4, NC, NB_BLK, NC, 4);
    run_variant("f16acc G=4 w@ddr-s", gemv_f16acc4, a, w_s, y4, NC, NB_BLK, NFULL, 4);

    /* ---- 时钟自检 ---- */
    {
        uint64_t c0 = HAP_perf_get_pcycles();
        int64_t t0 = HAP_perf_get_time_us();
        volatile uint32_t spin = 0;
        while (HAP_perf_get_time_us() - t0 < 20000) spin++;
        g_sink ^= spin;
        ex_log("clock self-check: %.1f cyc/us (PCYCLE vs wall)",
               (double)(HAP_perf_get_pcycles() - c0) / 20000.0);
    }

    free(a); free(w_c); free(w_s); free(y1); free(y4); free(y8); free(y16);
    free(yv); free(chk);
    return ex_summary();
}
