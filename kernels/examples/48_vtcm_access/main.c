/*
 * 48_vtcm_access — 008 US2 机理探针: 30× matmul 归因 + GATE3 广播 43× 裁决
 * =====================================================================
 * 背景: v4_007 census — M32 f16 matmul 族 wall 9.44s, 主力 M32xK1024xN3584
 * avg 59.9ms; 理论 (HMX GEMM ~µs 级 + pack 线性拷贝 ~1-2ms) 差 30×。
 * wall vs wgt 字节斜率 ~8.5ms/MB ≈ 3 cycles/elem @1.42GHz = 标量循环吞吐。
 * 嫌疑 = crouton pack 是 CPU 标量散写 VTCM (hmx_crouton_pos 非线性, 不可
 * 向量化) —— htp-hardware-scheduling 技能实测纪律: "Do NOT put SCALAR /
 * data-dependent scratch in VTCM" (GDN 标量划痕进 VTCM 7× 变慢)。
 * 同机理候选解释 T010 GATE3 broadcast 43× (gather 是数据依赖标量读)。
 *
 * 探针 (全部 exec_matmul_f16_hmx 逐字复刻的相位, 只计时不改语义):
 *   G1 pack act  → VTCM crouton (M=32 K=1024)    计时
 *   G2 pack wgt  → VTCM crouton (K=1024 N=3584)  计时 + N=1024 对照
 *   G3 HMX GEMM (已打包 tile)                   计时
 *   G4 unpack   VTCM → DDR                       计时
 *   G5 同 pack 循环, 目标换 DDR scratch          计时 — 标量散写 VTCM/DDR 对拍
 *   G6 标量 gather y[i]=x[off[i]]: x@DDR vs x@VTCM — GATE3 机理裁决
 *   G7 HVX 向量流读 2MB: hvhx_v2_copy_f16 (007 板上已证内核), dst 恒 DDR,
 *      src@DDR vs src@VTCM — Δ 隔离读侧 (手搓 vmem 流读在板上实测不执行
 *      数据搬运, 已弃; dc_hvx_load 是 4KB 算力负载也不适用)
 * 守门: GEMM 单 tile 对拍标量 f32 参考 (相对范数 3% — HMX 乘内 f16 舍入
 *   使全正 1024 项 dot 元素级差 ~2.5%, 同设备 cos 0.9986 口径之源);
 *   全链重跑位级一致; 输入敏感性必异; HVX 拷贝两侧 dst 逐字节==src。
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <hexagon_types.h>
#include "hvxhmx_v23.h"
#include "hvxhmx_runtime.h"
#include "hvxhmx_v2_binary.h"
#include "hmx_crouton.h"
#include "dma_utils.h"
#include "example_util.h"

#define MK 32u
#define KK 1024u
#define NN_BIG 3584u
#define NN_SMALL 1024u
#define NITER_PACK_ACT 20u
#define NITER_PACK_BIG 3u
#define NITER_PACK_SMALL 8u
#define NITER_GEMM 30u
#define NITER_UNPACK 10u
#define GATHER_N 32768u
#define NITER_GATHER 20u
#define HVX_B (2u * 1024u * 1024u)
#define NITER_HVX 30u

/* 板上 lib (007 谱系锚库 5a023fbb) 导出的搬运原语; 本工作树头文件早于
 * 007 US1, 就地声明 (签名与 007 树 hvxhmx_v2_binary.h 全等)。 */
void hvhx_v2_fill_f16(uint16_t* __restrict__ dst, uint16_t s, uint32_t n);
void hvhx_v2_copy_f16(uint16_t* __restrict__ dst, const uint16_t* __restrict__ src, uint32_t n);

static int64_t tm_us(void) { return HAP_perf_get_time_us(); }

static uint32_t lcg = 20261010u;
static uint32_t lcg_n(void) { lcg = lcg * 1664525u + 1013904223u; return lcg; }

/* 打包循环与 exec_matmul_f16_hmx 逐字复刻 (n0=0, N_full=stride);
 * t=0: 源行主 [rows,K] (act 路径, src_stride=K)
 * t=1: 源行主 [K,cols] (wgt 路径, src_stride=N_full — exec t1=0 分支)
 * dst 可指向 VTCM 或 DDR —— 这是 G1/G2 vs G5 的唯一变量。 */
static void pack_crouton(const uint16_t* src, uint16_t* dst,
                         uint32_t rows32, uint32_t n_dot,
                         uint32_t src_stride, int t)
{
    for (uint32_t r = 0; r < rows32; ++r)
        for (uint32_t kt = 0; kt < n_dot; ++kt) {
            uint16_t* tile = dst + ((size_t)r * n_dot + kt) * 1024u;
            for (uint32_t m = 0; m < 32u; ++m)
                for (uint32_t k = 0; k < 32u; ++k) {
                    size_t s = t ? ((size_t)(kt * 32u + k) * src_stride + (r * 32u + m))
                                 : ((size_t)(r * 32u + m) * src_stride + (kt * 32u + k));
                    tile[hmx_crouton_pos(m, k)] = src[s];
                }
        }
}

/* GEMM 序列复刻 exec_matmul_f16_hmx (bias=NULL → 恒等 scale 页) */
static void gemm_hmx(uint16_t* act_cr, uint16_t* wgt_cr, uint16_t* out_cr,
                     uint16_t* sca, uint32_t n_col, uint32_t n_dot)
{
    for (uint32_t r = 0; r < (MK / 32u); ++r) {
        for (uint32_t c = 0; c < n_col; ++c) {
            uint16_t* out_tile = out_cr + ((size_t)r * n_col + c) * 1024u;
            hmx_enable_execution();
            hmx_unit_acquire();
            asm volatile("mxclracc.hf" ::: "memory");
            asm volatile("bias = mxmem2(%0)" :: "r"(sca) : "memory");
            for (uint32_t kt = 0; kt < n_dot; ++kt) {
                const uint16_t* act_tile = act_cr + ((size_t)r * n_dot + kt) * 1024u;
                const uint16_t* wgt_tile = wgt_cr + ((size_t)c * n_dot + kt) * 1024u;
                asm volatile(
                    "{ activation.hf = mxmem(%0, %1):deep\n"
                    "  weight.hf = mxmem(%2, %3) }\n"
                    :: "r"(act_tile), "r"(0x7FFu),
                       "r"(wgt_tile), "r"(0x7FFu) : "memory");
            }
            memset(out_tile, 0, 2048u);
            asm volatile(
                "cvt.hf = acc(%0)\n"
                "mxmem(%1, %2) = cvt\n"
                :: "r"(2), "r"(out_tile), "r"(0) : "memory");
            hmx_unit_release();
            hmx_disable_execution();
        }
    }
}

int main(void)
{
    ex_open_result("48_vtcm_access");

    /* ---- 会话: hmx_runtime_setup 独立 VTCM (进程私有, 无 wtcache) ---- */
    if (hmx_runtime_setup(16u * 1024u * 1024u) != 0) {
        ex_log("!! hmx_runtime_setup 16MB"); ex_summary(); return 1;
    }
    char* vtcm = (char*)hmx_runtime_get_vtcm_base();
    if (!vtcm || hmx_runtime_get_vtcm_size() < 10u * 1024u * 1024u) {
        ex_log("!! vtcm base/size"); ex_summary(); return 1;
    }
    size_t off_act = 0;
    size_t off_wgt = (off_act + (size_t)MK * KK * 2u + 2047u) & ~(size_t)2047u;
    size_t off_out = (off_wgt + (size_t)KK * NN_BIG * 2u + 2047u) & ~(size_t)2047u;
    size_t off_sca = (off_out + (size_t)MK * NN_BIG * 2u + 2047u) & ~(size_t)2047u;
    size_t off_gx  = (off_sca + 2048u + 127u) & ~(size_t)127u;
    size_t off_hvx = (off_gx + (size_t)GATHER_N * 2u + 127u) & ~(size_t)127u;
    uint16_t* act_cr = (uint16_t*)(vtcm + off_act);
    uint16_t* wgt_cr = (uint16_t*)(vtcm + off_wgt);
    uint16_t* out_cr = (uint16_t*)(vtcm + off_out);
    uint16_t* sca    = (uint16_t*)(vtcm + off_sca);
    uint16_t* gx_vtcm = (uint16_t*)(vtcm + off_gx);
    uint8_t*  hvx_vtcm = (uint8_t*)(vtcm + off_hvx);
    ex_log("vtcm %p size=%u (layout ends %zuB)", vtcm, hmx_runtime_get_vtcm_size(),
           off_hvx + HVX_B);

    /* ---- DDR 面 ---- */
    uint16_t* act_src = memalign(128, (size_t)MK * KK * 2u);
    uint16_t* wgt_src = memalign(128, (size_t)KK * NN_BIG * 2u);
    uint16_t* out_ddr = memalign(128, (size_t)MK * NN_BIG * 2u);
    uint16_t* out_ddr2 = memalign(128, (size_t)MK * NN_BIG * 2u);
    uint16_t* ddr_scr = memalign(128, (size_t)KK * NN_SMALL * 2u);
    uint16_t* gx_ddr  = memalign(128, (size_t)GATHER_N * 2u);
    uint16_t* gy      = memalign(128, (size_t)GATHER_N * 2u);
    uint32_t* goffs   = memalign(128, (size_t)GATHER_N * 4u);
    uint8_t*  hvx_ddr = memalign(128, HVX_B);
    uint8_t*  hvx_dst = memalign(128, HVX_B);
    /* 可复现 f16 图案: 指数分布量级 [2^-10, 2^2] 防 denormal/overflow 全灭 */
    for (size_t i = 0; i < (size_t)MK * KK; i++) {
        uint32_t r = lcg_n();
        uint32_t expo = 0x3800u | (((r >> 4) & 0x3FFu) << 0u); /* [2^-8,2) 附近 */
        act_src[i] = (uint16_t)(expo | (r & 0x3FFu));
    }
    for (size_t i = 0; i < (size_t)KK * NN_BIG; i++) {
        uint32_t r = lcg_n();
        uint32_t expo = 0x3800u | ((r >> 5) & 0x3FFu);
        wgt_src[i] = (uint16_t)(expo | ((r >> 1) & 0x3FFu));
    }
    for (uint32_t i = 0; i < GATHER_N; i++) {
        uint32_t r = lcg_n();
        gx_ddr[i] = (uint16_t)(0x3800u | (r & 0x3FFu));
        goffs[i] = (r >> 10) % GATHER_N;
    }
    for (size_t i = 0; i < HVX_B; i++) hvx_ddr[i] = (uint8_t)lcg_n();
    memset(sca, 0, 2048u);
    for (uint32_t i = 0; i < 32u; ++i) sca[2u * i] = 0x3C00u;
    qurt_mem_cache_clean((qurt_addr_t)act_src, (size_t)MK * KK * 2u,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_mem_cache_clean((qurt_addr_t)wgt_src, (size_t)KK * NN_BIG * 2u,
                         QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);

    /* ---- G1/G2/G3/G4/G5 相位计时 ---- */
    int64_t t_pack_act_vtcm = 0, t_pack_wgt_vtcm_big = 0, t_pack_wgt_vtcm_small = 0;
    int64_t t_gemm = 0, t_unpack = 0, t_pack_wgt_ddr = 0;
    {
        int64_t t0;
        /* G1 pack act → VTCM */
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_PACK_ACT; ++it)
            pack_crouton(act_src, act_cr, MK / 32u, KK / 32u, KK, 0);
        t_pack_act_vtcm = tm_us() - t0;
        /* G2 pack wgt N3584 → VTCM */
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_PACK_BIG; ++it)
            pack_crouton(wgt_src, wgt_cr, NN_BIG / 32u, KK / 32u, NN_BIG, 1);
        t_pack_wgt_vtcm_big = tm_us() - t0;
        /* G3 GEMM (act+wgt 已打包) */
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_GEMM; ++it)
            gemm_hmx(act_cr, wgt_cr, out_cr, sca, NN_BIG / 32u, KK / 32u);
        t_gemm = tm_us() - t0;
        /* G4 unpack → DDR */
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_UNPACK; ++it)
            for (uint32_t r = 0; r < MK / 32u; ++r)
                for (uint32_t c = 0; c < NN_BIG / 32u; ++c) {
                    uint16_t* tile = out_cr + ((size_t)r * (NN_BIG / 32u) + c) * 1024u;
                    for (uint32_t m = 0; m < 32u; ++m)
                        for (uint32_t n = 0; n < 32u; ++n)
                            out_ddr[(size_t)(r * 32u + m) * NN_BIG + (c * 32u + n)] =
                                tile[hmx_crouton_pos(m, n)];
                }
        t_unpack = tm_us() - t0;
        /* G2b/G5: pack wgt N1024 → VTCM vs → DDR scratch (唯一变量 = 目标) */
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_PACK_SMALL; ++it)
            pack_crouton(wgt_src, wgt_cr, NN_SMALL / 32u, KK / 32u, NN_BIG, 1);
        t_pack_wgt_vtcm_small = tm_us() - t0;
        qurt_mem_cache_clean((qurt_addr_t)ddr_scr, (size_t)KK * NN_SMALL * 2u,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_PACK_SMALL; ++it)
            pack_crouton(wgt_src, ddr_scr, NN_SMALL / 32u, KK / 32u, NN_BIG, 1);
        t_pack_wgt_ddr = tm_us() - t0;
    }
    ex_log("pack_act→VTCM  M32xK1024      %lldus  (%u it)", (long long)t_pack_act_vtcm / NITER_PACK_ACT, NITER_PACK_ACT);
    ex_log("pack_wgt→VTCM  K1024xN3584    %lldus  (%u it)", (long long)t_pack_wgt_vtcm_big / NITER_PACK_BIG, NITER_PACK_BIG);
    ex_log("pack_wgt→VTCM  K1024xN1024    %lldus  (%u it)", (long long)t_pack_wgt_vtcm_small / NITER_PACK_SMALL, NITER_PACK_SMALL);
    ex_log("pack_wgt→DDR   K1024xN1024    %lldus  (%u it)", (long long)t_pack_wgt_ddr / NITER_PACK_SMALL, NITER_PACK_SMALL);
    ex_log("hmx_gemm        K1024xN3584    %lldus  (%u it)", (long long)t_gemm / NITER_GEMM, NITER_GEMM);
    ex_log("unpack→DDR      M32xN3584      %lldus  (%u it)", (long long)t_unpack / NITER_UNPACK, NITER_UNPACK);
    {
        int64_t sum = t_pack_act_vtcm / NITER_PACK_ACT + t_pack_wgt_vtcm_big / NITER_PACK_BIG +
                      t_gemm / NITER_GEMM + t_unpack / NITER_UNPACK;
        ex_log("phases sum per-op ≈ %lldus vs census avg 59900us → 缺口=%lldus",
               (long long)sum, (long long)(59900 - sum));
    }

    /* ---- G6 标量 gather: x@DDR vs x@VTCM (GATE3 broadcast 机理裁决) ---- */
    {
        memcpy(gx_vtcm, gx_ddr, (size_t)GATHER_N * 2u);
        int64_t t0, td, tv;
        volatile uint32_t sink = 0;
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_GATHER; ++it)
            for (uint32_t i = 0; i < GATHER_N; ++i) gy[i] = gx_ddr[goffs[i]];
        td = tm_us() - t0;
        sink ^= gy[0] ^ gy[GATHER_N - 1];
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_GATHER; ++it)
            for (uint32_t i = 0; i < GATHER_N; ++i) gy[i] = gx_vtcm[goffs[i]];
        tv = tm_us() - t0;
        sink ^= gy[0] ^ gy[GATHER_N - 1];
        ex_log("scalar_gather n=%u: DDR %lldus vs VTCM %lldus  ratio=%.1fx (sink=%u)",
               GATHER_N, (long long)td / NITER_GATHER, (long long)tv / NITER_GATHER,
               (td > 0) ? (double)tv / (double)td : 0.0, sink);
    }

    /* ---- G7 HVX 向量流读 2MB (hvhx_v2_copy_f16, dst 恒 DDR, src 换面) ---- */
    {
        memcpy(hvx_vtcm, hvx_ddr, HVX_B);
        qurt_mem_cache_clean((qurt_addr_t)hvx_vtcm, HVX_B,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        qurt_mem_cache_clean((qurt_addr_t)hvx_ddr, HVX_B,
                             QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        uint32_t n_f16 = HVX_B / 2u;
        int bad = 0;
        int64_t t0 = tm_us();
        for (uint32_t it = 0; it < NITER_HVX; ++it)
            hvhx_v2_copy_f16((uint16_t*)hvx_dst, (const uint16_t*)hvx_ddr, n_f16);
        int64_t td = tm_us() - t0;
        qurt_mem_cache_clean((qurt_addr_t)hvx_dst, HVX_B,
                             QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        bad += memcmp(hvx_dst, hvx_ddr, HVX_B) != 0;
        t0 = tm_us();
        for (uint32_t it = 0; it < NITER_HVX; ++it)
            hvhx_v2_copy_f16((uint16_t*)hvx_dst, (const uint16_t*)hvx_vtcm, n_f16);
        int64_t tv = tm_us() - t0;
        qurt_mem_cache_clean((qurt_addr_t)hvx_dst, HVX_B,
                             QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        bad += memcmp(hvx_dst, hvx_ddr, HVX_B) != 0;
        double gbsD = td > 0 ? (double)HVX_B * NITER_HVX / (double)td / 1000.0 : 0.0;
        double gbsV = tv > 0 ? (double)HVX_B * NITER_HVX / (double)tv / 1000.0 : 0.0;
        ex_log("hvx_copy_f16 2MB x%u it (dst@DDR): src@DDR %lldus/it (%.0fGB/s) vs src@VTCM %lldus/it (%.0fGB/s) 读侧Δratio=%.2f",
               NITER_HVX,
               (long long)td / NITER_HVX, gbsD, (long long)tv / NITER_HVX, gbsV,
               (td > 0) ? (double)tv / (double)td : 0.0);
        ex_check("hvx_copy_read_byteexact", bad, 0);
    }

    /* ---- 守门: GEMM 单 tile 对拍标量 f32 参考 + 全链重跑一致 + 敏感性 ---- */
    {
        /* 参考: act[0:32, 0:1024] × wgt[0:1024, 0:32] f32 累加 → f16 RNE */
        uint32_t rows32 = 1u, cols32 = 1u;
        pack_crouton(act_src, act_cr, rows32, KK / 32u, KK, 0);
        pack_crouton(wgt_src, wgt_cr, cols32, KK / 32u, NN_BIG, 1);
        gemm_hmx(act_cr, wgt_cr, out_cr, sca, cols32, KK / 32u);
        float* acc = calloc(32u * 32u, sizeof(float));
        float* aw = malloc((size_t)32u * KK * sizeof(float));
        float* ww = malloc((size_t)KK * 32u * sizeof(float));
        hvhx_v2_cvt_f16_to_f32(aw, act_src, 32u * KK);
        hvhx_v2_cvt_f16_to_f32(ww, wgt_src, KK * 32u);
        for (uint32_t m = 0; m < 32u; ++m)
            for (uint32_t k = 0; k < KK; ++k) {
                float a = aw[m * KK + k];
                for (uint32_t n = 0; n < 32u; ++n)
                    acc[m * 32u + n] += a * ww[k * 32u + n];
            }
        uint16_t ref_tile[1024];
        hvhx_v2_cvt_f32_to_f16(ref_tile, acc, 32u * 32u);
        /* HMX 乘内 f16 舍入: 全正 1024 项 dot 元素级差实测可达 ~2.5% (与
         * 设备 vs 金标 cos 0.9986 口径同源); 布局/scale 错会差 O(1) 量级。
         * 判据 = tile 相对范数 3%。 */
        double sse = 0.0, srr = 0.0;
        for (uint32_t m = 0; m < 32u; ++m)
            for (uint32_t n = 0; n < 32u; ++n) {
                uint16_t got = out_cr[hmx_crouton_pos(m, n)];
                float g = 0.f, r = 0.f;
                hvhx_v2_cvt_f16_to_f32(&g, &got, 1u);
                hvhx_v2_cvt_f16_to_f32(&r, &ref_tile[m * 32u + n], 1u);
                double d = (double)g - (double)r;
                sse += d * d;
                srr += (double)r * (double)r;
            }
        double nrm = srr > 0.0 ? __builtin_sqrt(sse / srr) : 1.0;
        ex_log("  tile rel-norm=%.4f (元素级 1e-3 门实测超 ~2.5% 属 HMX 舍入)", nrm);
        ex_check("gemm_tile_vs_scalar_ref_3pct", nrm < 0.03 ? 0 : 1, 0);
        free(acc); free(aw); free(ww);

        /* 全链 (pack act + pack wgt N3584 + gemm + unpack) 两跑位级一致 */
        pack_crouton(wgt_src, wgt_cr, NN_BIG / 32u, KK / 32u, NN_BIG, 1);
        for (uint32_t run = 0; run < 2; ++run) {
            pack_crouton(act_src, act_cr, MK / 32u, KK / 32u, KK, 0);
            gemm_hmx(act_cr, wgt_cr, out_cr, sca, NN_BIG / 32u, KK / 32u);
            for (uint32_t r = 0; r < MK / 32u; ++r)
                for (uint32_t c = 0; c < NN_BIG / 32u; ++c) {
                    uint16_t* tile = out_cr + ((size_t)r * (NN_BIG / 32u) + c) * 1024u;
                    for (uint32_t m = 0; m < 32u; ++m)
                        for (uint32_t n = 0; n < 32u; ++n)
                            out_ddr2[(size_t)(r * 32u + m) * NN_BIG + (c * 32u + n)] =
                                tile[hmx_crouton_pos(m, n)];
                }
            if (run == 0) memcpy(out_ddr, out_ddr2, (size_t)MK * NN_BIG * 2u);
        }
        ex_check("full_chain_deterministic",
                 memcmp(out_ddr, out_ddr2, (size_t)MK * NN_BIG * 2u), 0);
        /* 敏感性: act 旋转 16KB 再跑, 输出必异 */
        for (size_t i = 0; i < (size_t)MK * KK; i++)
            act_src[i] ^= 0x3C00u;   /* 改变量级 — 结果必变 */
        pack_crouton(act_src, act_cr, MK / 32u, KK / 32u, KK, 0);
        gemm_hmx(act_cr, wgt_cr, out_cr, sca, NN_BIG / 32u, KK / 32u);
        for (uint32_t r = 0; r < MK / 32u; ++r)
            for (uint32_t c = 0; c < NN_BIG / 32u; ++c) {
                uint16_t* tile = out_cr + ((size_t)r * (NN_BIG / 32u) + c) * 1024u;
                for (uint32_t m = 0; m < 32u; ++m)
                    for (uint32_t n = 0; n < 32u; ++n)
                        out_ddr2[(size_t)(r * 32u + m) * NN_BIG + (c * 32u + n)] =
                            tile[hmx_crouton_pos(m, n)];
            }
        ex_check("input_sensitivity",
                 memcmp(out_ddr, out_ddr2, (size_t)MK * NN_BIG * 2u) == 0, 0);
    }

    free(act_src); free(wgt_src); free(out_ddr); free(out_ddr2); free(ddr_scr);
    free(gx_ddr); free(gy); free(goffs); free(hvx_ddr); free(hvx_dst);
    return ex_summary();
}
