/* dc_parts.h — 4C 部件层: VTCM arena / HVX 负载 / DMA 流 / W4A16 引擎封装
 *
 * 本模块不新写任何数值 kernel (plan §5) — W4A16 用 t10 已闭合件原样,
 * norm/dot 是确定性整数元素运算 (自动向量化, bit-exact 可期)。
 */
#ifndef DC_PARTS_H
#define DC_PARTS_H

#include <stdint.h>
#include <stddef.h>
#include "dc_threads.h"

/* ---- VTCM arena (主线程 carve, 指针分发; R-D4) ---- */
struct dc_arena {
    uint8_t* base;
    uint32_t size;
    uint32_t off;
};
void dc_arena_init(struct dc_arena* a, uint8_t* base, uint32_t size);
uint8_t* dc_arena_alloc(struct dc_arena* a, uint32_t bytes, uint32_t align);

/* ---- 文件读入 (DDR staging, 128B 对齐) ---- */
uint8_t* dc_read_file(const char* path, uint32_t* bytes);

/* ---- HVX 负载 (P1/C1): 显式 HVX intrinsic, 4 独立链防 DCE ----
 * scratch: 4KB VTCM (>=32 向量), iters 外层循环。返回折叠校验和。 */
uint32_t dc_hvx_load(uint8_t* scratch4k, uint32_t iters);

/* ---- 确定性整数运算 (C1/C4/M-Gate; 自动向量化, 输出 bit-exact) ---- */
void dc_norm_i16(const int16_t* x, int16_t* y, uint32_t n);  /* clip((x>>2)+64) */
uint64_t dc_dot_u64(const int16_t* a, const int16_t* b, uint32_t n);

/* ---- DMA 流 (P2/C2/C4/M) ----
 * R-D1 对策: submit 恒在 mutex 内、且 poll 到 IDLE 才 submit → dma_desc_submit
 * 只走 dmstart 分支, 绝不 dmlink — 跨线程 g_last_desc 别名不可能发生。
 * desc 每流持久持有 (会话期不 free), g_last_desc 悬空风险为零。
 */
struct dc_dma {
    uint8_t* src;            /* DDR (CPU 写后由本函数清 cache) */
    uint8_t* dst;            /* VTCM 2KB 对齐 */
    uint32_t bytes;
    void* desc;              /* 16B 1D desc (DDR, posix_memalign 16) */
    dc_mutex_t* mu;          /* 共享 submit 锁 */
};
int dc_dma_init(struct dc_dma* d, uint8_t* src, uint8_t* dst, uint32_t bytes,
                dc_mutex_t* mu);
void dc_dma_destroy(struct dc_dma* d);
int dc_dma_once(struct dc_dma* d);          /* 单次 DDR→VTCM (契约同 1-C) */
/* src 一次性清 cache (CPU 写过 DDR 后、首次 DMA 前调一次;
 * 每轮清会把 4MB×N 的 dcacheopma CPU 开销算进带宽 — P2 首跑教训) */
void dc_dma_clean_src(struct dc_dma* d);
void dc_clean_ddr(const void* p, uint32_t bytes);
uint64_t dc_dma_checksum(const struct dc_dma* d); /* dst 校验和 (cache 安全读) */

/* 会话隔离: 引擎 IDLE + 清 g_last_desc (wtcache_dma_fence 同款, 自持实现) */
void dc_dma_fence(void);
/* PROF W-P3: UserDMA 累加器 (每 op 前 reset, 后 get) — dc_dma_once 内累计 */
void dc_dma_reset(void);
void dc_dma_get(int64_t* us, int64_t* bytes);

/* ---- W4A16 引擎 (P3/C2/C3/C4/M) ----
 * 每个 dc_w4 一套独立 VTCM 面 (act/out/wt/bias 表面全 2KB 对齐 — T10 教训),
 * 可被任一线程 invoke; 表重写+FLUSH+kernel 由 w4a16_invoke 承担 (t10 原样)。
 */
struct dc_w4 {
    uint32_t m, k, n;
    uint8_t* act;      /* M*K*2 */
    uint8_t* out;      /* M*N*2 (HMX 写) */
    uint8_t* wt;       /* K*N/2 */
    uint8_t* bias;     /* (N/32)*512 */
    uint8_t* atbl;     /* 8*(K/32)*4 */
    uint8_t* otbl;     /* 8*(N/32)*4 */
    uint8_t* mask;     /* 32 */
    uint8_t* extra;    /* 16 */
    const uint8_t* atbl_ddr;  /* 重写源 (每次 invoke 前 memcpy 进 atbl) */
    const uint8_t* otbl_ddr;
    const uint8_t* scale_ddr; /* 列 scale f16 N*2 (host 参考/设备出面反量化用;
                                 kernel 固定 ÷7 域不消费) */
    /* 分块重绑激活: 同一 act 多块 GEMM (lm_head 61 块) 时, act 量化+crouton
     * 面 + 输出域因子只算一次, 各块 invoke 复用。valid=面已备。 */
    const uint8_t* act_ddr; /* 已绑定的 DDR act 源 (位地址判 repack) */
    float    act_scale; /* 已量化的 a_scale = as·f */
    int      act_valid;
};
/* 从 arena 一性 carve 全部面 (HMX 面 2KB 对齐) */
int dc_w4_carve(struct dc_w4* e, struct dc_arena* a, uint32_t m, uint32_t k,
                uint32_t n, const uint8_t* atbl_ddr, const uint8_t* otbl_ddr);
/* 表回填 + invoke。act/wt/bias 此刻必须已在各自面上 (DMA 或 CPU)。 */
int dc_w4_invoke(struct dc_w4* e);
/* out 面 CPU 读回 (HMX 写绕过 dcache → 先 INVALIDATE) */
void dc_w4_read_out(const struct dc_w4* e, void* recv);
/* 全链: f16 DDR act → 量化(a16 域)+M→256 pad+crouton → kernel →
 * 出面反量化(A_s·S[n]) f16 DDR。carve 须按 pad 后 M (e->m == pad256(m))。
 * out_row_bytes = 输出行跨度字节 (分块写全宽 N_full 时传 N_full*2)。
 * wq_rms = 权重列 RMS 均值 (发射器 scale 槽尾 f16; 自适应输出域因子用)。
 * f_fixed > 0 = 固定域因子 (测试/闭包对拍); 0 = 运行时自适应
 *   f = pow2ceil(4·√k·RMS(a/max|a|)·wq_rms/7) — kernel ±1 输出域限制
 *   (闭包金标自身 14.1% 饱和), f 内部抵消 (dequant 用 a_scale 含 f)。 */
int dc_w4_run(struct dc_w4* e, const uint8_t* act_ddr, uint8_t* out_ddr,
              uint32_t m, uint32_t k, uint32_t n, const uint8_t* scale_ddr,
              uint32_t out_row_bytes, float wq_rms, float f_fixed);

/* lm_head N 分块优化: 复用 act 量化+crouton 面.
 * 调用序列:
 *   1. dc_w4_run_prep(e, act_src, M, K, wq_rms, f_fixed) — 量化一次
 *   2. for each chunk c0: dc_w4_run_invoke(e, scale+c0*2, nc,
 *        out_ddr+c0*2, N*2, M, wq_rms, f_fixed) — 只 kernel+dequant
 *   3. dc_w4_run_fini(e) — 释放重绑状态 (下次 prep 重新量化)
 * m_out = 真实行数 M (e->m 是 pad256(M), 反量化只用前 M 行);
 * out_ddr 由调用方给到本块列偏移。复用期内 e->n 须由 invoke 临时改回
 * (dc_w4_invoke 按 e->n refill 表; carve 宽 n_eff ≥ 块宽 nc)。
 * 数值契约与逐块 dc_w4_run 完全等价 (同 a_scale、同 32768 pad、同
 * w4a16_dequant_crouton); 仅消除 61 次重复的 act 全面 pass。 */
int dc_w4_run_prep(struct dc_w4* e, const uint8_t* act_src, uint32_t m,
                     uint32_t k, float wq_rms, float f_fixed);
int dc_w4_run_invoke(struct dc_w4* e, const uint8_t* scale, uint32_t n,
                       uint8_t* out_ddr, uint32_t out_row_bytes,
                       uint32_t m_out, float wq_rms, float f_fixed);
void dc_w4_run_fini(struct dc_w4* e);

#endif
