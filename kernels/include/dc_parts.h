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
};
/* 从 arena 一性 carve 全部面 (HMX 面 2KB 对齐) */
int dc_w4_carve(struct dc_w4* e, struct dc_arena* a, uint32_t m, uint32_t k,
                uint32_t n, const uint8_t* atbl_ddr, const uint8_t* otbl_ddr);
/* 表回填 + invoke。act/wt/bias 此刻必须已在各自面上 (DMA 或 CPU)。 */
int dc_w4_invoke(struct dc_w4* e);
/* out 面 CPU 读回 (HMX 写绕过 dcache → 先 INVALIDATE) */
void dc_w4_read_out(const struct dc_w4* e, void* recv);

#endif
