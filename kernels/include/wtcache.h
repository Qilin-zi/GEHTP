/* wtcache.h — 模块 1-C: 权重 cache/pin 公共接口
 *
 * 三件独立闭合的事 (MODULE_1C_WEIGHT_CACHE_PIN_CLOSURE_PLAN.md):
 *   1-C-α  wtcache_pin_weight     热权重一次 DMA 进 VTCM, 整会话不搬出 (tcm_migration 等价)
 *   1-C-β  wtcache_ring_*         prefetch/move_back 环, depth=4+4 (svf0_dma_cfg 等价)
 *   1-C-γ  wtcache_warm_file      page cache warm (host 侧 posix_fadvise)
 *
 * 物理基础 (vtcm_dma_bench REPORT 实测, 52f67807):
 *   - 16MB VTCM, base 0xFF000000, 2KB 对齐
 *   - UserDMA 描述符: DDR→VTCM 65.5 GB/s, VTCM→DDR 69.9 GB/s
 *   - HMX mxmem 只访 VTCM (DDR 直访 fault); VTCM→HMX ~1 TB/s
 *   - DMA 与 HMX 是独立硬件单元 → 可真重叠 (§2.5 null 是 HVX-copy+HVX-compute 冲突)
 *
 * DMA 引擎模型 (dma_utils.h):
 *   - 单 UserDMA 引擎; dma_desc_submit 菊花链 N 个 desc, 可在途追加 (g_last_desc link)
 *   - dma_desc_is_done(desc) 逐符非阻塞查询; dma_wait_for_idle() 全阻塞
 *   - 重叠靠 depth≥2 在途窗口 + 不每 submit 立刻 wait, 不是靠"两通道"
 */
#ifndef WTCACHE_H
#define WTCACHE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 错误码 ---- */
#define WTC_OK                 0
#define WTC_ERR_POWER         -1   /* HVX/HMX power on 失败 */
#define WTC_ERR_VTCM_ACQUIRE  -2   /* HAP_compute_res_acquire 返回 0 */
#define WTC_ERR_VTCM_PTR      -3   /* get_vtcm_ptr NULL */
#define WTC_ERR_HMX_LOCK      -4   /* hmx_lock 失败 */
#define WTC_ERR_OOM           -5   /* VTCM 偏移超出 16MB */
#define WTC_ERR_DMA           -6   /* dma_desc_init/submit 失败 */
#define WTC_ERR_VERIFY        -7   /* pin verify 不 bit-exact */
#define WTC_ERR_RANGE         -8   /* 参数越界 */
#define WTC_ERR_NOT_READY     -9   /* ring_next 但槽未就绪 */

/* ============================================================
 * 会话与 VTCM 池
 * ============================================================ */
struct wtcache_ctx;
struct wtcache_ring;

/* 16MB VTCM 布局 (运行时从 base 起 128B 对齐顺序切):
 *   [pin_pool]  pin_cap_bytes (默认 12MB) — 1-C-α 热权重常驻区
 *   [ring_in]   depth_prefetch × tile_bytes — 1-C-β prefetch 槽
 *   [ring_out]  depth_moveback × tile_bytes — 1-C-β move_back 槽
 *   [temp]      剩余 — 临时分配/干扰模拟 (T1/T6 用)
 * acquire VTCM + power on HVX/HMX + hmx_lock。
 * pin_cap_bytes=0 → 默认 12MB。 */
int  wtcache_open(struct wtcache_ctx** out, uint32_t pin_cap_bytes);
int  wtcache_close(struct wtcache_ctx* ctx);
/* 4C 假设B实验: hmx_lock 可能是"持有线程"属性 — 允许主线程 unlock 后
 * 由工作线程重新 lock, 以测试 HMX 指令是否只对持锁线程放行。 */
int  wtcache_hmx_lock(struct wtcache_ctx* ctx);
int  wtcache_hmx_unlock(struct wtcache_ctx* ctx);

/* 诊断: 返回 VTCM base/总大小/pin 区起止, 供 host 验证布局 (T6) */
void wtcache_layout(const struct wtcache_ctx* ctx,
                    void** vtcm_base, uint32_t* vtcm_size,
                    void** pin_base, uint32_t* pin_cap);

/* ============================================================
 * 1-C-α: 权重 pin (tcm_migration 等价)
 * ============================================================ */
/* 把 ddr_src (bytes 字节, align 对齐) 一次 DMA 进 pin 区固定槽, 返回 VTCM 指针。
 * vtcm_out 跨 decode 步稳定 (同一指针), 内容不被 ring/temp 破坏。
 * align: 0=默认 128B; 否则按 align 对齐 (≤128 且 2 的幂)。
 * 内部: DDR src FLUSH_INVALIDATE → DMA → VTCM dst INVALIDATE。 */
int  wtcache_pin_weight(struct wtcache_ctx* ctx,
                        const void* ddr_src, size_t bytes, int align,
                        void** vtcm_out);

/* 完整性自检: pin 区内容 vs DDR 源 bit-exact。T1 用。
 * 返回 WTC_OK=全一致, WTC_ERR_VERIFY+mismatch 首字节的偏移编码在 *first_bad_off。 */
int  wtcache_pin_verify(struct wtcache_ctx* ctx, const void* ddr_src,
                        const void* vtcm, size_t bytes,
                        uint32_t* first_bad_off);

/* ============================================================
 * 1-C-β: prefetch/move_back 环 (svf0 4+4 等价)
 * ============================================================ */
/* 初始化环: depth_prefetch 个 DDR→VTCM 槽 + depth_moveback 个 VTCM→DDR 槽。
 * tile_bytes 必须是 128 倍数 (DMA/cache 对齐)。默认 depth 4,4。
 * 槽从 VTCM pin 区之后顺序切。 */
int  wtcache_ring_init(struct wtcache_ctx* ctx, struct wtcache_ring** r,
                       size_t tile_bytes, int depth_prefetch, int depth_moveback);

/* 预填: 把 n 个 ddr tile 首地址登记进环 (prime 阶段批量 prefetch)。
 * n ≤ depth_prefetch。触发 n 个 DMA, 等全完成 (prime 不追求重叠)。 */
int  wtcache_ring_prime(struct wtcache_ring* r, const void* ddr_tiles[], int n);

/* 切换 prefetch 提交模式 (T5 重叠门用):
 *   enable=1 → 非阻塞菊花链 (submit_chained), depth≥2 在途, HMX compute 与 DMA 真重叠;
 *              会话内禁 move_back (ring_next 的 ddr_out_target 传 NULL)。
 *   enable=0 → 阻塞 submit_one (默认), 每 submit 排空引擎, 与 move_back 安全共存。
 * 切入 overlap 前 fence (清 g_last_desc + 排空引擎)。 */
void wtcache_ring_set_overlap(struct wtcache_ring* r, int enable);

/* 主循环原语 (稳态):
 *   1. 等 cur_prefetch 槽 DMA 完成 (dma_desc_is_done 轮询, 不全局 wait)
 *   2. 后台提交下一个 ddr tile 的 prefetch (保持 depth 在途)
 *   3. 若有已消费的输出槽, 提交 move_back (VTCM→DDR) 并回收
 *   返回: cur_vtcm_in = 给 HMX 读的当前 tile VTCM 地址
 *         cur_vtcm_out = 给 HMX 写输出的 VTCM 地址 (下轮 move_back 回 DDR)
 *   ddr_next: 下一 tile 的 DDR 源 (NULL = 无更多 prefetch, 仅排空)
 *   ddr_out_target: 当前输出要回搬的 DDR 目标 (NULL = 这轮不 move_back)
 *   invalidate: 1=读前 INVALIDATE 输入槽 (T3 一致性开关); 0=跳过 (复现 bug) */
int  wtcache_ring_next(struct wtcache_ring* r,
                       const void* ddr_next, void* ddr_out_target,
                       int invalidate,
                       void** cur_vtcm_in, void** cur_vtcm_out);

/* 收尾: 回搬所有残留输出, 等全部 DMA idle。T2/T4 用。 */
int  wtcache_ring_drain(struct wtcache_ring* r);

/* 释放 ring (desc 池 + 结构). ctx close 前可显式调; 单次跑可省略. */
void wtcache_ring_destroy(struct wtcache_ring* r);

/* 诊断: 当前在途 prefetch 数 / move_back 数 / 峰值 VTCM 占用 (T7) */
void wtcache_ring_stats(const struct wtcache_ring* r,
                        int* inflight_prefetch, int* inflight_moveback,
                        uint32_t* peak_vtcm_bytes);

/* ============================================================
 * 1-C-γ: page cache warm (host 侧, 便宜 bonus)
 * ============================================================ */
/* posix_fadvise(POSIX_FADV_WILLNEED) 或 readahead。host 进程调用。 */
int  wtcache_warm_file(const char* path);

#ifdef __cplusplus
}
#endif
#endif /* WTCACHE_H */
