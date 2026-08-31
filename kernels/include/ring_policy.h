/* ring_policy.h — U20 通用预取环策略库 (GENERIC_A §4.2)
 *
 * 生产侧 ring_enqueue 登记组 (FIFO); 消费侧 ring_need 声明即将 P_bulk,
 * READY 拿槽地址直读; ring_release 回收。任何 NOT_READY/DEGRADED 消费者
 * 都能走直读 DDR 的降级路径 —— 环是加速器不是正确性依赖。
 *
 * 实现纪律 (build52 canonical submit, §6.4):
 *   1. 锁内绝不触发引擎操作 (门铃只在锁外经串行化器);
 *   2. claim (填描述符) 与 fire (发门铃) 分离; 等待路径重驱动 fire;
 *   3. 跨条目禁止明链: 条目内 span 预链接 (整 span 一次门铃),
 *      条目间接续只靠门铃;
 *   4. re-kick 仅经 dmwait 确认真空闲后 dmstart (幂等);
 *   5. claim 深度门 n_qsub < n_fired + FIRE_DEPTH ∧ n_qsub - n_rel < N。
 */
#ifndef HVXHMX_V23_RING_POLICY_H
#define HVXHMX_V23_RING_POLICY_H

#include "engine_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RING_OK        0
#define RING_E_ARG    -1
#define RING_E_RANGE  -2
#define RING_E_NOMEM  -3
#define RING_E_FATAL  -4   /* 引擎断言层已冻结 (策略层立即 DEGRADED) */

typedef enum {
    RING_READY = 0,        /* 组就位, 可直读槽 */
    RING_NOT_READY,        /* 超时; 走降级直读 DDR */
    RING_DEGRADED          /* 环不可用, 后续全部降级 */
} ring_need_result;

#define RING_MAX_GROUPS      4096u
#define RING_MAX_DESCS_SLOT  64u
#define RING_FIRE_DEPTH      3u     /* claim 深度窗 (I3/槽安全联合) */

typedef struct ring_stats {
    uint64_t n_doorbells, n_groups, n_degrades, n_watchdogs;
    uint64_t n_rekicks;
    double   stall_us_total;       /* need 等待累计 (虚拟时钟) */
    uint64_t bytes_prefetched;
} ring_stats;

typedef struct ring_policy ring_policy;

ring_policy *ring_create(const engine_adapter *eng,
                         unsigned n_slots,
                         unsigned max_descs_per_slot,
                         uint32_t slot_bytes,
                         uint8_t *slots_base /* 调用方提供 N×slot_bytes */);

void ring_destroy(ring_policy *r);

/* 生产侧: 登记组 → 内部驱动 claim+fire。组号 = FIFO 序 (0 起)。 */
int ring_enqueue(ring_policy *r, uint64_t src, uint32_t bytes,
                 uint32_t nrows, uint32_t row_size, uint32_t src_stride);

/* 消费侧: 等组 g READY (最多 deadline_us 虚拟时间)。READY 时 *slot_out=
 * 目的槽地址。内部自旋驱动 fire/done 轮询 (等待即驱动, I7)。 */
ring_need_result ring_need(ring_policy *r, uint32_t g,
                           uint64_t deadline_us, uint64_t *slot_out);

/* 消费侧: 组用完 (幂等)。 */
void ring_release(ring_policy *r, uint32_t g);

/* P_serial 窗口通知: est_us 内无人消费 → 提升提交深度到满环。 */
void ring_on_serial_phase(ring_policy *r, uint32_t est_us);

/* 重量级: 等 [0..g] 全 READY 并 release (消息边界)。 */
int ring_drain_to(ring_policy *r, uint32_t g, uint64_t watchdog_us);

void ring_get_stats(const ring_policy *r, ring_stats *out);

/* 不变量检查 (GENERIC_A §6.2 I1-I8 的可运行子集; 返回违规编号 0=ok)。
 * I1 游标单调序  n_rel≤n_ready≤n_fired≤n_qsub≤n_enq
 * I2 槽占用      n_qsub-n_rel ≤ N_SLOTS
 * I5 链尾合法    fire_tail ∈ 池 (L5 的策略侧影子) */
int ring_check_invariants(const ring_policy *r);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_RING_POLICY_H */
