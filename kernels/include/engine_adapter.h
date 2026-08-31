/* engine_adapter.h — U20 引擎适配层: 所有门铃的唯一合法入口面
 *
 * 问题 GENERIC_A §4.1 (docs/GENERIC_A_DMA_PREFETCH_RING_PROBLEM.md):
 * 策略库对引擎的全部触碰收敛为 4 个操作 + 描述符池; 仿真/真实器件
 * 双跑靠替换 adapter 实现。
 *
 * 引擎状态机 (§2.2, 仿真必须建模):
 *   IDLE ⇄ RUN → PARK_RETIRE(软件不可见) → IDLE
 *   dmpoll 在 RETIRE 窗内说谎返回 IDLE; dmwait 是唯一真理来源。
 *
 * 门铃定律 Law1-8 摘要 (完整版见问题书 §3):
 *   L1 dmstart 仅真空闲, 全链 walk
 *   L2 parked-tail dmlink 只执行一条
 *   L3 RUN 态 dmlink 延长 walk
 *   L4 两次门铃间隔 ≥ W_ACCEPT (dmwait 确认真空闲除外)
 *   L5 cur 必须 ∈ 描述符池且对齐
 *   L6 RETIRE 窗内任何门铃 = 损坏
 *   L7 dmlink 需 slack≥2 (tail 与前驱都未 done)
 *   L8 IDLE 分支必先 dmwait 再 dmstart
 *
 * 器件血泪参照 (wt-bp-ring build52 canonical submit, run59/61/62/71):
 *   claim/fire 分离 + 每 entry 预链接 span + 跨 entry 永不明链;
 *   fire_tail = 最后已提交 span 尾; parked 尾永不 dmlink。
 */
#ifndef HVXHMX_V23_ENGINE_ADAPTER_H
#define HVXHMX_V23_ENGINE_ADAPTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ENG_STATUS_IDLE 0u
#define ENG_STATUS_RUN  1u
#define ENG_STATUS_ERR  2u

typedef struct eng_desc {
    struct eng_desc *next;      /* 链指针; 链尾必须 NULL */
    volatile uint32_t done;     /* 引擎写 1 表示完成 */
    /* ---- 拷贝语义字段, 策略库不解释只透传 ---- */
    uint64_t src, dst;
    uint32_t bytes;
    uint32_t nrows, row_size, src_stride, dst_stride;
    uint32_t flags;
} eng_desc;

typedef struct engine_adapter {
    int      (*dmstart)(eng_desc *d);
    int      (*dmlink)(eng_desc *cur, eng_desc *nx);
    uint32_t (*dmpoll)(void);
    void     (*dmwait)(void);
    eng_desc *(*pool_alloc)(unsigned n_slots, unsigned descs_per_slot);
    void     (*pool_free)(eng_desc *pool);
    /* 自旋/延时钩子: 策略库的等待节拍 (仿真=虚拟时钟推进; 器件=真延时) */
    void     (*spin)(uint32_t us);
    /* 单调 µs 时钟 (仿真=虚拟时钟; 器件=qtimer) */
    uint64_t (*clock_us)(void);
} engine_adapter;

/* ---- 仿真 adapter (ring_sim.c): 轨迹驱动 + Law 断言 + 故障注入 ---- */

/* 仿真参数 (µs / GB/s, 附录 A 默认值) */
typedef struct {
    double   bw_eng;              /* 55 GB/s */
    double   t_desc;              /* 0.5 µs/描述符 */
    double   w_retire;            /* 2 µs park-retire 窗 */
    double   w_accept;            /* 1 µs 门铃接受窗 */
    double   done_to_retire;      /* 0.3 µs done 可见→retire 开始 */
    double   t_doorbell;          /* 2 µs 门铃软件成本 (计入虚拟时钟) */
} sim_params;

/* 故障注入 (GENERIC_A §7.1, 0=off) */
#define SIM_FAULT_F2_DOORBELL_LOST 0x1u   /* dmlink 吞唤醒一次 */
#define SIM_FAULT_F4_EARLY_DONE    0x2u   /* done 提前假置 (数据未真到位) */

const sim_params *sim_default_params(void);

/* 建立/销毁仿真 adapter (含断言层)。out_fatal 非空时: 违反 Law 的操作
 * 记录定律号 + 时刻, 仿真立即冻结 (引擎损坏不可恢复), 后续门铃全部
 * 返回 -1。harness 用它证明断言有效 (F1 门)。 */
typedef struct {
    int      law;          /* 违反的定律号 1-8; 0=无 */
    double   t_us;         /* 虚拟时刻 */
    char     op[16];       /* dmstart/dmlink */
    uint64_t n_doorbells;  /* 累计门铃数 (dmstart+唤醒) */
    uint64_t n_bytes;      /* 累计搬运字节 */
    double   t_busy_us;    /* RUN+RETIRE 累计 (引擎占用) */
} sim_report;

int  sim_adapter_init(const sim_params *p, engine_adapter *out);
void sim_adapter_report(sim_report *rep);
void sim_adapter_fault(uint32_t mask);      /* 置位注入 (bit 清除) */
/* 轨迹驱动: 虚拟时钟推进 us 微秒 (处理途中引擎事件)。返回推进后时钟。 */
double sim_advance(double us);
double sim_now(void);

#ifdef __cplusplus
}
#endif
#endif /* HVXHMX_V23_ENGINE_ADAPTER_H */
