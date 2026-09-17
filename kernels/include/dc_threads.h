/* dc_threads.h — 线程底座 (plan §3, 吸收 M0 四坑)
 *
 * M0 已知坑的对策:
 *   - QURT 线程必须显式栈 (set_stack_addr+size), 64KB 起 — dc_spawn 内置
 *   - HAP_compute_res_* 只在主线程做一次 (R-D3) — 工作线程只拿裸指针
 *   - FARF 不达 logcat (R-D5) — 结论全走 result.jsonl
 *   - handle_invoke 3 参 — 本模块不涉及 (纯 DSP 内多线程)
 */
#ifndef DC_THREADS_H
#define DC_THREADS_H

#include <stdint.h>
#include <stddef.h>
#include <qurt_thread.h>
#include <qurt_barrier.h>
#include <qurt_mutex.h>

typedef struct {
    qurt_thread_t tid;
    void*         stack;      /* memalign(4096), dc_join 释放 */
    uint32_t      stack_bytes;
} dc_thread_t;

int  dc_spawn(dc_thread_t* t, const char* name, void (*fn)(void*), void* arg,
              uint32_t stack_bytes);
void dc_join(dc_thread_t* t);

/* ---- 同步原语 (dc_sync.c) ---- */
typedef qurt_barrier_t dc_barrier_t;
typedef qurt_mutex_t   dc_mutex_t;

void dc_barrier_init(dc_barrier_t* b, int n);
void dc_barrier_wait(dc_barrier_t* b);
void dc_mutex_init(dc_mutex_t* m);
void dc_mutex_lock(dc_mutex_t* m);
void dc_mutex_unlock(dc_mutex_t* m);

/* VTCM 自旋旗标 (跨 NSP 通信最廉价原语, P4 定其 cache 契约) */
void dc_flag_set(volatile uint32_t* f, uint32_t v);      /* 写 + FLUSH */
uint32_t dc_flag_wait(volatile uint32_t* f, uint32_t v); /* INVALIDATE 轮询, 返回观测值 */
uint32_t dc_flag_wait_ge(volatile uint32_t* f, uint32_t v); /* 流水线 >= 等待 (防消费者越位挂死) */

/* cache 操作直达 (P4 四象限用) */
void dc_flush(void* p, uint32_t bytes);
void dc_invalidate(void* p, uint32_t bytes);

#endif
