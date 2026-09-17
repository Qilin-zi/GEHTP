/* wpool.h — U12 常驻工人池 (dc_threads 原语之上, op82 结论的 DSP 内形态)
 *
 * 动机 (DFlashSpecCppRunner op82 常驻 worker 池): spawn-per-op 在短 job 下
 * 被 90-336x 的线程创建开销淹没; 常驻池一次 spawn, job 到达即派发。
 *
 * 纪律:
 *   - worker 显式栈 (dc_spawn 内置, 64KB 起自 M0 坑)
 *   - 队列互斥保护; 无 condvar → mutex 轮询 (短测试面, 不 sleep)
 *   - hmx 门 (模块 C P3): invoke HMX 的线程必须持有 hmx_lock —
 *     主线程 wpool_open 前 wtcache_hmx_unlock 交接, job 内 batch lock,
 *     wpool_close 后主线程 re-lock
 */
#ifndef HVXHMX_V23_WPOOL_H
#define HVXHMX_V23_WPOOL_H

#include <stdint.h>
#include "dc_threads.h"

#define WPOOL_MAX_WORKERS 4
#define WPOOL_QUEUE_DEPTH 64

typedef void (*wpool_job_fn)(void* ud);

struct wpool {
    dc_thread_t th[WPOOL_MAX_WORKERS];
    int         nworkers;
    int         spawned;
    dc_mutex_t  mx;
    wpool_job_fn qfn[WPOOL_QUEUE_DEPTH];
    void*       qud[WPOOL_QUEUE_DEPTH];
    uint32_t    qh, qt, qcount;
    uint32_t    done;        /* 已完成 job 数 (mx 保护) */
    uint32_t    executed;    /* 实际弹出执行数 */
    uint32_t    stop;
};

/* nworkers ≤ WPOOL_MAX_WORKERS; 返回 0 成功 */
int  wpool_open(struct wpool* p, int nworkers, uint32_t stack_bytes);
/* 投递 job (队满返回 -1, 调用方决定等待) */
int  wpool_submit(struct wpool* p, wpool_job_fn fn, void* ud);
/* 自旋等 done ≥ njobs */
void wpool_wait_all(struct wpool* p, uint32_t njobs);
/* 停止 + join 全部 worker (完成后必须调) */
void wpool_close(struct wpool* p);

#endif
