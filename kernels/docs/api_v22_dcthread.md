# U3 dcthread — QURT 线程与同步原语 功能手册

声明: [`include/dc_threads.h`](../include/dc_threads.h) · 源: `src/v22/dc_threads.c`,
`src/v22/dc_sync.c` (源模块 dualcore_v81 · 4C 闭合) · 测试: [examples/22_dualcore_threads](../examples/22_dualcore_threads/main.c)

## 功能

CDSP 进程内多线程底座, 吸收 M0 四坑后的安全封装:

| M0 坑 | 对策 |
|-------|------|
| QURT 线程必须显式栈 | `dc_spawn` 内置 set_stack_addr (64KB 起) |
| `HAP_compute_res_*` 只主线程一次 | 工作线程只拿裸指针 (VTCM 由主线程 wtcache_open 分好) |
| FARF 不达 logcat | 结论走 result 文件 |
| handle_invoke 3 参 | 本单元纯 DSP 内多线程, 不涉及 |

## API

### 线程

```c
int  dc_spawn(dc_thread_t* t, const char* name, void (*fn)(void*), void* arg,
              uint32_t stack_bytes);
void dc_join(dc_thread_t* t);
```
stack `memalign(4096)` 内置, join 释放。

### 同步 (= qurt 原语薄别名)

```c
typedef qurt_barrier_t dc_barrier_t;   /* dc_barrier_init(b, n); dc_barrier_wait(b); */
typedef qurt_mutex_t   dc_mutex_t;     /* dc_mutex_init/lock/unlock (pimutex) */
void dc_flush(void* p, uint32_t bytes);        /* FLUSH   dcache */
void dc_invalidate(void* p, uint32_t bytes);   /* INVALIDATE dcache */
```

### VTCM 自旋旗标

```c
void     dc_flag_set(volatile uint32_t* f, uint32_t v);   /* 写 + FLUSH */
uint32_t dc_flag_wait(volatile uint32_t* f, uint32_t v);  /* INVALIDATE 轮询到 ==v */
uint32_t dc_flag_wait_ge(volatile uint32_t* f, uint32_t v); /* 轮询到 >=v */
```

**契约 (P4 四象限实测)**: 本芯片 VTCM CPU 读写跨线程天然一致 (100/100) — 纯 volatile
即可, set/wait 里的 FLUSH/INVALIDATE 是保守开销。**流水线必须用 `wait_ge`**: 消费者可
领先任意步, `wait` 等值会永久挂死 (C4 实测)。

## 使用范例 (example 22 摘)

```c
dc_mutex_t mu;   dc_mutex_init(&mu);          /* 共享 DMA submit 锁 */
dc_barrier_t b;  dc_barrier_init(&b, 2);
volatile uint32_t* flag = (volatile uint32_t*)dc_arena_alloc(&ar, 4, 128);  *flag = 0;

dc_thread_t t0, t1;
dc_spawn(&t0, "w0", worker, &wa[0], 64*1024);
dc_spawn(&t1, "w1", worker, &wa[1], 64*1024);
/* worker 内: barrier_wait → flag_wait(go,1) → dc_dma_once(&dma) [共享 mu]
 *            → dc_w4_invoke(e) → 旗标握手 flag: 0→1→2 */
dc_join(&t0);  dc_join(&t1);
```

## 并发模型结论 (4C 设备实测, 写进调度决策)

- **单 DMA 引擎**: 跨线程 submit 必须同一把 mutex (R-D1)。
- **HMX 锁不缩放**: 两线程两引擎并发 wall ≈ 串行 (数值 byte-exact 无损);
  吞吐扩展走 U7 双域 (两 CDSP 进程, 2.001×)。
- **双 CDSP 零退化**: 同一二进制 dom3/dom4 各起一进程互不干扰 (example 20)。
