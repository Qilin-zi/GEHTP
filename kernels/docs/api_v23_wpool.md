# api_v23_wpool — U12 常驻 worker 池 (dd_worker + hmx_lock 交接)

源: `src/runtime/wpool.c` · 头: `include/wpool.h` · 例: `26_wpool` (5 门 PASS)

## API

```c
typedef void (*wpool_job_fn)(void* ud);
int  wpool_open(struct wpool* p, int nworkers, uint32_t stack_bytes);
int  wpool_submit(struct wpool* p, wpool_job_fn fn, void* ud);  /* 队满 -1 */
void wpool_wait_all(struct wpool* p, uint32_t njobs);
void wpool_close(struct wpool* p);          /* 清队 → stop → join */
```

- 互斥轮询队列 (无 condvar/sleep — QuRT 紧睡眠劣化, 同 op82 常驻池教训),
  workers 忙轮 job 槽; `stack_bytes < 64K` 强制 64K (QURT 线程显式栈坑)。
- 计数器: `done` (完成) / `executed` (真跑, 排除 stop 消费), 压力门要求两者相等。

## hmx_lock 交接模板 (模块 C P3 纪律)

**invoke HMX 的线程必须持 hmx_lock。** 主线程 `wtcache_open` 后已持锁 → 串行参考
不可再 lock; 池路径交接:

```
主线程: hmx_unlock() → submit(job) → wait_all → hmx_lock()
job 内: hmx_lock() → memcpy act → fence(FC_CPU,FC_HMX,FM_VTCM)
        → dc_w4_invoke → dc_w4_read_out → hmx_unlock()
```

## 设备门 (26_wpool, 2026-08-16)

- `randarr_24jobs_value_exact`: norm/dot/hvxload 混合 24 job 随机到达值精确
- `hmx_gate_pool_vs_serial_byteexact`: 池 HMX 路径 vs 串行 byte-exact
- `spawn16_value_exact` / `pool_faster_than_spawn`: **spawn-per-op 488us vs 池 110us (4.4×)**
- `stress_5rounds_counters_exact`: 5 轮压力 done=executed=total=76

## 坑

- 重开池后计数归零 — 断言目标要按轮重算。
- G2 串行参考在主线程**不加锁** (已持); 随便加会自锁死。
