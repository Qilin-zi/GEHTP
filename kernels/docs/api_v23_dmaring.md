# U20 dmaring — DMA 预取环 (GENERIC_A 问题书实现)

> 源: `docs(外部)/GENERIC_A_DMA_PREFETCH_RING_PROBLEM.md`
> 单元: `include/engine_adapter.h` + `include/ring_policy.h`
> 实现: `src/runtime/ring_sim.c` (引擎 FSM 仿真 + Law 断言层) +
> `src/runtime/ring_policy.c` (通用预取环策略)
> 例: `examples/34_dmaring` (26 门, 设备 52f67807 全绿)

## 1. 问题与解法骨架

相位交替消费者: P_bulk 起步即要数据 / P_serial 毫秒级不碰引擎。
目标 = P_serial 窗口预填 staging, 消费者不停顿;
硬约束 = 门铃纪律 Law1-8 (违反 = 域崩溃)。

```
生产侧 ring_enqueue ──► claim (锁内填描述符, span 预链) ──► fire (锁外门铃)
                                   │                            │ IDLE → dmwait+dmstart (L8/L1)
                                   │                            │ RUN  → dmlink 仅 slack≥2 (L7)
消费侧 ring_need ◄── advance_ready (done 前缀推进) ◄── 引擎 FSM (惰性结算)
```

## 2. engine_adapter.h — 引擎触碰收敛面

策略库对引擎的全部触碰 = 4 操作 + 描述符池 + spin/clock 钩子
(仿真 = 虚拟时钟; 器件 = qtimer/真延时, 双跑靠替换 adapter):

| 钩子 | 语义 | Law |
|---|---|---|
| `dmstart(d)` | 空闲起步, 全链 walk | L1/L4/L5/L6/L8 |
| `dmlink(cur,nx)` | RUN 续链 / IDLE 唤醒 | L2/L3/L4/L7 |
| `dmpoll()` | **RETIRE 窗内说谎返回 IDLE** (仿真灵魂, 不许修) | — |
| `dmwait()` | 阻塞至真空闲, 唯一真理来源 | L8 |
| `pool_alloc/free` | 描述符池 (8B 对齐) | L5 |
| `spin(us)/clock_us()` | 等待节拍 / 单调时钟 | — |

`sim_params` (附录 A 默认): bw_eng=55GB/s, t_desc=0.5µs, w_retire=2µs,
w_accept=1µs, done_to_retire=0.3µs, t_doorbell=2µs (软件成本计入虚拟时钟,
天然满足 L4 间隔)。

## 3. ring_policy.h — API

```c
ring_policy *ring_create(const engine_adapter *, n_slots, max_descs_per_slot,
                         slot_bytes, slots_base);   /* 调用方给 N×slot_bytes */
int   ring_enqueue(r, src, bytes, nrows, row_size, src_stride);  /* 组号=FIFO */
ring_need_result ring_need(r, g, deadline_us, &slot_out);
        /* READY=槽就位直读 / NOT_READY=超时走直读 DDR 降级 / DEGRADED=环不可用 */
void  ring_release(r, g);                    /* 幂等; RELEASE 先于 CLAIM */
void  ring_on_serial_phase(r, est_us);       /* 空窗通知 → 提交深度提升到满环 */
int   ring_drain_to(r, g, watchdog_us);      /* 消息边界重量级等待 */
int   ring_check_invariants(r);              /* I1 游标序 / I2 槽占用 / I5 fire_tail∈池 */
```

## 4. canonical submit 纪律 (wt-bp-ring build52 器件血泪)

1. **锁内绝不触发引擎操作** (门铃只在锁外);
2. claim 与 fire 分离; 所有等待路径重驱动 fire → 无死锁;
3. **跨条目禁止明链**: 条目内 span 预链接 (整 span 一次门铃),
   条目间接续只靠门铃; fire_tail = 最后已提交 span 尾;
4. re-kick 仅经 dmwait 确认真空闲后 dmstart (幂等, F2 孤儿救援);
5. claim 深度门 `n_qsub < n_fired + FIRE_DEPTH ∧ n_qsub − n_rel < N_SLOTS`。

## 5. 仿真器要点 (ring_sim.c)

- 惰性事件结算: 引擎不主动跑, poll/wait/advance 按当前时刻结算完成
  (done 位 = memcpy 落槽时刻; F4 注入可撕开两者);
- RETIRE 窗 = [链尾done+delay, +w_retire): dmpoll 谎报 IDLE;
- Law 断言: L1/L4/L5/L6/L7/L8 违规 → FATAL 冻结 (rep.law≠0, 后续门铃全 -1);
- 故障注入: `SIM_FAULT_F2_DOORBELL_LOST` (dmstart/dmlink 吞唤醒一次 →
  孤儿 → 超时 re-kick 救回), `SIM_FAULT_F4_EARLY_DONE` (done 提前假置,
  数据未拷 → 消费者 memcmp 检出)。

## 6. 例 34 门 (设备全绿 26/26)

| 门 | 内容 | 关键数字 |
|---|---|---|
| G1 | 单组: READY + 1 门铃 + bit-exact | 门铃=1 |
| G2 | 环回绕×100, I1/I2/I5 每步零违例 | 0 FATAL |
| G3 | 满 serial 预填 → bulk 零停顿 | stall<1µs |
| G4 | F2 孤儿 → re-kick 救回; G4c F4 假 done 检出 | n_rekicks≥1 |
| G5 | 断言自证: 合法零 FATAL + RETIRE 窗 dmstart 报 law6 | dmpoll 谎报 IDLE |
| G6 | 轨迹 T_eff vs oracle 上界 ≥0.95 | serial 1.000 / burst 0.999 |
| G7 | 参数扫描 BW40-70 × W_RETIRE 2/20 × N 2/4/8 | 18 组零 FATAL |
| G8 | 门铃经济性 (V7) | 门铃/组=1.000 |

## 7. 已知边界

- 仿真传输时间 = `bytes / (bw_eng×1000)` µs (bw 按 B/ns 口径);
- ring_need 的 deadline 为虚拟时间 (器件 adapter 下为真实 µs);
- re-kick 成功后 deadline 宽限一周期 (一次性, 防饿死也防死循环)。
