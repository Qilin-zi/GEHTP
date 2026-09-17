# api_v23_graphstep — U16 整步执行 (oplist_exec 演进)

源: `src/runtime/oplist_exec.c` (V2.2 演进, 无新文件) · 头: `include/oplist_exec.h`
· 例: `30_graph_step` (8 门 PASS)

## V2.2 → V2.3 变化

blob 解析 (oplist_parse) 与权重格式不动; 执行层新增**分段执行 + 统计**:

```c
struct wt_exec_stats {
    uint32_t ops, nop;         /* 执行 op 数 / 空转 (MATMUL 前置 pin 未就绪) */
    uint32_t matmul, rmsnorm, silu;
    uint32_t pin, pin_skipped; /* 真 staged / 引擎未建时记账跳过 */
};
int  wt_exec_run_range(struct wt_exec_ctx* x, uint32_t from, uint32_t to);
void wt_exec_get_stats(const struct wt_exec_ctx* x, struct wt_exec_stats* st);
/* wt_exec_run 不变 (全量), 内部 = run_range(0, n_ops) + stats 清零 */
```

- `OP_SILU_F16` (=4, arity 3: x_temp/y_temp/n_elem) 执行体: f16 面
  `x/(1+e^(−x))`, 与标量 oracle 半 ULP。
- **整步 vs 逐算子下发**: `wt_exec_run` 与循环 `wt_exec_run_range(i,i+1)` 输出
  **逐字节恒等** (三 temp 面 memcmp) — 图切分自由。
- pin-skip 语义: 引擎未建立时 PIN 只记账 `pin_skipped++`, 建立后真 staged
  (`pin_skipped==0` 恒等门)。

## 设备门 (30_graph_step, 2026-08-16)

blob_parse_ok / fused_run_ok / stats_fused_exact (`ops=6 mm=1 rms=1 silu=1 pin=2 skipped=1`)
/ pin_skip_engine_not_ready / split_run_ok / **split_vs_fused_temps_byteexact** /
pin_skip_engine_ready_zero / silu_vs_oracle_halfulp。
(合成 blob: 6 slots 512B 就地 LCG; split 总 8543us vs fused op 累计 26554us —
仅供参考, 非门。)

## 坑

- stats 是 ctx 生命周期内累计; shutdown 清零, run 前必须 memset (`wt_exec_run` 已内置)。
- `%u` 打印 uint32_t 位型注意强转 (device printf 警告即错)。
