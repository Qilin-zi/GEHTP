# US4 收口 digest — T022 门裁决 (1010)

**源**：`.scratch_008/t21A.op_ts.bin`（默认全量）+ `t21B.op_ts.bin`（op_ts_sample=8）
+ `.scratch_008/t21A.device.txt` / `t21B.device.txt`，lib 私有名 `libhvxhmx_008t21.so`。
**锚**：md5 `f87461e2a5089c48413e70463f4aa7dc`（v4 金标，双跑位级对拍恒 PASS）。

## 门裁决：PASS

| 门 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 幂等门（默认模式） | 行为/位级同现库 | md5 f87461e2；op_ts ver=4 n=11504 全量 | ✓ |
| 开销门（抽样口径） | ≤0.5s | ~0（见下） | ✓ |

## 双跑明细（d0f1784, t19 blob 复用, emit 未变）

| run | op_ts 口径 | exec wall | sum_op_us | 缝隙 | md5 |
|---|---|---|---|---|---|
| A 默认 | full（每 op 记录） | 6.308s | 6.292s | 16.1ms | f87461e2 ✓ |
| B 抽样 | sample=8（i%8==0） | 6.310s | 6.292s | 17.9ms | f87461e2 ✓ |
| t20 对照（US4 前） | full | 6.308s | 6.290s | 18.1ms | f87461e2 |

op_ts.bin 头校验（pad 字段承载抽样间隔 K，消费方显式口径）：
- A：`ver=4 n=11504 pad(K)=0 nonzero=11504`（全量逐条）
- B：`ver=4 n=11504 pad(K)=8 nonzero=1438 zero=10066`（11504/8=1438 精确）

## 关键结论：~1.4s 是 v4 时代陈旧估计

spec/research 里 "逐 op 裁决 + op_ts 记录 ≈1.4s" 出自 v4（31.8s wall）时代，
当时 `run_range` 有 **运行时引擎自决**（ver≤2 逐 op shape 门 + assign_engine）。
**003 ver=3 静态计划已把「逐 op 引擎重裁决」整个消掉**——本轮实查证据：

- `wt_annot_engine/fb_allowed/fb_engine` 全 `static inline`（oplist_parse.h:291-293），
  计划引擎从 footer 一次 `wt_rd_u32` 加载 + 掩码，零函数调用、零重裁决。
- 抽样旋钮（剥掉 dc_dma_get + 8 字段 ts 填充，7/8 op）实测 **exec Δ=+2ms（噪声）**，
  即 ts 记录开销本身 ~0；`sum_op_us` 与 `wall_us` 缝隙仅 **16-18ms**（1.4µs/op 循环簿记）。
- 残余 per-op 固定税 = t0/t1 HAP 打点 + dc_dma_reset + 逐 op 复位（~1-2µs/op ≈ 0.02s），
  是 op_us 聚合与 DMA 计账的必要成本，远低于 0.5s 门。

**US4 的正确交付 = 契约旋钮（T020）+ 幂等保障（默认位级不回归），收益垫底符合 P4 定位。
「逐 op 重裁决批化」在 003 已实质完成，T021 本轮以取证收口，未强行加批化冗余。**

## 落地改动

1. **T020 抽样旋钮**：`kernels/examples/42_gehtp_runner/main.c` — job.txt 键
   `op_ts_sample K`（K≥2 生效，0/缺省=全量）；`wt_exec_set_ts_sample(K)` 下发；
   op_ts.bin **pad 字段复用承载 K**（0=全量，非破坏）；设备日志 `[prof] op_ts full/sample=K`。
   + `scripts/gehtp_prof.py` 读 pad=K 并 stderr 显式口径警告（防 census 类对拍误读）。
2. **T021 run_range**：`kernels/src/runtime/oplist_exec.c` — ts 填充加抽样门
   `(g_ts_sample==0 || i%g_ts_sample==0)`；裁决/偏差不变量仍逐 op 结算（未动 003 契约）。
   取证：静态计划已消重裁决（见上），无冗余可再批化。
3. 头 `oplist_exec.h` + `oplist_exec.c` 加 `wt_exec_set_ts_sample`。

## 残余 / 边界

- 抽样模式 op_ts 的「偏差报告」只对采样 op 有实义（非采样槽恒零 → 计为按计划 0），
  gehtp_prof.py 已 warn 口径；**偏差不变量在运行时仍逐 op 结算**（不在 ts 产物里体现非采样项）。
- 幂等门 = 默认模式（无 `op_ts_sample` 键）与现库逐条一致，md5 位级同。生产可关（`op_ts_sample` 缺省即全量，零扰动）。
