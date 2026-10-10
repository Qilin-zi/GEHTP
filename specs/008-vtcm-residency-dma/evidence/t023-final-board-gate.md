# T023 终局板跑 digest — SC-003 ≤8s + 双口径判分 (1010)

**源**：`.scratch_008/t21A.device.txt`（默认全量）+ `t21B.device.txt`（op_ts_sample=8），
lib 私有名 `libhvxhmx_008t21.so`（commit de5574a），blob `qwen35_08b_t19.wtop` 复用（emit 未变）。
**设备**：d0f1784，Permissive。

## SC-003 总门 ≤8s — PASS

| 口径 | 007 锚 | 008 终局 | 判定 |
|---|---|---|---|
| exec wall（`[timing] wall_us`，007 锚同口径） | 24,240,344 µs（24.24s） | **6,307,990 µs（6.31s）** | ✓ ≤8s，3.84× |

```
[timing] ops=11504 sum_op_us=6291924 wall_us=6307990 ops_per_sec=1823.7
[run] seg_us read=4467765 parse=16752 wgt=26869 exec=6307990 write=26971 wall=11435291
```

- **口径说明（勿混淆）**：SC-003「全链 wall」= `[timing] wall_us`（exec = wt_exec_run 时长），
  与 007 锚 `wall_us=24,240,344`（budget-baseline.md:121）同口径。`[run] seg_us wall=11.44s`
  含 blob 冷读 4.47s（2GB f16 一次性 I/O，非计算）；007 锚亦只计 exec，故不回扣。
  服务化（004 T020 `GEHTP_PERSISTENT=1`）blob 常驻后该冷读摊销消失。
- A（默认）6.308s / B（抽样）6.310s，Δ=2ms（噪声），见 t022 digest。

## SC-004 精度不回归 — 双口径判分登记

### f16 列（位级门）— PASS（不回归）

| 项 | 冻结锚（007/004） | 实测 | 判定 |
|---|---|---|---|
| md5（逐字节） | `f87461e2a5089c48413e70463f4aa7dc` | 同（A/B 双跑恒等） | ✓ |
| cos | 0.998642 族 | **0.998642** | ✓ 不回归 |
| top1 | 30/32 | **30/32** | ✓ 不回归 |
| max\|d\| | — | 1.0660 | 供判读 |

判分命令：`python3 test_assets/qwen35_08b/judge_logits.py .scratch_008/output_008t21_A.f16.raw 0`
→ `prompt0: shape=(32, 248320) cos=0.998642 max|d|=1.0660 top1=30/32`。
judge 脚本内置硬门 `cos≥0.9999` 报 CHECK 属**判据口径差异**（那是 M4.4 长链门，
非 008 的「不回归」门）：008 f16 门 = 位级同 v4 锚，cos 与锚逐位一致即通过。

### W4 列（q4_0 golden 门）— 未消费（沿 007 冻结值）

- 007 冻结门 = 每 prompt **pooled cos(dev, cpu_golden/q4_0) ≥ 0.98 + top1 ≥ 20/32**
  （`gehtp-007-t016-q4-gate`；spec 里 0.9986/29 是 f16-vs-hf 占位量级、按构造不可达）。
- 008 Phase 1（US1–US4）**纯 f16 线**，未触及 W4A16 路径（本图无 W4，staging_sum=0），
  故 W4 列「不降」平凡成立，无需重锚。**Phase 2 W4 fleet 开工前不消费此列**。

## SC 汇总

| SC | 目标 | 实测 | 判定 | 出处 |
|---|---|---|---|---|
| SC-003 全链 wall | ≤8s | 6.31s | ✓ | 本文 |
| SC-004 精度 | f16 位级不动 / W4 沿 007 冻结 | md5 恒等 + cos 0.998642 | ✓ | 本文 |
| SC-005 per-op 开销 | ≤0.5s | ~0（16-18ms 簿记） | ✓ | t022 digest |
| SC-005 残余三族 | ≤2s | 0.56s | ✓ | t019 digest |

## 落地资产

- `.scratch_008/t21A.device.txt` / `t21B.device.txt` / `t21A.op_ts.bin` / `t21B.op_ts.bin`
- `.scratch_008/output_008t21_A.f16.raw` / `output_008t21_B.f16.raw`（md5 f87461e2）
- 判分脚本：`test_assets/qwen35_08b/judge_logits.py`（f16 列）；W4 列判分脚本沿 007
  `.scratch_007/judge_q4d.py`（pooled cos + top1，本线 Phase 2 起用）。
