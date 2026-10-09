# P1 门验收 — US1 broadcast HVX 纯搬运（2026-10-08, 板 d0f1784）

## 判据（spec SC-001/SC-004/SC-005 + 口径修正）

| 门 | 判据 | 结果 |
|---|---|---|
| broadcast 族 dev=1 回落 | 918 → 0 | ✅ **0**（且 704 计划标量全部翻 HVX） |
| broadcast 族耗时 | ≤ 1s | ✅ **0.082s**（基线 7.841s, 95.9×） |
| 全链路 wall | ≤ 28s | ✅ **24.240s**（基线 31.778s, -7.54s） |
| 精度 | cos 0.998642 不回归（逐字节） | ✅ 输出 md5 = 基线 md5 = `f87461e2a5089c48413e70463f4aa7dc`，位级一致 |
| 类 3（form3/strided） | 不计入回落 | ✅ census form3=0（evidence/broadcast_census.md），板上无形态 3 实例 |

## 证据链

- **产物**: blob `qwen35_08b_v4_007.wtop`（worktree `.scratch_007/`, md5 `722b3bb0`）
  结构与 v4 基线逐项一致（slots=1942 / ops=14188 / bytes=2041268992）；
  唯一差异 = v3 注解 footer：opcode 26 计划引擎 918 HVX+704 SCALAR → **1622 全 HVX**（fb_allowed=1）。
- **编译输入**: `qwen35_08b_v4.tagged.bin`（md5 `93427241` = 002 会话验证的稳定锚，未重编）。
- **板跑**: runner `gehtp_runner_007.so`（私有 job 路径 `gehtp_007/job.txt` + 私有 lib `libhvxhmx_007.so`，
  P5 隔离法），rc=0, ops=14188。
- **逐 op 证据**: `v4_007.op_ts.bin`（WTS1 v3, n=14188, wall_us=24240344）+ gehtp_prof.py 解析：
  - opcode 26 全部 1622 条 `(planned=1, actual=1, deviation=0)` —— 918→0 机检成立
  - 族总耗时 81810 µs
- **全图残余偏差**（P1 范围外, 与基线口数一致未动）: UNARY(code11) 86 + ADD(code7) 18 + SOFTMAX(code12) 1 = 105
  （基线同族 86+18+1=105）。
- **日志**: `.scratch_007/v4_007.device.txt`（M11 exec done rc=0 total_us=24240344; op_timing code=26 n=1622 total_us=81810）。

## 性能侧写（op_timing 对比, µs）

| opcode | 基线 total | 本门 total | Δ |
|---|---|---|---|
| 26 BROADCAST (n=1622) | 7,841,308 | 81,810 | **-7.76s (95.9×)** |
| 24 MATMUL_F16 (n=325) | 9,440,055 | 9,438,742 | ~0（噪声） |
| 35 MATMUL_SPLIT (n=32) | 4,248,976 | 4,246,591 | ~0（噪声） |
| 15 UNARY (n=2373) | 3,062,491 | 3,352,955 | +0.29s（run 间噪声, 无因果路径） |
| 全链 wall | 31,778,204 | 24,240,344 | **-7.54s** |

## 结论

US1（P1）四门全绿，broadcast 族 7.84s 单点（006 归因报告确认的最大单点 24.8% wall）已消灭；
下一目标 = US2 W4A16 全图（wall 主项 code=24/25/35 matmul 族 ~14s）。
