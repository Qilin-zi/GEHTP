# 主线合并后设备验证 — 007 prefill（2026-10-10, 板 d0f1784）

## 背景

007 全量（T001-T027）回合主线 `task/prof-wp`（快进 c55cc14→dd2462d）后，合并树设备库
= **007 broadcast HVX + Q4 LUT + 004 状态 op 三套叠加**（283888B 无符号 / 283944B SWIV 签名）。
宿主编过 + ctest 49/49 已验证，本文件记录**上板数值门**：合并库跑 007 prefill 是否位级回归。

## 验证对象

| 项 | 值 |
|---|---|
| 合并库（签名版） | `kernels/lib/libhvxhmx_v23.signed.so` = 283944B |
| 板上私有名 | `libhvxhmx_007.so`（推前 6ebf39fa 274600B → 推后 **4b1711ab** 283944B） |
| runner | `gehtp_runner_007.so`（007 私有名，NEEDED=libhvxhmx_007.so，未重编） |
| ABI 兼容性 | 旧 007 runner 直接加载合并库 rc=0（`wt_exec_run_io` 签名未变，004 的 `wt_exec_load_states` 是内部调用非导出符号） |

## 结果：两 prefill 位级一致（合并库未回归 007）

| prefill | 合并库输出 md5 | 合并前锚 md5 | 判 |
|---|---|---|---|
| f16 v4（broadcast HVX 全链） | `f87461e2a5089c48413e70463f4aa7dc` | `f87461e2a5089c48413e70463f4aa7dc` | **位级一致**（= US1 锚 md5） |
| Q4（LUT dequant 全链） | `d4b96641a5d002cce02d148525a7b715` | `d4b96641a5d002cce02d148525a7b715` | **位级一致**（= T017 LUT 锚 md5） |

## 时序（exec = 数值无关，仅证无路径破坏）

| 项 | 合并前锚 | 合并库 | 判 |
|---|---|---|---|
| f16 v4 exec | 24.24s | 23.55s | ~噪声（略快） |
| f16 v4 code=26 broadcast (n=1622) | 81.8ms | 77.7ms | HVX 在位 |
| f16 v4 code=35 lm_head (n=32) | 4.247s | 4.242s | ~噪声 |
| Q4 exec | 25.7s | 25.1s | ~噪声 |

## broadcast 回落不变量（gehtp_prof.py 偏差报告）

- op_ts.bin ver=3 偏差报告：**按计划 14083 | 允许回落 105 | 已中止 0**，不变量 PASS。
- broadcast(opcode 26) 1622 条 **deviation=0**（全 HVX，回落 0），与 US1 锚一致。
- 105 残余回落 = 基线口径 UNARY(86)+ADD(18)+SOFTMAX(1)，非 007 范围，与合并前同。

## 结论

**007 三套设备侧改动（broadcast HVX / Q4 LUT / 计划引擎注解）在合并库（007+004）下位级无回归。**
004 状态 op 是加法式新增（新 opcode 29/30 + 内部 `wt_exec_load_states`），prefill 图不含状态 op，
`wt_exec_load_states` 空表即返，不与 007 路径交互 —— 实测两 prefill 输出逐字节同即证。

## 未覆盖（留 004 会话）

- **004 decode 多步（state op 实义路径）未在本记录复验**：需合并树重编的 persistent runner
  （含 004 `steps` 循环码）+ decode 图 blob + 多步输入序列。007 旧 runner 不含 `steps` 码，
  只能跑 prefill（n_steps=1），decode 无法经它触达。属 004 域，合并前 004 会话已自验 decode；
  合并为 clean auto-merge（唯一冲突 wtop_ops.hpp 两独立 helper 并存），kernel 运行时零冲突。
