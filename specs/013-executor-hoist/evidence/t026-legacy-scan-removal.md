# T026 — 删 legacy 6-arity 扫描路径（设备门）

日期 2026-10-19；板 d0f1784；worktree `013-usd`（base bca460a，GDN P1/P2 tip）。

## 改动

`kernels/src/runtime/oplist_exec.c` `exec_matmul`（W4A16，opcode 1）：删除 `n_args==6`
分支的 O(n_slots) 槽表按长度扫描，改为 **9-arity 直读三槽（bias/atbl/otbl），非 9-arity
硬拒**。parse 侧 `legacy_arity`（T005/T007/T008）**刻意保留**（blob 装载仍放行 6-arity，
执行期才拒），避免 bump WT_BLOB_VER 与破坏向后兼容解析。

硬拒信息：`[FAIL] wt_exec_run_io rc=4 w4a16 arity 6 (legacy 6-arity scan removed)`。

## 门 1：板上 6-arity blob 依赖普查（先决）

`/data/local/tmp/hvxhmx23/gehtp/` 全量 `ls` 核对：唯一 6-arity 产物 = 合成
`blob_w4_6ar.wtop`（T010 遗留，4643328B）；无任何生产/私有 6-arity blob。删除安全。

## 门 2：W4A16 合成 blob 双跑（lib `libhvxhmx_013usd.so` / runner `gehtp_runner_013usd.so`）

| blob | arity | 结果 |
|---|---|---|
| `blob_w4_v9.wtop` | 9 | **[PASS] out 1310720B md5 `7654a8f64a0d66f2e93985ef34f10016`** == dev_w4.f16.raw（已知答案位级同） |
| `blob_w4_6ar.wtop` | 6 | **[FAIL] `wt_exec_run_io rc=4 w4a16 arity 6 (legacy 6-arity scan removed)`**（预期硬拒） |

- 9-arity 直读路径在删 6-arity 扫描后**位级不变**（直读 == T009 时代直读，扫描从未参与 9-arity 路径）。
- 6-arity 现于执行期硬拒（装载期 parse 仍放行，兼容不破坏）。

## 门 3：三线重跑（零回归，US5）

| 线 | 锚 | 实测 | 裁决 |
|---|---|---|---|
| prefill（gdn-ON 默认，usd_base.wtop） | exec 0.490s / md5 `2c09d8ba` | **exec 0.489s / md5 `2c09d8ba`** | ✓ 位级同 + 无回退 |
| seq1 decode（GEMV 主场，seq1_base.wtop） | exec 1.89s | **exec 1.885s**（rc=0, 12394 ops） | ✓ 无回退 |
| W4A16 合成（blob_w4_v9.wtop） | md5 `7654a8f6` | **`7654a8f6…`** | ✓ 位级同 |

**零回归机理**：T026 改动仅触 W4A16（opcode 1）执行路径——prefill/seq1 两线无 opcode 1
（全 f16 matmul），改动为死代码；W4A16 9-arity 合成 blob 位级同背书直读路径未破。

## 瞬时回归诚实记录

首跑 prefill 出现 **7.85s**（输出 md5 e0d81d41，4 个 gdn GEMM code=24 回落标量，
`engine_fallback reason=matmul_f16 hmx setup fail`）。根因=**瞬时板态**（legacy
`hmx_runtime_setup(2MB)` 因 HMX/VTCM 残留坏态失败，非 T026——hmx_common.c 未动且
W4A16 exec 路径 prefill 不触）。重跑即恢复锚 0.489s/2c09d8ba，两次位级同。

## ctest 阻塞（预存，非 T026 引入）

host ctest 全「Not Run」：编译器测试二进制为容器构建产物，宿主无 `test_*` 可执行文件
（预存环境问题）。T026 改动为设备侧死代码，宿主 ctest 本不可见。
