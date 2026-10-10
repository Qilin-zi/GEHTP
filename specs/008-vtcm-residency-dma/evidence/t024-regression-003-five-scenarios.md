# T024 回归复验 digest — 003 blob-v3 五场景 + ctest (1010)

**源**：008 当前 lib `libhvxhmx_008t21.so`（commit de5574a 系，US1-US4 全含）+ 重编 compiler
`compiler/build_linux`（10-10 08:01，含 US3 emit 改动）。合成 blob 源 = `/tmp/{devblob,w4blob,transformer2l.wtop}`
（003 会话产物，沿用，未重生成）。板跑脚本 `.scratch_008/run_008_t024_scenarios.sh`，全私有名 `t024_*`。
**设备**：d0f1784。

## 门裁决：PASS（五场景逐位复现 + ctest 全绿）

### A. 003 blob-v3 五场景（003 锚逐字节对照）

| 场景 | 003 锚 (Oct 8) | 008 实测 | 判定 |
|---|---|---|---|
| devA 显式回落 | rc=0 dev=[0,1,1] md5 2c08b292 | 按计划 1 / 允许回落 2 / 已中止 0，md5 **2c08b292** | ✓ |
| devB 回落禁止 | rc=2 die `fallback disallowed: n<64 (add HVX gate)` | rc=2 同文案逐字复现，按计划 1 / 已中止 1 | ✓ |
| w4v3 carve+DMA | md5 7654a8f6 (5 op, code33 DMA) | md5 **7654a8f6** | ✓ |
| w4v1 ver1 legacy | md5 7654a8f6 (4 op) | md5 **7654a8f6** | ✓ |
| t2l ver1 legacy | md5 7940320b (37 op) | md5 **7940320b** | ✓ |

偏差不变量全 **PASS**（`gehtp_prof.py` 机验），已中止 0（除 devB 构造性中止）。

**坑（本会话修复）**：`run_scenarios_2345.sh`（003 遗留）写 `t2l out_temp=11`，实测报
`[FAIL] out temp 11 empty`；003 原 device.txt 证据为 **out temp 4**（`M12 wrote 1024 bytes`）。
008 脚本已改为 4 并加注。其余 out_temp：devA=2 / devB=0 / w4v3=0 / w4v1=0。

### B. ctest（重编后，当前源码）

`compiler/build_linux` 增量重编（US3 emit 改动此前未入 host 二进制，原 `test_wtop_emit`
= 10-10 02:02 早于 US3 提交 05:22-06:13）后：**48/48 全绿**（`100% tests passed, 0 failed`）。
003 锚为 47/47；+1 = 008/004/005 线新增测试（`test_wtop_emit`/`test_static_input`/
`test_runtime_alloc`/`test_tiling_rule` 等，非本线引入缺口）。

### C. 引擎指派 + 数值（场景 1/4，由整网板跑覆盖）

- 引擎指派：t21 整网偏差报告 **按计划 11399 / 允许回落 105 / 已中止 0，不变量 PASS**；
  105 回落全为 Eltwise_Unary HVX→scalar（US3 刻意标量子型 SQRT/SOFTPLUS/NEG/SIN/COS）。
- 数值：T023 已验 cos 0.998642 / top1 30/32 / md5 f87461e2（f16 位级）。

## 结论

008 的 US1-US4（broadcast HVX / slice·transpose·unary HVX / op_ts 抽样）**未破坏 003 契约**：
回落裁决（`engine_fallback_or_die`）、carve footer、ver1 legacy 逐字节、偏差报告不变量全保留。
五场景按「同库对照」口径全过。

## 落地资产

- `.scratch_008/t024_scenarios/{devA,devB,w4v3,w4v1,t2l}.{out.raw,device.txt,op_ts.bin,run.log}`
- `.scratch_008/run_008_t024_scenarios.sh`（t2l out_temp 修正版）
- 重编产物：`compiler/build_linux/hnnx_compile` / `host_run` / 全 test 二进制（10-10 08:01）
