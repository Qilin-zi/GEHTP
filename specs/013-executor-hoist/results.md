# Results: 013-executor-hoist

**口径**: 每段收口门数值引用 contracts/ 原文；cos-min 门设锚值之下防假 FAIL（[[gehtp-08b-e2e-canonical-flags]] 边界坑）。
**板**: d0f1784；私有名纪律（lib/runner/blob/结果名全私有）；输出判据 = md5 位级 + judge_logits.py cos/top1。

---

## US-B — W4A16 供给侧直连（MVP，已收口 2026-10-13）

**范围裁决**: 原生 W4A16 0.8B E2E **移出 013 范围**（M%256==0 硬约束，M=32/128 原生不可达 → NO-GO）。B 段交付 = 供给侧**机制**（4 表生成 + 9-arity + 删扫描 + M 门槛），合成 M=256 blob 验证，不设 w4 全图性能门。

### 落地项

| 项 | 状态 | 证据 |
|---|---|---|
| requant_w4a16.cc（atbl/otbl 闭式） | ✓ | ctest `test_requant_w4a16`（s256/s2560/K≠N 全绿） |
| packed_w / folded_bias | 文档化 gap | oracle 已失（DFlashSpecCppRunner/htpw4a16_v81），见 `requant_w4a16.h` 尾注 |
| op_spec.inc arity=9 + legacy_arity=6 | ✓ | 单点注册表纪律（R1），不 bump WT_BLOB_VER（R2） |
| op_matmul.cpp 6→9 参发射 + M%256 门槛 | ✓ | T006/T011 |
| oplist_parse.c 双 arity 校验 | ✓ | T007 |
| oplist_exec.c 9-arity 直读三槽 + 删 O(n_slots) 扫描 | ✓ | T009（6-arity legacy 路径逐字节不动） |
| test_wtop_emit 9-arity 往返 + legacy 等价 | ✓ | 修 5 预存红（65c4f2e 010 merge 引入） |

### T010 合成 M=256 blob 设备门（已过）

- 9-arity blob（`blob_w4_v9.wtop`，matmul op3 `[0,1,0,256,2560,2560,2,3,4]`）设备输出 md5 `7654a8f6` == `dev_w4.f16.raw`（P5 dc_w4 已知答案）**位级一致**。
- 同源 6-arity blob 在新 lib 双跑 → 同 md5 `7654a8f6`（T009 直读 == legacy 扫描，legacy 路径未破）。

### T012 B 段零回归门（已过，US5）

| 门 | 目标 | 实测 | 裁决 |
|---|---|---|---|
| f16 默认线（非融合） | 输出 md5 == `f87461e2`（位级） | **`f87461e2`**（blob `qwen35_08b_010.wtop` md5 `3f857926`，out_temp 2796） | ✓ 位级 |
| f16 线 exec | 5.99s 锚 | 5.98s（total_us 5982080，ops 11504） | ✓ 无回退 |
| gdn 线（融合 pp） | cos ≥0.9999 / top1 32/32 | cos **0.999930** / top1 **32/32**，输出 md5 `9d1c0877` == 锚 | ✓ PASS |
| gdn 线 exec | 2.76s 锚 | 2.76s（total_us 2755877，ops 884） | ✓ 无回退 |
| host ctest | 全绿 | **50/50**（含 test_requant_w4a16 + test_wtop_emit 5 红已修） | ✓ |

**零回归机理**: B 段改动仅触 W4A16（opcode 1）路径——f16/gdn 两线无 opcode 1，设备侧 oplist_parse/oplist_exec 改动为死代码，编译器侧 op_matmul W4A16 emit 永不触发。f16 位级 + gdn 双门实测背书。

### B 段收益

- **供给侧机制可达**：W4A16 E2E 链（编译器产三槽 + 9-arity 直读 + 删扫描）落地，合成 M=256 位级验证。
- **删 O(n_slots) 扫描**：9-arity 路径装载期不再扫槽表（热路径常数级）。
- 无性能门（重定范围裁决）；原生 W4A16 M%256 NO-GO 归档见 T028。

**证据**: `evidence/t012-f16-{device.txt,op_ts.bin}` / `evidence/t012-gdn-{device.txt,op_ts.bin}`；`baseline/baseline-20261013.md`。

---

## US-A — VTCM 静态布局（未开始）

## US-C — carve 静态布局表（未开始）

## US-D — 形态 flag 化（未开始）

## 终局双门（T028 汇总，未开始）
