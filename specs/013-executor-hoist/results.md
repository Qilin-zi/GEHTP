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

## US-A — VTCM 静态布局（已收口 2026-10-14：T013-T017 全绿）

**T013/T014（地基，commit `cdc6610`）**：

- **T013** `wtop_emit.cpp`：TEMPOFF temp 表后追加 32B 布局段 `[magic 'VTCL'][ver 1][region_pin{off,size}][region_arena{off,size}][region_resident{off,size}]`。三区 VTCM-base 相对、互斥：pin=[0,pin_cap)、arena=[align2048(pin_cap), +hmx_arena_max)、resident 尾倒划到 16MB 顶。`hmx_arena_max_of()` 覆盖 f16 matmul / W4A16 / SPLIT 三形态。旧运行时 `len<16+n*12` 是 `<` 非 `==`，尾随段向后兼容。
- **T014** `hmx_common.c`：新增 `hmx_runtime_setup_region(void* base, unsigned size)`（base=0→全量 setup；base!=0→attach 共享 scratch）。不改 `hmx_runtime_setup(size)` 签名（会毁 ~20 旧调用点 UB），旧调用点零扰动。

**T015/T016（执行器上移，2026-10-14）**：

- **T015** `wtcache.h`/`wtcache_impl.c`：`wtcache_flush_count()`（wtcache_open 计数）；执行器装载期按表 `wtcache_open(pin_sz)`，f16/conv 消费点 `hmx_runtime_setup_region(vb+arena_off, arena_sz)` 表驱动有界 attach——零运行时布局决策。
- **T016** `oplist_exec.c`：VTCL 解析 → 装载期一次分区 + 三重 drift 硬校验（三区互斥 / arena==align2048(pin) / resid_sz==驻留池）；驻留池基址取表 resid_off；探戈计数全挂 `!has_layout`（布局图恒 0）；re-carve 仅 flush arena 段；carve 基址查表；cap=0（全 VTCM 驻留图）契约修复；关机 FARF+optrace 双通道打 layout/full_flush/arena_flush/tango。

**T017 设备门（板 d0f1784，证据 `evidence/t016-a-seg-gate.md`）**：

- 合成双 matmul 驻留 blob（VTCL+n_vtcm=2，host 逐字节核验）新旧 lib 双跑**位级同**（md5 `ce338fe6`）
- optrace：`layout=1 full_vtcm_flush=1 arena_flush=0 tango=0`（探戈零发生、装载期恰一次全 VTCM flush；W4A16 re-carve 段 flush 编译在未触发，待 C 段覆盖）
- f16 线 `f87461e2` 位级同锚（exec 5.99s）；gdn 线 `9d1c0877` 位级同锚（=cos 0.999930/top1 32/32，exec 2.76s）

### A 段收益与诚实定位

- **机制性前置设施落地**：装载期一次分区/探戈消解/flush 收敛/查表 attach 全实跑验证；cap=0 全驻留图合法产物契约修复。
- **对本轮基准无性能收益也无回归**（位级双确认）：0.8B f16/gdn 基准图无 VTCL 段走 legacy 路径；收益兑现依赖 C 段 + emit 侧驻留规划打开。
- **过程坑入册**：fastrpc 按 lib 名缓存（重推必须换名）；板上 lib 必须 SWIV 签名版；FARF 不进 logcat（取证走 optrace）。黑话词典 `docs/GEHTP_GLOSSARY.md` 随本段建立。



## US-C — carve 静态布局表（未开始）

## US-D — 形态 flag 化（未开始）

## 终局双门（T028 汇总，未开始）
