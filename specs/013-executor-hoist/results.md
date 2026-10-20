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



## US-C — carve 静态布局表（已收口 2026-10-14：T018-T020 全绿）

> 白话：以前每次遇到 W4A16 矩阵乘，执行器都要现场"画格子"（在 VTCM 高速内存里逐块
> 计算激活/输出/权重/偏置各放哪、各占多少，即 carve）。现在编译器把格子画好后直接
> 写进 blob 尾部（CRVL 表），执行器装载时按公式复核一遍（对不上就拒载），热路径只做
> 八个指针赋值——现场画格子的扫描和初始化全部从热路径删除。

**T018 `wtop_emit.cpp`（commit 待归档）**：

- CRVL footer：`[magic 'CRVL' 0x4C565243][n_layouts][n×40B 布局记录][n_refs][n×8B {op_index, layout_id}]`，位于 gdn footer 之后、128B 权重对齐之前，magic 探测确定性（填充全零）。
- 布局记录 {layout_id, M, K, N, n0, off_wt, off_bias, off_act, off_out, arena_hi}；去重键 (M,K,N,n0)（同形 W4A16 共享一布局）；发射门=em.ddr_static（有 TEMPOFF/VTCL）且存在 W4A16 op；M/K/N 异常（0 或 ≥2²¹）留 legacy carve。
- 偏移=镜像 `dc_w4_carve` 分配序的闭式公式（host 已证公式与逐块顺序分配逐字节等价）。

**T019 `oplist_parse.h/.c` + `oplist_exec.c`**：

- 解析侧仅边界校验；执行器装载期逐布局：记录合法性（lid/n0/对 32 取模）+ 闭式重算五偏移与线值对拍——**漂移即拒载**（err="cl drift #N: … 线值/算值"）+ arena_hi≤布局 arena 段 + 每 ref 的 op 边界/opcode/args[3..5] 形状一致性；随后一次性展开 cl[] 八指针。
- 热路径：查 cl_ref 命中 → 布局不变时零工作（仅 atbl/otbl 供给槽逐 op 刷新，因 w4a16_invoke 破坏性重写表）；布局切换 → 驻留槽失效 + 八指针直赋 + arena 段 FLUSH（arena_flush++）。**热路径已无 carve_live_top 扫描 / dc_arena_init / dc_w4_carve**；未引用的 W4A16 op 落 legacy carve 兜底。

**T020 设备门（板 d0f1784，证据 `evidence/t020-cl-device-gate.md`，lib `libhvxhmx_018.so`/runner `gehtp_runner_018b.so` 签名版指纹 ed4e46d6/347a3c56）**：

- 合成 M=256,K=N=2560 W4A16（v3+TEMPOFF/VTCL+CRVL）三跑：twin（无 CRVL，legacy 路径）与 cl（CRVL 快路径）输出均 `7654a8f6` **位级同 003 时代已知答案**；bad（CRVL off_wt 破坏 1 字节）装载期硬拒 `cl drift #0: wt 2621695/2621440`，零输出。
- 计数器（optrace）：cl 跑 `layout=1 arena_flush=1 tango=0`（快路径触发恰一次 arena 段 flush——补上 A 段"re-carve flush 编译在未触发"缺口）；twin `layout=0`；bad 拒载于热路径前。
- 0.8B 零回归：f16 线 `f87461e2` 位级同锚（11504 ops，exec 5.97s）；gdn 线 `9d1c0877` 位级同锚（exec 2.75s）。
- **收益量化（单 op 对照）**：W4A16 op 13.47ms(legacy)→6.92ms(CL)=**1.95×**；整 exec 34.7→32.1ms（单 op 图收益被读权重摊薄；多 W4A16 op 真图收益随 op 数放大）。

### C 段诚实定位

- 机制+位级门全绿；**0.8B 基准图当前无 W4A16 op**（lm_head 为 f16），收益兑现在真图待 W4A16 供给侧 emit 打开（013 记忆已记"收益未兑"缺口，与 B 段同口径）。
- 过程坑：`/data/local/tmp/hvxhmx23/gehtp` 已变 root:755（他会话 root adb 活跃）——绕行=私有目录 cl020 + runner 重编私有名烘焙 job 路径；板上库**必须推 SWIV 签名版、但以普通 .so 名落地**（推未签名版 dlopen 直接死，且 FARF 不可见）；`run_main_on_hexagon.farf` 掩码文件可开 AP 侧详细日志（用后已删）。


## 附加实测 — w4 线 0.8B 当前树全链重编（2026-10-14，证据 evidence/q4-line-rerun-device.md）

013 原 spec 承诺「w4 线对 007 q4 锚 25.7s 实质下降」，T011 移出的是**原生 W4A16** E2E（M%256 NO-GO）。本次以当前树**重编译**（gdn 融合 18/18 烤入）+ q4 发射（--gguf，w4 占比 99.7%，blob 5be4bd7f）补齐：

| 门 | 结果 |
|---|---|
| 精度（冻结门 cos≥0.98+top1≥20/32，vs cpu_golden/q4_0） | **4/4 PASS**（cos 0.99976–0.99996 / top1 31–32） |
| exec vs 25.7s 锚 | **15.09s，−41.3%**（wall 27.4→16.3s） |
| op_timing | MATMUL_F16 9.39s（锚 10.6s）；lm_head 4.59s（持平）；**gdn_block 融合 18 块共 0.65s**（旧图散在杂项 ~10.5s，主刀） |

诚实归属：收益=**全链重编**（gdn 融合+编译期折叠，920 ops vs 旧 11504），非 013 A/C 机制（0.8B VTCM auto 撤退无 TAG_MEM_PLAN、q4 线无 W4A16 op，本就不触）。007 锚的「旧图+当代 lib」欠账就此补齐，w4 线与 f16 线同源同代。

## US-D — 形态 flag 化（已收口 2026-10-19：T021-T025 全绿）

**Goal**: broadcast 形态 / GEMV 引擎 / Q4 槽判别编译期入 annotation flag，运行时零重推导。

**落地项（commit `9871577`）**：

- **T021** 前置检查：012 已落地主线（bca460a 含 exec_matmul_f16_hmx 会话外提段），分发段稳定。
- **T022** `wtop_ops.hpp` `broadcast_engine_annot`：形态类(1 整块/2 行循环/3 strided)入
  annotation bit[6:7]（`wt_annot_set_bcast_form`）。
- **T023** `op_matmul.cpp` + `oplist_parse.h`：GEMV 走 planned_engine 编码(M<32 编译期已知)；
  Q4_0 权重槽判别入 bit[8]（`wt_annot_set_q4`），替代运行期 `len==(k/32)*(nn/32)*640` 槽长探针。
- **T024** `oplist_exec.c`：注解路径直用 emit 形态（exec_broadcast 直读 form、exec_matmul_f16_impl
  直读 q4 位），删运行时 shape-class 重算与 Q4 探针；双侧判据漂移检测挪装载期一次校验。

**T025 D 段门（板 d0f1784，正典旗标）**：

| 门 | 锚 | 实测 | 裁决 |
|---|---|---|---|
| prefill（gdn-ON 默认线）精度 | cos 0.999921 / top1 32 | md5 `2c09d8ba` == 锚（位级同） | ✓ |
| prefill 性能 | exec 0.490s | **0.489s** | ✓ 无回退 |
| seq1 decode（GEMV 主场） | cos 0.999975 / 1.89s | 1.885s rc=0 | ✓ 非回归 |

**收益与诚实定位**：执行器热路径零运行时形态规划达成（SC-004）；本段是**机制性等价改写**
（编译期 flag == 运行期旧判据，位级同锚背书），无独立性能收益——与 A/C 段同口径。

## Polish — T026 删 legacy 6-arity 扫描（已收口 2026-10-19）

`exec_matmul`（W4A16）删 `n_args==6` O(n_slots) 槽表扫描，仅 9-arity 直读三槽，非 9-arity
执行期硬拒；parse 侧 `legacy_arity` 保留（装载仍放行 6-arity，向后兼容不 bump WT_BLOB_VER）。

**设备门（板 d0f1784，lib `libhvxhmx_013usd.so`）**：

| blob | arity | 结果 |
|---|---|---|
| `blob_w4_v9.wtop` | 9 | md5 `7654a8f6…` == 已知答案（位级同，直读路径未破） |
| `blob_w4_6ar.wtop` | 6 | `w4a16 arity 6 (legacy 6-arity scan removed)` 硬拒（预期） |

三线重跑零回退（prefill 0.489s/`2c09d8ba` + seq1 1.885s + W4A16 合成）。证据
`evidence/t026-legacy-scan-removal.md`。

## T027 — 回合主线 merge（已收口并 push 2026-10-19：`95188e1`）

merge/013-into-mainline（双亲 graph_prepare 拆分 `29a931e` + 013 `9871577`）**零冲突**
（文件集 disjoint）→ origin/task/prof-wp 快进。host compiler 全量构建 exit 0；ctest
**48/50**（唯二红 = test_wtop_emit/test_vtcm_auto 的 VTCM 驻留 auto 预存失败，013 之前已在
主线，非本回合引入，见 `evidence/t027-merge-mainline.md`）；三线板测位级同锚。

## 终局双门（T028 汇总）

**精度不回退**（默认线 cos 0.999921 / top1 32，位级同锚 md5 `2c09d8ba`）+ **性能实测**
（prefill exec **0.490s** 无回退；seq1 decode 1.89s 非回归）。

| 段 | 收益 | 性质 |
|---|---|---|
| B（W4A16 供给侧） | 机制可达（4 表+9-arity+删扫描+M 门槛） | 无性能门（M%256 NO-GO 裁决归档） |
| A（VTCM 静态布局） | 装载期一次分区/探戈消解 | 机制前置，本基准无收益 |
| C（carve 静态表） | W4A16 op 1.95×（单 op 对照） | 收益未兑（0.8B 无 W4A16 op） |
| D（形态 flag） | 热路径零运行时形态规划 | 机制等价，无独立收益 |
| Polish T026 | 删 6-arity 扫描 | 卫生 |

原生 W4A16 0.8B E2E 收益（M%256 NO-GO）显式移出 013 范围，归档于此。
