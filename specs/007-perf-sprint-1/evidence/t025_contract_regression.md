# T025 — 003 契约回归（SC-006 / FR-006）收口（2026-10-10）

## 结论

**PASS**：ctest 48/48 全绿；007 改动经静态 diff 证明**未触碰任何 003 契约面**
（序列化 / opcode / slot 布局 / carve footer / 回落协议 / ver1 legacy 分发）；
被改动面（broadcast planned-engine 注解 + 执行、Q4 LUT pack）已由本 sprint 的
设备门 T011 / T017 直接复验，偏差报告不变量成立。

## 一、ctest 终局（host 回归）

`compiler/build_host/ctest --output-on-failure` → **48/48 passed, 0 failed**
（spec 写「47/47 终局」，实际 47→48：新增 `test_tiling_rule`，非本 sprint 引入）。
重点契约项 `test_wtop_emit`（含 003 v3 注解表解析 + broadcast 三类形态 planned
engine 断言 + `fallback_allowed=0` 硬报错负例）与 `test_e2e`（融合契约）均 PASS。

## 二、静态契约不变证明（diff 范围审计）

007 sprint 全量改动 `git diff 7deb63b..HEAD` 仅 12 文件（+1023/-96）：

| 文件 | 改动性质 | 是否触契约面 |
|---|---|---|
| `compiler/tools/wtop_emit.cpp` | FR-003 GGUF/match 预检硬门、w4report sidecar、`ensure_weight_slot` 实参 `od->grouping`→`od` | 否（无 opcode/slot/序列化格式变更） |
| `compiler/tools/wtop_ops/op_matmul.cpp` | 同上实参 + bias 槽 `false`（T016 根因2） | 否（非 carve 发射） |
| `op_dwconv/op_rmsnorm.cpp` | 同上实参 | 否 |
| `op_eltwise_binary/ternary.cpp` | 注入的 broadcast 子 op 加 `em.cur_engine = broadcast_engine_annot(...)` | 复用既有引擎注解机制（`cur_engine`），非格式变更 |
| `wtop_ops.hpp` | `broadcast_engine_annot` + gguf 2-D 布局分族（T016 根因3） | 否 |
| `compiler/tests/test_wtop_emit.cpp` | 契约测试 | 测试面 |
| `kernels/src/hvx/hvx_binary.c` + `hvxhmx_v2_binary.h` | broadcast 形态1/2 HVX 块拷贝 | 否（新执行体） |
| `kernels/src/runtime/oplist_exec.c` | `exec_broadcast` 三形态分派 + `exec_matmul_f16_hmx` Q4 LUT + `exec_gather` Q4 索引修 | 见下 |
| `scripts/gguf_wtlink.py` | 匹配表生成 | 否 |

`oplist_exec.c` 改动段核对（`git diff` hunk 定位）：`exec_gather`(:1648)、
`exec_matmul_f16_hmx`(:1757-1900)、`exec_matmul_f16_impl`(:1944-2045)、
`exec_rmsnorm2`(:2075)、`exec_broadcast`(:2085-2294)。**均未触及**：
`xop_matmul_w4a16`（carve 执行体）、`run_range` 兜底网、`ver<=2` legacy 分发、
`engine_fallback_or_die` 函数体本身（仅 broadcast 新增一处同机制调用点，
003 已先落该机制，见 003 tasks.md 0930 收口注）。

**结论**：序列化契约「本线不新增 opcode、不改 slot 布局」逐条成立。

## 三、blob-v3 五场景逐条映射（003 quickstart.md 五场景）

| # | 003 场景 | 007 影响面 | 007 侧验证证据 |
|---|---|---|---|
| 1 | 引擎指派（planned==actual，deviation==0） | **有变更**：broadcast planned engine 类1/2→HVX、类3→scalar | T011 P1 gate：回落 918→0、类3 计划 scalar=dev0（合法非回落）、cos 0.998642 位级不动 → 偏差报告不变量 PASS |
| 2 | 显式回落不静默（dev==1 允许 / dev==2 中止） | 无变更（协议函数体未动；broadcast 回落反减 918→0） | 协议字节同 7deb63b；devA/devB 合成 ADD blob 走 exec_add 路径，本 sprint 未动 |
| 3 | VTCM carve + 显式搬移（重叠不变量） | 无变更（`xop_matmul_w4a16` carve 未动） | T015 0.8B W4A16 全图发射 + T016/T017 Q4 板跑在位；carve 不变量由 ctest 负例锁死（T022） |
| 4 | 数值 + 性能门 | **有变更**：broadcast HVX + Q4 LUT | T011 cos 0.998642 位级（max\|d\|=0）；T017 Q4 LUT 位级 max\|d\|=0、cos 0.993484 不变；wall 35.31→27.4s（T024 实测） |
| 5 | 向后兼容 legacy（ver1 逐字节） | 无变更（`ver<=2` legacy 分发未动） | ver1 blob 走 legacy 路径，本 sprint 未改该分支 |

## 四、回归口径说明（诚实披露）

- 场景 1/2/3/5 的**合成 blob**（devA/devB ADD、carve w4_v3、ver1 legacy w4）所测代码路径
  与基线 7deb63b **字节相同**（diff 未触及），故未在板上重跑——重跑为字节相同路径，
  无增量信息。若后续要求逐位复验，按 003 quickstart.md 场景 1-5 命令重放即可。
- 场景 4 及场景 1 的 broadcast 子面是 007 **实际改动面**，已由本 sprint 设备门
  T011（broadcast 回落 918→0 + cos 位级）/ T016-T017（Q4 4-prompt 门 + LUT 位级）直接覆盖。
- 003 五场景的「同库对照」锚已在校准（cos 0.998642/30/32/wall 35.31s），本 sprint
  无回退、仅前进（wall 27.4s、broadcast 族 ≤1s），未破该锚。

**SC-006 判定**：PASS（ctest 48/48 全绿 + 003 契约静态不变 + 改动面设备复验）。
