---
description: "Task list for 0.8B decode 上板（M=1 GEMV + 命名状态表 KV/SSM + persistent runtime + 端到端）"
---

# Tasks: 0.8B decode 上板

**Input**: Design documents from `/specs/004-decode-on-device/`

**Prerequisites**: plan.md (required), spec.md (required)

**Tests**: 各 story 含验证门（host 合成验证 + decode==prefill 金标 + 设备上板 optrace/cos/wall），按 plan.md「验证门」节执行。

**Organization**: 按 user story 分组；两阶段交付——P1（Foundational+US1/US2/US3 P1 切片）先落地 M=1 单步打通（MVP），P2（Phase 6）叠 KV+SSM+persistent+端到端。

**阶段验收铁律（SC-008）**：每个 Phase 的 Checkpoint 都含 **0.8B 上板双门——精度（decode 门：设备自洽 cos≥0.9999 + 跨设备基线 cos≥0.998643/top1 29/32 不回归）+ 性能（wall 记录对照基线/上一阶段值）**，双过才算该阶段完成。

> ⚠️ **基线精度现实（2026-09-29 实测）**：本树 0.8B 现状基线 cos=**0.998643** / top1=**29/32**（v17，HEAD afb56c7），**未达**判据脚本的黄金门（cos≥0.9999 且 top1 32/32）。这是 0.8B 线**既有的精度缺口**（根因见 [[gehtp-08b-accuracy-bisect]] / [[gehtp-f16-denormal-convert-bug]] / [[gehtp-08b-select-gather-fix]]），与本特性（decode 上板）正交。**本特性精度门语义 = 不回归基线**：跨设备门保持 cos≥0.998643/top1 29/32；**设备自洽门（decode==prefill 末位，同权重同来源）另设 cos≥0.9999** 以精确捕状态 bug——两者是不同门槛，前者是「0.8B 既有缺口」的现实，后者是「decode 状态正确性」的强判据。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无依赖）
- **[Story]**: US1=编译上板 / US2=精度保证 / US3=性能优化
- 设备任务前置惯例：先 `adb shell 'ps -A | grep hexagon'` 查共享设备冲突；推签名版 lib（[[gehtp-p5-device-gate]] / [[gehtp-build-env]]）

## Path Conventions

- kernels: `kernels/src/runtime/`, `kernels/include/`
- compiler: `compiler/src/`, `compiler/tools/`
- examples: `kernels/examples/`
- host 循环/golden: `scripts/`（`gehtp_server.py`/`device_runtime.py`/`golden_qwen35.py`/`judge_logits.py`）
- Spec artifacts: `specs/004-decode-on-device/`

---

## Phase 1: Setup（基线锚 + deploy loop + git settle）

**Purpose**: 锁定基线数据 + settle git + 摸清 seq=1 编译入口，为后续所有验证门铺路

- [ ] T001 抓基线：0.8B prefill v17 上板 d0f1784 一轮，落 `.scratch_004_baseline/`（勿入 git），记录 **cos=0.998643/top1=29/32 + wall** 与 sha——这是 decode==prefill 门 + 双门的对照锚。取证链 `.scratch_004_baseline/prof/`（op_ts.bin/device.txt）
- [X] T002 git settle：实测主树=`task/prof-wp`@a5bdc70（dad4b48+afb56c7 已为祖先，ff-merge 不需要）且留有 003 会话未提交 WIP（blob v3 引擎注解，其 T007 与本特性 T004/T005 同改 `exec_matmul_f16` 段）→ **004 开独立 worktree** `/disk1/gehtp-wt/004-decode`（分支 `004-decode-on-device` off a5bdc70）隔离，spec 三件套随分支首 commit 落位；回合主线时与 003 reconcile（[[gehtp-cross-session-coordination]]）
- [X] T003 [P] seq=1 编译入口排查：`scripts/export_qwen35_onnx.py` 现状 seq 参数化路径（能否产出 `[1,248320]`）；persistent socket 协议 host/C 两侧 key 对齐现状（host 发 `handle_id`/C 找 `model_id` 的当前不符点，`42_gehtp_persistent/main.c`）
  - **排查结论（2026-09-29）**：① `--seq` 已存在（`export_qwen35_onnx.py:97`），seq=1 零改动可出图，logits 实形 `[1,1,248320]`（元素数等价）；② mask/pos raw 文件不在脚本产出内，需手工备 seq=1 `.i32.raw`（各 4B，命名铁律 `wtop_emit.cpp:174-183`）；③ seq=1 **禁 `--net-json` 直通**（seq=32 dims 不可复用），须走 converter 全链（`scripts/gehtp:66-73`）；④ mask/pos 编译期固化进 blob（换 position=重编译，T018 范围）；⑤ persistent 6 bug 全部仍在且**另有 4 阻塞项**：响应缺 4B BE 长度前缀（host 必解析失败）、input_ids NULL 指针写（必 segfault）、double close、PING/PONG 未实现；⑥ `gehtp_server.py::_run_generation` 当前用 `DeviceRuntime` 占位（persistent 是死代码，`gehtp_server.py:255/:400`）；⑦ 协议 key 对照：`handle_id`↔`model_id` 错位 + host 从不发 `wtop_path`（C 永不加载 blob）

**Checkpoint**: 基线数据在手，git 干净，seq=1 入口已知

---

## Phase 2: Foundational（M=1 GEMV 快路径 + dispatch + seq=1 编译）

**Purpose**: 核心 enabler——US1（跑通 M=1）、US2（M=1 数值）、US3（M=1 加速）共同依赖；阻塞一切 story

**⚠️ CRITICAL**: 无 seq=1 blob 就无 decode 可测；无 GEMV 则 US3 性能目标为空

- [X] T004 [P] GEMV kernel：`kernels/src/runtime/oplist_exec.c` 新增 `exec_matmul_f16_gemv`（:1450 附近）——HVX f32-accumulate、M<32、stream W 行主序**不 crouton repack**、支持 `batched`/`Bn`/`t0`/`t1`/SPLIT `n0`/`N_full`（FR-001/002）
  - **实现（2026-09-29，:1453-1523）**：HVX 体内 `HVX_VectorPair` f32 累加，64 f16 列/向量，尾列走标量；`t0`/`t1` 置位时不做向量化（转置布局 W 的 N 维不连续，向量读会越界）而落标量；标量回落体索引与 `exec_matmul_f16` 原标量路径逐字段同构（batched `a[((bb*M)+m)*K+k]`、SPLIT `n0` 偏移、`N_full` 行距）
- [X] T005 dispatch：`oplist_exec.c:1580-1588` HMX 门**之前**插 `if (M<32) { if (gemv==0) return 0; }`——gemv 失败回落原标量（**全有或全无**，禁止混合路径，FR-001）
  - **实现（:1663-1672）**：分支在 ref/tmp 解析后、`M%32&&K%32&&N%32` HMX 门之前；`M>=32` 逐字节走原路径（SC-005）；`g_last_engine` 由 gemv 设置（HVX/SCALAR），optrace 可读
- [X] T006 host GEMV 验证：M=1/16/32 GEMV vs 标量参考 cos≥0.99999；batched/Bn=1、t0/t1、SPLIT n0 合成矩阵各一（compiler/tests 或独立 host 合成）
  - **落法（2026-09-29）**：独立 host 合成 `kernels/host/test_gemv_f16.c`——被测体按 `exec_matmul_f16_gemv` 标量回落的 base-pointer 写法逐字移植，参考体用绝对索引独立书写，两者对照 + `memcmp` 逐字节；14 例（M1/16/32、t0/t1 组合、batched Bn=1/2/3/4、SPLIT n0=128/256 + N_full=512、lm_head K=1024/N_full=4096 形状）全绿 cos=1.0 bit-exact
  - `gcc -O2 -std=gnu11 kernels/host/test_gemv_f16.c -o /tmp/test_gemv -lm && /tmp/test_gemv`
  - **覆盖边界**：本测试不链接 DSP runtime（`oplist_exec.c` 需 `HAP_perf.h` + `dc_*`/`wtcache_*`/`hvhx_v2_*` 桩，本树无 `host/shims/`），且 `#if __HVX__` 令主机永不走向量体——**HVX 向量体的数值验证归 T013（设备 optrace）**，本任务只钉住索引不变量
- [X] T007 seq=1 编译冒烟：`scripts/export_qwen35_onnx.py --seq 1` → converter 全链（**禁 `--net-json` 直通**，T003③）→ `hnnx_compile` → `wtop_emit` → blob 产出 `[1,1,248320]` logits（FR-003）
  - **绕道落法（2026-09-29/30）**：export 环境阻塞未解（见下），幸存 seq=1 ONNX（`/disk1/qwen35_0.8b/model_seq1_fixed.onnx` + 外置权重 + `model_net.json`/`model.params.bin` converter 产物）直通 `wtop_emit` 冒烟：**slots=1888 / ops=16212 / blob 12,372,736 B + ext weights 2,028,604,672 B**（`/tmp/seq1_copy/seq1.wtop`）
  - **host 数值门（2026-09-30）**：`host_run --net-json` 对 ORT golden（`model_seq1_fixed.onnx`, ids=178/pos=0/mask=1, top1=90637）**cos=0.999999 / top1=90637** 达门（≥0.9986/top1 一致）。过程中修 4 个 seq=1 形状角 bug（Gather 前导 1 剥除错位 / DepthWiseConv2d 居中 pad 自猜 vs 参数驱动 / **输入槽位序**: net.json tensor id 序≠JSON 键序致 pos/mask 槽对调（`host_input_order()` 声明序修复）/ **Softmax 显式 axis** 前导压缩后右对齐钳末维——layer3 首发散 cos 0.485 根因）；seq=32 老链（`--bin fcbias`）逐位一致回归通过
  - **遗留阻塞（export 环境仍缺）**：① 挂载 Qwen35dev 或准备 conda venv 跑 `export_qwen35_onnx.py`；② seq=1 mask/pos `.i32.raw` 手工生成已具备（`/tmp/s1_{ids,pos,mask}.i64.raw` 声明序喂法）；③ 上板喂 blob 输入时注意**槽位序 = id 序 ≠ 声明序**（input_ids=1, attention_mask=2, position_ids=3），喂前对 `[mdiag] in_ids` 核对

**Checkpoint**: M=1 有 GEMV 快路径 + seq=1 blob 可产出

---

## Phase 3: User Story 1 - 编译上板（Priority: P1）🎯 MVP

**Goal**: seq=1 decode 单 token 前向上板跑通，logits 正确；deploy loop 干净

**Independent Test**: seq=1 blob 上板 d0f1784，`[1,248320]` logits vs prefill 首行 cos≥0.9986/top1 一致（decode==prefill 首位置不变式）

- [ ] T008 [US1] seq=1 上板 d0f1784：单 token 前向 logits `[1,248320]` vs `cpu_golden/hf/logits_{i}[0]`（prefill 首行）cos≥0.998643/top1 一致
- [ ] T009 [US1] deploy loop 验证：`build_libs.sh`（full，勿 `one`）+ `gehtp setup --device d0f1784`（勿信 `[ -f signed.so ] || build` 短路 → stale arity），复现 prefill cos 0.998643/top1 29/32（G10）

**Checkpoint**: US1 独立交付——decode 单 token 前向上板正确，deploy loop 无 stale arity。**0.8B 双门（decode 门 + wall 对照 T001）过**。可 commit。

---

## Phase 4: User Story 2 - 精度保证（Priority: P1）

**Goal**: decode 双门判据 + per-op hook 取证链，挡住状态 bug 与 M=1 数值回归

**Independent Test**: decode 步 logits vs 设备自身 prefill 末位 cos≥0.9999/top1 精确（同权重同来源，不需 HF）

- [ ] T010 [US2] tier1 金标 helper：`scripts/golden_qwen35.py` 加切末行 helper——decode 步 logits 与设备自身 prefill **末位** logits 对拍（prefill seq=32 末位 == 第 32 token 的 logits）
- [ ] T011 [US2] judge 门接线：`scripts/judge_logits.py` 接 decode 步，双门并列——设备自洽主门（cos≥0.9999/top1 精确）+ 跨设备基线门（cos≥0.998643/top1 29/32，FR-010）
- [ ] T012 [US2] hook 取证链：`wt_exec_set_hook`/`cmp_hook.py`/`cmp_l0.py` 对 M=1 decode 步定位到具体 op（复用 [[gehtp-08b-accuracy-bisect]] 既有取证链）；异常注入（故意错一个 op）能被定位

**Checkpoint**: US2 独立交付——decode 门绿（tier1 主门），异常可定位到 op。可 commit。

---

## Phase 5: User Story 3 - 性能优化（Priority: P1）

**Goal**: M=1 全走 GEMV（无标量回落），prefill 逐字节不变

**Independent Test**: decode 步 optrace 全 GEMV；prefill（M=32/64）逐字节不变（additive 分支不碰原路径）

- [ ] T013 [US3] optrace 验证：decode 步（M=1）所有 matmul 走 `exec_matmul_f16_gemv`，无标量回落（optrace engine 列）
- [ ] T014 [US3] prefill 逐字节回归：M=32/64 prefill 路径逐字节不变（GEMV 分支 additive，不碰原 HMX 门/标量回落）

**Checkpoint**: US3 独立交付——M=1 加速落地，prefill 零回归。**0.8B 双门（cos 不回归 + wall 对照 T001）过**。可 commit。

---

## Phase 6: P2 状态化 + 端到端（US1/US2/US3 共同 P2 切片）

**Goal**: KV+SSM 状态接线 → decode 图发射 → persistent runtime → host 循环 → 端到端 autoregressive；tier2 金标；DDR 布局 + wall

**Independent Test**: socket 发 prefill+N 次 decode，权重只 load 一次、状态跨请求存活，返回每步 logits 连贯

### 状态表 + 状态 op（先于一切 P2）

- [ ] T015 [US1] 命名持久状态表（FR-004）：`oplist_exec.c` 统一 `g_kv[6]`/`g_ssm_state[18]` → `state_table[name]`；`qnn_ir_loader.cpp` 新增「state」tensor 分类（现只读 type、丢 memType，:98-133/:230-239）；编译期 TAG_MEM_PLAN/TEMPOFF 下发尺寸，运行时「查尺寸→一次分配→声明不可覆盖」；op 按名查槽
- [ ] T016 [US1] KV ops（FR-005）：`exec_kv_append`/`exec_kv_gather` 替换 `xop_kv_unimpl`（`oplist_exec.c:2049-2053,2097-2098`）；复用 `kvcache.h` ring（append-only + `kvc_set_consumer(FM_DDR,FC_DMA)`）；仿 `examples/29_kvcache` 验证 append→gather 往返逐字节
- [ ] T017 [US1] SSM state ops（FR-006）：`OP_CONV1D_SSM_STEP_F16`→`conv1d_step_f16`、`OP_DELTA_CHUNK_F16`→`delta_chunk_f16`（`gdn_sm.h`），state-in/out；验证「状态跨 2 chunk == 单 chunk oracle」（复用 `43_gdn_sm_08b` 一致性）

### decode 图发射 + persistent

- [ ] T018 [US1] decode 图发射：`scripts/export_qwen35_onnx.py` decode 变体（use_cache、KV append/gather + SSM step op 图）+ `compiler/src/ir/graph_prepare.cpp`/`compiler/tools/wtop_emit.cpp` state tensor 分类 + 状态尺寸下发（R3/R5 绿地，先做 host 执行验证）
- [ ] T019 [US1] persistent runtime（FR-007）：`kernels/examples/42_gehtp_persistent/main.c` 修 6 处硬 bug——`load_blob_from_path` 补 `wt_parse`+buf 挂载（现只读不进还泄漏 :118-143）、`parse_and_exec` 去自覆盖接 `wt_exec_run_io`（:145-160）、input_ids 栈数组+`strtol` 多数字解析（:185/:214-227）、`MAX_LOGITS 32768→248320`（栈数组改静态/堆）+`MAX_OPS 1024→16384`、协议 key 对齐（host 发 `handle_id`/C 找 `model_id`；host 补发 `wtop_path`）、token-id→host 侧 embedding 注入（FR-008，G7）；**T003 补充 4 阻塞项并入**：响应补 4B BE 长度前缀（:88-106）、double close（:104/:275）、PING/PONG、socket 路径统一（main.c:34 vs scripts/gehtp:21-22）
- [ ] T020 [US1] host 循环接线：`scripts/persistent_runtime.py`/`device_runtime.py`/`gehtp_server.py::_run_generation` 端到端生成连贯 token；socket 冒烟 prefill→N decode 不 reload、状态存活

### 精度 + 性能收口

- [ ] T021 [US2] tier2 金标：`scripts/golden_qwen35.py::run_decode`（HF `use_cache=True` 续写多步；blocked on R1——HF 对 GDN/SSM 层是否缓存 recurrent state 未验证，若否改 `use_cache=False` 每步全序列重跑）
- [ ] T022 [US3] KV/SSM DDR 布局（FR-009）：状态落 DDR（KV 12KB/token、SSM ~19MB），不挤 VTCM 16MB（VTCM 留给 HMX scratch）；上板确认
- [ ] T023 [US3] 端到端 wall：autoregressive 每步 wall 记录对照（目标：单步 decode 远小于 prefill 34s，lm_head 步不再全标量 4.24s）

**Checkpoint**: P2 全绿——`42_gehtp_persistent` 每 socket 请求跑单 token 前向（权重+KV+SSM 常驻），`gehtp serve` 端到端 autoregressive。**0.8B 双门（decode 门 + wall 对照）过**。可 commit + push。

---

## Phase 7: Polish & 收尾

- [ ] T024 [P] 更新记忆文件：[[gehtp-08b-baseline-002]] 补 decode 进度；新建 decode 状态记忆（若 T015-T023 落地）
- [ ] T025 [P] 更新 docs/GEHTP_PARALLEL_TASKBOOK.md 战役日志（decode 两阶段结果 + 各阶段 wall 数据）
- [ ] T026 全量回归：prefill 逐字节不回归 + decode 门（tier1+tier2）+ host 合成三连，确认 SC-001~008 全绿；commit + push（SSH 直连，[[gehtp-github-access]]）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，立即开始
- **Foundational (Phase 2)**: 依赖 Phase 1 基线；**阻塞所有 story**（无 seq=1 blob 无 decode 可测；无 GEMV 无性能）
- **US1 (Phase 3)**: 依赖 Foundational；**P1 MVP，独立可交付**
- **US2 (Phase 4)**: 依赖 US1（要有 seq=1 logits 才能对拍）
- **US3 (Phase 5)**: 依赖 Foundational（GEMV 已就绪）；建议 US1 后做（optrace 要在设备上跑 decode 步）
- **P2 (Phase 6)**: **硬依赖 US1 P1**（要有能在设备跑的 M=1 前向才谈得上接 KV/SSM 状态）；T021 依赖 T020（端到端）；T023 依赖 T022（DDR 布局）
- **Polish (Phase 7)**: 依赖全部

### Within Each Story

- Foundational 的 GEMV host 验证（T006）先于上板
- 状态表（T015）先于 KV/SSM op（T016/T017）——op 要按名查槽
- decode 图发射（T018）先于 persistent（T019）先于 host loop（T020）
- host 验证过才上板；设备任务先查共享冲突
- **每阶段收口必过 0.8B 双门（精度 cos + 性能 wall）**

### Parallel Opportunities

- T003 与 T001/T002 可并行
- T004（GEMV kernel）与 T007（seq=1 编译）可并行（不同文件）
- T016（KV ops）与 T017（SSM ops）可并行（不同 op，都依赖 T015）
- T024 / T025 可并行

## Implementation Strategy

### MVP First（P1 = M=1 单步打通）

1. Phase 1 Setup → Phase 2 Foundational（GEMV + seq=1）→ Phase 3 US1 完成 → **停下来过双门**（T008/T009）
2. 单独 commit：decode 单 token 前向上板正确，风险最低的一步先落袋
3. 叠 US2（精度门）+ US3（GEMV 加速）→ 双门 → commit

### 增量交付

1. P1（Foundational+US1/US2/US3）→ commit（MVP：M=1 单步上板 + 精度门 + 加速）
2. P2 状态表 + KV/SSM → commit（状态 op 就绪）
3. P2 decode 图 + persistent + host loop → commit（端到端生成）
4. P2 tier2 金标 + DDR + wall → commit（精度 + 性能收口）
5. Polish → push

### 风险集中点

- T018（decode 图发射）是全套件的风险阀门：state tensor 分类是 emit 层无先例的绿地（R3），先做 host 执行验证，fail 则回退到「per-op 手写状态表 + 运行时手动初始化」的简化路径
- T019（persistent runtime 6 bug）是端到端的咽喉，逐个 bug 修+测，勿一次性改完
- R4（token-id >2048 错嵌）必须在 T019/T020 前解决，否则生成的 token 会静默错
- R7（GEMV ulp 微扰）若 tier1 主门 cos<0.9999，先 host 对拍 GEMV vs 标量精确定位

---

## Notes

- 每完成一个 Checkpoint commit 一次；设备验证任务遵循 [[gehtp-p5-device-gate]] 惯例（签名 lib、查共享、备份 job.txt）
- kernels 改动 host 验证：先 `source env.sh` 再 `build_libs.sh`（full，勿 `one`）；compiler 单 target host 重编再跑，勿信残留容器二进制（[[gehtp-build-env]] / [[gehtp-device-lib-stale-arity]]）
- 数值门一律逐字节/cos 判据，不接受「看起来对」
- 命名状态表 name↔buffer 映射（R5）在 oplist_parse.h 与 emit 侧同一 name 枚举常量定义，round-trip 契约测试兜底
