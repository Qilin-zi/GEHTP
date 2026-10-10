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

- [X] T008 [US1] seq=1 上板 d0f1784：单 token 前向 logits `[1,248320]` vs `cpu_golden/hf/logits_{i}[0]`（prefill 首行）cos≥0.998643/top1 一致
  - **绿（2026-10-08，d0f1784，库指纹 a42720a5）**：`seq1.dev4.wtop`（直通 emit，`/tmp/seq1_copy/`）设备单发 → vs HF `logits_0[0]` **cos=0.999960 / top1=90637 精确一致**（top5 全同）；decode==prefill-首位置不变式 cos=0.999981。远超 0.998643 门
  - **根因双 bug（修复后达标）**：
    ① **emit Bn 剪尾**（`op_matmul.cpp`）：attn q·kᵀ 输出 `[1,8,1,1]`（M=N=1）被 `while(dims[rk-1]==1) rk--` 剪掉真实批维 8 → Bn=1 → 执行器只写 1/8 元素，其余 7 头读陈旧 temp（首发散 gid 4560/idx 2711，顺流污染全部 6 个 full-attn 层 + GDN scan 回路）。修法：批维 = 除末两维外全部前导维之积，不裁剪
    ② **GEMV HVX 向量体数值错**（`oplist_exec.c`）：K=1/N=256/Bn=8 实锤输出 = W 偶元素 stride-2 抽样（`hvx_vec_mpyacc_f32_f16` pair-interleave × `hvx_vec_f32_to_f16` 的 vdeal 组合不匹配；对照 `hvx-mm-kernels-flat.h` vec_dot 收尾是 lo+hi 归约无逐元素回写先例）。`#if __HVX__` 体 host 编译不参与 → host 单测覆盖不到。**禁用（`if (0 && ...)`）全走标量**（标量公式已 numpy 逐位证实），设备侧单测补齐前不回开
  - **取证链**：host netjson dump（`/tmp/s1diag5` 14697 张量）+ 设备 hook（`wt_exec_set_hook` 分段下发，1023B 截断上限）+ last-of-gid 对拍（code-26 分解中间步非检查点）+ 层哨兵阶梯（L0/L1/L2 exact → L3 起 BAD 定位窗口）
  - **归因修正（T012 异常注入反证）**：①Bn 剪尾单独在 seq=1 是端到端**良性**的（softmax 单元素退化掩蔽，见 T012）——旧库端到端崩的即时数值杀手是 ②GEMV 向量体（probs·v 的 v 抽样错）；①是潜伏形状 bug（seq>1 爆发）+ hook SIZE 签名，②是即时数值 bug。两修都对，但取证判据不同（SIZE vs BAD-cos）
- [X] T009 [US1] deploy loop 验证：`build_libs.sh`（full，勿 `one`）+ `gehtp setup --device d0f1784`（勿信 `[ -f signed.so ] || build` 短路 → stale arity），复现 prefill cos 0.998643/top1 29/32（G10）
  - **绿（2026-10-08，d0f1784，新库 a42720a5 = full build + setup 重推）**：prefill v17 blob（`qwen35_08b_v17.wtop`，i32 ids=prompt0）→ **cos=0.998642 / top1=30/32 / exec 34.1s / wall 37.6s**；历史基线（9/28 旧库产物 `output_v17bhmx_0.f16.raw`）cos=0.998646/29/32。Δcos=4e-6（f16 噪声级）、top1 反升、wall 持平（基线 33.9s）——**无回归**
  - **Δ来源（已知且良性）**：v17 全部 175 个 matmul_f16 均 M<32 → 旧库部分走 GEMV HVX 向量体（N≥64 的 !t0/!t1 分支），新库禁用后走标量 → row0-4 逐字节同、position 5 起微差（HVX 向量体与标量 f32 累加的舍入差）；这正是 T008 ② 修复的副作用，设备侧 GEMV 单测补齐后回开可收回 +0.2s
  - **口径修正（T013 轮复核，parse_wtop 真源）**：上条归因不成立——v17 的 matmul_f16（opcode 24）实为 **325 个、M 直方图 {32:199, 64:126}，M<32 计数 = 0**；"175 个"实为 OP_TRANSPOSE_GEN_F16=27 的条数，"M<32"是把 args[3]（d0 维度）误当 M。v17 从不进 GEMV（M<32）分支，新旧库在该图上 matmul 路径逐字节相同 → Δ 4e-6 是跨构建 f16 噪声（T011 族带宽 ±4e-6 内），非 GEMV 禁用副作用。T014 已实证 GEMV 回开后 v17 逐字节不变，双重确认 additive

**Checkpoint**: US1 独立交付——decode 单 token 前向上板正确，deploy loop 无 stale arity。**0.8B 双门（decode 门 + wall 对照 T001）过**。可 commit。

---

## Phase 4: User Story 2 - 精度保证（Priority: P1）

**Goal**: decode 双门判据 + per-op hook 取证链，挡住状态 bug 与 M=1 数值回归

**Independent Test**: decode 步 logits vs 设备自身 prefill 末位 cos≥0.9999/top1 精确（同权重同来源，不需 HF）

- [X] T010 [US2] tier1 金标 helper：`scripts/golden_qwen35.py` 加切末行 helper——decode 步 logits 与设备自身 prefill **末位** logits 对拍（prefill seq=32 末位 == 第 32 token 的 logits）
  - **绿（2026-10-08）**：`--cut-row R` 模式（numpy-only 裸环境可用，不拉模型）+ `load_golden_row()` 落地；产物 `cpu_golden/hf/logits_0_row0.f32.raw`（top1=90637，**P1 行锚 = row 0**，SC-001）与 `logits_0_row31.f32.raw`（top1=197，**P2 行锚 = row 31**，SC-002 真 decode 步）双锚齐备
  - 顺带修复：模块级 `from export_qwen35_onnx import QPM_ROOT` 断链（export 转 transformers≥4.57 后该常量已删，import 期即崩）→ lazy import 化；`--model-dir` 改可选（切行模式不需要）
- [X] T011 [US2] judge 门接线：`scripts/judge_logits.py` 接 decode 步，双门并列——设备自洽主门（cos≥0.9999/top1 精确）+ 跨设备基线门（cos≥0.998643/top1 29/32，FR-010）
  - **绿（2026-10-08，d0f1784）**：`judge_logits.py <out> --row 0 --device-prefill <v17_out>` → 主门（dev vs dev-prefill 行0）**cos=0.999981 / top1 精确**；基线门（dev vs HF 行0）**cos=0.999960 / top1=90637 精确**，双 PASS。prefill 门复验：v17 新库 cos=0.998642/30/32 PASS
  - 门常量定夺：`COS_BASELINE=0.99864`——FR-010 字面 0.998643 为单次记录值，同基线族实测 0.998642/45/46（跨构建 f16 噪声带 ±4e-6），字面门会误杀族内合法复现，取族下沿
- [X] T012 [US2] hook 取证链：`wt_exec_set_hook`/`cmp_hook.py`/`cmp_l0.py` 对 M=1 decode 步定位到具体 op（复用 [[gehtp-08b-accuracy-bisect]] 既有取证链）；异常注入（故意错一个 op）能被定位
  - **绿（2026-10-08，d0f1784）**：`cmp_hook.py`（新增 **SIZE 判据**：dev/host 元素数不匹配 = Bn/batch/形状 bug 签名）与 `cmp_l0.py`（`gehtp_prof.parse_wtop` 重写，last-writer 溯源自动挑检查点，去外部二进制依赖）参数化入树
  - **异常注入（dev3/dev4 理想对：同图同 n_ops=16212 同 gid，唯一差异 op2711 flags 0x10004=Bn1 vs 0x80004=Bn8）**：设备轮 hook 链 `2695:263 → 2711:183 → 2714:9` = RmsNorm OK(cos 0.999999) → **MatMul op2711 SIZE（dev 1 元素 vs host 8 元素，cos=-1.0）** → softmax cos=1.000000；双工具一致定位**首个 BAD = op2711**
  - **关键发现——端到门盲区**：dev3（Bn=1）输出过全部门（主门 0.999981/基线门 0.999960 双 PASS）——seq=1 softmax 对单元素归一恒 1.0，scores 数值被**数学抹除**，异常不传导 logits（op2714 cos=1.0 精确佐证）。结论：端到门对此类形状/状态 bug 盲，per-op hook SIZE 判据补盲——正是 US2 取证链的存在理由；Bn 截断危害是**潜伏的**（seq>1 scores 参与归一时爆发），修复仍必须（T008 ①）

**Checkpoint**: US2 独立交付——decode 门绿（tier1 主门），异常可定位到 op。可 commit。

---

## Phase 5: User Story 3 - 性能优化（Priority: P1）

**Goal**: M=1 全走 GEMV（无标量回落），prefill 逐字节不变

**Independent Test**: decode 步 optrace 全 GEMV；prefill（M=32/64）逐字节不变（additive 分支不碰原路径）

- [X] T013 [US3] optrace 验证：decode 步（M=1）所有 matmul 走 `exec_matmul_f16_gemv`，无标量回落（optrace engine 列）
  - **GEMV 向量体修复 + 回开（2026-10-08，d0f1784，全量重建库）**：T008 禁用的 `!t0&&!t1` 体回开 + 新增 t1 vec_dot 体（`oplist_exec.c::exec_matmul_f16_gemv`）
  - **根因实锤——native mpyacc acc 布局是「连续两半」**：`Q6_Wsf_vmpyacc_WsfVhfVhf`（v81 构建）的 acc = lo=第 0..31 列输出、hi=第 32..63 列，正确 epilogue = 纯 concat（`hvx_vec_f32_to_f16_shuff(lo,hi)`）。三观测自洽：vdeal → 偶元素 stride-2 抽样（T008 实锤）、vshuff → [w0,w32,w1,w33..] 交错（本轮实锤）、shuff 无 shuffle → **2048/2048 元素精确匹配**。教训：`<79` 模拟路径（Q6_Wqf32_vmpy）是 pair-interleave，两代指令布局不同，**勿以模拟码推 native 语义**
  - **t1 体**：W 存 [N,K]（输出行 K 连续）→ 逐输出列 vec_dot：mpyacc 累加 → `HVX_OP_ADD_F32(lo,hi)` 通道加 → `hvx_vec_reduce_sum_f32` 水平归约（hvx-fa-kernels.h vec_dot 同款收尾）；对 lo/hi 布局不敏感（求和无序），一次即对；K 尾数标量免 OOB
  - **数值验证**：杀手形状 op2718（K=1/N=256/Bn=8）hook cos **0.999999**；全部 M=1 t1 FC OK（lm_head op16211 K=1024/N=248320 cos 0.999974）；seq1 judge 双门 **PASS 且优于标量版**——主门 0.999987（标量 0.999981）/ 基线门 0.999970（标量 0.999960），top1 全精确
  - **optrace（op_ts.bin ver=2 engine 列）**：M<32 matmul **200/200 engine=HVX**（WT_ENG_HVX=1），无标量回落（scalar 参照轮 200/200=SCALAR）；该类 op 总耗时 **19.66s→0.69s（28.5×）**；lm_head op16211 单 op **6.86s→240ms**；seq1 整图 **wall 36.72s→17.74s（2.07×）**（exec 17.74s / 端到 wall 21.2s）
  - **取证教训**：hook 对拍 BAD 时先验证参照链自洽——本轮 host_id_4560 与 host 自身 4557(q)/4559(k) 矛盾（cos 0.644），离线证明 dev 输出 == q_dump·k_dump 精确一致后判定参照坏而非设备错；T008"首发散 gid 4560"部分是坏参照伪影
- [X] T014 [US3] prefill 逐字节回归：M=32/64 prefill 路径逐字节不变（GEMV 分支 additive，不碰原 HMX 门/标量回落）
  - **绿（2026-10-08，d0f1784）**：v17 prefill（`qwen35_08b_v17.wtop`）用 GEMV 回开后的同一新库跑 → 输出与 T009 基线（GEMV 禁用库）**逐字节一致**（cmp 全同）；judge prefill 门 PASS **cos=0.998642 / top1=30/32**
  - **结构性解释**：v17 的 325 个 matmul_f16 全 M=32/64（见 T009 口径修正）→ GEMV 分支（M<32 门）从不进入，字节一致是构造使然 + 实测确认，additive 成立

**Checkpoint**: US3 独立交付——M=1 加速落地，prefill 零回归。**0.8B 双门（cos 不回归 + wall 对照 T001）过**。可 commit。

---

## Phase 6: P2 状态化 + 端到端（US1/US2/US3 共同 P2 切片）

**Goal**: KV+SSM 状态接线 → decode 图发射 → persistent runtime → host 循环 → 端到端 autoregressive；tier2 金标；DDR 布局 + wall

**Independent Test**: socket 发 prefill+N 次 decode，权重只 load 一次、状态跨请求存活，返回每步 logits 连贯

### 状态表 + 状态 op（先于一切 P2）

- [X] T015 [US1] 命名持久状态表（FR-004）：`oplist_exec.c` 统一 `g_kv[6]`/`g_ssm_state[18]` → `state_table[name]`；`qnn_ir_loader.cpp` 新增「state」tensor 分类（现只读 type、丢 memType，:98-133/:230-239）；编译期 TAG_MEM_PLAN/TEMPOFF 下发尺寸，运行时「查尺寸→一次分配→声明不可覆盖」；op 按名查槽
  - **落地形态（FR-004 实现取舍）**：blob 内紧凑 idx 而非按名查槽——`WT_SLOT_STATE=0xFFFFFFFB` 槽（槽数据 `[n u32][n×{bytes u32}]`，n≤64）+ op 引用编码 `0x6000|state_idx`（与 `0x8000|slot`/`0x4000|VTCM-temp` 同族不重叠：VTCM 值域 [0x4000,0x4FFF]，`(arg & 0xE000)==0x6000` 精确判定）；名称留 emit 侧 manifest json（R5 的 name↔buffer 映射归 T018 emit 层同口径定义）。运行时 init-once 装载（`wt_exec_load_states`：memalign(128)+零初始化，无 state 槽也置位），生命期=进程级——`wt_exec_shutdown` 不回收，新序列用新 API `wt_exec_state_reset()` 清零+重建 ring；`run_range` 入口接线。IR/loader 侧 state tensor 分类（`qnn_ir_loader.cpp`）归 T018（emit 侧绿地）。
- [X] T016 [US1] KV ops（FR-005）：`exec_kv_append`/`exec_kv_gather` 替换 `xop_kv_unimpl`（`oplist_exec.c:2049-2053,2097-2098`）；复用 `kvcache.h` ring（append-only + `kvc_set_consumer(FM_DDR,FC_DMA)`）；仿 `examples/29_kvcache` 验证 append→gather 往返逐字节
  - **实现**：`state_kvc()` 惰性几何推导（slot_bytes=kv_elems·2·2 须 ≥256 且 128 倍数、nslots=st_bytes/slot_bytes 整除，append/gather 双侧同式，不一致硬错误）；append `[k,v,state_ref,pos]` 写 pos%nslots 槽 K|V 双面；GATHER 实义化 arity 7 `[state,kT_out,v_out,n_kv,n_heads,kv_heads,head_dim]` 出 GQA repeat_interleave 展开双面（kT=[heads,hd,n_kv] 供 batched scores 的 W、v=[heads,n_kv,hd] 供 probs·V），槽 miss=硬错误。consumer 设 FC_CPU/FM_DDR（CPU memcpy 语义，暂无 DMA 需求）。
  - **验证**：①host `kernels/host/test_kv_ssm_ops.c`（链真 kvcache.c+fence.c）：append 8 pos→GQA 展开往返**逐字节** vs 独立下标重建，含回绕语义（满槽覆写后旧 pos 自动 miss）全绿；②设备合成 blob（下条）双面 16384B **逐字节一致**。
- [X] T017 [US1] SSM state ops（FR-006）：`OP_CONV1D_SSM_STEP_F16`→`conv1d_step_f16`、`OP_DELTA_CHUNK_F16`→`delta_chunk_f16`（`gdn_sm.h`），state-in/out；验证「状态跨 2 chunk == 单 chunk oracle」（复用 `43_gdn_sm_08b` 一致性）
  - **实现**：opcode 29 `[x_ref,w_s,y_t,state_ref,C,d_conv]`（**kernel 内含 SiLU**，≠op19 纯卷积——decode 图发射端不接 Sigmoid/Mul）；opcode 30 `[k,v,q,beta,g,y_t,state_ref,c,ntok,H,D]`（S=[H·D·D] f32 原地，q/√D 在 kernel，c≤ntok≤GDN_C_MAX=64 校验）。占用 29/30（本树与主线均空闲；主线 003/005 已占 36，回合不撞）。
  - **验证**：①host 同测试文件：conv ×3 vs 独立 oracle（窗序+SiLU+窗推进 f32 位级 memcmp）全绿；delta 跨 2 chunk (c=2→1) vs 单 chunk (c=3) cos=1.000000/maxdS=1.5e-4 + vs ref_delta_token per-head oracle cos=1.000000（oracle 陷阱：`st->s` 是单 head [d,d] 面，须逐 head 独立 rec_state_t）。测试侧曾 3 bug（f16 位模式拷进 float 窗/cos 只除 |A|/共享状态基址）——修后全绿，kernel 无恙。
  - **设备门（T015/16/17 合并，d0f1784）**：python 手写 wtop 发射器（/tmp/mk_state_blob.py，12800B blob：EXT_IN+9 常量槽+STATE 槽(4 状态: KV 16KB/conv 1152B/delta×2 4KB)+27 ops=3 slice×x + conv_step×3 + delta c=2→1 vs c=3 + append×4 + gather）→ `gehtp run` + hook 8 点对拍 python 期望：**conv×3 cos=1.000000 maxd=0、delta A/B vs f64 oracle cos 均 1.000000、A vs B 跨 chunk 一致性 cos=1.000000、KV kT/v 双面逐字节一致、output 回传 cos=1.000000 == 9/9 全绿**。途中修 2 bug：发射器状态表漏第 4 项（n=4 只打 3 项→parse 硬拒，符合设计）；`exec_conv1d_ssm_step` 把 `0x8000|slot` 整值传 `slot_ptr`（须 `ref_ptr`+len 校验剥旗标）——上板首跑即炸、修后全绿（合成 blob 硬门的价值实证）。期望值侧 1 bug：f16→u16 用 `astype`（数值截断）而非 `view`（位重释），设备输出实际正确、期望错——对拍脚本修后 9/9。

### decode 图发射 + persistent

- [x] T018 [US1] decode 图发射——**2026-10-09 完成**（host `55aeb8d` + 设备 `2789fd9`）：`scripts/surgery_decode_graph.py` 把 prefill net.json 手术成 decode 图（6 新 op：DecodeFaceF16/KvAppendF16/KvGatherKT/V/Conv1dSsmStepF16/DeltaChunkF16；42 状态槽 ≈21MB；双输入合并 388B 面 [ids i32 ×1 | face i32 ×96]）。host 双路径 `--bin` vs net.json/golden **cos=1.000000、top1=90637 一致**；T018e 32 步 host 循环 ALL GREEN；设备（t18f runner steps 模式）32 步 cos 全 ≥0.9986（最低 0.999885 无漂移=状态积累正确）、top1 31/32（step6 f16 尾部近并列）、≈1.08s/步。三大根因：① tensor_param_ids 不序列化（tagged.bin 尾随序列化修复，向后兼容双方向）；② rebuild decode 分支无条件 return 跳过 op_data 共享赋值；③ ref_elem_count 表外 temp 复用高估（改用 g_last_bytes）。state_idx 全局重映射 kv→0..5/conv→6..23/delta→24..41 双侧同口径（`decode_state_gid`）。
- [x] T019 [US1] persistent runtime（FR-007）——**2026-10-10 完成，架构修正后整体重写**：原 socket 设计在 qurt 上编译不可能（posix 层无 sys/socket.h；run_main_on_hexagon 生命周期=一次 FastRPC 调用内 main() 存活）→ 改为 **DSP 常驻文件队列 runner**（`42_gehtp_persistent/main.c` 全重写）：blob/权重/状态一次装载，轮询 req 文件→执行→写 resp 文件，host 侧 `scripts/persistent_runtime.py` 经 adb 文件往返驱动。协议帧与原 socket 设计同形（4B BE 长度前缀+JSON+二进制体、ping/reset/shutdown/run、无双 close、socket 路径问题随架构消失）；原 6 处硬 bug 随重写全消（wt_parse 挂载/run_io 接线/JSON 多数字解析/vocab 级 argmax/key 对齐）；FR-008 embedding 注入被 T018 手术天然覆盖（i32 ids 直接喂 embedding Gather）。**设备门全绿（d0f1784）**：32 步 persistent decode 31/32 top1（step6=t18f 已知并列）、cos 全 ≥0.999885、reset 复现 step0 位级一致、wall 41s/32 步（含 adb 往返 1.28s/步，设备 1.10s/步）。**三根因**：① optrace 默认开致步时 2.66s（job `trace` 键默认 0，2.4×）；② 4B BE 前缀致 strstr 空 C 串（take_request lseek(fd,4)）；③ **非对齐输入指针进 run_io 杀 DSP 进程**（req+4+hdr_len 对齐随 JSON 头长漂移；runner 喂 malloc 8B 对齐从不出事——输入体先拷 128B 对齐静态缓冲再喂）。运维铁律：kill 留幽灵 DSP 会话（仍应答文件请求数十秒后被 reaper 回收，期间新实例 -2147482611）；`ps -A` NAME 是 15 字符 comm，判活必须 `pgrep -f`；setup 重推 .so 前先 pkill。构建并入 `gehtp setup`（[3/5] persistent target）。
- [x] T020 [US1] host 循环接线——**2026-10-10 完成，设备 E2E 全绿（d0f1784）**：
  - **接线四件**：`PersistentRuntimeError` 子类化 `DeviceRuntimeError`（服务层统一捕获）；`device_runtime.get_runtime()` 按 `GEHTP_PERSISTENT=1` 选 persistent 后端（默认仍原型占位 → host 测试零设备依赖）；`persistent_runtime.init()` 懒启动 DSP 常驻 runner（env `GEHTP_DECODE_WTOP`/`GEHTP_DECODE_OUT_TEMP`/`GEHTP_PERSIST_IDLE_S`，`start_server` 幂等复用）；lifespan 退出调 `runtime.shutdown()`（实测 SIGTERM→shutdown op→DSP 进程消失）。
  - **`_run_generation` 重构**（decode 图单 token/步）：prompt 相位逐 token 喂入（中间 logits 丢弃，末位 prompt token 输出=首生成 token）；生成相位惰性取步（stop 提前返回不多跑一步）；生成前 `reset()`=新序列清 KV/SSM（persistent 状态跨请求存活故必须）；生成计数==max_tokens 契约保持（contract 测试零改动全绿）。
  - **注册 API 增 `input_length` 覆盖**：decode manifest `input_elems=1` 是单步输入非上下文上限（首跑即被 max_tokens≤1 拒）；显式传 64（=编译期 MAX_POS，设备侧 run() 尚有 position≥MAX_POS 硬闸兜底）。
  - **测试**：`test_persistent_runtime.py` 从 socket 契约整体重写为文件队列契约（face 位级不变式、run 请求体==make_step_input、shutdown 优雅/kill 双路径、env 后端选择）30/30；全套 **65/65**。
  - **设备 E2E**（`GEHTP_PERSISTENT=1` 起 serve）：注册 21s 懒启动（含 2GB blob 装载）；`"user: Hello"`→`", I want to know how the \`nltk"`；`"user: The capital of France is"`→`" Paris."`（语义正确；B 紧跟 A 请求=reset/状态隔离生效）；流式与非流式逐字一致；finish=length、completion_tokens==max_tokens；≈1.15s/步。**直连链对拍 PASS**：serve 输出 == `PersistentDeviceRuntime` 直驱镜像链（同 tokenizer 编码/同位置序列），A/B 双 prompt 位级一致。

### 精度 + 性能收口

- [x] T021 [US2] tier2 金标——**2026-10-10 完成，设备门双 PASS（d0f1784）**：**R1 解除**——GPU 对拍实证 HF Qwen3_5 `DynamicCache` 正确携带 GDN/SSM recurrent state（增量 32 行 vs 全序列行 cos≥0.999999、argmax 全同；金标环境 `PYTHONPATH=/disk1/qnn08b_official/tfenv` + llmserv torch，plan 期 QPM_ROOT_DEFAULT 路径已失效）。`golden_qwen35.py::run_decode`（增量贪心续写，返回 rows+tokens）+ CLI `--decode N --decode-prompt I`；金标落 `cpu_golden/hf/tier2_0.{logits.f32,tokens.i64}.raw`（random prompt0 32+16 步）+ `text_tier2.*`（"user: The capital of France is" 7+12 步）。**设备门**：persistent 链 reset→喂 prompt→贪心续写逐行对拍 tier2——random **16/16 token 一致、worst cos=0.999991**；text **12/12 一致、worst cos=0.999994**（" Paris"=11751 起逐 token 全对，含重复结构；与 T020 E2E serve 输出同串）。decode 步越过 prompt 边界仍与 HF 位级对齐 = KV+SSM 状态跨步积累正确（tier1 门只覆盖 prompt 内）。
- [x] T022 [US3] KV/SSM DDR 布局（FR-009）——**2026-10-10 确认（代码路径 + blob 扫描 + 设备门佐证）**：`wt_exec_load_states`（oplist_exec.c:2071）经 `memalign(128, bytes)` 落 **CDSP DDR 堆**（DSP C 库堆；VTCM 只能经 `g_exec.vtcm_pool` wtcache 显式开——状态代码从不触碰），init-once 进程级生命期、`wt_exec_state_reset` 清零重建 ring。decode.wtop STATE 槽扫描（`gehtp_prof.parse_wtop` + oplist_parse.c:139 同式 weight_off）：**n=42、total 20.02 MiB**——kv 6×131072B（**=12.0 KiB/token @ MAX_POS=64**，G4 精确吻合）+ conv 18×73728B（3×6144 f32，G5 吻合）+ delta 18×1MiB（16×128×128 f32，G5 吻合）。上板佐证 = T019/T021 设备门（状态跨 32/48 步累积位级正确，此布局实跑）；20MiB ≪ CDSP 堆 2-2.9GB（008 探针），VTCM 16MB 留给计算暂存不受挤。
- [x] T023 [US3] 端到端 wall——**2026-10-10 记录（d0f1784，与 T019/T020/T021 同批测量）**：
  - prefill 基线（T009，v17 seq=32 单发）：exec **34.1s**、wall 37.6s ≈1.07s/token。
  - persistent decode 步：设备 exec **1.10s/步**（optrace 关，job `trace` 默认 0；trace 开 2.66s 勿混口径）；含 adb 文件队列往返 **1.28s/步**（T019 门 41.0s/32 步）；serve 端到端 ≈**1.15s/步**（T020 E2E 17.0s/(3+12 tok)）。
  - **门判**：单步 decode 1.10s ≪ prefill 34.1s（**31×**）✓；lm_head 在 1.10s 步内算术上不可能为 4.24s 全标量（US1 split 门：M=1 全走 GEMV 无标量回落，seq1 wall 36.7→17.7s 28.5×）✓。
  - 诚实注记：per-token 口径 decode 1.10s ≈ prefill 1.07s/token（本阶段赢在权重一次装载+状态常驻+流式，非 per-token 加速）；每步 adb 往返 ~0.18s（占 16%），AP 侧 socket 桥若 wall 需要再立项（T019 备忘）。

**Checkpoint**: P2 全绿——`42_gehtp_persistent` 每 socket 请求跑单 token 前向（权重+KV+SSM 常驻），`gehtp serve` 端到端 autoregressive。**0.8B 双门（decode 门 + wall 对照）过**。可 commit + push。

---

## Phase 7: Polish & 收尾

- [x] T024 [P] 更新记忆文件：[[gehtp-08b-baseline-002]] 补 decode 进度；新建 decode 状态记忆（若 T015-T023 落地）——**2026-10-10 完成**：新建 [[gehtp-004-decode-feature-complete]]（总状态+门锚速查）、[[gehtp-004-t021-tier2-golden]]（R1 解除+金标环境）、[[gehtp-004-t020-serving-wiring]]、[[gehtp-004-t019-persistent-filequeue]]；002 基线锚补 decode 进度段；MEMORY.md 索引同步。
- [x] T025 [P] 更新 docs/GEHTP_PARALLEL_TASKBOOK.md 战役日志（decode 两阶段结果 + 各阶段 wall 数据）——**2026-10-10 完成**：追加 004 线战役条目（P1/P2 战果、tier2 金标、各阶段 wall、架构定论、运维铁律、commits 清单）。
- [x] T026 全量回归：prefill 逐字节不回归 + decode 门（tier1+tier2）+ host 合成三连，确认 SC-001~008 全绿；commit + push（SSH 直连，[[gehtp-github-access]]）——**2026-10-10 回归完成，SC-001~008 全绿（push 待用户确认）**：
  - **host 套 65/65**（含 persistent 文件队列契约 30/30、contract/integration）。
  - **v17 prefill 门复验（当前含状态 op 的库）**：d0f1784 跑 seq=32 v17 → vs HF golden **cos=0.998642 / top1=30/32 / exec 33.75s**——与 T009 基线**逐位复现到第 6 位小数**，prefill 路径对 T015+ 状态 op **零回归**（SC-003/005 绿）。注记：与 Sep-28 `.scratch_eqfix` 旧输出字节不一致系跨构建谱系（该血统未证），数值门为正典判据。
  - **decode 门**：tier1（T018f/T019 31/32+cos≥0.9986）+ tier2（T021 16/16+12/12 token 位级，worst cos≥0.999991）均绿。
  - **SC 盘点**：SC-001/002 精度门绿、SC-004 GEMV 无标量回落（T014 回开实证）、SC-005 prefill 不变（本次复验）、SC-006 状态 DDR（T022）、SC-007 wall 对照（T023）、SC-008 双门每阶段过。

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
