---
description: "Feature spec: 0.8B decode（autoregressive 单 token 生成）编译上板 + 精度保证 + 性能优化"
---

# Feature Specification: 0.8B decode 上板

**Feature Branch**: `004-decode-on-device`

**Created**: 2026-09-29

**Status**: Draft

**Input**: 用户要求「spec 三件套规划 0.8B decode 部分编译上板、精度保证、性能优化」，并确认：① 完整含 KV cache，分两阶段（P1=M=1 单步打通，P2=KV+SSM 状态+persistent+端到端）；② 一个 feature spec、三个 P1 故事。

## 背景与问题

0.8B（Qwen3.5-0.8B）目前设备端只有 **prefill**（单发 seq=32 全图前向），**autoregressive decode（逐 token 生成）是纯 stub**：

1. `42_gehtp_runner` 只做单发 `wt_exec_run_io`，无逐 token 循环、无 KV 状态
2. `42_gehtp_persistent` 的 `parse_and_exec()` 返回占位 token 0（`main.c:145-160`），`load_blob_from_path` 不调 `wt_parse`（`:118-143`），MAX_LOGITS=32768/MAX_OPS=1024 偏小
3. KV 操作码 `OP_KV_APPEND_F16=22`/`OP_KV_GATHER_F16=23` 是 `xop_kv_unimpl`（`oplist_exec.c:2049-2053,2097-2098`）
4. host 侧逐 token 循环在 `scripts/gehtp_server.py::_generate()`，但底层 runtime 返回硬编码 0

**头号性能陷阱**：HMX 快路径门 `M%32==0 && K%32==0 && N%32==0`（`oplist_exec.c:1588`），decode M=1 全落标量三循环（`:1627-1653`）。lm_head N=248320 每步全标量 = 每 token 秒级开销。

**本特性目标**：让 decode 真正编译上板、有精度门、有性能目标——三件事分别由三个 P1 故事承载（编译上板 / 精度保证 / 性能优化），分两阶段交付。

### 已钉死的 ground truth（探查结论，改动设计以此为准）

| # | 结论 | 证据 |
|---|------|------|
| G1 | decode 纯 stub（见上 4 条） | `42_gehtp_persistent/main.c:118-160` |
| G2 | **M=1 陷阱**：HMX 门 M/K/N 全 32 对齐，decode M=1 全落标量 | `oplist_exec.c:1588`（门）、`:1627-1653`（标量回落） |
| G3 | 模型 24 层 = **18 linear-attn(GDN/SSM) + 6 full-attn**；hidden=1024，vocab=248320，intermediate=3584；full-attn heads=8 kv_heads=2 head_dim=256；linear-attn 16 heads/128 dim | `/disk1/qwen35_0.8b/model_hf/config.json` |
| G4 | KV 足迹 12KB/token（6 full-attn × K512+V512 f16）；4k context=48MB > 16MB VTCM → **KV 必须 DDR** | 计算 |
| G5 | SSM 状态 ~19.3MB（18 × (conv 3×6144 f32=72KB + delta S 16×128×128 f32=1MB)）→ DDR | `kernels/include/gdn_sm.h` |
| G6 | **状态化 kernel 已存在但未接线**：`conv1d_step_f16`/`delta_chunk_f16`/`solve_tri_f16`，host 验证于 `examples/43_gdn_sm_08b`（cos≥0.9999）；现有 `exec_conv1d_ssm` 是 stateless 全序列 | `gdn_sm.h`、`oplist_exec.c:1319-1346` |
| G7 | token-id → f16 编码仅 ≤2048 精确，vocab 248320 不够；prefill 只用了 id 0..2000 | `judge_logits.py`、`golden_qwen35.py` |
| G8 | `18_smallm_gemv` 是 **W4A16** 路径（`dc_w4_invoke`），对 f16 `OP_MATMUL_F16` decode matmul 不可复用 | `examples/18_smallm_gemv` |
| G9 | 基线精度 cos=0.998643/top1=29/32（v17，HEAD afb56c7）；黄金门 cos≥0.9999/32-32 未达；**精度门语义 = 不回归基线** | commit afb56c7/dad4b48/b66ed12 |
| G10 | deploy 陷阱：`build_libs.sh one` + `gehtp setup` 的 `[ -f signed.so ] || build` 短路 → stale arity；当前好板 **d0f1784**（默认 52f67807 陈旧） | `scripts/gehtp:255`、`kernels/build_libs.sh` |
| G11 | 004 在独立 worktree `/disk1/gehtp-wt/004-decode`（分支 `004-decode-on-device` off `task/prof-wp`@a5bdc70；dad4b48+afb56c7 均为祖先，无需 ff-merge）。主树留有 003 会话未提交 WIP（blob v3 引擎注解，其 T007 与本特性 T004/T005 同改 `exec_matmul_f16` 分发段）——worktree 隔离正是为此，回合主线时须与 003 reconcile（[[gehtp-cross-session-coordination]]） | git |
| G12 | QNN SDK 无 `SHELF` memtype（仅 RAW/MEMHANDLE/RETRIEVE_RAW）；生命期由 tensor **type** 表达；长活张量两条真实机制 = LPAI `PERSISTENT_MEM` 命名 arena + HTP 显式 I/O `past_key_*_in/out`；GEHTP `qnn_ir_loader.cpp:98-133` 只读 type、**丢弃 memType** | `/disk1/toolchains/qairt_2.48.40.260702/` |
| G13 | QNN 逆向文档：storage_class `{DDR,VTCM,CONST,VTCM_PERSISTENT}` + `lifetime_start/end` 图着色——**`VTCM_PERSISTENT` 是图内驻留非跨执行**（002 的 0x4000 同款）；序列化 bin 命名 KV 张量 `past_key_N_out`；M=1 decode 瓶颈=权重带宽（Q4_0 压 25GB/s，~90μs/token） | `compiler/reference/GRAPH_COMPILER_PRINCIPLES.md`、`SERIALIZED_BIN_ANALYSIS_GUIDE.md:161`、`HTPACC_GUIDE 3.md:1774-1787` |

## User Scenarios & Testing

### User Story 1 - 编译上板：M=1 单 token 前向上板跑通（Priority: P1）🎯 MVP

模型部署工程师要让设备能**逐 token 生成**：不只是单发 prefill，而是把 seq=1 的 M=1 前向图（prefill 退化，无 KV）编译出来、上板跑一个 M=1 前向、拿到正确的 `[1,248320]` logits，最终通过常驻 runtime 实现「权重只 load 一次、状态跨请求存活」的 autoregressive 循环。

**Why this priority**: 一切 decode 工作的地基；现在连「跑一个 M=1 前向、拿对 logits」都做不到（M=1 全落标量 + persistent 是 stub + 无 seq=1 编译路径）。

**Independent Test**: 编译 seq=1 产物上板 d0f1784，输出 `[1,248320]` logits 与 `cpu_golden/hf/logits_{i}[0]`（prefill 首行）cos≥0.998643/top1 一致——M=1 前向不变式（锚 prefill 首行，位置 0 因果输出与后续位置无关），无需新 HF 调用即可独立验证。

**Acceptance Scenarios**:

1. **Given** seq=1 编译产物（`--seq 1`），**When** 上板 d0f1784，**Then** 输出 `[1,248320]` logits 与 prefill 首行 cos≥0.998643/top1 一致
2. **Given** 干净构建（`build_libs.sh` full + `gehtp setup --device d0f1784`），**When** 跑 prefill v17，**Then** 复现 cos 0.998643/top1 29/32（deploy loop 打通、无 stale arity）
3. **[P2] Given** persistent runtime，**When** socket 发 `prefill`+N 次 `decode`，**Then** 权重只 load 一次、KV/SSM 状态跨请求存活、返回每步 logits

---

### User Story 2 - 精度保证：decode 精度门（Priority: P1）

精度工程师需要一门判据，挡住「状态携带 bug」与「M=1 数值回归」。decode 用同一套 exec 函数但 M=1/batched-attention 形状几乎未测，必须有一门判据把错误定位到具体 op。

**Why this priority**: 状态（KV/SSM）跨 `wt_exec_run_io` 存活是全新代码路径，最易出「前一步正确、后一步静默错」的 bug；没有精度门，decode 上的所有性能优化都无法信任。

**Independent Test**: 用 decode 步 logits 与设备自身 prefill 末位 logits 对拍（同权重同来源），cos≥0.9999 且 top1 精确即可精确捕状态 bug，不依赖 HF。

**Acceptance Scenarios**:

1. **Given** decode 步 logits，**When** 与设备自身 prefill 末位 logits 对拍，**Then** cos≥0.9999 且 top1 精确（设备自洽主门，同权重同来源）
2. **Given** decode 步 `[1,248320]` logits，**When** 与 `cpu_golden/hf/logits_{i}` 末行对拍，**Then** 保持不回归基线 **cos≥0.998643/top1 29/32**（跨设备基线门，非 0.9999 黄金门）
3. **Given** 精度异常，**When** 走 per-op hook 取证链（`wt_exec_set_hook`/`cmp_hook.py`/`cmp_l0.py`），**Then** 定位到具体 op

---

### User Story 3 - 性能优化：M=1 GEMV + KV/SSM DDR 布局 + 端到端 wall（Priority: P1）

性能工程师要消除 G2 的 M=1 标量回落（decode 头号性能杀手），让 KV/SSM 状态落 DDR 不挤 VTCM，并建立 autoregressive 每步 wall 的对照基线。

**Why this priority**: decode 每步都要算一遍 lm_head（N=248320）；M=1 标量回落会让每 token 秒级；状态若不落 DDR，4k context 的 48MB KV 会顶爆 16MB VTCM。

**Independent Test**: decode 步（M=1）optrace 确认所有 matmul 走 `exec_matmul_f16_gemv`（无标量回落），且 prefill（M=32/64）逐字节不变（additive 分支，不碰原 HMX 门/标量回落）。

**Acceptance Scenarios**:

1. **Given** decode 步（M=1），**When** optrace，**Then** 所有 matmul 走 `exec_matmul_f16_gemv`（无标量回落），prefill 逐字节不变
2. **Given** KV/SSM 状态，**When** 上板，**Then** 状态落 DDR（KV 12KB/token、SSM ~19MB），不挤 VTCM 16MB（VTCM 留给 HMX scratch）
3. **[P2] 端到端 wall**：autoregressive 每步 wall 记录并对照（目标：单步 decode 远小于 prefill 的 34s，lm_head 步不再全标量）

---

### Edge Cases

- M=1 但 K 非 32 对齐（本模型 K=1024 对齐，但防御性回落标量）
- GQA：kv_heads=2 → heads=8 的 `repeat_interleave(4)`，GATHER 不复制存储
- 首个 decode token 时 KV 为空（pos=0）：GATHER 读空集不崩
- token-id >2048 的 f16 编码溢出（vocab 248320）——必须走 host embedding 注入（G7/R4）
- 状态跨多请求存活时的电源/悬空指针（persistent runtime 生命周期）
- batched attention 的 Bn=1（单 token 注意力）与 prefill Bn=32 分派差异

## Requirements

### Functional Requirements

- **FR-001**: 新增 `exec_matmul_f16_gemv`（M<32 专用 HVX f32-accumulate 快路径）；`exec_matmul_f16` 内、HMX 门 `:1588` **之前**插 `if (M<32)` 分支，gemv 失败回落原标量（**全有或全无**，禁止混合路径）
- **FR-002**: GEMV **不 crouton repack**——stream W 行主序（GEMV 是 memory-bound，repack 翻倍 W 流量，lm_head W≈508MB 不可接受）；支持 `batched`/`Bn`、`t0/t1` 转置、SPLIT 的 `n0/N_full`
- **FR-003**: seq=1 编译路径：`export_qwen35_onnx.py --seq 1` → `hnnx_compile` → `wtop_emit` → blob 产出 `[1,248320]` logits
- **FR-004**: **命名持久状态表** `state_table[name]={buffer,size,grow_policy,lifetime}`——统一 `g_kv[6]`/`g_ssm_state[18]` 为一张表；编译期把 KV/SSM 状态尺寸算进 blob manifest（TAG_MEM_PLAN/TEMPOFF 下发），运行时「查尺寸→一次分配→声明不可覆盖」，op **按名**读写状态槽（对齐 QNN 的 `past_key_N_out` 命名 + LPAI `PERSISTENT_MEM` 查询式 arena，取代裸魔法数）。**wire 格式**：blob op 引用用 `0x6000|state_idx`（平行 `0x8000|slot`/`0x4000|temp`，`oplist_exec.c:168-179` ref 解码器加一分支）；`state_idx` 指向 manifest 的 `state_table[idx]={name,size}`，name 由编译器 emit 并随 TAG_MEM_PLAN/TEMPOFF 扩展段下发，运行时按 idx 查槽分配——「命名」语义保留在 manifest，blob 里是紧凑 idx
- **FR-005**: `OP_KV_APPEND_F16=22`（arity 4 `[k_t,v_t,state_id,pos]`，写新 token，RoPE append 时施加）/ `OP_KV_GATHER_F16=23`（arity 5 `[state_id,out_t,pos,n_kv,head_dim]`，读 `[0..pos)`，GQA repeat_interleave(4)）替换 `xop_kv_unimpl`；复用 `kvcache.h` ring（`nslots=max_pos` 退化 append-only）
- **FR-006**: `OP_CONV1D_SSM_STEP_F16`/`OP_DELTA_CHUNK_F16` 接线 `gdn_sm.h`（`conv1d_step_f16`/`delta_chunk_f16`），state-in/out、per-op 只读不覆盖
- **FR-007**: persistent runtime（fuse runner+persistent）：`wt_blob` parse 一次、external weights、`state_table`、position 持久化；每请求重算单 token embedding、scratch temp、op_us/op_ts。修 6 处硬 bug：`load_blob_from_path` 补 `wt_parse`、`parse_and_exec` 去自覆盖、input_ids 解析、`MAX_LOGITS 32768→248320`+`MAX_OPS 1024→16384`、协议 key 对齐（host 发 `handle_id`/C 找 `model_id`）、token-id→host 侧 embedding 注入
- **FR-008**: token-id→host 侧 embedding 注入（tie_word_embeddings=true，lookup embed_tokens/lm_head 权重注入 f16 `[1,1024]`），不靠 f16 存 id（G7）
- **FR-009**: KV/SSM 状态落 DDR（KV 12KB/token、SSM ~19MB），VTCM 16MB 留给 HMX scratch
- **FR-010**: 精度门语义 = 不回归基线（cos≥0.998643/top1 29/32），设备自洽门 cos≥0.9999 精确捕状态 bug；不接受「看起来对」，一律逐字节/cos 判据

### Key Entities

- **命名持久状态表 `state_table`**: `name → {buffer, size, grow_policy, lifetime}`；name 由编译器 emit（`past_key_N`/`ssm_conv_N`/`ssm_delta_N`），运行时按名查槽
- **KV cache**: `K/V[l][kv_head=2][head_dim=256][max_pos]` f16，6 full-attn 层，12KB/token，DDR，复用 `kvcache.h` ring
- **SSM state**: 18 linear 层，conv 窗 3×6144 f32 + delta S 16×128×128 f32，~19.3MB，DDR
- **decode 图**: seq=1 变体（use_cache、KV append/gather + SSM step op），与 prefill 图共享权重 slot

## Success Criteria

- **SC-001**: seq=1 单 token 前向 logits vs prefill **首位** cos≥0.998643/top1 一致（M=1 前向不变式；锚首行，因位置 0 因果输出与后续位置无关）
- **SC-002**: 设备自洽主门 cos≥0.9999/top1 精确（**真 decode 步**==prefill 末位，同权重同来源；与 SC-001 的「seq=1 prefill vs 首位」是不同阶段不同锚）
- **SC-003**: 跨设备基线门 cos≥0.998643/top1 29/32 不回归
- **SC-004**: decode 步（M=1）所有 matmul 走 `exec_matmul_f16_gemv`，无标量回落
- **SC-005**: prefill（M=32/64）逐字节不变（additive 分支不碰原路径）
- **SC-006**: KV/SSM 状态落 DDR，不挤 VTCM 16MB
- **SC-007**: autoregressive 每步 wall 记录对照（单步 decode 远小于 prefill 34s）
- **SC-008**: **每个 User Story 的阶段验收门都含 0.8B 双门**：精度（cos ≥ 基线判据）+ 性能（wall 记录对照基线/上一阶段值）——双过才算该阶段完成

## Out of Scope（明确不做）

- **M=1 GEMV 的权重量化**（Q4_0 压 DDR 带宽，QNN 的 decode 优化手段）——0.8B 精度要求全 f16（[[gehtp-08b-accuracy-bisect]]：W4A16 噪声按层乘性累积），量化压缩留后续精度攻关
- **KV cache strided 零拷贝视图**——v1 物理 gather 到 temp，零拷贝留后续 perf
- **prefill 的 KV 复用 / prefix cache**——本特性只做 decode 逐 token append
- **多序列 batch 并行 decode**——单序列 autoregressive 先行
- **SSM kernel 的 HMX 化**——先标量/f16 正确性，加速留后续
- **decode 权重的 W4A16 化**——lm_head 本 blob 是 f16（G8），W4A16 decode 不在本特性
