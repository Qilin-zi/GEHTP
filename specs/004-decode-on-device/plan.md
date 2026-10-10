# Implementation Plan: 0.8B decode 上板

**Branch**: `004-decode-on-device` | **Date**: 2026-09-29 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/004-decode-on-device/spec.md`

## Summary

把 0.8B 从「只跑单发 prefill」升级到「能逐 token autoregressive 生成」，分两阶段：**P1（M=1 单步打通）**——seq=1 编译上板、M=1 GEMV 快路径、decode==prefill 精度门；**P2（状态 + 端到端）**——KV cache + SSM 状态 + persistent runtime + host 循环，端到端生成连贯 token。三个 P1 故事（编译上板/精度保证/性能优化）横跨两阶段。

核心设计决策（QNN 对照结论，见 spec G12/G13）：KV/SSM 这类**跨执行长活张量**用「命名持久状态表 `state_table[name]`」表达（对齐 QNN 的 `past_key_N_out` 命名图张量 + LPAI `PERSISTENT_MEM`），而不是魔法指针；M=1 decode 瓶颈是**权重带宽**，故 GEMV 不 repack、stream W，但 0.8B 精度锁 f16（Q4_0 量化留后续）。

## Technical Context

**Language/Version**: C99（kernels，Hexagon DSP + host 双编译）/ C++17（compiler host 工具）/ Python（host 循环 + golden）

**Primary Dependencies**: 现有 GEHTP 工具链（`hnnx_compile` → `wtop_emit` → blob；runner `42_gehtp_runner` / `42_gehtp_persistent`）；状态化 kernel `gdn_sm.h`；KV ring `kvcache.h`

**Storage**: KV cache（12KB/token，DDR）+ SSM state（~19.3MB，DDR）——跨 `wt_exec_run_io` 存活；blob manifest（TAG_MEM_PLAN/TEMPOFF）下发状态尺寸

**Testing**: host 合成验证（GEMV vs 标量 cos、KV append→gather 往返、SSM 跨 chunk 一致性）+ decode==prefill 金标 + 设备上板（optrace / cos / wall）

**Target Platform**: SA8797P V81（VTCM 16MB，DDR 大块状态）+ host x86 模拟；好板 **d0f1784**

**Performance Goals**: 单步 decode wall 远小于 prefill 的 34s（lm_head 步不再全标量 4.24s）；autoregressive 每步 wall 建立对照基线

**Constraints**:
- prefill（M=32/64）逐字节不变（GEMV 是 additive 分支）
- 精度门 = 不回归基线 cos 0.998643/top1 29/32（非 0.9999 黄金门）
- KV/SSM 状态落 DDR，不挤 VTCM 16MB
- 设备 lib 推签名版；多会话共享设备先查进程冲突（[[gehtp-p5-device-gate]]）

**Scale/Scope**: 0.8B（24 层 = 18 linear + 6 full-attn，hidden 1024，vocab 248320）为验证模型

## 目标架构：persistent runtime + 命名状态表

```
host (gehtp_server) ──socket──▶ 42_gehtp_persistent (常驻)
                                   │ 一次 wt_parse + external weights load
                                   │ state_table[name] (KV[6] + SSM[18], DDR, 跨请求存活)
                                   │  ┌─ prefill 请求: seq=32 全图前向 → 填 KV + SSM
                                   │  └─ decode 请求: 单 token embedding → 逐 op(M=1 GEMV)
                                   │       → KV APPEND + GATHER → SSM step → lm_head
                                   └─▶ 返回每步 logits [1,248320]
```

**分工原则（对齐 QNN 跨执行模型）**：编译期把 KV/SSM 状态尺寸算进 blob manifest，运行时「查尺寸→一次分配→声明不可覆盖」，op 按名读写状态槽。设备侧仍是「按名查表」得到 buffer 指针，但名字由编译器统一 emit（`past_key_N`/`ssm_conv_N`/`ssm_delta_N`），不靠 `0x6000|state_id` 魔法数、也不靠 memType（QNN 自己也从不用 memType 表达生命期）。

## Constitution Check

`.specify/memory/constitution.md` 为空模板，无治理约束，跳过。

## Phase 结构总览

依赖链（两阶段 + 三故事交叉）：

```
Setup（基线 + deploy loop + git settle）
   │
Foundational（M=1 GEMV + dispatch + seq=1 编译）  ← US1/US2/US3 共同 enabler，阻塞一切
   │
US1 编译上板（P1）── seq=1 上板 + single-token 门    🎯 MVP
US2 精度保证（P1）── tier1 金标 + judge + hook       （随 US1 P1）
US3 性能优化（P1）── optrace GEMV + prefill 回归      （随 US1 P1）
   │
P2 状态化 + 端到端（US1/US2/US3 共同 P2 切片）
   │   KV/SSM 接线 → decode 图发射 → persistent → host loop → tier2 金标 → DDR + wall
   │
Polish
```

**关键依赖**：P2 的所有切片硬依赖 P1 的 seq=1 上板（要有能在设备跑的 M=1 前向，才谈得上接 KV/SSM 状态与端到端）。GEMV（US3 P1）独立，可与 seq=1 编译（US1 P1）并行。

---

## Phase 0 / Foundational：M=1 GEMV 快路径 + seq=1 编译

### 设计

**1. GEMV kernel（FR-001/002）**

`kernels/src/runtime/oplist_exec.c` 新增 `exec_matmul_f16_gemv`（`:1450` 附近）：

- HVX f32-accumulate，M<32（1/16/32 边界），stream W 行主序，**不 crouton repack**
- 支持 `batched`/`Bn`、`t0/t1` 转置、SPLIT 的 `n0/N_full`（复用 exec_matmul_f16 的参数解析）
- 输出 scatter 同 HMX 路径：`y[(r*32+m)*N_full + n0 + c*32+n]`

**2. dispatch（FR-001）**

`exec_matmul_f16` 内、HMX 门 `:1588` **之前**插：

```c
if (M < 32u) { if (exec_matmul_f16_gemv(...) == 0) return 0; /* 失败回落原标量 */ }
```

**全有或全无**：gemv 失败（任何 chunk）→ 整 op 走原标量，禁止部分 gemv + 部分标量的混合路径（数值纪律）。

**3. seq=1 编译冒烟**

`scripts/export_qwen35_onnx.py` 加 `--seq 1`（现状 seq 参数化路径），产出 `[1,248320]` logits 的 blob。**金标 tier1 前提**：decode==prefill 首位置不变式。

### 验证门

1. host：M=1/16/32 GEMV vs 标量参考 cos≥0.99999
2. host：GEMV 支持 batched/Bn=1、t0/t1、SPLIT n0 的合成矩阵各一
3. 设备：M=1 decode 步 logits vs prefill 首位 cos≥0.9986/top1 一致
4. 回归：prefill（M=32/64）逐字节不变（GEMV 分支 additive）

---

## Phase A / US1：编译上板（P1）

### 设计

**seq=1 上板（T008）**：`--seq 1` 编译产物上板 d0f1784，单 token 前向 logits 与 prefill 首行对拍。

**deploy loop（T009）**：干净构建 `build_libs.sh`（full，勿 `one`）+ `gehtp setup --device d0f1784`（勿信 `[ -f signed.so ] || build` 短路 → stale arity），复现 prefill cos 0.998643/29-32。

P2 切片（KV/SSM 接线、decode 图发射、persistent、host loop）在 Phase D 展开。

### 验证门

1. seq=1 blob 上板，logits `[1,248320]` vs prefill 首行 cos≥0.9986/top1
2. deploy loop 复现 prefill cos 0.998643/29-32

---

## Phase B / US2：精度保证（P1）

### 设计

**tier1 金标（decode==prefill 末位）**：`golden_qwen35.py` 加切末行 helper——decode 步 logits 应与设备自身 prefill **末位** logits 一致（因为 prefill seq=32 的末位就是「第 32 个 token 的 logits」，decode 单步算的也是「当前 token 的 logits」）。同权重同来源 → cos≥0.9999 精确捕状态 bug。

**judge 门接线**：`judge_logits.py` 接 decode 步，双门并列——设备自洽主门（cos≥0.9999）+ 跨设备基线门（cos≥0.998643/29-32）。

**hook 取证链**：`wt_exec_set_hook`/`cmp_hook.py`/`cmp_l0.py` 对 M=1 decode 步定位到具体 op（复用 prefill 的既有取证链，[[gehtp-08b-accuracy-bisect]]）。

### 验证门

1. decode 步 vs prefill 末位 cos≥0.9999/top1 精确（主门）
2. decode 步 vs HF 末行 cos≥0.998643/29-32（基线门）
3. 异常注入（故意错一个 op）能被 hook 链定位

---

## Phase C / US3：性能优化（P1）

### 设计

**optrace 验证（T013）**：decode 步所有 matmul 走 `exec_matmul_f16_gemv`，无标量回落（optrace engine 列全 GEMV/HMX）。

**prefill 逐字节回归（T014）**：M=32/64 路径未动（additive 分支不碰原 HMX 门/标量回落）。

P2 切片（KV/SSM DDR 布局、端到端 wall）在 Phase D 展开。

### 验证门

1. optrace：M=1 无标量回落
2. prefill 逐字节回归

---

## Phase D / P2：状态化 + 端到端（US1/US2/US3 共同 P2 切片）

### 设计

**1. 命名持久状态表（FR-004，先于 KV/SSM op 接线）**

`oplist_exec.c` 把 `g_kv[6]`/`g_ssm_state[18]` 统一为 `state_table[name]`：

- 编译器在 decode 图里为每层 KV/SSM emit 状态张量名（`past_key_N`/`ssm_conv_N`/`ssm_delta_N`）
- blob manifest（TAG_MEM_PLAN/TEMPOFF）下发尺寸；运行时「查尺寸→一次分配→声明不可覆盖」
- op 按名查槽拿 buffer 指针（对齐 QNN `past_key_N_out` 命名 + LPAI 查询式 arena）——wire 格式：blob op 引用 `0x6000|state_idx`（`oplist_exec.c:168-179` 加分支），`state_idx` 指向 manifest `state_table[idx]={name,size}`，name 由 emit 下发，运行时按 idx 查槽
- **集成 hook**：`qnn_ir_loader.cpp` 已按 tensor `type` 三态分派（`:230-239`）但丢弃 memType（`:98-133`）——IR 层新增「state」tensor 分类即可贯通

**2. KV ops（FR-005）**

`exec_kv_append`/`exec_kv_gather` 替换 `xop_kv_unimpl`（`oplist_exec.c:2049-2053,2097-2098`）：

- 复用 `kvcache.h:20-42` ring（`nslots=max_pos` 退化 append-only，`kvc_set_consumer(FM_DDR,FC_DMA)` 白拿 fence）
- APPEND 写新 token（RoPE append 时施加），GATHER 读 `[0..pos)`，GQA `repeat_interleave(4)` 不复制存储
- 仿 `examples/29_kvcache` 验证 append→gather 往返逐字节

**3. SSM state ops（FR-006）**

`OP_CONV1D_SSM_STEP_F16`→`conv1d_step_f16`、`OP_DELTA_CHUNK_F16`→`delta_chunk_f16`，state-in/out；验证「状态跨 2 chunk == 单 chunk oracle」（复用 `43_gdn_sm_08b` 一致性）。

**4. decode 图发射（T018）**

`export_qwen35_onnx.py` decode 变体（use_cache、KV append/gather + SSM step op 图）+ `graph_prepare.cpp`/`wtop_emit.cpp` 新增 state tensor 分类 + 状态尺寸下发。

**5. persistent runtime（FR-007）**

`42_gehtp_persistent/main.c` 修 6 处硬 bug（见 FR-007）；持久化 `wt_blob`/weights/`state_table`/position，每请求重算单 token embedding + scratch。

**6. host 循环接线（T020）**

`persistent_runtime.py`/`device_runtime.py`/`gehtp_server.py::_run_generation` 端到端生成连贯 token；decode 门绿。

**7. tier2 金标（T021）**

`golden_qwen35.py::run_decode`（HF `use_cache=True` 续写多步；blocked on R1——HF 对 GDN/SSM 层是否缓存 recurrent state 未验证）。

**8. DDR 布局 + wall（FR-009，T022/T023）**

KV/SSM 状态落 DDR，VTCM 16MB 留给 HMX scratch；autoregressive 每步 wall 记录对照。

### 验证门

1. KV append→gather 往返逐字节；SSM 跨 chunk == 单 chunk oracle
2. socket 冒烟：prefill→N decode 步，不 reload，状态存活
3. `gehtp serve` 端到端生成连贯 token；decode 门绿（tier1 主门 + 基线门）
4. 状态落 DDR（不挤 VTCM 16MB）；每步 wall 记录对照

---

## 风险登记

| 风险 | 等级 | 缓解 |
|------|------|------|
| R1 HF `use_cache` 对 GDN/SSM 层是否携带 recurrent state 未验证（modeling 代码在 QPM_ROOT，off-tree 不可读） | 中 | tier1 金标（decode==prefill）不需 HF cache；tier2 若 HF 不缓存 SSM 状态，改 `use_cache=False` 每步全序列重跑（慢但正确） |
| R2 `OP_GATHER_F16`(20) 已重载（embedding 查表 + GDN state 读回）；新增 SSM state ref 不得与其冲突 | 中 | 命名状态表独立编码（`state_table[name]`，独立于 `0x8000\|slot`/`0x4000\|VTCM-temp`，`oplist_exec.c:170-179`），不改 opcode 20 语义 |
| R3 decode 图发射（T018）是最大、最欠规格的一块，需引入跨 `wt_exec_run_io` 存活的「state tensor」类，emit 层无先例 | 高 | opcode 槽 + `gdn_sm.h` kernel 已就绪；graph-lowering 是绿地，单独成 phase 并先做 host 侧执行验证 |
| R4 token-id→f16 编码 >2048 会静默错嵌（vocab 248320） | 高 | P2 加 host 侧 embedding 注入（tie_word_embeddings=true，lookup embed_tokens/lm_head 权重注入 f16 `[1,1024]`），不靠 f16 存 id |
| R5 状态引用 wire 格式（`0x6000|state_idx`）需 emit/runtime 同口径（idx 枚举 + manifest name 一致） | 中 | oplist_parse.h 与 emit 侧同一 idx 枚举常量 + round-trip 契约测试兜底（FR-004 已定格式，此处只管一致性） |
| R6 多会话共享设备 d0f1784 撞车 | 中 | 上板前 `adb shell 'ps -A \| grep hexagon'`；job.txt 备份 .pre_ 后缀（[[gehtp-p5-device-gate]]） |
| R7 GEMV 数值 vs HMX/标量有 ulp 级差异，可能微扰 cos | 低 | GEMV f32-accumulate 与 HMX 同为精确和；tier1 主门 cos≥0.9999 兜底；异常则对比 GEMV vs 标量 host 精确对拍 |

## 工作量估算

| 阶段 | 规模 | 主要文件 |
|------|------|---------|
| Foundational GEMV + dispatch | ~200 行 | oplist_exec.c, oplist_parse.h |
| US1 编译上板 | ~50 行 + 上板 | export_qwen35_onnx.py, build_libs.sh, scripts/gehtp |
| US2 精度保证 | ~100 行 | golden_qwen35.py, judge_logits.py, cmp_hook.py/cmp_l0.py |
| US3 性能优化 | 验证为主 | optrace |
| P2 状态 + 端到端 | ~600 行 | oplist_exec.c, kvcache.h, gdn_sm.h, 42_gehtp_persistent/main.c, export/graph_prepare/wtop_emit, *.py |

## 参考

- 探查 ground truth G1-G13：spec.md §背景
- 状态化 kernel：`kernels/include/gdn_sm.h`、`examples/43_gdn_sm_08b`
- KV ring：`kernels/include/kvcache.h`、`examples/29_kvcache`
- QNN 长活张量两源：SDK `/disk1/toolchains/qairt_2.48.40.260702/include/QNN/` + 逆向 `compiler/reference/{GRAPH_COMPILER_PRINCIPLES.md, SERIALIZED_BIN_ANALYSIS_GUIDE.md, HTPACC_GUIDE 3.md}`
- 精度取证链：[[gehtp-08b-accuracy-bisect]]、[[gehtp-f16-denormal-convert-bug]]
- 基线锚：[[gehtp-08b-baseline-002]]、[[gehtp-us1-split-doublegate]]、[[gehtp-batched-attn-hmx]]
- 部署/设备坑：[[gehtp-p5-device-gate]]、[[gehtp-device-lib-stale-arity]]、[[gehtp-build-env]]、[[gehtp-integ-unified-merge]]
