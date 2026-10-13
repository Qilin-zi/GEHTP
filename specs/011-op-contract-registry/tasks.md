# Tasks: 算子输入契约单点注册表

**Input**: Design documents from `/specs/011-op-contract-registry/`

**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: 不采用 TDD 红绿循环（这是逐值转录重构），但 **parity 对拍单测是硬门**，且须在切换前跑通。

**Organization**: 按用户故事分组。US1= P0（arity+身份单一事实源），US2= P1（语义机器化）。**US3（P2 按名取参）明确范围外，本文件不生成其任务**（见底部「范围外」注）。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无依赖）
- **[Story]**: 归属的用户故事（US1/US2）

---

## Phase 1: Setup

**Purpose**: 注册表类型定义 + 共享头接线（本重构唯一的「新脚手架」）

- [X] T001 创建 `kernels/include/op_spec.inc`，包含：`enum wt_arg_kind { WT_ARG_SLOT, WT_ARG_REF, WT_ARG_TEMP, WT_ARG_STATE, WT_ARG_IMM, WT_ARG_IMM_PACKED }`、`struct wt_arg_spec { const char* name; uint8_t kind; int8_t slot_check; }`、`struct wt_op_spec { int8_t arity; struct wt_arg_spec args[WT_MAX_ARGS]; }`，以及一个零初始化的 `static const struct wt_op_spec wt_op_spec[]` 占位。**`arity` 用 `int8_t`（有符号）**，以承载保留空洞 31/32 与越界 opcode 的 `-1`（无效哨兵），等价于现 `arity_of()` 的 `default: return -1`。然后在 `kernels/include/oplist_parse.h` 的 `struct wt_blob` 之后、错误码 enum 之前加 `#include "op_spec.inc"`。验证 `oplist_parse.c`（C）与编译器构建（C++）都能编译通过。

---

## Phase 2: User Story 1 - arity 与参数身份单一事实源 (Priority: P1) 🎯 MVP

**Goal**: `arity_of()` 与 `arg_is_slot()` 改从 `wt_op_spec[]` 读取，`#define WT_ARITY_*` 与两张 switch 表删除，取值逐值等价。

**Independent Test**: parity 单测全绿 + 切换后 byte-exact 回归（35 例 + 261 门 + W4A16 65536/65536）逐位一致。

### 转录（parity-first：先建表、后切换）

- [X] T002 [US1] 在 `kernels/include/op_spec.inc` 填充 `wt_op_spec[].arity`，逐值转录自 `#define WT_ARITY_*`（`oplist_parse.h:265-300`）与 `arity_of()`（`oplist_parse.c:10-50`）。保留空洞 31/32 的 `arity = -1` 哨兵，与 `arity_of()` 的 `default: return -1` 语义一致（越界 opcode → `WT_ERR_OPCODE`）。
- [X] T003 [US1] 在 `kernels/include/op_spec.inc` 填充 `wt_op_spec[].args[].slot_check`，逐值转录自 `arg_is_slot()`（`oplist_parse.c:53-88`），保留 `default: return 0` 语义（未列出的 arg → 0）。`slot_check` 必须显式转录、**不得从 kind 推导**（research §2：out_t=false 而 state_ref=true 的反例）。

### 对拍 + 切换 + 删除

- [X] T004 [US1] 在 `compiler/tests/test_wtop_emit.cpp` 加 parity 单测：先把 `oplist_parse.c` 里的旧函数重命名为 `arity_of_old` / `arg_is_slot_old`（保留 switch 体），然后遍历 `op = 0..37` 断言 `wt_op_spec[op].arity == arity_of_old(op)`，遍历 `i = 0..WT_MAX_ARGS-1` 断言 `wt_op_spec[op].args[i].slot_check == arg_is_slot_old(op, i)`。跑通 = 转录零误差，才可进入 T005/T006。
- [X] T005 [US1] 在 `kernels/src/runtime/oplist_parse.c` 把 `arity_of()` 改为 `if (opcode > OP_MAX) return -1; return wt_op_spec[opcode].arity;`，删除旧 switch 体（T004 通过后）。
- [X] T006 [US1] 在 `kernels/src/runtime/oplist_parse.c` 把 `arg_is_slot()` 改为 `return wt_op_spec[opcode].args[idx].slot_check;`，删除旧 switch 体（T004 通过后）。
- [X] T007 [US1] 删除 `kernels/include/oplist_parse.h:265-300` 的 `#define WT_ARITY_*` 宏（arity 已单点化进 `wt_op_spec[]`）。同时删除 `oplist_parse.c` 里的 `arity_of_old` / `arg_is_slot_old` 临时函数。
- [X] T008 [US1] 跑 byte-exact 回归（35 例 + 261 门 + W4A16 65536/65536），逐位一致；并验证 `wt_parse` 仍对未知 opcode 返回 `WT_ERR_OPCODE`、对 arity 不匹配返回 `WT_ERR_ARITY`（见 `quickstart.md` 步骤 3）。**设备侧步骤见 `device-runbook.md` §2**。**已做**（2026-10-13，d0f1784）：三处预存基建阻断已修——①例 19/43/25 `delta_chunk_f16` 补 `q_raw=0`（009 参数未同步）②`pull_result` 结果路径对齐 hrt/gehtp ③`wt_w3.c` 槽行 `%u` 吃 `uint64_t offset` 的 varargs 对齐 bug（Hexagon 上 offset/addr/sha256 三字段漂移；本会话独立发现并修，上游 010 收口 c77665a 已用 `%llu` + `(unsigned long long)` 同根修复落地，取上游版）。全量 `build_examples.sh all`：**312 pass / 10 fail**，10 fail 全为预存范围外例（13_compat_dlsym 遗留 v2.so、37_conv2d_add 1ULP、38_transformer_ops mini-arity+gather、39_probe/41_qwen_layer0 wt_parse rc、40_qwen_layer3 unary subtype），非 011 回归；例 21（唯一直接对拍 wt_parse）5/5 + `w3_host_device_lines_identical` PASS。W4A16（例 31）不调 wt_parse，数值路径 011 构造上不可触达。**0.8B E2E 加固**（2026-10-13，主线 FF 到 c77665a 后重编 lib 300808B 全量重推）：qwen35_08b_gdn_default blob（884 ops）cos=0.999930 / top1=32/32 / max|d|=0.2047，exec 2.771s（锚点 2.76s 持平）— 精度性能双无回归。

**Checkpoint**: US1 完成后，arity 与参数身份唯一事实源就位，两类坑（arity 6→9 耦合、陈旧 arity）的源级漂移根除。

---

## Phase 3: User Story 2 - 参数语义机器可读 (Priority: P2)

**Goal**: 每个参数加 `name` + `kind` 字段，把 enum 注释里的人工语义升级为程序可校验的字段（FR-004 / FR-005）。

**Independent Test**: 覆盖率检查（T011）全绿 + byte-exact 回归仍逐位一致（name/kind 是纯元数据，不触碰解析路径）。

### 语义富化

- [X] T009 [US2] 在 `kernels/include/op_spec.inc` 为每个 arg 填 `name`（语义名），转录自 enum 注释（`oplist_parse.h:133-262`）。例：`OP_MATMUL_F16` → `act`/`wgt`/`out`/`M`/`K`/`N`/`flags`。
- [X] T010 [US2] 在 `kernels/include/op_spec.inc` 为每个 arg 填 `kind`（`wt_arg_kind`），按 data-model.md 的 6 类分类；`WT_ARG_IMM_PACKED` 用于 Gather 的 `axis|xrank<<8|irank<<16` 与 matmul flags。覆盖 FR-005 的 5 类编码（0x8000 slot / 裸 temp / 0x4000 VTCM / 0x6000 state / imm）。
- [X] T011 [US2] 加一个单测（放 `compiler/tests/test_wtop_emit.cpp` 或新文件）断言：每个已定义 opcode 的前 `arity` 个 arg 都有非空 `name` 且 `kind != 0`（无未分类 arg），落实 FR-005 的 100% 覆盖。
- [X] T012 [US2] 更新 `oplist_parse.h:133-262` 的 enum 注释为「语义以 `op_spec.inc` 为准」的短指针（注释降级为导读，不再做真源）。

**Checkpoint**: US2 完成后，语义描述从「仅注释」变为「机器可读字段」，语义-解码不一致可被程序校验。

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: 收尾与文档

- [X] T013 重建设备侧 `.signed.so` 并验证无 stale-arity（research §4：注册表把陈旧变成 parity 可检测，但设备二进制重建仍是 build 系统责任）。**步骤见 `device-runbook.md` §1**。**已做**（2026-10-13，d0f1784）：全量 `build_libs.sh`（lib 278376→300808B，op_spec.inc 表落地）+ `gehtp setup` 推 + 例 21 冒烟 55/55、无 rc=-11/-10。
- [X] T014 更新 `docs/GEHTP_TOOLS.md`（或架构文档），记录「算子契约单点注册表」与已删除的散落表，供后续维护者知悉。

---

## 范围外（US3 / P2，后续单独立项）

- **US3 按名取参**：`exec_*` 按 `args[n]` 硬下标解码 → 按 `name` 取参；emit 侧 `weight_input_index()` + `op_*.cpp` 改读注册表。**本 spec 明确排除**（FR-008 / spec US3），由后续独立 spec 用独立 byte-exact 门压着做。

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，立即可开始。
- **US1 (Phase 2)**: 依赖 T001 类型定义。内部严格串行：T002/T003（转录）→ T004（对拍）→ T005/T006（切换）→ T007（删除）→ T008（回归）。
- **US2 (Phase 3)**: 依赖 US1 完成（表必须已就位且已切换）。T009/T010 富化同表字段，T011 覆盖率检查，T012 注释降级。
- **Polish (Phase 4)**: 依赖 US1+US2 完成。

### 串行性说明（诚实）

本重构主体在**同一个文件**（`op_spec.inc` 表 + `oplist_parse.c` 两函数）上逐步转录，天然强串行，**并行机会少**。唯一可并行的点是：
- T011（覆盖率单测，新文件）可在 T009/T010 进行时并行写。
- T013（设备重编）与 T014（文档）可并行，且可与 US2 收尾重叠。

### Parallel Opportunities

- `T011`（覆盖率单测）与 `T009`/`T010`（表富化）可并行（不同文件）。
- `T013` 与 `T014` 可并行（不同领域）。

---

## Parallel Example

```bash
# US2 阶段：表富化 与 覆盖率单测 并行
Task: "T009/T010 在 op_spec.inc 填 name/kind"
Task: "T011 在 test_wtop_emit.cpp 写覆盖率单测"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. T001 类型定义 → T002/T003 转录 → T004 parity 对拍（**必须绿**）
2. T005/T006 切换 → T007 删旧 → T008 byte-exact 回归（**必须逐位一致**）
3. **STOP & VALIDATE**: US1 独立验收（parity 绿 + 回归逐位不变）

### Incremental Delivery

1. US1（P0）= 单一事实源，根治源级漂移 → 独立验收
2. US2（P1）= 语义机器化 → 独立验收（byte-exact 仍不变）
3. US3（P2）= 范围外，后续单独立项

---

## Notes

- parity 对拍（T004）是**硬门**：转录零误差才准切换，任何一例差异即回退定位。
- `arity` 用 `int8_t` 带 -1 哨兵（data-model.md 里写 `uint8_t` 是笔误，实际需有符号以承载空洞/越界的 `-1`）。
- `slot_check` 显式转录，不从 `kind` 推导（research §2）。
- 每完成一个 checkpoint 停一下做独立验收；byte-exact 回归是唯一裁判口径。
