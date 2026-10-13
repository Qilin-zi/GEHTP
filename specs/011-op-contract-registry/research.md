# Research: 算子输入契约单点注册表

**日期**: 2026-10-12 | **来源**: 代码实测（非猜测）

## 1. 契约散落地图（6 处，实测 file:line）

| # | 位置 | 内容 | 机器可读 |
|---|---|---|---|
| ① | `kernels/include/oplist_parse.h:265-300` | `#define WT_ARITY_*`（36 个 arity 宏） | ✅ 常量 |
| ② | `kernels/src/runtime/oplist_parse.c:10-50` | `static int arity_of(opcode)` switch（34 case + default -1） | ✅ |
| ③ | `kernels/src/runtime/oplist_parse.c:53-88` | `static int arg_is_slot(opcode,idx)` switch（32 case + default 0） | ✅ |
| ④ | `kernels/include/oplist_parse.h:133-262` | enum 里的 arg 语义注释（`[a_ref,w_s,out_t,M,K,N,flags]`…） | ❌ **纯注释** |
| ⑤ | `kernels/src/runtime/oplist_exec.c` | 各 `exec_*` 按 `args[n]` 硬下标解码 | ✅（与④靠人工对齐） |
| ⑥ | `compiler/tools/wtop_ops/wtop_ops.hpp` + `op_*.cpp` | emit 编码：`weight_input_index()` + 各 handler | ✅（⑥是③的镜像） |

**关键发现**：④（语义注释）是唯一「真源」，但它是注释，编译器看不见。①②③⑤⑥ 是四份各自手工维护、靠约定对齐的机器表，漂移即 bug。

## 2. `arg_is_slot()` 的真实语义（重要修正）

实测 `arg_is_slot()` 并非「slot 下标」——它实际编码的是 **「该参数是否需做 slot 空间校验」**（即：是否可能携带 `0x8000|slot` 引用）。证据：

- `wt_parse`（oplist_parse.c:204-221）只对 `arg_is_slot()==true` 的参数做 `0x8000` 范围校验。
- `OP_MATMUL_F16 [a_ref,w_s,out_t,M,K,N,flags]`：`idx 0,1` 返回 true（a_ref 可 temp 可 slot；w_s 恒 slot），`idx 2(out_t),3..6(imm)` 返回 false。
- `OP_KV_APPEND_F16 [k_t,v_t,state_ref,pos]`：`idx 2(state_ref=0x6000)` 返回 true，但其值走 `0x6000` 编码、不设 `0x8000` 位，范围校验自动跳过。

**结论**：`arg_is_slot` 是「引用可校验位」，不是「槽/临时」二分类。因此注册表必须**显式**存这个布尔（`slot_check`），**不能**从 `kind` 推导——推导会因 out_t=false 而 state_ref=true 这类 quirk 出错。

## 3. 共享源：编译器与执行器已同源（简化决策）

实测 `compiler/build_pass/` 的依赖文件显示：

- 编译器 `#include "oplist_parse.h"`（`../kernels/include/oplist_parse.h`）。
- 编译器**直接编译** `kernels/src/runtime/oplist_parse.c`（`test_wtop_emit` 链接了它，用于 round-trip 解析）。

**结论**：宿主编译器与设备执行器**已经共享同一份** `oplist_parse.h` + `oplist_parse.c`。单点注册表放进 `kernels/include/`（现有共享头），两棵树自动读到同一张表，**不需要新建共享通道**。

## 4. 两类坑的精确归因（哪些被真正根治）

| 坑 | 根因 | 注册表能否根治 |
|---|---|---|
| arity 6→9 强耦合（010 US2 blocked） | arity 散在 `WT_ARITY_*` + `arity_of()` + emit 三处，改 6→9 时三处失同步 | ✅ **能**——arity 单点化，改一处 |
| 陈旧 arity rc=-11（设备库 stale arity） | 设备 `.signed.so` 未重编（build one 只重编单 .o、跳过 .signed.so 重建），设备二进制 arity 落后 | ⚠️ **部分**——单点化让漂移**可检测**（parity 断言），但设备二进制重建仍是 build 系统的责任 |

**诚实结论**：注册表直接根治「源级多表漂移」（arity 耦合）；对「设备二进制陈旧」，它把隐蔽漂移变成**构建期可 fail-fast 的 parity 断言**，但 `.signed.so` 的强制重建仍是独立 build 工程点。计划不夸大：SC-004 的「根除」精确措辞为「源级多表漂移根除 + 设备陈旧可检测」。

## 5. 决策记录

### D1: 注册表形态 —— 静态常量数组（非代码生成）
- **Decision**: `static const struct wt_op_spec wt_op_spec[] = {...}`，按 opcode 下标索引，放入共享头。
- **Rationale**: 36 个 opcode 固定且低频变更；直接 C 数组最简单，无新增生成步骤/构建依赖，host+设备同源自动共享。
- **Alternatives**: X-macro 生成 / 脚本生成 `.inc`。均引入额外构建步骤，收益（省一次手写）不抵复杂度。

### D2: `slot_check` 显式布尔（不从 kind 推导）
- **Decision**: `wt_arg_spec` 含显式 `slot_check` 字段，逐值转录现有 `arg_is_slot()` 32 个 case。
- **Rationale**: 见 §2——`arg_is_slot` 语义有 quirk，推导必出错；显式转录保证 byte-exact 等价。
- **Alternatives**: 从 `kind` 推导（`ARG_SLOT||ARG_REF||ARG_STATE → true`）。被 out_t=false/state_ref=true 反例否决。

### D3: kind 5 类编码（P1 语义机器化）
- **Decision**: `kind ∈ {ARG_SLOT, ARG_REF, ARG_TEMP, ARG_STATE, ARG_IMM}`（另有 `ARG_IMM_PACKED` 表位域，如 Gather 的 `axis|xrank<<8|irank<<16`、matmul flags）。
- **Rationale**: 覆盖 spec FR-005 的 5 类编码（0x8000 slot / 裸 temp / 0x4000 VTCM / 0x6000 state / imm）。
- **Alternatives**: 只做 slot-vs-imm 二值。不足以支撑 FR-004（语义机器可读）。

### D4: P2 边界
- **Decision**: `exec_*`（⑤）与 emit `weight_input_index`（⑥）本 spec **不动**，仅让注册表就位供其后续读。
- **Rationale**: ⑤触及 34 个 exec 函数、⑥触及 19 个 emit handler，属最高风险改动，须单独 byte-exact 门压着做（spec FR-008）。

## 6. 迁移正确性策略（parity-first）

迁移分两步走，先「对拍」后「切换」：

1. **Parity 单测**：新表就位后，写一个单测遍历 36 opcode × 16 arg，断言 `wt_op_spec[op].arity == arity_of_old(op)` 且 `wt_op_spec[op].args[i].slot_check == arg_is_slot_old(op,i)`。**在切换前跑通**，证明表转录零误差。
2. **切换**：`arity_of()` / `arg_is_slot()` 改读表，删除 `#define WT_ARITY_*` 与旧 switch。
3. **Byte-exact 回归**：35 例 + 261 门 + W4A16 65536/65536 逐位对比，零回归后合入。
