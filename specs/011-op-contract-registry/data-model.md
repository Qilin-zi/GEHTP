# Data Model: 算子输入契约注册表

**日期**: 2026-10-12

## 实体

### 1. `wt_arg_spec` —— 单个参数

| 字段 | 类型 | 说明 | 来源（迁移前） |
|---|---|---|---|
| `name` | `const char*` | 语义名（`act`/`wgt`/`out`/`M`/`K`/`N`/`bias`…） | ④ enum 注释 |
| `kind` | `uint8_t` enum | 5 类编码（见下） | ④ enum 注释（机器化） |
| `slot_check` | `uint8_t` bool | 是否需 slot 空间校验（引用可校验位） | ③ `arg_is_slot()` 逐值转录 |

### 2. `wt_op_spec` —— 单个算子

| 字段 | 类型 | 说明 | 来源（迁移前） |
|---|---|---|---|
| `arity` | `uint8_t` | 参数个数（≤16 = WT_MAX_ARGS） | ① `WT_ARITY_*` + ② `arity_of()` |
| `args` | `wt_arg_spec[WT_MAX_ARGS]` | 前 `arity` 个有效，其余为哨兵 | ③④ |

### 3. 注册表本体

`static const struct wt_op_spec wt_op_spec[OP_MAX+1]`，下标 = opcode。覆盖 36 个已定义 opcode（0–37，31/32 空洞哨兵）。

## kind 枚举（5 类 + 打包位域）

| 值 | 含义 | 线编码 | 典型参数 |
|---|---|---|---|
| `ARG_SLOT` | 权重/常量槽 | `0x8000\|slot` 或裸 slot id | `w_s`、`b_s`、`tbl` |
| `ARG_REF` | 槽或 temp（运行时按 flag 定） | 裸 temp id 或 `0x8000\|slot` | `a_ref`、OP_DMA src/dst |
| `ARG_TEMP` | 运行时临时量（输出） | 裸 id，可 `0x4000` VTCM | `out_t`、`y_t` |
| `ARG_STATE` | 外置状态槽 | `0x6000\|idx` | `state_ref`（KV/SSM） |
| `ARG_IMM` | 标量 | 裸 u32 | `M`、`K`、`N`、`axis` |
| `ARG_IMM_PACKED` | 打包位域标量 | 裸 u32 | Gather `axis\|xrank<<8\|irank<<16`、matmul flags |

## 关系

- `wt_op_spec[opcode].args[i]` ↔ `opcode` 的第 `i` 个参数（0 ≤ i < arity）。
- `arity_of(opcode)` ≡ `wt_op_spec[opcode].arity`（FR-002 逐值等价）。
- `arg_is_slot(opcode,i)` ≡ `wt_op_spec[opcode].args[i].slot_check`（FR-003 逐值等价）。
- `slot_check` 与 `kind` **无推导关系**（见 research §2）——二者独立字段。

## 校验规则（来自 spec FR）

- **FR-002 一致性**：`wt_op_spec[op].arity` 必须等于迁移前 `WT_ARITY_*` / `arity_of()` 的值，逐 opcode 一致。
- **FR-003 一致性**：`slot_check` 必须逐 (opcode, idx) 等价于迁移前 `arg_is_slot()`（含 default 0 语义）。
- **FR-005 覆盖**：全部 36 opcode 的每个有效 arg 必须有 `kind`（不得留未分类）。
- **FR-006 byte-exact**：迁移前后 wt_parse 的 arity 校验（`WT_ERR_ARITY`）与 slot 范围校验（`WT_ERR_BAD_REF`）行为不变。
- **FR-008 边界**：`exec_*` 解码逻辑、emit `weight_input_index` 编码逻辑不得改动。

## 状态转换

无运行时状态——注册表是只读编译期常量，无状态机。
