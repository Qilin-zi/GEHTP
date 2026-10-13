# Contract: 算子契约注册表 Schema

**日期**: 2026-10-12 | **面向**: 编译器(emit) ↔ 执行器(parse/exec) 双边的唯一事实源

## 1. 目的

`wt_op_spec[]` 是算子输入契约的**单点声明**。编译器与执行器都从它读 arity / 参数身份 / 参数语义，取代目前散落的两张 switch 表 + 宏 + 注释。

## 2. Schema（C 兼容，C++ 可 include）

```c
enum wt_arg_kind {
    WT_ARG_SLOT,        /* 权重/常量槽     0x8000|slot 或裸 slot id */
    WT_ARG_REF,         /* 槽或 temp       裸 id 或 0x8000|slot（运行时按 flag） */
    WT_ARG_TEMP,        /* 输出/临时       裸 id，可 0x4000 VTCM */
    WT_ARG_STATE,       /* 外置状态槽      0x6000|idx */
    WT_ARG_IMM,         /* 标量            裸 u32 */
    WT_ARG_IMM_PACKED,  /* 打包位域标量    裸 u32（axis|xrank<<8|irank<<16 等） */
};

struct wt_arg_spec {
    const char *name;       /* 语义名，如 "act" / "wgt" / "out" / "M" */
    uint8_t     kind;       /* 上述 enum */
    uint8_t     slot_check; /* 逐值转录 arg_is_slot() 布尔 */
};

struct wt_op_spec {
    uint8_t            arity;               /* ≤ WT_MAX_ARGS(16) */
    struct wt_arg_spec args[WT_MAX_ARGS];
};

/* 下标 = opcode，覆盖 0..37（31/32 为空洞哨兵） */
static const struct wt_op_spec wt_op_spec[];
```

## 3. 消费方（迁移后）

| 消费者 | 读取 | 等价于（迁移前） |
|---|---|---|
| `arity_of(opcode)` | `wt_op_spec[opcode].arity` | `WT_ARITY_*` 宏 + switch |
| `arg_is_slot(opcode,idx)` | `wt_op_spec[opcode].args[idx].slot_check` | switch 32 case |
| parity 单测 | 全表 | 新旧逐值对拍 |
| （P2，后续）`exec_*` / emit | `name` + `kind` | ⑤⑥ |

## 4. 参数编码分类法（`kind` 的线格式约定）

| kind | 线编码 | 运行时如何解析 |
|---|---|---|
| `WT_ARG_SLOT` | `0x8000\|slot`（发射器统一 flagged）或裸 slot id | `(v & 0x8000)` → slot 空间，`v & 0x7FFF` 为槽 id |
| `WT_ARG_REF` | 裸 temp id 或 `0x8000\|slot` | 按 `0x8000` 位判定槽/临时 |
| `WT_ARG_TEMP` | 裸 id；VTCM 驻留 = `0x4000\|temp` | 运行时 temp 表查表 |
| `WT_ARG_STATE` | `0x6000\|idx` | 状态表索引 |
| `WT_ARG_IMM` / `IMM_PACKED` | 裸 u32 | 直接读标量；PACKED 按位域拆解 |

> 注意：`slot_check` 与 `kind` 无推导关系（见 data-model.md）。OP_DMA 的 `0xC000|引擎面` 编码是 P5 特有、由 `exec_dma` 自行边界检查，不属本 6 类。

## 5. 不变量（迁移正确性判据）

1. `arity_of(op)` 与迁移前逐值一致（否则 `WT_ERR_ARITY` 行为变）。
2. `arg_is_slot(op,i)` 与迁移前逐值一致（否则 `WT_ERR_BAD_REF` 行为变）。
3. blob 线格式（wt_op/wt_slot/wt_blob）零改动。
4. 35 例 + 261 门 + W4A16 65536/65536 回归逐位不变。
