# Quickstart: 算子契约注册表迁移验证

**日期**: 2026-10-12 | **前提**: 本特性是内部重构，验证 = 证明「迁移前后逐位等价」。

## 验证总览

| 步骤 | 验证什么 | 通过判据 |
|---|---|---|
| 1. Parity 单测 | 新表 == 旧表（arity + slot_check 逐值） | 36 opcode × 16 arg 零差异 |
| 2. Host 单测 | wt_parse 行为不变 | `oplist_parse.c` 单测全绿 |
| 3. Byte-exact 回归 | 编译产物逐位不变 | 35 例 + 261 门 + W4A16 65536/65536 |

## 步骤 1：Parity 对拍（迁移正确性的核心门）

新表 `wt_op_spec[]` 就位后、切换消费方之前，先跑 parity 单测证明转录零误差：

```c
/* 伪代码：断言新表逐值等价于旧 arity_of()/arg_is_slot() */
for (int op = 0; op <= OP_MAX; op++) {
    assert(wt_op_spec[op].arity == arity_of_old(op));          /* 含 default -1 语义 */
    for (int i = 0; i < WT_MAX_ARGS; i++)
        assert(wt_op_spec[op].args[i].slot_check == arg_is_slot_old(op, i));
}
```

- 位置：`compiler/tests/test_wtop_emit.cpp` 已链接 `oplist_parse.c`，可直接加此断言。
- 通过后再删除 `arity_of_old` / `arg_is_slot_old` 的 switch，改由 `arity_of()` / `arg_is_slot()` 读表。

## 步骤 2：Host 单测

`oplist_parse.c` 是平台无关纯 C（host/设备同一份源码），host 单测即覆盖 wt_parse 的 arity 校验（`WT_ERR_ARITY`）与 slot 范围校验（`WT_ERR_BAD_REF`）。

## 步骤 3：Byte-exact 回归

沿用既有验证线（详见 `docs/GEHTP_TOOLS.md` 与 [[gehtp-validation-line]]）：

- 35 例 host 数值对拍
- 261 门（含 cos 门 / top1 门）
- W4A16 256³ 65536/65536

**判据**：全部与迁移前逐位一致，零回归。任一例出现差异即回退定位（本迁移只动解析层元数据，不应触碰任何数值路径）。

## 预期结果

- Parity 单测：全绿（转录等价）。
- 迁移后 `arity_of()` / `arg_is_slot()` 行为不变 → blob 解析不变 → 回归逐位不变。
- 无 wall 时间变化（arity/身份查询是每 op 一次 O(1)，非热点）。

## 完成后

- 删除 `#define WT_ARITY_*` 与旧 switch（或改派生宏）。
- 更新 enum 注释（④）指向注册表（语义名以注册表为准，注释仅作导读）。
- 记录设备 `.signed.so` 需重编（契约源虽未变，但二进制需跟上——见 research §4 的「部分根治」边界）。
