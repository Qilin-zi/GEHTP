# Contract: OP_MATMUL_W4A16 v9（B 段）

**契约主体**: blob 级算子序列化格式（opcode 28）

## 参数序列（v9 正典）

```
args[0]=act_ref  args[1]=w_slot(0x8000|s)  args[2]=out_temp
args[3]=M  args[4]=K  args[5]=N
args[6]=bias_slot(0x8000|s)  args[7]=atbl_slot(0x8000|s)  args[8]=otbl_slot(0x8000|s)
```

## 注册表（op_spec.inc 单点）

- `arity = 9`；新增列 `legacy_arity = 6`（仅本 opcode 非 0）。
- parse 校验：`n_args == arity || (legacy_arity && n_args == legacy_arity)`。

## 分派语义（执行器）

- `n_args == 9`：三槽直读，**禁止**槽表扫描；槽长度校验保留（emit 契约哨兵：bias=(N/32)*512、atbl=8*(K/32)*4、otbl=8*(N/32)*4）。
- `n_args == 6`：legacy 长度匹配扫描路径（行为与现状逐字节一致；D 段收口后删除）。

## 供给侧（requant_w4a16.cc）

- 输入：Q4_0 GGUF 权重（经 gguf_wtlink 读取层）；输出：packed_w/atbl/otbl/folded_bias 四槽。
- atbl/otbl 布局 = i*M*8（010 已验证契约，otbl 按 N 列分列——K≠N 扫描 bug 修正版）。
- 折叠语义：folded_bias 吸收后续 Add 节点（对齐 fc_bias_fold 口径，0.8B 正典须 --no-pass fc_bias_fold 防双折）。

## 兼容性

- 不 bump WT_BLOB_VER（v3 内自识别）。
- 板上共享库须同时跑 6/9 两代 blob（混跑现实，009/010 先例）。

## 验收断言（test_wtop_emit + 合成 blob）

1. serialize→deserialize 往返：9 参全槽引用无损。
2. legacy 6 参 blob 在新 parse 下逐字节等价执行。
3. 合成单 op blob（003 T014 法）设备输出 == P5 dc_w4 已知答案。
