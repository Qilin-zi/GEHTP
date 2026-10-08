# Data Model: 005-graph-identity-passes

编译器 pass 特性——实体是**改写前后的图对象与中间判据**，无持久化存储（全部落在既有 tagged .bin 序列化框架内，零新字段）。

## 实体

### SliceSpec（P1 内部值类型，pass_view_fold.cpp 自由函数域）

两个 StridedSlice 合成的中间表示。

| 字段 | 类型 | 语义 | 校验 |
|---|---|---|---|
| begin[N] | i64 | 每轴起始（已消化 begin_mask：mask 位=0 起） | 解析后无 0x80000000 哨兵残留 |
| end[N] | i64 | 每轴终止（已消化 end_mask：mask 位=dim） | 同上 |
| strides[N] | i64 | 每轴步长（≠0） | 合成后仍 ≠0 |
| shrink | u32 位掩码 | 该轴折叠为标量维 | 合成时维度代数精确消解 |
| new_axes | u32 位掩码 | 插入的新 1 维 | 同上 |
| shape_in[N] | u32 | 输入静态形状（吸收 Reshape 后的底层形状） | 与 opdef.output_def 对账 |

**状态转换**: `compose(a: SliceSpec, b: SliceSpec) → SliceSpec ∪ {不可合}`——纯函数、无副作用；结果为全区间恒等 → 上层改线删 op。

### ViewOp（P1 匹配目标，非新类型）

复用 OpDef；匹配谓词按 `name_tag`（"StridedSlice"/"Reshape"/"Transpose"）+ 单消费者门（`consumers.size()==1`）。

### ConstScalar（P2 合成产物）

同型合并产生的新 const：`EB(c1)∘EB(c2) → EB(c')`，c' = f64 运算后 cast 回原 const dtype（f16 池）。**校验**: c' 写入 const 池（append_node），消费者改线，旧双 const 由 fixpoint DCE 回收——与 fc_bias_fold 的 bias 并入完全同构。

### BiasMark（P3 复用，零新增）

op_data scalar `has_bias=1`（e4d1dbb 机制）。MatMul 链头写入同一标记；发射门与 host execute 只认标记（extract_fc_extra 口径不变，MatMul 走 extract_matmul_extra + 同名 scalar）。

## 数据流

```text
net.json → loader → GraphPrepare(含 inject_htp_prepare_inputs)
  → run_optimize_passes: PHASE_0 const_fold → PHASE_1 shape_fold
  → 【新】PHASE_1 追加: view_fold(P1) / eltwise_const_fold(P2)
  → PHASE_3: fc_bias_fold(P3a 匹配器含 MatMul 链头)
  → fixpoint(DCE/order/CSE) → 序列化 tagged .bin → wtop_emit → 设备
```

**关键不变量**:
1. P1/P2 改写只发生在 prepare 阶段（序列化之前），序列化往返不感知 pass 存在。
2. 所有改写满足单消费者门 → fixpoint DCE 可安全回收死 op。
3. `--opt-stats` 的 matched/rewritten 计数是唯一观感面（无日志文件输出）。

## 验证规则映射

| FR | 实体/不变量 | 验证 |
|---|---|---|
| FR-001/002 | SliceSpec 合成位级恒等 | test_opt_passes 图 + host md5 |
| FR-003 | ConstScalar ≤1 ULP | host 对拍 max\|d\| |
| FR-004 | BiasMark（FC+MatMul 链头同标记） | 单测 [15a] 非对齐折 / [15b] 32 对齐折（US3b 撤门后正例）/ [15c] batched 反例 |
| FR-006/007 | 幂等 + stats | 二跑零改写 + stats_line |
| FR-008 | 序列化不感知 | ctest 序列化往返（既有） |
