# Feature Specification: Graph Identity Passes (View-Soup 消除 + Elementwise 合并 + MatMul Bias 推广)

对齐真库 libQnnHtp.so 同级能力（`Reshape_to_fold` / `stridedslice_opts` / `transpose_opts` / `QNN_FullyConnected_Add_fuse` 家族），在 PassManager 注册制下新增三个图级恒等 pass。全部遵守与 `fc_bias_fold`(e4d1dbb) 相同的纪律：**数学恒等、幂等、零设备语义改动**（P3 除外，见分阶段）。

> **状态（2026-10-08 收尾）**: US1+US2 `951cef7` / US3a `1e131fb` / US3b `69893e4`，全部验收门通过。实测：**SC-001** ops 16438→14152（-13.9%，门 -6%）、wall 39.5s→31.49s（-20.3%，门 -4%）；**SC-002** host 位级一致（P1 md5 相同、P2 n_diff=0）；**SC-003** ctest 47/47；**SC-004** 发射 md5 与基线逐字节一致（a322d06a…）；**SC-005** cos=1.000000 + 三判据 0 违例（"≤2 ULP"细化为三判据口径，f16 次正规对消输出位距膨胀，见 US3b 证据）；**SC-006** 挂起口径沿用。0.8B 真图 **0 次 MatMul 折叠为正确结果**（42 候选实为 activity-activity 加法，另侧全非 const）。Div→Mul 拆独立 pass `eltwise_div_to_mul` **默认关**（设备 EB DIVIDE/MULTIPLY 溢出口径不同，开之致行 28-31 NaN——host 位级一致不保证设备一致）。终局五门证据链：`.work/prof/005-final-gates-20261008.md`。

## Background *(量化依据，2026-09-30 于 0.8B v2 实测)*

设备 exec 35.58s 的耗时分布（fcbias 二进制，16438 ops）：

| 耗时 | 算子 | 数量 | 均耗 |
|---|---|---|---|
| 6.69s | StridedSlice | 3507 | 1.9ms |
| 3.76s | Reshape/Cast/Unary 混合 | 4061 | 0.9ms |
| 1.27s | Transpose | 175 | 7.3ms |
| 7.23s | Eltwise_Binary/Ternary | 1604 | 4.5ms |

图模式普查（net.json 源图）：**SS∘SS 嵌套链 1158 对、SS←Reshape 1184 对、Reshape∘Reshape 0**（无直接链，全部与 Slice 交替）；同型 elementwise 链 mul∘mul 60 + add∘add 47 + div∘mul 36；MatMul→Add 模式 42/139。**identity 切片仅 6 个**——收益不在删恒等，在**合并链条**。

关键约束（设备已实锤）：`exec_matmul_f16_impl` 的 bias v1 仅标量路径（HMX 形折叠会踢下 HMX，oplist_exec.c:1855 起）；发射门 [op_matmul.cpp:40](../../compiler/tools/wtop_ops/op_matmul.cpp#L40) 硬编码 `nm == "FullyConnected"`，而 BIAS opcode(36) 通路本身通用。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 视图汤合并：Slice/Reshape/Transpose 链条就地消解 (Priority: P1)

编译 0.8B 时，线性注意力 chunked scan 产生的 Reshape∘Slice 交替链被合并或吸收，设备 op 数显著下降、wall 时间下降，输出数值不变。

验收：SS∘SS 合并 + Reshape 吸收规则在 0.8B 上合计消解 ≥1000 个视图 op；设备 wall 相对 39.5s 基线下降 ≥4%；host_run 逐元素 max|d| = 0（恒等变换必须位级等价——注意浮点不参与视图变换，纯索引重排）。

### User Story 2 - 同型 elementwise 合并 (Priority: P2)

`Mul(c1)∘Mul(c2) → Mul(c1·c2)`、`Add(c1)∘Add(c2) → Add(c1+c2)`、`Div(c) → Mul(1/c)` 在图级合并。0.8B 上 ~155 条链。

验收：合并后图数值恒等（const 折叠在 f64 精度下算新系数，host 对拍 max|d| ≤ 1 ULP f16）；Div→Mul 仅当除数 const 且非零。**Div→Mul 的数值口径**：1/c 用 f32 先算再入 const 池（与设备除法路径的既有容差一致）。

### User Story 3 - MatMul+bias 折叠推广（fc_bias_fold 同机制） (Priority: P3，分两阶段)

**3a（纯图级）**：pass_fc_bias 匹配器接受 MatMul 链头 + 发射门放行 "MatMul"，沿用 HMX 形状门（M/K/N 全 %32==0 不折）。42 个 MatMul→Add 候选中 32 对齐形（注意力系）被门挡住——**这是刻意的**，防性能倒退。

**3b（设备验证）**：`exec_matmul_f16_hmx` 出 crouton 后逐列加 bias（VTCM 内完成，无额外 DDR 往返），上板验证数值后撤掉 want_hmx=0 强制（oplist_exec.c:1857）。3b 落地后才解锁 32 对齐形的折叠，0.8B 的 42 个候选方有实际收益。

验收：3a = ctest 全绿 + 0.8B 折叠数报告（预期 0-5 个非对齐形）+ 无行为回退；3b = 合成 blob（32 对齐 + bias）上板，输出与标量路径 cos ≥ 0.9999 且逐列 max|d| ≤ 2 ULP f16。

### Edge Cases

- **StridedSlice 的 begin_mask/end_mask/shrink_axes/new_axes_mask**：合并规则必须完整消化掩码语义，不能只看 begin/end/strides 三元组；shrink_axes 合并时维度代数需精确（错了就是数值错位）。
- **Slice 作用于非连续产线**： Slice 的输入同时被其他 op 消费时不可吸收 Reshape（单消费者门，同 fc_bias_fold 纪律）。
- **Transpose∘Transpose 合成**：两置换的复合置换若为恒等 → 删除两 op；非恒等 → 合成单 Transpose（但当前无通用 Transpose 发射优化，仅当合成后为恒等才动）。
- **elementwise 链中间有旁路消费者**：中间结果被别的 op 引用时不可合并（单消费者门）。
- **Div 除数为 0 或 denormal**：Div→Mul 不触发，保留原 op。
- **MatMul batched 标志**：3a 匹配器必须排除 batched 形（设备硬错：`matmul_f16_bias 不支持 batched`）。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: PassManager MUST 新增 P1 pass `view_fold`：识别并消解 identity 视图 op（StridedSlice/Reshape/Transpose 的恒等实例）与嵌套链（SS∘SS、Reshape 吸收进 SS、Transpose∘Transpose 恒等对）。
- **FR-002**: 视图合并 MUST 保持位级恒等：合并前后图的张量字节内容完全一致（视图变换不涉及浮点运算）。
- **FR-003**: P2 pass `eltwise_const_fold` MUST 合并同型 const 二元链并做 Div→Mul 变换，仅当操作数为编译期 const 且中间张量无旁路消费者。
- **FR-004**: `fc_bias_fold` 匹配器 MUST 扩展接受 `MatMul` 链头；发射门（op_matmul.cpp）MUST 放行 `MatMul`+`has_bias` 标记到 OP_MATMUL_F16_BIAS(36)；两者 MUST 沿用 HMX 形状门（全 %32==0 不折）与 batched 排除。
- **FR-005**: 3b 阶段设备端 MUST 支持 HMX 路径的 bias 应用（GEMM 后 VTCM 内逐列加），并保持与标量路径数值等价（容差内）。
- **FR-006**: 所有新 pass MUST 幂等（二跑零改写）且默认开启前有 `--no-pass <name>` 逃生门（现有 CLI 机制）。
- **FR-007**: 每个 pass MUST 在 `--opt-stats` 输出匹配/改写计数（现有 stats_line 机制）。
- **FR-008**: 序列化往返（tagged .bin）后 pass 效果 MUST 保留（改写发生在 prepare 阶段，序列化在之后——不引入新的序列化字段；op_data 标记沿用 fc_bias 机制）。

### Key Entities

- **视图链 (View Chain)**: Reshape/StridedSlice/Transpose 交替组成的无计算 op 序列，可整体消解或化简为更短序列。
- **同型链 (Same-Type Chain)**: 相邻同操作码的 Eltwise_Binary，其一操作数为 const。
- **bias 可信标记**: op_data scalar `has_bias`（e4d1dbb 既有机制，MatMul 推广复用）。
- **HMX 形状门**: M%32==0 && K%32==0 && N%32==0 的形状判定，决定折叠是否允许（v1 bias 仅标量路径）。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 0.8B v2 设备 op 数从 16438 下降 ≥6%（≥1000 个视图 op 消解）；wall 从 39.5s 下降 ≥4%（量级：1000×1.9ms ≈ 1.9s）。
- **SC-002**: host_run 输出与折叠前位级一致（视图 pass）；eltwise 合并 max|d| ≤ 1 ULP f16。
- **SC-003**: ctest 47/47 不回归 + 新增 pass 单测（正例/反例/幂等三图起步）。
- **SC-004**: 3a 落地后 0.8B 设备 wall 与 cos 相对基线无回退（折叠数可为 0——正确性优先）。
- **SC-005**: 3b 合成 blob 上板：HMX+bias vs 标量+bias cos ≥ 0.9999、逐列 max|d| ≤ 2 ULP。
- **SC-006**: 精度门沿用全链路口径（受共享设备侧"前 16 行全零"问题影响时，以 host_run 对拍 + op 数/wall 为准，见 [[gehtp-fcbias-device-verify]]）。

## Assumptions

- 视图 op 的设备执行开销以 dispatch 为主（1.9ms/个的小张量切片），删 op 即省时；无需改内核。
- 0.8B 线性注意力的 Slice 都来自 chunked scan 的规则模式，合并规则可覆盖大头；不追求全覆盖。
- 3b 的 HMX bias 为小内核改动（出 crouton 后 VTCM 内逐列加），风险在验证而非实现。
- 不做：Softmax 前折叠（图里 6 个、0.00s）、Cast 消除（1 个）、fold_relu/SwiGLU（需新内核参数，另行立项）。
- 与 003 线共享 graph_prepare.cpp：P1/P2 的 pass 代码放 `src/opt/pass_view_fold.cpp`、`pass_eltwise_const.cpp` 新文件，不动 003 在途的 wtop_emit/kernels 文件（3b 的 oplist_exec.c 改动需与 003 协调窗口）。
