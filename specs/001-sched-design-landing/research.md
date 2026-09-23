# Phase 0 Research: 调度设计原则落地 — 五个落地件的技术决策

> 2026-09-17。每条 = Decision / Rationale / Alternatives。所有代码锚点已对源码逐条核实。

## R1（对应 US1/FR-001/FR-002）：访问类型分类器落在哪、怎么分类

**现状实证**：`compiler/src/vtcm/ddr_offsets.cpp:83-107` 的两级分配中，`vtcm_budget > 0` 时
FancyAllocator 对**全部** alloc 请求先试 VTCM 池——**装得下就驻留，没有任何访问类型谓词**。
这就是 E3 要补的洞。

**Decision**: 在 `compute_ddr_offsets` 构建 `reqs` 之后、VTCM 池分配之前，加一个
**消费方驱动的分类 pass**：
- 分类粒度 = alloc 请求（张量 = 生产 op 的输出）
- 分类依据 = **消费方**算子的访问模式：`opdef->consumers` 已存在
  （`graph_prepare.cpp:1184` 实证可用），查静态表 `opcode → 每操作数访问类`
  （VECTOR / HMX / DMA / SCALAR / UNKNOWN）
- 规则：任一消费者为 SCALAR（gather 索引、data-dependent slice、标量循环 scratch 等）
  或 UNKNOWN → 该张量禁入 VTCM（保守 DDR）；全部消费者为 VECTOR/HMX/DMA → 可驻留
- 静态表放 `include/hnnx/ir/op_def.hpp` 旁的独立小表（不改既有 OpDef 布局，零 ABI 风险）

**Rationale**: 访问类型是**消费者属性**而非生产者属性（同一张量被 MatMul 读 = DMA 友好，
被 Gather 当索引 = data-dependent 标量访问）；消费方驱动是唯一能抓住这个语义的分类法。
保守默认（UNKNOWN→DDR）符合"宁慢勿错"。

**Alternatives considered**:
- 生产者属性表（按产出 op 分类）——否：抓不住"被谁怎么用"，Gather 的索引输入就是反例
- 自动数据流分析推断访问模式——否：过度工程，20 型 opcode 家谱一张静态表就够
- 设备侧运行时再纠正——否：违反"规划唯一真相源在编译器"（MEMPLAN §0 决策）

## R2（对应 US2/FR-003）：去重门禁的实现形态

**现状实证**：去重只在 `wtop_emit.cpp:316` 的权重打包（"建 f16 槽，按 const id 去重"）。
可写 scratch/中间 temp 走 `fresh_temp`（`wtop_ops.hpp:136-163`），**不经过任何去重路径**。
即当前不存在 C3 所述的合并行为；风险是**未来**有人给"优化"加合并。

**Decision**: 三件套——(a) 单测：两个内容全同的可写 scratch 经编译后仍是独立规划条目；
(b) `wtop_emit.cpp:316` 去重循环处加注释门禁（"仅只读权重常量；可写 buffer 永不合入"）；
(c) 回归门进 ctest。

**Rationale**: 行为保持约束下不应为一个不存在的行为改写代码；门禁的价值是防回归。
QNN 把全零 scratch dedup 成共享 VTCM buffer 造成假 WAW 串行的案例（skill 实录）证明
这个坑一旦出现极难排查，值得花小钱设防。

**Alternatives considered**:
- 主动重构 emit 加"可写性"字段——否：过度设计，现有路径本就分离
- 只靠 code review 约定——否：无测试的约定在多会话并行下会破

## R3（对应 US3/FR-004）：HMX/HVX 规则表=记录+守门，不改派发

**现状实证**：wtop_ops/ 16 个 handler 按 opcode 固定选择 kernel/单元——**没有显式规则表，
也没有窄输出守卫**。

**Decision**: 初版只做两件事（行为保持的红线内）：
- 在 op lowering 处记录每个算子的单元指派及理由（审计日志，编译期 `--audit` 输出）
- 规则表 + 一个**警告级**守门：窄输出（输出宽度 ≤ 阈值，初版 128）被派到 HMX 时
  stderr 警告（不 fail，不改派发）。阈值参数化，43_hwinfo 回填后校准
- 真正按规则**改变**派发是后续独立特性（需设备 A/B 验证，M7 期）

**Rationale**: 改变派发 = 行为变化，违反本特性"行为保持"约束；先把可观测性和规则表
立起来，后续改派发才有对照基线。

**Alternatives considered**:
- 本期直接改派发——否：违反行为保持；且无 V81 实测阈值（设备故障中）
- 不做——否：US3 的可审计性是 M7 调优的前置输入

## R4（对应 US4/FR-005/FR-006）：dominant-path 下界走 manifest，不进 blob

**现状实证**：manifest.json 由 `wtop_emit.cpp:617-646` 写出（input_slot/output_temp/n_ops
等）；tagged.bin tag 表（`serializer.hpp:108-131`）改动会波及 deserialize 三方消费
（host_run/wtop_emit/round-trip 测试）。

**Decision**: manifest.json 新增可选字段：
```json
"dompath_estimate": {"cycles": <u64>, "source": "table"|"measured", "per_op_table_version": 1}
```
- 值 = 沿定稿序（TAG_PLAN_ORDER 已保证拓扑）按关键链求和；初版用编译器内置
  per-opcode 粗粒度周期表（`table` 标注），43_hwinfo 与后续 optrace 测量校准后可升级
  `measured`
- **tagged.bin 与 .wtop blob 零改动**——兼容性问题直接消失（FR-006 自动满足）

**Rationale**: 下界估计是给 host 侧 M7 用的标尺，设备执行器根本不需要它；manifest 是
它的天然载体。"下界 ≤ 实测 wall" 的不变式（SC-005）在 host 即可校验（有 host_run
对拍与历史设备数字）。

**Alternatives considered**:
- 新 tag 进 tagged.bin（如 0x4450 'DP'）——否：违反 blob 兼容优先，且无设备侧消费者
- 等实测数据齐了再做——否：估计表先上车，字段与口径先稳定，校准是数据演进不是格式演进

## R5（对应 US5/FR-007）：报告门禁 = lint 脚本 + 模板，既有报告不追溯

**Decision**: 新增 `scripts/perf_report_lint.py`：
- 校验性能报告含 真算/装料/卸料 三分类分项，每项带 口径字段（dompath/cycles_used/
  graph-wall/retire-interval 之一）+ 形状 + 场景（单发/序列）标注
- 缺项 → 非零退出 + 指出缺失项；附报告模板生成（`--init`）
- 既有 `kernels/PERF_REPORT.md`、`results/*.txt` 不追溯（门禁只拦新报告）
- 口径词汇表引用 `docs/GEHTP_SCHED_HWFACTS.md` §4，不复制定义（单一事实源）

**Rationale**: 纪律如果没有 CI 式门禁就会在多会话并行下退化；lint 是最便宜的门禁形态。

**Alternatives considered**:
- 改造既有报告生成器统一出口——否：现有多为手工报告，没有统一出口可改
- 追溯修订历史报告——否：沉没成本，无行动价值

## 横向：多会话错峰（FR-008 的执行约束）

- 本特性 P1 触碰 `graph_prepare.cpp`（分类 pass 调用点）——VL 线在途文件，
  动工前必须查 mtime + 会话排序（MEMPLAN §4.4 同规则）
- `wtop_emit.cpp`、`ddr_offsets.cpp`、`op_def.hpp` 旁新表、`scripts/` 新脚本均为
  低冲突面
- 设备相关复核（quickstart、43_hwinfo 回填）在 52f67807 恢复前挂起，不阻塞 host 门合入
