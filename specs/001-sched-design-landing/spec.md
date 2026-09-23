# Feature Specification: GEHTP 调度设计原则落地（编译器+执行器）

**Feature Branch**: `001-sched-design-landing`

**Created**: 2026-09-17

**Status**: Draft

**Input**: 把 docs/GEHTP_SCHED_HWFACTS.md 已定稿的调度/测量设计原则（E1–E7、C1–C5）转为编译器与执行器的工程实现与门禁，约束：行为保持、blob 格式兼容、每阶段独立可验收可回退；硬件实测锚点由 kernels/examples/43_hwinfo 探针提供（V81 52f67807）。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - VTCM 放置增加访问类型分类（E3） (Priority: P1)

编译器做 VTCM 驻留决策时，按张量的**访问方式**分类：只有被 HVX 向量/HMX/DMA 访问的张量才允许驻留 VTCM；被标量代码或 data-dependent 索引密集访问的 scratch 必须留 DDR。这样 M4 之后开 VTCM 驻留时不会出现"越驻留越慢"的 7 倍反噬（v75 实测教训）。

**Why this priority**: 错误的放置直接把性能做负，且 VTCM 驻留是编译期内存规划阶段二（M6 尾/M7）的核心动作；该谓词是其前置。

**Independent Test**: 用含两类张量的合成图编译，检查内存规划输出：标量/scratch 类 100% 落 DDR，向量/DMA 类可按预算驻留；存量 conv_add/L3 对拍零回归。

**Acceptance Scenarios**:

1. **Given** 一张图中同时含 GEMM 激活（向量访问）与索引 scratch（标量访问），**When** 编译产出内存规划，**Then** scratch 条目不带驻留标记、GEMM 激活可按预算带驻留标记
2. **Given** 访问类型无法判定的张量，**When** 分类器保守处理，**Then** 该张量落 DDR 且不报错
3. **Given** conv_add / spill 变体 / L3 81-op 基线，**When** 重新编译，**Then** 产物与基线逐字节一致（行为保持）

---

### User Story 2 - 可写 buffer 防去重门禁（C3） (Priority: P2)

编译/打包链路中，**可写** scratch/中间 buffer 即使字节内容全同也绝不参与常量去重合并；只读权重常量的去重保持现状。防止并行链被假共享依赖强行串行（QNN 常量 dedup 的前车之鉴）。

**Why this priority**: 代价小、防御的是隐蔽且难排查的并行性坍塌；不依赖设备即可验证。

**Independent Test**: 构造两个分支各持一份全零 scratch 的合成图，编译后检查规划产物中两份 scratch 仍为独立条目；权重去重行为不变。

**Acceptance Scenarios**:

1. **Given** 图中有两个内容全同的可写 scratch，**When** 编译，**Then** 产物中两者地址/条目独立，未被合并
2. **Given** 图中有两个内容全同的只读权重常量，**When** 编译，**Then** 去重行为与基线一致

---

### User Story 3 - HMX/HVX 分配规则编码化（C2） (Priority: P2)

编译器为算子选择执行单元时按规则决策：大矩阵（≥512 dim 级）优先 HMX；窄输出或小块（输出窄 ⇒ readback 主导）直接 HVX。规则显式可查，不允许"默认都丢给 HMX"。

**Why this priority**: 避免在小/窄算子上被 HMX 固定开销拖慢；纯编译期启发式，host 侧可测。

**Independent Test**: 用一组不同形状（大方阵/窄输出/小块）的算子编译，检查每个算子的单元指派符合规则表。

**Acceptance Scenarios**:

1. **Given** K×K≥512 维的矩阵乘，**When** 编译，**Then** 指派 HMX
2. **Given** 输出宽度 ≤128 的算子，**When** 编译，**Then** 指派 HVX 并给出可审计的决策记录

---

### User Story 4 - dominant-path 静态下界估计进产物（C1） (Priority: P3)

编译器沿定稿执行序的关键依赖链输出图的 dominant-path 静态下界估计（"理想重叠地板"），写入产物的可选元数据字段；供 M7 性能收敛期做"实测 wall − 下界 = 可调度空间"的标尺。无逐 op 实测数据时允许用内置估计表，须标注来源。

**Why this priority**: 是 M7 的标尺工具，不阻塞当前正确性战役；价值在"不用跑设备就有下界"。

**Independent Test**: 编译 conv_add 与 L3 层，检查产物含下界字段；host 侧推算的下界 ≤ 同图实测 wall（设备可用时复核）。

**Acceptance Scenarios**:

1. **Given** 任意可编译图，**When** 编译完成，**Then** 产物元数据含下界估计值与估计来源标注
2. **Given** 同一图的下界与设备实测 wall，**When** 对比，**Then** 下界 ≤ 实测 wall（不出现物理不可能的估计）

---

### User Story 5 - 性能报告三分类口径门禁（C4+§4 纪律） (Priority: P3)

一切性能报告按 真算/装料/卸料 三分类分列输出，每个数字带口径字段+形状+场景；禁止单值 lumped 报告。A/B 对比须同口径同形状同场景、取重复运行中位数。

**Why this priority**: 测量纪律是 M7 不出 phantom gap 的前提；纯报告层改造，host 可验。

**Independent Test**: 对任意基准跑输出报告，检查三分类齐全且每数字带口径/形状/场景标注；旧式单值报告被门禁拒绝。

**Acceptance Scenarios**:

1. **Given** 一次基准运行，**When** 生成报告，**Then** 输出含三分类分项且每分项带完整标注
2. **Given** 一份缺分类/缺标注的旧格式报告，**When** 门禁检查，**Then** 被拒并指出缺失项

---

### Edge Cases

- 访问类型分类器遇到无法判定的张量 → 保守落 DDR（宁慢勿错），并记可审计日志
- 无逐 op 实测数据的 dominant-path 估计 → 用内置估计表并在产物中标注 "estimated"，禁止与实测值混报
- 设备不可用（当前 52f67807 虚拟化 SSR 通道故障中）→ host 侧门禁全部可跑；设备复核项挂起不阻塞合入
- 存量 blob 不含新元数据字段 → 消费方按可选字段缺失处理，行为不变（向后兼容）

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 编译器内存规划 MUST 在 VTCM 驻留决策前对每个候选张量做访问类型分类（向量/HVX、HMX、DMA 访问 → 可驻留；标量或 data-dependent 访问 → 必须 DDR）
- **FR-002**: 分类结果 MUST 进入规划产物的可审计记录（哪个张量、哪一类、依据是什么）
- **FR-003**: 编译/打包链路 MUST 保证可写 scratch/中间 buffer 不参与常量去重合并，只读权重去重行为不变
- **FR-004**: 编译器 MUST 按显式规则表为算子指派 HMX 或 HVX（大矩阵→HMX；窄输出/小块→HVX），并输出可审计的指派记录
- **FR-005**: 编译产物 MUST 可携带 dominant-path 静态下界估计（可选字段），含估计来源标注（实测表/估计表）
- **FR-006**: 旧产物（无新字段）MUST 被所有消费方按向后兼容处理，行为逐字节不变
- **FR-007**: 性能报告生成路径 MUST 输出真算/装料/卸料三分类，且每个数字带口径字段+形状+场景标注；单值 lumped 报告 MUST 被门禁拒绝
- **FR-008**: 全部改动 MUST 满足行为保持：conv_add 三输入轮换 byte-exact、spill 变体 byte-exact、L3 81-op 对拍 |d|≤0.001、host 侧 ctest 全绿
- **FR-009**: 硬件相关常量（HVX 单元数、VTCM 上限、钟频）MUST 从实测探针（43_hwinfo）读取或标注来源，禁止硬编码他机型（v75）数字

### Key Entities *(include if feature involves data)*

- **内存规划条目 (MemPlan entry)**: 张量的放置决策记录——偏移、尺寸、驻留标记、访问类型分类、分类依据
- **下界估计 (Dominant-path estimate)**: 图级静态下界值 + 估计来源 + 关联的定稿执行序
- **性能报告 (Perf report)**: 一次基准的三分类分项集合，每项含数值、口径字段、形状、场景
- **去重门禁规则 (Dedup guard)**: 常量去重的适用谓词——仅只读常量可合并，可写 buffer 豁免

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 存量验收门 100% 零回归：conv_add 3×32768/32768 byte-exact、spill 变体 byte-exact、L3 81-op 对拍 |d|≤0.001、ctest 44/44
- **SC-002**: 合成验证图中标量/scratch 类张量 100% 不落 VTCM；向量/DMA 类张量驻留率受预算控制且可审计
- **SC-003**: 全同可写 scratch 合并事件为 0；只读权重去重命中率与基线一致
- **SC-004**: 算子单元指派 100% 符合规则表并可审计；窄输出算子误派 HMX 事件为 0
- **SC-005**: 100% 新编产物携带下界估计字段（或显式标注缺失原因）；下界估计值 ≤ 对应实测 wall 的比例 100%（无物理不可能估计）
- **SC-006**: 新格式性能报告三分类齐出率 100%；lumped 单值报告门禁拦截率 100%

## Assumptions

- 设备侧 WTOP blob 格式保持兼容优先：新元数据以可选字段/可选 tag 方式加入，旧消费方读到缺失即走原路径
- 43_hwinfo 探针数字（HVX 单元数、PCYCLE/µs、锁语义）待 52f67807 恢复后回填；回填前编译器侧改造不依赖这些具体数值（规则表以形状阈值表达，不以绝对频率表达）
- 43_hwinfo 复测命令：`cd kernels && ./examples/build_examples.sh 43`
- 多会话错峰纪律沿用（动工前查 mtime + 会话排序），VL 线在途文件（graph_prepare.cpp/ops.cpp/host_run.cpp）需避让
- 本特性不做跨 op 软件流水（E7 所指的 supertile 级重叠）——那是 M7 之后的独立评估项，本期只建立测量标尺
