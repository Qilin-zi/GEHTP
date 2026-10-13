# Feature Specification: 算子输入契约单点注册表

**Feature Branch**: `011-op-contract-registry`

**Created**: 2026-10-12

**Status**: Draft

**Input**: User description: "起一个新的 011 spec：算子输入契约单点注册表迁移。P0（合并 arity + 身份两张 switch 表）低风险高收益，根治 arity 6→9 强耦合（010 US2 blocked）与陈旧 arity 致 rc=-11（设备库 stale arity）两类坑；P1（语义机器化）把 arg 语义注释变成校验；P2（按名取参）收益最大风险最高，单独立项用 byte-exact 门压着做。"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - arity 与参数身份单一事实源 (Priority: P1)

作为编译器/执行器维护者，当我新增或修改一个算子（opcode）时，我希望它的 arity（参数个数）与每个参数的「身份」（槽引用 / temp / 标量 imm / 状态槽）只在一处声明，使运行时 arity 校验、slot 身份判定表、以及编译侧 emit 编码三者之间不再可能因分开手工维护而漂移。

**Why this priority**: 这是 P0 步——风险最低、收益最高。它从根上消除两类已记录的坑：(a) arity 6→9 强耦合（010 US2 blocked：W4A16 的 bias/atbl/otbl 三槽与 6 参 emit 强耦合）；(b) 陈旧 arity 致 rc=-11（设备侧执行器库 stale arity：opcode 28 的 arity 12→14 陈旧）。两者同源：arity 与身份散落在 2–3 张靠约定手工对齐的表里。

**Independent Test**: 合并后，改动任一算子的 arity 或身份只改一处；现有 byte-exact 回归套件（35 例 + 261 门 + W4A16 65536/65536）的编译产物逐位不变。

**Acceptance Scenarios**:

1. **Given** 36 个 opcode 的 arity 目前分散在 `#define WT_ARITY_*` 宏与 `arity_of()` switch 两处，**When** 合并为单一声明表，**Then** 任一 opcode 的 arity 只在一处声明，且运行时查询与宏取值逐值一致。
2. **Given** `arg_is_slot()` switch 手工维护每个 opcode 的 slot 下标，**When** 身份信息并入同一张声明表，**Then** 结构上不可能出现「arity 声明 6 个参数而身份表标 9 个下标」这类自相矛盾。
3. **Given** 现有 blob 解析（wt_parse）依赖 arity 校验，**When** 用新表替换旧 arity 来源，**Then** 所有已编译 blob 仍可解析，不再出现陈旧 arity 导致的 rc=-11。

---

### User Story 2 - 参数语义机器可读 (Priority: P2)

作为编译器/执行器维护者，我希望每个参数的语义含义（激活、权重、M、K、N、bias 等）以机器可读的命名字段记录在同一注册表里——而不是只写在 enum 注释里——使「文档语义」与「实际解码」的不一致在构建阶段即可被校验，而不是等到运行期以静默数值错误浮现。

**Why this priority**: 这是 P1 步。当前唯一的权威语义描述活在 enum 注释（散落地图第④处），编译器看不到它，也就无法防漂移。语义机器化把「约定」升级为「校验」。

**Independent Test**: 对每个 opcode 的每个参数，注册表中的语义名可与 exec_* 实际解码含义自动比对；引入一处故意的语义-解码不一致会在构建/校验阶段报错。

**Acceptance Scenarios**:

1. **Given** 每个参数的语义目前只写在 enum 注释里，**When** 语义以命名字段进入注册表，**Then** 每个参数的语义可被程序读取，而非仅人类可读。
2. **Given** exec_* 目前按 `args[n]` 硬下标解码，**When** 注册表提供按语义名索引的能力，**Then** 语义名与下标位置的对应关系存在单点校验，不再依赖人工对齐。

---

### User Story 3 - 按名取参（本 spec 范围外，后续单独立项） (Priority: P3)

作为编译器/执行器维护者，我最终希望 exec_* 函数按语义名访问参数，而非按硬编码的位置下标 `args[n]`，使新增/删除一个参数不再需要逐一审计 37 个 exec_* 函数里的每个硬下标。

**Why this priority**: 这是 P2 步——收益最大（加算子不再手写位置解码），但风险也最高（触及全部 34 个算子的解码路径）。按特性描述，此步**建议单独立项、用 byte-exact 门压着做**，因此**明确不在本 011 spec 范围内**，作为后续独立项目。

**Independent Test**: 不适用——延后至独立项目。

**Acceptance Scenarios**: 不适用——延后。（仅在此标注范围边界。）

---

### Edge Cases

- 变长结构算子（CONCAT/SPLIT 的 segment 数组、Gather 的打包位域 `axis|xrank<<8|irank<<16`）：声明表须能表达「向量参数 / 打包位域」，不能被强制拆成固定 arity。
- 保留空洞 opcode（31、32 未使用）：声明表须容纳「无实现」的 opcode 而不报 arity 错误。
- 设备侧执行器库（.signed.so）与宿主编译器 lib 的陈旧 arity 不一致：合并后须保证两边读到同一份契约，而非各自编译各自的陈旧副本。
- 无 slot 参数的算子（SILU、IM2COL、ADD、TRANSPOSE_F16）：`arg_is_slot()` 目前靠 default 返回 0，合并后须等价保留「默认无 slot」语义。

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 提供单一声明式注册表，为每个 opcode 记录 {arity, 每个参数的 {类型, 语义名}}。
- **FR-002**: 运行时 arity 查询（当前 `arity_of()`）与编译期 arity 宏（`#define WT_ARITY_*`）MUST 改从该注册表读取，两者取值 MUST 与迁移前逐值一致。
- **FR-003**: slot/temp 身份判定（当前 `arg_is_slot()`）MUST 改从该注册表的参数类型字段读取，并 MUST 保留「未声明 slot 下标则默认无 slot」的既有语义。
- **FR-004**: 每个参数的语义 MUST 以机器可读的命名字段记录（替换 enum 注释里的人工描述），使语义可被程序校验。
- **FR-005**: 参数编码分类 MUST 覆盖全部 5 类：slot 引用（0x8000）、裸 temp id、VTCM 驻留 temp（0x4000）、状态槽（0x6000）、标量/打包位域 imm。
- **FR-006**: 迁移 MUST 保持 byte-exact——现有全部回归套件（35 例 + 261 门 + W4A16 65536/65536）的编译产物 MUST 与迁移前逐位一致。
- **FR-007**: 迁移 MUST 在结构上消除 arity 与身份两表之间的自相矛盾——「分开维护导致的不一致」MUST 不可能再发生。
- **FR-008**: 本 spec MUST NOT 改动 exec_* 函数的参数解码方式（`args[n]` 硬下标）——按名取参属 P2，归后续单独项目。

### Key Entities *(include if feature involves data)*

- **算子 (opcode)**: 36 个已定义算子（编号 0–37，31/32 为保留空洞）。属性：opcode 编号、arity（参数个数）。
- **算子参数 (arg)**: 每个 opcode 的每个参数。属性：类型（slot / temp / imm / state）、语义名（激活、权重、M、K、N、bias 等）。
- **算子契约注册表 (op contract registry)**: 上述两者的单一声明表，是 arity / 身份 / 语义的唯一事实源。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 全部现有回归（35 例 + 261 门 + W4A16 65536/65536）在迁移后**逐位一致**（byte-exact），零回归。
- **SC-002**: arity 与身份的定义从 2 张 switch 表（外加宏）合并为 1 张声明表——新增或修改任一算子的 arity/身份只需改动一处。
- **SC-003**: 语义描述从「仅 enum 注释」变为「机器可读字段」，100% 覆盖 36 个 opcode 的全部参数。
- **SC-004**: 两类已知坑（arity 6→9 强耦合、陈旧 arity rc=-11）在结构上被根除——不再依赖「人工保持多表同步」。

## Assumptions

- 目标读者是编译器/执行器维护者（GEHTP 团队），而非产品终端用户——这是内部工具链重构，spec 以维护者视角书写。
- P2（按名取参，改动 exec_* 解码路径）明确不在本 spec 范围内，作为后续独立项目。
- 现有 byte-exact 回归套件（35 例 + 261 门 + W4A16 65536/65536）是迁移正确性的唯一裁判口径，迁移前后产物逐位对比。
- 设备侧执行器库与宿主编译器 lib 须共享同一份契约源（或由同一生成步骤产出），以避免陈旧 arity 复发。
- 5 类参数编码分类（slot / temp / VTCM temp / state / imm）足以覆盖当前全部 36 个 opcode。
