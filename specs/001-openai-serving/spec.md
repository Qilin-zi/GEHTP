# Feature Specification: OpenAI 兼容推理服务接口

**Feature Branch**: `task/prof-wp`

**Created**: 2026-09-23

**Status**: Draft

**Input**: User description: "为 GEHTP 增加一个面向内部测试部署的 OpenAI 兼容推理服务接口。服务需要支持多模型注册与选择、POST /v1/chat/completions、流式 SSE 响应；host 负责 tokenizer、请求会话、采样和 KV-cache 元数据管理，KV-cache 实际数据常驻 device。当前 GEHTP 是 ONNX/QNN IR 编译为静态 .wtop 后由 V81 设备 runtime 执行的工具链；服务目标是比较不同编译模型的效果和性能，不是生产级多租户平台。需要覆盖模型生命周期、请求并发/排队、错误处理、指标记录，以及从现有 adb/file runner 过渡到可复用的 device runtime 会话。"

## User Scenarios & Testing

### User Story 1 - 提交单次对话请求并获得回复 (Priority: P1)

**场景**: 研究人员希望快速测试某个已编译模型的推理效果，提交一个对话请求并得到流式回复。

**Why this priority**: 这是最核心的使用场景，其他功能都是围绕此展开的基础设施。

**Independent Test**: 可以通过调用 `/v1/chat/completions` 接口，传入一个消息和已注册的模型 ID，验证是否返回符合 OpenAI 格式的 SSE 流，并能正确解码出完整回复。

**Acceptance Scenarios**:

1. **Given** 有一个已注册的模型 "qwen35-0.8b"，**When** 发送包含 `{"model": "qwen35-0.8b", "messages": [{"role": "user", "content": "你好"}]}` 的请求，**Then** 返回 SSE 流的最终消息内容
2. **Given** 模型正在编译中或未注册，**When** 使用该模型 ID 发送请求，**Then** 返回明确的错误信息说明模型不可用
3. **Given** 请求包含无效的消息格式，**When** 发送到 API，**Then** 返回 4xx 错误并说明问题所在

---

### User Story 2 - 注册和管理多个模型 (Priority: P2)

**场景**: 工程师编译了多个版本或配置的模型，需要在服务中注册以便对比它们的性能和效果。

**Why this priority**: 多模型对比是服务的主要目的，必须在完成 P1 之后实现。

**Independent Test**: 可以通过 API 或配置文件注册多个模型，查询模型列表能获取所有已注册模型及其状态信息。

**Acceptance Scenarios**:

1. **Given** 有一个编译好的 `.wtop` 文件和对应的 manifest，**When** 通过 API 注册模型，**Then** 模型被添加到可用列表中并可被选择
2. **Given** 已有多个模型，**When** 请求模型列表，**Then** 返回所有模型的基本信息和状态（如：可用/加载中/不可用）
3. **Given** 某个模型文件损坏或无法加载，**When** 尝试使用该模型，**Then** 返回清晰错误并标记该模型为不可用状态

---

### User Story 3 - 流式响应以实时观察生成分 (Priority: P3)

**场景**: 研究者希望在生成过程中实时看到每个 token 的输出，用于调试和分析模型行为。

**Why this priority**: 增强体验而非核心功能，P1 完成后即可提供基础可用性。

**Independent Test**: 设置 `stream=true` 参数，验证服务器返回多条 SSE `data:` 事件，每条包含部分文本。

**Acceptance Scenarios**:

1. **Given** 发起一个流式请求 (`stream=true`)，**When** 接收响应，**Then** 每秒至少收到 1-3 条增量 token 的 SSE 事件
2. **Given** 流式响应进行中，**When** 客户端断开连接，**Then** 服务端停止生成并释放资源

---

### User Story 4 - 查看请求指标和日志 (Priority: P4)

**场景**: 管理员希望了解系统的负载情况、各模型的响应时间和成功率。

**Why this priority**: 用于分析性能瓶颈和模型优劣对比，属于运营监控层面。

**Independent Test**: 发送若干请求后，查询指标端点能看到请求计数、平均延迟、错误率等数据。

**Acceptance Scenarios**:

1. **Given** 系统已处理若干请求，**When** 查询 `/metrics` 或访问日志，**Then** 能看到各模型的请求数、平均/分位数延迟、成功/失败计数
2. **Given** 某次请求失败，**When** 查看日志，**Then** 记录失败的请求 ID、模型、错误原因和时间

---

### Edge Cases

Given that this is for internal model comparison, these are the expected edge cases:

- Device becomes busy or unresponsive: Service queues requests or marks them as error after timeout
- KV-cache exceeds device capacity: Host-side GC triggers; for exceedingly long contexts, truncate or split across batches
- Model fails to load or weights missing: Return FR-007 style error, mark model status as "unavailable"  
- Stream connection lost mid-generation: Server stops generation, client can reconnect with fresh request
- Multiple concurrent requests targeting same model: Serialize at host level with per-request KV-cache isolation
- Tokenizer vocabulary mismatch or missing: Host-side error with clear message
## Requirements

### Functional Requirements

- **FR-001**: 系统 MUST 暴露 REST 接口兼容 OpenAI 的 `/v1/chat/completions` API
- **FR-002**: 系统 MUST 支持多模型注册，每个模型有唯一标识和状态（可用/加载中/不可用）
- **FR-003**: 系统 MUST 在 host 侧管理 KV-cache 元数据和采样逻辑
- **FR-004**: 系统 MUST 将 KV-cache 实际数据驻留在 device 以减少传输开销
- **FR-005**: 系统 MUST 支持 `stream=true` 参数并以 SSE 形式返回增量 token
- **FR-006**: 系统 MUST 在 host 侧完成 tokenizer 编码/解码
- **FR-007**: 系统 MUST 记录每次推理请求的指标（延迟、token 数、错误）
- **FR-008**: 系统 MUST 提供模型列表查询接口
- **FR-009**: 系统 MUST 支持 graceful shutdown 和资源释放

### Key Entities

- **Model**: 代表一个已编译的 GEHTP 模型，属性包括：ID、路径（.wtop + manifest）、状态、元数据（输入长度、输出格式、编译时间）
- **Session**: 代表一个推理会话，属性包括：Session ID、关联模型、KV-cache 句柄、消息历史、状态（active/finished/error）
- **Request**: 代表一次推理请求，属性包括：Request ID、Session ID、消息、参数（max_tokens, temperature, top_p, stream）、结果

## Success Criteria

### Measurable Outcomes

- **SC-001**: 研究人员能在 5 分钟内完成模型注册并提交第一次推理请求
- **SC-002**: 单次对话请求（prompt ≤ 128 tokens, 生成 ≤ 64 tokens）端到端延迟 ≤ 2 秒（在 V81 设备上）
- **SC-003**: 系统能同时服务 ≥ 4 个并发推理请求而不崩溃
- **SC-004**: 流式响应首 token 延迟 ≤ 500ms（设备冷启动除外）
- **SC-005**: 支持至少 3 个不同模型的同时在线（如 qwen35_0.8b, qwen35_4b, minicpm5_2b）
- **SC-006**: 服务正常运行时间 ≥ 8 小时无需重启（内部测试期间）
- **SC-007**: 错误发生时用户能得到清晰的错误信息（≥ 95% 的错误场景可解释）

## Assumptions

- 假设已有可用的编译工具链（compiler + kernels）能够产出有效的 `.wtop` 文件
- 假设 V81 设备已经配置好且能够通过 adb 访问
- 假设 tokenizer 文件已存在于模型目录或可通过 HF transformers 加载
- 假设当前没有现成的 device runtime 常驻进程，需要先改造或封装现有的 `gehtp run`
- 假设内部测试场景下，设备是独占的或竞争可控
- 假设编译好的模型数量在 5-10 个以内，不会频繁更换
- 假设不需要多租户隔离，同一时间只有少数几个研究人员使用
- 假设流式响应以 SSE 为标准格式即可，不需要特殊的断线重连协议
