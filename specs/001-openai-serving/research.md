# Research Report: OpenAI 兼容推理服务接口

**Date**: 2026-09-23
**Feature**: specs/001-openai-serving

## Research Tasks

### R1: KV-cache 在 device 上的持久化机制

**决策**: 使用现有 `OP_KV_APPEND_F16` / `OP_KV_GATHER_F16` opcode (见 `oplist_parse.h:109-110`) 作为 KV 写入/读取基础；host 端管理 cache 元数据（位置、长度、归属 session），device 端仅存储实际 tensor 数据，通过 slot 描述符访问。

**Rationale**: 
- 设备 runtime (`oplist_exec.c`) 已有 KV opcode 定义，compiler 侧 `tools/wtop_ops/` 尚未有 emitter，但这属于 compiler 开发问题，不阻塞 runtime 服务层设计
- 已有实验工件 `blobs_kv/kv_blob.wtop` 证明 KV 流程可跑
- 常驻 device runtime 需要在 init 时接收 KV buffer 分配参数，并在 decode 时接收 cache position

**Alternatives considered**:
- Host 端维护完整 KV tensor 并每次下发：可行但传输开销极大（~256MB+ per context），排除
- Device 端自建 cache 管理：增加 device runtime 复杂度，违反"host 管元数据"约束

### R2: 常驻 device runtime 的实现路径

**决策**: 分两步实现——
1. 原型期：封装现有 `gehtp run`（adb/file 方式）为可复用的 `device_runtime` 模块
2. 正式期：扩展 `kernels/examples/42_gehtp_runner/main.c` 增加常驻循环和 slot 复用

**Rationale**:
- 原型期验证协议和模型正确性，耗时短
- 正式期减少 adb/file 开销，首 token 延迟从秒级降到亚秒级
- `oplist_exec.h` 的 `wt_exec_run_io` 已支持多输入/输出槽，常驻 runtime 可复用

**Alternatives considered**:
- 直接在 host 上跑 `host_run`（x86 simulator）：性能太低，不适合流式服务
- 完全替换为 FastRPC/skel：需额外 Hexagon SDK 配置和通信层，超出内部测试范围

### R3: 多模型并发与调度策略

**决策**: 采用 host 端请求队列 + 并发上限（默认 4）。每个推理请求独占一个 KV-cache session；同一模型的多个请求可共享编译产物但 KV 隔离。

**Rationale**:
- 设备是独占硬件资源，同时执行多个推理会争用 HMX/HVX/VTCM
- 内部测试场景下 ≤ 4 并发是合理上限
- KV-cache 的持久 buffer 按 session 分配，decode 步骤通过 position 区分

**Alternatives considered**:
- 无限制并发：可能导致 device OOM 或推理质量下降
- 单请求独占设备：资源利用率太低，内部测试不必要

### R4: Host 侧 tokenizer 集成

**决策**: 优先使用 `tokenizers` Python 库（已在 `scripts/golden_qwen35.py` 中使用 `Tokenizer.from_file`），HF transformers 作为 fallback。chat template 从模型目录加载。

**Rationale**:
- `tokenizers` 库速度快，与 HF 模型兼容性好
- `golden_qwen35.py` 已有 tokenizers 使用先例
- HF transformers 已安装在 `scripts/venv_qpm210` 中

**Alternatives considered**:
- 完全自定义 tokenizer：工作量大且易错，不必要
- 仅用 HF 加载：首次 import 慢，tokenizers 更快

### R5: 指标与日志方案

**决策**: API server 内部维护请求计数器（内存中），提供 `/metrics` 端点返回 JSON。同时输出结构化日志到文件供分析。

**Rationale**:
- 内部测试不需要 Prometheus 等重型监控
- 内存计数器足够满足"请求数、延迟、错误率"指标
- 日志便于离线分析模型对比

**Alternatives considered**:
- Prometheus + Grafana：过重，内部测试不需要
- 完全依赖设备端 optrace：host 侧指标更直接

### R6: SSE 流式响应实现

**决策**: FastAPI 的 StreamingResponse + SSE 格式。每生成一个 token 输出一行 `data: {json}\n\n`。客户端断开时停止生成并释放 device 资源。

**Rationale**:
- FastAPI 对 SSE 的支持成熟
- 每 token 输出一条事件，客户端可增量渲染
- 断开检测通过 StreamingResponse 的 `close()` 回调实现

**Alternatives considered**:
- WebSocket：OpenAI 风格是 SSE，更兼容
- 批处理返回：不符合流式响应要求
