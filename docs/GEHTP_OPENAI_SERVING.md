# GEHTP OpenAI 兼容推理服务 — 边界情况与已知限制

**Feature**: `specs/001-openai-serving`
**实现**: `scripts/gehtp_server.py` + `scripts/{device_runtime,model_registry,sessions,tokenizer,sampler,metrics,errors,logging_config}.py`
**运行环境**: `/disk1/miniforge3/envs/llmserv/bin/python`（fastapi/uvicorn/tokenizers/httpx + pytest 已安装）

## 架构分工

- **host**：tokenizer、请求会话、采样（当 runtime 返回 logits 时）、KV-cache 元数据、请求队列、指标。
- **device**：KV-cache tensor 实际数据、模型执行。通过 `DeviceRuntime` 接口访问（`init/run/cancel/reset/release/health/shutdown`）。

## 已知限制（原型期）

### 1. 原型 runtime 的返回值约定

默认 `DeviceRuntime` 是 host-only 原型，`run()` 返回固定 token id `0`，不代表真实模型执行。
约定如下：

- 返回 **int** → 视为 device 已完成的采样决策，host 直接使用该 token，不再过 sampler。
- 返回 **list[float]** → 视为 logits，host 按 `temperature`/`top_p` 采样（`temperature=0` 走 greedy）。

接入常驻 V81 runtime 时按上述约定返回其中一种即可。

### 2. KV-cache 容量

- 模型注册加载时分配 device slot，KV 预算 `kv_max = max(4096, input_length + 1024)`。
- 请求时校验 `prompt_tokens > input_length` → 400 `prompt_too_long`；
  `prompt_tokens + max_tokens > kv_max` → 400 `kv_budget_exceeded`。
- 每个请求的 session KV 元数据独立（`SessionManager`），但**同一模型的并发请求共享一个 model-level device slot**（原型期限制）。
  常驻 runtime 落地后应改为 per-session device handle。

### 3. cancel 语义（共享 slot 限制）

`runtime.cancel()` 以 handle 为粒度。由于同模型请求共享 slot，断开连接的 cancel 会影响该 slot 上的其它请求。
原型 runtime 的 `run()` 是同步的，cancel 无法中断已开始的调用；服务在 cancel 后立即 `reset()` 清除标志，
避免毒化该 slot 上的下一个请求。常驻 runtime 应实现 per-session cancel 后移除 `reset()` 调用。

### 4. 流式 stop 序列可能泄漏前缀

stop 检测针对已拼接的完整文本做 `find()`，并精确截断、只发送 stop 之前的增量。
但若 stop 序列跨越多个 token（如序列 `"abc"`，token 分别为 `"ab"`、`"c"`），
第一个 token 的内容块**已经发出**，客户端会看到 stop 序列的前缀片段。
非流式响应不受影响（始终精确截断）。

### 5. 客户端断开

- 流式：每 token 前检查 `request.is_disconnected()`；断开时 `runtime.cancel()` + session 标记 error + 释放 KV 元数据。
  StreamingResponse 被关闭（GeneratorExit）时在 `finally` 中执行同样的清理。
- 断开的请求**不计入** metrics 的 success/failure（避免污染成功率），只记录日志。

### 6. 并发与排队

- 并发上限 4（`MAX_CONCURRENT_REQUESTS`），等待队列上限 16（`MAX_QUEUED_REQUESTS`）。
- 队列满 → 429 `queue_full`（`rate_limit_error`）。
- 等待时间计入 `queue_latency_ms` 指标。

### 7. 错误码映射

| HTTP | 场景 |
|------|------|
| 400 | 参数非法（model/messages/max_tokens/temperature/top_p/stop/stream）、JSON 格式错误、prompt 超长、KV 预算超出、wtop/manifest 路径不存在 |
| 404 | 模型未注册 |
| 409 | 模型 loading / unavailable / 重复注册 / 重复加载 |
| 429 | 等待队列已满 |
| 500 | tokenizer 加载/编码失败 |
| 503 | device runtime 错误（`DeviceError`） |
| 504 | 预留给 device 执行超时（原型 runtime 同步执行，暂未使用） |

### 8. 注册是异步的

`POST /v1/models/register` 立即返回 202 + `status: loading`；
manifest 解析、device init、tokenizer 预加载在后台完成。
结果通过 `GET /v1/models` 观察（`available` / `unavailable` / `error`，失败原因在 `error` 字段）。
TestClient 会等待后台任务完成，因此测试中注册后立即可用。

### 9. 优雅关闭

lifespan shutdown：释放所有 session 的 KV 元数据（`session_manager.clear()`），
对每个模型 `runtime.release(device_slot)` 并置为 `unavailable`。
uvicorn 收到 SIGTERM/SIGINT 后走完该流程（已实测验证）。

### 11. 持久化 runtime（PersistentDeviceRuntime）

`scripts/persistent_runtime.py` 提供 `PersistentDeviceRuntime`，通过 Unix socket 与设备端 `42_gehtp_persistent` 通信，避免每次请求的 adb push/pull。

**协议**：
- 请求：4 字节长度（big-endian）+ JSON header（`{"handle_id": str, "input_ids": [int], "position": int, "kv_position": int}`）
- 响应：4 字节长度 + JSON（`{"token": int}` 或 `{"logits": [float]}`）

**生命周期**：
- `init(model_id, kv_max)` 创建 `PersistentHandle`，注册到 runtime 内部表
- `run(handle, input_ids, position, kv_position)` 通过 socket 发送请求并解析响应
- `cancel(handle)` 标记取消，下次 `run()` 抛出 `PersistentRuntimeError`
- `reset(handle)` 清除取消标志
- `release(handle)` 移除 handle
- `health()` 返回 socket 连通性与活跃 handle 数

**设备端**：`kernels/examples/42_gehtp_persistent/main.c` 监听 `/data/local/tmp/gehtp/runtime.sock`，维护最多 4 个模型槽位，持久化 `wt_blob` 避免重复解析。

**测试**：`tests/test_persistent_runtime.py`（19 项，全部通过）。

**限制**：
- 原型期 `parse_and_exec()` 返回占位 token `0`，不代表真实模型执行
- 同一模型的并发请求共享一个 model-level device slot（与 `DeviceRuntime` 一致）
- 设备端 socket server 需手动启动：`adb shell "/data/local/tmp/hvxhmx23/gehtp_persistent"`

## 测试

```bash
cd /disk1/gehtp
/disk1/miniforge3/envs/llmserv/bin/python -m pytest tests/ -q
```

- `tests/contract/test_chat_completions.py` — 请求/响应格式、SSE 事件格式、错误码（35 项中的 23 项）
- `tests/integration/test_us1_single_request.py` — 注册→列表→对话→指标全链路、注册冲突、unload、多模型
- `tests/test_persistent_runtime.py` — 持久化 runtime 协议、生命周期、重连（19 项）
