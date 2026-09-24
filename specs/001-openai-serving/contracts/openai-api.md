# OpenAI 兼容 API Contract

**Base path**: `/v1`
**Purpose**: 面向内部模型效果和性能测试的兼容接口。

## GET /v1/models

返回已注册模型。

### Response 200

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen35-0.8b",
      "object": "model",
      "created": 1720000000,
      "owned_by": "gehtp",
      "status": "available"
    }
  ]
}
```

`status` 取值：`loading`、`available`、`unavailable`、`error`。

## POST /v1/chat/completions

### Request

```json
{
  "model": "qwen35-0.8b",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true,
  "stop": ["<|im_end|>"]
}
```

### Request rules

- `model` 必填，必须匹配 `GET /v1/models` 中状态为 `available` 的模型。
- `messages` 必填且非空；支持 `system`、`user`、`assistant` 角色。
- `max_tokens` 默认 16，必须为正整数且不超过模型配置的上限。
- `temperature` 默认 1.0，范围为 [0, 2]；值为 0 时使用 greedy sampling。
- `top_p` 默认 1.0，范围为 (0, 1]。
- `stream` 默认 false；本期仅支持 SSE 流式输出，不支持 WebSocket。

### Non-stream response (200)

```json
{
  "id": "chatcmpl-gehtp-<request-id>",
  "object": "chat.completion",
  "created": 1720000000,
  "model": "qwen35-0.8b",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "回复文本"},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 4,
    "total_tokens": 7
  }
}
```

### Stream response (200)

Response `Content-Type` 为 `text/event-stream`，每个增量事件以空行结束：

```text
data: {"id":"chatcmpl-gehtp-<request-id>","object":"chat.completion.chunk","model":"qwen35-0.8b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-gehtp-<request-id>","object":"chat.completion.chunk","model":"qwen35-0.8b","choices":[{"index":0,"delta":{"content":"你"},"finish_reason":null}]}

data: {"id":"chatcmpl-gehtp-<request-id>","object":"chat.completion.chunk","model":"qwen35-0.8b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

客户端断开后，服务端取消当前请求并释放或回收对应的 KV-cache。

## POST /v1/models/register

内部管理接口，用于注册已编译模型。

### Request

```json
{
  "id": "qwen35-0.8b",
  "wtop_path": "/models/qwen35-0.8b/model.wtop",
  "manifest_path": "/models/qwen35-0.8b/model.wtop.manifest.json",
  "weights_path": "/models/qwen35-0.8b/model.weights.bin",
  "tokenizer_path": "/models/qwen35-0.8b/tokenizer.json"
}
```

### Response

返回模型 ID 和 `loading` 状态；加载完成后通过 `/v1/models` 观察状态。

## GET /metrics

返回内部测试指标 JSON。至少包含：

- 按模型统计的请求总数、成功数、失败数
- 排队时间、首 token 延迟、总生成延迟的平均值和 p50/p95
- prompt token 数、completion token 数
- 当前队列长度、活跃 session 数、device runtime 状态

## Error format

```json
{
  "error": {
    "message": "model is unavailable",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_unavailable"
  }
}
```

HTTP 状态码：

| 状态码 | 场景 |
|--------|------|
| 400 | 请求格式或参数非法 |
| 404 | 模型不存在或 endpoint 不存在 |
| 409 | 模型正在加载或资源冲突 |
| 429 | 队列已满 |
| 500 | 服务内部错误 |
| 503 | device runtime 不可用 |
| 504 | device 执行超时 |
