# Quickstart: OpenAI 兼容推理服务接口

**Purpose**: 证明从模型注册到流式回复的全链路可运行。

## Prerequisites

- V81 设备（52f67807）通过 adb 可达
- `gehtp setup` 已完成（kernels 库和 runner 已推送到 `/data/local/tmp/hvxhmx23`）
- 已有一个编译好的 `.wtop` 模型（如 `qwen35_08b`），附带 manifest 和 tokenizer
- Python 3.10+，pip 安装 `fastapi uvicorn tokenizers`

## Step 1: 启动 device runtime 会话

在 device 端启动常驻 runner（或使用封装后的常驻模式）：

```bash
# 方式 A: 使用现有 gehtp run（验证期，每次 adb push/pull）
gehtp run qwen35_08b/model.wtop --input prompt.f16.raw --output out.f16.raw --device 52f67807

# 方式 B: 常驻模式（开发阶段）
# 需要先改造或封装 42_gehtp_runner 支持常驻循环
adb shell "/data/local/tmp/hvxhmx23/gehtp_runner --persist"
```

## Step 2: 启动 API server

```bash
cd /disk1/gehtp
# 假设 server 实现为 scripts/gehtp_server.py
python3 scripts/gehtp_server.py --host 0.0.0.0 --port 8000
```

## Step 3: 注册模型

```bash
curl -X POST http://localhost:8000/v1/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "qwen35-0.8b",
    "wtop_path": "/disk1/gehtp/test_models/qwen35_08b/model.wtop",
    "manifest_path": "/disk1/gehtp/test_models/qwen35_08b/model.wtop.manifest.json",
    "weights_path": "/disk1/gehtp/test_models/qwen35_08b/model.weights.bin",
    "tokenizer_path": "/disk1/gehtp/test_models/qwen35_08b/tokenizer.json"
  }'
```

验证模型状态：`GET http://localhost:8000/v1/models`，返回 `"status": "available"`。

## Step 4: 提交推理请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-0.8b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32,
    "stream": true
  }'
```

验证结果：
- 非流式：返回包含 `choices[0].message.content` 的 JSON
- 流式：返回多条 `data:` 事件，最终 `data: [DONE]`

## Step 5: 查看指标

```bash
curl http://localhost:8000/metrics
```

返回包含请求计数、平均延迟、队列长度等字段的 JSON。

## Validation Checklist

- [ ] 模型注册后状态变为 available
- [ ] 单次非流式请求返回正确格式回复
- [ ] 流式请求逐 token 输出 SSE 事件
- [ ] 错误模型 ID 返回 4xx 错误
- [ ] 超过并发上限时返回 429 或排队
- [ ] 指标端点包含请求数和延迟数据
