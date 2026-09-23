# Data Model: OpenAI 兼容推理服务接口

**Date**: 2026-09-23
**Feature**: specs/001-openai-serving

## Entities

### Model

代表一个已编译的 GEHTP 模型。每个模型在服务中有一个唯一标识，对应一组设备可执行的工件。

**Fields**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识，如 "qwen35-0.8b" |
| `wtop_path` | string | .wtop 文件路径（设备可见） |
| `manifest_path` | string | .wtop.manifest.json 路径（host 可见） |
| `weights_path` | string | 可选，.weights.bin 路径（route B 外置权重） |
| `status` | enum | "available" / "loading" / "unavailable" / "error" |
| `input_length` | int | 编译时最大输入 token 数 |
| `compiled_at` | datetime | 编译时间 |
| `device_slot` | string | 常驻 device runtime 中的 slot ID |

**Validation rules**:
- `id` 必须唯一，命名遵循模型名+版本格式
- `wtop_path` 必须指向存在的 .wtop 文件
- `input_length` 由 manifest 的 `input_elems` 决定

**State transitions**:
```
loading → available (init 成功)
loading → unavailable (init 失败)
available → unavailable (runtime error)
unavailable → loading (重新加载)
```

### Session

代表一个持续的推理会话，包含 KV-cache 元数据和消息历史。

**Fields**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 唯一标识，UUID |
| `model_id` | string | 关联的模型 ID |
| `status` | enum | "active" / "finished" / "error" |
| `messages` | array | 对话消息历史 [{role, content}] |
| `kv_position` | int | 当前 KV-cache 已写入位置（token 数） |
| `kv_handle` | string | device 端 KV buffer 的引用句柄 |
| `created_at` | datetime | 会话创建时间 |
| `last_active` | datetime | 最后活动时间 |
| `max_tokens` | int | 本次会话最大生成 token 数 |

**Validation rules**:
- 一个 session 绑定一个 model，不可切换
- `kv_position` 在 decode 时递增
- `kv_handle` 由常驻 device runtime 分配

**State transitions**:
```
active → finished (达到 max_tokens 或用户 stop)
active → error (device error / timeout)
finished → active (复用 session 继续生成)
error → active (重试)
```

### Request

代表一次推理请求，从 API 接收参数并驱动 device 执行。

**Fields**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `request_id` | string | 唯一标识，UUID |
| `session_id` | string | 关联的 session |
| `messages` | array | 本次请求的消息（仅首条 or 完整历史） |
| `params` | object | {max_tokens, temperature, top_p, stream, stop} |
| `input_ids` | array | 编码后的 token ID 序列 |
| `attention_mask` | array | 注意力掩码 |
| `position_ids` | array | 位置索引 |
| `status` | enum | "queued" / "running" / "completed" / "error" |
| `output_tokens` | array | 生成的 token ID 序列 |
| `output_text` | string | 解码后的文本 |
| `created_at` | datetime | 请求创建时间 |
| `started_at` | datetime | 设备开始执行时间 |
| `completed_at` | datetime | 完成时间 |
| `error` | string | 错误信息（如有） |

**Validation rules**:
- `input_ids` 长度不得超过模型的 `input_length`
- `params.temperature` ∈ [0, 2]
- `params.top_p` ∈ (0, 1]
- 流式请求的 `stream=true`

**State transitions**:
```
queued → running (device 就绪)
running → completed (生成完毕)
running → error (device error)
running → cancelled (客户端断开)
```

### KVCacheHandle

host 端对 device KV-cache 的元数据引用。

**Fields**:
| 字段 | 类型 | 说明 |
|------|------|------|
| `handle_id` | string | 唯一标识 |
| `session_id` | string | 归属 session |
| `model_id` | string | 模型 ID |
| `kv_base_addr` | int | device 端 KV buffer 基地址 |
| `slot_id` | int | device runtime 中的 slot 编号 |
| `current_position` | int | 当前已写入 token 数 |
| `max_position` | int | KV cache 容量上限 |
| `k_shape` | array | K tensor 形状 [layers, heads, seq, head_dim] |
| `v_shape` | array | V tensor 形状 |
| `created_at` | datetime | 创建时间 |

**Validation rules**:
- `current_position` ≤ `max_position`
- 同一 session 不可创建多个 KVCacheHandle

**State transitions**:
```
active → released (session 结束或 GC)
active → full (current_position == max_position)
```

## Relationships

```
Model 1 --- * Session   (一个模型可有多个活跃 session)
Session 1 --- * Request  (一个 session 产生多个请求)
Session 1 --- 1 KVCacheHandle (一个 session 有一个 KV cache 句柄)
Request   * --- * InputSlot (一个请求包含多个输入张量)
```

## Device Runtime Contract

常驻 device runtime 提供以下接口（host 端调用）：

1. `init(model_id, kv_max_position)` → 返回 `device_handle`
2. `run(device_handle, input_ids, position, kv_position)` → 返回 logits 或 token_id
3. `cancel(device_handle)` → 停止当前执行
4. `release(device_handle)` → 释放 KV buffer 和 slot

host 端在以下时机调用：
- Model 注册时：`init()` 分配 slot 和 KV buffer
- 请求开始时：`run()` 执行 prefill/decode
- 请求结束时：`release()` 释放资源（session 复用时可保留）

