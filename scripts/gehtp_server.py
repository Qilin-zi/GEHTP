#!/usr/bin/env python3
"""
GEHTP OpenAI-Compatible Inference Service

FastAPI server providing /v1/chat/completions for internal model testing on V81 device.

Host responsibilities: tokenizer, request sessions, sampling, KV-cache metadata.
Device responsibilities: KV-cache tensor data and model execution via device_runtime.
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.status import HTTP_200_OK, HTTP_202_ACCEPTED

try:
    from .errors import (
        DeviceError,
        ModelError,
        QueueFullError,
        ValidationError,
        install_error_handlers,
    )
    from .logging_config import configure_logging
    from .model_registry import Model, ModelRegistry, ModelStatus
    from .device_runtime import get_runtime, DeviceRuntimeError
    from .persistent_runtime import get_persistent_runtime, PersistentRuntimeError
except ImportError:  # pragma: no cover - supports ``python scripts/gehtp_server.py``
    from errors import (
        DeviceError,
        ModelError,
        QueueFullError,
        ValidationError,
        install_error_handlers,
    )
    from logging_config import configure_logging
    from model_registry import Model, ModelRegistry, ModelStatus
    from device_runtime import get_runtime, DeviceRuntimeError
    from persistent_runtime import get_persistent_runtime, PersistentRuntimeError
    from metrics import MetricsCollector
    from sessions import Session, SessionManager, SessionStatus
    from tokenizer import ChatTokenizer, TokenizerConfig, TokenizerError
    from sampler import Sampler, SamplerError, greedy_sample


logger = configure_logging()


# Global state (in-memory, no database)
model_registry = ModelRegistry()
metrics = MetricsCollector()
session_manager = SessionManager()

# Concurrency limit (research.md R3): host-side queue + limit
MAX_CONCURRENT_REQUESTS = 4
# Waiting requests beyond the concurrency limit; excess gets 429 (contract)
MAX_QUEUED_REQUESTS = 16


class RequestQueue:
    """Asyncio semaphore + bounded wait queue for device concurrency control."""

    def __init__(self, max_concurrent: int, max_queued: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_queued = max_queued
        self._queued = 0

    @property
    def queued(self) -> int:
        return self._queued

    async def acquire(self) -> float:
        """Wait for a device slot; returns queue wait time in ms.

        Raises QueueFullError (HTTP 429) when the wait queue is full.
        """
        if self._queued >= self._max_queued:
            raise QueueFullError()
        self._queued += 1
        metrics.set_queue_length(self._queued)
        enqueued_at = time.monotonic()
        try:
            await self._semaphore.acquire()
        except BaseException:
            self._queued -= 1
            metrics.set_queue_length(self._queued)
            raise
        self._queued -= 1
        metrics.set_queue_length(self._queued)
        return (time.monotonic() - enqueued_at) * 1000.0

    def release(self) -> None:
        self._semaphore.release()


request_queue = RequestQueue(MAX_CONCURRENT_REQUESTS, MAX_QUEUED_REQUESTS)

# Per-model tokenizer cache (tokenizers are immutable after load)
_tokenizers: Dict[str, ChatTokenizer] = {}


def generate_request_id() -> str:
    return f"chatcmpl-gehtp-{uuid.uuid4().hex[:8]}"


def _get_tokenizer(model: Model) -> ChatTokenizer:
    cached = _tokenizers.get(model.id)
    if cached is not None:
        return cached
    if not model.tokenizer_path:
        raise ModelError(
            f"model {model.id} has no tokenizer configured",
            status_code=409,
            code="model_unavailable",
            param="model",
        )
    tokenizer = ChatTokenizer(TokenizerConfig(tokenizer_path=model.tokenizer_path))
    tokenizer.load()  # raises TokenizerError
    _tokenizers[model.id] = tokenizer
    return tokenizer


def _validate_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValidationError(
            "messages must be a non-empty array", param="messages", code="invalid_messages"
        )

    validated = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValidationError(
                f"messages[{index}] must be an object", param="messages", code="invalid_messages"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValidationError(
                f"messages[{index}].role must be system, user, or assistant",
                param="messages",
                code="invalid_messages",
            )
        if not isinstance(content, str):
            raise ValidationError(
                f"messages[{index}].content must be a string",
                param="messages",
                code="invalid_messages",
            )
        validated.append({"role": role, "content": content})
    return validated


def _validate_generation_params(body: Dict[str, Any], model: Model) -> Dict[str, Any]:
    max_tokens = body.get("max_tokens", 16)
    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 1.0)
    stream = body.get("stream", False)

    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise ValidationError(
            "max_tokens must be a positive integer", param="max_tokens", code="invalid_max_tokens"
        )
    if model.input_length and max_tokens > model.input_length:
        raise ValidationError(
            f"max_tokens must not exceed model limit ({model.input_length})",
            param="max_tokens",
            code="max_tokens_exceeded",
        )
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValidationError(
            "temperature must be a number", param="temperature", code="invalid_temperature"
        )
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ValidationError("top_p must be a number", param="top_p", code="invalid_top_p")
    if not isinstance(stream, bool):
        raise ValidationError("stream must be a boolean", param="stream", code="invalid_stream")
    try:
        Sampler(temperature=float(temperature), top_p=float(top_p))
    except SamplerError as exc:
        param = "temperature" if "temperature" in str(exc) else "top_p"
        raise ValidationError(str(exc), param=param, code=f"invalid_{param}") from exc

    stop = body.get("stop")
    if stop is not None and not isinstance(stop, (str, list)):
        raise ValidationError("stop must be a string or array", param="stop", code="invalid_stop")
    if isinstance(stop, list) and not all(isinstance(item, str) for item in stop):
        raise ValidationError(
            "stop array entries must be strings", param="stop", code="invalid_stop"
        )

    return {
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": stream,
        "stop": stop,
    }


async def _read_json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise ValidationError(
            "request body must be valid JSON", code="invalid_json"
        ) from exc
    if not isinstance(body, dict):
        raise ValidationError("request body must be a JSON object", code="invalid_json")
    return body


def _find_stop(generated_text: str, stop_sequences: List[str]) -> int:
    """Return the earliest index where any stop sequence starts, or -1."""
    stop_at = -1
    for seq in stop_sequences:
        if not seq:
            continue
        idx = generated_text.find(seq)
        if idx >= 0 and (stop_at < 0 or idx < stop_at):
            stop_at = idx
    return stop_at


async def _run_generation(
    *,
    request: Request,
    model: Model,
    session: Session,
    input_ids: List[int],
    params: Dict[str, Any],
    tokenizer: ChatTokenizer,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Drive prefill/decode on the device runtime.

    Yields events:
      {"type": "token", "token_id": int, "text": str}
      {"type": "done", "finish_reason": "stop"|"length"|"cancelled"}
      {"type": "error", "message": str}

    The prototype runtime returns a token id (already the device's sampling
    decision); a persistent runtime may return logits, in which case the host
    sampler applies temperature/top_p.
    """
    runtime = get_runtime()
    stop = params["stop"]
    stop_sequences = stop if isinstance(stop, list) else ([stop] if stop else [])
    max_tokens = params["max_tokens"]
    sampler = Sampler(temperature=params["temperature"], top_p=params["top_p"])
    greedy = params["temperature"] == 0

    position = 0
    kv_position = 0
    generated_text = ""
    last_token_id: Optional[int] = None

    for _ in range(max_tokens):
        if await request.is_disconnected():
            try:
                runtime.cancel(model.device_slot or "")
                runtime.reset(model.device_slot or "")
            except DeviceRuntimeError:
                pass
            session.fail()
            yield {"type": "done", "finish_reason": "cancelled"}
            return

        step_input = input_ids if last_token_id is None else [last_token_id]
        try:
            result = await asyncio.to_thread(
                runtime.run,
                model.device_slot or "",
                step_input,
                position,
                kv_position,
            )
        except DeviceRuntimeError as exc:
            session.fail()
            yield {"type": "error", "message": f"device error: {exc}"}
            return
        except Exception as exc:  # noqa: BLE001 - host-side safety net
            session.fail()
            yield {"type": "error", "message": f"runtime error: {exc}"}
            return

        if isinstance(result, (list, tuple)):
            logits = [float(x) for x in result]
            if not logits:
                session.fail()
                yield {"type": "error", "message": "device returned empty logits"}
                return
            token_id = greedy_sample(logits) if greedy else sampler.step(logits)[0]
        else:
            # Token id from the device: already the sampling decision.
            token_id = int(result)

        try:
            token_text = tokenizer.decode([token_id])
        except TokenizerError as exc:
            session.fail()
            yield {"type": "error", "message": f"tokenizer decode failed: {exc}"}
            return

        merged = generated_text + token_text
        stop_at = _find_stop(merged, stop_sequences)
        if stop_at >= 0:
            emitted = merged[:stop_at][len(generated_text):]
            generated_text = merged[:stop_at]
            try:
                session.advance(len(input_ids) if last_token_id is None else 1)
            except ValueError:
                pass
            if emitted:
                yield {"type": "token", "token_id": token_id, "text": emitted}
            yield {"type": "done", "finish_reason": "stop"}
            return

        generated_text = merged
        try:
            session.advance(len(input_ids) if last_token_id is None else 1)
        except ValueError:
            yield {"type": "done", "finish_reason": "length"}
            return
        position += len(input_ids) if last_token_id is None else 1
        kv_position += len(input_ids) if last_token_id is None else 1
        last_token_id = token_id
        yield {"type": "token", "token_id": token_id, "text": token_text}

    yield {"type": "done", "finish_reason": "length"}


def _sse_chunk(request_id: str, model_id: str, delta: Dict[str, Any], finish_reason: Optional[str]) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "model": model_id,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GEHTP server starting", extra={"status": "starting"})
    yield
    # T030: graceful shutdown — cancel in-flight work, release device slots
    logger.info("GEHTP server shutting down", extra={"status": "shutting_down"})
    runtime = get_runtime()
    session_manager.clear()
    for model in model_registry.list_all():
        if model.device_slot:
            try:
                runtime.release(model.device_slot)
            except DeviceRuntimeError:
                pass
            model.device_slot = None
            model.status = ModelStatus.UNAVAILABLE
    logger.info("GEHTP server shutdown complete", extra={"status": "stopped"})


def create_app() -> FastAPI:
    app = FastAPI(
        title="GEHTP OpenAI-Compatible Inference Service",
        version="0.1.0",
        description="Internal testing interface for comparing compiled models on V81 device",
        docs_url="/docs",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)

    return app


app = create_app()


def _load_model_task(model_id: str) -> None:
    """Background task: manifest parse + device init + tokenizer preload."""
    try:
        model = model_registry.load(model_id, get_runtime())
    except ModelError as exc:
        logger.warning(
            f"model load failed: {exc.message}",
            extra={"model": model_id, "status": "error"},
        )
        return
    except Exception as exc:  # noqa: BLE001
        model_registry.set_error(model_id, f"unexpected load failure: {exc}")
        logger.warning(
            f"model load failed: {exc}",
            extra={"model": model_id, "status": "error"},
        )
        return

    if model.tokenizer_path:
        try:
            _get_tokenizer(model)
        except TokenizerError as exc:
            model_registry.set_error(model_id, f"tokenizer load failed: {exc}")
            logger.warning(
                f"tokenizer load failed: {exc}",
                extra={"model": model_id, "status": "error"},
            )
            return

    logger.info(
        f"model {model_id} available",
        extra={"model": model_id, "status": "available"},
    )


@app.get("/v1/models", response_model=Dict[str, Any])
async def list_models() -> Dict[str, Any]:
    models = model_registry.list_all()
    data = [model.to_dict() for model in models]
    return {"object": "list", "data": data}


@app.post("/v1/models/register")
async def register_model(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    body = await _read_json_body(request)
    model_id = body.get("id")
    wtop_path = body.get("wtop_path")
    manifest_path = body.get("manifest_path")
    weights_path = body.get("weights_path")
    tokenizer_path = body.get("tokenizer_path")

    if not model_id or not isinstance(model_id, str):
        raise ValidationError("model id required", param="id", code="model_id_required")
    if not wtop_path or not isinstance(wtop_path, str):
        raise ValidationError("wtop_path is required", param="wtop_path", code="wtop_path_required")
    if not manifest_path or not isinstance(manifest_path, str):
        raise ValidationError(
            "manifest_path is required", param="manifest_path", code="manifest_path_required"
        )

    model = model_registry.register(
        model_id,
        wtop_path,
        manifest_path,
        weights_path=weights_path,
        tokenizer_path=tokenizer_path,
    )
    background_tasks.add_task(_load_model_task, model.id)
    logger.info(
        f"model {model_id} registered",
        extra={"model": model_id, "status": "loading"},
    )
    return JSONResponse(content=model.to_dict(), status_code=HTTP_202_ACCEPTED)


@app.post("/v1/models/{model_id}/unload")
async def unload_model(model_id: str) -> JSONResponse:
    model = model_registry.unload(model_id, get_runtime())
    _tokenizers.pop(model_id, None)
    logger.info(
        f"model {model_id} unloaded",
        extra={"model": model_id, "status": "unavailable"},
    )
    return JSONResponse(content=model.to_dict(), status_code=HTTP_200_OK)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await _read_json_body(request)
    request_id = generate_request_id()

    model_id = body.get("model")
    if not model_id or not isinstance(model_id, str):
        raise ValidationError("model is required", param="model", code="model_required")

    model = model_registry.get(model_id)  # 404 ModelError

    if model.status == ModelStatus.LOADING:
        raise ModelError(
            f"model is loading: {model_id}",
            status_code=409,
            code="model_loading",
            param="model",
        )
    if model.status != ModelStatus.AVAILABLE:
        raise ModelError(
            f"model is unavailable: {model.status.value}",
            status_code=409,
            code="model_unavailable",
            param="model",
        )

    validated_messages = _validate_messages(body.get("messages"))
    params = _validate_generation_params(body, model)

    try:
        tokenizer = _get_tokenizer(model)
    except TokenizerError as exc:
        raise ModelError(
            f"tokenizer load failed: {exc}", status_code=500, code="tokenizer_error"
        ) from exc

    prompt_text = tokenizer.apply_chat_template(validated_messages)
    try:
        input_ids = tokenizer.encode(prompt_text)
    except TokenizerError as exc:
        raise ModelError(
            f"tokenizer encode failed: {exc}", status_code=500, code="tokenizer_error"
        ) from exc
    prompt_tokens = len(input_ids)
    if model.input_length and prompt_tokens > model.input_length:
        raise ValidationError(
            f"prompt has {prompt_tokens} tokens, exceeds model input_length ({model.input_length})",
            param="messages",
            code="prompt_too_long",
        )
    if prompt_tokens + params["max_tokens"] > model.kv_max:
        raise ValidationError(
            f"prompt_tokens + max_tokens exceeds KV budget ({model.kv_max})",
            param="max_tokens",
            code="kv_budget_exceeded",
        )

    logger.info(
        "request queued",
        extra={"request_id": request_id, "model": model_id, "status": "queued"},
    )
    queue_time_ms = await request_queue.acquire()  # 429 QueueFullError
    started_at = time.monotonic()
    logger.info(
        "request running",
        extra={
            "request_id": request_id,
            "model": model_id,
            "status": "running",
            "latency_ms": round(queue_time_ms, 3),
        },
    )

    session = session_manager.create(
        model_id,
        max_tokens=params["max_tokens"],
        messages=validated_messages,
        kv_max_position=model.kv_max,
        slot_id=model.slot_id,
    )
    metrics.set_active_sessions(session_manager.active_count())

    first_token_latency_ms: Optional[float] = None
    completion_tokens = 0
    generated_parts: List[str] = []
    finish_reason = "stop"

    if params["stream"]:
        async def generate() -> AsyncGenerator[str, None]:
            nonlocal first_token_latency_ms, completion_tokens, finish_reason
            completed = False
            errored = False
            try:
                yield _sse_chunk(request_id, model_id, {"role": "assistant"}, None)
                async for event in _run_generation(
                    request=request,
                    model=model,
                    session=session,
                    input_ids=input_ids,
                    params=params,
                    tokenizer=tokenizer,
                ):
                    if event["type"] == "token":
                        if first_token_latency_ms is None:
                            first_token_latency_ms = (time.monotonic() - started_at) * 1000.0
                        completion_tokens += 1
                        generated_parts.append(event["text"])
                        yield _sse_chunk(request_id, model_id, {"content": event["text"]}, None)
                    elif event["type"] == "done":
                        finish_reason = event["finish_reason"]
                        if finish_reason == "cancelled":
                            logger.info(
                                "request cancelled (client disconnected)",
                                extra={
                                    "request_id": request_id,
                                    "model": model_id,
                                    "status": "cancelled",
                                },
                            )
                            return
                        completed = True
                        yield _sse_chunk(request_id, model_id, {}, finish_reason)
                        yield "data: [DONE]\n\n"
                    elif event["type"] == "error":
                        errored = True
                        logger.warning(
                            f"request error: {event['message']}",
                            extra={
                                "request_id": request_id,
                                "model": model_id,
                                "status": "error",
                            },
                        )
                        metrics.record_request(
                            model_id,
                            success=False,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            queue_time_ms=queue_time_ms,
                        )
                        yield _sse_chunk(request_id, model_id, {}, "error")
                        yield "data: [DONE]\n\n"
                        return
                generation_latency_ms = (time.monotonic() - started_at) * 1000.0
                metrics.record_request(
                    model_id,
                    success=True,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    queue_time_ms=queue_time_ms,
                    first_token_latency_ms=first_token_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                )
                session_manager.finish(session.session_id)
                logger.info(
                    "request completed",
                    extra={
                        "request_id": request_id,
                        "model": model_id,
                        "tokens": prompt_tokens + completion_tokens,
                        "latency_ms": round(generation_latency_ms, 3),
                        "status": "completed",
                    },
                )
            finally:
                if not completed and not errored:
                    # Client disconnected mid-stream (or generator closed):
                    # cancel the device op and recycle the KV-cache (T024).
                    # reset() clears the flag afterwards because prototype
                    # runs are synchronous — otherwise the pending cancel
                    # would poison the next request on this model slot.
                    try:
                        get_runtime().cancel(model.device_slot or "")
                        get_runtime().reset(model.device_slot or "")
                    except DeviceRuntimeError:
                        pass
                    if session.status == SessionStatus.ACTIVE:
                        session.fail()
                session_manager.release(session.session_id)
                request_queue.release()
                metrics.set_active_sessions(session_manager.active_count())

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"X-Request-ID": request_id},
        )

    # Non-streaming path (T016)
    try:
        async for event in _run_generation(
            request=request,
            model=model,
            session=session,
            input_ids=input_ids,
            params=params,
            tokenizer=tokenizer,
        ):
            if event["type"] == "token":
                if first_token_latency_ms is None:
                    first_token_latency_ms = (time.monotonic() - started_at) * 1000.0
                completion_tokens += 1
                generated_parts.append(event["text"])
            elif event["type"] == "done":
                finish_reason = event["finish_reason"]
            elif event["type"] == "error":
                metrics.record_request(
                    model_id,
                    success=False,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    queue_time_ms=queue_time_ms,
                )
                logger.warning(
                    f"request error: {event['message']}",
                    extra={"request_id": request_id, "model": model_id, "status": "error"},
                )
                raise DeviceError(event["message"])
    except Exception:
        session_manager.release(session.session_id)
        request_queue.release()
        metrics.set_active_sessions(session_manager.active_count())
        raise

    if finish_reason == "cancelled":
        session_manager.release(session.session_id)
        request_queue.release()
        metrics.set_active_sessions(session_manager.active_count())
        raise DeviceError("client disconnected", code="client_disconnected")

    generation_latency_ms = (time.monotonic() - started_at) * 1000.0
    generated_text = "".join(generated_parts)
    metrics.record_request(
        model_id,
        success=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        queue_time_ms=queue_time_ms,
        first_token_latency_ms=first_token_latency_ms,
        generation_latency_ms=generation_latency_ms,
    )
    session_manager.finish(session.session_id)
    session_manager.release(session.session_id)
    request_queue.release()
    metrics.set_active_sessions(session_manager.active_count())
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "model": model_id,
            "tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(generation_latency_ms, 3),
            "status": "completed",
        },
    )

    return JSONResponse(
        content={
            "id": request_id,
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        status_code=HTTP_200_OK,
        headers={"X-Request-ID": request_id},
    )


def _empty_model_metrics() -> Dict[str, Any]:
    return {
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "success_rate": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "queue_latency_ms": {"avg": 0.0, "p50": 0.0, "p95": 0.0},
        "first_token_latency_ms": {"avg": 0.0, "p50": 0.0, "p95": 0.0},
        "generation_latency_ms": {"avg": 0.0, "p50": 0.0, "p95": 0.0},
    }


@app.get("/metrics")
async def metrics_endpoint() -> Dict[str, Any]:
    metrics.set_active_sessions(session_manager.active_count())
    metrics.set_queue_length(request_queue.queued)
    snapshot = metrics.snapshot(device_runtime=get_runtime().health())
    models_snapshot = snapshot["models"]
    sessions = session_manager.list_sessions()
    for model in model_registry.list_all():
        entry = models_snapshot.setdefault(model.id, _empty_model_metrics())
        entry["status"] = model.status.value
        entry["kv_tokens_in_use"] = sum(
            s.kv_handle.current_position
            for s in sessions
            if s.model_id == model.id and s.kv_handle is not None
        )
    return snapshot


@app.get("/health")
async def health_endpoint() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device_runtime": get_runtime().health(),
        "active_sessions": session_manager.active_count(),
        "queue_length": request_queue.queued,
    }


def main():
    parser = argparse.ArgumentParser(description="GEHTP OpenAI-Compatible Inference Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    configure_logging(getattr(logging, args.log_level.upper()))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
