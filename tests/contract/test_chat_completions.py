"""Contract tests for POST /v1/chat/completions per contracts/openai-api.md.

Validates request rules, response schemas (stream + non-stream), SSE event
format, and the structured error format with HTTP status codes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import gehtp_server  # noqa: E402
from model_registry import ModelStatus  # noqa: E402
from tests.conftest import parse_sse  # noqa: E402


def _chat_payload(model_id: str, **overrides):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hello world"}],
    }
    payload.update(overrides)
    return payload


# --- Non-stream response format (contract: Non-stream response 200) ---


def test_non_stream_response_schema(client, registered_model):
    response = client.post("/v1/chat/completions", json=_chat_payload(registered_model))
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["id"].startswith("chatcmpl-gehtp-")
    assert body["object"] == "chat.completion"
    assert isinstance(body["created"], int)
    assert body["model"] == registered_model

    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")

    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_finish_reason_length_when_max_tokens_reached(client, registered_model):
    response = client.post(
        "/v1/chat/completions", json=_chat_payload(registered_model, max_tokens=3)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3


def test_stop_sequence_truncates_output(client, registered_model):
    # The placeholder device runtime always emits token id 0 ("<unk>").
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(registered_model, max_tokens=8, stop=["<unk>"]),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "<unk>" not in body["choices"][0]["message"]["content"]


def test_greedy_temperature_zero(client, registered_model):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(registered_model, max_tokens=2, temperature=0),
    )
    assert response.status_code == 200, response.text
    assert response.json()["usage"]["completion_tokens"] == 2


# --- Stream response format (contract: Stream response 200) ---


def test_stream_response_sse_format(client, registered_model):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(registered_model, max_tokens=3, stream=True),
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")

    events, done = parse_sse(response.text)
    assert done, "stream must terminate with data: [DONE]"
    assert len(events) >= 3  # role chunk + content chunk(s) + finish chunk

    first = events[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["id"].startswith("chatcmpl-gehtp-")
    assert first["model"] == registered_model
    assert first["choices"][0]["delta"] == {"role": "assistant"}
    assert first["choices"][0]["finish_reason"] is None

    content_chunks = events[1:-1]
    assert content_chunks, "expected at least one content chunk"
    for chunk in content_chunks:
        assert chunk["object"] == "chat.completion.chunk"
        assert isinstance(chunk["choices"][0]["delta"].get("content"), str)
        assert chunk["choices"][0]["finish_reason"] is None

    final = events[-1]
    assert final["choices"][0]["delta"] == {}
    assert final["choices"][0]["finish_reason"] in ("stop", "length")


def test_stream_per_token_chunks(client, registered_model):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(registered_model, max_tokens=4, stream=True),
    )
    assert response.status_code == 200, response.text
    events, done = parse_sse(response.text)
    assert done
    content_chunks = events[1:-1]
    # one SSE event per generated token (R6)
    assert len(content_chunks) == 4


# --- Error format and HTTP status codes (contract: Error format) ---


def _assert_error(body, *, param=None, code=None, error_type="invalid_request_error"):
    assert "error" in body
    error = body["error"]
    assert isinstance(error["message"], str) and error["message"]
    assert error["type"] == error_type
    if param is not None:
        assert error["param"] == param
    if code is not None:
        assert error["code"] == code


def test_missing_model_field(client, registered_model):
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 400
    _assert_error(response.json(), param="model")


def test_unknown_model_returns_404(client):
    response = client.post("/v1/chat/completions", json=_chat_payload("no-such-model"))
    assert response.status_code == 404
    _assert_error(response.json(), param="model", code="model_not_found")


def test_loading_model_returns_409(client, registered_model):
    gehtp_server.model_registry.set_status(registered_model, ModelStatus.LOADING)
    response = client.post("/v1/chat/completions", json=_chat_payload(registered_model))
    assert response.status_code == 409
    _assert_error(response.json(), param="model", code="model_loading")


def test_unavailable_model_returns_409(client, registered_model):
    gehtp_server.model_registry.set_status(registered_model, ModelStatus.UNAVAILABLE)
    response = client.post("/v1/chat/completions", json=_chat_payload(registered_model))
    assert response.status_code == 409
    _assert_error(response.json(), param="model", code="model_unavailable")


def test_invalid_json_body(client, registered_model):
    response = client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    _assert_error(response.json(), code="invalid_json")


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user"}],
        [{"role": "bad", "content": "hi"}],
        [{"role": "user", "content": 42}],
        ["not-a-dict"],
    ],
)
def test_invalid_messages(client, registered_model, messages):
    response = client.post(
        "/v1/chat/completions", json=_chat_payload(registered_model, messages=messages)
    )
    assert response.status_code == 400
    _assert_error(response.json(), param="messages")


@pytest.mark.parametrize("max_tokens", [0, -1, "16", True])
def test_invalid_max_tokens(client, registered_model, max_tokens):
    response = client.post(
        "/v1/chat/completions", json=_chat_payload(registered_model, max_tokens=max_tokens)
    )
    assert response.status_code == 400
    _assert_error(response.json(), param="max_tokens")


def test_max_tokens_above_model_limit(client, registered_model):
    # Fixture manifest sets input_elems = 128
    response = client.post(
        "/v1/chat/completions", json=_chat_payload(registered_model, max_tokens=129)
    )
    assert response.status_code == 400
    _assert_error(response.json(), param="max_tokens", code="max_tokens_exceeded")


@pytest.mark.parametrize("temperature", [-0.1, 2.1, "hot"])
def test_invalid_temperature(client, registered_model, temperature):
    response = client.post(
        "/v1/chat/completions", json=_chat_payload(registered_model, temperature=temperature)
    )
    assert response.status_code == 400
    _assert_error(response.json(), param="temperature")


@pytest.mark.parametrize("top_p", [0, 1.5, "wide"])
def test_invalid_top_p(client, registered_model, top_p):
    response = client.post(
        "/v1/chat/completions", json=_chat_payload(registered_model, top_p=top_p)
    )
    assert response.status_code == 400
    _assert_error(response.json(), param="top_p")


@pytest.mark.parametrize("stop", [123, ["ok", 5]])
def test_invalid_stop(client, registered_model, stop):
    response = client.post("/v1/chat/completions", json=_chat_payload(registered_model, stop=stop))
    assert response.status_code == 400
    _assert_error(response.json(), param="stop")


def test_prompt_exceeding_input_length(client, registered_model):
    long_prompt = " ".join(["hello"] * 200)
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(
            registered_model,
            messages=[{"role": "user", "content": long_prompt}],
        ),
    )
    assert response.status_code == 400
    _assert_error(response.json(), param="messages", code="prompt_too_long")


def test_queue_full_returns_429(client, registered_model):
    gehtp_server.request_queue._max_queued = 0
    try:
        response = client.post("/v1/chat/completions", json=_chat_payload(registered_model))
    finally:
        gehtp_server.request_queue._max_queued = gehtp_server.MAX_QUEUED_REQUESTS
    assert response.status_code == 429
    _assert_error(response.json(), code="queue_full", error_type="rate_limit_error")
