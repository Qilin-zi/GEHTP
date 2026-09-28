"""Integration test for User Story 1: single request end-to-end.

Flow: register model → loading→available via device_runtime init →
GET /v1/models → POST /v1/chat/completions (stream + non-stream) →
GET /metrics reflects the requests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from tests.conftest import parse_sse  # noqa: E402


def test_register_list_chat_metrics_flow(client, model_dir):
    # 1. Register: returns the model id with status "loading" (contract).
    response = client.post(
        "/v1/models/register",
        json={
            "id": "integration-model",
            "wtop_path": str(model_dir / "model.wtop"),
            "manifest_path": str(model_dir / "model.wtop.manifest.json"),
            "tokenizer_path": str(model_dir / "tokenizer.json"),
        },
    )
    assert response.status_code == 202, response.text
    registered = response.json()
    assert registered["id"] == "integration-model"
    assert registered["status"] == "loading"

    # Background load (manifest parse + device init) completes after the
    # response; the model becomes available with manifest-derived metadata.
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    entry = next(m for m in models["data"] if m["id"] == "integration-model")
    assert entry["status"] == "available"
    assert entry["input_length"] == 128
    assert entry["device_slot"]

    # 2. Non-stream request through device_runtime.
    payload = {
        "model": "integration-model",
        "messages": [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "hello world"},
        ],
        "max_tokens": 4,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "length"
    content_non_stream = body["choices"][0]["message"]["content"]
    assert content_non_stream
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] == 4

    # 3. Streaming request: SSE deltas reassemble to the same content
    # (the placeholder device runtime is deterministic).
    response = client.post("/v1/chat/completions", json={**payload, "stream": True})
    assert response.status_code == 200, response.text
    events, done = parse_sse(response.text)
    assert done
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    streamed = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in events[1:-1]
    )
    assert streamed == content_non_stream
    assert events[-1]["choices"][0]["finish_reason"] == "length"

    # 4. Metrics reflect both requests.
    metrics = client.get("/metrics").json()
    model_metrics = metrics["models"]["integration-model"]
    assert model_metrics["requests"] == 2
    assert model_metrics["successes"] == 2
    assert model_metrics["failures"] == 0
    assert model_metrics["prompt_tokens"] == 2 * body["usage"]["prompt_tokens"]
    assert model_metrics["completion_tokens"] == 8
    assert model_metrics["status"] == "available"
    assert metrics["queue_length"] == 0
    assert metrics["active_sessions"] == 0  # sessions released after completion
    assert metrics["device_runtime"]["status"] == "available"


def test_register_validation_and_conflict(client, model_dir):
    # Missing required fields → 400
    response = client.post("/v1/models/register", json={"id": "x"})
    assert response.status_code == 400

    # Nonexistent wtop → 400
    response = client.post(
        "/v1/models/register",
        json={
            "id": "missing-wtop",
            "wtop_path": str(model_dir / "nope.wtop"),
            "manifest_path": str(model_dir / "model.wtop.manifest.json"),
        },
    )
    assert response.status_code == 400

    # Duplicate registration → 409
    payload = {
        "id": "dup-model",
        "wtop_path": str(model_dir / "model.wtop"),
        "manifest_path": str(model_dir / "model.wtop.manifest.json"),
    }
    assert client.post("/v1/models/register", json=payload).status_code == 202
    response = client.post("/v1/models/register", json=payload)
    assert response.status_code == 409


def test_unload_transitions_to_unavailable(client, model_dir):
    payload = {
        "id": "unload-model",
        "wtop_path": str(model_dir / "model.wtop"),
        "manifest_path": str(model_dir / "model.wtop.manifest.json"),
        "tokenizer_path": str(model_dir / "tokenizer.json"),
    }
    assert client.post("/v1/models/register", json=payload).status_code == 202

    response = client.post("/v1/models/unload-model/unload")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "unavailable"
    assert response.json()["device_slot"] is None

    # Chat requests against an unloaded model are rejected with 409.
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "unload-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 409


def test_multi_model_registration(client, model_dir, tmp_path):
    # Second fake model with its own artifacts.
    other = tmp_path / "model2"
    other.mkdir()
    (other / "model.wtop").write_bytes(b"WTOP-DUMMY-2")
    (other / "model.wtop.manifest.json").write_text('{"input_elems": 64}')

    for model_id, directory in (("model-a", model_dir), ("model-b", other)):
        response = client.post(
            "/v1/models/register",
            json={
                "id": model_id,
                "wtop_path": str(directory / "model.wtop"),
                "manifest_path": str(directory / "model.wtop.manifest.json"),
            },
        )
        assert response.status_code == 202, response.text

    models = client.get("/v1/models").json()["data"]
    by_id = {m["id"]: m for m in models}
    assert by_id["model-a"]["status"] == "available"
    assert by_id["model-b"]["status"] == "available"
    assert by_id["model-a"]["input_length"] == 128
    assert by_id["model-b"]["input_length"] == 64
    # Each model gets its own device slot.
    assert by_id["model-a"]["device_slot"] != by_id["model-b"]["device_slot"]
    assert by_id["model-a"]["slot_id"] != by_id["model-b"]["slot_id"]
