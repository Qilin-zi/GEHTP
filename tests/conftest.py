"""Shared fixtures for the OpenAI-compatible serving tests.

Creates a fake compiled model (dummy .wtop + manifest + a tiny WordLevel
tokenizer.json) and resets all server global state between tests.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

import gehtp_server  # noqa: E402


VOCAB = {
    "<unk>": 0,
    "user:": 1,
    "system:": 2,
    "assistant:": 3,
    "hello": 4,
    "world": 5,
    "hi": 6,
    "there": 7,
    "test": 8,
    "model": 9,
}


def _write_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path))


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """A fake compiled model directory with wtop, manifest, tokenizer."""
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "model.wtop").write_bytes(b"WTOP-DUMMY")
    (directory / "model.wtop.manifest.json").write_text(json.dumps({"input_elems": 128}))
    _write_tokenizer(directory / "tokenizer.json")
    return directory


@pytest.fixture()
def client():
    """TestClient with all global server state reset around each test."""
    gehtp_server.model_registry.clear()
    gehtp_server.session_manager.clear()
    gehtp_server.metrics.reset()
    gehtp_server._tokenizers.clear()
    gehtp_server.get_runtime().shutdown()
    with TestClient(gehtp_server.app) as test_client:
        yield test_client
    gehtp_server.model_registry.clear()
    gehtp_server.session_manager.clear()
    gehtp_server.metrics.reset()
    gehtp_server._tokenizers.clear()
    gehtp_server.get_runtime().shutdown()


@pytest.fixture()
def registered_model(client: TestClient, model_dir: Path) -> str:
    """Register a model; background load completes before the client returns."""
    model_id = "test-model"
    response = client.post(
        "/v1/models/register",
        json={
            "id": model_id,
            "wtop_path": str(model_dir / "model.wtop"),
            "manifest_path": str(model_dir / "model.wtop.manifest.json"),
            "tokenizer_path": str(model_dir / "tokenizer.json"),
        },
    )
    assert response.status_code == 202, response.text
    return model_id


def parse_sse(body: str):
    """Parse an SSE body into (events, done) where events are decoded JSON chunks."""
    events = []
    done = False
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), f"bad SSE block: {block!r}"
        data = block[len("data: "):]
        if data == "[DONE]":
            done = True
            continue
        events.append(json.loads(data))
    return events, done
