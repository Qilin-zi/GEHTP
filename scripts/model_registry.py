"""Model registry and entity management for the OpenAI-compatible service.

The Model entity tracks compiled GEHTP model artifacts, lifecycle status,
and manifest-derived metadata (e.g. input_length).

Lifecycle (per data-model.md):

    loading → available      (device init succeeded)
    loading → unavailable    (device init failed)
    loading → error          (manifest parse failed)
    available → unavailable  (unload / runtime error)
    unavailable → loading    (reload)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .device_runtime import DeviceHandle, DeviceRuntime, DeviceRuntimeError
    from .persistent_runtime import get_persistent_runtime
except ImportError:  # pragma: no cover - supports ``python scripts/gehtp_server.py``
    from device_runtime import DeviceHandle, DeviceRuntime, DeviceRuntimeError
    from persistent_runtime import get_persistent_runtime


try:
    from .errors import ModelError
except ImportError:
    from errors import ModelError


DEFAULT_KV_MAX = 4096


class ModelStatus(str, Enum):
    LOADING = "loading"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class Model:
    """Represents a compiled GEHTP model."""

    id: str
    wtop_path: str
    manifest_path: str
    status: ModelStatus = ModelStatus.LOADING
    weights_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    input_length: int = 0
    compiled_at: Optional[datetime] = None
    device_slot: Optional[str] = None
    slot_id: int = 0
    kv_max: int = DEFAULT_KV_MAX
    error: Optional[str] = None

    @property
    def manifest_path_obj(self) -> Path:
        return Path(self.manifest_path)

    @property
    def wtop_path_obj(self) -> Path:
        return Path(self.wtop_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "created": int(self.compiled_at.timestamp()) if self.compiled_at else 0,
            "owned_by": "gehtp",
            "status": self.status.value,
            "input_length": self.input_length,
            "device_slot": self.device_slot,
            "slot_id": self.slot_id,
            "error": self.error,
        }


class ModelRegistry:
    """Thread-safe registry for loaded models."""

    def __init__(self) -> None:
        self._models: Dict[str, Model] = {}
        self._lock = threading.RLock()
        self._next_slot_id = 0

    def register(
        self,
        model_id: str,
        wtop_path: str,
        manifest_path: str,
        *,
        weights_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> Model:
        """Create a model entry in ``loading`` state.

        Validation of request fields happens here so the API can fail fast;
        the heavy load (manifest parse + device init) is done by :meth:`load`.
        """
        if not model_id:
            raise ModelError(
                "model id required",
                status_code=400,
                code="model_id_required",
                param="id",
            )

        wtop = Path(wtop_path) if wtop_path else None
        manifest = Path(manifest_path) if manifest_path else None
        if not wtop or not wtop.is_file():
            raise ModelError(
                f"wtop file not found: {wtop_path}",
                status_code=400,
                code="wtop_not_found",
                param="wtop_path",
            )
        if not manifest or not manifest.is_file():
            raise ModelError(
                f"manifest file not found: {manifest_path}",
                status_code=400,
                code="manifest_not_found",
                param="manifest_path",
            )

        with self._lock:
            if model_id in self._models:
                raise ModelError(
                    f"model {model_id} already registered",
                    status_code=409,
                    code="model_exists",
                )
            model = Model(
                id=model_id,
                wtop_path=wtop_path,
                manifest_path=manifest_path,
                weights_path=weights_path,
                tokenizer_path=tokenizer_path,
                compiled_at=datetime.now(timezone.utc),
                slot_id=self._next_slot_id,
            )
            self._next_slot_id += 1
            self._models[model_id] = model
        return model

    def load(
        self,
        model_id: str,
        device_runtime: Optional[DeviceRuntime] = None,
        *,
        kv_max: int = DEFAULT_KV_MAX,
    ) -> Model:
        """Parse the manifest and initialize the device slot.

        Transitions ``loading``/``unavailable``/``error`` → ``available``.
        On failure the model stays registered with status ``unavailable``
        (device init failed) or ``error`` (manifest invalid) so the state is
        visible via ``GET /v1/models``.
        """
        model = self.get(model_id)
        with self._lock:
            if model.status == ModelStatus.AVAILABLE:
                raise ModelError(
                    f"model {model_id} is already loaded",
                    status_code=409,
                    code="model_already_loaded",
                )
            model.status = ModelStatus.LOADING
            model.error = None

        try:
            model.input_length = self._parse_manifest_input_length(model.manifest_path)
        except Exception as exc:
            with self._lock:
                model.status = ModelStatus.ERROR
                model.error = f"failed to parse manifest: {exc}"
            raise ModelError(
                model.error,
                status_code=400,
                code="invalid_manifest",
                param="manifest_path",
            ) from exc

        kv_budget = max(kv_max, model.input_length + 1024)

        if device_runtime is None:
            device_runtime = get_persistent_runtime()

        try:
            handle = device_runtime.init(
                model_id,
                kv_max=kv_budget,
                wtop_path=model.wtop_path,
                manifest_path=model.manifest_path,
            )
        except DeviceRuntimeError as exc:
            with self._lock:
                model.status = ModelStatus.UNAVAILABLE
                model.error = f"device init failed: {exc}"
            raise ModelError(
                model.error,
                status_code=503,
                code="device_init_failed",
            ) from exc
        with self._lock:
            model.device_slot = handle.handle_id
            model.kv_max = handle.kv_max_position

        with self._lock:
            model.status = ModelStatus.AVAILABLE
        return model

    def unload(
        self,
        model_id: str,
        device_runtime: Optional[DeviceRuntime] = None,
    ) -> Model:
        """Release the device slot and transition to ``unavailable``."""
        model = self.get(model_id)
        with self._lock:
            slot = model.device_slot
            model.device_slot = None
            model.status = ModelStatus.UNAVAILABLE
        if slot and device_runtime is not None:
            try:
                device_runtime.release(slot)
            except DeviceRuntimeError:
                pass
        return model

    def get(self, model_id: str) -> Model:
        with self._lock:
            try:
                return self._models[model_id]
            except KeyError as exc:
                raise ModelError(f"model not found: {model_id}", code="model_not_found") from exc

    def list_all(self) -> List[Model]:
        with self._lock:
            return list(self._models.values())

    def set_status(self, model_id: str, status: ModelStatus) -> None:
        with self._lock:
            model = self.get(model_id)
            model.status = status

    def set_error(self, model_id: str, message: str) -> None:
        with self._lock:
            model = self.get(model_id)
            model.status = ModelStatus.ERROR
            model.error = message

    def clear(self) -> None:
        with self._lock:
            self._models.clear()

    @staticmethod
    def _parse_manifest_input_length(manifest_path: str) -> int:
        """Parse input_length from the manifest JSON file."""
        with open(manifest_path) as f:
            manifest = json.load(f)
        input_elems = manifest.get("input_elems")
        if input_elems is None:
            raise ModelError(
                "manifest missing 'input_elems' field",
                status_code=400,
                code="invalid_manifest",
                param="manifest_path",
            )
        return int(input_elems)
