"""Device runtime abstraction for GEHTP inference.

The prototype backend invokes the existing ``scripts/gehtp`` file runner. A
persistent device implementation can be supplied through the same interface.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional


class DeviceRuntimeError(RuntimeError):
    """Raised when the device runner cannot execute a request."""


@dataclass
class DeviceHandle:
    handle_id: str
    model_id: str
    kv_max_position: int
    wtop_path: Optional[str] = None
    manifest_path: Optional[str] = None


class DeviceRuntime:
    """Small, serializable interface around the GEHTP device runner."""

    def __init__(
        self,
        runner_path: Optional[str] = None,
        device: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.runner_path = runner_path or str(Path(__file__).with_name("gehtp"))
        self.device = device or os.environ.get("ANDROID_SERIAL", "52f67807")
        self.timeout = timeout
        self._handles: dict[str, DeviceHandle] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()

    def init(
        self,
        model_id: str,
        kv_max: int,
        *,
        wtop_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> DeviceHandle:
        """Allocate host metadata for a model/device slot."""
        if kv_max <= 0:
            raise ValueError("kv_max must be positive")
        if wtop_path and not Path(wtop_path).is_file():
            raise DeviceRuntimeError(f"wtop file not found: {wtop_path}")
        if manifest_path and not Path(manifest_path).is_file():
            raise DeviceRuntimeError(f"manifest file not found: {manifest_path}")

        handle = DeviceHandle(
            handle_id=str(uuid.uuid4()),
            model_id=model_id,
            kv_max_position=kv_max,
            wtop_path=wtop_path,
            manifest_path=manifest_path,
        )
        with self._lock:
            self._handles[handle.handle_id] = handle
        return handle

    def run(
        self,
        handle: DeviceHandle | str,
        input_ids: Iterable[int],
        position: int,
        kv_position: int,
    ) -> Any:
        """Run a prefill/decode step through the configured backend.

        The prototype file runner expects raw float input and produces raw
        output, so callers may provide a persistent backend by subclassing
        this class. The default implementation returns a deterministic token
        placeholder and keeps the public contract usable in host-only tests.
        """
        resolved = self._resolve_handle(handle)
        if position < 0 or kv_position < 0:
            raise ValueError("position and kv_position must be non-negative")
        if kv_position >= resolved.kv_max_position:
            raise DeviceRuntimeError("KV cache capacity exceeded")
        with self._lock:
            if resolved.handle_id in self._cancelled:
                self._cancelled.remove(resolved.handle_id)
                raise DeviceRuntimeError("device request cancelled")
        # A persistent runner can override this method. Returning a token ID
        # makes the prototype backend useful without claiming to execute a
        # model when no persistent runtime is available.
        return 0

    def cancel(self, handle: DeviceHandle | str) -> None:
        """Cancel the active operation associated with a handle."""
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.add(resolved.handle_id)

    def reset(self, handle: DeviceHandle | str) -> None:
        """Clear a pending cancellation for a handle.

        Prototype runs are synchronous, so a cancellation flag has no
        in-flight operation to abort; without a reset it would poison the
        next ``run()`` on the same (model-level) slot. A persistent runtime
        should cancel the in-flight op first and reset once it has aborted.
        """
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.discard(resolved.handle_id)

    def release(self, handle: DeviceHandle | str) -> None:
        """Release host metadata and any associated device resources."""
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.discard(resolved.handle_id)
            self._handles.pop(resolved.handle_id, None)

    def health(self) -> dict[str, Any]:
        """Return basic runtime health information."""
        with self._lock:
            active = len(self._handles)
        return {"status": "available", "active_handles": active, "device": self.device}

    def shutdown(self) -> None:
        """Release all handles (server shutdown and test teardown)."""
        with self._lock:
            self._handles.clear()
            self._cancelled.clear()

    def _resolve_handle(self, handle: DeviceHandle | str) -> DeviceHandle:
        if isinstance(handle, DeviceHandle):
            resolved = handle
        else:
            with self._lock:
                resolved = self._handles.get(handle)
            if resolved is None:
                raise DeviceRuntimeError(f"unknown device handle: {handle}")
        with self._lock:
            if resolved.handle_id not in self._handles:
                raise DeviceRuntimeError(f"released device handle: {resolved.handle_id}")
        return resolved


_default_runtime: Optional[DeviceRuntime] = None


def get_runtime() -> DeviceRuntime:
    """Return the process-wide runtime instance."""
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = DeviceRuntime()
    return _default_runtime
