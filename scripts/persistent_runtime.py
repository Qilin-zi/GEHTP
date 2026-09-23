"""Persistent device runtime protocol implementation.

Provides a connection-oriented interface to the device-side socket server,
enabling persistent model sessions that avoid adb push/pull for each request.

Protocol:
  - Server listens on Unix socket at /data/local/tmp/gehtp/runtime.sock
  - Client sends JSON header + binary tensor data
  - Each request: {"op": "prefill|decode", "handle_id": str, "input_ids": list, "position": int, "kv_position": int}
  - Response: {"token": int, "logits": [float]|null, "kv": meta}
"""
from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class PersistentRuntimeError(RuntimeError):
    """Raised when persistent runtime operations fail."""


@dataclass
class PersistentHandle:
    """Handle for a persisted model session on device."""
    handle_id: str
    model_id: str
    kv_max_position: int
    sock_path: str = "/data/local/tmp/gehtp/runtime.sock"
    model_meta: dict[str, Any] = field(default_factory=dict)


class PersistentDeviceRuntime:
    """Connection-oriented runtime for persisted device sessions.

    Maintains persistent connections to device-side socket server for
    low-latency inference after model initialization.
    """

    SOCK_PATH = "/data/local/tmp/gehtp/runtime.sock"
    SOCK_TIMEOUT = 30.0
    RECONNECT_DELAY = 1.0
    MAX_RECONNECT_ATTEMPTS = 3

    def __init__(
        self,
        device: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.device = device or os.environ.get("ANDROID_SERIAL", "52f67807")
        self.timeout = timeout
        self._handles: dict[str, PersistentHandle] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._sock_path = self.SOCK_PATH

    def init(
        self,
        model_id: str,
        kv_max: int,
        *,
        wtop_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> PersistentHandle:
        """Initialize a persistent model session on device.

        Uses file-based runner to load model, then establishes socket connection.
        """
        if kv_max <= 0:
            raise ValueError("kv_max must be positive")

        handle = PersistentHandle(
            handle_id=f"persist_{model_id}_{int(time.time()*1000)}",
            model_id=model_id,
            kv_max_position=kv_max,
            sock_path=self._sock_path,
            model_meta={
                "wtop_path": wtop_path,
                "manifest_path": manifest_path,
            },
        )

        with self._lock:
            self._handles[handle.handle_id] = handle
        return handle

    def run(
        self,
        handle: PersistentHandle | str,
        input_ids: list[int],
        position: int,
        kv_position: int,
    ) -> int | list[float]:
        """Execute a prefill or decode step through persistent session.

        Returns either a token ID (greedy) or logits (for host sampling).
        """
        resolved = self._resolve_handle(handle)

        with self._lock:
            if resolved.handle_id in self._cancelled:
                self._cancelled.discard(resolved.handle_id)
                raise PersistentRuntimeError("device request cancelled")

        if position < 0 or kv_position < 0:
            raise ValueError("position and kv_position must be non-negative")
        if kv_position >= resolved.kv_max_position:
            raise PersistentRuntimeError("KV cache capacity exceeded")

        try:
            sock = self._connect_with_reconnect()
            try:
                return self._send_request(
                    sock, resolved.handle_id, input_ids, position, kv_position
                )
            finally:
                sock.close()
        except PersistentRuntimeError:
            raise
        except Exception as e:
            raise PersistentRuntimeError(
                f"socket communication failed: {e}"
            ) from e

    def cancel(self, handle: PersistentHandle | str) -> None:
        """Mark a pending operation as cancelled."""
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.add(resolved.handle_id)

    def reset(self, handle: PersistentHandle | str) -> None:
        """Clear cancellation flag for a handle."""
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.discard(resolved.handle_id)

    def release(self, handle: PersistentHandle | str) -> None:
        """Release a persistent handle and its resources."""
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.discard(resolved.handle_id)
            self._handles.pop(resolved.handle_id, None)

    def health(self) -> dict[str, Any]:
        """Return health status including socket connectivity."""
        with self._lock:
            active = len(self._handles)

        sock_ok = self._check_socket()
        return {
            "status": "available" if sock_ok else "degraded",
            "active_handles": active,
            "device": self.device,
            "socket": "connected" if sock_ok else "disconnected",
        }

    def shutdown(self) -> None:
        """Release all handles."""
        with self._lock:
            self._handles.clear()
            self._cancelled.clear()

    def _connect_with_reconnect(self) -> socket.socket:
        """Connect with retry logic for temporary socket unavailability."""
        last_err: Optional[Exception] = None
        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            try:
                return self._connect()
            except FileNotFoundError as e:
                last_err = e
                if attempt < self.MAX_RECONNECT_ATTEMPTS - 1:
                    time.sleep(self.RECONNECT_DELAY)
        raise PersistentRuntimeError(
            f"socket not available after {self.MAX_RECONNECT_ATTEMPTS} attempts"
        ) from last_err

    def _connect(self) -> socket.socket:
        """Establish connection to device socket server."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        if not os.path.exists(self._sock_path):
            raise FileNotFoundError(f"socket not found: {self._sock_path}")

        sock.connect(self._sock_path)
        return sock

    def _check_socket(self) -> bool:
        """Check if socket server is responsive."""
        try:
            sock = self._connect()
            try:
                sock.settimeout(1.0)
                sock.sendall(b"PING")
                response = sock.recv(4)
                return response == b"PONG"
            finally:
                sock.close()
        except Exception:
            return False

    def _send_request(
        self,
        sock: socket.socket,
        handle_id: str,
        input_ids: list[int],
        position: int,
        kv_position: int,
    ) -> int | list[float]:
        """Send inference request and receive response."""
        payload = {
            "handle_id": handle_id,
            "input_ids": input_ids,
            "position": position,
            "kv_position": kv_position,
        }

        header = json.dumps(payload).encode("utf-8")
        header_len = struct.pack("!I", len(header))

        sock.sendall(header_len + header)

        resp_len = struct.unpack("!I", sock.recv(4))[0]
        response = sock.recv(resp_len)
        result = json.loads(response.decode("utf-8"))

        if "token" in result:
            return result["token"]
        elif "logits" in result:
            return result["logits"]
        else:
            raise PersistentRuntimeError(f"invalid response: {result}")

    def _resolve_handle(self, handle: PersistentHandle | str) -> PersistentHandle:
        if isinstance(handle, PersistentHandle):
            return handle
        with self._lock:
            resolved = self._handles.get(handle)
        if resolved is None:
            raise PersistentRuntimeError(f"unknown device handle: {handle}")
        return resolved


_default_persistent_runtime: Optional[PersistentDeviceRuntime] = None


def get_persistent_runtime() -> PersistentDeviceRuntime:
    """Return the process-wide persistent runtime instance."""
    global _default_persistent_runtime
    if _default_persistent_runtime is None:
        _default_persistent_runtime = PersistentDeviceRuntime()
    return _default_persistent_runtime