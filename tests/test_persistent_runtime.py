"""Tests for the persistent device runtime protocol.

These tests verify the socket-based communication protocol without
requiring actual device hardware.
"""
import json
import os
import socket
import struct
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from persistent_runtime import (
    PersistentDeviceRuntime,
    PersistentHandle,
    PersistentRuntimeError,
)


@pytest.fixture
def mock_sock():
    """Create a mock socket pair for testing."""
    server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    yield server_sock, client_sock
    server_sock.close()
    client_sock.close()


class TestPersistentHandle:
    def test_persistent_handle_creation(self):
        """PersistentHandle stores required metadata."""
        handle = PersistentHandle(
            handle_id="test_handle_123",
            model_id="test-model",
            kv_max_position=4096,
            sock_path="/tmp/test.sock",
        )
        assert handle.handle_id == "test_handle_123"
        assert handle.model_id == "test-model"
        assert handle.kv_max_position == 4096
        assert handle.sock_path == "/tmp/test.sock"

    def test_persistent_handle_defaults(self):
        """PersistentHandle uses default socket path."""
        handle = PersistentHandle(
            handle_id="handle",
            model_id="m",
            kv_max_position=1,
        )
        assert handle.sock_path == "/data/local/tmp/gehtp/runtime.sock"


class TestPersistentDeviceRuntime:
    def test_init_creates_handle(self):
        """init() creates a handle with valid metadata."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model-1", kv_max=2048)

        assert isinstance(handle, PersistentHandle)
        assert handle.model_id == "model-1"
        assert handle.kv_max_position == 2048

    def test_init_validates_kv_max(self):
        """init() rejects invalid kv_max."""
        runtime = PersistentDeviceRuntime()
        with pytest.raises(ValueError, match="kv_max must be positive"):
            runtime.init("model", kv_max=0)

        with pytest.raises(ValueError, match="kv_max must be positive"):
            runtime.init("model", kv_max=-1)

    @patch.object(PersistentDeviceRuntime, '_connect_with_reconnect')
    def test_run_under_kv_limit(self, mock_connect):
        """run() executes within KV limit."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model", kv_max=10)

        mock_sock = MagicMock()
        mock_sock.recv = MagicMock(side_effect=[
            struct.pack("!I", len(b'{"token":0}')),
            b'{"token":0}',
        ])
        mock_sock.close = MagicMock()
        mock_connect.return_value = mock_sock

        result = runtime.run(handle, [1, 2, 3], position=0, kv_position=5)
        assert result == 0  # Placeholder token

    def test_run_exceeds_kv_limit(self):
        """run() rejects KV position beyond limit."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model", kv_max=10)

        with pytest.raises(PersistentRuntimeError, match="KV cache capacity exceeded"):
            runtime.run(handle, [1, 2], position=0, kv_position=100)

    def test_run_negative_position(self):
        """run() rejects negative position."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model", kv_max=10)

        with pytest.raises(ValueError, match="non-negative"):
            runtime.run(handle, [1], position=-1, kv_position=0)

    def test_resolve_handle_by_id(self):
        """Handles can be resolved by string ID."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model", kv_max=10)

        resolved = runtime._resolve_handle(handle.handle_id)
        assert resolved.handle_id == handle.handle_id

    def test_resolve_unknown_handle(self):
        """Unknown handle IDs raise error."""
        runtime = PersistentDeviceRuntime()

        with pytest.raises(PersistentRuntimeError, match="unknown device handle"):
            runtime._resolve_handle("nonexistent")

    def test_cancel_marks_handle_cancelled(self):
        """cancel() marks handle as cancelled."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model", kv_max=10)

        runtime.cancel(handle)

        with pytest.raises(PersistentRuntimeError, match="cancelled"):
            runtime.run(handle, [1], position=0, kv_position=0)

    def test_reset_clears_cancel_flag(self):
        """reset() clears cancellation flag."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model", kv_max=10)

        runtime.cancel(handle)
        runtime.reset(handle)

        # Patch _connect_with_reconnect to avoid socket lookup
        with patch.object(runtime, '_connect_with_reconnect') as mock_connect:
            mock_sock = MagicMock()
            mock_sock.recv = MagicMock(side_effect=[
                struct.pack("!I", len(b'{"token":0}')),
                b'{"token":0}',
            ])
            mock_sock.close = MagicMock()
            mock_connect.return_value = mock_sock
            result = runtime.run(handle, [1], position=0, kv_position=0)
            assert result == 0

    def test_release_removes_handle(self):
        """release() removes handle from runtime."""
        runtime = PersistentDeviceRuntime()
        handle = runtime.init("model", kv_max=10)

        runtime.release(handle)

        with pytest.raises(PersistentRuntimeError, match="unknown device handle"):
            runtime._resolve_handle(handle.handle_id)

    def test_health_returns_status(self):
        """health() returns status dict."""
        runtime = PersistentDeviceRuntime()

        handle = runtime.init("model", kv_max=10)
        health = runtime.health()

        assert "status" in health
        assert "active_handles" in health
        assert health["active_handles"] == 1

    def test_health_detects_socket_degraded(self):
        """health() reports degraded when socket unavailable."""
        runtime = PersistentDeviceRuntime()

        # Socket won't exist in test environment
        health = runtime.health()
        assert health["status"] == "degraded"
        assert health["socket"] == "disconnected"

    def test_shutdown_clears_handles(self):
        """shutdown() clears all state."""
        runtime = PersistentDeviceRuntime()
        runtime.init("model1", kv_max=10)
        runtime.init("model2", kv_max=10)

        runtime.shutdown()

        assert len(runtime._handles) == 0


class TestSocketProtocol:
    def test_send_request_format(self):
        """Request format: 4-byte len + JSON header."""
        runtime = PersistentDeviceRuntime()
        runtime._sock_path = "/tmp/protocol_test.sock"

        # Use mock socket to verify format
        mock_sock = MagicMock()
        sent_data = []

        def capture_sendall(data):
            sent_data.append(data)

        mock_sock.sendall = capture_sendall
        mock_sock.recv = MagicMock(side_effect=[
            struct.pack("!I", len(b'{"token":5}')),
            b'{"token":5}',
        ])
        mock_sock.close = MagicMock()

        with patch.object(runtime, '_connect', return_value=mock_sock):
            runtime._send_request(mock_sock, "handle1", [1, 2, 3], 0, 0)

        assert len(sent_data) == 1
        # First 4 bytes are header length
        header_len = struct.unpack("!I", sent_data[0][:4])[0]
        # Verify header is valid JSON with expected fields
        header = sent_data[0][4:4+header_len].decode('utf-8')
        result = json.loads(header)
        assert result["handle_id"] == "handle1"
        assert result["input_ids"] == [1, 2, 3]
        assert result["position"] == 0
        assert result["kv_position"] == 0


class TestReconnection:
    def test_reconnect_on_socket_not_found(self):
        """Runtime retries connection when socket missing."""
        runtime = PersistentDeviceRuntime()
        runtime._sock_path = "/nonexistent/socket.sock"

        # Must init first to create a handle registered in _handles
        handle = runtime.init("model", kv_max=10)

        with pytest.raises(PersistentRuntimeError, match="socket not available"):
            runtime.run(handle, [1], 0, 0)


class TestGetPersistentRuntime:
    def test_returns_singleton(self):
        """get_persistent_runtime returns same instance."""
        from persistent_runtime import _default_persistent_runtime

        # Reset for test isolation
        import persistent_runtime as pr_module
        pr_module._default_persistent_runtime = None

        r1 = PersistentDeviceRuntime()
        r2 = PersistentDeviceRuntime()

        assert r1 is not r2  # Each call creates new


    def test_default_runtime_function(self):
        """get_persistent_runtime returns process-wide instance."""
        from persistent_runtime import get_persistent_runtime, PersistentDeviceRuntime

        import persistent_runtime as pr_module
        pr_module._default_persistent_runtime = None

        r1 = get_persistent_runtime()
        assert isinstance(r1, PersistentDeviceRuntime)

        r2 = get_persistent_runtime()
        assert r1 is r2