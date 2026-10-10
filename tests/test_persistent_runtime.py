"""Tests for the persistent device runtime (file-queue transport, T019/T020).

协议不再走 socket: host 写 req 文件 (4B BE len + JSON + 二进制体) → DSP
常驻 runner 应答 resp 文件。传输缝 = ``_rpc``, adb 全部 mock, 不需要设备。
"""
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from device_runtime import DeviceRuntimeError, get_runtime as get_default_runtime
from persistent_runtime import (
    IN_BYTES,
    MAX_POS,
    PersistentDeviceRuntime,
    PersistentHandle,
    PersistentRuntimeError,
    get_persistent_runtime,
    make_face,
    make_step_input,
)


@pytest.fixture()
def runtime():
    """PersistentDeviceRuntime with the device-server boot disabled."""
    r = PersistentDeviceRuntime()
    with patch.object(PersistentDeviceRuntime, "_ensure_device_server"):
        yield r


class TestStepInput:
    def test_step_input_length_and_token(self):
        body = make_step_input(1234, 5)
        assert len(body) == IN_BYTES == 388
        assert struct.unpack("<i", body[:4])[0] == 1234

    def test_face_position_zero(self):
        face = np.frombuffer(make_face(0), dtype=np.int32)
        h = face.view(np.uint16)
        cos = h[0:64].view(np.float16)
        sin = h[64:128].view(np.float16)
        mask = h[128:192].view(np.float16)
        assert np.all(cos == np.float16(1.0))
        assert np.all(sin == np.float16(0.0))
        assert mask[0] == np.float16(0.0)
        assert np.all(mask[1:] == np.float16(-np.inf))

    def test_face_mask_causal(self):
        pos = 7
        mask = np.frombuffer(make_face(pos), dtype=np.int32).view(np.uint16)[128:192].view(np.float16)
        assert np.all(mask[: pos + 1] == np.float16(0.0))
        assert np.all(mask[pos + 1 :] == np.float16(-np.inf))

    def test_face_cos_matches_formula(self):
        from persistent_runtime import _INV

        pos = 13
        h = np.frombuffer(make_face(pos), dtype=np.int32).view(np.uint16)
        cos = h[0:64].view(np.float16)
        sin = h[64:128].view(np.float16)
        expect_c = np.concatenate([np.cos(pos * _INV)] * 2).astype(np.float16)
        expect_s = np.concatenate([np.sin(pos * _INV)] * 2).astype(np.float16)
        assert np.array_equal(cos, expect_c)
        assert np.array_equal(sin, expect_s)

    def test_face_position_bounds(self):
        with pytest.raises(ValueError):
            make_face(-1)
        with pytest.raises(ValueError):
            make_face(MAX_POS)

    def test_face_deterministic(self):
        assert make_face(9) == make_face(9)


class TestErrorHierarchy:
    def test_persistent_error_is_device_error(self):
        assert issubclass(PersistentRuntimeError, DeviceRuntimeError)


class TestHandle:
    def test_handle_fields(self):
        h = PersistentHandle(
            handle_id="h1", model_id="m", kv_max_position=64,
            model_meta={"wtop_path": "/x"},
        )
        assert h.handle_id == "h1"
        assert h.kv_max_position == 64
        assert h.model_meta["wtop_path"] == "/x"


class TestInit:
    def test_init_creates_handle_without_device(self, runtime):
        handle = runtime.init("model-1", kv_max=2048)
        assert isinstance(handle, PersistentHandle)
        assert handle.model_id == "model-1"
        assert handle.kv_max_position == 2048

    def test_init_validates_kv_max(self, runtime):
        with pytest.raises(ValueError, match="kv_max must be positive"):
            runtime.init("model", kv_max=0)

    def test_init_boots_device_server(self):
        r = PersistentDeviceRuntime()
        with patch.object(r, "_ensure_device_server") as boot:
            r.init("model", kv_max=10)
        boot.assert_called_once()


class TestRun:
    def test_run_returns_device_token(self, runtime):
        handle = runtime.init("model", kv_max=100)
        with patch.object(
            runtime, "_rpc", return_value={"rc": 0, "token": 42}
        ) as rpc:
            assert runtime.run(handle, [7], position=3, kv_position=3) == 42
        hdr = rpc.call_args.args[0]
        assert hdr["op"] == "run"
        assert hdr["position"] == 3
        body = rpc.call_args.kwargs["body"]
        assert body == make_step_input(7, 3)

    def test_run_takes_first_token_only(self, runtime):
        handle = runtime.init("model", kv_max=100)
        with patch.object(runtime, "_rpc", return_value={"rc": 0, "token": 1}) as rpc:
            runtime.run(handle, [9, 8, 7], position=0, kv_position=0)
        assert rpc.call_args.kwargs["body"] == make_step_input(9, 0)

    def test_run_exceeds_kv_limit(self, runtime):
        handle = runtime.init("model", kv_max=10)
        with pytest.raises(PersistentRuntimeError, match="KV cache capacity exceeded"):
            runtime.run(handle, [1], position=0, kv_position=100)

    def test_run_exceeds_max_pos(self, runtime):
        handle = runtime.init("model", kv_max=4096)
        with pytest.raises(PersistentRuntimeError, match="MAX_POS"):
            runtime.run(handle, [1], position=MAX_POS, kv_position=0)

    def test_run_negative_position(self, runtime):
        handle = runtime.init("model", kv_max=10)
        with pytest.raises(ValueError, match="non-negative"):
            runtime.run(handle, [1], position=-1, kv_position=0)

    def test_run_empty_input(self, runtime):
        handle = runtime.init("model", kv_max=10)
        with pytest.raises(ValueError, match="non-empty"):
            runtime.run(handle, [], position=0, kv_position=0)

    def test_run_rpc_error_propagates(self, runtime):
        handle = runtime.init("model", kv_max=10)
        with patch.object(runtime, "_rpc", side_effect=PersistentRuntimeError("boom")):
            with pytest.raises(PersistentRuntimeError, match="boom"):
                runtime.run(handle, [1], position=0, kv_position=0)


class TestLifecycle:
    def test_cancel_marks_handle_cancelled(self, runtime):
        handle = runtime.init("model", kv_max=10)
        runtime.cancel(handle)
        with pytest.raises(PersistentRuntimeError, match="cancelled"):
            runtime.run(handle, [1], position=0, kv_position=0)

    def test_reset_clears_cancel_and_rpcs(self, runtime):
        handle = runtime.init("model", kv_max=10)
        runtime.cancel(handle)
        with patch.object(runtime, "_rpc", return_value={"rc": 0}) as rpc:
            runtime.reset(handle)
        assert rpc.call_args.args[0]["op"] == "reset"
        with patch.object(runtime, "_rpc", return_value={"rc": 0, "token": 1}):
            assert runtime.run(handle, [1], position=0, kv_position=0) == 1

    def test_release_removes_handle(self, runtime):
        handle = runtime.init("model", kv_max=10)
        runtime.release(handle)
        with pytest.raises(PersistentRuntimeError, match="unknown device handle"):
            runtime._resolve_handle(handle.handle_id)

    def test_resolve_unknown_handle(self, runtime):
        with pytest.raises(PersistentRuntimeError, match="unknown device handle"):
            runtime._resolve_handle("nonexistent")

    def test_resolve_by_id(self, runtime):
        handle = runtime.init("model", kv_max=10)
        assert runtime._resolve_handle(handle.handle_id) is handle

    def test_shutdown_stops_server(self):
        r = PersistentDeviceRuntime()
        with patch.object(r, "_adb") as adb, patch.object(r, "_rpc") as rpc:
            # pgrep 恒空 → 优雅路径: 只发 shutdown op, 不 kill
            adb.return_value = MagicMock(stdout="\n", returncode=0)
            r.shutdown()
        assert rpc.call_args.args[0]["op"] == "shutdown"
        assert not any("kill" in str(c) for c in adb.call_args_list)

    def test_shutdown_kills_survivor(self):
        r = PersistentDeviceRuntime()
        with patch.object(r, "_rpc"), patch.object(r, "_adb") as adb, \
             patch("persistent_runtime.time") as fake_time:
            # pgrep 恒见 pid → 等待循环到时退出 → post-loop pgrep 仍见 → kill
            adb.return_value = MagicMock(stdout="123\n", returncode=0)
            fake_time.monotonic.side_effect = [0.0, 5.0, 10.0]
            r.shutdown()
        assert any("kill 123" in str(c) for c in adb.call_args_list)


class TestHealth:
    def test_health_degraded_when_no_server(self, runtime):
        with patch.object(runtime, "server_running", return_value=False):
            health = runtime.health()
        assert health["status"] == "degraded"
        assert health["transport"] == "file-queue"

    def test_health_available_when_pong(self, runtime):
        runtime.init("model", kv_max=10)
        with patch.object(runtime, "server_running", return_value=True), \
             patch.object(runtime, "_rpc", return_value={"rc": 0, "pong": 1}):
            health = runtime.health()
        assert health["status"] == "available"
        assert health["active_handles"] == 1


class TestGetRuntimeSelection:
    def test_env_selects_persistent(self, monkeypatch):
        import device_runtime
        import persistent_runtime

        monkeypatch.setenv("GEHTP_PERSISTENT", "1")
        monkeypatch.setattr(device_runtime, "_default_runtime", None)
        monkeypatch.setattr(persistent_runtime, "_default_persistent_runtime", None)
        r = get_default_runtime()
        assert isinstance(r, PersistentDeviceRuntime)
        assert device_runtime._default_runtime is r

    def test_env_default_is_prototype(self, monkeypatch):
        import device_runtime
        from device_runtime import DeviceRuntime

        monkeypatch.delenv("GEHTP_PERSISTENT", raising=False)
        monkeypatch.setattr(device_runtime, "_default_runtime", None)
        r = get_default_runtime()
        assert type(r) is DeviceRuntime

    def test_persistent_runtime_singleton(self, monkeypatch):
        import persistent_runtime

        monkeypatch.setattr(persistent_runtime, "_default_persistent_runtime", None)
        r1 = get_persistent_runtime()
        assert get_persistent_runtime() is r1
