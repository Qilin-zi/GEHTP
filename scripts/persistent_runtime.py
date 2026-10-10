"""Persistent device runtime protocol implementation (004 T019).

架构修正 (2026-10-09): 原设计假设设备侧 Unix socket server, 但 qurt (DSP)
无 socket 支持, 且 run_main_on_hexagon 生命周期 = 一次 FastRPC 调用内
main() 存活。持久化语义落在 DSP 侧常驻进程的文件请求循环上
(``kernels/examples/42_gehtp_persistent/main.c``):

  - start_server(): adb 起 DSP 常驻进程 (blob 解析 + KV/SSM 状态一次装载)
  - 每请求: host 写 req 文件 (4B BE hdr_len + JSON + 输入体) → runner
    读全后 unlink 认领 → 执行 → 写 resp 文件 (4B BE len + JSON) → host
    读后 unlink。协议帧与原 socket 设计同形 (FR-007), 传输层换文件。

decode 图请求体 = 388B 合并输入 [ids i32 ×1 | face i32 ×96], face 由
position 在 host 侧构造 (RoPE cos/sin + causal mask, 与
surgery_decode_graph.py 的 u16_face_layout 同口径; f64→f16 位级与
T018 host 驱动一致)。
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# decode 图几何 (surgery_decode_graph.py / model config 同源)
VOCAB = 248320
MAX_POS = 64          # surgery --max-pos
ROT_DIM = 64          # partial_rotary 0.25 × head_dim 256
FACE_N32 = 96         # (cos 64 + sin 64 + mask 64) u16 = 96 i32
IN_BYTES = 4 + 4 * FACE_N32   # ids 4B + face 384B = 388B (EXT_IN 槽 len)
# RoPE 频率表: 1/1e7^(2j/64), j=0..31 (Qwen3.5 0.8B)
_INV = 1.0 / (1e7 ** (np.arange(0, 64, 2, dtype=np.float64) / 64))


def make_face(position: int) -> bytes:
    """构造 decode 图融合输入面 (384B): [cos|cos | sin|sin | mask] u16 → i32。"""
    if not 0 <= position < MAX_POS:
        raise ValueError(f"position {position} out of [0,{MAX_POS})")
    h = np.zeros(192, dtype=np.uint16)
    c = np.cos(position * _INV)
    s = np.sin(position * _INV)
    h[0:64] = np.concatenate([c, c]).astype(np.float16).view(np.uint16)
    h[64:128] = np.concatenate([s, s]).astype(np.float16).view(np.uint16)
    m = np.zeros(MAX_POS)
    m[position + 1:] = -np.inf
    h[128:192] = m.astype(np.float16).view(np.uint16)
    return h.view(np.int32).tobytes()


def make_step_input(token_id: int, position: int) -> bytes:
    """单步 decode 请求体: ids i32[1] + face i32[96] = 388B。"""
    ids = np.array([token_id], dtype=np.int32).tobytes()
    return ids + make_face(position)


class PersistentRuntimeError(RuntimeError):
    """Raised when persistent runtime operations fail."""


@dataclass
class PersistentHandle:
    """Handle for a persisted model session on device."""
    handle_id: str
    model_id: str
    kv_max_position: int
    model_meta: dict[str, Any] = field(default_factory=dict)


class PersistentDeviceRuntime:
    """File-queue transport to the DSP-resident persistent runner.

    Public contract mirrors the original socket design (init/run/cancel/
    reset/release/health) so the serving layer stays transport-agnostic.
    """

    PERSIST_DIR = "/data/local/tmp/hvxhmx23/gehtp_persist"
    DEVROOT = "/data/local/tmp/hvxhmx23"
    POLL_INTERVAL = 0.05
    STARTUP_TIMEOUT = 300.0     # 2GB blob 读入 + parse 需要数十秒

    def __init__(
        self,
        device: Optional[str] = None,
        timeout: float = 180.0,
    ) -> None:
        self.device = device or os.environ.get("ANDROID_SERIAL", "52f67807")
        self.timeout = timeout
        self._handles: dict[str, PersistentHandle] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.RLock()
        self._server_pid: Optional[int] = None
        self._job: dict[str, Any] = {}

    # ---- adb helpers ----
    def _adb(self, *args: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        cmd = ["adb", "-s", self.device, *args]
        return subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout or self.timeout,
        )

    def _adb_ok(self, *args: str, timeout: Optional[float] = None) -> str:
        r = self._adb(*args, timeout=timeout)
        if r.returncode != 0:
            raise PersistentRuntimeError(
                f"adb {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}"
            )
        return r.stdout

    # ---- server lifecycle ----
    def server_running(self) -> bool:
        # [g] 方括号防 pgrep -f 自匹配; setsid 后 $! 不可靠, liveness 恒以 pgrep 实测
        r = self._adb("shell", "pgrep -f '[g]ehtp_persistent.so' || true")
        return bool(r.stdout.split())

    def start_server(
        self,
        blob: str,
        out_temp: int,
        in_bytes: int = IN_BYTES,
        *,
        weights_size: Optional[int] = None,
        idle_exit_s: int = 1800,
        so_name: str = "gehtp_persistent.so",
    ) -> dict[str, Any]:
        """启动 DSP 常驻 runner。blob 为设备侧 .wtop 绝对路径 (须已 push)。"""
        if self.server_running():
            # 已有服务器 (含同配置复用场景): ping 通即接管
            try:
                self._rpc({"op": "ping"}, timeout=30.0)
                self._job = {"blob": blob, "out_temp": out_temp, "in_bytes": in_bytes}
                return {"pid": None, "reused": True, **self._job}
            except PersistentRuntimeError:
                raise PersistentRuntimeError(
                    "persistent server already running but not responding")
        p = self.PERSIST_DIR
        r = self._adb("shell", f"wc -c < {blob} 2>/dev/null")
        size = r.stdout.strip()
        if not size.isdigit() or int(size) == 0:
            raise PersistentRuntimeError(f"blob not on device: {blob} (size={size!r})")
        self._adb_ok("shell", f"mkdir -p {p}")
        r = self._adb("shell", f"ls {self.DEVROOT}/{so_name} 2>/dev/null")
        if not r.stdout.strip():
            raise PersistentRuntimeError(
                f"{self.DEVROOT}/{so_name} missing — run `gehtp setup` (persistent target)")

        lines = [
            f"blob {blob}",
            f"out_temp {out_temp}",
            f"in_bytes {in_bytes}",
            f"req {p}/req.bin",
            f"resp {p}/resp.bin",
            f"out {p}/out.f16.raw",
            f"idle_exit_s {idle_exit_s}",
        ]
        if weights_size:
            lines.append(f"weights_size {weights_size}")
        with tempfile.NamedTemporaryFile("w", suffix=".job.txt", delete=False) as f:
            f.write("\n".join(lines) + "\n")
            job_local = f.name
        try:
            self._adb_ok("push", job_local, f"{p}/job.txt", timeout=60.0)
        finally:
            os.unlink(job_local)

        launch = (
            f"cd {self.DEVROOT} && ADSP_LIBRARY_PATH={self.DEVROOT} "
            f"CDSP_LIBRARY_PATH={self.DEVROOT} setsid nohup ./run_main_on_hexagon 3 "
            f"{so_name} > {p}/server.log 2>&1 < /dev/null & echo $!"
        )
        # setsid 已分离会话+全流重定向, 但 adbd 仍可能不立即返回 — 超时视为
        # 非致命, 以 pgrep 实际进程为准 ($! 经 setsid 后不可靠, 弃用)
        try:
            self._adb("shell", launch, timeout=20.0)
        except subprocess.TimeoutExpired:
            pass
        r = self._adb("shell", "pgrep -f '[g]ehtp_persistent.so' || true")
        pids = r.stdout.split()
        if not pids:
            raise PersistentRuntimeError("server launch failed (pgrep empty)")
        self._server_pid = int(pids[-1])
        self._job = {"blob": blob, "out_temp": out_temp, "in_bytes": in_bytes}

        # 等 ping 就绪 (blob 2GB 读入 + wt_parse)
        t0 = time.monotonic()
        last_err: Optional[Exception] = None
        while time.monotonic() - t0 < self.STARTUP_TIMEOUT:
            if not self.server_running():
                log = self._adb("shell", f"tail -20 {p}/server.log").stdout
                raise PersistentRuntimeError(f"server died during startup:\n{log}")
            try:
                self._rpc({"op": "ping"}, timeout=30.0)
                return {"pid": self._server_pid, **self._job}
            except PersistentRuntimeError as e:
                last_err = e
                time.sleep(1.0)
        raise PersistentRuntimeError(f"server not ready in {self.STARTUP_TIMEOUT}s: {last_err}")

    def stop_server(self) -> None:
        # 恒先发 shutdown op: kill AP 进程会留 DSP 侧幽灵会话 (仍应答文件请求,
        # 数十秒后被 reaper 回收; 期间新实例启动报 -2147482611, 回收时刻撞上
        # 请求即表现为"run 崩溃")。优雅退出由 runner 自身做 wt_exec_shutdown。
        try:
            self._rpc({"op": "shutdown"}, timeout=30.0)
        except PersistentRuntimeError:
            pass
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            r = self._adb("shell", "pgrep -f '[g]ehtp_persistent.so' || true")
            if not r.stdout.split():
                self._server_pid = None
                return
            time.sleep(0.5)
        r = self._adb("shell", "pgrep -f '[g]ehtp_persistent.so' || true")
        pids = r.stdout.split()
        if pids:
            self._adb("shell", "kill " + " ".join(pids))
            # kill 后 DSP 会话回收需要数秒; 不等会污染下一次启动
            time.sleep(5.0)
        self._server_pid = None

    # ---- file-queue RPC ----
    def _rpc(self, hdr: dict[str, Any], body: bytes = b"", timeout: Optional[float] = None) -> dict[str, Any]:
        p = self.PERSIST_DIR
        timeout = timeout or self.timeout
        self._adb_ok("shell", f"rm -f {p}/resp.bin")
        j = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
        payload = struct.pack("!I", len(j)) + j + body
        with tempfile.NamedTemporaryFile("wb", suffix=".req", delete=False) as f:
            f.write(payload)
            req_local = f.name
        try:
            self._adb_ok("push", req_local, f"{p}/req.bin", timeout=timeout)
        finally:
            os.unlink(req_local)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self._adb("shell", f"test -s {p}/resp.bin && echo Y || echo N")
            if r.stdout.strip() == "Y":
                with tempfile.NamedTemporaryFile(suffix=".resp", delete=False) as f:
                    resp_local = f.name
                try:
                    self._adb_ok("pull", f"{p}/resp.bin", resp_local, timeout=timeout)
                    self._adb_ok("shell", f"rm -f {p}/resp.bin")
                    with open(resp_local, "rb") as g:
                        data = g.read()
                finally:
                    os.unlink(resp_local)
                if len(data) < 4:
                    raise PersistentRuntimeError(f"short resp ({len(data)}B)")
                n = struct.unpack("!I", data[:4])[0]
                if len(data) < 4 + n:
                    raise PersistentRuntimeError(f"truncated resp ({len(data)}B < {4+n})")
                out = json.loads(data[4:4 + n].decode("utf-8"))
                if out.get("rc", 0) != 0:
                    raise PersistentRuntimeError(
                        f"device rc={out.get('rc')}: {out.get('err', 'unknown')}")
                return out
            time.sleep(self.POLL_INTERVAL)
        raise PersistentRuntimeError(f"rpc timeout after {timeout}s (op={hdr.get('op')})")

    # ---- public contract (原 socket 设计同名接口) ----
    def init(
        self,
        model_id: str,
        kv_max: int,
        *,
        wtop_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
    ) -> PersistentHandle:
        if kv_max <= 0:
            raise ValueError("kv_max must be positive")
        handle = PersistentHandle(
            handle_id=f"persist_{model_id}_{int(time.time()*1000)}",
            model_id=model_id,
            kv_max_position=kv_max,
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
        *,
        want_out: bool = False,
    ) -> int:
        """单步 decode。input_ids=[tok] (多 id 仅取首元素 — decode 图单 token)。

        返回设备侧 argmax token (贪心决策); want_out=True 时设备另写
        PERSIST_DIR/out.f16.raw (f16 logits), 由验证/取证侧 pull。
        """
        resolved = self._resolve_handle(handle)
        with self._lock:
            if resolved.handle_id in self._cancelled:
                self._cancelled.discard(resolved.handle_id)
                raise PersistentRuntimeError("device request cancelled")
        if position < 0 or kv_position < 0:
            raise ValueError("position and kv_position must be non-negative")
        if position >= MAX_POS:
            raise PersistentRuntimeError(f"position exceeds decode graph MAX_POS={MAX_POS}")
        if kv_position >= resolved.kv_max_position:
            raise PersistentRuntimeError("KV cache capacity exceeded")
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        body = make_step_input(int(input_ids[0]), position)
        out = self._rpc(
            {
                "op": "run",
                "handle_id": resolved.handle_id,
                "position": position,
                "kv_position": kv_position,
                "want_out": 1 if want_out else 0,
            },
            body=body,
        )
        return int(out["token"])

    def pull_out(self, local_path: str) -> str:
        """拉取最近一步 want_out 写下的 f16 logits。"""
        self._adb_ok("pull", f"{self.PERSIST_DIR}/out.f16.raw", local_path)
        return local_path

    def cancel(self, handle: PersistentHandle | str) -> None:
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.add(resolved.handle_id)

    def reset(self, handle: PersistentHandle | str) -> None:
        """清零 KV/SSM 状态 (新序列)。"""
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.discard(resolved.handle_id)
        self._rpc({"op": "reset", "handle_id": resolved.handle_id})

    def release(self, handle: PersistentHandle | str) -> None:
        resolved = self._resolve_handle(handle)
        with self._lock:
            self._cancelled.discard(resolved.handle_id)
            self._handles.pop(resolved.handle_id, None)

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = len(self._handles)
        pong = False
        if self.server_running():
            try:
                self._rpc({"op": "ping"}, timeout=30.0)
                pong = True
            except PersistentRuntimeError:
                pong = False
        return {
            "status": "available" if pong else "degraded",
            "active_handles": active,
            "device": self.device,
            "server_pid": self._server_pid,
            "transport": "file-queue",
        }

    def shutdown(self) -> None:
        with self._lock:
            self._handles.clear()
            self._cancelled.clear()
        self.stop_server()

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
