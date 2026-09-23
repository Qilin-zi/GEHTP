"""In-memory session and device KV-cache metadata management."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionStatus(str, Enum):
    ACTIVE = "active"
    FINISHED = "finished"
    ERROR = "error"


class KVCacheStatus(str, Enum):
    ACTIVE = "active"
    FULL = "full"
    RELEASED = "released"


@dataclass
class KVCacheHandle:
    handle_id: str
    session_id: str
    model_id: str
    kv_base_addr: int = 0
    slot_id: int = 0
    current_position: int = 0
    max_position: int = 4096
    k_shape: List[int] = field(default_factory=list)
    v_shape: List[int] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: KVCacheStatus = KVCacheStatus.ACTIVE

    def advance(self, tokens: int = 1) -> None:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        next_position = self.current_position + tokens
        if next_position > self.max_position:
            raise ValueError("KV cache capacity exceeded")
        self.current_position = next_position
        if self.current_position == self.max_position:
            self.status = KVCacheStatus.FULL

    def release(self) -> None:
        self.status = KVCacheStatus.RELEASED


@dataclass
class Session:
    session_id: str
    model_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    messages: List[Dict[str, str]] = field(default_factory=list)
    kv_position: int = 0
    kv_handle: Optional[KVCacheHandle] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_tokens: int = 16

    def touch(self) -> None:
        self.last_active = datetime.now(timezone.utc)

    def append_messages(self, messages: List[Dict[str, str]]) -> None:
        self.messages.extend(messages)
        self.touch()

    def advance(self, tokens: int = 1) -> None:
        if self.kv_handle is not None:
            self.kv_handle.advance(tokens)
        self.kv_position += tokens
        self.touch()

    def finish(self) -> None:
        self.status = SessionStatus.FINISHED
        self.touch()

    def fail(self) -> None:
        self.status = SessionStatus.ERROR
        self.touch()

    def reactivate(self) -> None:
        if self.status in (SessionStatus.FINISHED, SessionStatus.ERROR):
            self.status = SessionStatus.ACTIVE
            self.touch()


class SessionManager:
    """Thread-safe manager for sessions and one KV handle per session."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._handles: Dict[str, KVCacheHandle] = {}
        self._lock = threading.RLock()

    def create(
        self,
        model_id: str,
        *,
        max_tokens: int = 16,
        messages: Optional[List[Dict[str, str]]] = None,
        kv_max_position: int = 4096,
        kv_base_addr: int = 0,
        slot_id: int = 0,
    ) -> Session:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if kv_max_position <= 0:
            raise ValueError("kv_max_position must be positive")
        session_id = str(uuid.uuid4())
        handle = KVCacheHandle(
            handle_id=str(uuid.uuid4()),
            session_id=session_id,
            model_id=model_id,
            max_position=kv_max_position,
            kv_base_addr=kv_base_addr,
            slot_id=slot_id,
        )
        session = Session(
            session_id=session_id,
            model_id=model_id,
            max_tokens=max_tokens,
            messages=list(messages or []),
            kv_handle=handle,
        )
        with self._lock:
            self._sessions[session_id] = session
            self._handles[handle.handle_id] = handle
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def get_handle(self, session_id: str) -> KVCacheHandle:
        session = self.get(session_id)
        if session.kv_handle is None:
            raise KeyError(f"session has no KV cache: {session_id}")
        return session.kv_handle

    def finish(self, session_id: str) -> None:
        self.get(session_id).finish()

    def fail(self, session_id: str) -> None:
        self.get(session_id).fail()

    def release(self, session_id: str, *, remove: bool = False) -> None:
        with self._lock:
            session = self.get(session_id)
            if session.kv_handle is not None:
                session.kv_handle.release()
                self._handles.pop(session.kv_handle.handle_id, None)
                session.kv_handle = None
            if remove:
                self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        with self._lock:
            return sum(s.status == SessionStatus.ACTIVE for s in self._sessions.values())

    def list_sessions(self) -> List[Session]:
        with self._lock:
            return list(self._sessions.values())

    def clear(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                if session.kv_handle is not None:
                    session.kv_handle.release()
            self._sessions.clear()
            self._handles.clear()
