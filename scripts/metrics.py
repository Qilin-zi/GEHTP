"""In-memory request metrics for the internal inference service."""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Deque, Dict, Optional


@dataclass
class ModelMetrics:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    queue_times_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    first_token_latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))
    generation_latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=10_000))


class MetricsCollector:
    """Thread-safe counters and bounded latency samples."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelMetrics] = defaultdict(ModelMetrics)
        self._queue_length = 0
        self._active_sessions = 0
        self._lock = threading.RLock()

    def record_request(
        self,
        model: str,
        *,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        queue_time_ms: Optional[float] = None,
        first_token_latency_ms: Optional[float] = None,
        generation_latency_ms: Optional[float] = None,
    ) -> None:
        with self._lock:
            stats = self._models[model]
            stats.requests += 1
            stats.successes += int(success)
            stats.failures += int(not success)
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens
            if queue_time_ms is not None:
                stats.queue_times_ms.append(float(queue_time_ms))
            if first_token_latency_ms is not None:
                stats.first_token_latencies_ms.append(float(first_token_latency_ms))
            if generation_latency_ms is not None:
                stats.generation_latencies_ms.append(float(generation_latency_ms))

    def set_queue_length(self, value: int) -> None:
        with self._lock:
            self._queue_length = max(0, value)

    def set_active_sessions(self, value: int) -> None:
        with self._lock:
            self._active_sessions = max(0, value)

    @staticmethod
    def _summary(values: Deque[float]) -> Dict[str, float]:
        if not values:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
        ordered = sorted(values)

        def percentile(percent: float) -> float:
            rank = (len(ordered) - 1) * percent
            lower = math.floor(rank)
            upper = math.ceil(rank)
            if lower == upper:
                return ordered[lower]
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)

        return {
            "avg": mean(ordered),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
        }

    def snapshot(self, *, device_runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            models = {}
            for model_id, stats in self._models.items():
                models[model_id] = {
                    "requests": stats.requests,
                    "successes": stats.successes,
                    "failures": stats.failures,
                    "success_rate": stats.successes / stats.requests if stats.requests else 0.0,
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "queue_latency_ms": self._summary(stats.queue_times_ms),
                    "first_token_latency_ms": self._summary(stats.first_token_latencies_ms),
                    "generation_latency_ms": self._summary(stats.generation_latencies_ms),
                }
            return {
                "models": models,
                "queue_length": self._queue_length,
                "active_sessions": self._active_sessions,
                "device_runtime": device_runtime or {},
            }

    def reset(self) -> None:
        with self._lock:
            self._models.clear()
            self._queue_length = 0
            self._active_sessions = 0
