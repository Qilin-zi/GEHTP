"""Sampling strategies for GEHTP inference.

Implements greedy, top-p (nucleus), and temperature-based sampling.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple


class SamplerError(RuntimeError):
    """Raised when sampling parameters are invalid."""


def validate_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> None:
    """Validate sampling parameters."""
    if temperature < 0 or temperature > 2.0:
        raise SamplerError("temperature must be in range [0, 2]")
    if top_p <= 0 or top_p > 1.0:
        raise SamplerError("top_p must be in range (0, 1]")
    if top_k < 0:
        raise SamplerError("top_k must be non-negative")


def softmax(logits: List[float], _inv_temp: float = 1.0) -> List[float]:
    """Compute softmax probabilities from logits (vectorized approximation)."""
    if _inv_temp != 1.0:
        logits = [x * _inv_temp for x in logits]
    max_logit = max(logits) if logits else 0.0
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    if total == 0:
        return [1.0 / len(logits)] * len(logits) if logits else []
    return [e / total for e in exp_logits]


def top_k_top_p_filtering(
    probs: List[float],
    top_p: float = 1.0,
    top_k: int = 0,
) -> List[float]:
    """Filter distribution to top-p nucleus and/or top-k candidates."""
    vocab_size = len(probs)
    if top_k == 0 and top_p >= 1.0:
        return probs[:]

    indexed = list(enumerate(probs))
    indexed.sort(key=lambda x: x[1], reverse=True)

    if top_k > 0:
        indexed = indexed[:top_k]

    cumulative = 0.0
    cutoff_idx = 0
    for idx, (orig_idx, prob) in enumerate(indexed):
        cumulative += prob
        cutoff_idx = idx + 1
        if cumulative >= top_p:
            break

    allowed = set(orig_idx for orig_idx, _ in indexed[:cutoff_idx])

    filtered = [0.0] * vocab_size
    for orig_idx, prob in zip(allowed, [p for _, p in indexed[:cutoff_idx]]):
        filtered[orig_idx] = prob

    total = sum(filtered)
    if total > 0:
        filtered = [p / total for p in filtered]
    return filtered


def sample_from_logits(
    logits: List[float],
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    rng: Optional[random.Random] = None,
) -> Tuple[int, float]:
    """Sample a token ID from logits with given sampling parameters.

    Returns ``(token_id, probability)`` tuple.
    """
    validate_params(temperature, top_p, top_k)

    inv_temp = 1.0 / temperature if temperature > 0 else float("inf")
    probs = softmax(logits, inv_temp)
    probs = top_k_top_p_filtering(probs, top_p=top_p, top_k=top_k)

    r = (rng or random.Random()).random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i, p

    return len(probs) - 1, probs[-1] if probs else (0, 0.0)


def greedy_sample(logits: List[float]) -> int:
    """Return the argmax token ID (equivalent to temperature=0)."""
    if not logits:
        raise SamplerError("empty logits")
    return max(range(len(logits)), key=lambda i: logits[i])


class Sampler:
    """Stateful sampler holding temperature/top-p configuration."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> None:
        validate_params(temperature, top_p, top_k)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._rng = random.Random()

    def step(self, logits: List[float]) -> Tuple[int, float]:
        """Sample from logits with the configured parameters."""
        if self.temperature == 0:
            return greedy_sample(logits), 1.0
        return sample_from_logits(
            logits,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            rng=self._rng,
        )

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility."""
        self._rng.seed(seed)
