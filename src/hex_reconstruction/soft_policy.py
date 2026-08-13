"""Dependency-free reference for the future soft policy loss."""

from __future__ import annotations

import math
from typing import Sequence


def log_softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("logits cannot be empty")
    maximum = max(logits)
    log_normalizer = maximum + math.log(sum(math.exp(value - maximum) for value in logits))
    return [value - log_normalizer for value in logits]


def soft_policy_cross_entropy(logits: Sequence[float], pi: Sequence[float]) -> float:
    """Compute -sum(pi * log_softmax(logits)) without reducing pi to argmax."""

    if len(logits) != len(pi):
        raise ValueError("logits and pi must have the same length")
    if not math.isclose(sum(pi), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("pi must sum to one")
    return -sum(target * log_probability for target, log_probability in zip(pi, log_softmax(logits)))

