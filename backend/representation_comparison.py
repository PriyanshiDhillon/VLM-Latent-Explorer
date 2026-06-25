"""Metrics for comparing aligned token hidden-state representations."""

from __future__ import annotations

import numpy as np


def cosine_change(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return 1 - cosine similarity for two token activations.

    Zero means identical direction; larger values indicate a stronger internal
    representation shift.  ``None`` is returned for incompatible/empty input.
    """
    first = np.asarray(first, dtype=np.float32).reshape(-1)
    second = np.asarray(second, dtype=np.float32).reshape(-1)
    if first.size == 0 or first.shape != second.shape:
        return None
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None
    similarity = float(np.dot(first, second) / denominator)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))

