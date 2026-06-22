"""Utilities for generated-token attention analysis."""

from __future__ import annotations

import numpy as np


def generated_attention_matrix(
    attention_steps: list,
    prompt_length: int,
    token_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lower-triangular generated-history attention matrices.

    Attention for generated token ``t`` is captured on generation forward pass
    ``t+1``, after that token is processed. The displayed matrix excludes its
    diagonal self-attention and is normalized over earlier generated tokens.
    Raw weights and total generated-history mass are returned for hover details.
    """
    raw = np.full((token_count, token_count), np.nan, dtype=np.float32)
    normalized = np.full_like(raw, np.nan)
    history_mass = np.zeros(token_count, dtype=np.float32)

    for row in range(min(token_count, max(0, len(attention_steps) - 1))):
        weights = attention_steps[row + 1]
        if weights is None:
            continue
        weights = np.asarray(weights, dtype=np.float32)
        if weights.size == 0:
            continue
        if weights.ndim == 2:
            weights = weights.mean(axis=0)
        weights = weights.reshape(-1)

        available = min(row, token_count, max(0, len(weights) - prompt_length))
        if available == 0:
            continue
        generated_weights = weights[prompt_length : prompt_length + available]
        raw[row, :available] = generated_weights
        mass = float(generated_weights.sum())
        history_mass[row] = mass
        if mass > 0:
            normalized[row, :available] = generated_weights / mass

    return normalized, raw, history_mass
