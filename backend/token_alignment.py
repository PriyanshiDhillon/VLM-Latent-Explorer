"""Order-preserving alignment for generated token sequences."""

from __future__ import annotations


def _normalise(token: str) -> str:
    return str(token).strip().casefold()


def align_token_sequences(current_tokens, reference_tokens, current_types=None, reference_types=None):
    """Align token sequences without allowing reordering."""
    current_types = current_types or ["text"] * len(current_tokens)
    reference_types = reference_types or ["text"] * len(reference_tokens)
    n, m, gap = len(current_tokens), len(reference_tokens), -2
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): scores[i][0], trace[i][0] = i * gap, "up"
    for j in range(1, m + 1): scores[0][j], trace[0][j] = j * gap, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same_text = _normalise(current_tokens[i-1]) == _normalise(reference_tokens[j-1])
            same_type = current_types[i-1] == reference_types[j-1]
            diagonal = 3 if same_text and same_type else (-1 if same_type else -2)
            scores[i][j], trace[i][j] = max(
                (scores[i-1][j-1] + diagonal, "diag"),
                (scores[i-1][j] + gap, "up"),
                (scores[i][j-1] + gap, "left"), key=lambda item: item[0])
    rows, i, j = [], n, m
    while i or j:
        direction = trace[i][j]
        if i and j and direction == "diag":
            ci, ri = i-1, j-1
            same = (_normalise(current_tokens[ci]) == _normalise(reference_tokens[ri])
                    and current_types[ci] == reference_types[ri])
            rows.append(dict(current_index=ci, reference_index=ri,
                             current_token=current_tokens[ci], reference_token=reference_tokens[ri],
                             operation="match" if same else "replace"))
            i -= 1; j -= 1
        elif i and direction == "up":
            ci = i-1
            rows.append(dict(current_index=ci, reference_index=None,
                             current_token=current_tokens[ci], reference_token=None, operation="insert"))
            i -= 1
        else:
            ri = j-1
            rows.append(dict(current_index=None, reference_index=ri,
                             current_token=None, reference_token=reference_tokens[ri], operation="delete"))
            j -= 1
    return list(reversed(rows))
