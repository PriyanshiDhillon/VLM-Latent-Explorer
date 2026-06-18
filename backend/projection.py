"""
Projection utilities.

- Project new token activations onto a precomputed UMAP manifold.
- Run on-demand t-SNE on a user-selected subset.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.manifold import TSNE

PRECOMPUTED_DIR = Path("precomputed")


def project_onto_manifold(activations: np.ndarray, model_name: str) -> np.ndarray:
    """
    Project new activations (T, D) onto the precomputed UMAP manifold.

    Returns coords (T, 2).
    """
    umap_path = PRECOMPUTED_DIR / f"umap_{model_name}.pkl"
    if not umap_path.exists():
        raise FileNotFoundError(
            f"UMAP model not found at {umap_path}. Run the offline pipeline first."
        )
    umap_model = joblib.load(umap_path)
    coords = umap_model.transform(activations.astype(np.float32))
    return coords


def tsne_reproject(points_2d: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Run t-SNE on a subset of 2D UMAP points (the bounding-box selection).

    We use the 2D UMAP coords as input (cheap; avoids re-running on raw
    high-dim activations which we may not have for corpus reference points).

    Returns:
        coords_tsne : np.ndarray (N, 2)
        labels      : list[str]  (passed through unchanged)
    """
    if len(points_2d) < 5:
        return points_2d, labels

    perplexity = min(30, max(2, len(points_2d) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=500,
        init="pca" if len(points_2d) > 10 else "random",
    )
    coords_tsne = tsne.fit_transform(points_2d)
    return coords_tsne, labels


def compute_selection_stats(
    token_types: list[str],
    attn_weights_at_step: np.ndarray | None,
    correct_flags: list[bool],
) -> dict:
    """
    Compute summary statistics for the points inside the bounding box.

    Parameters
    ----------
    token_types        : type label per selected point
    attn_weights_at_step : (num_heads, src_len) attention for the current step, or None
    correct_flags      : per-instance correctness booleans

    Returns
    -------
    dict with human-readable stat strings
    """
    counts = {"text": 0, "visual": 0, "latent": 0}
    for t in token_types:
        if t in counts:
            counts[t] += 1

    total = sum(counts.values())
    stats = {
        "total_points": total,
        "text_count":   counts["text"],
        "visual_count": counts["visual"],
        "latent_count": counts["latent"],
    }

    if attn_weights_at_step is not None:
        mean_attn = attn_weights_at_step.mean(axis=0) 
        stats["max_attention_position"] = int(mean_attn.argmax())
        stats["mean_attention_entropy"] = float(
            -(mean_attn * np.log(mean_attn + 1e-9)).sum()
        )

    if correct_flags:
        stats["accuracy"] = f"{sum(correct_flags)}/{len(correct_flags)}"

    return stats