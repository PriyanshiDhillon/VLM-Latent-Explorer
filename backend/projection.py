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
MAX_LEGACY_UMAP_BYTES = 512 * 1024 * 1024


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
    if umap_path.stat().st_size > MAX_LEGACY_UMAP_BYTES:
        raise RuntimeError(
            f"{umap_path} is a legacy high-memory UMAP artifact. "
            "Regenerate the offline projections with experiment/fit_umap.py."
        )
    umap_model = joblib.load(umap_path)
    coords = umap_model.transform(activations.astype(np.float32))
    return coords


def tsne_reproject(points_2d: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Run t-SNE on a shared set of 2D UMAP points.

    We use the 2D UMAP coordinates as input because raw high-dimensional
    activations are not retained in the dashboard's corpus store. All points
    that will be overlaid must be passed in one call.

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


def find_k_nearest_text_neighbors(
    query_coord: np.ndarray,
    corpus_coords: np.ndarray,
    corpus_types: list[str],
    corpus_labels: list[str],
    corpus_example_ids: list | None = None,
    k: int = 10,
) -> list[dict]:
    """Return the k nearest corpus text tokens to a single 2D query point."""
    from scipy.spatial import cKDTree

    text_indices = [i for i, t in enumerate(corpus_types) if t == "text"]
    if not text_indices or query_coord is None:
        return []

    text_coords = corpus_coords[np.array(text_indices)]
    text_labels = [corpus_labels[i] for i in text_indices]
    text_eids = (
        [corpus_example_ids[i] for i in text_indices]
        if corpus_example_ids is not None
        else [None] * len(text_indices)
    )

    k = min(k, len(text_indices))
    tree = cKDTree(text_coords)
    distances, indices = tree.query(
        query_coord.reshape(1, -1).astype(np.float64), k=k
    )
    distances = np.atleast_2d(distances)[0]
    indices = np.atleast_2d(indices)[0]

    return [
        {
            "label": text_labels[int(idx)],
            "distance": float(dist),
            "example_id": text_eids[int(idx)],
        }
        for idx, dist in zip(indices, distances)
    ]


def find_nearest_text_neighbors(
    query_coords: np.ndarray,
    corpus_coords: np.ndarray,
    corpus_types: list[str],
    corpus_labels: list[str],
) -> list[dict | None]:
    """
    For each query point, return the nearest corpus point of type 'text'.

    Parameters
    ----------
    query_coords  : (K, 2) 2D UMAP coordinates of the tokens to look up
    corpus_coords : (N, 2) 2D UMAP coordinates of the full corpus
    corpus_types  : length-N token type per corpus point
    corpus_labels : length-N token string per corpus point

    Returns
    -------
    list of length K — {"label": str, "distance": float} or None per entry
    """
    from scipy.spatial import cKDTree

    text_indices = [i for i, t in enumerate(corpus_types) if t == "text"]
    if not text_indices or len(query_coords) == 0:
        return [None] * len(query_coords)

    text_coords = corpus_coords[np.array(text_indices)]
    text_labels = [corpus_labels[i] for i in text_indices]

    tree = cKDTree(text_coords)
    distances, indices = tree.query(query_coords.astype(np.float64), k=1)

    return [
        {"label": text_labels[int(idx)], "distance": float(dist)}
        for idx, dist in zip(indices, distances)
    ]
