"""
Projection utilities.

- Project new token activations onto a precomputed UMAP manifold.
- Run on-demand t-SNE on a user-selected subset.
"""

from __future__ import annotations
import os

import numpy as np
import joblib
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTED_DIR = Path(os.environ.get("PRECOMPUTED_DIR", "precomputed"))
if not PRECOMPUTED_DIR.is_absolute():
    PRECOMPUTED_DIR = PROJECT_ROOT / PRECOMPUTED_DIR

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


def compute_neighborhood_preservation_scores(
    activations: np.ndarray,
    coords_2d: np.ndarray,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Return a local projection-reliability score for every token.

    Each score is the fraction of nearest neighbours shared between the
    original hidden-state space and the 2D projection. A score of 1 means the
    local neighbourhood is fully preserved; 0 means it is completely changed.
    This is a per-token diagnostic, not scikit-learn's global trustworthiness.
    """
    high_dim = np.asarray(activations, dtype=np.float32)
    low_dim = np.asarray(coords_2d, dtype=np.float32)

    if high_dim.ndim != 2 or low_dim.ndim != 2:
        raise ValueError("activations and coords_2d must both be 2D arrays")
    if len(high_dim) != len(low_dim):
        raise ValueError(
            "activations and coords_2d must contain the same number of tokens"
        )
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")

    token_count = len(high_dim)
    if token_count == 0:
        return np.empty(0, dtype=np.float32)
    if token_count == 1:
        return np.ones(1, dtype=np.float32)

    k = min(n_neighbors, token_count - 1)
    high_dist = pairwise_distances(high_dim, metric="euclidean")
    low_dist = pairwise_distances(low_dim, metric="euclidean")
    np.fill_diagonal(high_dist, np.inf)
    np.fill_diagonal(low_dist, np.inf)
    high_neighbors = np.argsort(high_dist, axis=1)[:, :k]
    low_neighbors = np.argsort(low_dist, axis=1)[:, :k]

    scores = np.empty(token_count, dtype=np.float32)
    for index in range(token_count):
        overlap = np.intersect1d(
            high_neighbors[index], low_neighbors[index], assume_unique=True
        ).size
        scores[index] = overlap / k
    return scores

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


def _attention_entropy(attn_weights: np.ndarray | list | None) -> float | None:
    if attn_weights is None:
        return None

    arr = np.asarray(attn_weights, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim >= 2:
        arr = arr.mean(axis=0)

    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        return None

    probs = arr / total
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    # Numerical guard: entropy should be >= 0; clamp tiny negative noise.
    entropy = float(max(0.0, entropy))
    return entropy



def _pairwise_distance_summary(vectors: np.ndarray | list | None) -> dict[str, float] | None:
    if vectors is None:
        return None

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or len(arr) < 2:
        return None

    distances = pairwise_distances(arr, metric="euclidean")
    upper = distances[np.triu_indices(len(arr), k=1)]
    if upper.size == 0:
        return None

    return {
        "mean": float(upper.mean()),
        "std": float(upper.std()),
        "min": float(upper.min()),
        "max": float(upper.max()),
    }


def compute_selection_stats(
    token_types: list[str],
    attn_weights: list | np.ndarray | None,
    correct_flags: list[bool],
    selection_indices: list[int] | None = None,
    trust_scores: list[float] | None = None,
    activations: np.ndarray | list | None = None,
    coords_2d: np.ndarray | list | None = None,
    nearest_text: list[dict | None] | None = None,
) -> dict:
    """
    Compute summary statistics for the points inside the bounding box.

    Parameters
    ----------
    token_types        : type label per selected point
    attn_weights       : per-token attention weights (full sequence, if available)
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
        "total_points": int(total),
        "text_count":   int(counts["text"]),
        "visual_count": int(counts["visual"]),
        "latent_count": int(counts["latent"]),
    }

    # Percentages per type
    # if total > 0:
    #     stats.update({
    #         "text_pct":   f"{counts['text']/total:.2%}",
    #         "visual_pct": f"{counts['visual']/total:.2%}",
    #         "latent_pct": f"{counts['latent']/total:.2%}",
    #     })

    selected_indices = selection_indices or []

    # Entropy split by token type, computed from each selected token's own attention.
    # if attn_weights is not None and selected_indices and len(token_types) == len(selected_indices):
    #     entropy_by_type: dict[str, list[float]] = {"text": [], "visual": [], "latent": []}
    #     all_entropies: list[float] = []
    #     attn_seq = list(attn_weights)
        # for idx, token_type in zip(selected_indices, token_types):
        #     if token_type not in entropy_by_type or idx >= len(attn_seq):
        #         continue
        #     entropy = _attention_entropy(attn_seq[idx])
        #     if entropy is None:
        #         continue
        #     entropy_by_type[token_type].append(entropy)
        #     all_entropies.append(entropy)

        # for token_type, values in entropy_by_type.items():
        #     if not values:
        #         continue
        #     prefix = f"{token_type}_attention_entropy"
        #     stats[f"{prefix}_mean"] = float(np.mean(values))
        #     stats[f"{prefix}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        #     stats[f"{prefix}_min"] = float(np.min(values))
        #     stats[f"{prefix}_max"] = float(np.max(values))
        #     stats[f"{prefix}_count"] = len(values)

        # if all_entropies:
        #     stats["attention_entropy_mean"] = float(np.mean(all_entropies))
        #     stats["attention_entropy_std"] = float(np.std(all_entropies)) if len(all_entropies) > 1 else 0.0

    # Spatial dispersion in original activation space and projected space.
    if selected_indices and activations is not None:
        activation_arr = np.asarray(activations, dtype=np.float32)
        valid_indices = [i for i in selected_indices if 0 <= i < len(activation_arr)]
        if len(valid_indices) >= 2:
            activation_summary = _pairwise_distance_summary(activation_arr[valid_indices])
            if activation_summary:
                stats["selection_activation_pairwise_distance_mean"] = activation_summary["mean"]
                stats["selection_activation_pairwise_distance_std"] = activation_summary["std"]
                stats["selection_activation_pairwise_distance_min"] = activation_summary["min"]
                stats["selection_activation_pairwise_distance_max"] = activation_summary["max"]

    projected_summary = None
    activation_summary = None
    if selected_indices and coords_2d is not None:
        coords_arr = np.asarray(coords_2d, dtype=np.float32)
        valid_indices = [i for i in selected_indices if 0 <= i < len(coords_arr)]
        if len(valid_indices) >= 2:
            projected_summary = _pairwise_distance_summary(coords_arr[valid_indices])
            if projected_summary:
                stats["selection_projection_pairwise_distance_mean"] = projected_summary["mean"]
                stats["selection_projection_pairwise_distance_std"] = projected_summary["std"]
                stats["selection_projection_pairwise_distance_min"] = projected_summary["min"]
                stats["selection_projection_pairwise_distance_max"] = projected_summary["max"]

    if selected_indices and activations is not None:
        activation_arr = np.asarray(activations, dtype=np.float32)
        valid_indices = [i for i in selected_indices if 0 <= i < len(activation_arr)]
        if len(valid_indices) >= 2:
            activation_summary = _pairwise_distance_summary(activation_arr[valid_indices])
            if activation_summary and projected_summary:
                stats["selection_projection_to_activation_distance_ratio"] = (
                    projected_summary["mean"] / activation_summary["mean"]
                    if activation_summary["mean"] > 0 else None
                )

    # Aggregate nearest-corpus text distance for the selected region.
    if nearest_text is not None and selected_indices:
        selected_distances = []
        selected_by_type: dict[str, list[float]] = {"text": [], "visual": [], "latent": []}
        for idx, token_type in zip(selected_indices, token_types):
            if idx >= len(nearest_text):
                continue
            neighbor = nearest_text[idx]
            if not neighbor:
                continue
            distance = neighbor.get("distance")
            if distance is None:
                continue
            distance = float(distance)
            selected_distances.append(distance)
            if token_type in selected_by_type:
                selected_by_type[token_type].append(distance)

        if selected_distances:
            stats["mean_nearest_corpus_text_distance"] = float(np.mean(selected_distances))
            stats["nearest_corpus_text_distance_std"] = float(np.std(selected_distances)) if len(selected_distances) > 1 else 0.0
            stats["nearest_corpus_text_distance_min"] = float(np.min(selected_distances))
            stats["nearest_corpus_text_distance_max"] = float(np.max(selected_distances))

        for token_type, values in selected_by_type.items():
            if values:
                stats[f"{token_type}_mean_nearest_corpus_text_distance"] = float(np.mean(values))

    # Trustworthiness / uncertainty for selected points
    if trust_scores is not None and selection_indices:
        ts = np.asarray(trust_scores, dtype=np.float64)
        sel = [i for i in selection_indices if 0 <= i < len(ts)]
        if sel:
            vals = ts[sel]
            stats["mean_trustworthiness"] = round(float(vals.mean()), 2)
            stats["median_trustworthiness"] = round(float(np.median(vals)), 2)
            stats["trustworthiness_std"] = round(float(vals.std()), 2) if len(vals) > 1 else 0.0
        else:
            stats["mean_trustworthiness"] = None
            stats["median_trustworthiness"] = None
            stats["trustworthiness_std"] = None

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
