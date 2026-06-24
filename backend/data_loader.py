import os
import json
import joblib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRECOMPUTED_DIR = Path(os.environ.get("PRECOMPUTED_DIR", "precomputed"))
if not PRECOMPUTED_DIR.is_absolute():
    PRECOMPUTED_DIR = PROJECT_ROOT / PRECOMPUTED_DIR

DATA_DIR = Path("data/subset")

MODELS = ["qwen", "monet", "lvr"]


def load_umap_model(model_name: str):
    """Load a pre-fitted UMAP model from disk."""
    path = PRECOMPUTED_DIR / f"umap_{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"UMAP model not found at {path}. Run offline pipeline first.")
    return joblib.load(path)


def load_corpus_embeddings(model_name: str) -> dict:
    """
    Load the precomputed 2D UMAP projections for the reference corpus.

    Returns a dict with keys:
        - 'coords':     np.ndarray (N, 2)  — 2D UMAP coordinates
        - 'types':      list[str]          — 'text' | 'visual' | 'latent' per point
        - 'example_ids': list[str]         — which example each point came from
        - 'labels':     list[str]          — for hover: token string or description
    """
    path = PRECOMPUTED_DIR / f"corpus_2d_{model_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Corpus embeddings not found at {path}. Run offline pipeline first.")
    data = np.load(path, allow_pickle=True)
    out = {
            "coords":      data["coords"],
            "types":       data["types"].tolist(),
            "example_ids": data["example_ids"].tolist(),
            "labels":      data["labels"].tolist(),
        }
    for k in ("gen_index", "token_index"):
        if k in data:
            out[k] = data[k].tolist()
    return out


def load_example_cache(example_id: str, model_name: str) -> dict:
    """
    Load all precomputed per-example data for one model run.

    Returns a dict with keys:
        - 'activations':     np.ndarray (T, D)    — raw hidden states, one per token
        - 'coords_2d':       np.ndarray (T, 2)    — UMAP projection of activations
        - 'token_types':     list[str]            — 'text' | 'visual' | 'latent' per token
        - 'attn_weights':    np.ndarray (T, H, W) — cross-attention to image grid per step
        - 'generated_text':  str                  — full model output (reasoning + answer)
        - 'token_strings':   list[str]            — decoded string for each generated token
        - 'correct':         bool                 — did model get right answer
    """
    path = PRECOMPUTED_DIR / "corpus_embeddings" / model_name / f"{example_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Example cache not found at {path}.")
    data = np.load(path, allow_pickle=True)
    return {
        "activations":    data["activations"],
        "coords_2d":      data["coords_2d"],
        "token_types":    data["token_types"].tolist(),
        "attn_weights":   data["attn_weights"],
        "generated_text": str(data["generated_text"]),
        "token_strings":  data["token_strings"].tolist(),
        "correct":        bool(data["correct"]),
    }


def list_examples() -> list[dict]:
    """
    Return metadata for all examples in the local subset.

    Each dict has: id, image_path, question, answer
    """
    meta_path = DATA_DIR / "metadata.json"
    if not meta_path.exists():
        return []
    with open(meta_path) as f:
        return json.load(f)


def load_example_image(example_id: str) -> str:
    """Return the filesystem path to the image for a given example."""
    return str(DATA_DIR / "images" / f"{example_id}.jpg")


def corpus_embeddings_exist(model_name: str) -> bool:
    return (PRECOMPUTED_DIR / f"corpus_2d_{model_name}.npz").exists()


def umap_model_exists(model_name: str) -> bool:
    return (PRECOMPUTED_DIR / f"umap_{model_name}.pkl").exists()