import numpy as np
import joblib
import umap
from pathlib import Path

ROOT = Path.home() / "VLM-Latent-Explorer"
PRECOMPUTED = ROOT / "precomputed"
MODELS = ["qwen", "monet", "lvr"]

for model_name in MODELS:
    cache_dir = PRECOMPUTED / "corpus_embeddings" / model_name
    npz_files = sorted(cache_dir.glob("*.npz"))

    if not npz_files:
        print(f"[{model_name}] No .npz files found — skipping")
        continue

    print(f"[{model_name}] Loading {len(npz_files)} files...")

    all_activations = []
    all_types = []
    all_labels = []
    all_example_ids = []

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        acts = data["activations"].astype(np.float32)

        # Remove rows with NaN (filler tokens from padding)
        valid_mask = ~np.isnan(acts).any(axis=1)
        acts = acts[valid_mask]

        token_types   = data["token_types"].tolist()
        token_strings = data["token_strings"].tolist()

        # Trim lists to match valid rows
        token_types   = [t for t, v in zip(token_types, valid_mask) if v]
        token_strings = [s for s, v in zip(token_strings, valid_mask) if v]

        all_activations.append(acts)
        all_types.extend(token_types)
        all_labels.extend(token_strings)
        all_example_ids.extend([npz_path.stem] * len(acts))

    activations = np.concatenate(all_activations, axis=0)
    print(f"[{model_name}] Fitting UMAP on {activations.shape} ...")

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer.fit_transform(activations)  # shape: (N, 2)

    # Save the fitted model — used later to project NEW inference points
    joblib.dump(reducer, PRECOMPUTED / f"umap_{model_name}.pkl")
    print(f"[{model_name}] Saved umap_{model_name}.pkl")

    # Save the 2D coords of the corpus — these are the background dots in the dashboard
    np.savez_compressed(
        PRECOMPUTED / f"corpus_2d_{model_name}.npz",
        coords=coords_2d.astype(np.float32),
        types=np.array(all_types, dtype=object),
        labels=np.array(all_labels, dtype=object),
        example_ids=np.array(all_example_ids, dtype=object),
    )
    print(f"[{model_name}] Saved corpus_2d_{model_name}.npz  ({len(activations)} tokens)")

print("Done.")
