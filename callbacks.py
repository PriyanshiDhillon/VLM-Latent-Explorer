"""
All Dash callbacks.

Wiring:
  run-btn            → runs inference, populates stores + trace + slider
  step-slider        → updates heatmap + UMAP highlight
  model-selector     → reloads corpus embeddings manifold
  umap-graph         → bounding-box selection triggers t-SNE + stats
  instance-list      → toggle active instance
  image-upload       → preview image
  example-selector   → pre-fill question + image
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import numpy as np
from pathlib import Path
from dash import Input, Output, Patch, State, callback, ctx, html, no_update, ALL
from dash.exceptions import PreventUpdate
from PIL import Image, ImageDraw

from backend import heatmap as hm
from backend import projection as proj
from backend import data_loader as dl
from backend import inference as inf
from backend import token_attention as token_attn
from backend import token_alignment
from backend import representation_comparison

import plotly.graph_objects as go


TOKEN_COLORS = {
    "text":   "#3b82f6",   
    "visual": "#22c55e", 
    "latent": "#f97316", 
}

INSTANCE_LINE_STYLES = ["solid", "dash", "dot", "dashdot"]
_TSNE_CACHE: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

_BLANK_MASK_FIGURE = {
    "data": [],
    "layout": {
        "template": "plotly_white",
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "xaxis": {"showgrid": False, "showticklabels": False, "zeroline": False},
        "yaxis": {"showgrid": False, "showticklabels": False, "zeroline": False},
        "paper_bgcolor": "#1e293b",
        "plot_bgcolor": "#1e293b",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_pil(b64_str: str) -> Image.Image:
    """Convert a base64 data-URI to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _apply_image_mask(image: Image.Image, mask_region: dict) -> Image.Image:
    """Black out a normalised rectangular region of an image."""
    img = image.copy().convert("RGB")
    W, H = img.size
    x0 = int(max(0.0, mask_region["x0_norm"]) * W)
    y0 = int(max(0.0, mask_region["y0_norm"]) * H)
    x1 = int(min(1.0, mask_region["x1_norm"]) * W)
    y1 = int(min(1.0, mask_region["y1_norm"]) * H)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])
    if x1 > x0 and y1 > y0:
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    return img


def _execute_inference_and_update(
    image: Image.Image,
    question: str,
    model_name: str,
    existing_instances: dict,
    corpus: dict,
    instance_prefix: str = "instance",
    mask_region: dict | None = None,
) -> tuple:
    """Run inference and package results; shared by Run Inference and Run Intervention."""
    if mask_region:
        image = _apply_image_mask(image, mask_region)

    result = inf.run_inference(image, question, model_name)
    result["model_name"] = model_name
    inst_id = f"{instance_prefix}_{len(existing_instances) + 1}"

    cache_dir = dl.PRECOMPUTED_DIR / "online_cache" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        cache_dir / f"{inst_id}.npz",
        activations=result["activations"],
        token_types=np.array(result["token_types"]),
        token_strings=np.array(result["token_strings"]),
        token_ids=np.array(result["token_ids"]),
        generated_text=result["generated_text"],
        prompt_length=result["prompt_length"],
        image_grid_hw=np.array(result["image_grid_hw"]),
    )

    attn_arr = np.empty(len(result["attn_weights"]), dtype=object)
    for i, w in enumerate(result["attn_weights"]):
        attn_arr[i] = w if w is not None else np.array([])
    np.save(cache_dir / f"{inst_id}_attn_weights.npy", attn_arr, allow_pickle=True)

    layer_attn_arr = np.empty(len(result["layer_attn_weights"]), dtype=object)
    for li, layer_steps in enumerate(result["layer_attn_weights"]):
        step_arr = np.empty(len(layer_steps), dtype=object)
        for si, w in enumerate(layer_steps):
            step_arr[si] = w if w is not None else np.array([])
        layer_attn_arr[li] = step_arr
    np.save(cache_dir / f"{inst_id}_layer_attn_weights.npy", layer_attn_arr, allow_pickle=True)

    if dl.umap_model_exists(model_name):
        try:
            result["coords_2d"] = proj.project_onto_manifold(result["activations"], model_name)
            result["trustworthiness_scores"] = proj.compute_neighborhood_preservation_scores(
                result["activations"], result["coords_2d"]
            )
        except Exception as exc:
            result["coords_2d"] = None
            result["trustworthiness_scores"] = None
            result["projection_error"] = str(exc)
            print(f"[projection] {model_name}/{inst_id}: {exc}")
    else:
        result["coords_2d"] = None
        result["trustworthiness_scores"] = None

    nearest_text: list = [None] * len(result["token_types"])
    if result.get("coords_2d") is not None and corpus and corpus.get("coords") and corpus.get("labels"):
        corpus_coords_arr = np.array(corpus["coords"], dtype=np.float32)
        latent_indices = [i for i, t in enumerate(result["token_types"]) if t == "latent"]
        if latent_indices:
            latent_coords = result["coords_2d"][np.array(latent_indices)]
            neighbors = proj.find_nearest_text_neighbors(
                latent_coords, corpus_coords_arr, corpus["types"], corpus["labels"]
            )
            for idx, neighbor in zip(latent_indices, neighbors):
                nearest_text[idx] = neighbor
    result["nearest_text"] = nearest_text

    if mask_region:
        result["mask_region"] = mask_region

    updated = dict(existing_instances)
    updated[inst_id] = _result_to_serialisable(
        {k: v for k, v in result.items() if k != "activations"}
    )
    n_steps = len(result["token_strings"])
    marks = {i: str(i) for i in range(0, n_steps, max(1, n_steps // 10))}
    trace_children = _build_reasoning_trace(
        result["token_strings"], result["token_types"], result.get("nearest_text")
    )
    instance_children = _build_instance_list(updated, inst_id)
    return updated, inst_id, n_steps - 1, 0, marks, trace_children, instance_children


def _result_to_serialisable(result: dict) -> dict:
    """Convert numpy arrays in inference result to lists for dcc.Store."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(result)


def _result_from_serialisable(result: dict) -> dict:
    """Restore numpy arrays from dcc.Store dict."""
    out = {}
    for k, v in result.items():
        if k in ("activations", "coords_2d") and v is not None:
            out[k] = np.array(v)
        else:
            out[k] = v
    return out

def _passage(example_id):
    """Map an example id like 'example_000007' to a 1-based passage number (8)."""
    try:
        return int(str(example_id).split("_")[-1]) + 1
    except (ValueError, IndexError):
        return example_id


def _joint_tsne_projection(corpus: dict, instances: dict) -> tuple[dict, dict]:
    """Fit corpus and live points together so every trace shares one t-SNE frame."""
    corpus_coords = np.asarray(corpus.get("coords", []), dtype=np.float32)
    instance_coords = {}
    digest = hashlib.blake2b(digest_size=16)
    digest.update(corpus_coords.tobytes())

    for inst_id, inst_data in instances.items():
        result = _result_from_serialisable(inst_data)
        coords = result.get("coords_2d")
        if coords is None or len(coords) == 0:
            continue
        coords = np.asarray(coords, dtype=np.float32)
        instance_coords[inst_id] = coords
        digest.update(inst_id.encode("utf-8"))
        digest.update(coords.tobytes())

    cache_key = digest.hexdigest()
    cached = _TSNE_CACHE.get(cache_key)
    if cached is None:
        blocks = [corpus_coords, *instance_coords.values()]
        combined = np.concatenate(blocks, axis=0)
        transformed, _ = proj.tsne_reproject(combined, [])
        offset = len(corpus_coords)
        transformed_instances = {}
        for inst_id, coords in instance_coords.items():
            transformed_instances[inst_id] = transformed[offset : offset + len(coords)]
            offset += len(coords)
        cached = (transformed[: len(corpus_coords)], transformed_instances)
        if len(_TSNE_CACHE) >= 4:
            _TSNE_CACHE.pop(next(iter(_TSNE_CACHE)))
        _TSNE_CACHE[cache_key] = cached

    corpus_tsne, instances_tsne = cached
    projected_corpus = dict(corpus)
    projected_corpus["coords"] = corpus_tsne.tolist()
    projected_instances = dict(instances)
    for inst_id, coords in instances_tsne.items():
        result = _result_from_serialisable(instances[inst_id])
        result["coords_2d"] = coords
        projected_instances[inst_id] = _result_to_serialisable(result)
    return projected_corpus, projected_instances


def _attention_token_label(index: int, token: str, token_type: str) -> str:
    type_code = {"text": "T", "latent": "L", "visual": "V"}.get(token_type, "?")
    visible = token.replace("\n", "↵").replace(" ", "␠") or "∅"
    if len(visible) > 15:
        visible = visible[:12] + "…"
    return f"{index} {type_code}:{visible}"


def _build_token_attention_figure(
    result: dict,
    current_step: int,
    layer_index: int | None = None,
) -> go.Figure:
    token_strings = result.get("token_strings", [])
    token_types = result.get("token_types", [])
    layer_attention = result.get("layer_attn_weights", [])
    layer_count = len(layer_attention)
    selected_layer = (
        max(0, min(int(layer_index), layer_count - 1))
        if layer_index is not None and layer_count
        else layer_count - 1
    )
    attention_steps = layer_attention[selected_layer] if layer_count else []
    prompt_length = 0
    token_count = min(len(token_strings), len(token_types))

    fig = go.Figure()
    if token_count == 0 or not attention_steps:
        fig.add_annotation(
            text=(
                "Run a new inference to inspect attention across decoder layers."
                if not layer_attention
                else "No generated-token attention was captured for this layer."
            ),
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(color="#6b7e78", size=13),
        )
        fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=30, b=30))
        return fig

    normalized, raw, history_mass = token_attn.generated_attention_matrix(
        attention_steps, prompt_length, token_count
    )
    labels = [
        _attention_token_label(i, token_strings[i], token_types[i])
        for i in range(token_count)
    ]
    hover = np.empty((token_count, token_count), dtype=object)
    hover[:] = ""
    for query in range(token_count):
        for source in range(query):
            if np.isfinite(normalized[query, source]):
                hover[query, source] = (
                    f"query: {labels[query]}<br>source: {labels[source]}<br>"
                    f"relative history attention: {normalized[query, source]:.2%}<br>"
                    f"raw attention: {raw[query, source]:.6f}<br>"
                    f"total generated-history mass: {history_mass[query]:.4f}"
                )

    indices = np.arange(token_count)
    fig.add_trace(go.Heatmap(
        z=normalized,
        x=indices,
        y=indices,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0, "#f8fafc"],
            [0.15, "#fde68a"],
            [0.45, "#fb923c"],
            [1.0, "#9a3412"],
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(title="Relative<br>attention", tickformat=".0%", thickness=14),
        xgap=0.35,
        ygap=0.35,
    ))

    selected_step = max(0, min(int(current_step or 0), token_count - 1))
    fig.add_shape(
        type="rect",
        x0=-0.5,
        x1=token_count - 0.5,
        y0=selected_step - 0.5,
        y1=selected_step + 0.5,
        line=dict(color="#111827", width=2),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f4f7f5",
        margin=dict(l=145, r=30, t=45, b=120),
        xaxis=dict(
            title="Attended-to generated token (source)",
            tickmode="array", tickvals=indices, ticktext=labels,
            tickangle=-55, tickfont=dict(size=9),
            range=[-0.5, token_count - 0.5],
        ),
        yaxis=dict(
            title="Generated token producing attention (query)",
            tickmode="array", tickvals=indices, ticktext=labels,
            tickfont=dict(size=9), autorange="reversed",
        ),
        annotations=[dict(
            text=(
                f"Selected row {selected_step} · generated-history attention mass "
                f"{history_mass[selected_step]:.4f} · decoder layer {selected_layer}"
            ),
            x=0, y=1.08, xref="paper", yref="paper", showarrow=False,
            xanchor="left", font=dict(size=11, color="#6b7e78"),
        )],
    )
    return fig

def _build_umap_figure(
    corpus: dict | None,
    instances: dict,
    active_id: str | None,
    projection_mode: str = "umap",
    highlight_mode: str = "all",
    show_trace: bool = True,
    corpus_visible_ids: list | None = None,
    line_target: str = "__active__",
) -> tuple[go.Figure, dict]:
    """Build the UMAP base figure (no step-dependent elements).

    Returns (fig, trace_info). Step-specific overlays (star marker, trajectory
    line tail) are added separately by update_umap_step via Patch().
    """
    fig = go.Figure()

    if projection_mode == "tsne" and corpus and corpus.get("coords"):
        corpus, instances = _joint_tsne_projection(corpus, instances)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f4f7f5",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        dragmode="select",
        uirevision="umap",
    )

    # ── Unpack corpus arrays ─────────────────────────────────────────────────
    corpus_coords = corpus_genidx = corpus_tokidx = corpus_example_ids = None
    corpus_types = corpus_labels = None
    if corpus and corpus.get("coords"):
        corpus_coords      = np.array(corpus["coords"])
        corpus_types       = corpus["types"]
        corpus_labels      = corpus.get("labels")
        corpus_example_ids = corpus.get("example_ids")
        corpus_genidx      = corpus.get("gen_index")
        corpus_tokidx      = corpus.get("token_index")

    # Pre-compute sorted sequence for corpus line_target (step-trimming is
    # handled in update_umap_step; here we show ALL highlighted points).
    sorted_ex_mask: list[int] = []
    if line_target != "__active__" and corpus_coords is not None and corpus_example_ids is not None:
        def _seq_key(i: int) -> tuple:
            gi = int(corpus_genidx[i]) if corpus_genidx is not None else 0
            ti = int(corpus_tokidx[i]) if corpus_tokidx is not None else 0
            return (1, gi) if gi >= 0 else (0, ti)
        ex_all = [i for i, e in enumerate(corpus_example_ids) if str(e) == str(line_target)]
        sorted_ex_mask = sorted(ex_all, key=_seq_key)

    highlighted_corpus_set = set(sorted_ex_mask)

    # ── Corpus background ────────────────────────────────────────────────────
    if corpus_coords is not None and corpus_visible_ids is not None and len(corpus_visible_ids) > 0:
        visible_set = set(str(v) for v in corpus_visible_ids)
        have_meta   = corpus_example_ids is not None and corpus_genidx is not None

        for ttype, color in TOKEN_COLORS.items():
            hi_mask  = []
            bg_mask  = []
            for i, t in enumerate(corpus_types):
                if t != ttype:
                    continue
                eid = str(corpus_example_ids[i]) if corpus_example_ids is not None else None
                if eid not in visible_set:
                    continue
                if eid == str(line_target) and i in highlighted_corpus_set:
                    hi_mask.append(i)
                else:
                    bg_mask.append(i)

            def _make_corpus_trace(mask, size, opacity, name):
                tk = dict(
                    x=corpus_coords[mask, 0], y=corpus_coords[mask, 1],
                    mode="markers",
                    marker=dict(color=color, size=size, opacity=opacity,
                                line=dict(width=1 if size > 4 else 0, color="#111827")),
                    name=name,
                    legendgroup=f"corpus:{ttype}",
                    showlegend=(size <= 4),
                )
                if have_meta:
                    cd = []
                    for i in mask:
                        gi  = corpus_genidx[i]
                        seq = f"gen {gi}" if gi >= 0 else \
                              (f"input {corpus_tokidx[i]}" if corpus_tokidx is not None else "input")
                        lab = (corpus_labels[i] if corpus_labels else "").strip() or "·"
                        cd.append([_passage(corpus_example_ids[i]), seq, lab])
                    tk["customdata"] = cd
                    tk["hovertemplate"] = (
                        f"<b>Passage %{{customdata[0]}}</b> · {ttype}<br>"
                        "seq: %{customdata[1]}<br>"
                        "token: %{customdata[2]}<br>"
                        "(%{x:.2f}, %{y:.2f})<extra></extra>"
                    )
                return tk

            if bg_mask:
                fig.add_trace(go.Scattergl(**_make_corpus_trace(
                    bg_mask, size=3, opacity=0.25, name=f"corpus:{ttype}")))
            if hi_mask:
                fig.add_trace(go.Scattergl(**_make_corpus_trace(
                    hi_mask, size=8, opacity=0.85, name=f"corpus:{ttype}:highlight")))

    # ── Instance traces ──────────────────────────────────────────────────────
    # Uniform appearance — no step-specific star/size. update_umap_step patches these.
    active_type_traces: dict[str, dict] = {}
    active_model = None
    if active_id and active_id in instances:
        _ar = _result_from_serialisable(instances[active_id])
        active_model = _ar.get("model_name")
    if show_trace:
        for idx, (inst_id, inst_data) in enumerate(instances.items()):
            result = _result_from_serialisable(inst_data)
            # Instances from a different model live in a different UMAP coordinate space
            # and cannot be meaningfully overlaid on this one.
            inst_model = result.get("model_name")
            if active_model and inst_model and inst_model != active_model:
                continue
            coords_2d = result.get("coords_2d")
            if coords_2d is None or len(coords_2d) == 0:
                continue
            coords_2d     = np.array(coords_2d)
            token_types   = result.get("token_types", [])
            token_strings = result.get("token_strings", [])
            is_active     = inst_id == active_id

            for ttype, color in TOKEN_COLORS.items():
                mask = [i for i, t in enumerate(token_types) if t == ttype]
                if not mask:
                    continue

                if is_active:
                    sizes     = [10] * len(mask)
                    opacities = [0.95] * len(mask)
                else:
                    sizes     = [5] * len(mask)
                    opacities = [0.2] * len(mask)

                trace_idx = len(fig.data)
                if is_active:
                    active_type_traces[ttype] = {"idx": trace_idx, "mask": mask}

                customdata = [
                    [inst_id, i, (token_strings[i] if i < len(token_strings) else "")]
                    for i in mask
                ]
                fig.add_trace(go.Scattergl(
                    x=coords_2d[mask, 0], y=coords_2d[mask, 1],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=sizes,
                        symbol="circle",
                        opacity=opacities,
                        line=dict(width=2 if is_active else 1, color="#111827"),
                    ),
                    name=f"{inst_id}:{ttype}",
                    legendgroup=inst_id,
                    showlegend=True,
                    customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · " + ttype + "<br>"
                        "step: %{customdata[1]}<br>"
                        "token: %{customdata[2]}<br>"
                        "(%{x:.2f}, %{y:.2f})<extra></extra>"
                    ),
                ))

    # ── Step-overlay placeholder traces (filled by update_umap_step) ─────────
    traj_start_idx = len(fig.data)
    for ttype, color in TOKEN_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode="lines",
            line=dict(color=color, dash="solid", width=2),
            name=f"traj:{ttype}",
            legendgroup="trajectory",
            showlegend=False,
        ))

    # Corpus trajectory: pre-filled with full path; step callback trims when needed
    corpus_traj_idx = len(fig.data)
    if sorted_ex_mask and corpus_coords is not None and len(sorted_ex_mask) >= 2:
        lc = corpus_coords[sorted_ex_mask]
        fig.add_trace(go.Scatter(
            x=lc[:, 0].tolist(), y=lc[:, 1].tolist(),
            mode="lines",
            line=dict(color="#8b5cf6", dash="solid", width=2),
            name=f"line:passage {_passage(line_target)}",
            showlegend=False,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode="lines",
            line=dict(color="#8b5cf6", dash="solid", width=2),
            name="traj:corpus",
            showlegend=False,
        ))

    # Current-step star marker (position filled by update_umap_step)
    star_idx = len(fig.data)
    fig.add_trace(go.Scattergl(
        x=[], y=[],
        mode="markers",
        marker=dict(symbol="star", size=16, color="white",
                    line=dict(width=2.5, color="#111827")),
        name="current-step",
        showlegend=False,
        hoverinfo="skip",
    ))

    traj_indices = {ttype: traj_start_idx + i for i, ttype in enumerate(TOKEN_COLORS)}
    trace_info = {
        "active_id":      active_id,
        "type_traces":    active_type_traces,
        "traj_indices":   traj_indices,
        "corpus_traj_idx": corpus_traj_idx,
        "star_idx":       star_idx,
        "sorted_ex_mask": sorted_ex_mask,
        "line_target":    line_target,
    }
    return fig, trace_info


def _build_step_patch(
    step: int,
    trace_info: dict,
    instances: dict,
    active_id: str,
    highlight_mode: str,
    corpus: dict | None = None,
) -> Patch:
    """Return a Patch() that updates only the step-sensitive UMAP overlay traces."""
    patched = Patch()

    result    = _result_from_serialisable(instances[active_id])
    coords_2d = result.get("coords_2d")
    token_types = result.get("token_types", [])

    if coords_2d is None:
        return patched

    coords_2d = np.array(coords_2d)
    step      = int(step or 0)
    inst_idx  = list(instances.keys()).index(active_id)
    dash      = INSTANCE_LINE_STYLES[inst_idx % len(INSTANCE_LINE_STYLES)]

    # Instance trajectory lines
    for ttype, traj_fig_idx in trace_info.get("traj_indices", {}).items():
        if highlight_mode == "step":
            traj_mask = sorted([i for i, t in enumerate(token_types) if t == ttype and i <= step])
        else:
            traj_mask = sorted([i for i, t in enumerate(token_types) if t == ttype])

        if len(traj_mask) >= 2:
            lc = coords_2d[traj_mask]
            patched["data"][traj_fig_idx]["x"] = lc[:, 0].tolist()
            patched["data"][traj_fig_idx]["y"] = lc[:, 1].tolist()
            patched["data"][traj_fig_idx]["line"]["dash"] = dash
        else:
            patched["data"][traj_fig_idx]["x"] = []
            patched["data"][traj_fig_idx]["y"] = []

    # Corpus trajectory: trim to current step when highlight_mode == "step"
    corpus_traj_idx = trace_info.get("corpus_traj_idx")
    sorted_ex_mask  = trace_info.get("sorted_ex_mask", [])
    line_target     = trace_info.get("line_target", "__active__")
    if (corpus_traj_idx is not None and line_target != "__active__"
            and sorted_ex_mask and highlight_mode == "step"
            and corpus and corpus.get("coords")):
        corpus_coords = np.array(corpus["coords"])
        traj = sorted_ex_mask[: step + 1]
        if len(traj) >= 2:
            lc = corpus_coords[traj]
            patched["data"][corpus_traj_idx]["x"] = lc[:, 0].tolist()
            patched["data"][corpus_traj_idx]["y"] = lc[:, 1].tolist()
        else:
            patched["data"][corpus_traj_idx]["x"] = []
            patched["data"][corpus_traj_idx]["y"] = []

    # Star marker at current step
    star_idx = trace_info.get("star_idx")
    if star_idx is not None:
        if step < len(token_types):
            patched["data"][star_idx]["x"] = [float(coords_2d[step, 0])]
            patched["data"][star_idx]["y"] = [float(coords_2d[step, 1])]
            patched["data"][star_idx]["marker"]["color"] = TOKEN_COLORS.get(token_types[step], "#ffffff")
        else:
            patched["data"][star_idx]["x"] = []
            patched["data"][star_idx]["y"] = []

    # Dim/reveal tokens for highlight_mode == "step"
    if highlight_mode == "step":
        highlighted = set(range(step + 1))
        for ttype, tinfo in trace_info.get("type_traces", {}).items():
            tidx = tinfo["idx"]
            mask = tinfo["mask"]
            patched["data"][tidx]["marker"]["opacity"] = [
                0.95 if m in highlighted else 0.4 for m in mask
            ]
            patched["data"][tidx]["marker"]["size"] = [
                10 if m in highlighted else 5 for m in mask
            ]

    return patched


# ── Callbacks ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("uploaded-image-preview", "src"),
        Output("uploaded-image-preview", "style"),  
        Output("store-current-image-b64", "data"),
        Input("image-upload", "contents"),
        prevent_initial_call=True,
    )
    def preview_image(contents):
        if not contents:
            raise PreventUpdate
        img = _b64_to_pil(contents)
        print(f"[image-upload] Uploaded image size: {img.width}x{img.height} px")
        return contents, {"display": "block"}, contents

    @app.callback(
        Output("question-input", "value"),
        Output("uploaded-image-preview", "src", allow_duplicate=True),
        Output("store-current-image-b64", "data", allow_duplicate=True),
        Input("example-selector", "value"),
        prevent_initial_call=True,
    )
    def load_example(example_id):
        if not example_id:
            raise PreventUpdate
        examples = dl.list_examples()
        ex = next((e for e in examples if e["id"] == example_id), None)
        if not ex:
            raise PreventUpdate
        question = ex.get("question", "")
        img_path = dl.load_example_image(example_id)
        try:
            img = Image.open(img_path).convert("RGB")
            b64 = "data:image/jpeg;base64," + base64.b64encode(
                open(img_path, "rb").read()
            ).decode()
            return question, b64, b64
        except Exception:
            return question, no_update, no_update


    @app.callback(
        Output("model-load-status",   "children"),
        Output("model-load-progress", "value"),
        Output("model-load-progress", "style"),
        Input("load-model-btn", "n_clicks"),
        State("model-selector", "value"),
        prevent_initial_call=True,
    )
    def load_model_on_click(n_clicks, model_name):
        if not model_name:
            return "No model selected.", 0, {"display": "none"}
        print(f"[load-model] Loading model: {model_name}")
        current = inf.get_loaded_model()
        if current and current != model_name:
            inf.unload_model(current)
        if model_name in inf._model_cache:
            return f"✓ {model_name} ready.", 100, {"display": "none"}
        try:
            inf._get_model_and_processor(model_name)
            return f"✓ {model_name} ready.", 100, {"display": "none"}
        except Exception as e:
            return f"✗ Failed: {e}", 0, {"display": "none"}

    @app.callback(
        Output("store-corpus-embeddings", "data"),
        Input("model-selector", "value"),
        Input("store-active-instance", "data"),
        State("store-instances", "data"),
    )
    def reload_corpus(model_name, active_id, instances):
        # Always use the active instance's model when one exists — the corpus
        # must match the coordinate space of the displayed instance.
        if active_id and instances and active_id in instances:
            inst_model = (instances[active_id] or {}).get("model_name")
            if inst_model:
                model_name = inst_model
        if not model_name or not dl.corpus_embeddings_exist(model_name):
            return {}
        try:
            corpus = dl.load_corpus_embeddings(model_name)
            store = {
                "coords": corpus["coords"].tolist(),
                "types":  corpus["types"],
            }
            for k in ("labels", "example_ids", "gen_index", "token_index"):
                if k in corpus:
                    store[k] = corpus[k]
            return store
        except Exception:
            return {}

    @app.callback(
        Output("store-instances",       "data"),
        Output("store-active-instance", "data"),
        Output("step-slider",           "max"),
        Output("step-slider",           "value"),
        Output("step-slider",           "marks"),
        Output("reasoning-trace",       "children"),
        Output("instance-list",         "children"),
        Input("run-btn", "n_clicks"),
        State("store-current-image-b64",   "data"),
        State("question-input",            "value"),
        State("model-selector",            "value"),
        State("store-instances",           "data"),
        State("store-corpus-embeddings",   "data"),
        prevent_initial_call=True,
    )
    def run_inference(n_clicks, img_b64, question, model_name, existing_instances, corpus):
        if not model_name or not img_b64 or not question:
            raise PreventUpdate
        image = _b64_to_pil(img_b64)
        return _execute_inference_and_update(
            image, question, model_name, existing_instances, corpus or {},
        )

    @app.callback(
        Output("umap-line-target", "options"),
        Input("store-corpus-embeddings", "data"),
    )
    def update_line_target_options(corpus):
        opts = [{"label": "Current trace", "value": "__active__"}]
        if corpus and corpus.get("example_ids"):
            seen: dict = {}
            for eid in corpus["example_ids"]:
                key = str(eid)
                if key not in seen:
                    seen[key] = True
                    opts.append({"label": f"Passage {_passage(eid)}", "value": key})
        return opts

    @app.callback(
        Output("corpus-passage-dropdown", "options"),
        Output("corpus-passage-dropdown", "value"),
        Input("store-corpus-embeddings",  "data"),
    )
    def init_corpus_passages(corpus):
        opts = []
        if corpus and corpus.get("example_ids"):
            seen: dict = {}
            for eid in corpus["example_ids"]:
                key = str(eid)
                if key not in seen:
                    seen[key] = True
                    opts.append({"label": f"Passage {_passage(eid)}", "value": key})
        return opts, [o["value"] for o in opts]

    @app.callback(
        Output("corpus-passage-dropdown", "value", allow_duplicate=True),
        Input("btn-corpus-all",  "n_clicks"),
        Input("btn-corpus-none", "n_clicks"),
        State("corpus-passage-dropdown", "options"),
        prevent_initial_call=True,
    )
    def handle_corpus_buttons(n_all, n_none, options):
        if ctx.triggered_id == "btn-corpus-all":
            return [o["value"] for o in (options or [])]
        return []

    @app.callback(
        Output("corpus-selection-count",  "children"),
        Input("corpus-passage-dropdown",  "value"),
        State("corpus-passage-dropdown",  "options"),
    )
    def update_corpus_count(selected, options):
        return f"{len(selected or [])} of {len(options or [])}"

    @app.callback(
        Output("heatmap-image",       "src"),
        Output("param-display",       "children"),
        Output("uncertainty-display", "children"),
        Input("step-slider",           "value"),
        Input("store-active-instance", "data"),
        State("store-instances",           "data"),
        State("store-active-projection",   "data"),
        State("store-current-image-b64",   "data"),
        prevent_initial_call=True,
    )
    def update_views(step, active_id, instances, projection_mode, img_b64):
        if not active_id or active_id not in instances:
            raise PreventUpdate

        result = _result_from_serialisable(instances[active_id])
        model_name = result.get("model_name", "unknown")
        attn_list = result.get("attn_weights", [])
        grid_hw   = tuple(result.get("image_grid_hw", (16, 16)))

        attn_at_step = None
        if attn_list and step < len(attn_list) and attn_list[step] is not None:
            attn_at_step = np.array(attn_list[step])

        if img_b64:
            image = _b64_to_pil(img_b64)
            mask_region = result.get("mask_region")
            if mask_region:
                image = _apply_image_mask(image, mask_region)
            image_token_range = result.get("image_token_range")
            heatmap_src = hm.attn_to_heatmap_overlay(image, attn_at_step, grid_hw,
                                                      image_token_range=image_token_range)
        else:
            heatmap_src = ""

        token_str = ""
        if step < len(result.get("token_strings", [])):
            token_str = result["token_strings"][step]
        token_type = ""
        if step < len(result.get("token_types", [])):
            token_type = result["token_types"][step]

        nearest_text_list = result.get("nearest_text", [])
        nearest_at_step = (
            nearest_text_list[step]
            if nearest_text_list and step < len(nearest_text_list)
            else None
        )

        param_children = [
            _stat_row("Step",       str(step)),
            _stat_row("Token",      repr(token_str)),
            _stat_row("Type",       token_type),
            _stat_row("Model",      model_name),
            _stat_row("Instance",   active_id),
            _stat_row("Projection", projection_mode or "umap"),
        ]
        if nearest_at_step:
            label = nearest_at_step["label"].strip() or nearest_at_step["label"]
            param_children.insert(3, _stat_row("Nearest text", f'"{label}" (d={nearest_at_step["distance"]:.2f})'))
        if result.get("projection_error"):
            param_children.append(_stat_row("Projection error", result["projection_error"]))

        uncertainty_children = _build_uncertainty_display(result, step)
        print(f"[heatmap] Attention heatmap rendered for step={step}, instance={active_id}")

        return heatmap_src, param_children, uncertainty_children

    @app.callback(
        Output("umap-graph",      "figure"),
        Output("store-umap-base", "data"),
        Input("store-active-instance",   "data"),
        Input("store-active-projection", "data"),
        Input("highlight-mode",          "value"),
        Input("umap-show-trace",         "value"),
        Input("corpus-passage-dropdown", "value"),
        Input("umap-line-target",        "value"),
        State("store-instances",         "data"),
        State("store-corpus-embeddings", "data"),
        prevent_initial_call=True,
    )
    def update_umap_base(active_id, projection_mode, highlight_mode, show_trace_val,
                         corpus_visible_ids, line_target, instances, corpus):
        if not active_id or active_id not in instances:
            raise PreventUpdate

        fig, trace_info = _build_umap_figure(
            corpus, instances, active_id, projection_mode,
            highlight_mode=highlight_mode or "all",
            show_trace="trace" in (show_trace_val or []),
            corpus_visible_ids=corpus_visible_ids or [],
            line_target=line_target or "__active__",
        )
        print(f"[umap] UMAP graph rendered: mode={projection_mode}, instance={active_id}, traces={len(fig.data)}")
        return fig, trace_info

    @app.callback(
        Output("umap-graph", "figure", allow_duplicate=True),
        Input("step-slider",     "value"),
        Input("store-umap-base", "data"),
        State("store-instances",          "data"),
        State("store-active-instance",    "data"),
        State("highlight-mode",           "value"),
        State("store-corpus-embeddings",  "data"),
        prevent_initial_call=True,
    )
    def update_umap_step(step, trace_info, instances, active_id, highlight_mode, corpus):
        if not trace_info or not active_id or not instances:
            raise PreventUpdate
        if active_id not in instances:
            raise PreventUpdate
        if trace_info.get("active_id") != active_id:
            raise PreventUpdate

        return _build_step_patch(
            step or 0, trace_info, instances, active_id,
            highlight_mode or "all", corpus,
        )

    @app.callback(
        Output("token-attention-matrix", "figure"),
        Input("step-slider", "value"),
        Input("attention-layer-slider", "value"),
        Input("store-active-instance", "data"),
        Input("store-instances", "data"),
    )
    def update_token_attention(step, layer_index, active_id, instances):
        if not active_id or active_id not in instances:
            return _build_token_attention_figure({}, step or 0, layer_index)
        result = _result_from_serialisable(instances[active_id])
        fig = _build_token_attention_figure(result, step or 0, layer_index)
        print(f"[token-attention] Token attention matrix rendered: step={step}, layer={layer_index}, instance={active_id}")
        return fig

    @app.callback(
        Output("attention-layer-slider", "max"),
        Output("attention-layer-slider", "value"),
        Output("attention-layer-slider", "marks"),
        Input("store-active-instance", "data"),
        State("store-instances", "data"),
    )
    def update_attention_layer_slider(active_id, instances):
        result = instances.get(active_id, {}) if active_id else {}
        layer_count = int(result.get("attention_layer_count", 0))
        maximum = max(0, layer_count - 1)
        if not layer_count:
            return 0, 0, {0: "rerun"}
        stride = max(1, maximum // 6)
        marks = {i: str(i) for i in range(0, maximum + 1, stride)}
        marks[maximum] = str(maximum)
        return maximum, maximum, marks

    @app.callback(
        Output("attention-layer-slider", "value", allow_duplicate=True),
        Output("attention-layer-interval", "disabled"),
        Output("attention-layer-playback", "children"),
        Output("attention-layer-playback", "title"),
        Output("attention-layer-playback", "aria-label"),
        Input("attention-layer-playback", "n_clicks"),
        Input("attention-layer-interval", "n_intervals"),
        Input("store-active-instance", "data"),
        State("attention-layer-slider", "value"),
        State("attention-layer-slider", "max"),
        State("attention-layer-interval", "disabled"),
        prevent_initial_call=True,
    )
    def control_attention_layer_playback(
        n_clicks, n_intervals, active_id, current_layer, maximum_layer, disabled
    ):
        play_icon = html.Span(className="playback-icon playback-icon--play")
        pause_icon = html.Span(className="playback-icon playback-icon--pause")

        if ctx.triggered_id == "store-active-instance":
            return no_update, True, play_icon, "Play decoder layers", "Play decoder layers"

        maximum = int(maximum_layer or 0)
        if ctx.triggered_id == "attention-layer-playback":
            if not disabled:
                return no_update, True, play_icon, "Play decoder layers", "Play decoder layers"
            if maximum <= 0:
                return 0, True, play_icon, "Play decoder layers", "Play decoder layers"
            return 0, False, pause_icon, "Pause decoder layers", "Pause decoder layers"

        if ctx.triggered_id == "attention-layer-interval" and not disabled:
            next_layer = int(current_layer or 0) + 1
            if next_layer >= maximum:
                return maximum, True, play_icon, "Play decoder layers", "Play decoder layers"
            return next_layer, False, pause_icon, "Pause decoder layers", "Pause decoder layers"

        raise PreventUpdate

    @app.callback(
        Output("token-context-panel", "children"),
        Input("step-slider",            "value"),
        Input("store-active-instance",  "data"),
        State("store-instances",        "data"),
        State("store-corpus-embeddings","data"),
        prevent_initial_call=True,
    )
    def update_token_context(step, active_id, instances, corpus):
        if not active_id or active_id not in instances:
            raise PreventUpdate

        result = _result_from_serialisable(instances[active_id])
        coords_2d     = result.get("coords_2d")
        token_strings = result.get("token_strings", [])
        token_types   = result.get("token_types", [])

        step = int(step or 0)
        current_token = token_strings[step] if step < len(token_strings) else ""
        current_type  = token_types[step]   if step < len(token_types)   else "text"

        neighbors = []
        if (coords_2d is not None and corpus
                and corpus.get("coords") and corpus.get("labels")):
            query_coord      = np.array(coords_2d)[step]
            corpus_coords_arr = np.array(corpus["coords"], dtype=np.float32)
            neighbors = proj.find_k_nearest_text_neighbors(
                query_coord,
                corpus_coords_arr,
                corpus["types"],
                corpus["labels"],
                corpus_example_ids=corpus.get("example_ids"),
                k=10,
            )

        print(f"[token-context] Nearest corpus neighbours rendered: "
              f"step={step}, instance={active_id}, found={len(neighbors)}")
        return _build_token_context_panel(current_token, current_type, neighbors)

    @app.callback(
        Output("stats-display", "children"),
        Input("umap-graph",     "selectedData"),
        Input("highlight-mode", "value"),
        State("store-active-instance", "data"),
        State("store-instances",       "data"),
        State("step-slider",           "value"),
        prevent_initial_call=True,
    )
    def update_stats(selected_data, highlight_mode, active_id, instances, step):
        if not selected_data or not selected_data.get("points"):
            raise PreventUpdate
        if not active_id or active_id not in instances:
            raise PreventUpdate

        result      = _result_from_serialisable(instances[active_id])
        token_types = result.get("token_types", [])
        coords_2d   = result.get("coords_2d")

        selected_indices = [
            int(pt["customdata"][1])
            for pt in selected_data["points"]
            if isinstance(pt.get("customdata"), (list, tuple))
            and len(pt["customdata"]) >= 2
            and pt["customdata"][0] == active_id
        ]
        if not selected_indices:
            raise PreventUpdate

        if coords_2d is not None:
            n = len(coords_2d)
            selected_indices = [i for i in selected_indices if i < n]

        selected_types = [token_types[i] for i in selected_indices if i < len(token_types)]

        attn_list    = result.get("attn_weights", [])
        attn_at_step = None
        if attn_list and step < len(attn_list) and attn_list[step] is not None:
            attn_at_step = np.array(attn_list[step])

        stats = proj.compute_selection_stats(selected_types, attn_at_step, [])
        return [_stat_row(k, str(v)) for k, v in stats.items()]

    @app.callback(
        Output("store-active-instance", "data", allow_duplicate=True),
        Input({"type": "instance-badge", "index": ALL}, "n_clicks"),
        State("store-instances", "data"),
        prevent_initial_call=True,
    )
    def switch_instance(n_clicks_list, instances):
        triggered = ctx.triggered_id
        if triggered is None:
            raise PreventUpdate
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            raise PreventUpdate
        return triggered["index"]
        
    @app.callback(
        Output("instance-list",  "children", allow_duplicate=True),
        Output("step-slider",    "max",      allow_duplicate=True),
        Output("step-slider",    "value",    allow_duplicate=True),
        Output("step-slider",    "marks",    allow_duplicate=True),
        Input("store-active-instance", "data"),
        State("store-instances",       "data"),
        prevent_initial_call=True,
    )
    def update_on_instance_switch(active_id, instances):
        if not active_id or active_id not in instances:
            raise PreventUpdate
        result = instances[active_id]
        n_steps = len(result.get("token_strings", []))
        marks = {i: str(i) for i in range(0, n_steps, max(1, n_steps // 10))}
        badges = _build_instance_list(instances, active_id)
        return badges, max(0, n_steps - 1), 0, marks


    # ── Mask graph: render uploaded image so user can draw a selection ────────
    @app.callback(
        Output("comparison-instance-selector", "options"),
        Output("comparison-instance-selector", "value"),
        Input("store-instances", "data"),
        Input("store-active-instance", "data"),
        State("comparison-instance-selector", "value"),
    )
    def update_comparison_instance_options(instances, active_id, selected_id):
        options = [{"label": key, "value": key} for key in (instances or {}) if key != active_id]
        valid = {option["value"] for option in options}
        if selected_id not in valid:
            selected_id = options[0]["value"] if options else None
        return options, selected_id

    @app.callback(
        Output("token-comparison-display", "children"),
        Input("step-slider", "value"),
        Input("store-active-instance", "data"),
        Input("comparison-instance-selector", "value"),
        Input("store-instances", "data"),
        State("model-selector", "value"),
    )
    def update_token_comparison(step, active_id, reference_id, instances, model_name):
        if not active_id or not reference_id or not instances:
            return html.Div("Run an intervention to compare generated tokens.", className="comparison-placeholder")
        if active_id not in instances or reference_id not in instances:
            raise PreventUpdate
        current, reference = instances[active_id], instances[reference_id]
        alignment = token_alignment.align_token_sequences(
            current.get("token_strings", []), reference.get("token_strings", []),
            current.get("token_types", []), reference.get("token_types", []))

        def load_activations(instance_id):
            path = dl.PRECOMPUTED_DIR / "online_cache" / str(model_name) / f"{instance_id}.npz"
            try:
                with np.load(path, allow_pickle=False) as cached:
                    return np.asarray(cached["activations"], dtype=np.float32)
            except (OSError, KeyError, ValueError):
                return None

        current_activations = load_activations(active_id)
        reference_activations = load_activations(reference_id)
        for row in alignment:
            ci, ri = row["current_index"], row["reference_index"]
            row["representation_change"] = None
            if (current_activations is not None and reference_activations is not None
                    and ci is not None and ri is not None
                    and ci < len(current_activations) and ri < len(reference_activations)):
                row["representation_change"] = representation_comparison.cosine_change(
                    current_activations[ci], reference_activations[ri])
        representation_changes = [row["representation_change"] for row in alignment
                                  if row["representation_change"] is not None]
        step = int(step or 0)
        selected = next((row for row in alignment if row["current_index"] == step), None)
        counts = {op: sum(row["operation"] == op for row in alignment)
                  for op in ("match", "replace", "insert", "delete")}
        if selected:
            ref_label = (f"step {selected['reference_index']}: {selected['reference_token']!r}"
                         if selected["reference_index"] is not None else "no corresponding token")
            current_summary = html.Div([
                html.Strong(f"Current step {step}: {selected['current_token']!r}"),
                html.Span("->", className="comparison-arrow"), html.Strong(f"{reference_id} {ref_label}"),
                html.Span(selected["operation"].upper(),
                          className=f"comparison-status comparison-status--{selected['operation']}"),
                html.Span(
                    f"Hidden-state change: {selected['representation_change']:.3f}"
                    if selected["representation_change"] is not None else "Hidden-state change unavailable",
                    className="comparison-hidden-change"),
            ], className="comparison-current-summary")
        else:
            current_summary = html.Div("Current step could not be aligned.")
        summary = html.Div([
            html.Span(f"Current: {active_id}", className="comparison-instance-name"),
            html.Span(f"Reference: {reference_id}", className="comparison-instance-name"),
            *[html.Span(f"{op.title()} {counts[op]}", className=f"comparison-count comparison-count--{op}")
              for op in ("match", "replace", "insert", "delete")],
            html.Span(f"Mean hidden-state change {np.mean(representation_changes):.3f}",
                      className="comparison-count comparison-count--representation")
            if representation_changes else html.Span("Hidden states unavailable", className="comparison-count"),
        ], className="comparison-summary")
        rows = []
        for row in alignment:
            ci = "-" if row["current_index"] is None else str(row["current_index"])
            ri = "-" if row["reference_index"] is None else str(row["reference_index"])
            ct = "[none]" if row["current_token"] is None else repr(row["current_token"])
            rt = "[none]" if row["reference_token"] is None else repr(row["reference_token"])
            change = row["representation_change"]
            status_label = row["operation"] if change is None else f"{row['operation']} | d={change:.3f}"
            rows.append(html.Div([
                html.Span(ci, className="comparison-step"), html.Span(ct, className="comparison-token"),
                html.Span("->", className="comparison-arrow"),
                html.Span(ri, className="comparison-step"), html.Span(rt, className="comparison-token"),
                html.Span(status_label, title="1 - cosine similarity of the original hidden-state vectors",
                          className=f"comparison-status comparison-status--{row['operation']}"),
            ], className="comparison-row comparison-row--active"
               if row["current_index"] == step else "comparison-row"))
        return html.Div([summary, current_summary, html.Div(rows, className="comparison-list")])


    @app.callback(
        Output("mask-image-graph", "figure"),
        Input("store-current-image-b64", "data"),
        Input("store-mask-region",       "data"),
        prevent_initial_call=True,
    )
    def update_mask_image(img_b64, mask_region):
        if not img_b64:
            return _BLANK_MASK_FIGURE
        img = _b64_to_pil(img_b64)
        W, H = img.size
        fig = go.Figure()
        fig.add_trace(go.Image(z=np.array(img)))
        if mask_region:
            x0 = mask_region["x0_norm"] * W
            y0 = mask_region["y0_norm"] * H
            x1 = mask_region["x1_norm"] * W
            y1 = mask_region["y1_norm"] * H
            fig.add_shape(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="#ef4444", width=2),
                fillcolor="rgba(239,68,68,0.2)",
            )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#1e293b",
            plot_bgcolor="#1e293b",
            dragmode="select",
            xaxis=dict(range=[-0.5, W - 0.5], showgrid=False,
                       showticklabels=False, zeroline=False, visible=False),
            yaxis=dict(range=[H - 0.5, -0.5], showgrid=False,
                       showticklabels=False, zeroline=False, visible=False,
                       scaleanchor="x"),
        )
        return fig

    @app.callback(
        Output("store-mask-region", "data"),
        Input("mask-image-graph", "selectedData"),
        State("mask-image-graph", "figure"),
        prevent_initial_call=True,
    )
    def capture_mask_region(selected_data, figure):
        if not selected_data or not selected_data.get("range"):
            raise PreventUpdate  # figure re-render resets selectedData; don't clear stored mask
        rng = selected_data["range"]
        x_sel = rng.get("x", [0, 1])
        y_sel = rng.get("y", [0, 1])
        layout = (figure or {}).get("layout", {})
        x_ax_range = layout.get("xaxis", {}).get("range", [-0.5, 1])
        y_ax_range = layout.get("yaxis", {}).get("range", [1, -0.5])
        W = x_ax_range[1] + 0.5
        H = y_ax_range[0] + 0.5   # y is reversed: range[0] = H − 0.5
        if W <= 0 or H <= 0:
            raise PreventUpdate
        return {
            "x0_norm": max(0.0, min(x_sel) / W),
            "x1_norm": min(1.0, max(x_sel) / W),
            "y0_norm": max(0.0, min(y_sel) / H),
            "y1_norm": min(1.0, max(y_sel) / H),
        }

    @app.callback(
        Output("store-mask-region", "data", allow_duplicate=True),
        Input("store-current-image-b64", "data"),
        prevent_initial_call=True,
    )
    def clear_mask_on_new_image(_):
        return None

    # ── Causal intervention run ───────────────────────────────────────────────
    @app.callback(
        Output("intervention-reasoning-trace", "children"),
        Output("store-instances",             "data",  allow_duplicate=True),
        Output("store-active-instance",       "data",  allow_duplicate=True),
        Output("step-slider",                 "max",   allow_duplicate=True),
        Output("step-slider",                 "value", allow_duplicate=True),
        Output("step-slider",                 "marks", allow_duplicate=True),
        Output("instance-list",               "children", allow_duplicate=True),
        Input("run-intervention-btn", "n_clicks"),
        State("store-current-image-b64",        "data"),
        State("question-input",                 "value"),
        State("intervention-question-input",    "value"),
        State("model-selector",                 "value"),
        State("store-instances",                "data"),
        State("store-corpus-embeddings",        "data"),
        State("store-mask-region",              "data"),
        prevent_initial_call=True,
    )
    def run_intervention(
        n_clicks, img_b64, question, intervention_question,
        model_name, existing_instances, corpus, mask_region,
    ):
        if not model_name or not img_b64:
            raise PreventUpdate
        effective_question = (intervention_question or "").strip() or (question or "").strip()
        if not effective_question:
            raise PreventUpdate
        image = _b64_to_pil(img_b64)
        updated, inst_id, max_step, val, marks, trace_children, instance_children = \
            _execute_inference_and_update(
                image, effective_question, model_name,
                existing_instances, corpus or {},
                instance_prefix="intervention",
                mask_region=mask_region,
            )
        return trace_children, updated, inst_id, max_step, val, marks, instance_children

    @app.callback(
        Output("eval-display", "children"),
        Input("store-instances", "data"),
    )
    def update_eval(instances):
        if not instances:
            return "No instances yet."
        rows = []
        for inst_id, data in instances.items():
            token_strings = data.get("token_strings", [])
            token_types   = data.get("token_types", [])
            text = "".join(
                tok for tok, ttype in zip(token_strings, token_types)
                if ttype != "latent"
            ).strip()
            snippet = text[:200] + "…" if len(text) > 200 else text
            is_intervention = inst_id.startswith("intervention_")
            rows.append(html.Div([
                html.Div(
                    inst_id,
                    style={
                        "fontSize": "0.68rem", "fontWeight": "700",
                        "color": "#7c3aed" if is_intervention else "#1a2e28",
                        "marginBottom": "3px",
                    },
                ),
                html.Div(
                    snippet or "–",
                    style={
                        "fontSize": "0.72rem", "color": "#374151",
                        "lineHeight": "1.5", "marginBottom": "10px",
                        "wordBreak": "break-word",
                    },
                ),
            ]))
        return rows
    

    # ── Projection toggle (UMAP ↔ t-SNE) ─────────────────────────────────
    @app.callback(
        Output("store-active-projection",   "data"),
        Output("btn-umap",                  "color"),
        Output("btn-umap",                  "outline"),
        Output("btn-tsne",                  "color"),
        Output("btn-tsne",                  "outline"),
        Output("projection-panel-title",    "children"),
        Output("projection-panel-subtitle", "children"),
        Input("btn-umap",  "n_clicks"),
        Input("btn-tsne",  "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_projection(n_umap, n_tsne):
        triggered = ctx.triggered_id
        if triggered == "btn-umap":
            return (
                "umap",
                "primary", False,
                "secondary", True,
                "UMAP Projection",
                "Token embedding space — draw a box to zoom into a region",
            )
        else:
            return (
                "tsne",
                "secondary", True,
                "primary", False,
                "Joint t-SNE Re-layout",
                "Shared t-SNE of corpus + live UMAP coordinates",
            )


# ── UI helpers ───────────────────────────────────────────────────────────────

def _build_token_context_panel(current_token: str, current_type: str, neighbors: list) -> html.Div:
    """Build the token-context neighbour panel for the current step."""
    token_display = repr(current_token) if current_token else "–"
    if len(token_display) > 40:
        token_display = token_display[:38] + "…'"

    header = html.Div([
        html.Span("Current token: ", className="stat-label"),
        html.Span(token_display, className=f"trace-token trace-token--{current_type}"),
    ], className="stat-row", style={"marginBottom": "10px"})

    if not neighbors:
        return html.Div([
            header,
            html.Div(
                "No corpus data — run the offline pipeline to enable neighbour lookup.",
                style={"color": "#9ca3af", "fontStyle": "italic", "fontSize": "0.78rem"},
            ),
        ])

    max_dist = max((n["distance"] for n in neighbors), default=1) or 1

    rows = []
    for rank, neighbor in enumerate(neighbors, 1):
        label = neighbor["label"].strip() or neighbor["label"] or "·"
        if len(label) > 35:
            label = label[:33] + "…"
        dist        = neighbor["distance"]
        example_id  = neighbor.get("example_id")
        passage_num = _passage(example_id) if example_id is not None else "?"
        bar_pct     = max(4, int((1.0 - dist / max_dist) * 100))

        rows.append(html.Div([
            html.Span(str(rank), className="neighbor-rank"),
            html.Div([
                html.Div([
                    html.Span(f'"{label}"', className="neighbor-label"),
                    html.Span(f"P{passage_num}", className="neighbor-passage"),
                    html.Span(f"d={dist:.2f}", className="neighbor-dist"),
                ], className="neighbor-meta"),
                html.Div(
                    html.Div(style={
                        "width": f"{bar_pct}%",
                        "height": "3px",
                        "background": "#3b82f6",
                        "borderRadius": "2px",
                    }),
                    className="neighbor-bar-bg",
                ),
            ], className="neighbor-content"),
        ], className="neighbor-row"))

    return html.Div([header, html.Div(rows, className="neighbor-list")])


def _build_reasoning_trace(token_strings: list[str], token_types: list[str], nearest_text: list | None = None):
    """Build clickable token spans for the reasoning trace panel."""
    from dash import html
    spans = []
    for i, (tok, ttype) in enumerate(zip(token_strings, token_types)):
        nt = nearest_text[i] if nearest_text and i < len(nearest_text) else None
        title = f"Step {i} | {ttype}"
        if ttype == "latent" and nt:
            label = nt["label"].strip() or nt["label"]
            title += f" | ≈{label} (d={nt['distance']:.2f})"
            children: list = [tok, html.Span(f"≈{label}", className="nearest-text-hint")]
        else:
            children = [tok]
        spans.append(
            html.Span(
                children,
                id={"type": "trace-token", "index": i},
                className=f"trace-token trace-token--{ttype}",
                title=title,
            )
        )
    return spans


def _build_instance_list(instances: dict, active_id: str):
    """Build the instance toggle badges."""
    from dash import html
    import dash_bootstrap_components as dbc
    badges = []
    for inst_id in instances:
        is_active = inst_id == active_id
        is_intervention = inst_id.startswith("intervention_")
        label = inst_id
        css = "me-1 instance-badge"
        if is_active:
            css += " active-badge"
        if is_intervention:
            css += " intervention-badge"
        badges.append(
            dbc.Badge(
                label,
                id={"type": "instance-badge", "index": inst_id},
                color="primary",
                className=css,
                n_clicks=0,
            )
        )
    return badges


def _stat_row(label: str, value: str):
    from dash import html
    return html.Div(
        [html.Span(label + ": ", className="stat-label"),
         html.Span(value,        className="stat-value")],
        className="stat-row",
    )

def _build_uncertainty_display(result: dict, current_step: int):
    """
    Build a per-token uncertainty bar strip.

    Uses 'trustworthiness_scores' if available in result (list[float], one per token),
    otherwise shows a placeholder. Scores are in [0, 1]; higher = more trustworthy
    (i.e. lower projection distortion / better neighbour preservation).
    """
    from dash import html

    scores = result.get("trustworthiness_scores")   # list[float] | None
    token_types = result.get("token_types", [])

    if scores is None or len(scores) == 0:
        return html.Div(
            "Uncertainty scores not available for this run.",
            className="stat-row",
            style={"color": "#9ca3af", "fontStyle": "italic"},
        )

    scores = [float(score) for score in scores]
    selected_score = scores[current_step] if 0 <= current_step < len(scores) else None
    mean_score = float(np.mean(scores))
    summary = html.Div(
        [
            html.Span(
                f"Current: {selected_score:.2f}"
                if selected_score is not None
                else "Current: n/a"
            ),
            html.Span(f"Mean: {mean_score:.2f}", style={"marginLeft": "16px"}),
        ],
        className="stat-row",
        title="Fraction of nearest neighbours preserved after projection to 2D.",
    )

    items = []
    for i, (score, ttype) in enumerate(zip(scores, token_types)):
        if score >= 0.75:
            color = "#22c55e"
        elif score >= 0.5:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        is_current = i == current_step
        bar_width = f"{max(4, int(score * 100))}%"
        items.append(
            html.Div(
                className=(
                    "uncertainty-token uncertainty-token--active"
                    if is_current else "uncertainty-token"
                ),
                title=f"Step {i} | {ttype} | neighbour preservation={score:.2f}",
                children=[
                    html.Div(
                        style={
                            "width":        bar_width,
                            "height":       "6px",
                            "background":   color,
                            "borderRadius": "3px",
                            "opacity":      "1" if is_current else "0.55",
                            "transition":   "width 0.2s",
                        }
                    )
                ],
            )
        )
    return html.Div(
        [summary, html.Div(items, className="uncertainty-bar-strip")]
    )
