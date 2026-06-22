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
from dash import Input, Output, State, callback, ctx, html, no_update, ALL
from dash.exceptions import PreventUpdate
from PIL import Image

from backend import heatmap as hm
from backend import projection as proj
from backend import data_loader as dl
from backend import inference as inf
from backend import token_attention as token_attn

import plotly.graph_objects as go


TOKEN_COLORS = {
    "text":   "#3b82f6",   
    "visual": "#22c55e", 
    "latent": "#f97316", 
}

INSTANCE_LINE_STYLES = ["solid", "dash", "dot", "dashdot"]
_TSNE_CACHE: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_pil(b64_str: str) -> Image.Image:
    """Convert a base64 data-URI to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


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
    current_step: int,
    projection_mode: str = "umap",
    highlight_mode: str = "all",
    show_trace: bool = True,
    corpus_visible_ids: list | None = None,
    line_target: str = "__active__",
) -> go.Figure:
    """Build the UMAP scatter figure with corpus background + instance trajectories."""
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

    # ── Unpack corpus arrays (always, so trajectory code can use them) ───────
    corpus_coords = corpus_genidx = corpus_tokidx = corpus_example_ids = None
    corpus_types = corpus_labels = None
    if corpus and corpus.get("coords"):
        corpus_coords      = np.array(corpus["coords"])
        corpus_types       = corpus["types"]
        corpus_labels      = corpus.get("labels")
        corpus_example_ids = corpus.get("example_ids")
        corpus_genidx      = corpus.get("gen_index")
        corpus_tokidx      = corpus.get("token_index")

    # Pre-compute sorted sequence for the corpus line_target example so
    # highlight_mode="step" can trim it to current_step tokens.
    highlighted_corpus_set: set[int] = set()
    sorted_ex_mask: list[int] = []
    if line_target != "__active__" and corpus_coords is not None and corpus_example_ids is not None:
        def _seq_key(i: int) -> tuple:
            gi = int(corpus_genidx[i]) if corpus_genidx is not None else 0
            ti = int(corpus_tokidx[i]) if corpus_tokidx is not None else 0
            return (1, gi) if gi >= 0 else (0, ti)
        ex_all = [i for i, e in enumerate(corpus_example_ids) if str(e) == str(line_target)]
        sorted_ex_mask = sorted(ex_all, key=_seq_key)
        limit = current_step + 1 if highlight_mode == "step" else len(sorted_ex_mask)
        highlighted_corpus_set = set(sorted_ex_mask[:limit])

    # ── Corpus background ────────────────────────────────────────────────────
    if corpus_coords is not None and corpus_visible_ids is not None and len(corpus_visible_ids) > 0:
        visible_set = set(str(v) for v in corpus_visible_ids)
        have_meta   = corpus_example_ids is not None and corpus_genidx is not None

        for ttype, color in TOKEN_COLORS.items():
            # Split into highlighted (line_target) and background points
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
    if show_trace:
        for idx, (inst_id, inst_data) in enumerate(instances.items()):
            result = _result_from_serialisable(inst_data)
            coords_2d = result.get("coords_2d")
            if coords_2d is None or len(coords_2d) == 0:
                continue
            coords_2d    = np.array(coords_2d)
            token_types  = result.get("token_types", [])
            token_strings = result.get("token_strings", [])
            is_active = inst_id == active_id

            # Highlight only when active instance is the line target
            if line_target == "__active__" and is_active:
                if highlight_mode == "step":
                    highlighted = set(range(current_step + 1))
                else:
                    highlighted = set(range(len(token_types)))
            else:
                highlighted = set()  # dim — not the current line/highlight target

            for ttype, color in TOKEN_COLORS.items():
                mask = [i for i, t in enumerate(token_types) if t == ttype]
                if not mask:
                    continue
                is_highlighted = bool(highlighted)
                sizes    = [16 if i == current_step and is_highlighted else
                             (10 if i in highlighted else 5) for i in mask]
                opacities = [0.95 if i in highlighted else
                              (0.4 if is_active else 0.2) for i in mask]
                symbols  = ["star" if i == current_step and is_highlighted
                             else "circle" for i in mask]
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
                        symbol=symbols,
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

    # ── Trajectory line ──────────────────────────────────────────────────────
    if line_target == "__active__" and active_id and active_id in instances:
        result    = _result_from_serialisable(instances[active_id])
        coords_2d = result.get("coords_2d")
        if coords_2d is not None and len(coords_2d) > 0:
            coords_2d   = np.array(coords_2d)
            token_types  = result.get("token_types", [])
            inst_idx     = list(instances.keys()).index(active_id)
            dash         = INSTANCE_LINE_STYLES[inst_idx % len(INSTANCE_LINE_STYLES)]
            for ttype, color in TOKEN_COLORS.items():
                traj_idx = sorted([i for i, t in enumerate(token_types)
                                   if t == ttype and i <= current_step])
                if len(traj_idx) < 2:
                    continue
                lc = coords_2d[traj_idx]
                fig.add_trace(go.Scatter(
                    x=lc[:, 0], y=lc[:, 1],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=2),
                    name=f"line:{ttype}",
                    legendgroup="line",
                    showlegend=False,
                ))

    elif line_target != "__active__" and sorted_ex_mask:
        limit = current_step + 1 if highlight_mode == "step" else len(sorted_ex_mask)
        traj  = sorted_ex_mask[:limit]
        if len(traj) >= 2:
            lc = corpus_coords[traj]
            fig.add_trace(go.Scatter(
                x=lc[:, 0], y=lc[:, 1],
                mode="lines",
                line=dict(color="#8b5cf6", dash="solid", width=2),
                name=f"line:passage {_passage(line_target)}",
                showlegend=False,
            ))

    return fig


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
    )
    def reload_corpus(model_name):
        if not dl.corpus_embeddings_exist(model_name):
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
        State("store-current-image-b64", "data"),
        State("question-input",          "value"),
        State("model-selector",          "value"),
        State("store-instances",         "data"),
        prevent_initial_call=True,
    )
    def run_inference(n_clicks, img_b64, question, model_name, existing_instances):
        if not model_name or not img_b64 or not question:
            raise PreventUpdate

        image = _b64_to_pil(img_b64)
        result = inf.run_inference(image, question, model_name)
        inst_id = f"instance_{len(existing_instances) + 1}"   # ← define first

        cache_dir = Path("precomputed/online_cache") / model_name
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

        attn_weights_saveable = [
            w if w is not None else np.array([])
            for w in result["attn_weights"]
        ]
        attn_arr = np.empty(len(attn_weights_saveable), dtype=object)
        for i, w in enumerate(attn_weights_saveable):
            attn_arr[i] = w
        np.save(
            cache_dir / f"{inst_id}_attn_weights.npy",
            attn_arr,
            allow_pickle=True,
        )
        layer_attn_arr = np.empty(len(result["layer_attn_weights"]), dtype=object)
        for layer_index, layer_steps in enumerate(result["layer_attn_weights"]):
            step_arr = np.empty(len(layer_steps), dtype=object)
            for step_index, weights in enumerate(layer_steps):
                step_arr[step_index] = weights if weights is not None else np.array([])
            layer_attn_arr[layer_index] = step_arr
        np.save(
            cache_dir / f"{inst_id}_layer_attn_weights.npy",
            layer_attn_arr,
            allow_pickle=True,
        )
        if dl.umap_model_exists(model_name):
            try:
                coords_2d = proj.project_onto_manifold(result["activations"], model_name)
                result["coords_2d"] = coords_2d
            except Exception as exc:
                result["coords_2d"] = None
                result["projection_error"] = str(exc)
                print(f"[projection] {model_name}/{inst_id}: {exc}")
        else:
            result["coords_2d"] = None

       # inst_id = f"instance_{len(existing_instances) + 1}"
        updated_instances = dict(existing_instances)
        serialisable = _result_to_serialisable(
            {k: v for k, v in result.items() if k != "activations"}
        )
        updated_instances[inst_id] = serialisable
        n_steps = len(result["token_strings"])
        marks = {i: str(i) for i in range(0, n_steps, max(1, n_steps // 10))}

        trace_children = _build_reasoning_trace(result["token_strings"], result["token_types"])
        instance_children = _build_instance_list(updated_instances, inst_id)

        return (
            updated_instances,
            inst_id,
            n_steps - 1,
            0,
            marks,
            trace_children,
            instance_children,
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
        Output("umap-graph",          "figure"),
        Output("param-display",       "children"),
        Output("uncertainty-display", "children"),
        Input("step-slider",            "value"),
        Input("store-active-instance",  "data"),
        Input("store-active-projection","data"),
        Input("highlight-mode",         "value"),
        Input("umap-show-trace",        "value"),
        Input("corpus-passage-dropdown","value"),
        Input("umap-line-target",       "value"),
        State("store-instances",        "data"),
        State("store-corpus-embeddings","data"),
        State("store-current-image-b64","data"),
        State("model-selector",         "value"),
        prevent_initial_call=True,
    )
    def update_views(step, active_id, projection_mode, highlight_mode,
                     show_trace_val, corpus_visible_ids, line_target,
                     instances, corpus, img_b64, model_name):
        if not active_id or active_id not in instances:
            raise PreventUpdate

        result = _result_from_serialisable(instances[active_id])
        attn_list = result.get("attn_weights", [])
        grid_hw   = tuple(result.get("image_grid_hw", (16, 16)))

        attn_at_step = None
        if attn_list and step < len(attn_list) and attn_list[step] is not None:
            attn_at_step = np.array(attn_list[step])

        if img_b64:
            image = _b64_to_pil(img_b64)
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

        fig = _build_umap_figure(
            corpus, instances, active_id, step, projection_mode,
            highlight_mode=highlight_mode or "all",
            show_trace="trace" in (show_trace_val or []),
            corpus_visible_ids=corpus_visible_ids or [],
            line_target=line_target or "__active__",
        )

        param_children = [
            _stat_row("Step",       str(step)),
            _stat_row("Token",      repr(token_str)),
            _stat_row("Type",       token_type),
            _stat_row("Model",      model_name),
            _stat_row("Instance",   active_id),
            _stat_row("Projection", projection_mode or "umap"),
        ]
        if result.get("projection_error"):
            param_children.append(_stat_row("Projection error", result["projection_error"]))

        # ── Uncertainty (neighbour-preservation per token) ────────────────
        uncertainty_children = _build_uncertainty_display(result, step)

        return heatmap_src, fig, param_children, uncertainty_children

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
        return _build_token_attention_figure(result, step or 0, layer_index)

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
        Output("tsne-graph",    "figure"),
        Output("stats-display", "children"),
        Input("umap-graph",     "selectedData"),
        Input("highlight-mode", "value"),
        State("store-active-instance", "data"),
        State("store-instances",        "data"),
        State("step-slider",            "value"),
        prevent_initial_call=True,
    )
    def update_tsne(selected_data, highlight_mode, active_id, instances, step):
        if not selected_data or not selected_data.get("points"):
            raise PreventUpdate
        if not active_id or active_id not in instances:
            raise PreventUpdate

        result = _result_from_serialisable(instances[active_id])
        coords_2d    = result.get("coords_2d")
        token_types  = result.get("token_types", [])
        token_strings = result.get("token_strings", [])

        if coords_2d is None:
            raise PreventUpdate

        coords_2d = np.array(coords_2d)

        selected_indices = []
        for pt in selected_data["points"]:
            customdata = pt.get("customdata")
            if (
                isinstance(customdata, (list, tuple))
                and len(customdata) >= 2
                and customdata[0] == active_id
            ):
                selected_indices.append(int(customdata[1]))

        if not selected_indices:
            raise PreventUpdate

        selected_indices = [i for i in selected_indices if i < len(coords_2d)]
        selected_coords  = coords_2d[selected_indices]
        selected_types   = [token_types[i] for i in selected_indices if i < len(token_types)]

        tsne_coords, _ = proj.tsne_reproject(selected_coords, selected_types)

        # Which original indices count as "highlighted" based on highlight_mode
        highlighted = set(selected_indices if highlight_mode == "all" else
                          [i for i in selected_indices if i <= step])

        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f4f7f5",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        for ttype, color in TOKEN_COLORS.items():
            mask = [j for j, t in enumerate(selected_types) if t == ttype]
            if not mask:
                continue
            orig_indices = [selected_indices[j] for j in mask]
            sizes    = [12 if orig_indices[k] in highlighted else 6  for k in range(len(mask))]
            opacities = [0.95 if orig_indices[k] in highlighted else 0.35 for k in range(len(mask))]
            customdata = [
                [orig_indices[k], token_strings[orig_indices[k]] if orig_indices[k] < len(token_strings) else ""]
                for k in range(len(mask))
            ]
            fig.add_trace(go.Scatter(
                x=tsne_coords[mask, 0], y=tsne_coords[mask, 1],
                mode="markers",
                marker=dict(
                    color=color,
                    size=sizes,
                    opacity=opacities,
                    line=dict(width=1, color="#111827"),
                ),
                name=ttype,
                customdata=customdata,
                hovertemplate=(
                    "<b>" + ttype + "</b><br>"
                    "step: %{customdata[0]}<br>"
                    "token: %{customdata[1]}<br>"
                    "(%{x:.2f}, %{y:.2f})<extra></extra>"
                ),
            ))

        attn_list = result.get("attn_weights", [])
        attn_at_step = None
        if attn_list and step < len(attn_list) and attn_list[step] is not None:
            attn_at_step = np.array(attn_list[step])

        stats = proj.compute_selection_stats(selected_types, attn_at_step, [])
        stats_children = [_stat_row(k, str(v)) for k, v in stats.items()]

        return fig, stats_children

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
        Output("reasoning-trace", "children", allow_duplicate=True),
        Output("instance-list",   "children", allow_duplicate=True),
        Input("store-active-instance", "data"),
        State("store-instances",       "data"),
        prevent_initial_call=True,
    )
    def update_trace_on_switch(active_id, instances):
        if not active_id or active_id not in instances:
            raise PreventUpdate
        result = instances[active_id]
        trace = _build_reasoning_trace(
            result.get("token_strings", []),
            result.get("token_types", []),
        )
        badges = _build_instance_list(instances, active_id)
        return trace, badges


    @app.callback(
        Output("eval-display", "children"),
        Input("store-instances", "data"),
    )
    def update_eval(instances):
        if not instances:
            return "No instances yet."
        rows = []
        for inst_id, data in instances.items():
            correct = data.get("correct", None)
            label = "✓" if correct else ("✗" if correct is False else "?")
            rows.append(_stat_row(inst_id, label))
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

def _build_reasoning_trace(token_strings: list[str], token_types: list[str]):
    """Build clickable token spans for the reasoning trace panel."""
    from dash import html
    spans = []
    for i, (tok, ttype) in enumerate(zip(token_strings, token_types)):
        spans.append(
            html.Span(
                tok,
                id={"type": "trace-token", "index": i},
                className=f"trace-token trace-token--{ttype}",
                title=f"Step {i} | {ttype}",
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
        badges.append(
            dbc.Badge(
                inst_id,
                id={"type": "instance-badge", "index": inst_id},
                color="primary" if is_active else "secondary",
                className="me-1 instance-badge",
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

    if not scores:
        return html.Div(
            "Uncertainty scores not available for this run.",
            className="stat-row",
            style={"color": "#9ca3af", "fontStyle": "italic"},
        )

    items = []
    for i, (score, ttype) in enumerate(zip(scores, token_types)):
        color = TOKEN_COLORS.get(ttype, "#6b7280")
        is_current = i == current_step
        bar_width = f"{max(4, int(score * 100))}%"
        items.append(
            html.Div(
                className=(
                    "uncertainty-token uncertainty-token--active"
                    if is_current else "uncertainty-token"
                ),
                title=f"Step {i} | {ttype} | trust={score:.2f}",
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
    return html.Div(items, className="uncertainty-bar-strip")
