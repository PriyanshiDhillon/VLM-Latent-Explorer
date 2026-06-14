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
import io
import json
import numpy as np
from dash import Input, Output, State, callback, ctx, no_update, ALL
from dash.exceptions import PreventUpdate
from PIL import Image

from backend import heatmap as hm
from backend import projection as proj
from backend import data_loader as dl
from backend import inference as inf

import plotly.graph_objects as go


TOKEN_COLORS = {
    "text":   "#3b82f6",   
    "visual": "#22c55e", 
    "latent": "#f97316", 
}

INSTANCE_LINE_STYLES = ["solid", "dash", "dot", "dashdot"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64_to_pil(b64_str: str) -> Image.Image:
    """Convert a base64 data-URI to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def _result_to_serialisable(result: dict) -> dict:
    """Convert numpy arrays in inference result to lists for dcc.Store."""
    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            out[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        else:
            out[k] = v
    return out


def _result_from_serialisable(result: dict) -> dict:
    """Restore numpy arrays from dcc.Store dict."""
    out = {}
    for k, v in result.items():
        if k in ("activations", "coords_2d") and v is not None:
            out[k] = np.array(v)
        else:
            out[k] = v
    return out


def _build_umap_figure(
    corpus: dict | None,
    instances: dict,
    active_id: str | None,
    current_step: int,
) -> go.Figure:
    """Build the UMAP scatter figure with corpus background + instance trajectories."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f4f7f5",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        dragmode="select",
        uirevision="umap",
    )

    if corpus and corpus.get("coords"):
        coords = np.array(corpus["coords"])
        types  = corpus["types"]
        for ttype, color in TOKEN_COLORS.items():
            mask = [i for i, t in enumerate(types) if t == ttype]
            if not mask:
                continue
            fig.add_trace(go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1],
                mode="markers",
                marker=dict(color=color, size=3, opacity=0.25),
                name=f"corpus:{ttype}",
                showlegend=True,
            ))


    for idx, (inst_id, inst_data) in enumerate(instances.items()):
        result = _result_from_serialisable(inst_data)
        coords_2d = result.get("coords_2d")
        if coords_2d is None or len(coords_2d) == 0:
            continue
        coords_2d = np.array(coords_2d)
        token_types = result.get("token_types", [])
        is_active = inst_id == active_id


        for ttype, color in TOKEN_COLORS.items():
            mask = [i for i, t in enumerate(token_types) if t == ttype]
            if not mask:
                continue
            sizes = [10 if i == current_step else 6 for i in mask]
            symbols = ["star" if i == current_step else "circle" for i in mask]
            fig.add_trace(go.Scatter(
                x=coords_2d[mask, 0], y=coords_2d[mask, 1],
                mode="markers",
                marker=dict(
                    color=color,
                    size=sizes,
                    symbol=symbols,
                    opacity=0.9 if is_active else 0.5,
                    line=dict(width=1, color="white") if is_active else dict(width=0),
                ),
                name=f"{inst_id}:{ttype}",
                showlegend=True,
            ))

        latent_idx = sorted([i for i, t in enumerate(token_types) if t == "latent"])
        if latent_idx:
            lc = coords_2d[latent_idx]
            dash = INSTANCE_LINE_STYLES[idx % len(INSTANCE_LINE_STYLES)]
            fig.add_trace(go.Scatter(
                x=lc[:, 0], y=lc[:, 1],
                mode="lines",
                line=dict(color=TOKEN_COLORS["latent"], dash=dash, width=2),
                name=f"{inst_id}:trajectory",
                showlegend=True,
            ))

    return fig


# ── Callbacks ────────────────────────────────────────────────────────────────

def register_callbacks(app):

    @app.callback(
        Output("uploaded-image-preview", "src"),
        Output("store-current-image-b64", "data"),
        Input("image-upload", "contents"),
        prevent_initial_call=True,
    )
    def preview_image(contents):
        if not contents:
            raise PreventUpdate
        return contents, contents

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
        Output("store-corpus-embeddings", "data"),
        Input("model-selector", "value"),
    )
    def reload_corpus(model_name):
        if not dl.corpus_embeddings_exist(model_name):
            return {}
        try:
            corpus = dl.load_corpus_embeddings(model_name)
            return {
                "coords": corpus["coords"].tolist(),
                "types":  corpus["types"],
                "labels": corpus["labels"],
            }
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
        if not img_b64 or not question:
            raise PreventUpdate

        image = _b64_to_pil(img_b64)
        result = inf.run_inference(image, question, model_name)

        if dl.umap_model_exists(model_name):
            try:
                coords_2d = proj.project_onto_manifold(result["activations"], model_name)
                result["coords_2d"] = coords_2d
            except Exception:
                result["coords_2d"] = None
        else:
            result["coords_2d"] = None

        inst_id = f"instance_{len(existing_instances) + 1}"
        updated_instances = dict(existing_instances)
        updated_instances[inst_id] = _result_to_serialisable(result)

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
        Output("heatmap-image",  "src"),
        Output("umap-graph",     "figure"),
        Output("param-display",  "children"),
        Input("step-slider",           "value"),
        Input("store-active-instance", "data"),
        State("store-instances",        "data"),
        State("store-corpus-embeddings","data"),
        State("store-current-image-b64","data"),
        State("model-selector",         "value"),
        prevent_initial_call=True,
    )
    def update_views(step, active_id, instances, corpus, img_b64, model_name):
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
            heatmap_src = hm.attn_to_heatmap_overlay(image, attn_at_step, grid_hw)
        else:
            heatmap_src = ""

        fig = _build_umap_figure(corpus, instances, active_id, step)

        token_str = ""
        if step < len(result.get("token_strings", [])):
            token_str = result["token_strings"][step]
        token_type = ""
        if step < len(result.get("token_types", [])):
            token_type = result["token_types"][step]

        param_children = [
            _stat_row("Step",       str(step)),
            _stat_row("Token",      repr(token_str)),
            _stat_row("Type",       token_type),
            _stat_row("Model",      model_name),
            _stat_row("Instance",   active_id),
        ]

        return heatmap_src, fig, param_children

    @app.callback(
        Output("tsne-graph",    "figure"),
        Output("stats-display", "children"),
        Input("umap-graph",     "selectedData"),
        State("store-active-instance", "data"),
        State("store-instances",        "data"),
        State("step-slider",            "value"),
        prevent_initial_call=True,
    )
    def update_tsne(selected_data, active_id, instances, step):
        if not selected_data or not selected_data.get("points"):
            raise PreventUpdate
        if not active_id or active_id not in instances:
            raise PreventUpdate

        result = _result_from_serialisable(instances[active_id])
        coords_2d   = result.get("coords_2d")
        token_types = result.get("token_types", [])

        if coords_2d is None:
            raise PreventUpdate

        coords_2d = np.array(coords_2d)

        selected_indices = []
        for pt in selected_data["points"]:
            if "pointIndex" in pt:
                selected_indices.append(pt["pointIndex"])

        if not selected_indices:
            raise PreventUpdate

        selected_indices = [i for i in selected_indices if i < len(coords_2d)]
        selected_coords  = coords_2d[selected_indices]
        selected_types   = [token_types[i] for i in selected_indices if i < len(token_types)]

        tsne_coords, _ = proj.tsne_reproject(selected_coords, selected_types)

        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f4f7f5",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        for ttype, color in TOKEN_COLORS.items():
            mask = [i for i, t in enumerate(selected_types) if t == ttype]
            if not mask:
                continue
            fig.add_trace(go.Scatter(
                x=tsne_coords[mask, 0], y=tsne_coords[mask, 1],
                mode="markers",
                marker=dict(color=color, size=7),
                name=ttype,
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
        return triggered["index"]

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
    

    @app.callback(
        Output("kpi-model", "children"),
        Output("kpi-step",  "children"),
        Output("kpi-type",  "children"),
        Output("kpi-inst",  "children"),
        Input("model-selector",        "value"),
        Input("step-slider",           "value"),
        Input("store-active-instance", "data"),
        Input("store-instances",       "data"),
    )
    def update_kpis(model_name, step, active_id, instances):
        inst_count = len(instances) if instances else 0
        token_type = "—"
        if active_id and instances and active_id in instances:
            types = instances[active_id].get("token_types", [])
            if types and step < len(types):
                token_type = types[step]
        return (
            model_name or "—",
            str(step),
            token_type,
            str(inst_count),
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