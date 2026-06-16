from dash import dcc, html
import dash_bootstrap_components as dbc

MODEL_OPTIONS = [
    {"label": "Qwen2.5-VL (baseline)", "value": "qwen"},
    {"label": "Monet-7B",              "value": "monet"},
    {"label": "LVR",                   "value": "lvr"},
]


# ── Top-level Reasoning Step slider bar ───────────────────────────────────────
def step_slider_bar() -> html.Div:
    return html.Div(
        className="global-step-bar",
        style={"marginBottom": "30px"},
        children=[
            html.Div("Reasoning Step", className="step-bar-label"),
            dcc.Slider(
                id="step-slider",
                min=0, max=0, step=1, value=0,
                marks={},
                tooltip={"placement": "bottom", "always_visible": False},
                className="center-slider global-step-slider",
            ),
        ],
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar() -> html.Div:
    return html.Div(
        className="sidebar",
        children=[

            html.Div([
                html.Div("VLM Latent Token Explorer", className="sidebar-title"),
                html.Div("Visual Latent Reasoning Analysis", className="sidebar-subtitle"),
            ]),

            html.Div("Model", className="sidebar-label"),
            dbc.Select(
                id="model-selector",
                options=MODEL_OPTIONS,
                value=None,
                placeholder="Select a model",
                className="sidebar-select mb-2",
            ),
            dcc.Loading(
                type="circle",
                children=html.Div(id="model-load-status",
                                  style={"fontSize": "0.8rem", "color": "#9ca3af"}),
            ),
            dbc.Progress(id="model-load-progress", value=0, style={"display": "none"}, className="mb-2"),

            html.Div("Example", className="sidebar-label"),
            dbc.Select(
                id="example-selector",
                options=[],
                value=None,
                className="sidebar-select mb-2",
            ),

            html.Hr(className="sidebar-divider"),

            # ── INPUT SECTION ────────────────────────────────────────────
            html.Div("Question", className="sidebar-label"),
            dbc.Textarea(
                id="question-input",
                placeholder="Enter a question...",
                rows=2,
                className="mb-2 sidebar-input",
            ),

            html.Div("Upload Image", className="sidebar-label"),
            dcc.Upload(
                id="image-upload",
                children=html.Div([
                    "Drag & Drop or ",
                    html.A("select file", className="upload-link"),
                ]),
                className="upload-box mb-2",
            ),
            html.Img(
                id="uploaded-image-preview",
                className="preview-img mb-2",
                style={"display": "none"},
            ),

            html.Div("Reasoning Trace", className="sidebar-label"),
            html.Div(id="reasoning-trace", className="reasoning-trace mb-2"),

            html.Hr(className="sidebar-divider"),

            html.Div("Instances", className="sidebar-label"),
            html.Div(id="instance-list", className="instance-list mb-2"),

            html.Div(style={"flex": "1"}),

            dbc.Button(
                "▶  Run Inference",
                id="run-btn",
                className="run-btn w-100",
            ),
        ],
    )


# ── Row 1: Heatmap | Parameters + Eval ───────────────────────────────────────
def row_one() -> dbc.Row:
    return dbc.Row(className="g-3 mb-3", children=[

        # ── HEATMAP ───────────────────────────────────────────────────────
        dbc.Col(width=6, children=[
            html.Div(className="content-card h-100", children=[
                html.Div("Attention Heatmap",
                         className="content-card-title"),
                html.Div("Cross-attention overlay on image per reasoning step",
                         className="content-card-subtitle"),
                html.Div(
                    html.Img(
                        id="heatmap-image",
                        src="",
                        className="heatmap-img",
                    ),
                    className="heatmap-wrapper",
                ),
            ]),
        ]),

        # ── PARAMETERS + EVALUATION ───────────────────────────────────────
        dbc.Col(width=3, children=[
            html.Div(className="content-card mb-3 h-100", children=[
                html.Div("Token Parameters",
                         className="content-card-title"),
                html.Div("Info for the token at the current step",
                         className="content-card-subtitle"),
                html.Div(id="param-display", className="param-display"),
            ]),
        ]),
        dbc.Col(width=3, children=[
            html.Div(className="content-card h-100", children=[
                html.Div("Evaluation",
                         className="content-card-title"),
                html.Div("Per-instance accuracy",
                         className="content-card-subtitle"),
                html.Div(
                    id="eval-display",
                    className="eval-display",
                    children="No instances yet.",
                ),
            ]),
        ]),
    ])


# ── Row 2: Projection (UMAP/t-SNE toggle) + Zoom ─────────────────────────────
def row_two() -> dbc.Row:
    return dbc.Row(className="g-3 mb-3", children=[

        # ── MAIN PROJECTION PANEL (full width) ───────────────────────────
        dbc.Col(width=12, children=[
            html.Div(className="content-card", children=[

                # ── Header row: title + projection toggle ─────────────────
                html.Div(
                    className="d-flex justify-content-between align-items-start mb-1",
                    children=[
                        html.Div([
                            html.Div(
                                id="projection-panel-title",
                                children="UMAP Projection",
                                className="content-card-title",
                                style={"marginBottom": "2px"},
                            ),
                            html.Div(
                                id="projection-panel-subtitle",
                                children="Token embedding space — draw a box to zoom into a region",
                                className="content-card-subtitle",
                            ),
                        ]),
                        dbc.ButtonGroup([
                            dbc.Button(
                                "UMAP",
                                id="btn-umap",
                                color="primary",
                                size="sm",
                                className="proj-toggle-btn active",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                "t-SNE",
                                id="btn-tsne",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="proj-toggle-btn",
                                n_clicks=0,
                            ),
                        ], className="proj-toggle-group"),
                    ],
                ),

                # ── Main projection graph ─────────────────────────────────
                dcc.Graph(
                    id="umap-graph",
                    config={
                        "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                        "displayModeBar": True,
                        "scrollZoom": True,
                    },
                    style={"height": "380px"},
                    figure={
                        "data": [],
                        "layout": {
                            "template":      "plotly_white",
                            "paper_bgcolor": "#ffffff",
                            "plot_bgcolor":  "#f4f7f5",
                        },
                    },
                ),

                # ── Uncertainty strip ─────────────────────────────────────
                html.Hr(className="card-divider"),
                html.Div(
                    className="d-flex align-items-center gap-2 mb-1",
                    children=[
                        html.Div("Projection Uncertainty", className="panel-label"),
                        html.Span(
                            "ⓘ",
                            id="uncertainty-info-icon",
                            className="uncertainty-info-icon",
                            title=(
                                "Neighbour-preservation score per token: "
                                "high = trustworthy position, low = compressed/distorted region."
                            ),
                        ),
                    ],
                ),
                html.Div(id="uncertainty-display", className="uncertainty-display"),
            ]),
        ]),
    ])


# ── Row 3: Zoom / detail panel + Stats ───────────────────────────────────────
def row_three() -> dbc.Row:
    return dbc.Row(className="g-3", children=[

        # ── ZOOM DETAIL (re-projection of selected box) ───────────────────
        dbc.Col(width=6, children=[
            html.Div(className="content-card", children=[
                html.Div("Zoom — Selected Region", className="content-card-title"),
                html.Div(
                    "Re-projection of the bounding-box selection above",
                    className="content-card-subtitle",
                ),
                dcc.Graph(
                    id="tsne-graph",
                    style={"height": "320px"},
                    figure={
                        "data": [],
                        "layout": {
                            "template":      "plotly_white",
                            "paper_bgcolor": "#ffffff",
                            "plot_bgcolor":  "#f4f7f5",
                        },
                    },
                ),
            ]),
        ]),

        # ── SELECTION STATISTICS ──────────────────────────────────────────
        dbc.Col(width=6, children=[
            html.Div(className="content-card", children=[
                html.Div("Selection Statistics", className="content-card-title"),
                html.Div(
                    "Aggregate metrics for the tokens in the selected region",
                    className="content-card-subtitle",
                ),
                html.Div(id="stats-display", className="stats-display"),
            ]),
        ]),
    ])


# ── Root layout ───────────────────────────────────────────────────────────────
def build_layout() -> html.Div:
    return html.Div(
        className="dashboard-root",
        children=[

            sidebar(),

            html.Div(
                className="main-content",
                children=[
                    html.Div(
                        className="page-header",
                        children=[
                            html.H4(
                                "Visual Latent Token Reasoning Explorer",
                                className="page-title",
                            ),
                            step_slider_bar(),
                        ],
                    ),
                    row_one(),
                    row_two(),
                    row_three(),
                ],
            ),

            # ── Stores ───────────────────────────────────────────────────
            dcc.Store(id="store-instances",         data={}),
            dcc.Store(id="store-active-instance",   data=None),
            dcc.Store(id="store-corpus-embeddings", data={}),
            dcc.Store(id="store-current-image-b64", data=None),
            dcc.Store(id="store-active-projection", data="umap"),
        ],
    )