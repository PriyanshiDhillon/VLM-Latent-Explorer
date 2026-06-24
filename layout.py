from dash import dcc, html
import dash_bootstrap_components as dbc

_BLANK_FIGURE = {
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

            # ── MODEL ─────────────────────────────────────────────────────
            html.Div("Model", className="sidebar-label"),
            dbc.Select(
                id="model-selector",
                options=MODEL_OPTIONS,
                value=None,
                placeholder="Select a model",
                className="sidebar-select mb-2",
            ),
            dbc.Button(
                "Load Model",
                id="load-model-btn",
                className="w-100 mb-2",
                color="secondary",
                size="sm",
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

            # ── INSTANCES ─────────────────────────────────────────────────
            html.Div("Instances", className="sidebar-label"),
            html.Div(id="instance-list", className="instance-list mb-2"),

            html.Hr(className="sidebar-divider"),

            # ── INFERENCE ─────────────────────────────────────────────────
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

            dbc.Button(
                "▶  Run Inference",
                id="run-btn",
                className="run-btn w-100 mb-2",
            ),

            html.Hr(className="sidebar-divider"),

            # ── CAUSAL INTERVENTION ───────────────────────────────────────
            html.Div("Causal Intervention", className="sidebar-label"),
            html.Div(
                "Answer text can read question text, generated latent tokens, and answer history, but not image patches.",
                className="intervention-hint",
            ),

            html.Div("Modified Question", className="intervention-sublabel"),
            dbc.Textarea(
                id="intervention-question-input",
                placeholder="Leave empty to reuse current question…",
                rows=2,
                className="mb-2 sidebar-input",
            ),

            html.Div("Image Mask", className="intervention-sublabel"),
            html.Div(
                "Draw a box on the image to black out a region",
                className="intervention-hint",
            ),
            dcc.Graph(
                id="mask-image-graph",
                config={
                    "modeBarButtonsToAdd": ["select2d"],
                    "modeBarButtonsToRemove": [
                        "zoom2d", "pan2d", "zoomIn2d", "zoomOut2d",
                        "autoScale2d", "resetScale2d", "lasso2d", "toImage",
                    ],
                    "displayModeBar": True,
                    "scrollZoom": False,
                    "displaylogo": False,
                },
                style={"height": "150px"},
                figure=_BLANK_FIGURE,
            ),

            html.Div("Intervention Trace", className="intervention-sublabel"),
            html.Div(
                id="intervention-reasoning-trace",
                className="reasoning-trace mb-2",
                children="Run an intervention to see its trace here.",
                style={"color": "#9ca3af", "fontStyle": "italic"},
            ),

            dbc.Button(
                "▶  Run Latent Bottleneck",
                id="run-intervention-btn",
                className="intervention-btn w-100 mb-3",
                n_clicks=0,
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


def attention_row() -> dbc.Row:
    return dbc.Row(className="g-3 mb-3", children=[
        dbc.Col(width=12, children=[
            html.Div(className="content-card", children=[
                html.Div(
                    className="d-flex justify-content-between align-items-start",
                    children=[
                        html.Div([
                            html.Div("Generated Token Attention", className="content-card-title"),
                            html.Div(
                                "Rows query previous generated tokens; values are normalized within generated history",
                                className="content-card-subtitle",
                            ),
                            html.Div(
                                className="attention-layer-control",
                                children=[
                                    html.Div(
                                        className="attention-layer-heading",
                                        children=[
                                            html.Span(
                                                "Decoder layer",
                                                className="attention-layer-label",
                                            ),
                                            html.Button(
                                                html.Span(className="playback-icon playback-icon--play"),
                                                id="attention-layer-playback",
                                                className="attention-layer-playback",
                                                title="Play decoder layers",
                                                **{"aria-label": "Play decoder layers"},
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="attention-layer-slider-wrap",
                                        children=dcc.Slider(
                                            id="attention-layer-slider",
                                            min=0,
                                            max=0,
                                            step=1,
                                            value=0,
                                            marks={0: "rerun"},
                                            updatemode="mouseup",
                                        ),
                                    ),
                                    dcc.Interval(
                                        id="attention-layer-interval",
                                        interval=1000,
                                        n_intervals=0,
                                        disabled=True,
                                    ),
                                ],
                            ),
                        ]),
                        html.Div(className="token-type-legend", children=[
                            html.Span("T · text", className="token-type-chip token-type-chip--text"),
                            html.Span("L · latent", className="token-type-chip token-type-chip--latent"),
                            html.Span("V · visual", className="token-type-chip token-type-chip--visual"),
                        ]),
                    ],
                ),
                dcc.Graph(
                    id="token-attention-matrix",
                    config={"displayModeBar": True, "scrollZoom": True},
                    style={"height": "520px"},
                    figure={"data": [], "layout": {"template": "plotly_white"}},
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

                # ── Controls strip ────────────────────────────────────────
                html.Div(
                    className="d-flex flex-wrap align-items-center gap-3 mb-1",
                    style={"fontSize": "0.78rem", "color": "#374151"},
                    children=[
                        # Highlight mode
                        html.Div([
                            html.Span("Highlight:", style={"color": "#6b7280", "marginRight": "5px"}),
                            dcc.RadioItems(
                                id="highlight-mode",
                                options=[
                                    {"label": "All tokens",   "value": "all"},
                                    {"label": "Up to step",   "value": "step"},
                                ],
                                value="all",
                                inline=True,
                                inputStyle={"marginRight": "3px"},
                                labelStyle={"marginRight": "10px"},
                            ),
                        ], className="d-flex align-items-center"),
                        # Show trace toggle
                        html.Div([
                            html.Span("Show trace:", style={"color": "#6b7280", "marginRight": "5px"}),
                            dcc.Checklist(
                                id="umap-show-trace",
                                options=[{"label": "", "value": "trace"}],
                                value=["trace"],
                                inputStyle={"cursor": "pointer"},
                            ),
                        ], className="d-flex align-items-center"),
                        # Line follows / highlight target
                        html.Div([
                            html.Span(
                                "Line / highlight:",
                                style={"color": "#6b7280", "marginRight": "5px", "whiteSpace": "nowrap"},
                            ),
                            dcc.Dropdown(
                                id="umap-line-target",
                                options=[{"label": "Current trace", "value": "__active__"}],
                                value="__active__",
                                clearable=False,
                                style={"width": "170px", "fontSize": "0.78rem"},
                            ),
                        ], className="d-flex align-items-center"),
                    ],
                ),
                # ── Corpus passage visibility ──────────────────────────────
                html.Div(
                    className="d-flex align-items-center gap-2 mb-2",
                    children=[
                        html.Span(
                            "Corpus:",
                            style={"fontSize": "0.78rem", "color": "#6b7280", "whiteSpace": "nowrap"},
                        ),
                        html.Span(
                            id="corpus-selection-count",
                            children="",
                            style={"fontSize": "0.78rem", "color": "#374151", "whiteSpace": "nowrap"},
                        ),
                        dbc.Button(
                            "Select All", id="btn-corpus-all",
                            size="sm", color="secondary", outline=True,
                            style={"fontSize": "0.72rem", "padding": "1px 8px"},
                            n_clicks=0,
                        ),
                        dbc.Button(
                            "Deselect All", id="btn-corpus-none",
                            size="sm", color="secondary", outline=True,
                            style={"fontSize": "0.72rem", "padding": "1px 8px"},
                            n_clicks=0,
                        ),
                        dcc.Dropdown(
                            id="corpus-passage-dropdown",
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="Toggle individual passages…",
                            className="corpus-passage-dropdown",
                            style={"flex": "1", "fontSize": "0.78rem", "minWidth": "0"},
                        ),
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
                    style={"height": "360px"},
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


# ── Row 3: Token context + Stats ─────────────────────────────────────────────
def row_three() -> dbc.Row:
    return dbc.Row(className="g-3", children=[

        # ── TOKEN CONTEXT PANEL ───────────────────────────────────────────
        dbc.Col(width=6, children=[
            html.Div(className="content-card", children=[
                html.Div("Token Context — Nearest Corpus Neighbours", className="content-card-title"),
                html.Div(
                    "10 closest corpus text tokens to the current step in UMAP space",
                    className="content-card-subtitle",
                ),
                html.Div(id="token-context-panel", className="token-context-panel"),
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
                    attention_row(),
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
            dcc.Store(id="store-umap-base",         data=None),
            dcc.Store(id="store-mask-region",       data=None),
        ],
    )
