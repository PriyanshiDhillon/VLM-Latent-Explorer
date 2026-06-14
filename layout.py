from dash import dcc, html
import dash_bootstrap_components as dbc

MODEL_OPTIONS = [
    {"label": "Qwen2.5-VL (baseline)", "value": "qwen"},
    {"label": "Monet-7B",              "value": "monet"},
    {"label": "LVR",                   "value": "lvr"},
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar() -> html.Div:
    return html.Div(
        className="sidebar",
        children=[

            html.Div([
                html.Div("VLM Explorer",      className="sidebar-title"),
                html.Div("Latent Reasoning",  className="sidebar-subtitle"),
            ]),

            html.Div("Model", className="sidebar-label"),
            dbc.Select(
                id="model-selector",
                options=MODEL_OPTIONS,
                value="qwen",
                className="sidebar-select mb-2",
            ),

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

            html.Div("Reasoning Step", className="sidebar-label"),
            dcc.Slider(
                id="step-slider",
                min=0, max=0, step=1, value=0,
                marks={},
                tooltip={"placement": "right", "always_visible": True},
                className="sidebar-slider mb-2",
            ),

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


# ── KPI strip ─────────────────────────────────────────────────────────────────
# def kpi_strip() -> dbc.Row:
    # cards = [
    #     ("Active Model",   "kpi-model", "kpi-card"),
    #     ("Current Step",   "kpi-step",  "kpi-card dark"),
    #     ("Token Type",     "kpi-type",  "kpi-card dark"),
    #     ("Instances",      "kpi-inst",  "kpi-card"),
    # ]
    # cols = []
    # for label, cid, cls in cards:
    #     cols.append(dbc.Col(
    #         html.Div(className=cls, children=[
    #             html.Div(label,             className="kpi-card-label"),
    #             html.Div("—", id=cid,      className="kpi-card-value"),
    #         ]),
    #         width=3,
    #     ))
    # return dbc.Row(cols, className="g-3 mb-3")


# ── Row 1: Input | Heatmap | Parameters+Eval ─────────────────────────────────
def row_one() -> dbc.Row:
    return dbc.Row(className="g-3 mb-3", children=[

        # ── HEATMAP (left, wider) ─────────────────────────────────────────
        dbc.Col(width=6, children=[
            html.Div(className="content-card h-100", children=[
                html.Div("Attention Heatmap",
                         className="content-card-title"),
                html.Div("Cross-attention to image per step",
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

        # ── PARAMETERS + EVALUATION (right) ──────────────────────────────
        dbc.Col(width=3, children=[
            html.Div(className="content-card mb-3 h-100", children=[
                html.Div("Parameters",
                         className="content-card-title"),
                html.Div("Current token info",
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


# ── Row 2: UMAP | t-SNE + Stats ──────────────────────────────────────────────
def row_two() -> dbc.Row:
    return dbc.Row(className="g-3", children=[

        # ── UMAP (larger left) ────────────────────────────────────────────
        dbc.Col(width=6, children=[
            html.Div(className="content-card", children=[
                html.Div("UMAP Projection",                     className="content-card-title"),
                html.Div("Token embedding space — draw a box to select", className="content-card-subtitle"),
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
            ]),
        ]),

        # ── t-SNE + Statistics (right) ────────────────────────────────────
        dbc.Col(width=6, children=[
            html.Div(className="content-card", children=[
                html.Div("t-SNE — Bounding Box Selection", className="content-card-title"),
                html.Div("Re-projection of selected region", className="content-card-subtitle"),
                dcc.Graph(
                    id="tsne-graph",
                    style={"height": "340px"},
                    figure={
                        "data": [],
                        "layout": {
                            "template":      "plotly_white",
                            "paper_bgcolor": "#ffffff",
                            "plot_bgcolor":  "#f4f7f5",
                        },
                    },
                ),
                html.Hr(className="card-divider"),
                html.Div("Selection Statistics", className="panel-label"),
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
                    html.H4("Latent Reasoning Explorer", className="page-title"),
                    # kpi_strip(),
                    row_one(),
                    row_two(),
                ],
            ),

            dcc.Store(id="store-instances",         data={}),
            dcc.Store(id="store-active-instance",   data=None),
            dcc.Store(id="store-corpus-embeddings", data={}),
            dcc.Store(id="store-current-image-b64", data=None),
        ],
    )