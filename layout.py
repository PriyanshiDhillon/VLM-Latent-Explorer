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
    {"label": "Qwen2.5-VL", "value": "qwen", "short": "Qwen"},
    {"label": "Monet-7B", "value": "monet", "short": "Monet"},
    {"label": "LVR", "value": "lvr", "short": "LVR"},
]

EXPERIMENT_OPTIONS = [
    {
        "label": "Normal inference",
        "value": "normal",
    },
    {
        "label": "Latent bottleneck",
        "value": "latent_bottleneck",
    },
    {
        "label": "Image mask",
        "value": "image_mask",
    },
]


def step_slider_bar() -> html.Div:
    return html.Div(
        className="global-step-bar",
        children=[
            html.Div("Reasoning Step", className="step-bar-label"),
            dcc.Slider(
                id="step-slider",
                min=0,
                max=0,
                step=1,
                value=0,
                marks={},
                tooltip={"placement": "bottom", "always_visible": False},
                className="center-slider global-step-slider",
            ),
        ],
    )


def experiment_control_panel() -> html.Div:
    return html.Div(
        className="experiment-control-panel glass-panel",
        children=[
            html.Div("Choose a model", className="section-title"),
            html.Div("Select the backbone before running experiments.", className="section-subtitle"),
            html.Div(
                className="model-card-row",
                children=[
                    html.Button(
                        option["short"],
                        id={"type": "model-card", "index": option["value"]},
                        className="model-card",
                        n_clicks=0,
                        title=option["label"],
                    )
                    for option in MODEL_OPTIONS
                ],
            ),
            html.Div(id="selected-model-label", className="selected-model-label", children="No model selected"),
            dbc.Select(
                id="model-selector",
                options=[{"label": o["label"], "value": o["value"]} for o in MODEL_OPTIONS],
                value=None,
                placeholder="Select a model",
                className="hidden-control",
            ),
            dbc.Button(
                "Load selected model",
                id="load-model-btn",
                className="load-model-btn",
                color="secondary",
                size="sm",
            ),
            dcc.Loading(
                type="circle",
                children=html.Div(id="model-load-status", className="model-load-status"),
            ),
            dbc.Progress(id="model-load-progress", value=0, style={"display": "none"}),
            html.Div("Choose an experiment", className="section-title experiment-choice-title"),
            html.Div("Choose how the model should be probed.", className="section-subtitle"),
            dcc.RadioItems(
                id="experiment-selector",
                options=EXPERIMENT_OPTIONS,
                value="normal",
                className="experiment-radio",
                inputClassName="experiment-radio-input",
                labelClassName="experiment-radio-label",
            ),
            html.Div(
                "Latent bottleneck lets answer text read question tokens, generated latent tokens, "
                "and previous answer tokens, but blocks direct image-patch access.",
                className="experiment-note",
            ),
        ],
    )


def input_panel() -> html.Div:
    return html.Div(
        className="input-panel glass-panel",
        children=[
            html.Div("Input", className="section-title"),
            html.Div("Image", className="input-label"),
            dcc.Upload(
                id="image-upload",
                children=html.Div(["Drag & Drop or ", html.A("select file", className="upload-link")]),
                className="upload-box mb-2",
            ),
            html.Img(
                id="uploaded-image-preview",
                className="preview-img mb-2",
                style={"display": "none"},
            ),
            html.Div("Question", className="input-label"),
            dbc.Textarea(
                id="question-input",
                placeholder="Type your question...",
                rows=4,
                className="question-input",
            ),
            dbc.Textarea(
                id="intervention-question-input",
                placeholder="Hidden intervention question mirror",
                rows=1,
                className="hidden-control",
            ),
            html.Div(
                id="mask-panel",
                className="mask-panel",
                style={"display": "none"},
                children=[
                    html.Div("Image mask", className="input-label input-label-muted"),
                    html.Div("Draw a box on the image to black out a region.", className="experiment-note"),
                    dcc.Graph(
                        id="mask-image-graph",
                        config={
                            "modeBarButtonsToAdd": ["select2d"],
                            "modeBarButtonsToRemove": [
                                "zoom2d",
                                "pan2d",
                                "zoomIn2d",
                                "zoomOut2d",
                                "autoScale2d",
                                "resetScale2d",
                                "lasso2d",
                                "toImage",
                            ],
                            "displayModeBar": True,
                            "scrollZoom": False,
                            "displaylogo": False,
                        },
                        style={"height": "160px"},
                        figure=_BLANK_FIGURE,
                    ),
                ],
            ),
            dbc.Button("▶  Run experiment", id="run-btn", className="run-btn experiment-run-btn"),
            dbc.Button(
                "▶  Run Latent Bottleneck",
                id="run-intervention-btn",
                className="hidden-control",
                n_clicks=0,
            ),
        ],
    )


def welcome_page() -> html.Section:
    return html.Section(
        id="welcome-page",
        className="story-page welcome-page",
        children=[
            html.Div(
                className="welcome-copy",
                children=[
                    html.Div("Welcome to the", className="welcome-kicker"),
                    html.H1("Visual Latent Reasoning Explorer"),
                    html.Div(className="welcome-orb", children="◌"),
                ],
            ),
            html.A("→", href="#experiments-page", className="page-arrow page-arrow-right", title="Go to experiments"),
        ],
    )


def experiments_page() -> html.Section:
    return html.Section(
        id="experiments-page",
        className="story-page experiments-page",
        children=[
            html.Div(
                className="experiments-grid",
                children=[
                    html.Div(className="experiments-title", children="Experiments"),
                    experiment_control_panel(),
                    input_panel(),
                    html.Div(
                        className="experiments-nav-rail",
                        children=html.A(
                            "To dashboard →",
                            href="#dashboard-page",
                            className="page-cta dashboard-cta",
                        ),
                    ),
                ],
            ),
            dbc.Select(id="example-selector", options=[], value=None, className="hidden-control"),
        ],
    )


def attention_row() -> dbc.Row:
    return dbc.Row(
        className="g-3 mb-3",
        children=[
            dbc.Col(
                width=12,
                children=[
                    html.Div(
                        className="content-card",
                        children=[
                            html.Div(
                                className="d-flex justify-content-between align-items-start",
                                children=[
                                    html.Div(
                                        [
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
                                                            html.Span("Decoder layer", className="attention-layer-label"),
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
                                        ]
                                    ),
                                    html.Div(
                                        className="token-type-legend",
                                        children=[
                                            html.Span("T · text", className="token-type-chip token-type-chip--text"),
                                            html.Span("L · latent", className="token-type-chip token-type-chip--latent"),
                                            html.Span("V · visual", className="token-type-chip token-type-chip--visual"),
                                        ],
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="token-attention-matrix",
                                config={"displayModeBar": True, "scrollZoom": True},
                                style={"height": "520px"},
                                figure={"data": [], "layout": {"template": "plotly_white"}},
                            ),
                        ],
                    )
                ],
            )
        ],
    )


def row_one() -> dbc.Row:
    return dbc.Row(
        className="g-3 mb-3",
        children=[
            dbc.Col(
                width=6,
                children=[
                    html.Div(
                        className="content-card h-100",
                        children=[
                            html.Div("Attention Heatmap", className="content-card-title"),
                            html.Div("Cross-attention overlay on image per reasoning step", className="content-card-subtitle"),
                            html.Div(
                                html.Img(id="heatmap-image", src="", className="heatmap-img"),
                                className="heatmap-wrapper",
                            ),
                        ],
                    )
                ],
            ),
            dbc.Col(
                width=3,
                children=[
                    html.Div(
                        className="content-card mb-3 h-100",
                        children=[
                            html.Div("Token Parameters", className="content-card-title"),
                            html.Div("Info for the token at the current step", className="content-card-subtitle"),
                            html.Div(id="param-display", className="param-display"),
                        ],
                    )
                ],
            ),
            dbc.Col(
                width=3,
                children=[
                    html.Div(
                        className="content-card h-100",
                        children=[
                            html.Div("Evaluation", className="content-card-title"),
                            html.Div("Per-instance output snippets", className="content-card-subtitle"),
                            html.Div(id="eval-display", className="eval-display", children="No instances yet."),
                        ],
                    )
                ],
            ),
        ],
    )


def row_two() -> dbc.Row:
    return dbc.Row(
        className="g-3 mb-3",
        children=[
            dbc.Col(
                width=12,
                children=[
                    html.Div(
                        className="content-card",
                        children=[
                            html.Div(
                                className="d-flex justify-content-between align-items-start mb-1",
                                children=[
                                    html.Div(
                                        [
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
                                        ]
                                    ),
                                    dbc.ButtonGroup(
                                        [
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
                                        ],
                                        className="proj-toggle-group",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="d-flex flex-wrap align-items-center gap-3 mb-1",
                                style={"fontSize": "0.78rem", "color": "#374151"},
                                children=[
                                    html.Div(
                                        [
                                            html.Span("Highlight:", style={"color": "#6b7280", "marginRight": "5px"}),
                                            dcc.RadioItems(
                                                id="highlight-mode",
                                                options=[
                                                    {"label": "All tokens", "value": "all"},
                                                    {"label": "Up to step", "value": "step"},
                                                ],
                                                value="all",
                                                inline=True,
                                                inputStyle={"marginRight": "3px"},
                                                labelStyle={"marginRight": "10px"},
                                            ),
                                        ],
                                        className="d-flex align-items-center",
                                    ),
                                    html.Div(
                                        [
                                            html.Span("Show trace:", style={"color": "#6b7280", "marginRight": "5px"}),
                                            dcc.Checklist(
                                                id="umap-show-trace",
                                                options=[{"label": "", "value": "trace"}],
                                                value=["trace"],
                                                inputStyle={"cursor": "pointer"},
                                            ),
                                        ],
                                        className="d-flex align-items-center",
                                    ),
                                    html.Div(
                                        [
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
                                        ],
                                        className="d-flex align-items-center",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="d-flex align-items-center gap-2 mb-2",
                                children=[
                                    html.Span("Corpus:", style={"fontSize": "0.78rem", "color": "#6b7280", "whiteSpace": "nowrap"}),
                                    html.Span(
                                        id="corpus-selection-count",
                                        children="",
                                        style={"fontSize": "0.78rem", "color": "#374151", "whiteSpace": "nowrap"},
                                    ),
                                    dbc.Button(
                                        "Select All",
                                        id="btn-corpus-all",
                                        size="sm",
                                        color="secondary",
                                        outline=True,
                                        style={"fontSize": "0.72rem", "padding": "1px 8px"},
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Deselect All",
                                        id="btn-corpus-none",
                                        size="sm",
                                        color="secondary",
                                        outline=True,
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
                                        "template": "plotly_white",
                                        "paper_bgcolor": "#ffffff",
                                        "plot_bgcolor": "#f4f7f5",
                                    },
                                },
                            ),
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
                        ],
                    )
                ],
            )
        ],
    )


def row_three() -> dbc.Row:
    return dbc.Row(
        className="g-3",
        children=[
            dbc.Col(
                width=6,
                children=[
                    html.Div(
                        className="content-card",
                        children=[
                            html.Div("Token Context — Nearest Corpus Neighbours", className="content-card-title"),
                            html.Div("10 closest corpus text tokens to the current step in UMAP space", className="content-card-subtitle"),
                            html.Div(id="token-context-panel", className="token-context-panel"),
                        ],
                    )
                ],
            ),
            dbc.Col(
                width=6,
                children=[
                    html.Div(
                        className="content-card",
                        children=[
                            html.Div("Selection Statistics", className="content-card-title"),
                            html.Div("Aggregate metrics for the tokens in the selected region", className="content-card-subtitle"),
                            html.Div(id="stats-display", className="stats-display"),
                        ],
                    )
                ],
            ),
        ],
    )


def dashboard_page() -> html.Section:
    return html.Section(
        id="dashboard-page",
        className="story-page dashboard-page",
        children=[
            html.Div(
                className="dashboard-shell",
                children=[
                    html.Div(
                        className="dashboard-rail",
                        children=[
                            html.Div(
                                className="instances-card glass-panel",
                                children=[
                                    html.Div("Instances", className="section-title"),
                                    html.Div(id="instance-list", className="instance-list"),
                                ],
                            ),
                            html.A("← To experiments", href="#experiments-page", className="page-cta back-cta"),
                            html.Div(
                                className="dashboard-trace-card glass-panel",
                                children=[
                                    html.Div("Reasoning trace", className="section-title"),
                                    html.Div(id="reasoning-trace", className="reasoning-trace"),
                                    html.Div(
                                        id="intervention-reasoning-trace",
                                        className="hidden-control",
                                        children="Run a latent bottleneck experiment to see its trace here.",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="dashboard-workspace",
                        children=[
                            html.Div(className="dashboard-top", children=[step_slider_bar()]),
                            row_one(),
                            attention_row(),
                            row_two(),
                            row_three(),
                        ],
                    ),
                ],
            )
        ],
    )


def build_layout() -> html.Div:
    return html.Div(
        className="dashboard-root paged-root",
        children=[
            html.Div(
                className="page-deck",
                children=[
                    welcome_page(),
                    experiments_page(),
                    dashboard_page(),
                ],
            ),
            dcc.Store(id="store-instances", data={}),
            dcc.Store(id="store-active-instance", data=None),
            dcc.Store(id="store-corpus-embeddings", data={}),
            dcc.Store(id="store-current-image-b64", data=None),
            dcc.Store(id="store-active-projection", data="umap"),
            dcc.Store(id="store-umap-base", data=None),
            dcc.Store(id="store-mask-region", data=None),
        ],
    )
