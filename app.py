import dash
import dash_bootstrap_components as dbc
from layout import build_layout
from callbacks import register_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "VLM Latent Reasoning Explorer"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                min-height: 100vh;
                background: #eef2f0 !important;
                margin: 0; padding: 0;
                font-family: Inter, sans-serif;
            }
        </style>
    </head>
    <body style="background:#eef2f0; min-height:100vh;">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = build_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9001)