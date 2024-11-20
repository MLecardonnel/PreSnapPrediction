import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
from pre_snap_prediction.plays_designer import callbacks  # noqa: F401
from pre_snap_prediction.plays_designer.layout import header, play_editor, play_selection

app: Dash = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="NFL Plays Designer",
    suppress_callback_exceptions=True,
)

server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=True),
        html.Div(header),
        html.Div(play_selection, style={"margin-left": 20, "margin-right": 20}),
        html.Div(play_editor, style={"margin-left": 20, "margin-right": 20, "margin-top": 20}),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False, port=8052, threaded=True)
