import dash_bootstrap_components as dbc
from dash import dcc, html

header = (
    html.Header(
        dbc.Row(
            [
                html.H1(
                    ["NFL Plays ", html.Span("Designer", className="highlight")],
                    className="title",
                    style={"text-align": "center"},
                ),
            ],
            justify="center",
            align="center",
            style={"minHeight": 100},
        ),
        id="header",
    ),
)


play_selection = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Formation", className="title"),
                            dcc.Dropdown(
                                clearable=False,
                                id="offensive_formation",
                                className="text",
                                style={"max-width": 400},
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H4("Alignment", className="title"),
                            dcc.Dropdown(
                                clearable=False,
                                id="receiver_alignment",
                                className="text",
                                style={"max-width": 400},
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [dbc.Button("Edit Play", id="edit_play", size="lg")],
                        className="text-center",
                        width=4,
                    ),
                ],
                align="center",
            ),
        ]
    )
)


play_editor = dbc.Col(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id="zone_designer", config={"edits": {"shapePosition": True}}, style={"max-width": 500}),
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(dbc.CardBody(dbc.Row([], id="route_type_container"))),
                        dbc.Card(
                            dbc.CardBody(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.H4("Absolute Yardline", className="title"),
                                                dcc.Input(
                                                    value=60,
                                                    type="number",
                                                    min=11,
                                                    max=109,
                                                    required=True,
                                                    id="absolute_yardline",
                                                    className="text",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4("Yards To Go", className="title"),
                                                dcc.Input(
                                                    value=10,
                                                    type="number",
                                                    min=1,
                                                    required=True,
                                                    id="yards_to_go",
                                                    className="text",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                        dbc.Col(
                                            [
                                                html.H4("Down", className="title"),
                                                dcc.Input(
                                                    value=1,
                                                    type="number",
                                                    min=1,
                                                    max=4,
                                                    required=True,
                                                    id="down",
                                                    className="text",
                                                ),
                                            ],
                                            width=4,
                                        ),
                                    ]
                                )
                            ),
                            style={"margin-top": 20},
                        ),
                    ],
                    width=6,
                ),
            ],
            align="center",
        ),
        dbc.Row(
            dbc.Col(
                [dbc.Button("Compute Play", id="compute_play", size="lg", style={"margin-left": 20})],
            ),
            style={"margin-top": 50},
        ),
        dbc.Row(
            dcc.Graph(id="play_visualization"),
        ),
    ],
    id="play_editor",
)
