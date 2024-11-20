import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
from dash import ALL, Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from plotly import colors
from pre_snap_prediction.data import process_orpsp
from pre_snap_prediction.modeling import orpsp_classification, route_time_regression
from pre_snap_prediction.plays_designer.utils import (
    classification_model,
    default_formations,
    encoder,
    find_intermediate_point,
    time_encoder,
    time_model,
)
from pre_snap_prediction.utils.constants import FIELD_LENGTH, FIELD_WIDTH
from pre_snap_prediction.visualization import Field


@callback(
    [Output("offensive_formation", "options"), Output("offensive_formation", "value")],
    Input("url", "pathname"),
)
def update_formation_selector(pathname: str) -> tuple[list, str]:
    """Updates the offensive formation selector dropdown based on the provided URL pathname.

    Parameters
    ----------
    pathname : str
        The current URL pathname, used as the input trigger for the callback.

    Returns
    -------
    tuple[list, str]
        A tuple containing:
        - A list of dictionaries with all formation options.
        - A string representing the default selected formation value.
    """
    formations = list(default_formations.keys())
    options = [{"label": value, "value": value} for value in formations]
    return options, formations[0]


@callback(
    [Output("receiver_alignment", "options"), Output("receiver_alignment", "value")],
    Input("offensive_formation", "value"),
)
def update_alignment_selector(formation: str) -> tuple[list, str]:
    """Updates the receiver alignment selector dropdown based on the provided offensive formation.

    Parameters
    ----------
    formation : str
        The current selected offensive formation value.

    Returns
    -------
    tuple[list, str]
        A tuple containing:
        - A list of dictionaries with all alignment options.
        - A string representing the default selected alignment value.
    """
    alignments = list(default_formations[formation].keys())
    options = [{"label": value, "value": value} for value in alignments]
    return options, alignments[0]


@callback(
    Output("play_editor", "style"),
    [Input("edit_play", "n_clicks"), Input("offensive_formation", "value"), Input("receiver_alignment", "value")],
)
def display_editor(n_clicks: int | None, formation: str, alignment: str) -> dict:
    """Controls the visibility of the play editor interface based on user interactions.

    Parameters
    ----------
    n_clicks : int | None
        The number of times the edit play button has been clicked.
    formation : str
        The current selected offensive formation value.
    alignment : str
        The current selected receiver alignment value.

    Returns
    -------
    dict
        A dictionary representing the CSS style for the play editor interface to controle the visibility.
    """
    if n_clicks is not None and ctx.triggered_id == "edit_play":
        return {}
    else:
        return {"display": "none"}


@callback(
    Output("route_type_container", "children"),
    Input("edit_play", "n_clicks"),
    [State("offensive_formation", "value"), State("receiver_alignment", "value")],
)
def create_route_type_selectors(n_clicks: int | None, formation: str, alignment: str) -> list:
    """Generates dropdown selectors for route types based on the offensive formation and receiver alignment.

    Parameters
    ----------
    n_clicks : int | None
        The number of times the edit play button has been clicked.
    formation : str
        The current selected offensive formation value.
    alignment : str
        The current selected receiver alignment value.

    Returns
    -------
    list
        A list of dbc.Row components, each containing a labeled dropdown for route type selection,
        corresponding to each receiver running a route.

    Raises
    ------
    PreventUpdate
        Preventing the callback from updating the output if n_clicks is None.
    """
    if n_clicks is not None:
        children = []
        nb_receiver = sum(default_formations[formation][alignment]["wasRunningRoute"])
        for i in range(1, nb_receiver + 1):
            children += [
                dbc.Row(
                    dbc.Col(
                        [
                            html.H4(
                                f"Receiver {i}", className="title", style={"color": colors.qualitative.Plotly[i - 1]}
                            ),
                            dcc.Dropdown(
                                options=[
                                    {"label": "straight", "value": "straight"},
                                    {"label": "45angle", "value": "45angle"},
                                    {"label": "90angle", "value": "90angle"},
                                ],
                                value="straight",
                                clearable=False,
                                id={"type": "route_type", "index": i},
                                className="text",
                                style={"max-width": 400},
                            ),
                        ]
                    ),
                    style={"margin-bottom": 20},
                )
            ]

        return children
    else:
        raise PreventUpdate


@callback(
    Output("zone_designer", "figure"),
    [
        Input("edit_play", "n_clicks"),
        Input("zone_designer", "relayoutData"),
        Input({"type": "route_type", "index": ALL}, "value"),
    ],
    [
        State("offensive_formation", "value"),
        State("receiver_alignment", "value"),
        State("zone_designer", "figure"),
    ],
)
def update_zone_designer(
    n_clicks: int | None, relayout_data: dict, route_types: list, formation: str, alignment: str, fig: go.Figure
) -> go.Figure:
    """Updates the zone designer figure to reflect current play configurations, receiver routes, receiver and zone adjustments.

    Parameters
    ----------
    n_clicks : int | None
        The number of times the edit play button has been clicked.
    relayout_data : dict
        Data containing changes made to the layout, such as dragged shapes in the figure.
    route_types : list
        List of selected route types for each receiver, corresponding to dropdown inputs.
    formation : str
        The current selected offensive formation value.
    alignment : str
        The current selected receiver alignment value.
    fig : go.Figure
        The current state of the zone designer figure to be updated.

    Returns
    -------
    go.Figure
        Updated Plotly figure reflecting the current configuration and adjustments.
    """
    if n_clicks is None:
        fig = go.Figure()
        fig.update_layout(
            title="Drag Reception Zones",
            xaxis={"range": [-10, 25], "fixedrange": True},
            yaxis={
                "range": [0, FIELD_WIDTH],
                "fixedrange": True,
                "scaleanchor": "x",
                "scaleratio": 1,
                "showgrid": False,
            },
            height=600,
        )

    elif len(ctx.triggered) > 1:
        data = []
        players = pl.DataFrame(default_formations[formation][alignment])
        non_receiver = players.filter(pl.col("wasRunningRoute") == 0)
        data.append(
            go.Scatter(
                x=non_receiver["x"].to_numpy(),
                y=non_receiver["y"].to_numpy(),
                mode="markers",
                marker={"size": 10, "color": "white"},
                name="non_receiver",
                showlegend=False,
                hoverinfo="none",
            )
        )

        fig = go.Figure(data=data)

        receiver = players.filter(pl.col("wasRunningRoute") == 1)
        for i in range(len(receiver)):
            fig.add_shape(
                type="circle",
                x0=receiver["x"][i] - 0.5,
                x1=receiver["x"][i] + 0.5,
                y0=receiver["y"][i] - 0.5,
                y1=receiver["y"][i] + 0.5,
                fillcolor=colors.qualitative.Plotly[i],
                line_color=colors.qualitative.Plotly[i],
                name=f"receiver_{i+1}",
            )

            fig.add_shape(
                type="circle",
                x0=receiver["x"][i] + 10 - 1.5,
                x1=receiver["x"][i] + 10 + 1.5,
                y0=receiver["y"][i] - 1.5,
                y1=receiver["y"][i] + 1.5,
                line_color=colors.qualitative.Plotly[i],
                name=f"reception_zone_{i+1}",
            )

            fig.add_trace(
                go.Scatter(
                    x=[receiver["x"][i], receiver["x"][i] + 5, receiver["x"][i] + 10],
                    y=[receiver["y"][i], receiver["y"][i], receiver["y"][i]],
                    mode="lines",
                    line_color=colors.qualitative.Plotly[i],
                    name=f"route_{i+1}",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        fig.update_layout(
            title="Drag Receivers and Reception Zones",
            xaxis={"range": [-10, 25], "fixedrange": True},
            yaxis={
                "range": [0, FIELD_WIDTH],
                "fixedrange": True,
                "scaleanchor": "x",
                "scaleratio": 1,
                "showgrid": False,
            },
            height=700,
        )

    elif ctx.triggered_id == "zone_designer":
        shapes = fig["layout"]["shapes"]

        for i, shape in enumerate(shapes):
            if f"shapes[{i}].x0" in relayout_data:
                shape["x0"] = relayout_data[f"shapes[{i}].x0"]
                shape["y0"] = relayout_data[f"shapes[{i}].y0"]
                shape["x1"] = relayout_data[f"shapes[{i}].x1"]
                shape["y1"] = relayout_data[f"shapes[{i}].y1"]

            if "receiver" in shape.get("name"):
                route_point = 0
                shape_size = 1
                shape["x0"] = min(shape["x0"], -0.5)
                shape["x1"] = min(shape["x1"], 0.5)
            else:
                route_point = 2
                shape_size = 3
                shape["x0"] = min(shape["x0"], 22)
                shape["x1"] = min(shape["x1"], 25)

            shape["x0"] = max(shape["x0"], -10)
            shape["x1"] = max(shape["x1"], -10 + shape_size)
            shape["y0"] = max(shape["y0"], 0)
            shape["y1"] = max(shape["y1"], 0 + shape_size)
            shape["y0"] = min(shape["y0"], FIELD_WIDTH - shape_size)
            shape["y1"] = min(shape["y1"], FIELD_WIDTH)

            position = next(
                (
                    index
                    for index, item in enumerate(fig["data"])
                    if item["name"].split("_")[-1] == shape.get("name").split("_")[-1]
                ),
                None,
            )

            center_x = (shape["x0"] + shape["x1"]) / 2
            center_y = (shape["y0"] + shape["y1"]) / 2

            if position is not None:
                fig["data"][position]["x"][route_point] = center_x
                fig["data"][position]["y"][route_point] = center_y

                xi, yi = find_intermediate_point(
                    fig["data"][position]["x"][0],
                    fig["data"][position]["y"][0],
                    fig["data"][position]["x"][2],
                    fig["data"][position]["y"][2],
                    route_types[int(shape.get("name").split("_")[-1]) - 1],
                )

                fig["data"][position]["x"][1] = xi
                fig["data"][position]["y"][1] = yi

    elif isinstance(ctx.triggered_id, dict) and ctx.triggered_id["type"] == "route_type":
        position = next(
            (
                index
                for index, item in enumerate(fig["data"])
                if item["name"].split("_")[-1] == str(ctx.triggered_id["index"])
            ),
            None,
        )

        if position is not None:
            xi, yi = find_intermediate_point(
                fig["data"][position]["x"][0],
                fig["data"][position]["y"][0],
                fig["data"][position]["x"][2],
                fig["data"][position]["y"][2],
                route_types[ctx.triggered_id["index"] - 1],
            )

            fig["data"][position]["x"][1] = xi
            fig["data"][position]["y"][1] = yi

    return fig


@callback(
    Output("absolute_yardline", "value"),
    Input("edit_play", "n_clicks"),
)
def update_absolute_yardline(n_clicks: int | None) -> int:
    """Reset the value of the absolute yardline input field.

    Parameters
    ----------
    n_clicks : int | None
        The number of times the edit play button has been clicked.

    Returns
    -------
    int
        The updated value of the absolute yardline, which is set to 60 by default.
    """
    return 60


@callback(
    Output("yards_to_go", "max"),
    Input("absolute_yardline", "value"),
)
def update_max_yards_to_go(absolute_yard_line: int) -> int:
    """Updates the maximum value for the yards to go input field based on the absolute yardline.

    Parameters
    ----------
    absolute_yard_line : int
        The current position of the ball on the field, expressed as an absolute yardline value.

    Returns
    -------
    int
        The maximum number of yards to go, calculated as the remaining distance to the goal line.

    Raises
    ------
    PreventUpdate
        Preventing the callback from updating the output if absolute_yard_line is None.
    """
    if absolute_yard_line is not None:
        return FIELD_LENGTH - 10 - absolute_yard_line
    else:
        raise PreventUpdate


@callback(
    Output("yards_to_go", "value"),
    [Input("edit_play", "n_clicks"), Input("yards_to_go", "max")],
    State("yards_to_go", "value"),
)
def update_yards_to_go(n_clicks: int | None, max_yards_to_go: int, yards_to_go: int) -> int:
    """Updates the yards to go input value based on user interaction and constraints.

    Parameters
    ----------
    n_clicks : int | None
        The number of times the edit play button has been clicked.
    max_yards_to_go : int
        The maximum allowable value for yards to go, determined by the field constraints.
    yards_to_go : int
        The current value of the yards to go input field.

    Returns
    -------
    int
        The updated yards to go value, reseted to 10 if edit play is clicked.

    Raises
    ------
    PreventUpdate
        If no relevant changes occur, no update is performed.
    """
    if n_clicks is None or ctx.triggered_id == "edit_play":
        return 10
    elif max_yards_to_go is not None and max_yards_to_go < yards_to_go:
        return max_yards_to_go
    else:
        raise PreventUpdate


@callback(
    Output("down", "value"),
    Input("edit_play", "n_clicks"),
)
def update_down(n_clicks: int | None) -> int:
    """Reset the value of the down input field.

    Parameters
    ----------
    n_clicks : int | None
        The number of times the edit play button has been clicked.

    Returns
    -------
    int
        The updated value of the down, which is set to 1 by default.
    """
    return 1


@callback(
    Output("play_visualization", "figure"),
    [Input("edit_play", "n_clicks"), Input("compute_play", "n_clicks")],
    [
        State("offensive_formation", "value"),
        State("receiver_alignment", "value"),
        State("zone_designer", "figure"),
        State({"type": "route_type", "index": ALL}, "value"),
        State("absolute_yardline", "value"),
        State("yards_to_go", "value"),
        State("down", "value"),
    ],
)
def update_play_visualization(
    edit_play: int | None,
    compute_play: int | None,
    formation: str,
    alignment: str,
    zone_designer: dict,
    route_types: list,
    absolute_yardline: int,
    yards_to_go: int,
    down: int,
) -> go.Figure:
    """Updates the play visualization based on the given inputs, including offensive formations,
    receiver alignments, designed zones, route types, and play-specific data.

    Parameters
    ----------
    edit_play : int | None
        The number of times the edit play button has been clicked.
    compute_play : int | None
        The number of times the compute play button has been clicked.
    formation : str
        The current selected offensive formation value.
    alignment : str
        The current selected receiver alignment value.
    zone_designer : dict
        The current zone designer figure.
    route_types : list
        List of the current route types assigned to each receiver.
    absolute_yardline : int
        The current value of the absolute yardline input field.
    yards_to_go : int
        The current value of the yards to go input field.
    down : int
        The current value of the down input field.

    Returns
    -------
    go.Figure
        A Plotly figure representing the play visualization, including the field,
        player positions, routes, and reception zones.
    """
    if edit_play is None or ctx.triggered_id == "edit_play":
        fig = Field(is_animated=False).fig

    else:
        data = pl.DataFrame(default_formations[formation][alignment])
        data = data.with_columns(
            pl.lit(formation).alias("offenseFormation"),
            pl.lit(alignment).alias("receiverAlignment"),
            pl.lit(absolute_yardline).alias("absoluteYardlineNumber"),
            pl.lit(yards_to_go).alias("yardsToGo"),
            pl.lit(down).alias("down"),
            pl.lit(False).alias("playAction"),
            pl.lit("right").alias("playDirection"),
            pl.lit(0).alias("gameId"),
            pl.lit(0).alias("playId"),
            pl.Series("nflId", range(1, len(data) + 1)),
            pl.lit(1).alias("frameId"),
            pl.lit("BEFORE_SNAP").alias("frameType"),
            pl.lit(False).alias("is_defense"),
            pl.lit("NFL").alias("club"),
            pl.Series("displayName", [f"Player {i}" for i in range(1, len(data) + 1)]),
            pl.lit(None).alias("x_recep_zone"),
            pl.lit(None).alias("y_recep_zone"),
            pl.lit(None).alias("xi"),
            pl.lit(None).alias("yi"),
            pl.lit(absolute_yardline).alias("x_ball"),
            pl.lit(FIELD_WIDTH / 2).alias("y_ball"),
        )

        receiver = data.filter(pl.col("wasRunningRoute") == 1)
        receiver = receiver.with_columns(
            pl.Series("receiverId", range(1, len(receiver) + 1)),
            pl.Series("route_type", route_types),
            pl.lit(0).alias("cluster"),
        )

        data = data.join(
            receiver.select(["displayName", "receiverId", "route_type", "cluster"]),
            on=["displayName"],
            how="left",
        )

        shapes = zone_designer["layout"]["shapes"]

        for shape in shapes:
            shape_name = shape.get("name").split("_")
            if shape_name[0] == "receiver":
                data = data.with_columns(
                    pl.when(pl.col("receiverId") == int(shape_name[-1]))
                    .then((shape["x0"] + shape["x1"]) / 2)
                    .otherwise(pl.col("x"))
                    .alias("x"),
                    pl.when(pl.col("receiverId") == int(shape_name[-1]))
                    .then((shape["y0"] + shape["y1"]) / 2)
                    .otherwise(pl.col("y"))
                    .alias("y"),
                )
            elif shape_name[0] == "reception":
                data = data.with_columns(
                    pl.when(pl.col("receiverId") == int(shape_name[-1]))
                    .then((shape["x0"] + shape["x1"]) / 2)
                    .otherwise(pl.col("x_recep_zone"))
                    .alias("x_recep_zone"),
                    pl.when(pl.col("receiverId") == int(shape_name[-1]))
                    .then((shape["y0"] + shape["y1"]) / 2)
                    .otherwise(pl.col("y_recep_zone"))
                    .alias("y_recep_zone"),
                )

        scatters = zone_designer["data"]
        routes_traces = []
        for scatter in scatters:
            scatter_name = scatter.get("name").split("_")
            if scatter_name[0] == "route":
                data = data.with_columns(
                    pl.when(pl.col("receiverId") == int(scatter_name[-1]))
                    .then(scatter["x"][1])
                    .otherwise(pl.col("xi"))
                    .alias("xi"),
                    pl.when(pl.col("receiverId") == int(scatter_name[-1]))
                    .then(scatter["y"][1])
                    .otherwise(pl.col("yi"))
                    .alias("yi"),
                )

                routes_traces += [
                    go.Scatter(
                        x=[x + absolute_yardline for x in scatter["x"]],
                        y=scatter["y"],
                        mode="lines",
                        line={
                            "color": "Teal",
                            "width": 3,
                        },
                        opacity=0.5,
                        hoverinfo="none",
                        showlegend=False,
                    )
                ]

        data = data.with_columns(
            pl.when((pl.col("x_recep_zone") < 1) & (pl.col("y_recep_zone") - pl.col("y") < 5))
            .then(pl.lit("screen"))
            .when(pl.col("x_recep_zone") < 3)
            .then(pl.lit("flat"))
            .when((pl.col("route_type") == "straight") & (pl.col("x_recep_zone") < 10))
            .then(pl.lit("shortstraight"))
            .when(
                (pl.col("route_type") == "45angle") & (pl.col("xi") - pl.col("x") > 10) & (pl.col("yi") == pl.col("y"))
            )
            .then(pl.lit("late45angle"))
            .when(pl.col("route_type") == "45angle")
            .then(pl.lit("early45angle"))
            .otherwise(pl.col("route_type"))
            .alias("route_mode")
        )

        data = data.with_columns(
            (pl.col("x") + absolute_yardline).alias("x"),
            (pl.col("x_recep_zone") + absolute_yardline).alias("x_recep_zone"),
        )

        data = data.with_columns(
            (pl.col("x_recep_zone") - pl.col("x") - 3).alias("relative_x_min"),
            (pl.col("x_recep_zone") - pl.col("x") + 3).alias("relative_x_max"),
            (pl.col("y_recep_zone") - pl.col("y") - 3).alias("relative_y_min"),
            (pl.col("y_recep_zone") - pl.col("y") + 3).alias("relative_y_max"),
        )

        data = pl.concat(
            [data, pl.DataFrame({"x": [float(absolute_yardline)], "y": [FIELD_WIDTH / 2], "frameId": [1]})],
            how="diagonal_relaxed",
        )

        orpsp_features = data.filter(pl.col("wasRunningRoute") == 1)
        orpsp_features = process_orpsp.compute_orpsp_features(orpsp_features)

        time_data = route_time_regression.select_route_time_features(orpsp_features)
        encoded_time_data = route_time_regression.transform_encoder(time_data, time_encoder)
        predictions = route_time_regression.predict_route_time(encoded_time_data, time_model)

        data = data.join(
            predictions,
            on=["gameId", "playId", "nflId"],
            how="left",
        )
        orpsp_features = orpsp_features.join(
            predictions,
            on=["gameId", "playId", "nflId"],
            how="left",
        )

        features = orpsp_classification.select_orpsp_features(orpsp_features)
        features_encoded = orpsp_classification.transform_encoder(features, encoder)
        orpsp_predictions = orpsp_classification.predict_orpsp(features_encoded, classification_model)

        data = data.join(
            orpsp_predictions,
            on=["gameId", "playId", "nflId"],
            how="left",
        )

        field = Field(is_animated=False)
        field.fig.add_traces(routes_traces)
        field.create_animation(data)

        fig = field.fig

    return fig
