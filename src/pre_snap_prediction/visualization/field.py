import copy
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib.cm import Reds
from matplotlib.colors import to_hex
from pre_snap_prediction.utils.constants import FIELD_LENGTH, FIELD_WIDTH

animations_path = (Path(__file__).parents[3] / "reports/animations").as_posix() + "/"


class Field:
    """Class for visualizing a football field with tracking data for a play"""

    def __init__(
        self,
        field_width: float = FIELD_WIDTH,
        field_length: float = FIELD_LENGTH,
        step_duration: int = 50,
        is_animated: bool = True,
    ):
        """Initialize the Field object.

        Parameters
        ----------
        field_width : float, optional
            Width of the football field, by default 53.3
        field_length : float, optional
            Length of the football field, by default 120.0
        step_duration : int, optional
            Duration for animation steps, by default 50
        is_animated : bool, optional
            Is the figure animated, by default True
        """
        self.field_width = field_width
        self.field_length = field_length
        self.field_subdivision = field_length / 12

        self.step_duration = step_duration

        self.is_animated = is_animated

        self.color_field = "#96B78C"
        self.color_endzone = "#6F976D"
        self.color_lines = "white"

        self.fig = go.Figure()

        self._draw_rectangle_on_field(x0=0, y0=0, x1=field_length, y1=field_width, color=self.color_field)
        self._draw_rectangle_on_field(x0=0, y0=0, x1=self.field_subdivision, y1=field_width, color=self.color_endzone)
        self._draw_rectangle_on_field(
            x0=field_length - self.field_subdivision,
            y0=0,
            x1=field_length,
            y1=field_width,
            color=self.color_endzone,
        )

        self._draw_numbers_on_field(field_width - 5)
        self._draw_numbers_on_field(5)

        for i in range(2, 23):
            self._draw_line_on_field(
                x=i * self.field_subdivision / 2,
                color=self.color_lines,
                width=2 if i % 2 == 0 else 1,
            )

        self.fig.update_layout(
            xaxis={"range": [-5, field_length + 5], "visible": False},
            yaxis={"range": [-5, field_width + 5], "visible": False, "scaleanchor": "x", "scaleratio": 1},
            height=600,
            updatemenus=(
                [
                    {
                        "buttons": [
                            {
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": self.step_duration, "redraw": False},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": "⏵",
                                "method": "animate",
                            },
                            {
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": "⏸",
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top",
                    }
                ]
                if self.is_animated
                else None
            ),
            sliders=(
                [
                    {
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 20},
                            "prefix": "frameId: ",
                            "visible": True,
                            "xanchor": "right",
                        },
                        "transition": {"duration": self.step_duration, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [],
                    }
                ]
                if self.is_animated
                else None
            ),
        )

    def _draw_numbers_on_field(self, y: float) -> None:
        numbers_on_field = ["10", "20", "30", "40", "50", "40", "30", "20", "10"]

        for i in range(len(numbers_on_field)):
            self.fig.add_shape(
                x0=(i + 2) * self.field_subdivision,
                x1=(i + 2) * self.field_subdivision,
                y0=y,
                y1=y,
                layer="below",
                label={
                    "text": numbers_on_field[i],
                    "font": {
                        "size": 25,
                        "color": self.color_lines,
                    },
                },
                opacity=0,
            )

    def _draw_rectangle_on_field(self, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
        self.fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line_width=0, layer="below", fillcolor=color)

    def _draw_line_on_field(self, x: float, color: str, width: int) -> None:
        self.fig.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=self.field_width,
            layer="below",
            line={
                "color": color,
                "width": width,
            },
        )

    def _create_step(self, frame_id: str) -> dict:
        step = {
            "args": [
                [frame_id],
                {
                    "frame": {"duration": self.step_duration, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": frame_id,
            "method": "animate",
        }
        return step

    def _draw_scrimmage_and_first_down(self, absoluteYardlineNumber: int, yardsToGo: int, playDirection: str) -> None:
        """Draw scrimmage line and first down marker on the football field.

        Parameters
        ----------
        absoluteYardlineNumber : int
            Absolute yardline number on the football field
        yardsToGo : int
            Yards to go for a first down
        playDirection : str
            Direction of play
        """
        if playDirection == "right":
            yard_line_first_down = absoluteYardlineNumber + yardsToGo
        elif playDirection == "left":
            yard_line_first_down = absoluteYardlineNumber - yardsToGo
        else:
            raise ValueError

        self._draw_line_on_field(absoluteYardlineNumber, "#0070C0", 2)
        self._draw_line_on_field(yard_line_first_down, "#E9D11F", 2)

    def _draw_reception_zone(
        self,
        x: float,
        y: float,
        relative_x_min: float,
        relative_x_max: float,
        relative_y_min: float,
        relative_y_max: float,
        route_time_mean: float,
    ) -> None:
        if x + relative_x_min > self.field_length:
            x1 = self.field_length + (relative_x_max - relative_x_min) / 2
            x0 = self.field_length - (relative_x_max - relative_x_min) / 2
        elif x + relative_x_min < 0:
            x1 = (relative_x_max - relative_x_min) / 2
            x0 = -(relative_x_max - relative_x_min) / 2
        else:
            x1 = x + relative_x_max
            x0 = x + relative_x_min

        if y + relative_y_min > self.field_width:
            y1 = self.field_width + (relative_y_max - relative_y_min) / 2
            y0 = self.field_width - (relative_y_max - relative_y_min) / 2
        elif y + relative_y_min < 0:
            y1 = (relative_y_max - relative_y_min) / 2
            y0 = -(relative_y_max - relative_y_min) / 2
        else:
            y1 = y + relative_y_max
            y0 = y + relative_y_min

        self.fig.add_shape(
            type="circle",
            layer="below",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            opacity=0.7,
            fillcolor="PaleTurquoise",
            line_color="LightSeaGreen",
            label={
                "text": f"~{round(route_time_mean,1)}s",
                "padding": abs(relative_x_max - relative_x_min) * 5,
                "font": {"color": "Teal"},
            },
        )

    def _create_scatter_route(self, route_tracking: pl.DataFrame) -> go.Scatter:
        route_tracking = route_tracking.with_columns(pl.col("frameId").rank().alias("route_frameId"))
        route_tracking = route_tracking.filter(pl.col("route_frameId") <= pl.col("route_frameId_mean"))

        return go.Scatter(
            x=route_tracking["x"].to_numpy(),
            y=route_tracking["y"].to_numpy(),
            mode="lines",
            line={
                "color": "Teal",
                "width": 3,
            },
            opacity=0.5,
            hoverinfo="none",
            showlegend=False,
        )

    def _get_color(self, value: float) -> str:
        return to_hex(Reds(value))

    def _create_orpsp_card(self, nb_cards: int, card: int, displayName: str, orpsp: float):
        card_prop = self.field_width / nb_cards
        card_center = card_prop * card - card_prop / 2

        self.fig.add_shape(
            type="rect",
            layer="below",
            x0=-30,
            x1=-5,
            y0=card_center - (card_prop / 2 - 1),
            y1=card_center + (card_prop / 2 - 1),
            opacity=0.7,
            fillcolor=self._get_color(orpsp),
            line_color="Black",
            label={
                "text": f"{displayName}<br><b>ORPSP:</b> {round(orpsp, 2)}",
                "font": {"color": "Black"},
            },
        )

    def create_animation(self, play_tracking: pl.DataFrame) -> None:
        """Create an animation for player movements during a play.

        Parameters
        ----------
        play_tracking : pl.DataFrame
            A Polars DataFrame containing tracking data for players during the play.
        """
        self._draw_scrimmage_and_first_down(
            **play_tracking.select(["absoluteYardlineNumber", "yardsToGo", "playDirection"]).rows(named=True)[0]
        )

        routes_traces = []
        if "cluster" in play_tracking.columns:
            route_players = (
                play_tracking.filter(pl.col("frameType") == "BEFORE_SNAP", pl.col("cluster").is_not_null())
                .group_by(["gameId", "playId", "nflId"])
                .last()
            )

            for i in range(len(route_players)):
                x, y, relative_x_min, relative_x_max, relative_y_min, relative_y_max, route_time_mean = (
                    route_players.select(
                        [
                            "x",
                            "y",
                            "relative_x_min",
                            "relative_x_max",
                            "relative_y_min",
                            "relative_y_max",
                            "route_time_mean",
                        ]
                    ).row(i)
                )
                self._draw_reception_zone(
                    x, y, relative_x_min, relative_x_max, relative_y_min, relative_y_max, route_time_mean
                )

            routes_tracking = play_tracking.filter(pl.col("frameType") == "AFTER_SNAP", pl.col("cluster").is_not_null())

            for player in routes_tracking["nflId"].unique():
                route_tracking = routes_tracking.filter(pl.col("nflId") == player)
                routes_traces.append(self._create_scatter_route(route_tracking))

        if "orpsp" in play_tracking.columns:
            receiver_first_frame = (
                play_tracking.filter(pl.col("frameType") == "BEFORE_SNAP", pl.col("orpsp").is_not_null())
                .group_by(["gameId", "playId", "nflId"])
                .last()
            )
            receiver_first_frame = receiver_first_frame.sort("y")
            for i in range(len(receiver_first_frame)):
                self._create_orpsp_card(
                    len(receiver_first_frame),
                    i + 1,
                    receiver_first_frame["displayName"][i],
                    receiver_first_frame["orpsp"][i],
                )

        frames = []
        steps = []
        for frame_id in play_tracking["frameId"].unique():
            frame_tracking = play_tracking.filter(pl.col("frameId") == frame_id)

            ball_tracking = frame_tracking.filter(pl.col("nflId").is_null())

            players_tracking = frame_tracking.filter(pl.col("nflId").is_not_null())
            defense_tracking = players_tracking.filter(pl.col("is_defense"))
            offense_tracking = players_tracking.filter(~pl.col("is_defense"))

            data = [] + routes_traces

            data.append(
                go.Scatter(
                    x=defense_tracking["x"].to_numpy(),
                    y=defense_tracking["y"].to_numpy(),
                    mode="markers",
                    marker={"size": 10, "color": "black"},
                    name="defense",
                    hoverinfo="none",
                ),
            )

            if "orpsp" in offense_tracking.columns:
                offense_tracking = offense_tracking.with_columns(
                    pl.col("orpsp").fill_null(0).map_elements(self._get_color, return_dtype=str).alias("hex_color")
                )

                marker = {
                    "size": 10,
                    "color": offense_tracking["hex_color"].to_numpy(),
                    "cmin": 0,
                    "cmax": 1,
                    "colorscale": "Reds",
                    "colorbar": dict(title="offense", thickness=10, x=1.03, y=0.4, len=0.85),
                }
                showlegend = False
                customdata = np.stack(
                    (
                        offense_tracking["nflId"].to_numpy(),
                        offense_tracking["displayName"].to_numpy(),
                        offense_tracking["club"].to_numpy(),
                        offense_tracking["orpsp"].fill_null(0).round(2).to_numpy(),
                    ),
                    axis=-1,
                )
                hovertemplate = "<br><b>ORPSP:</b> %{customdata[3]}"

            else:
                marker = {"size": 10, "color": "white"}
                showlegend = True
                customdata = np.stack(
                    (
                        offense_tracking["nflId"].to_numpy(),
                        offense_tracking["displayName"].to_numpy(),
                        offense_tracking["club"].to_numpy(),
                    ),
                    axis=-1,
                )
                hovertemplate = ""

            data.append(
                go.Scatter(
                    x=offense_tracking["x"].to_numpy(),
                    y=offense_tracking["y"].to_numpy(),
                    mode="markers",
                    marker=marker,
                    name="offense",
                    showlegend=showlegend,
                    customdata=customdata,
                    hovertemplate="<b>nflId:</b> %{customdata[0]}<br>"
                    "<b>Player:</b> %{customdata[1]}<br>"
                    "<b>Team:</b> %{customdata[2]}<br>" + hovertemplate,
                ),
            )

            data.append(
                go.Scatter(
                    x=ball_tracking["x"].to_numpy(),
                    y=ball_tracking["y"].to_numpy(),
                    mode="markers",
                    marker={"size": 10, "color": "brown", "symbol": "diamond-wide"},
                    name="ball",
                    hoverinfo="none",
                ),
            )

            steps.append(self._create_step(str(frame_id)))
            frames.append(
                {
                    "data": data,
                    "name": str(frame_id),
                }
            )
        for trace in frames[0]["data"]:
            self.fig.add_trace(trace)
        self.fig.frames = frames
        if self.is_animated:
            self.fig.layout.sliders[0]["steps"] = steps

    def save_as_gif(self, name: str = "animated_play") -> None:
        """Save the animated play as a GIF file.

        Parameters
        ----------
        name : str, optional
            Name of the saved GIF file, by default "animated_play"
        """
        layout = copy.deepcopy(self.fig.layout)
        frames = []
        for i, frame in enumerate(self.fig.frames):
            layout.sliders[0]["active"] = i
            fig = go.Figure(frame.data, layout=layout)
            frames.append(iio.imread(fig.to_image(format="png", height=600, width=1350)))

        iio.imwrite(animations_path + f"{name}.gif", frames, loop=0, format="GIF")
