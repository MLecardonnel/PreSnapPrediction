from pathlib import Path

import numpy as np
import polars as pl

data_path = (Path(__file__).parents[3] / "data").as_posix() + "/"


def read_tracking_csv(weeks: int = 9) -> pl.DataFrame:
    """Reads and combines tracking data CSV files for a specified number of weeks.

    Parameters
    ----------
    weeks : int, optional
        The number of weeks of tracking data to read, by default 9

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the concatenated tracking data for the specified number of weeks.

    Raises
    ------
    ValueError
        If the `weeks` parameter is not between 1 and 9.
    """
    if weeks >= 1 and weeks <= 9:
        tracking = pl.concat(
            [
                pl.read_csv(data_path + f"tracking_week_{i}.csv", null_values="NA").with_columns(
                    pl.lit(i).alias("week")
                )
                for i in range(1, weeks + 1)
            ]
        )
        return tracking
    else:
        raise ValueError("weeks should be between 1 and 9")


def inverse_left_directed_plays(data: pl.DataFrame) -> pl.DataFrame:
    """Adjusts the coordinates of plays where the play direction is "left" to be consistent
    with plays moving to the right by inverting the x and y coordinates accordingly.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing player tracking data.

    Returns
    -------
    pl.DataFrame
        A modified Polars DataFrame where the x and y coordinates have been adjusted.
    """
    left_playDirection = pl.col("playDirection") == "left"

    data = data.with_columns(
        [
            pl.when(left_playDirection).then(120.0 - pl.col("x")).otherwise(pl.col("x")).alias("x"),
            pl.when(left_playDirection).then(53.3 - pl.col("y")).otherwise(pl.col("y")).alias("y"),
        ]
    )

    return data


def get_route_tracking(tracking: pl.DataFrame, player_play: pl.DataFrame) -> pl.DataFrame:
    """Filters and joins tracking data with player play information to obtain route-specific tracking data.

    Parameters
    ----------
    tracking : pl.DataFrame
        A Polars DataFrame containing player tracking data.
    player_play : pl.DataFrame
        A Polars DataFrame containing player play information

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the tracking data for players who were running routes.
    """
    filtered_player_play = player_play.filter(pl.col("wasRunningRoute") == 1)

    route_tracking = tracking.join(
        filtered_player_play.select(["gameId", "playId", "nflId", "routeRan"]),
        on=["gameId", "playId", "nflId"],
        how="inner",
    )

    return route_tracking


def get_route_direction(data: pl.DataFrame, max_route_frame: int = 50) -> pl.DataFrame:
    """Determines the direction of player routes by comparing player positions before and after the snap.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing tracking data
    max_route_frame : int, optional
        The maximum number of frames after the snap to consider when calculating route direction, by default 50

    Returns
    -------
    pl.DataFrame
        A modified Polars DataFrame with a route direction column.
    """
    start_positions = (
        data.filter(pl.col("frameType") == "BEFORE_SNAP")
        .group_by(["gameId", "playId", "nflId"])
        .agg([pl.col("y").last().alias("y_start"), pl.col("frameId").last().alias("frameId_start")])
    )

    after_snap = data.filter(pl.col("frameType") == "AFTER_SNAP")

    route_direction = after_snap.join(start_positions, on=["gameId", "playId", "nflId"], how="inner")

    route_direction = route_direction.with_columns(
        [(pl.col("frameId") - pl.col("frameId_start") - 1).alias("route_frameId")]
    )

    route_direction = (
        route_direction.filter(pl.col("route_frameId") <= max_route_frame)
        .group_by(["gameId", "playId", "nflId"])
        .agg([pl.col("y").last(), pl.col("y_start").last()])
    )

    route_direction = route_direction.with_columns([(pl.col("y") > pl.col("y_start")).alias("route_left")])

    data = data.join(
        route_direction.select(["gameId", "playId", "nflId", "route_left"]),
        on=["gameId", "playId", "nflId"],
        how="left",
    )

    return data


def inverse_right_route(data: pl.DataFrame) -> pl.DataFrame:
    """Adjusts the y-coordinate of routes that move to the right to mirror those moving to the left.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing player tracking data.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame where the y-coordinate has been adjusted for players running routes to the right.
    """
    right_route = ~pl.col("route_left")

    data = data.with_columns([pl.when(right_route).then(53.3 - pl.col("y")).otherwise(pl.col("y")).alias("y")])

    return data


def process_route_tracking(
    route_tracking: pl.DataFrame, player_play: pl.DataFrame, max_route_frame: int = 50
) -> pl.DataFrame:
    """Processes player route tracking data by calculating relative positions from their starting positions.

    Parameters
    ----------
    route_tracking : pl.DataFrame
        A Polars DataFrame containing tracking data.
    player_play : pl.DataFrame
        A Polars DataFrame containing player play information
    max_route_frame : int, optional
        The maximum number of frames to consider after the snap for analyzing the route, by default 50

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the processed route tracking data
    """
    targeted_receiver = player_play.filter(pl.col("wasTargettedReceiver") == 1)

    start_positions = (
        route_tracking.filter(pl.col("frameType") == "BEFORE_SNAP")
        .group_by(["gameId", "playId", "nflId"])
        .agg(
            [
                pl.col("x").last().alias("x_start"),
                pl.col("y").last().alias("y_start"),
                pl.col("frameId").last().alias("frameId_start"),
            ]
        )
    )

    after_snap = route_tracking.filter(pl.col("frameType") == "AFTER_SNAP")

    processed_route_tracking = after_snap.join(start_positions, on=["gameId", "playId", "nflId"], how="inner")

    processed_route_tracking = processed_route_tracking.with_columns(
        [
            (pl.col("x") - pl.col("x_start")).alias("relative_x"),
            (pl.col("y") - pl.col("y_start")).alias("relative_y"),
            (pl.col("frameId") - pl.col("frameId_start") - 1).alias("route_frameId"),
        ]
    )

    processed_route_tracking = processed_route_tracking.filter(pl.col("route_frameId") <= max_route_frame)

    reception_clusters_frames = processed_route_tracking.filter(pl.col("event") == "pass_arrived")
    reception_clusters_frames = reception_clusters_frames.with_columns(
        pl.col("route_frameId").alias("reception_frameId")
    )
    reception_clusters_frames = reception_clusters_frames.select(
        ["gameId", "playId", "nflId", "reception_frameId"]
    ).join(
        targeted_receiver.select(["gameId", "playId", "nflId"]),
        on=["gameId", "playId", "nflId"],
        how="inner",
    )

    processed_route_tracking = processed_route_tracking.join(
        reception_clusters_frames,
        on=["gameId", "playId", "nflId"],
        how="left",
    )

    processed_route_tracking = processed_route_tracking.filter(
        (pl.col("reception_frameId").is_null()) | (pl.col("route_frameId") <= pl.col("reception_frameId"))
    ).drop("reception_frameId")

    return processed_route_tracking


def _quadratic_fit(args: list[pl.Series]) -> list[float]:
    coefficients = np.polyfit(args[0].to_numpy(), args[1].to_numpy(), 2)
    return [coefficients[0], coefficients[1], coefficients[2]]


def _get_position_from_percent(col: str, percent: float, non_negative: bool = False) -> pl.Series:
    position = (pl.col(col).count() * percent).cast(pl.Int64)

    result = pl.col(col).get(position)

    if non_negative:
        result = pl.when(result < 0).then(0).otherwise(result)

    return result


def compute_route_features(processed_route_tracking: pl.DataFrame) -> pl.DataFrame:
    """Computes advanced route features for players based on their processed route tracking data.

    Parameters
    ----------
    processed_route_tracking : pl.DataFrame
        A Polars DataFrame containing the processed tracking data.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing various computed route features for each player in each play.
    """
    quadratic_fit_results = processed_route_tracking.group_by(["gameId", "playId", "nflId"]).agg(
        pl.map_groups(exprs=["relative_x", "relative_y"], function=_quadratic_fit).alias("coefficients")
    )

    quadratic_fit_results = quadratic_fit_results.with_columns(
        pl.col("coefficients").list.to_struct(fields=["coef_a", "coef_b", "coef_c"])
    ).unnest("coefficients")

    route_features = processed_route_tracking.group_by(["week", "gameId", "playId", "nflId"]).agg(
        [
            pl.col("relative_x").median().alias("x_median"),
            pl.col("relative_x").std().alias("x_std"),
            _get_position_from_percent("relative_x", 0.2, True).alias("x_20"),
            _get_position_from_percent("relative_x", 0.5, True).alias("x_50"),
            _get_position_from_percent("relative_x", 0.8, True).alias("x_80"),
            pl.col("relative_y").median().alias("y_median"),
            pl.col("relative_y").std().alias("y_std"),
            _get_position_from_percent("relative_y", 0.2).alias("y_10"),
            _get_position_from_percent("relative_y", 0.5).alias("y_50"),
            _get_position_from_percent("relative_y", 0.8).alias("y_80"),
        ]
    )

    route_features = route_features.join(quadratic_fit_results, on=["gameId", "playId", "nflId"], how="inner")

    return route_features
