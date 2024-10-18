import numpy as np
import polars as pl


def inverse_left_directed_plays(data: pl.DataFrame) -> pl.DataFrame:
    """_summary_

    Parameters
    ----------
    data : pl.DataFrame
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
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
    """_summary_

    Parameters
    ----------
    tracking : pl.DataFrame
        _description_
    player_play : pl.DataFrame
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    filtered_player_play = player_play.filter(pl.col("wasRunningRoute") == 1)

    route_tracking = tracking.join(
        filtered_player_play.select(["gameId", "playId", "nflId", "routeRan"]),
        on=["gameId", "playId", "nflId"],
        how="inner",
    )

    return route_tracking


def get_route_direction(data: pl.DataFrame, max_route_frame: int = 30) -> pl.DataFrame:
    """_summary_

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    max_route_frame : int, optional
        _description_, by default 30

    Returns
    -------
    pl.DataFrame
        _description_
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
    """_summary_

    Parameters
    ----------
    data : pl.DataFrame
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    right_route = ~pl.col("route_left")

    data = data.with_columns([pl.when(right_route).then(53.3 - pl.col("y")).otherwise(pl.col("y")).alias("y")])

    return data


def process_route_tracking(route_tracking: pl.DataFrame, max_route_frame: int = 30) -> pl.DataFrame:
    """_summary_

    Parameters
    ----------
    route_tracking : pl.DataFrame
        _description_
    max_route_frame : int, optional
        _description_, by default 30

    Returns
    -------
    pl.DataFrame
        _description_
    """
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
    """_summary_

    Parameters
    ----------
    processed_route_tracking : pl.DataFrame
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    quadratic_fit_results = processed_route_tracking.group_by(["gameId", "playId", "nflId"]).agg(
        pl.map_groups(exprs=["relative_x", "relative_y"], function=_quadratic_fit).alias("coefficients")
    )

    quadratic_fit_results = quadratic_fit_results.with_columns(
        pl.col("coefficients").list.to_struct(fields=["coef_a", "coef_b", "coef_c"])
    ).unnest("coefficients")

    route_features = processed_route_tracking.group_by(["gameId", "playId", "nflId"]).agg(
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
