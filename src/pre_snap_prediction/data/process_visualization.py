import polars as pl
from pre_snap_prediction.data import process_data


def add_clusters_to_visualization(
    visualization_tracking: pl.DataFrame, clusters_route: pl.DataFrame, clusters_reception_zone: pl.DataFrame
) -> pl.DataFrame:
    """Enhances the visualization tracking data by adding route cluster information and reception zones.

    Parameters
    ----------
    visualization_tracking : pl.DataFrame
        A Polars DataFrame containing the visualization tracking data.
    clusters_route : pl.DataFrame
        A Polars DataFrame containing route clusters.
    clusters_reception_zone : pl.DataFrame
        A Polars DataFrame containing reception zone data for the clusters.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the enhanced visualization tracking data.
    """
    clusters_route_reception = clusters_route.select(["gameId", "playId", "nflId", "cluster"]).join(
        clusters_reception_zone.select(
            [
                "cluster",
                "relative_x_min",
                "relative_x_max",
                "relative_y_min",
                "relative_y_max",
                "route_frameId_mean",
                "route_time_mean",
            ]
        ),
        on=["cluster"],
        how="left",
    )

    visualization_tracking = visualization_tracking.join(
        clusters_route_reception,
        on=["gameId", "playId", "nflId"],
        how="left",
    )

    visualization_tracking = visualization_tracking.with_columns(
        pl.when(pl.col("playDirection") == "left")
        .then(-pl.col("relative_x_min"))
        .otherwise(pl.col("relative_x_min"))
        .alias("relative_x_min"),
        pl.when(pl.col("playDirection") == "left")
        .then(-pl.col("relative_x_max"))
        .otherwise(pl.col("relative_x_max"))
        .alias("relative_x_max"),
        pl.when(~pl.col("route_left"))
        .then(-pl.col("relative_y_min"))
        .otherwise(pl.col("relative_y_min"))
        .alias("relative_y_min"),
        pl.when(~pl.col("route_left"))
        .then(-pl.col("relative_y_max"))
        .otherwise(pl.col("relative_y_max"))
        .alias("relative_y_max"),
    )

    return visualization_tracking


def add_orpsp_to_visualization(visualization_tracking: pl.DataFrame, orpsp_predictions: pl.DataFrame) -> pl.DataFrame:
    """Enhances the visualization tracking data by adding ORPSP predictions.

    Parameters
    ----------
    visualization_tracking : pl.DataFrame
        _description_
    orpsp_predictions : pl.DataFrame
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    visualization_tracking = visualization_tracking.join(
        orpsp_predictions,
        on=["gameId", "playId", "nflId"],
        how="left",
    )

    return visualization_tracking


def compute_visualization_tracking(
    tracking: pl.DataFrame,
    plays: pl.DataFrame,
    clusters_route: pl.DataFrame | None = None,
    clusters_reception_zone: pl.DataFrame | None = None,
    orpsp_predictions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Prepares tracking data for visualization by incorporating play details and optional clustering data.

    Parameters
    ----------
    tracking : pl.DataFrame
        A Polars DataFrame containing player tracking data.
    plays : pl.DataFrame
        A Polars DataFrame containing player play information.
    clusters_route : pl.DataFrame | None, optional
        A Polars DataFrame containing route clusters, by default None
    clusters_reception_zone : pl.DataFrame | None, optional
        A Polars DataFrame containing reception zone data for the clusters, by default None

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame prepared for visualization?
    """
    visualization_tracking = process_data.get_route_direction(
        tracking.select(
            ["gameId", "playId", "nflId", "displayName", "frameId", "frameType", "club", "x", "y", "playDirection"]
        )
    )
    visualization_tracking = visualization_tracking.join(
        plays.select(["gameId", "playId", "defensiveTeam", "absoluteYardlineNumber", "yardsToGo"]),
        on=["gameId", "playId"],
        how="left",
    )

    visualization_tracking = visualization_tracking.with_columns(
        (pl.col("club") == pl.col("defensiveTeam")).alias("is_defense")
    )

    if clusters_route is not None and clusters_reception_zone is not None:
        visualization_tracking = add_clusters_to_visualization(
            visualization_tracking, clusters_route, clusters_reception_zone
        )

    if orpsp_predictions is not None:
        visualization_tracking = add_orpsp_to_visualization(visualization_tracking, orpsp_predictions)

    return visualization_tracking
