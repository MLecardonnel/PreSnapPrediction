import polars as pl
from pre_snap_prediction.data import process_data
from pre_snap_prediction.modeling import route_clustering
from pre_snap_prediction.utils.constants import FIELD_LENGTH, FIELD_WIDTH, ROUTES_CONVERSION


def create_orpsp_target(complete_plays: pl.DataFrame, plays: pl.DataFrame, player_play: pl.DataFrame) -> pl.DataFrame:
    """Creates an Open Receiver Pre Snap Probability (ORPSP) target DataFrame.

    Parameters
    ----------
    complete_plays : pl.DataFrame
        A DataFrame containing the unique 'gameId' and 'playId' pairs for complete plays.
    plays : pl.DataFrame
        A Polars DataFrame containing plays information.
    player_play : pl.DataFrame
        A Polars DataFrame containing player play information.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the calculated ORPSP target.
    """
    complete_plays_data = route_clustering.join_data_to_complete_plays(complete_plays, plays)

    plays_target_data = complete_plays_data.filter(
        pl.col("passResult").is_not_null(),
        (~pl.col("qbSpike") | pl.col("qbSpike").is_null()),
        ((pl.col("timeToSack") > 4) | pl.col("timeToSack").is_null()),
    )

    plays_target_data = plays_target_data.select(["gameId", "playId", "passResult", "prePenaltyYardsGained"])

    complete_player_play_data = route_clustering.join_data_to_complete_plays(complete_plays, player_play)

    complete_player_play_data = complete_player_play_data.filter(pl.col("wasRunningRoute") == 1)

    complete_player_play_data = complete_player_play_data.select(["gameId", "playId", "nflId", "wasTargettedReceiver"])

    players_target_data = plays_target_data.join(
        complete_player_play_data,
        on=["gameId", "playId"],
        how="inner",
    ).filter(
        ((pl.col("passResult").is_in(["C", "I", "IN"])) & pl.col("wasTargettedReceiver") == 1)
        | (pl.col("passResult").is_in(["S", "R"]))
    )

    players_target_data = players_target_data.with_columns(
        pl.when(pl.col("passResult") == "C", pl.col("prePenaltyYardsGained") > 2)
        .then(1)
        .otherwise(0)
        .alias("orpsp_target")
    )

    players_target_data = players_target_data.select(["gameId", "playId", "nflId", "orpsp_target"])

    print(players_target_data["orpsp_target"].value_counts().sort("orpsp_target"))

    return players_target_data


def get_plays_features(complete_plays: pl.DataFrame, plays: pl.DataFrame) -> pl.DataFrame:
    """Extracts and returns essential features for each play.

    Parameters
    ----------
    complete_plays : pl.DataFrame
        A DataFrame containing the unique 'gameId' and 'playId' pairs for complete plays.
    plays : pl.DataFrame
        A Polars DataFrame containing plays information.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing plays features.
    """
    complete_plays_data = route_clustering.join_data_to_complete_plays(complete_plays, plays)
    plays_features = complete_plays_data.select(
        [
            "gameId",
            "playId",
            "down",
            "yardsToGo",
            "absoluteYardlineNumber",
            "offenseFormation",
            "receiverAlignment",
            "playAction",
        ]
    )

    return plays_features


def get_clusters_features(
    complete_plays: pl.DataFrame,
    clusters_route: pl.DataFrame,
    clusters_route_mode: pl.DataFrame,
    clusters_reception_zone: pl.DataFrame,
) -> pl.DataFrame:
    """Compiles and returns detailed features integrating route and reception zone details.

    Parameters
    ----------
    complete_plays : pl.DataFrame
        A DataFrame containing the unique 'gameId' and 'playId' pairs for complete plays.
    clusters_route : pl.DataFrame
        A Polars DataFrame containing route clusters.
    clusters_route_mode : pl.DataFrame
        A Polars DataFrame that contains the most common route type for each cluster.
    clusters_reception_zone : pl.DataFrame
        A Polars DataFrame containing reception zone data for the clusters.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing route cluster features.
    """
    complete_clusters_route = route_clustering.join_data_to_complete_plays(complete_plays, clusters_route)
    clusters_features = complete_clusters_route.select(["gameId", "playId", "nflId", "cluster"])
    clusters_features = clusters_features.join(
        clusters_route_mode,
        on=["cluster"],
        how="inner",
    )
    clusters_features = clusters_features.join(
        clusters_reception_zone.select(["cluster", "relative_x_mean", "relative_y_mean", "route_time_mean"]),
        on=["cluster"],
        how="inner",
    )

    return clusters_features


def get_tracking_features(complete_plays: pl.DataFrame, tracking: pl.DataFrame) -> pl.DataFrame:
    """Extracts and returns key tracking features for each player-frame.

    Parameters
    ----------
    complete_plays : pl.DataFrame
        A DataFrame containing the unique 'gameId' and 'playId' pairs for complete plays.
    tracking : pl.DataFrame
        A Polars DataFrame containing player tracking data.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing tracking features for each player-frame.
    """
    complete_tracking = route_clustering.join_data_to_complete_plays(complete_plays, tracking)
    complete_tracking = process_data.inverse_left_directed_plays(complete_tracking)
    complete_tracking = process_data.get_route_direction(complete_tracking)

    tracking_features = complete_tracking.select(
        ["gameId", "playId", "nflId", "frameType", "frameId", "playDirection", "x", "y", "o", "event", "route_left"]
    )

    return tracking_features


def get_start_features(tracking_features: pl.DataFrame) -> pl.DataFrame:
    """Extracts starting features for each player at the "BEFORE_SNAP" frame and appends ball position at this frame.

    Parameters
    ----------
    tracking_features : pl.DataFrame
        A Polars DataFrame containing tracking features for each player-frame.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing starting features of players before the snap.
    """
    start_features = (
        tracking_features.filter(pl.col("frameType") == "BEFORE_SNAP").group_by(["gameId", "playId", "nflId"]).last()
    )

    ball_features = start_features.filter(pl.col("nflId").is_null())
    ball_features = ball_features.select(["gameId", "playId", pl.col("x").alias("x_ball"), pl.col("y").alias("y_ball")])

    start_features = start_features.join(
        ball_features,
        on=["gameId", "playId"],
        how="inner",
    )

    return start_features


def _compute_euclidian_distance(
    x1: pl.Expr | float, y1: pl.Expr | float, x2: pl.Expr | float, y2: pl.Expr | float
) -> pl.Expr:
    if isinstance(x1, float):
        x1 = pl.lit(x1)
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2).sqrt().alias("euclidian_distance")


def _compute_direction_angle(x1: pl.Expr | float, y1: pl.Expr | float, x2: pl.Expr, y2: pl.Expr) -> pl.Expr:
    return (((pl.arctan2(x2 - x1, y2 - y1)).degrees() + 360) % 360).alias("direction_angle")


def _convert_dis_to_nearest_sideline(col: pl.Expr) -> pl.Expr:
    half_field_width = FIELD_WIDTH / 2
    return (half_field_width * (col // half_field_width) - col % half_field_width).abs().alias("dis_nearest_sideline")


def _compute_plays_route_features(data: pl.DataFrame) -> pl.DataFrame:
    unique_routes = set(ROUTES_CONVERSION.values())
    data = data.with_columns(
        (pl.col("x_recep_zone") - pl.col("absoluteYardlineNumber")).alias("recep_zone_from_yard_line")
    )
    data = data.with_columns(
        [pl.when(pl.col("route_mode") == route).then(1).otherwise(0).alias(route) for route in unique_routes]
        + [
            pl.when(pl.col("recep_zone_from_yard_line") < 0).then(1).otherwise(0).alias("recep_zone_negative"),
            pl.when(pl.col("recep_zone_from_yard_line") >= 0, pl.col("recep_zone_from_yard_line") < 5)
            .then(1)
            .otherwise(0)
            .alias("recep_zone_5"),
            pl.when(pl.col("recep_zone_from_yard_line") >= 5, pl.col("recep_zone_from_yard_line") < 10)
            .then(1)
            .otherwise(0)
            .alias("recep_zone_10"),
            pl.when(pl.col("recep_zone_from_yard_line") >= 10, pl.col("recep_zone_from_yard_line") < 20)
            .then(1)
            .otherwise(0)
            .alias("recep_zone_20"),
            pl.when(pl.col("recep_zone_from_yard_line") >= 20).then(1).otherwise(0).alias("recep_zone_inf"),
        ]
    )

    route_features = data.group_by(["gameId", "playId"]).agg(
        [pl.col(route).sum().alias(f"nb_{route}") for route in unique_routes]
        + [
            pl.col("route_mode").count().alias("nb_routes"),
            pl.col("recep_zone_negative").sum().alias("nb_recep_zone_negative"),
            pl.col("recep_zone_5").sum().alias("nb_recep_zone_5"),
            pl.col("recep_zone_10").sum().alias("nb_recep_zone_10"),
            pl.col("recep_zone_20").sum().alias("nb_recep_zone_20"),
            pl.col("recep_zone_inf").sum().alias("nb_recep_zone_inf"),
        ]
    )

    return route_features


def _get_absolute_reception_zone(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        (pl.col("x") + pl.col("relative_x_mean")).alias("x_recep_zone"),
        (pl.col("y") + pl.col("relative_y_mean")).alias("y_recep_zone"),
    )
    data = data.with_columns(
        pl.when(pl.col("x_recep_zone") > FIELD_LENGTH)
        .then(FIELD_LENGTH - 0.01)
        .when(pl.col("x_recep_zone") < 0)
        .then(0)
        .otherwise(pl.col("x_recep_zone"))
        .alias("x_recep_zone"),
        pl.when(pl.col("y_recep_zone") > FIELD_WIDTH)
        .then(FIELD_WIDTH - 0.01)
        .when(pl.col("y_recep_zone") < 0)
        .then(0)
        .otherwise(pl.col("y_recep_zone"))
        .alias("y_recep_zone"),
    )

    return data


def preprocess_orpsp_features(
    plays_features: pl.DataFrame, clusters_features: pl.DataFrame, start_features: pl.DataFrame, players: pl.DataFrame
) -> pl.DataFrame:
    """Pre-process Open Receiver Pre Snap Probability (ORPSP) features for each play.

    Parameters
    ----------
    plays_features : pl.DataFrame
        A Polars DataFrame containing plays features.
    clusters_features : pl.DataFrame
        A Polars DataFrame containing route cluster features.
    start_features : pl.DataFrame
        A Polars DataFrame containing starting features of players before the snap.
    players : pl.DataFrame
        A Polars DataFrame containing players information.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame of pre-processed ORPSP features.
    """
    features = clusters_features.join(
        plays_features,
        on=["gameId", "playId"],
        how="inner",
    )

    start_features = start_features.join(
        players.select(["nflId", "position"]),
        on=["nflId"],
        how="inner",
    )

    features = features.join(
        start_features,
        on=["gameId", "playId", "nflId"],
        how="inner",
    )

    features = features.with_columns(
        pl.when(pl.col("playDirection") == "left")
        .then(FIELD_LENGTH - pl.col("absoluteYardlineNumber"))
        .otherwise(pl.col("absoluteYardlineNumber"))
        .alias("absoluteYardlineNumber"),
        pl.when(~pl.col("route_left"))
        .then(-pl.col("relative_y_mean"))
        .otherwise(pl.col("relative_y_mean"))
        .alias("relative_y_mean"),
    )

    features = _get_absolute_reception_zone(features)

    return features


def compute_orpsp_features(features: pl.DataFrame) -> pl.DataFrame:
    """Computes Open Receiver Pre Snap Probability (ORPSP) features for each play.

    Parameters
    ----------
    features : pl.DataFrame
        Polars DataFrame of pre-processed ORPSP features.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame of ORPSP features.
    """
    features = features.with_columns(
        _compute_euclidian_distance(pl.col("x"), pl.col("y"), pl.col("x_recep_zone"), pl.col("y_recep_zone")).alias(
            "dis_recep_zone"
        ),
        _compute_direction_angle(pl.col("x"), pl.col("y"), pl.col("x_recep_zone"), pl.col("y_recep_zone")).alias(
            "dir_recep_zone"
        ),
        _convert_dis_to_nearest_sideline(_compute_euclidian_distance(0, 0, 0, pl.col("y_recep_zone"))).alias(
            "recep_zone_dis_out_of_bounds"
        ),
        _compute_euclidian_distance(pl.col("x_recep_zone"), 0, pl.col("absoluteYardlineNumber"), 0).alias(
            "recep_zone_dis_yard_line"
        ),
        _compute_euclidian_distance(
            pl.col("x_ball"), pl.col("y_ball"), pl.col("x_recep_zone"), pl.col("y_recep_zone")
        ).alias("dis_ball_to_recep_zone"),
        _compute_direction_angle(
            pl.col("x_ball"), pl.col("y_ball"), pl.col("x_recep_zone"), pl.col("y_recep_zone")
        ).alias("dir_ball_to_recep_zone"),
    )

    features = features.with_columns(
        _convert_dis_to_nearest_sideline(_compute_euclidian_distance(0, 0, 0, pl.col("y"))).alias("dis_out_of_bounds"),
        _compute_euclidian_distance(pl.col("x"), 0, FIELD_LENGTH, 0).alias("dis_back_of_endzone"),
        _compute_euclidian_distance(pl.col("x"), 0, pl.col("absoluteYardlineNumber"), 0).alias("dis_yard_line"),
    )

    plays_route_features = _compute_plays_route_features(features)
    features = features.join(
        plays_route_features,
        on=["gameId", "playId"],
        how="inner",
    )

    return features
