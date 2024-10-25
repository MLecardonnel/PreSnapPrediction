import polars as pl
from pre_snap_prediction.modeling import route_clustering


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

    plays_target_data = plays_target_data.with_columns(
        pl.when(pl.col("passResult") == "C", pl.col("prePenaltyYardsGained") > 0)
        .then(1)
        .otherwise(0)
        .alias("orpsp_target")
    )

    plays_target_data = plays_target_data.select(["gameId", "playId", "orpsp_target", "passResult"])

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

    players_target_data = players_target_data.select(["gameId", "playId", "nflId", "orpsp_target"])

    print(players_target_data["orpsp_target"].value_counts())

    return players_target_data
