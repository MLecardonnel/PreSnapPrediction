import polars as pl
from catboost import CatBoostRegressor
from pre_snap_prediction.utils.constants import ROUTES_CONVERSION
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import IsolationForest
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


def train_outliers_model(data: pl.DataFrame) -> IsolationForest:
    """Trains an Isolation Forest model to detect outliers in the provided dataset.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing the dataset to train the outlier detection model.

    Returns
    -------
    IsolationForest
        An Isolation Forest model trained on the input data.
    """
    outliers_model = IsolationForest(random_state=0).fit(data.drop(["week", "gameId", "playId", "nflId"]).to_numpy())

    return outliers_model


def predict_outliers(data: pl.DataFrame, outliers_model: IsolationForest) -> pl.DataFrame:
    """Uses a trained Isolation Forest model to predict outliers (anomalies) in the dataset.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing the dataset for which outliers need to be predicted.
    outliers_model : IsolationForest
        A pre-trained Isolation Forest model that will be used to predict outliers in the dataset.

    Returns
    -------
    pl.DataFrame
        The input Polars DataFrame with an additional "anomaly" column.
    """
    predictions = outliers_model.predict(data.drop(["week", "gameId", "playId", "nflId"]).to_numpy())

    data = data.with_columns(pl.Series("anomaly", predictions))

    print(data["anomaly"].value_counts().sort("count", descending=True))

    return data


def remove_outliers(data: pl.DataFrame) -> pl.DataFrame:
    """Removes outlier data points from the DataFrame based on the "anomaly" column.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing the dataset with an "anomaly" column.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with outliers removed and the "anomaly" column dropped.
    """
    valid_data = data.filter(pl.col("anomaly") == 1).drop("anomaly")

    return valid_data


def train_route_clustering(data: pl.DataFrame, damping=0.9, preference=-50) -> AffinityPropagation:
    """Trains an Affinity Propagation model to cluster player route data based on their movement patterns.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing the route data to be clustered.
    damping : float, optional
        A value between 0.5 and 1 that controls the extent to which the current clustering is influenced
        by the previous iteration, by default 0.9
    preference : int, optional
        Controls the number of clusters, by default -50

    Returns
    -------
    AffinityPropagation
        A trained Affinity Propagation model that can be used to predict clusters for route data.
    """
    clustering_model = AffinityPropagation(random_state=0, damping=damping, preference=preference).fit(
        data.drop(["week", "gameId", "playId", "nflId"]).to_numpy()
    )

    return clustering_model


def predict_route_cluters(data: pl.DataFrame, clustering_model: AffinityPropagation) -> pl.DataFrame:
    """Predicts route clusters for player data using a pre-trained Affinity Propagation model.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing the dataset for which route clusters need to be predicted.
    clustering_model : AffinityPropagation
        A pre-trained Affinity Propagation model used to predict the clusters for the route data.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with an additional "cluster" column indicating the cluster assigned to each route.
    """
    predictions = clustering_model.predict(data.drop(["week", "gameId", "playId", "nflId"]).to_numpy())

    data = data.with_columns(pl.Series("cluster", predictions))

    print(data["cluster"].value_counts().sort("count", descending=True))

    return data


def join_clusters_to_data(data: pl.DataFrame, clusters_route: pl.DataFrame, how: str = "left") -> pl.DataFrame:
    """Joins the cluster labels with the original dataset based on game, play, and player identifiers.

    Parameters
    ----------
    data : pl.DataFrame
        The original Polars DataFrame containing the dataset with game, play, and player-level data.
    clusters_route : pl.DataFrame
        A Polars DataFrame containing the route clustering results.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with the original data and an additional "cluster" column.
    """
    clusters_data = data.join(
        clusters_route.select(["gameId", "playId", "nflId", "cluster"]), on=["gameId", "playId", "nflId"], how=how
    )

    return clusters_data


def get_modified_route_mode(player_play: pl.DataFrame, clusters_route_tracking: pl.DataFrame) -> pl.DataFrame:
    """Determines the most common (mode) modified route type for each route cluster.

    Parameters
    ----------
    player_play : pl.DataFrame
        A Polars DataFrame containing player play information.
    clusters_route_tracking : pl.DataFrame
        A Polars DataFrame containing the route clustering results.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame that contains the most common route type for each cluster.
    """
    data = join_clusters_to_data(
        player_play.select(["gameId", "playId", "nflId", "routeRan"]),
        clusters_route_tracking,
        how="inner",
    )

    clusters_route_mode = data.group_by(["cluster"]).agg(
        pl.col("routeRan").mode().get(0).replace(ROUTES_CONVERSION).alias("route_mode"),
    )

    return clusters_route_mode


def get_clusters_reception_zones(player_play: pl.DataFrame, clusters_route_tracking: pl.DataFrame) -> pl.DataFrame:
    """Computes the reception zones for each route cluster based on targeted receiver data.

    Parameters
    ----------
    player_play : pl.DataFrame
        A Polars DataFrame containing player play information.
    clusters_route_tracking : pl.DataFrame
        A Polars DataFrame containing the route clustering results.

    Returns
    -------
    pl.DataFrame
         Polars DataFrame containing the calculated reception zones for each route cluster.
    """
    targeted_receiver = player_play.filter(pl.col("wasTargettedReceiver") == 1)

    reception_clusters_frames = clusters_route_tracking.filter(
        pl.col("event") == "pass_arrived", pl.col("cluster").is_not_null()
    )

    targeted_clusters = targeted_receiver.select(["gameId", "playId", "nflId"]).join(
        reception_clusters_frames,
        on=["gameId", "playId", "nflId"],
        how="inner",
    )

    clusters_reception_zones = targeted_clusters.group_by(["cluster"]).agg(
        pl.col("relative_x").min().alias("relative_x_min"),
        pl.col("relative_x").max().alias("relative_x_max"),
        pl.col("relative_x").mean().alias("relative_x_mean"),
        pl.col("relative_y").min().alias("relative_y_min"),
        pl.col("relative_y").max().alias("relative_y_max"),
        pl.col("relative_y").mean().alias("relative_y_mean"),
        pl.col("route_frameId").mean().alias("route_frameId_mean"),
    )

    clusters_reception_zones = clusters_reception_zones.with_columns(
        [
            pl.when((pl.col("relative_x_mean") - pl.col("relative_x_min")).abs() < 1)
            .then(pl.col("relative_x_mean") - 3)
            .otherwise(pl.col("relative_x_min"))
            .alias("relative_x_min"),
            pl.when((pl.col("relative_x_max") - pl.col("relative_x_mean")).abs() < 1)
            .then(pl.col("relative_x_mean") + 3)
            .otherwise(pl.col("relative_x_max"))
            .alias("relative_x_max"),
            pl.when((pl.col("relative_y_mean") - pl.col("relative_y_min")).abs() < 1)
            .then(pl.col("relative_y_mean") - 3)
            .otherwise(pl.col("relative_y_min"))
            .alias("relative_y_min"),
            pl.when((pl.col("relative_y_max") - pl.col("relative_y_mean")).abs() < 1)
            .then(pl.col("relative_y_mean") + 3)
            .otherwise(pl.col("relative_y_max"))
            .alias("relative_y_max"),
            (pl.col("route_frameId_mean") * 0.1).alias("route_time_mean"),
        ]
    )

    return clusters_reception_zones


def predict_missing_reception_zone(clusters_route: pl.DataFrame, clusters_reception_zone: pl.DataFrame) -> pl.DataFrame:
    """Predict missing reception zone values for route clusters.

    Parameters
    ----------
    clusters_route : pl.DataFrame
        A Polars DataFrame containing route clusters.
    clusters_reception_zone : pl.DataFrame
        A Polars DataFrame containing reception zone data for the clusters.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the predicted reception zone data for the clusters.
    """
    columns_to_predict = ["relative_x_mean", "relative_y_mean", "route_frameId_mean"]

    clusters_data = clusters_route.drop(["week", "gameId", "playId", "nflId"]).join(
        clusters_reception_zone.select(["cluster"] + columns_to_predict),
        on=["cluster"],
        how="left",
    )

    clusters_learning = clusters_data.filter(pl.col("relative_x_mean").is_not_null())

    clusters_to_predict = clusters_data.filter(pl.col("relative_x_mean").is_null())
    clusters_to_predict = clusters_to_predict.group_by("cluster").mean()

    for col in columns_to_predict:
        x_train, x_test, y_train, y_test = train_test_split(
            clusters_learning.drop(["cluster"] + columns_to_predict),
            clusters_learning.select([col]),
            test_size=0.3,
        )

        model = CatBoostRegressor().fit(
            x_train.to_numpy(),
            y_train.to_numpy(),
            verbose=False,
        )

        y_train_pred = model.predict(x_train.to_numpy())
        y_pred = model.predict(x_test.to_numpy())

        print(f"{col} Train RMSE: {root_mean_squared_error(y_train.to_numpy(), y_train_pred)}")
        print(f"{col} Train RMSE: {root_mean_squared_error(y_test.to_numpy(), y_pred)}")

        missing_prediction = model.predict(clusters_to_predict.drop(["cluster"] + columns_to_predict).to_numpy())

        clusters_to_predict = clusters_to_predict.with_columns(pl.Series(missing_prediction).alias(col))

    clusters_to_predict = clusters_to_predict.with_columns(
        (pl.col("relative_x_mean") - 3).alias("relative_x_min"),
        (pl.col("relative_x_mean") + 3).alias("relative_x_max"),
        (pl.col("relative_y_mean") - 3).alias("relative_y_min"),
        (pl.col("relative_y_mean") + 3).alias("relative_y_max"),
        (pl.col("route_frameId_mean") * 0.1).alias("route_time_mean"),
    )

    clusters_reception_zone = pl.concat(
        [clusters_reception_zone, clusters_to_predict.select(clusters_reception_zone.columns)]
    )

    return clusters_reception_zone


def get_complete_plays(player_play: pl.DataFrame, clusters_route: pl.DataFrame) -> pl.DataFrame:
    """Returns a DataFrame containing the set of plays where all players who ran a route
    have been assigned to a specific cluster.

    Parameters
    ----------
    player_play : pl.DataFrame
        A Polars DataFrame containing player play information.
    clusters_route : pl.DataFrame
        A Polars DataFrame containing route clusters.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the unique 'gameId' and 'playId' pairs for complete plays.
    """
    filtered_player_play = player_play.filter(pl.col("wasRunningRoute") == 1)

    player_play_clusters = filtered_player_play.select(["gameId", "playId", "nflId"]).join(
        clusters_route.select(["gameId", "playId", "nflId", "cluster"]),
        on=["gameId", "playId", "nflId"],
        how="left",
    )

    plays_missing_clusters = (
        player_play_clusters.filter(pl.col("cluster").is_null()).select(["gameId", "playId"]).unique()
    )

    print(f"Incomplete plays: {len(plays_missing_clusters)}")

    complete_plays = (
        player_play_clusters.select(["gameId", "playId"])
        .unique()
        .join(
            plays_missing_clusters,
            on=["gameId", "playId"],
            how="anti",
        )
    )

    return complete_plays


def join_data_to_complete_plays(complete_plays: pl.DataFrame, data: pl.DataFrame) -> pl.DataFrame:
    """Joins additional data to complete plays based on game and play identifiers.

    Parameters
    ----------
    complete_plays : pl.DataFrame
        A DataFrame containing the unique 'gameId' and 'playId' pairs for complete plays.
    data : pl.DataFrame
        A Polars DataFrame to be joined with complete plays.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame that contains complete plays data.
    """
    complete_plays_data = complete_plays.join(
        data,
        on=["gameId", "playId"],
        how="inner",
    )

    return complete_plays_data
