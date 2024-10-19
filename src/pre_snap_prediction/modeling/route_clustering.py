import polars as pl
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import IsolationForest


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
    outliers_model = IsolationForest(random_state=0).fit(data.drop(["gameId", "playId", "nflId"]).to_numpy())

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
    predictions = outliers_model.predict(data.drop(["gameId", "playId", "nflId"]).to_numpy())

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
        data.drop(["gameId", "playId", "nflId"]).to_numpy()
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
    predictions = clustering_model.predict(data.drop(["gameId", "playId", "nflId"]).to_numpy())

    data = data.with_columns(pl.Series("cluster", predictions))

    print(data["cluster"].value_counts().sort("count", descending=True))

    return data


def join_clusters_to_data(data: pl.DataFrame, clusters_route: pl.DataFrame) -> pl.DataFrame:
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
        clusters_route.select(["gameId", "playId", "nflId", "cluster"]), on=["gameId", "playId", "nflId"], how="left"
    )

    return clusters_data
