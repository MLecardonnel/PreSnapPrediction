import polars as pl
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import IsolationForest


def train_outliers_model(data: pl.DataFrame, contamination: float = 0.1) -> IsolationForest:
    """_summary_

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    contamination : float, optional
        _description_, by default 0.1

    Returns
    -------
    IsolationForest
        _description_
    """
    outliers_model = IsolationForest(random_state=0, contamination=contamination).fit(
        data.drop(["gameId", "playId", "nflId"]).to_numpy()
    )

    return outliers_model


def predict_outliers(data: pl.DataFrame, outliers_model: IsolationForest) -> pl.DataFrame:
    """_summary_

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    outliers_model : IsolationForest
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    predictions = outliers_model.predict(data.drop(["gameId", "playId", "nflId"]).to_numpy())

    data = data.with_columns(pl.Series("anomaly", predictions))

    print(data["anomaly"].value_counts().sort("count", descending=True))

    return data


def remove_outliers(data: pl.DataFrame) -> pl.DataFrame:
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
    valid_data = data.filter(pl.col("anomaly") == 1).drop("anomaly")

    return valid_data


def train_route_clustering(data: pl.DataFrame, damping=0.9, preference=-50) -> AffinityPropagation:
    """_summary_

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    damping : float, optional
        _description_, by default 0.9
    preference : int, optional
        _description_, by default -50

    Returns
    -------
    AffinityPropagation
        _description_
    """
    clustering_model = AffinityPropagation(random_state=0, damping=damping, preference=preference).fit(
        data.drop(["gameId", "playId", "nflId"]).to_numpy()
    )

    return clustering_model


def predict_route_cluters(data: pl.DataFrame, clustering_model: AffinityPropagation) -> pl.DataFrame:
    """_summary_

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    clustering_model : AffinityPropagation
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    predictions = clustering_model.predict(data.drop(["gameId", "playId", "nflId"]).to_numpy())

    data = data.with_columns(pl.Series("cluster", predictions))

    print(data["cluster"].value_counts().sort("count", descending=True))

    return data


def join_clusters_to_data(data, clusters_route):
    """_summary_

    Parameters
    ----------
    data : _type_
        _description_
    clusters_route : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    clusters_data = data.join(
        clusters_route.select(["gameId", "playId", "nflId", "cluster"]), on=["gameId", "playId", "nflId"], how="left"
    )

    return clusters_data
