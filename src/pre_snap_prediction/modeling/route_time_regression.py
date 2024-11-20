import polars as pl
from catboost import CatBoostRegressor
from pre_snap_prediction.utils.constants import ROUTE_TIME_FEATURES, ROUTE_TIME_FEATURES_TO_ENCODE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def select_route_time_features(features: pl.DataFrame, target: bool = False) -> pl.DataFrame:
    """Selects a specific subset of features for route time prediction.

    Parameters
    ----------
    features : pl.DataFrame
        A Polars DataFrame containing the full set of computed ORPSP features.
    target : bool, optional
        Should the target be select or not, by default False

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with only the selected features.
    """
    index_columns = ["gameId", "playId", "nflId"]
    if target:
        index_columns += ["route_time_mean"]
    return features.select(index_columns + ROUTE_TIME_FEATURES)


def train_encoder(data: pl.DataFrame) -> OrdinalEncoder:
    """Trains an OrdinalEncoder on specified categorical features.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing the categorical features to be encoded.

    Returns
    -------
    OrdinalEncoder
        A trained encoder ready for transforming the categorical columns into ordinal values.
    """
    encoder = OrdinalEncoder().fit(data.select(ROUTE_TIME_FEATURES_TO_ENCODE))

    return encoder


def transform_encoder(data: pl.DataFrame, encoder: OrdinalEncoder) -> pl.DataFrame:
    """Transforms specified categorical features in the DataFrame using a fitted OrdinalEncoder.

    Parameters
    ----------
    data : pl.DataFrame
        A Polars DataFrame containing the features to encode.
    encoder : OrdinalEncoder
        A trained OrdinalEncoder used to transform the categorical features.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame identical in structure to the input data, but with encoded categorical features.
    """
    data_to_encode = data.select(ROUTE_TIME_FEATURES_TO_ENCODE)

    other_data = data.drop(ROUTE_TIME_FEATURES_TO_ENCODE)
    encoded_data = pl.DataFrame(encoder.transform(data_to_encode), schema=data_to_encode.columns)
    encoded_data = pl.concat([other_data, encoded_data], how="horizontal").select(data.columns)

    return encoded_data


def route_time_train_test_split(
    encoded_data: pl.DataFrame, test_size: float = 0.3
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits the route time dataset into training and testing sets.

    Parameters
    ----------
    encoded_data : pl.DataFrame
        A Polars DataFrame containing encoded features.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.3

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Four DataFrames representing the training and testing splits.
    """
    learning_data = encoded_data.filter(pl.col("route_time_mean").is_not_null())

    x_train, x_test, y_train, y_test = train_test_split(
        learning_data.drop("route_time_mean"),
        learning_data.select("route_time_mean"),
        test_size=test_size,
        random_state=0,
    )

    return x_train, x_test, y_train, y_test


def train_route_time_regression(x_train: pl.DataFrame, y_train: pl.DataFrame) -> CatBoostRegressor:
    """Trains a CatBoost regression model to predict route time.

    Parameters
    ----------
    x_train : pl.DataFrame
        A Polars DataFrame containing training features data.
    y_train : pl.DataFrame
        A Polars DataFrame containing training labels for the route time regression.

    Returns
    -------
    CatBoostRegressor
        A trained CatBoostRegressor model ready for route time prediction.
    """
    regression_model = CatBoostRegressor().fit(
        x_train.drop(["gameId", "playId", "nflId"]).to_numpy(),
        y_train.to_numpy(),
        verbose=False,
    )

    return regression_model


def predict_route_time(x: pl.DataFrame, regression_model: CatBoostRegressor) -> pl.DataFrame:
    """Generates route time predictions using a trained CatBoost regression model.

    Parameters
    ----------
    x : pl.DataFrame
        A Polars DataFrame containing features data.
    regression_model : CatBoostRegressor
        A trained CatBoostRegressor model ready for route time prediction.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the predicted route time.
    """
    predictions = x.select(["gameId", "playId", "nflId"])
    predictions = predictions.with_columns(
        pl.Series("route_time_mean", regression_model.predict(x.drop(["gameId", "playId", "nflId"]).to_numpy()))
    )
    return predictions
