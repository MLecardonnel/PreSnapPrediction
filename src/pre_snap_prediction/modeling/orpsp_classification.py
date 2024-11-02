import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from pre_snap_prediction.utils.constants import FEATURES, FEATURES_TO_ENCODE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def select_orpsp_features(features: pl.DataFrame) -> pl.DataFrame:
    """Selects a specific subset of ORPSP features.

    Parameters
    ----------
    features : pl.DataFrame
        A Polars DataFrame containing the full set of computed ORPSP features.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with only the selected ORPSP features.
    """
    return features.select(["gameId", "playId", "nflId"] + FEATURES)


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
    encoder = OrdinalEncoder().fit(data.select(FEATURES_TO_ENCODE))

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
    data_to_encode = data.select(FEATURES_TO_ENCODE)

    other_data = data.drop(FEATURES_TO_ENCODE)
    encoded_data = pl.DataFrame(encoder.transform(data_to_encode), schema=data_to_encode.columns)
    encoded_data = pl.concat([other_data, encoded_data], how="horizontal").select(data.columns)

    return encoded_data


def orpsp_train_test_split(
    encoded_data: pl.DataFrame, target_data: pl.DataFrame, test_size: float = 0.3
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Splits the ORPSP dataset into training and testing sets.

    Parameters
    ----------
    encoded_data : pl.DataFrame
        A Polars DataFrame containing encoded features.
    target_data : pl.DataFrame
        A Polars DataFrame containing the ORPSP target variable
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.3

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Four DataFrames representing the training and testing splits.
    """
    learning_data = encoded_data.join(
        target_data,
        on=["gameId", "playId", "nflId"],
        how="left",
    )
    learning_data = learning_data.filter(pl.col("orpsp_target").is_not_null())

    x_train, x_test, y_train, y_test = train_test_split(
        learning_data.drop("orpsp_target"),
        learning_data.select("orpsp_target"),
        test_size=test_size,
        random_state=0,
    )

    return x_train, x_test, y_train, y_test


def train_orpsp_classification(x_train: pl.DataFrame, y_train: pl.DataFrame, max_depth: int = 6) -> CatBoostClassifier:
    """Trains a CatBoost classification model to predict ORPSP.

    Parameters
    ----------
    x_train : pl.DataFrame
        A Polars DataFrame containing training features data.
    y_train : pl.DataFrame
        A Polars DataFrame containing training labels for the ORPSP classification.
    max_depth : int, optional
        Maximum depth of the tree in the CatBoostClassifier, by default 6

    Returns
    -------
    CatBoostClassifier
        A trained CatBoostClassifier model ready for ORPSP prediction.
    """
    classification_model = CatBoostClassifier(max_depth=max_depth, learning_rate=0.01).fit(
        x_train.drop(["gameId", "playId", "nflId"]).to_numpy(),
        y_train.to_numpy(),
        verbose=False,
    )

    return classification_model


def predict_orpsp_class(x: pl.DataFrame, classification_model: CatBoostClassifier) -> np.ndarray:
    """Uses a trained CatBoost classification model to predict ORPSP target classes.

    Parameters
    ----------
    x : pl.DataFrame
        A Polars DataFrame containing features data.
    classification_model : CatBoostClassifier
        A trained CatBoostClassifier model ready for ORPSP prediction.

    Returns
    -------
    np.ndarray
        Array of predicted class labels for the ORPSP target.
    """
    return classification_model.predict(x.drop(["gameId", "playId", "nflId"]).to_numpy())


def predict_orpsp(x: pl.DataFrame, classification_model: CatBoostClassifier) -> pl.DataFrame:
    """Generates ORPSP probability predictions using a trained CatBoost classification model.

    Parameters
    ----------
    x : pl.DataFrame
        A Polars DataFrame containing features data.
    classification_model : CatBoostClassifier
        A trained CatBoostClassifier model ready for ORPSP prediction.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the predicted ORPSP.
    """
    predictions = x.select(["gameId", "playId", "nflId"])
    predictions = predictions.with_columns(
        pl.Series("orpsp", classification_model.predict_proba(x.drop(["gameId", "playId", "nflId"]).to_numpy())[:, 1])
    )
    return predictions
