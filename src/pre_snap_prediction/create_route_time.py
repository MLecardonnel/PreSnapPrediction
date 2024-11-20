import pickle
from pathlib import Path

import polars as pl
from pre_snap_prediction.modeling import route_time_regression

data_path = (Path(__file__).parents[2] / "data").as_posix() + "/"
models_path = (Path(__file__).parents[2] / "models").as_posix() + "/"


if __name__ == "__main__":
    orpsp_features = pl.read_csv(data_path + "orpsp_features.csv", null_values=["NA", ""])

    route_time_features = route_time_regression.select_route_time_features(orpsp_features, target=True)

    time_encoder = route_time_regression.train_encoder(route_time_features)
    pickle.dump(time_encoder, open(models_path + "time_encoder.pkl", "wb"))

    encoded_time_features = route_time_regression.transform_encoder(route_time_features, time_encoder)
    x_train, x_test, y_train, y_test = route_time_regression.route_time_train_test_split(encoded_time_features)

    time_model = route_time_regression.train_route_time_regression(x_train, y_train)
    pickle.dump(time_model, open(models_path + "time_model.pkl", "wb"))
