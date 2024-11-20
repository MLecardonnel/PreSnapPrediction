import pickle
from pathlib import Path

import polars as pl
from pre_snap_prediction.data import process_data, process_orpsp
from pre_snap_prediction.modeling import orpsp_classification
from sklearn.metrics import balanced_accuracy_score

data_path = (Path(__file__).parents[2] / "data").as_posix() + "/"
models_path = (Path(__file__).parents[2] / "models").as_posix() + "/"


if __name__ == "__main__":
    complete_plays = pl.read_csv(data_path + "complete_plays.csv", null_values=["NA", ""])

    plays = pl.read_csv(data_path + "plays.csv", null_values=["NA", ""])
    player_play = pl.read_csv(data_path + "player_play.csv", null_values=["NA", ""])

    orpsp_target = process_orpsp.create_orpsp_target(complete_plays, plays, player_play)
    orpsp_target.write_csv(data_path + "orpsp_target.csv", null_value="NA")

    clusters_route = pl.read_csv(data_path + "clusters_route.csv", null_values=["NA", ""])
    clusters_route_mode = pl.read_csv(data_path + "clusters_route_mode.csv", null_values=["NA", ""])
    clusters_reception_zone = pl.read_csv(data_path + "clusters_reception_zone.csv", null_values=["NA", ""])

    tracking = process_data.read_tracking_csv()
    players = pl.read_csv(data_path + "players.csv", null_values=["NA", ""])

    plays_features = process_orpsp.get_plays_features(complete_plays, plays)
    clusters_features = process_orpsp.get_clusters_features(
        complete_plays, clusters_route, clusters_route_mode, clusters_reception_zone
    )
    tracking_features = process_orpsp.get_tracking_features(complete_plays, tracking)
    start_features = process_orpsp.get_start_features(tracking_features)

    orpsp_features = process_orpsp.preprocess_orpsp_features(plays_features, clusters_features, start_features, players)
    orpsp_features = process_orpsp.compute_orpsp_features(orpsp_features)
    orpsp_features.write_csv(data_path + "orpsp_features.csv", null_value="NA")

    features = orpsp_classification.select_orpsp_features(orpsp_features)

    encoder = orpsp_classification.train_encoder(features)
    pickle.dump(encoder, open(models_path + "encoder.pkl", "wb"))

    features_encoded = orpsp_classification.transform_encoder(features, encoder)
    x_train, x_test, y_train, y_test = orpsp_classification.orpsp_train_test_split(features_encoded, orpsp_target)

    classification_model = orpsp_classification.train_orpsp_classification(x_train, y_train)
    pickle.dump(classification_model, open(models_path + "classification_model.pkl", "wb"))

    y_pred = orpsp_classification.predict_orpsp_class(x_test, classification_model)
    y_train_pred = orpsp_classification.predict_orpsp_class(x_train, classification_model)
    print("Train Balanced Accuracy: " + str(balanced_accuracy_score(y_train, y_train_pred)))
    print("Test Balanced Accuracy: " + str(balanced_accuracy_score(y_test, y_pred)))

    orpsp_predictions = orpsp_classification.predict_orpsp(features_encoded, classification_model)
    orpsp_predictions.write_csv(data_path + "orpsp_predictions.csv", null_value="NA")
