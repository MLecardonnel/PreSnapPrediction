import pickle
from pathlib import Path

import polars as pl
from pre_snap_prediction.data import process_data
from pre_snap_prediction.modeling import route_clustering

data_path = (Path(__file__).parents[2] / "data").as_posix() + "/"
models_path = (Path(__file__).parents[2] / "models").as_posix() + "/"


if __name__ == "__main__":
    tracking = process_data.read_tracking_csv()
    player_play = pl.read_csv(data_path + "player_play.csv", null_values="NA")

    inverse_tracking = process_data.inverse_left_directed_plays(tracking)

    route_tracking = process_data.get_route_tracking(inverse_tracking, player_play)
    route_tracking = process_data.get_route_direction(route_tracking)
    route_tracking = process_data.inverse_right_route(route_tracking)
    route_tracking = process_data.process_route_tracking(route_tracking, player_play)

    route_features = process_data.compute_route_features(route_tracking)

    outliers_model = route_clustering.train_outliers_model(route_features.filter(pl.col("week") == 1))
    pickle.dump(outliers_model, open(models_path + "outliers_model.pkl", "wb"))

    outliers_route = route_clustering.predict_outliers(route_features, outliers_model)
    valid_route_features = route_clustering.remove_outliers(outliers_route)

    clustering_model = route_clustering.train_route_clustering(valid_route_features.filter(pl.col("week") == 1))
    pickle.dump(clustering_model, open(models_path + "clustering_model.pkl", "wb"))

    clusters_route = route_clustering.predict_route_cluters(valid_route_features, clustering_model)
    clusters_route.write_csv(data_path + "clusters_route.csv", null_value="NA")

    clusters_route_mode = route_clustering.get_modified_route_mode(player_play, clusters_route)
    clusters_route_mode.write_csv(data_path + "clusters_route_mode.csv", null_value="NA")

    clusters_route_tracking = route_clustering.join_clusters_to_data(route_tracking, clusters_route)

    clusters_reception_zone = route_clustering.get_clusters_reception_zones(player_play, clusters_route_tracking)
    clusters_reception_zone = route_clustering.predict_missing_reception_zone(clusters_route, clusters_reception_zone)
    clusters_reception_zone.write_csv(data_path + "clusters_reception_zone.csv", null_value="NA")
