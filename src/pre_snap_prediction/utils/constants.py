FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3


ROUTES_CONVERSION = {
    "GO": "straight",
    "HITCH": "shortstraight",
    "SLANT": "early45angle",
    "CROSS": "early45angle",
    "POST": "late45angle",
    "CORNER": "late45angle",
    "OUT": "90angle",
    "IN": "90angle",
    "FLAT": "flat",
    "SCREEN": "screen",
    "ANGLE": "wheelangle",
    "WHEEL": "wheelangle",
}


FEATURES = [
    "gameId",
    "down",
    "yardsToGo",
    "offenseFormation",
    "receiverAlignment",
    "playAction",
    "position",
    "route_mode",
    "route_time_mean",
    "dis_recep_zone",
    "dir_recep_zone",
    "recep_zone_dis_out_of_bounds",
    "recep_zone_dis_yard_line",
    "dis_out_of_bounds",
    "dis_back_of_endzone",
    "dis_yard_line",
    "nb_early45angle",
    "nb_flat",
    "nb_90angle",
    "nb_late45angle",
    "nb_shortstraight",
    "nb_wheelangle",
    "nb_screen",
    "nb_straight",
    "nb_routes",
    "nb_recep_zone_negative",
    "nb_recep_zone_5",
    "nb_recep_zone_10",
    "nb_recep_zone_20",
    "nb_recep_zone_inf",
    "dis_ball_to_recep_zone",
    "dir_ball_to_recep_zone",
]


FEATURES_TO_ENCODE = [
    "offenseFormation",
    "receiverAlignment",
    "playAction",
    "route_mode",
    "position",
]
