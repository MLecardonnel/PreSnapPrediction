import json
import pickle
from pathlib import Path

with open(Path(__file__).parents[0].as_posix() + "/default_formations.json") as file:
    default_formations: dict = json.load(file)


models_path = (Path(__file__).parents[3] / "models").as_posix() + "/"
encoder = pickle.load(open(models_path + "encoder.pkl", "rb"))
classification_model = pickle.load(open(models_path + "classification_model.pkl", "rb"))
time_encoder = pickle.load(open(models_path + "time_encoder.pkl", "rb"))
time_model = pickle.load(open(models_path + "time_model.pkl", "rb"))


def _find_intermediate_straight_point(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    xi = (x1 + x2) / 2
    yi = (y1 + y2) / 2
    return xi, yi


def _find_intermediate_45angle_point(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    if x1 <= x2:
        xi = x2 - abs(y1 - y2)
        yi = y1
        if xi < x1:
            yi = y2 - xi * ((y1 - y2) / abs(y1 - y2))
            xi = x2
    else:
        xi = x2 + abs(y1 - y2)
        yi = y1
        if xi > x1:
            yi = y2 + xi * ((y1 - y2) / abs(y1 - y2))
            xi = x2
    return xi, yi


def _find_intermediate_90angle_point(x2: float, y1: float) -> tuple[float, float]:
    xi = x2
    yi = y1
    return xi, yi


def find_intermediate_point(x1: float, y1: float, x2: float, y2: float, route_type: str) -> tuple[float, float]:
    """Determines the intermediate point between two coordinates based on the specified route type.

    Parameters
    ----------
    x1 : float
        The x-coordinate of the starting point.
    y1 : float
        The y-coordinate of the starting point.
    x2 : float
        The x-coordinate of the ending point.
    y2 : float
        The y-coordinate of the ending point.
    route_type : str
        The type of route used to calculate the intermediate point.

    Returns
    -------
    tuple[float, float]
        A tuple representing the x and y coordinates of the intermediate point.

    Raises
    ------
    ValueError
        If route_type is not one of the supported values: ['straight', '45angle', '90angle'].
    """
    if route_type == "straight":
        xi, yi = _find_intermediate_straight_point(x1, y1, x2, y2)
    elif route_type == "45angle":
        xi, yi = _find_intermediate_45angle_point(x1, y1, x2, y2)
    elif route_type == "90angle":
        xi, yi = _find_intermediate_90angle_point(x2, y1)
    else:
        raise ValueError("Invalid route_type, should be a value among ['straight', '45angle', '90angle']")
    return xi, yi
