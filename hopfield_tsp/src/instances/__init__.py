"""Instancias del TSP y construcción de la matriz de distancias."""

from .cities import (
    FIVE_CITIES,
    FOUR_CITIES,
    INSTANCES,
    Instance,
    get_instance,
)
from .distance_matrix import (
    distance_matrix_for,
    euclidean_distance_matrix,
    is_valid_distance_matrix,
)

__all__ = [
    "Instance",
    "FOUR_CITIES",
    "FIVE_CITIES",
    "INSTANCES",
    "get_instance",
    "euclidean_distance_matrix",
    "distance_matrix_for",
    "is_valid_distance_matrix",
]
