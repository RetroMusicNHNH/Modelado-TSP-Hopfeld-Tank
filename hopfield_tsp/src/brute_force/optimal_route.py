"""Ruta óptima por fuerza bruta (valor de referencia objetivo).

Para instancias pequeñas es factible enumerar todas las rutas posibles y
elegir la de menor longitud. Esto proporciona el valor contra el cual se
contrasta la calidad de las soluciones del modelo dinámico.

Valores de referencia esperados (informe):
    - 4 ciudades: A-B-C-D-A, longitud ~ 6.56
    - 5 ciudades: A-B-C-D-E-A, longitud ~ 11.72
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass
class OptimalResult:
    """Resultado de la búsqueda exhaustiva.

    Attributes:
        route: Ruta óptima como lista de índices (empieza en 0).
        length: Longitud total del ciclo óptimo.
        num_routes_evaluated: Cantidad de rutas distintas evaluadas.
    """

    route: list[int]
    length: float
    num_routes_evaluated: int


def _cycle_length(route: tuple[int, ...], d: np.ndarray) -> float:
    """Longitud de un ciclo cerrado dado por una secuencia de índices."""
    total = 0.0
    n = len(route)
    for k in range(n):
        total += d[route[k], route[(k + 1) % n]]
    return total


def brute_force_optimal(d: np.ndarray) -> OptimalResult:
    """Encuentra la ruta óptima evaluando todas las permutaciones.

    Se fija la ciudad 0 como inicio y se enumeran las permutaciones de las
    restantes. No se eliminan los recorridos invertidos (tienen la misma
    longitud), pero el conteo refleja las permutaciones evaluadas.

    Args:
        d: Matriz de distancias (N, N).

    Returns:
        Un ``OptimalResult`` con la ruta y la longitud óptimas.
    """
    d = np.asarray(d, dtype=float)
    n = d.shape[0]
    rest = list(range(1, n))

    best_route: tuple[int, ...] = tuple(range(n))
    best_length = float("inf")
    count = 0

    for perm in permutations(rest):
        route = (0,) + perm
        length = _cycle_length(route, d)
        count += 1
        if length < best_length:
            best_length = length
            best_route = route

    return OptimalResult(
        route=list(best_route),
        length=float(best_length),
        num_routes_evaluated=count,
    )
