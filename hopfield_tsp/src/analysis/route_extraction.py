"""Extracción de la ruta a partir de la matriz de activaciones final.

La matriz V codifica, en cada columna i, qué ciudad ocupa la posición i de
la ruta. Para leer la ruta se recorre cada posición y se elige la ciudad con
mayor activación, resolviendo empates/duplicados de forma codiciosa.
"""

from __future__ import annotations

import numpy as np

from ..instances.cities import Instance


def extract_route(V: np.ndarray) -> list[int]:
    """Extrae la secuencia de ciudades leyendo cada posición de la ruta.

    Para cada posición (columna) se asigna la ciudad disponible con mayor
    activación. La asignación codiciosa evita repetir ciudades incluso si la
    matriz no es una permutación perfecta.

    Args:
        V: Matriz de activaciones (N, N).

    Returns:
        Lista de índices de ciudades en el orden de la ruta.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    route: list[int] = []
    used: set[int] = set()

    for pos in range(n):
        column = V[:, pos].copy()
        # Descarta las ciudades ya asignadas.
        for city in used:
            column[city] = -np.inf
        city = int(np.argmax(column))
        route.append(city)
        used.add(city)

    return route


def route_to_labels(route: list[int], instance: Instance) -> list[str]:
    """Traduce una ruta de índices a etiquetas de ciudades (A, B, C, ...)."""
    return [instance.labels[i] for i in route]


def route_as_cycle_string(route: list[int], instance: Instance) -> str:
    """Representa la ruta como cadena cerrada, p. ej. 'A-B-C-D-A'."""
    labels = route_to_labels(route, instance)
    return "-".join(labels + [labels[0]])
