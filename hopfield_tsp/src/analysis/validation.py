"""Validación de soluciones: ¿es V una matriz de permutación válida?

Una solución del TSP corresponde a una matriz de permutación: cada fila y
cada columna contienen exactamente un uno. Aquí se verifica esa condición
sobre el estado final del modelo (tras redondear las activaciones) y se
calcula la longitud de la ruta resultante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationReport:
    """Resumen de la validez de una solución.

    Attributes:
        is_permutation: True si V (redondeada) es matriz de permutación.
        row_violations: Número de filas cuya suma no es 1.
        col_violations: Número de columnas cuya suma no es 1.
        total_activation: Suma total de la matriz redondeada.
    """

    is_permutation: bool
    row_violations: int
    col_violations: int
    total_activation: float


def binarize(V: np.ndarray) -> np.ndarray:
    """Convierte V en 0/1 tomando el máximo de cada columna (posición).

    Cada posición de la ruta se asigna a la ciudad con mayor activación en
    esa columna. Esto produce una matriz binaria que luego se valida.

    Args:
        V: Matriz de activaciones (N, N).

    Returns:
        Matriz binaria (N, N) de tipo entero.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    binary = np.zeros_like(V, dtype=int)
    winners = np.argmax(V, axis=0)  # ciudad ganadora por cada posición
    binary[winners, np.arange(n)] = 1
    return binary


def validate(V: np.ndarray, threshold: float = 0.5) -> ValidationReport:
    """Verifica si V representa una matriz de permutación válida.

    Args:
        V: Matriz de activaciones (N, N).
        threshold: Umbral para considerar una activación como "encendida"
            al evaluar las sumas por filas y columnas.

    Returns:
        Un ``ValidationReport`` con el detalle de las violaciones.
    """
    V = np.asarray(V, dtype=float)
    binary = (V >= threshold).astype(int)

    row_sums = binary.sum(axis=1)
    col_sums = binary.sum(axis=0)
    row_violations = int(np.sum(row_sums != 1))
    col_violations = int(np.sum(col_sums != 1))

    return ValidationReport(
        is_permutation=(row_violations == 0 and col_violations == 0),
        row_violations=row_violations,
        col_violations=col_violations,
        total_activation=float(binary.sum()),
    )


def route_length(route: list[int] | np.ndarray, d: np.ndarray) -> float:
    """Calcula la longitud total de una ruta cerrada.

    Args:
        route: Secuencia de índices de ciudades (sin repetir el inicio al
            final); se asume que la ruta regresa al punto de partida.
        d: Matriz de distancias (N, N).

    Returns:
        Longitud total del ciclo, incluyendo el regreso al origen.
    """
    route = list(route)
    d = np.asarray(d, dtype=float)
    total = 0.0
    for k in range(len(route)):
        a = route[k]
        b = route[(k + 1) % len(route)]
        total += d[a, b]
    return float(total)
