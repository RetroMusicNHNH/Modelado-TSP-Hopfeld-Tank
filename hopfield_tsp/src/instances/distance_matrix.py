"""Construcción de la matriz de distancias euclidianas.

La información geométrica del problema se condensa en una matriz simétrica
con ceros en la diagonal, donde la entrada (x, y) es la distancia euclidiana

    d_xy = sqrt((x1 - x2)^2 + (y1 - y2)^2)

Esta matriz alimenta el cuarto término de la función de energía.
"""

from __future__ import annotations

import numpy as np

from .cities import Instance


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calcula la matriz de distancias euclidianas entre puntos del plano.

    Args:
        coords: Arreglo (N, 2) de coordenadas (x, y).

    Returns:
        Matriz (N, N) simétrica, con diagonal nula y entradas no negativas.
    """
    coords = np.asarray(coords, dtype=float)
    # Diferencias por pares mediante broadcasting: (N, 1, 2) - (1, N, 2).
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def distance_matrix_for(instance: Instance) -> np.ndarray:
    """Devuelve la matriz de distancias de una instancia dada."""
    return euclidean_distance_matrix(instance.coords)


def is_valid_distance_matrix(d: np.ndarray, tol: float = 1e-9) -> bool:
    """Verifica las tres propiedades de una matriz de distancias válida.

    Es simétrica, tiene diagonal nula y entradas fuera de la diagonal
    estrictamente positivas.

    Args:
        d: Matriz cuadrada a verificar.
        tol: Tolerancia numérica para las comparaciones.

    Returns:
        True si la matriz cumple las tres propiedades.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        return False
    simetrica = np.allclose(d, d.T, atol=tol)
    diagonal_nula = np.allclose(np.diag(d), 0.0, atol=tol)
    fuera = d[~np.eye(d.shape[0], dtype=bool)]
    positiva = bool(np.all(fuera > tol))
    return bool(simetrica and diagonal_nula and positiva)
