"""Función de energía de Hopfield-Tank para el TSP.

La energía tiene cuatro términos (ec. de energía del informe):

    E(V) = A/2 * sum_x ( sum_i V_xi - 1 )^2          # cada ciudad una vez
         + B/2 * sum_i ( sum_x V_xi - 1 )^2          # cada posición una vez
         + C/2 * ( sum_xi V_xi - N )^2               # total de activaciones = N
         + D/2 * sum_x sum_{y!=x} d_xy * sum_i V_xi (V_{y,i+1} + V_{y,i-1})

V es una matriz (N, N): la fila x es la ciudad, la columna i es la posición.
Los índices i+1 e i-1 son cíclicos porque la ruta es un ciclo cerrado.
"""

from __future__ import annotations

import numpy as np

from .parameters import HopfieldParameters


def neighbor_sum(V: np.ndarray) -> np.ndarray:
    """Suma de activaciones en posiciones adyacentes (cíclicas).

    Para cada (y, i) devuelve V_{y,i+1} + V_{y,i-1} con índices módulo N.

    Args:
        V: Matriz de activaciones (N, N).

    Returns:
        Matriz (N, N) con la suma de vecinos por posición.
    """
    # roll(-1) trae la columna i+1 a la posición i; roll(+1) trae la i-1.
    return np.roll(V, -1, axis=1) + np.roll(V, 1, axis=1)


def energy(V: np.ndarray, d: np.ndarray, params: HopfieldParameters) -> float:
    """Evalúa la función de energía total E(V).

    Args:
        V: Matriz de activaciones (N, N) con valores en [0, 1].
        d: Matriz de distancias (N, N).
        params: Parámetros del modelo (A, B, C, D, ...).

    Returns:
        Valor escalar de la energía.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]

    row_sums = V.sum(axis=1)  # sum_i V_xi  por ciudad x
    col_sums = V.sum(axis=0)  # sum_x V_xi  por posición i
    total = V.sum()

    term_a = 0.5 * params.A * np.sum((row_sums - 1.0) ** 2)
    term_b = 0.5 * params.B * np.sum((col_sums - 1.0) ** 2)
    term_c = 0.5 * params.C * (total - n) ** 2
    term_d = 0.5 * params.D * np.sum(V * (d @ neighbor_sum(V)))

    return float(term_a + term_b + term_c + term_d)


def energy_terms(
    V: np.ndarray, d: np.ndarray, params: HopfieldParameters
) -> dict[str, float]:
    """Descompone la energía en sus cuatro términos por separado.

    Útil para diagnóstico: permite ver qué penalización domina durante la
    evolución temporal.

    Returns:
        Diccionario con las claves 'A', 'B', 'C', 'D' y 'total'.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]

    row_sums = V.sum(axis=1)
    col_sums = V.sum(axis=0)
    total = V.sum()

    term_a = 0.5 * params.A * np.sum((row_sums - 1.0) ** 2)
    term_b = 0.5 * params.B * np.sum((col_sums - 1.0) ** 2)
    term_c = 0.5 * params.C * (total - n) ** 2
    term_d = 0.5 * params.D * np.sum(V * (d @ neighbor_sum(V)))

    return {
        "A": float(term_a),
        "B": float(term_b),
        "C": float(term_c),
        "D": float(term_d),
        "total": float(term_a + term_b + term_c + term_d),
    }
