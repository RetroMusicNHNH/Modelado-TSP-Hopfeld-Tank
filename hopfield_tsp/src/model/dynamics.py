"""Dinámica del modelo: sigmoide y sistema de EDOs acopladas.

A cada activación V_xi se asocia un potencial interno u_xi mediante la
sigmoide

    V_xi = 0.5 * (1 + tanh(u_xi / u0)).

La evolución del potencial sigue el gradiente negativo de la energía
(ec. de la EDO del informe):

    du_xi/dt = -u_xi/tau
               - A * sum_{j!=i} V_xj
               - B * sum_{y!=x} V_yi
               - C * ( sum_{y,j} V_yj - N )
               - D * sum_{y!=x} d_xy ( V_{y,i+1} + V_{y,i-1} ).

Hay una ecuación por cada par (x, i): N^2 EDOs acopladas en total.
"""

from __future__ import annotations

import numpy as np

from .energy import neighbor_sum
from .parameters import HopfieldParameters


def activation(u: np.ndarray, u0: float) -> np.ndarray:
    """Aplica la sigmoide V = 0.5 (1 + tanh(u / u0)) elemento a elemento.

    Args:
        u: Matriz (N, N) de potenciales internos.
        u0: Parámetro de pendiente de la sigmoide.

    Returns:
        Matriz (N, N) de activaciones en [0, 1].
    """
    return 0.5 * (1.0 + np.tanh(np.asarray(u, dtype=float) / u0))


def du_dt(u: np.ndarray, d: np.ndarray, params: HopfieldParameters) -> np.ndarray:
    """Calcula la derivada temporal du/dt para todo el sistema.

    Implementa, de forma vectorizada, la ecuación de la dinámica del informe.
    Los términos por filas y columnas usan la convención sum_{j!=i}, que se
    expresa como (suma total de la fila/columna) - V_xi.

    Args:
        u: Matriz (N, N) de potenciales internos.
        d: Matriz de distancias (N, N).
        params: Parámetros del modelo.

    Returns:
        Matriz (N, N) con du_xi/dt para cada par (x, i).
    """
    u = np.asarray(u, dtype=float)
    n = u.shape[0]
    V = activation(u, params.u0)

    # Término A: -A * sum_{j!=i} V_xj = -A * (suma de la fila - V_xi).
    row_sums = V.sum(axis=1, keepdims=True)
    term_a = -params.A * (row_sums - V)

    # Término B: -B * sum_{y!=x} V_yi = -B * (suma de la columna - V_xi).
    col_sums = V.sum(axis=0, keepdims=True)
    term_b = -params.B * (col_sums - V)

    # Término C: -C * (suma total - N), idéntico para todas las neuronas.
    term_c = -params.C * (V.sum() - n)

    # Término D: -D * sum_{y!=x} d_xy (V_{y,i+1} + V_{y,i-1}).
    # d tiene diagonal nula, así que incluir y=x no afecta el resultado.
    term_d = -params.D * (d @ neighbor_sum(V))

    # Decaimiento pasivo -u/tau (relajación disipativa).
    decay = -u / params.tau

    return decay + term_a + term_b + term_c + term_d


def random_initial_potentials(
    n: int, u0: float, amplitude: float = 0.1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Genera potenciales iniciales pequeños y aleatorios alrededor de cero.

    Se centra el estado cerca de V ~ 0.5 (la sigmoide en su punto medio) y se
    añade una perturbación pequeña para romper la simetría, tal como recomienda
    la literatura de Hopfield-Tank.

    Args:
        n: Número de ciudades (la matriz será n x n).
        u0: Parámetro de pendiente; escala la perturbación inicial.
        amplitude: Amplitud relativa del ruido respecto de u0.
        rng: Generador opcional de números aleatorios (para reproducibilidad).

    Returns:
        Matriz (N, N) de potenciales iniciales.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Pequeño sesgo que ya satisface aproximadamente la restricción de suma N.
    bias = u0 * np.arctanh(2.0 * (1.0 / n) - 1.0)
    noise = amplitude * u0 * rng.standard_normal((n, n))
    return bias + noise
