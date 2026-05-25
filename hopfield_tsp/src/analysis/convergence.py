"""Análisis de convergencia y decrecimiento de la energía.

La curva E(t) es el principal instrumento de diagnóstico del modelo: si la
energía decrece monótonamente y se estabiliza, el sistema se comporta como
predice la teoría (la energía actúa como función de Lyapunov).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvergenceReport:
    """Diagnóstico de la trayectoria de energía.

    Attributes:
        is_monotonic: True si E(t) no aumenta (dentro de la tolerancia).
        max_increase: Mayor aumento puntual de energía observado.
        converged: True si la energía se estabilizó al final.
        final_energy: Valor de energía en el último tiempo.
        settle_index: Índice muestreado en el que la energía se considera
            estabilizada (o -1 si no se detectó).
    """

    is_monotonic: bool
    max_increase: float
    converged: bool
    final_energy: float
    settle_index: int


def is_monotonic_decreasing(energies: np.ndarray, tol: float = 1e-6) -> bool:
    """Indica si la sucesión de energías es no creciente.

    Args:
        energies: Vector de energías muestreadas E(t).
        tol: Tolerancia para aumentos numéricos despreciables.

    Returns:
        True si ningún paso aumenta la energía por encima de la tolerancia.
    """
    energies = np.asarray(energies, dtype=float)
    diffs = np.diff(energies)
    return bool(np.all(diffs <= tol))


def analyze_convergence(
    energies: np.ndarray,
    settle_tol: float = 1e-4,
    window: int = 10,
) -> ConvergenceReport:
    """Analiza la trayectoria de energía: monotonía y estabilización.

    Args:
        energies: Vector de energías muestreadas E(t).
        settle_tol: Variación relativa máxima en la ventana final para
            declarar convergencia.
        window: Número de muestras finales usadas para evaluar la estabilidad.

    Returns:
        Un ``ConvergenceReport`` con el diagnóstico completo.
    """
    energies = np.asarray(energies, dtype=float)
    diffs = np.diff(energies)
    max_increase = float(np.max(diffs)) if diffs.size else 0.0

    settle_index = -1
    converged = False
    if energies.size >= window:
        tail = energies[-window:]
        spread = float(np.max(tail) - np.min(tail))
        scale = max(abs(float(energies[-1])), 1.0)
        if spread / scale < settle_tol:
            converged = True
            settle_index = int(energies.size - window)

    return ConvergenceReport(
        is_monotonic=is_monotonic_decreasing(energies),
        max_increase=max_increase,
        converged=converged,
        final_energy=float(energies[-1]) if energies.size else float("nan"),
        settle_index=settle_index,
    )
