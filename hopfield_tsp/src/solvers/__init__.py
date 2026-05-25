"""Solucionadores numéricos para el sistema de EDOs de Hopfield-Tank.

Expone un tipo de resultado común (``SimulationResult``) y los dos métodos
de integración estudiados en el informe: Euler explícito y Runge-Kutta 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model.parameters import HopfieldParameters


@dataclass
class SimulationResult:
    """Resultado de una integración numérica del sistema.

    Attributes:
        method: Nombre del método empleado ('euler' o 'rk4').
        times: Tiempos muestreados (vector de longitud T).
        energies: Energía E(t) en cada tiempo muestreado.
        V_history: Activaciones muestreadas, forma (T, N, N).
        u_final: Potenciales internos en el estado final (N, N).
        V_final: Activaciones en el estado final (N, N).
        distance_matrix: Matriz de distancias usada (N, N).
        params: Parámetros del modelo empleados.
    """

    method: str
    times: np.ndarray
    energies: np.ndarray
    V_history: np.ndarray
    u_final: np.ndarray
    V_final: np.ndarray
    distance_matrix: np.ndarray
    params: HopfieldParameters = field(default_factory=HopfieldParameters)

    @property
    def final_energy(self) -> float:
        """Energía del estado final."""
        return float(self.energies[-1])

    @property
    def n(self) -> int:
        """Número de ciudades de la instancia simulada."""
        return self.V_final.shape[0]


# Las importaciones van después de definir SimulationResult para que los
# submódulos puedan referenciarlo sin provocar importaciones circulares.
from .euler import euler_integrate  # noqa: E402
from .runge_kutta import rk4_integrate  # noqa: E402

__all__ = [
    "SimulationResult",
    "euler_integrate",
    "rk4_integrate",
]
