"""Parámetros del modelo de Hopfield-Tank para el TSP.

Los valores por defecto corresponden a los clásicos propuestos en la
literatura original (Hopfield & Tank, 1985) y adoptados en el informe:

    A = B = 500,  C = 200,  D = 500,  tau = 1,  u0 = 0.02

Estos parámetros equilibran las penalizaciones de factibilidad (A, B, C)
contra el término de distancia (D). Si el sistema no converge a soluciones
válidas, conviene explorar variaciones de estos valores.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HopfieldParameters:
    """Conjunto de parámetros que rigen la energía y la dinámica.

    Attributes:
        A: Penaliza que una ciudad ocupe más de una posición (término por filas).
        B: Penaliza que una posición sea ocupada por más de una ciudad (columnas).
        C: Restricción global: el total de activaciones debe ser N.
        D: Pondera la distancia recorrida (término de optimización).
        tau: Constante de tiempo del decaimiento pasivo -u/tau.
        u0: Controla la pendiente de la sigmoide V = 0.5(1 + tanh(u/u0)).
        dt: Paso de integración temporal usado por los solucionadores.
        steps: Número de pasos de integración por defecto.
    """

    A: float = 500.0
    B: float = 500.0
    C: float = 200.0
    D: float = 500.0
    tau: float = 1.0
    u0: float = 0.02
    dt: float = 1e-5
    steps: int = 20_000

    def as_dict(self) -> dict[str, float]:
        """Devuelve los parámetros como diccionario (útil para registro)."""
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "D": self.D,
            "tau": self.tau,
            "u0": self.u0,
            "dt": self.dt,
            "steps": self.steps,
        }

    def __str__(self) -> str:  # pragma: no cover - solo presentación
        return (
            f"A={self.A:g}, B={self.B:g}, C={self.C:g}, D={self.D:g}, "
            f"tau={self.tau:g}, u0={self.u0:g}, dt={self.dt:g}, steps={self.steps}"
        )


# Conjunto de parámetros por defecto reutilizable en todo el proyecto.
DEFAULT_PARAMETERS = HopfieldParameters()
