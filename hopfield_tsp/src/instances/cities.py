"""Instancias del problema: coordenadas de las ciudades en el plano.

Define las dos instancias estudiadas en el informe:

- Instancia de 4 ciudades (cuadrilátero irregular). Ruta óptima de contorno.
- Instancia de 5 ciudades (pentágono irregular). Ruta óptima no evidente.

Las etiquetas (A, B, C, ...) se generan a partir del orden de las filas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    """Una instancia del TSP: nombre, etiquetas y coordenadas.

    Attributes:
        name: Identificador legible de la instancia.
        labels: Etiquetas de las ciudades (p. ej. ['A', 'B', 'C', 'D']).
        coords: Arreglo (N, 2) con las coordenadas (x, y) de cada ciudad.
    """

    name: str
    labels: tuple[str, ...]
    coords: np.ndarray

    @property
    def n(self) -> int:
        """Número de ciudades de la instancia."""
        return len(self.labels)


def _make_labels(n: int) -> tuple[str, ...]:
    """Genera etiquetas A, B, C, ... para n ciudades."""
    return tuple(chr(ord("A") + i) for i in range(n))


# ---------------------------------------------------------------------------
# Instancia de 4 ciudades: cuadrilátero irregular.
# A=(0,0), B=(2,0), C=(2,1.5), D=(0,1). Ruta óptima: A-B-C-D-A (~6.56).
# ---------------------------------------------------------------------------
FOUR_CITIES = Instance(
    name="4-cities",
    labels=_make_labels(4),
    coords=np.array(
        [
            [0.0, 0.0],  # A
            [2.0, 0.0],  # B
            [2.0, 1.5],  # C
            [0.0, 1.0],  # D
        ]
    ),
)


# ---------------------------------------------------------------------------
# Instancia de 5 ciudades: pentágono irregular.
# A=(0,0), B=(3,0), C=(4,2), D=(2,3), E=(0,2). Ruta óptima: A-B-C-D-E-A (~11.72).
# ---------------------------------------------------------------------------
FIVE_CITIES = Instance(
    name="5-cities",
    labels=_make_labels(5),
    coords=np.array(
        [
            [0.0, 0.0],  # A
            [3.0, 0.0],  # B
            [4.0, 2.0],  # C
            [2.0, 3.0],  # D
            [0.0, 2.0],  # E
        ]
    ),
)


INSTANCES = {
    FOUR_CITIES.name: FOUR_CITIES,
    FIVE_CITIES.name: FIVE_CITIES,
}


def get_instance(name: str) -> Instance:
    """Recupera una instancia registrada por su nombre.

    Args:
        name: Nombre de la instancia ('4-cities' o '5-cities').

    Returns:
        La instancia correspondiente.

    Raises:
        KeyError: Si el nombre no está registrado.
    """
    if name not in INSTANCES:
        disponibles = ", ".join(INSTANCES)
        raise KeyError(f"Instancia '{name}' desconocida. Disponibles: {disponibles}")
    return INSTANCES[name]
