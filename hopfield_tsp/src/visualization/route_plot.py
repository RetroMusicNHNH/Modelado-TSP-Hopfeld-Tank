"""Visualización de una ruta sobre el plano de las ciudades.

Dibuja las ciudades como puntos etiquetados y la ruta como un ciclo cerrado,
permitiendo contrastar visualmente la solución del modelo con la óptima.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_route(
    coords: np.ndarray,
    route: list[int],
    labels: list[str] | tuple[str, ...] | None = None,
    title: str = "Ruta",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    color: str = "#003366",
) -> plt.Axes:
    """Dibuja una ruta cerrada sobre las coordenadas de las ciudades.

    Args:
        coords: Coordenadas (N, 2) de las ciudades.
        route: Secuencia de índices que define el orden de visita.
        labels: Etiquetas de las ciudades (opcional).
        title: Título de la gráfica.
        save_path: Si se indica, guarda la figura en esa ruta.
        ax: Ejes existentes sobre los que dibujar (opcional).
        color: Color de las aristas de la ruta.

    Returns:
        Los ejes de Matplotlib utilizados.
    """
    coords = np.asarray(coords, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    # Cierra el ciclo repitiendo la primera ciudad al final.
    cycle = list(route) + [route[0]]
    path = coords[cycle]
    ax.plot(path[:, 0], path[:, 1], "-", color=color, lw=1.8, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], s=120, color="#FF8C00",
               edgecolors="black", zorder=2)

    for idx, (x, y) in enumerate(coords):
        label = labels[idx] if labels is not None else str(idx)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
