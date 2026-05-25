"""Visualización de la matriz de activaciones V como mapa de calor.

Cada fila es una ciudad y cada columna una posición de la ruta. En una
solución válida el mapa muestra exactamente un valor cercano a 1 por fila y
por columna (una matriz de permutación).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_activation_matrix(
    V: np.ndarray,
    labels: list[str] | tuple[str, ...] | None = None,
    title: str = "Matriz de activaciones $V$",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja la matriz V como mapa de calor con anotaciones.

    Args:
        V: Matriz de activaciones (N, N).
        labels: Etiquetas de las ciudades para el eje de filas.
        title: Título de la gráfica.
        save_path: Si se indica, guarda la figura en esa ruta.
        ax: Ejes existentes sobre los que dibujar (opcional).

    Returns:
        Los ejes de Matplotlib utilizados.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    im = ax.imshow(V, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Anota el valor numérico en cada celda.
    for x in range(n):
        for i in range(n):
            value = V[x, i]
            color = "white" if value > 0.5 else "black"
            ax.text(i, x, f"{value:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    if labels is not None:
        ax.set_yticks(range(n))
        ax.set_yticklabels(list(labels))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{i + 1}" for i in range(n)])
    ax.set_xlabel("Posición en la ruta")
    ax.set_ylabel("Ciudad")
    ax.set_title(title)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
