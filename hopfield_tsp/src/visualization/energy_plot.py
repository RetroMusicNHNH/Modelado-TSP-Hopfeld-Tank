"""Gráfica de la energía E(t) frente al tiempo.

El decrecimiento monótono de esta curva confirma que la función de energía
actúa como función de Lyapunov del sistema.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_energy(
    times: np.ndarray,
    energies: np.ndarray,
    title: str = "Energía E(t)",
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja la curva de energía frente al tiempo.

    Args:
        times: Tiempos muestreados.
        energies: Energía en cada tiempo.
        title: Título de la gráfica.
        save_path: Si se indica, guarda la figura en esa ruta.
        ax: Ejes existentes sobre los que dibujar (opcional).

    Returns:
        Los ejes de Matplotlib utilizados.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(times, energies, color="#003366", lw=1.8)
    ax.set_xlabel("Tiempo $t$")
    ax.set_ylabel("Energía $E(t)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax


def plot_energy_comparison(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Comparación de energía: Euler vs RK4",
    save_path: str | Path | None = None,
) -> plt.Axes:
    """Superpone varias curvas de energía para comparar métodos.

    Args:
        results: Diccionario {etiqueta: (tiempos, energías)}.
        title: Título de la gráfica.
        save_path: Si se indica, guarda la figura en esa ruta.

    Returns:
        Los ejes de Matplotlib utilizados.
    """
    _, ax = plt.subplots(figsize=(7, 4.5))
    for label, (times, energies) in results.items():
        ax.plot(times, energies, lw=1.8, label=label)
    ax.set_xlabel("Tiempo $t$")
    ax.set_ylabel("Energía $E(t)$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    return ax
