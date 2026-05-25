"""Semana 4 - Comparación entre Euler y RK4 en ambas instancias.

Integra el sistema con los dos métodos partiendo del mismo estado inicial,
superpone las curvas de energía y resume en una tabla la energía final, la
validez de la solución, la ruta obtenida y la brecha respecto al óptimo.

Uso (desde la carpeta hopfield_tsp/):
    python experiments/week4_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from src.analysis import (
    analyze_convergence,
    extract_route,
    route_as_cycle_string,
    route_length,
    validate,
)
from src.brute_force import brute_force_optimal
from src.instances import FIVE_CITIES, FOUR_CITIES, distance_matrix_for
from src.model import HopfieldParameters, random_initial_potentials
from src.solvers import euler_integrate, rk4_integrate
from src.visualization import plot_energy_comparison

FIGURES = ROOT / "results" / "figures"


def run_methods(instance, steps: int):
    """Corre Euler y RK4 desde el mismo estado inicial y devuelve resultados."""
    d = distance_matrix_for(instance)
    params = HopfieldParameters(dt=1e-5, steps=steps)
    rng = np.random.default_rng(seed=7)
    u0 = random_initial_potentials(instance.n, params.u0, rng=rng)

    return {
        "euler": euler_integrate(u0.copy(), d, params, record_every=100),
        "rk4": rk4_integrate(u0.copy(), d, params, record_every=100),
    }, d


def summarize(instance, results, d) -> None:
    """Imprime una tabla comparativa y guarda la gráfica de energía."""
    optimal = brute_force_optimal(d)
    print(f"\n=== Instancia {instance.name} ===")
    print(f"Óptimo: {route_as_cycle_string(optimal.route, instance)}  "
          f"L={optimal.length:.4f}\n")
    header = f"{'método':>7} | {'E_final':>10} | {'monót.':>6} | {'válida':>6} | {'ruta':>14} | {'L':>7} | {'gap%':>7}"
    print(header)
    print("-" * len(header))

    energy_curves = {}
    for name, res in results.items():
        conv = analyze_convergence(res.energies)
        report = validate(res.V_final)
        route = extract_route(res.V_final)
        length = route_length(route, d)
        gap = 100.0 * (length - optimal.length) / optimal.length
        energy_curves[name.upper()] = (res.times, res.energies)
        print(f"{name:>7} | {res.final_energy:>10.3f} | "
              f"{str(conv.is_monotonic):>6} | {str(report.is_permutation):>6} | "
              f"{route_as_cycle_string(route, instance):>14} | {length:>7.3f} | "
              f"{gap:>+7.2f}")

    plot_energy_comparison(
        energy_curves,
        title=f"Euler vs RK4 - Energía, {instance.name}",
        save_path=FIGURES / "energy" / f"week4_comparison_{instance.name}.png",
    )


def main() -> None:
    results4, d4 = run_methods(FOUR_CITIES, steps=30_000)
    summarize(FOUR_CITIES, results4, d4)

    results5, d5 = run_methods(FIVE_CITIES, steps=40_000)
    summarize(FIVE_CITIES, results5, d5)

    print(f"\nGráficas comparativas guardadas en {FIGURES / 'energy'}.")


if __name__ == "__main__":
    main()
