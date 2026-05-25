"""Semana 2 - Integración con Euler explícito, instancia de 4 ciudades.

Ejecuta el modelo de Hopfield-Tank sobre el cuadrilátero irregular usando el
método de Euler, contrasta el resultado con la ruta óptima (A-B-C-D-A, ~6.56)
y guarda las figuras y los datos en results/.

Uso (desde la carpeta hopfield_tsp/):
    python experiments/week2_euler_4cities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite importar el paquete `src` al ejecutar el script directamente.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, apto para guardar figuras
import numpy as np

from src.analysis import (
    analyze_convergence,
    extract_route,
    route_as_cycle_string,
    route_length,
    validate,
)
from src.brute_force import brute_force_optimal
from src.instances import FOUR_CITIES, distance_matrix_for
from src.model import HopfieldParameters, random_initial_potentials
from src.solvers import euler_integrate
from src.visualization import plot_activation_matrix, plot_energy, plot_route

FIGURES = ROOT / "results" / "figures"
DATA = ROOT / "results" / "data"


def main() -> None:
    instance = FOUR_CITIES
    d = distance_matrix_for(instance)
    params = HopfieldParameters(dt=1e-5, steps=30_000)

    optimal = brute_force_optimal(d)
    print(f"== {instance.name} | método=euler ==")
    print(f"Óptimo (fuerza bruta): {route_as_cycle_string(optimal.route, instance)}"
          f"  L={optimal.length:.4f}")

    rng = np.random.default_rng(seed=7)
    u0 = random_initial_potentials(instance.n, params.u0, rng=rng)
    result = euler_integrate(u0, d, params, record_every=100)

    conv = analyze_convergence(result.energies)
    report = validate(result.V_final)
    route = extract_route(result.V_final)
    length = route_length(route, d)

    print(f"Energía final: {result.final_energy:.4f}  "
          f"(monótona={conv.is_monotonic}, convergió={conv.converged})")
    print(f"Ruta del modelo: {route_as_cycle_string(route, instance)}  L={length:.4f}")
    print(f"¿Permutación válida? {report.is_permutation}  "
          f"(viol. filas={report.row_violations}, columnas={report.col_violations})")
    gap = 100.0 * (length - optimal.length) / optimal.length
    print(f"Brecha respecto al óptimo: {gap:+.2f}%")

    # Figuras.
    plot_energy(result.times, result.energies,
                title="Euler - Energía E(t), 4 ciudades",
                save_path=FIGURES / "energy" / "week2_euler_4cities.png")
    plot_activation_matrix(result.V_final, labels=instance.labels,
                           title="Euler - Matriz V final, 4 ciudades",
                           save_path=FIGURES / "matrices" / "week2_euler_4cities.png")
    plot_route(instance.coords, route, labels=instance.labels,
               title=f"Euler - Ruta {route_as_cycle_string(route, instance)}",
               save_path=FIGURES / "routes" / "week2_euler_4cities.png")

    # Datos.
    (DATA / "trajectories").mkdir(parents=True, exist_ok=True)
    (DATA / "final_states").mkdir(parents=True, exist_ok=True)
    np.savez(DATA / "trajectories" / "week2_euler_4cities.npz",
             times=result.times, energies=result.energies, V_history=result.V_history)
    np.savez(DATA / "final_states" / "week2_euler_4cities.npz",
             V_final=result.V_final, u_final=result.u_final, route=np.array(route))
    print(f"Figuras y datos guardados en {FIGURES} y {DATA}.")


if __name__ == "__main__":
    main()
