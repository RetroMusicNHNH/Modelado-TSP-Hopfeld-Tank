"""Pruebas de los solucionadores numéricos (Euler y RK4)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.analysis import analyze_convergence
from src.instances import FOUR_CITIES, distance_matrix_for
from src.model import HopfieldParameters, random_initial_potentials
from src.solvers import euler_integrate, rk4_integrate


def _setup(steps=3000):
    d = distance_matrix_for(FOUR_CITIES)
    params = HopfieldParameters(dt=1e-5, steps=steps)
    rng = np.random.default_rng(7)
    u0 = random_initial_potentials(FOUR_CITIES.n, params.u0, rng=rng)
    return u0, d, params


def test_euler_decreases_energy():
    u0, d, params = _setup()
    res = euler_integrate(u0, d, params, record_every=50)
    assert res.energies[-1] <= res.energies[0]


def test_rk4_decreases_energy():
    u0, d, params = _setup()
    res = rk4_integrate(u0, d, params, record_every=50)
    assert res.energies[-1] <= res.energies[0]


def test_rk4_energy_decreases_and_settles():
    # La dinámica documentada usa la convención sum_{j!=i}, que NO es el
    # gradiente exacto de la energía; por eso la energía puede sobrepasar y
    # reacomodarse en lugar de decrecer de forma estrictamente monótona.
    # Lo que sí debe cumplirse es una caída global grande y una estabilización
    # final, con incrementos puntuales pequeños frente a esa caída.
    u0, d, params = _setup()
    res = rk4_integrate(u0, d, params, record_every=50)
    assert res.energies[-1] < res.energies[0]
    assert analyze_convergence(res.energies).converged
    increase = max(0.0, float(np.max(np.diff(res.energies))))
    total_drop = float(res.energies[0] - res.energies[-1])
    assert increase < 0.05 * total_drop


def test_result_shapes():
    u0, d, params = _setup(steps=1000)
    res = euler_integrate(u0, d, params, record_every=100)
    assert res.V_final.shape == (4, 4)
    assert res.u_final.shape == (4, 4)
    assert res.times.shape[0] == res.energies.shape[0]
    assert res.V_history.shape[1:] == (4, 4)


def test_euler_and_rk4_agree_for_small_step():
    # Con dt muy pequeño ambos métodos deben coincidir aproximadamente.
    d = distance_matrix_for(FOUR_CITIES)
    params = HopfieldParameters(dt=1e-6, steps=2000)
    rng = np.random.default_rng(7)
    u0 = random_initial_potentials(FOUR_CITIES.n, params.u0, rng=rng)
    e = euler_integrate(u0.copy(), d, params, record_every=2000)
    r = rk4_integrate(u0.copy(), d, params, record_every=2000)
    assert np.isclose(e.final_energy, r.final_energy, rtol=1e-2)
