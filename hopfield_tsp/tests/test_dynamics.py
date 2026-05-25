"""Pruebas de la dinámica: sigmoide y sistema de EDOs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.instances import FOUR_CITIES, distance_matrix_for
from src.model import HopfieldParameters, activation, du_dt


def test_activation_bounds():
    u = np.array([[-100.0, 0.0], [100.0, 5.0]])
    V = activation(u, u0=0.02)
    assert np.all(V >= 0.0)
    assert np.all(V <= 1.0)


def test_activation_midpoint_at_zero():
    u = np.zeros((3, 3))
    V = activation(u, u0=0.02)
    assert np.allclose(V, 0.5)


def test_du_dt_shape():
    d = distance_matrix_for(FOUR_CITIES)
    params = HopfieldParameters()
    u = np.zeros((4, 4))
    deriv = du_dt(u, d, params)
    assert deriv.shape == (4, 4)


def test_pure_decay_when_penalties_off():
    # Con A=B=C=D=0 y d=0, la dinámica se reduce al decaimiento -u/tau.
    n = 4
    d = np.zeros((n, n))
    params = HopfieldParameters(A=0, B=0, C=0, D=0, tau=2.0)
    rng = np.random.default_rng(3)
    u = rng.standard_normal((n, n))
    deriv = du_dt(u, d, params)
    assert np.allclose(deriv, -u / params.tau)
