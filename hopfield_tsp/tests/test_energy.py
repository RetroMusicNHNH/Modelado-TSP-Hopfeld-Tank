"""Pruebas de la función de energía."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.analysis.validation import route_length
from src.instances import FOUR_CITIES, distance_matrix_for
from src.model import HopfieldParameters, energy, energy_terms, neighbor_sum


def test_neighbor_sum_is_cyclic():
    V = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ns = neighbor_sum(V)
    # Para la fila 0 (un 1 en la posición 0), los vecinos cíclicos son las
    # posiciones 1 y 2 (i+1 e i-1 con N=3).
    expected = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    assert np.allclose(ns, expected)


def test_permutation_matrix_zeroes_feasibility_terms():
    d = distance_matrix_for(FOUR_CITIES)
    params = HopfieldParameters()
    V = np.eye(4)  # matriz de permutación válida
    terms = energy_terms(V, d, params)
    assert terms["A"] == 0.0
    assert terms["B"] == 0.0
    assert terms["C"] == 0.0
    assert terms["D"] > 0.0


def test_distance_term_equals_D_times_tour_length():
    d = distance_matrix_for(FOUR_CITIES)
    params = HopfieldParameters()
    V = np.eye(4)
    route = [0, 1, 2, 3]  # ruta codificada por la identidad
    terms = energy_terms(V, d, params)
    assert np.isclose(terms["D"], params.D * route_length(route, d))


def test_terms_sum_to_total():
    d = distance_matrix_for(FOUR_CITIES)
    params = HopfieldParameters()
    rng = np.random.default_rng(0)
    V = rng.random((4, 4))
    terms = energy_terms(V, d, params)
    assert np.isclose(terms["total"], energy(V, d, params))
    assert np.isclose(terms["A"] + terms["B"] + terms["C"] + terms["D"], terms["total"])


def test_energy_is_non_negative_for_activations():
    d = distance_matrix_for(FOUR_CITIES)
    params = HopfieldParameters()
    rng = np.random.default_rng(1)
    for _ in range(5):
        V = rng.random((4, 4))
        assert energy(V, d, params) >= 0.0
