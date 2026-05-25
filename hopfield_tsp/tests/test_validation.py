"""Pruebas de validación, extracción de ruta y fuerza bruta."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.analysis import extract_route, route_length, validate
from src.brute_force import brute_force_optimal
from src.instances import FIVE_CITIES, FOUR_CITIES, distance_matrix_for


def test_permutation_matrix_is_valid():
    V = np.eye(4)
    report = validate(V)
    assert report.is_permutation
    assert report.row_violations == 0
    assert report.col_violations == 0


def test_invalid_matrix_is_detected():
    # Dos ciudades en la misma posición, una posición vacía.
    V = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    report = validate(V)
    assert not report.is_permutation


def test_extract_route_is_a_permutation():
    rng = np.random.default_rng(0)
    V = rng.random((5, 5))
    route = extract_route(V)
    assert sorted(route) == list(range(5))


def test_four_city_optimal_route():
    d = distance_matrix_for(FOUR_CITIES)
    result = brute_force_optimal(d)
    # La ruta óptima es A-B-C-D-A (índices 0-1-2-3) con longitud ~6.56.
    assert np.isclose(result.length, 6.5616, atol=1e-3)
    assert result.num_routes_evaluated == 6  # (4-1)! = 6 permutaciones


def test_five_city_optimal_route():
    d = distance_matrix_for(FIVE_CITIES)
    result = brute_force_optimal(d)
    # La ruta óptima es A-B-C-D-E-A con longitud ~11.72.
    assert np.isclose(result.length, 11.708, atol=1e-2)
    assert result.num_routes_evaluated == 24  # (5-1)! = 24 permutaciones


def test_route_length_matches_manual_sum():
    d = distance_matrix_for(FOUR_CITIES)
    route = [0, 1, 2, 3]
    manual = d[0, 1] + d[1, 2] + d[2, 3] + d[3, 0]
    assert np.isclose(route_length(route, d), manual)
