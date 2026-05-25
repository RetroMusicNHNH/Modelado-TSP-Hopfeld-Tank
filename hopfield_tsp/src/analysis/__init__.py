"""Análisis de resultados: convergencia, extracción de ruta y validación."""

from .convergence import (
    ConvergenceReport,
    analyze_convergence,
    is_monotonic_decreasing,
)
from .route_extraction import (
    extract_route,
    route_as_cycle_string,
    route_to_labels,
)
from .validation import (
    ValidationReport,
    binarize,
    route_length,
    validate,
)

__all__ = [
    "ConvergenceReport",
    "analyze_convergence",
    "is_monotonic_decreasing",
    "extract_route",
    "route_to_labels",
    "route_as_cycle_string",
    "ValidationReport",
    "validate",
    "binarize",
    "route_length",
]
