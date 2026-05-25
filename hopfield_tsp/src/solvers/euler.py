"""Método de Euler explícito para integrar el sistema de Hopfield-Tank.

El esquema de Euler es el más simple: avanza el estado mediante

    u_{k+1} = u_k + dt * f(u_k),

donde f(u) = du/dt. Es de primer orden y sensible al tamaño de paso, pero
sirve como punto de partida para diagnosticar el decrecimiento de la energía.
"""

from __future__ import annotations

import numpy as np

from ..model.dynamics import activation, du_dt
from ..model.energy import energy
from ..model.parameters import HopfieldParameters


def euler_integrate(
    u_init: np.ndarray,
    d: np.ndarray,
    params: HopfieldParameters,
    record_every: int = 100,
):
    """Integra el sistema con Euler explícito y registra la trayectoria.

    Args:
        u_init: Potenciales iniciales (N, N).
        d: Matriz de distancias (N, N).
        params: Parámetros del modelo (usa params.dt y params.steps).
        record_every: Cada cuántos pasos se guarda un punto de la trayectoria.

    Returns:
        Un ``SimulationResult`` con tiempos, energías e historial de V.
    """
    from . import SimulationResult  # import diferido para evitar ciclos

    u = np.array(u_init, dtype=float, copy=True)
    dt = params.dt
    n_steps = params.steps

    times: list[float] = []
    energies: list[float] = []
    history: list[np.ndarray] = []

    for step in range(n_steps + 1):
        if step % record_every == 0 or step == n_steps:
            V = activation(u, params.u0)
            times.append(step * dt)
            energies.append(energy(V, d, params))
            history.append(V.copy())
        if step == n_steps:
            break
        u = u + dt * du_dt(u, d, params)

    V_final = activation(u, params.u0)
    return SimulationResult(
        method="euler",
        times=np.asarray(times),
        energies=np.asarray(energies),
        V_history=np.asarray(history),
        u_final=u,
        V_final=V_final,
        distance_matrix=np.asarray(d, dtype=float),
        params=params,
    )
