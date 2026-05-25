"""Método de Runge-Kutta de cuarto orden (RK4) para Hopfield-Tank.

RK4 promedia cuatro evaluaciones de la pendiente en cada paso:

    k1 = f(u)
    k2 = f(u + dt/2 * k1)
    k3 = f(u + dt/2 * k2)
    k4 = f(u + dt   * k3)
    u_{k+1} = u_k + dt/6 (k1 + 2 k2 + 2 k3 + k4).

Es de cuarto orden, por lo que reproduce la trayectoria continua con mucha
mayor fidelidad que Euler para el mismo tamaño de paso.
"""

from __future__ import annotations

import numpy as np

from ..model.dynamics import activation, du_dt
from ..model.energy import energy
from ..model.parameters import HopfieldParameters


def rk4_integrate(
    u_init: np.ndarray,
    d: np.ndarray,
    params: HopfieldParameters,
    record_every: int = 100,
):
    """Integra el sistema con RK4 y registra la trayectoria.

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

    def f(state: np.ndarray) -> np.ndarray:
        return du_dt(state, d, params)

    for step in range(n_steps + 1):
        if step % record_every == 0 or step == n_steps:
            V = activation(u, params.u0)
            times.append(step * dt)
            energies.append(energy(V, d, params))
            history.append(V.copy())
        if step == n_steps:
            break
        k1 = f(u)
        k2 = f(u + 0.5 * dt * k1)
        k3 = f(u + 0.5 * dt * k2)
        k4 = f(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    V_final = activation(u, params.u0)
    return SimulationResult(
        method="rk4",
        times=np.asarray(times),
        energies=np.asarray(energies),
        V_history=np.asarray(history),
        u_final=u,
        V_final=V_final,
        distance_matrix=np.asarray(d, dtype=float),
        params=params,
    )
