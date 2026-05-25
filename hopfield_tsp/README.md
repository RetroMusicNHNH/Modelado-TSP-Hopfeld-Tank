# Modelo de Hopfield-Tank para el Problema del Viajero (TSP)

Implementación en Python del modelo continuo de Hopfield-Tank (1985) aplicado
a dos instancias pequeñas del TSP: un cuadrilátero de 4 ciudades y un pentágono
de 5 ciudades. El proyecto acompaña al informe del curso **MO-0006 Ecuaciones
Diferenciales Ordinarias con Modelos** (Universidad de Costa Rica).

## Idea del modelo

El problema combinatorio se codifica en una **función de energía** cuyos mínimos
corresponden a rutas válidas y cortas. Un sistema de `N²` EDOs acopladas hace
decrecer esa energía hasta un estado de equilibrio que se interpreta como la
ruta solución. La energía actúa como **función de Lyapunov** del sistema.

- Variables `V_xi ∈ [0,1]`: certeza de que la ciudad `x` ocupa la posición `i`.
- Sigmoide: `V = ½(1 + tanh(u/u₀))`.
- Energía con cuatro términos (factibilidad por filas `A`, por columnas `B`,
  restricción global `C` y distancia `D`).
- Parámetros por defecto: `A = B = 500`, `C = 200`, `D = 500`, `τ = 1`, `u₀ = 0.02`.

### Valores de referencia (fuerza bruta)

| Instancia   | Ruta óptima   | Longitud |
|-------------|---------------|----------|
| 4 ciudades  | A-B-C-D-A     | ≈ 6.56   |
| 5 ciudades  | A-B-C-D-E-A   | ≈ 11.72  |

## Estructura

```
hopfield_tsp/
├── src/
│   ├── instances/      # coordenadas y matriz de distancias
│   ├── model/          # parámetros, energía y dinámica (EDOs)
│   ├── solvers/        # Euler y Runge-Kutta 4
│   ├── analysis/       # convergencia, extracción de ruta, validación
│   ├── brute_force/    # ruta óptima exacta (referencia)
│   └── visualization/  # energía, matriz V y rutas
├── experiments/        # scripts por semana
├── tests/              # pruebas con pytest
├── notebooks/          # exploración interactiva
└── results/            # figuras y datos generados
```

## Instalación

```bash
cd hopfield_tsp
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

## Uso

Ejecutar los experimentos desde la carpeta `hopfield_tsp/`:

```bash
python experiments/week2_euler_4cities.py
python experiments/week2_euler_5cities.py
python experiments/week3_rk4_4cities.py
python experiments/week3_rk4_5cities.py
python experiments/week4_comparison.py
```

Cada script imprime un resumen (energía final, validez de la solución, ruta
obtenida y brecha respecto al óptimo) y guarda figuras en `results/figures/` y
datos en `results/data/`.

### Ejemplo mínimo

```python
import numpy as np
from src.instances import FOUR_CITIES, distance_matrix_for
from src.model import HopfieldParameters, random_initial_potentials
from src.solvers import rk4_integrate
from src.analysis import extract_route, route_as_cycle_string, route_length

d = distance_matrix_for(FOUR_CITIES)
params = HopfieldParameters(dt=1e-5, steps=30_000)
u0 = random_initial_potentials(FOUR_CITIES.n, params.u0,
                               rng=np.random.default_rng(7))
res = rk4_integrate(u0, d, params)
route = extract_route(res.V_final)
print(route_as_cycle_string(route, FOUR_CITIES), route_length(route, d))
```

## Pruebas

```bash
cd hopfield_tsp
pytest
```

## Referencias

- Hopfield, J. J., & Tank, D. W. (1985). "Neural" computation of decisions in
  optimization problems. *Biological Cybernetics*.
- Talaván, P. M., & Yáñez, J. (2002). Parameter setting of the Hopfield network
  applied to TSP. *Neural Networks*.
