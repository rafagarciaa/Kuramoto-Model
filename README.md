# KuramotoFinal

Simulador del modelo de Kuramoto con cuatro escenarios distintos (campo medio, red modular, red jerárquica y conectoma cerebral binarizado o pesado). Se basa en un único *driver* (`Kuramoto.py`) que despacha a la simulación elegida según un archivo de configuración (`params.json`). Todas las funciones internas viven en el paquete `kuramoto_scripts/`.

---

## 1. Instalación

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS / Git Bash:
source .venv/bin/activate

pip install -r ../requirements.txt
```

Probado con Python 3.12 y 3.13. Dependencias: `numpy`, `scipy`, `matplotlib`, `joblib`, `numba`.

---

## 2. Uso básico

1. Edita `params.json` (todo se controla desde ahí; **no se tocan parámetros en el `.py`**).
2. Elige `sim_type` (0–4).
3. Ejecuta:

```bash
cd KuramotoFinal
python Kuramoto.py
```

Los resultados se guardan en `resultados/<tipo>/<NombreAutomatico>/`, donde `<NombreAutomatico>` se construye como `N{N}_K{n_K}_runs{n_runs}_t{t_max}`.

---

## 3. Tipos de simulación (`sim_type`)

| `sim_type` | Escenario | Sección de `params.json` |
|---|---|---|
| 0 | **Campo medio**: barrido en `sigma` (anchura de la distribución de frecuencias). Soporta *finite-size scaling* para extrapolar `Kc` al límite termodinámico. | `tipo0_mean_field` |
| 1 | **Red modular**: `n_modules` bloques densos enlazados débilmente. | `tipo1_modular` |
| 2 | **Red jerárquica**: dos niveles (módulos y submódulos). | `tipo2_hierarchical` |
| 3 | **Conectoma binarizado** vs. red aleatoria que preserva grado. | `tipo3_connectome` |
| 4 | **Conectoma con pesos** vs. red aleatoria que preserva *strength* (4-cycle swap). | `tipo4_connectome_weighted` |

---

## 4. Parámetros (`params.json`)

### 4.1 Bloque `general` — común a todos los tipos

| Campo | Significado |
|---|---|
| `N` | Número de osciladores. Ignorado en `sim_type=3,4` (lo fija el conectoma, N=90). |
| `dt` | Paso de integración de Euler. |
| `n_runs` | Condiciones iniciales independientes por cada celda `(sigma, K)`. Se promedia sobre ellas. |
| `n_jobs` | Workers de `joblib`. `-1` = todos los cores. |
| `seed` | Semilla maestra (`null` = aleatoria en cada corrida). |

### 4.2 Bloque `convergence`

| Campo | Significado |
|---|---|
| `block_size` | Pasos por bloque. Cada bloque se promedia y se compara con el anterior. |
| `conv_threshold` | Si `|media_actual − media_anterior| < umbral`, la simulación para antes de `t_max`. |
| `t_max` | Tope de tiempo (en unidades naturales). `max_steps = t_max / dt`. |

### 4.3 Bloque `K_sweep`

| Campo | Significado |
|---|---|
| `n_K` | Nº de valores de K en el barrido. |
| `K_width_factor` | Anchura de la densidad t-Student que concentra puntos K cerca de Kc. |

Los valores de K se eligen con `K_values_tstudent` (tipo 0, lineal) o `K_values_log_tstudent` (tipos 1–4, log) para muestrear más finamente la región crítica.

### 4.4 `tipo0_mean_field`

| Campo | Significado |
|---|---|
| `n_sigmas`, `sigma_min`, `sigma_max` | Barrido lineal en `sigma`. |
| `K_min`, `K_max` | Rango de K. K se construye centrado en `Kc_teorica(sigma)`. |
| `scaling` | `true` → corre varios `N` para extrapolar `Kc(N) → Kc_inf` (finite-size scaling). |
| `scaling_fracs` | Lista de fracciones de `N` a simular cuando `scaling=true`. |
| `fss_method` | Método para extrapolar: `"linear_invN"` (α=1, robusto, sesgado), `"powerlaw_2_5"` (α=2/5, teórico de Hong et al. 2007), `"powerlaw_free"` (α ajustado). El plot muestra siempre los tres; este campo elige cuál se guarda como “oficial”. |

### 4.5 `tipo1_modular`

| Campo | Significado |
|---|---|
| `n_modules` | Nº de bloques. |
| `p_intra` | Probabilidad de arista intra-módulo. |
| `n_edges_inter` | Nº de aristas entre cada par de módulos. |
| `sigma` | Anchura fija de la distribución de frecuencias. |
| `K_min`, `K_max`, `K_center` | Rango log-uniforme. `K_max=null` → cota por estabilidad lineal de Euler. `K_center=null` → log uniforme. |

### 4.6 `tipo2_hierarchical`

| Campo | Significado |
|---|---|
| `submodules_per_module` | Lista: cada entrada es el nº de submódulos dentro de ese módulo. Su longitud = nº de módulos. |
| `p_intra_submodule` | Densidad dentro de cada submódulo. |
| `n_edges_inter_submodule` | Aristas entre submódulos del mismo módulo. |
| `n_edges_inter_module` | Aristas entre módulos. |
| `sigma`, `K_min`, `K_max`, `K_center` | Igual que `tipo1`. |

### 4.7 `tipo3_connectome`

| Campo | Significado |
|---|---|
| `mat_file` | `.mat` dentro de `data/` (por defecto `SCmatrices88healthy.mat`, 88 sujetos sanos). |
| `threshold` | Umbral para binarizar. `"auto"` = el más alto que mantiene el grafo conexo; o un float. |
| `n_swaps_factor` | Nº de double-edge swaps = `n_swaps_factor × |aristas|` para construir la red aleatoria que **preserva grado**. |
| `sigma`, `K_min`, `K_max`, `K_center` | Como antes. |

### 4.8 `tipo4_connectome_weighted`

Conectoma **sin binarizar**: K solo multiplica los pesos `W`.

| Campo | Significado |
|---|---|
| `mat_file` | Mismo `.mat`. |
| `approximation` | `"matriz_real"` (W = media de los 88 sujetos, valores brutos) o `"intervalos"` (W normalizada y discretizada en `n_levels` valores). |
| `n_levels` | Solo en `"intervalos"`: nº de niveles en `[0,1]`. |
| `log_transform` | Solo en `"intervalos"`: aplica `log(1+W)` antes de normalizar (recomendado: aplana la distribución sesgada del conectoma). |
| `n_swaps_factor` | 4-cycle swaps = `n_swaps_factor × |aristas no nulas|`. Cada swap **preserva la strength exacta** de los 4 nodos. |
| `sigma`, `K_min`, `K_max`, `K_center` | Como antes. |

---

## 5. Salidas

Cada corrida produce en `resultados/<tipo>/<nombre>/`:

| Archivo | Contenido |
|---|---|
| `params.txt` | Copia plana de los parámetros usados (incluye `max_steps` derivado). |
| `log.txt` | Stdout/stderr completos de la corrida. |
| `barrido.npz` | Datos numéricos del barrido. Claves: `K_grid`, `R_mean`, `R_sigma`, `R_err`, `n_steps`, `lvl{l}_rm_*` (por cada nivel jerárquico), `sigmas`, y los extras propios del tipo (`N_values`, `Kc_per_N`, `Kc_inf`, `fits_*`, `module_id`, `submodule_id`, `rand_R_*`, `threshold`, `K_max`, `strength_max`, …). |
| Figuras `.png` | Distintas según el tipo (ver abajo). |
| `A.png` / `W_*.png` | Matriz(es) de adyacencia/pesos usadas. |

### Observables principales

- **`R_mean(sigma, K)`** — parámetro de orden global de Kuramoto promediado sobre las `n_runs` réplicas.
- **`R_sigma(sigma, K)`** — desviación temporal de R: proxy directo de **metaestabilidad**.
- **`R_err`** — error estándar entre réplicas.
- **`lvlX_rm_*`** — los mismos observables medidos dentro de cada comunidad de nivel X (módulo, submódulo, hemisferio…).

### Figuras por tipo

- **Tipo 0**: `plot_mean_field` (R y σ_R suavizados con spline PCHIP, una curva por sigma). Si `scaling=true`, además `plot_scaling_Kc` con dos paneles (`Kc` vs `1/N` y vs `N^(−2/5)`) y los tres ajustes superpuestos.
- **Tipo 1**: `plot_modular` (R global + R medio por módulo).
- **Tipo 2**: `plot_hierarchical` (R global, R por módulo y R por submódulo).
- **Tipo 3** y **Tipo 4**: `plot_connectome` superpone curvas para conectoma y red aleatoria; permite ver si el cableado real eleva σ_R (más metaestabilidad) respecto al control con mismo grado/strength.

### Cómo se calcula `Kc_experimental`

Se localiza el K donde σ_R es máximo (pico de fluctuaciones del parámetro de orden ⇒ transición). En el tipo 0 con `scaling=true` se ajusta `Kc(N) = Kc_inf + a · N^(−α)` con tres convenciones de α y se reporta una tabla comparativa de `Kc_inf`, error relativo a `Kc_teorica(sigma)` y el `α` ajustado libremente.

---

## 6. Datos requeridos

En `data/`:

- `SCmatrices88healthy.mat` — matrices estructurales (88 sujetos, parcelación AAL90).
- `AAL_regions.csv` — etiquetas de las 90 regiones (usado para asignar hemisferio izquierdo/derecho como nivel jerárquico en los tipos 3 y 4).

---

## 7. Flujo interno (resumen)

`Kuramoto.py` → `load_params()` → `DISPATCH[sim_type]` → prepara A/W e ICs → `barrido(...)` (núcleo paralelo en `sweep.py`, con kernels Numba en `integrators.py` y observables en `observables.py`) → guarda `.npz` → genera figuras (`plotting.py`).

La parada es **adaptativa**: cada `block_size` pasos compara la media de R con el bloque previo; si la diferencia cae por debajo de `conv_threshold`, corta antes de `max_steps`. Esto ahorra tiempo lejos de Kc, donde R converge rápido.
