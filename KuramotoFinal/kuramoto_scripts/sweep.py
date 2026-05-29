"""
kuramoto_scripts.sweep
======================

Motor de barrido en K (y en sigma para campo medio), paralelizado con joblib.

Una sola funcion `barrido` cubre los 4 sim_type:
    - Campo medio (A_runs=None): barrido en sigma; K_grid (n_sigmas, n_K).
    - Red (A_runs por run): sigma escalar -> n_sigmas=1; niveles opcionales.

Common Random Numbers (CRN):
    Para cada run r, las ICs (omega_r, theta0_r) y la matriz A_r se generan
    UNA vez fuera y se reutilizan en todos los K. Reduce varianza entre
    celdas y suaviza las curvas.

Devuelve un dict con (nombres nuevos):
    R_mean   (n_sigmas, n_K)   : ⟨R⟩ medio sobre runs.
    R_sigma  (n_sigmas, n_K)   : metaestabilidad (sigma_R temporal medio).
    R_err    (n_sigmas, n_K)   : dispersion run-a-run de ⟨R⟩ (banda de error).
    n_steps  (n_sigmas, n_K)   : pasos medios usados (diagnostico de la parada).
    levels   : lista (una entrada por nivel) de dicts con
               {rm_mean, rm_sigma, rm_err} de shape (n_sigmas, n_K, n_grupos).
"""

import time
import numpy as np
from joblib import Parallel, delayed

from kuramoto_scripts.system import run_simulation


# ----------------------------------------------------------------------------
# Tarea elemental
# ----------------------------------------------------------------------------

def _run_one(i, j, r, N, K, sigma, dt, max_steps, block_size, conv_threshold,
             omega_ic, theta0_ic, A, level_ids):
    """Una simulacion con ICs/A precomputadas.

    Devuelve (i, j, r, mean_R, sigma_R, n_steps_used, mean_rm, sigma_rm)
    donde mean_rm/sigma_rm son listas (una por nivel) o None en campo medio.
    """
    sys = run_simulation(N, K, sigma, dt, max_steps, block_size, conv_threshold,
                         A=A, level_ids=level_ids,
                         omega=omega_ic, theta_0=theta0_ic)
    if A is None:
        return (i, j, r, sys.mean_R, sys.sigma_R, sys.n_steps_used, None, None)
    return (i, j, r, sys.mean_R, sys.sigma_R, sys.n_steps_used,
            sys.mean_rm, sys.sigma_rm)


# ----------------------------------------------------------------------------
# Barrido
# ----------------------------------------------------------------------------

def barrido(N, dt, max_steps, block_size, conv_threshold,
            K_grid, sigmas, n_runs,
            omegas_IC, thetas_IC,
            A_runs=None, level_ids=None,
            n_jobs=-1, verbose_joblib=10):
    n_sigmas, n_K = K_grid.shape

    # Numero de grupos por nivel (a partir de las particiones estructurales).
    if level_ids is not None:
        n_groups = [int(np.asarray(lid).max()) + 1 for lid in level_ids]
        n_levels = len(level_ids)
    else:
        n_groups, n_levels = [], 0

    print(f"Barrido: {n_sigmas} sigmas x {n_K} K x {n_runs} runs = "
          f"{n_sigmas * n_K * n_runs} simulaciones")
    print(f"  Niveles: {n_levels}  (grupos por nivel: {n_groups})")
    print(f"  Parada: block_size={block_size}, conv_threshold={conv_threshold}, "
          f"max_steps={max_steps}")
    print(f"  Workers (n_jobs): {n_jobs}\n")

    # --- Construir tareas ---
    tareas = []
    for i in range(n_sigmas):
        sigma = float(sigmas[i])
        for j in range(n_K):
            K = float(K_grid[i, j])
            for r in range(n_runs):
                A_r = A_runs[r] if A_runs is not None else None
                tareas.append((i, j, r, N, K, sigma, dt,
                               max_steps, block_size, conv_threshold,
                               omegas_IC[i, r], thetas_IC[i, r], A_r, level_ids))

    print(f"Lanzando {len(tareas)} simulaciones en paralelo...")
    t0 = time.perf_counter()
    resultados = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
        delayed(_run_one)(*t) for t in tareas
    )
    print(f"Tiempo del barrido: {(time.perf_counter()-t0)/60:.1f} min\n")

    # --- Reagregacion ---
    R_mean  = np.zeros((n_sigmas, n_K))
    R_sigma = np.zeros((n_sigmas, n_K))
    R_err   = np.zeros((n_sigmas, n_K))
    n_steps = np.zeros((n_sigmas, n_K))

    meanR_cell  = {}
    sigmaR_cell = {}
    steps_cell  = {}
    rm_mean_cell  = {}   # (i,j) -> list[level] de listas (una por run) de arrays
    rm_sigma_cell = {}

    for (i, j, r, mean_R, sigma_R, nsteps, mean_rm, sigma_rm) in resultados:
        meanR_cell.setdefault((i, j), []).append(mean_R)
        sigmaR_cell.setdefault((i, j), []).append(sigma_R)
        steps_cell.setdefault((i, j), []).append(nsteps)
        if mean_rm is not None:
            rm_mean_cell.setdefault((i, j), [[] for _ in range(n_levels)])
            rm_sigma_cell.setdefault((i, j), [[] for _ in range(n_levels)])
            for l in range(n_levels):
                rm_mean_cell[(i, j)][l].append(mean_rm[l])
                rm_sigma_cell[(i, j)][l].append(sigma_rm[l])

    for (i, j), vals in meanR_cell.items():
        R_mean[i, j]  = np.mean(vals)
        R_sigma[i, j] = np.mean(sigmaR_cell[(i, j)])
        R_err[i, j]   = np.std(vals)
        n_steps[i, j] = np.mean(steps_cell[(i, j)])

    # Niveles.
    levels = []
    for l in range(n_levels):
        ng = n_groups[l]
        rm_mean  = np.zeros((n_sigmas, n_K, ng))
        rm_sigma = np.zeros((n_sigmas, n_K, ng))
        rm_err   = np.zeros((n_sigmas, n_K, ng))
        for (i, j) in rm_mean_cell:
            arr_mean  = np.asarray(rm_mean_cell[(i, j)][l])   # (n_runs, ng)
            arr_sigma = np.asarray(rm_sigma_cell[(i, j)][l])
            rm_mean[i, j]  = np.mean(arr_mean, axis=0)
            rm_sigma[i, j] = np.mean(arr_sigma, axis=0)
            rm_err[i, j]   = np.std(arr_mean, axis=0)
        levels.append({'rm_mean': rm_mean, 'rm_sigma': rm_sigma, 'rm_err': rm_err})

    return {
        'R_mean': R_mean, 'R_sigma': R_sigma, 'R_err': R_err,
        'n_steps': n_steps, 'levels': levels,
    }
