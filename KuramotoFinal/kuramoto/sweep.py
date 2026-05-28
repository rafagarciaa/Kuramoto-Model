"""
kuramoto.sweep
==============

Motor de barrido en K. Una funcion unica `barrido_completo` que cubre
los dos modos:

    Modo campo medio (A is None):
        Acepta un VECTOR de sigmas y un grid de K POR sigma. Devuelve
        R_means, R_stds, R_mean_stds en formato (n_sigmas, num_K).

    Modo red (A ndarray, una matriz por run):
        Acepta sigma ESCALAR (la red es la fuente de heterogeneidad) y
        un unico grid de K. Devuelve R_means, R_stds, ... en formato
        (num_K,) y opcionalmente rm_means en (num_K, num_modules).

Common Random Numbers (CRN):
    Para cada run r, las IC (omega_r, theta0_r) y la red A_r se generan
    UNA sola vez fuera y se reutilizan en todos los K del barrido. Esto
    reduce drasticamente la varianza cruzada entre celdas y suaviza la
    curva R(K).
"""

import time
import numpy as np
from joblib import Parallel, delayed

from kuramoto.system import Simulacion_Kuramoto


# ----------------------------------------------------------------------------
# Tarea elemental: una simulacion, devuelve el indexado para reagrupar
# ----------------------------------------------------------------------------

def _una_sim_indexada(i, j, r, N, K, sigma, dt, t_max,
                      omega_ic, theta0_ic, A, module_id):
    """Ejecuta una simulacion con ICs/A precomputadas.

    Devuelve una tupla con el indexado (i, j, r) para que el llamante
    sepa donde meter el resultado, mas los observables del estacionario.

    En campo medio A es None y r_m_mean/r_m_std vienen vacios.
    """
    sys = Simulacion_Kuramoto(N, K, sigma, dt, t_max,
                              A=A, module_id=module_id,
                              omega=omega_ic, theta_0=theta0_ic)
    if A is None:
        return (i, j, r, sys.R_mean, sys.R_std, None, None)
    return (i, j, r, sys.R_mean, sys.R_std, sys.r_m_mean, sys.r_m_std)


# ----------------------------------------------------------------------------
# Barrido unificado
# ----------------------------------------------------------------------------

def barrido_completo(
    N, dt,
    K_values_per_sigma, T_per_sigma_K,
    sigma_values,
    num_runs,
    omegas_IC, thetas_IC,
    As=None, module_ids=None,
    n_jobs=-1, verbose_joblib=10,
):
    """Lanza num_sigmas x num_K x num_runs simulaciones en paralelo.

    Forma de los inputs:

        K_values_per_sigma : ndarray (n_sigmas, num_K)
            Grid de K para cada sigma. En modo red sera (1, num_K).
        T_per_sigma_K      : ndarray (n_sigmas, num_K)
            t_max(K) por celda. Tiempo adaptativo: mas largo cerca de Kc.
        sigma_values       : array shape (n_sigmas,)
            En modo red basta con [sigma_unico].
        num_runs           : int
            Numero de IC independientes por (sigma, K).
        omegas_IC          : ndarray (n_sigmas, num_runs, N)
        thetas_IC          : ndarray (n_sigmas, num_runs, N)
        As                 : None  -> campo medio
                           : (num_runs, N, N) -> red, una matriz por run.
        module_ids         : None  -> sin info de modulos
                           : (num_runs, N) -> r_m por modulo.

    Devuelve un dict con:
        R_means      : (n_sigmas, num_K)
        R_stds       : (n_sigmas, num_K)   <- media de sigma_R por run
        R_mean_stds  : (n_sigmas, num_K)   <- variabilidad de <R> entre runs
        rm_means     : (n_sigmas, num_K, num_modules)  o None
        rm_stds      : (n_sigmas, num_K, num_modules)  o None
        rm_mean_stds : (n_sigmas, num_K, num_modules)  o None
    """
    n_sigmas, num_K = K_values_per_sigma.shape

    # Modo red: detectamos el numero de modulos a partir del primer module_id.
    if As is not None and module_ids is not None:
        num_modules = int(module_ids[0].max()) + 1
    elif As is not None:
        # Red sin info de modulos: un unico "modulo" trivial.
        num_modules = 1
    else:
        num_modules = 0  # campo medio: no hay r_m.

    # ----- Resumen del coste estimado ------------------------------------
    total_ut = T_per_sigma_K.sum() * num_runs
    print(f"Barrido: {n_sigmas} sigmas x {num_K} K x {num_runs} runs = "
          f"{n_sigmas * num_K * num_runs} simulaciones")
    print(f"  Total simulado: {total_ut:.0f} u.t. "
          f"({total_ut*int(1/dt):.2e} pasos aprox.)")
    print(f"  Workers (n_jobs): {n_jobs}\n")

    # ----- Construir la lista de tareas ----------------------------------
    tareas = []
    for i, sigma in enumerate(sigma_values):
        for j in range(num_K):
            K       = float(K_values_per_sigma[i, j])
            t_max_j = float(T_per_sigma_K[i, j])
            for r in range(num_runs):
                A_r   = As[r]         if As         is not None else None
                mid_r = module_ids[r] if module_ids is not None else None
                tareas.append((i, j, r, N, K, sigma, dt, t_max_j,
                               omegas_IC[i, r], thetas_IC[i, r],
                               A_r, mid_r))

    # Balanceo: las tareas con mas pasos arrancan primero para que las
    # cortas rellenen los huecos al final. Indice 7 = t_max_j.
    tareas.sort(key=lambda tarea: -tarea[7])

    print(f"Lanzando {len(tareas)} simulaciones en paralelo...")
    t_inicio = time.perf_counter()
    resultados = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
        delayed(_una_sim_indexada)(*tarea) for tarea in tareas
    )
    print(f"Tiempo total del barrido: {(time.perf_counter()-t_inicio)/60:.1f} min\n")

    # ----- Reagregacion en arrays ----------------------------------------
    R_means     = np.zeros((n_sigmas, num_K))
    R_stds      = np.zeros((n_sigmas, num_K))
    R_mean_stds = np.zeros((n_sigmas, num_K))

    means_por_celda = {}
    stds_por_celda  = {}
    rm_means_por_celda = {}
    rm_stds_por_celda  = {}

    for (i, j, r, r_mean, r_std, rm_mean, rm_std) in resultados:
        means_por_celda.setdefault((i, j), []).append(r_mean)
        stds_por_celda .setdefault((i, j), []).append(r_std)
        if rm_mean is not None:
            rm_means_por_celda.setdefault((i, j), []).append(rm_mean)
            rm_stds_por_celda .setdefault((i, j), []).append(rm_std)

    for (i, j), means in means_por_celda.items():
        R_means[i, j]     = np.mean(means)
        R_stds[i, j]      = np.mean(stds_por_celda[(i, j)])
        R_mean_stds[i, j] = np.std(means)

    if num_modules >= 1 and As is not None:
        rm_means     = np.zeros((n_sigmas, num_K, num_modules))
        rm_stds      = np.zeros((n_sigmas, num_K, num_modules))
        rm_mean_stds = np.zeros((n_sigmas, num_K, num_modules))
        for (i, j), lst in rm_means_por_celda.items():
            arr_means = np.asarray(lst)             # (num_runs, num_modules)
            arr_stds  = np.asarray(rm_stds_por_celda[(i, j)])
            rm_means[i, j]     = np.mean(arr_means, axis=0)
            rm_stds[i, j]      = np.mean(arr_stds,  axis=0)
            rm_mean_stds[i, j] = np.std(arr_means,  axis=0)
    else:
        rm_means = rm_stds = rm_mean_stds = None

    return {
        'R_means'     : R_means,
        'R_stds'      : R_stds,
        'R_mean_stds' : R_mean_stds,
        'rm_means'    : rm_means,
        'rm_stds'     : rm_stds,
        'rm_mean_stds': rm_mean_stds,
    }
