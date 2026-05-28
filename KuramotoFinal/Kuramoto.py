"""
Kuramoto.py
===========

Script unico para las Tareas 1, 2, 3 y 4. Toda la diferencia entre
tareas vive en UNA sola variable: `MatrixOp`.

    MatrixOp = None        -> Tarea 1 (campo medio).
                              Integrador rapido O(N), malla de K lineal
                              en torno al Kc teorico. Barrido en sigma.

    MatrixOp = A (ndarray) -> Tareas 2, 3, 4 (red).
                              Integrador O(N^2), malla de K log,
                              sigma escalar fijo, opcional module_id
                              para descomponer r_m por modulos.

Como usarlo:

    1. Decides la TAREA tocando el bloque MATRIXOP_CONFIG mas abajo.
       Solo hay un MatrixOp activo y el resto comentados.

    2. Ajustas parametros (N, sigma, dt, num_K, num_runs, ...) en el
       bloque PARAMETROS DEL SISTEMA. Los defaults son razonables.

    3. python Kuramoto.py

Outputs (en resultados/<subdir>/<nombre_base>/):
    - params.txt          : copia legible de los parametros usados.
    - log.txt             : duplicado completo de la salida por consola.
    - condiciones_iniciales.npz : ICs (+ As + module_ids si red).
    - barrido.npz         : K_values, T_per_K, R_means, R_stds, ...
    - R_vs_K.png, sigmaR_vs_K.png, combinado.png
    - A.png               : visualizacion de la matriz (solo modo red).
"""

# Limitamos threads de BLAS/MKL/OMP ANTES de importar numpy/numba. Cada
# worker de joblib es un proceso, y si BLAS abre threads propios dentro
# de cada worker tendremos oversubscription: el rendimiento CAE. Con todo
# a 1, joblib es el unico que reparte trabajo y tenemos exactamente n_jobs
# threads efectivos.
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import time
import numpy as np

from kuramoto import (
    # Barrido y observables
    barrido_completo,
    Kc_teorica, Kc_experimental,
    # Grids de K y t_max
    K_values_tstudent,     t_max_per_K,
    K_values_log_tstudent, t_max_per_K_log,
    # Generadores
    generar_ICs, generar_ICs_por_sigma,
    generar_As_modular, crear_matriz_modular,
    # Conectoma
    cargar_y_preparar_A,
    # IO
    crear_carpeta_resultados, guardar_params_txt,
    iniciar_log, cerrar_log, _ruta,
    # Plots
    plot_R_vs_K, plot_sigmaR_vs_K, plot_combined,
    plot_matriz_adyacencia,
)


# =============================================================================
# PARAMETROS DEL SISTEMA
# =============================================================================

# --- Tamano y tiempo (validos para los dos modos) ---
N             = 3000          # numero de osciladores
dt            = 0.025         # paso de integracion
t_max_base    = 400.0         # t_max en las colas (K lejos de Kc)
t_max_peak    = 1500.0        # t_max en el pico (K ~ Kc)

# --- Barrido en K ---
num_K         = 300           # numero de valores de K
num_runs      = 1             # IC independientes por (sigma, K)
width_factor  = 0.3           # anchura de la t-Student en torno a Kc
                              # (compartido por K_values y t_max(K))

# --- Tarea 1 (campo medio): barrido en sigma ---
num_sigmas    = 3
sigma_min     = 0.5
sigma_max     = 1.5
K_min         = 0.25          # rango lineal de K para campo medio
K_max         = 4.0

# --- Tareas 2-4 (red): sigma escalar + rango LOG de K ---
sigma_red     = 1.0
K_min_red     = 5e-3
# Limite superior: bound de estabilidad del Euler para el grado tipico.
# Por defecto se reajusta para el caso modular en el bloque de abajo;
# si vas a usar Tarea 4 conectoma, conviene poner un K_max_red explicito.
K_max_red     = None          # None -> autocalcular segun la matriz.
K_center_red  = None          # None -> logspace uniforme. Si conoces ~Kc,
                              # ponlo aqui para concentrar puntos.

# --- Paralelismo ---
n_jobs_default = 16
n_jobs = int(os.environ.get('KURAMOTO_N_JOBS', n_jobs_default))

# --- Semilla maestra (None -> aleatoria cada vez) ---
SEED = None


# =============================================================================
# MATRIXOP_CONFIG: elige la TAREA descomentando UNO de los bloques
# =============================================================================

# ---- Tarea 1: campo medio ---------------------------------------------------
MatrixOp  = None
module_id = None
As        = None
module_ids = None
ETIQUETA_TAREA = 'Tarea1'


# ---- Tarea 2: red modular sintetica -----------------------------------------
# from numpy.random import default_rng
# rng_red    = default_rng(SEED)
# num_modules = 2
# p_intra     = 1.0
# n_aristas   = 1
# # Una matriz POR RUN (mismo perfil estructural, realizaciones distintas).
# As, module_ids = generar_As_modular(num_runs, N, num_modules, n_aristas,
#                                      p_intra, seed=SEED)
# MatrixOp  = As[0]                # representativa para info_box; el barrido
#                                  # usa As[r] por run.
# module_id = module_ids[0]
# ETIQUETA_TAREA = 'Tarea2'


# ---- Tarea 3: red jerarquica con A propia (la pasas tu) ---------------------
# A_propia        = np.load('estructuras/mi_matriz.npy')      # tu matriz
# module_id_propio = np.load('estructuras/mi_module_id.npy')  # opcional
# MatrixOp  = A_propia
# module_id = module_id_propio
# # En modo red sin generador, replicamos la misma A para todas las runs:
# As         = np.broadcast_to(MatrixOp[None, ...], (num_runs,)+MatrixOp.shape).copy()
# module_ids = np.broadcast_to(module_id[None, ...], (num_runs,)+module_id.shape).copy()
# ETIQUETA_TAREA = 'Tarea3'


# ---- Tarea 4: conectoma (lectura del .mat + threshold automatico) -----------
# A_conectoma, thr = cargar_y_preparar_A(
#     ruta_mat='data/SCmatrices88healthy.mat',
#     threshold='auto',          # o un float explicito.
# )
# MatrixOp  = A_conectoma
# module_id = None               # Sin descomposicion en modulos (decision
#                                # consciente: la Tarea 4 solo reporta R global).
# As         = np.broadcast_to(MatrixOp[None, ...], (num_runs,)+MatrixOp.shape).copy()
# module_ids = None
# # Reajusta N al tamano del conectoma (override del default de arriba).
# N = MatrixOp.shape[0]
# ETIQUETA_TAREA = 'Tarea4'


# =============================================================================
# DESPACHO INTERNO (NO TOCAR salvo que sepas lo que haces)
# =============================================================================

def _setup_campo_medio():
    """Construye K_values_per_sigma y T_per_sigma_K para Tarea 1.

    Un grid de K t-Student por cada sigma, centrado en Kc_teorica(sigma).
    """
    sigma_values = np.linspace(sigma_min, sigma_max, num_sigmas)
    K_values_per_sigma = np.zeros((num_sigmas, num_K))
    T_per_sigma_K      = np.zeros((num_sigmas, num_K))
    for i, sigma in enumerate(sigma_values):
        Kc = Kc_teorica(sigma)
        K_values_per_sigma[i] = K_values_tstudent(num_K, K_min, K_max, Kc,
                                                   width_factor)
        T_per_sigma_K[i]      = t_max_per_K(K_values_per_sigma[i], Kc,
                                             t_max_base, t_max_peak,
                                             width_factor)
    return sigma_values, K_values_per_sigma, T_per_sigma_K


def _setup_red(A_ref):
    """Construye K_values_per_sigma y T_per_sigma_K para modo red.

    Sigma escalar -> n_sigmas = 1. K-grid log-tstudent.
    """
    # Limite por estabilidad: dt * K * lambda_max(L) < 2. Como cota
    # conservadora usamos lambda_max(L) ~ grado_max = max degree de A.
    if K_max_red is None:
        grado_max = float(A_ref.sum(axis=1).max())
        K_max_eff = 0.95 * (2.0 / (dt * max(grado_max, 1.0)))
    else:
        K_max_eff = float(K_max_red)

    sigma_values       = np.array([sigma_red])
    K_values_per_sigma = K_values_log_tstudent(num_K, K_min_red, K_max_eff,
                                                K_center_red, width_factor)
    K_values_per_sigma = K_values_per_sigma[None, :]    # (1, num_K)
    T_per_sigma_K      = t_max_per_K_log(K_values_per_sigma[0], K_center_red,
                                          t_max_base, t_max_peak, width_factor)
    T_per_sigma_K      = T_per_sigma_K[None, :]
    return sigma_values, K_values_per_sigma, T_per_sigma_K, K_max_eff


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.perf_counter()

    es_campo_medio = (MatrixOp is None)

    # ----- Setup del barrido segun modo -----
    if es_campo_medio:
        sigma_values, K_vals_ps, T_ps_K = _setup_campo_medio()
        K_max_eff = K_max
    else:
        sigma_values, K_vals_ps, T_ps_K, K_max_eff = _setup_red(MatrixOp)

    n_sigmas = len(sigma_values)

    # ----- Carpeta de resultados + log -----
    t_max_label   = f"{int(t_max_base)}-{int(t_max_peak)}"
    if es_campo_medio:
        nombre_base = f"N{N}_sigmas{num_sigmas}_K{num_K}_Runs{num_runs}_t{t_max_label}"
    else:
        nombre_base = f"N{N}_K{num_K}_Runs{num_runs}_t{t_max_label}"
    run_dir = crear_carpeta_resultados(ETIQUETA_TAREA, nombre_base)
    log_file, stdout_orig, stderr_orig = iniciar_log(run_dir)

    try:
        print(f"Tarea:                {ETIQUETA_TAREA}")
        print(f"Modo:                 {'CAMPO MEDIO' if es_campo_medio else 'RED'}")
        print(f"Carpeta resultados:   {run_dir}")
        print(f"Log:                  {os.path.join(run_dir, 'log.txt')}\n")

        guardar_params_txt(run_dir, {
            'tarea'         : ETIQUETA_TAREA,
            'modo'          : 'campo_medio' if es_campo_medio else 'red',
            'N'             : N,
            'dt'            : dt,
            't_max_base'    : t_max_base,
            't_max_peak'    : t_max_peak,
            'num_K'         : num_K,
            'num_runs'      : num_runs,
            'width_factor'  : width_factor,
            'sigma_values'  : list(sigma_values),
            'K_min'         : K_min if es_campo_medio else K_min_red,
            'K_max'         : K_max_eff,
            'K_center_red'  : K_center_red,
            'n_jobs'        : n_jobs,
            'seed'          : SEED,
            'A_shape'       : None if MatrixOp is None else tuple(MatrixOp.shape),
            'module_id'     : None if module_id is None else 'set',
        })

        # ----- Generar ICs (pareadas, CRN) -----
        print("Generando condiciones iniciales pareadas...")
        if es_campo_medio:
            omegas_IC, thetas_IC = generar_ICs_por_sigma(n_sigmas, num_runs,
                                                          N, sigma_values,
                                                          seed=SEED)
        else:
            omegas_IC, thetas_IC = generar_ICs(num_runs, N, sigma_red, seed=SEED)
            # Adaptamos al formato (n_sigmas, num_runs, N): n_sigmas = 1.
            omegas_IC = omegas_IC[None, ...]
            thetas_IC = thetas_IC[None, ...]

        # ----- Plot diagnostico de la matriz A (solo modo red) -----
        if not es_campo_medio:
            plot_matriz_adyacencia(
                MatrixOp,
                module_id=module_id,
                save_path=_ruta(run_dir, 'A.png'),
                titulo=fr'Matriz de adyacencia ({ETIQUETA_TAREA})  $N={N}$',
            )

        # ----- Guardar ICs (y matrices, si modo red) en disco -----
        npz_ICs = {
            'omegas': omegas_IC,
            'thetas': thetas_IC,
            'sigma_values': np.asarray(sigma_values),
        }
        if As is not None:
            npz_ICs['As'] = As
        if module_ids is not None:
            npz_ICs['module_ids'] = module_ids
        np.savez_compressed(os.path.join(run_dir, 'condiciones_iniciales.npz'),
                            **npz_ICs)

        # ----- Resumen del coste -----
        total_ut = T_ps_K.sum() * num_runs
        print(f"\nBarrido: {n_sigmas} sigmas x {num_K} K x {num_runs} runs")
        print(f"Total simulado: {total_ut:.0f} u.t. ({total_ut*int(1/dt):.2e} pasos)\n")

        # ----- Lanzar el barrido -----
        out = barrido_completo(
            N=N, dt=dt,
            K_values_per_sigma=K_vals_ps,
            T_per_sigma_K=T_ps_K,
            sigma_values=sigma_values,
            num_runs=num_runs,
            omegas_IC=omegas_IC, thetas_IC=thetas_IC,
            As=As, module_ids=module_ids,
            n_jobs=n_jobs,
        )

        R_means      = out['R_means']
        R_stds       = out['R_stds']
        R_mean_stds  = out['R_mean_stds']
        rm_means     = out['rm_means']
        rm_stds      = out['rm_stds']
        rm_mean_stds = out['rm_mean_stds']

        # ----- Guardar barrido en disco -----
        np.savez_compressed(
            os.path.join(run_dir, 'barrido.npz'),
            K_values_per_sigma=K_vals_ps,
            T_per_sigma_K=T_ps_K,
            R_means=R_means, R_stds=R_stds, R_mean_stds=R_mean_stds,
            rm_means=(rm_means if rm_means is not None else np.zeros(0)),
            rm_stds =(rm_stds  if rm_stds  is not None else np.zeros(0)),
            rm_mean_stds=(rm_mean_stds if rm_mean_stds is not None else np.zeros(0)),
            sigma_values=np.asarray(sigma_values),
        )

        # ----- Plots -----
        log_x       = (not es_campo_medio)
        num_modules = rm_means.shape[-1] if rm_means is not None else 0

        plot_R_vs_K(K_vals_ps, sigma_values, R_means, R_mean_stds,
                    N, num_runs, run_dir,
                    log_x=log_x, mostrar_Kc_teorica=es_campo_medio,
                    rm_means=rm_means, rm_mean_stds=rm_mean_stds,
                    num_modules=num_modules)

        plot_sigmaR_vs_K(K_vals_ps, sigma_values, R_stds,
                         N, num_runs, run_dir,
                         log_x=log_x,
                         rm_stds=rm_stds, num_modules=num_modules)

        plot_combined(K_vals_ps, sigma_values, R_means, R_stds, R_mean_stds,
                      N, num_runs, run_dir,
                      log_x=log_x, mostrar_Kc_teorica=es_campo_medio,
                      rm_means=rm_means, rm_stds=rm_stds, rm_mean_stds=rm_mean_stds,
                      num_modules=num_modules)

        # ----- Tabla comparativa final -----
        print("\n" + "=" * 60)
        if es_campo_medio:
            print(f"{'sigma':>6} | {'Kc teorica':>11} | {'Kc experimental':>15}")
            print("-" * 60)
            for i, sigma in enumerate(sigma_values):
                Kc_th  = Kc_teorica(sigma)
                Kc_exp = Kc_experimental(K_vals_ps[i], R_stds[i], log=False)
                print(f"{sigma:>6.2f} | {Kc_th:>11.4f} | {Kc_exp:>15.4f}")
        else:
            Kc_exp = Kc_experimental(K_vals_ps[0], R_stds[0], log=True)
            print(f"Kc experimental (global, log fit): {Kc_exp:.4g}")
        print("=" * 60)

        elapsed = time.perf_counter() - t0
        print(f"\nTiempo total: {elapsed/60:.1f} min ({elapsed:.1f} s)")
        print(f"Resultados en: {run_dir}")

    finally:
        cerrar_log(log_file, stdout_orig, stderr_orig)


if __name__ == "__main__":
    main()
