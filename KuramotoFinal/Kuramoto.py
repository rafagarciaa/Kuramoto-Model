"""
Kuramoto.py
===========

Driver unico para los 4 tipos de simulacion. NO se tocan parametros aqui:
todo se edita en params.json. El tipo se elige con "sim_type" (0..3).

    python Kuramoto.py

Salidas en resultados/tipo<N>/<nombre>/ :
    params.txt, log.txt, barrido.npz, las figuras y (en red) A.png.
"""

# Limitar threads de BLAS/OMP ANTES de importar numpy/numba (evita
# oversubscription con los workers de joblib).
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import time
import numpy as np
from scipy.optimize import curve_fit

from kuramoto_scripts import (
    load_params,
    barrido, Kc_teorica, Kc_experimental,
    K_values_tstudent, K_values_log_tstudent,
    generar_ICs, generar_ICs_por_sigma,
    generar_As_modular, generar_As_jerarquica,
    cargar_y_preparar_A, hemisferio_ids, randomize_preserving_degree,
    prepare_W_real, prepare_W_intervalos, randomize_preserving_strength,
    crear_carpeta_resultados, guardar_params_txt,
    iniciar_log, cerrar_log, _ruta,
    plot_mean_field, plot_modular, plot_hierarchical, plot_connectome,
    plot_scaling_Kc, plot_matriz_adyacencia,
)

# Exponente teorico FSS para Kuramoto campo medio con g(omega) Gaussiana
# (Hong, Chate, Park PRL 2007). El metodo "linear_invN" asume alpha=1
# (lineal en 1/N) y es robusto al ruido pero sesgado. El metodo
# "powerlaw_2_5" usa el alpha correcto pero amplifica el ruido cuando
# los Kc(N) tienen dispersion comparable al shift residual.
ALPHA_PL = 2.0 / 5.0


def _fit_powerlaw(N_arr, Kc_arr, alpha=ALPHA_PL):
    """Ajusta Kc(N) = Kc_inf + a * N^(-alpha) con alpha FIJO.

    Devuelve (Kc_inf, a) o (NaN, NaN) si el fit falla."""
    def model(N, kc_inf, a):
        return kc_inf + a * np.power(N, -alpha)
    p0 = [float(Kc_arr[-1]), float(Kc_arr[0] - Kc_arr[-1])]
    try:
        popt, _ = curve_fit(model, N_arr, Kc_arr, p0=p0, maxfev=5000)
        return popt
    except Exception:
        return np.array([np.nan, np.nan])


def _fit_powerlaw_free(N_arr, Kc_arr):
    """Ajusta Kc(N) = Kc_inf + a * N^(-alpha) con alpha LIBRE.

    Devuelve (Kc_inf, a, alpha) o (NaN, NaN, NaN). Requiere >=4 puntos N
    para que el fit tenga grados de libertad. alpha esta acotado a
    [0.05, 2.0] para evitar fugar a extremos sin sentido fisico.
    """
    if len(N_arr) < 4:
        return np.array([np.nan, np.nan, np.nan])
    def model(N, kc_inf, a, alpha):
        return kc_inf + a * np.power(N, -alpha)
    p0 = [float(Kc_arr[-1]), float(Kc_arr[0] - Kc_arr[-1]), ALPHA_PL]
    try:
        popt, _ = curve_fit(model, N_arr, Kc_arr, p0=p0,
                            bounds=([0.0, -np.inf, 0.05],
                                    [np.inf, np.inf, 2.0]),
                            maxfev=10000)
        return popt
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def _fmt(v, spec='.4f'):
    """Formatea un float, devolviendo 'n/a' si no es finito."""
    return f"{v:{spec}}" if np.isfinite(v) else "n/a"

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AQUI, 'data')


def _K_max_estable(A_ref, dt, factor=0.95):
    """Cota superior de K por estabilidad lineal de Euler: dt*K*grado_max < 2."""
    grado_max = float(np.asarray(A_ref).sum(axis=1).max())
    return factor * (2.0 / (dt * max(grado_max, 1.0)))


def _guardar(run_dir, K_grid, out, extra=None):
    datos = {'K_grid': K_grid,
             'R_mean': out['R_mean'], 'R_sigma': out['R_sigma'],
             'R_err': out['R_err'], 'n_steps': out['n_steps']}
    for l, lvl in enumerate(out['levels']):
        datos[f'lvl{l}_rm_mean']  = lvl['rm_mean']
        datos[f'lvl{l}_rm_sigma'] = lvl['rm_sigma']
        datos[f'lvl{l}_rm_err']   = lvl['rm_err']
    if extra:
        datos.update(extra)
    np.savez_compressed(os.path.join(run_dir, 'barrido.npz'), **datos)


# ============================================================================
# Tipo 0: campo medio
# ============================================================================

def _barrido_campo_medio(p, N, sigmas, K_grid):
    """Un barrido de campo medio para un N dado. Devuelve el dict `out`."""
    g, c = p.general, p.convergence
    omegas_IC, thetas_IC = generar_ICs_por_sigma(len(sigmas), g.n_runs, N,
                                                 sigmas, seed=g.seed)
    return barrido(N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                   K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                   A_runs=None, level_ids=None, n_jobs=g.n_jobs)


def run_tipo0(p, run_dir):
    g, ks, s0 = p.general, p.K_sweep, p.tipo0_mean_field
    sigmas = np.linspace(s0.sigma_min, s0.sigma_max, s0.n_sigmas)

    # K es lineal en torno al Kc teorico de cada sigma. NO depende de N,
    # asi que la misma rejilla sirve para todos los tamaños del scaling.
    K_grid = np.zeros((s0.n_sigmas, ks.n_K))
    for i, sigma in enumerate(sigmas):
        K_grid[i] = K_values_tstudent(ks.n_K, s0.K_min, s0.K_max,
                                      Kc_teorica(sigma), ks.K_width_factor)

    scaling = getattr(s0, 'scaling', False)

    # --- Caso simple: un solo N ---
    if not scaling:
        out = _barrido_campo_medio(p, g.N, sigmas, K_grid)
        _guardar(run_dir, K_grid, out, extra={'sigmas': sigmas})
        plot_mean_field(K_grid, sigmas, out, g.N, g.n_runs, run_dir)
        return

    # --- Finite-size scaling: varias N, extrapolar Kc a 1/N = 0 ---
    fracs = getattr(s0, 'scaling_fracs', [0.2, 0.4, 0.6, 0.8, 1.0])
    N_values = [max(int(round(f * g.N)), 2) for f in fracs]
    print(f"Finite-size scaling. N = {N_values}\n")

    Kc_per_N = np.zeros((s0.n_sigmas, len(N_values)))
    out_full = None
    for k, Nk in enumerate(N_values):
        print(f"--- N = {Nk}  ({k+1}/{len(N_values)}) ---")
        outk = _barrido_campo_medio(p, Nk, sigmas, K_grid)
        for i in range(s0.n_sigmas):
            Kc_per_N[i, k] = Kc_experimental(K_grid[i], outk['R_sigma'][i], log=False)
        out_full = outk  # el ultimo (N completa) se usa para las graficas estandar

    # --- Calculamos los TRES ajustes FSS. ---
    # El usuario los compara visualmente en el plot (2 paneles fijos:
    # 1/N y N^(-2/5)) y numericamente en la tabla. fss_method en params.json
    # selecciona cual es "el oficial" (cual va a Kc_inf):
    #   "linear_invN"   -> alpha = 1     (lineal en 1/N, bajo varianza, sesgado)
    #   "powerlaw_2_5"  -> alpha = 2/5   (Hong 2007, teorico)
    #   "powerlaw_free" -> alpha libre   (ajustado a partir de los datos)
    inv_N = 1.0 / np.array(N_values, dtype=float)
    N_arr = np.array(N_values, dtype=float)

    fits_lin  = np.zeros((s0.n_sigmas, 2))          # (Kc_inf, slope_en_1/N)
    fits_pl   = np.zeros((s0.n_sigmas, 2))          # (Kc_inf, a) para alpha=2/5
    fits_free = np.zeros((s0.n_sigmas, 3))          # (Kc_inf, a, alpha) libre
    for i in range(s0.n_sigmas):
        slope, intercept = np.polyfit(inv_N, Kc_per_N[i], 1)
        fits_lin[i]  = (intercept, slope)           # Kc = intercept + slope/N
        fits_pl[i]   = _fit_powerlaw(N_arr, Kc_per_N[i], alpha=ALPHA_PL)
        fits_free[i] = _fit_powerlaw_free(N_arr, Kc_per_N[i])

    fss_method = getattr(s0, 'fss_method', 'linear_invN')
    Kc_inf_by_method = {
        'linear_invN':   fits_lin[:, 0],
        'powerlaw_2_5':  fits_pl[:, 0],
        'powerlaw_free': fits_free[:, 0],
    }
    if fss_method not in Kc_inf_by_method:
        print(f"AVISO: fss_method='{fss_method}' desconocido, uso 'linear_invN'.")
        fss_method = 'linear_invN'
    Kc_inf = Kc_inf_by_method[fss_method]

    # Estrella en la columna del metodo oficial.
    star = {'linear_invN':   ('*', ' ', ' '),
            'powerlaw_2_5':  (' ', '*', ' '),
            'powerlaw_free': (' ', ' ', '*')}[fss_method]

    print(f"\nFSS method oficial: '{fss_method}'  (marcado con * en la tabla)")
    print("=" * 108)
    print(f"{'sigma':>6} | {'Kc th':>7} | {'Kc(Nmax)':>9} | "
          f"{'Kc 1/N'+star[0]:>9} | {'err%':>5} | "
          f"{'Kc 2/5'+star[1]:>9} | {'err%':>5} | "
          f"{'Kc free'+star[2]:>10} | {'err%':>5} | "
          f"{'alpha':>6} | {'err(a)%':>8}")
    print("-" * 108)
    for i, sigma in enumerate(sigmas):
        Kc_th = Kc_teorica(sigma)
        e_lin  = 100.0 * abs(fits_lin[i, 0] - Kc_th) / Kc_th
        e_pl   = (100.0 * abs(fits_pl[i, 0]   - Kc_th) / Kc_th
                  if np.isfinite(fits_pl[i, 0])   else float('nan'))
        e_free = (100.0 * abs(fits_free[i, 0] - Kc_th) / Kc_th
                  if np.isfinite(fits_free[i, 0]) else float('nan'))
        alpha_fit = fits_free[i, 2]
        e_alpha   = (100.0 * abs(alpha_fit - ALPHA_PL) / ALPHA_PL
                     if np.isfinite(alpha_fit) else float('nan'))
        print(f"{sigma:>6.2f} | {Kc_th:>7.4f} | {Kc_per_N[i, -1]:>9.4f} | "
              f"{fits_lin[i, 0]:>9.4f} | {e_lin:>4.2f}% | "
              f"{_fmt(fits_pl[i, 0]):>9} | {_fmt(e_pl, '.2f'):>4}% | "
              f"{_fmt(fits_free[i, 0]):>10} | {_fmt(e_free, '.2f'):>4}% | "
              f"{_fmt(alpha_fit, '.3f'):>6} | {_fmt(e_alpha, '.2f'):>7}%")
    print("=" * 108)

    _guardar(run_dir, K_grid, out_full, extra={
        'sigmas': sigmas, 'N_values': N_arr, 'inv_N': inv_N,
        'Kc_per_N': Kc_per_N, 'Kc_inf': Kc_inf,
        'fss_method': fss_method,
        'fits_linear_invN':   fits_lin,
        'fits_powerlaw_2_5':  fits_pl,
        'fits_powerlaw_free': fits_free,
        'alpha_teorico': ALPHA_PL,
    })
    plot_mean_field(K_grid, sigmas, out_full, g.N, g.n_runs, run_dir)
    plot_scaling_Kc(N_arr, Kc_per_N, fits_lin, fits_pl, sigmas, run_dir,
                    fss_method=fss_method, fits_free=fits_free)


# ============================================================================
# Tipo 1: red modular
# ============================================================================

def run_tipo1(p, run_dir):
    g, c, ks, s1 = p.general, p.convergence, p.K_sweep, p.tipo1_modular

    As, module_id = generar_As_modular(g.n_runs, g.N, s1.n_modules,
                                       s1.n_edges_inter, s1.p_intra, seed=g.seed)
    level_ids = [module_id]

    K_max = s1.K_max if s1.K_max is not None else _K_max_estable(As[0], g.dt)
    K_grid = K_values_log_tstudent(ks.n_K, s1.K_min, K_max,
                                   s1.K_center, ks.K_width_factor)[None, :]
    sigmas = np.array([s1.sigma])

    omegas_IC, thetas_IC = generar_ICs(g.n_runs, g.N, s1.sigma, seed=g.seed)
    omegas_IC, thetas_IC = omegas_IC[None, ...], thetas_IC[None, ...]

    plot_matriz_adyacencia(As[0], module_id, _ruta(run_dir, 'A.png'),
                           titulo=fr'Red modular  $N={g.N}$, $M={s1.n_modules}$')

    out = barrido(g.N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                  K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                  A_runs=As, level_ids=level_ids, n_jobs=g.n_jobs)

    _guardar(run_dir, K_grid, out, extra={'module_id': module_id, 'K_max': K_max})
    plot_modular(K_grid, out, g.N, g.n_runs, s1.n_modules, run_dir)


# ============================================================================
# Tipo 2: red jerarquica
# ============================================================================

def run_tipo2(p, run_dir):
    g, c, ks, s2 = p.general, p.convergence, p.K_sweep, p.tipo2_hierarchical
    subs = list(s2.submodules_per_module)

    As, module_id, submodule_id = generar_As_jerarquica(
        g.n_runs, g.N, subs, s2.p_intra_submodule,
        s2.n_edges_inter_submodule, s2.n_edges_inter_module, seed=g.seed)
    level_ids = [module_id, submodule_id]

    K_max = s2.K_max if s2.K_max is not None else _K_max_estable(As[0], g.dt)
    K_grid = K_values_log_tstudent(ks.n_K, s2.K_min, K_max,
                                   s2.K_center, ks.K_width_factor)[None, :]
    sigmas = np.array([s2.sigma])

    omegas_IC, thetas_IC = generar_ICs(g.n_runs, g.N, s2.sigma, seed=g.seed)
    omegas_IC, thetas_IC = omegas_IC[None, ...], thetas_IC[None, ...]

    plot_matriz_adyacencia(As[0], submodule_id, _ruta(run_dir, 'A.png'),
                           titulo=fr'Red jerarquica  $N={g.N}$, submodulos={int(submodule_id.max())+1}')

    out = barrido(g.N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                  K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                  A_runs=As, level_ids=level_ids, n_jobs=g.n_jobs)

    _guardar(run_dir, K_grid, out,
             extra={'module_id': module_id, 'submodule_id': submodule_id, 'K_max': K_max})
    plot_hierarchical(K_grid, out, module_id, submodule_id, g.N, g.n_runs, run_dir)


# ============================================================================
# Tipo 3: conectoma vs aleatoria
# ============================================================================

def run_tipo3(p, run_dir):
    g, c, ks, s3 = p.general, p.convergence, p.K_sweep, p.tipo3_connectome

    ruta_mat = os.path.join(DATA, s3.mat_file)
    A_conn, thr = cargar_y_preparar_A(ruta_mat, threshold=s3.threshold)
    N = A_conn.shape[0]                      # el conectoma fija N (90)

    hemi = hemisferio_ids(os.path.join(DATA, 'AAL_regions.csv'))
    level_ids = [hemi]

    K_max = s3.K_max if s3.K_max is not None else _K_max_estable(A_conn, g.dt)
    K_grid = K_values_log_tstudent(ks.n_K, s3.K_min, K_max,
                                   s3.K_center, ks.K_width_factor)[None, :]
    sigmas = np.array([s3.sigma])

    # ICs compartidas entre conectoma y aleatoria (comparacion justa).
    omegas_IC, thetas_IC = generar_ICs(g.n_runs, N, s3.sigma, seed=g.seed)
    omegas_IC, thetas_IC = omegas_IC[None, ...], thetas_IC[None, ...]

    # Matrices por run: conectoma fijo; aleatoria = una randomizacion por run.
    rng = np.random.default_rng(g.seed)
    A_runs_conn = [A_conn for _ in range(g.n_runs)]
    A_runs_rand = [randomize_preserving_degree(A_conn, s3.n_swaps_factor, rng)
                   for _ in range(g.n_runs)]

    plot_matriz_adyacencia(A_conn, hemi, _ruta(run_dir, 'A_conectoma.png'),
                           titulo=fr'Conectoma  $N={N}$  (thr={thr:.4g})')
    plot_matriz_adyacencia(A_runs_rand[0], hemi, _ruta(run_dir, 'A_aleatoria.png'),
                           titulo=fr'Aleatoria mismo grado  $N={N}$')

    print("\n--- Barrido CONECTOMA ---")
    out_conn = barrido(N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                       K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                       A_runs=A_runs_conn, level_ids=level_ids, n_jobs=g.n_jobs)
    print("\n--- Barrido ALEATORIA (mismo grado) ---")
    out_rand = barrido(N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                       K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                       A_runs=A_runs_rand, level_ids=level_ids, n_jobs=g.n_jobs)

    _guardar(run_dir, K_grid, out_conn,
             extra={'threshold': thr, 'K_max': K_max,
                    'rand_R_mean': out_rand['R_mean'],
                    'rand_R_sigma': out_rand['R_sigma'],
                    'rand_R_err': out_rand['R_err']})
    plot_connectome(K_grid, out_conn, out_rand, N, g.n_runs, run_dir)


# ============================================================================
# Tipo 4: conectoma con pesos (vs aleatoria que conserva strength)
# ============================================================================

def run_tipo4(p, run_dir):
    """Conectoma con pesos. K es solo multiplicador escalar sobre la matriz
    de pesos W. Dos preparaciones disponibles:

       approximation = "matriz_real" -> W = promedio de los 88 sujetos,
                                        diagonal=0, valores brutos.
       approximation = "intervalos"  -> W normalizada (con log_transform
                                        opcional) y aproximada a n_levels
                                        valores en {0, 1/(n-1), ..., 1}.

    Comparacion con red aleatoria que preserva strength por nodo (4-cycle
    swap). Si la modularidad jerarquica del cerebro potencia la
    metaestabilidad, sigma_R deberia ser mayor en el conectoma que en
    su version aleatoria con misma strength.
    """
    g, c, ks, s4 = p.general, p.convergence, p.K_sweep, p.tipo4_connectome_weighted
    ruta_mat = os.path.join(DATA, s4.mat_file)

    # --- Preparar matriz de pesos W segun el modo elegido ---
    approx = getattr(s4, 'approximation', 'matriz_real')
    if approx == 'intervalos':
        n_levels      = getattr(s4, 'n_levels', 5)
        log_transform = bool(getattr(s4, 'log_transform', True))
        W = prepare_W_intervalos(ruta_mat, n_levels, log_transform=log_transform)
        n_distintos = int(len(np.unique(W)))
        descripcion = (f"intervalos (n_levels={n_levels}, "
                       f"log_transform={log_transform}; "
                       f"{n_distintos} valores distintos en W)")
    elif approx == 'matriz_real':
        W = prepare_W_real(ruta_mat)
        descripcion = "matriz real (valores brutos)"
    else:
        raise ValueError(f"approximation='{approx}' desconocido. "
                         "Usa 'matriz_real' o 'intervalos'.")

    N = W.shape[0]
    n_nonzero = int((W != 0).sum() // 2)
    strength_max = float(W.sum(axis=1).max())
    print(f"Conectoma preparado: {descripcion}")
    print(f"  shape={W.shape}, |E_nonzero|={n_nonzero}, "
          f"max(W)={W.max():.4g}, max(strength)={strength_max:.4g}")

    hemi = hemisferio_ids(os.path.join(DATA, 'AAL_regions.csv'))
    level_ids = [hemi]

    # --- K-grid: K es solo multiplicador, asi que el rango es el habitual ---
    K_max = s4.K_max if s4.K_max is not None else _K_max_estable(W, g.dt)
    K_grid = K_values_log_tstudent(ks.n_K, s4.K_min, K_max,
                                   s4.K_center, ks.K_width_factor)[None, :]
    sigmas = np.array([s4.sigma])

    # ICs compartidas entre conectoma y aleatoria (comparacion justa).
    omegas_IC, thetas_IC = generar_ICs(g.n_runs, N, s4.sigma, seed=g.seed)
    omegas_IC, thetas_IC = omegas_IC[None, ...], thetas_IC[None, ...]

    # Matrices por run: conectoma fijo; aleatoria = una randomizacion por run.
    rng = np.random.default_rng(g.seed)
    W_runs_conn = [W for _ in range(g.n_runs)]
    n_swaps_factor = getattr(s4, 'n_swaps_factor', 20)
    W_runs_rand = [randomize_preserving_strength(W, n_swaps_factor, rng)
                   for _ in range(g.n_runs)]

    # Plots de la matriz W (no binaria -> el vmax adapta automaticamente).
    plot_matriz_adyacencia(W, hemi, _ruta(run_dir, 'W_conectoma.png'),
                           titulo=fr'Conectoma {approx}  $N={N}$')
    plot_matriz_adyacencia(W_runs_rand[0], hemi,
                           _ruta(run_dir, 'W_aleatoria.png'),
                           titulo=fr'Aleatoria misma strength  $N={N}$')

    print("\n--- Barrido CONECTOMA pesado ---")
    out_conn = barrido(N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                       K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                       A_runs=W_runs_conn, level_ids=level_ids, n_jobs=g.n_jobs)
    print("\n--- Barrido ALEATORIA (misma strength) ---")
    out_rand = barrido(N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                       K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                       A_runs=W_runs_rand, level_ids=level_ids, n_jobs=g.n_jobs)

    _guardar(run_dir, K_grid, out_conn,
             extra={'approximation': approx, 'K_max': K_max,
                    'strength_max': strength_max,
                    'rand_R_mean':  out_rand['R_mean'],
                    'rand_R_sigma': out_rand['R_sigma'],
                    'rand_R_err':   out_rand['R_err']})
    plot_connectome(K_grid, out_conn, out_rand, N, g.n_runs, run_dir)


# ============================================================================
# Main
# ============================================================================

DISPATCH = {0: run_tipo0, 1: run_tipo1, 2: run_tipo2,
            3: run_tipo3, 4: run_tipo4}
SUBDIR   = {0: 'tipo0_campo_medio',     1: 'tipo1_modular',
            2: 'tipo2_jerarquico',      3: 'tipo3_conectoma',
            4: 'tipo4_conectoma_pesado'}


def main():
    t0 = time.perf_counter()
    p = load_params()

    if p.max_steps < 2 * p.convergence.block_size:
        raise ValueError("max_steps (t_max/dt) debe ser >= 2*block_size.")

    nombre = f"N{p.general.N}_K{p.K_sweep.n_K}_runs{p.general.n_runs}_t{int(p.convergence.t_max)}"
    run_dir = crear_carpeta_resultados(SUBDIR[p.sim_type], nombre)
    log_file, so, se = iniciar_log(run_dir)
    try:
        print(f"sim_type = {p.sim_type}  ({SUBDIR[p.sim_type]})")
        print(f"Resultados: {run_dir}\n")
        guardar_params_txt(run_dir, p.as_dict_plano())

        DISPATCH[p.sim_type](p, run_dir)

        dt_min = (time.perf_counter() - t0) / 60
        print(f"\nListo en {dt_min:.1f} min. Resultados en: {run_dir}")
    finally:
        cerrar_log(log_file, so, se)


if __name__ == "__main__":
    main()
