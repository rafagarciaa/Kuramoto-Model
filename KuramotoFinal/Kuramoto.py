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

from kuramoto_scripts import (
    load_params,
    barrido, Kc_teorica,
    K_values_tstudent, K_values_log_tstudent,
    generar_ICs, generar_ICs_por_sigma,
    generar_As_modular, generar_As_jerarquica,
    cargar_y_preparar_A, hemisferio_ids, randomize_preserving_degree,
    crear_carpeta_resultados, guardar_params_txt,
    iniciar_log, cerrar_log, _ruta,
    plot_mean_field, plot_modular, plot_hierarchical, plot_connectome,
    plot_matriz_adyacencia,
)

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

def run_tipo0(p, run_dir):
    g, c, ks, s0 = p.general, p.convergence, p.K_sweep, p.tipo0_mean_field
    sigmas = np.linspace(s0.sigma_min, s0.sigma_max, s0.n_sigmas)

    K_grid = np.zeros((s0.n_sigmas, ks.n_K))
    for i, sigma in enumerate(sigmas):
        K_grid[i] = K_values_tstudent(ks.n_K, s0.K_min, s0.K_max,
                                      Kc_teorica(sigma), ks.K_width_factor)

    omegas_IC, thetas_IC = generar_ICs_por_sigma(s0.n_sigmas, g.n_runs, g.N,
                                                 sigmas, seed=g.seed)
    out = barrido(g.N, g.dt, p.max_steps, c.block_size, c.conv_threshold,
                  K_grid, sigmas, g.n_runs, omegas_IC, thetas_IC,
                  A_runs=None, level_ids=None, n_jobs=g.n_jobs)

    _guardar(run_dir, K_grid, out, extra={'sigmas': sigmas})
    plot_mean_field(K_grid, sigmas, out, g.N, g.n_runs, run_dir)


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
# Main
# ============================================================================

DISPATCH = {0: run_tipo0, 1: run_tipo1, 2: run_tipo2, 3: run_tipo3}
SUBDIR   = {0: 'tipo0_campo_medio', 1: 'tipo1_modular',
            2: 'tipo2_jerarquico', 3: 'tipo3_conectoma'}


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
