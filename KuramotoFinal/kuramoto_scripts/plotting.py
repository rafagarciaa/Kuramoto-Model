"""
kuramoto_scripts.plotting
==========================

Figuras de los barridos, una orquestacion por sim_type:

    plot_mean_field   (sim_type 0): R(K) y sigma_R(K) por sigma.
    plot_modular      (sim_type 1): R + r_m por modulo; sigma.
    plot_hierarchical (sim_type 2): figura global (R + r^1) y una figura por
                                    modulo de nivel 1 (r^1_m + sus r^2 hijos).
    plot_connectome   (sim_type 3): R + r por hemisferio, y comparacion de
                                    sigma_R conectoma vs red aleatoria.

Convencion de datos: el dict `out` viene de sweep.barrido y trae
R_mean/R_sigma/R_err (n_sigmas, n_K) y levels[l] = {rm_mean, rm_sigma, rm_err}
de shape (n_sigmas, n_K, n_grupos). Para red n_sigmas = 1 (fila 0).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from kuramoto_scripts.observables import Kc_teorica, Kc_experimental
from kuramoto_scripts.io          import _ruta


# ----------------------------------------------------------------------------
# Suavizado para visualizacion (PCHIP)
# ----------------------------------------------------------------------------

def _smooth(x, y, log_x=False, n=300):
    """Spline monotono por tramos (PCHIP) para suavizar la curva visual.

    Mantenemos los marcadores como dato; esto es solo decoracion. PCHIP
    se prefiere al cubico clasico porque no produce overshoots en
    transiciones abruptas (R pasando de ~0 a ~1 cerca de Kc).

    En eje log el x_fine es geomspace para que la curva quede suave en
    pantalla con escala logaritmica.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4:
        return x, y
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    keep = np.concatenate([[True], np.diff(xs) > 0])
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 4:
        return xs, ys
    spline = PchipInterpolator(xs, ys, extrapolate=False)
    x_fine = (np.geomspace(xs[0], xs[-1], n) if log_x
              else np.linspace(xs[0], xs[-1], n))
    return x_fine, spline(x_fine)


# ----------------------------------------------------------------------------
# Estilo
# ----------------------------------------------------------------------------

def setup_plot_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'DejaVu Serif'],
        'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
        'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': 1.0, 'axes.grid': True, 'grid.alpha': 0.25,
        'grid.linestyle': '--', 'grid.linewidth': 0.5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'legend.frameon': True, 'legend.framealpha': 0.95,
        'legend.edgecolor': 'black', 'figure.dpi': 100,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })


def _info_box(ax, text, loc='bottom'):
    y, va = (0.05, 'bottom') if loc == 'bottom' else (0.95, 'top')
    ax.text(0.98, y, text, transform=ax.transAxes, ha='right', va=va,
            fontsize=9, style='italic', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.7))


def _set_K_axis(ax, log_x):
    if log_x:
        ax.set_xscale('log')
        ax.set_xlabel(r'Acoplamiento $K$  (escala log)')
    else:
        ax.set_xlim(left=0)
        ax.set_xlabel(r'Acoplamiento $K$')


# ----------------------------------------------------------------------------
# sim_type 0: campo medio
# ----------------------------------------------------------------------------

def plot_mean_field(K_grid, sigmas, out, N, n_runs, save_dir):
    setup_plot_style()
    R_mean, R_sigma, R_err = out['R_mean'], out['R_sigma'], out['R_err']
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigmas)))

    def draw_R(ax):
        for i, sigma in enumerate(sigmas):
            Kc = Kc_teorica(sigma)
            ax.fill_between(K_grid[i], R_mean[i] - R_err[i], R_mean[i] + R_err[i],
                            color=colors[i], alpha=0.25)
            # Marcadores = dato; spline (PCHIP) = decoracion visual.
            ax.plot(K_grid[i], R_mean[i], 'o', ms=4, color=colors[i])
            xs, ys = _smooth(K_grid[i], R_mean[i], log_x=False)
            ax.plot(xs, ys, '-', lw=1.5, color=colors[i],
                    label=fr'$\sigma={sigma:.2f}$  $K_c^{{th}}={Kc:.2f}$')
            ax.axvline(Kc, color=colors[i], ls='--', lw=1.0, alpha=0.6)
        _set_K_axis(ax, log_x=False)
        ax.set_ylabel(r'$\langle R \rangle$'); ax.set_ylim(-0.02, 1.02)
        ax.set_title('Transicion de sincronizacion (campo medio)')
        ax.legend(loc='lower right', title=r'Lineas: $K_c$ teorica')

    def draw_sigma(ax):
        for i, sigma in enumerate(sigmas):
            Kc_exp = Kc_experimental(K_grid[i], R_sigma[i], log=False)
            ax.plot(K_grid[i], R_sigma[i], 'o', ms=4, color=colors[i])
            xs, ys = _smooth(K_grid[i], R_sigma[i], log_x=False)
            ax.plot(xs, ys, '-', lw=1.5, color=colors[i],
                    label=fr'$\sigma={sigma:.2f}$  $K_c^{{exp}}={Kc_exp:.3g}$')
            ax.axvline(Kc_exp, color=colors[i], ls=':', lw=1.2, alpha=0.8)
        _set_K_axis(ax, log_x=False)
        ax.set_ylabel(r'$\sigma_R$'); ax.set_title('Metaestabilidad')
        ax.legend(loc='upper right', title=r'Lineas: $K_c$ experimental')

    fig, ax = plt.subplots(figsize=(7.5, 5.2)); draw_R(ax)
    _info_box(ax, fr'$N={N}$, runs$={n_runs}$', 'bottom')
    fig.savefig(_ruta(save_dir, 'R_vs_K.png')); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2)); draw_sigma(ax)
    _info_box(ax, fr'$N={N}$, runs$={n_runs}$', 'top')
    fig.savefig(_ruta(save_dir, 'sigmaR_vs_K.png')); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    draw_R(a1); draw_sigma(a2)
    fig.tight_layout(); fig.savefig(_ruta(save_dir, 'combinado.png')); plt.close(fig)


# ----------------------------------------------------------------------------
# sim_type 0 con scaling: extrapolacion de Kc a N->inf (1/N -> 0)
# ----------------------------------------------------------------------------

def plot_scaling_Kc(inv_N, N_values, Kc_per_N, Kc_inf, lines, sigmas, save_dir):
    """Finite-size scaling: Kc(N) vs 1/N + recta extrapolada a 1/N=0.

    inv_N     : (n_N,)               -> 1/N de cada tamaño.
    Kc_per_N  : (n_sigmas, n_N)      -> Kc experimental para cada (sigma, N).
    Kc_inf    : (n_sigmas,)          -> corte de la recta en 1/N=0.
    lines     : lista de (pendiente, ordenada) por sigma.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigmas)))
    x_fit = np.array([0.0, float(np.max(inv_N))])

    for i, sigma in enumerate(sigmas):
        c = colors[i]
        slope, intercept = lines[i]
        # Puntos Kc(N).
        ax.plot(inv_N, Kc_per_N[i], 'o', ms=6, color=c)
        # Recta de ajuste, extendida hasta 1/N = 0.
        ax.plot(x_fit, intercept + slope * x_fit, '--', lw=1.3, color=c)
        # Punto extrapolado en 1/N = 0.
        ax.plot(0.0, Kc_inf[i], '*', ms=15, color=c, zorder=5)
        # Kc teorico de referencia.
        Kc_th = Kc_teorica(sigma)
        ax.axhline(Kc_th, ls=':', lw=1.0, color=c, alpha=0.5)
        ax.plot([], [], '-', color=c,
                label=fr'$\sigma={sigma:.2f}$: $K_c^\infty={Kc_inf[i]:.3f}$, '
                      fr'$K_c^{{th}}={Kc_th:.3f}$')

    ax.axvline(0.0, color='gray', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$1/N$')
    ax.set_ylabel(r'$K_c$')
    ax.set_xlim(left=-0.02 * float(np.max(inv_N)))
    ax.set_title(r'Finite-size scaling: extrapolacion de $K_c$ a $N\to\infty$')
    ax.legend(loc='best', title=r'$\bigstar$ = $K_c$ extrapolado ($1/N=0$)')
    _info_box(ax, fr'$N$: {", ".join(str(n) for n in N_values)}', 'top')
    fig.savefig(_ruta(save_dir, 'scaling_Kc.png'))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Helpers para red (una sola sigma -> fila 0)
# ----------------------------------------------------------------------------

def _draw_global_R(ax, K, R_mean, R_err, log_x, label=r'$\langle R\rangle$ (global)'):
    ax.fill_between(K, R_mean - R_err, R_mean + R_err, color='k', alpha=0.2)
    ax.plot(K, R_mean, 'o', ms=4, color='k')
    xs, ys = _smooth(K, R_mean, log_x=log_x)
    ax.plot(xs, ys, '-', lw=1.8, color='k', label=label)


def _draw_groups(ax, K, rm_mean, rm_err, labels, log_x=False, cmap_lims=(0.2, 0.85)):
    ng = rm_mean.shape[-1]
    colors = plt.cm.viridis(np.linspace(*cmap_lims, ng))
    for g in range(ng):
        ax.fill_between(K, rm_mean[:, g] - rm_err[:, g], rm_mean[:, g] + rm_err[:, g],
                        color=colors[g], alpha=0.15)
        ax.plot(K, rm_mean[:, g], 'o', ms=3, color=colors[g])
        xs, ys = _smooth(K, rm_mean[:, g], log_x=log_x)
        ax.plot(xs, ys, '-', lw=1.2, color=colors[g], label=labels[g])
    return colors


# ----------------------------------------------------------------------------
# sim_type 1: modular
# ----------------------------------------------------------------------------

def plot_modular(K_grid, out, N, n_runs, n_modules, save_dir):
    setup_plot_style()
    K = K_grid[0]
    R_mean, R_sigma, R_err = out['R_mean'][0], out['R_sigma'][0], out['R_err'][0]
    lvl = out['levels'][0]
    labels = [fr'$\langle r_{{{g+1}}}\rangle$' for g in range(n_modules)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))

    _draw_groups(a1, K, lvl['rm_mean'][0], lvl['rm_err'][0], labels, log_x=True)
    _draw_global_R(a1, K, R_mean, R_err, log_x=True)
    _set_K_axis(a1, log_x=True)
    a1.set_ylabel('Parametros de orden'); a1.set_ylim(-0.02, 1.02)
    a1.set_title('Sincronizacion local vs global (red modular)')
    a1.legend(loc='lower right')
    _info_box(a1, fr'$N={N}$, $M={n_modules}$, runs$={n_runs}$', 'bottom')

    ng = n_modules
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, ng))
    for g in range(ng):
        Kc_m = Kc_experimental(K, lvl['rm_sigma'][0][:, g], log=True)
        a2.plot(K, lvl['rm_sigma'][0][:, g], 'o', ms=3, color=colors[g])
        xs, ys = _smooth(K, lvl['rm_sigma'][0][:, g], log_x=True)
        a2.plot(xs, ys, '-', lw=1.0, color=colors[g],
                label=fr'$\sigma_{{r_{{{g+1}}}}}$  $K_c={Kc_m:.3g}$')
    Kc_g = Kc_experimental(K, R_sigma, log=True)
    a2.plot(K, R_sigma, 'o', ms=4, color='k')
    xs, ys = _smooth(K, R_sigma, log_x=True)
    a2.plot(xs, ys, '-', lw=1.8, color='k',
            label=fr'$\sigma_R$ (global)  $K_c={Kc_g:.3g}$')
    a2.axvline(Kc_g, color='k', ls='--', lw=1.2, alpha=0.7)
    _set_K_axis(a2, log_x=True)
    a2.set_ylabel(r'$\sigma$'); a2.set_title('Metaestabilidad')
    a2.legend(loc='upper right')
    _info_box(a2, fr'$N={N}$, $M={n_modules}$, runs$={n_runs}$', 'top')

    fig.tight_layout(); fig.savefig(_ruta(save_dir, 'combinado.png')); plt.close(fig)


# ----------------------------------------------------------------------------
# sim_type 2: jerarquico
# ----------------------------------------------------------------------------

def plot_hierarchical(K_grid, out, module_id, submodule_id, N, n_runs, save_dir):
    setup_plot_style()
    K = K_grid[0]
    R_mean, R_sigma, R_err = out['R_mean'][0], out['R_sigma'][0], out['R_err'][0]
    nivel_mod = out['levels'][0]   # modulos (nivel 1)
    nivel_sub = out['levels'][1]   # submodulos (nivel 2)

    n_modules = int(module_id.max()) + 1

    # Mapa submodulo_global -> modulo padre.
    n_subs = int(submodule_id.max()) + 1
    parent = np.zeros(n_subs, dtype=int)
    for s in range(n_subs):
        parent[s] = module_id[np.where(submodule_id == s)[0][0]]

    # --- Figura global: R + r^1 por modulo ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))
    labels = [fr'$\langle r^1_{{{m+1}}}\rangle$' for m in range(n_modules)]
    _draw_groups(a1, K, nivel_mod['rm_mean'][0], nivel_mod['rm_err'][0], labels, log_x=True)
    _draw_global_R(a1, K, R_mean, R_err, log_x=True)
    _set_K_axis(a1, log_x=True)
    a1.set_ylabel('Parametros de orden'); a1.set_ylim(-0.02, 1.02)
    a1.set_title('Global y modulos de nivel 1')
    a1.legend(loc='lower right')
    _info_box(a1, fr'$N={N}$, $M={n_modules}$, runs$={n_runs}$', 'bottom')

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_modules))
    for m in range(n_modules):
        a2.plot(K, nivel_mod['rm_sigma'][0][:, m], 'o', ms=3, color=colors[m])
        xs, ys = _smooth(K, nivel_mod['rm_sigma'][0][:, m], log_x=True)
        a2.plot(xs, ys, '-', lw=1.2, color=colors[m],
                label=fr'$\sigma_{{r^1_{{{m+1}}}}}$')
    Kc_g = Kc_experimental(K, R_sigma, log=True)
    a2.plot(K, R_sigma, 'o', ms=4, color='k')
    xs, ys = _smooth(K, R_sigma, log_x=True)
    a2.plot(xs, ys, '-', lw=1.8, color='k',
            label=fr'$\sigma_R$ (global)  $K_c={Kc_g:.3g}$')
    a2.axvline(Kc_g, color='k', ls='--', lw=1.2, alpha=0.7)
    _set_K_axis(a2, log_x=True)
    a2.set_ylabel(r'$\sigma$'); a2.set_title('Metaestabilidad (nivel 1)')
    a2.legend(loc='upper right')
    _info_box(a2, fr'$N={N}$, $M={n_modules}$, runs$={n_runs}$', 'top')

    fig.tight_layout(); fig.savefig(_ruta(save_dir, 'global_nivel1.png')); plt.close(fig)

    # --- Una figura por modulo de nivel 1: su r^1 + los r^2 de sus submodulos ---
    for m in range(n_modules):
        subs_m = np.where(parent == m)[0]
        fig, ax = plt.subplots(figsize=(8, 5.2))

        colors = plt.cm.plasma(np.linspace(0.15, 0.8, len(subs_m)))
        for k, s in enumerate(subs_m):
            ax.plot(K, nivel_sub['rm_mean'][0][:, s], 'o', ms=3, color=colors[k])
            xs, ys = _smooth(K, nivel_sub['rm_mean'][0][:, s], log_x=True)
            ax.plot(xs, ys, '-', lw=1.1, color=colors[k],
                    label=fr'$\langle r^2_{{{s+1}}}\rangle$')

        # r^1 del modulo padre, en negro grueso.
        ax.plot(K, nivel_mod['rm_mean'][0][:, m], 's', ms=4, color='k')
        xs, ys = _smooth(K, nivel_mod['rm_mean'][0][:, m], log_x=True)
        ax.plot(xs, ys, '-', lw=2.0, color='k',
                label=fr'$\langle r^1_{{{m+1}}}\rangle$ (modulo)')

        _set_K_axis(ax, log_x=True)
        ax.set_ylabel('Parametros de orden'); ax.set_ylim(-0.02, 1.02)
        ax.set_title(fr'Modulo {m+1}: nivel 1 vs submodulos (nivel 2)')
        ax.legend(loc='lower right')
        _info_box(ax, fr'$N={N}$, runs$={n_runs}$', 'bottom')
        fig.savefig(_ruta(save_dir, f'modulo_{m+1}.png')); plt.close(fig)


# ----------------------------------------------------------------------------
# sim_type 3: conectoma vs red aleatoria
# ----------------------------------------------------------------------------

def plot_connectome(K_grid, out_conn, out_rand, N, n_runs, save_dir,
                    hemi_labels=('izq', 'der')):
    setup_plot_style()
    K = K_grid[0]

    # --- Figura 1: R global + r por hemisferio (conectoma) ---
    R_mean, R_sigma, R_err = out_conn['R_mean'][0], out_conn['R_sigma'][0], out_conn['R_err'][0]
    hemi = out_conn['levels'][0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))

    labels = [fr'$\langle r_{{{h}}}\rangle$' for h in hemi_labels]
    _draw_groups(a1, K, hemi['rm_mean'][0], hemi['rm_err'][0], labels, log_x=True)
    _draw_global_R(a1, K, R_mean, R_err, log_x=True)
    _set_K_axis(a1, log_x=True)
    a1.set_ylabel('Parametros de orden'); a1.set_ylim(-0.02, 1.02)
    a1.set_title('Conectoma: global y hemisferios')
    a1.legend(loc='lower right')
    _info_box(a1, fr'$N={N}$, runs$={n_runs}$', 'bottom')

    for h in range(hemi['rm_sigma'].shape[-1]):
        a2.plot(K, hemi['rm_sigma'][0][:, h], 'o', ms=3)
        xs, ys = _smooth(K, hemi['rm_sigma'][0][:, h], log_x=True)
        a2.plot(xs, ys, '-', lw=1.2,
                label=fr'$\sigma_{{r_{{{hemi_labels[h]}}}}}$')
    a2.plot(K, R_sigma, 'o', ms=4, color='k')
    xs, ys = _smooth(K, R_sigma, log_x=True)
    a2.plot(xs, ys, '-', lw=1.8, color='k', label=r'$\sigma_R$ (global)')
    _set_K_axis(a2, log_x=True)
    a2.set_ylabel(r'$\sigma$'); a2.set_title('Metaestabilidad (conectoma)')
    a2.legend(loc='upper right')
    _info_box(a2, fr'$N={N}$, runs$={n_runs}$', 'top')
    fig.tight_layout(); fig.savefig(_ruta(save_dir, 'conectoma.png')); plt.close(fig)

    # --- Figura 2: comparacion conectoma vs aleatoria ---
    Rc, Sc = out_conn['R_mean'][0], out_conn['R_sigma'][0]
    Rr, Sr = out_rand['R_mean'][0], out_rand['R_sigma'][0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2))

    a1.plot(K, Rc, 'o', ms=4, color='#1f77b4')
    xs, ys = _smooth(K, Rc, log_x=True)
    a1.plot(xs, ys, '-', lw=1.8, color='#1f77b4', label='Conectoma')
    a1.plot(K, Rr, 's', ms=4, color='#d62728')
    xs, ys = _smooth(K, Rr, log_x=True)
    a1.plot(xs, ys, '--', lw=1.6, color='#d62728', label='Aleatoria (mismo grado)')
    _set_K_axis(a1, log_x=True)
    a1.set_ylabel(r'$\langle R \rangle$'); a1.set_ylim(-0.02, 1.02)
    a1.set_title(r'$\langle R\rangle$: conectoma vs aleatoria')
    a1.legend(loc='lower right')

    Kc_c = Kc_experimental(K, Sc, log=True)
    Kc_r = Kc_experimental(K, Sr, log=True)
    a2.plot(K, Sc, 'o', ms=4, color='#1f77b4')
    xs, ys = _smooth(K, Sc, log_x=True)
    a2.plot(xs, ys, '-', lw=1.8, color='#1f77b4',
            label=fr'Conectoma  $K_c={Kc_c:.3g}$')
    a2.plot(K, Sr, 's', ms=4, color='#d62728')
    xs, ys = _smooth(K, Sr, log_x=True)
    a2.plot(xs, ys, '--', lw=1.6, color='#d62728',
            label=fr'Aleatoria  $K_c={Kc_r:.3g}$')
    _set_K_axis(a2, log_x=True)
    a2.set_ylabel(r'$\sigma_R$')
    a2.set_title('Metaestabilidad: conectoma vs aleatoria')
    a2.legend(loc='upper right')
    _info_box(a2, fr'$N={N}$, runs$={n_runs}$', 'top')
    fig.tight_layout(); fig.savefig(_ruta(save_dir, 'comparacion.png')); plt.close(fig)


# ----------------------------------------------------------------------------
# Diagnostico de matriz (reusado por todos los modos de red)
# ----------------------------------------------------------------------------

def plot_matriz_adyacencia(A, group_id=None, save_path=None, titulo=None):
    setup_plot_style()
    N = A.shape[0]
    n_edges = int(A.sum() // 2)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(A, cmap='Greys', aspect='equal', interpolation='nearest', vmin=0, vmax=1)
    if group_id is not None:
        sizes = np.bincount(np.asarray(group_id), minlength=int(group_id.max()) + 1)
        for c in np.cumsum(sizes)[:-1]:
            ax.axhline(c - 0.5, color='red', lw=0.8, alpha=0.7)
            ax.axvline(c - 0.5, color='red', lw=0.8, alpha=0.7)
    ax.set_title(titulo or fr'Matriz de adyacencia  $N={N}$, $|E|={n_edges}$')
    ax.set_xlabel('Nodo'); ax.set_ylabel('Nodo')
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path
