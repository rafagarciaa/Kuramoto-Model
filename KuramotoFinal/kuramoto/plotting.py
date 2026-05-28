"""
kuramoto.plotting
=================

Estilo y plots para los barridos en K.

Convenciones:

    - El estilo (setup_plot_style) se llama UNA vez al inicio de cada
      plot publico. Asi puedes ejecutar plots sueltos sin acordarte.

    - Los `_draw_*` son helpers privados que dibujan sobre un `ax` ya
      creado. Permiten componer figuras combinadas sin duplicar codigo.

    - Si la rejilla de K viene de un modo red (log K), pasamos `log_x=True`
      y los plots usan eje log. En campo medio (lineal) no.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend no interactivo: thread-safe y sin GUI.
import matplotlib.pyplot as plt

from kuramoto.observables import Kc_teorica, Kc_experimental
from kuramoto.io          import _ruta


def setup_plot_style():
    """Estilo profesional uniforme para todas las figuras."""
    plt.rcParams.update({
        'font.family'       : 'serif',
        'font.serif'        : ['Computer Modern Roman', 'DejaVu Serif'],
        'font.size'         : 11,
        'axes.labelsize'    : 13,
        'axes.titlesize'    : 14,
        'legend.fontsize'   : 10,
        'xtick.labelsize'   : 10,
        'ytick.labelsize'   : 10,
        'axes.linewidth'    : 1.0,
        'axes.grid'         : True,
        'grid.alpha'        : 0.25,
        'grid.linestyle'    : '--',
        'grid.linewidth'    : 0.5,
        'xtick.direction'   : 'in',
        'ytick.direction'   : 'in',
        'xtick.top'         : True,
        'ytick.right'       : True,
        'xtick.major.size'  : 5,
        'ytick.major.size'  : 5,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'legend.frameon'    : True,
        'legend.framealpha' : 0.95,
        'legend.edgecolor'  : 'black',
        'legend.fancybox'   : False,
        'figure.dpi'        : 100,
        'savefig.dpi'       : 300,
        'savefig.bbox'      : 'tight',
    })


# ----------------------------------------------------------------------------
# Cajitas con info de la corrida
# ----------------------------------------------------------------------------

def _add_info_box(ax, info_text, loc='bottom'):
    """Cajita esquina inferior/superior derecha con info textual."""
    y, va = (0.05, 'bottom') if loc == 'bottom' else (0.95, 'top')
    ax.text(0.98, y, info_text, transform=ax.transAxes,
            ha='right', va=va, fontsize=9, style='italic', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.7))


# ----------------------------------------------------------------------------
# Helpers de dibujo: <R> vs K  y  sigma_R vs K
# ----------------------------------------------------------------------------

def _draw_R_vs_K(ax, K_values_per_sigma, sigma_values,
                 R_means, R_mean_stds, colors,
                 log_x=False, mostrar_Kc_teorica=True,
                 rm_means=None, rm_mean_stds=None, num_modules=0):
    """Dibuja <R>(K) con banda +/- sigma sobre runs.

    Si rm_means esta dado, dibuja tambien las curvas <r_m>(K) por modulo
    para CADA sigma (esto solo tiene sentido si len(sigma_values)==1, que
    es el caso de red, asi que en la practica solo se usa para Tarea 2).
    """
    for i, sigma in enumerate(sigma_values):
        # Curvas por modulo (solo modo red).
        if rm_means is not None and num_modules > 0:
            cmap = plt.cm.viridis(np.linspace(0.2, 0.85, num_modules))
            for m in range(num_modules):
                ax.fill_between(K_values_per_sigma[i],
                                rm_means[i, :, m] - rm_mean_stds[i, :, m],
                                rm_means[i, :, m] + rm_mean_stds[i, :, m],
                                color=cmap[m], alpha=0.15)
                ax.plot(K_values_per_sigma[i], rm_means[i, :, m],
                        marker='o', markersize=3, linewidth=1.2,
                        color=cmap[m],
                        label=fr'$\langle r_{{{m+1}}} \rangle$')

        # Curva global <R>(K).
        ax.fill_between(K_values_per_sigma[i],
                        R_means[i] - R_mean_stds[i],
                        R_means[i] + R_mean_stds[i],
                        color=colors[i], alpha=0.25)
        if mostrar_Kc_teorica:
            Kc_th = Kc_teorica(sigma)
            label = (fr'$\sigma = {sigma:.2f}$   '
                     fr'$K_c^{{\mathrm{{th}}}} = {Kc_th:.2f}$')
            ax.axvline(Kc_th, color=colors[i], linestyle='--',
                       linewidth=1.0, alpha=0.6)
        else:
            label = fr'$\langle R \rangle$ (global)'
        ax.plot(K_values_per_sigma[i], R_means[i],
                marker='o', markersize=4, linewidth=1.5,
                color=colors[i], label=label)

    if log_x:
        ax.set_xscale('log')
        ax.set_xlabel(r'Acoplamiento $K$  (escala log)')
    else:
        ax.set_xlim(left=0)
        ax.set_xlabel(r'Acoplamiento $K$')
    ax.set_ylabel(r'Parametro de orden $\langle R \rangle$')
    ax.set_title(r'Transicion de sincronizacion')
    ax.set_ylim(-0.02, 1.02)
    leg_title = r'Lineas: $K_c$ teorica' if mostrar_Kc_teorica else None
    ax.legend(loc='lower right', title=leg_title)


def _draw_sigmaR_vs_K(ax, K_values_per_sigma, sigma_values,
                       R_stds, colors,
                       log_x=False,
                       rm_stds=None, num_modules=0):
    """Dibuja sigma_R(K) y marca el Kc experimental (vertice parabolico)."""
    for i, sigma in enumerate(sigma_values):
        if rm_stds is not None and num_modules > 0:
            cmap = plt.cm.viridis(np.linspace(0.2, 0.85, num_modules))
            for m in range(num_modules):
                Kc_m = Kc_experimental(K_values_per_sigma[i], rm_stds[i, :, m],
                                       log=log_x)
                ax.plot(K_values_per_sigma[i], rm_stds[i, :, m],
                        marker='o', markersize=3, linewidth=1.0,
                        color=cmap[m],
                        label=fr'$\sigma_{{r_{{{m+1}}}}}$   '
                              fr'$K_c^{{({m+1})}} = {Kc_m:.3g}$')
                ax.axvline(Kc_m, color=cmap[m], linestyle=':',
                           linewidth=1.0, alpha=0.6)

        Kc_exp = Kc_experimental(K_values_per_sigma[i], R_stds[i], log=log_x)
        ax.plot(K_values_per_sigma[i], R_stds[i],
                marker='o', markersize=4, linewidth=1.5,
                color=colors[i],
                label=fr'$\sigma = {sigma:.2f}$   '
                      fr'$K_c^{{\mathrm{{exp}}}} = {Kc_exp:.3g}$')
        ax.axvline(Kc_exp, color=colors[i], linestyle=':',
                   linewidth=1.2, alpha=0.8)

    if log_x:
        ax.set_xscale('log')
        ax.set_xlabel(r'Acoplamiento $K$  (escala log)')
    else:
        ax.set_xlim(left=0)
        ax.set_xlabel(r'Acoplamiento $K$')
    ax.set_ylabel(r'Desviacion estandar $\sigma_R$')
    ax.set_title(r'Metaestabilidad: fluctuaciones de $R$')
    ax.legend(loc='upper right', title=r'Lineas: $K_c$ experimental')


# ----------------------------------------------------------------------------
# Plots publicos
# ----------------------------------------------------------------------------

def _info_text(N, num_runs, num_modules=0):
    if num_modules > 0:
        return fr'$N = {N}$, $M = {num_modules}$, runs $= {num_runs}$'
    return fr'$N = {N}$, runs $= {num_runs}$'


def plot_R_vs_K(K_values_per_sigma, sigma_values, R_means, R_mean_stds,
                N, num_runs, save_dir,
                log_x=False, mostrar_Kc_teorica=True,
                rm_means=None, rm_mean_stds=None, num_modules=0,
                nombre='R_vs_K.png'):
    """Una figura: <R> vs K para cada sigma."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_R_vs_K(ax, K_values_per_sigma, sigma_values, R_means, R_mean_stds,
                 colors, log_x=log_x, mostrar_Kc_teorica=mostrar_Kc_teorica,
                 rm_means=rm_means, rm_mean_stds=rm_mean_stds,
                 num_modules=num_modules)
    _add_info_box(ax, _info_text(N, num_runs, num_modules), loc='bottom')

    fig.savefig(_ruta(save_dir, nombre))
    plt.close(fig)


def plot_sigmaR_vs_K(K_values_per_sigma, sigma_values, R_stds,
                      N, num_runs, save_dir,
                      log_x=False,
                      rm_stds=None, num_modules=0,
                      nombre='sigmaR_vs_K.png'):
    """Una figura: sigma_R(K)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_sigmaR_vs_K(ax, K_values_per_sigma, sigma_values, R_stds, colors,
                       log_x=log_x, rm_stds=rm_stds, num_modules=num_modules)
    _add_info_box(ax, _info_text(N, num_runs, num_modules), loc='top')

    fig.savefig(_ruta(save_dir, nombre))
    plt.close(fig)


def plot_combined(K_values_per_sigma, sigma_values,
                   R_means, R_stds, R_mean_stds,
                   N, num_runs, save_dir,
                   log_x=False, mostrar_Kc_teorica=True,
                   rm_means=None, rm_stds=None, rm_mean_stds=None,
                   num_modules=0,
                   nombre='combinado.png'):
    """Figura combinada: <R>(K) a la izquierda, sigma_R(K) a la derecha."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_R_vs_K(ax1, K_values_per_sigma, sigma_values, R_means, R_mean_stds,
                 colors, log_x=log_x, mostrar_Kc_teorica=mostrar_Kc_teorica,
                 rm_means=rm_means, rm_mean_stds=rm_mean_stds,
                 num_modules=num_modules)
    _add_info_box(ax1, _info_text(N, num_runs, num_modules), loc='bottom')

    _draw_sigmaR_vs_K(ax2, K_values_per_sigma, sigma_values, R_stds, colors,
                       log_x=log_x, rm_stds=rm_stds, num_modules=num_modules)
    _add_info_box(ax2, _info_text(N, num_runs, num_modules), loc='top')

    fig.tight_layout()
    fig.savefig(_ruta(save_dir, nombre))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plot de la matriz de adyacencia (diagnostico)
# ----------------------------------------------------------------------------

def plot_matriz_adyacencia(A, module_id=None, save_path=None, titulo=None):
    """Visualiza A como imagen, con separadores de modulo si module_id.

    Si module_id es None, dibuja solo la matriz (caso conectoma). Si esta
    dado, anade lineas rojas separando bloques y un panel a la derecha con
    la densidad por bloque (intra/inter modulo).
    """
    setup_plot_style()
    N = A.shape[0]
    n_edges = int(A.sum() // 2)

    if module_id is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(A, cmap='Greys', aspect='equal',
                  interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(titulo or fr'Matriz de adyacencia  $N={N}$, $|E|={n_edges}$')
        ax.set_xlabel('Nodo')
        ax.set_ylabel('Nodo')
    else:
        from kuramoto.networks import stats_matriz_adyacencia
        num_modules = int(module_id.max()) + 1

        fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                                 gridspec_kw={'width_ratios': [2, 1]})
        ax = axes[0]
        ax.imshow(A, cmap='Greys', aspect='equal',
                  interpolation='nearest', vmin=0, vmax=1)

        sizes = np.bincount(module_id, minlength=num_modules)
        cum   = np.cumsum(sizes)
        for c in cum[:-1]:
            ax.axhline(c - 0.5, color='red', linewidth=1.0, alpha=0.8)
            ax.axvline(c - 0.5, color='red', linewidth=1.0, alpha=0.8)

        centers, start = [], 0
        for m in range(num_modules):
            centers.append(start + sizes[m] / 2 - 0.5 if sizes[m] > 0 else np.nan)
            start += sizes[m]
        visibles = [(c, m) for c, m in zip(centers, range(num_modules))
                    if not np.isnan(c)]
        if visibles:
            ticks, labs = zip(*[(c, f'M{m}') for c, m in visibles])
            ax.set_xticks(ticks); ax.set_xticklabels(labs)
            ax.set_yticks(ticks); ax.set_yticklabels(labs)

        ax.set_title(titulo or fr'Matriz de adyacencia  $N={N}$, $|E|={n_edges}$')

        # Panel derecho: densidades por bloque.
        ax = axes[1]
        dens = stats_matriz_adyacencia(A, module_id)
        im = ax.imshow(dens, cmap='viridis', aspect='equal', vmin=0, vmax=1)
        for i in range(num_modules):
            for j in range(num_modules):
                ax.text(j, i, f'{dens[i,j]:.3g}', ha='center', va='center',
                        color='white' if dens[i,j] < 0.5 else 'black',
                        fontsize=11)
        ax.set_xticks(range(num_modules))
        ax.set_xticklabels([f'M{m}' for m in range(num_modules)])
        ax.set_yticks(range(num_modules))
        ax.set_yticklabels([f'M{m}' for m in range(num_modules)])
        ax.set_title('Densidad por bloque')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        txt = '  '.join([fr'$|M_{m}|={sizes[m]}$' for m in range(num_modules)])
        fig.text(0.5, 0.01, txt, ha='center', fontsize=10,
                 style='italic', alpha=0.8)

    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return save_path
