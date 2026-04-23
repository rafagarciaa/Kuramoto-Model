import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit
from scipy.stats import norm
from scipy.stats import t as t_dist
from joblib import Parallel, delayed

# Resumen del script: Este código se encarga fundamentalmente de la primera tarea.
# Los primero que se hace es configurar el estilo de las gráficas, para que se vean profesionales.
# Después configuramos el PATH para que se organicen de forma inteligente y ordenada.
# La clase KuramotoSystem es una forma de variable que nos permite guardar toda una simulación
# completa en UNA SOLA VARIABLE. Esto es increíblemente útil y elegante para organizar los datos. 
# De esta forma no tenemos que estar trabajando con arrays independientes, o perdiendo memoria dedicada a 
# almacenar datos inútiles en estructuras ineficientes. Aquí guardamos TODA la información
# necesaria para los cálculos. Dentro de la clase definimos varios métodos que nos servirán para
# incializar los elementos, ejecutar las simulaciones y demás. Luego tenemos algunas @properties. 
# Una property, es en realidad un método, pero que al definir como property añadimos la sintaxis
# de un atributo. Es decir, en vez de llamar a la funcion y tener que indicarle qué valores usar, 
# se supone que la función utiliza los valores de la clase (y el objeto particular con el que esté 
# trabajando en ese instante). De esta forma tratamos un método como un atributo, ya que no modifica
# el objeto, si no que hace algún cálculo sobre él. Usamos esto para la media y la desviación estándar
# de R. Luego tenemos la función _integrar, es el corazón del programa, integrando el método de euler para
# los cálculos básicos. Usamos para ella el decorador njit (no python) para optimizar el código.
# Más adelante tenemos la función Simulacion_Kuramoto, que es simplemente una envoltura
# de Simulacion_Kuramoto que a su vez utiliza _integrar para resolver el sistema, junto algunos métodos del objeto 
# que hemos llamado "sys". Luego tenemos la función _una_simulacion_indexada. Esta función surge únicamente
# del intento de optimizar el código aún más con joblib, donde ejecutamos varias simulaciones individualmente
# para poder usar varios núcleos de la CPU. Estas simulaciones se indexan para ser ordenadas luego
# de forma lógica por el índice. A continuación tenemos el cálculo de la Kc teórica y experimental. 
# Para la teórica usamos la aproximación estandar. Mientras que para calcular la experimental hemos
# optado por usar el máximo de la desviación estándar de R, que como veremos es una muy buena aproximación. 
# Finalmente, antes del main(), tenemos dos secciones, la primera básicamente estamos creando una distribución 
# no uniforme de las K para obtener mayor nivel de detalle cuando la curva R_vs_K tiene pendiente, ya que
# si usamos una distribución lineal en las zonas constantes, tendremos un equiespaciado sobre el camino de la
# curva, pero en las zonas de gran pendiente, tendremos lo opuesto. Entonces usamos varias funciones para ver cual 
# crea una distribución más detallada. Luego tenemos una función que integra todo un barrido completo en una
# sola función. Finalmente tenemos los plots, que organizan las imágenes en las dos gráficas
# que se presentan, y las almacena por separado, y luego muestra y guarda una imagen de ambas juntas
# para poder comparar sencillamente los valores de Kc teórica y experimental. Por último tenemos el main()
# que se encarga de ejecutar todo el programa y generar las gráficas.


def setup_plot_style():
    """
    Configura matplotlib con estilo profesional para publicaciones.
    """
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

# Path relativo al script: funciona desde cualquier directorio de ejecución
FIGURES_DIR = os.path.join(r'c:\Users\Rafa\Desktop\IMPORTANTE_UNIVERSIDAD\Asignaturas\5to\2doCuatri\SistemasComplejos\Kuramoto', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def _params_folder_name(N, t_max, num_K, num_runs, num_sigmas):
    """Nombre de carpeta basado en los parámetros distintivos de la corrida."""
    return f'N{N}_t{t_max}_K{num_K}_Runs{num_runs}_sigmas{num_sigmas}'


def _unique_path(directory, filename):
    """Si el archivo ya existe, añade (1), (2), ... para no sobreescribir.
    Crea el directorio si no existe."""
    os.makedirs(directory, exist_ok=True)
    base, ext = os.path.splitext(filename)
    path = os.path.join(directory, filename)
    n = 1
    while os.path.exists(path):
        path = os.path.join(directory, f'{base}({n}){ext}')
        n += 1
    return path


class KuramotoSystem:

    def __init__(self, N: int, steps: int, dt: float):
        self.N     = N          # Número de osciladores
        self.steps = steps      # Número de pasos temporales
        self.dt    = dt         # Paso de tiempo

        # Parámetro de orden R(t) y fase media ψ(t), shape (steps+1,)
        self.R   = np.zeros(steps + 1, dtype=np.float64)
        self.psi = np.zeros(steps + 1, dtype=np.float64)

        # Frecuencias naturales ω, shape (N,). Estáticas (no cambian en t)
        self.omega = np.zeros(N, dtype=np.float64)

        # Buffers de fases para Euler: actual y siguiente
        self.theta_curr = np.zeros(N, dtype=np.float64)
        self.theta_next = np.zeros(N, dtype=np.float64)

    def initialize(self, sigma: float = 1.0):
        """omega ~ N(0, sigma), theta inicial ~ U(-π, π)."""
        self.omega      = np.random.normal(0, sigma, self.N)
        self.theta_curr = np.random.uniform(-np.pi, np.pi, self.N)

    def run(self, K: float):
        """Ejecuta la integración. Numba modifica los arrays in-place."""
        _integrar(self.theta_curr, self.theta_next,
                  self.omega, K, self.dt, self.steps, self.R, self.psi)

    @property
    def R_mean(self):
        """⟨R⟩ descartando el primer 25% (transitorio)."""
        n_trans = self.steps // 4
        return np.mean(self.R[n_trans:])

    @property
    def R_std(self) -> float:
        """σ_R: mide la metaestabilidad (pico en Kc)."""
        n_trans = self.steps // 4
        return float(np.std(self.R[n_trans:]))


@njit(fastmath=True, cache=True)
def _integrar(theta_curr, theta_next, omega, K, dt, steps, R, psi):
    N = theta_curr.shape[0]

    for t in range(steps):

        # R(t) y ψ(t) a partir de R·e^(iψ) = (1/N) Σ e^(iθ_j)
        re, im = 0.0, 0.0
        for j in range(N):
            re += math.cos(theta_curr[j])
            im += math.sin(theta_curr[j])

        R[t]   = math.sqrt(re**2 + im**2) / N
        psi[t] = math.atan2(im, re)

        # Euler: θ̇_i = ω_i + K·R·sin(ψ - θ_i)
        for i in range(N):
            theta_next[i] = theta_curr[i] + dt * (omega[i] + K * R[t] * math.sin(psi[t] - theta_curr[i]))

        # Swap de buffers: ahora theta_curr contiene t+1
        for i in range(N):
            theta_curr[i] = theta_next[i]

    # Último paso: calculamos R y ψ en t = steps
    re, im = 0.0, 0.0
    for j in range(N):
        re += math.cos(theta_curr[j])
        im += math.sin(theta_curr[j])
    R[steps]   = math.sqrt(re**2 + im**2) / N
    psi[steps] = math.atan2(im, re)


def Simulacion_Kuramoto(N, K, sigma, dt, t_max):
    num_pasos = int(t_max / dt)
    sys = KuramotoSystem(N=N, steps=num_pasos, dt=dt)
    sys.initialize(sigma=sigma)
    sys.run(K=K)
    return sys


def _una_simulacion_indexada(i, j, N, K, sigma, dt, t_max):
    """Ejecuta una simulación y devuelve su posición (i,j) en la matriz."""
    sys = Simulacion_Kuramoto(N, K, sigma, dt, t_max)
    return (i, j, sys.R_mean, sys.R_std)


def Kc_teorica(sigma):
    """Kc teórica para distribución gaussiana: Kc = σ·√(8/π)."""
    return sigma * math.sqrt(8 / math.pi)


def Kc_experimental(K_values, R_stds):
    """Kc como argmax de σ_R: máxima variabilidad = transición."""
    return K_values[np.argmax(R_stds)]


def K_values_gaussiano(num_K, K_min, K_max, Kc, width_factor=0.3):
    """Genera num_K valores de K con densidad gaussiana centrada en ~Kc."""
    K_center = 1.05 * Kc
    sigma_K  = width_factor * Kc

    # Invertimos la CDF: uniforme en [u_min, u_max] → K no uniforme
    u_min = norm.cdf(K_min, loc=K_center, scale=sigma_K)
    u_max = norm.cdf(K_max, loc=K_center, scale=sigma_K)

    u_values = np.linspace(u_min, u_max, num_K)
    K_values = norm.ppf(u_values, loc=K_center, scale=sigma_K)
    return K_values

def K_values_curva(num_K, K_min, K_max, Kc, n_fine=10000):
    """Genera num_K valores de K equiespaciados en longitud de arco
    de la curva teórica R(K) = sqrt((K - Kc) / K), con ejes normalizados."""
    K_fine = np.linspace(K_min, K_max, n_fine)

    R_fine = np.where(K_fine > Kc, np.sqrt((K_fine - Kc) / K_fine), 0.0)

    # Normalizar ambos ejes a [0, 1] para que la geometría sea visual
    K_norm = (K_fine - K_min) / (K_max - K_min)
    R_max  = np.max(R_fine) if np.max(R_fine) > 0 else 1.0
    R_norm = R_fine / R_max

    dK_n = np.diff(K_norm)
    dR_n = np.diff(R_norm)

    ds = np.sqrt(dK_n**2 + dR_n**2)
    s_cumul = np.concatenate([[0.0], np.cumsum(ds)])

    s_uniform = np.linspace(0, s_cumul[-1], num_K)
    K_values = np.interp(s_uniform, s_cumul, K_fine)
    return K_values


def K_values_curva_tstudent(num_K, K_min, K_max, Kc, width_factor, n_fine, df):
    """Genera num_K valores de K distribuidos sobre la curva teórica
    R(K) = sqrt((K - Kc) / K), con densidad t-Student centrada en ~Kc."""
    K_center = Kc
    sigma_K  = width_factor * Kc
    K_fine = np.linspace(K_min, K_max, n_fine)
    R_fine = np.where(K_fine > Kc, np.sqrt((K_fine - Kc) / K_fine), 0.0)
    # Normalizar ambos ejes a [0, 1]
    K_norm = (K_fine - K_min) / (K_max - K_min)
    R_max  = np.max(R_fine) if np.max(R_fine) > 0 else 1.0
    R_norm = R_fine / R_max
    dK_n = np.diff(K_norm)
    dR_n = np.diff(R_norm)
    ds = np.sqrt(dK_n**2 + dR_n**2)
    s_cumul = np.concatenate([[0.0], np.cumsum(ds)])
    # Peso t-Student en cada segmento (colas más pesadas que gaussiana)
    K_mid = 0.5 * (K_fine[:-1] + K_fine[1:])
    w = t_dist.pdf(K_mid, df=df, loc=K_center, scale=sigma_K)
    w_ds = w * ds
    F = np.concatenate([[0.0], np.cumsum(w_ds)])
    F /= F[-1]
    u_values = np.linspace(0, 1, num_K)
    s_samples = np.interp(u_values, F, s_cumul)
    K_values  = np.interp(s_samples, s_cumul, K_fine)
    return K_values


def barrido_completo(N, sigma_values, num_K, K_min, K_max, num_runs, dt, t_max, width_factor, n_fine, n_jobs=-1):
    n_sigmas = len(sigma_values)

    # Matrices de resultados
    K_values_per_sigma = np.zeros((n_sigmas, num_K))
    R_means            = np.zeros((n_sigmas, num_K))
    R_stds             = np.zeros((n_sigmas, num_K))
    R_mean_stds        = np.zeros((n_sigmas, num_K))

    # Pre-calculamos los K_values para cada sigma (cada uno centrado en su Kc)
    for i, sigma in enumerate(sigma_values):
        Kc = Kc_teorica(sigma)
        # K_values_per_sigma[i] = K_values_gaussiano(num_K, K_min, K_max, Kc, width_factor)
        # K_values_per_sigma[i] = K_values_curva(num_K, K_min, K_max, Kc, n_fine)
        K_values_per_sigma[i] = K_values_curva_tstudent(num_K, K_min, K_max, Kc, width_factor, n_fine, df=2)

    # Generamos TODAS las tareas de golpe, con sus índices (i, j)
    tareas = []
    for i, sigma in enumerate(sigma_values):
        for j, K in enumerate(K_values_per_sigma[i]):
            for _ in range(num_runs):
                tareas.append((i, j, N, K, sigma, dt, t_max))

    print(f"Lanzando {len(tareas)} simulaciones en paralelo...")

    # UN solo Parallel para todo: el pool se reutiliza
    resultados = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_una_simulacion_indexada)(*tarea) for tarea in tareas
    )

    # Agregamos los resultados por (i, j)
    means_por_punto = {}
    stds_por_punto  = {}
    for (i, j, r_mean, r_std) in resultados:
        means_por_punto.setdefault((i, j), []).append(r_mean)
        stds_por_punto .setdefault((i, j), []).append(r_std)

    for (i, j), means in means_por_punto.items():
        R_means[i, j]     = np.mean(means)
        R_stds[i, j]      = np.mean(stds_por_punto[(i, j)])
        R_mean_stds[i, j] = np.std(means)

    return K_values_per_sigma, R_means, R_stds, R_mean_stds


def _draw_R_vs_K(ax, K_values_per_sigma, sigma_values, R_means, R_mean_stds, colors):
    for i, sigma in enumerate(sigma_values):
        Kc_th = Kc_teorica(sigma)
        ax.fill_between(K_values_per_sigma[i], R_means[i] - R_mean_stds[i], R_means[i] + R_mean_stds[i], color=colors[i], alpha=0.2)
        ax.plot(K_values_per_sigma[i], R_means[i], marker='o', markersize=4, linewidth=1.5, color=colors[i], label=fr'$\sigma = {sigma:.2f}$   $K_c^{{\mathrm{{th}}}} = {Kc_th:.2f}$')
        ax.axvline(Kc_th, color=colors[i], linestyle='--', linewidth=1.0, alpha=0.6)

    ax.set_xlabel(r'Acoplamiento $K$')
    ax.set_ylabel(r'Parámetro de orden $\langle R \rangle$')
    ax.set_title(r'Transición de sincronización en el modelo de Kuramoto')
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right', title=r'Líneas: $K_c$ teórica')


def _draw_sigmaR_vs_K(ax, K_values_per_sigma, sigma_values, R_stds, colors):
    for i, sigma in enumerate(sigma_values):
        Kc_exp = K_values_per_sigma[i, np.argmax(R_stds[i])]
        ax.plot(K_values_per_sigma[i], R_stds[i], marker='o', markersize=4, linewidth=1.5, color=colors[i], label=fr'$\sigma = {sigma:.2f}$   $K_c^{{\mathrm{{exp}}}} = {Kc_exp:.2f}$')
        ax.axvline(Kc_exp, color=colors[i], linestyle=':', linewidth=1.2, alpha=0.8)

    ax.set_xlabel(r'Acoplamiento $K$')
    ax.set_ylabel(r'Desviación estándar $\sigma_R$')
    ax.set_title(r'Metaestabilidad: fluctuaciones del parámetro de orden')
    ax.set_xlim(left=0)
    ax.legend(loc='upper right', title=r'Líneas: $K_c$ experimental')


def _add_info_box(ax, N, num_runs, loc='bottom'):
    y, va = (0.05, 'bottom') if loc == 'bottom' else (0.95, 'top')
    ax.text(0.98, y, fr'$N = {N}$, $\langle \mathrm{{runs}} \rangle = {num_runs}$', transform=ax.transAxes, ha='right', va=va, fontsize=9, style='italic', alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))


def plot_R_vs_K(K_values_per_sigma, sigma_values, R_means, R_mean_stds, N, t_max, num_K, num_runs, guardar=False):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_R_vs_K(ax, K_values_per_sigma, sigma_values, R_means, R_mean_stds, colors)
    _add_info_box(ax, N, num_runs, loc='bottom')

    if guardar:
        num_sigmas = len(sigma_values)
        params_folder = _params_folder_name(N, t_max, num_K, num_runs, num_sigmas)
        save_dir = os.path.join(FIGURES_DIR, 'R_vs_K', params_folder)
        fname = f'R_vs_K_N{N}_t{t_max}_K{num_K}_Runs{num_runs}_sigmas{num_sigmas}.png'
        fig.savefig(_unique_path(save_dir, fname))


def plot_sigmaR_vs_K(K_values_per_sigma, sigma_values, R_stds, N, t_max, num_K, num_runs, guardar=False):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_sigmaR_vs_K(ax, K_values_per_sigma, sigma_values, R_stds, colors)
    _add_info_box(ax, N, num_runs, loc='top')

    if guardar:
        num_sigmas = len(sigma_values)
        params_folder = _params_folder_name(N, t_max, num_K, num_runs, num_sigmas)
        save_dir = os.path.join(FIGURES_DIR, 'sigmaR_vs_K', params_folder)
        fname = f'sigmaR_vs_K_N{N}_t{t_max}_K{num_K}_Runs{num_runs}_sigmas{num_sigmas}.png'
        fig.savefig(_unique_path(save_dir, fname))


def plot_combined(K_values_per_sigma, sigma_values, R_means, R_stds, R_mean_stds, N, t_max, num_K, num_runs, guardar=False):
    """Ambas gráficas lado a lado, en una sola imagen."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_R_vs_K(ax1, K_values_per_sigma, sigma_values, R_means, R_mean_stds, colors)
    _add_info_box(ax1, N, num_runs, loc='bottom')

    _draw_sigmaR_vs_K(ax2, K_values_per_sigma, sigma_values, R_stds, colors)
    _add_info_box(ax2, N, num_runs, loc='top')

    fig.tight_layout()

    if guardar:
        num_sigmas = len(sigma_values)
        params_folder = _params_folder_name(N, t_max, num_K, num_runs, num_sigmas)
        save_dir = os.path.join(FIGURES_DIR, 'resultados', params_folder)
        fname = f'resultados_N{N}_t{t_max}_K{num_K}_Runs{num_runs}_sigmas{num_sigmas}.png'
        fig.savefig(_unique_path(save_dir, fname))
    plt.show()



def main():
    # Parámetros del sistema
    N     = 1000
    dt    = 0.01
    t_max = 150.0

    # Parámetros del barrido
    num_K        = 50
    num_sigmas   = 4
    num_runs     = 15
    width_factor = 0.6
    n_fine       = 10000

    # Rangos
    K_min,     K_max     = 0.5, 4.0
    sigma_min, sigma_max = 0.5, 1.5

    sigma_values = np.linspace(sigma_min, sigma_max, num_sigmas)

    print(f"Barrido: {num_sigmas} sigmas × {num_K} K × {num_runs} runs = {num_sigmas * num_K * num_runs} simulaciones\n")

    K_values_per_sigma, R_means, R_stds, R_mean_stds = barrido_completo(N, sigma_values, num_K, K_min, K_max, num_runs, dt, t_max, width_factor, n_fine)

    plot_R_vs_K(K_values_per_sigma, sigma_values, R_means, R_mean_stds, N, t_max, num_K, num_runs, guardar=True)
    plot_sigmaR_vs_K(K_values_per_sigma, sigma_values, R_stds, N, t_max, num_K, num_runs, guardar=True)
    plot_combined(K_values_per_sigma, sigma_values, R_means, R_stds, R_mean_stds, N, t_max, num_K, num_runs, guardar=True)

    # Tabla comparativa
    print("\n" + "="*60)
    print(f"{'σ':>6} | {'Kc teórica':>11} | {'Kc experimental':>15} | {'<σ(R_mean)>':>12}")
    print("-"*60)
    for i, sigma in enumerate(sigma_values):
        Kc_th            = Kc_teorica(sigma)
        Kc_exp           = K_values_per_sigma[i, np.argmax(R_stds[i])]
        mean_variability = np.mean(R_mean_stds[i])
        print(f"{sigma:>6.2f} | {Kc_th:>11.3f} | {Kc_exp:>15.3f} | {mean_variability:>12.5f}")
    print("="*60)

    global_variability = np.mean(R_mean_stds)
    print(f"\n<σ(R_mean)> global = {global_variability:.5f}")

if __name__ == "__main__":
    main()
