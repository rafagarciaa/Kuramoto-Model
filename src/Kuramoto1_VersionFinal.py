import os
import sys
import math
import time
import matplotlib
matplotlib.use('Agg')          # backend no interactivo, sin Tk, thread-safe
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

# ----------------------------------------------------------------------------
# Utilidades de carpetas (mismo esquema que Kuramoto2)
# ----------------------------------------------------------------------------

# Carpeta raíz donde se acumulan todas las ejecuciones.
RESULTADOS_BASE = 'resultados\\Tarea1'


def crear_carpeta_resultados(N, num_sigmas, num_K, num_runs, t_max):
    """Crea una carpeta 'resultados/Tarea1/N{N}_sigmas{s}_K{k}_Runs{r}_t{t}/'.
    Si ya existía, añade sufijo (1), (2)... para no sobreescribir.
    Devuelve la ruta absoluta de la carpeta creada."""
    os.makedirs(RESULTADOS_BASE, exist_ok=True)
    base = f"N{N}_sigmas{num_sigmas}_K{num_K}_Runs{num_runs}_t{t_max}"
    path = os.path.join(RESULTADOS_BASE, base)
    n = 1
    while os.path.exists(path):
        path = os.path.join(RESULTADOS_BASE, f"{base}({n})")
        n += 1
    os.makedirs(path)
    return os.path.abspath(path)

def generar_ICs(num_sigmas, num_runs, N, sigma_values, seed=None):
    """
    Genera condiciones iniciales pareadas: para cada sigma, num_runs ternas
    (omega, theta_0) que se reutilizarán en todos los valores de K.
    Esto implementa Common Random Numbers (CRN): la run r del sigma i
    usa siempre las mismas ICs en todos los K_j, lo que reduce varianza
    cruzada entre celdas y suaviza la curva R(K).
    
    Returns
    -------
    omegas_IC : ndarray, shape (num_sigmas, num_runs, N)
    thetas_IC : ndarray, shape (num_sigmas, num_runs, N)
    """
    rng = np.random.default_rng(seed)
    omegas_IC = np.zeros((num_sigmas, num_runs, N), dtype=np.float64)
    thetas_IC = np.zeros((num_sigmas, num_runs, N), dtype=np.float64)
    for i, sigma in enumerate(sigma_values):
        omegas_IC[i] = rng.normal(0.0, sigma, size=(num_runs, N))
        thetas_IC[i] = rng.uniform(-np.pi, np.pi, size=(num_runs, N))
    return omegas_IC, thetas_IC

def guardar_ICs(omegas_IC, thetas_IC, sigma_values, run_dir):
    """Guarda las ICs en run_dir/condiciones_iniciales.npz para reproducibilidad."""
    ruta = os.path.join(run_dir, 'condiciones_iniciales.npz')
    np.savez_compressed(ruta,
                        omegas=omegas_IC,
                        thetas=thetas_IC,
                        sigma_values=np.asarray(sigma_values))
    return ruta

def _ruta(directorio, nombre):
    """Crea el directorio si no existe y devuelve la ruta completa al fichero."""
    os.makedirs(directorio, exist_ok=True)
    return os.path.join(directorio, nombre)


def guardar_params_txt(run_dir, params_dict):
    """Guarda los parámetros de la ejecución en run_dir/params.txt."""
    with open(os.path.join(run_dir, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write("Parametros de la ejecucion\n")
        f.write("=" * 40 + "\n")
        for k, v in params_dict.items():
            f.write(f"{k:25s} = {v}\n")


# ----------------------------------------------------------------------------
# Logging: duplica todo lo que se imprime a un fichero log.txt en run_dir
# ----------------------------------------------------------------------------

class Tee:
    """Duplica la escritura a varios streams (consola + archivo).
    Hace flush en cada write para que el log quede grabado aunque el
    script se caiga a mitad de ejecucion."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # Algunas librerias (tqdm, joblib) preguntan esto para decidir
        # si usar barras de progreso o salida plana.
        return any(getattr(s, 'isatty', lambda: False)() for s in self.streams)


def iniciar_log(run_dir):
    """Redirige stdout y stderr a un Tee que escribe tambien en run_dir/log.txt.
    Devuelve (log_file, stdout_orig, stderr_orig) para poder restaurarlos al final."""
    log_path = os.path.join(run_dir, 'log.txt')
    log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # line-buffered
    log_file.write(f"Log de ejecucion - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 60 + "\n\n")
    log_file.flush()
    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    sys.stdout = Tee(stdout_orig, log_file)
    sys.stderr = Tee(stderr_orig, log_file)
    return log_file, stdout_orig, stderr_orig


def cerrar_log(log_file, stdout_orig, stderr_orig):
    """Restaura stdout/stderr originales y cierra el archivo de log."""
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    log_file.close()


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

    def initialize(self, sigma: float = 1.0, omega=None, theta_0=None):
        """omega ~ N(0, sigma) y theta inicial ~ U(-π, π) si no se pasan ICs.
        Si se pasan `omega` y/o `theta_0`, se usan esos en su lugar."""
        if omega is None:
            self.omega = np.random.normal(0, sigma, self.N)
        else:
            self.omega = np.asarray(omega, dtype=np.float64).copy()
        if theta_0 is None:
            self.theta_curr = np.random.uniform(-np.pi, np.pi, self.N)
        else:
            # .copy() es crítico: _integrar modifica theta_curr in-place y
            # no queremos corromper la IC compartida entre workers.
            self.theta_curr = np.asarray(theta_0, dtype=np.float64).copy()

    def run(self, K: float):
        """Ejecuta la integración. Numba modifica los arrays in-place."""
        _integrar(self.theta_curr, self.theta_next, self.omega, K, self.dt, self.steps, self.R, self.psi)

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


def Simulacion_Kuramoto(N, K, sigma, dt, t_max, omega=None, theta_0=None):
    num_pasos = int(t_max / dt)
    sys = KuramotoSystem(N=N, steps=num_pasos, dt=dt)
    sys.initialize(sigma=sigma, omega=omega, theta_0=theta_0)
    sys.run(K=K)
    return sys


def _una_simulacion_indexada(i, j, N, K, sigma, dt, t_max, omega_ic, theta0_ic):
    """Ejecuta una simulación con ICs precomputadas. Devuelve (i, j, R_mean, R_std)."""
    sys = Simulacion_Kuramoto(N, K, sigma, dt, t_max, omega=omega_ic, theta_0=theta0_ic)
    return (i, j, sys.R_mean, sys.R_std)


def Kc_teorica(sigma):
    """Kc teórica para distribución gaussiana: Kc = σ·√(8/π)."""
    return sigma * math.sqrt(8 / math.pi)

def Kc_experimental(K_values, R_stds, window=3):
    """Kc estimado por ajuste parabolico al maximo de sigma_R.

    En lugar de tomar simplemente argmax(sigma_R), que esta atado a la rejilla
    discreta de K, ajustamos una parabola a los (2*window+1) puntos centrados
    en el maximo y devolvemos el vertice analitico. Esto da una Kc_exp continua
    (no atada a la rejilla) y reduce el ruido local del estimador.

    Si la ventana no llega a 3 puntos o el ajuste no es concavo hacia abajo
    (a >= 0), o si el vertice cae fuera de la ventana, caemos al argmax
    discreto como salvaguarda.

    Parameters
    ----------
    K_values : array_like
        Valores de K del barrido.
    R_stds : array_like
        sigma_R correspondiente a cada K.
    window : int
        Numero de puntos a cada lado del maximo a usar en el ajuste.
        Total de puntos ajustados = 2*window + 1.

    Returns
    -------
    Kc : float
        Estimacion continua de Kc.
    """
    K_values = np.asarray(K_values)
    R_stds   = np.asarray(R_stds)
    idx      = int(np.argmax(R_stds))

    # Ventana centrada en idx, recortada a los limites del array.
    lo = max(0, idx - window)
    hi = min(len(K_values), idx + window + 1)

    if hi - lo < 3:
        return float(K_values[idx])

    K_win = K_values[lo:hi]
    R_win = R_stds[lo:hi]

    # Ajuste por minimos cuadrados: y = a*K^2 + b*K + c
    a, b, _ = np.polyfit(K_win, R_win, 2)

    # Sin concavidad hacia abajo no hay vertice util.
    if a >= 0:
        return float(K_values[idx])

    K_vertex = -b / (2.0 * a)

    # Si el vertice escapa de la ventana, el ajuste no es fiable.
    if K_vertex < K_win[0] or K_vertex > K_win[-1]:
        return float(K_values[idx])

    return float(K_vertex)

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

def K_values_tstudent(num_K, K_min, K_max, Kc, width_factor, df):
    """Densidad t-Student centrada en Kc, simétrica, sin guía geométrica."""
    K_center = Kc
    sigma_K  = width_factor * Kc
    u_min = t_dist.cdf(K_min, df=df, loc=K_center, scale=sigma_K)
    u_max = t_dist.cdf(K_max, df=df, loc=K_center, scale=sigma_K)
    u_values = np.linspace(u_min, u_max, num_K)
    K_values = t_dist.ppf(u_values, df=df, loc=K_center, scale=sigma_K)
    return K_values

def t_max_per_K(K_values, Kc, t_max_base, t_max_peak, width_factor=0.3, df=2):
    """Asigna t_max(K) con la MISMA forma que la densidad t-Student usada
    para colocar los K_values en `K_values_tstudent`.

    Si los K-points se colocan con densidad proporcional a pdf_t(K; Kc, sigma_K, df),
    entonces el tiempo por punto sigue exactamente esa misma forma:
        T(K) = T_base + (T_peak - T_base) * pdf_t(K) / pdf_t(Kc)

    Asi un unico (width_factor, df) controla simultaneamente DONDE colocas los K
    y CUANTO tiempo se simula cada K. La concentracion de potencia de computo
    por punto sigue la concentracion de puntos.

    Comparado con un perfil gaussiano, las colas pesadas de la t-Student (df=2)
    mantienen tiempos algo mas altos en los flancos (1.5-2 sigma_K de Kc), donde
    sigma_R aun es no-trivial y la transicion esta ocurriendo.

    Parameters
    ----------
    K_values : ndarray
        Valores de K del barrido (ya colocados con K_values_tstudent).
    Kc : float
        Acoplamiento critico (teorico) en torno al cual concentrar el tiempo.
    t_max_base : float
        Tiempo minimo de simulacion (en las colas, lejos de Kc).
    t_max_peak : float
        Tiempo maximo de simulacion (justo en Kc).
    width_factor : float
        Anchura de la t-Student como fraccion de Kc. DEBE coincidir con el
        valor pasado a K_values_tstudent para que las dos densidades
        (K-points y T) queden alineadas.
    df : int
        Grados de libertad de la t-Student. DEBE coincidir con K_values_tstudent.

    Returns
    -------
    T : ndarray
        Array shape (len(K_values),) con t_max[j] para cada K_j.
    """
    sigma_K = width_factor * Kc
    pdf     = t_dist.pdf(K_values, df=df, loc=Kc, scale=sigma_K)
    pdf_max = t_dist.pdf(Kc,        df=df, loc=Kc, scale=sigma_K)
    weight  = pdf / pdf_max
    return t_max_base + (t_max_peak - t_max_base) * weight


def barrido_completo(N, sigma_values, num_K, K_min, K_max, num_runs, dt, t_max_base, t_max_peak, width_factor, n_fine, omegas_IC, thetas_IC, n_jobs=-1):
    n_sigmas = len(sigma_values)

    K_values_per_sigma = np.zeros((n_sigmas, num_K))
    T_per_sigma_K      = np.zeros((n_sigmas, num_K))   # t_max(K) por celda
    R_means            = np.zeros((n_sigmas, num_K))
    R_stds             = np.zeros((n_sigmas, num_K))
    R_mean_stds        = np.zeros((n_sigmas, num_K))

    for i, sigma in enumerate(sigma_values):
        Kc = Kc_teorica(sigma)
        K_values_per_sigma[i] = K_values_tstudent(num_K, K_min, K_max, Kc, width_factor, df=2)
        T_per_sigma_K[i]      = t_max_per_K(K_values_per_sigma[i], Kc, t_max_base, t_max_peak, width_factor, df=2)

    # Resumen del coste estimado: total de u.t. simuladas y desglose por sigma.
    print("Asignacion de t_max(K) (perfil t-Student, misma densidad que K-points):")
    print(f"  t_max_base = {t_max_base:.0f} u.t.   t_max_peak = {t_max_peak:.0f} u.t.   width_factor = {width_factor} (compartido con K_values)")
    for i, sigma in enumerate(sigma_values):
        T = T_per_sigma_K[i]
        print(f"  sigma={sigma:.2f}: t_max in [{T.min():.0f}, {T.max():.0f}] u.t., <t_max>={T.mean():.0f}, suma={T.sum()*num_runs:.0f} u.t.")
    total_ut = T_per_sigma_K.sum() * num_runs
    print(f"  Total simulado (todas las tareas): {total_ut:.0f} u.t. ({total_ut*int(1/dt):.2e} pasos)\n")

    # Tareas con ICs pareadas: run r del sigma i usa siempre (omegas_IC[i,r], thetas_IC[i,r]).
    # Cada tarea lleva su propio t_max[j] segun el perfil gaussiano.
    tareas = []
    for i, sigma in enumerate(sigma_values):
        for j, K in enumerate(K_values_per_sigma[i]):
            t_max_ij = float(T_per_sigma_K[i, j])
            for r in range(num_runs):
                tareas.append((i, j, N, K, sigma, dt, t_max_ij, omegas_IC[i, r], thetas_IC[i, r]))

    # Balanceo de carga: ordenamos las tareas de mayor a menor t_max para que
    # las pesadas arranquen primero y las cortas rellenen los huecos al final.
    # El indice 6 de cada tupla es t_max_ij.
    tareas.sort(key=lambda tarea: -tarea[6])

    print(f"Lanzando {len(tareas)} simulaciones en paralelo (ordenadas por t_max desc)...")

    resultados = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_una_simulacion_indexada)(*tarea) for tarea in tareas
    )

    means_por_punto = {}
    stds_por_punto  = {}
    for (i, j, r_mean, r_std) in resultados:
        means_por_punto.setdefault((i, j), []).append(r_mean)
        stds_por_punto .setdefault((i, j), []).append(r_std)

    for (i, j), means in means_por_punto.items():
        R_means[i, j]     = np.mean(means)
        R_stds[i, j]      = np.mean(stds_por_punto[(i, j)])
        R_mean_stds[i, j] = np.std(means)

    return K_values_per_sigma, T_per_sigma_K, R_means, R_stds, R_mean_stds

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
        Kc_exp = Kc_experimental(K_values_per_sigma[i], R_stds[i])
        ax.plot(K_values_per_sigma[i], R_stds[i], marker='o', markersize=4, linewidth=1.5, color=colors[i], label=fr'$\sigma = {sigma:.2f}$   $K_c^{{\mathrm{{exp}}}} = {Kc_exp:.3f}$')
        ax.axvline(Kc_exp, color=colors[i], linestyle=':', linewidth=1.2, alpha=0.8)

    ax.set_xlabel(r'Acoplamiento $K$')
    ax.set_ylabel(r'Desviación estándar $\sigma_R$')
    ax.set_title(r'Metaestabilidad: fluctuaciones del parámetro de orden')
    ax.set_xlim(left=0)
    ax.legend(loc='upper right', title=r'Líneas: $K_c$ experimental')


def _add_info_box(ax, N, num_runs, loc='bottom'):
    y, va = (0.05, 'bottom') if loc == 'bottom' else (0.95, 'top')
    ax.text(0.98, y, fr'$N = {N}$, $\langle \mathrm{{runs}} \rangle = {num_runs}$', transform=ax.transAxes, ha='right', va=va, fontsize=9, style='italic', alpha=0.7, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))


def plot_R_vs_K(K_values_per_sigma, sigma_values, R_means, R_mean_stds, N, num_runs, save_dir):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_R_vs_K(ax, K_values_per_sigma, sigma_values, R_means, R_mean_stds, colors)
    _add_info_box(ax, N, num_runs, loc='bottom')

    fig.savefig(_ruta(save_dir, 'R_vs_K.png'))
    plt.close(fig)


def plot_sigmaR_vs_K(K_values_per_sigma, sigma_values, R_stds, N, num_runs, save_dir):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_sigmaR_vs_K(ax, K_values_per_sigma, sigma_values, R_stds, colors)
    _add_info_box(ax, N, num_runs, loc='top')

    fig.savefig(_ruta(save_dir, 'sigmaR_vs_K.png'))
    plt.close(fig)


def plot_combined(K_values_per_sigma, sigma_values, R_means, R_stds, R_mean_stds, N, num_runs, save_dir):
    """Ambas gráficas lado a lado, en una sola imagen."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sigma_values)))

    _draw_R_vs_K(ax1, K_values_per_sigma, sigma_values, R_means, R_mean_stds, colors)
    _add_info_box(ax1, N, num_runs, loc='bottom')

    _draw_sigmaR_vs_K(ax2, K_values_per_sigma, sigma_values, R_stds, colors)
    _add_info_box(ax2, N, num_runs, loc='top')

    fig.tight_layout()

    fig.savefig(_ruta(save_dir, 'combinado.png'))
    plt.close(fig)

def main():
    t0 = time.perf_counter()

    # Parámetros del sistema
    N     = 3000        #Fijo
    dt    = 0.025       #Fijo



    t_max_base = 400.0
    t_max_peak = 1500.0

    # Parámetros del barrido
    num_K        = 300
    num_sigmas   = 3
    num_runs     = 1
    width_factor = 0.3         #Fijo (anchura de la densidad de K en torno a Kc)
    n_fine       = 10000       #Fijo

    # Rangos
    K_min,     K_max     = 0.25, 4.0
    sigma_min, sigma_max = 0.5, 1.5

    sigma_values = np.linspace(sigma_min, sigma_max, num_sigmas)

    # ============================================================
    # Setup: carpeta de resultados, log y params.txt
    # ============================================================
    # Usamos un identificador "base-peak" para distinguir ejecuciones con
    # distintos perfiles de t_max(K).
    t_max_label = f"{int(t_max_base)}-{int(t_max_peak)}"
    run_dir = crear_carpeta_resultados(N, num_sigmas, num_K, num_runs, t_max_label)

    # Activamos el log: a partir de aqui todo lo que se imprima por consola
    # se escribe tambien en run_dir/log.txt
    log_file, stdout_orig, stderr_orig = iniciar_log(run_dir)

    try:
        print(f"Carpeta de resultados: {run_dir}")
        print(f"Log de salida:         {os.path.join(run_dir, 'log.txt')}\n")

        guardar_params_txt(run_dir, {
            'N': N, 'dt': dt,
            't_max_base': t_max_base, 't_max_peak': t_max_peak,
            'num_K': num_K, 'num_sigmas': num_sigmas, 'num_runs': num_runs,
            'width_factor': width_factor, 'n_fine': n_fine,
            'K_min': K_min, 'K_max': K_max,
            'sigma_min': sigma_min, 'sigma_max': sigma_max,
            'sigma_values': list(sigma_values),
            'ICs_pareadas': True,
            't_max_adaptativo': 't-Student (misma densidad que K-points)',
        })

        # Generamos las ICs UNA sola vez por (sigma, run) y las reutilizamos en todos los K
        print("Generando condiciones iniciales pareadas (Common Random Numbers)...")
        omegas_IC, thetas_IC = generar_ICs(num_sigmas, num_runs, N, sigma_values)
        ruta_ICs = guardar_ICs(omegas_IC, thetas_IC, sigma_values, run_dir)
        print(f"  Guardadas en: {ruta_ICs}\n")

        print(f"Barrido: {num_sigmas} sigmas x {num_K} K x {num_runs} runs = {num_sigmas * num_K * num_runs} simulaciones\n")

        K_values_per_sigma, T_per_sigma_K, R_means, R_stds, R_mean_stds = barrido_completo(
            N, sigma_values, num_K, K_min, K_max, num_runs, dt,
            t_max_base, t_max_peak, width_factor,
            n_fine, omegas_IC, thetas_IC,
        )

        # Guardamos K_values y T_per_sigma_K en disco para reproducibilidad y postproc.
        np.savez_compressed(
            os.path.join(run_dir, 'barrido.npz'),
            K_values_per_sigma=K_values_per_sigma,
            T_per_sigma_K=T_per_sigma_K,
            R_means=R_means, R_stds=R_stds, R_mean_stds=R_mean_stds,
            sigma_values=np.asarray(sigma_values),
        )

        plot_R_vs_K(K_values_per_sigma, sigma_values, R_means, R_mean_stds, N, num_runs, run_dir)
        plot_sigmaR_vs_K(K_values_per_sigma, sigma_values, R_stds, N, num_runs, run_dir)
        plot_combined(K_values_per_sigma, sigma_values, R_means, R_stds, R_mean_stds, N, num_runs, run_dir)

        # Tabla comparativa
        print("\n" + "="*60)
        print(f"{'sigma':>6} | {'Kc teorica':>11} | {'Kc experimental':>15} | {'<sigma(R_mean)>':>12}")
        print("-"*60)
        for i, sigma in enumerate(sigma_values):
            Kc_th            = Kc_teorica(sigma)
            Kc_exp           = Kc_experimental(K_values_per_sigma[i], R_stds[i])
            mean_variability = np.mean(R_mean_stds[i])
            print(f"{sigma:>6.2f} | {Kc_th:>11.4f} | {Kc_exp:>15.4f} | {mean_variability:>12.5f}")
        print("="*60)

        global_variability = np.mean(R_mean_stds)
        print(f"\n<sigma(R_mean)> global = {global_variability:.5f}")

        print("\n" + "=" * 60)
        print(f"Listo. Resultados en: {run_dir}")
        print("=" * 60)

        elapsed = time.perf_counter() - t0
        print(f"\nTiempo total de ejecucion: {elapsed/60:.1f} min ({elapsed:.1f} s)")

    finally:
        # Restauramos stdout/stderr y cerramos el log pase lo que pase
        cerrar_log(log_file, stdout_orig, stderr_orig)


if __name__ == "__main__":
    main()